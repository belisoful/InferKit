//
//  NFKMLXLaMa.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXFFT
import MLXNN

// LaMa is a single-forward inpainter built on Fast Fourier Convolutions (FFC): each layer splits its
// channels into a local (spatial) branch and a global (spectral) branch, where the global branch runs
// a 2-D FFT, a 1×1 convolution on the stacked real/imaginary parts, and an inverse FFT. It runs
// through `NFKMLXModuleBackend`-shaped I/O: a plate under `NFKInputImage` and a mask under
// `NFKInputMask`, returning the inpainted image. Tensors flow NHWC.
//
// The convolutions reflect-pad as the reference does (`padding_mode='reflect'`), and the FFT uses
// orthogonal normalization. The big-lama checkpoint stores its weights under a flat `model.N.*`
// Sequential index, which `remapReferenceKey` translates.

/// LaMa generator dimensions. The defaults are the released big-lama geometry, read off its own
/// `config.yaml`: 64 base channels, three downsampling stages, **18** residual blocks, and a global
/// channel fraction of 0.75 that appears only at the last downsample and through the trunk.
public struct NFKMLXLaMaConfiguration: Sendable {
    public var baseChannels: Int = 64
    public var downsampling: Int = 3
    public var blocks: Int = 18
    public var ratio: Float = 0.75                              // global-channel fraction in the FFC trunk
    public init() {}
}

/// Splits channels into `(local, global)` counts for a given global ratio.
private func split(_ channels: Int, ratio: Float) -> (local: Int, global: Int) {
    let global = Int(Float(channels) * ratio)
    return (channels - global, global)
}

/// The spectral branch of an FFC: FFT → 1×1 conv on stacked real/imag → inverse FFT.
final class NFKLaMaFourierUnit: Module {
    @ModuleInfo(key: "conv_layer") var convLayer: Conv2d
    let bn: BatchNorm

    init(channels: Int) {
        _convLayer.wrappedValue = Conv2d(inputChannels: channels * 2, outputChannels: channels * 2, kernelSize: 1, bias: false)
        bn = BatchNorm(featureCount: channels * 2)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (n, h, w, c) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        let scale = 1.0 / sqrtf(Float(h * w))                   // orthogonal FFT normalization
        let spectrum = rfft2(x, axes: [1, 2]) * scale           // complex [N, H, W/2+1, C]
        let wHalf = spectrum.shape[2]

        // Interleave real/imag per channel to match the reference [c0r, c0i, c1r, c1i, …] channel order.
        let stacked = stacked([spectrum.realPart(), spectrum.imaginaryPart()], axis: -1)
        let interleaved = stacked.reshaped([n, h, wHalf, c * 2])

        let mixed = relu(bn(convLayer(interleaved)))
        let unpacked = mixed.reshaped([n, h, wHalf, c, 2])
        let real = unpacked[0..., 0..., 0..., 0..., 0]
        let imag = unpacked[0..., 0..., 0..., 0..., 1]
        let complex = real.asType(.complex64) + imag.asType(.complex64) * MLXArray(real: 0, imaginary: 1)
        return irfft2(complex, s: [h, w], axes: [1, 2]) * Float(h * w) * scale
    }
}

/// The `convg2g` path: a narrowing convolution → Fourier unit → widening convolution over a residual.
/// The reference keeps the narrowing as a `Sequential` of convolution, normalization, and ReLU, so its
/// keys are `conv1.0` and `conv1.1`; the remap translates those onto these names.
final class NFKLaMaSpectralTransform: Module {
    final class Narrowing: Module {
        @ModuleInfo(key: "conv") var conv: Conv2d
        @ModuleInfo(key: "bn") var bn: BatchNorm

        init(inChannels: Int, outChannels: Int) {
            _conv.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: 1, bias: false)
            _bn.wrappedValue = BatchNorm(featureCount: outChannels)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray { relu(bn(conv(x))) }
    }

    @ModuleInfo(key: "conv1") var conv1: Narrowing
    let fu: NFKLaMaFourierUnit
    @ModuleInfo(key: "conv2") var conv2: Conv2d

    init(inChannels: Int, outChannels: Int) {
        _conv1.wrappedValue = Narrowing(inChannels: inChannels, outChannels: outChannels / 2)
        fu = NFKLaMaFourierUnit(channels: outChannels / 2)
        _conv2.wrappedValue = Conv2d(inputChannels: outChannels / 2, outputChannels: outChannels, kernelSize: 1, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let reduced = conv1(x)
        return conv2(reduced + fu(reduced))
    }
}

/// A Fast Fourier Convolution: local↔local and local↔global spatial convs plus the global↔global
/// spectral transform. `global == 0` collapses to a plain convolution.
final class NFKLaMaFFC: Module {
    @ModuleInfo(key: "convl2l") var convL2L: Conv2d?
    @ModuleInfo(key: "convl2g") var convL2G: Conv2d?
    @ModuleInfo(key: "convg2l") var convG2L: Conv2d?
    @ModuleInfo(key: "convg2g") var convG2G: NFKLaMaSpectralTransform?

    /// The reference gives every spatial convolution `padding_mode='reflect'`, so the border is
    /// mirrored here and the convolutions themselves run at padding zero. A border approximation is
    /// not cosmetic in this network — the same substitution cost style transfer a 0.049 mean pixel
    /// error, because normalization carries it inward from the edge.
    let reflectPad: Int

    init(inLocal: Int, inGlobal: Int, outLocal: Int, outGlobal: Int, kernelSize: Int, stride: Int) {
        let padding = 0
        reflectPad = kernelSize / 2
        if inLocal > 0, outLocal > 0 {
            _convL2L.wrappedValue = Conv2d(inputChannels: inLocal, outputChannels: outLocal, kernelSize: IntOrPair(kernelSize), stride: IntOrPair(stride), padding: IntOrPair(padding), bias: false)
        }
        if inLocal > 0, outGlobal > 0 {
            _convL2G.wrappedValue = Conv2d(inputChannels: inLocal, outputChannels: outGlobal, kernelSize: IntOrPair(kernelSize), stride: IntOrPair(stride), padding: IntOrPair(padding), bias: false)
        }
        if inGlobal > 0, outLocal > 0 {
            _convG2L.wrappedValue = Conv2d(inputChannels: inGlobal, outputChannels: outLocal, kernelSize: IntOrPair(kernelSize), stride: IntOrPair(stride), padding: IntOrPair(padding), bias: false)
        }
        if inGlobal > 0, outGlobal > 0 {
            _convG2G.wrappedValue = NFKLaMaSpectralTransform(inChannels: inGlobal, outChannels: outGlobal)
        }
    }

    func callAsFunction(_ local: MLXArray, _ global: MLXArray?) -> (local: MLXArray?, global: MLXArray?) {
        let paddedLocal = NFKMLXResample.reflectPadded(local, reflectPad)
        let paddedGlobal = global.map { NFKMLXResample.reflectPadded($0, reflectPad) }
        var outLocal: MLXArray?
        var outGlobal: MLXArray?
        if let convL2L { outLocal = convL2L(paddedLocal) }
        if let convG2L, let paddedGlobal { outLocal = (outLocal.map { $0 + convG2L(paddedGlobal) }) ?? convG2L(paddedGlobal) }
        if let convL2G { outGlobal = convL2G(paddedLocal) }
        // The spectral branch takes the unpadded feature: its convolutions are 1×1 and its transform is
        // global, so a mirrored border would be extra signal the reference never sees.
        if let convG2G, let global { outGlobal = (outGlobal.map { $0 + convG2G(global) }) ?? convG2G(global) }
        return (outLocal, outGlobal)
    }
}

/// FFC followed by batch norm and ReLU on each present branch.
final class NFKLaMaFFCBNAct: Module {
    let ffc: NFKLaMaFFC
    @ModuleInfo(key: "bn_l") var bnL: BatchNorm?
    @ModuleInfo(key: "bn_g") var bnG: BatchNorm?

    init(inChannels: Int, outChannels: Int, kernelSize: Int, stride: Int, inRatio: Float, outRatio: Float) {
        let (inLocal, inGlobal) = split(inChannels, ratio: inRatio)
        let (outLocal, outGlobal) = split(outChannels, ratio: outRatio)
        ffc = NFKLaMaFFC(inLocal: inLocal, inGlobal: inGlobal, outLocal: outLocal, outGlobal: outGlobal,
                         kernelSize: kernelSize, stride: stride)
        if outLocal > 0 { _bnL.wrappedValue = BatchNorm(featureCount: outLocal) }
        if outGlobal > 0 { _bnG.wrappedValue = BatchNorm(featureCount: outGlobal) }
    }

    func callAsFunction(_ local: MLXArray, _ global: MLXArray?) -> (local: MLXArray?, global: MLXArray?) {
        let (outLocal, outGlobal) = ffc(local, global)
        let activatedLocal = outLocal.map { array in relu((bnL?.callAsFunction(array)) ?? array) }
        let activatedGlobal = outGlobal.map { array in relu((bnG?.callAsFunction(array)) ?? array) }
        return (activatedLocal, activatedGlobal)
    }
}

/// A residual FFC block: two FFC-BN-ReLU layers, added to the input on both branches.
final class NFKLaMaResnetBlock: Module {
    @ModuleInfo(key: "conv1") var conv1: NFKLaMaFFCBNAct
    @ModuleInfo(key: "conv2") var conv2: NFKLaMaFFCBNAct

    init(channels: Int, ratio: Float) {
        _conv1.wrappedValue = NFKLaMaFFCBNAct(inChannels: channels, outChannels: channels, kernelSize: 3, stride: 1, inRatio: ratio, outRatio: ratio)
        _conv2.wrappedValue = NFKLaMaFFCBNAct(inChannels: channels, outChannels: channels, kernelSize: 3, stride: 1, inRatio: ratio, outRatio: ratio)
    }

    func callAsFunction(_ local: MLXArray, _ global: MLXArray?) -> (local: MLXArray?, global: MLXArray?) {
        let (l1, g1) = conv1(local, global)
        let (l2, g2) = conv2(l1 ?? local, g1)
        let outLocal = (l2 ?? local) + local
        let outGlobal: MLXArray? = {
            guard let g2 else { return global }
            return global.map { $0 + g2 } ?? g2
        }()
        return (outLocal, outGlobal)
    }
}

/// The LaMa FFC-ResNet generator: (masked plate + mask) → inpainted image.
final class NFKMLXLaMaNet: Module {
    @ModuleInfo(key: "init_conv") var initConv: NFKLaMaFFCBNAct
    let downsample: [NFKLaMaFFCBNAct]
    let blocks: [NFKLaMaResnetBlock]
    @ModuleInfo(key: "up") var up: [ConvTransposed2d]
    @ModuleInfo(key: "up_bn") var upBN: [BatchNorm]
    @ModuleInfo(key: "out_conv") var outConv: Conv2d

    let configuration: NFKMLXLaMaConfiguration

    init(_ configuration: NFKMLXLaMaConfiguration) {
        self.configuration = configuration
        let base = configuration.baseChannels
        let ratio = configuration.ratio

        _initConv.wrappedValue = NFKLaMaFFCBNAct(inChannels: 4, outChannels: base, kernelSize: 7, stride: 1, inRatio: 0, outRatio: 0)

        var downs = [NFKLaMaFFCBNAct]()
        for i in 0 ..< configuration.downsampling {
            let inChannels = base * (1 << i)
            let outChannels = base * (1 << (i + 1))
            let outRatio: Float = i == configuration.downsampling - 1 ? ratio : 0
            let inRatio: Float = i == 0 ? 0 : 0                 // global appears only after the last downsample
            downs.append(NFKLaMaFFCBNAct(inChannels: inChannels, outChannels: outChannels, kernelSize: 3, stride: 2, inRatio: inRatio, outRatio: outRatio))
        }
        downsample = downs

        let trunk = base * (1 << configuration.downsampling)
        blocks = (0 ..< configuration.blocks).map { _ in NFKLaMaResnetBlock(channels: trunk, ratio: ratio) }

        var ups = [ConvTransposed2d]()
        var upBNs = [BatchNorm]()
        for i in 0 ..< configuration.downsampling {
            let inChannels = trunk / (1 << i)
            let outChannels = trunk / (1 << (i + 1))
            // `outputPadding` 1 is the reference's: without it a transposed convolution lands one
            // pixel short of doubling, which a resize afterwards can only paper over.
            ups.append(ConvTransposed2d(inputChannels: inChannels, outputChannels: outChannels,
                                        kernelSize: 3, stride: 2, padding: 1, outputPadding: 1))
            upBNs.append(BatchNorm(featureCount: outChannels))
        }
        _up.wrappedValue = ups
        _upBN.wrappedValue = upBNs
        _outConv.wrappedValue = Conv2d(inputChannels: base, outputChannels: 3, kernelSize: 7)
    }

    /// - Parameters:
    ///   - image: the plate `[H, W, 3]` in `0...1`.
    ///   - mask: `[H, W, 1]`, 1 where the region is regenerated.
    /// - Returns: the inpainted image `[H, W, 3]` in `0...1`.
    func inpaint(_ image: MLXArray, mask: MLXArray) -> MLXArray {
        let (height, width) = (image.shape[0], image.shape[1])
        let batchedImage = image.reshaped([1, height, width, image.shape[2]])
        let batchedMask = mask.reshaped([1, height, width, 1])
        let masked = batchedImage * (1 - batchedMask)
        let input = concatenated([masked, batchedMask], axis: 3)   // the FFC layers carry their own padding

        var (local, global) = initConv(input, nil)
        for layer in downsample {
            (local, global) = layer(local ?? input, global)
        }
        for block in blocks {
            (local, global) = block(local!, global)
        }
        var feature = global.map { concatenated([local!, $0], axis: 3) } ?? local!
        for (transpose, norm) in zip(up, upBN) {
            feature = relu(norm(transpose(feature)))
        }
        // The reference reflection-pads by three before the 7×7 output convolution, and its output
        // activation is a sigmoid (`add_out_act: sigmoid` in big-lama's own config).
        let output = sigmoid(outConv(NFKMLXResample.reflectPadded(feature, 3)))

        // Composite: keep the known region from the source, take the masked region from the network.
        let result = batchedMask * output + (1 - batchedMask) * batchedImage
        return result.reshaped([height, width, 3])
    }
}

/// LaMa inpainting as an InferKit backend, and its registration for the Objective-C path.
@objc(NFKMLXLaMa)
public final class NFKMLXLaMa: NSObject {

    @objc public static let modelName = "lama-inpaint"

    static func makeNet(_ configuration: NFKMLXLaMaConfiguration = NFKMLXLaMaConfiguration()) -> NFKMLXLaMaNet {
        NFKMLXLaMaNet(configuration)
    }

    /// Builds a LaMa inpainting backend directly from optional local weights — no registry required.
    /// A nil `weightsURL` builds random weights (`isReady` is true). The backend reads the plate under
    /// `NFKInputImage` and the mask under `NFKInputMask`. Run inference
    /// off the render thread.
    @objc(backendWithWeightsURL:error:)
    public static func backend(weightsURL: URL?) throws -> any NFKInferenceBackend {
        let net = NFKMLXLaMaNet(NFKMLXLaMaConfiguration())
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        let holder = NFKMLXLaMaHolder(net)
        var configuration = NFKMattingConfiguration()
        configuration.plateChannels = 3
        return NFKMLXMattingBackend(identifier: modelName, isReady: true, configuration: configuration) { plate, hint in
            let mask = hint ?? (plate[0..., 0..., 0] * 0).reshaped([plate.shape[0], plate.shape[1], 1])
            let inpainted = holder.net.inpaint(plate, mask: mask)
            let alpha = MLXArray.ones([plate.shape[0], plate.shape[1], 1])
            return concatenated([inpainted, alpha], axis: 2)
        }
    }

    /// Downloads the checkpoint from Hugging Face, then builds — no registry required.
    /// Blocking on the network; run off the render thread.
    @objc(backendWithRepo:weightsPath:revision:cacheDirectoryURL:error:)
    public static func backend(repo: String, weightsPath: String, revision: String?, cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        let url = try NFKMLXDownload.weightsURL(repo: repo, weightsPath: weightsPath, revision: revision, cacheDirectoryURL: cacheDirectoryURL)
        return try backend(weightsURL: url)
    }

    /// The asynchronous form of the download factory: downloads on a background queue, then builds and
    /// delivers the backend (or an error) to `completionHandler`.
    @objc(backendWithRepo:weightsPath:revision:cacheDirectoryURL:completionHandler:)
    public static func backend(repo: String, weightsPath: String, revision: String?, cacheDirectoryURL: URL?,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXDownload.backend(repo: repo, weightsPath: weightsPath, revision: revision,
                               cacheDirectoryURL: cacheDirectoryURL,
                               build: { try backend(weightsURL: $0) },
                               completionHandler: completionHandler)
    }

    /// Registers LaMa (`lama-inpaint`) with `NFKMLXModelRegistry`, delegating to `backend(weightsURL:)`.
    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName) { weightsURL in try backend(weightsURL: weightsURL) }
    }

    /// Maps the reference's flat `model.N` Sequential onto the module's names. The index layout comes
    /// straight from how the reference builds the list: a reflection pad, the init convolution, the
    /// downsampling stages, the residual blocks, a tuple concatenation, then three
    /// (transposed convolution, normalization, activation) triples and the output pad and convolution.
    /// The parameter-free entries still consume an index, which is why the upsampling triples start at
    /// 24 rather than 23.
    static func remapReferenceKey(_ key: String, configuration: NFKMLXLaMaConfiguration) -> String {
        var key = key
        // The spectral branch's narrowing is a Sequential of convolution and normalization. Scoped by
        // the digit that follows, so the residual blocks' own `conv1.ffc…` keys are left alone.
        key = key.replacingOccurrences(of: ".conv1.0.", with: ".conv1.conv.")
        key = key.replacingOccurrences(of: ".conv1.1.", with: ".conv1.bn.")

        guard key.hasPrefix("model.") else { return key }
        let rest = key.dropFirst("model.".count)
        guard let dot = rest.firstIndex(of: "."), let index = Int(rest[..<dot]) else { return key }
        let tail = String(rest[rest.index(after: dot)...])

        let downsampling = configuration.downsampling
        let firstDown = 2
        let firstBlock = firstDown + downsampling
        let firstUp = firstBlock + configuration.blocks + 1                 // + the tuple concatenation
        let finalConv = firstUp + 3 * downsampling + 1                      // + the output reflection pad

        if index == 1 {
            return "init_conv." + tail
        }
        if index >= firstDown, index < firstBlock {
            return "downsample.\(index - firstDown)." + tail
        }
        if index >= firstBlock, index < firstBlock + configuration.blocks {
            return "blocks.\(index - firstBlock)." + tail
        }
        if index >= firstUp, index < firstUp + 3 * downsampling {
            let offset = index - firstUp
            let stage = offset / 3
            return offset % 3 == 0 ? "up.\(stage)." + tail : "up_bn.\(stage)." + tail
        }
        if index == finalConv {
            return "out_conv." + tail
        }
        return key
    }

    /// Loads a safetensors checkpoint, translating the reference's flat `model.N.*` Sequential names
    /// and transposing 4-D convolution weights to MLX's layout. `remap` overrides the built-in
    /// translation for a checkpoint that stores its weights some other way.
    static func loadWeights(into net: NFKMLXLaMaNet, from url: URL, remap: ((String) -> String)? = nil) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let raw = checkpoint.arrays
        let mapped = raw.map { key, value -> (String, MLXArray) in
            // The raw Lightning checkpoint prefixes the generator's keys; the offline converter
            // strips the prefix, so a converted file never carries it.
            var reference = key
            if reference.hasPrefix("generator.") {
                reference = String(reference.dropFirst("generator.".count))
            }
            let name = remap?(key) ?? remapReferenceKey(reference, configuration: net.configuration)
            // A transposed convolution stores `[in, out, kH, kW]`, so it takes a different axis order
            // from the forward convolutions' `[out, in, kH, kW]`.
            if checkpoint.needsConvTranspose, value.ndim == 4 {
                return (name, name.hasPrefix("up.") ? value.transposed(1, 2, 3, 0) : value.transposed(0, 2, 3, 1))
            }
            return (name, value)
        }
        try NFKMLXWeights.apply(mapped, to: net)
    }
}

private final class NFKMLXLaMaHolder: @unchecked Sendable {
    let net: NFKMLXLaMaNet
    init(_ net: NFKMLXLaMaNet) { self.net = net }
}
