//
//  NFKMLXColorizer.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXNN

// Colorful Image Colorization (Zhang et al., ECCV 2016) predicts color for a grayscale photo. The
// model works in CIELAB space: the network sees only the lightness channel L and predicts, per output
// location, a distribution over 313 quantized ab chroma bins; the annealed mean of that distribution
// (a softmax followed by a fixed 1×1 convolution whose weights encode the bin centers, stored in the
// checkpoint as `model_out`) yields the ab channels. The predicted ab recombines with the original
// full-resolution L, so luminance detail is preserved exactly and only chroma is synthesized.
//
// The network is a VGG-style stack of eight convolution blocks, each closed by a BatchNorm; blocks 5
// and 6 use dilation 2. The module names here are flat per-layer names (`conv1_1`, `norm1`,
// `deconv8_1`, `out_ab`); the reference checkpoint stores `nn.Sequential` indices (`model1.0`,
// `model8.6`, `model_out`), and `Tools/colorizer-to-safetensors/convert.py` performs that exact,
// deterministic rename (including the ConvTranspose axis swap), so a converted checkpoint loads
// directly. L is normalized as (L − 50) / 100 on input; predicted ab is scaled by 110. Tensors flow in
// NHWC. Resampling is nearest (bilinear is a sweep item, as for depth).

/// sRGB ↔ CIELAB conversion on `[H, W, 3]` tensors, D65 white point. Values: sRGB in `0...1`, L in
/// `0...100`, ab roughly `-110...110`.
enum NFKLabColor {

    private static let epsilon: Float = 0.008856452         // (6/29)³
    private static let kappaInv: Float = 0.12841855         // 3·(6/29)²
    private static let offset: Float = 4.0 / 29.0
    private static let white: [Float] = [0.95047, 1.0, 1.08883]

    private static func matrix(_ values: [Float]) -> MLXArray {
        values.withUnsafeBufferPointer { MLXArray($0, [3, 3]) }
    }

    // Rows produce (X, Y, Z) from linear (R, G, B); the transpose right-multiplies pixel rows.
    private static let rgbToXYZ = matrix([0.4124564, 0.3575761, 0.1804375,
                                          0.2126729, 0.7151522, 0.0721750,
                                          0.0193339, 0.1191920, 0.9503041]).transposed(1, 0)
    private static let xyzToRGB = matrix([3.2404542, -1.5371385, -0.4985314,
                                          -0.9692660, 1.8760108, 0.0415560,
                                          0.0556434, -0.2040259, 1.0572252]).transposed(1, 0)
    private static let whitePoint = white.withUnsafeBufferPointer { MLXArray($0, [1, 3]) }

    /// Converts sRGB `[H, W, 3]` (`0...1`) to Lab `[H, W, 3]`.
    static func toLab(_ rgb: MLXArray) -> MLXArray {
        let (height, width) = (rgb.shape[0], rgb.shape[1])
        let srgb = rgb.reshaped([height * width, 3])

        let linear = which(srgb .> 0.04045, pow((srgb + 0.055) / 1.055, 2.4), srgb / 12.92)
        let xyz = linear.matmul(rgbToXYZ) / whitePoint

        // f(t) = t^(1/3) above the CIE knee, a matched linear segment below it.
        let f = which(xyz .> epsilon, pow(xyz, 1.0 / 3.0), xyz / kappaInv + offset)
        let fx = f[0..., 0 ..< 1]
        let fy = f[0..., 1 ..< 2]
        let fz = f[0..., 2 ..< 3]

        let l = 116.0 * fy - 16.0
        let a = 500.0 * (fx - fy)
        let b = 200.0 * (fy - fz)
        return concatenated([l, a, b], axis: 1).reshaped([height, width, 3])
    }

    /// Converts Lab `[H, W, 3]` to sRGB `[H, W, 3]`, clipped to `0...1`. Out-of-gamut chroma clips at
    /// the linear-RGB stage.
    static func toRGB(_ lab: MLXArray) -> MLXArray {
        let (height, width) = (lab.shape[0], lab.shape[1])
        let flat = lab.reshaped([height * width, 3])
        let l = flat[0..., 0 ..< 1]
        let a = flat[0..., 1 ..< 2]
        let b = flat[0..., 2 ..< 3]

        let fy = (l + 16.0) / 116.0
        let fx = fy + a / 500.0
        let fz = fy - b / 200.0
        let f = concatenated([fx, fy, fz], axis: 1)

        let delta: Float = 6.0 / 29.0
        let xyz = which(f .> delta, f * f * f, kappaInv * (f - offset)) * whitePoint

        let linear = clip(xyz.matmul(xyzToRGB), min: 0, max: 1)
        let srgb = which(linear .> 0.0031308, 1.055 * pow(linear, 1.0 / 2.4) - 0.055, 12.92 * linear)
        return clip(srgb, min: 0, max: 1).reshaped([height, width, 3])
    }
}

/// Colorizer dimensions. `eccv16` matches the reference checkpoint; `tiny` keeps tests fast.
public struct NFKMLXColorizerConfiguration: Sendable {
    public var resolution: Int
    public var channels: [Int]
    public var binCount: Int

    public init(resolution: Int = 256, channels: [Int] = [64, 128, 256, 512], binCount: Int = 313) {
        self.resolution = resolution
        self.channels = channels
        self.binCount = binCount
    }

    public static let eccv16 = NFKMLXColorizerConfiguration()

    public static let tiny = NFKMLXColorizerConfiguration(resolution: 16, channels: [4, 8, 8, 8], binCount: 7)
}

/// The eccv16 colorization network: eight convolution blocks over the L channel, a 313-bin chroma
/// classifier, and the annealed-mean readout convolution.
final class NFKMLXColorizerNet: Module {

    @ModuleInfo(key: "conv1_1") var conv1_1: Conv2d
    @ModuleInfo(key: "conv1_2") var conv1_2: Conv2d
    @ModuleInfo(key: "norm1") var norm1: BatchNorm

    @ModuleInfo(key: "conv2_1") var conv2_1: Conv2d
    @ModuleInfo(key: "conv2_2") var conv2_2: Conv2d
    @ModuleInfo(key: "norm2") var norm2: BatchNorm

    @ModuleInfo(key: "conv3_1") var conv3_1: Conv2d
    @ModuleInfo(key: "conv3_2") var conv3_2: Conv2d
    @ModuleInfo(key: "conv3_3") var conv3_3: Conv2d
    @ModuleInfo(key: "norm3") var norm3: BatchNorm

    @ModuleInfo(key: "conv4_1") var conv4_1: Conv2d
    @ModuleInfo(key: "conv4_2") var conv4_2: Conv2d
    @ModuleInfo(key: "conv4_3") var conv4_3: Conv2d
    @ModuleInfo(key: "norm4") var norm4: BatchNorm

    @ModuleInfo(key: "conv5_1") var conv5_1: Conv2d
    @ModuleInfo(key: "conv5_2") var conv5_2: Conv2d
    @ModuleInfo(key: "conv5_3") var conv5_3: Conv2d
    @ModuleInfo(key: "norm5") var norm5: BatchNorm

    @ModuleInfo(key: "conv6_1") var conv6_1: Conv2d
    @ModuleInfo(key: "conv6_2") var conv6_2: Conv2d
    @ModuleInfo(key: "conv6_3") var conv6_3: Conv2d
    @ModuleInfo(key: "norm6") var norm6: BatchNorm

    @ModuleInfo(key: "conv7_1") var conv7_1: Conv2d
    @ModuleInfo(key: "conv7_2") var conv7_2: Conv2d
    @ModuleInfo(key: "conv7_3") var conv7_3: Conv2d
    @ModuleInfo(key: "norm7") var norm7: BatchNorm

    @ModuleInfo(key: "deconv8_1") var deconv8_1: ConvTransposed2d
    @ModuleInfo(key: "conv8_2") var conv8_2: Conv2d
    @ModuleInfo(key: "conv8_3") var conv8_3: Conv2d
    @ModuleInfo(key: "conv8_313") var conv8_313: Conv2d

    @ModuleInfo(key: "out_ab") var outAb: Conv2d

    let configuration: NFKMLXColorizerConfiguration

    init(_ c: NFKMLXColorizerConfiguration) {
        configuration = c
        let (c1, c2, c3, c4) = (c.channels[0], c.channels[1], c.channels[2], c.channels[3])

        _conv1_1.wrappedValue = Conv2d(inputChannels: 1, outputChannels: c1, kernelSize: 3, padding: 1)
        _conv1_2.wrappedValue = Conv2d(inputChannels: c1, outputChannels: c1, kernelSize: 3, stride: 2, padding: 1)
        _norm1.wrappedValue = BatchNorm(featureCount: c1)

        _conv2_1.wrappedValue = Conv2d(inputChannels: c1, outputChannels: c2, kernelSize: 3, padding: 1)
        _conv2_2.wrappedValue = Conv2d(inputChannels: c2, outputChannels: c2, kernelSize: 3, stride: 2, padding: 1)
        _norm2.wrappedValue = BatchNorm(featureCount: c2)

        _conv3_1.wrappedValue = Conv2d(inputChannels: c2, outputChannels: c3, kernelSize: 3, padding: 1)
        _conv3_2.wrappedValue = Conv2d(inputChannels: c3, outputChannels: c3, kernelSize: 3, padding: 1)
        _conv3_3.wrappedValue = Conv2d(inputChannels: c3, outputChannels: c3, kernelSize: 3, stride: 2, padding: 1)
        _norm3.wrappedValue = BatchNorm(featureCount: c3)

        _conv4_1.wrappedValue = Conv2d(inputChannels: c3, outputChannels: c4, kernelSize: 3, padding: 1)
        _conv4_2.wrappedValue = Conv2d(inputChannels: c4, outputChannels: c4, kernelSize: 3, padding: 1)
        _conv4_3.wrappedValue = Conv2d(inputChannels: c4, outputChannels: c4, kernelSize: 3, padding: 1)
        _norm4.wrappedValue = BatchNorm(featureCount: c4)

        _conv5_1.wrappedValue = Conv2d(inputChannels: c4, outputChannels: c4, kernelSize: 3, padding: IntOrPair(2), dilation: IntOrPair(2))
        _conv5_2.wrappedValue = Conv2d(inputChannels: c4, outputChannels: c4, kernelSize: 3, padding: IntOrPair(2), dilation: IntOrPair(2))
        _conv5_3.wrappedValue = Conv2d(inputChannels: c4, outputChannels: c4, kernelSize: 3, padding: IntOrPair(2), dilation: IntOrPair(2))
        _norm5.wrappedValue = BatchNorm(featureCount: c4)

        _conv6_1.wrappedValue = Conv2d(inputChannels: c4, outputChannels: c4, kernelSize: 3, padding: IntOrPair(2), dilation: IntOrPair(2))
        _conv6_2.wrappedValue = Conv2d(inputChannels: c4, outputChannels: c4, kernelSize: 3, padding: IntOrPair(2), dilation: IntOrPair(2))
        _conv6_3.wrappedValue = Conv2d(inputChannels: c4, outputChannels: c4, kernelSize: 3, padding: IntOrPair(2), dilation: IntOrPair(2))
        _norm6.wrappedValue = BatchNorm(featureCount: c4)

        _conv7_1.wrappedValue = Conv2d(inputChannels: c4, outputChannels: c4, kernelSize: 3, padding: 1)
        _conv7_2.wrappedValue = Conv2d(inputChannels: c4, outputChannels: c4, kernelSize: 3, padding: 1)
        _conv7_3.wrappedValue = Conv2d(inputChannels: c4, outputChannels: c4, kernelSize: 3, padding: 1)
        _norm7.wrappedValue = BatchNorm(featureCount: c4)

        _deconv8_1.wrappedValue = ConvTransposed2d(inputChannels: c4, outputChannels: c3, kernelSize: 4, stride: 2, padding: 1)
        _conv8_2.wrappedValue = Conv2d(inputChannels: c3, outputChannels: c3, kernelSize: 3, padding: 1)
        _conv8_3.wrappedValue = Conv2d(inputChannels: c3, outputChannels: c3, kernelSize: 3, padding: 1)
        _conv8_313.wrappedValue = Conv2d(inputChannels: c3, outputChannels: c.binCount, kernelSize: 1)

        _outAb.wrappedValue = Conv2d(inputChannels: c.binCount, outputChannels: 2, kernelSize: 1, bias: false)
    }

    /// Runs the eight blocks over a normalized L map `[1, res, res, 1]`, returning chroma-bin logits
    /// `[1, res/4, res/4, binCount]`.
    func logits(_ l: MLXArray) -> MLXArray {
        var x = norm1(relu(conv1_2(relu(conv1_1(l)))))
        x = norm2(relu(conv2_2(relu(conv2_1(x)))))
        x = norm3(relu(conv3_3(relu(conv3_2(relu(conv3_1(x)))))))
        x = norm4(relu(conv4_3(relu(conv4_2(relu(conv4_1(x)))))))
        x = norm5(relu(conv5_3(relu(conv5_2(relu(conv5_1(x)))))))
        x = norm6(relu(conv6_3(relu(conv6_2(relu(conv6_1(x)))))))
        x = norm7(relu(conv7_3(relu(conv7_2(relu(conv7_1(x)))))))
        x = relu(conv8_3(relu(conv8_2(relu(deconv8_1(x))))))
        return conv8_313(x)
    }

    /// Colorizes a bridged image `[H, W, 3]` (`0...1`): the original L drives the network at the model
    /// resolution, the annealed-mean ab comes back to full resolution, and the pair converts to sRGB.
    func colorize(_ image: MLXArray) -> MLXArray {
        let lightness = NFKLabColor.toLab(image)[0..., 0..., 0 ..< 1]    // [H, W, 1], 0...100
        return NFKLabColor.toRGB(concatenated([lightness, ab(from: lightness)], axis: 2))
    }

    /// The network's ab prediction for a bridged image, at the input resolution and on the reference's
    /// ±110 scale — the colorized result before the original lightness is put back.
    func abPrediction(_ image: MLXArray) -> MLXArray {
        ab(from: NFKLabColor.toLab(image)[0..., 0..., 0 ..< 1])
    }

    private func ab(from lightness: MLXArray) -> MLXArray {
        let (height, width) = (lightness.shape[0], lightness.shape[1])
        let batched = lightness.reshaped([1, height, width, 1])
        let small = NFKMLXResample.resizeBilinear(batched, height: configuration.resolution, width: configuration.resolution)
        let distribution = softmax(logits((small - 50.0) / 100.0), axis: -1)
        let scaled = outAb(distribution) * 110.0                 // annealed mean over the bin centers
        // The reference brings ab back with a bilinear `nn.Upsample`; nearest neighbour costs a
        // quarter of the ab agreement (cosine 0.96 against the reference, versus 0.9999999998).
        return NFKMLXResample.resizeBilinear(scaled, height: height, width: width).reshaped([height, width, 2])
    }
}

/// Holds the network for capture in the backend's `@Sendable` forward closure.
private final class NFKColorizerHolder: @unchecked Sendable {
    let net: NFKMLXColorizerNet
    init(_ net: NFKMLXColorizerNet) { self.net = net }
}

/// ECCV-16 colorization as an InferKit backend, and its registration for the Objective-C path.
///
/// `NFKMLXColorizerNet` is the real Zhang et al. model. Random weights run (proving the pipeline); a
/// trained checkpoint colorizes plausibly. The 313 ab bin centers ship inside the checkpoint as the
/// `model_out` convolution, so no separate cluster file is needed. Convert the release with
/// `Tools/colorizer-to-safetensors/convert.py`, which renames the `nn.Sequential` keys and swaps the
/// ConvTranspose axes; the loader then transposes 4-D weights `[out, in, kH, kW]` to MLX's
/// `[out, kH, kW, in]`.
@objc(NFKMLXColorizer)
public final class NFKMLXColorizer: NSObject {

    /// The registry name the model builds under.
    @objc public static let modelName = "colorizer-eccv16"

    /// Builds a colorization backend directly from optional local weights — no registry required. A
    /// nil `weightsURL` builds random weights (`isReady` is true). Run
    /// inference off the render thread.
    static func makeNet(_ configuration: NFKMLXColorizerConfiguration = .eccv16) -> NFKMLXColorizerNet {
        let net = NFKMLXColorizerNet(configuration)
        // The checkpoint's BatchNorm running statistics only apply in evaluation mode.
        net.train(false)
        return net
    }

    @objc(backendWithWeightsURL:error:)
    public static func backend(weightsURL: URL?) throws -> any NFKInferenceBackend {
        let net = makeNet()
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        let holder = NFKColorizerHolder(net)
        return NFKMLXModuleBackend(identifier: modelName, isReady: true) { image in holder.net.colorize(image) }
    }

    /// Downloads the checkpoint from Hugging Face, then builds the backend — no registry required.
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

    /// Registers the colorizer (`colorizer-eccv16`) with `NFKMLXModelRegistry`, delegating to
    /// `backend(weightsURL:)`.
    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName) { weightsURL in
            try backend(weightsURL: weightsURL)
        }
    }

    /// Loads a safetensors checkpoint into `net`, transposing 4-D convolution weights from the on-disk
    /// `[out, in, kH, kW]` to MLX's channels-last `[out, kH, kW, in]`. The converter has already
    /// swapped ConvTranspose weights into the same on-disk layout, so one transpose covers both.
    static func loadWeights(into net: NFKMLXColorizerNet, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let raw = checkpoint.arrays
        let mapped = raw.compactMap { key, value -> (String, MLXArray)? in
            guard !key.hasSuffix("num_batches_tracked") else { return nil }
            let name = remapReferenceKey(key)
            guard checkpoint.needsConvTranspose, value.ndim == 4 else { return (name, value) }
            // The reference stores its one transposed convolution as `[in, out, kH, kW]`; a
            // converted file already carries it permuted to the forward layout, so only a raw
            // checkpoint takes the transposed-convolution order.
            let isRawTransposedConv = name == "deconv8_1.weight" && checkpoint.isNativeTorch
            return (name, isRawTransposedConv ? value.transposed(1, 2, 3, 0) : value.transposed(0, 2, 3, 1))
        }
        try NFKMLXWeights.apply(mapped, to: net)
    }

    /// The reference `nn.Sequential` slots per block, in the order the reference lists them.
    private static let referenceBlocks: [String: [Int: String]] = [
        "model1": [0: "conv1_1", 2: "conv1_2", 4: "norm1"],
        "model2": [0: "conv2_1", 2: "conv2_2", 4: "norm2"],
        "model3": [0: "conv3_1", 2: "conv3_2", 4: "conv3_3", 6: "norm3"],
        "model4": [0: "conv4_1", 2: "conv4_2", 4: "conv4_3", 6: "norm4"],
        "model5": [0: "conv5_1", 2: "conv5_2", 4: "conv5_3", 6: "norm5"],
        "model6": [0: "conv6_1", 2: "conv6_2", 4: "conv6_3", 6: "norm6"],
        "model7": [0: "conv7_1", 2: "conv7_2", 4: "conv7_3", 6: "norm7"],
        "model8": [0: "deconv8_1", 2: "conv8_2", 4: "conv8_3", 6: "conv8_313"],
    ]

    /// Translates a reference key (`model1.0.weight` → `conv1_1.weight`, `model_out.weight` →
    /// `out_ab.weight`) — the rename `Tools/colorizer-to-safetensors` applies offline — so the raw
    /// eccv16 release loads directly. Any other key, a converted file's included, passes through
    /// unchanged.
    static func remapReferenceKey(_ key: String) -> String {
        if key == "model_out.weight" {
            return "out_ab.weight"
        }
        let parts = key.split(separator: ".").map(String.init)
        guard parts.count >= 3, let block = referenceBlocks[parts[0]],
              let slot = Int(parts[1]), let name = block[slot] else { return key }
        return ([name] + parts[2...]).joined(separator: ".")
    }
}

// MARK: - siggraph17 (user-hint colorization)
//
// The second released colorizer is a separate network from eccv16, not a configuration of it: a
// U-shaped stack of sixteen blocks with three shortcut paths, a four-channel input (the lightness, an
// ab hint, and a mask marking where that hint applies), and a head that regresses ab directly instead
// of classifying it into quantized bins. With an empty hint it colorizes automatically; with one it
// follows the user's strokes.

/// One VGG-style block: convolutions with ReLU between them, optionally ending in a normalization.
/// The reference stores each as a numbered `Sequential`, so `remapReferenceKey` translates positions.
final class NFKSiggraphBlock: Module {
    @ModuleInfo(key: "convs") var convs: [Conv2d]
    @ModuleInfo(key: "norm") var norm: BatchNorm?
    /// The reference opens the decoder blocks with a ReLU, before their first convolution.
    let leadsWithReLU: Bool
    /// The last block finishes on a leaky ReLU rather than a normalization.
    let endsWithLeakyReLU: Bool

    init(channels: [(Int, Int)], dilation: Int = 1, normalized: Bool = true,
         leadsWithReLU: Bool = false, endsWithLeakyReLU: Bool = false) {
        _convs.wrappedValue = channels.map { input, output in
            Conv2d(inputChannels: input, outputChannels: output, kernelSize: 3,
                   padding: IntOrPair(dilation), dilation: IntOrPair(dilation))
        }
        if normalized {
            _norm.wrappedValue = BatchNorm(featureCount: channels.last!.1)
        }
        self.leadsWithReLU = leadsWithReLU
        self.endsWithLeakyReLU = endsWithLeakyReLU
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = leadsWithReLU ? relu(x) : x
        for (index, conv) in convs.enumerated() {
            out = conv(out)
            if index < convs.count - 1 {
                out = relu(out)
            }
        }
        if endsWithLeakyReLU {
            return leakyRelu(out, negativeSlope: 0.2)
        }
        return norm.map { $0(relu(out)) } ?? out
    }
}

/// The siggraph17 generator: seven encoder blocks, three upsampling stages each fused with a
/// shortcut from the matching encoder depth, and the ab regression head.
final class NFKMLXSiggraphNet: Module {
    @ModuleInfo(key: "model1") var model1: NFKSiggraphBlock
    @ModuleInfo(key: "model2") var model2: NFKSiggraphBlock
    @ModuleInfo(key: "model3") var model3: NFKSiggraphBlock
    @ModuleInfo(key: "model4") var model4: NFKSiggraphBlock
    @ModuleInfo(key: "model5") var model5: NFKSiggraphBlock
    @ModuleInfo(key: "model6") var model6: NFKSiggraphBlock
    @ModuleInfo(key: "model7") var model7: NFKSiggraphBlock
    @ModuleInfo(key: "model8up") var model8up: ConvTransposed2d
    @ModuleInfo(key: "model3short8") var model3short8: Conv2d
    @ModuleInfo(key: "model8") var model8: NFKSiggraphBlock
    @ModuleInfo(key: "model9up") var model9up: ConvTransposed2d
    @ModuleInfo(key: "model2short9") var model2short9: Conv2d
    @ModuleInfo(key: "model9") var model9: NFKSiggraphBlock
    @ModuleInfo(key: "model10up") var model10up: ConvTransposed2d
    @ModuleInfo(key: "model1short10") var model1short10: Conv2d
    @ModuleInfo(key: "model10") var model10: NFKSiggraphBlock
    @ModuleInfo(key: "model_out") var modelOut: Conv2d

    override init() {
        _model1.wrappedValue = NFKSiggraphBlock(channels: [(4, 64), (64, 64)])
        _model2.wrappedValue = NFKSiggraphBlock(channels: [(64, 128), (128, 128)])
        _model3.wrappedValue = NFKSiggraphBlock(channels: [(128, 256), (256, 256), (256, 256)])
        _model4.wrappedValue = NFKSiggraphBlock(channels: [(256, 512), (512, 512), (512, 512)])
        // Blocks five and six dilate rather than downsample, holding the resolution while the
        // receptive field keeps growing.
        _model5.wrappedValue = NFKSiggraphBlock(channels: [(512, 512), (512, 512), (512, 512)], dilation: 2)
        _model6.wrappedValue = NFKSiggraphBlock(channels: [(512, 512), (512, 512), (512, 512)], dilation: 2)
        _model7.wrappedValue = NFKSiggraphBlock(channels: [(512, 512), (512, 512), (512, 512)])

        _model8up.wrappedValue = ConvTransposed2d(inputChannels: 512, outputChannels: 256,
                                                  kernelSize: 4, stride: 2, padding: 1)
        _model3short8.wrappedValue = Conv2d(inputChannels: 256, outputChannels: 256, kernelSize: 3, padding: 1)
        _model8.wrappedValue = NFKSiggraphBlock(channels: [(256, 256), (256, 256)], leadsWithReLU: true)

        _model9up.wrappedValue = ConvTransposed2d(inputChannels: 256, outputChannels: 128,
                                                  kernelSize: 4, stride: 2, padding: 1)
        _model2short9.wrappedValue = Conv2d(inputChannels: 128, outputChannels: 128, kernelSize: 3, padding: 1)
        _model9.wrappedValue = NFKSiggraphBlock(channels: [(128, 128)], leadsWithReLU: true)

        _model10up.wrappedValue = ConvTransposed2d(inputChannels: 128, outputChannels: 128,
                                                   kernelSize: 4, stride: 2, padding: 1)
        _model1short10.wrappedValue = Conv2d(inputChannels: 64, outputChannels: 128, kernelSize: 3, padding: 1)
        _model10.wrappedValue = NFKSiggraphBlock(channels: [(128, 128)], normalized: false,
                                                 leadsWithReLU: true, endsWithLeakyReLU: true)
        _modelOut.wrappedValue = Conv2d(inputChannels: 128, outputChannels: 2, kernelSize: 1)
    }

    /// The reference downsamples by taking every other pixel rather than pooling.
    private static func subsampled(_ x: MLXArray) -> MLXArray {
        x[0..., .stride(by: 2), .stride(by: 2), 0...]
    }

    /// Colorizes a bridged image `[H, W, 3]` (`0...1`), preserving its lightness exactly and taking
    /// only the ab channels from the network — the same contract as the eccv16 colorizer.
    func colorize(_ image: MLXArray) -> MLXArray {
        let (height, width) = (image.shape[0], image.shape[1])
        // The Lab conversion works on an unbatched `[H, W, 3]`; the network takes a batch.
        let lab = NFKLabColor.toLab(image)
        let lightness = lab[0..., 0..., 0 ..< 1]
        let ab = predictAB(lightness: lightness.reshaped([1, height, width, 1]))
        let combined = concatenated([lightness, ab.reshaped([height, width, 2])], axis: 2)
        return NFKLabColor.toRGB(combined)
    }

    /// Predicted ab for a lightness map `[1, H, W, 1]` in CIELAB, with an optional hint. `hint` is
    /// `[1, H, W, 2]` of ab values and `mask` `[1, H, W, 1]` marking where the hint applies; passing
    /// nil for both colorizes automatically, as the reference does.
    func predictAB(lightness: MLXArray, hint: MLXArray? = nil, mask: MLXArray? = nil) -> MLXArray {
        let shape = lightness.shape
        let hintValue = hint ?? MLXArray.zeros([shape[0], shape[1], shape[2], 2])
        let maskValue = mask ?? MLXArray.zeros([shape[0], shape[1], shape[2], 1])
        // The reference normalizes lightness by (L − 50)/100 and the ab hint by 110.
        let input = concatenated([(lightness - 50) / 100, hintValue / 110, maskValue], axis: 3)

        let conv1 = model1(input)
        let conv2 = model2(Self.subsampled(conv1))
        let conv3 = model3(Self.subsampled(conv2))
        let conv4 = model4(Self.subsampled(conv3))
        let conv7 = model7(model6(model5(conv4)))

        let up8 = model8up(conv7) + model3short8(conv3)
        let conv8 = model8(up8)
        let up9 = model9up(conv8) + model2short9(conv2)
        let conv9 = model9(up9)
        let up10 = model10up(conv9) + model1short10(conv1)
        let conv10 = model10(up10)
        // The head is a tanh, and the reference scales its output back by 110.
        return tanh(modelOut(conv10)) * 110
    }
}

/// Holds the network for capture in the backend's `@Sendable` closure.
private final class NFKSiggraphHolder: @unchecked Sendable {
    let net: NFKMLXSiggraphNet
    init(_ net: NFKMLXSiggraphNet) { self.net = net }
}

/// The siggraph17 user-hint colorizer as an InferKit backend.
///
/// A grayscale or colour image under `NFKInputImage` is colorized; the lightness is preserved exactly
/// and only the ab channels are predicted, as in `NFKMLXColorizer`.
@objc(NFKMLXSiggraphColorizer)
public final class NFKMLXSiggraphColorizer: NSObject {

    /// The registry name the model builds under.
    @objc public static let modelName = "colorizer-siggraph17"

    static func makeNet() -> NFKMLXSiggraphNet { NFKMLXSiggraphNet() }

    /// Builds a colorization backend directly from optional local weights — no registry required.
    /// A nil `weightsURL` builds random weights (`isReady` is true).
    @objc(backendWithWeightsURL:error:)
    public static func backend(weightsURL: URL?) throws -> any NFKInferenceBackend {
        let net = NFKMLXSiggraphNet()
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        net.train(false)
        let holder = NFKSiggraphHolder(net)
        return NFKMLXModuleBackend(identifier: modelName, isReady: true) { image in
            holder.net.colorize(image)
        }
    }

    /// Downloads the checkpoint, then builds the backend. Blocking on the network; run off the render
    /// thread.
    @objc(backendWithRepo:weightsPath:revision:cacheDirectoryURL:error:)
    public static func backend(repo: String, weightsPath: String, revision: String?, cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        let url = try NFKMLXDownload.weightsURL(repo: repo, weightsPath: weightsPath, revision: revision, cacheDirectoryURL: cacheDirectoryURL)
        return try backend(weightsURL: url)
    }

    /// The asynchronous form of the download factory.
    @objc(backendWithRepo:weightsPath:revision:cacheDirectoryURL:completionHandler:)
    public static func backend(repo: String, weightsPath: String, revision: String?, cacheDirectoryURL: URL?,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXDownload.backend(repo: repo, weightsPath: weightsPath, revision: revision,
                               cacheDirectoryURL: cacheDirectoryURL,
                               build: { try backend(weightsURL: $0) },
                               completionHandler: completionHandler)
    }

    /// Registers the model (`colorizer-siggraph17`) with `NFKMLXModelRegistry`.
    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName) { weightsURL in
            try backend(weightsURL: weightsURL)
        }
    }

    /// Maps the reference's numbered `Sequential` slots onto the module's names. Which slot holds what
    /// depends on the block: the encoder blocks open with a convolution, the decoder blocks open with
    /// a ReLU, and only some end in a normalization — so the convolution indices are counted per block
    /// rather than assumed.
    static func remapReferenceKey(_ key: String) -> String {
        // Convolution slots per block, in the order the reference lists them.
        let convolutionSlots: [String: [Int]] = [
            "model1": [0, 2], "model2": [0, 2], "model3": [0, 2, 4], "model4": [0, 2, 4],
            "model5": [0, 2, 4], "model6": [0, 2, 4], "model7": [0, 2, 4],
            "model8": [1, 3], "model9": [1], "model10": [1],
        ]
        let normalizationSlot: [String: Int] = [
            "model1": 4, "model2": 4, "model3": 6, "model4": 6,
            "model5": 6, "model6": 6, "model7": 6, "model8": 5, "model9": 3,
        ]
        guard let dot = key.firstIndex(of: ".") else { return key }
        let block = String(key[..<dot])
        let rest = key[key.index(after: dot)...]
        guard let slotEnd = rest.firstIndex(of: "."), let slot = Int(rest[..<slotEnd]) else { return key }
        let tail = String(rest[rest.index(after: slotEnd)...])

        if let slots = convolutionSlots[block], let index = slots.firstIndex(of: slot) {
            return "\(block).convs.\(index).\(tail)"
        }
        if normalizationSlot[block] == slot {
            return "\(block).norm.\(tail)"
        }
        // The shortcuts, upsamplers, and the output head hold a single layer at slot zero.
        return slot == 0 ? "\(block).\(tail)" : key
    }

    /// Loads a safetensors checkpoint, remapping the reference's names and transposing 4-D weights.
    /// The upsamplers are transposed convolutions and take the other axis order.
    static func loadWeights(into net: NFKMLXSiggraphNet, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let raw = checkpoint.arrays
        let mapped = raw.compactMap { key, value -> (String, MLXArray)? in
            // The 529-class auxiliary head supervises training and inference never reads it.
            guard !key.hasPrefix("model_class") else { return nil }
            let name = remapReferenceKey(key)
            if checkpoint.needsConvTranspose, value.ndim == 4 {
                let isTransposed = name.hasSuffix("up.weight")
                return (name, isTransposed ? value.transposed(1, 2, 3, 0) : value.transposed(0, 2, 3, 1))
            }
            return (name, value)
        }
        try NFKMLXWeights.apply(mapped, to: net)
    }
}
