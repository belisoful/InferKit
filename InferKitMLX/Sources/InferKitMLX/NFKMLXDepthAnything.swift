//
//  NFKMLXDepthAnything.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXNN

// Depth Anything V2 is a DINOv2 ViT encoder with a DPT (dense prediction transformer) head. It runs as
// a single forward through `NFKMLXModuleBackend`: an RGB image in, a grayscale depth map out. The
// module structure and parameter names mirror the reference PyTorch model
// (`depth-anything/Depth-Anything-V2-Small`): the encoder under `pretrained.*`, the head under
// `depth_head.*`. Tensors flow NHWC (MLX's channels-last layout).
//
// The encoder processes a fixed 518×518 (37×37 patches of 14), so `pos_embed` matches without
// interpolation; the depth map resizes back to the request's image size. `NFKMLXDepthConfiguration`
// carries the ViT-Small dimensions; the Base/Large variants change `embedDimensions`, `depth`,
// `heads`, and the DPT channel widths.

/// The ViT + DPT dimensions. Defaults are Depth Anything V2 **Small**.
public struct NFKMLXDepthConfiguration: Sendable {
    public var patchSize: Int = 14
    public var inputSize: Int = 518                             // 37 × 14
    public var embedDimensions: Int = 384
    public var depth: Int = 12
    public var heads: Int = 6
    public var mlpRatio: Int = 4
    /// The four encoder blocks whose outputs feed the DPT head.
    public var hooks: [Int] = [2, 5, 8, 11]
    /// The DPT fusion width.
    public var features: Int = 64
    /// The per-hook reassemble widths (fine → coarse).
    public var outChannels: [Int] = [48, 96, 192, 384]

    public init() {}

    /// ViT-Small (the default): 384-d, 12 blocks, 6 heads.
    public static var small: NFKMLXDepthConfiguration { NFKMLXDepthConfiguration() }

    /// ViT-Base: 768-d, 12 blocks, 12 heads, wider DPT head.
    public static var base: NFKMLXDepthConfiguration {
        var configuration = NFKMLXDepthConfiguration()
        configuration.embedDimensions = 768
        configuration.heads = 12
        configuration.features = 128
        configuration.outChannels = [96, 192, 384, 768]
        return configuration
    }

    /// ViT-Large: 1024-d, 24 blocks, 16 heads, hooks at 4/11/17/23, widest DPT head.
    public static var large: NFKMLXDepthConfiguration {
        var configuration = NFKMLXDepthConfiguration()
        configuration.embedDimensions = 1024
        configuration.depth = 24
        configuration.heads = 16
        configuration.hooks = [4, 11, 17, 23]
        configuration.features = 256
        configuration.outChannels = [256, 512, 1024, 1024]
        return configuration
    }

    var tokenGrid: Int { inputSize / patchSize }
}

/// Multi-head self-attention (`attn.qkv`, `attn.proj`).
final class NFKDinoAttention: Module {
    let qkv: Linear
    let proj: Linear
    let heads: Int

    init(dimensions: Int, heads: Int) {
        self.heads = heads
        qkv = Linear(dimensions, dimensions * 3, bias: true)
        proj = Linear(dimensions, dimensions, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (batch, tokens, dimensions) = (x.shape[0], x.shape[1], x.shape[2])
        let headDim = dimensions / heads
        let scale = 1.0 / sqrtf(Float(headDim))

        let fused = qkv(x).reshaped([batch, tokens, 3, heads, headDim]).transposed(2, 0, 3, 1, 4)
        let parts = fused.split(parts: 3, axis: 0)
        let q = parts[0].reshaped([batch * heads, tokens, headDim])
        let k = parts[1].reshaped([batch * heads, tokens, headDim])
        let v = parts[2].reshaped([batch * heads, tokens, headDim])

        let scores = softmax((q * scale).matmul(k.transposed(0, 2, 1)), axis: -1)
        let context = scores.matmul(v)
            .reshaped([batch, heads, tokens, headDim]).transposed(0, 2, 1, 3)
            .reshaped([batch, tokens, dimensions])
        return proj(context)
    }
}

/// The per-channel learned scale DINOv2 applies after attention and the MLP (`ls1.gamma`, `ls2.gamma`).
final class NFKDinoLayerScale: Module {
    let gamma: MLXArray
    init(dimensions: Int) { gamma = MLXArray.ones([dimensions]) }
    func callAsFunction(_ x: MLXArray) -> MLXArray { x * gamma }
}

/// One transformer block: norm → attention → scale (residual), norm → MLP → scale (residual).
final class NFKDinoBlock: Module {
    let norm1: LayerNorm
    let attn: NFKDinoAttention
    let ls1: NFKDinoLayerScale
    let norm2: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: NFKDinoMLP
    let ls2: NFKDinoLayerScale

    init(dimensions: Int, heads: Int, mlpRatio: Int) {
        norm1 = LayerNorm(dimensions: dimensions)
        attn = NFKDinoAttention(dimensions: dimensions, heads: heads)
        ls1 = NFKDinoLayerScale(dimensions: dimensions)
        norm2 = LayerNorm(dimensions: dimensions)
        _mlp.wrappedValue = NFKDinoMLP(dimensions: dimensions, hidden: dimensions * mlpRatio)
        ls2 = NFKDinoLayerScale(dimensions: dimensions)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = x + ls1(attn(norm1(x)))
        out = out + ls2(mlp(norm2(out)))
        return out
    }
}

/// The block MLP (`mlp.fc1`, `mlp.fc2`).
final class NFKDinoMLP: Module {
    let fc1: Linear
    let fc2: Linear
    init(dimensions: Int, hidden: Int) {
        fc1 = Linear(dimensions, hidden, bias: true)
        fc2 = Linear(hidden, dimensions, bias: true)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(gelu(fc1(x))) }
}

/// The DINOv2 patch-embedding convolution (`patch_embed.proj`).
final class NFKDinoPatchEmbed: Module {
    @ModuleInfo(key: "proj") var proj: Conv2d
    init(patchSize: Int, dimensions: Int) {
        _proj.wrappedValue = Conv2d(inputChannels: 3, outputChannels: dimensions,
                                    kernelSize: IntOrPair(patchSize), stride: IntOrPair(patchSize))
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { proj(x) }
}

/// The DINOv2 ViT encoder (`pretrained.*`). Returns the token features at the configured hook blocks.
final class NFKDinoEncoder: Module {
    @ModuleInfo(key: "patch_embed") var patchEmbed: NFKDinoPatchEmbed
    @ModuleInfo(key: "cls_token") var clsToken: MLXArray
    @ModuleInfo(key: "pos_embed") var posEmbed: MLXArray
    let blocks: [NFKDinoBlock]
    let norm: LayerNorm

    private let hooks: Set<Int>

    init(_ configuration: NFKMLXDepthConfiguration) {
        let dimensions = configuration.embedDimensions
        let grid = configuration.tokenGrid
        _patchEmbed.wrappedValue = NFKDinoPatchEmbed(patchSize: configuration.patchSize, dimensions: dimensions)
        _clsToken.wrappedValue = MLXArray.zeros([1, 1, dimensions])
        _posEmbed.wrappedValue = MLXArray.zeros([1, grid * grid + 1, dimensions])
        blocks = (0 ..< configuration.depth).map { _ in
            NFKDinoBlock(dimensions: dimensions, heads: configuration.heads, mlpRatio: configuration.mlpRatio)
        }
        norm = LayerNorm(dimensions: dimensions)
        hooks = Set(configuration.hooks)
    }

    /// - Parameter x: `[1, inputSize, inputSize, 3]`.
    /// - Returns: the hooked features, each `[1, grid*grid, dimensions]` (class token dropped).
    func hookedFeatures(_ x: MLXArray) -> [MLXArray] {
        let dimensions = x.shape[3] == 3 ? patchEmbed.proj.weight.shape[0] : x.shape[3]
        let patches = patchEmbed(x)                             // [1, grid, grid, dimensions]
        let grid = patches.shape[1]
        var tokens = patches.reshaped([1, grid * grid, dimensions])
        tokens = concatenated([clsToken, tokens], axis: 1) + posEmbed

        var outputs = [MLXArray]()
        for (index, block) in blocks.enumerated() {
            tokens = block(tokens)
            if hooks.contains(index) {
                // DINOv2's `get_intermediate_layers` normalizes every hooked layer with the final
                // LayerNorm before handing it to the head (`norm=True` is its default). Skipping it feeds
                // the DPT head unnormalized features — the map still looks plausible, so only a numerical
                // comparison against the reference catches it.
                outputs.append(norm(tokens)[0..., 1...])       // normalize, then drop the class token
            }
        }
        return outputs
    }
}

/// A DPT residual convolution unit (`resConfUnitN`): relu → conv → relu → conv, added to the input.
final class NFKDPTResidualUnit: Module {
    let conv1: Conv2d
    let conv2: Conv2d
    init(features: Int) {
        conv1 = Conv2d(inputChannels: features, outputChannels: features, kernelSize: 3, padding: 1)
        conv2 = Conv2d(inputChannels: features, outputChannels: features, kernelSize: 3, padding: 1)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        x + conv2(relu(conv1(relu(x))))
    }
}

/// A DPT feature-fusion block (`refinenetN`): fuse a coarse path with a same-resolution skip, then
/// resize to the next level's size. The reassembled pyramid is not clean 2× multiples, so the resize
/// targets an explicit size rather than a fixed factor.
final class NFKDPTFusion: Module {
    @ModuleInfo(key: "resConfUnit1") var unit1: NFKDPTResidualUnit
    @ModuleInfo(key: "resConfUnit2") var unit2: NFKDPTResidualUnit
    @ModuleInfo(key: "out_conv") var outConv: Conv2d

    init(features: Int) {
        _unit1.wrappedValue = NFKDPTResidualUnit(features: features)
        _unit2.wrappedValue = NFKDPTResidualUnit(features: features)
        _outConv.wrappedValue = Conv2d(inputChannels: features, outputChannels: features, kernelSize: 1)
    }

    func callAsFunction(_ x: MLXArray, skip: MLXArray?, height: Int, width: Int) -> MLXArray {
        var out = x
        if let skip {
            out = out + unit1(skip)
        }
        out = unit2(out)
        out = NFKMLXResample.resizeNearest(out, height: height, width: width)
        return outConv(out)
    }
}

/// The DPT head (`depth_head.*`): reassemble the four hooked features to a pyramid, fuse coarse → fine,
/// and read out a single-channel depth map.
final class NFKDPTHead: Module {
    @ModuleInfo(key: "projects") var projects: [Conv2d]
    @ModuleInfo(key: "resize_layers") var resizeLayers: [Module]
    @ModuleInfo(key: "scratch") var scratch: NFKDPTScratch

    private let grid: Int

    init(_ configuration: NFKMLXDepthConfiguration) {
        let dimensions = configuration.embedDimensions
        let out = configuration.outChannels
        _projects.wrappedValue = out.map { Conv2d(inputChannels: dimensions, outputChannels: $0, kernelSize: 1) }
        _resizeLayers.wrappedValue = [
            ConvTransposed2d(inputChannels: out[0], outputChannels: out[0], kernelSize: 4, stride: 4),
            ConvTransposed2d(inputChannels: out[1], outputChannels: out[1], kernelSize: 2, stride: 2),
            Identity(),
            Conv2d(inputChannels: out[3], outputChannels: out[3], kernelSize: 3, stride: 2, padding: 1),
        ]
        _scratch.wrappedValue = NFKDPTScratch(features: configuration.features, outChannels: out)
        grid = configuration.tokenGrid
    }

    /// - Parameter features: four hooked token features, each `[1, grid*grid, dimensions]`.
    /// - Returns: a depth map `[1, H, W, 1]` at the finest fusion resolution.
    func callAsFunction(_ features: [MLXArray]) -> MLXArray {
        var pyramid = [MLXArray]()
        for (index, tokens) in features.enumerated() {
            let dimensions = tokens.shape[2]
            let spatial = tokens.reshaped([1, grid, grid, dimensions])
            var reassembled = projects[index](spatial)
            reassembled = (resizeLayers[index] as? UnaryLayer)?.callAsFunction(reassembled) ?? reassembled
            pyramid.append(reassembled)
        }
        return scratch(pyramid)
    }
}

/// The DPT `scratch`: per-level input convolutions, four fusion blocks, and the output convolutions.
final class NFKDPTScratch: Module {
    @ModuleInfo(key: "layer1_rn") var layer1: Conv2d
    @ModuleInfo(key: "layer2_rn") var layer2: Conv2d
    @ModuleInfo(key: "layer3_rn") var layer3: Conv2d
    @ModuleInfo(key: "layer4_rn") var layer4: Conv2d
    @ModuleInfo(key: "refinenet1") var refine1: NFKDPTFusion
    @ModuleInfo(key: "refinenet2") var refine2: NFKDPTFusion
    @ModuleInfo(key: "refinenet3") var refine3: NFKDPTFusion
    @ModuleInfo(key: "refinenet4") var refine4: NFKDPTFusion
    @ModuleInfo(key: "output_conv1") var outputConv1: Conv2d
    @ModuleInfo(key: "output_conv2") var outputConv2: [Module]

    init(features: Int, outChannels: [Int]) {
        _layer1.wrappedValue = Conv2d(inputChannels: outChannels[0], outputChannels: features, kernelSize: 3, padding: 1, bias: false)
        _layer2.wrappedValue = Conv2d(inputChannels: outChannels[1], outputChannels: features, kernelSize: 3, padding: 1, bias: false)
        _layer3.wrappedValue = Conv2d(inputChannels: outChannels[2], outputChannels: features, kernelSize: 3, padding: 1, bias: false)
        _layer4.wrappedValue = Conv2d(inputChannels: outChannels[3], outputChannels: features, kernelSize: 3, padding: 1, bias: false)
        _refine1.wrappedValue = NFKDPTFusion(features: features)
        _refine2.wrappedValue = NFKDPTFusion(features: features)
        _refine3.wrappedValue = NFKDPTFusion(features: features)
        _refine4.wrappedValue = NFKDPTFusion(features: features)
        _outputConv1.wrappedValue = Conv2d(inputChannels: features, outputChannels: features / 2, kernelSize: 3, padding: 1)
        // A Sequential(conv, relu, conv, relu) in the reference, so the convolutions carry indices 0 and 2.
        _outputConv2.wrappedValue = [
            Conv2d(inputChannels: features / 2, outputChannels: 32, kernelSize: 3, padding: 1),
            ReLU(),
            Conv2d(inputChannels: 32, outputChannels: 1, kernelSize: 1),
            ReLU(),
        ]
    }

    private func size(_ x: MLXArray) -> (Int, Int) { (x.shape[1], x.shape[2]) }

    /// - Parameter pyramid: reassembled features fine → coarse (`layer1` … `layer4`).
    func callAsFunction(_ pyramid: [MLXArray]) -> MLXArray {
        let l1 = layer1(pyramid[0])
        let l2 = layer2(pyramid[1])
        let l3 = layer3(pyramid[2])
        let l4 = layer4(pyramid[3])

        var path = refine4(l4, skip: nil, height: size(l3).0, width: size(l3).1)
        path = refine3(path, skip: l3, height: size(l2).0, width: size(l2).1)
        path = refine2(path, skip: l2, height: size(l1).0, width: size(l1).1)
        path = refine1(path, skip: l1, height: size(l1).0 * 2, width: size(l1).1 * 2)

        let depth = outputConv1(path)
        return (outputConv2[3] as! ReLU)((outputConv2[2] as! Conv2d)(
            (outputConv2[1] as! ReLU)((outputConv2[0] as! Conv2d)(depth))))
    }
}

/// Nearest-neighbor resampling helpers for NHWC tensors.
enum NFKMLXResample {
    /// Pads `[N, H, W, C]` by mirroring the border inward without repeating the edge sample — the
    /// reference `ReflectionPad2d`. MLX pads with a constant or the edge value only, and the difference
    /// is not cosmetic: a network that normalizes over whole feature maps carries a border approximation
    /// into every pixel.
    static func reflectPadded(_ x: MLXArray, _ pad: Int) -> MLXArray {
        guard pad > 0 else { return x }
        func mirrored(_ size: Int) -> MLXArray {
            MLXArray((-pad ..< size + pad).map { index -> Int32 in
                if index < 0 { return Int32(-index) }
                if index >= size { return Int32(2 * size - 2 - index) }
                return Int32(index)
            })
        }
        return take(take(x, mirrored(x.shape[1]), axis: 1), mirrored(x.shape[2]), axis: 2)
    }

    /// Pads the height and width of `[N, H, W, C]` on both sides, leaving batch and channels alone.
    static func spatiallyPadded(_ x: MLXArray, _ pad: IntOrPair, value: Float) -> MLXArray {
        guard pad.first > 0 || pad.second > 0 else { return x }
        let widths = [IntOrPair(0), IntOrPair(pad.first), IntOrPair(pad.second), IntOrPair(0)]
        return padded(x, widths: widths, mode: .constant, value: MLXArray(value).asType(x.dtype))
    }

    /// Max-pools `[N, H, W, C]`, adding the border here rather than through the pooling layer.
    ///
    /// `Pool.callAsFunction` in mlx-swift 0.31.6 builds its pad widths as `[0, 0] + padding + [0, 0]`,
    /// which is two entries too many: a four-axis input gets the first four, so a 2-D pool pads width
    /// and channels instead of height and width. It raises nothing — the output shape is silently wrong,
    /// and the failure surfaces later as a channel mismatch from an innocent layer. Every `Pool`
    /// subclass shares that initializer, so any pooling that needs a border comes through here and
    /// `averagePooled` instead of passing `padding:` to the framework layer.
    static func maxPooled(_ x: MLXArray, kernel: IntOrPair, stride: IntOrPair,
                          padding: IntOrPair = 0) -> MLXArray {
        MaxPool2d(kernelSize: kernel, stride: stride)(spatiallyPadded(x, padding, value: -.infinity))
    }

    /// Average-pools `[N, H, W, C]`, adding the border here rather than through the pooling layer.
    /// The zero border counts toward each window's mean, as PyTorch's `count_include_pad` default does.
    static func averagePooled(_ x: MLXArray, kernel: IntOrPair, stride: IntOrPair,
                              padding: IntOrPair = 0) -> MLXArray {
        AvgPool2d(kernelSize: kernel, stride: stride)(spatiallyPadded(x, padding, value: 0))
    }

    static func upsampleNearest(_ x: MLXArray, scale: Int) -> MLXArray {
        let (n, h, w, c) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        let expanded = x.reshaped([n, h, 1, w, 1, c])
        return broadcast(expanded, to: [n, h, scale, w, scale, c]).reshaped([n, h * scale, w * scale, c])
    }

    /// Resamples `[N, H, W, C]` to `height` × `width` by nearest neighbor.
    static func resizeNearest(_ x: MLXArray, height: Int, width: Int) -> MLXArray {
        let (n, h, w, c) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        let rowIndex = (0 ..< height).map { Int32($0 * h / height) }
        let colIndex = (0 ..< width).map { Int32($0 * w / width) }
        let rows = MLXArray(rowIndex)
        let cols = MLXArray(colIndex)
        return x[0..., rows][0..., 0..., cols].reshaped([n, height, width, c])
    }

    /// Resamples `[N, H, W, C]` to `height` × `width` by bilinear interpolation with `align_corners=false`
    /// (PyTorch's `F.interpolate(mode: "bilinear")` default), matching networks trained with bilinear
    /// upsampling. Nearest neighbor introduces aliasing those models were not trained for.
    static func resizeBilinear(_ x: MLXArray, height: Int, width: Int) -> MLXArray {
        let (n, h, w, c) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        func axis(_ outSize: Int, _ inSize: Int) -> (lo: MLXArray, hi: MLXArray, frac: [Float]) {
            let scale = Float(inSize) / Float(outSize)
            var lo = [Int32](), hi = [Int32](), frac = [Float]()
            for o in 0 ..< outSize {
                let src = min(max((Float(o) + 0.5) * scale - 0.5, 0), Float(inSize - 1))
                let low = Int(src.rounded(.down))
                lo.append(Int32(low))
                hi.append(Int32(min(low + 1, inSize - 1)))
                frac.append(src - Float(low))
            }
            return (MLXArray(lo), MLXArray(hi), frac)
        }
        let (y0, y1, yf) = axis(height, h)
        let (x0, x1, xf) = axis(width, w)
        let rows0 = x[0..., y0], rows1 = x[0..., y1]
        let c00 = rows0[0..., 0..., x0], c01 = rows0[0..., 0..., x1]
        let c10 = rows1[0..., 0..., x0], c11 = rows1[0..., 0..., x1]
        let wy = MLXArray(yf).reshaped([1, height, 1, 1])
        let wx = MLXArray(xf).reshaped([1, 1, width, 1])
        let top = c00 * (1 - wx) + c01 * wx
        let bottom = c10 * (1 - wx) + c11 * wx
        return (top * (1 - wy) + bottom * wy).reshaped([n, height, width, c])
    }
}

/// Depth Anything V2 as an InferKit backend, and its registration for the Objective-C path.
///
/// `NFKMLXDepthAnythingNet` is the real DINOv2 + DPT model. Random weights run (proving the pipeline);
/// a trained **safetensors** checkpoint makes the depth meaningful. Because the network is large and
/// the reference key layout is intricate, confirm a real checkpoint with the self-validating converter
/// (`Tools/depth-anything-to-safetensors`), which reports any key/shape mismatch before use.
/// The Depth Anything V2 size, for the Objective-C factory.
@objc(NFKMLXDepthVariant)
public enum NFKMLXDepthVariant: Int {
    case small
    case base
    case large
}

@objc(NFKMLXDepthAnything)
public final class NFKMLXDepthAnything: NSObject {

    @objc public static let modelName = "depth-anything-v2-small"
    @objc public static let baseModelName = "depth-anything-v2-base"
    @objc public static let largeModelName = "depth-anything-v2-large"

    static func makeNet(_ configuration: NFKMLXDepthConfiguration = NFKMLXDepthConfiguration()) -> NFKMLXDepthAnythingNet {
        NFKMLXDepthAnythingNet(configuration)
    }

    private static func specs(for variant: NFKMLXDepthVariant) -> (name: String, configuration: NFKMLXDepthConfiguration) {
        switch variant {
        case .small: return (modelName, .small)
        case .base: return (baseModelName, .base)
        case .large: return (largeModelName, .large)
        }
    }

    /// Builds a Depth Anything V2 backend directly from optional local weights — no registry required.
    /// A nil `weightsURL` builds random weights (`isReady` is true).
    /// Run inference off the render thread.
    @objc(backendWithVariant:weightsURL:error:)
    public static func backend(variant: NFKMLXDepthVariant, weightsURL: URL?) throws -> any NFKInferenceBackend {
        let spec = specs(for: variant)
        let net = NFKMLXDepthAnythingNet(spec.configuration)
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        let holder = NFKMLXDepthHolder(net)
        return NFKMLXModuleBackend(identifier: spec.name, isReady: true) { image in holder.net.depth(image) }
    }

    /// Downloads the checkpoint from Hugging Face, then builds — no registry required.
    /// Blocking on the network; run off the render thread.
    @objc(backendWithVariant:repo:weightsPath:revision:cacheDirectoryURL:error:)
    public static func backend(variant: NFKMLXDepthVariant, repo: String, weightsPath: String,
                               revision: String?, cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        let url = try NFKMLXDownload.weightsURL(repo: repo, weightsPath: weightsPath, revision: revision, cacheDirectoryURL: cacheDirectoryURL)
        return try backend(variant: variant, weightsURL: url)
    }

    /// The asynchronous form of the download factory: downloads on a background queue, then builds and
    /// delivers the backend (or an error) to `completionHandler`.
    @objc(backendWithVariant:repo:weightsPath:revision:cacheDirectoryURL:completionHandler:)
    public static func backend(variant: NFKMLXDepthVariant, repo: String, weightsPath: String,
                               revision: String?, cacheDirectoryURL: URL?,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXDownload.backend(repo: repo, weightsPath: weightsPath, revision: revision,
                               cacheDirectoryURL: cacheDirectoryURL,
                               build: { try backend(variant: variant, weightsURL: $0) },
                               completionHandler: completionHandler)
    }

    /// Registers all three Depth Anything V2 sizes with `NFKMLXModelRegistry`, each delegating to
    /// `backend(variant:weightsURL:)`.
    @objc public static func register() {
        for variant in [NFKMLXDepthVariant.small, .base, .large] {
            NFKMLXModelRegistry.register(name: specs(for: variant).name) { weightsURL in
                try backend(variant: variant, weightsURL: weightsURL)
            }
        }
    }

    /// Loads a safetensors checkpoint, transposing 4-D convolution weights from PyTorch
    /// `[out, in, kH, kW]` to MLX `[out, kH, kW, in]`. `remap` renames keys when a checkpoint differs.
    static func loadWeights(into net: NFKMLXDepthAnythingNet, from url: URL,
                            remap: (String) -> String = { $0 }) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let raw = checkpoint.arrays
        let mapped = raw.map { key, value in
            (remap(key), checkpoint.needsConvTranspose && value.ndim == 4 ? value.transposed(0, 2, 3, 1) : value)
        }
        try NFKMLXWeights.apply(mapped, to: net)
    }
}

/// The full model: encoder (`pretrained`) + DPT head (`depth_head`).
final class NFKMLXDepthAnythingNet: Module {
    @ModuleInfo(key: "pretrained") var pretrained: NFKDinoEncoder
    @ModuleInfo(key: "depth_head") var depthHead: NFKDPTHead

    let configuration: NFKMLXDepthConfiguration

    init(_ configuration: NFKMLXDepthConfiguration) {
        self.configuration = configuration
        _pretrained.wrappedValue = NFKDinoEncoder(configuration)
        _depthHead.wrappedValue = NFKDPTHead(configuration)
    }

    /// Maps a bridged image `[H, W, 3]` (`0...1`) to a grayscale depth image `[H, W, 3]` (`0...1`),
    /// near = bright. The encoder runs at the fixed input size; the map resizes back and normalizes.
    func depth(_ image: MLXArray) -> MLXArray {
        let (height, width) = (image.shape[0], image.shape[1])
        let resized = NFKMLXResample.resizeBilinear(
            image.reshaped([1, height, width, image.shape[2]]),
            height: configuration.inputSize, width: configuration.inputSize)
        // Depth Anything's transform normalizes with ImageNet statistics before the encoder. The final
        // min-max scaling of the depth map hides a missing normalization — relative near/far ordering
        // survives while the values themselves are wrong — so this is easy to overlook.
        let mean = MLXArray([Float(0.485), 0.456, 0.406])
        let standardDeviation = MLXArray([Float(0.229), 0.224, 0.225])
        let prepared = (resized - mean) / standardDeviation
        let features = pretrained.hookedFeatures(prepared)
        let map = depthHead(features)                          // [1, h, w, 1]
        let full = NFKMLXResample.resizeBilinear(map, height: height, width: width)

        let minimum = full.min()
        let span = maximum(full.max() - minimum, MLXArray(1e-6))
        let normalized = ((full - minimum) / span).reshaped([height, width, 1])
        return concatenated([normalized, normalized, normalized], axis: 2)
    }
}

private final class NFKMLXDepthHolder: @unchecked Sendable {
    let net: NFKMLXDepthAnythingNet
    init(_ net: NFKMLXDepthAnythingNet) { self.net = net }
}
