//
//  NFKMLXSegFormer.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXNN

// SegFormer assigns a class to every pixel. A hierarchical transformer encoder (MiT) produces features
// at four scales using efficient self-attention (keys and values are spatially reduced) and a Mix-FFN
// (a depthwise convolution inside the feed-forward, so no positional embedding is needed); a light
// all-MLP head fuses the four scales and classifies each location. The argmax over classes is a label
// map, emitted as a grayscale image whose pixel value encodes the class index.
//
// Module structure follows the reference SegFormer. Names are grouped into clean stages rather than the
// reference's flat `encoder.block.N.M.*` layout, so the exact key remap and the BN in the decode head
// are validation-sweep items. Tensors flow in NHWC; spatial tokens are `[1, h·w, C]`.

/// SegFormer dimensions. Defaults size the MiT-B0 encoder; `tiny` keeps tests fast.
public struct NFKMLXSegFormerConfiguration: Sendable {
    public var embedDimensions: [Int]
    public var heads: [Int]
    public var reductions: [Int]
    public var depths: [Int]
    public var decodeDimensions: Int
    public var classCount: Int

    public init(embedDimensions: [Int] = [32, 64, 160, 256], heads: [Int] = [1, 2, 5, 8],
                reductions: [Int] = [8, 4, 2, 1], depths: [Int] = [2, 2, 2, 2],
                decodeDimensions: Int = 256, classCount: Int = 150) {
        self.embedDimensions = embedDimensions
        self.heads = heads
        self.reductions = reductions
        self.depths = depths
        self.decodeDimensions = decodeDimensions
        self.classCount = classCount
    }

    public static let mitB0 = NFKMLXSegFormerConfiguration()

    public static let tiny = NFKMLXSegFormerConfiguration(embedDimensions: [8, 16, 24, 32], heads: [1, 2, 3, 4],
                                                          reductions: [4, 2, 1, 1], depths: [1, 1, 1, 1],
                                                          decodeDimensions: 16, classCount: 4)
}

/// Overlapping patch embedding: a strided convolution that also overlaps neighbors, then a LayerNorm.
final class NFKSegFormerPatchEmbed: Module {
    @ModuleInfo(key: "proj") var proj: Conv2d
    @ModuleInfo(key: "norm") var norm: LayerNorm

    init(inChannels: Int, outChannels: Int, patch: Int, stride: Int) {
        _proj.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: IntOrPair(patch),
                                    stride: IntOrPair(stride), padding: IntOrPair(patch / 2))
        _norm.wrappedValue = LayerNorm(dimensions: outChannels)
    }

    /// Returns tokens `[1, h·w, C]` and the grid size `(h, w)`.
    func callAsFunction(_ image: MLXArray) -> (tokens: MLXArray, height: Int, width: Int) {
        let x = proj(image)                                     // [1, h, w, C]
        let (h, w, c) = (x.shape[1], x.shape[2], x.shape[3])
        return (norm(x.reshaped([1, h * w, c])), h, w)
    }
}

/// Efficient self-attention: queries at full resolution, keys and values from a spatially reduced map.
final class NFKSegFormerAttention: Module {
    @ModuleInfo(key: "q") var q: Linear
    @ModuleInfo(key: "kv") var kv: Linear
    @ModuleInfo(key: "proj") var proj: Linear
    @ModuleInfo(key: "sr") var sr: Conv2d?
    @ModuleInfo(key: "sr_norm") var srNorm: LayerNorm?
    private let heads: Int

    init(dimensions: Int, heads: Int, reduction: Int) {
        self.heads = heads
        _q.wrappedValue = Linear(dimensions, dimensions)
        _kv.wrappedValue = Linear(dimensions, dimensions * 2)
        _proj.wrappedValue = Linear(dimensions, dimensions)
        if reduction > 1 {
            _sr.wrappedValue = Conv2d(inputChannels: dimensions, outputChannels: dimensions,
                                      kernelSize: IntOrPair(reduction), stride: IntOrPair(reduction))
            _srNorm.wrappedValue = LayerNorm(dimensions: dimensions)
        }
    }

    func callAsFunction(_ x: MLXArray, height: Int, width: Int) -> MLXArray {
        let (tokens, dimensions) = (x.shape[1], x.shape[2])
        let headDim = dimensions / heads
        let scale = 1.0 / sqrtf(Float(headDim))

        var context = x
        if let sr, let srNorm {
            let spatial = x.reshaped([1, height, width, dimensions])
            let reduced = sr(spatial)
            context = srNorm(reduced.reshaped([1, reduced.shape[1] * reduced.shape[2], dimensions]))
        }
        let contextTokens = context.shape[1]

        let queries = q(x).reshaped([1, tokens, heads, headDim]).transposed(0, 2, 1, 3).reshaped([heads, tokens, headDim])
        let kvPair = kv(context).reshaped([1, contextTokens, 2, heads, headDim]).transposed(2, 0, 3, 1, 4)
        let keys = kvPair[0].reshaped([heads, contextTokens, headDim])
        let values = kvPair[1].reshaped([heads, contextTokens, headDim])

        let scores = softmax(queries.matmul(keys.transposed(0, 2, 1)) * scale, axis: -1)
        let attended = scores.matmul(values).reshaped([1, heads, tokens, headDim]).transposed(0, 2, 1, 3).reshaped([1, tokens, dimensions])
        return proj(attended)
    }
}

/// Mix-FFN: a feed-forward with a depthwise convolution between the two linear layers, which supplies
/// positional information in place of a positional embedding.
final class NFKSegFormerMixFFN: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "dwconv") var dwconv: Conv2d
    @ModuleInfo(key: "fc2") var fc2: Linear

    init(dimensions: Int, hidden: Int) {
        _fc1.wrappedValue = Linear(dimensions, hidden)
        _dwconv.wrappedValue = Conv2d(inputChannels: hidden, outputChannels: hidden, kernelSize: 3, padding: 1, groups: hidden)
        _fc2.wrappedValue = Linear(hidden, dimensions)
    }

    func callAsFunction(_ x: MLXArray, height: Int, width: Int) -> MLXArray {
        var h = fc1(x)
        let hidden = h.shape[2]
        h = dwconv(h.reshaped([1, height, width, hidden])).reshaped([1, height * width, hidden])
        return fc2(gelu(h))
    }
}

/// A transformer block: pre-norm efficient attention, then pre-norm Mix-FFN.
final class NFKSegFormerBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "attn") var attn: NFKSegFormerAttention
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "ffn") var ffn: NFKSegFormerMixFFN

    init(dimensions: Int, heads: Int, reduction: Int) {
        _norm1.wrappedValue = LayerNorm(dimensions: dimensions)
        _attn.wrappedValue = NFKSegFormerAttention(dimensions: dimensions, heads: heads, reduction: reduction)
        _norm2.wrappedValue = LayerNorm(dimensions: dimensions)
        _ffn.wrappedValue = NFKSegFormerMixFFN(dimensions: dimensions, hidden: dimensions * 4)
    }

    func callAsFunction(_ x: MLXArray, height: Int, width: Int) -> MLXArray {
        let h = x + attn(norm1(x), height: height, width: width)
        return h + ffn(norm2(h), height: height, width: width)
    }
}

/// One encoder stage: patch embedding, a run of transformer blocks, and a final norm.
final class NFKSegFormerStage: Module {
    @ModuleInfo(key: "patch_embed") var patchEmbed: NFKSegFormerPatchEmbed
    @ModuleInfo(key: "blocks") var blocks: [NFKSegFormerBlock]
    @ModuleInfo(key: "norm") var norm: LayerNorm

    init(inChannels: Int, outChannels: Int, patch: Int, stride: Int, heads: Int, reduction: Int, depth: Int) {
        _patchEmbed.wrappedValue = NFKSegFormerPatchEmbed(inChannels: inChannels, outChannels: outChannels, patch: patch, stride: stride)
        _blocks.wrappedValue = (0 ..< depth).map { _ in NFKSegFormerBlock(dimensions: outChannels, heads: heads, reduction: reduction) }
        _norm.wrappedValue = LayerNorm(dimensions: outChannels)
    }

    /// Returns the stage feature map `[1, h, w, C]`.
    func callAsFunction(_ image: MLXArray) -> MLXArray {
        var (x, h, w) = patchEmbed(image)
        for block in blocks {
            x = block(x, height: h, width: w)
        }
        x = norm(x)
        return x.reshaped([1, h, w, x.shape[2]])
    }
}

/// The SegFormer network: a four-stage MiT encoder and an all-MLP decode head.
///
/// Fine-tuning works on this type directly: build one with ``NFKMLXSegFormer/network(weightsURL:classCount:)``,
/// train it with ``NFKMLXSegFormer/fineTune(_:examples:trainable:objective:optimizer:steps:clipGradientNorm:checkpoint:observer:)``,
/// then save it for ``NFKMLXSegFormer/backend(weightsURL:)``.
public final class NFKMLXSegFormerNet: Module {
    @ModuleInfo(key: "stage1") var stage1: NFKSegFormerStage
    @ModuleInfo(key: "stage2") var stage2: NFKSegFormerStage
    @ModuleInfo(key: "stage3") var stage3: NFKSegFormerStage
    @ModuleInfo(key: "stage4") var stage4: NFKSegFormerStage
    @ModuleInfo(key: "linear_c") var linearC: [Linear]
    @ModuleInfo(key: "linear_fuse") var linearFuse: Conv2d
    @ModuleInfo(key: "batch_norm") var batchNorm: BatchNorm
    @ModuleInfo(key: "classifier") var classifier: Conv2d

    let configuration: NFKMLXSegFormerConfiguration

    init(_ c: NFKMLXSegFormerConfiguration) {
        configuration = c
        let e = c.embedDimensions
        _stage1.wrappedValue = NFKSegFormerStage(inChannels: 3, outChannels: e[0], patch: 7, stride: 4, heads: c.heads[0], reduction: c.reductions[0], depth: c.depths[0])
        _stage2.wrappedValue = NFKSegFormerStage(inChannels: e[0], outChannels: e[1], patch: 3, stride: 2, heads: c.heads[1], reduction: c.reductions[1], depth: c.depths[1])
        _stage3.wrappedValue = NFKSegFormerStage(inChannels: e[1], outChannels: e[2], patch: 3, stride: 2, heads: c.heads[2], reduction: c.reductions[2], depth: c.depths[2])
        _stage4.wrappedValue = NFKSegFormerStage(inChannels: e[2], outChannels: e[3], patch: 3, stride: 2, heads: c.heads[3], reduction: c.reductions[3], depth: c.depths[3])
        _linearC.wrappedValue = e.map { Linear($0, c.decodeDimensions) }
        // The reference fuses with a bias-free 1×1 convolution followed by BatchNorm and ReLU.
        _linearFuse.wrappedValue = Conv2d(inputChannels: c.decodeDimensions * 4, outputChannels: c.decodeDimensions,
                                          kernelSize: 1, bias: false)
        _batchNorm.wrappedValue = BatchNorm(featureCount: c.decodeDimensions)
        _classifier.wrappedValue = Conv2d(inputChannels: c.decodeDimensions, outputChannels: c.classCount, kernelSize: 1)
    }

    /// Applies the input normalization SegFormer's image processor performs, taking a bridged image
    /// `[H, W, 3]` in `0...1` to the batched, ImageNet-normalized `[1, H, W, 3]` the encoder expects.
    ///
    /// Inference and fine-tuning both go through this, because a training run that normalized
    /// differently would optimize for a distribution the model never sees at inference. Four models
    /// in this package have already shipped that bug.
    func normalized(_ image: MLXArray) -> MLXArray {
        let mean = MLXArray([Float(0.485), 0.456, 0.406])
        let standardDeviation = MLXArray([Float(0.229), 0.224, 0.225])
        return (image.reshaped([1, image.shape[0], image.shape[1], 3]) - mean) / standardDeviation
    }

    /// Produces class logits `[1, h, w, classCount]` at the finest (stage-1) resolution, from an
    /// already-normalized batch.
    public func logits(_ image: MLXArray) -> MLXArray {
        let f1 = stage1(image)
        let f2 = stage2(f1)
        let f3 = stage3(f2)
        let f4 = stage4(f3)
        let features = [f1, f2, f3, f4]

        let (targetH, targetW) = (f1.shape[1], f1.shape[2])
        let projected = features.enumerated().map { index, feature -> MLXArray in
            let (h, w) = (feature.shape[1], feature.shape[2])
            let mapped = linearC[index](feature.reshaped([1, h * w, feature.shape[3]])).reshaped([1, h, w, configuration.decodeDimensions])
            return NFKMLXResample.resizeBilinear(mapped, height: targetH, width: targetW)
        }
        // The reference concatenates the projected stages COARSEST FIRST (`all_hidden_states[::-1]`), so
        // the fuse convolution's input channels are ordered stage 4 → stage 1.
        let fused = relu(batchNorm(linearFuse(concatenated(projected.reversed(), axis: 3))))
        return classifier(fused)
    }

    /// Segments a bridged image `[H, W, 3]` (`0...1`), returning a grayscale label map `[H, W, 1]` whose
    /// value is the class index scaled to `0...1`.
    func segment(_ image: MLXArray) -> MLXArray {
        let (height, width) = (image.shape[0], image.shape[1])
        let scores = logits(normalized(image))
        let labels = scores.argMax(axis: -1)                    // [1, h, w]
        let normalized = labels.asType(.float32) / Float(max(configuration.classCount - 1, 1))
        let map = normalized.reshaped([1, scores.shape[1], scores.shape[2], 1])
        let full = NFKMLXResample.resizeNearest(map, height: height, width: width)
        return full.reshaped([height, width, 1])
    }
}

/// Holds the network for capture in the backend's `@Sendable` closure.
private final class NFKSegFormerHolder: @unchecked Sendable {
    let net: NFKMLXSegFormerNet
    init(_ net: NFKMLXSegFormerNet) { self.net = net }
}

/// SegFormer semantic segmentation as an InferKit backend, and its registration for the Objective-C path.
///
/// `NFKMLXSegFormerNet` is the real MiT + MLP-head model. Random weights run (proving the pipeline); a
/// trained checkpoint segments accurately. The output is a grayscale label map under `NFKOutputImage`;
/// a consumer recovers the class index as `round(gray · (classCount − 1))`. Load a **safetensors**
/// checkpoint; the loader transposes 4-D convolution weights `[out, in, kH, kW]` to MLX's `[out, kH,
/// kW, in]`.
@objc(NFKMLXSegFormer)
public final class NFKMLXSegFormer: NSObject {

    /// The registry name the model builds under.
    @objc public static let modelName = "segformer-b0"

    /// Builds a segmentation backend directly from optional local weights — no registry required. A nil
    /// `weightsURL` builds random weights (`isReady` is true). Run
    /// inference off the render thread.
    @objc(backendWithWeightsURL:error:)
    public static func backend(weightsURL: URL?) throws -> any NFKInferenceBackend {
        let net = NFKMLXSegFormerNet(.mitB0)
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        let holder = NFKSegFormerHolder(net)
        return NFKMLXModuleBackend(identifier: modelName, isReady: true) { image in holder.net.segment(image) }
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

    /// Registers SegFormer (`segformer-b0`) with `NFKMLXModelRegistry`, delegating to `backend(weightsURL:)`.
    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName) { weightsURL in
            try backend(weightsURL: weightsURL)
        }
    }

    /// Loads a safetensors checkpoint into `net`, transposing 4-D convolution weights from PyTorch's
    /// `[out, in, kH, kW]` to MLX's channels-last `[out, kH, kW, in]`.
    static func loadWeights(into net: NFKMLXSegFormerNet, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let raw = checkpoint.arrays
        var mapped = raw.compactMap { key, value -> (String, MLXArray)? in
            // key/value fold into one tensor below; drop them here so they are not applied twice.
            if key.contains(".attention.self.key.") || key.contains(".attention.self.value.") { return nil }
            let name = remapReferenceKey(key)
            guard !name.isEmpty else { return nil }
            return (name, checkpoint.needsConvTranspose && value.ndim == 4 ? value.transposed(0, 2, 3, 1) : value)
        }
        mapped += fusedKeyValueWeights(from: raw)
        try NFKMLXWeights.apply(mapped, to: net)
    }

    /// The reference keeps separate `key` and `value` projections where this port uses one fused `kv`
    /// Linear, so the two tensors concatenate along the output dimension — the order the fused projection
    /// unpacks them in. A 1:1 key remap cannot express a two-into-one fold.
    private static func fusedKeyValueWeights(from raw: [String: MLXArray]) -> [(String, MLXArray)] {
        var result = [(String, MLXArray)]()
        for (key, keyValue) in raw where key.contains(".attention.self.key.") {
            let valueName = key.replacingOccurrences(of: ".attention.self.key.", with: ".attention.self.value.")
            guard let valueValue = raw[valueName] else { continue }
            let suffix = key.hasSuffix(".weight") ? "weight" : "bias"
            let target = remapReferenceKey(key)
                .replacingOccurrences(of: ".attn.k.\(suffix)", with: ".attn.kv.\(suffix)")
            result.append((target, concatenated([keyValue, valueValue], axis: 0)))
        }
        return result
    }

    /// Maps a Hugging Face SegFormer checkpoint's names onto this module's. The reference nests the
    /// encoder under `segformer.encoder` with `block.<stage>.<index>` and separate `patch_embeddings` and
    /// `layer_norm` lists, while this port groups each stage together.
    static func remapReferenceKey(_ key: String) -> String {
        var name = key

        // Encoder: patch embeddings and the per-stage final norms live in their own indexed lists.
        for stage in 0 ..< 4 {
            name = name.replacingOccurrences(of: "segformer.encoder.patch_embeddings.\(stage).proj.",
                                             with: "stage\(stage + 1).patch_embed.proj.")
            name = name.replacingOccurrences(of: "segformer.encoder.patch_embeddings.\(stage).layer_norm.",
                                             with: "stage\(stage + 1).patch_embed.norm.")
            name = name.replacingOccurrences(of: "segformer.encoder.layer_norm.\(stage).",
                                             with: "stage\(stage + 1).norm.")
            name = name.replacingOccurrences(of: "segformer.encoder.block.\(stage).",
                                             with: "stage\(stage + 1).blocks.")
        }
        name = name.replacingOccurrences(of: ".layer_norm_1.", with: ".norm1.")
        name = name.replacingOccurrences(of: ".layer_norm_2.", with: ".norm2.")
        name = name.replacingOccurrences(of: ".attention.self.query.", with: ".attn.q.")
        name = name.replacingOccurrences(of: ".attention.self.key.", with: ".attn.k.")
        name = name.replacingOccurrences(of: ".attention.self.sr.", with: ".attn.sr.")
        name = name.replacingOccurrences(of: ".attention.self.layer_norm.", with: ".attn.sr_norm.")
        name = name.replacingOccurrences(of: ".attention.output.dense.", with: ".attn.proj.")
        name = name.replacingOccurrences(of: ".mlp.dwconv.dwconv.", with: ".ffn.dwconv.")
        name = name.replacingOccurrences(of: ".mlp.dense1.", with: ".ffn.fc1.")
        name = name.replacingOccurrences(of: ".mlp.dense2.", with: ".ffn.fc2.")

        // Decode head. The `.proj.` strip must stay scoped to `linear_c`: the encoder's patch embeddings
        // also end in `.proj.weight`, and an unscoped replacement would silently rename those too.
        name = name.replacingOccurrences(of: "decode_head.linear_c.", with: "linear_c.")
        if name.hasPrefix("linear_c.") {
            name = name.replacingOccurrences(of: ".proj.", with: ".")
        }
        name = name.replacingOccurrences(of: "decode_head.", with: "")
        return name
    }
}
