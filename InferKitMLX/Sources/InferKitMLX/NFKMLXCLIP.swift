//
//  NFKMLXCLIP.swift
//  InferKitMLX
//

import Foundation
import CoreGraphics
import InferKit
import MLX
import MLXNN

// CLIP encodes an image and a text string into a shared embedding space, so their similarity is a dot
// product. The image encoder is a Vision Transformer (patch-embedding convolution, class token,
// transformer blocks); the text encoder is a causal Transformer over byte-level-BPE tokens. Both
// project to a common dimension and L2-normalize.
//
// Module structure and parameter names mirror the reference OpenAI CLIP (`visual.*` for the image
// tower, `token_embedding` / `transformer` / `ln_final` / `text_projection` for the text tower), so a
// converted checkpoint loads by name. Attention keeps the reference's fused `in_proj_weight` /
// `in_proj_bias` and `out_proj`. Tensors flow in NHWC for the convolution and `[batch, tokens, width]`
// for the transformers.

/// CLIP dimensions. The defaults size the ViT-B/32 model; `tiny` keeps tests fast.
public struct NFKMLXCLIPConfiguration: Sendable {
    public var imageResolution: Int
    public var patchSize: Int
    public var visionWidth: Int
    public var visionLayers: Int
    public var visionHeads: Int
    public var embedDimensions: Int
    public var vocabularySize: Int
    public var contextLength: Int
    public var textWidth: Int
    public var textLayers: Int
    public var textHeads: Int

    public init(imageResolution: Int = 224, patchSize: Int = 32, visionWidth: Int = 768,
                visionLayers: Int = 12, visionHeads: Int = 12, embedDimensions: Int = 512,
                vocabularySize: Int = 49408, contextLength: Int = 77, textWidth: Int = 512,
                textLayers: Int = 12, textHeads: Int = 8) {
        self.imageResolution = imageResolution
        self.patchSize = patchSize
        self.visionWidth = visionWidth
        self.visionLayers = visionLayers
        self.visionHeads = visionHeads
        self.embedDimensions = embedDimensions
        self.vocabularySize = vocabularySize
        self.contextLength = contextLength
        self.textWidth = textWidth
        self.textLayers = textLayers
        self.textHeads = textHeads
    }

    /// ViT-B/32, the common base model.
    public static let base = NFKMLXCLIPConfiguration()

    /// A small configuration for offline structure and round-trip tests.
    public static let tiny = NFKMLXCLIPConfiguration(imageResolution: 16, patchSize: 8, visionWidth: 16,
                                                     visionLayers: 1, visionHeads: 2, embedDimensions: 8,
                                                     vocabularySize: 64, contextLength: 12, textWidth: 16,
                                                     textLayers: 1, textHeads: 2)
}

/// Multi-head attention with the reference's fused input projection: a single `in_proj_weight`
/// (`[3·dim, dim]`) and `in_proj_bias`, then a separate `out_proj`.
final class NFKCLIPAttention: Module {

    @ModuleInfo(key: "in_proj_weight") var inProjWeight: MLXArray
    @ModuleInfo(key: "in_proj_bias") var inProjBias: MLXArray
    @ModuleInfo(key: "out_proj") var outProj: Linear
    private let heads: Int

    init(dimensions: Int, heads: Int) {
        self.heads = heads
        _inProjWeight.wrappedValue = NFKCLIPInit.parameter([3 * dimensions, dimensions])
        _inProjBias.wrappedValue = MLXArray.zeros([3 * dimensions])
        _outProj.wrappedValue = Linear(dimensions, dimensions)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        let (batch, tokens, dimensions) = (x.shape[0], x.shape[1], x.shape[2])
        let headDim = dimensions / heads
        let scale = 1.0 / sqrtf(Float(headDim))

        let projected = x.matmul(inProjWeight.transposed(1, 0)) + inProjBias   // [B, L, 3D]
        let parts = split(projected, parts: 3, axis: -1)
        let shaped = { (t: MLXArray) in
            t.reshaped([batch, tokens, self.heads, headDim]).transposed(0, 2, 1, 3)
                .reshaped([batch * self.heads, tokens, headDim])
        }
        let q = shaped(parts[0])
        let k = shaped(parts[1])
        let v = shaped(parts[2])

        var scores = q.matmul(k.transposed(0, 2, 1)) * scale
        if let mask { scores = scores + mask }
        let attended = softmax(scores, axis: -1).matmul(v)
            .reshaped([batch, heads, tokens, headDim]).transposed(0, 2, 1, 3).reshaped([batch, tokens, dimensions])
        return outProj(attended)
    }
}

/// The activation inside a CLIP residual block. OpenAI's CLIP and the Stable Diffusion 1.x text
/// encoder use QuickGELU; the OpenCLIP towers SD 2.x and SDXL condition on use plain GELU.
public enum NFKCLIPActivation: Sendable {
    case quickGELU
    case gelu

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        switch self {
        case .quickGELU: return NFKCLIPInit.quickGELU(x)
        // The reference's `gelu` is the exact error-function form, not the tanh approximation its
        // neighbour `gelu_new` selects.
        case .gelu: return MLXNN.gelu(x)
        }
    }
}

/// The MLP inside a residual block: `c_fc` up to the intermediate width, an activation, `c_proj` back.
final class NFKCLIPMLP: Module {
    @ModuleInfo(key: "c_fc") var cFc: Linear
    @ModuleInfo(key: "c_proj") var cProj: Linear
    let activation: NFKCLIPActivation

    init(dimensions: Int, intermediate: Int? = nil, activation: NFKCLIPActivation = .quickGELU) {
        self.activation = activation
        _cFc.wrappedValue = Linear(dimensions, intermediate ?? dimensions * 4)
        _cProj.wrappedValue = Linear(intermediate ?? dimensions * 4, dimensions)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        cProj(activation(cFc(x)))
    }
}

/// A residual attention block: pre-norm self-attention, then pre-norm MLP.
final class NFKCLIPResidualAttentionBlock: Module {
    @ModuleInfo(key: "ln_1") var ln1: LayerNorm
    @ModuleInfo(key: "attn") var attn: NFKCLIPAttention
    @ModuleInfo(key: "ln_2") var ln2: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: NFKCLIPMLP

    init(dimensions: Int, heads: Int, intermediate: Int? = nil,
         activation: NFKCLIPActivation = .quickGELU) {
        _ln1.wrappedValue = LayerNorm(dimensions: dimensions)
        _attn.wrappedValue = NFKCLIPAttention(dimensions: dimensions, heads: heads)
        _ln2.wrappedValue = LayerNorm(dimensions: dimensions)
        _mlp.wrappedValue = NFKCLIPMLP(dimensions: dimensions, intermediate: intermediate, activation: activation)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        let h = x + attn(ln1(x), mask: mask)
        return h + mlp(ln2(h))
    }
}

/// A stack of residual attention blocks, named `resblocks` to match the reference.
final class NFKCLIPTransformer: Module {
    @ModuleInfo(key: "resblocks") var resblocks: [NFKCLIPResidualAttentionBlock]

    init(width: Int, layers: Int, heads: Int, intermediate: Int? = nil,
         activation: NFKCLIPActivation = .quickGELU) {
        _resblocks.wrappedValue = (0 ..< layers).map { _ in
            NFKCLIPResidualAttentionBlock(dimensions: width, heads: heads, intermediate: intermediate,
                                          activation: activation)
        }
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
        var h = x
        for block in resblocks {
            h = block(h, mask: mask)
        }
        return h
    }
}

/// The image tower: patch-embedding convolution, class token, positional embedding, transformer, and
/// a final projection to the shared embedding dimension.
final class NFKCLIPVisionTransformer: Module {
    @ModuleInfo(key: "conv1") var conv1: Conv2d
    @ModuleInfo(key: "class_embedding") var classEmbedding: MLXArray
    @ModuleInfo(key: "positional_embedding") var positionalEmbedding: MLXArray
    @ModuleInfo(key: "ln_pre") var lnPre: LayerNorm
    @ModuleInfo(key: "transformer") var transformer: NFKCLIPTransformer
    @ModuleInfo(key: "ln_post") var lnPost: LayerNorm
    @ModuleInfo(key: "proj") var proj: MLXArray

    private let resolution: Int

    init(_ c: NFKMLXCLIPConfiguration) {
        resolution = c.imageResolution
        let grid = c.imageResolution / c.patchSize
        _conv1.wrappedValue = Conv2d(inputChannels: 3, outputChannels: c.visionWidth,
                                     kernelSize: IntOrPair(c.patchSize), stride: IntOrPair(c.patchSize), bias: false)
        _classEmbedding.wrappedValue = NFKCLIPInit.parameter([c.visionWidth])
        _positionalEmbedding.wrappedValue = NFKCLIPInit.parameter([grid * grid + 1, c.visionWidth])
        _lnPre.wrappedValue = LayerNorm(dimensions: c.visionWidth)
        _transformer.wrappedValue = NFKCLIPTransformer(width: c.visionWidth, layers: c.visionLayers, heads: c.visionHeads)
        _lnPost.wrappedValue = LayerNorm(dimensions: c.visionWidth)
        _proj.wrappedValue = NFKCLIPInit.parameter([c.visionWidth, c.embedDimensions])
    }

    /// Encodes a bridged image `[H, W, 3]` (`0...1`) into an L2-normalized embedding `[embedDimensions]`.
    func encode(_ image: MLXArray) -> MLXArray {
        let batched = image.reshaped([1, image.shape[0], image.shape[1], image.shape[2]])
        let resized = NFKMLXResample.resizeNearest(batched, height: resolution, width: resolution)
        let normalized = (resized - NFKCLIPInit.imageMean) / NFKCLIPInit.imageStd

        let patches = conv1(normalized)                                     // [1, grid, grid, width]
        let width = patches.shape[3]
        var x = patches.reshaped([1, patches.shape[1] * patches.shape[2], width])
        let classToken = classEmbedding.reshaped([1, 1, width])
        x = concatenated([classToken, x], axis: 1)                          // prepend the class token
        x = x + positionalEmbedding
        x = lnPre(x)
        x = transformer(x, mask: nil)
        let pooled = lnPost(x[0..., 0])                                     // the class token
        let embedding = pooled.matmul(proj)
        return NFKCLIPInit.l2Normalize(embedding).reshaped([proj.shape[1]])
    }
}

/// The full CLIP model: the image tower under `visual`, and the text tower at the top level.
///
/// Both towers stay frozen for a linear probe, so a consumer's own classifier trains on cached
/// embeddings from ``NFKMLXCLIP/embeddings(for:using:colorSpace:)``.
public final class NFKMLXCLIPNet: Module {
    @ModuleInfo(key: "visual") var visual: NFKCLIPVisionTransformer
    @ModuleInfo(key: "token_embedding") var tokenEmbedding: Embedding
    @ModuleInfo(key: "positional_embedding") var positionalEmbedding: MLXArray
    @ModuleInfo(key: "transformer") var transformer: NFKCLIPTransformer
    @ModuleInfo(key: "ln_final") var lnFinal: LayerNorm
    @ModuleInfo(key: "text_projection") var textProjection: MLXArray
    @ModuleInfo(key: "logit_scale") var logitScale: MLXArray

    let configuration: NFKMLXCLIPConfiguration

    init(_ configuration: NFKMLXCLIPConfiguration) {
        self.configuration = configuration
        _visual.wrappedValue = NFKCLIPVisionTransformer(configuration)
        _tokenEmbedding.wrappedValue = Embedding(embeddingCount: configuration.vocabularySize, dimensions: configuration.textWidth)
        _positionalEmbedding.wrappedValue = NFKCLIPInit.parameter([configuration.contextLength, configuration.textWidth])
        _transformer.wrappedValue = NFKCLIPTransformer(width: configuration.textWidth, layers: configuration.textLayers, heads: configuration.textHeads)
        _lnFinal.wrappedValue = LayerNorm(dimensions: configuration.textWidth)
        _textProjection.wrappedValue = NFKCLIPInit.parameter([configuration.textWidth, configuration.embedDimensions])
        _logitScale.wrappedValue = MLXArray(logf(1.0 / 0.07))
    }

    /// Encodes an image `[H, W, 3]` in `0...1` into an L2-normalized embedding.
    public func encodeImage(_ image: MLXArray) -> MLXArray {
        visual.encode(image)
    }

    /// Encodes token ids into an L2-normalized embedding. The pooled feature is the position of the
    /// highest token id (the end-of-text token, the reference's pooling rule).
    func encodeText(_ tokenIds: [Int]) -> MLXArray {
        let length = tokenIds.count
        let tokens = MLXArray(tokenIds.map { Int32($0) }).reshaped([1, length])
        var x = tokenEmbedding(tokens) + positionalEmbedding[0 ..< length]
        x = transformer(x, mask: NFKCLIPInit.causalMask(length))
        x = lnFinal(x)
        let endIndex = tokenIds.firstIndex(of: tokenIds.max() ?? 0) ?? (length - 1)
        let pooled = x[0, endIndex].reshaped([1, configuration.textWidth])
        let embedding = pooled.matmul(textProjection)
        return NFKCLIPInit.l2Normalize(embedding).reshaped([textProjection.shape[1]])
    }
}

/// Shared initialization, normalization, and small building blocks for CLIP.
enum NFKCLIPInit {

    /// CLIP's image normalization constants, shaped for NHWC broadcasting.
    static let imageMean: MLXArray = [Float(0.48145466), 0.4578275, 0.40821073]
        .withUnsafeBufferPointer { MLXArray($0, [1, 1, 1, 3]) }
    static let imageStd: MLXArray = [Float(0.26862954), 0.26130258, 0.27577711]
        .withUnsafeBufferPointer { MLXArray($0, [1, 1, 1, 3]) }

    /// A deterministic small-magnitude parameter, so a random-weights model runs without a non-zero
    /// requirement on a random source. A loaded checkpoint overwrites it.
    static func parameter(_ shape: [Int]) -> MLXArray {
        let count = shape.reduce(1, *)
        var values = [Float](repeating: 0, count: count)
        for i in 0 ..< count {
            values[i] = (Float((i * 733) % 2003) / 2003.0 - 0.5) * 0.04       // roughly ±0.02, varied
        }
        return values.withUnsafeBufferPointer { MLXArray($0, shape) }
    }

    static func quickGELU(_ x: MLXArray) -> MLXArray {
        x * sigmoid(x * 1.702)
    }

    static func l2Normalize(_ x: MLXArray) -> MLXArray {
        x / sqrt(sum(x * x, axis: -1, keepDims: true))
    }

    static func causalMask(_ length: Int) -> MLXArray {
        var values = [Float](repeating: 0, count: length * length)
        for i in 0 ..< length {
            for j in (i + 1) ..< length {
                values[i * length + j] = -1e9
            }
        }
        return values.withUnsafeBufferPointer { MLXArray($0, [length, length]) }
    }
}

/// Holds the network for capture in the backend's `@Sendable` closures.
private final class NFKCLIPHolder: @unchecked Sendable {
    let net: NFKMLXCLIPNet
    init(_ net: NFKMLXCLIPNet) { self.net = net }
}

/// A CLIP embedding backend. An image under `NFKInputImage` encodes to an embedding under
/// `NFKOutputEmbedding`. A text prompt encodes when a `tokenizer` is supplied (text → token ids); the
/// byte-level-BPE vocabulary is a load-time artifact, so a caller that tokenizes elsewhere passes the
/// ids through `encodeText`.
@objc(NFKMLXCLIPBackend)
public final class NFKMLXCLIPBackend: NSObject, NFKInferenceBackend {

    private let holder: NFKCLIPHolder
    private let tokenizer: (@Sendable (String) -> [Int])?
    private let identifier: String

    init(net: NFKMLXCLIPNet, identifier: String, tokenizer: (@Sendable (String) -> [Int])? = nil) {
        holder = NFKCLIPHolder(net)
        self.identifier = identifier
        self.tokenizer = tokenizer
        super.init()
    }

    @objc public var isReady: Bool { true }
    @objc public var backendIdentifier: String { identifier }

    @objc(runInferenceForRequest:error:)
    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        let job = submitInferenceJob(for: request)
        let semaphore = DispatchSemaphore(value: 0)
        job.completionHandler = { _ in semaphore.signal() }
        semaphore.wait()
        if let result = job.result {
            return result
        }
        if let error = job.error {
            throw error
        }
        throw NFKMLXError.noOutput
    }

    @objc(submitInferenceJobForRequest:)
    public func submitInferenceJob(for request: NFKInferenceRequest) -> NFKInferenceJob {
        let job = NFKInferenceJob()
        let holder = self.holder
        let tokenizer = self.tokenizer
        Task.detached(priority: .userInitiated) {
            do {
                let embedding = try NFKMLXCLIPBackend.embed(request, net: holder.net, tokenizer: tokenizer)
                job.finish(with: NFKInferenceResult(outputs: [NFKOutputEmbedding: embedding]))
            } catch {
                job.finish(withError: error as NSError)
            }
        }
        return job
    }

    private static func embed(_ request: NFKInferenceRequest, net: NFKMLXCLIPNet,
                              tokenizer: (@Sendable (String) -> [Int])?) throws -> [NSNumber] {
        if let value = request.input(forKey: NFKInputImage) {
            let image = try NFKMLXImageBridge.tensor(from: value, channels: 3, colorSpace: CGColorSpaceCreateDeviceRGB())
            return numbers(net.encodeImage(image))
        }
        if let prompt = request.prompt, let tokenizer {
            return numbers(net.encodeText(tokenizer(prompt)))
        }
        throw NFKMLXError.unsupportedInput
    }

    private static func numbers(_ embedding: MLXArray) -> [NSNumber] {
        eval(embedding)
        return embedding.asArray(Float.self).map { NSNumber(value: $0) }
    }
}

/// CLIP as an InferKit backend, and its registration for the Objective-C path.
///
/// `NFKMLXCLIPNet` is the real CLIP model, not a stand-in. Random weights run (proving the pipeline);
/// a trained checkpoint makes the embeddings meaningful. Load a **safetensors** checkpoint whose
/// parameter names match the reference OpenAI CLIP (`visual.conv1.weight`, `visual.transformer.resblocks.N.*`,
/// `token_embedding.weight`, `text_projection`); the loader transposes the 4-D patch-embedding
/// convolution weight `[out, in, kH, kW]` to MLX's `[out, kH, kW, in]`.
@objc(NFKMLXCLIP)
public final class NFKMLXCLIP: NSObject {

    /// The registry name the model builds under.
    @objc public static let modelName = "clip-vit-b-32"

    /// Builds a CLIP image-embedding backend directly from optional local weights — no registry
    /// required. A nil `weightsURL` builds random weights (`isReady` is true). The text path needs a
    /// tokenizer, so this factory serves image embedding; a Swift caller adds a tokenizer through
    /// `NFKMLXCLIPBackend`. Run inference off the render thread.
    @objc(backendWithWeightsURL:error:)
    public static func backend(weightsURL: URL?) throws -> any NFKInferenceBackend {
        let net = NFKMLXCLIPNet(.base)
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        return NFKMLXCLIPBackend(net: net, identifier: modelName)
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

    /// Registers CLIP (`clip-vit-b-32`) with `NFKMLXModelRegistry`, delegating to `backend(weightsURL:)`.
    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName) { weightsURL in
            try backend(weightsURL: weightsURL)
        }
    }

    /// Loads a safetensors checkpoint into `net`, transposing the 4-D patch-embedding convolution
    /// weight from PyTorch's `[out, in, kH, kW]` to MLX's channels-last `[out, kH, kW, in]`.
    static func loadWeights(into net: NFKMLXCLIPNet, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let raw = checkpoint.arrays
        let mapped = raw.map { key, value in
            (key, checkpoint.needsConvTranspose && value.ndim == 4 ? value.transposed(0, 2, 3, 1) : value)
        }
        try NFKMLXWeights.apply(mapped, to: net)
    }
}
