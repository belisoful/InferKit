//
//  NFKMLXSigLIP2.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXFast
import MLXNN

// SigLIP 2: a contrastive image-text model, the CLIP upgrade, and the vision tower a VLM reads. The
// vision and text towers are the same transformer the SmolVLM SigLIP encoder uses (`NFKSigLIPLayer` and
// friends are reused); SigLIP 2 adds an attention-pooling head over the vision patches, a text tower over
// a 256k multilingual vocabulary, and a learned logit scale and bias for the sigmoid similarity.
//
// The image embedding is the pooling head's output; the text embedding is the last token's hidden state
// through a projection. Both are L2-normalized, and `logit = scale·(text·image) + bias`.

/// Text-tower dimensions for SigLIP 2.
public struct NFKMLXSigLIP2TextConfiguration: Sendable {
    public var hiddenSize: Int
    public var layerCount: Int
    public var headCount: Int
    public var intermediateSize: Int
    public var vocabularySize: Int
    public var maxPositions: Int
    public var layerNormEpsilon: Float

    public init(hiddenSize: Int = 768, layerCount: Int = 12, headCount: Int = 12, intermediateSize: Int = 3072,
                vocabularySize: Int = 256000, maxPositions: Int = 64, layerNormEpsilon: Float = 1e-6) {
        self.hiddenSize = hiddenSize
        self.layerCount = layerCount
        self.headCount = headCount
        self.intermediateSize = intermediateSize
        self.vocabularySize = vocabularySize
        self.maxPositions = maxPositions
        self.layerNormEpsilon = layerNormEpsilon
    }

    /// The `NFKMLXSigLIPConfiguration` the shared encoder layers take (the text tower has no patches).
    var encoderConfiguration: NFKMLXSigLIPConfiguration {
        NFKMLXSigLIPConfiguration(hiddenSize: hiddenSize, layerCount: layerCount, headCount: headCount,
                                  intermediateSize: intermediateSize, layerNormEpsilon: layerNormEpsilon)
    }
}

/// SigLIP 2 geometry: the vision configuration, the text configuration, and the image resolution.
public struct NFKMLXSigLIP2Configuration: Sendable {
    public var vision: NFKMLXSigLIPConfiguration
    public var text: NFKMLXSigLIP2TextConfiguration

    public init(vision: NFKMLXSigLIPConfiguration = NFKMLXSigLIPConfiguration(imageSize: 224),
                text: NFKMLXSigLIP2TextConfiguration = NFKMLXSigLIP2TextConfiguration()) {
        self.vision = vision
        self.text = text
    }

    /// The released `siglip2-base-patch16-224`.
    public static let base = NFKMLXSigLIP2Configuration()

    /// A small configuration for weight-free tests.
    public static let tiny = NFKMLXSigLIP2Configuration(
        vision: NFKMLXSigLIPConfiguration(hiddenSize: 32, layerCount: 2, headCount: 2, intermediateSize: 64,
                                          patchSize: 16, imageSize: 64),
        text: NFKMLXSigLIP2TextConfiguration(hiddenSize: 32, layerCount: 2, headCount: 2, intermediateSize: 64,
                                             vocabularySize: 128, maxPositions: 16))
}

/// SigLIP 2's vision patch embedding: a patch convolution plus a learned position embedding read in
/// row-major order (SigLIP 2 does not use SmolVLM's fractional position buckets).
final class NFKSigLIP2VisionEmbeddings: Module {
    @ModuleInfo(key: "patch_embedding") var patchEmbedding: Conv2d
    @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding

    let positionCount: Int

    init(_ c: NFKMLXSigLIPConfiguration) {
        positionCount = c.positionCount
        _patchEmbedding.wrappedValue = Conv2d(inputChannels: 3, outputChannels: c.hiddenSize,
                                              kernelSize: IntOrPair(c.patchSize), stride: IntOrPair(c.patchSize),
                                              bias: true)
        _positionEmbedding.wrappedValue = Embedding(embeddingCount: c.positionCount, dimensions: c.hiddenSize)
    }

    /// `[tiles, H, W, 3]` → `[tiles, patches, hidden]`.
    func callAsFunction(_ pixelValues: MLXArray) -> MLXArray {
        let patches = patchEmbedding(pixelValues)
        let tiles = patches.dim(0)
        let embedded = patches.reshaped([tiles, positionCount, patches.dim(3)])
        return embedded + positionEmbedding.weight
    }
}

/// `nn.MultiheadAttention` with a fused query/key/value projection: a probe query attends over the patch
/// features. `attention` is a real submodule so the checkpoint's dotted keys (`attention.in_proj_weight`,
/// `attention.out_proj.*`) nest correctly.
final class NFKSigLIP2ProbeAttention: Module {
    @ParameterInfo(key: "in_proj_weight") var inProjWeight: MLXArray
    @ParameterInfo(key: "in_proj_bias") var inProjBias: MLXArray
    @ModuleInfo(key: "out_proj") var outProj: Linear

    let heads: Int
    let headDimensions: Int

    init(_ c: NFKMLXSigLIPConfiguration) {
        heads = c.headCount
        headDimensions = c.headDimensions
        _inProjWeight.wrappedValue = MLXArray.zeros([3 * c.hiddenSize, c.hiddenSize])
        _inProjBias.wrappedValue = MLXArray.zeros([3 * c.hiddenSize])
        _outProj.wrappedValue = Linear(c.hiddenSize, c.hiddenSize, bias: true)
    }

    /// Query `[B, 1, hidden]`, key/value `[B, N, hidden]` → `[B, 1, hidden]`.
    func callAsFunction(_ query: MLXArray, _ keyValue: MLXArray) -> MLXArray {
        let (batch, length, hidden) = (keyValue.shape[0], keyValue.shape[1], keyValue.shape[2])
        let q = matmul(query, inProjWeight[0 ..< hidden, 0...].transposed(1, 0)) + inProjBias[0 ..< hidden]
        let k = matmul(keyValue, inProjWeight[hidden ..< 2 * hidden, 0...].transposed(1, 0)) + inProjBias[hidden ..< 2 * hidden]
        let v = matmul(keyValue, inProjWeight[2 * hidden ..< 3 * hidden, 0...].transposed(1, 0)) + inProjBias[2 * hidden ..< 3 * hidden]

        func split(_ t: MLXArray, _ len: Int) -> MLXArray {
            t.reshaped([batch, len, self.heads, headDimensions]).transposed(0, 2, 1, 3)
        }
        let attended = MLXFast.scaledDotProductAttention(
            queries: split(q, 1), keys: split(k, length), values: split(v, length),
            scale: 1 / sqrt(Float(headDimensions)), mask: nil)
        return outProj(attended.transposed(0, 2, 1, 3).reshaped([batch, 1, hidden]))
    }
}

/// The multi-head attention pooling head: a learned probe token attends over the patch features, then a
/// residual layer-norm and feed-forward. Its pooled output is the image embedding.
final class NFKSigLIP2PoolingHead: Module {
    @ParameterInfo(key: "probe") var probe: MLXArray
    @ModuleInfo(key: "attention") var attention: NFKSigLIP2ProbeAttention
    @ModuleInfo(key: "layernorm") var layerNorm: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: NFKSigLIPMLP

    init(_ c: NFKMLXSigLIPConfiguration) {
        _probe.wrappedValue = MLXArray.zeros([1, 1, c.hiddenSize])
        _attention.wrappedValue = NFKSigLIP2ProbeAttention(c)
        _layerNorm.wrappedValue = LayerNorm(dimensions: c.hiddenSize, eps: c.layerNormEpsilon)
        _mlp.wrappedValue = NFKSigLIPMLP(c)
    }

    /// Patch features `[B, N, hidden]` → the pooled embedding `[B, hidden]`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let pooled = attention(broadcast(probe, to: [x.shape[0], 1, x.shape[2]]), x)
        return (pooled + mlp(layerNorm(pooled)))[0..., 0, 0...]                     // [B, hidden]
    }
}

/// The SigLIP 2 vision tower: patch embedding, the shared encoder, a post layer-norm, and the pooling
/// head. Returns the pooled image embedding.
final class NFKSigLIP2VisionNet: Module {
    @ModuleInfo(key: "embeddings") var embeddings: NFKSigLIP2VisionEmbeddings
    @ModuleInfo(key: "encoder") var encoder: NFKSigLIPEncoder
    @ModuleInfo(key: "post_layernorm") var postLayerNorm: LayerNorm
    @ModuleInfo(key: "head") var head: NFKSigLIP2PoolingHead

    init(_ c: NFKMLXSigLIPConfiguration) {
        _embeddings.wrappedValue = NFKSigLIP2VisionEmbeddings(c)
        _encoder.wrappedValue = NFKSigLIPEncoder(c)
        _postLayerNorm.wrappedValue = LayerNorm(dimensions: c.hiddenSize, eps: c.layerNormEpsilon)
        _head.wrappedValue = NFKSigLIP2PoolingHead(c)
    }

    func callAsFunction(_ pixelValues: MLXArray) -> MLXArray {
        var hidden = embeddings(pixelValues)
        for layer in encoder.layers { hidden = layer(hidden) }
        return head(postLayerNorm(hidden))
    }
}

/// The SigLIP 2 text tower: token and position embeddings, the shared encoder, a final layer-norm, and a
/// projection over the last token. Returns the text embedding.
final class NFKSigLIP2TextNet: Module {
    @ModuleInfo(key: "embeddings") var embeddings: NFKSigLIP2TextEmbeddings
    @ModuleInfo(key: "encoder") var encoder: NFKSigLIPEncoder
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm
    @ModuleInfo(key: "head") var head: Linear

    init(_ c: NFKMLXSigLIP2TextConfiguration) {
        _embeddings.wrappedValue = NFKSigLIP2TextEmbeddings(c)
        _encoder.wrappedValue = NFKSigLIPEncoder(c.encoderConfiguration)
        _finalLayerNorm.wrappedValue = LayerNorm(dimensions: c.hiddenSize, eps: c.layerNormEpsilon)
        _head.wrappedValue = Linear(c.hiddenSize, c.hiddenSize, bias: true)
    }

    /// Token ids `[B, T]` → the text embedding `[B, hidden]`. SigLIP pools the LAST position.
    func callAsFunction(_ tokens: MLXArray) -> MLXArray {
        var hidden = embeddings(tokens)
        for layer in encoder.layers { hidden = layer(hidden) }
        hidden = finalLayerNorm(hidden)
        return head(hidden[0..., -1, 0...])
    }
}

/// SigLIP 2's text embedding: a token embedding plus a learned position embedding.
final class NFKSigLIP2TextEmbeddings: Module {
    @ModuleInfo(key: "token_embedding") var tokenEmbedding: Embedding
    @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding

    init(_ c: NFKMLXSigLIP2TextConfiguration) {
        _tokenEmbedding.wrappedValue = Embedding(embeddingCount: c.vocabularySize, dimensions: c.hiddenSize)
        _positionEmbedding.wrappedValue = Embedding(embeddingCount: c.maxPositions, dimensions: c.hiddenSize)
    }

    func callAsFunction(_ tokens: MLXArray) -> MLXArray {
        tokenEmbedding(tokens) + positionEmbedding.weight[0 ..< tokens.shape[1]]
    }
}

/// The SigLIP 2 model: the vision and text towers and the learned logit scale and bias.
final class NFKMLXSigLIP2Net: Module {
    @ModuleInfo(key: "vision") var vision: NFKSigLIP2VisionNet
    @ModuleInfo(key: "text") var text: NFKSigLIP2TextNet
    @ParameterInfo(key: "logit_scale") var logitScale: MLXArray
    @ParameterInfo(key: "logit_bias") var logitBias: MLXArray

    let configuration: NFKMLXSigLIP2Configuration

    init(_ c: NFKMLXSigLIP2Configuration) {
        configuration = c
        _vision.wrappedValue = NFKSigLIP2VisionNet(c.vision)
        _text.wrappedValue = NFKSigLIP2TextNet(c.text)
        _logitScale.wrappedValue = MLXArray.zeros([1])
        _logitBias.wrappedValue = MLXArray.zeros([1])
    }

    /// The L2-normalized image embedding for `[tiles, H, W, 3]` pixel values.
    func imageEmbedding(_ pixelValues: MLXArray) -> MLXArray {
        let embedding = vision(pixelValues)
        return embedding / sqrt(embedding.square().sum(axis: -1, keepDims: true))
    }

    /// The L2-normalized text embedding for token ids `[B, T]`.
    func textEmbedding(_ tokens: MLXArray) -> MLXArray {
        let embedding = text(tokens)
        return embedding / sqrt(embedding.square().sum(axis: -1, keepDims: true))
    }

    /// The sigmoid-similarity logits per text: `scale·(text · image) + bias`.
    func logits(image: MLXArray, text tokens: MLXArray) -> MLXArray {
        let imageEmbeds = imageEmbedding(image)
        let textEmbeds = textEmbedding(tokens)
        return matmul(textEmbeds, imageEmbeds.transposed(1, 0)) * exp(logitScale) + logitBias
    }
}

/// Holds the network for capture in the backend's `@Sendable` closure.
private final class NFKSigLIP2Holder: @unchecked Sendable {
    let net: NFKMLXSigLIP2Net
    init(_ net: NFKMLXSigLIP2Net) { self.net = net }
}

/// SigLIP 2 image embedding as an InferKit backend: reads `NFKInputImage`, returns the L2-normalized
/// image embedding under `NFKOutputEmbedding`. Text embeddings are reached through `NFKMLXSigLIP2`.
@objc(NFKMLXSigLIP2Backend)
public final class NFKMLXSigLIP2Backend: NSObject, NFKInferenceBackend {

    private let holder: NFKSigLIP2Holder
    private let identifier: String

    init(net: NFKMLXSigLIP2Net, identifier: String) {
        holder = NFKSigLIP2Holder(net)
        self.identifier = identifier
        super.init()
    }

    @objc public var isReady: Bool { true }
    @objc public var backendIdentifier: String { identifier }

    @objc(runInferenceForRequest:error:)
    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        guard let value = request.input(forKey: NFKInputImage), CFGetTypeID(value as CFTypeRef) == CGImage.typeID else {
            throw NFKMLXError.unsupportedInput
        }
        let pixels = try NFKMLXSigLIP2.pixelValues(from: value as! CGImage, imageSize: holder.net.configuration.vision.imageSize)
        let embedding = holder.net.imageEmbedding(pixels)
        eval(embedding)
        let numbers = embedding.reshaped([-1]).asArray(Float.self).map { NSNumber(value: $0) }
        return NFKInferenceResult(outputs: [NFKOutputEmbedding: numbers])
    }

    @objc(submitInferenceJobForRequest:)
    public func submitInferenceJob(for request: NFKInferenceRequest) -> NFKInferenceJob {
        let job = NFKInferenceJob()
        Task.detached(priority: .userInitiated) {
            do {
                job.finish(with: try self.runInference(for: request))
            } catch {
                job.finish(withError: error as NSError)
            }
        }
        return job
    }
}

/// Registration, embeddings, and weight loading for SigLIP 2.
@objc(NFKMLXSigLIP2)
public final class NFKMLXSigLIP2: NSObject {

    /// The registry name the model builds under.
    @objc public static let modelName = "siglip2-base-patch16-224"

    private let holder: NFKSigLIP2Holder

    init(net: NFKMLXSigLIP2Net) { holder = NFKSigLIP2Holder(net) }

    static func makeNet(_ configuration: NFKMLXSigLIP2Configuration = .base) -> NFKMLXSigLIP2Net {
        NFKMLXSigLIP2Net(configuration)
    }

    /// The L2-normalized image embedding for a `CGImage`.
    public func imageEmbedding(_ image: CGImage) throws -> [Float] {
        let pixels = try Self.pixelValues(from: image, imageSize: holder.net.configuration.vision.imageSize)
        let embedding = holder.net.imageEmbedding(pixels)
        eval(embedding)
        return embedding.reshaped([-1]).asArray(Float.self)
    }

    /// The L2-normalized text embedding for token ids (padded to the tower's context length by the
    /// caller, as SigLIP's tokenizer does).
    public func textEmbedding(tokens: [Int]) -> [Float] {
        let ids = tokens.map(Int32.init).withUnsafeBufferPointer { MLXArray($0, [1, tokens.count]) }
        let embedding = holder.net.textEmbedding(ids)
        eval(embedding)
        return embedding.reshaped([-1]).asArray(Float.self)
    }

    /// `CGImage` → `[1, imageSize, imageSize, 3]` pixel values in `-1…1`, SigLIP's normalization.
    static func pixelValues(from image: CGImage, imageSize: Int) throws -> MLXArray {
        let rgb = try NFKMLXImageBridge.tensor(from: image, channels: 3, colorSpace: CGColorSpaceCreateDeviceRGB())
        let batched = rgb.reshaped([1, rgb.shape[0], rgb.shape[1], rgb.shape[2]])                // [1, H, W, 3] in 0…1
        let resized = NFKMLXResample.resizeBilinear(batched, height: imageSize, width: imageSize)
        return resized * 2 - 1
    }

    /// Builds a SigLIP 2 backend directly from optional local weights — no registry required.
    ///
    /// - Since: InferKit 0.3.1
    @objc(backendWithWeightsURL:error:)
    public static func backend(weightsURL: URL?) throws -> any NFKInferenceBackend {
        NFKMLXSigLIP2Backend(net: try loadedNet(.base, weightsURL: weightsURL), identifier: modelName)
    }

    /// Builds a SigLIP 2 model object (with `imageEmbedding`/`textEmbedding`) from optional local weights.
    public static func model(configuration: NFKMLXSigLIP2Configuration = .base, weightsURL: URL?) throws -> NFKMLXSigLIP2 {
        NFKMLXSigLIP2(net: try loadedNet(configuration, weightsURL: weightsURL))
    }

    private static func loadedNet(_ configuration: NFKMLXSigLIP2Configuration, weightsURL: URL?) throws -> NFKMLXSigLIP2Net {
        let net = makeNet(configuration)
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        return net
    }

    /// Downloads the checkpoint from Hugging Face, then builds — no registry required.
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

    /// Registers SigLIP 2 with `NFKMLXModelRegistry`, delegating to `backend(weightsURL:)`.
    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName) { weightsURL in try backend(weightsURL: weightsURL) }
    }

    /// Loads a checkpoint, transposing 4-D Conv2d weights `[out, in, kH, kW]` → MLX's `[out, kH, kW, in]`
    /// and mapping the reference's `vision_model.`/`text_model.` prefixes onto the module's `vision`/`text`.
    /// Linear and embedding weights and the pooling head's fused projection are 2-D and pass through.
    static func loadWeights(into net: NFKMLXSigLIP2Net, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let transpose = checkpoint.needsConvTranspose
        let mapped = checkpoint.arrays.map { key, value -> (String, MLXArray) in
            let name = remapReferenceKey(key)
            return (transpose && value.ndim == 4) ? (name, value.transposed(0, 2, 3, 1)) : (name, value)
        }
        try NFKMLXWeights.apply(mapped, to: net)
    }

    /// Maps the reference's `vision_model.`/`text_model.` prefixes onto the module's `vision`/`text`; the
    /// top-level `logit_scale`/`logit_bias` already match.
    static func remapReferenceKey(_ key: String) -> String {
        if key.hasPrefix("vision_model.") { return "vision." + key.dropFirst("vision_model.".count) }
        if key.hasPrefix("text_model.") { return "text." + key.dropFirst("text_model.".count) }
        return key
    }
}
