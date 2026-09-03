//
//  NFKMLXSmolVLM.swift
//  InferKitMLX
//
//  SmolVLM2-500M: the package's first vision-language model, and its first path from an image and a
//  question to text. A VLM is three parts: a vision encoder that turns an image into feature tokens, a
//  connector that projects those tokens to the decoder's width, and a language decoder that reads the
//  text with the image tokens spliced in and answers.
//
//  SmolVLM2 is a SigLIP vision encoder, a pixel-shuffle connector, and a Llama decoder. The decoder is
//  the dense stack `NFKMLXLanguageNet` already runs (`model_type` llama), so only the vision encoder and
//  the connector are new here; the fusion splices the projected vision tokens into the decoder's input
//  embeddings at the image-token positions.
//

import Foundation
import CoreGraphics
import InferKit
import MLX
import MLXFast
import MLXNN

/// The geometry of the SigLIP vision encoder SmolVLM uses.
public struct NFKMLXSigLIPConfiguration: Sendable {
    public var hiddenSize: Int
    public var layerCount: Int
    public var headCount: Int
    public var intermediateSize: Int
    public var patchSize: Int
    public var imageSize: Int
    public var layerNormEpsilon: Float

    public init(hiddenSize: Int = 768, layerCount: Int = 12, headCount: Int = 12,
                intermediateSize: Int = 3072, patchSize: Int = 16, imageSize: Int = 512,
                layerNormEpsilon: Float = 1e-6) {
        self.hiddenSize = hiddenSize
        self.layerCount = layerCount
        self.headCount = headCount
        self.intermediateSize = intermediateSize
        self.patchSize = patchSize
        self.imageSize = imageSize
        self.layerNormEpsilon = layerNormEpsilon
    }

    /// The released SmolVLM2-500M vision geometry.
    public static let smolVLM = NFKMLXSigLIPConfiguration()

    /// A small configuration for tests.
    public static let tiny = NFKMLXSigLIPConfiguration(
        hiddenSize: 32, layerCount: 2, headCount: 2, intermediateSize: 64, patchSize: 16, imageSize: 64)

    var grid: Int { imageSize / patchSize }
    var positionCount: Int { grid * grid }
    var headDimensions: Int { hiddenSize / headCount }
}

/// SigLIP's patch embedding: a convolution over 16×16 patches plus a learned position embedding. There
/// is no class token.
final class NFKSigLIPEmbeddings: Module {
    @ModuleInfo(key: "patch_embedding") var patchEmbedding: Conv2d
    @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding

    let positionCount: Int
    /// The position id of each patch. SmolVLM does not read the row-major grid straight: it buckets a
    /// patch's fractional coordinate against `1/side … (side-1)/side` with a `1 - 1e-6` factor, so a
    /// full `side × side` tile maps a row to `[0, 0, 1, …, side-2]` rather than `0 … side-1`. Every tile
    /// here is full resolution, so these ids are fixed. Held as a plain array so a stored `MLXArray`
    /// does not enter the module's parameters.
    private let positionIds: [Int32]

    init(_ c: NFKMLXSigLIPConfiguration) {
        positionCount = c.positionCount
        _patchEmbedding.wrappedValue = Conv2d(inputChannels: 3, outputChannels: c.hiddenSize,
                                              kernelSize: IntOrPair(c.patchSize),
                                              stride: IntOrPair(c.patchSize), bias: true)
        _positionEmbedding.wrappedValue = Embedding(embeddingCount: c.positionCount, dimensions: c.hiddenSize)

        let side = c.grid
        let boundaries = (1 ..< side).map { Float($0) / Float(side) }
        let bucket = (0 ..< side).map { k -> Int32 in
            let coordinate = Float(k) / Float(side) * (1 - 1e-6)
            return Int32(boundaries.filter { $0 <= coordinate }.count)
        }
        var ids = [Int32]()
        for row in 0 ..< side {
            for column in 0 ..< side {
                ids.append(bucket[row] * Int32(side) + bucket[column])
            }
        }
        positionIds = ids
        super.init()
    }

    /// `pixelValues` is `[tiles, height, width, 3]` (channels last). Returns `[tiles, patches, hidden]`.
    func callAsFunction(_ pixelValues: MLXArray) -> MLXArray {
        let patches = patchEmbedding(pixelValues)                 // [tiles, grid, grid, hidden]
        let tiles = patches.dim(0)
        let embedded = patches.reshaped([tiles, positionCount, patches.dim(3)])
        return embedded + positionEmbedding(MLXArray(positionIds))
    }
}

/// SigLIP attention: separate query, key, value, and output projections, each with a bias.
final class NFKSigLIPAttention: Module {
    @ModuleInfo(key: "q_proj") var queryProjection: Linear
    @ModuleInfo(key: "k_proj") var keyProjection: Linear
    @ModuleInfo(key: "v_proj") var valueProjection: Linear
    @ModuleInfo(key: "out_proj") var outputProjection: Linear

    let heads: Int
    let headDimensions: Int
    let scale: Float

    init(_ c: NFKMLXSigLIPConfiguration) {
        heads = c.headCount
        headDimensions = c.headDimensions
        scale = 1 / sqrt(Float(c.headDimensions))
        _queryProjection.wrappedValue = Linear(c.hiddenSize, c.hiddenSize, bias: true)
        _keyProjection.wrappedValue = Linear(c.hiddenSize, c.hiddenSize, bias: true)
        _valueProjection.wrappedValue = Linear(c.hiddenSize, c.hiddenSize, bias: true)
        _outputProjection.wrappedValue = Linear(c.hiddenSize, c.hiddenSize, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (batch, length) = (x.shape[0], x.shape[1])
        let shape = { (t: MLXArray) in
            t.reshaped([batch, length, self.heads, self.headDimensions]).transposed(0, 2, 1, 3)
        }
        let queries = shape(queryProjection(x))
        let keys = shape(keyProjection(x))
        let values = shape(valueProjection(x))
        let attention = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: nil)
        return outputProjection(attention.transposed(0, 2, 1, 3).reshaped([batch, length, heads * headDimensions]))
    }
}

/// SigLIP's feed-forward: a projection up, the tanh-approximate GELU, a projection back.
final class NFKSigLIPMLP: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    init(_ c: NFKMLXSigLIPConfiguration) {
        _fc1.wrappedValue = Linear(c.hiddenSize, c.intermediateSize, bias: true)
        _fc2.wrappedValue = Linear(c.intermediateSize, c.hiddenSize, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(geluApproximate(fc1(x))) }
}

/// One SigLIP encoder layer: pre-normalized attention and feed-forward, each added back.
final class NFKSigLIPLayer: Module {
    @ModuleInfo(key: "layer_norm1") var norm1: LayerNorm
    @ModuleInfo(key: "self_attn") var attention: NFKSigLIPAttention
    @ModuleInfo(key: "layer_norm2") var norm2: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: NFKSigLIPMLP

    init(_ c: NFKMLXSigLIPConfiguration) {
        _norm1.wrappedValue = LayerNorm(dimensions: c.hiddenSize, eps: c.layerNormEpsilon)
        _attention.wrappedValue = NFKSigLIPAttention(c)
        _norm2.wrappedValue = LayerNorm(dimensions: c.hiddenSize, eps: c.layerNormEpsilon)
        _mlp.wrappedValue = NFKSigLIPMLP(c)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let attended = x + attention(norm1(x))
        return attended + mlp(norm2(attended))
    }
}

/// The `encoder` submodule, a stack of SigLIP layers under the reference's key.
final class NFKSigLIPEncoder: Module {
    @ModuleInfo(key: "layers") var layers: [NFKSigLIPLayer]

    init(_ c: NFKMLXSigLIPConfiguration) {
        _layers.wrappedValue = (0 ..< c.layerCount).map { _ in NFKSigLIPLayer(c) }
        super.init()
    }
}

/// The SigLIP vision encoder. Each image tile becomes a grid of patch features, bidirectionally
/// attended and normalized. For SmolVLM every patch is visible, so there is no patch attention mask.
public final class NFKMLXSigLIPNet: Module {
    @ModuleInfo(key: "embeddings") var embeddings: NFKSigLIPEmbeddings
    @ModuleInfo(key: "encoder") var encoder: NFKSigLIPEncoder
    @ModuleInfo(key: "post_layernorm") var postLayerNorm: LayerNorm

    let configuration: NFKMLXSigLIPConfiguration

    init(_ c: NFKMLXSigLIPConfiguration) {
        configuration = c
        _embeddings.wrappedValue = NFKSigLIPEmbeddings(c)
        _encoder.wrappedValue = NFKSigLIPEncoder(c)
        _postLayerNorm.wrappedValue = LayerNorm(dimensions: c.hiddenSize, eps: c.layerNormEpsilon)
        super.init()
    }

    /// `pixelValues` is `[tiles, height, width, 3]`. Returns each tile's patch features `[tiles, patches, hidden]`.
    func callAsFunction(_ pixelValues: MLXArray) -> MLXArray {
        var hidden = embeddings(pixelValues)
        for layer in encoder.layers {
            hidden = layer(hidden)
        }
        return postLayerNorm(hidden)
    }
}

/// SmolVLM's connector: a pixel shuffle that folds a `scaleFactor × scaleFactor` neighborhood of patches
/// into one token with `scaleFactor²` times the channels, then a bias-free projection to the decoder's
/// width. This is what reduces 1024 patch tokens per tile to 64 vision tokens.
public final class NFKMLXSmolVLMConnector: Module {
    @ModuleInfo(key: "modality_projection") var modalityProjection: NFKSmolVLMProjection

    let scaleFactor: Int

    init(visionHidden: Int, decoderHidden: Int, scaleFactor: Int) {
        self.scaleFactor = scaleFactor
        _modalityProjection.wrappedValue = NFKSmolVLMProjection(
            inputSize: visionHidden * scaleFactor * scaleFactor, outputSize: decoderHidden)
        super.init()
    }

    /// `x` is `[tiles, patches, visionHidden]`. Returns `[tiles, patches / scaleFactor², decoderHidden]`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        modalityProjection(pixelShuffle(x))
    }

    /// The Idefics3/SmolVLM pixel shuffle, ported step for step so the token order matches the reference.
    private func pixelShuffle(_ input: MLXArray) -> MLXArray {
        let (batch, sequence, embed) = (input.dim(0), input.dim(1), input.dim(2))
        let height = Int(Double(sequence).squareRoot())
        let width = height
        var x = input.reshaped([batch, height, width, embed])
        x = x.reshaped([batch, height, width / scaleFactor, embed * scaleFactor])
        x = x.transposed(0, 2, 1, 3)
        x = x.reshaped([batch, width / scaleFactor, height / scaleFactor, embed * scaleFactor * scaleFactor])
        x = x.transposed(0, 2, 1, 3)
        return x.reshaped([batch, sequence / (scaleFactor * scaleFactor), embed * scaleFactor * scaleFactor])
    }
}

/// The connector's projection: a single bias-free linear layer under the reference's `proj` key.
final class NFKSmolVLMProjection: Module {
    @ModuleInfo(key: "proj") var proj: Linear

    init(inputSize: Int, outputSize: Int) {
        _proj.wrappedValue = Linear(inputSize, outputSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { proj(x) }
}

/// The Llama decoder geometry SmolVLM2-500M carries in its `text_config`.
extension NFKMLXLanguageConfiguration {
    public static let smolVLM2Decoder: NFKMLXLanguageConfiguration = {
        // The released checkpoint ships a separate `lm_head` that is NOT identical to the embedding, so
        // the decoder keeps its own output projection rather than tying.
        var configuration = NFKMLXLanguageConfiguration(
            hiddenSize: 960, layerCount: 32, headCount: 15, keyValueHeadCount: 5, headDimensions: 64,
            intermediateSize: 2560, vocabularySize: 49_280, ropeTheta: 100_000, rmsEpsilon: 1e-5,
            tiesWordEmbeddings: false)
        // A Llama decoder does not normalize its queries and keys; that is Qwen3.
        configuration.normalizesQueryAndKey = false
        return configuration
    }()
}

/// SmolVLM2 as a whole: the vision encoder, the connector, and the Llama decoder, plus the fusion that
/// splices the projected vision tokens into the decoder's input embeddings at the image-token positions.
public final class NFKMLXSmolVLMNet {
    let vision: NFKMLXSigLIPNet
    let connector: NFKMLXSmolVLMConnector
    let decoder: NFKMLXLanguageNet
    let imageTokenId: Int

    init(vision: NFKMLXSigLIPNet, connector: NFKMLXSmolVLMConnector, decoder: NFKMLXLanguageNet,
         imageTokenId: Int) {
        self.vision = vision
        self.connector = connector
        self.decoder = decoder
        self.imageTokenId = imageTokenId
    }

    /// The projected vision tokens for a batch of tiles, `[tiles, tokensPerTile, decoderWidth]`.
    /// `pixelValues` is the reference's `[tiles, 3, height, width]` (channels first).
    func imageFeatures(pixelValues: MLXArray) -> MLXArray {
        // The vision encoder takes channels-last input; the released pixel values are channels-first.
        connector(vision(pixelValues.transposed(0, 2, 3, 1)))
    }

    /// The decoder's input embeddings with the flattened vision tokens spliced in at the image-token
    /// positions, `[1, sequence, decoderWidth]`.
    func fusedEmbeddings(inputIds: [Int], imageFeatures features: MLXArray) -> MLXArray {
        let width = decoder.configuration.hiddenSize
        let flat = features.reshaped([-1, width])
        let sequence = inputIds.count
        let textEmbeddings = decoder.embed(MLXArray(inputIds.map(Int32.init)).reshaped([1, sequence]))[0]

        // Each image position reads the next vision token in order; a text position reads its embedding.
        var featureIndex = [Int32](repeating: 0, count: sequence)
        var isImage = [Float](repeating: 0, count: sequence)
        var counter: Int32 = 0
        for position in 0 ..< sequence where inputIds[position] == imageTokenId {
            featureIndex[position] = counter
            counter += 1
            isImage[position] = 1
        }
        let gathered = flat.take(MLXArray(featureIndex), axis: 0)                 // [sequence, width]
        let mask = MLXArray(isImage).reshaped([sequence, 1]) .> 0
        return MLX.where(mask, gathered, textEmbeddings).reshaped([1, sequence, width])
    }

    /// The decoder logits over a fused image-and-text sequence, `[1, sequence, vocabulary]`.
    func logits(inputIds: [Int], pixelValues: MLXArray) -> MLXArray {
        let embeddings = fusedEmbeddings(inputIds: inputIds,
                                         imageFeatures: imageFeatures(pixelValues: pixelValues))
        return decoder.logits(fromHidden: decoder.hiddenStates(fromEmbeddings: embeddings))
    }

    /// Greedy continuation from a fused image-and-text prompt: prefill through a cache, then decode one
    /// token at a time until `maxTokens` or an end token.
    func generate(inputIds: [Int], pixelValues: MLXArray, maxTokens: Int, endTokens: Set<Int>) -> [Int] {
        let cache = NFKMLXKeyValueCache(layerCount: decoder.configuration.layerCount)
        let embeddings = fusedEmbeddings(inputIds: inputIds,
                                         imageFeatures: imageFeatures(pixelValues: pixelValues))
        var hidden = decoder.hiddenStates(fromEmbeddings: embeddings, cache: cache)
        var produced = [Int]()
        for _ in 0 ..< maxTokens {
            let lastHidden = hidden[0..., (hidden.dim(1) - 1)...]
            let next = decoder.logits(fromHidden: lastHidden).reshaped([-1]).argMax().item(Int.self)
            if endTokens.contains(next) { break }
            produced.append(next)
            hidden = decoder.hiddenStates(fromEmbeddings: decoder.embed(MLXArray([Int32(next)]).reshaped([1, 1])),
                                          cache: cache)
        }
        return produced
    }
}

/// SmolVLM's image processor: an image becomes a set of 512×512 tiles plus a global thumbnail, each
/// normalized to `-1 … 1`, matching the sequence the vision encoder and the prompt expansion expect.
///
/// @discussion The resize is CoreGraphics rather than the reference's PIL LANCZOS, so the pixel values
/// differ slightly and a caption may not be token-identical to the reference; the tile geometry and the
/// prompt structure match exactly, and the network is at reference parity on the reference's own pixel
/// values. The image's longest edge is scaled to 2048, split into `⌈h/512⌉ × ⌈w/512⌉` sub-tiles, and the
/// whole image is resized to one 512×512 global tile appended last.
public enum NFKMLXSmolVLMImageProcessor {
    static let tileSize = 512
    static let longestEdge = 2048

    /// The pixel values `[tiles, 3, 512, 512]` and the tile grid for one image.
    public static func process(_ image: CGImage) -> (pixelValues: MLXArray, rows: Int, cols: Int) {
        let scale = Double(longestEdge) / Double(max(image.width, image.height))
        let resizedWidth = Swift.max(tileSize, Int((Double(image.width) * scale).rounded()))
        let resizedHeight = Swift.max(tileSize, Int((Double(image.height) * scale).rounded()))
        let cols = Int(ceil(Double(resizedWidth) / Double(tileSize)))
        let rows = Int(ceil(Double(resizedHeight) / Double(tileSize)))

        let grid = resample(image, width: cols * tileSize, height: rows * tileSize)
        var tiles = [Float]()
        for row in 0 ..< rows {
            for column in 0 ..< cols {
                appendNormalizedTile(from: grid.bytes, gridWidth: cols * tileSize,
                                     originX: column * tileSize, originY: row * tileSize, into: &tiles)
            }
        }
        let global = resample(image, width: tileSize, height: tileSize)
        appendNormalizedTile(from: global.bytes, gridWidth: tileSize, originX: 0, originY: 0, into: &tiles)

        let count = rows * cols + 1
        return (MLXArray(tiles).reshaped([count, 3, tileSize, tileSize]), rows, cols)
    }

    /// Draws `image` into a `width × height` RGBA context.
    private static func resample(_ image: CGImage, width: Int, height: Int) -> (bytes: [UInt8], width: Int, height: Int) {
        var bytes = [UInt8](repeating: 0, count: width * height * 4)
        let context = CGContext(data: &bytes, width: width, height: height, bitsPerComponent: 8,
                                bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
                                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        context?.interpolationQuality = .high
        context?.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        return (bytes, width, height)
    }

    /// Appends one 512×512 tile at `(originX, originY)` of an RGBA buffer as planar `[3, 512, 512]`
    /// normalized to `-1 … 1`.
    private static func appendNormalizedTile(from bytes: [UInt8], gridWidth: Int, originX: Int, originY: Int,
                                             into tiles: inout [Float]) {
        for channel in 0 ..< 3 {
            for y in 0 ..< tileSize {
                for x in 0 ..< tileSize {
                    let pixel = ((originY + y) * gridWidth + (originX + x)) * 4 + channel
                    tiles.append(Float(bytes[pixel]) / 127.5 - 1)
                }
            }
        }
    }
}

/// Holds the network and tokenizer for capture off the render thread.
private final class NFKSmolVLMHolder: @unchecked Sendable {
    let net: NFKMLXSmolVLMNet
    let tokenizer: NFKTokenizer?
    init(_ net: NFKMLXSmolVLMNet, _ tokenizer: NFKTokenizer?) {
        self.net = net
        self.tokenizer = tokenizer
    }
}

/// SmolVLM2 as an image-and-text model: an image and a question in, an answer out.
///
/// `NFKMLXSmolVLM` is the released `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` image path: a SigLIP
/// vision encoder, a pixel-shuffle connector, and a Llama decoder. The decoder is the dense stack
/// `NFKMLXLanguage` already runs, loaded from the checkpoint's `text_model` subtree. Run inference off
/// the render thread.
@objc(NFKMLXSmolVLM)
public final class NFKMLXSmolVLM: NSObject {

    /// A name for the model the factories produce.
    @objc public static let modelName = "smolvlm2-500m"

    private let holder: NFKSmolVLMHolder
    private let endToken: Int

    init(net: NFKMLXSmolVLMNet, tokenizer: NFKTokenizer?, endToken: Int) {
        holder = NFKSmolVLMHolder(net, tokenizer)
        self.endToken = endToken
        super.init()
    }

    /// Builds the model from a downloaded release directory, ready to answer questions about an image.
    @objc(smolVLMWithDirectoryURL:error:)
    public static func load(directoryURL: URL) throws -> NFKMLXSmolVLM {
        let net = try model(directoryURL: directoryURL)
        let tokenizer = tokenizer(inDirectory: directoryURL)
        let end = specialToken("<end_of_utterance>", inDirectory: directoryURL) ?? 49_279
        return NFKMLXSmolVLM(net: net, tokenizer: tokenizer, endToken: end)
    }

    /// Answers `question` about `image`, greedily decoding up to `maxTokens` tokens.
    @objc(answerForImage:question:maxTokens:)
    public func answer(image: CGImage, question: String, maxTokens: Int) -> String {
        guard let tokenizer = holder.tokenizer else { return "" }
        let (pixelValues, rows, cols) = NFKMLXSmolVLMImageProcessor.process(image)
        let prompt = NFKMLXSmolVLM.prompt(rows: rows, cols: cols, question: question)
        let ids = tokenizer.encode(prompt).map(\.intValue)
        let produced = holder.net.generate(inputIds: ids, pixelValues: pixelValues,
                                            maxTokens: maxTokens, endTokens: [endToken])
        return tokenizer.decode(produced.map { NSNumber(value: $0) })
    }

    /// Answers `question` about `image` with up to 128 tokens.
    @objc(answerForImage:question:)
    public func answer(image: CGImage, question: String) -> String {
        answer(image: image, question: question, maxTokens: 128)
    }

    /// The id of a special token literal in a release's `tokenizer.json`.
    static func specialToken(_ literal: String, inDirectory directory: URL) -> Int? {
        guard let data = try? Data(contentsOf: directory.appendingPathComponent("tokenizer.json")),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
        for entry in (json["added_tokens"] as? [[String: Any]]) ?? [] where entry["content"] as? String == literal {
            return entry["id"] as? Int
        }
        return nil
    }

    /// Builds the network from a downloaded release directory. Run inference off the render thread.
    public static func model(directoryURL: URL) throws -> NFKMLXSmolVLMNet {
        let configURL = directoryURL.appendingPathComponent("config.json")
        let json = (try? JSONSerialization.jsonObject(with: Data(contentsOf: configURL)) as? [String: Any]) ?? [:]
        let imageTokenId = (json["image_token_id"] as? NSNumber)?.intValue ?? 49_190
        let scaleFactor = (json["scale_factor"] as? NSNumber)?.intValue ?? 4

        let vision = NFKMLXSigLIPNet(.smolVLM)
        let connector = NFKMLXSmolVLMConnector(visionHidden: NFKMLXSigLIPConfiguration.smolVLM.hiddenSize,
                                               decoderHidden: NFKMLXLanguageConfiguration.smolVLM2Decoder.hiddenSize,
                                               scaleFactor: scaleFactor)
        let decoder = NFKMLXLanguage.makeNet(.smolVLM2Decoder)
        try loadWeights(vision: vision, connector: connector, decoder: decoder, directoryURL: directoryURL)
        return NFKMLXSmolVLMNet(vision: vision, connector: connector, decoder: decoder,
                                imageTokenId: imageTokenId)
    }

    /// Partitions the single checkpoint by prefix into the three networks: `model.vision_model.` (the
    /// patch-embedding convolution transposed to MLX's channels-last layout), `model.connector.`, and
    /// `model.text_model.` (remapped onto the decoder's `model.` layout, its tied `lm_head` dropped).
    static func loadWeights(vision: NFKMLXSigLIPNet, connector: NFKMLXSmolVLMConnector,
                            decoder: NFKMLXLanguageNet, directoryURL: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(
            url: directoryURL.appendingPathComponent("model.safetensors"))
        let transpose = checkpoint.needsConvTranspose

        let visionWeights = checkpoint.arrays.compactMap { key, value -> (String, MLXArray)? in
            guard key.hasPrefix("model.vision_model.") else { return nil }
            let stripped = String(key.dropFirst("model.vision_model.".count))
            return (stripped, transpose && value.ndim == 4 ? value.transposed(0, 2, 3, 1) : value)
        }
        let connectorWeights = checkpoint.arrays.compactMap { key, value -> (String, MLXArray)? in
            key.hasPrefix("model.connector.")
                ? (String(key.dropFirst("model.connector.".count)), value) : nil
        }
        let decoderWeights = checkpoint.arrays.compactMap { key, value -> (String, MLXArray)? in
            if key.hasPrefix("model.text_model.") {
                return ("model." + key.dropFirst("model.text_model.".count), value)
            }
            // The output projection lives at the top level and is not tied to the embedding here.
            return key == "lm_head.weight" ? (key, value) : nil
        }
        try NFKMLXWeights.apply(visionWeights, to: vision, verifyShapes: true)
        try NFKMLXWeights.apply(connectorWeights, to: connector, verifyShapes: true)
        try NFKMLXWeights.apply(decoderWeights, to: decoder, verifyShapes: true)
    }

    /// The number of `<image>` tokens each tile expands to (64 = 1024 patches shuffled by 4×4).
    static let tokensPerTile = 64

    /// The full expanded prompt string for a `rows × cols` tiling of one image and a question, ready for
    /// the byte-level BPE tokenizer to turn into ids. It reproduces the processor's structure: each
    /// sub-tile carries a `<row_r_col_c>` marker and `<image>` tokens, a row ends with a newline, and a
    /// global thumbnail tile follows before the text.
    static func prompt(rows: Int, cols: Int, question: String) -> String {
        let images = String(repeating: "<image>", count: tokensPerTile)
        var structure = ""
        for row in 1 ... rows {
            for column in 1 ... cols {
                structure += "<fake_token_around_image><row_\(row)_col_\(column)>" + images
            }
            structure += "\n"
        }
        structure += "\n<fake_token_around_image><global-img>" + images + "<fake_token_around_image>"
        return "User:" + structure + question + "<end_of_utterance>\nAssistant:"
    }

    /// The byte-level BPE tokenizer a release describes, with every added token (`<image>`, the
    /// `<row_r_col_c>` markers, `<end_of_utterance>`) registered so the expanded prompt segments on them.
    static func tokenizer(inDirectory directory: URL) -> NFKTokenizer? {
        guard let data = try? Data(contentsOf: directory.appendingPathComponent("tokenizer.json")),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let model = json["model"] as? [String: Any],
              let vocabulary = model["vocab"] as? [String: Int],
              let merges = model["merges"] as? [Any] else { return nil }

        var specials = [String: Int]()
        for entry in (json["added_tokens"] as? [[String: Any]]) ?? [] {
            if let content = entry["content"] as? String, let id = entry["id"] as? Int {
                specials[content] = id
            }
        }

        let scratch = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        guard (try? FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)) != nil,
              let vocabularyData = try? JSONSerialization.data(withJSONObject: vocabulary),
              (try? vocabularyData.write(to: scratch.appendingPathComponent("vocab.json"))) != nil else {
            return nil
        }
        var mergesText = "#version: 0.2\n"
        for entry in merges {
            if let pair = entry as? [String], pair.count == 2 {
                mergesText += pair[0] + " " + pair[1] + "\n"
            } else if let text = entry as? String {
                mergesText += text + "\n"
            }
        }
        guard (try? mergesText.write(to: scratch.appendingPathComponent("merges.txt"),
                                     atomically: true, encoding: .utf8)) != nil else { return nil }
        let manifest: [String: Any] = ["tokenizer": ["type": "bpe-bytelevel", "pretokenizer": "gpt2",
                                                     "specialTokens": specials]]
        return try? NFKTokenizer(forManifest: manifest, directory: scratch)
    }
}
