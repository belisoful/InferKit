//
//  NFKMLXPixtral.swift
//  InferKitMLX
//
//  Pixtral 12B's vision tower, the third vision-language architecture beside SmolVLM's SigLIP and
//  Qwen3-VL's 2D-rotary ViT. Pixtral's encoder is a from-scratch ViT with native variable-resolution
//  input: an image is convolved into patches at its own aspect ratio, laid out row-major, and read with
//  a 2D rotary embedding over the (height, width) patch grid. The blocks are pre-normalized with
//  RMSNorm, attend over the whole image (a block-diagonal mask when several images are packed, which for
//  a single image is full attention), and feed forward through a SiLU-gated MLP. A two-layer GELU
//  connector projects the image tokens to the decoder width, and the decoder is the Mistral-Nemo dense
//  stack `NFKMLXLanguageNet` already runs.
//

import Foundation
import CoreGraphics
import InferKit
import MLX
import MLXFast
import MLXNN

/// The geometry of the Pixtral vision encoder. Pixtral 12B ships the 1024-wide, 24-block tower; the
/// fields that a release's `vision_config` omits take the reference `PixtralVisionConfig` defaults.
public struct NFKMLXPixtralVisionConfiguration: Sendable {
    public var hiddenSize: Int
    public var depth: Int
    public var headCount: Int
    public var intermediateSize: Int
    public var patchSize: Int
    public var imageSize: Int
    public var ropeTheta: Float
    public var rmsEpsilon: Float

    public init(hiddenSize: Int = 1024, depth: Int = 24, headCount: Int = 16, intermediateSize: Int = 4096,
                patchSize: Int = 16, imageSize: Int = 1024, ropeTheta: Float = 10_000, rmsEpsilon: Float = 1e-5) {
        self.hiddenSize = hiddenSize
        self.depth = depth
        self.headCount = headCount
        self.intermediateSize = intermediateSize
        self.patchSize = patchSize
        self.imageSize = imageSize
        self.ropeTheta = ropeTheta
        self.rmsEpsilon = rmsEpsilon
    }

    public static let pixtral12B = NFKMLXPixtralVisionConfiguration()

    /// Reads a release's `vision_config` from its `config.json`.
    public static func configuration(fromHuggingFace url: URL) throws -> NFKMLXPixtralVisionConfiguration {
        let data = try Data(contentsOf: url)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw NFKMLXError.unsupportedConfiguration("\(url.lastPathComponent) is not a JSON object")
        }
        return try configuration(fromJSON: json)
    }

    static func configuration(fromJSON json: [String: Any]) throws -> NFKMLXPixtralVisionConfiguration {
        guard let vision = json["vision_config"] as? [String: Any] else {
            throw NFKMLXError.unsupportedConfiguration("the config carries no vision_config")
        }
        let kind = (vision["model_type"] as? String) ?? ""
        guard kind == "pixtral" else {
            throw NFKMLXError.unsupportedConfiguration("this reads a Pixtral vision tower, not \(kind)")
        }
        func integer(_ key: String, _ fallback: Int) -> Int { (vision[key] as? NSNumber)?.intValue ?? fallback }
        func real(_ key: String, _ fallback: Float) -> Float { (vision[key] as? NSNumber)?.floatValue ?? fallback }
        return NFKMLXPixtralVisionConfiguration(
            hiddenSize: integer("hidden_size", 1024), depth: integer("num_hidden_layers", 24),
            headCount: integer("num_attention_heads", 16), intermediateSize: integer("intermediate_size", 4096),
            patchSize: integer("patch_size", 16), imageSize: integer("image_size", 1024),
            ropeTheta: real("rope_theta", 10_000))
    }

    var headDimensions: Int { hiddenSize / headCount }
    var patchInputSize: Int { 3 * patchSize * patchSize }
    /// The reference's `max_patches_per_side`: the rotary table is indexed against this square grid,
    /// not the image's own grid, so a patch's position id is `row · maxSide + column`.
    var maxSide: Int { imageSize / patchSize }
}

/// Pixtral vision attention: separate q/k/v/o projections without bias, 2D rotary on the queries and
/// keys, and full attention over an image's patches.
final class NFKPixtralAttention: Module {
    @ModuleInfo(key: "q_proj") var queryProjection: Linear
    @ModuleInfo(key: "k_proj") var keyProjection: Linear
    @ModuleInfo(key: "v_proj") var valueProjection: Linear
    @ModuleInfo(key: "o_proj") var outputProjection: Linear

    let heads: Int
    let headDimensions: Int
    let scale: Float

    init(_ c: NFKMLXPixtralVisionConfiguration) {
        heads = c.headCount
        headDimensions = c.headDimensions
        scale = 1 / sqrt(Float(c.headDimensions))
        _queryProjection.wrappedValue = Linear(c.hiddenSize, c.hiddenSize, bias: false)
        _keyProjection.wrappedValue = Linear(c.hiddenSize, c.hiddenSize, bias: false)
        _valueProjection.wrappedValue = Linear(c.hiddenSize, c.hiddenSize, bias: false)
        _outputProjection.wrappedValue = Linear(c.hiddenSize, c.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let length = x.dim(0)
        var queries = queryProjection(x).reshaped([length, heads, headDimensions]).transposed(1, 0, 2)
        var keys = keyProjection(x).reshaped([length, heads, headDimensions]).transposed(1, 0, 2)
        let values = valueProjection(x).reshaped([length, heads, headDimensions]).transposed(1, 0, 2)

        queries = applyRotary(queries, cos: cos, sin: sin)
        keys = applyRotary(keys, cos: cos, sin: sin)

        let attention = MLXFast.scaledDotProductAttention(
            queries: queries.expandedDimensions(axis: 0), keys: keys.expandedDimensions(axis: 0),
            values: values.expandedDimensions(axis: 0), scale: scale, mask: nil)[0]
        return outputProjection(attention.transposed(1, 0, 2).reshaped([length, heads * headDimensions]))
    }

    /// `x·cos + rotateHalf(x)·sin`, with `cos`/`sin` `[length, headDim]` shared across heads.
    private func applyRotary(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let half = x.dim(2) / 2
        let rotated = concatenated([-x[0..., 0..., half...], x[0..., 0..., 0 ..< half]], axis: -1)
        return x * cos + rotated * sin
    }
}

/// Pixtral vision feed-forward: a SiLU-gated MLP, `down(silu(gate(x)) · up(x))`.
final class NFKPixtralMLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(_ c: NFKMLXPixtralVisionConfiguration) {
        _gate.wrappedValue = Linear(c.hiddenSize, c.intermediateSize, bias: false)
        _up.wrappedValue = Linear(c.hiddenSize, c.intermediateSize, bias: false)
        _down.wrappedValue = Linear(c.intermediateSize, c.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { down(silu(gate(x)) * up(x)) }
}

/// One Pixtral vision block: pre-normalized attention and feed-forward, each added back.
final class NFKPixtralLayer: Module {
    @ModuleInfo(key: "attention_norm") var attentionNorm: RMSNorm
    @ModuleInfo(key: "attention") var attention: NFKPixtralAttention
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm
    @ModuleInfo(key: "feed_forward") var feedForward: NFKPixtralMLP

    init(_ c: NFKMLXPixtralVisionConfiguration) {
        _attentionNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsEpsilon)
        _attention.wrappedValue = NFKPixtralAttention(c)
        _ffnNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsEpsilon)
        _feedForward.wrappedValue = NFKPixtralMLP(c)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let attended = x + attention(attentionNorm(x), cos: cos, sin: sin)
        return attended + feedForward(ffnNorm(attended))
    }
}

/// The stack of vision blocks, named to match the reference's `transformer.layers`.
final class NFKPixtralTransformer: Module {
    @ModuleInfo(key: "layers") var layers: [NFKPixtralLayer]

    init(_ c: NFKMLXPixtralVisionConfiguration) {
        _layers.wrappedValue = (0 ..< c.depth).map { _ in NFKPixtralLayer(c) }
        super.init()
    }
}

/// The Pixtral vision encoder. It takes an image's normalized pixels and returns the per-patch features
/// the connector projects to the decoder width.
public final class NFKMLXPixtralVisionNet: Module {
    @ModuleInfo(key: "patch_conv") var patchConv: Linear
    @ModuleInfo(key: "ln_pre") var lnPre: RMSNorm
    @ModuleInfo(key: "transformer") var transformer: NFKPixtralTransformer

    let configuration: NFKMLXPixtralVisionConfiguration

    init(_ c: NFKMLXPixtralVisionConfiguration) {
        configuration = c
        // The patch convolution has kernel = stride = patch, so it is one linear projection over each
        // flattened `[channels, patch, patch]` patch. The loader reshapes the 4-D conv weight to match.
        _patchConv.wrappedValue = Linear(c.patchInputSize, c.hiddenSize, bias: false)
        _lnPre.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsEpsilon)
        _transformer.wrappedValue = NFKPixtralTransformer(c)
        super.init()
    }

    /// `pixelValues` is `[1, 3, height, width]` normalized pixels. Returns the per-patch features
    /// `[gridH · gridW, hidden]` in row-major patch order, the reference's `last_hidden_state`.
    public func callAsFunction(_ pixelValues: MLXArray) -> MLXArray {
        intermediateSeams(pixelValues).output
    }

    /// The patch embedding, the ln_pre output, the first block's output, and the last hidden state —
    /// what the isolation harness compares seam by seam.
    func intermediateSeams(_ pixelValues: MLXArray)
        -> (patch: MLXArray, lnPre: MLXArray, layer0: MLXArray, output: MLXArray) {
        let patch = configuration.patchSize
        let gridH = pixelValues.dim(2) / patch, gridW = pixelValues.dim(3) / patch
        let embedded = patchConv(patchify(pixelValues, gridH: gridH, gridW: gridW))
        var hidden = lnPre(embedded)
        let normalized = hidden
        let (cos, sin) = rotaryEmbedding(gridH: gridH, gridW: gridW)
        var first = hidden
        for (index, layer) in transformer.layers.enumerated() {
            hidden = layer(hidden, cos: cos, sin: sin)
            if index == 0 { first = hidden }
        }
        return (embedded, normalized, first, hidden)
    }

    /// Flattens `[1, 3, H, W]` into row-major patches `[gridH · gridW, 3 · patch²]`, each patch laid out
    /// `(channel, kernelRow, kernelColumn)` to match the reshaped convolution kernel.
    func patchify(_ pixelValues: MLXArray, gridH: Int, gridW: Int) -> MLXArray {
        let patch = configuration.patchSize
        return pixelValues.reshaped([3, gridH, patch, gridW, patch])
            .transposed(1, 3, 0, 2, 4)
            .reshaped([gridH * gridW, 3 * patch * patch])
    }

    /// The 2D rotary cosine and sine tables for the patch grid, `[1, gridH · gridW, headDim]`. A patch
    /// at grid position `(row, column)` takes the first quarter of the head dimension from the row's
    /// frequencies and the second quarter from the column's, doubled — the reference's precomputed
    /// `inv_freq` indexed by `row · maxSide + column`.
    func rotaryEmbedding(gridH: Int, gridW: Int) -> (cos: MLXArray, sin: MLXArray) {
        let dimensions = configuration.headDimensions
        let quarter = dimensions / 4
        let theta = configuration.ropeTheta
        let rowFrequencies = (0 ..< quarter).map { 1 / powf(theta, Float(4 * $0) / Float(dimensions)) }
        let columnFrequencies = (0 ..< quarter).map { 1 / powf(theta, Float(4 * $0 + 2) / Float(dimensions)) }

        var table = [Float]()
        table.reserveCapacity(gridH * gridW * dimensions / 2)
        for row in 0 ..< gridH {
            for column in 0 ..< gridW {
                table.append(contentsOf: rowFrequencies.map { Float(row) * $0 })
                table.append(contentsOf: columnFrequencies.map { Float(column) * $0 })
            }
        }
        let count = gridH * gridW
        let rotary = MLXArray(table).reshaped([count, dimensions / 2])
        let doubled = concatenated([rotary, rotary], axis: -1)              // [count, headDim]
        return (cos(doubled).expandedDimensions(axis: 0), sin(doubled).expandedDimensions(axis: 0))
    }
}

/// The Pixtral connector (`multi_modal_projector`): two biased linear layers around an exact GELU, from
/// the vision width to the decoder width.
public final class NFKMLXPixtralConnector: Module {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(visionSize: Int, textSize: Int) {
        _linear1.wrappedValue = Linear(visionSize, textSize, bias: true)
        _linear2.wrappedValue = Linear(textSize, textSize, bias: true)
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray { linear2(gelu(linear1(x))) }
}

/// Building and loading Pixtral 12B.
///
/// `NFKMLXPixtral` is the released `mistral-experimental/pixtral-12b`. The vision tower is
/// ``NFKMLXPixtralVisionNet``, the connector ``NFKMLXPixtralConnector``, and the decoder the
/// Mistral-Nemo dense stack `NFKMLXLanguage` already runs, loaded from the checkpoint's
/// `language_model.` subtree.
@objc(NFKMLXPixtral)
public final class NFKMLXPixtral: NSObject {

    /// A name for the model the factories produce.
    @objc public static let modelName = "pixtral-12b"
    static let requiredFiles = ["config.json", "tokenizer.json", "tokenizer_config.json"]
    static let optionalFiles = ["vocab.json", "merges.txt", "added_tokens.json"]
    static let weightFiles = ["model.safetensors", "model.safetensors.index.json"]

    /// The `[IMG]` placeholder id the projected vision tokens splice into (`image_token_index`).
    static let imageTokenId = 10

    private let visionNet: NFKMLXPixtralVisionNet
    private let connector: NFKMLXPixtralConnector
    private let textDecoder: NFKMLXLanguageNet
    private let tokenizer: NFKTokenizer
    private let processor: NFKMLXPixtralImageProcessor
    private let endTokens: Set<Int>

    init(visionNet: NFKMLXPixtralVisionNet, connector: NFKMLXPixtralConnector, decoder: NFKMLXLanguageNet,
         tokenizer: NFKTokenizer, endTokens: Set<Int>, processor: NFKMLXPixtralImageProcessor) {
        self.visionNet = visionNet
        self.connector = connector
        self.textDecoder = decoder
        self.tokenizer = tokenizer
        self.endTokens = endTokens
        self.processor = processor
        super.init()
    }

    /// Loads the whole model — the vision tower, the connector, the Mistral decoder, and the release
    /// tokenizer — from a downloaded release directory, ready to answer a question about an image.
    @objc(modelWithDirectoryURL:error:)
    public static func model(directoryURL: URL) throws -> NFKMLXPixtral {
        let vision = try visionNet(directoryURL: directoryURL)
        let connector = try connector(directoryURL: directoryURL)
        let decoder = try decoder(directoryURL: directoryURL)
        guard let tokenizer = NFKMLXLanguage.releaseTokenizer(inDirectory: directoryURL) else {
            throw NFKMLXError.weightsMismatch("the release carries no tokenizer files")
        }
        let (specials, endToken) = NFKMLXLanguage.specialTokens(inDirectory: directoryURL)
        var stops = Set([specials["</s>"], endToken].compactMap { $0 })
        if stops.isEmpty { stops = [2] }
        return NFKMLXPixtral(visionNet: vision, connector: connector, decoder: decoder, tokenizer: tokenizer,
                             endTokens: stops, processor: NFKMLXPixtralImageProcessor())
    }

    /// Downloads a release into the hub cache and loads the whole model.
    ///
    /// @discussion The download fetches `config.json`, `tokenizer.json`, `tokenizer_config.json`, the
    /// `vocab.json`, `merges.txt`, and `added_tokens.json` the repo serves, and the weights, single-file
    /// or sharded. A file the cache already holds is not fetched again. The call blocks on the network
    /// for about 25 GB on a first download; call it off the render thread. The public release is
    /// `mistral-experimental/pixtral-12b`, not gated; its former name, `mistral-community/pixtral-12b`,
    /// redirects there.
    ///
    /// Introduced in InferKit 0.4.0.
    @objc(modelWithRepo:revision:cacheDirectoryURL:error:)
    public static func model(repo: String, revision: String?, cacheDirectoryURL: URL?) throws -> NFKMLXPixtral {
        try model(directoryURL: try NFKMLXReleaseDownload.directory(
            repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL,
            required: requiredFiles, optional: optionalFiles, weights: weightFiles))
    }

    /// The asynchronous form of ``model(repo:revision:cacheDirectoryURL:)``. The download and the build
    /// run at user-initiated quality of service off the calling thread.
    ///
    /// Introduced in InferKit 0.4.0.
    @objc(modelWithRepo:revision:cacheDirectoryURL:completionHandler:)
    public static func model(repo: String, revision: String?, cacheDirectoryURL: URL?,
                             completionHandler: @escaping (NFKMLXPixtral?, Error?) -> Void) {
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                completionHandler(try model(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL), nil)
            } catch {
                completionHandler(nil, error)
            }
        }
    }

    /// Answers `question` about `image`: the processor resizes and normalizes it, the vision tower and
    /// connector encode it, the projected tokens splice into the prompt at the `[IMG]` positions, and the
    /// Mistral decoder generates a reply. The CoreGraphics resize differs from the reference's, so the
    /// answer is a close approximation of the reference pipeline's rather than token-identical.
    @objc(answerForImage:question:maxTokens:)
    public func answer(image: CGImage, question: String, maxTokens: Int) -> String {
        let (pixelValues, gridH, gridW) = processor.process(image)
        let features = connector(visionNet(pixelValues))
        let prompt = "<s>[INST]" + imagePlaceholders(gridH: gridH, gridW: gridW) + question + "[/INST]"
        let inputIds = tokenizer.encode(prompt).map(\.intValue)
        let generated = NFKMLXPixtral.generate(decoder: textDecoder, inputIds: inputIds, features: features,
                                               maxTokens: maxTokens, endTokens: endTokens)
        return tokenizer.decode(generated.map { NSNumber(value: $0) })
    }

    /// The `[IMG]` / `[IMG_BREAK]` / `[IMG_END]` expansion the Pixtral processor writes for one image:
    /// `gridW` image tokens per row, a break after each row, and an end token last.
    func imagePlaceholders(gridH: Int, gridW: Int) -> String {
        var text = ""
        for row in 0 ..< gridH {
            text += String(repeating: "[IMG]", count: gridW)
            text += row == gridH - 1 ? "[IMG_END]" : "[IMG_BREAK]"
        }
        return text
    }

    /// Builds the vision tower from a release directory, at the geometry its `config.json` describes.
    public static func visionNet(directoryURL: URL) throws -> NFKMLXPixtralVisionNet {
        let configuration = try NFKMLXPixtralVisionConfiguration.configuration(
            fromHuggingFace: directoryURL.appendingPathComponent("config.json"))
        let net = NFKMLXPixtralVisionNet(configuration)
        let arrays = try NFKMLXReleaseWeights.arrays(inDirectory: directoryURL) { key in
            key.hasPrefix("vision_tower.") ? String(key.dropFirst("vision_tower.".count)) : nil
        }
        // The patch convolution weight is stored 4-D (`[out, channels, patch, patch]`) and flattens to a
        // linear weight `[out, channels · patch²]`.
        let mapped = arrays.map { key, value -> (String, MLXArray) in
            key == "patch_conv.weight" && value.ndim == 4 ? (key, value.reshaped([value.dim(0), -1])) : (key, value)
        }
        try NFKMLXWeights.apply(mapped, to: net, verifyShapes: true)
        return net
    }

    /// Builds the connector from a release directory. Its input width is the vision hidden size, its
    /// output the decoder width.
    public static func connector(directoryURL: URL) throws -> NFKMLXPixtralConnector {
        let visionConfiguration = try NFKMLXPixtralVisionConfiguration.configuration(
            fromHuggingFace: directoryURL.appendingPathComponent("config.json"))
        let textConfiguration = try decoderConfiguration(directoryURL: directoryURL)
        let net = NFKMLXPixtralConnector(visionSize: visionConfiguration.hiddenSize,
                                         textSize: textConfiguration.hiddenSize)
        let arrays = try NFKMLXReleaseWeights.arrays(inDirectory: directoryURL) { key in
            key.hasPrefix("multi_modal_projector.")
                ? String(key.dropFirst("multi_modal_projector.".count)) : nil
        }
        try NFKMLXWeights.apply(arrays, to: net, verifyShapes: true)
        return net
    }

    /// The decoder configuration a release's `config.json` describes: its `text_config` read through the
    /// dense reader (`mistral`).
    ///
    /// @discussion Pixtral's `text_config` states `head_dim` (128) but omits `num_attention_heads`, and
    /// the head width does not divide the hidden size (5120 / 128 = 40, but the release has 32 heads at a
    /// 4096-wide q projection). The dense reader derives the count that fills the hidden size, so the head
    /// count is corrected here to the transformers `MistralConfig` default the release inherits.
    public static func decoderConfiguration(directoryURL: URL) throws -> NFKMLXLanguageConfiguration {
        let data = try Data(contentsOf: directoryURL.appendingPathComponent("config.json"))
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let text = json["text_config"] as? [String: Any] else {
            throw NFKMLXError.unsupportedConfiguration("the config carries no text_config")
        }
        var configuration = try NFKMLXLanguage.configuration(fromJSON: text)
        if text["num_attention_heads"] == nil {
            configuration.headCount = 32
        }
        return configuration
    }

    /// Builds the Mistral decoder at the release's geometry and loads the `language_model.` subtree —
    /// the dense stack under `language_model.model.` and the untied `language_model.lm_head.weight`.
    public static func decoder(directoryURL: URL) throws -> NFKMLXLanguageNet {
        let configuration = try decoderConfiguration(directoryURL: directoryURL)
        let net = NFKMLXLanguageNet(configuration)
        let arrays = try NFKMLXReleaseWeights.arrays(inDirectory: directoryURL) { key in
            key.hasPrefix("language_model.") ? String(key.dropFirst("language_model.".count)) : nil
        }
        try NFKMLXWeights.apply(arrays, to: net)
        return net
    }

    /// The decoder logits over a fused image-and-text sequence, `[1, sequence, vocabulary]`. The projected
    /// vision features splice into the decoder's input embeddings at the `[IMG]` positions; the Mistral
    /// rotary is the ordinary 1-D one.
    public static func logits(decoder: NFKMLXLanguageNet, inputIds: [Int], features: MLXArray) -> MLXArray {
        decoder.logits(fromHidden: fusedEmbeddings(decoder: decoder, inputIds: inputIds, features: features)
            .hidden)
    }

    /// Embeds the token ids and scatters the projected vision features into the `[IMG]` positions in
    /// order. Returns the fused embeddings run through the decoder to post-norm hidden states.
    static func fusedEmbeddings(decoder: NFKMLXLanguageNet, inputIds: [Int], features: MLXArray,
                                cache: NFKMLXKeyValueCache? = nil) -> (hidden: MLXArray, imageCount: Int) {
        let sequence = inputIds.count
        let width = decoder.configuration.hiddenSize
        var embeddings = decoder.embed(MLXArray(inputIds.map(Int32.init)).reshaped([1, sequence]))[0]

        var featureIndex = [Int32](repeating: 0, count: sequence)
        var isImage = [Float](repeating: 0, count: sequence)
        var counter: Int32 = 0
        for position in 0 ..< sequence where inputIds[position] == imageTokenId {
            featureIndex[position] = counter
            counter += 1
            isImage[position] = 1
        }
        if counter > 0 {
            let gathered = features.reshaped([-1, width]).take(MLXArray(featureIndex), axis: 0)
            let mask = MLXArray(isImage).reshaped([sequence, 1]) .> 0
            embeddings = MLX.where(mask, gathered, embeddings)
        }
        let hidden = decoder.hiddenStates(fromEmbeddings: embeddings.reshaped([1, sequence, width]),
                                          cache: cache)
        return (hidden, Int(counter))
    }

    /// Greedy continuation from a fused image-and-text prompt, cached: a prefill over the fused
    /// embeddings, then one token at a time.
    public static func generate(decoder: NFKMLXLanguageNet, inputIds: [Int], features: MLXArray,
                                maxTokens: Int, endTokens: Set<Int>) -> [Int] {
        let cache = NFKMLXKeyValueCache(layerCount: decoder.configuration.layerCount)
        var hidden = fusedEmbeddings(decoder: decoder, inputIds: inputIds, features: features, cache: cache).hidden
        var produced = [Int]()
        for _ in 0 ..< maxTokens {
            let lastHidden = hidden[0..., (hidden.dim(1) - 1)...]
            let next = decoder.logits(fromHidden: lastHidden).reshaped([-1]).argMax().item(Int.self)
            if endTokens.contains(next) { break }
            produced.append(next)
            hidden = decoder.hiddenStates(
                fromEmbeddings: decoder.embed(MLXArray([Int32(next)]).reshaped([1, 1])), cache: cache)
        }
        return produced
    }
}

/// Turns a `CGImage` into the normalized pixels the Pixtral vision tower reads. The image is resized so
/// its longest edge is at most `imageSize` and both sides are a multiple of the patch, then rescaled and
/// normalized with the CLIP mean and standard deviation.
///
/// The resize is CoreGraphics rather than the reference's bicubic, so the pixels are a close
/// approximation, the documented difference the SmolVLM and Qwen3-VL processors also carry.
public struct NFKMLXPixtralImageProcessor {
    public let patchSize: Int
    public let longestEdge: Int
    public let mean: [Float]
    public let standardDeviation: [Float]

    public init(patchSize: Int = 16, longestEdge: Int = 1024,
                mean: [Float] = [0.48145466, 0.4578275, 0.40821073],
                standardDeviation: [Float] = [0.26862954, 0.26130258, 0.27577711]) {
        self.patchSize = patchSize
        self.longestEdge = longestEdge
        self.mean = mean
        self.standardDeviation = standardDeviation
    }

    /// The reference resize: scale down so neither side passes `longestEdge`, then round each side up to
    /// a multiple of the patch.
    public func resize(height: Int, width: Int) -> (height: Int, width: Int) {
        let ratio = Swift.max(Double(height) / Double(longestEdge), Double(width) / Double(longestEdge))
        var scaledHeight = height, scaledWidth = width
        if ratio > 1 {
            scaledHeight = Int((Double(height) / ratio).rounded(.down))
            scaledWidth = Int((Double(width) / ratio).rounded(.down))
        }
        func toPatch(_ value: Int) -> Int { Int((Double(value) / Double(patchSize)).rounded(.up)) * patchSize }
        return (toPatch(scaledHeight), toPatch(scaledWidth))
    }

    /// The normalized pixels `[1, 3, height, width]` and the patch grid.
    public func process(_ image: CGImage) -> (pixelValues: MLXArray, gridH: Int, gridW: Int) {
        let (height, width) = resize(height: image.height, width: image.width)
        var bytes = [UInt8](repeating: 0, count: width * height * 4)
        let context = CGContext(data: &bytes, width: width, height: height, bitsPerComponent: 8,
                                bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
                                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        context?.interpolationQuality = .high
        context?.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

        var planar = [Float](repeating: 0, count: 3 * height * width)
        for y in 0 ..< height {
            for x in 0 ..< width {
                let base = (y * width + x) * 4
                for channel in 0 ..< 3 {
                    let value = Float(bytes[base + channel]) / 255
                    planar[channel * height * width + y * width + x] =
                        (value - mean[channel]) / standardDeviation[channel]
                }
            }
        }
        return (MLXArray(planar).reshaped([1, 3, height, width]), height / patchSize, width / patchSize)
    }
}
