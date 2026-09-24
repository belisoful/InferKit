// Sa2VA (ByteDance) is a segmentation-capable vision-language model: an InternVL2.5 understanding core
// — an InternViT-300M image encoder, a pixel-shuffle MLP projector, and a Qwen2.5-3B decoder — whose
// text stream can emit a `[SEG]` token, and a SAM 2 grounding branch that turns the decoder's hidden
// state at each `[SEG]` position into a segmentation mask. The understanding half and the SAM 2 half
// are ported elsewhere in the package and reused here: the decoder is `NFKMLXLanguageNet` (Qwen2.5 is a
// plain qwen2 dense decoder) and the grounding encoder is `NFKMLXSAM2TrackerNet` (the checkpoint's
// `grounding_encoder.sam2_model.*` is a standard Hiera-Large SAM 2). What this file adds is the
// InternViT tower, the projector, the `[SEG]`-hidden-state bridge, and the fusion that splices vision
// tokens into the decoder.
//
// The one seam without a neighbor is the InternViT tower: a pre-norm ViT with per-channel LayerScale on
// each residual, a class token and a learned position table, qkv bias, no query/key normalization, and
// no final layer norm (the features are the last block's output). The projector's pixel shuffle is
// InternVL's own, which transposes differently from Idefics3's, so it is written out here rather than
// borrowed from `NFKMLXSmolVLMConnector`.

import Foundation
import MLX
import MLXNN
import MLXFast

// MARK: - Configuration

/// The instruction template a Sa2VA release names in its `config.json` (`template`), as the release's
/// `templates.py` spells it. The reference formats one instruction with no system turn and stops at the
/// tokenizer's end token or at any of the template's stop words.
public struct NFKMLXSa2VATemplate: Sendable, Equatable {
    /// The template's name in `config.json`.
    public let name: String
    /// The user turn, `{input}` marking where the prompt goes.
    public let instruction: String
    /// The texts that end the assistant's turn.
    public let stopWords: [String]

    public static let phi3 = NFKMLXSa2VATemplate(name: "phi3_chat", instruction: "<|user|>\n{input}<|end|>\n<|assistant|>\n",
                                                 stopWords: ["<|end|>"])
    public static let qwen = NFKMLXSa2VATemplate(name: "qwen_chat",
                                                 instruction: "<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n",
                                                 stopWords: ["<|im_end|>", "<|endoftext|>"])
    public static let internLM2 = NFKMLXSa2VATemplate(name: "internlm2_chat",
                                                      instruction: "<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n",
                                                      stopWords: ["<|im_end|>"])
    public static let vicuna = NFKMLXSa2VATemplate(name: "vicuna", instruction: "USER: {input} ASSISTANT:", stopWords: [])

    /// The template `config.json` names; a release that names none is Sa2VA-4B's phi3.
    public static func named(_ name: String?) throws -> NFKMLXSa2VATemplate {
        guard let name else { return .phi3 }
        guard let template = [phi3, qwen, internLM2, vicuna].first(where: { $0.name == name }) else {
            throw NFKMLXError.unsupportedConfiguration("Sa2VA template \(name) is not one this port formats")
        }
        return template
    }

    /// The instruction around `input`.
    public func formatted(_ input: String) -> String {
        instruction.replacingOccurrences(of: "{input}", with: input)
    }

    /// The reference's `StopWordStoppingCriteria`: whether the decoded generation, with carriage returns
    /// and newlines removed, ends with a stop word. A stop word can span tokens in a way no fixed token
    /// run matches (phi3's `<|end|>` merges with the text before it in a Qwen vocabulary), so the test
    /// is on the text. Only the tail is decoded, which is enough to hold the longest stop word.
    public func stops(after tokens: [Int], decode: ([Int]) -> String) -> Bool {
        guard !stopWords.isEmpty else { return false }
        let longest = stopWords.map(\.count).max() ?? 0
        let text = decode(Array(tokens.suffix(longest + 8)))
            .replacingOccurrences(of: "\r", with: "").replacingOccurrences(of: "\n", with: "")
        return stopWords.contains { text.hasSuffix($0) }
    }
}

/// Geometry for a Sa2VA release on InternVL. The vision and projector fields describe the InternViT tower and the
/// pixel-shuffle MLP; the decoder is carried as a `NFKMLXLanguageConfiguration` (qwen2), and the
/// grounding encoder uses `NFKMLXSAM2TrackerConfiguration.large`.
public struct NFKMLXSa2VAConfiguration: Sendable {
    // InternViT
    public var visionHiddenSize: Int
    public var visionLayers: Int
    public var visionHeads: Int
    public var visionIntermediateSize: Int
    public var patchSize: Int
    public var imageSize: Int
    public var visionLayerNormEps: Float
    public var qkvBias: Bool
    /// RMSNorm in place of LayerNorm (`norm_type` `rms_norm`, InternViT-6B).
    public var visionRMSNorm: Bool = false
    /// Per-token RMS normalization of the queries and keys across all heads (`qk_normalization`,
    /// InternViT-6B).
    public var visionQueryKeyNormalization: Bool = false

    // Projector + fusion
    public var downsampleRatio: Double
    public var decoderHiddenSize: Int

    // Decoder + special tokens
    public var decoder: NFKMLXLanguageConfiguration
    public var imageContextTokenId: Int
    public var segmentationTokenId: Int
    /// The release's instruction template.
    public var template: NFKMLXSa2VATemplate = .phi3
    /// The tokenizer's end token (`tokenizer_config.json`'s `eos_token`).
    public var endTokenId: Int = 151_645

    /// The number of vision tokens one 448×448 tile contributes after pixel shuffle: (448/14)² · (1/2)².
    public var tokensPerTile: Int {
        let grid = imageSize / patchSize
        return Int(Double(grid * grid) * downsampleRatio * downsampleRatio)
    }

    /// The SAM 2 grounding encoder's geometry. Sa2VA's checkpoint is a SAM 2.0 Hiera-Large, without the
    /// 2.1-only object-pointer temporal projection and occlusion embedding, so it builds at `.sam2`.
    public var groundingConfiguration: NFKMLXSAM2TrackerConfiguration {
        NFKMLXSAM2TrackerConfiguration.geometry(.large, release: .sam2)
    }

    public init(visionHiddenSize: Int = 1024, visionLayers: Int = 24, visionHeads: Int = 16,
                visionIntermediateSize: Int = 4096, patchSize: Int = 14, imageSize: Int = 448,
                visionLayerNormEps: Float = 1e-6, qkvBias: Bool = true,
                downsampleRatio: Double = 0.5, decoderHiddenSize: Int = 2048,
                decoder: NFKMLXLanguageConfiguration, imageContextTokenId: Int,
                segmentationTokenId: Int) {
        self.visionHiddenSize = visionHiddenSize
        self.visionLayers = visionLayers
        self.visionHeads = visionHeads
        self.visionIntermediateSize = visionIntermediateSize
        self.patchSize = patchSize
        self.imageSize = imageSize
        self.visionLayerNormEps = visionLayerNormEps
        self.qkvBias = qkvBias
        self.downsampleRatio = downsampleRatio
        self.decoderHiddenSize = decoderHiddenSize
        self.decoder = decoder
        self.imageContextTokenId = imageContextTokenId
        self.segmentationTokenId = segmentationTokenId
    }
}

// MARK: - InternViT tower

/// The patch embedding, class token, and learned position table. The patch conv runs on channels-last
/// input; the position table is added unchanged because the tile size is fixed at the native grid, so
/// the reference's bicubic resize to that same grid is the identity.
final class NFKSa2VAVisionEmbeddings: Module {
    @ModuleInfo(key: "patch_embedding") var patchEmbedding: Conv2d
    @ParameterInfo(key: "class_embedding") var classEmbedding: MLXArray
    @ParameterInfo(key: "position_embedding") var positionEmbedding: MLXArray

    init(_ c: NFKMLXSa2VAConfiguration) {
        _patchEmbedding.wrappedValue = Conv2d(
            inputChannels: 3, outputChannels: c.visionHiddenSize,
            kernelSize: IntOrPair(c.patchSize), stride: IntOrPair(c.patchSize))
        let grid = c.imageSize / c.patchSize
        _classEmbedding.wrappedValue = MLXArray.zeros([1, 1, c.visionHiddenSize])
        _positionEmbedding.wrappedValue = MLXArray.zeros([1, grid * grid + 1, c.visionHiddenSize])
        super.init()
    }

    /// `pixelValues` is channels-last `[batch, height, width, 3]`. Returns `[batch, 1 + patches, hidden]`.
    func callAsFunction(_ pixelValues: MLXArray) -> MLXArray {
        let patches = patchEmbedding(pixelValues)                       // [B, gh, gw, hidden]
        let batch = patches.shape[0]
        let hidden = patches.shape[3]
        let flat = patches.reshaped([batch, -1, hidden])                // [B, patches, hidden]
        let cls = broadcast(classEmbedding, to: [batch, 1, hidden])
        let tokens = concatenated([cls, flat], axis: 1)
        return tokens + positionEmbedding
    }
}

/// Multi-head self-attention with a fused qkv projection. InternViT-6B also RMS-normalizes each token's
/// queries and keys across all heads together (`q_norm` / `k_norm` over the full width) before they
/// split; the 300M tower does not.
final class NFKSa2VAAttention: Module {
    @ModuleInfo(key: "qkv") var qkv: Linear
    @ModuleInfo(key: "proj") var proj: Linear
    @ModuleInfo(key: "q_norm") var queryNorm: RMSNorm?
    @ModuleInfo(key: "k_norm") var keyNorm: RMSNorm?

    let heads: Int
    let scale: Float

    init(_ c: NFKMLXSa2VAConfiguration) {
        heads = c.visionHeads
        scale = pow(Float(c.visionHiddenSize / c.visionHeads), -0.5)
        _qkv.wrappedValue = Linear(c.visionHiddenSize, 3 * c.visionHiddenSize, bias: c.qkvBias)
        _proj.wrappedValue = Linear(c.visionHiddenSize, c.visionHiddenSize)
        let normalizes = c.visionQueryKeyNormalization
        _queryNorm.wrappedValue = normalizes ? RMSNorm(dimensions: c.visionHiddenSize, eps: c.visionLayerNormEps) : nil
        _keyNorm.wrappedValue = normalizes ? RMSNorm(dimensions: c.visionHiddenSize, eps: c.visionLayerNormEps) : nil
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (batch, length, width) = (x.shape[0], x.shape[1], x.shape[2])
        let headDim = width / heads
        let projected = qkv(x).reshaped([batch, length, 3, width])
        var queries = projected[0..., 0..., 0], keys = projected[0..., 0..., 1]
        if let queryNorm, let keyNorm {
            queries = queryNorm(queries)
            keys = keyNorm(keys)
        }
        func split(_ t: MLXArray) -> MLXArray { t.reshaped([batch, length, heads, headDim]).transposed(0, 2, 1, 3) }
        let q = split(queries), k = split(keys), v = split(projected[0..., 0..., 2])
        let out = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale,
                                                    mask: .none)
        return proj(out.transposed(0, 2, 1, 3).reshaped([batch, length, width]))
    }
}

/// The feed-forward block: `fc2(gelu(fc1(x)))`.
final class NFKSa2VAMLP: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    init(_ c: NFKMLXSa2VAConfiguration) {
        _fc1.wrappedValue = Linear(c.visionHiddenSize, c.visionIntermediateSize)
        _fc2.wrappedValue = Linear(c.visionIntermediateSize, c.visionHiddenSize)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(gelu(fc1(x))) }
}

/// A pre-norm block with per-channel LayerScale on each residual. LayerScale multiplies the sub-block
/// output by a learned per-channel vector before it is added back, which is the InternViT feature the
/// plainer SigLIP blocks do not carry.
final class NFKSa2VAEncoderLayer: Module {
    @ModuleInfo(key: "norm1") var norm1: UnaryLayer
    @ModuleInfo(key: "norm2") var norm2: UnaryLayer
    @ModuleInfo(key: "attn") var attention: NFKSa2VAAttention
    @ModuleInfo(key: "mlp") var mlp: NFKSa2VAMLP
    @ParameterInfo(key: "ls1") var layerScale1: MLXArray
    @ParameterInfo(key: "ls2") var layerScale2: MLXArray

    init(_ c: NFKMLXSa2VAConfiguration) {
        func norm() -> UnaryLayer {
            c.visionRMSNorm ? RMSNorm(dimensions: c.visionHiddenSize, eps: c.visionLayerNormEps)
                : LayerNorm(dimensions: c.visionHiddenSize, eps: c.visionLayerNormEps)
        }
        _norm1.wrappedValue = norm()
        _norm2.wrappedValue = norm()
        _attention.wrappedValue = NFKSa2VAAttention(c)
        _mlp.wrappedValue = NFKSa2VAMLP(c)
        _layerScale1.wrappedValue = MLXArray.ones([c.visionHiddenSize])
        _layerScale2.wrappedValue = MLXArray.ones([c.visionHiddenSize])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x + attention(norm1(x)) * layerScale1
        h = h + mlp(norm2(h)) * layerScale2
        return h
    }
}

/// The InternViT-300M image encoder. There is no final layer norm: the features are the last block's
/// output, and `extract_feature` reads `last_hidden_state` directly.
public final class NFKMLXSa2VAVisionNet: Module {
    @ModuleInfo(key: "embeddings") var embeddings: NFKSa2VAVisionEmbeddings
    @ModuleInfo(key: "encoder") var encoder: NFKSa2VAVisionEncoder

    init(_ c: NFKMLXSa2VAConfiguration) {
        _embeddings.wrappedValue = NFKSa2VAVisionEmbeddings(c)
        _encoder.wrappedValue = NFKSa2VAVisionEncoder(c)
        super.init()
    }

    /// `pixelValues` is channels-last `[tiles, height, width, 3]`. Returns `[tiles, 1 + patches, hidden]`.
    func callAsFunction(_ pixelValues: MLXArray) -> MLXArray {
        encoder(embeddings(pixelValues))
    }
}

/// The transformer stack under the checkpoint's `encoder.layers` key.
final class NFKSa2VAVisionEncoder: Module {
    @ModuleInfo(key: "layers") var layers: [NFKSa2VAEncoderLayer]

    init(_ c: NFKMLXSa2VAConfiguration) {
        _layers.wrappedValue = (0 ..< c.visionLayers).map { _ in NFKSa2VAEncoderLayer(c) }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for layer in layers { h = layer(h) }
        return h
    }
}

// MARK: - Projector

/// The `mlp1` projector: InternVL pixel shuffle then a `LayerNorm → Linear → GELU → Linear` stack. The
/// checkpoint stores the stack as a numeric `nn.Sequential` (slots 0/1/3); the loader renames those to
/// `norm`/`fc1`/`fc2` because MLX reads an all-integer-keyed module as an array.
final class NFKSa2VAProjector: Module {
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    let scaleFactor: Double

    init(_ c: NFKMLXSa2VAConfiguration) {
        scaleFactor = c.downsampleRatio
        let shuffledWidth = Int(Double(c.visionHiddenSize) / (c.downsampleRatio * c.downsampleRatio))
        _norm.wrappedValue = LayerNorm(dimensions: shuffledWidth)
        _fc1.wrappedValue = Linear(shuffledWidth, c.decoderHiddenSize)
        _fc2.wrappedValue = Linear(c.decoderHiddenSize, c.decoderHiddenSize)
        super.init()
    }

    /// `x` is the vision features with the class token already dropped, `[tiles, patches, hidden]`.
    /// Returns `[tiles, patches · scale², decoderHidden]`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(gelu(fc1(norm(pixelShuffle(x)))))
    }

    /// InternVL's pixel shuffle (ps_version v2), ported step for step. It folds a `scale` factor of the
    /// width into the channel axis, transposes, folds the height, and — unlike v1 — transposes back so
    /// the token order is row-major rather than transposed.
    private func pixelShuffle(_ input: MLXArray) -> MLXArray {
        let (tiles, sequence, embed) = (input.dim(0), input.dim(1), input.dim(2))
        let height = Int(Double(sequence).squareRoot())
        let width = height
        let shrunkWidth = Int(Double(width) * scaleFactor)
        let shrunkHeight = Int(Double(height) * scaleFactor)
        var x = input.reshaped([tiles, height, width, embed])
        // N, W, H, C -> N, W, H*scale, C/scale
        x = x.reshaped([tiles, height, shrunkWidth, Int(Double(embed) / scaleFactor)])
        // -> N, H*scale, W, C/scale
        x = x.transposed(0, 2, 1, 3)
        // -> N, H*scale, W*scale, C/scale²
        x = x.reshaped([tiles, shrunkWidth, shrunkHeight,
                        Int(Double(embed) / (scaleFactor * scaleFactor))])
        // ps v2 swaps the two spatial axes back
        x = x.transposed(0, 2, 1, 3)
        return x.reshaped([tiles, shrunkWidth * shrunkHeight,
                           Int(Double(embed) / (scaleFactor * scaleFactor))])
    }
}

/// The `[SEG]`-hidden-state bridge: `Linear → ReLU → Linear` mapping a decoder hidden state (2048) to a
/// SAM 2 prompt embedding (256). The checkpoint stores it as an `nn.Sequential` (slots 0/2); the loader
/// renames those to `fc1`/`fc2`.
final class NFKSa2VATextHiddenFCS: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    init(inputSize: Int, outputSize: Int) {
        _fc1.wrappedValue = Linear(inputSize, inputSize)
        _fc2.wrappedValue = Linear(inputSize, outputSize)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(relu(fc1(x))) }
}

// MARK: - The whole model

/// Sa2VA-4B: the InternViT tower, the projector, the Qwen2.5 decoder, the `[SEG]` bridge, and the SAM 2
/// grounding encoder, with the fusion that splices projected vision tokens into the decoder at the
/// image-context positions and the segmentation path that drives SAM 2 from a `[SEG]` hidden state.
public final class NFKMLXSa2VANet: Module {
    @ModuleInfo(key: "vision_model") var vision: NFKMLXSa2VAVisionNet
    @ModuleInfo(key: "mlp1") var projector: NFKSa2VAProjector
    @ModuleInfo(key: "language_model") var language: NFKMLXLanguageNet
    @ModuleInfo(key: "text_hidden_fcs") var textHiddenFCS: NFKSa2VATextHiddenFCS
    @ModuleInfo(key: "grounding_encoder") var grounding: NFKSa2VAGroundingEncoder

    public let configuration: NFKMLXSa2VAConfiguration

    public init(_ c: NFKMLXSa2VAConfiguration) {
        configuration = c
        _vision.wrappedValue = NFKMLXSa2VAVisionNet(c)
        _projector.wrappedValue = NFKSa2VAProjector(c)
        _language.wrappedValue = NFKMLXLanguageNet(c.decoder)
        _textHiddenFCS.wrappedValue = NFKSa2VATextHiddenFCS(
            inputSize: c.decoderHiddenSize, outputSize: c.groundingConfiguration.hiddenDimensions)
        _grounding.wrappedValue = NFKSa2VAGroundingEncoder(c.groundingConfiguration)
        super.init()
    }

    /// The projected vision tokens for a batch of tiles, `[tiles, tokensPerTile, decoderHidden]`.
    /// `pixelValues` is the reference's channels-first `[tiles, 3, imageSize, imageSize]`.
    public func imageFeatures(pixelValues: MLXArray) -> MLXArray {
        let features = vision(pixelValues.transposed(0, 2, 3, 1))       // [tiles, 1+patches, hidden]
        let dropped = features[0..., 1...]                             // drop the class token
        return projector(dropped)                                      // [tiles, tokensPerTile, hidden]
    }

    /// The decoder's input embeddings with the flattened vision tokens spliced in at the image-context
    /// positions, `[1, sequence, decoderHidden]`.
    public func fusedEmbeddings(inputIds: [Int], imageFeatures features: MLXArray) -> MLXArray {
        let width = configuration.decoderHiddenSize
        let flat = features.reshaped([-1, width])
        let sequence = inputIds.count
        let textEmbeddings = language.embed(MLXArray(inputIds.map(Int32.init))
            .reshaped([1, sequence]))[0]

        var featureIndex = [Int32](repeating: 0, count: sequence)
        var isImage = [Float](repeating: 0, count: sequence)
        var counter: Int32 = 0
        for position in 0 ..< sequence where inputIds[position] == configuration.imageContextTokenId {
            featureIndex[position] = counter
            counter += 1
            isImage[position] = 1
        }
        let gathered = flat.take(MLXArray(featureIndex), axis: 0)       // [sequence, width]
        let mask = MLXArray(isImage).reshaped([sequence, 1]) .> 0
        let fused = MLX.where(mask, gathered, textEmbeddings)
        return fused.reshaped([1, sequence, width])
    }

    /// Greedy decoding from a fused prefill. Returns the generated token ids (excluding the stop token)
    /// and the decoder hidden state at each `[SEG]` position, ready for the grounding branch. The
    /// segmentation hidden states are read from one teacher-forced pass over the realized sequence,
    /// which is numerically the causal equal of the per-step states the reference gathers.
    ///
    /// `stopsAfter` sees the tokens generated so far after each one and ends generation, keeping them,
    /// when it returns true: the reference's stop-word criterion is a test of the decoded text (see
    /// ``NFKMLXSa2VATemplate/stops(after:decode:)``). `endToken` stops generation and is reported as
    /// `ending` rather than kept in `tokens`; the reference's decoded answer keeps it.
    public func generate(inputIds: [Int], imageFeatures: MLXArray, maximumTokens: Int, endToken: Int,
                         stopsAfter: (([Int]) -> Bool)? = nil)
        -> (tokens: [Int], ending: Int?, segmentationHiddenStates: [MLXArray]) {
        let cache = NFKMLXKeyValueCache(layerCount: configuration.decoder.layerCount)
        let fused = fusedEmbeddings(inputIds: inputIds, imageFeatures: imageFeatures)
        var hidden = language.hiddenStates(fromEmbeddings: fused, cache: cache)
        var logits = language.logits(fromHidden: hidden[0..., (hidden.dim(1) - 1)...])
        var next = argMax(logits[0, 0], axis: -1).item(Int.self)
        var tokens = [Int]()
        var ending: Int?
        for _ in 0 ..< maximumTokens {
            if next == endToken {
                ending = next
                break
            }
            tokens.append(next)
            if let stopsAfter, stopsAfter(tokens) { break }
            let embedding = language.embed(MLXArray([Int32(next)]).reshaped([1, 1]))
            hidden = language.hiddenStates(fromEmbeddings: embedding, cache: cache)
            logits = language.logits(fromHidden: hidden)
            next = argMax(logits[0, 0], axis: -1).item(Int.self)
        }

        let full = inputIds + tokens
        let fullHidden = language.hiddenStates(
            fromEmbeddings: fusedEmbeddings(inputIds: full, imageFeatures: imageFeatures))
        var segments = [MLXArray]()
        for (position, token) in full.enumerated() where token == configuration.segmentationTokenId {
            segments.append(fullHidden[0, position])
        }
        return (tokens, ending, segments)
    }

    /// A single mask from a `[SEG]` hidden state and the SAM-resolution grounding image. `segHidden` is
    /// one decoder hidden state `[decoderHidden]`; `groundingImage` is the normalized SAM input
    /// `[1, 1024, 1024, 3]`. Returns the best-of-three mask logits at the SAM image resolution
    /// `[1, imageSize, imageSize]` and at the quarter-resolution decoder stride `[1, grid, grid]`.
    public func segment(segHidden: MLXArray, groundingImage: MLXArray)
        -> (highResolution: MLXArray, lowResolution: MLXArray) {
        let embedding = textHiddenFCS(segHidden.reshaped([1, -1]))      // [1, 256]
        return grounding.segment(image: groundingImage,
                                 languageEmbedding: embedding.reshaped([1, 1, -1]))
    }
}

/// The SAM 2 grounding encoder, `grounding_encoder.sam2_model.*` in the checkpoint. It wraps the shared
/// SAM 2 tracker and drives it the way Sa2VA does: the initial conditioning frame adds the tracker's
/// `no_mem_embed` (no memory attention runs), the `[SEG]` embedding is concatenated to the empty-point
/// sparse prompt, and the object-score suppression the tracker applies to a click-driven frame is
/// dropped, matching the reference's commented-out `torch.where`.
public final class NFKSa2VAGroundingEncoder: Module {
    @ModuleInfo(key: "sam2_model") var tracker: NFKMLXSAM2TrackerNet

    let configuration: NFKMLXSAM2TrackerConfiguration

    init(_ c: NFKMLXSAM2TrackerConfiguration) {
        configuration = c
        _tracker.wrappedValue = NFKMLXSAM2TrackerNet(c)
        super.init()
    }

    func segment(image: MLXArray, languageEmbedding: MLXArray)
        -> (highResolution: MLXArray, lowResolution: MLXArray) {
        segment(levels: tracker.imageEncoder.features(image), languageEmbedding: languageEmbedding)
    }

    /// The image's feature levels, computed once for every object a step segments.
    func imageLevels(_ image: MLXArray) -> [MLXArray] { tracker.imageEncoder.features(image) }

    /// ``segment(image:languageEmbedding:)`` from already-computed feature levels.
    func segment(levels: [MLXArray], languageEmbedding: MLXArray)
        -> (highResolution: MLXArray, lowResolution: MLXArray) {
        let hidden = configuration.hiddenDimensions
        let grid = configuration.featureGrid
        let conditioned = levels[levels.count - 1]
            + tracker.noMemoryEmbedding.reshaped([1, 1, 1, hidden])

        // The empty point the reference pads with, then the `[SEG]` token concatenated onto the sparse
        // prompt exactly as `_forward_sam_heads` does.
        let emptyPoint = tracker.promptEncoder.sparse(points: [(x: Float(0), y: Float(0), label: -1)])
        let sparse = concatenated([emptyPoint, languageEmbedding], axis: 1)
        let dense = tracker.promptEncoder.dense(grid: grid)
        let positional = tracker.promptEncoder.positionEncoding.grid(grid, grid)
        let decoded = tracker.maskDecoder(features: conditioned, positional: positional, sparse: sparse,
                                          dense: dense, highResolution: Array(levels.dropLast()))

        // Multimask: three candidate masks in slots 1…3, choose the one the decoder scores highest.
        // No object-score suppression — Sa2VA leaves the mask untouched.
        let candidateIoU = decoded.iou[0..., 1...]
        let best = argMax(candidateIoU, axis: -1).item(Int.self)
        let low = decoded.masks[0..., 1 + best]                        // [1, grid, grid]
        let high = NFKMLXResample.resizeBilinear(
            low.expandedDimensions(axis: 3),
            height: configuration.imageSize, width: configuration.imageSize)
        return (high.squeezed(axis: 3), low)
    }
}
