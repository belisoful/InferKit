//
//  NFKMLXSa2VALLaVA.swift
//  InferKitMLX
//
//  Sa2VA on LLaVA-1.5 (`ByteDance/Sa2VA-LLaVA-1.5-7B`): transformers' `LlavaForConditionalGeneration`
//  under the checkpoint's `model.` — a CLIP ViT-L/14-336 read at its second-to-last layer with the class
//  token dropped, a Linear → GELU → Linear projector, and a Vicuna-7B (Llama) decoder — with Sa2VA's
//  `[SEG]` bridge and SAM 2 Hiera-Large grounding. The vision tower here keeps transformers' CLIP
//  layout (`vision_model.encoder.layers.N.self_attn.q_proj`, `layer_norm1`, `pre_layrnorm`), which the
//  toolkit's OpenAI-layout CLIP does not load.
//

import Foundation
import CoreGraphics
import InferKit
import MLX
import MLXNN

/// The geometry of a transformers CLIP vision model.
struct NFKLLaVAVisionConfiguration: Sendable {
    var hiddenSize = 1024
    var layers = 24
    var heads = 16
    var intermediateSize = 4096
    var imageSize = 336
    var patchSize = 14
    var layerNormEps: Float = 1e-5

    init(json: [String: Any]) {
        func int(_ key: String, _ fallback: Int) -> Int { (json[key] as? NSNumber)?.intValue ?? fallback }
        hiddenSize = int("hidden_size", hiddenSize)
        layers = int("num_hidden_layers", layers)
        heads = int("num_attention_heads", heads)
        intermediateSize = int("intermediate_size", intermediateSize)
        imageSize = int("image_size", imageSize)
        patchSize = int("patch_size", patchSize)
        layerNormEps = (json["layer_norm_eps"] as? NSNumber)?.floatValue ?? layerNormEps
    }

    var grid: Int { imageSize / patchSize }
}

/// One CLIP encoder layer: pre-norm attention with biased projections, then a quick-GELU MLP.
final class NFKLLaVAVisionLayer: Module {
    final class Attention: Module {
        @ModuleInfo(key: "q_proj") var q: Linear
        @ModuleInfo(key: "k_proj") var k: Linear
        @ModuleInfo(key: "v_proj") var v: Linear
        @ModuleInfo(key: "out_proj") var out: Linear
        init(_ width: Int) {
            _q.wrappedValue = Linear(width, width)
            _k.wrappedValue = Linear(width, width)
            _v.wrappedValue = Linear(width, width)
            _out.wrappedValue = Linear(width, width)
        }
    }
    final class MLP: Module {
        @ModuleInfo(key: "fc1") var fc1: Linear
        @ModuleInfo(key: "fc2") var fc2: Linear
        init(_ width: Int, _ inner: Int) {
            _fc1.wrappedValue = Linear(width, inner)
            _fc2.wrappedValue = Linear(inner, width)
        }
        /// CLIP's `quick_gelu`, `x · σ(1.702 x)`.
        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let h = fc1(x)
            return fc2(h * sigmoid(1.702 * h))
        }
    }
    @ModuleInfo(key: "layer_norm1") var norm1: LayerNorm
    @ModuleInfo(key: "self_attn") var attention: Attention
    @ModuleInfo(key: "layer_norm2") var norm2: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: MLP
    let heads: Int

    init(_ c: NFKLLaVAVisionConfiguration) {
        heads = c.heads
        _norm1.wrappedValue = LayerNorm(dimensions: c.hiddenSize, eps: c.layerNormEps)
        _attention.wrappedValue = Attention(c.hiddenSize)
        _norm2.wrappedValue = LayerNorm(dimensions: c.hiddenSize, eps: c.layerNormEps)
        _mlp.wrappedValue = MLP(c.hiddenSize, c.intermediateSize)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let n = norm1(x)
        let headDim = x.dim(2) / heads
        func split(_ t: MLXArray) -> MLXArray { t.reshaped([t.dim(0), t.dim(1), heads, headDim]).transposed(0, 2, 1, 3) }
        let attended = MLXFast.scaledDotProductAttention(
            queries: split(attention.q(n)), keys: split(attention.k(n)), values: split(attention.v(n)),
            scale: 1 / sqrt(Float(headDim)), mask: nil)
        let h = x + attention.out(attended.transposed(0, 2, 1, 3).reshaped(x.shape))
        return h + mlp(norm2(h))
    }
}

/// The transformers CLIP vision model under `vision_model.`.
final class NFKLLaVAVisionModel: Module {
    final class Embeddings: Module {
        @ModuleInfo(key: "patch_embedding") var patchEmbedding: Conv2d
        @ParameterInfo(key: "class_embedding") var classEmbedding: MLXArray
        @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding
        init(_ c: NFKLLaVAVisionConfiguration) {
            _patchEmbedding.wrappedValue = Conv2d(inputChannels: 3, outputChannels: c.hiddenSize,
                                                  kernelSize: IntOrPair(c.patchSize), stride: IntOrPair(c.patchSize),
                                                  bias: false)
            _classEmbedding.wrappedValue = MLXArray.zeros([c.hiddenSize])
            _positionEmbedding.wrappedValue = Embedding(embeddingCount: c.grid * c.grid + 1, dimensions: c.hiddenSize)
        }
    }
    final class Encoder: Module {
        @ModuleInfo(key: "layers") var layers: [NFKLLaVAVisionLayer]
        init(_ c: NFKLLaVAVisionConfiguration) {
            _layers.wrappedValue = (0 ..< c.layers).map { _ in NFKLLaVAVisionLayer(c) }
        }
    }
    @ModuleInfo(key: "embeddings") var embeddings: Embeddings
    @ModuleInfo(key: "pre_layrnorm") var preNorm: LayerNorm
    @ModuleInfo(key: "encoder") var encoder: Encoder
    @ModuleInfo(key: "post_layernorm") var postNorm: LayerNorm

    init(_ c: NFKLLaVAVisionConfiguration) {
        _embeddings.wrappedValue = Embeddings(c)
        _preNorm.wrappedValue = LayerNorm(dimensions: c.hiddenSize, eps: c.layerNormEps)
        _encoder.wrappedValue = Encoder(c)
        _postNorm.wrappedValue = LayerNorm(dimensions: c.hiddenSize, eps: c.layerNormEps)
    }

    /// The hidden state after `layerCount` encoder layers (transformers' `hidden_states[layerCount]`),
    /// `[B, 1 + patches, hidden]`, from normalized pixels `[B, H, W, 3]`.
    func hiddenState(_ pixels: MLXArray, afterLayers layerCount: Int) -> MLXArray {
        let patches = embeddings.patchEmbedding(pixels)
        let batch = patches.dim(0), width = patches.dim(3)
        var x = patches.reshaped([batch, -1, width])
        let cls = broadcast(embeddings.classEmbedding.reshaped([1, 1, width]), to: [batch, 1, width])
        x = concatenated([cls, x], axis: 1)
        x = x + embeddings.positionEmbedding.weight[0 ..< x.dim(1)]
        x = preNorm(x)
        for layer in encoder.layers.prefix(layerCount) { x = layer(x) }
        return x
    }
}

/// The LLaVA projector (`multi_modal_projector`): `linear_2(gelu(linear_1(x)))`.
final class NFKLLaVAProjector: Module {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear
    init(visionWidth: Int, decoderWidth: Int) {
        _linear1.wrappedValue = Linear(visionWidth, decoderWidth)
        _linear2.wrappedValue = Linear(decoderWidth, decoderWidth)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { linear2(gelu(linear1(x))) }
}

/// Sa2VA on LLaVA-1.5: the CLIP tower, the projector, the Llama decoder, the `[SEG]` bridge, and SAM 2
/// grounding.
///
/// @discussion The reference resizes the image to 336 bicubic (PIL, through torchvision), reads the CLIP
/// tower's second-to-last hidden state without its class token, projects the 576 patch features, and
/// splices them at the 576 `<image>` tokens of a Vicuna instruction (`USER: … ASSISTANT:`). Generation is
/// greedy to `</s>`; each `[SEG]` hidden state drives SAM 2 as in the other releases.
///
/// Introduced in InferKit 0.4.0.
public final class NFKMLXSa2VALLaVANet: Module {
    @ModuleInfo(key: "vision") var vision: NFKLLaVAVisionModel
    @ModuleInfo(key: "projector") var projector: NFKLLaVAProjector
    @ModuleInfo(key: "language_model") var language: NFKMLXLanguageNet
    @ModuleInfo(key: "text_hidden_fcs") var textHiddenFCS: NFKSa2VATextHiddenFCS
    @ModuleInfo(key: "grounding_encoder") var grounding: NFKSa2VAGroundingEncoder

    let visionConfiguration: NFKLLaVAVisionConfiguration
    /// The `<image>` id the vision features splice at.
    public let imageTokenId: Int
    /// The `[SEG]` id.
    public let segmentationTokenId: Int
    /// The hidden-state index the projector reads (`vision_feature_layer`, -2: the second-to-last).
    let featureLayer: Int

    init(vision: NFKLLaVAVisionConfiguration, decoder: NFKMLXLanguageConfiguration, imageTokenId: Int,
         segmentationTokenId: Int, featureLayer: Int) {
        visionConfiguration = vision
        self.imageTokenId = imageTokenId
        self.segmentationTokenId = segmentationTokenId
        self.featureLayer = featureLayer
        let grounding = NFKMLXSAM2TrackerConfiguration.geometry(.large, release: .sam2)
        _vision.wrappedValue = NFKLLaVAVisionModel(vision)
        _projector.wrappedValue = NFKLLaVAProjector(visionWidth: vision.hiddenSize, decoderWidth: decoder.hiddenSize)
        _language.wrappedValue = NFKMLXLanguageNet(decoder)
        _textHiddenFCS.wrappedValue = NFKSa2VATextHiddenFCS(inputSize: decoder.hiddenSize,
                                                           outputSize: grounding.hiddenDimensions)
        _grounding.wrappedValue = NFKSa2VAGroundingEncoder(grounding)
        super.init()
    }

    /// Builds the network from a released directory and loads it at float32.
    public static func load(directoryURL: URL) throws -> NFKMLXSa2VALLaVANet {
        let data = try Data(contentsOf: directoryURL.appendingPathComponent("config.json"))
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let text = json["text_config"] as? [String: Any],
              let visionJSON = json["vision_config"] as? [String: Any] else {
            throw NFKMLXError.unsupportedConfiguration("Sa2VA-LLaVA config.json lacks text_config or vision_config")
        }
        var decoder = try NFKMLXLanguage.configuration(fromJSON: text)
        decoder.vocabularySize = (json["vocab_size"] as? NSNumber)?.intValue ?? decoder.vocabularySize
        decoder.tiesWordEmbeddings = false
        let layer = (json["vision_feature_layer"] as? NSNumber)?.intValue ?? -2
        let net = NFKMLXSa2VALLaVANet(
            vision: NFKLLaVAVisionConfiguration(json: visionJSON), decoder: decoder,
            imageTokenId: (json["image_token_index"] as? NSNumber)?.intValue ?? 32_000,
            segmentationTokenId: NFKMLXSa2VAProcessor.tokenId("[SEG]", inDirectory: directoryURL) ?? 32_004,
            featureLayer: layer)
        try net.loadWeights(fromDirectory: directoryURL)
        return net
    }

    /// The checkpoint's names in this module's: `model.model.vision_tower.vision_model.` → `vision.`,
    /// `model.model.multi_modal_projector.` → `projector.`, `model.model.language_model.` →
    /// `language_model.model.`, `model.lm_head.` → `language_model.lm_head.`, and the bridge and
    /// grounding through Sa2VA-4B's remaps.
    func loadWeights(fromDirectory directory: URL) throws {
        // A directory `NFKMLXSa2VA.save` wrote (a fine-tune) holds this module's own names and layout.
        let saved = directory.appendingPathComponent("model.safetensors")
        if FileManager.default.fileExists(atPath: saved.path),
           let checkpoint = try? NFKMLXWeights.loadCheckpoint(url: saved), !checkpoint.needsConvTranspose {
            try NFKMLXWeights.apply(checkpoint.arrays.map { ($0.key, $0.value) }, to: self)
            return
        }
        let arrays = try NFKMLXReleaseWeights.arrays(inDirectory: directory)
        var mapped = [(String, MLXArray)]()
        let grounding = "grounding_encoder.sam2_model."
        let renames = [("model.model.vision_tower.vision_model.", "vision."),
                       ("model.model.multi_modal_projector.", "projector."),
                       ("model.model.language_model.", "language_model.model."),
                       ("model.lm_head.", "language_model.lm_head.")]
        for (key, value) in arrays {
            if key.hasPrefix(grounding) {
                guard let remapped = NFKMLXSAM2.remapTrackerKey(String(key.dropFirst(grounding.count))) else { continue }
                let named = remapped.replacingOccurrences(of: ".g_weight", with: ".gamma")
                mapped.append((grounding + named, NFKMLXSa2VANet.groundingTransposed(named, value)))
            } else if key.hasPrefix("text_hidden_fcs.") {
                mapped.append((NFKMLXSa2VANet.sequentialName(key), value))
            } else if let (from, to) = renames.first(where: { key.hasPrefix($0.0) }) {
                let name = to + key.dropFirst(from.count)
                let converted = name == "vision.embeddings.patch_embedding.weight" && value.ndim == 4
                    ? value.transposed(0, 2, 3, 1) : value
                mapped.append((name, converted.asType(.float32)))
            }
        }
        try NFKMLXWeights.apply(mapped, to: self)
    }

    /// The projected patch features `[1, patches, decoderHidden]` of normalized pixels `[1, 336, 336, 3]`.
    public func imageFeatures(_ pixels: MLXArray) -> MLXArray {
        let layers = featureLayer < 0 ? visionConfiguration.layers + 1 + featureLayer : featureLayer
        let hidden = vision.hiddenState(pixels, afterLayers: layers)
        return projector(hidden[0..., 1...])
    }

    /// The decoder's input embeddings with the patch features spliced at the `<image>` positions.
    public func fusedEmbeddings(inputIds: [Int], imageFeatures features: MLXArray) -> MLXArray {
        let width = language.configuration.hiddenSize
        let flat = features.reshaped([-1, width])
        let sequence = inputIds.count
        let text = language.embed(MLXArray(inputIds.map(Int32.init)).reshaped([1, sequence]))[0]
        var index = [Int32](repeating: 0, count: sequence)
        var isImage = [Float](repeating: 0, count: sequence)
        var counter: Int32 = 0
        for position in 0 ..< sequence where inputIds[position] == imageTokenId {
            index[position] = counter
            counter += 1
            isImage[position] = 1
        }
        let gathered = flat.take(MLXArray(index), axis: 0)
        let mask = MLXArray(isImage).reshaped([sequence, 1]) .> 0
        return MLX.where(mask, gathered, text).reshaped([1, sequence, width])
    }

    /// Greedy generation to `endToken`, then each `[SEG]`'s final hidden state from one teacher-forced
    /// pass over the realized sequence.
    public func generate(inputIds: [Int], imageFeatures features: MLXArray, maximumTokens: Int, endToken: Int)
        -> (tokens: [Int], ending: Int?, segmentationHiddenStates: [MLXArray]) {
        let cache = NFKMLXKeyValueCache(layerCount: language.configuration.layerCount)
        var hidden = language.hiddenStates(fromEmbeddings: fusedEmbeddings(inputIds: inputIds, imageFeatures: features),
                                           cache: cache)
        var tokens = [Int]()
        var ending: Int?
        for _ in 0 ..< maximumTokens {
            let next = argMax(language.logits(fromHidden: hidden[0..., (hidden.dim(1) - 1)...])[0, 0], axis: -1)
                .item(Int.self)
            if next == endToken {
                ending = next
                break
            }
            tokens.append(next)
            hidden = language.hiddenStates(fromEmbeddings: language.embed(MLXArray([Int32(next)]).reshaped([1, 1])),
                                           cache: cache)
        }
        let full = inputIds + tokens
        guard full.contains(segmentationTokenId) else { return (tokens, ending, []) }
        let fullHidden = language.hiddenStates(fromEmbeddings: fusedEmbeddings(inputIds: full, imageFeatures: features))
        return (tokens, ending, full.enumerated().filter { $0.element == segmentationTokenId }.map { fullHidden[0, $0.offset] })
    }

    /// A mask from a `[SEG]` hidden state and the normalized SAM input `[1, 1024, 1024, 3]`.
    public func segment(segHidden: MLXArray, groundingImage: MLXArray)
        -> (highResolution: MLXArray, lowResolution: MLXArray) {
        let embedding = textHiddenFCS(segHidden.reshaped([1, -1]))
        return grounding.segment(image: groundingImage, languageEmbedding: embedding.reshaped([1, 1, -1]))
    }
}

/// The LLaVA release's processing: the Vicuna instruction and the CLIP image transform.
public enum NFKMLXSa2VALLaVA {
    static let clipMean: [Float] = [0.48145466, 0.4578275, 0.40821073]
    static let clipStd: [Float] = [0.26862954, 0.26130258, 0.27577711]

    /// The Vicuna instruction with `<image>` expanded to `imageTokens` image tokens and a newline.
    public static func promptText(_ text: String, imageTokens: Int) -> String {
        let images = String(repeating: "<image>", count: imageTokens) + "\n"
        let body = text.contains("<image>") ? text.replacingOccurrences(of: "<image>", with: images) : images + text
        return "USER: " + body + " ASSISTANT:"
    }

    /// The normalized `[1, side, side, 3]` pixels: the 8-bit image resized to `side` with PIL's bicubic
    /// (torchvision's `Resize` on a PIL image), scaled to `0…1`, and CLIP-normalized.
    public static func pixelValues(_ image: Any, side: Int = 336) throws -> MLXArray {
        let (rgb, width, height) = try NFKMLXSa2VAProcessor.rgbBytes(image)
        let resized = NFKMLXPILResample.resampled(rgb, width: width, height: height, toWidth: side, toHeight: side,
                                                  filter: .bicubic)
        let pixels = MLXArray(resized.map { Float($0) / 255 }).reshaped([1, side, side, 3])
        return (pixels - MLXArray(clipMean).reshaped([1, 1, 1, 3])) / MLXArray(clipStd).reshaped([1, 1, 1, 3])
    }
}
