//
//  NFKMLXGemma3Vision.swift
//  InferKitMLX
//
//  The vision side of the multimodal Gemma 3 releases (4B, 12B, 27B): a SigLIP so400m vision tower at
//  896×896, a projector that average-pools its 4096 patch features to the 256 soft tokens the decoder
//  reads, and the image processor and prompt expansion that put them in front of the text.
//
//  The tower is the same SigLIP transformer SmolVLM and SigLIP 2 run (`NFKSigLIPEncoder` and its layers
//  are reused), read with row-major position ids; nothing in it is new. The projector is Gemma's:
//  an average pool over the 64×64 patch grid in 4×4 cells, a Gemma `(1 + w)` RMS norm over the pooled
//  features, and one bias-free matrix into the decoder's width.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import CoreGraphics
import InferKit
import MLX
import MLXNN
import MLXRandom

public extension NFKMLXSigLIPConfiguration {
    /// The released Gemma 3 vision tower: SigLIP so400m at 896×896 with 14-pixel patches, 27 layers of
    /// 16 heads over 1152 channels, no pooling head.
    static let gemma3 = NFKMLXSigLIPConfiguration(
        hiddenSize: 1152, layerCount: 27, headCount: 16, intermediateSize: 4304, patchSize: 14,
        imageSize: 896)
}

/// The Gemma 3 vision tower: the SigLIP patch embedding read row-major, the shared encoder, and the
/// post layer-norm. Returns the patch features `[images, patches, hidden]`.
public final class NFKMLXGemma3VisionNet: Module {
    @ModuleInfo(key: "embeddings") var embeddings: NFKSigLIP2VisionEmbeddings
    @ModuleInfo(key: "encoder") var encoder: NFKSigLIPEncoder
    @ModuleInfo(key: "post_layernorm") var postLayerNorm: LayerNorm

    public let configuration: NFKMLXSigLIPConfiguration

    public init(_ c: NFKMLXSigLIPConfiguration = .gemma3) {
        configuration = c
        _embeddings.wrappedValue = NFKSigLIP2VisionEmbeddings(c)
        _encoder.wrappedValue = NFKSigLIPEncoder(c)
        _postLayerNorm.wrappedValue = LayerNorm(dimensions: c.hiddenSize, eps: c.layerNormEpsilon)
        super.init()
    }

    /// `pixelValues` is `[images, height, width, 3]` (channels last) in `-1 … 1`.
    public func callAsFunction(_ pixelValues: MLXArray) -> MLXArray {
        var hidden = embeddings(pixelValues)
        for layer in encoder.layers { hidden = layer(hidden) }
        return postLayerNorm(hidden)
    }
}

/// Gemma 3's multimodal projector: the patch features are average-pooled from the `side × side` patch
/// grid to `tokensPerSide × tokensPerSide` cells, normalized by a Gemma RMS norm, and multiplied into
/// the decoder's width.
public final class NFKMLXGemma3MultimodalProjector: Module {
    @ModuleInfo(key: "mm_soft_emb_norm") var norm: NFKGemma3Norm
    /// `[visionHidden, textHidden]`, applied as `x · W` (the reference's `matmul`, not a `Linear`).
    @ParameterInfo(key: "mm_input_projection_weight") var weight: MLXArray

    /// Patches along one side of the image (64 for 896 / 14).
    public let patchesPerSide: Int
    /// Soft tokens along one side (16 for the 256 tokens per image).
    public let tokensPerSide: Int

    public init(visionHidden: Int, textHidden: Int, patchesPerSide: Int, tokensPerImage: Int = 256,
                eps: Float = 1e-6) {
        _norm.wrappedValue = NFKGemma3Norm(dimensions: visionHidden, eps: eps)
        _weight.wrappedValue = MLXArray.zeros([visionHidden, textHidden])
        self.patchesPerSide = patchesPerSide
        tokensPerSide = Int(Double(tokensPerImage).squareRoot())
        super.init()
    }

    /// `[images, patches, visionHidden]` → `[images, tokens, textHidden]`.
    public func callAsFunction(_ features: MLXArray) -> MLXArray {
        let (images, hidden) = (features.shape[0], features.shape[2])
        let kernel = patchesPerSide / tokensPerSide
        // The reference reshapes to a `[side, side]` grid and average-pools in `kernel × kernel` cells;
        // the pooled grid then flattens row-major into the token order.
        let grid = features.reshaped([images, tokensPerSide, kernel, tokensPerSide, kernel, hidden])
        let pooled = grid.mean(axes: [2, 4]).reshaped([images, tokensPerSide * tokensPerSide, hidden])
        return matmul(norm(pooled), weight)
    }
}

/// Gemma 3's image processor: a `CGImage` becomes one `[1, 896, 896, 3]` tile in `-1 … 1`
/// (`rescale 1/255`, mean 0.5, std 0.5), the aspect ratio squashed to the square.
///
/// @discussion The resize is CoreGraphics bilinear rather than the reference's PIL bilinear, so the
/// pixel values differ slightly and an answer may not be token-identical to the reference's; the
/// network is at reference parity on the reference's own pixel values. Pan-and-scan (the optional
/// crops the reference processor can add) is off in every release and is not reproduced.
public enum NFKMLXGemma3ImageProcessor {
    /// `image` is a `CGImage`, `CVPixelBuffer`, or `MTLTexture` — what `NFKInputImage` carries.
    public static func pixelValues(from image: Any, imageSize: Int = 896) throws -> MLXArray {
        let rgb = try NFKMLXImageBridge.tensor(from: image, channels: 3, colorSpace: CGColorSpaceCreateDeviceRGB())
        let batched = rgb.reshaped([1, rgb.shape[0], rgb.shape[1], rgb.shape[2]])          // [1, H, W, 3] in 0…1
        return NFKMLXResample.resizeBilinear(batched, height: imageSize, width: imageSize) * 2 - 1
    }
}

/// The token ids a Gemma 3 release names for its markers.
public struct NFKMLXGemma3Tokens: Sendable {
    public var beginOfSequence: Int
    public var endOfSequence: Int
    public var startOfTurn: Int
    public var endOfTurn: Int
    public var startOfImage: Int
    public var endOfImage: Int
    public var imageSoftToken: Int
    public var padToken: Int
    /// Soft tokens the processor expands one image into (`mm_tokens_per_image`).
    public var tokensPerImage: Int

    public init(beginOfSequence: Int = 2, endOfSequence: Int = 1, startOfTurn: Int = 105, endOfTurn: Int = 106,
                startOfImage: Int = 255_999, endOfImage: Int = 256_000, imageSoftToken: Int = 262_144,
                padToken: Int = 0, tokensPerImage: Int = 256) {
        self.beginOfSequence = beginOfSequence
        self.endOfSequence = endOfSequence
        self.startOfTurn = startOfTurn
        self.endOfTurn = endOfTurn
        self.startOfImage = startOfImage
        self.endOfImage = endOfImage
        self.imageSoftToken = imageSoftToken
        self.padToken = padToken
        self.tokensPerImage = tokensPerImage
    }

    /// The ids one image occupies in a prompt: the reference processor's
    /// `\n\n<start_of_image>` + 256 × `<image_soft_token>` + `<end_of_image>\n\n`, with the two
    /// double-newlines supplied as `newlineIds` (what the tokenizer encodes `"\n\n"` to).
    public func imageSequence(newlineIds: [Int]) -> [Int] {
        newlineIds + [startOfImage] + Array(repeating: imageSoftToken, count: tokensPerImage)
            + [endOfImage] + newlineIds
    }
}

/// A Gemma 3 release as one model: the decoder, the tokenizer, the chat template, and — in a
/// multimodal release — the vision tower and projector, with the fusion that puts an image's soft
/// tokens in front of a question and the bidirectional attention among them.
///
/// @discussion Text runs through ``generate(tokens:image:options:onToken:)``: the prompt (with the
/// image's soft tokens spliced at its placeholders) is prefilled through a hybrid cache, then one
/// token at a time decodes against it. An image's placeholder run attends to itself in both
/// directions, the reference's blockwise rule, which the prefill masks carry; the decode steps are
/// plain causal, as the reference's are.
public final class NFKMLXGemma3Model {
    public let decoder: NFKMLXGemma3Net
    public let vision: NFKMLXGemma3VisionNet?
    public let projector: NFKMLXGemma3MultimodalProjector?
    let tokenizer: NFKMLXGemmaTokenizer
    public let tokens: NFKMLXGemma3Tokens
    /// The release's own Jinja chat template, or nil to build Gemma's turns by hand.
    let chatTemplate: String?

    init(decoder: NFKMLXGemma3Net, vision: NFKMLXGemma3VisionNet?, projector: NFKMLXGemma3MultimodalProjector?,
         tokenizer: NFKMLXGemmaTokenizer, tokens: NFKMLXGemma3Tokens, chatTemplate: String?) {
        self.decoder = decoder
        self.vision = vision
        self.projector = projector
        self.tokenizer = tokenizer
        self.tokens = tokens
        self.chatTemplate = chatTemplate
    }

    /// Whether the release carries a vision tower, so an image can be asked about.
    public var acceptsImages: Bool { vision != nil && projector != nil }

    // MARK: Prompts

    /// The reference processor's spelling of one image in a prompt: `\n\n<start_of_image>`, the soft
    /// token repeated `tokensPerImage` times, `<end_of_image>\n\n`.
    ///
    /// @discussion The expansion happens in the TEXT, before tokenizing, as the processor does it —
    /// the newlines around the markers merge with the neighbouring text's (`user\n` + `\n\n` is ONE
    /// token, 109), so expanding after tokenizing would read a different sentence.
    public var imageSequenceText: String {
        "\n\n<start_of_image>" + String(repeating: "<image_soft_token>", count: tokens.tokensPerImage)
            + "<end_of_image>\n\n"
    }

    /// Every lone `<start_of_image>` in `text` replaced by the full image sequence; a marker already
    /// followed by a soft token is left as it is.
    public func expandingImageMarkers(in text: String) -> String {
        let marker = "<start_of_image>"
        guard text.contains(marker) else { return text }
        var result = ""
        var rest = Substring(text)
        while let range = rest.range(of: marker) {
            result += rest[..<range.lowerBound]
            let after = rest[range.upperBound...]
            if after.hasPrefix("<image_soft_token>") {
                result += marker
            } else {
                result += imageSequenceText
            }
            rest = after
        }
        return result + rest
    }

    /// The ids of a raw prompt: BOS, then the text. With an image attached, the prompt's own
    /// `<start_of_image>` marks where the image goes; a prompt without one gets the image before its
    /// text, as the reference processor places a leading image.
    public func promptTokens(_ prompt: String, withImage: Bool = false) -> [Int] {
        var text = prompt
        if withImage, !text.contains("<start_of_image>") { text = "<start_of_image>" + text }
        return [tokens.beginOfSequence] + tokenizer.encode(expandingImageMarkers(in: text))
    }

    /// The ids of a chat: the release's template rendered over the messages with the assistant turn
    /// opened, the image markers expanded, then tokenized with the markers as ids. An attached image
    /// goes where the LAST user message spells `<start_of_image>`, or at the start of its content
    /// when it spells none (the reference processor's image-then-text order).
    public func chatTokens(messages: [[AnyHashable: Any]], withImage: Bool = false) -> [Int] {
        var rendered = [[String: Any]]()
        let lastUser = messages.lastIndex { ($0["role"] as? String ?? "user") == "user" }
        for (index, message) in messages.enumerated() {
            let role = message["role"] as? String ?? "user"
            var content = (message["content"] as? String) ?? ""
            if withImage, index == lastUser, !content.contains("<start_of_image>") {
                content = "<start_of_image>" + content
            }
            rendered.append(["role": role, "content": content])
        }
        let text: String
        if let chatTemplate,
           let output = try? NFKMLXChatTemplateRenderer.render(chatTemplate, messages: rendered,
                                                               addGenerationPrompt: true,
                                                               bosToken: "<bos>", eosToken: "<eos>") {
            text = output
        } else {
            text = Self.handRenderedChat(rendered)
        }
        var ids = tokenizer.encode(expandingImageMarkers(in: text))
        if ids.first != tokens.beginOfSequence { ids.insert(tokens.beginOfSequence, at: 0) }
        return ids
    }

    /// Gemma's turn format, for a release without a readable template: `<bos>`, then each turn as
    /// `<start_of_turn>role\ncontent<end_of_turn>\n` (assistant spelled `model`), then the model turn opened.
    static func handRenderedChat(_ messages: [[String: Any]]) -> String {
        var text = "<bos>"
        var firstUserPrefix = ""
        var turns = messages
        if let first = turns.first, (first["role"] as? String) == "system" {
            firstUserPrefix = ((first["content"] as? String) ?? "") + "\n\n"
            turns.removeFirst()
        }
        for (index, message) in turns.enumerated() {
            let role = (message["role"] as? String) == "assistant" ? "model" : (message["role"] as? String ?? "user")
            let content = ((message["content"] as? String) ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
            text += "<start_of_turn>\(role)\n" + (index == 0 ? firstUserPrefix : "") + content + "<end_of_turn>\n"
        }
        return text + "<start_of_turn>model\n"
    }

    /// Each position's image index, or -1 for text: consecutive soft tokens form one block, a new
    /// block beginning wherever a soft token follows a non-soft one — the reference's
    /// `get_block_sequence_ids_for_mask`.
    public func blockIds(for ids: [Int]) -> [Int]? {
        guard ids.contains(tokens.imageSoftToken) else { return nil }
        var blocks = [Int](repeating: -1, count: ids.count)
        var current = -1
        for (index, id) in ids.enumerated() where id == tokens.imageSoftToken {
            if index == 0 || ids[index - 1] != tokens.imageSoftToken { current += 1 }
            blocks[index] = current
        }
        return blocks
    }

    // MARK: Fusion

    /// The projected soft tokens for an image, `[1, tokensPerImage, hidden]`.
    public func softTokens(for image: Any) throws -> MLXArray {
        guard let vision, let projector else { throw NFKMLXError.unsupportedInput }
        let pixels = try NFKMLXGemma3ImageProcessor.pixelValues(from: image, imageSize: vision.configuration.imageSize)
        return projector(vision(pixels))
    }

    /// The decoder's input embeddings with the soft tokens spliced at the placeholder positions.
    public func fusedEmbeddings(tokens ids: [Int], softTokens: MLXArray?) -> MLXArray {
        // A placeholder id past the embedding table (the 262144 image token against a 262144-row
        // table, as the text-only sizes carry) embeds as the pad token, as the reference substitutes.
        let rows = decoder.configuration.vocabularySize
        let embedded = ids.map { $0 == tokens.imageSoftToken && $0 >= rows ? tokens.padToken : $0 }
        let embeddings = decoder.embed(MLXArray(embedded.map(Int32.init)).reshaped([1, ids.count]))
        guard let softTokens else { return embeddings }
        return NFKMLXGemma4Fusion.fuse(textEmbeddings: embeddings, softTokens: softTokens,
                                       isPlaceholder: ids.map { $0 == tokens.imageSoftToken })
    }

    /// The logits over a whole fused sequence, `[1, length, vocabulary]` — the prefill with no cache,
    /// which is what a parity record compares against.
    public func logits(tokens ids: [Int], softTokens: MLXArray?) -> MLXArray {
        let hidden = decoder.hiddenStates(fromEmbeddings: fusedEmbeddings(tokens: ids, softTokens: softTokens),
                                          blockIds: blockIds(for: ids))
        return decoder.logits(fromHidden: hidden)
    }

    // MARK: Generation

    /// Generates a continuation of `ids` (with `image`'s soft tokens at its placeholders), one token
    /// at a time through a hybrid cache, reporting each token to `onToken` and stopping at
    /// `maxTokens`, a stop token, or when `onToken` returns false.
    @discardableResult
    public func generate(tokens ids: [Int], image: Any? = nil,
                         options: NFKMLXGenerationOptions = NFKMLXGenerationOptions(),
                         onToken: (Int) -> Bool = { _ in true }) throws -> [Int] {
        let soft = try image.map { try softTokens(for: $0) }
        let cache = NFKMLXGemma3Cache(layerCount: decoder.configuration.layerCount,
                                      slidingWindow: decoder.configuration.slidingWindow)
        if let seed = options.seed { MLXRandom.seed(seed) }
        let stops = options.stopTokens.isEmpty ? Set([tokens.endOfSequence, tokens.endOfTurn]) : options.stopTokens

        var hidden = decoder.hiddenStates(fromEmbeddings: fusedEmbeddings(tokens: ids, softTokens: soft),
                                          cache: cache, blockIds: blockIds(for: ids))
        var produced = [Int]()
        for _ in 0 ..< Swift.max(options.maxTokens, 0) {
            let last = decoder.logits(fromHidden: hidden[0..., (hidden.dim(1) - 1)...]).reshaped([-1])
            let next = NFKMLXLanguageNet.sample(last, options: options)
            if stops.contains(next) { break }
            produced.append(next)
            if !onToken(next) { break }
            hidden = decoder.hiddenStates(
                fromEmbeddings: decoder.embed(MLXArray([Int32(next)]).reshaped([1, 1])), cache: cache)
        }
        return produced
    }

    /// The text of a token sequence, the markers left out.
    public func decode(_ ids: [Int]) -> String { tokenizer.decode(ids, skipSpecial: true) }

    /// Answers a question about an image (or, with no image, a plain question) through the chat
    /// template, greedily.
    public func answer(image: Any?, question: String, maxTokens: Int = 64) throws -> String {
        var options = NFKMLXGenerationOptions()
        options.maxTokens = maxTokens
        options.temperature = 0
        let ids = chatTokens(messages: [["role": "user", "content": question]], withImage: image != nil)
        return decode(try generate(tokens: ids, image: image, options: options))
    }
}
