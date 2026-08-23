//
//  NFKMLXSDTextEncoder.swift
//  InferKitMLX
//
//  The CLIP text tower the Stable Diffusion releases cross-attend to.
//
//  Every released text-to-image model here conditions on one or two of these towers. They differ in
//  scalars a configuration carries — width, depth, head count, the MLP's activation, and which hidden
//  state the pipeline reads — not in structure, so the blocks are `NFKMLXCLIP`'s own.
//
//  The released weights are a `transformers` CLIPTextModel, whose keys differ from the OpenAI CLIP
//  layout `NFKMLXCLIP` loads: the attention stores separate `q_proj` / `k_proj` / `v_proj` where the
//  module keeps the reference's fused projection, so the remap concatenates three tensors into one.
//

import Foundation
import InferKit
import MLX
import MLXNN

/// Which hidden state a pipeline conditions on.
public enum NFKSDTextOutput: Sendable {
    /// Every layer, then the final layer normalization. Stable Diffusion 1.x and 2.x — the 2.x
    /// releases drop the tower's last layer in their own configuration rather than skipping it here.
    case lastHiddenState
    /// Every layer but the last, and no final layer normalization. SDXL reads both of its towers this
    /// way.
    case penultimateHiddenState
}

/// The geometry of a released text encoder.
public struct NFKMLXSDTextEncoderConfiguration: Sendable {
    public var width: Int = 768
    public var layers: Int = 12
    public var heads: Int = 12
    public var intermediate: Int = 3072
    public var vocabularySize: Int = 49408
    public var contextLength: Int = 77
    public var activation: NFKCLIPActivation = .quickGELU
    public var output: NFKSDTextOutput = .lastHiddenState
    /// The width a pooled embedding projects to, for the towers that carry a projection. SDXL's second
    /// tower supplies the pooled embedding its UNet conditions on; the others have none.
    public var projectionDimensions: Int?

    public init() {}

    /// Stable Diffusion 1.x: CLIP ViT-L/14's text tower.
    public static let stableDiffusion15 = NFKMLXSDTextEncoderConfiguration()

    /// Stable Diffusion 2.x: OpenCLIP ViT-H/14's text tower, with the release's own layer count — the
    /// checkpoint carries 23 of the tower's 24 layers, which is how the penultimate hidden state
    /// reaches the UNet through the last-hidden-state path.
    public static let stableDiffusion21: NFKMLXSDTextEncoderConfiguration = {
        var c = NFKMLXSDTextEncoderConfiguration()
        c.width = 1024
        c.layers = 23
        c.heads = 16
        c.intermediate = 4096
        c.activation = .gelu
        return c
    }()

    /// SDXL's first tower: the same weights as Stable Diffusion 1.x, read one layer earlier.
    public static let sdxlPrimary: NFKMLXSDTextEncoderConfiguration = {
        var c = NFKMLXSDTextEncoderConfiguration()
        c.output = .penultimateHiddenState
        return c
    }()

    /// SDXL's second tower: OpenCLIP bigG, which also supplies the pooled embedding.
    public static let sdxlSecondary: NFKMLXSDTextEncoderConfiguration = {
        var c = NFKMLXSDTextEncoderConfiguration()
        c.width = 1280
        c.layers = 32
        c.heads = 20
        c.intermediate = 5120
        c.activation = .gelu
        c.output = .penultimateHiddenState
        c.projectionDimensions = 1280
        return c
    }()

    /// A small configuration for offline structure and round-trip tests.
    public static let tiny: NFKMLXSDTextEncoderConfiguration = {
        var c = NFKMLXSDTextEncoderConfiguration()
        c.width = 32
        c.layers = 2
        c.heads = 2
        c.intermediate = 64
        c.vocabularySize = 64
        c.contextLength = 12
        return c
    }()
}

/// What a text encoder produces: the sequence the UNet cross-attends to, and — where the tower
/// carries a projection — the pooled embedding SDXL folds into its timestep.
public struct NFKSDTextEmbedding {
    /// `[1, contextLength, width]`.
    public let hidden: MLXArray
    /// `[1, projectionDimensions]`, or nil for a tower without a projection.
    public let pooled: MLXArray?
}

/// The CLIP text tower, in the module layout `NFKMLXCLIP` already uses.
public final class NFKMLXSDTextEncoderNet: Module {
    @ModuleInfo(key: "token_embedding") var tokenEmbedding: Embedding
    @ModuleInfo(key: "positional_embedding") var positionalEmbedding: MLXArray
    @ModuleInfo(key: "transformer") var transformer: NFKCLIPTransformer
    @ModuleInfo(key: "ln_final") var lnFinal: LayerNorm
    @ModuleInfo(key: "text_projection") var textProjection: MLXArray?

    public let configuration: NFKMLXSDTextEncoderConfiguration

    public init(configuration: NFKMLXSDTextEncoderConfiguration) {
        self.configuration = configuration
        _tokenEmbedding.wrappedValue = Embedding(embeddingCount: configuration.vocabularySize,
                                                 dimensions: configuration.width)
        _positionalEmbedding.wrappedValue = NFKCLIPInit.parameter([configuration.contextLength,
                                                                   configuration.width])
        _transformer.wrappedValue = NFKCLIPTransformer(width: configuration.width,
                                                       layers: configuration.layers,
                                                       heads: configuration.heads,
                                                       intermediate: configuration.intermediate,
                                                       activation: configuration.activation)
        _lnFinal.wrappedValue = LayerNorm(dimensions: configuration.width)
        _textProjection.wrappedValue = configuration.projectionDimensions.map {
            NFKCLIPInit.parameter([configuration.width, $0])
        }
    }

    /// Encodes a padded token sequence. The pooled embedding is read at the end-of-text token, which
    /// the reference locates as the sequence's highest id — the padding repeats that same id, so the
    /// FIRST occurrence is the one the reference takes.
    public func encode(_ tokenIds: [Int]) -> NFKSDTextEmbedding {
        let length = tokenIds.count
        let tokens = MLXArray(tokenIds.map { Int32($0) }).reshaped([1, length])
        let mask = NFKCLIPInit.causalMask(length)
        var h = tokenEmbedding(tokens) + positionalEmbedding[0 ..< length]

        var penultimate = h
        for block in transformer.resblocks {
            penultimate = h
            h = block(h, mask: mask)
        }

        let hidden: MLXArray
        switch configuration.output {
        case .lastHiddenState: hidden = lnFinal(h)
        case .penultimateHiddenState: hidden = penultimate
        }
        guard let textProjection else {
            return NFKSDTextEmbedding(hidden: hidden, pooled: nil)
        }
        // The pooled embedding always runs the whole stack through the final normalization, even when
        // the sequence the UNet reads stops a layer short.
        let end = tokenIds.firstIndex(of: tokenIds.max() ?? 0) ?? (length - 1)
        let pooled = lnFinal(h)[0, end].reshaped([1, configuration.width]).matmul(textProjection)
        return NFKSDTextEmbedding(hidden: hidden, pooled: pooled)
    }
}

/// The prompt tokenizer a Stable Diffusion release ships, and the model input it produces.
///
/// The vocabulary is a load-time artifact rather than part of the network, so a release carries it as
/// files beside the weights: `vocab.json`, `merges.txt`, and a `tokenizer_config.json` naming the
/// markers. This reads those and drives the core's `NFKTokenizer` CLIP variant, so text-to-image
/// takes a prompt rather than token ids.
public struct NFKMLXSDPromptTokenizer: @unchecked Sendable {

    private let tokenizer: NFKTokenizer
    /// The start-of-text marker's id.
    public let startTokenId: Int
    /// The end-of-text marker's id.
    public let endTokenId: Int
    /// The id a short prompt is padded with. Most releases pad with the end marker; SDXL's second
    /// tokenizer names a different one, so it is read from the release rather than assumed.
    public let paddingTokenId: Int

    /// Reads a release's `tokenizer/` directory.
    public init(directoryURL: URL) throws {
        let vocabURL = directoryURL.appendingPathComponent("vocab.json")
        guard let data = try? Data(contentsOf: vocabURL),
              let vocabulary = try JSONSerialization.jsonObject(with: data) as? [String: Int] else {
            throw NFKMLXError.weightsMismatch("the tokenizer directory carries no readable vocab.json")
        }
        let start = "<|startoftext|>", end = "<|endoftext|>"
        guard let startId = vocabulary[start], let endId = vocabulary[end] else {
            throw NFKMLXError.weightsMismatch("the vocabulary names no start and end markers")
        }
        startTokenId = startId
        endTokenId = endId
        paddingTokenId = Self.paddingToken(in: directoryURL).flatMap { vocabulary[$0] } ?? endId

        let manifest: [String: Any] = [
            "bosTokenId": startId,
            "eosTokenId": endId,
            "tokenizer": [
                "type": "clip",
                "vocab": "vocab.json",
                "merges": "merges.txt",
                "specialTokens": [start: startId, end: endId],
            ],
        ]
        tokenizer = try NFKTokenizer(forManifest: manifest, directory: directoryURL)
    }

    /// The ids for the prompt's text alone, without the markers a model input carries.
    public func encode(_ prompt: String) -> [Int] {
        tokenizer.encode(prompt).map(\.intValue)
    }

    /// The model input: the start marker, the prompt, the end marker, then padding to the tower's
    /// context length. A prompt longer than that is truncated with the end marker kept last, which is
    /// what the reference's `truncation=True` does.
    public func tokens(for prompt: String, contextLength: Int) -> [Int] {
        var ids = [startTokenId] + encode(prompt)
        if ids.count >= contextLength {
            ids = Array(ids.prefix(contextLength - 1))
        }
        ids.append(endTokenId)
        return ids + Array(repeating: paddingTokenId, count: contextLength - ids.count)
    }

    /// The padding token a release names.
    ///
    /// Two files can name it, and `special_tokens_map.json` wins where both do — which is how Stable
    /// Diffusion 2.x pads with `!` while its `tokenizer_config.json` still says the end marker. Reading
    /// only the config pads a short prompt with 75 end markers where the release trained on 75 `!`,
    /// and the model then reads a different sentence.
    private static func paddingToken(in directoryURL: URL) -> String? {
        for name in ["special_tokens_map.json", "tokenizer_config.json"] {
            guard let data = try? Data(contentsOf: directoryURL.appendingPathComponent(name)),
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                continue
            }
            if let token = json["pad_token"] as? String {
                return token
            }
            if let token = (json["pad_token"] as? [String: Any])?["content"] as? String {
                return token
            }
        }
        return nil
    }
}

/// Building a text encoder and loading a released checkpoint into it.
public enum NFKMLXSDTextEncoder {

    /// Builds the tower and loads `weightsURL` when one is given.
    public static func net(configuration: NFKMLXSDTextEncoderConfiguration,
                           weightsURL: URL? = nil,
                           precision: NFKMLXWeightPrecision = .float32) throws -> NFKMLXSDTextEncoderNet {
        let net = NFKMLXSDTextEncoderNet(configuration: configuration)
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL, precision: precision)
        }
        net.train(false)
        return net
    }

    /// Loads a released `text_encoder` checkpoint, fusing the reference's separate query, key, and
    /// value projections into the one the module keeps.
    public static func loadWeights(into net: NFKMLXSDTextEncoderNet, from url: URL,
                                   precision: NFKMLXWeightPrecision = .float32) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        try NFKMLXWeights.apply(NFKMLXWeights.converted(remap(checkpoint.arrays), to: precision), to: net)
    }

    /// Translates the `transformers` CLIPTextModel layout onto the module's names.
    ///
    /// A 1:1 key map cannot express the attention: the reference stores `q_proj`, `k_proj`, and
    /// `v_proj` separately where the module keeps the fused `in_proj_weight` the OpenAI layout uses,
    /// so the three are concatenated in that order. Keys the module has no place for — the
    /// `position_ids` buffer, a vision tower travelling in the same file — are dropped.
    static func remap(_ arrays: [String: MLXArray]) -> [(String, MLXArray)] {
        var mapped = [(String, MLXArray)]()
        var attention = [String: [String: MLXArray]]()      // "resblocks.N" -> part -> tensor

        for (key, value) in arrays {
            guard key.hasPrefix("text_model.") || key == "text_projection.weight" else { continue }
            if key == "text_projection.weight" {
                // The reference projects with a `Linear`, so its weight is the transpose of the
                // matrix this module multiplies by.
                mapped.append(("text_projection", value.transposed(1, 0)))
                continue
            }
            let name = String(key.dropFirst("text_model.".count))
            switch name {
            case "embeddings.token_embedding.weight":
                mapped.append(("token_embedding.weight", value))
            case "embeddings.position_embedding.weight":
                mapped.append(("positional_embedding", value))
            case "final_layer_norm.weight", "final_layer_norm.bias":
                mapped.append(("ln_final." + (name.hasSuffix("weight") ? "weight" : "bias"), value))
            default:
                guard name.hasPrefix("encoder.layers.") else { continue }
                let parts = name.split(separator: ".").map(String.init)
                guard parts.count >= 5, let index = Int(parts[2]) else { continue }
                let block = "transformer.resblocks.\(index)"
                let tail = parts.dropFirst(3).joined(separator: ".")
                if let renamed = blockKey(tail) {
                    mapped.append((block + "." + renamed, value))
                } else if tail.hasPrefix("self_attn.") {
                    attention[block, default: [:]][tail] = value
                }
            }
        }

        for (block, parts) in attention {
            for suffix in ["weight", "bias"] {
                let fused = ["q_proj", "k_proj", "v_proj"].compactMap { parts["self_attn.\($0).\(suffix)"] }
                guard fused.count == 3 else { continue }
                let name = suffix == "weight" ? "in_proj_weight" : "in_proj_bias"
                mapped.append((block + ".attn." + name, concatenated(fused, axis: 0)))
            }
        }
        return mapped
    }

    /// The names that translate one for one inside a layer.
    private static func blockKey(_ tail: String) -> String? {
        switch tail {
        case "layer_norm1.weight": return "ln_1.weight"
        case "layer_norm1.bias": return "ln_1.bias"
        case "layer_norm2.weight": return "ln_2.weight"
        case "layer_norm2.bias": return "ln_2.bias"
        case "mlp.fc1.weight": return "mlp.c_fc.weight"
        case "mlp.fc1.bias": return "mlp.c_fc.bias"
        case "mlp.fc2.weight": return "mlp.c_proj.weight"
        case "mlp.fc2.bias": return "mlp.c_proj.bias"
        case "self_attn.out_proj.weight": return "attn.out_proj.weight"
        case "self_attn.out_proj.bias": return "attn.out_proj.bias"
        default: return nil
        }
    }
}
