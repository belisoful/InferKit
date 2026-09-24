//
//  NFKMLXInternLM2.swift
//  InferKitMLX
//
//  InternLM2 in the dense decoder's terms. InternLM2 is a Llama-layout decoder under other names: a
//  fused `wqkv`, `wo`, a SwiGLU feed-forward named `w1` (gate), `w3` (up), and `w2` (down),
//  `attention_norm` / `ffn_norm`, `tok_embeddings`, and `output`. The fused projection stores, for each
//  key-value head, its group of query heads followed by the one key head and the one value head, so the
//  split regroups rather than slices in three.
//

import Foundation
import InferKit
import MLX

enum NFKMLXInternLM2 {

    /// Whether a decoder `config.json` block is InternLM2's.
    static func describes(_ json: [String: Any]) -> Bool {
        (json["model_type"] as? String) == "internlm2"
    }

    /// The decoder block as the dense configuration reader takes it. InternLM2's `rope_scaling` is
    /// dynamic NTK, which rescales the rotary only once a sequence passes `max_position_embeddings`
    /// (32,768); below that the rotary is the base one, so the block is dropped for a model whose
    /// sequences stay under it.
    static func denseConfigurationJSON(_ json: [String: Any], sequencesStayBelowWindow: Bool) -> [String: Any] {
        var adjusted = sequencesStayBelowWindow ? NFKMLXRoPEScaling.droppingDynamic(json) : json
        adjusted["attention_bias"] = json["bias"] ?? false
        return adjusted
    }

    /// The dense decoder's name for an InternLM2 tensor name under `prefix`, or nil for the fused
    /// `wqkv` (which ``splitQueryKeyValue(_:configuration:)`` maps) and for a name that is not InternLM2's.
    static func denseKey(_ key: String, prefix: String) -> String? {
        guard key.hasPrefix(prefix) else { return nil }
        let name = String(key.dropFirst(prefix.count))
        let renames: [(String, String)] = [
            ("model.tok_embeddings.", "model.embed_tokens."), ("output.", "lm_head."),
            (".attention.wo.", ".self_attn.o_proj."),
            (".feed_forward.w1.", ".mlp.gate_proj."), (".feed_forward.w3.", ".mlp.up_proj."),
            (".feed_forward.w2.", ".mlp.down_proj."),
            (".attention_norm.", ".input_layernorm."), (".ffn_norm.", ".post_attention_layernorm."),
        ]
        if name.contains(".attention.wqkv.") { return nil }
        for (from, to) in renames where name.hasPrefix(from) || name.contains(from) {
            return prefix + name.replacingOccurrences(of: from, with: to)
        }
        return key
    }

    /// The query, key, and value weights of a fused `wqkv` `[(kvHeads · (groups + 2) · headDim), hidden]`.
    static func splitQueryKeyValue(_ fused: MLXArray, configuration: NFKMLXLanguageConfiguration)
        -> (query: MLXArray, key: MLXArray, value: MLXArray) {
        let kvHeads = configuration.keyValueHeadCount
        let groups = configuration.headCount / kvHeads
        let headDim = configuration.headDimensions
        let hidden = fused.dim(-1)
        let grouped = fused.reshaped([kvHeads, groups + 2, headDim, hidden])
        return (grouped[0..., 0 ..< groups].reshaped([kvHeads * groups * headDim, hidden]),
                grouped[0..., groups].reshaped([kvHeads * headDim, hidden]),
                grouped[0..., groups + 1].reshaped([kvHeads * headDim, hidden]))
    }

    /// An InternLM2 checkpoint's tensors under `prefix` in the dense decoder's names, the fused
    /// projections split; tensors outside `prefix` pass through.
    static func denseWeights(_ pairs: [(String, MLXArray)], prefix: String,
                             configuration: NFKMLXLanguageConfiguration) -> [(String, MLXArray)] {
        pairs.flatMap { key, value -> [(String, MLXArray)] in
            guard key.hasPrefix(prefix) else { return [(key, value)] }
            if key.hasSuffix(".attention.wqkv.weight") {
                let layer = key.replacingOccurrences(of: ".attention.wqkv.weight", with: ".self_attn.")
                let split = splitQueryKeyValue(value, configuration: configuration)
                return [(layer + "q_proj.weight", split.query), (layer + "k_proj.weight", split.key),
                        (layer + "v_proj.weight", split.value)]
            }
            return [(denseKey(key, prefix: prefix) ?? key, value)]
        }
    }
}

// MARK: - Tokenizer

/// InternLM2's own `InternLM2Tokenizer` (the slow one; its releases ship no fast tokenizer): a
/// SentencePiece model with the added tokens of `added_tokens.json` and `tokenizer_config.json`. With
/// ``Decoding/llamaFast`` it is also the Llama-2 / Vicuna tokenizer LLaVA-1.5 carries, whose fast form
/// encodes the same way.
///
/// @discussion Encoding prepends `<s>`, splits the text at every special token (longest match; the
/// releases strip no whitespace around them), and encodes each remaining run with the SentencePiece
/// model on its own, as `PreTrainedTokenizer.tokenize` does. Decoding follows `_decode` with its defaults:
/// added tokens past the model's pieces stand alone and are joined with spaces, and the rest decode
/// through `convert_tokens_to_string`, which spaces a special token after ordinary text. The Llama fast
/// decoder instead writes every token in place (`▁` as a space, byte pieces fused) and strips one
/// leading space.
///
/// Introduced in InferKit 0.4.0.
@objc(NFKMLXInternLM2Tokenizer)
public final class NFKMLXInternLM2Tokenizer: NFKTokenizer {
    /// How ids decode to text.
    @objc(NFKMLXInternLM2TokenizerDecoding)
    public enum Decoding: Int, Sendable {
        /// InternLM2's slow `_decode`.
        case internLM2Slow
        /// The Llama fast tokenizer's decoder chain.
        case llamaFast
    }

    public let segmenter: NFKMLXSentencePieceSegmenter
    public let decoding: Decoding
    private let specialIds: [String: Int]
    private let specialsLongestFirst: [String]
    private let specialPieces: [Int: String]
    private let allSpecial: Set<String>
    private let startId: Int
    private let endId: Int
    private let addsStart: Bool

    /// Reads `tokenizer.model`, `added_tokens.json`, and `tokenizer_config.json` from `directory`.
    @objc(initWithDirectoryURL:decoding:error:)
    public init(directory: URL, decoding: Decoding = .internLM2Slow) throws {
        self.decoding = decoding
        let segmenter = try NFKMLXSentencePieceSegmenter(contentsOf: directory.appendingPathComponent("tokenizer.model"))
        self.segmenter = segmenter
        var ids = [String: Int]()
        if let data = try? Data(contentsOf: directory.appendingPathComponent("added_tokens.json")),
           let added = try? JSONSerialization.jsonObject(with: data) as? [String: Int] {
            ids.merge(added) { first, _ in first }
        }
        var additional = [String]()
        var config = [String: Any]()
        if let data = try? Data(contentsOf: directory.appendingPathComponent("tokenizer_config.json")),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            config = json
            additional = json["additional_special_tokens"] as? [String] ?? []
            for (key, entry) in json["added_tokens_decoder"] as? [String: [String: Any]] ?? [:] {
                if let content = entry["content"] as? String, let id = Int(key) { ids[content] = ids[content] ?? id }
            }
        }
        func named(_ key: String, _ fallback: String) -> String { config[key] as? String ?? fallback }
        let core = [named("bos_token", "<s>"), named("eos_token", "</s>"), named("unk_token", "<unk>"),
                    named("pad_token", "</s>")]
        for token in core + additional where ids[token] == nil {
            if let id = segmenter.id(of: token) { ids[token] = id }
        }
        specialIds = ids
        specialsLongestFirst = ids.keys.sorted { $0.count > $1.count }
        specialPieces = Dictionary(ids.map { ($1, $0) }, uniquingKeysWith: { first, _ in first })
        allSpecial = Set(core + additional)
        startId = ids[named("bos_token", "<s>")] ?? 1
        endId = ids[named("eos_token", "</s>")] ?? 2
        addsStart = (config["add_bos_token"] as? Bool) ?? true
        super.init()
    }

    public override var eosTokenId: Int { endId }
    public override var bosTokenId: Int { startId }

    /// The id of a special or added token, or of a SentencePiece piece.
    public func id(ofToken token: String) -> Int? { specialIds[token] ?? segmenter.id(of: token) }

    public override func encode(_ text: String) -> [NSNumber] {
        encodeIds(text, addingStart: addsStart).map { NSNumber(value: $0) }
    }

    /// The ids of `text`, with `<s>` first when `addingStart`.
    public func encodeIds(_ text: String, addingStart: Bool) -> [Int] {
        var ids = addingStart ? [startId] : []
        var pending = ""
        var rest = Substring(text)
        func flush() {
            if !pending.isEmpty { ids += segmenter.encode(pending, dummyPrefix: nil) }
            pending = ""
        }
        while !rest.isEmpty {
            if let special = specialsLongestFirst.first(where: { rest.hasPrefix($0) }) {
                flush()
                ids.append(specialIds[special]!)
                rest = rest.dropFirst(special.count)
            } else {
                pending.append(rest.removeFirst())
            }
        }
        flush()
        return ids
    }

    public override func decode(_ tokenIds: [NSNumber]) -> String {
        decodeIds(tokenIds.map(\.intValue))
    }

    /// The text of `ids` with special tokens kept, as the release's tokenizer decodes it.
    public func decodeIds(_ ids: [Int]) -> String {
        guard decoding == .internLM2Slow else { return llamaFastText(ids) }
        let pieceCount = segmenter.pieceCount
        var texts = [String]()
        var run = [Int]()
        for id in ids {
            if id >= pieceCount, let token = specialPieces[id] {
                if !run.isEmpty { texts.append(tokensToString(run)) }
                run.removeAll()
                texts.append(token)
            } else {
                run.append(id)
            }
        }
        if !run.isEmpty { texts.append(tokensToString(run)) }
        return texts.joined(separator: " ")
    }

    /// The Llama fast decoder: special and added tokens written as themselves, runs of pieces through
    /// the SentencePiece model (`▁` to a space, byte pieces fused), and one leading space stripped.
    private func llamaFastText(_ ids: [Int]) -> String {
        var output = ""
        var pieces = [Int]()
        for id in ids {
            if let token = specialPieces[id] {
                output += pieceText(pieces) + token
                pieces.removeAll()
            } else {
                pieces.append(id)
            }
        }
        output += pieceText(pieces)
        return output.hasPrefix(" ") ? String(output.dropFirst()) : output
    }

    /// Pieces spelled out with their spaces kept: the segmenter drops a leading dummy-prefix space, which
    /// the fast decoder does not, so a run that opens with `▁` gets it back.
    private func pieceText(_ ids: [Int]) -> String {
        guard let first = ids.first else { return "" }
        let text = segmenter.decode(ids)
        let opensWithSpace = segmenter.piece(at: first)?.unicodeScalars.first == NFKMLXSentencePieceSegmenter.space
        return opensWithSpace && !text.hasPrefix(" ") ? " " + text : text
    }

    /// `convert_tokens_to_string`: special pieces written literally and spaced after ordinary text, the
    /// rest decoded by the SentencePiece model, and the result's spacing cleaned up (a space before
    /// punctuation and contractions removed). The reference then prepends a space and drops the first
    /// character, which cancel: its `_maybe_add_prefix_space` tests a token string against a set of ids,
    /// so it always prepends.
    private func tokensToString(_ ids: [Int]) -> String {
        var output = ""
        var pieces = [Int]()
        var previousSpecial = false
        for id in ids {
            let token = specialPieces[id] ?? segmenter.piece(at: id) ?? ""
            if allSpecial.contains(token) {
                if !previousSpecial { output += " " }
                output += segmenter.decode(pieces) + token
                previousSpecial = true
                pieces.removeAll()
            } else {
                pieces.append(id)
                previousSpecial = false
            }
        }
        output += segmenter.decode(pieces)
        // transformers' `clean_up_tokenization`, which this method calls whatever the tokenizer's
        // `clean_up_tokenization_spaces` says.
        for (from, to) in [(" .", "."), (" ?", "?"), (" !", "!"), (" ,", ","), (" ' ", "'"), (" n't", "n't"),
                           (" 'm", "'m"), (" 's", "'s"), (" 've", "'ve"), (" 're", "'re")] {
            output = output.replacingOccurrences(of: from, with: to)
        }
        return output
    }
}
