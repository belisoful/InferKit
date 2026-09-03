//
//  NFKMLXGGUFLanguage.swift
//  InferKitMLX
//
//  Wiring the native GGUF reader into the language-model loader, so a GGUF release generates text end
//  to end: its metadata becomes an `NFKMLXLanguageConfiguration`, its tensors are remapped from the
//  llama.cpp naming onto the module's HF-style keys and loaded (dequantized by the reader), and its
//  embedded tokenizer is rebuilt for the core byte-level BPE reader. GGUF is the dominant distribution
//  format for quantized language models, and until this it could be read but not run.
//

import Foundation
import InferKit
import MLX

extension NFKMLXLanguage {

    /// Builds a text-generation backend from a GGUF file, reading its geometry, weights, and tokenizer.
    ///
    /// @discussion The GGUF carries everything a release directory would: the architecture and
    /// hyperparameters in its metadata, the (possibly quantized) weights in its tensors, and the
    /// tokenizer's vocabulary and merges in its metadata arrays. Only the dense decoder families this
    /// network implements are accepted (`llama`, `qwen2`, `qwen3`); another architecture throws rather
    /// than loading under the wrong structure. Run inference off the render thread.
    public static func backend(ggufURL: URL,
                               options: NFKMLXGenerationOptions = NFKMLXGenerationOptions())
        throws -> any NFKInferenceBackend {
        let (net, tokenizer) = try loadedGGUF(at: ggufURL)
        return NFKMLXLanguageBackend(net: net, tokenizer: tokenizer, identifier: modelName,
                                     options: options)
    }

    /// The Objective-C entry: builds a backend from a GGUF file. Generation options are set per request
    /// through ``NFKMLXGenerationParameterKey``.
    @objc(backendWithGGUFURL:error:)
    public static func backend(ggufURL: URL) throws -> any NFKInferenceBackend {
        try backend(ggufURL: ggufURL, options: NFKMLXGenerationOptions())
    }

    /// The network and tokenizer a GGUF file describes.
    static func loadedGGUF(at url: URL) throws -> (NFKMLXLanguageNet, NFKTokenizer?) {
        let gguf = try NFKMLXGGUF.gguf(contentsOf: url)
        let configuration = try self.configuration(fromGGUF: gguf)
        let net = makeNet(configuration)
        try loadWeights(into: net, fromGGUF: gguf)
        let tokenizer = ggufTokenizer(gguf)
        return (net, tokenizer)
    }

    /// Reads a GGUF's metadata into a decoder configuration.
    ///
    /// @discussion The hyperparameters live under the architecture's own prefix (`llama.block_count`,
    /// `qwen3.embedding_length`). Two structural facts are read from the tensors rather than the
    /// metadata, which carries no flag for either: a model is tied when it ships no `output.weight`
    /// (it reuses the token embedding as the logit head), and it normalizes queries and keys when it
    /// ships `blk.0.attn_q_norm.weight` (Qwen3 does, Qwen2 and Llama do not).
    public static func configuration(fromGGUF gguf: NFKMLXGGUF) throws -> NFKMLXLanguageConfiguration {
        guard let architecture = gguf.metadataString(forKey: "general.architecture") else {
            throw NFKMLXError.unsupportedConfiguration("the GGUF names no general.architecture")
        }
        guard ["llama", "qwen2", "qwen3"].contains(architecture) else {
            throw NFKMLXError.unsupportedConfiguration(
                "this reads a dense llama/qwen decoder, not a \(architecture) GGUF")
        }
        let names = Set(gguf.tensorNames)
        let hidden = gguf.metadataInteger(forKey: "\(architecture).embedding_length", defaultValue: 0)
        let heads = gguf.metadataInteger(forKey: "\(architecture).attention.head_count", defaultValue: 0)
        guard hidden > 0, heads > 0 else {
            throw NFKMLXError.unsupportedConfiguration("the GGUF omits its embedding length or head count")
        }
        let vocabulary = gguf.metadataInteger(forKey: "\(architecture).vocab_size",
            defaultValue: gguf.info(forTensor: "token_embd.weight")?.shape.first ?? 0)

        var configuration = NFKMLXLanguageConfiguration(
            hiddenSize: hidden,
            layerCount: gguf.metadataInteger(forKey: "\(architecture).block_count", defaultValue: 0),
            headCount: heads,
            keyValueHeadCount: gguf.metadataInteger(forKey: "\(architecture).attention.head_count_kv",
                                                    defaultValue: heads),
            headDimensions: gguf.metadataInteger(forKey: "\(architecture).attention.key_length",
                                                 defaultValue: hidden / max(heads, 1)),
            intermediateSize: gguf.metadataInteger(forKey: "\(architecture).feed_forward_length",
                                                   defaultValue: 0),
            vocabularySize: vocabulary,
            ropeTheta: gguf.metadataFloat(forKey: "\(architecture).rope.freq_base", defaultValue: 10_000),
            rmsEpsilon: gguf.metadataFloat(forKey: "\(architecture).attention.layer_norm_rms_epsilon",
                                           defaultValue: 1e-5),
            // A tied model ships no separate output projection; it reuses the token embedding.
            tiesWordEmbeddings: !names.contains("output.weight"),
            // Qwen2 ships attention biases; Llama and Qwen3 do not.
            attentionBias: names.contains("blk.0.attn_q.bias"))
        // Qwen3 ships per-head query/key norms; Qwen2 and Llama do not.
        configuration.normalizesQueryAndKey = names.contains("blk.0.attn_q_norm.weight")
        return configuration
    }

    /// Loads a GGUF's tensors into the network, remapping the llama.cpp names onto the module's keys.
    ///
    /// @discussion llama.cpp PERMUTES the query and key projections during conversion so its
    /// interleaved rotary reads adjacent channels, where this decoder (and the HF checkpoint the GGUF
    /// came from) rotate split halves. The permutation is undone here, per head, or the rotary reads
    /// the wrong channel pairs and the model runs mostly-right and subtly wrong.
    static func loadWeights(into net: NFKMLXLanguageNet, fromGGUF gguf: NFKMLXGGUF) throws {
        let arrays = try gguf.arrays()
        let heads = net.configuration.headCount
        let keyValueHeads = net.configuration.keyValueHeadCount
        var mapped = [(String, MLXArray)]()
        for (name, value) in arrays {
            // A tied model still may not carry `output.weight`; a mapped `lm_head.weight` is dropped
            // when the network ties, exactly as the release loader drops the duplicate.
            guard let key = huggingFaceKey(forGGUF: name) else { continue }
            if key == "lm_head.weight" && net.configuration.tiesWordEmbeddings { continue }
            if key.hasSuffix("self_attn.q_proj.weight") || key.hasSuffix("self_attn.q_proj.bias") {
                mapped.append((key, unpermuteRotary(value, heads: heads)))
            } else if key.hasSuffix("self_attn.k_proj.weight") || key.hasSuffix("self_attn.k_proj.bias") {
                mapped.append((key, unpermuteRotary(value, heads: keyValueHeads)))
            } else {
                mapped.append((key, value))
            }
        }
        try NFKMLXWeights.apply(mapped, to: net)
    }

    /// Undoes llama.cpp's per-head rotary permutation of a query or key projection: the leading axis is
    /// `heads · headDim`, reshaped to `[heads, headDim/2, 2, …]`, the split-half axes exchanged, and
    /// reshaped back. It applies to a weight `[heads·headDim, in]` and a bias `[heads·headDim]` alike.
    static func unpermuteRotary(_ weights: MLXArray, heads: Int) -> MLXArray {
        let halfDimension = weights.shape[0] / heads / 2
        let trailing = Array(weights.shape.dropFirst())
        let reshaped = weights.reshaped([heads, halfDimension, 2] + trailing)
        return reshaped.swappedAxes(1, 2).reshaped(weights.shape)
    }

    /// The module key a GGUF tensor name corresponds to, or nil for one the decoder does not read.
    ///
    /// @discussion GGUF names a llama-family model with `token_embd`, `output_norm`, `output`, and a
    /// per-block `blk.N.*` set; the module keeps the checkpoint's HF names (`model.layers.N.self_attn`).
    static func huggingFaceKey(forGGUF name: String) -> String? {
        switch name {
        case "token_embd.weight": return "model.embed_tokens.weight"
        case "output_norm.weight": return "model.norm.weight"
        case "output.weight": return "lm_head.weight"
        default: break
        }
        guard name.hasPrefix("blk.") else { return nil }
        let components = name.split(separator: ".", maxSplits: 2)
        guard components.count == 3, let layer = Int(components[1]) else { return nil }
        let prefix = "model.layers.\(layer)."
        switch String(components[2]) {
        case "attn_norm.weight": return prefix + "input_layernorm.weight"
        case "ffn_norm.weight": return prefix + "post_attention_layernorm.weight"
        case "attn_q.weight": return prefix + "self_attn.q_proj.weight"
        case "attn_k.weight": return prefix + "self_attn.k_proj.weight"
        case "attn_v.weight": return prefix + "self_attn.v_proj.weight"
        case "attn_output.weight": return prefix + "self_attn.o_proj.weight"
        case "attn_q.bias": return prefix + "self_attn.q_proj.bias"
        case "attn_k.bias": return prefix + "self_attn.k_proj.bias"
        case "attn_v.bias": return prefix + "self_attn.v_proj.bias"
        case "attn_q_norm.weight": return prefix + "self_attn.q_norm.weight"
        case "attn_k_norm.weight": return prefix + "self_attn.k_norm.weight"
        case "ffn_gate.weight": return prefix + "mlp.gate_proj.weight"
        case "ffn_up.weight": return prefix + "mlp.up_proj.weight"
        case "ffn_down.weight": return prefix + "mlp.down_proj.weight"
        default: return nil
        }
    }

    /// Rebuilds the tokenizer a GGUF embeds, for the core byte-level BPE reader.
    ///
    /// @discussion GGUF stores a GPT-2-family tokenizer as its already byte-encoded token strings
    /// (`Ġcapital`), its merge rules, and a per-token type that marks the special tokens. These are
    /// written to the `vocab.json`/`merges.txt` the core reader takes, the same route the ModernBERT
    /// tokenizer uses. The pre-tokenization pattern is the release's own (`qwen2` for a Qwen vocabulary,
    /// `gpt2` otherwise), because a merge cannot cross a pre-token boundary.
    static func ggufTokenizer(_ gguf: NFKMLXGGUF) -> NFKTokenizer? {
        guard let tokens = gguf.metadataStringArray(forKey: "tokenizer.ggml.tokens"),
              let merges = gguf.metadataStringArray(forKey: "tokenizer.ggml.merges") else { return nil }
        var vocabulary = [String: Int]()
        vocabulary.reserveCapacity(tokens.count)
        for (index, token) in tokens.enumerated() { vocabulary[token] = index }

        // Token types: CONTROL (3) and USER_DEFINED (4) are the special tokens, which must resolve to a
        // single id rather than being split into ordinary byte-level pieces.
        var specials = [String: Int]()
        if let types = gguf.metadataIntegerArray(forKey: "tokenizer.ggml.token_type") {
            for (index, type) in types.enumerated() where (type == 3 || type == 4) && index < tokens.count {
                specials[tokens[index]] = index
            }
        }

        let scratch = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        guard (try? FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)) != nil,
              let vocabularyData = try? JSONSerialization.data(withJSONObject: vocabulary),
              (try? vocabularyData.write(to: scratch.appendingPathComponent("vocab.json"))) != nil else {
            return nil
        }
        var mergesText = "#version: 0.2\n"
        for merge in merges { mergesText += merge + "\n" }
        guard (try? mergesText.write(to: scratch.appendingPathComponent("merges.txt"),
                                     atomically: true, encoding: .utf8)) != nil else { return nil }

        let pre = gguf.metadataString(forKey: "tokenizer.ggml.pre") ?? ""
        let pattern = pre.contains("qwen") ? "qwen2" : "gpt2"
        var manifest: [String: Any] = ["tokenizer": ["type": "bpe-bytelevel", "pretokenizer": pattern,
                                                     "specialTokens": specials]]
        let eos = gguf.metadataInteger(forKey: "tokenizer.ggml.eos_token_id", defaultValue: -1)
        if eos >= 0 { manifest["eosTokenId"] = eos }
        return try? NFKTokenizer(forManifest: manifest, directory: scratch)
    }
}
