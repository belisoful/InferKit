//
//  NFKMLXVoxtral.swift
//  InferKitMLX
//
//  Voxtral-Mini 3B (`VoxtralForConditionalGeneration`, Mistral, Apache-2.0): a speech language model.
//  A Whisper large-v3 audio encoder turns log-mel features into acoustic embeddings, a two-layer
//  projector groups every four frames and maps them to the decoder width, and a Llama decoder generates
//  text with the projected audio embeddings scattered into the prompt at the audio-token positions.
//
//  Built almost entirely from parts already at parity: the audio encoder reuses `NFKWhisperEncoder`
//  (the released `audio_tower` is the Whisper encoder, remapped from its transformers names to the
//  openai names the Whisper port loads), and the decoder reuses `NFKMLXGraniteTextNet` with Granite's
//  four scalar multipliers set to identity — the dense Granite decoder is a superset of a Llama decoder.
//  The projector is the only new module.
//
//  Reference: HF `transformers` `VoxtralForConditionalGeneration`.
//

import Foundation
import InferKit
import MLX
import MLXNN

/// The geometry of a Voxtral speech language model.
public struct NFKMLXVoxtralConfiguration: Sendable {
    // The Whisper audio encoder.
    public var audioMels: Int
    public var audioState: Int
    public var audioHeads: Int
    public var audioLayers: Int
    /// The projector's input width: the encoder state times the number of frames it groups.
    public var projectorInputSize: Int
    public var audioTokenId: Int

    // The Llama text decoder, expressed as a Granite decoder with identity multipliers.
    public var text: NFKMLXGraniteTextConfiguration

    public init(audioMels: Int = 128, audioState: Int = 1280, audioHeads: Int = 20, audioLayers: Int = 32,
                projectorInputSize: Int = 5120, audioTokenId: Int = 24,
                text: NFKMLXGraniteTextConfiguration) {
        self.audioMels = audioMels
        self.audioState = audioState
        self.audioHeads = audioHeads
        self.audioLayers = audioLayers
        self.projectorInputSize = projectorInputSize
        self.audioTokenId = audioTokenId
        self.text = text
    }

    /// The Whisper geometry the reused encoder reads.
    var whisperConfiguration: NFKMLXWhisperConfiguration {
        var c = NFKMLXWhisperConfiguration()
        c.nMels = audioMels
        c.nAudioState = audioState
        c.nAudioHead = audioHeads
        c.nAudioLayer = audioLayers
        return c
    }

    /// The released `mistralai/Voxtral-Mini-3B-2507` geometry: the Whisper large-v3 encoder over a
    /// Ministral 3B decoder (head_dim 128, so the query width exceeds the hidden size).
    public static let mini3B = NFKMLXVoxtralConfiguration(
        text: NFKMLXGraniteTextConfiguration(
            hiddenSize: 3072, layerCount: 30, headCount: 32, keyValueHeadCount: 8, headDimensions: 128,
            intermediateSize: 8192, vocabularySize: 131_072, ropeTheta: 100_000_000, rmsEpsilon: 1e-5,
            embeddingMultiplier: 1, residualMultiplier: 1, attentionMultiplier: 1.0 / Float(128).squareRoot(),
            logitsScaling: 1, tiesWordEmbeddings: false))

    /// Reads a Voxtral geometry from a Hugging Face `config.json` dictionary.
    public static func configuration(fromHuggingFace json: [String: Any]) throws
        -> NFKMLXVoxtralConfiguration {
        guard (json["model_type"] as? String) == "voxtral" else {
            throw NFKMLXError.unsupportedConfiguration(
                "expected model_type voxtral, found \(json["model_type"] as? String ?? "nil")")
        }
        func intIn(_ d: [String: Any], _ key: String, _ fallback: Int) -> Int { (d[key] as? Int) ?? fallback }
        func floatIn(_ d: [String: Any], _ key: String, _ fallback: Float) -> Float {
            if let v = d[key] as? Double { return Float(v) }
            if let v = d[key] as? Int { return Float(v) }
            return fallback
        }
        let audio = json["audio_config"] as? [String: Any] ?? [:]
        let textJSON = json["text_config"] as? [String: Any] ?? [:]
        let heads = intIn(textJSON, "num_attention_heads", 32)
        let headDim = (textJSON["head_dim"] as? Int) ?? (intIn(textJSON, "hidden_size", 3072) / heads)
        let text = NFKMLXGraniteTextConfiguration(
            hiddenSize: intIn(textJSON, "hidden_size", 3072), layerCount: intIn(textJSON, "num_hidden_layers", 30),
            headCount: heads, keyValueHeadCount: intIn(textJSON, "num_key_value_heads", 8),
            headDimensions: headDim, intermediateSize: intIn(textJSON, "intermediate_size", 8192),
            vocabularySize: intIn(textJSON, "vocab_size", 131_072),
            ropeTheta: floatIn(textJSON, "rope_theta", 100_000_000), rmsEpsilon: floatIn(textJSON, "rms_norm_eps", 1e-5),
            embeddingMultiplier: 1, residualMultiplier: 1,
            attentionMultiplier: 1.0 / Float(headDim).squareRoot(), logitsScaling: 1,
            tiesWordEmbeddings: (textJSON["tie_word_embeddings"] as? Bool) ?? false)
        let audioState = intIn(audio, "hidden_size", 1280)
        return NFKMLXVoxtralConfiguration(
            audioMels: intIn(audio, "num_mel_bins", 128), audioState: audioState,
            audioHeads: intIn(audio, "num_attention_heads", 20), audioLayers: intIn(audio, "num_hidden_layers", 32),
            projectorInputSize: intIn(audio, "intermediate_size", audioState * 4),
            audioTokenId: intIn(json, "audio_token_id", 24), text: text)
    }
}

/// The two-layer multimodal projector: `linear_1 → gelu → linear_2`, mapping grouped acoustic frames to
/// the decoder width.
final class NFKVoxtralProjector: Module {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(inputSize: Int, hiddenSize: Int) {
        _linear1.wrappedValue = Linear(inputSize, hiddenSize, bias: false)
        _linear2.wrappedValue = Linear(hiddenSize, hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { linear2(NFKReferenceRounding.wide(linear1(x)) { gelu($0) }) }
}

/// Voxtral end to end: the Whisper encoder and the projector turn log-mel features into audio
/// embeddings, which scatter into the Llama decoder's prompt at the audio-token positions.
public final class NFKMLXVoxtralNet: Module {
    public let config: NFKMLXVoxtralConfiguration

    @ModuleInfo(key: "audio_tower") var audioTower: NFKWhisperEncoder
    @ModuleInfo(key: "multi_modal_projector") var projector: NFKVoxtralProjector
    @ModuleInfo(key: "language_model") var languageModel: NFKMLXGraniteTextNet

    public init(_ config: NFKMLXVoxtralConfiguration) {
        self.config = config
        _audioTower.wrappedValue = NFKWhisperEncoder(config.whisperConfiguration)
        _projector.wrappedValue = NFKVoxtralProjector(inputSize: config.projectorInputSize,
                                                      hiddenSize: config.text.hiddenSize)
        _languageModel.wrappedValue = NFKMLXGraniteTextNet(config.text)
        super.init()
    }

    /// The projected audio embeddings for a batch of log-mel features `[B, frames, mels]`: the encoder
    /// output is grouped in fours (`reshape(-1, projectorInputSize)`), then projected.
    public func audioEmbeddings(_ mel: MLXArray) -> MLXArray {
        let encoded = audioTower(mel)                                        // [B, T, audioState]
        let grouped = encoded.reshaped([-1, config.projectorInputSize])      // [B·T/4, 4·audioState]
        return projector(grouped)                                           // [B·T/4, textHidden]
    }

    public func callAsFunction(_ tokens: MLXArray, mel: MLXArray) -> MLXArray {
        logits(tokens: tokens, audioEmbeddings: audioEmbeddings(mel))
    }

    /// Fuses audio embeddings into the token embeddings at the audio-token positions, then decodes.
    public func logits(tokens: MLXArray, audioEmbeddings audio: MLXArray) -> MLXArray {
        let safeTokens = MLX.where(tokens .== MLXArray(Int32(config.audioTokenId)), MLXArray(Int32(0)), tokens)
        var embeddings = languageModel.tokenEmbeddings(safeTokens)
        let ids = tokens.asArray(Int32.self)
        let hidden = embeddings.dim(2)
        let flatAudio = audio.reshaped([-1, hidden])
        var audioIndex = 0
        var rows = [MLXArray]()
        rows.reserveCapacity(ids.count)
        for (position, id) in ids.enumerated() {
            if Int(id) == config.audioTokenId {
                rows.append(flatAudio[audioIndex].reshaped([1, 1, hidden]))
                audioIndex += 1
            } else {
                rows.append(embeddings[0..., position, 0...].reshaped([1, 1, hidden]))
            }
        }
        embeddings = concatenated(rows, axis: 1)
        return languageModel.logits(fromHidden: languageModel.hiddenStates(fromEmbeddings: embeddings))
    }
}

/// Builders and the released-weight loader for Voxtral-Mini.
@objc(NFKMLXVoxtral)
public final class NFKMLXVoxtral: NSObject {
    public static func makeNet(_ config: NFKMLXVoxtralConfiguration) -> NFKMLXVoxtralNet {
        NFKMLXVoxtralNet(config)
    }

    public static func net(fromDirectory directory: URL) throws -> NFKMLXVoxtralNet {
        let url = directory.appendingPathComponent("config.json")
        let json = try JSONSerialization.jsonObject(with: try Data(contentsOf: url)) as? [String: Any] ?? [:]
        return NFKMLXVoxtralNet(try NFKMLXVoxtralConfiguration.configuration(fromHuggingFace: json))
    }

    /// Remaps a released Voxtral key to this module's own names. The audio encoder follows the openai
    /// Whisper naming the `NFKWhisperEncoder` loads, so the released transformers `audio_tower` names are
    /// rewritten; the learned `embed_positions` is dropped (the encoder computes its sinusoids); the
    /// decoder and projector keep their names. Returns nil to skip a tensor.
    static func remap(_ key: String) -> String? {
        if key.hasPrefix("audio_tower.") {
            if key.contains("embed_positions") { return nil }               // sinusoids are computed
            var k = key
            k = k.replacingOccurrences(of: ".self_attn.q_proj", with: ".attn.query")
            k = k.replacingOccurrences(of: ".self_attn.k_proj", with: ".attn.key")
            k = k.replacingOccurrences(of: ".self_attn.v_proj", with: ".attn.value")
            k = k.replacingOccurrences(of: ".self_attn.out_proj", with: ".attn.out")
            k = k.replacingOccurrences(of: ".self_attn_layer_norm", with: ".attn_ln")
            k = k.replacingOccurrences(of: ".final_layer_norm", with: ".mlp_ln")
            k = k.replacingOccurrences(of: ".fc1", with: ".mlp.0")
            k = k.replacingOccurrences(of: ".fc2", with: ".mlp.2")
            k = k.replacingOccurrences(of: ".layers.", with: ".blocks.")
            // The encoder's trailing layer norm: `audio_tower.layer_norm` → `audio_tower.ln_post`.
            if k == "audio_tower.layer_norm.weight" { k = "audio_tower.ln_post.weight" }
            if k == "audio_tower.layer_norm.bias" { k = "audio_tower.ln_post.bias" }
            return k
        }
        return key
    }

    /// Loads a released Voxtral-Mini decoder from its directory. The audio encoder's Conv1d weights are
    /// transposed to channels-last; a tied release ships no `lm_head`.
    public static func loadWeights(into net: NFKMLXVoxtralNet, fromDirectory directory: URL,
                                   precision: NFKMLXWeightPrecision = .checkpoint) throws {
        let tied = net.languageModel.lmHead == nil
        let read = try NFKMLXReleaseWeights.arrays(
            inDirectory: directory, precision: precision == .float32 ? .checkpoint : precision) { key in
            if tied && key.hasPrefix("language_model.lm_head.") { return nil }
            return remap(key)
        }
        let merged = read.map { name, value -> (String, MLXArray) in
            var array = value.ndim == 3 && name.hasSuffix(".weight") ? value.transposed(0, 2, 1) : value
            if precision == .float32 && widensAtFloat32(name, array) {
                array = array.asType(.float32)
                eval(array)
            }
            return (name, array)
        }
        try NFKMLXWeights.apply(merged, to: net)
    }

    /// Whether a tensor widens to float32 in a `.float32` load. The encoder and the projector widen
    /// whole. The decoder widens its embedding table and its norms and keeps each projection matrix at
    /// the release's bfloat16: the float32 embeddings carry every layer in float32, and each product
    /// promotes its bfloat16 operand exactly, so the arithmetic equals a decoder widened throughout at
    /// about half its memory.
    static func widensAtFloat32(_ name: String, _ value: MLXArray) -> Bool {
        !name.hasPrefix("language_model.") || value.ndim == 1 || name == "language_model.model.embed_tokens.weight"
    }
}
