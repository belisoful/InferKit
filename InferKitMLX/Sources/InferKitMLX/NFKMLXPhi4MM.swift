//
//  NFKMLXPhi4MM.swift
//  InferKitMLX
//
//  Phi-4-multimodal (`Phi4MMForCausalLM`, Microsoft, MIT): a Phi-4-mini decoder shared across three
//  input modes, with a SigLIP image tower and a Conformer audio tower feeding embeddings into the
//  decoder at placeholder positions, and a per-modality LoRA folded onto the decoder's projections.
//
//  The decoder reuses ``NFKMLXLanguageNet``. Phi-4-mini differs from the Qwen/Llama stacks that class
//  already serves in three ways, all handled here: it rotates only the first `partial_rotary_factor`
//  of each head (``NFKMLXLanguageConfiguration/rotaryDimensions``), it scales the rotary with LongRoPE
//  (``NFKMLXRoPEScaling/Kind/longrope``), and it stores its query/key/value and gate/up projections
//  fused, wrapped as PEFT LoRA modules. The released checkpoint bakes both adapters (`vision`, `speech`)
//  into the base shards, so a modality is served by folding its adapter into the base projection and
//  splitting the result into the separate projections the decoder holds.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import MLX
import MLXNN
import MLXRandom

/// Which input mode the decoder is serving, which selects the LoRA folded onto its projections.
///
/// @discussion The reference selects the adapter from the input: an image (with or without audio) uses
/// the `vision` adapter, audio alone uses the `speech` adapter, and text alone uses none. The audio
/// projector also has a `speech` and a `vision` head, chosen the same way.
public enum NFKMLXPhi4MMModality: Sendable {
    case language
    case vision
    case speech

    /// The adapter name the release stores for this mode, or nil for the no-adapter language mode.
    var adapterName: String? {
        switch self {
        case .language: return nil
        case .vision: return "vision"
        case .speech: return "speech"
        }
    }
}

/// A projection holding both released LoRA adapters beside its base weight, with one active at a time:
/// `y = Wx + scale·(x·A)·B` for the active adapter, and `Wx` alone in the language mode. The adapters
/// run beside the base rather than folding into it, so switching modes costs nothing and the base is
/// stored once.
final class NFKPhi4MMMixtureLoRALinear: Linear {
    @ParameterInfo(key: "vision_a") var visionA: MLXArray      // [in, rank]
    @ParameterInfo(key: "vision_b") var visionB: MLXArray      // [rank, out]
    @ParameterInfo(key: "speech_a") var speechA: MLXArray
    @ParameterInfo(key: "speech_b") var speechB: MLXArray
    let visionScale: Float
    let speechScale: Float
    var active: NFKMLXPhi4MMModality = .language

    init(base: Linear, vision: (a: MLXArray, b: MLXArray, scale: Float),
         speech: (a: MLXArray, b: MLXArray, scale: Float)) {
        _visionA.wrappedValue = vision.a
        _visionB.wrappedValue = vision.b
        _speechA.wrappedValue = speech.a
        _speechB.wrappedValue = speech.b
        visionScale = vision.scale
        speechScale = speech.scale
        super.init(weight: base.weight, bias: base.bias)
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = super.callAsFunction(x)
        switch active {
        case .language: return y
        case .vision: return y + matmul(matmul(x, visionA), visionB) * visionScale
        case .speech: return y + matmul(matmul(x, speechA), speechB) * speechScale
        }
    }
}

/// Building and loading Phi-4-multimodal from a downloaded release directory.
@objc(NFKMLXPhi4MM)
public final class NFKMLXPhi4MM: NSObject {

    /// The registry name a Phi-4-multimodal backend reports.
    @objc public static let modelName = "phi-4-multimodal"

    static let imageTokenId = 200010            // <|endoftext10|>, the image placeholder
    static let audioTokenId = 200011            // <|endoftext11|>, the audio placeholder
    static let userTokenId = 200021             // <|user|>
    static let endTokenId = 200020              // <|end|>
    static let assistantTokenId = 200019        // <|assistant|>
    static let endOfTextTokenId = 199999        // <|endoftext|>
    /// The release's `generation_config` stops on either the turn end or the end of text.
    static let stopTokens: Set<Int> = [endTokenId, endOfTextTokenId]

    // MARK: Decoder

    /// The Phi-4-mini decoder geometry, read from the release `config.json`: 3072 wide over 32 layers of
    /// 24 query and 8 key/value heads, a 96-channel partial rotary scaled by LongRoPE, tied embeddings,
    /// and the o200k vocabulary. The fused projections are split on load, so the decoder itself is the
    /// ordinary dense stack.
    public static func decoderConfiguration(directoryURL: URL) throws -> NFKMLXLanguageConfiguration {
        let data = try Data(contentsOf: directoryURL.appendingPathComponent("config.json"))
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw NFKMLXError.malformedCheckpoint("Phi-4-multimodal config.json is not an object")
        }
        func integer(_ key: String, _ fallback: Int) -> Int { (json[key] as? NSNumber)?.intValue ?? fallback }
        func real(_ key: String, _ fallback: Float) -> Float { (json[key] as? NSNumber)?.floatValue ?? fallback }

        let heads = integer("num_attention_heads", 24)
        let hidden = integer("hidden_size", 3072)
        let headDim = hidden / heads
        let maxPositions = integer("max_position_embeddings", 131072)
        let partial = real("partial_rotary_factor", 1)

        var configuration = NFKMLXLanguageConfiguration(
            hiddenSize: hidden,
            layerCount: integer("num_hidden_layers", 32),
            headCount: heads,
            keyValueHeadCount: integer("num_key_value_heads", 8),
            headDimensions: headDim,
            intermediateSize: integer("intermediate_size", 8192),
            vocabularySize: integer("vocab_size", 200064),
            ropeTheta: real("rope_theta", 10000),
            rmsEpsilon: real("rms_norm_eps", 1e-5),
            tiesWordEmbeddings: (json["tie_word_embeddings"] as? NSNumber)?.boolValue ?? true,
            normalizesQueryAndKey: false,
            attentionBias: (json["attention_bias"] as? NSNumber)?.boolValue ?? false)
        configuration.rotaryDimensions = partial < 1 ? Int(Float(headDim) * partial) : nil
        configuration.ropeScaling = try NFKMLXRoPEScaling.read(json["rope_scaling"], maximumPositions: maxPositions)
        // Phi carries `original_max_position_embeddings` at the top level of the config, not inside the
        // `rope_scaling` block. LongRoPE's attention scale and its short/long table switch both derive
        // from it, so it must be lifted from there rather than defaulting to the extended window.
        if configuration.ropeScaling?.kind == .longrope {
            configuration.ropeScaling?.originalMaxPositionEmbeddings =
                integer("original_max_position_embeddings", 4096)
            configuration.ropeScaling?.maximumPositionEmbeddings = maxPositions
        }
        return configuration
    }

    /// The LoRA scale a modality folds at, `lora_alpha / r`, read from the release config's
    /// `vision_lora` / `speech_lora` blocks.
    static func loRAScale(directoryURL: URL, adapter: String) throws -> Float {
        let data = try Data(contentsOf: directoryURL.appendingPathComponent("config.json"))
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        guard let block = json?["\(adapter)_lora"] as? [String: Any],
              let rank = (block["r"] as? NSNumber)?.floatValue,
              let alpha = (block["lora_alpha"] as? NSNumber)?.floatValue, rank > 0 else {
            throw NFKMLXError.malformedCheckpoint("Phi-4-multimodal config carries no \(adapter)_lora r/alpha")
        }
        return alpha / rank
    }

    /// The decoder's stored tensors by their release names: the embedding, the norms, each projection's
    /// `base_layer`, and the low-rank pairs of the named adapters. The towers and any adapter not named
    /// are skipped.
    static func decoderTensors(directoryURL: URL, precision: NFKMLXWeightPrecision,
                               adapters: Set<String>) throws -> [String: MLXArray] {
        let raw = try NFKMLXReleaseWeights.arrays(inDirectory: directoryURL, precision: precision) { key -> String? in
            guard isDecoderTensor(key) else { return nil }
            if key.contains(".lora_A.") || key.contains(".lora_B.") {
                return adapters.contains(where: { key.contains(".\($0).") }) ? key : nil
            }
            return key
        }
        var byKey = [String: MLXArray](minimumCapacity: raw.count)
        for (key, value) in raw { byKey[key] = value }
        return byKey
    }

    /// The decoder tensors for one modality, in ``NFKMLXLanguageNet``'s names: the selected adapter is
    /// folded into each base projection, then the fused query/key/value and gate/up projections are
    /// split into the separate ones the decoder holds. The other modality's adapter and the two towers
    /// are skipped, which is what lets the apply stay strict.
    static func decoderArrays(directoryURL: URL,
                              configuration: NFKMLXLanguageConfiguration,
                              modality: NFKMLXPhi4MMModality,
                              precision: NFKMLXWeightPrecision = .float32) throws -> [(String, MLXArray)] {
        let adapter = modality.adapterName
        let byKey = try decoderTensors(directoryURL: directoryURL, precision: precision,
                                       adapters: adapter.map { [$0] } ?? [])
        let scale = try adapter.map { try loRAScale(directoryURL: directoryURL, adapter: $0) } ?? 0
        return try splitArrays(byKey: byKey, configuration: configuration) { base in
            guard let weight = byKey[base + ".base_layer.weight"] else {
                throw NFKMLXError.malformedCheckpoint("Phi-4-multimodal is missing \(base).base_layer.weight")
            }
            guard let adapter else { return weight }
            guard let a = byKey[base + ".lora_A.\(adapter).weight"],
                  let b = byKey[base + ".lora_B.\(adapter).weight"] else {
                throw NFKMLXError.malformedCheckpoint("Phi-4-multimodal is missing the \(adapter) adapter for \(base)")
            }
            return weight + matmul(b, a) * scale
        }
    }

    /// The decoder tensors in ``NFKMLXLanguageNet``'s names, each fused projection produced by `weight`
    /// (a base or an adapter-folded base) and split into the separate projections the decoder holds.
    static func splitArrays(byKey: [String: MLXArray], configuration: NFKMLXLanguageConfiguration,
                            weight fold: (String) throws -> MLXArray) throws -> [(String, MLXArray)] {
        func tensor(_ name: String) throws -> MLXArray {
            guard let value = byKey[name] else {
                throw NFKMLXError.malformedCheckpoint("Phi-4-multimodal is missing \(name)")
            }
            return value
        }
        var out = [(String, MLXArray)]()
        out.append(("model.embed_tokens.weight", try tensor("model.embed_tokens.weight")))
        out.append(("model.norm.weight", try tensor("model.norm.weight")))

        let qDim = configuration.headCount * configuration.headDimensions
        let kvDim = configuration.keyValueHeadCount * configuration.headDimensions
        let intermediate = configuration.intermediateSize
        for layer in 0 ..< configuration.layerCount {
            let base = "model.layers.\(layer)"
            out.append(("\(base).input_layernorm.weight", try tensor("\(base).input_layernorm.weight")))
            out.append(("\(base).post_attention_layernorm.weight", try tensor("\(base).post_attention_layernorm.weight")))

            let qkv = try fold("\(base).self_attn.qkv_proj")
            out.append(("\(base).self_attn.q_proj.weight", qkv[0 ..< qDim]))
            out.append(("\(base).self_attn.k_proj.weight", qkv[qDim ..< (qDim + kvDim)]))
            out.append(("\(base).self_attn.v_proj.weight", qkv[(qDim + kvDim) ..< (qDim + 2 * kvDim)]))
            out.append(("\(base).self_attn.o_proj.weight", try fold("\(base).self_attn.o_proj")))

            let gateUp = try fold("\(base).mlp.gate_up_proj")
            out.append(("\(base).mlp.gate_proj.weight", gateUp[0 ..< intermediate]))
            out.append(("\(base).mlp.up_proj.weight", gateUp[intermediate ..< (2 * intermediate)]))
            out.append(("\(base).mlp.down_proj.weight", try fold("\(base).mlp.down_proj")))
        }
        return out
    }

    /// The Phi-4-mini decoder loaded for one modality: the base projections with that modality's adapter
    /// folded in, the fused projections split. A `.language` load folds nothing.
    public static func decoder(directoryURL: URL,
                               modality: NFKMLXPhi4MMModality = .language,
                               precision: NFKMLXWeightPrecision = .float32) throws -> NFKMLXLanguageNet {
        let configuration = try decoderConfiguration(directoryURL: directoryURL)
        let net = NFKMLXLanguageNet(configuration)
        let arrays = try decoderArrays(directoryURL: directoryURL, configuration: configuration,
                                       modality: modality, precision: precision)
        try NFKMLXWeights.apply(arrays, to: net, verifyShapes: true)
        return net
    }

    /// The Phi-4-mini decoder with BOTH adapters live beside the base projections, one active at a time:
    /// the released mixture of LoRAs. One decoder serves every input mode; ``select(_:in:)`` chooses the
    /// adapter, and `.language` runs the base alone. Each fused projection's adapter splits with it — the
    /// query, key, and value share the query/key/value adapter's down-projection and take their rows of
    /// its up-projection.
    public static func mixtureDecoder(directoryURL: URL,
                                      precision: NFKMLXWeightPrecision = .float32) throws -> NFKMLXLanguageNet {
        let configuration = try decoderConfiguration(directoryURL: directoryURL)
        let net = NFKMLXLanguageNet(configuration)
        let byKey = try decoderTensors(directoryURL: directoryURL, precision: precision, adapters: ["vision", "speech"])
        try NFKMLXWeights.apply(try splitArrays(byKey: byKey, configuration: configuration) { base in
            guard let weight = byKey[base + ".base_layer.weight"] else {
                throw NFKMLXError.malformedCheckpoint("Phi-4-multimodal is missing \(base).base_layer.weight")
            }
            return weight
        }, to: net, verifyShapes: true)

        let scales = [try loRAScale(directoryURL: directoryURL, adapter: "vision"),
                      try loRAScale(directoryURL: directoryURL, adapter: "speech")]
        let qDim = configuration.headCount * configuration.headDimensions
        let kvDim = configuration.keyValueHeadCount * configuration.headDimensions
        let intermediate = configuration.intermediateSize
        for (index, block) in net.model.layers.enumerated() {
            let base = "model.layers.\(index)"
            func mixture(_ layer: Linear, _ projection: String, rows: Range<Int>? = nil) throws -> NFKPhi4MMMixtureLoRALinear {
                var pairs = [(a: MLXArray, b: MLXArray, scale: Float)]()
                for (adapter, scale) in zip(["vision", "speech"], scales) {
                    guard let a = byKey["\(base).\(projection).lora_A.\(adapter).weight"],
                          var b = byKey["\(base).\(projection).lora_B.\(adapter).weight"] else {
                        throw NFKMLXError.malformedCheckpoint("Phi-4-multimodal is missing the \(adapter) adapter for \(base).\(projection)")
                    }
                    if let rows { b = b[rows] }
                    pairs.append((a: a.transposed(), b: b.transposed(), scale: scale))
                }
                return NFKPhi4MMMixtureLoRALinear(base: layer, vision: pairs[0], speech: pairs[1])
            }
            let attention = block.attention
            try attention.update(modules: ModuleChildren.unflattened([
                ("q_proj", try mixture(attention.queryProjection, "self_attn.qkv_proj", rows: 0 ..< qDim)),
                ("k_proj", try mixture(attention.keyProjection, "self_attn.qkv_proj", rows: qDim ..< (qDim + kvDim))),
                ("v_proj", try mixture(attention.valueProjection, "self_attn.qkv_proj", rows: (qDim + kvDim) ..< (qDim + 2 * kvDim))),
                ("o_proj", try mixture(attention.outputProjection, "self_attn.o_proj")),
            ]), verify: .none)
            guard let feedForward = block.feedForward as? NFKLMFeedForward else { continue }
            try feedForward.update(modules: ModuleChildren.unflattened([
                ("gate_proj", try mixture(feedForward.gate, "mlp.gate_up_proj", rows: 0 ..< intermediate)),
                ("up_proj", try mixture(feedForward.up, "mlp.gate_up_proj", rows: intermediate ..< (2 * intermediate))),
                ("down_proj", try mixture(feedForward.down, "mlp.down_proj")),
            ]), verify: .none)
        }
        return net
    }

    /// Makes `modality`'s adapter the active one on every projection of a ``mixtureDecoder(directoryURL:precision:)``.
    public static func select(_ modality: NFKMLXPhi4MMModality, in decoder: NFKMLXLanguageNet) {
        for (_, module) in decoder.leafModules().flattened() {
            (module as? NFKPhi4MMMixtureLoRALinear)?.active = modality
        }
    }

    /// Greedy continuation from a text prompt, cached: a prefill over the token embeddings, then one
    /// token at a time. Stops at an end token or `maxTokens`.
    public static func generateText(decoder: NFKMLXLanguageNet, inputIds: [Int],
                                    maxTokens: Int, endTokens: Set<Int>) -> [Int] {
        let cache = NFKMLXKeyValueCache(layerCount: decoder.configuration.layerCount)
        let sequence = inputIds.count
        var hidden = decoder.hiddenStates(
            fromEmbeddings: decoder.embed(MLXArray(inputIds.map(Int32.init)).reshaped([1, sequence])),
            cache: cache)
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

    // MARK: Where each released tensor goes

    static let imagePrefix = "model.embed_tokens_extend.image_embed."
    static let audioPrefix = "model.embed_tokens_extend.audio_embed."

    /// Whether a released tensor belongs to the decoder: the embedding, the norms, each projection's base
    /// weight, and both adapters' low-rank pairs.
    static func isDecoderTensor(_ key: String) -> Bool {
        key.hasPrefix("model.") && !key.hasPrefix("model.embed_tokens_extend.")
    }

    /// Whether a released tensor is part of the SigLIP tower the image feature never reaches: its final
    /// layer, its post-layer-norm, and its attention-pooling head. The feature is the penultimate hidden
    /// state, so no network loads these.
    static func isUnusedImageTensor(_ key: String, imageLayers: Int = NFKMLXSigLIPConfiguration.phi4mm.layerCount) -> Bool {
        guard key.hasPrefix(imagePrefix) else { return false }
        let name = key.dropFirst(imagePrefix.count)
        return name.hasPrefix("img_processor.encoder.layers.\(imageLayers).")
            || name.hasPrefix("img_processor.post_layernorm.") || name.hasPrefix("img_processor.head.")
    }

    /// A released tensor's name in ``NFKMLXPhi4MMImageNet``, or nil when it is not an image-tower
    /// tensor that network loads.
    static func imageModuleKey(forRelease key: String,
                               imageLayers: Int = NFKMLXSigLIPConfiguration.phi4mm.layerCount) -> String? {
        guard key.hasPrefix(imagePrefix), !isUnusedImageTensor(key, imageLayers: imageLayers) else { return nil }
        return String(key.dropFirst(imagePrefix.count))
    }

    /// A released tensor's name in ``NFKMLXPhi4MMAudioNet``, or nil when it is not a speech-tower tensor.
    static func audioModuleKey(forRelease key: String) -> String? {
        key.hasPrefix(audioPrefix) ? String(key.dropFirst(audioPrefix.count)) : nil
    }

    /// PyTorch stores a convolution channel-first; MLX's `Conv1d`/`Conv2d` hold it channels-last.
    static func channelsLast(_ key: String, _ value: MLXArray) -> MLXArray {
        guard key.hasSuffix(".weight") else { return value }
        if value.ndim == 4 { return value.transposed(0, 2, 3, 1) }
        if value.ndim == 3 { return value.transposed(0, 2, 1) }
        return value
    }

    // MARK: Speech tower

    /// The speech tower's tensors in the module's names, with the convolutions channels-last; the biases
    /// and the normalization statistics pass through.
    static func audioArrays(directoryURL: URL) throws -> [(String, MLXArray)] {
        try NFKMLXReleaseWeights.arrays(inDirectory: directoryURL, remap: audioModuleKey(forRelease:))
            .map { key, value in (key, channelsLast(key, value)) }
    }

    /// The speech tower loaded from a release directory.
    public static func audioNet(directoryURL: URL,
                                configuration: NFKMLXPhi4MMAudioConfiguration = .released) throws -> NFKMLXPhi4MMAudioNet {
        let net = NFKMLXPhi4MMAudioNet(configuration)
        try NFKMLXWeights.apply(audioArrays(directoryURL: directoryURL), to: net, verifyShapes: true)
        return net
    }

    // MARK: Image tower

    /// The image tower's tensors in the module's names, with the patch-embedding convolution
    /// channels-last. The SigLIP parts the penultimate-layer feature never reaches are skipped.
    static func imageArrays(directoryURL: URL,
                            configuration: NFKMLXSigLIPConfiguration = .phi4mm) throws -> [(String, MLXArray)] {
        try NFKMLXReleaseWeights.arrays(inDirectory: directoryURL) {
            imageModuleKey(forRelease: $0, imageLayers: configuration.layerCount)
        }.map { key, value in (key, channelsLast(key, value)) }
    }

    /// The image tower loaded from a release directory.
    public static func imageNet(directoryURL: URL,
                                configuration: NFKMLXSigLIPConfiguration = .phi4mm) throws -> NFKMLXPhi4MMImageNet {
        let net = NFKMLXPhi4MMImageNet(configuration)
        try NFKMLXWeights.apply(imageArrays(directoryURL: directoryURL, configuration: configuration), to: net,
                                verifyShapes: true)
        return net
    }

    // MARK: Fusion

    /// Embeds `inputIds`, scatters `features` (projected image or audio embeddings, in sequence order)
    /// into the positions holding `placeholder`, and runs the decoder to post-norm hidden states.
    static func fusedHidden(decoder: NFKMLXLanguageNet, inputIds: [Int], features: MLXArray,
                            placeholder: Int, cache: NFKMLXKeyValueCache? = nil) -> MLXArray {
        fusedHidden(decoder: decoder, inputIds: inputIds, features: [(placeholder, features)], cache: cache)
    }

    /// Embeds `inputIds` and scatters each placeholder's features into that placeholder's positions, in
    /// sequence order; an image and an audio clip in one prompt fill their own placeholders.
    static func fusedHidden(decoder: NFKMLXLanguageNet, inputIds: [Int],
                            features: [(placeholder: Int, features: MLXArray)],
                            cache: NFKMLXKeyValueCache? = nil) -> MLXArray {
        let sequence = inputIds.count
        let width = decoder.configuration.hiddenSize
        var embeddings = decoder.embed(MLXArray(inputIds.map(Int32.init)).reshaped([1, sequence]))[0]
        for (placeholder, values) in features {
            var featureIndex = [Int32](repeating: 0, count: sequence)
            var isFeature = [Float](repeating: 0, count: sequence)
            var counter: Int32 = 0
            for position in 0 ..< sequence where inputIds[position] == placeholder {
                featureIndex[position] = counter
                counter += 1
                isFeature[position] = 1
            }
            guard counter > 0 else { continue }
            let gathered = values.reshaped([-1, width]).take(MLXArray(featureIndex), axis: 0)
            let mask = MLXArray(isFeature).reshaped([sequence, 1]) .> 0
            embeddings = MLX.where(mask, gathered.asType(embeddings.dtype), embeddings)
        }
        return decoder.hiddenStates(fromEmbeddings: embeddings.reshaped([1, sequence, width]), cache: cache)
    }

    /// Greedy continuation from a fused multimodal prompt: a prefill over the fused embeddings, then one
    /// token at a time from the decoder alone.
    static func generateFused(decoder: NFKMLXLanguageNet, inputIds: [Int], features: MLXArray,
                              placeholder: Int, maxTokens: Int, endTokens: Set<Int>) -> [Int] {
        generateFused(decoder: decoder, inputIds: inputIds, features: [(placeholder, features)],
                      maxTokens: maxTokens, endTokens: endTokens)
    }

    static func generateFused(decoder: NFKMLXLanguageNet, inputIds: [Int],
                              features: [(placeholder: Int, features: MLXArray)],
                              maxTokens: Int, endTokens: Set<Int>) -> [Int] {
        var options = NFKMLXGenerationOptions()
        options.maxTokens = maxTokens
        return generateFused(decoder: decoder, inputIds: inputIds, features: features, options: options,
                             endTokens: endTokens)
    }

    /// A continuation from a fused multimodal prompt, sampled as `options` asks (greedy at temperature
    /// 0). `onToken` sees each id as it is produced; returning false stops the run.
    static func generateFused(decoder: NFKMLXLanguageNet, inputIds: [Int],
                              features: [(placeholder: Int, features: MLXArray)],
                              options: NFKMLXGenerationOptions, endTokens: Set<Int>,
                              onToken: ((Int) -> Bool)? = nil) -> [Int] {
        if let seed = options.seed { MLXRandom.seed(seed) }
        let cache = NFKMLXKeyValueCache(layerCount: decoder.configuration.layerCount)
        var hidden = fusedHidden(decoder: decoder, inputIds: inputIds, features: features, cache: cache)
        var produced = [Int]()
        for _ in 0 ..< options.maxTokens {
            let logits = decoder.logits(fromHidden: hidden[0..., (hidden.dim(1) - 1)...]).reshaped([-1])
            let next = NFKMLXLanguageNet.sample(logits, options: options)
            if endTokens.contains(next) { break }
            produced.append(next)
            if let onToken, !onToken(next) { break }
            hidden = decoder.hiddenStates(
                fromEmbeddings: decoder.embed(MLXArray([Int32(next)]).reshaped([1, 1])), cache: cache)
        }
        return produced
    }
}
