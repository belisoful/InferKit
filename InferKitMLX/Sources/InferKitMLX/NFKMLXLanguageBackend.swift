//
//  NFKMLXLanguageBackend.swift
//  InferKitMLX
//
//  Text generation over `NFKMLXLanguageNet`, and the InferKit backend around it.
//

import Foundation
import InferKit
import MLX
import MLXNN

/// How a generation run samples.
public struct NFKMLXGenerationOptions: Sendable {
    /// The most tokens to produce, not counting the prompt.
    public var maxTokens: Int = 256
    /// 0 samples greedily; above 0 divides the logits before sampling.
    public var temperature: Float = 0
    /// Nucleus sampling: the smallest set of tokens whose probability sums past this is sampled from.
    /// Ignored at temperature 0.
    public var topP: Float = 1
    /// A seed makes a sampled run repeatable.
    public var seed: UInt64?
    /// Generation stops when one of these is produced.
    public var stopTokens: Set<Int> = []
    /// The most cached positions to retain, or `nil` to retain every one.
    ///
    /// @discussion A cache that keeps everything grows with the conversation, and past a certain
    /// length that is what ends the run. A window bounds it by dropping the oldest positions.
    ///
    /// This changes what the model reads, so it is off by default. It costs nothing while the
    /// conversation is shorter than the window, and past that the model stops seeing its beginning —
    /// which for a model whose attention is not natively windowed is an approximation, not a
    /// configuration. Size it from what the machine can hold; ``NFKMLXGPU/recommendedWorkingSetSize``
    /// is the budget to divide.
    public var contextWindow: Int?

    public init() {}
}

/// Holds the network for capture in a `@Sendable` body.
private final class NFKLMHolder: @unchecked Sendable {
    let net: NFKMLXLanguageNet
    let tokenizer: NFKTokenizer?
    init(_ net: NFKMLXLanguageNet, _ tokenizer: NFKTokenizer?) {
        self.net = net
        self.tokenizer = tokenizer
    }
}

public extension NFKMLXLanguageNet {

    /// Generates continuation tokens for `prompt`, calling `onToken` with each as it is produced.
    ///
    /// @discussion The prompt runs as one forward pass that fills the cache, then each step feeds back
    /// a single token. Returning false from `onToken` stops the run, which is how a job's cancellation
    /// reaches generation.
    func generate(prompt: [Int], options: NFKMLXGenerationOptions = NFKMLXGenerationOptions(),
                  onToken: ((Int) -> Bool)? = nil) -> [Int] {
        guard !prompt.isEmpty else { return [] }
        if let seed = options.seed { MLXRandom.seed(seed) }

        let cache = NFKMLXKeyValueCache(layerCount: configuration.layerCount,
                                        window: options.contextWindow)
        var logits = self(MLXArray(prompt.map { Int32($0) }).reshaped([1, prompt.count]), cache: cache)
        var produced = [Int]()

        for _ in 0 ..< options.maxTokens {
            let next = NFKMLXLanguageNet.sample(logits[0, -1], options: options)
            if options.stopTokens.contains(next) { break }
            produced.append(next)
            if let onToken, !onToken(next) { break }
            logits = self(MLXArray([Int32(next)]).reshaped([1, 1]), cache: cache)
        }
        return produced
    }

    /// Picks the next token from a `[vocabulary]` row of logits.
    static func sample(_ logits: MLXArray, options: NFKMLXGenerationOptions) -> Int {
        guard options.temperature > 0 else {
            return logits.argMax().item(Int.self)
        }
        var probabilities = softmax(logits / options.temperature, axis: -1)
        if options.topP < 1 {
            probabilities = nucleus(probabilities, topP: options.topP)
        }
        // `categorical` samples from logits, so the filtered distribution goes back through a log.
        return MLXRandom.categorical(log(probabilities)).item(Int.self)
    }

    /// Zeroes every token outside the smallest set whose probability passes `topP`, then renormalizes.
    private static func nucleus(_ probabilities: MLXArray, topP: Float) -> MLXArray {
        let order = argSort(probabilities, axis: -1)
        let sorted = take(probabilities, order, axis: -1)
        let cumulative = cumsum(sorted, axis: -1)
        // Keeping where the cumulative sum from the LOW end is under (1 - topP) discards the tail,
        // which is the same set the reference keeps from the high end.
        let keep = cumulative .> (1 - topP)
        let filtered = MLX.where(keep, sorted, MLXArray(Float(0)))
        // Undo the sort so the probabilities line up with their token ids again.
        let restored = zeros(like: probabilities)
        let scattered = take(filtered, argSort(order, axis: -1), axis: -1)
        return (restored + scattered) / scattered.sum()
    }
}

/// On-device text generation as an InferKit backend.
///
/// Reads `NFKInputPrompt` or `NFKInputMessages` and returns `NFKOutputText`. `NFKParameterTemperature`,
/// `NFKParameterTopP`, `NFKParameterMaxTokens`, and `NFKParameterSeed` override the defaults.
/// Generation is many forward passes; run it off the render thread, or submit a job, which reports
/// each token through `partialResult`.
public final class NFKMLXLanguageBackend: NSObject, NFKInferenceBackend {

    private let holder: NFKLMHolder
    private let identifier: String
    private let defaults: NFKMLXGenerationOptions

    init(net: NFKMLXLanguageNet, tokenizer: NFKTokenizer?, identifier: String,
         options: NFKMLXGenerationOptions = NFKMLXGenerationOptions()) {
        self.holder = NFKLMHolder(net, tokenizer)
        self.identifier = identifier
        self.defaults = options
        super.init()
    }

    public var isReady: Bool { true }
    public var backendIdentifier: String { identifier }

    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        guard let tokenizer = holder.tokenizer else {
            throw NFKMLXError.unsupportedConfiguration(
                "text generation needs a tokenizer; build the backend with one")
        }
        guard let text = Self.prompt(from: request) else { throw NFKMLXError.unsupportedInput }

        var options = defaults
        if let value = request.parameter(forKey: NFKParameterTemperature) as? NSNumber {
            options.temperature = value.floatValue
        }
        if let value = request.parameter(forKey: NFKParameterTopP) as? NSNumber {
            options.topP = value.floatValue
        }
        if let value = request.parameter(forKey: NFKParameterMaxTokens) as? NSNumber {
            options.maxTokens = value.intValue
        }
        if let value = request.parameter(forKey: NFKParameterSeed) as? NSNumber {
            options.seed = value.uint64Value
        }

        let tokens = tokenizer.encode(text).map(\.intValue)
        let produced = holder.net.generate(prompt: tokens, options: options)
        return NFKInferenceResult(outputs: [NFKOutputText: tokenizer.decode(produced.map {
            NSNumber(value: $0)
        })])
    }

    /// The prompt text, from either input key. A message list is flattened in order, which is the
    /// plain-text form a base model expects; a chat-tuned model wants its own template applied first.
    static func prompt(from request: NFKInferenceRequest) -> String? {
        if let prompt = request.prompt { return prompt }
        guard let messages = request.messages else { return nil }
        return messages.compactMap { $0["content"] as? String }.joined(separator: "\n")
    }
}

/// Building, loading, and registering the language model.
@objc(NFKMLXLanguage)
public final class NFKMLXLanguage: NSObject {

    /// The registry name the model builds under.
    @objc public static let modelName = "qwen3"

    static func makeNet(_ configuration: NFKMLXLanguageConfiguration = .qwen3_0_6B) -> NFKMLXLanguageNet {
        NFKMLXLanguageNet(configuration)
    }

    /// Reads a released `config.json` into a configuration.
    ///
    /// @discussion Only the dense decoder is read. A config naming a hybrid or mixture-of-experts
    /// architecture is rejected rather than approximated: `Qwen3_5ForConditionalGeneration` interleaves
    /// linear-attention layers with full attention, which this network does not implement, and loading
    /// its weights into a dense stack would produce fluent-looking nonsense.
    public static func configuration(fromHuggingFace url: URL) throws -> NFKMLXLanguageConfiguration {
        let data = try Data(contentsOf: url)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw NFKMLXError.unsupportedConfiguration("\(url.lastPathComponent) is not a JSON object")
        }
        if let architectures = json["architectures"] as? [String],
           let name = architectures.first,
           !(name.hasSuffix("ForCausalLM")) {
            throw NFKMLXError.unsupportedConfiguration(
                "\(name) is not a dense causal language model; this network implements the dense "
                + "decoder only")
        }
        // Newer transformers writes `layer_types` even for a homogeneous dense stack (the MiniMax
        // Music 3 language model lists 36 × "full_attention"), so only a MIXED stack is rejected.
        let layerTypes = (json["layer_types"] as? [String]) ?? []
        if json["num_experts"] != nil || layerTypes.contains(where: { $0 != "full_attention" }) {
            throw NFKMLXError.unsupportedConfiguration(
                "the config describes a mixture-of-experts or hybrid-attention model, which this "
                + "network does not implement")
        }

        func integer(_ key: String, _ fallback: Int) -> Int { (json[key] as? NSNumber)?.intValue ?? fallback }
        func real(_ key: String, _ fallback: Float) -> Float { (json[key] as? NSNumber)?.floatValue ?? fallback }

        let hidden = integer("hidden_size", 1024)
        let heads = integer("num_attention_heads", 16)
        var configuration = NFKMLXLanguageConfiguration(
            hiddenSize: hidden,
            layerCount: integer("num_hidden_layers", 28),
            headCount: heads,
            keyValueHeadCount: integer("num_key_value_heads", heads),
            headDimensions: integer("head_dim", hidden / max(heads, 1)),
            intermediateSize: integer("intermediate_size", 3072),
            vocabularySize: integer("vocab_size", 151_936),
            // Transformers 5.x nests the rotary base under `rope_parameters`.
            ropeTheta: ((json["rope_parameters"] as? [String: Any])?["rope_theta"] as? NSNumber)?
                .floatValue ?? real("rope_theta", 1_000_000),
            rmsEpsilon: real("rms_norm_eps", 1e-6),
            tiesWordEmbeddings: (json["tie_word_embeddings"] as? NSNumber)?.boolValue ?? false,
            attentionBias: (json["attention_bias"] as? NSNumber)?.boolValue ?? false)
        // Qwen3 normalizes queries and keys per head; Qwen2 and Llama do not. The model type is what
        // says so — the config carries no flag for it.
        let modelType = (json["model_type"] as? String) ?? ""
        configuration.normalizesQueryAndKey = modelType.hasPrefix("qwen3")
        // A release that extended its window says so here. An unimplemented kind throws rather than
        // loading under the wrong rotary, which would run and be wrong.
        configuration.ropeScaling = try NFKMLXRoPEScaling.read(
            json["rope_scaling"],
            maximumPositions: integer("max_position_embeddings", 32_768))
        return configuration
    }

    /// Loads a released checkpoint. The module's keys are the checkpoint's, so nothing is remapped.
    ///
    /// Every weight here is at most two-dimensional, so none of the convolution transposes the vision
    /// models apply have any part in this path.
    static func loadWeights(into net: NFKMLXLanguageNet, from url: URL,
                            precision: NFKMLXWeightPrecision = .float32) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        // A quantized checkpoint reshapes the module to match and loads at its stored dtypes: the
        // packed weights are uint32 whatever the request, and the scales keep the precision the
        // quantization was computed at.
        NFKMLXQuantization.matchStructure(of: checkpoint, on: net)
        let keepStored = precision == .checkpoint || checkpoint.quantization != nil
        let tied = net.lmHead == nil
        let mapped = checkpoint.arrays.compactMap { key, value -> (String, MLXArray)? in
            // A tied release still ships `lm_head.weight`: in Qwen3-0.6B it is byte-identical to
            // `model.embed_tokens.weight`, so the module keeps one copy and the file's duplicate is
            // dropped rather than loaded into a projection the tied model does not have.
            if tied && key.hasPrefix("lm_head.") { return nil }
            let keeps = keepStored || (value.dtype != .float16 && value.dtype != .bfloat16)
            return (key, keeps ? value : value.asType(.float32))
        }
        try NFKMLXWeights.apply(mapped, to: net)
    }

    /// Builds a text-generation backend from local weights and a tokenizer.
    ///
    /// A nil `weightsURL` builds random weights, which run and produce nonsense — useful for proving
    /// the pipeline without a download. Without a tokenizer the backend cannot encode a prompt and
    /// reports so rather than guessing.
    public static func backend(weightsURL: URL?, tokenizer: NFKTokenizer?,
                               configuration: NFKMLXLanguageConfiguration = .qwen3_0_6B,
                               options: NFKMLXGenerationOptions = NFKMLXGenerationOptions())
        throws -> any NFKInferenceBackend {
        let net = makeNet(configuration)
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        return NFKMLXLanguageBackend(net: net, tokenizer: tokenizer, identifier: modelName,
                                     options: options)
    }

    /// Loads a release from its directory, following the shard index when there is one.
    ///
    /// @discussion A release past about a gigabyte is split into `model-0000N-of-0000M.safetensors`
    /// with a `model.safetensors.index.json` naming which shard holds each tensor. Every size above
    /// 0.6B ships that way, so a loader that only reads `model.safetensors` covers the smallest model
    /// and nothing else.
    static func loadWeights(into net: NFKMLXLanguageNet, fromDirectory directory: URL,
                            precision: NFKMLXWeightPrecision = .float32) throws {
        // A single-file release routes through the checkpoint reader, which is where a quantized
        // save's metadata lives; only a sharded release takes the merge path (a quantized module
        // saves as one file, so a quantized sharded release does not arise).
        let files = try NFKMLXReleaseWeights.files(inDirectory: directory)
        if files.count == 1 {
            try loadWeights(into: net, from: files[0], precision: precision)
            return
        }
        // A tied release still ships `lm_head.weight`, byte-identical to the embedding, and a tied
        // module has no projection to put it in.
        let tied = net.lmHead == nil
        let merged = try NFKMLXReleaseWeights.arrays(inDirectory: directory, precision: precision) {
            tied && $0.hasPrefix("lm_head.") ? nil : $0
        }
        try NFKMLXWeights.apply(merged, to: net)
    }

    /// Builds from a downloaded release directory holding the weights, `config.json`, and the
    /// tokenizer files.
    public static func backend(directoryURL: URL,
                               options: NFKMLXGenerationOptions = NFKMLXGenerationOptions())
        throws -> any NFKInferenceBackend {
        let configuration = try self.configuration(
            fromHuggingFace: directoryURL.appendingPathComponent("config.json"))
        // Qwen ships a byte-level BPE vocabulary trained on its OWN pre-tokenization splits — a
        // letter run absorbing one leading punctuation character, digits split singly — so the
        // pattern is named: the GPT-2 default encodes the same text to different, valid-looking ids.
        let tokenizer = try? NFKTokenizer(forManifest: ["tokenizer": ["type": "bpe-bytelevel",
                                                                      "pretokenizer": "qwen2"]],
                                          directory: directoryURL)
        let net = makeNet(configuration)
        try loadWeights(into: net, fromDirectory: directoryURL)
        return NFKMLXLanguageBackend(net: net, tokenizer: tokenizer, identifier: modelName,
                                     options: options)
    }
}
