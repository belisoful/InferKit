//
//  NFKMLXMusic3Backend.swift
//  InferKitMLX
//
//  The consumer surface of the MiniMax Music 3 port: the prompt contract (the release's cleaners,
//  special-token template, and byte-level BPE through the core NFKTokenizer), and the backend that
//  chains the five measured stages behind NFKInputPrompt + NFKInputLyrics → NFKOutputAudio.
//
//  The weights are NOT permissively licensed — see Docs/companions.md.
//

import Foundation
import InferKit
import MLX
import MLXNN

// MARK: - Prompt contract

/// The prompt assembly the checkpoint was trained on. Even whitespace-level changes to the
/// assembled prompt change the generated audio, so the cleaners reproduce the Diffusers pipeline's
/// own (`_clean_caption`, `_normalize_lyrics`) and are measured token for token against them.
enum NFKMusic3Prompt {

    static let imStart = "<|im_start|>", imEnd = "<|im_end|>"
    static let captionStart = "<|caption_start|>", captionEnd = "<|caption_end|>"
    static let lyricsStart = "<|lyrics_start|>", lyricsEnd = "<|lyrics_end|>"
    static let audioStart = "<|audio_start|>"

    /// The caption cleaner: `<|tag value|>` forms rewrite to "tag is value", the markdown forms the
    /// model's input contract accepts are stripped (headers, bullets, bold, italics, rules), lines
    /// lose trailing whitespace, and blank runs collapse.
    static func cleanedCaption(_ caption: String) -> String {
        var text = rewritingSpecialTags(caption)
        var lines = [String]()
        for line in splitLines(text) {
            var cleaned = replacing(line, pattern: "^\\s{0,3}#{1,6}\\s+", with: "")
            cleaned = replacing(cleaned, pattern: "^\\s*[*+-]\\s+", with: "")
            cleaned = replacing(cleaned, pattern: "^\\s*\\*\\s+", with: "")
            while cleaned.contains("**") {
                let updated = replacing(cleaned, pattern: "\\*\\*([^*]+)\\*\\*", with: "$1")
                if updated == cleaned { break }
                cleaned = updated
            }
            cleaned = replacing(cleaned, pattern: "(?<!\\*)\\*([^*\\n]+)\\*(?!\\*)", with: "$1")
            while let last = cleaned.last, last.isWhitespace { cleaned.removeLast() }
            lines.append(cleaned)
        }
        text = lines.joined(separator: "\n")
        text = replacing(text, pattern: "^\\s*[-*_]{3,}\\s*$", with: "", options: [.anchorsMatchLines])
        text = text.replacingOccurrences(of: "• ", with: "")
        text = text.replacingOccurrences(of: "    ", with: "")
        return replacing(text, pattern: "\\n{2,}", with: "\n")
    }

    /// The lyrics normalizer: a line that OPENS with structure tags keeps only the tags (text on a
    /// tag line is dropped by the checkpoint's input contract), inline tags split onto their own
    /// lines, ` ^ ` is a line break, tags lowercase, and `[start]` leads.
    static func normalizedLyrics(_ lyrics: String) -> String {
        let leadingTags = try! NSRegularExpression(pattern: "^[ \\t]*((?:\\[[^\\]]+\\][ \\t]*)+)")
        var output = [String]()
        for line in lyrics.components(separatedBy: "\n") {
            let range = NSRange(line.startIndex..., in: line)
            if let match = leadingTags.firstMatch(in: line, range: range),
               let tags = Range(match.range(at: 1), in: line) {
                output.append(String(line[tags]).trimmingCharacters(in: .whitespaces))
            } else {
                output.append(line)
            }
        }
        var text = output.joined(separator: "\n")
        text = text.replacingOccurrences(of: "] ", with: "]\n")
        text = text.replacingOccurrences(of: " [", with: "\n[")
        text = text.replacingOccurrences(of: " ^ ", with: "\n")
        text = lowercasingTags(text)
        return "[start]\n" + text
    }

    /// The assembled special-token prompt.
    static func assembled(caption: String, lyrics: String) -> String {
        imStart + captionStart + cleanedCaption(caption) + captionEnd
            + lyricsStart + normalizedLyrics(lyrics) + lyricsEnd + imEnd + audioStart
    }

    /// The release's byte-level BPE through the core tokenizer: `vocab.json` + `merges.txt` with the
    /// special tokens from `added_tokens.json`, which the core splits on anywhere in the text — the
    /// reference tokenizer's own behavior.
    static func tokenizer(directory: URL) throws -> NFKTokenizer {
        let addedData = try Data(contentsOf: directory.appendingPathComponent("added_tokens.json"))
        guard let added = try JSONSerialization.jsonObject(with: addedData) as? [String: Int] else {
            throw NFKMLXError.unsupportedConfiguration("added_tokens.json is not a name → id object")
        }
        // The Qwen2 pre-tokenization, not the GPT-2 default: the vocabulary was trained on its own
        // splits ("-pop" is one pretoken, digits split singly), and encoding under the wrong pattern
        // produces different, valid-looking ids for the same text.
        let manifest: [String: Any] = ["tokenizer": [
            "type": "bpe-bytelevel",
            "pretokenizer": "qwen2",
            "specialTokens": added.mapValues { NSNumber(value: $0) },
        ]]
        return try NFKTokenizer(forManifest: manifest, directory: directory)
    }

    /// The `[2, L]` conditional/unconditional id pair: the assembled prompt, and its classifier-free
    /// counterpart with every token but the first and the two trailing structure tokens replaced by
    /// the audio-CFG token.
    static func textIDs(caption: String, lyrics: String, tokenizer: NFKTokenizer) throws -> MLXArray {
        let ids = tokenizer.encode(assembled(caption: caption, lyrics: lyrics))
            .map { Int32(truncating: $0) }
        guard ids.count <= NFKMusic3Contract.maxPromptTokens else {
            throw NFKMLXError.unsupportedConfiguration(
                "the assembled prompt has \(ids.count) tokens; the maximum is "
                + "\(NFKMusic3Contract.maxPromptTokens)")
        }
        var unconditional = ids
        if ids.count > 3 {
            for index in 1 ..< ids.count - 2 {
                unconditional[index] = Int32(NFKMusic3Contract.audioCFGToken)
            }
        }
        return stacked([MLXArray(ids), MLXArray(unconditional)], axis: 0)
    }

    // MARK: helpers

    /// Python `splitlines` semantics for the newline the cleaners see: a trailing newline yields no
    /// trailing empty line.
    private static func splitLines(_ text: String) -> [String] {
        var lines = text.components(separatedBy: "\n")
        if text.hasSuffix("\n") { lines.removeLast() }
        return lines
    }

    /// `<|tag value|>` → "tag is value"; `<|tag|>` → "tag". The remainder keeps its own internal
    /// whitespace, as the reference's `split(None, 1)` does.
    private static func rewritingSpecialTags(_ text: String) -> String {
        let regex = try! NSRegularExpression(pattern: "<\\|([^|]*)\\|>")
        let source = text as NSString
        var result = ""
        var cursor = 0
        for match in regex.matches(in: text, range: NSRange(location: 0, length: source.length)) {
            result += source.substring(with: NSRange(location: cursor,
                                                     length: match.range.location - cursor))
            let inner = source.substring(with: match.range(at: 1))
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if let firstSpace = inner.firstIndex(where: { $0.isWhitespace }) {
                let head = String(inner[..<firstSpace])
                let tail = String(inner[firstSpace...]).drop { $0.isWhitespace }
                result += tail.isEmpty ? inner : "\(head) is \(tail)"
            } else {
                result += inner
            }
            cursor = match.range.location + match.range.length
        }
        result += source.substring(from: cursor)
        return result
    }

    private static func lowercasingTags(_ text: String) -> String {
        let regex = try! NSRegularExpression(pattern: "\\[([^\\]]+)\\]")
        let source = text as NSString
        var result = ""
        var cursor = 0
        for match in regex.matches(in: text, range: NSRange(location: 0, length: source.length)) {
            result += source.substring(with: NSRange(location: cursor,
                                                     length: match.range.location - cursor))
            result += "[" + source.substring(with: match.range(at: 1)).lowercased() + "]"
            cursor = match.range.location + match.range.length
        }
        result += source.substring(from: cursor)
        return result
    }

    private static func replacing(_ text: String, pattern: String, with template: String,
                                  options: NSRegularExpression.Options = []) -> String {
        let regex = try! NSRegularExpression(pattern: pattern, options: options)
        return regex.stringByReplacingMatches(in: text, options: [],
                                              range: NSRange(location: 0, length: (text as NSString).length),
                                              withTemplate: template)
    }
}

// MARK: - Quantized releases

extension NFKMLXMusic3 {

    /// Writes a quantized copy of a downloaded release, in the release's own layout, so
    /// ``backend(directoryURL:)`` takes the result unchanged.
    ///
    /// The language model and depth decoder pack their `Linear` layers to affine `bits` and the DiT
    /// to `transformerBits`. The language model also packs its input embedding, its largest tensor:
    /// the model is untied, so the embedding is a lookup table separate from the packed `lm_head` and
    /// quantizing it leaves the logit head alone. Norms and convolutions stay at their stored
    /// precision, and the vocoder and condition encoder copy through unquantized — they are small, and
    /// the waveform passes through them last. The tokenizer files and the LICENSE copy through too: the
    /// license travels with the weights.
    ///
    /// The split default is MEASURED, not assumed: at 4-bit the language model's first-step logits
    /// hold a 0.9995 cosine against the full-precision parity record, and packing the embedding at
    /// 4-bit alongside holds 0.99933 while reclaiming 1.10 GiB, while the DiT's velocity falls to
    /// 0.978 — the flow field is the quantization-sensitive stage — so the DiT defaults to 8-bit,
    /// where the stack still fits the working set with room to stay resident.
    ///
    /// **Fallback precision — measured, for when stack size is the binding constraint.**
    /// `transformerBits: 6` is the DiT middle ground: velocity cosine 0.99844 against the record
    /// (against 0.99990 at 8-bit and 0.97751 at 4-bit — `testTheDiTQuantizationBitWidthSweep` is the
    /// record), reclaiming about 0.6 GB more. Because the DiT's error compounds over the sampling
    /// loop, cosine alone does not settle it — do a listening A/B before shipping a 6-bit DiT. The
    /// `bits`/`transformerBits` split is itself the mixed-precision recipe (4-bit language model,
    /// 8-bit DiT); there is no finer per-DiT-layer mixing, because whole-DiT 6-bit is the only point
    /// on the curve measured to be usable. The tied-model counterpart is the `includeEmbeddings`
    /// argument of `NFKMLXQuantization.quantize(module:bits:groupSize:includeEmbeddings:)`: safe to
    /// enable at 8-bit on a TIED language model (the tied-head cost is ~1e-5 there), a small ~0.006
    /// hit at 4-bit, and off by default because the wrong-width case is where packing the head costs
    /// most.
    ///
    /// One-time and blocking (the bf16 language model loads whole before packing); run it off the
    /// main thread.
    ///
    /// Introduced in InferKit 0.2.0.
    public static func quantizeRelease(at source: URL, to destination: URL,
                                       bits: Int = 4, transformerBits: Int = 8,
                                       groupSize: Int = 64) throws {
        let manager = FileManager.default
        func copyThrough(_ relative: String) throws {
            let from = source.appendingPathComponent(relative)
            guard manager.fileExists(atPath: from.path) else { return }
            let to = destination.appendingPathComponent(relative)
            try manager.createDirectory(at: to.deletingLastPathComponent(),
                                        withIntermediateDirectories: true)
            try? manager.removeItem(at: to)
            try manager.copyItem(at: from, to: to)
        }

        let languageDestination = destination.appendingPathComponent("language_model")
        try manager.createDirectory(at: languageDestination, withIntermediateDirectories: true)
        let configuration = try NFKMLXLanguage.configuration(
            fromHuggingFace: source.appendingPathComponent("language_model/config.json"))
        var language: NFKMLXLanguageNet? = NFKMLXLanguage.makeNet(configuration)
        try NFKMLXLanguage.loadWeights(into: language!,
                                       fromDirectory: source.appendingPathComponent("language_model"),
                                       precision: .checkpoint)
        // The LM is untied (its lm_head is a separate Linear, packed here too), so quantizing the
        // input embedding costs almost nothing — measured at logits cosine 0.99933 against 0.99952
        // for the bf16 embedding — and reclaims the 1.6 GiB it occupies unquantized, the largest
        // remaining tensor in the stack.
        NFKMLXQuantization.quantize(module: language!, bits: bits, groupSize: groupSize,
                                    includeEmbeddings: true)
        try NFKMLXWeights.save(language!, to: languageDestination.appendingPathComponent("model.safetensors"))
        language = nil
        NFKMLXGPU.clearCache()

        var depth: NFKMusic3DepthDecoderNet? = makeDepthDecoder()
        try loadDepthWeights(
            into: depth!,
            from: source.appendingPathComponent("rvq_depth_decoder/diffusion_pytorch_model.safetensors"),
            precision: .checkpoint)
        NFKMLXQuantization.quantize(module: depth!, bits: bits, groupSize: groupSize)
        try manager.createDirectory(at: destination.appendingPathComponent("rvq_depth_decoder"),
                                    withIntermediateDirectories: true)
        try NFKMLXWeights.save(depth!, to: destination
            .appendingPathComponent("rvq_depth_decoder/diffusion_pytorch_model.safetensors"))
        depth = nil
        NFKMLXGPU.clearCache()

        var transformer: NFKMusic3DiTNet? = makeDiT()
        try loadDiTWeights(into: transformer!, from: source.appendingPathComponent("transformer"))
        NFKMLXQuantization.quantize(module: transformer!, bits: transformerBits, groupSize: groupSize)
        try manager.createDirectory(at: destination.appendingPathComponent("transformer"),
                                    withIntermediateDirectories: true)
        try NFKMLXWeights.save(transformer!, to: destination
            .appendingPathComponent("transformer/diffusion_pytorch_model.safetensors"))
        transformer = nil
        NFKMLXGPU.clearCache()

        for relative in ["language_model/config.json", "rvq_depth_decoder/config.json",
                         "transformer/config.json", "scheduler/scheduler_config.json",
                         "vocoder/config.json", "vocoder/diffusion_pytorch_model.safetensors",
                         "condition_encoder/config.json",
                         "condition_encoder/diffusion_pytorch_model.safetensors",
                         "qwen_7B/qwen3-8B-tokenizer-music/vocab.json",
                         "qwen_7B/qwen3-8B-tokenizer-music/merges.txt",
                         "qwen_7B/qwen3-8B-tokenizer-music/added_tokens.json",
                         "LICENSE"] {
            try copyThrough(relative)
        }
    }
}

// MARK: - Backend

/// The loaded stages a music backend retains across runs when the whole stack fits the working
/// set. Runs serialize on the lock — the weights are GPU-scale, and two concurrent songs would
/// contend for everything.
private final class NFKMusic3StageCache: @unchecked Sendable {
    let lock = NSLock()
    var languageModel: NFKMLXLanguageNet?
    var depthDecoder: NFKMusic3DepthDecoderNet?
    var conditionEncoder: NFKMusic3ConditionEncoderNet?
    var transformer: NFKMusic3DiTNet?
    var vocoder: NFKMusic3VocoderNet?

    func drop() {
        languageModel = nil
        depthDecoder = nil
        conditionEncoder = nil
        transformer = nil
        vocoder = nil
    }

    var isResident: Bool { languageModel != nil && transformer != nil && vocoder != nil }
}

/// MiniMax Music 3 behind the InferKit contract: a music description under `NFKInputPrompt` and
/// lyrics under `NFKInputLyrics` become a stereo 44.1 kHz `NFKAudioAsset` under `NFKOutputAudio`.
///
/// Honored parameters: `NFKParameterDurationSeconds` (upper bound; the model may stop earlier;
/// capped at six minutes), `NFKParameterSeed`, `NFKParameterSteps` (flow-matching steps per window),
/// and `NFKParameterGuidanceScale` (the DiT's, default 1.7).
///
/// Whether the stages stay loaded between runs is DECIDED FROM THE WEIGHTS, not assumed: when the
/// stack's weight bytes plus a reserve for activations and the CFG pair's key-value cache fit the
/// machine's working set — which a release quantized by
/// ``NFKMLXMusic3/quantizeRelease(at:to:bits:transformerBits:groupSize:)`` does — every stage loads once and later
/// runs skip straight to generation. Otherwise the stages load from disk per run and each is freed
/// when its part is done: the full-precision language model (16 GiB at bf16) and float32 DiT
/// (9.7 GB) together exceed a 32 GB machine's working set, and the pipeline is strictly sequential,
/// so staging is what makes the model runnable there at all.
@objc(NFKMLXMusicBackend)
public final class NFKMLXMusicBackend: NSObject, NFKInferenceBackend {

    private let directoryURL: URL
    private let outputDirectory: URL
    private let stages = NFKMusic3StageCache()

    /// - Parameters:
    ///   - directoryURL: the release tree (`language_model/`, `rvq_depth_decoder/`,
    ///     `condition_encoder/`, `transformer/`, `vocoder/`, and the byte-level tokenizer files
    ///     under `qwen_7B/qwen3-8B-tokenizer-music/`), as `MiniMaxAI/MiniMax-Music3` publishes it.
    ///   - outputDirectory: where the generated WAV files are written.
    public init(directoryURL: URL,
                outputDirectory: URL = FileManager.default.temporaryDirectory) {
        self.directoryURL = directoryURL
        self.outputDirectory = outputDirectory
        super.init()
    }

    @objc public var isReady: Bool {
        FileManager.default.fileExists(
            atPath: directoryURL.appendingPathComponent("language_model/config.json").path)
    }

    @objc public var backendIdentifier: String { "minimax-music3" }

    @objc(runInferenceForRequest:error:)
    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        let job = submitInferenceJob(for: request)
        let semaphore = DispatchSemaphore(value: 0)
        job.completionHandler = { _ in semaphore.signal() }
        semaphore.wait()
        if let result = job.result { return result }
        if let error = job.error { throw error }
        throw NFKMLXError.noOutput
    }

    @objc(submitInferenceJobForRequest:)
    public func submitInferenceJob(for request: NFKInferenceRequest) -> NFKInferenceJob {
        let job = NFKInferenceJob()
        let directoryURL = self.directoryURL
        let outputDirectory = self.outputDirectory
        let stages = self.stages
        Task.detached(priority: .userInitiated) {
            do {
                if let result = try NFKMLXMusicBackend.run(request, directoryURL: directoryURL,
                                                           outputDirectory: outputDirectory,
                                                           stages: stages, job: job) {
                    job.finish(with: result)
                }
            } catch {
                job.finish(withError: error as NSError)
            }
        }
        return job
    }

    // MARK: Residency

    /// Bytes the stack's weight files occupy on disk — which is what a `.checkpoint`-precision load
    /// costs in memory, since the stored dtypes are adopted.
    static func stackWeightBytes(in directory: URL) -> Int {
        let manager = FileManager.default
        var total = 0
        for component in ["language_model", "rvq_depth_decoder", "condition_encoder",
                          "transformer", "vocoder"] {
            let folder = directory.appendingPathComponent(component)
            for name in (try? manager.contentsOfDirectory(atPath: folder.path)) ?? []
            where name.hasSuffix(".safetensors") {
                let path = folder.appendingPathComponent(name).path
                total += (try? manager.attributesOfItem(atPath: path)[.size] as? Int).flatMap { $0 } ?? 0
            }
        }
        return total
    }

    /// Whether the stages stay loaded between runs. The reserve covers activations, the CFG pair's
    /// key-value cache at the full position budget (~3 GB at bf16), and the sampler's scratch. The
    /// budget is the working set Metal recommends, NOT live free memory — a resident backend's own
    /// weights would count against "free" on the next check and evict themselves.
    static func keepsStagesResident(weightBytes: Int, workingSetBudget: Int) -> Bool {
        weightBytes + (4 << 30) <= workingSetBudget
    }

    private static func residencyBudget() -> Int {
        let recommended = NFKHardwareProfile.current.recommendedWorkingSetSize
        return recommended > 0 ? Int(Double(recommended) * 0.85) : 0
    }

    /// Whether this backend is holding its stages loaded (after at least one run on a stack that
    /// fits). Exposed for tests and for a host deciding when to construct a second heavy backend.
    public var isHoldingStagesResident: Bool {
        stages.lock.lock()
        defer { stages.lock.unlock() }
        return stages.isResident
    }

    /// Returns nil only when the job was cancelled.
    private static func run(_ request: NFKInferenceRequest, directoryURL: URL, outputDirectory: URL,
                            stages: NFKMusic3StageCache, job: NFKInferenceJob) throws -> NFKInferenceResult? {
        guard let caption = request.input(forKey: NFKInputPrompt) as? String, !caption.isEmpty,
              let lyrics = request.input(forKey: NFKInputLyrics) as? String, !lyrics.isEmpty else {
            throw NFKMLXError.unsupportedInput
        }
        func parameter(_ key: String) -> NSNumber? { request.parameter(forKey: key) as? NSNumber }
        let duration = parameter(NFKParameterDurationSeconds)?.doubleValue ?? 10
        guard duration > 0 else { throw NFKMLXError.unsupportedInput }
        let seed = parameter(NFKParameterSeed)?.uint64Value ?? 0
        let steps = max(parameter(NFKParameterSteps)?.intValue ?? 30, 1)
        let guidance = parameter(NFKParameterGuidanceScale)?.floatValue ?? 1.7
        let maxFrames = min(Int(duration * NFKMusic3Contract.framesPerSecond),
                            NFKMusic3Contract.maxAudioFrames)

        let tokenizer = try NFKMusic3Prompt.tokenizer(
            directory: directoryURL.appendingPathComponent("qwen_7B/qwen3-8B-tokenizer-music"))
        let textIDs = try NFKMusic3Prompt.textIDs(caption: caption, lyrics: lyrics,
                                                  tokenizer: tokenizer)

        stages.lock.lock()
        defer { stages.lock.unlock() }
        let resident = keepsStagesResident(weightBytes: stackWeightBytes(in: directoryURL),
                                           workingSetBudget: residencyBudget())
        if !resident { stages.drop() }

        // Stage A — the autoregressive pass, scoped so a non-resident run RELEASES the language
        // model before the DiT loads (locals live to function exit otherwise, and the bf16 model
        // plus the float32 DiT together are what does not fit). It loads at the checkpoint's stored
        // precision (bf16, or the quantized packing); float32 would not fit beside anything else.
        let generation: NFKMusic3AutoregressiveStage.Generation = try {
            let languageModel: NFKMLXLanguageNet
            let depthDecoder: NFKMusic3DepthDecoderNet
            if let cachedLanguage = stages.languageModel, let cachedDepth = stages.depthDecoder {
                languageModel = cachedLanguage
                depthDecoder = cachedDepth
            } else {
                let languageDirectory = directoryURL.appendingPathComponent("language_model")
                let configuration = try NFKMLXLanguage.configuration(
                    fromHuggingFace: languageDirectory.appendingPathComponent("config.json"))
                languageModel = NFKMLXLanguage.makeNet(configuration)
                try NFKMLXLanguage.loadWeights(into: languageModel, fromDirectory: languageDirectory,
                                               precision: .checkpoint)
                depthDecoder = NFKMLXMusic3.makeDepthDecoder()
                try NFKMLXMusic3.loadDepthWeights(
                    into: depthDecoder,
                    from: directoryURL.appendingPathComponent("rvq_depth_decoder/diffusion_pytorch_model.safetensors"),
                    precision: .checkpoint)
                if resident {
                    stages.languageModel = languageModel
                    stages.depthDecoder = depthDecoder
                }
            }
            let stage = NFKMusic3AutoregressiveStage(languageModel: languageModel,
                                                     depthDecoder: depthDecoder)
            job.reportProgress(0.05)
            return try stage.generate(textIDs: textIDs, maxFrames: maxFrames, seed: seed)
        }()
        NFKMLXGPU.clearCache()
        if job.status == .cancelled { return nil }
        job.reportProgress(0.5)

        // Stage B — windowed flow matching, scoped the same way. The AR stage's fused hidden
        // states arrive as float32.
        let denoised: [MLXArray]? = try {
            let conditionEncoder: NFKMusic3ConditionEncoderNet
            let transformer: NFKMusic3DiTNet
            if let cachedCondition = stages.conditionEncoder, let cachedTransformer = stages.transformer {
                conditionEncoder = cachedCondition
                transformer = cachedTransformer
            } else {
                conditionEncoder = NFKMLXMusic3.makeConditionEncoder()
                try NFKMLXMusic3.loadConditionWeights(
                    into: conditionEncoder,
                    from: directoryURL.appendingPathComponent("condition_encoder/diffusion_pytorch_model.safetensors"))
                transformer = NFKMLXMusic3.makeDiT()
                try NFKMLXMusic3.loadDiTWeights(into: transformer,
                                                from: directoryURL.appendingPathComponent("transformer"))
                if resident {
                    stages.conditionEncoder = conditionEncoder
                    stages.transformer = transformer
                }
            }
            let matcher = NFKMusic3FlowMatcher(transformer: transformer,
                                               conditionEncoder: conditionEncoder)
            matcher.guidanceScale = guidance
            let hiddens = generation.frameHiddens.asType(.float32)
            return matcher.latentChunks(frameHiddens: hiddens, steps: steps, seed: seed,
                                        progress: { step, total in
                job.reportProgress(0.5 + 0.45 * Double(step) / Double(total))
                return job.status != .cancelled
            })
        }()
        guard let chunks = denoised else { return nil }
        NFKMLXGPU.clearCache()

        // Stage C — decode, crop, and stitch.
        let vocoder: NFKMusic3VocoderNet
        if let cachedVocoder = stages.vocoder {
            vocoder = cachedVocoder
        } else {
            vocoder = NFKMLXMusic3.makeVocoder()
            try NFKMLXMusic3.loadVocoderWeights(
                into: vocoder,
                from: directoryURL.appendingPathComponent("vocoder/diffusion_pytorch_model.safetensors"))
            if resident { stages.vocoder = vocoder }
        }
        let hop = vocoder.configuration.hop
        var interleaved = [Float]()
        for (index, latents) in chunks.enumerated() {
            let wave = vocoder.waveform(latents)                    // [1, samples, 2]
            eval(wave)
            let kept = NFKMusic3FlowMatcher.keptRange(chunkIndex: index, chunkCount: chunks.count,
                                                      samples: wave.shape[1], hop: hop)
            let span = clip(wave[0..., kept], min: -1, max: 1)
            interleaved.append(contentsOf: span.reshaped([-1]).asArray(Float.self))
            if job.status == .cancelled { return nil }
        }
        NFKMLXGPU.clearCache()

        let sampleRate = vocoder.configuration.samplingRate
        let url = outputDirectory.appendingPathComponent("minimax-music3-\(UUID().uuidString).wav")
        try NFKMLXWaveFile.write(samples: interleaved, sampleRate: sampleRate, channels: 2, to: url)
        let asset = NFKAudioAsset(fileURL: url,
                                  durationSeconds: Double(interleaved.count / 2) / Double(sampleRate),
                                  sampleRate: Double(sampleRate),
                                  channelCount: 2)
        job.reportProgress(1)
        return NFKInferenceResult(outputs: [NFKOutputAudio: asset])
    }
}

// MARK: - Factories

extension NFKMLXMusic3 {

    /// The registry name the model builds under.
    @objc public static let modelName = "minimax-music3"

    /// Builds the music backend from a downloaded release directory (the `MiniMaxAI/MiniMax-Music3`
    /// tree). `isReady` reports whether the language model is present rather than failing the build,
    /// so a consumer can construct first and download later.
    ///
    /// Introduced in InferKit 0.2.0.
    @objc(backendWithDirectoryURL:error:)
    public static func backend(directoryURL: URL) throws -> any NFKInferenceBackend {
        NFKMLXMusicBackend(directoryURL: directoryURL)
    }

    /// Registers the factory under ``modelName``. The registry's `weightsURL` is the release
    /// DIRECTORY for this model; nil is refused, because a 27 GB stack has no useful
    /// random-weights form.
    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName) { weightsURL in
            guard let weightsURL else {
                throw NFKMLXError.unsupportedConfiguration(
                    "\(modelName) needs the release directory; there is no random-weights form")
            }
            return try backend(directoryURL: weightsURL)
        }
    }
}
