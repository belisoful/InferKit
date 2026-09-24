//
//  NFKMLXFlorence2Backend.swift
//  InferKitMLX
//

// The consumer surface for Florence-2: the processor (task-prompt expansion, the BART byte-level BPE
// tokenizer extended with the 1024 location tokens, the 768 image processor, and text / bounding-box
// post-processing) and an NFKInferenceBackend that runs one image plus a task token to a caption or a
// set of detections. Generation drives the shared seq2seq decoder from the fused image+text memory.

import Foundation
import CoreGraphics
import InferKit
import MLX
import MLXNN

// MARK: - Processor

public enum NFKMLXFlorence2Processor {

    /// The fixed task prompts. A task token is replaced by its natural-language prompt before tokenizing,
    /// exactly as `processing_florence2.py` does.
    public static let taskPromptsWithoutInput: [String: String] = [
        "<OCR>": "What is the text in the image?",
        "<OCR_WITH_REGION>": "What is the text in the image, with regions?",
        "<CAPTION>": "What does the image describe?",
        "<DETAILED_CAPTION>": "Describe in detail what is shown in the image.",
        "<MORE_DETAILED_CAPTION>": "Describe with a paragraph what is shown in the image.",
        "<OD>": "Locate the objects with category name in the image.",
        "<DENSE_REGION_CAPTION>": "Locate the objects in the image, with their descriptions.",
        "<REGION_PROPOSAL>": "Locate the region proposals in the image.",
    ]

    public static let taskPromptsWithInput: [String: String] = [
        "<CAPTION_TO_PHRASE_GROUNDING>": "Locate the phrases in the caption: {input}",
        "<REFERRING_EXPRESSION_SEGMENTATION>": "Locate {input} in the image with mask",
        "<REGION_TO_SEGMENTATION>": "What is the polygon mask of region {input}",
        "<OPEN_VOCABULARY_DETECTION>": "Locate {input} in the image.",
        "<REGION_TO_CATEGORY>": "What is the region {input}?",
        "<REGION_TO_DESCRIPTION>": "What does the region {input} describe?",
        "<REGION_TO_OCR>": "What text is in the region {input}?",
    ]

    /// The task tokens whose output is a set of `label<loc><loc><loc><loc>` detections.
    static let detectionTasks: Set<String> = [
        "<OD>", "<DENSE_REGION_CAPTION>", "<REGION_PROPOSAL>",
        "<CAPTION_TO_PHRASE_GROUNDING>", "<OPEN_VOCABULARY_DETECTION>",
    ]

    /// Replaces a task token with its prompt. A prompt that already carries no task token passes through.
    public static func expandPrompt(_ text: String) -> String {
        if let prompt = taskPromptsWithoutInput[text] { return prompt }
        for (token, template) in taskPromptsWithInput where text.hasPrefix(token) {
            let input = String(text.dropFirst(token.count))
            return template.replacingOccurrences(of: "{input}", with: input)
        }
        return text
    }

    /// The 1024 tokens Florence adds after the 50265-entry BART vocabulary: four OD/OCR markers, the
    /// 1000 location tokens, then twenty structural markers.
    static func addedTokens() -> [String] {
        var tokens = ["<od>", "</od>", "<ocr>", "</ocr>"]
        tokens += (0 ..< 1000).map { "<loc_\($0)>" }
        tokens += ["<cap>", "</cap>", "<ncap>", "</ncap>", "<dcap>", "</dcap>", "<grounding>", "</grounding>",
                   "<seg>", "</seg>", "<sep>", "<region_cap>", "</region_cap>", "<region_to_desciption>",
                   "</region_to_desciption>", "<proposal>", "</proposal>", "<poly>", "</poly>", "<and>"]
        return tokens
    }

    /// Builds the BART byte-level BPE tokenizer from `tokenizer.json`, adding the 1024 Florence tokens as
    /// specials (ids continue from 50265) so the generated location tokens decode to their literals.
    public static func tokenizer(inDirectory directory: URL) -> NFKTokenizer? {
        guard let data = try? Data(contentsOf: directory.appendingPathComponent("tokenizer.json")),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let model = json["model"] as? [String: Any],
              let vocabulary = model["vocab"] as? [String: Any],
              let merges = model["merges"] as? [Any] else { return nil }

        var specials = [String: Int]()
        for entry in (json["added_tokens"] as? [[String: Any]]) ?? [] {
            if let content = entry["content"] as? String, let id = entry["id"] as? Int { specials[content] = id }
        }
        let base = (vocabulary.count)  // 50265
        for (offset, token) in addedTokens().enumerated() { specials[token] = base + offset }

        let scratch = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try? FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
        guard let vocabularyData = try? JSONSerialization.data(withJSONObject: vocabulary),
              (try? vocabularyData.write(to: scratch.appendingPathComponent("vocab.json"))) != nil else { return nil }
        var mergesText = "#version: 0.2\n"
        for entry in merges {
            if let pair = entry as? [String], pair.count == 2 { mergesText += pair[0] + " " + pair[1] + "\n" }
            else if let text = entry as? String { mergesText += text + "\n" }
        }
        guard (try? mergesText.write(to: scratch.appendingPathComponent("merges.txt"),
                                     atomically: true, encoding: .utf8)) != nil else { return nil }
        let manifest: [String: Any] = ["tokenizer": ["type": "bpe-bytelevel", "specialTokens": specials]]
        return try? NFKTokenizer(forManifest: manifest, directory: scratch)
    }

    /// The `[1, 768, 768, 3]` NHWC pixel tensor: the image as `[H, W, 3]` in `0…1` through the shared
    /// bridge, bilinear-resized to 768, then ImageNet normalization, matching the reference preprocessor
    /// (the reference's bicubic resize differs only slightly).
    public static func pixelValues(_ image: Any, side: Int = 768) throws -> MLXArray {
        let native = try NFKMLXImageBridge.tensor(from: image, channels: 3,
                                                  colorSpace: CGColorSpaceCreateDeviceRGB())   // [H, W, 3] 0…1
        let batched = native.reshaped([1, native.dim(0), native.dim(1), 3])
        let resized = NFKMLXResample.resizeBilinear(batched, height: side, width: side)          // [1, side, side, 3]
        let mean = MLXArray([Float(0.485), 0.456, 0.406]).reshaped([1, 1, 1, 3])
        let std = MLXArray([Float(0.229), 0.224, 0.225]).reshaped([1, 1, 1, 3])
        return (resized - mean) / std
    }

    /// The BART-wrapped prompt ids `[1, T]`: `<s>` (id 0), the prompt's byte-level BPE ids, then `</s>`
    /// (the eos id). The core tokenizer returns the ids for the text alone, so the markers are added here.
    public static func encodePrompt(_ prompt: String, tokenizer: NFKTokenizer, eosTokenId: Int) -> MLXArray {
        var ids: [Int32] = [0]
        ids += tokenizer.encode(prompt).map { Int32(truncating: $0) }
        ids.append(Int32(eosTokenId))
        return MLXArray(ids).reshaped([1, ids.count])
    }

    /// The generated text with the BART/Florence special markers removed.
    public static func cleanText(_ text: String) -> String {
        var out = text
        for marker in ["<s>", "</s>", "<pad>"] { out = out.replacingOccurrences(of: marker, with: "") }
        return out.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Parses `label<loc_x1><loc_y1><loc_x2><loc_y2>…` detection text into `NFKDetection`s with boxes
    /// normalized to `0…1` (dequantizing each 0…999 bin to its bin centre). A leading run before the first
    /// location group is the first label; each following label is the text between two location groups.
    public static func parseDetections(_ text: String) -> [NFKDetection] {
        let cleaned = cleanText(text)
        let pattern = "<loc_(\\d+)><loc_(\\d+)><loc_(\\d+)><loc_(\\d+)>"
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return [] }
        let ns = cleaned as NSString
        let matches = regex.matches(in: cleaned, range: NSRange(location: 0, length: ns.length))
        var detections: [NFKDetection] = []
        var labelStart = 0
        var currentLabel = ""
        for match in matches {
            let labelRange = NSRange(location: labelStart, length: match.range.location - labelStart)
            let label = ns.substring(with: labelRange).trimmingCharacters(in: .whitespacesAndNewlines)
            if !label.isEmpty { currentLabel = label }
            func bin(_ index: Int) -> Double { (Double(ns.substring(with: match.range(at: index))) ?? 0) }
            let x1 = (bin(1) + 0.5) / 1000, y1 = (bin(2) + 0.5) / 1000
            let x2 = (bin(3) + 0.5) / 1000, y2 = (bin(4) + 0.5) / 1000
            detections.append(NFKDetection(label: currentLabel, classIndex: -1, confidence: 1,
                                           boundingBox: CGRect(x: x1, y: y1, width: max(0, x2 - x1), height: max(0, y2 - y1))))
            labelStart = match.range.location + match.range.length
        }
        return detections
    }
}

// MARK: - Generation wrapper

/// Drives the shared seq2seq decoder from a precomputed fused image+text memory, so `generate` runs the
/// decoder against the joined sequence without re-encoding from token ids.
final class NFKFlorence2DecodeModel: NFKMLXSeq2SeqDecodable {
    let net: NFKMLXFlorence2Net
    let memory: MLXArray
    init(net: NFKMLXFlorence2Net, memory: MLXArray) { self.net = net; self.memory = memory }
    func encodeSource(_ tokens: MLXArray) -> MLXArray { memory }
    func makeDecodingCache() -> NFKMLXSeq2SeqCache { net.language.makeCache() }
    func decodeStep(_ tokens: MLXArray, memory: MLXArray, cache: NFKMLXSeq2SeqCache) -> MLXArray {
        net.language.decode(tokens, memory: memory, cache: cache)
    }
    func reorderCache(_ cache: NFKMLXSeq2SeqCache, rows: MLXArray) { cache.reorder(rows) }
}

// MARK: - Backend

@objc(NFKMLXFlorence2)
public final class NFKMLXFlorence2: NSObject {

    @objc public static let modelName = "florence-2-large"
    static let requiredFiles = ["config.json", "tokenizer.json"]
    static let optionalFiles = [String]()
    static let weightFiles = ["model.safetensors"]

    /// Builds a backend from a released Florence-2 directory: `config.json` for the geometry (base or
    /// large), `model.safetensors`, and `tokenizer.json`.
    @objc(backendWithDirectoryURL:error:)
    public static func backend(directoryURL: URL) throws -> NFKMLXFlorence2Backend {
        guard let tokenizer = NFKMLXFlorence2Processor.tokenizer(inDirectory: directoryURL) else {
            throw NFKMLXError.unsupportedConfiguration("Florence-2 tokenizer.json is missing or unreadable")
        }
        let geometry = try NFKMLXFlorence2Net.configuration(fromConfigURL: directoryURL.appendingPathComponent("config.json"))
        let net = NFKMLXFlorence2Net(vision: geometry.vision, text: geometry.text)
        try net.loadWeights(from: directoryURL.appendingPathComponent("model.safetensors"))
        let decoding = try generationDefaults(fromConfigURL: directoryURL.appendingPathComponent("config.json"),
                                              text: geometry.text)
        return NFKMLXFlorence2Backend(net: net, tokenizer: tokenizer, decoding: decoding)
    }

    /// The release's generation settings, which transformers reads from `text_config`: both releases
    /// ask for three beams, early stopping, no repeated 3-gram, and `<s>` forced first and `</s>` forced
    /// at the length limit. Captioning and OCR need them; under plain greedy decoding the caption
    /// degenerates to a run of `<s>`.
    static func generationDefaults(fromConfigURL url: URL, text: NFKMLXSeq2SeqConfiguration) throws -> NFKMLXSeq2SeqDecoding {
        let json = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any]
        let t = json?["text_config"] as? [String: Any] ?? [:]
        func int(_ key: String) -> Int? { (t[key] as? NSNumber)?.intValue }
        return NFKMLXSeq2SeqDecoding(beams: int("num_beams") ?? 1, maxTokens: maximumTokens,
                                     lengthPenalty: (t["length_penalty"] as? NSNumber)?.floatValue ?? 1,
                                     earlyStopping: (t["early_stopping"] as? NSNumber)?.boolValue ?? false,
                                     startToken: text.decoderStartTokenId, endToken: text.eosTokenId,
                                     forcedFirstToken: int("forced_bos_token_id"),
                                     forcedLastToken: int("forced_eos_token_id"),
                                     noRepeatNgramSize: int("no_repeat_ngram_size") ?? 0)
    }

    /// Downloads a release into the hub cache and builds the backend.
    ///
    /// @discussion The download fetches `config.json`, `tokenizer.json`, and `model.safetensors`, the
    /// files the build reads; the geometry comes from the configuration. A file the cache already holds
    /// is not fetched again. The call blocks on the network; call it off the render thread. The public
    /// releases are `microsoft/Florence-2-base` and `microsoft/Florence-2-large`; neither is gated.
    ///
    /// Introduced in InferKit 0.4.0.
    @objc(backendWithRepo:revision:cacheDirectoryURL:error:)
    public static func backend(repo: String, revision: String?, cacheDirectoryURL: URL?) throws -> NFKMLXFlorence2Backend {
        try backend(directoryURL: try NFKMLXReleaseDownload.directory(
            repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL,
            required: requiredFiles, optional: optionalFiles, weights: weightFiles))
    }

    /// The asynchronous form of ``backend(repo:revision:cacheDirectoryURL:)``. The download and the
    /// build run at user-initiated quality of service off the calling thread.
    ///
    /// Introduced in InferKit 0.4.0.
    @objc(backendWithRepo:revision:cacheDirectoryURL:completionHandler:)
    public static func backend(repo: String, revision: String?, cacheDirectoryURL: URL?,
                               completionHandler: @escaping (NFKMLXFlorence2Backend?, Error?) -> Void) {
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                completionHandler(try backend(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL), nil)
            } catch {
                completionHandler(nil, error)
            }
        }
    }

    /// The maximum tokens generated after the start token when a request does not set
    /// `NFKParameterMaxTokens`: 1024, the release's sample code.
    @objc public static var maximumTokens = 1024
}

/// The Florence-2 inference backend: an image plus a task token in, a caption (`NFKOutputText`) out, with
/// detections (`NFKOutputDetections`) added for the localization tasks.
///
/// @discussion Generation follows the release's settings (three beams, no repeated 3-gram). A request
/// may set `NFKParameterMaxTokens` and ``NFKMLXTranslationParameterKey/beamCount`` (1 decodes greedily).
public final class NFKMLXFlorence2Backend: NSObject, NFKInferenceBackend {
    private let holder: Holder
    private let tokenizer: NFKTokenizer
    /// The decode a request starts from: the release's generation settings.
    public let defaultDecoding: NFKMLXSeq2SeqDecoding

    final class Holder: @unchecked Sendable { let net: NFKMLXFlorence2Net; init(_ n: NFKMLXFlorence2Net) { net = n } }

    init(net: NFKMLXFlorence2Net, tokenizer: NFKTokenizer, decoding: NFKMLXSeq2SeqDecoding) {
        self.holder = Holder(net)
        self.tokenizer = tokenizer
        self.defaultDecoding = decoding
    }

    public var supportedInputKeys: Set<String> { [NFKInputImage, NFKInputPrompt] }
    public var supportedParameterKeys: Set<String> { [NFKParameterMaxTokens, NFKMLXTranslationParameterKey.beamCount] }
    public var isReady: Bool { true }
    public var backendIdentifier: String { NFKMLXFlorence2.modelName }

    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        guard let imageValue = request.input(forKey: NFKInputImage) else {
            throw NFKMLXError.unsupportedConfiguration("Florence-2 needs an NFKInputImage")
        }
        let task = (request.input(forKey: NFKInputPrompt) as? String) ?? "<CAPTION>"

        let net = holder.net
        let pixels = try NFKMLXFlorence2Processor.pixelValues(imageValue)
        let prompt = NFKMLXFlorence2Processor.expandPrompt(task)
        let inputIds = NFKMLXFlorence2Processor.encodePrompt(prompt, tokenizer: tokenizer,
                                                             eosTokenId: net.textConfig.eosTokenId)

        let memory = net.encode(pixels: pixels, inputIds: inputIds)
        let model = NFKFlorence2DecodeModel(net: net, memory: memory)
        var decoding = defaultDecoding
        decoding.maxTokens = NFKMLXFlorence2.maximumTokens
        if let value = request.parameter(forKey: NFKParameterMaxTokens) as? NSNumber {
            decoding.maxTokens = max(1, value.intValue)
        }
        if let value = request.parameter(forKey: NFKMLXTranslationParameterKey.beamCount) as? NSNumber {
            decoding.beams = max(1, value.intValue)
        }
        let ids = NFKMLXSeq2SeqDecoder.generate(model, source: [net.textConfig.decoderStartTokenId], decoding: decoding)
        let text = tokenizer.decode(ids.map { NSNumber(value: $0) })

        var outputs: [String: Any] = [NFKOutputText: NFKMLXFlorence2Processor.cleanText(text)]
        let taskToken = task.hasPrefix("<") ? String(task.prefix(while: { $0 != ">" })) + ">" : task
        if NFKMLXFlorence2Processor.detectionTasks.contains(taskToken) {
            outputs[NFKOutputDetections] = NFKMLXFlorence2Processor.parseDetections(text)
        }
        return NFKInferenceResult(outputs: outputs)
    }
}
