//
//  NFKMLXLoadOverheadTests.swift
//  InferKitMLXTests
//
//  The measurement behind the catalog's `loadOverheadBytes`: one catalog variant per process, built at its
//  default load options, then given one reference request. MLX's own peak is read across the load and
//  across the request. The load's transient is its peak above the bytes the loaded model holds once
//  memory settles; the request's peak is read above that same settled level. Both, the settled bytes, and
//  the catalog's estimate at the same options go to a JSON file that `Tools/catalog/overhead.py` turns
//  into `NFKMLXCatalogLoadOverheads.swift`.
//
//  The suite runs only when `IK_LOAD_OVERHEAD_VARIANT` names a variant as `model/variant` and
//  `IK_LOAD_OVERHEAD_OUTPUT` names the JSON file to write. `IK_LOAD_OVERHEAD_LOCATION` names the
//  release to load; an empty or absent location builds the network on initial values through the
//  registry, which measures activations and leaves the load transient unmeasured. Large releases run
//  under the Large Model Coordination lock (`overhead.py run` behind `Tools/lmc/lmc.py`).
//

import Darwin
import XCTest
import CoreGraphics
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXLoadOverheadTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "the measurement runs MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    func testTheNamedVariantsLoadAndReferenceRequestPeaks() throws {
        let environment = ProcessInfo.processInfo.environment
        guard let name = environment["IK_LOAD_OVERHEAD_VARIANT"], !name.isEmpty,
              let output = environment["IK_LOAD_OVERHEAD_OUTPUT"], !output.isEmpty else {
            throw XCTSkip("IK_LOAD_OVERHEAD_VARIANT and IK_LOAD_OVERHEAD_OUTPUT name the variant and the record")
        }
        try requireMLXRuntime()
        let location = environment["IK_LOAD_OVERHEAD_LOCATION"].flatMap { $0.isEmpty ? nil : URL(fileURLWithPath: $0) }
        var record = NFKMLXOverheadRecord(name: name, location: location)
        let cap = environment["IK_LOAD_OVERHEAD_CAP"].flatMap(Int.init) ?? 18 << 30
        NFKMLXOverheadWatchdog.start(name: name, weights: location == nil ? "initial" : "stored", cap: cap,
                                     output: URL(fileURLWithPath: output))
        defer {
            record.write(to: URL(fileURLWithPath: output))
        }
        do {
            try record.measure()
        } catch {
            record.fields["error"] = error.localizedDescription
        }
        XCTAssertNil(record.fields["error"], "\(name): \(record.fields["error"] ?? "")")
    }
}

/// One variant's measurement and the record it writes.
private struct NFKMLXOverheadRecord {

    let name: String
    let location: URL?
    var fields: [String: Any] = [:]

    init(name: String, location: URL?) {
        self.name = name
        self.location = location
        fields["variantKey"] = name
        fields["weights"] = location == nil ? "initial" : "stored"
        fields["location"] = location?.path ?? NSNull()
    }

    mutating func measure() throws {
        let parts = name.split(separator: "/", maxSplits: 1).map(String.init)
        guard parts.count == 2 else {
            throw Self.failure("\(name) is not model/variant")
        }
        NFKMLXReferenceModels.registerAll()
        let catalog = NFKModelCatalog()
        catalog.addDescriptors(NFKMLXModelDescriptors.modelDescriptors())
        guard let descriptor = catalog.descriptor(withIdentifier: parts[0]),
              let variant = descriptor.variants.first(where: { $0.identifier == parts[1] }) else {
            throw Self.failure("the catalog has no variant \(name)")
        }
        fields["family"] = descriptor.family
        fields["tasks"] = descriptor.tasks
        var options: [String: Any] = descriptor.variants.count > 1 ? [NFKModelLoadOptionVariant: variant.identifier] : [:]
        let budget = NFKModelMemoryBudget.current()
        fields["budgetBytes"] = budget.byteCount
        let request: NFKModelBuildRequest
        do {
            request = try catalog.buildRequest(forModelIdentifier: descriptor.identifier, options: options,
                                               location: location ?? FileManager.default.temporaryDirectory)
        } catch let error as NSError {
            guard let fitting = error.userInfo[NFKModelMemoryFittingOptionsKey] as? [String: Any] else {
                throw error
            }
            fields["refusedDefaults"] = error.localizedDescription
            options = fitting
            request = try catalog.buildRequest(forModelIdentifier: descriptor.identifier, options: options,
                                               location: location ?? FileManager.default.temporaryDirectory)
        }
        fields["loadOptions"] = Self.jsonValues(request.loadOptions)
        if let estimate = try? descriptor.estimatedMemory(forOptions: options, memoryBudget: budget) {
            fields["estimate"] = estimate.dictionaryRepresentation()
        }

        NFKMLXOverheadWatchdog.phase = "load"
        NFKMLXOverheadMemory.warmUp()
        let before = NFKMLXOverheadMemory.settle()
        fields["activeBeforeLoad"] = before.active
        NFKMLXOverheadMemory.resetPeaks()
        var started = Date()
        let backend = try build(request)
        let object = backend as AnyObject
        if object.responds(to: #selector(NFKMLXOptionalBackendMethods.prepare)) {
            try unsafeBitCast(object, to: NFKMLXOptionalBackendMethods.self).prepare()
        }
        NFKMLXOverheadMemory.synchronize()
        let loadPeak = NFKMLXOverheadMemory.peaks()
        fields["loadSeconds"] = Date().timeIntervalSince(started)
        let loaded = NFKMLXOverheadMemory.settle()
        fields["loadPeakActive"] = loadPeak.active
        fields["loadPeakFootprint"] = loadPeak.footprint
        fields["residentAfterLoad"] = loaded.active - before.active
        fields["footprintAfterLoad"] = loaded.footprint - before.footprint
        fields["loadTransient"] = max(0, loadPeak.active - loaded.active)
        fields["modelInfoResidency"] = backend.modelInfo?[NFKModelInfoResidency] ?? NSNull()

        var reference = try NFKMLXReferenceRequest(descriptor: descriptor, variant: variant, backend: backend,
                                                    defaults: request.requestDefaults, loadOptions: request.loadOptions,
                                                    repeats: location == nil)
        var settled = loaded
        if location == nil {
            // A network on initial values creates its parameters on first use, so a first request
            // materializes them and the second is the one measured.
            NFKMLXOverheadWatchdog.phase = "materializing request"
            do {
                try Self.run(&reference, on: backend)
            } catch {
                fields["requestError"] = error.localizedDescription
            }
            settled = NFKMLXOverheadMemory.settle()
            fields["residentAfterLoad"] = settled.active - before.active
            fields["residentBeforeMaterializing"] = loaded.active - before.active
        }
        NFKMLXOverheadWatchdog.phase = "request"
        NFKMLXOverheadMemory.resetPeaks()
        started = Date()
        if fields["requestError"] == nil {
            do {
                try Self.run(&reference, on: backend)
            } catch {
                fields["requestError"] = error.localizedDescription
            }
        }
        NFKMLXOverheadMemory.synchronize()
        let requestPeak = NFKMLXOverheadMemory.peaks()
        fields["requestSeconds"] = Date().timeIntervalSince(started)
        fields["request"] = reference.summary
        let after = NFKMLXOverheadMemory.settle()
        fields["requestPeakActive"] = requestPeak.active
        fields["requestPeakFootprint"] = requestPeak.footprint
        fields["requestPeak"] = max(0, requestPeak.active - settled.active)
        fields["activeAfterRequest"] = after.active - before.active
        reference.removeFiles()
        if fields["requestError"] != nil {
            throw Self.failure("the reference request failed: \(fields["requestError"] ?? "")")
        }
    }

    /// Runs the reference request, adapting it to what a translator's refusal names, at most four times.
    private static func run(_ reference: inout NFKMLXReferenceRequest, on backend: any NFKInferenceBackend) throws {
        for attempt in 0 ..< 4 {
            do {
                _ = try backend.runInference(for: reference.request)
                return
            } catch {
                guard attempt < 3, reference.adapt(after: error) else {
                    throw error
                }
            }
        }
    }

    /// The backend the catalog builds from the stored release, or, without one, the registry's network on
    /// initial values with the same load options.
    private func build(_ request: NFKModelBuildRequest) throws -> any NFKInferenceBackend {
        guard location == nil else {
            return try NFKMLXModelDescriptors.backend(for: request)
        }
        let name = request.variant?.registryName ?? request.descriptor.registryName ?? request.descriptor.identifier
        let consumed = NFKMLXModelRegistry.consumedOptionKeys(forModelNamed: name)
        return try NFKMLXModelRegistry.backend(named: name, weightsURL: nil,
                                               loadOptions: NFKMLXLoadOptions(request.loadOptions, draftURL: nil)) {
            try NFKMLXModelDescriptors.applyPropertyOptions(of: request, to: $0, skipping: consumed)
        }
    }

    func write(to url: URL) {
        guard let data = try? JSONSerialization.data(withJSONObject: Self.jsonValues(fields),
                                                     options: [.prettyPrinted, .sortedKeys]) else {
            return
        }
        try? data.write(to: url, options: .atomic)
    }

    private static func failure(_ description: String) -> NSError {
        NSError(domain: "NFKMLXLoadOverheadTests", code: 1, userInfo: [NSLocalizedDescriptionKey: description])
    }

    /// The values as JSON: numbers, strings, arrays, and dictionaries as they are, anything else described.
    static func jsonValues(_ dictionary: [String: Any]) -> [String: Any] {
        dictionary.mapValues(jsonValue)
    }

    private static func jsonValue(_ value: Any) -> Any {
        switch value {
        case let dictionary as [String: Any]: return jsonValues(dictionary)
        case let array as [Any]: return array.map(jsonValue)
        case is NSNumber, is String, is NSNull: return value
        case let url as URL: return url.path
        default: return String(describing: value)
        }
    }
}

/// A thread that ends the process with a record when MLX's active memory passes a cap, so one
/// measurement cannot push a shared machine into swap. The driver reads the record as any other.
///
/// @discussion One allocation can carry the memory past the cap between two readings: a 30 s STORM
/// request stopped at 28.6 GB against a 22 GiB cap with a 0.2 s poll. MLX's own memory limit makes the
/// scheduler wait and fails no allocation, so it cannot bound the peak either.
private enum NFKMLXOverheadWatchdog {

    nonisolated(unsafe) static var phase = "start"

    static func start(name: String, weights: String, cap: Int, output: URL) {
        let thread = Thread {
            while true {
                let active = NFKMLXGPU.activeMemory
                if active > cap {
                    let record: [String: Any] = [
                        "variantKey": name, "weights": weights, "overCap": true, "activeBytes": active,
                        "capBytes": cap, "phase": phase,
                        "error": "MLX's active memory reached \(active) bytes, over the cap of \(cap), in the \(phase)",
                    ]
                    if let data = try? JSONSerialization.data(withJSONObject: record, options: [.sortedKeys]) {
                        try? data.write(to: output, options: .atomic)
                    }
                    exit(3)
                }
                usleep(20_000)
            }
        }
        thread.start()
    }
}

/// MLX's active memory and the kernel's footprint, read after both streams drain.
private enum NFKMLXOverheadMemory {

    struct Reading {
        let active: Int
        let footprint: Int
    }

    // The interval maximum resets to the current footprint and then tracks the kernel's maximum. Its reset
    // has no public header; without the symbol the footprint peak reads the lifetime maximum.
    private static let resetFootprintInterval: (@convention(c) (pid_t) -> Int32)? = {
        guard let symbol = dlsym(UnsafeMutableRawPointer(bitPattern: -2), "proc_reset_footprint_interval") else {
            return nil
        }
        return unsafeBitCast(symbol, to: (@convention(c) (pid_t) -> Int32).self)
    }()

    private static func footprint() -> (now: Int, intervalPeak: Int) {
        var info = rusage_info_v4()
        let status = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) {
                proc_pid_rusage(getpid(), RUSAGE_INFO_V4, $0)
            }
        }
        guard status == 0 else { return (0, 0) }
        return (Int(info.ri_phys_footprint), Int(info.ri_interval_max_phys_footprint))
    }

    static func synchronize() {
        Stream.gpu.synchronize()
        Stream.cpu.synchronize()
    }

    static func warmUp() {
        let small = MLXRandom.uniform(low: -1, high: 1, [64, 64])
        eval(matmul(small, small).asType(.bfloat16))
    }

    /// The reading once MLX's cache is empty and the footprint holds within 4 MiB for 200 ms, or after 30 s.
    /// A footprint read right after a large free still counts buffers MLX released
    /// (`NFKMLXLoadFootprintTests`).
    static func settle() -> Reading {
        let deadline = Date().addingTimeInterval(30)
        var last = -1
        var steady = 0
        while true {
            synchronize()
            NFKMLXGPU.clearCache()
            let now = footprint().now
            steady = last >= 0 && abs(last - now) < 4 << 20 ? steady + 1 : 0
            last = now
            if (NFKMLXGPU.cacheMemory == 0 && steady >= 4) || Date() > deadline {
                return Reading(active: NFKMLXGPU.activeMemory, footprint: now)
            }
            usleep(50_000)
        }
    }

    static func resetPeaks() {
        NFKMLXGPU.resetPeakAllocatedByteCount()
        _ = resetFootprintInterval?(getpid())
    }

    static func peaks() -> Reading {
        Reading(active: NFKMLXGPU.peakAllocatedByteCount, footprint: footprint().intervalPeak)
    }
}

/// The request the measurement runs for a variant: the descriptor's required inputs at the reference sizes,
/// its default request options, one denoising step where it takes steps, and one generated token where it
/// generates tokens.
///
/// @discussion The reference sizes:
/// - an image is 1024×1024; a model of a fixed size resizes it as it does any other input
/// - a decoder's prompt is 4096 tokens, or its context less 64 where that is shorter
/// - a text encoder's prompt is 480 tokens; a reranker ranks 8 such documents against a one-sentence query
/// - a translator's prompt is 200 words, within the 512 positions of the smallest encoders
/// - audio is 30 s
/// - a video is 1280×720 at 24 frames per second, its frame count the default where an option sets one
///   and 16 frames otherwise
/// - a generator's prompt is one sentence, and a speech synthesizer's about 30 s of speech
private struct NFKMLXReferenceRequest {

    static let imageSide = 1024
    static let decoderTokens = 4096
    static let encoderTokens = 480
    static let translatorWords = 200
    static let audioSeconds = 30
    static let videoSize = (width: 1280, height: 720)
    static let videoFrames = 16

    private static let decodingTasks: Set<String> = [
        "text-generation", "code-generation", "vision-question-answering", "decision",
    ]
    private static let generatingTasks: Set<String> = [
        "text-to-image", "video-generation", "music-generation", "audio-generation", "controlnet",
    ]
    /// The languages a translator is asked for in turn when its refusal names none it takes.
    private static let targetLanguages = ["fr", "en", "zh", "de", "es"]
    private static let reusesPromptCache = "NFKMLXParameterReusesPromptCache"
    private static let words = ["the", "river", "runs", "past", "old", "stone", "walls", "and", "quiet", "fields"]
    private static let sentence = "A lighthouse on a rocky coast at dusk, photographed with a long lens."
    private static let speech = [String](repeating: "The morning train left the station on time, and the "
        + "passengers settled in for the long ride north along the coast.", count: 4).joined(separator: " ")

    private(set) var request: NFKInferenceRequest
    private(set) var summary: [String: Any]
    private let files: [URL]
    /// The input that holds the reference prompt, and its words.
    private var promptKey: String?
    private var promptWords = 0

    /// `repeats` is true for a request run twice, which reads no kept prompt cache.
    init(descriptor: NFKModelDescriptor, variant: NFKModelVariant, backend: any NFKInferenceBackend,
         defaults: [String: Any], loadOptions: [String: Any], repeats: Bool) throws {
        let tasks = Set(descriptor.tasks)
        var parameters = defaults
        var summary = [String: Any]()
        if descriptor.option(withKey: NFKParameterSteps) != nil {
            parameters[NFKParameterSteps] = 1
            summary["steps"] = 1
        }
        if descriptor.option(withKey: NFKParameterMaxTokens) != nil {
            parameters[NFKParameterMaxTokens] = 1
            summary["maxTokens"] = 1
        }
        if repeats, descriptor.option(withKey: Self.reusesPromptCache) != nil {
            parameters[Self.reusesPromptCache] = false
        }
        for (key, fallback) in [(NFKParameterTargetLanguage, "fr"), (NFKParameterSourceLanguage, "en")]
            where parameters[key] == nil {
            guard let option = descriptor.option(withKey: key) else {
                continue
            }
            let taken = [NFKParameterTargetLanguage, NFKParameterSourceLanguage].compactMap { parameters[$0] as? String }
            let value = option.choices.map(\.value).first { !taken.contains(($0 as? String) ?? "") } ?? fallback
            parameters[key] = value
            summary[key] = value
        }
        let context = (loadOptions[NFKModelLoadOptionContextLength] as? NSNumber)?.intValue
            ?? variant.memoryFacts?.maximumContextLength?.intValue
        let decodes = variant.memoryFacts?.keyValueByteCountPerToken != nil || !tasks.isDisjoint(with: Self.decodingTasks)
        let promptTokens = tasks.contains("translation") && !decodes ? Self.translatorWords
            : decodes ? min(Self.decoderTokens, (context ?? Self.decoderTokens) - 64) : Self.encoderTokens
        let generates = !tasks.isDisjoint(with: Self.generatingTasks)
        let frames = (defaults[NFKParameterFrameCount] as? NSNumber)?.intValue ?? Self.videoFrames

        var inputs = [String: Any]()
        var files = [URL]()
        var promptKeys = [String]()
        for port in Self.ports(of: descriptor) {
            switch port.valueType {
            case "image":
                inputs[port.key] = Self.image(side: Self.imageSide)
                summary["imageSide"] = Self.imageSide
            case "mask":
                inputs[port.key] = Self.mask(side: Self.imageSide)
            case "audio":
                inputs[port.key] = Self.wave()
                summary["audioSeconds"] = Self.audioSeconds
            case "audios":
                inputs[port.key] = [Data](repeating: Self.wave(), count: 4)
                summary["audioSeconds"] = Self.audioSeconds
            case "video":
                let url = FileManager.default.temporaryDirectory
                    .appendingPathComponent("NFKMLXLoadOverheadTests-\(UUID().uuidString).mov")
                try NFKMLXVideoFile.write((0 ..< frames).map { Self.image(width: Self.videoSize.width,
                                                                               height: Self.videoSize.height, shift: $0) },
                                          framesPerSecond: 24, to: url)
                files.append(url)
                inputs[port.key] = NFKVideoAsset(fileURL: url)
                summary["videoFrames"] = frames
                summary["videoSize"] = "\(Self.videoSize.width)x\(Self.videoSize.height)"
            case "documents":
                inputs[port.key] = (0 ..< 8).map { Self.text(words: Self.encoderTokens, offset: $0) }
                summary["documents"] = 8
                if inputs[NFKInputPrompt] == nil, descriptor.inputs.contains(where: { $0.key == NFKInputPrompt }) {
                    inputs[NFKInputPrompt] = Self.sentence
                }
            case "questions":
                inputs[port.key] = ["urgent": NFKDecisionQuestion.noulQuestion(withInstructions: "Is this urgent?")]
                inputs[NFKInputState] = Self.text(words: promptTokens, offset: 0)
                promptKeys.append(NFKInputState)
            case "json":
                inputs[port.key] = (0 ..< 2048).map { NSNumber(value: sin(Double($0) / 24) * 10 + 50) }
                summary["seriesLength"] = 2048
            default:
                if port.key == NFKInputPrompt || port.valueType == "text" {
                    if tasks.contains("speech-synthesis") {
                        inputs[port.key] = Self.speech
                    } else if tasks.contains("reranking") || generates || port.key != NFKInputPrompt {
                        inputs[port.key] = inputs[port.key] ?? Self.sentence
                    } else {
                        inputs[port.key] = Self.text(words: promptTokens, offset: 0)
                        promptKeys.append(port.key)
                    }
                }
            }
        }
        var request = NFKInferenceRequest(inputs: inputs, parameters: parameters)
        if let key = promptKeys.first {
            var words = promptTokens
            // A tokenizer that splits the reference words differently gets the word count that gives the
            // token count, measured once and corrected once.
            for _ in 0 ..< 2 {
                guard let counted = Self.tokenCount(of: request, by: backend), counted > 0,
                      abs(counted - promptTokens) > 8 else {
                    break
                }
                words = max(1, words * promptTokens / counted)
                inputs[key] = Self.text(words: words, offset: 0)
                request = NFKInferenceRequest(inputs: inputs, parameters: parameters)
            }
            summary["promptTokensTarget"] = promptTokens
            summary["promptWords"] = words
            promptKey = key
            promptWords = words
            if let counted = Self.tokenCount(of: request, by: backend) {
                summary["promptTokens"] = counted
            }
        }
        self.request = request
        self.summary = summary
        self.files = files
    }

    /// The required inputs, or, where none is required (a prompt or messages, either one), the first the
    /// descriptor takes of an image for an image task, else of a prompt, an image, audio, and a video.
    private static func ports(of descriptor: NFKModelDescriptor) -> [NFKModelPort] {
        let required = descriptor.inputs.filter(\.isRequired)
        guard required.isEmpty else {
            return required
        }
        let imageTask = descriptor.tasks.contains { $0.hasPrefix("image-") }
        let order = imageTask ? [NFKInputImage, NFKInputPrompt, NFKInputAudio, NFKInputVideo]
            : [NFKInputPrompt, NFKInputImage, NFKInputAudio, NFKInputVideo]
        for key in order {
            if let port = descriptor.inputs.first(where: { $0.key == key }) {
                return [port]
            }
        }
        return []
    }

    /// The request adapted to a translator's refusal. False where the refusal names nothing to adapt to.
    ///
    /// @discussion
    /// - "… reads 512 source tokens, and this paragraph holds 521" → the prompt shortened in proportion,
    ///   with a twentieth to spare
    /// - "… does not translate into es; it translates into nan, …, zh, …" → a listed target, `zh` where
    ///   the list holds it, else the first
    /// - "… translates into eng, not fr" or "… translates from vie, not en" → the language it names
    /// - "… does not translate into fr" with no list → the next of ``targetLanguages``
    mutating func adapt(after error: Error) -> Bool {
        let message = error.localizedDescription
        var parameters = request.parameters
        if let limit = Self.integers(in: message, after: "reads ", before: " source tokens"),
           let held = Self.integers(in: message, after: "this paragraph holds ", before: nil),
           let key = promptKey, held > limit {
            promptWords = max(1, promptWords * limit * 19 / (held * 20))
            var inputs = request.inputs
            inputs[key] = Self.text(words: promptWords, offset: 0)
            request = NFKInferenceRequest(inputs: inputs, parameters: parameters)
            summary["promptWords"] = promptWords
            summary["promptTokensLimit"] = limit
            return true
        }
        if let range = message.range(of: "; it translates into ") {
            let listed = message[range.upperBound...].split(separator: ",").map {
                $0.trimmingCharacters(in: .whitespacesAndNewlines.union(.punctuationCharacters))
            }
            if let language = listed.contains("zh") ? "zh" : listed.first, !language.isEmpty,
               parameters[NFKParameterTargetLanguage] as? String != language {
                parameters[NFKParameterTargetLanguage] = language
                return adopt(parameters, key: NFKParameterTargetLanguage, language: language)
            }
        }
        for (phrase, key) in [("translates into ", NFKParameterTargetLanguage), ("translates from ", NFKParameterSourceLanguage)] {
            guard !message.contains("does not " + phrase.replacingOccurrences(of: "translates", with: "translate")),
                  let range = message.range(of: phrase),
                  let language = message[range.upperBound...].split(separator: ",").first.map(String.init),
                  parameters[key] as? String != language else {
                continue
            }
            parameters[key] = language
            return adopt(parameters, key: key, language: language)
        }
        guard message.contains("does not translate into "),
              let current = parameters[NFKParameterTargetLanguage] as? String,
              let index = Self.targetLanguages.firstIndex(of: current), index + 1 < Self.targetLanguages.count else {
            return false
        }
        let language = Self.targetLanguages[index + 1]
        parameters[NFKParameterTargetLanguage] = language
        if parameters[NFKParameterSourceLanguage] as? String == language {
            parameters[NFKParameterSourceLanguage] = Self.targetLanguages.first { $0 != language }
        }
        return adopt(parameters, key: NFKParameterTargetLanguage, language: language)
    }

    /// The integer between `prefix` and `suffix` (or the next non-digit) in `message`.
    private static func integers(in message: String, after prefix: String, before suffix: String?) -> Int? {
        guard let start = message.range(of: prefix)?.upperBound else {
            return nil
        }
        let rest = message[start...]
        let digits = suffix.flatMap { rest.range(of: $0) }.map { rest[..<$0.lowerBound] } ?? rest.prefix { $0.isNumber }
        return Int(digits)
    }

    private mutating func adopt(_ parameters: [String: Any], key: String, language: String) -> Bool {
        request = NFKInferenceRequest(inputs: request.inputs, parameters: parameters)
        summary[key] = language
        return true
    }

    // The backend protocol's optional throwing methods, called through `?`, crash the compiler's IR
    // generation (Swift 6.4); a protocol of their own selectors reaches the same methods.
    private static func tokenCount(of request: NFKInferenceRequest, by backend: any NFKInferenceBackend) -> Int? {
        let object = backend as AnyObject
        guard object.responds(to: #selector(NFKMLXOptionalBackendMethods.tokenCount(for:))) else {
            return nil
        }
        return (try? unsafeBitCast(object, to: NFKMLXOptionalBackendMethods.self).tokenCount(for: request))?.intValue
    }

    func removeFiles() {
        for url in files {
            try? FileManager.default.removeItem(at: url)
        }
    }

    private static func text(words count: Int, offset: Int) -> String {
        (0 ..< count).map { words[($0 + offset) % words.count] }.joined(separator: " ")
    }

    private static func wave() -> Data {
        let rate = 44100
        let samples = (0 ..< rate * audioSeconds).map { index -> Float in
            let time = Float(index) / Float(rate)
            return 0.3 * sin(2 * .pi * 220 * time) + 0.1 * sin(2 * .pi * 1375 * time)
        }
        return NFKMLXWaveFile.data(samples: samples, sampleRate: rate)
    }

    private static func image(side: Int) -> CGImage {
        image(width: side, height: side, shift: 0)
    }

    /// An RGBA gradient, shifted by `shift` pixels so the frames of a clip differ.
    private static func image(width: Int, height: Int, shift: Int) -> CGImage {
        var pixels = [UInt8](repeating: 255, count: width * height * 4)
        for y in 0 ..< height {
            for x in 0 ..< width {
                let index = (y * width + x) * 4
                pixels[index] = UInt8((x + shift) * 255 / max(1, width - 1 + shift))
                pixels[index + 1] = UInt8(y * 255 / max(1, height - 1))
                pixels[index + 2] = UInt8(((x + y + shift) / 8) % 256)
            }
        }
        return cgImage(pixels, width: width, height: height)
    }

    /// White in the middle half, black around it.
    private static func mask(side: Int) -> CGImage {
        var pixels = [UInt8](repeating: 0, count: side * side * 4)
        for y in side / 4 ..< side * 3 / 4 {
            for x in side / 4 ..< side * 3 / 4 {
                let index = (y * side + x) * 4
                pixels[index] = 255
                pixels[index + 1] = 255
                pixels[index + 2] = 255
            }
        }
        for index in stride(from: 3, to: pixels.count, by: 4) {
            pixels[index] = 255
        }
        return cgImage(pixels, width: side, height: side)
    }

    private static func cgImage(_ pixels: [UInt8], width: Int, height: Int) -> CGImage {
        let provider = CGDataProvider(data: Data(pixels) as CFData)!
        return CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
                       bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
                       bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
    }
}

/// `NFKInferenceBackend`'s optional throwing methods, as a backend that implements them answers.
@objc private protocol NFKMLXOptionalBackendMethods {
    @objc(prepareWithError:)
    func prepare() throws
    @objc(tokenCountForRequest:error:)
    func tokenCount(for request: NFKInferenceRequest) throws -> NSNumber
}
