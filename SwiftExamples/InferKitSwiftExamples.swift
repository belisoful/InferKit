//
//  InferKitSwiftExamples.swift
//  InferKitSwiftExamples
//
//  The Swift half of the compiled examples. `Examples/NFKExamples.m` is the Objective-C half; both
//  mirror Docs/examples.md, so CI catches a documented snippet that stops compiling. This file also
//  pins the shape the API takes once Swift's ObjC importer has renamed things — `runInference(for:)`
//  rather than `runInferenceForRequest:error:`, throwing rather than an `NSError **` out-parameter —
//  which is what a Swift consumer actually writes.
//
//  The weight-free paths run and assert; the model and network backends are exercised at the contract
//  level (construct, read identity), so their example code keeps compiling without weights or a server.
//

import CoreText
import XCTest
import CoreML
import InferKit

final class InferKitSwiftExamples: XCTestCase {

    // MARK: The shared contract (Docs/examples.md: The shared contract)

    func testExampleTheSharedContractWithPassthrough() throws {
        let backend = NFKPassthroughBackend()
        backend.outputMap = [NFKOutputText: NFKInputPrompt]   // each output key maps to an input key

        let request = NFKInferenceRequest(inputs: [NFKInputPrompt: "Explain diffraction in one sentence."],
                                          parameters: [NFKParameterMaxTokens: 64],
                                          outputModality: .text)

        // The ObjC `NSError **` out-parameter imports as `throws`.
        let result = try backend.runInference(for: request)
        XCTAssertEqual(result.text, "Explain diffraction in one sentence.")
    }

    // MARK: Jobs, submit, completion (Docs/examples.md: Subsystems — Jobs)

    func testExampleSubmittingAJob() {
        let backend = NFKPassthroughBackend()
        backend.outputMap = [NFKOutputImage: NFKInputImage]
        let request = NFKInferenceRequest(inputs: [NFKInputImage: "plate"])

        let finished = expectation(description: "job finished")
        let job = NFKInferenceSubmit(backend, request, nil)
        job.completionHandler = { completed in
            XCTAssertEqual(completed.status, .succeeded)
            XCTAssertEqual(completed.result?.output(forKey: NFKOutputImage) as? String, "plate")
            finished.fulfill()
        }
        wait(for: [finished], timeout: 5)
    }

    // MARK: Tensor conversion (Docs/examples.md: Subsystems — Tensor conversion)

    func testExampleTensorConversionRoundTrip() {
        var interleavedRGBA: [Float] = [
            0.1, 0.2, 0.3, 1.0,   0.4, 0.5, 0.6, 1.0,
            0.7, 0.8, 0.9, 1.0,   0.15, 0.25, 0.35, 1.0,
        ]
        let spec = NFKTensorSpecMake(2, 2, 3)                 // defaults: CHW, RGBA order, mean 0 / scale 1

        var tensor = [Float](repeating: 0, count: Int(NFKTensorElementCount(spec)))
        NFKInterleavedToTensor(&interleavedRGBA, &tensor, spec)
        XCTAssertEqual(NFKTensorElementCount(spec), 12)

        var restored = [Float](repeating: 0, count: 16)
        NFKTensorToInterleaved(&tensor, &restored, spec)
        XCTAssertEqual(restored[0], 0.1, accuracy: 1e-5)
    }

    // MARK: MLMultiArray bridge (Docs/examples.md: Subsystems — Tensor conversion)

    func testExampleMLMultiArrayBridge() throws {
        var interleavedRGBA: [Float] = [
            0.1, 0.2, 0.3, 1.0,   0.4, 0.5, 0.6, 1.0,
            0.7, 0.8, 0.9, 1.0,   0.15, 0.25, 0.35, 1.0,
        ]
        let spec = NFKTensorSpecMake(2, 2, 3)

        let array = try XCTUnwrap(NFKMultiArrayFromInterleaved(&interleavedRGBA, spec, nil))
        XCTAssertEqual(array.shape, [1, 3, 2, 2])             // [1, C, H, W]

        var restored = [Float](repeating: 0, count: 16)
        XCTAssertTrue(NFKInterleavedFromMultiArray(array, &restored, spec))
        XCTAssertEqual(restored[0], 0.1, accuracy: 1e-5)
    }

    // MARK: Tokenizer (Docs/examples.md: Subsystems — Tokenizers)

    func testExampleTokenizerEncodeDecode() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let vocabulary = ["h": 0, "e": 1, "l": 2, "o": 3, "he": 4, "ll": 5, "hell": 6, "hello": 7]
        try JSONSerialization.data(withJSONObject: vocabulary)
            .write(to: directory.appendingPathComponent("vocab.json"))
        try "#version: 0.2\nh e\nl l\nhe ll\nhell o\n"
            .write(to: directory.appendingPathComponent("merges.txt"), atomically: true, encoding: .utf8)

        let manifest = ["tokenizer": ["type": "bpe-bytelevel", "vocab": "vocab.json", "merges": "merges.txt"]]
        // A `nullable instancetype` factory imports as a throwing initializer.
        let tokenizer = try NFKTokenizer(forManifest: manifest, directory: directory)
        XCTAssertEqual(tokenizer.encode("hello"), [7])
        XCTAssertEqual(tokenizer.decode([7]), "hello")
    }

    // Docs/examples.md: Subsystems — Tokenizers. The CLIP variant, which the Stable Diffusion text
    // encoders take: text lowercases, and a word's last piece carries "</w>".
    func testExampleCLIPTokenizerEncodeDecode() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let vocabulary = ["h": 0, "e": 1, "l": 2, "o</w>": 3, "he": 4, "ll": 5, "hell": 6, "hello</w>": 7]
        try JSONSerialization.data(withJSONObject: vocabulary)
            .write(to: directory.appendingPathComponent("vocab.json"))
        try "#version: 0.2\nh e\nl l\nhe ll\nhell o</w>\n"
            .write(to: directory.appendingPathComponent("merges.txt"), atomically: true, encoding: .utf8)

        let manifest = ["tokenizer": ["type": "clip", "vocab": "vocab.json", "merges": "merges.txt"]]
        let tokenizer = try NFKTokenizer(forManifest: manifest, directory: directory)
        XCTAssertEqual(tokenizer.encode("HELLO"), [7])              // lowercased first
        XCTAssertEqual(tokenizer.decode([7]), "hello")
    }

    // MARK: Hugging Face hub (Docs/examples.md: Subsystems — Hugging Face hub)

    func testExampleHuggingFaceHubURLResolution() {
        let hub = NFKHFHub(cacheDirectoryURL: nil)
        // The URL is optional: an invalid repo or path resolves to nil rather than a bad request.
        let remote = hub.remoteURL(forRepo: "Qwen/Qwen2.5-0.5B-Instruct", revision: nil,
                                   path: "tokenizer.json")
        XCTAssertEqual(remote?.absoluteString.contains("Qwen/Qwen2.5-0.5B-Instruct"), true)
    }

    func testExampleHuggingFaceHubCachePolicy() throws {
        let cache = FileManager.default.temporaryDirectory.appendingPathComponent("NFKSwiftExamplesHubCache")
        try FileManager.default.createDirectory(at: cache, withIntermediateDirectories: true)
        let savedLimit = NFKHFHub.defaultCacheSizeLimit
        defer {
            NFKHFHub.defaultCacheSizeLimit = savedLimit
            try? FileManager.default.removeItem(at: cache)
        }

        NFKHFHub.defaultCacheSizeLimit = 20 * 1024 * 1024 * 1024    // every new hub, companions' included
        let hub = NFKHFHub(cacheDirectoryURL: cache)
        hub.cacheSizeLimit = NFKHFHubUnlimitedCacheSize             // this hub only
        XCTAssertTrue(hub.excludesCacheFromBackup)                  // the default

        try NFKHFHub.setExcludedFromBackup(true, for: cache)
        XCTAssertTrue(NFKHFHub.isExcluded(fromBackup: cache))
        try hub.pinCachedRepo("Qwen/Qwen2.5-0.5B-Instruct", revision: nil)   // never evicted
        XCTAssertTrue(hub.isCachedRepoPinned("Qwen/Qwen2.5-0.5B-Instruct", revision: nil))
        try hub.trimCacheToSizeLimit()                              // after lowering a limit
        try hub.removeCachedRepo("Qwen/Qwen2.5-0.5B-Instruct", revision: nil)
        XCTAssertEqual(hub.cacheSize(), 0)
    }

    // MARK: Backend contracts (Docs/examples.md: Text → text, Image → image)

    func testExampleLocalLanguageBackendContract() throws {
        guard #available(macOS 15.0, iOS 18.0, tvOS 18.0, *) else {
            throw XCTSkip("the Core ML language backend needs macOS 15 / iOS 18")
        }
        let backend = NFKCoreMLLanguageBackend(modelDirectoryURL: URL(fileURLWithPath: "/models/qwen"))
        backend.computeUnits = .all
        XCTAssertEqual(backend.backendIdentifier, "coreml-llm")
        XCTAssertFalse(backend.isReady)                       // not ready until prepared
    }

    // Docs/examples.md: Constraining the Core ML backend's output. The same core keys serve the MLX backend.
    func testExampleConstrainingTheLocalLanguageBackend() {
        let request = NFKInferenceRequest(inputs: [NFKInputPrompt: "Describe Paris as JSON."],
                                          parameters: [NFKParameterOutputFormat: "json-object",
                                                       NFKParameterMaxTokens: 96, NFKParameterTemperature: 0])
        XCTAssertEqual(request.parameters[NFKParameterOutputFormat] as? String, "json-object")
        let pick = NFKInferenceRequest(inputs: [NFKInputPrompt: "Is the sky blue? Answer yes or no."],
                                       parameters: [NFKParameterChoices: ["yes", "no"]])
        XCTAssertEqual((pick.parameters[NFKParameterChoices] as? [String])?.count, 2)

        var tokens = (0 ..< 256).map { Data([UInt8($0)]) }
        tokens.append(Data())                                          // the end token has no bytes
        let vocabulary = NFKTokenVocabulary(tokens: tokens, endToken: 256)
        let json = NFKJSONConstraint(vocabulary: vocabulary, root: .object)
        XCTAssertTrue(json.acceptsText("{\"city\": \"Par"))
        XCTAssertTrue(json.isCompleteText("{\"city\": \"Paris\"}"))
        XCTAssertFalse(json.acceptsText("[1, 2]"))
    }

    func testExampleRemoteBackendContract() {
        let backend = NFKRemoteBackend(endpointURL: URL(string: "http://localhost:11434/v1/chat/completions"))
        backend.modelName = "llama3.2"
        XCTAssertEqual(backend.backendIdentifier, "remote")
        XCTAssertTrue(backend.isReady)                        // an endpoint is set
    }

    func testExampleTranscriptionBackendContract() {
        // Audio → text: point at an OpenAI-compatible transcriptions endpoint, set the model, and pass
        // audio under `NFKInputAudio` (an `NFKAudioAsset` or `Data`). The call itself needs the network.
        let backend = NFKRemoteTranscriptionBackend(
            endpointURL: URL(string: "https://api.example.com/v1/audio/transcriptions"))
        backend.modelName = "whisper-1"
        XCTAssertEqual(backend.backendIdentifier, "remote-transcription")
        XCTAssertTrue(backend.isReady)
    }

    func testExampleCoreMLImageBackendContract() {
        let backend = NFKCoreMLBackend(modelURL: URL(fileURLWithPath: "/models/style.mlpackage"))
        XCTAssertEqual(backend.backendIdentifier, "coreml")
        XCTAssertFalse(backend.isReady)
    }

    // MARK: Dynamic discovery (Docs/examples.md: Choosing a backend at runtime)

    func testExampleACapabilityIsUnavailableWithoutItsCompanion() {
        // The core resolves a provider class by name, so an engine that is not linked is simply
        // absent rather than a link error. This target links only the core.
        XCTAssertFalse(NFKDynamicBackend.isCapabilityAvailable(NFKCapabilityControlNet))
        XCTAssertThrowsError(try NFKDynamicBackend.backend(forCapability: NFKCapabilityControlNet))
    }

    // Remote providers from Swift: the same presets, with the importer's own naming.
    func testRemoteProviderPresets() {
        let ollama = NFKRemoteProvider(identifier: "ollama")
        XCTAssertNotNil(ollama)
        XCTAssertFalse(ollama!.requiresAPIKey, "a local server needs no key")

        let backend = NFKRemoteProvider.backend(for: ollama!, apiKey: nil, modelName: "llama3.2")
        XCTAssertTrue(backend.isReady)

        // Anthropic is the one provider that is not OpenAI-compatible.
        let claude = NFKRemoteProvider.backend(for: .anthropic, apiKey: "sk-ant-…",
                                               modelName: "claude-sonnet-4-5")
        XCTAssertEqual(claude.backendIdentifier, "anthropic-messages")
    }

    // Model discovery from Swift: the importer's naming for the list, the re-based preset, and the
    // unreachable-runner error. A picker is filled from `models(withAPIKey:)`, which throws.
    func testRemoteModelDiscovery() {
        let ollama = NFKRemoteProvider.ollama
        XCTAssertEqual(ollama.modelsURL.absoluteString, "http://localhost:11434/v1/models")

        let lan = ollama.withBaseURL(URL(string: "http://192.168.1.20:11434/v1")!)
        XCTAssertEqual(lan.identifier, "ollama")
        XCTAssertEqual(lan.endpointURL.absoluteString, "http://192.168.1.20:11434/v1/chat/completions")

        // Nothing listens on the discard port: the runner is "not running", which is
        // `NFKInferenceError.remoteUnreachable`, not an empty list.
        let stopped = ollama.withBaseURL(URL(string: "http://127.0.0.1:9/v1")!)
        let catalog = NFKRemoteModelCatalog(for: stopped, apiKey: nil)
        catalog.timeout = 5
        XCTAssertThrowsError(try catalog.models()) { error in
            XCTAssertEqual((error as NSError).domain, NFKInferenceErrorDomain)
            XCTAssertEqual((error as NSError).code, NFKInferenceError.error_RemoteUnreachable.rawValue)
        }

        let model = NFKRemoteModel(entry: ["id": "llama3.2:latest", "owned_by": "library"])
        XCTAssertEqual(model?.identifier, "llama3.2:latest")
        XCTAssertEqual(model?.ownedBy, "library")
    }

    // Discovery from Swift. The blocking calls import as methods rather than properties, and the
    // reachability check imports as a throwing call because it reports its failure through NSError.
    func testFindingWhicheverLocalRunnerIsRunning() {
        XCTAssertEqual(NFKRemoteProvider.localProviders.map(\.identifier),
                       ["ollama", "lmstudio", "llamacpp", "vllm"])

        let running = NFKRemoteProvider.firstAvailableLocalProvider()
        let backend = NFKRemoteProvider.backendForFirstAvailableLocalProvider(withModelName: "llama3.2")
        XCTAssertTrue(running == nil || backend != nil)
        for provider in NFKRemoteProvider.availableLocalProviders() {
            XCTAssertFalse(provider.requiresAPIKey)
        }

        // A list of the caller's own, for other ports or other machines on the network.
        let elsewhere = [NFKRemoteProvider.ollama.withBaseURL(URL(string: "http://127.0.0.1:9/v1")!)]
        XCTAssertTrue(NFKRemoteProvider.availableProviders(among: elsewhere, timeout: 2).isEmpty)
        XCTAssertNil(NFKRemoteProvider.firstAvailableProvider(among: elsewhere, timeout: 2))

        let stopped = NFKRemoteProvider.ollama.withBaseURL(URL(string: "http://127.0.0.1:9/v1")!)
        XCTAssertThrowsError(try stopped.isReachable(withAPIKey: nil, timeout: 2)) { error in
            XCTAssertEqual((error as NSError).code, NFKInferenceError.error_RemoteUnreachable.rawValue)
        }
    }

    // The completion-handler forms import as async calls. Their names carry a probe prefix: the
    // importer drops the handler from the name, which would otherwise take the blocking call's.
    func testDiscoveryAwaited() async {
        let providers = await NFKRemoteProvider.probeAvailableLocalProviders()
        XCTAssertLessThanOrEqual(providers.count, NFKRemoteProvider.localProviders.count)
        let first = await NFKRemoteProvider.probeFirstAvailableLocalProvider()
        XCTAssertTrue(first == nil || providers.contains(first!), "a provider compares by value")
    }

    // The embeddings backend and the local-runner surface from Swift: the importer's naming for the
    // factory, the optional actions as optional protocol requirements, and the Ollama list entry.
    func testRemoteEmbeddingsAndLocalRunners() {
        // The class factory imports as a failable initializer.
        let embedder = NFKRemoteEmbeddingBackend(for: .ollama, apiKey: nil, modelName: "nomic-embed-text")
        XCTAssertEqual(embedder?.endpointURL?.absoluteString, "http://localhost:11434/v1/embeddings")
        XCTAssertNil(NFKRemoteEmbeddingBackend(for: .anthropic, apiKey: "k", modelName: "m"))

        let runner = NFKRemoteProvider.ollama.localRunner
        XCTAssertEqual(runner?.nativeBaseURL.absoluteString, "http://localhost:11434")
        // The actions that change the machine are optional requirements; check before offering one.
        XCTAssertTrue(runner!.responds(to: #selector(NFKLocalModelRunner.pullModel(_:))), "Ollama downloads")
        XCTAssertFalse(NFKRemoteProvider.lmStudio.localRunner!.responds(to: #selector(NFKLocalModelRunner.pullModel(_:))),
                       "LM Studio's REST surface does not")
        XCTAssertNil(NFKRemoteProvider.llamaCpp.localRunner, "nothing beyond the OpenAI surface to adapt")

        let installed = NFKRemoteModel(entry: [
            "name": "gpt-oss:20b", "size": 13_793_441_244,
            "details": ["quantization_level": "MXFP4", "context_length": 131_072],
            "capabilities": ["completion", "tools", "thinking"],
        ])
        XCTAssertEqual(installed?.identifier, "gpt-oss:20b")
        XCTAssertEqual(installed?.contextLength, 131_072)
        XCTAssertEqual(installed?.capabilities, ["completion", "tools", "thinking"])
    }

    // Speech, image, and the codec from Swift: both class factories import as failable initializers,
    // and NFKImageCoding's methods take the importer's `for:` spelling.
    func testRemoteSpeechImageAndVision() {
        let speaker = NFKRemoteSpeechBackend(for: .openAI, apiKey: "sk-…", modelName: "gpt-4o-mini-tts", voice: "alloy")
        XCTAssertEqual(speaker?.endpointURL?.absoluteString, "https://api.openai.com/v1/audio/speech")
        XCTAssertEqual(speaker?.responseFormat, "wav")

        let painter = NFKRemoteImageBackend(for: .xAI, apiKey: "xai-…", modelName: "grok-2-image")
        XCTAssertEqual(painter?.generationsURL?.absoluteString, "https://api.x.ai/v1/images/generations")
        XCTAssertEqual(painter?.editsURL?.absoluteString, "https://api.x.ai/v1/images/edits")
        XCTAssertNil(NFKRemoteImageBackend(for: .anthropic, apiKey: "k", modelName: "m"))

        let context = CGContext(data: nil, width: 2, height: 2, bitsPerComponent: 8, bytesPerRow: 8,
                                space: CGColorSpaceCreateDeviceRGB(),
                                bitmapInfo: CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.premultipliedFirst.rawValue)!
        context.setFillColor(red: 0, green: 0, blue: 1, alpha: 1)
        context.fill(CGRect(x: 0, y: 0, width: 2, height: 2))
        let square = context.makeImage()!

        let png = NFKImageCoding.pngData(forImage: square)
        XCTAssertNotNil(png)
        XCTAssertTrue(NFKImageCoding.dataURL(forImage: square)?.hasPrefix("data:image/png;base64,") == true)
        let decoded = NFKImageCoding.pixelBuffer(withImageData: png!)   // owned: CF_RETURNS_RETAINED imports managed
        XCTAssertNotNil(decoded)
        XCTAssertEqual(CVPixelBufferGetPixelFormatType(decoded!), kCVPixelFormatType_32BGRA)

        // A vision question: the ordinary chat request with an image beside the prompt.
        let look = NFKInferenceRequest(inputs: [NFKInputPrompt: "What is in this frame?", NFKInputImage: square])
        XCTAssertNotNil(look.input(forKey: NFKInputImage))
    }

    // Streaming, tools, and the retry knobs from Swift: the job form imports as
    // `submitInferenceJob(for:)`, the typed accessor as `toolCalls`, the knobs as class properties.
    func testRemoteStreamingToolsAndRetries() {
        let weather: [String: Any] = ["name": "get_weather",
                                      "parameters": ["type": "object", "properties": ["city": ["type": "string"]]]]
        let ask = NFKInferenceRequest(inputs: [NFKInputPrompt: "Weather in Paris?"],
                                      parameters: [NFKParameterTools: [weather]], outputModality: .text)
        let turn = NFKInferenceResult(outputs: [NFKOutputToolCalls: [["id": "call_1", "name": "get_weather",
                                                                       "arguments": ["city": "Paris"]]]])
        XCTAssertEqual(turn.toolCalls?.first?["name"] as? String, "get_weather")

        let backend = NFKRemoteBackend(endpointURL: URL(string: "http://127.0.0.1:9/v1/chat/completions"))
        backend.timeout = 5
        let job = backend.submitInferenceJob(for: ask)
        XCTAssertNotNil(job.cancellationHandler)
        let ended = expectation(description: "stream ended")
        job.completionHandler = { _ in ended.fulfill() }
        wait(for: [ended], timeout: 10)
        XCTAssertEqual(job.status, .failed)
        XCTAssertEqual((job.error as NSError?)?.code, NFKInferenceError.error_RemoteUnreachable.rawValue)

        XCTAssertEqual(NFKRemoteTransport.retryAttempts, 2)
        XCTAssertEqual(NFKRemoteTransport.maximumRetryDelay, 8)
    }

    // The remaining directions from Swift: the importer's names for the three new services and the
    // transcription options, and the request keys for documents, clips, and a spoken reply.
    func testRemoteMediaModes() {
        let ears = NFKRemoteTranscriptionBackend(for: .groq, apiKey: "gsk_…", modelName: "whisper-large-v3")
        ears?.emitsTimestamps = true
        ears?.translates = true
        XCTAssertEqual(ears?.endpointURL?.absoluteString, "https://api.groq.com/openai/v1/audio/transcriptions")

        let director = NFKRemoteVideoBackend(for: .googleGemini, apiKey: "AIza…", modelName: "veo-3.1-generate-preview")
        XCTAssertEqual(director?.submitURL?.absoluteString, "https://generativelanguage.googleapis.com/v1beta/openai/videos")
        XCTAssertEqual(director?.apiStyle, .geminiSoraCompatible)
        let veo = NFKRemoteVideoBackend(for: .googleGemini, apiStyle: .geminiVeo, apiKey: "AIza…", modelName: "veo-3.1-generate-preview")
        XCTAssertEqual(veo?.submitURL?.absoluteString, "https://generativelanguage.googleapis.com/v1beta/models")
        let ranker = NFKRemoteReranker(for: .together, apiKey: "k", modelName: "Salesforce/Llama-Rank-V1")
        XCTAssertEqual(ranker?.endpointURL?.absoluteString, "https://api.together.xyz/v1/rerank")
        let gate = NFKRemoteModerationBackend(for: .mistral, apiKey: "k", modelName: "mistral-moderation-latest")
        XCTAssertEqual(gate?.endpointURL?.absoluteString, "https://api.mistral.ai/v1/moderations")

        let watched = NFKInferenceRequest(inputs: [NFKInputPrompt: "What happens?",
                                                   NFKInputVideo: NFKVideoAsset(fileURL: URL(fileURLWithPath: "/tmp/clip.mp4"))],
                                          parameters: [NFKParameterVideoFrameCount: 8, NFKParameterAudioOutput: ["voice": "alloy"]],
                                          outputModality: .text)
        XCTAssertNotNil(watched.input(forKey: NFKInputVideo))
        XCTAssertEqual((watched.parameter(forKey: NFKParameterAudioOutput) as? [String: String])?["voice"], "alloy")
    }

    // One contract key asks a reasoning model how hard to think; each backend maps the three levels
    // to its provider's control. What came back is the chain and what the turn cost.
    // Every hosted mode reaches through the same request type; the backend names the service.
    func testEveryHostedMode() {
        let responses = NFKRemoteResponsesBackend(for: .openAI, apiKey: "sk-…", modelName: "gpt-5.6-sol")
        XCTAssertEqual(responses?.endpointURL?.absoluteString, "https://api.openai.com/v1/responses")

        let gemini = NFKGeminiInteractionsBackend(apiKey: "AIza…", modelName: "gemini-3.1-flash-image")
        let draw = NFKInferenceRequest(inputs: [NFKInputPrompt: "a lighthouse"],
                                       parameters: [NFKParameterAspectRatio: "16:9"],
                                       outputModality: .image)
        XCTAssertTrue(gemini.isReady)
        XCTAssertEqual(draw.outputModality, .image)

        let infill = NFKRemoteCompletionBackend(for: .mistral, apiKey: "k", modelName: "codestral-latest")
        XCTAssertEqual(infill?.apiStyle, .mistral)
        let counter = NFKRemoteTokenCounter(for: .anthropic, apiKey: "k", modelName: "claude-opus-5-5")
        XCTAssertEqual(counter?.apiStyle, .anthropic)
        XCTAssertNotNil(NFKRemoteOCRBackend(for: .mistral, apiKey: "k", modelName: "mistral-ocr-latest"))
        XCTAssertNotNil(NFKRemoteClassifierBackend(for: .vLLM, apiKey: nil, modelName: "m"))

        let live = NFKRealtimeSession(for: .googleGemini, apiStyle: .geminiLive, apiKey: "AIza…", modelName: "gemini-3.8-live")
        live?.textHandler = { text, kind in _ = (text, kind) }
        XCTAssertEqual(live?.inputSampleRate, 16000)

        let files = NFKRemoteFileStore(for: .googleGemini, apiKey: "AIza…")
        XCTAssertEqual(files?.apiStyle, .gemini)
        let library = NFKRemoteRetrievalStore(for: .xAI, apiKey: "xai-…")
        library?.managementAPIKey = "xai-mgmt-…"
        XCTAssertEqual(library?.apiStyle, .XAI)
        let usage = NFKRemoteUsageReporter(for: .openAI, apiKey: "sk-admin-…")
        XCTAssertEqual(usage?.apiStyle, .openAI)
    }

    func testReasoningEffortAndWhatTheTurnCost() {
        let request = NFKInferenceRequest(inputs: [NFKInputPrompt: "Why is the sky blue?"],
                                          parameters: [NFKParameterReasoningEffort: NFKReasoningEffortDeep])
        XCTAssertEqual(request.parameter(forKey: NFKParameterReasoningEffort) as? String, "deep")

        let result = NFKInferenceResult(outputs: [
            NFKOutputText: "Shorter wavelengths scatter more.",
            NFKOutputReasoning: "Rayleigh scattering goes as the inverse fourth power.",
            NFKOutputUsage: [NFKUsageInputTokens: 11, NFKUsageOutputTokens: 7, NFKUsageReasoningTokens: 5],
        ])
        let usage = result.output(forKey: NFKOutputUsage) as? [String: Int]
        XCTAssertEqual(usage?[NFKUsageInputTokens], 11)
        XCTAssertEqual(usage?[NFKUsageReasoningTokens], 5)
        XCTAssertNil(usage?[NFKUsageCachedTokens], "an unreported count is absent, not zero")
        XCTAssertNotNil(result.output(forKey: NFKOutputReasoning))
    }

    // MARK: Where Core ML runs (Docs/examples.md: Where Core ML actually runs)

    // `MLComputeUnitsCPUOnly` is zero, so an unset property would move every model off the
    // accelerators. The backend initializes it explicitly.
    func testExampleTheCoreMLBackendDefaultsToAllComputeUnits() {
        let backend = NFKCoreMLBackend(modelURL: nil)
        XCTAssertEqual(backend.computeUnits, .all)

        backend.computeUnits = .cpuAndNeuralEngine
        XCTAssertEqual(backend.computeUnits, .cpuAndNeuralEngine)
    }

    // The plan reads a compiled model's placement without running it. This pins the shape the ObjC
    // importer gives the factory, and the availability answer, without needing a model.
    func testExampleTheComputePlanReportsWhetherItCanAnswer() throws {
        if #available(macOS 14.4, iOS 17.4, tvOS 17.4, *) {
            XCTAssertTrue(NFKComputePlan.isAvailable)
        } else {
            XCTAssertFalse(NFKComputePlan.isAvailable)
        }

        // A model that is not there fails rather than reporting an empty plan, because "nothing is on
        // the Neural Engine" and "cannot tell" are different answers.
        let absent = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("inferkit-example-absent.mlmodelc")
        XCTAssertThrowsError(try NFKComputePlan(forCompiledModelAt: absent, computeUnits: .all))
    }

    /// Docs/examples.md: Audio → notes and structure, through the importer's Swift spelling. The core
    /// value types need no weights, and `standardMIDIFileData()` is a method here rather than the
    /// property Objective-C sees.
    func testExampleTranscriptionAndStructureResults() {
        let notes = [NFKMIDINote(pitch: 64, startSeconds: 0.5, endSeconds: 1.0, velocity: 90,
                                 program: 4, percussion: false, pitchBend: [0.0, 0.25, 0.5]),
                     NFKMIDINote(pitch: 60, startSeconds: 0.0, endSeconds: 0.5, velocity: 100)]
        let sequence = NFKMIDISequence(notes: notes, tempoBPM: 96)
        XCTAssertEqual(sequence.notes.first?.pitch, 60)
        XCTAssertEqual(sequence.standardMIDIFileData().prefix(4), Data("MThd".utf8))

        let result = NFKInferenceResult(outputs: [
            NFKOutputMIDI: sequence,
            NFKOutputSegments: [NFKAudioSegment(startSeconds: 0, endSeconds: 15.2, label: "intro", confidence: 0.82)],
            NFKOutputBeats: [NFKMusicBeat(timeSeconds: 0.5, positionInBar: 1)],
            NFKOutputTempo: 120.0,
        ])
        XCTAssertEqual(result.midi?.notes.count, 2)
        XCTAssertEqual(result.segments?.first?.label, "intro")
        XCTAssertEqual(result.beats?.first?.isDownbeat, true)
    }

    // Typed decisions from Swift: the question factories import as static methods, the convenience
    // as `answers(forState:questions:)` throwing, and the result's typed accessor as `answers`.
    func testTypedDecisions() throws {
        let jev = NFKRemoteProvider.backend(for: .typeSafe, apiKey: "ts-…", modelName: "jev-latest") as? NFKTypeSafeBackend
        XCTAssertEqual(jev?.endpointURL?.absoluteString, "https://api.typesafe.ai/v1/systemone")

        let questions = [
            "department": NFKDecisionQuestion.choiceQuestion(withInstructions: "Which team should handle this?",
                                                             options: ["billing", "technical", "sales"]),
            "severity": NFKDecisionQuestion.scoreQuestion(withInstructions: "How severe?", levels: ["low", "medium", "high"]),
            "urgent": NFKDecisionQuestion.noulQuestion(withInstructions: "The customer needs an answer today."),
        ]
        let ask = NFKInferenceRequest(inputs: [NFKInputState: "Help! My payouts have been failing for 3 days.",
                                               NFKInputQuestions: questions])
        XCTAssertEqual((ask.input(forKey: NFKInputQuestions) as? [String: NFKDecisionQuestion])?["urgent"]?.type, .noul)
        XCTAssertEqual(questions["severity"]?.dictionaryRepresentation()["criteria"] as? [String], ["low", "medium", "high"])

        // What comes back (needs network): `try jev.answers(forState: state, questions: questions)`.
        let severity = NFKDecisionAnswer(dictionary: ["type": "score", "score": 1.6,
                                                      "legend": ["0": "low", "1": "medium", "2": "high"], "confidence": 0.61])
        XCTAssertEqual(severity?.score, 1.6)
        let decided = NFKInferenceResult(outputs: [NFKOutputAnswers: ["severity": severity!]])
        XCTAssertEqual(decided.answers?["severity"]?.legend?["2"], "high")
    }

    // MARK: Apple's own engines

    // Docs/examples.md: Reading the text in an image (Vision, no weights)
    func testExampleReadingTextInAnImage() throws {
        let image = try XCTUnwrap(Self.imageOfText("INFERKIT"))
        let backend = NFKVisionTextBackend()   // +backend imports as init()
        // Swift gets the throwing form of runInferenceForRequest:error:.
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: image]))
        XCTAssertEqual(result.text, "INFERKIT")
        XCTAssertEqual(result.detections?.first?.label, "INFERKIT")
    }

    // Docs/examples.md: Apple's frame processors (VideoToolbox)
    func testExampleUpscalingAFrameWithVideoToolbox() {
        let backend = NFKVideoToolboxBackend(task: .superResolution)
        guard backend.isReady else {
            return   // the processors need Apple silicon and a recent OS
        }
        // 0 takes the smallest factor the machine offers, and a factor it lacks is refused by name.
        XCTAssertEqual(backend.scaleFactor, 0)
    }

    // Docs/examples.md: Transcribing with Apple's recognizer
    func testExampleTranscribingWithApplesRecognizer() {
        let backend = NFKSpeechRecognitionBackend()
        backend.requiresOnDeviceRecognition = true
        backend.locale = Locale(identifier: "en-US")
        XCTAssertEqual(backend.locale.identifier, "en-US")
        if !NFKSpeechRecognitionBackend.isAuthorized {
            XCTAssertFalse(backend.isReady)
        }
    }

    // Docs/examples.md: Apple's own engines — speaking text, and classifying what was spoken
    func testExampleSpeakingAndClassifying() throws {
        let voice = NFKSpeechSynthesisBackend()
        voice.language = "en-US"
        let spoken = try voice.runInference(
            for: NFKInferenceRequest(inputs: [NFKInputPrompt: "The quick brown fox jumps over the lazy dog."]))
        let asset = try XCTUnwrap(spoken.output(forKey: NFKOutputAudio) as? NFKAudioAsset)

        let sounds = NFKSoundClassificationBackend()
        sounds.minimumConfidence = 0.05
        let heard = try sounds.runInference(for: NFKInferenceRequest(inputs: [NFKInputAudio: asset]))
        XCTAssertGreaterThan(try XCTUnwrap(heard.segments).count, 0)
    }

    // Docs/examples.md: Apple's own engines — word and sentence vectors
    func testExampleEmbeddingTextWithNaturalLanguage() throws {
        let result = try NFKTextEmbeddingBackend().runInference(
            for: NFKInferenceRequest(inputs: [NFKInputPrompt: "a lighthouse at dawn"]))
        XCTAssertGreaterThan(try XCTUnwrap(result.embedding).count, 0)
    }

    private static func imageOfText(_ text: String) -> CGImage? {
        let space = CGColorSpaceCreateDeviceRGB()
        guard let context = CGContext(data: nil, width: 600, height: 160, bitsPerComponent: 8, bytesPerRow: 0,
                                      space: space,
                                      bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                                          | CGBitmapInfo.byteOrder32Little.rawValue) else {
            return nil
        }
        context.setFillColor(red: 1, green: 1, blue: 1, alpha: 1)
        context.fill(CGRect(x: 0, y: 0, width: 600, height: 160))
        let font = CTFontCreateWithName("Helvetica" as CFString, 72, nil)
        let line = CTLineCreateWithAttributedString(NSAttributedString(string: text,
                                                                       attributes: [.font: font]))
        context.textPosition = CGPoint(x: 24, y: 48)
        CTLineDraw(line, context)
        return context.makeImage()
    }
}
