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

    // MARK: Freeing a model's memory (Docs/examples.md: The shared contract — Freeing a model's memory)

    func testExampleFreeingAModelsMemory() {
        let backend = NFKPassthroughBackend()
        XCTAssertEqual(NFKInferenceResidentByteCount(backend), 0)   // 0 when the backend measures nothing
        XCTAssertEqual(NFKInferencePeakByteCount(backend), 0)

        NFKInferenceUnload(backend)                    // the passthrough holds no model, so it keeps running
        XCTAssertTrue(NFKInferenceIsLoaded(backend))
        XCTAssertTrue(NFKInferencePrepare(backend, nil))
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

    // MARK: Library and API version (Docs/examples.md: Subsystems — Library and API version)

    func testExampleLibraryAndAPIVersion() {
        let supported = NFKInferKit.apiMajorVersion == Int(NFK_API_VERSION_MAJOR)
            && NFKInferKit.apiMinorVersion >= Int(NFK_API_VERSION_MINOR)
        print("InferKit \(NFKInferKit.releaseVersion), API \(NFKInferKit.apiVersion)")
        XCTAssertTrue(supported)
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
        let cache = FileManager.default.temporaryDirectory.appendingPathComponent("NFKSwiftExamplesHubCache-\(UUID().uuidString)")
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

    // A URL source whose file is on this machine stands in for a Hugging Face repository, so the
    // example runs without a network.
    func testExampleDownloadingWithProgressThenRemoving() async throws {
        let scratch = FileManager.default.temporaryDirectory.appendingPathComponent("NFKSwiftExamplesDownloads-\(UUID().uuidString)")
        let cache = scratch.appendingPathComponent("cache")
        let release = scratch.appendingPathComponent("model.bin")
        try FileManager.default.createDirectory(at: cache, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: scratch) }
        try Data(count: 256 * 1024).write(to: release)
        let source = NFKModelSource(kind: NFKModelSourceKindURL, repository: nil, revision: nil, url: release,
                                    layout: NFKModelSourceLayoutFile, files: nil, requiresConversion: false,
                                    extensions: nil)
        let hub = NFKHFHub(cacheDirectoryURL: cache)

        // What would this download, and how large?
        let plan = try await hub.planDownload(source: source)
        XCTAssertEqual(plan.remainingByteCount, 256 * 1024)

        // Download it with progress; `download.cancel()` stops it and leaves no partial file.
        let localURL: URL? = try await withCheckedThrowingContinuation { continuation in
            hub.download(plan: plan, progressHandler: { progress in
                _ = progress.fractionCompleted                  // a progress bar
                _ = progress.currentFile?.path                  // "model.bin"
            }, completionHandler: { localURL, error in
                if let error {
                    continuation.resume(throwing: error)
                } else {
                    continuation.resume(returning: localURL)
                }
            })
        }
        XCTAssertEqual(localURL?.lastPathComponent, "model.bin")

        // "Downloaded · 256 KB" and a Delete button.
        let status = hub.downloadStatus(source: source)
        XCTAssertEqual(status.presence.rawValue, NFKDownloadPresence.complete.rawValue)
        _ = ByteCountFormatter.string(fromByteCount: status.byteCount, countStyle: .file)
        let removal = try hub.removalPlan(source: source)
        let removed = try await hub.removeFiles(in: removal)
        XCTAssertEqual(removed.freedByteCount, removal.freedByteCount)
        XCTAssertEqual(hub.downloadStatus(source: source).presence.rawValue, NFKDownloadPresence.absent.rawValue)
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

    // MARK: Model catalog (Docs/examples.md: Model catalog)

    func testExampleModelCatalogFromSwift() throws {
        let catalog = NFKModelCatalog()
        let upscaler = NFKModelDescriptor(identifier: "example.upscaler", displayName: "Example Upscaler",
                                          family: "example", summary: nil, engine: NFKModelEnginePlugin,
                                          registryName: nil, providerClassName: "ExampleUpscalerProvider",
                                          tasks: [NFKModelTaskUpscaling], inputModalities: nil, outputModalities: nil,
                                          inputs: [NFKModelPort(key: NFKInputImage, required: true)],
                                          outputs: [NFKModelPort.outputPort(withKey: NFKOutputImage, required: true)],
                                          options: [NFKModelOption(key: NFKParameterSeed)],
                                          variants: [NFKModelVariant(identifier: "x4", displayName: "×4")],
                                          license: nil, requirements: nil, residencies: [NFKModelResidencyResident],
                                          standIn: false, introducedVersion: nil, extensions: nil)
        catalog.addDescriptor(upscaler)

        // The manifest round trip: the `NSError **` out-parameters import as `throws`.
        let manifest = try catalog.manifestData()
        let reread = NFKModelCatalog()
        try reread.addDescriptors(fromManifestData: manifest)
        XCTAssertEqual(reread.descriptor(withIdentifier: "example.upscaler"), upscaler)
        XCTAssertEqual(reread.descriptors(forTask: NFKModelTaskUpscaling).map(\.identifier), ["example.upscaler"])
        XCTAssertEqual(upscaler.defaultVariant?.identifier, "x4")
        XCTAssertEqual(upscaler.options.first?.kind, NFKModelOptionKindInteger)
        XCTAssertEqual(NFKModelValueTypeForKey(NFKInputPrompt), NFKModelValueTypeText)

        // A variant's own license overrides its model's; a file can come from another repository.
        let vocoder = NFKModelFile(path: "vocoder.safetensors", byteCount: nil, sha256: nil,
                                   role: NFKModelFileRoleWeights, optional: false, repository: "org/vocoder",
                                   revision: nil, url: nil, extensions: nil)
        let pair = NFKModelVariant(identifier: "en-de", displayName: nil, isDefault: true, registryName: nil,
                                   source: nil, downloadByteCount: nil, residentByteCount: nil, peakByteCount: nil,
                                   parameterCount: nil, precision: nil, minimumMemoryByteCount: nil,
                                   license: NFKModelLicense(identifier: "CC-BY-4.0", name: "CC BY 4.0", url: nil),
                                   extensions: nil)
        XCTAssertTrue(vocoder.overridesSource)
        XCTAssertEqual(upscaler.license(for: pair)?.identifier, "CC-BY-4.0")
        XCTAssertEqual(NFKModelTaskDecision, "decision")

        let described = NFKModelDescriptor(identifier: "app.echo", derivedFrom: NFKPassthroughBackend())
        XCTAssertEqual(described.engine, NFKModelEnginePlugin)
        XCTAssertEqual(NFKModelCatalog.schemaVersion, 2)
        XCTAssertTrue(NFKModelCatalog.providerClassNames.contains("NFKMLXModelDescriptors"))
        _ = NFKModelCatalog.shared
    }

    // Memory estimates, constraints, and model names from Swift.
    func testExampleModelMemoryFromSwift() throws {
        let decoder = NFKModelStageFacts(identifier: "decoder",
                                         byteCountByPrecision: [NFKModelPrecisionCheckpoint: 2_000_000_000])
        let facts = NFKModelMemoryFacts(stages: [decoder], keyValueByteCountPerToken: 114_688,
                                        boundedKeyValueByteCount: nil, maximumContextLength: 32_768,
                                        vocabularySize: nil, loadOverheadByteCount: 0, basis: nil, extensions: nil)
        let variant = NFKModelVariant(identifier: "1b", displayName: nil, isDefault: true, registryName: nil,
                                      source: nil, downloadByteCount: nil, residentByteCount: nil, peakByteCount: nil,
                                      parameterCount: 1_000_000_000, precision: NFKModelPrecisionBFloat16,
                                      minimumMemoryByteCount: nil, license: nil, memoryFacts: facts, extensions: nil)
        let context = NFKModelOption(key: NFKModelLoadOptionContextLength, kind: NFKModelOptionKindInteger,
                                     title: "Context length", help: nil, defaultValue: 8192, minimum: 512,
                                     maximum: 32_768, step: 512, choices: nil, stage: NFKModelOptionStageLoad,
                                     group: nil, extensions: nil)
        let memoryModel = NFKModelMemoryModel(precisionOptionKey: nil, quantizationBitsOptionKey: nil,
                                              quantizationGroupSizeOptionKey: nil,
                                              contextLengthOptionKey: NFKModelLoadOptionContextLength,
                                              cacheBitsOptionKey: nil, cacheGroupSizeOptionKey: nil, draftOptionKey: nil,
                                              draftBitsOptionKey: nil, residencyOptionKey: nil, cacheStage: nil,
                                              draftStage: nil, stagePrecisions: nil, extensions: nil)
        let chat = NFKModelDescriptor(identifier: "example.chat", displayName: nil, family: "example", summary: nil,
                                      engine: NFKModelEnginePlugin, registryName: nil, providerClassName: nil,
                                      tasks: nil, inputModalities: nil, outputModalities: nil, inputs: nil,
                                      outputs: nil, options: [context], variants: [variant], license: nil,
                                      requirements: nil, residencies: nil, memoryModel: memoryModel, presets: nil,
                                      standIn: false, introducedVersion: nil, extensions: nil)

        // The `NSError **` estimate imports as `throws`; an option value may be a string.
        let budget = NFKModelMemoryBudget(workingSetByteCount: 32 << 30)
        let estimate = try chat.estimatedMemory(forOptions: [NFKModelLoadOptionContextLength: "4096"],
                                                memoryBudget: budget)
        XCTAssertEqual(estimate.peakByteCount, 2_000_000_000 + 114_688 * 4096)
        XCTAssertTrue(estimate.fits)
        let constraints = chat.constraints(for: budget)
        XCTAssertTrue(constraints.anyCombinationFits)
        XCTAssertEqual(NFKModelToolName(NFKModelPresetModelName("example.chat", "fast")), "example_chat__fast")
    }

    // The budget a catalog builds against, pinned to a known host and restored with nil (the property is
    // null_resettable, so Swift reads it as an implicitly unwrapped optional).
    func testExamplePinningTheBudgetACatalogBuildsAgainstFromSwift() {
        let catalog = NFKModelCatalog()
        catalog.memoryBudget = NFKModelMemoryBudget(workingSetByteCount: 48 << 30)
        XCTAssertEqual(catalog.memoryBudget.reserveByteCount, NFKModelMemoryBudget.maximumReserveByteCount)
        XCTAssertEqual(NFKModelMemoryBudget.reserveByteCount(forBudgetByteCount: 6 << 30), 3 << 30)
        catalog.memoryBudget = nil
        XCTAssertEqual(catalog.memoryBudget, NFKModelMemoryBudget.current())
    }

    // Resolving a Hugging Face repository from Swift (Docs/examples.md: Model catalog — Resolving a Hugging
    // Face repository): the route is NFKModelResolution.Route, and the synchronous form throws. A shipped
    // variant's repository resolves with no network request.
    func testExampleResolvingAHuggingFaceRepositoryFromSwift() throws {
        let manifest = """
            { "schemaVersion": 2, "kind": "inferkit.model-catalog", "models": [ {
              "id": "example.chat-1b", "family": "example", "engine": "example-engine", "modelTypes": ["example"],
              "variants": [{ "id": "1b", "isDefault": true, "residentBytes": 2000000000,
                             "source": { "kind": "huggingface", "repository": "example/chat-1b", "layout": "directory" } }]
            } ] }
            """
        let catalog = NFKModelCatalog()
        try catalog.addDescriptors(fromManifestData: Data(manifest.utf8))
        let resolver = NFKModelResolver(catalog: catalog, hub: nil)
        resolver.allowsUnverifiedRoutes = true
        let resolution = try resolver.resolution(forRepository: "example/chat-1b", revision: nil)
        XCTAssertEqual(resolution.route, .catalog)
        XCTAssertEqual(resolution.route.name, "catalog")
        XCTAssertEqual(resolution.verification, .measured)
        let restored = try NFKModelDescriptor(dictionary: try XCTUnwrap(resolution.descriptor).dictionaryRepresentation())
        XCTAssertEqual(restored.modelTypes, ["example"])
    }

    #if os(macOS)
    // Plugins from Swift (Docs/examples.md: Model catalog — Plugins): a folder with one executable plugin
    // whose program never runs, listed in a registry of its own catalog.
    func testExampleListingPluginsFromSwift() throws {
        let files = FileManager.default
        let folder = files.temporaryDirectory.appendingPathComponent("swift-plugins-\(UUID().uuidString)", isDirectory: true)
        let contents = folder.appendingPathComponent("Echo.inferkitplugin/Contents", isDirectory: true)
        let directoryMode: [FileAttributeKey: Any] = [.posixPermissions: 0o755]
        try files.createDirectory(at: contents.appendingPathComponent("MacOS"), withIntermediateDirectories: true,
                                  attributes: directoryMode)
        try files.createDirectory(at: contents.appendingPathComponent("Resources"), withIntermediateDirectories: true,
                                  attributes: directoryMode)
        addTeardownBlock { try? FileManager.default.removeItem(at: folder) }
        let info: NSDictionary = ["CFBundleIdentifier": "com.example.swift-echo", "CFBundleShortVersionString": "1.0.0",
                                  "CFBundleExecutable": "Echo", "CFBundlePackageType": "BNDL"]
        try info.write(to: contents.appendingPathComponent("Info.plist"))
        let manifest = """
            { "schemaVersion": 1, "kind": "inferkit.plugin", "id": "com.example.swift-echo", "version": "1.0.0",
              "compatibility": { "inferkitAPI": "1.0" }, "entry": { "kind": "executable", "executable": "Echo" },
              "options": [{ "key": "greeting", "kind": "text", "default": "hello" }],
              "models": [{ "id": "echo", "family": "example-echo",
                           "inputs": [{ "key": "prompt", "required": true }], "outputs": [{ "key": "text" }] }] }
            """
        try manifest.write(to: contents.appendingPathComponent("Resources/inferkit-plugin.json"), atomically: true, encoding: .utf8)
        let program = contents.appendingPathComponent("MacOS/Echo")
        try "#!/bin/sh\nexit 0\n".write(to: program, atomically: true, encoding: .utf8)
        try files.setAttributes([.posixPermissions: 0o755], ofItemAtPath: program.path)

        let catalog = NFKModelCatalog()
        catalog.memoryBudget = NFKModelMemoryBudget(workingSetByteCount: 64 << 30)  // not this machine's
        let registry = NFKPluginRegistry(catalog: catalog)
        var errors: NSArray?
        let added = registry.addPlugins(inDirectory: folder, errors: &errors)
        XCTAssertEqual(added.map(\.identifier), ["com.example.swift-echo"])
        XCTAssertEqual(errors?.count ?? 0, 0)
        // The catalog's build entry throws in Swift, as the registry's own entry does.
        let echo = try registry.catalog.backend(forModelIdentifier: "com.example.swift-echo/echo", options: nil)
        XCTAssertEqual(echo.backendIdentifier, "plugin:com.example.swift-echo/echo")
        let direct = try registry.backend(forModelIdentifier: "com.example.swift-echo/echo", options: nil)
        XCTAssertEqual(direct.isolation, .process)
        XCTAssertEqual(direct.process?.state, .notRunning)
        XCTAssertFalse(registry.isTrustedPluginIdentifier("com.example.swift-echo"))
        try registry.setOptionValues(["greeting": "hi"], forPluginIdentifier: "com.example.swift-echo")
        XCTAssertEqual(registry.optionValues(forPluginIdentifier: "com.example.swift-echo") as? [String: String], ["greeting": "hi"])
        XCTAssertThrowsError(try registry.setOptionValues(["volume": 3], forPluginIdentifier: "com.example.swift-echo"))
        let reloaded = expectation(description: "reloaded")
        registry.reloadPlugin(withIdentifier: "com.example.swift-echo") { error in
            XCTAssertNil(error)
            reloaded.fulfill()
        }
        wait(for: [reloaded], timeout: 10)
        registry.sharedMemoryPrefix = nil
        XCTAssertFalse(NFKPluginInProcessWarningFullAccess.isEmpty)
        _ = NFKPluginRegistry.shared
    }
    #endif

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

    // Gemini's generateContent API from Swift: the factory on the gemini preset.
    func testGeminiGenerateContent() {
        let gemini = NFKGeminiGenerateContentBackend(for: .googleGemini, apiKey: "AIza…", modelName: "gemini-3.8-flash")
        XCTAssertEqual(gemini?.endpointURL.absoluteString, "https://generativelanguage.googleapis.com/v1beta/models")
        XCTAssertEqual(gemini?.backendIdentifier, "gemini-generatecontent")
        XCTAssertNil(NFKGeminiGenerateContentBackend(for: .openAI, apiKey: "k", modelName: "m"))
    }

    // OpenCode from Swift: one preset, a backend per model's route, and a hand-made route.
    func testOpenCodeGateway() {
        let zen = NFKRemoteProvider.openCodeZen
        let gpt = NFKRemoteProvider.backend(for: zen, apiKey: "sk-…", modelName: "gpt-5.5")
        XCTAssertEqual(gpt.backendIdentifier, "remote-responses")
        XCTAssertEqual(zen.apiStyle(forModelName: "claude-sonnet-5-5"), .anthropicMessages)
        XCTAssertEqual(zen.apiStyle(forModelName: "gemini-3.5-flash"), .geminiGenerateContent)

        let gemini = NFKRemoteProvider.backend(for: zen, apiKey: "sk-…", modelName: "gemini-3.5-flash")
        XCTAssertEqual(gemini.backendIdentifier, "gemini-generatecontent")

        let routed = zen.withAPIStyle(.anthropicMessages, forModelName: "claude-next")
        XCTAssertEqual(routed.apiStyle(forModelName: "claude-next"), .anthropicMessages)
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
                       ["ollama", "lmstudio", "llamacpp", "vllm", "inferkit"])

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

    // MARK: Serving (Docs/examples.md: Serving a model to other machines)

    // The server from Swift: start() throws, addBackend keeps its label, and the discovery's awaited
    // form is probeInferKitServers(timeout:) so it does not take the blocking call's name.
    func testServingAModelToOtherMachines() async throws {
        let server = NFKInferenceServer()
        server.port = 0
        server.loopbackOnly = true
        server.addBackend(NFKPassthroughBackend(), forModelName: "echo")
        try server.start()
        defer { server.stop() }

        let provider = NFKRemoteProvider.inferKit.withBaseURL(try XCTUnwrap(server.localBaseURL))
        let echo = NFKRemoteProvider.backend(for: provider, apiKey: nil, modelName: "echo")
        let result = try echo.runInference(for: NFKInferenceRequest(inputs: [NFKInputPrompt: "hello"]))
        XCTAssertEqual(result.output(forKey: NFKInputPrompt) as? String, "hello")

        let found = await NFKRemoteProvider.probeInferKitServers(timeout: 0.2)
        XCTAssertTrue(found.allSatisfy { $0.apiStyle == .inferKit })
    }

    // A provider and a builder written in main-actor code: both import as Sendable, so neither closure
    // nor conformance is main-actor isolated, and the server's build on its own queue does not trap.
    // Tools/swift-isolation-probe compiles the same calls as Swift 6 with main-actor default isolation.
    final class EchoBuilder: NSObject, NFKServedBackendBuilder {
        func backend(forModelName modelName: String) throws -> any NFKInferenceBackend {
            guard modelName == "built" else {
                throw NSError(domain: NFKInferenceServerErrorDomain, code: NFKInferenceServerError.modelNotFound.rawValue)
            }
            return NFKPassthroughBackend()
        }
    }

    @MainActor
    func testServingModelsBuiltOnDemandFromMainActorCode() {
        let server = NFKInferenceServer()
        server.residencyCheckInterval = 0
        server.addBackendProvider({ _ in NFKPassthroughBackend() }, forModelName: "provided")
        server.addBackendBuilder(EchoBuilder(), forModelName: "built")
        server.addBackendBuilder(EchoBuilder(), forModelName: "refused")

        let loaded = expectation(description: "both models built")
        loaded.expectedFulfillmentCount = 2
        for name in ["provided", "built"] {
            server.loadModelNamed(name) { error in
                XCTAssertNil(error)
                loaded.fulfill()
            }
        }
        let refused = expectation(description: "the builder's error reaches the handler")
        server.loadModelNamed("refused") { error in
            XCTAssertEqual((error as NSError?)?.code, NFKInferenceServerError.modelNotFound.rawValue)
            refused.fulfill()
        }
        wait(for: [loaded, refused], timeout: 10)
        XCTAssertEqual(server.residency(forModelName: "provided"), .resident)
        XCTAssertEqual(server.residency(forModelName: "built"), .resident)
        XCTAssertEqual(server.residency(forModelName: "refused"), .notLoaded)
    }

    // Observation, scoped keys, and app routes from Swift: the delegate's methods import as
    // inferenceServer(_:didChangeModel:), addRoute(pathPrefix:handler:) throws, the MCP settings read
    // mcpPingInterval and mcpSessions, and currentStatus() is the status route read in process.
    // A catalog model hosted under a preset, a suggested preset, and an alias. The add methods take
    // NSError ** and import as throwing, and the native client's catalog read imports as fetchModelCatalog().
    final class PassthroughBuilder: NSObject, NFKModelBackendBuilding {
        func backend(for request: NFKModelBuildRequest) throws -> any NFKInferenceBackend {
            let backend = NFKPassthroughBackend()
            try request.applyPropertyOptions(toBackend: backend)
            return backend
        }
    }

    func testServingACatalogModelUnderPresetsFromSwift() throws {
        let catalog = NFKModelCatalog()
        try catalog.addDescriptors(fromManifestData: Data("""
            { "schemaVersion": 1, "kind": "inferkit.model-catalog", "models": [ {
              "id": "example.voice", "family": "example", "engine": "swift-example-engine",
              "options": [ { "key": "style", "kind": "text", "stage": "load", "default": "plain" },
                           { "key": "speed", "kind": "number", "default": 1.0 } ],
              "presets": [ { "id": "brisk", "suggested": true, "requestDefaults": { "speed": 1.5 } } ] } ] }
            """.utf8))
        NFKModelCatalog.registerBackendBuilder(PassthroughBuilder(), forEngine: "swift-example-engine")
        defer { NFKModelCatalog.registerBackendBuilder(nil, forEngine: "swift-example-engine") }

        catalog.memoryBudget = NFKModelMemoryBudget(workingSetByteCount: 64 << 30)  // the host the server plans for
        let server = NFKInferenceServer()
        server.port = 0
        server.loopbackOnly = true
        server.modelCatalog = catalog
        try server.addModel(identifier: "example.voice", options: nil, catalog: nil, forModelName: "voice")
        let loud = NFKModelPreset(identifier: "loud", title: nil, help: nil, baseModelIdentifier: nil,
                                  loadOptions: ["style": "loud"], requestDefaults: nil, suggested: false, extensions: nil)
        try server.addPreset(loud, forBaseModelName: "voice")
        try server.addAlias("speaker", forModelName: "voice", requestDefaults: ["speed": 0.8])
        XCTAssertEqual(server.modelNames, ["voice", "voice@loud"])
        XCTAssertEqual(server.aliasNames, ["speaker"])
        let tool = try server.addTool(forModelName: "voice@loud", outputModality: .text, purpose: nil)
        XCTAssertEqual(tool.name, "voice__loud")
        XCTAssertEqual(server.modelName(forToolName: "voice__loud"), "voice@loud")
        try server.start()
        defer { server.stop() }

        let client = NFKRemoteInferKitBackend(baseURL: try XCTUnwrap(server.localBaseURL))
        client.modelName = "voice@brisk"
        let result = try client.runInference(for: NFKInferenceRequest(inputs: [NFKInputPrompt: "hi"]))
        XCTAssertEqual(result.output(forKey: NFKInputPrompt) as? String, "hi")
        XCTAssertEqual(client.lastReportedLoad?.presetIdentifier, "brisk")
        let served = try client.fetchModelCatalog()
        XCTAssertNotNil(served.descriptor(withIdentifier: "example.voice"))
    }

    final class ModelObserver: NSObject, NFKInferenceServerDelegate {
        let loaded: XCTestExpectation
        init(loaded: XCTestExpectation) {
            self.loaded = loaded
        }
        func inferenceServer(_ server: NFKInferenceServer, didChangeModel event: NFKServerModelEvent) {
            if event.kind == .loaded && event.reason == .explicit {
                loaded.fulfill()
            }
        }
    }

    func testObservingTheServerFromSwift() throws {
        let server = NFKInferenceServer()
        server.port = 0
        server.loopbackOnly = true
        server.residencyCheckInterval = 0
        let observer = ModelObserver(loaded: expectation(description: "the model loaded"))
        server.delegate = observer
        server.addAPIKey("dashboard-key", named: "dashboard", scopes: [.status])
        server.loopbackExemptScopes = [.inference, .mcp, .appRoutes]
        try server.addRoute(pathPrefix: "/myapp/") { request, response in
            response.sendStatus(200, jsonObject: ["path": request.path], headers: nil)
        }
        server.mcpPingInterval = 15
        try server.start()
        defer { server.stop() }

        server.addBackendProvider({ _ in NFKPassthroughBackend() }, forModelName: "echo")
        server.loadModelNamed("echo", completionHandler: nil)
        wait(for: [observer.loaded], timeout: 10)
        XCTAssertEqual(server.currentStatus().modelNamed("echo")?.residency, .resident)
        XCTAssertEqual(server.apiKeyNames, ["dashboard"])
        XCTAssertEqual(server.routePathPrefixes, ["/myapp/"])
        XCTAssertTrue(server.mcpSessions.isEmpty)
    }

    // MARK: Models as tools (Docs/examples.md: Models as tools)

    // A Swift backend adopts the protocol by its imported names; this one calls the tool it is
    // offered on its first turn and answers from the tool's result on its second.
    final class ToolCallingModel: NSObject, NFKInferenceBackend {
        var isReady: Bool { true }
        var backendIdentifier: String { "swift-example-remote" }
        func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
            if let last = request.messages?.last, last["role"] as? String == "tool" {
                return NFKInferenceResult(outputs: [NFKOutputText: "Local: \(last["content"] as? String ?? "")"])
            }
            let tool = (request.parameter(forKey: NFKParameterTools) as? [[String: Any]])?.first?["name"] as? String ?? ""
            return NFKInferenceResult(outputs: [NFKOutputToolCalls: [["id": "call_1", "name": tool,
                                                                      "arguments": [NFKInputPrompt: "Say hi."]]]])
        }
    }

    // The importer's names: the class factories are initializers, a handler block takes the error
    // out-parameter as NSErrorPointer, and invokeWithArguments: is invoke(withArguments:).
    func testModelsAsTools() throws {
        let local = NFKPassthroughBackend()
        let askLocal = NFKBackendTool(backend: local, name: "ask_local_model", outputModality: .text,
                                      purpose: "the text must stay on this machine")
        XCTAssertTrue(askLocal.toolDescription.contains("Use it when the text must stay on this machine."))
        let weather = NFKInferenceTool(name: "get_weather",
                                       description: "Current weather in a city. Returns the temperature and sky.",
                                       parameters: ["type": "object",
                                                    "properties": ["city": ["type": "string"]],
                                                    "required": ["city"]],
                                       handler: { arguments, _ in
                                           NFKToolResult(text: "18 C in \(arguments["city"] as? String ?? "")")
                                       })
        XCTAssertTrue(weather.invoke(withArguments: [:]).isError)

        let loop = NFKToolLoop(backend: ToolCallingModel(), tools: [weather])
        loop.maximumTurns = 4
        loop.approvalHandler = { name, arguments in true }
        let result = try loop.runInference(for: NFKInferenceRequest(inputs: [NFKInputPrompt: "Weather?"],
                                                                    parameters: nil, outputModality: .text))
        let ran = result.output(forKey: NFKOutputToolInvocations) as? [NFKToolInvocation]
        XCTAssertEqual(ran?.first?.toolName, "get_weather")
        XCTAssertTrue(ran?.first?.result.isError ?? false, "the model's arguments lacked the city")
        XCTAssertEqual((result.output(forKey: NFKOutputMessages) as? [[String: Any]])?.count, 4)

        let turn = NFKToolMessages.assistantMessage(for: NFKInferenceResult(outputs: [NFKOutputText: "hi"]))
        XCTAssertEqual(turn["role"] as? String, "assistant")
        let answer = NFKToolMessages.message(forToolCallIdentifier: "call_1", name: "get_weather",
                                             result: NFKToolResult.errorResult(withText: "no city"))
        XCTAssertEqual(answer["role"] as? String, "tool")
    }

    // MCP from Swift: addTool(_:), localMCPURL, and the client's tools(), which throws.
    func testServingToolsOverMCP() throws {
        let server = NFKInferenceServer()
        server.port = 0
        server.loopbackOnly = true
        server.addTool(NFKInferenceTool(name: "get_weather", description: "Current weather in a city.",
                                        parameters: ["type": "object", "properties": ["city": ["type": "string"]]],
                                        handler: { arguments, _ in NFKToolResult(text: "18 C in \(arguments["city"] as? String ?? "")") }))
        try server.start()
        defer { server.stop() }

        let client = NFKMCPClient(endpointURL: try XCTUnwrap(server.localMCPURL), apiKey: nil)
        let tools = try client.tools()
        XCTAssertEqual(tools.map(\.name), ["get_weather"])
        let result = try client.resultOfTool(named: "get_weather", arguments: ["city": "Oslo"])
        XCTAssertEqual(result.text, "18 C in Oslo")
        XCTAssertEqual(client.serverInfo?["name"] as? String, "InferKit")
    }

    // A stdio MCP server from Swift: the command initializer, serverProcess (optional, nil over HTTP),
    // a @Sendable state handler, ping(), and disconnect(). The server is a stand-in written in sh,
    // and the short timeouts bound the example if it stops answering.
    func testAddingAStdioMCPServer() throws {
        let standIn = #"""
        while IFS= read -r line; do
          id=$(printf '%s' "$line" | /usr/bin/sed -n 's/.*"id":\([0-9]*\).*/\1/p')
          case "$line" in
          *'"notifications/'*) ;;
          *'"ping"'*) printf '{"jsonrpc":"2.0","id":%s,"result":{}}\n' "$id" ;;
          *'"initialize"'*) printf '{"jsonrpc":"2.0","id":%s,"result":{"protocolVersion":"2025-06-18","capabilities":{"tools":{}},"serverInfo":{"name":"stand-in","version":"1"}}}\n' "$id" ;;
          *'"tools/list"'*) printf '{"jsonrpc":"2.0","id":%s,"result":{"tools":[{"name":"read_note","inputSchema":{"type":"object"}}]}}\n' "$id" ;;
          *'"tools/call"'*) printf '{"jsonrpc":"2.0","id":%s,"result":{"content":[{"type":"text","text":"%s"}]}}\n' "$id" "$NOTE" ;;
          esac
        done
        """#
        let client = NFKMCPClient(command: "/bin/sh", arguments: ["-c", standIn],
                                  environment: ["NOTE": "Buy milk."], workingDirectoryURL: nil)
        XCTAssertEqual(client.transport, .stdio)
        client.protocolTimeout = 10
        client.timeout = 10
        let process = try XCTUnwrap(client.serverProcess)
        process.restartPolicy = .onCrash
        process.stateHandler = { process, state in
            if state == .exited {
                print("exited \(process.terminationStatus): \(process.recentStandardError)")
            }
        }
        let tools = try client.tools()
        XCTAssertEqual(tools.map(\.name), ["read_note"])
        XCTAssertEqual(try client.resultOfTool(named: "read_note", arguments: [:]).text, "Buy milk.")
        try client.ping()
        client.disconnect()
        XCTAssertEqual(process.state, .exited)
        XCTAssertEqual(process.terminationReason, .exit)
    }

    // An agent's CLI from Swift: link(_:name:scope:bearerTokenEnvironmentVariable:) throws, and
    // mcpLink(named:scope:) returns nil for an absent registration and throws only on a failed command.
    func testLinkingTheServerToAnAgentCLI() throws {
        let server = NFKInferenceServer()
        server.port = 0
        server.loopbackOnly = true
        try server.start()
        defer { server.stop() }

        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let stub = directory.appendingPathComponent("codex")
        try "#!/bin/sh\n[ \"$2\" = get ] && { echo \"Error: No MCP server named '$3' found.\"; exit 1; }\nexit 0\n"
            .write(to: stub, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: stub.path)

        let codex = NFKAgentCLI.codex
        codex.executableURL = stub
        XCTAssertFalse(codex.supportsScope(.project))
        try codex.link(server, name: "inferkit", scope: .user, bearerTokenEnvironmentVariable: nil)
        XCTAssertNil(try codex.mcpLink(named: "inferkit", scope: .user))
        try codex.unlinkMCPServer(named: "inferkit", scope: .user)
        XCTAssertThrowsError(try codex.linkMCPServer(named: "in ferkit", url: try XCTUnwrap(server.localMCPURL),
                                                     bearerTokenEnvironmentVariable: nil, scope: .user)) { error in
            XCTAssertEqual((error as? NFKAgentCLIError)?.code, .invalidArgument)
        }

        // Antigravity (agy) shares its registration with the Antigravity app and IDE; it takes a server
        // that needs no token, so naming a token variable throws before any command runs.
        let antigravity = NFKAgentCLI.antigravity
        antigravity.executableURL = stub
        XCTAssertEqual(antigravity.executableName, "agy")
        XCTAssertThrowsError(try antigravity.link(server, name: "inferkit", scope: .user, bearerTokenEnvironmentVariable: "INFERKIT_KEY")) { error in
            XCTAssertEqual((error as? NFKAgentCLIError)?.code, .invalidArgument)
        }
    }

    // The status client from Swift: the blocking fetch throws, and the awaited form is
    // fetchStatus(baseURL:apiKey:) so it does not take the blocking call's name.
    func testLoadAndModelStatus() async throws {
        let server = NFKInferenceServer()
        server.port = 0
        server.loopbackOnly = true
        server.addBackend(NFKPassthroughBackend(), forModelName: "echo")
        try server.start()
        defer { server.stop() }
        let base = try XCTUnwrap(server.localBaseURL)

        let status = try NFKServerStatus.fetch(baseURL: base, apiKey: nil)
        XCTAssertEqual(status.modelNamed("echo")?.estimatedWaitSeconds, 0)
        XCTAssertNil(status.modelNamed("echo")?.cachedInputShare)
        XCTAssertEqual(status.modelNamed("echo")?.backendStatus.count, 0)
        XCTAssertNotEqual(status.host?.thermalState, .unknown)
        let awaited = try await NFKServerStatus.fetchStatus(baseURL: base, apiKey: nil)
        XCTAssertEqual(awaited.modelNamed("echo")?.limit, 1)
    }

    // The balancer from Swift: the class factory imports as an initializer, and the policy as an enum.
    func testBalancingAcrossServers() throws {
        let server = NFKInferenceServer()
        server.port = 0
        server.loopbackOnly = true
        server.addBackend(NFKPassthroughBackend(), forModelName: "echo")
        try server.start()
        defer { server.stop() }

        let balanced = NFKBalancedBackend(modelName: "echo")
        balanced.policy = .fewestOutstanding
        balanced.addServer(withBaseURL: try XCTUnwrap(server.localBaseURL), apiKey: nil)
        let result = try balanced.runInference(for: NFKInferenceRequest(inputs: [NFKInputPrompt: "hello"]))
        XCTAssertEqual(result.output(forKey: NFKInputPrompt) as? String, "hello")
        XCTAssertEqual(balanced.lastServerBaseURL, server.localBaseURL)

        balanced.conversationWaitAllowance = 30
        let turn = NFKInferenceRequest(inputs: [NFKInputPrompt: "and the action items?"],
                                       parameters: [NFKParameterConversationKey: "meeting-42"])
        XCTAssertEqual(try balanced.runInference(for: turn).output(forKey: NFKInputPrompt) as? String,
                       "and the action items?")
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

        let gemini = NFKGeminiInteractionsBackend(apiKey: "AIza…", modelName: "gemini-nano-banana-2.1")
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

    // Gemini's voice library from Swift: the calls import as throwing methods, and the factory as an
    // optional initializer.
    func testGeminiVoiceLibrary() throws {
        let voices = NFKGeminiVoiceLibrary(for: .googleGemini, apiKey: "AIza…")
        XCTAssertEqual(voices?.endpointURL?.absoluteString, "https://generativelanguage.googleapis.com/v1beta/voices")
        XCTAssertNil(NFKGeminiVoiceLibrary(for: .openAI, apiKey: "sk-…"))

        let designed = NFKRemoteVoice(identifier: "voice_abc", name: "Astronomer", languages: ["en-GB"],
                                      raw: ["sample_audio": ["mime_type": "audio/wav", "data": "UklGRg=="]])
        XCTAssertEqual(designed.sampleAudioData?.count, 4)

        // What the calls look like (they need network, so this is compiled and not run).
        func manage(_ voices: NFKGeminiVoiceLibrary, clip: Data, consent: Data) throws {
            let prebuilt = try voices.voices(matchingFilters: ["type": "prebuilt", "language_code": ["en-US", "en-GB"]])
            let astronomer = try voices.designVoice(withDescription: "A warm astronomer with a gentle British accent.",
                                                    displayName: "Astronomer", attributes: ["gender": "male"])
            let replica = try voices.replicateVoice(fromAudio: clip, consentAudio: consent, displayName: nil, stores: false)
            try voices.deleteVoice(withIdentifier: astronomer.identifier)
            _ = (prebuilt, replica)
        }
    }

    // Gemini's Antigravity agent from Swift: it writes files in a hosted sandbox, which the environment
    // store reads back by the id the reply carries.
    func testGeminiAgentEnvironment() throws {
        let maker = NFKGeminiInteractionsBackend(apiKey: "AIza…", modelName: "gemini-4-argon")
        maker.agentName = "antigravity-preview-09-2026"
        XCTAssertTrue(maker.isReady)
        let environments = NFKGeminiEnvironmentStore(for: .googleGemini, apiKey: "AIza…")
        XCTAssertEqual(environments?.endpointURL?.absoluteString, "https://generativelanguage.googleapis.com/v1beta/environments")
        XCTAssertNil(NFKGeminiEnvironmentStore(for: .openAI, apiKey: "sk-…"))

        // What the calls look like (they need network, so this is compiled and not run).
        func fetchMesh(_ result: NFKInferenceResult, from environments: NFKGeminiEnvironmentStore) throws -> Data? {
            guard let sandbox = result.output(forKey: NFKGeminiOutputEnvironmentIdentifier) as? String else { return nil }
            let files = try environments.files(inEnvironment: sandbox, atPath: "out", recursive: true)
            return try environments.contentsOfFile(atPath: files.first { $0.path.hasSuffix(".stl") }?.path ?? "out/stand.stl",
                                                   inEnvironment: sandbox, recursive: false)
        }
        _ = fetchMesh
    }

    // OpenAI's Decisions API from Swift: the provider factory returns an optional, and a refusal
    // reads through `isRefused`.
    func testOpenAIDecisions() throws {
        let luna = NFKOpenAIDecisionsBackend(for: .openAI, apiKey: "sk-…", modelName: "gpt-6-luna")
        XCTAssertEqual(luna?.endpointURL?.absoluteString, "https://api.openai.com/v1/decisions")

        // What comes back (needs network): `try luna.answers(forState: state, questions: questions)`.
        let urgent = NFKDecisionAnswer(dictionary: ["type": "predicate", "name": "urgent", "probability": 0.91])
        XCTAssertEqual(urgent?.type, .noul)
        let declined = NFKDecisionAnswer.refusal(for: .choice, raw: ["type": "refusal", "name": "department"])
        XCTAssertTrue(declined.isRefused)
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
        let windows = try XCTUnwrap(heard.segments)
        if windows.isEmpty && NFKHardwareProfile.current.isVirtualMachine {
            throw XCTSkip("the classifier heard nothing in this virtual machine")
        }
        XCTAssertGreaterThan(windows.count, 0)
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
