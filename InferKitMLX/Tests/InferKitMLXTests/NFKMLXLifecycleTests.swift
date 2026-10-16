//
//  NFKMLXLifecycleTests.swift
//  InferKitMLXTests
//
//  Unloading a backend's model and loading it again, and the memory a load is measured to hold. The
//  models are small seeded random weights behind registered factories, so each build reproduces the
//  same weights the way a build from a weights file does.
//

import XCTest
import CoreGraphics
import InferKit
import InferKitMLXSupport
import MLX
@testable import InferKitMLX

/// A backend outside the lifecycle classes, holding one array, which the registry wraps.
private final class NFKMLXLifecycleTestBackend: NSObject, NFKInferenceBackend {
    let weights: MLXArray

    init(weights: MLXArray) {
        self.weights = weights
        super.init()
    }

    var isReady: Bool { true }

    var backendIdentifier: String { "lifecycle-test-plain" }

    var supportedOutputKeys: Set<String> { [NFKOutputText] }

    func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        NFKInferenceResult(outputs: [NFKOutputText: "\(weights.sum().item(Float.self))"])
    }
}

/// Seeded weights a `@Sendable` forward reads.
private final class NFKMLXWeightsHolder: @unchecked Sendable {
    let weights: MLXArray

    init(_ weights: MLXArray) {
        self.weights = weights
    }
}

/// A build count a registered factory raises from any thread.
private final class NFKMLXBuildCounter: @unchecked Sendable {
    private let lock = NSLock()
    private var count = 0

    func increment() {
        lock.withLock { count += 1 }
    }

    var value: Int { lock.withLock { count } }
}

/// A backend without weights, which a wrapper builds again after each unload.
private final class NFKMLXWeightlessTestBackend: NSObject, NFKInferenceBackend {
    var isReady: Bool { true }

    var backendIdentifier: String { "lifecycle-test-weightless" }

    func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        NFKInferenceResult(outputs: [NFKOutputText: "ran"])
    }
}

/// A backend without weights whose own method and token count signal `entered`, then wait at `gate`.
private final class NFKMLXGatedTestBackend: NSObject, NFKInferenceBackend {
    private let entered: DispatchSemaphore
    private let gate: DispatchSemaphore

    init(entered: DispatchSemaphore, gate: DispatchSemaphore) {
        self.entered = entered
        self.gate = gate
        super.init()
    }

    var isReady: Bool { true }

    var backendIdentifier: String { "lifecycle-test-gated-weightless" }

    func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        NFKInferenceResult(outputs: [NFKOutputText: "ran"])
    }

    @objc func waitAtGate() {
        entered.signal()
        _ = gate.wait(timeout: .now() + 30)
    }

    @objc(tokenCountForRequest:error:)
    func tokenCount(for request: NFKInferenceRequest) throws -> NSNumber {
        waitAtGate()
        return 3
    }

    @objc func raiseAnException() {
        NSException(name: .genericException, reason: "the held backend raised", userInfo: nil).raise()
    }

    @objc func scaled(_ value: Int, by factor: Int) -> Int {
        value * factor
    }

    @objc func label(_ prefix: String) -> String {
        "\(prefix) \(backendIdentifier)"
    }
}

/// Messages only the gated backend implements, sent to the wrapper.
@objc private protocol NFKMLXGatedMessages {
    func scaled(_ value: Int, by factor: Int) -> Int
    func label(_ prefix: String) -> String
}

/// The token count, sent as an Objective-C message: the Swift 6.4 compiler crashes generating a call
/// through the optional requirement of `NFKInferenceBackend`.
@objc private protocol NFKMLXTokenCountMessage {
    @objc(tokenCountForRequest:error:)
    func tokenCount(for request: NFKInferenceRequest) throws -> NSNumber
}

/// A value one thread sets and another reads.
private final class NFKMLXLockedValue<Value>: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: Value?

    func set(_ value: Value?) {
        lock.withLock { stored = value }
    }

    var value: Value? { lock.withLock { stored } }
}

final class NFKMLXLifecycleTests: XCTestCase {

    private static let moduleName = "lifecycle-test-module"
    private static let generationName = "lifecycle-test-generation"
    private static let plainName = "lifecycle-test-plain"
    private static let gatedName = "lifecycle-test-gated"
    private static let localFileName = "lifecycle-test-local-file"

    /// 2048 × 1024 float32 weights, 8 MiB, the same on every build.
    private static func seededWeights() -> MLXArray {
        let weights = MLXRandom.normal([2048, 1024], key: MLXRandom.key(7))
        eval(weights)
        return weights
    }

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    /// The URL the registry needs to treat a build as reproducible; the test factories read nothing from it.
    private var sourceURL: URL {
        FileManager.default.temporaryDirectory.appendingPathComponent("NFKMLXLifecycleTests-source")
    }

    override func tearDown() {
        for name in [Self.moduleName, Self.generationName, Self.plainName, Self.gatedName, Self.localFileName] {
            NFKMLXModelRegistry.unregisterModel(named: name)
        }
        if NFKMLXGPU.metalLibraryURL != nil {
            NFKMLXGPU.synchronize()
            NFKMLXGPU.clearCache()
        }
        super.tearDown()
    }

    private func registerModule() {
        NFKMLXModelRegistry.register(name: Self.moduleName) { _ in
            let holder = NFKMLXWeightsHolder(Self.seededWeights())
            return NFKMLXModuleBackend(identifier: Self.moduleName) { image in
                clip(image * 0.5 + holder.weights.mean(), min: 0, max: 1)
            }
        }
    }

    private func registerGeneration() {
        NFKMLXModelRegistry.register(name: Self.generationName) { _ in
            let weights = Self.seededWeights()
            return NFKMLXGenerationBackend(identifier: Self.generationName, inputKeys: [NFKInputPrompt],
                                           parameterKeys: [], outputKeys: [NFKOutputText]) { _, _ in
                NFKInferenceResult(outputs: [NFKOutputText: "\(weights.sum().item(Float.self))"])
            }
        }
    }

    private func imageBytes(_ result: NFKInferenceResult) throws -> [UInt8] {
        let image = try XCTUnwrap(result.output(forKey: NFKOutputImage))
        return try NFKMLXImageBridge.rgbaBytes(from: image, colorSpace: CGColorSpaceCreateDeviceRGB()).bytes
    }

    private var imageRequest: NFKInferenceRequest {
        NFKInferenceRequest(inputs: [NFKInputImage: NFKMLXBackendTestSupport.solidImage(width: 16, height: 16)])
    }

    // MARK: Without a source

    func testABackendBuiltFromAClosureKeepsItsModel() throws {
        let backend = NFKMLXModuleBackend(identifier: "closure") { $0 }
        XCTAssertTrue(backend.isLoaded)
        backend.unload()
        XCTAssertTrue(backend.isLoaded, "a closure alone has no source to load from")
        XCTAssertEqual(backend.residentByteCount, 0, "nothing measured the closure's build")
        XCTAssertEqual(NFKInferenceIsLoaded(backend), true)
    }

    func testARegistryBuildWithoutAURLKeepsItsModel() throws {
        try requireMLXRuntime()
        registerModule()
        let backend = try NFKMLXModelRegistry.backend(named: Self.moduleName, weightsURL: nil)
        backend.unload?()
        XCTAssertEqual(backend.isLoaded, true, "random weights would not come back the same")
    }

    // MARK: Module backend

    func testUnloadFreesTheMeasuredBytesAndARunLoadsTheModelAgain() throws {
        try requireMLXRuntime()
        registerModule()
        let backend = try XCTUnwrap(try NFKMLXModelRegistry.backend(named: Self.moduleName, weightsURL: sourceURL)
                                    as? NFKMLXModuleBackend, "a lifecycle backend keeps its class")
        let first = try imageBytes(try backend.runInference(for: imageRequest))
        let resident = backend.residentByteCount
        XCTAssertGreaterThanOrEqual(resident, 2048 * 1024 * 4, "the weights are measured")
        XCTAssertGreaterThanOrEqual(backend.peakByteCount, resident)
        XCTAssertEqual(backend.modelInfo[NFKModelInfoResidentBytes] as? Int, resident)

        NFKMLXGPU.synchronize()
        let before = NFKMLXGPU.activeMemory
        backend.unload()
        let freed = before - NFKMLXGPU.activeMemory
        XCTAssertFalse(backend.isLoaded)
        XCTAssertEqual(backend.residentByteCount, 0)
        XCTAssertEqual(freed, resident, accuracy: resident / 10 + (1 << 20), "unload frees what the load held")
        XCTAssertGreaterThan(backend.peakByteCount, 0, "the peak describes the last load and stays")

        let second = try imageBytes(try backend.runInference(for: imageRequest))
        XCTAssertTrue(backend.isLoaded, "the run loaded the model again")
        XCTAssertEqual(second, first, "the reloaded model answers as before")
        XCTAssertGreaterThan(backend.residentByteCount, 0, "the reload is measured")
    }

    func testPrepareLoadsAnUnloadedModel() throws {
        try requireMLXRuntime()
        registerModule()
        let backend = try NFKMLXModelRegistry.backend(named: Self.moduleName, weightsURL: sourceURL)
        backend.unload?()
        XCTAssertEqual(backend.isLoaded, false)
        XCTAssertTrue(NFKInferencePrepare(backend, nil))
        XCTAssertEqual(backend.isLoaded, true)
        XCTAssertGreaterThan(NFKInferenceResidentByteCount(backend), 0)
    }

    // MARK: Generation backend

    func testAGenerationBackendUnloadsAndLoadsAgain() throws {
        try requireMLXRuntime()
        registerGeneration()
        let backend = try XCTUnwrap(try NFKMLXModelRegistry.backend(named: Self.generationName, weightsURL: sourceURL)
                                    as? NFKMLXGenerationBackend)
        let request = NFKInferenceRequest(inputs: [NFKInputPrompt: "sum"])
        let first = try backend.runInference(for: request).text
        XCTAssertGreaterThan(backend.residentByteCount, 0)

        backend.unload()
        XCTAssertFalse(backend.isLoaded)
        XCTAssertEqual(backend.residentByteCount, 0)
        XCTAssertNotNil(backend.modelInfo[NFKModelInfoPeakBytes], "the description keeps the last load's figures")

        XCTAssertEqual(try backend.runInference(for: request).text, first)
        XCTAssertTrue(backend.isLoaded)
    }

    func testAnUnloadDuringARunFreesTheModelWhenTheRunEnds() throws {
        try requireMLXRuntime()
        let started = DispatchSemaphore(value: 0)
        let gate = DispatchSemaphore(value: 0)
        NFKMLXModelRegistry.register(name: Self.gatedName) { _ in
            let weights = Self.seededWeights()
            return NFKMLXGenerationBackend(identifier: Self.gatedName, inputKeys: [NFKInputPrompt],
                                           parameterKeys: [], outputKeys: [NFKOutputText]) { _, _ in
                started.signal()
                gate.wait()
                return NFKInferenceResult(outputs: [NFKOutputText: "\(weights.shape)"])
            }
        }
        let backend = try NFKMLXModelRegistry.backend(named: Self.gatedName, weightsURL: sourceURL)
        let job = NFKInferenceSubmit(backend, NFKInferenceRequest(inputs: [NFKInputPrompt: "x"]), nil)
        started.wait()

        backend.unload?()
        XCTAssertEqual(backend.isLoaded, true, "the run in flight keeps the model")

        let finished = expectation(description: "the run finished")
        job.completionHandler = { _ in finished.fulfill() }
        gate.signal()
        wait(for: [finished], timeout: 30)
        XCTAssertNotNil(job.result, "the run was not cut")
        XCTAssertEqual(backend.isLoaded, false, "the last run to end freed the model")
    }

    // MARK: Any other backend

    func testAnyOtherBackendIsWrappedAndBuiltAgain() throws {
        try requireMLXRuntime()
        let builds = NFKMLXBuildCounter()
        NFKMLXModelRegistry.register(name: Self.plainName) { _ in
            builds.increment()
            return NFKMLXLifecycleTestBackend(weights: Self.seededWeights())
        }
        let backend = try NFKMLXModelRegistry.backend(named: Self.plainName, weightsURL: sourceURL)
        let wrapper = try XCTUnwrap(backend as? NFKMLXReloadableBackend)
        XCTAssertTrue(wrapper.wrappedBackend is NFKMLXLifecycleTestBackend, "the model's own class is reachable")
        XCTAssertEqual(backend.backendIdentifier, "lifecycle-test-plain")
        XCTAssertEqual(backend.supportedOutputKeys, [NFKOutputText])
        XCTAssertFalse((backend as AnyObject).responds(to: NSSelectorFromString("backendStatus")),
                       "the wrapper answers for the wrapped class")
        let request = NFKInferenceRequest(inputs: [NFKInputPrompt: "sum"])
        let first = try backend.runInference(for: request).text
        XCTAssertGreaterThan(NFKInferenceResidentByteCount(backend), 0)

        NFKInferenceUnload(backend)
        XCTAssertFalse(NFKInferenceIsLoaded(backend))
        XCTAssertNil(wrapper.wrappedBackend, "nothing is held while unloaded")
        XCTAssertEqual(try backend.runInference(for: request).text, first)
        XCTAssertNotNil(wrapper.wrappedBackend)
        XCTAssertEqual(builds.value, 2, "one build, and one after the unload")
    }

    func testTheWrapperTakesADirectFactory() throws {
        try requireMLXRuntime()
        let backend = try NFKMLXReloadableBackend {
            NFKMLXLifecycleTestBackend(weights: Self.seededWeights())
        }
        XCTAssertTrue(backend.isLoaded)
        XCTAssertGreaterThan(backend.residentByteCount, 0)
        backend.unload()
        XCTAssertFalse(backend.isLoaded)
        XCTAssertEqual(backend.residentByteCount, 0)
        try backend.prepare()
        XCTAssertTrue(backend.isLoaded)
    }

    // MARK: Server

    func testTheServerReportsTheResidentBytesOfAnMLXModelAndUnloadsIt() throws {
        try requireMLXRuntime()
        registerModule()
        let server = NFKInferenceServer()
        server.residencyCheckInterval = 0
        let built = NSMutableArray()
        let sourceURL = self.sourceURL
        server.addBackendProvider({ _ in
            guard let backend = try? NFKMLXModelRegistry.backend(named: Self.moduleName, weightsURL: sourceURL) else {
                return nil
            }
            built.add(backend)
            return backend
        }, forModelName: "module")
        let loaded = expectation(description: "the provider built the model")
        server.loadModelNamed("module") { error in
            XCTAssertNil(error)
            loaded.fulfill()
        }
        wait(for: [loaded], timeout: 60)

        let status = try XCTUnwrap(server.currentStatus().models.first { $0.name == "module" })
        let resident = try XCTUnwrap(status.residentBytes).intValue
        XCTAssertGreaterThanOrEqual(resident, 2048 * 1024 * 4)
        XCTAssertGreaterThanOrEqual(try XCTUnwrap(status.peakBytes).intValue, resident)

        let backend = try XCTUnwrap(built.firstObject as? NFKMLXModuleBackend)
        XCTAssertTrue(server.unloadModelNamed("module"))
        XCTAssertFalse(backend.isLoaded, "the server unloads the backend before releasing it")
        let afterUnload = try XCTUnwrap(server.currentStatus().models.first { $0.name == "module" })
        XCTAssertEqual(afterUnload.residentBytes?.intValue, resident, "the figures describe the last load")
    }

    // MARK: Unload during a load

    private var textRequest: NFKInferenceRequest {
        NFKInferenceRequest(inputs: [NFKInputPrompt: "x"])
    }

    func testAnUnloadDuringALoadFreesTheModelWhenTheRunEnds() throws {
        let loading = DispatchSemaphore(value: 0)
        let gate = DispatchSemaphore(value: 0)
        let builds = NFKMLXBuildCounter()
        let backend = try NFKMLXReloadableBackend {
            builds.increment()
            if builds.value > 1 {
                loading.signal()
                _ = gate.wait(timeout: .now() + 30)
            }
            return NFKMLXWeightlessTestBackend()
        }
        backend.unload()
        XCTAssertFalse(backend.isLoaded)

        let result = NFKMLXLockedValue<NFKInferenceResult>()
        let ran = DispatchSemaphore(value: 0)
        let request = textRequest
        DispatchQueue.global(qos: .userInitiated).async {
            result.set(try? backend.runInference(for: request))
            ran.signal()
        }
        XCTAssertEqual(loading.wait(timeout: .now() + 30), .success, "the run started a load")
        backend.unload()
        gate.signal()
        XCTAssertEqual(ran.wait(timeout: .now() + 30), .success, "the run returned")

        XCTAssertNotNil(result.value, "the run that asked for the load completed")
        XCTAssertFalse(backend.isLoaded, "the unload during the load freed the model when the run ended")
        XCTAssertEqual(builds.value, 2)
    }

    func testAnUnloadOfAnUnloadedModelLeavesTheNextLoadResident() throws {
        let backend = try NFKMLXReloadableBackend { NFKMLXWeightlessTestBackend() }
        backend.unload()
        backend.unload()
        XCTAssertEqual(try backend.runInference(for: textRequest).text, "ran")
        XCTAssertTrue(backend.isLoaded, "an unload that found nothing loaded does not wait for the next load")
    }

    // MARK: Forwarded messages

    func testAnUnloadDuringAForwardedMessageFreesTheModelAfterItReturns() throws {
        let entered = DispatchSemaphore(value: 0)
        let gate = DispatchSemaphore(value: 0)
        let backend = try NFKMLXReloadableBackend { NFKMLXGatedTestBackend(entered: entered, gate: gate) }
        let selector = NSSelectorFromString("waitAtGate")
        XCTAssertTrue(backend.responds(to: selector), "the wrapper answers for the held class")

        let returned = DispatchSemaphore(value: 0)
        let workerMayEnd = DispatchSemaphore(value: 0)
        let loadedOnReturn = NFKMLXLockedValue<Bool>()
        DispatchQueue.global(qos: .userInitiated).async {
            _ = backend.perform(selector)
            loadedOnReturn.set(backend.isLoaded)
            returned.signal()
            _ = workerMayEnd.wait(timeout: .now() + 30)
        }
        XCTAssertEqual(entered.wait(timeout: .now() + 30), .success, "the forwarded message reached the held backend")
        backend.unload()
        XCTAssertTrue(backend.isLoaded, "the forwarded message in flight keeps the model")
        gate.signal()
        XCTAssertEqual(returned.wait(timeout: .now() + 30), .success, "the forwarded message returned")
        XCTAssertEqual(loadedOnReturn.value, false, "the model was freed when the message returned")
        XCTAssertFalse(backend.isLoaded, "the model is freed while the worker's block still runs")
        workerMayEnd.signal()
    }

    func testAForwardedMessageThatRaisesEndsItsRunAndAnUnloadFreesTheModel() throws {
        let backend = try NFKMLXReloadableBackend {
            NFKMLXGatedTestBackend(entered: DispatchSemaphore(value: 0), gate: DispatchSemaphore(value: 0))
        }
        let inFlight = NFKMLXActivity.inFlightCount
        let exception = NFKMLXCatchException {
            _ = backend.perform(NSSelectorFromString("raiseAnException"))
        }
        XCTAssertEqual(exception?.reason, "the held backend raised", "the exception reaches the sender")
        XCTAssertEqual(NFKMLXActivity.inFlightCount, inFlight, "the forwarded run ended during the unwind")
        backend.unload()
        XCTAssertFalse(backend.isLoaded, "no run is left counted, so the unload frees the model")
    }

    func testAForwardedMessageCarriesItsArgumentsAndResultAndLoadsAnUnloadedModel() throws {
        let backend = try NFKMLXReloadableBackend {
            NFKMLXGatedTestBackend(entered: DispatchSemaphore(value: 0), gate: DispatchSemaphore(value: 0))
        }
        backend.unload()
        XCTAssertFalse(backend.isLoaded)
        let messages = unsafeBitCast(backend, to: NFKMLXGatedMessages.self)
        XCTAssertEqual(messages.scaled(6, by: 7), 42)
        XCTAssertEqual(messages.label("from"), "from lifecycle-test-gated-weightless")
        XCTAssertTrue(backend.isLoaded, "the forwarded message built the model")
    }

    func testAnUnloadDuringATokenCountFreesTheModelAfterTheCount() throws {
        let entered = DispatchSemaphore(value: 0)
        let gate = DispatchSemaphore(value: 0)
        let backend: any NFKInferenceBackend = try NFKMLXReloadableBackend {
            NFKMLXGatedTestBackend(entered: entered, gate: gate)
        }
        let count = NFKMLXLockedValue<NSNumber>()
        let counted = DispatchSemaphore(value: 0)
        let request = textRequest
        DispatchQueue.global(qos: .userInitiated).async {
            count.set(try? unsafeBitCast(backend, to: NFKMLXTokenCountMessage.self).tokenCount(for: request))
            counted.signal()
        }
        XCTAssertEqual(entered.wait(timeout: .now() + 30), .success, "the count reached the held backend")
        backend.unload?()
        XCTAssertEqual(backend.isLoaded, true, "the count in flight keeps the model")
        gate.signal()
        XCTAssertEqual(counted.wait(timeout: .now() + 30), .success, "the count returned")
        XCTAssertEqual(count.value, 3)
        XCTAssertEqual(backend.isLoaded, false, "the unload freed the model after the count")
    }

    func testTheWrapperCountsTokensOnlyWhenTheHeldClassDoes() throws {
        let backend = try NFKMLXReloadableBackend { NFKMLXWeightlessTestBackend() }
        XCTAssertFalse(backend.responds(to: NSSelectorFromString("tokenCountForRequest:error:")),
                       "the wrapper answers for the held class")
        XCTAssertThrowsError(try backend.tokenCount(for: textRequest))
    }

    // MARK: Loading again from removed files

    private func temporaryFolder() throws -> URL {
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("NFKMLXLifecycleTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        return folder
    }

    private func writePlaceholder(_ url: URL) throws {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: url)
    }

    private func assertModelFilesRemoved(_ error: Error, path: String, file: StaticString = #filePath, line: UInt = #line) {
        let error = error as NSError
        XCTAssertEqual(error.domain, NFKMLXReloadError.errorDomain, file: file, line: line)
        XCTAssertEqual(error.code, NFKMLXReloadError.modelFilesRemoved.rawValue, file: file, line: line)
        XCTAssertEqual(error.userInfo[NSFilePathErrorKey] as? String, path, file: file, line: line)
        XCTAssertTrue(error.localizedDescription.contains("NFKDownload"), file: file, line: line)
    }

    func testAFetchDuringALoadReadsTheCacheOnly() throws {
        let cache = try temporaryFolder()
        defer { try? FileManager.default.removeItem(at: cache) }
        let hub = NFKFailingTransportHub()
        hub.cacheDirectoryURL = cache
        XCTAssertThrowsError(try NFKMLXReload.reloading {
            try hub.fetchFile(repo: "org/absent", revision: nil, path: "model.safetensors")
        }) { assertModelFilesRemoved($0, path: "model.safetensors") }
        XCTAssertEqual(hub.attempts, [], "a load after an unload makes no request")
    }

    func testALoadAfterTheReleaseIsRemovedFailsWithoutARequest() throws {
        let cache = try temporaryFolder()
        defer { try? FileManager.default.removeItem(at: cache) }
        let hub = NFKFailingTransportHub()
        hub.cacheDirectoryURL = cache
        let repo = "org/lifecycle-release"
        let required = ["config.json"]
        let weights = ["model.safetensors"]
        let seed = {
            for path in required + weights {
                try self.writePlaceholder(try XCTUnwrap(hub.localURL(forRepo: repo, revision: nil, path: path)))
            }
        }
        try seed()
        let snapshot = try XCTUnwrap(hub.localURL(forRepo: repo, revision: nil, path: "config.json")).deletingLastPathComponent()

        try NFKMLXReleaseDownload.using(hub) {
            let backend = try NFKMLXReloadableBackend {
                let folder = try NFKMLXReleaseDownload.directory(repo: repo, revision: nil, cacheDirectoryURL: cache,
                                                                 required: required, optional: ["tokenizer.json"],
                                                                 weights: weights)
                _ = try Data(contentsOf: folder.appendingPathComponent("config.json"))
                return NFKMLXWeightlessTestBackend()
            }
            let firstBuildAttempts = hub.attempts
            backend.unload()
            try FileManager.default.removeItem(at: snapshot)

            XCTAssertThrowsError(try backend.runInference(for: textRequest)) {
                assertModelFilesRemoved($0, path: "config.json")
            }
            XCTAssertEqual(hub.attempts, firstBuildAttempts, "the load after the removal made no request")
            XCTAssertFalse(backend.isLoaded)

            try seed()
            XCTAssertEqual(try backend.runInference(for: textRequest).text, "ran",
                           "files downloaded again load on the next run")
            XCTAssertTrue(backend.isLoaded)
            XCTAssertEqual(hub.attempts, firstBuildAttempts, "the load from the cache made no request")
        }
    }

    func testALoadAfterTheWeightsFileIsRemovedFails() throws {
        let folder = try temporaryFolder()
        defer { try? FileManager.default.removeItem(at: folder) }
        let weights = folder.appendingPathComponent("model.safetensors")
        try writePlaceholder(weights)
        NFKMLXModelRegistry.register(name: Self.localFileName) { url in
            _ = try Data(contentsOf: url ?? weights)
            return NFKMLXWeightlessTestBackend()
        }
        let backend = try NFKMLXModelRegistry.backend(named: Self.localFileName, weightsURL: weights)
        backend.unload?()
        try FileManager.default.removeItem(at: weights)

        XCTAssertThrowsError(try backend.runInference(for: textRequest)) {
            assertModelFilesRemoved($0, path: weights.path)
        }
        try writePlaceholder(weights)
        XCTAssertEqual(try backend.runInference(for: textRequest).text, "ran")
    }

    func testALoadAfterAFileInTheReleaseFolderIsRemovedFails() throws {
        let folder = try temporaryFolder()
        defer { try? FileManager.default.removeItem(at: folder) }
        let shard = folder.appendingPathComponent("transformer/model.safetensors")
        try writePlaceholder(folder.appendingPathComponent("config.json"))
        try writePlaceholder(shard)
        NFKMLXModelRegistry.register(name: Self.localFileName) { _ in
            _ = try Data(contentsOf: shard)
            return NFKMLXWeightlessTestBackend()
        }
        let backend = try NFKMLXModelRegistry.backend(named: Self.localFileName, weightsURL: folder)
        backend.unload?()
        try FileManager.default.removeItem(at: shard)

        XCTAssertThrowsError(try backend.runInference(for: textRequest)) { error in
            let error = error as NSError
            XCTAssertEqual(error.domain, NFKMLXReloadError.errorDomain)
            XCTAssertEqual(error.code, NFKMLXReloadError.modelFilesRemoved.rawValue)
            let path = error.userInfo[NSFilePathErrorKey] as? String ?? ""
            XCTAssertTrue(path.hasSuffix("transformer/model.safetensors"), path)
        }
        XCTAssertEqual(backend.isLoaded, false)
    }

    /// A safetensors file with an empty header, which `NFKMLXSafetensors.entries(inFile:)` reads.
    private func writeEmptySafetensors(_ url: URL) throws {
        let header = Data(#"{"__metadata__":{}}"#.utf8)
        var length = UInt64(header.count).littleEndian
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try (Data(bytes: &length, count: 8) + header).write(to: url)
    }

    func testASiblingRemovedFromTheFolderLeavesTheReloadWorkingAndTheOpenedFileIsRequired() throws {
        let folder = try temporaryFolder()
        defer { try? FileManager.default.removeItem(at: folder) }
        let weights = folder.appendingPathComponent("model.safetensors")
        let sibling = folder.appendingPathComponent("other-variant/model.safetensors")
        let partial = folder.appendingPathComponent("other-variant/model-2.safetensors.download")
        try writeEmptySafetensors(weights)
        try writePlaceholder(sibling)
        try writePlaceholder(partial)
        NFKMLXModelRegistry.register(name: Self.localFileName) { _ in
            _ = try NFKMLXSafetensors.entries(inFile: weights)
            return NFKMLXWeightlessTestBackend()
        }
        let backend = try NFKMLXModelRegistry.backend(named: Self.localFileName, weightsURL: folder)
        backend.unload?()
        try FileManager.default.removeItem(at: sibling)
        try FileManager.default.removeItem(at: partial)
        XCTAssertEqual(try backend.runInference(for: textRequest).text, "ran",
                       "another variant's file and a partial transfer are not the model's files")

        backend.unload?()
        try FileManager.default.removeItem(at: weights)
        XCTAssertThrowsError(try backend.runInference(for: textRequest)) {
            assertModelFilesRemoved($0, path: weights.path)
        }
    }

    func testAListedFileRemovedFailsTheReloadAndAnUnlistedOneDoesNot() throws {
        let folder = try temporaryFolder()
        defer { try? FileManager.default.removeItem(at: folder) }
        let weights = folder.appendingPathComponent("model.safetensors")
        let sibling = folder.appendingPathComponent("variant-b/model.safetensors")
        for url in [folder.appendingPathComponent("config.json"), weights, sibling] {
            try writePlaceholder(url)
        }
        NFKMLXModelRegistry.register(name: Self.localFileName) { _ in NFKMLXWeightlessTestBackend() }
        let backend = try NFKMLXModelRegistry.backend(named: Self.localFileName, weightsURL: folder, loadOptions: nil,
                                                      listing: ["config.json", "model.safetensors", "tokenizer.json"])
        backend.unload?()
        try FileManager.default.removeItem(at: sibling)
        XCTAssertEqual(try backend.runInference(for: textRequest).text, "ran",
                       "an unlisted file and a listed file absent at the build are not required")

        backend.unload?()
        try FileManager.default.removeItem(at: weights)
        XCTAssertThrowsError(try backend.runInference(for: textRequest)) {
            assertModelFilesRemoved($0, path: weights.path)
        }
    }

    // MARK: Snapshot holds across a build

    func testAReleaseFetchedInsideABuildStaysHeldUntilTheBuildReturns() throws {
        let cache = try temporaryFolder()
        defer { try? FileManager.default.removeItem(at: cache) }
        let hub = NFKFailingTransportHub()
        hub.cacheDirectoryURL = cache
        let repo = "org/held-release"
        for path in ["config.json", "model.safetensors"] {
            try writePlaceholder(try XCTUnwrap(hub.localURL(forRepo: repo, revision: nil, path: path)))
        }
        try NFKMLXReleaseDownload.using(hub) {
            try NFKMLXBuildHolds.keeping {
                _ = try NFKMLXReleaseDownload.directory(repo: repo, revision: nil, cacheDirectoryURL: cache,
                                                        required: ["config.json"], optional: [],
                                                        weights: ["model.safetensors"])
                XCTAssertThrowsError(try hub.removeCachedRepo(repo, revision: nil),
                                     "the fetch returned, and the build still holds the release") { error in
                    XCTAssertEqual((error as NSError).domain, NFKDownloadErrorDomain)
                    XCTAssertEqual((error as NSError).code, NFKDownloadError.inUse.rawValue)
                }
            }
        }
        XCTAssertNoThrow(try hub.removeCachedRepo(repo, revision: nil), "the hold ended when the build returned")
        XCTAssertEqual(hub.attempts, [], "the release came from the cache")
    }

    func testAReleaseFetchedOutsideABuildIsReleasedWhenTheFetchReturns() throws {
        let cache = try temporaryFolder()
        defer { try? FileManager.default.removeItem(at: cache) }
        let hub = NFKFailingTransportHub()
        hub.cacheDirectoryURL = cache
        let repo = "org/unheld-release"
        try writePlaceholder(try XCTUnwrap(hub.localURL(forRepo: repo, revision: nil, path: "config.json")))
        try NFKMLXReleaseDownload.using(hub) {
            _ = try NFKMLXReleaseDownload.directory(repo: repo, revision: nil, cacheDirectoryURL: cache,
                                                    required: ["config.json"], optional: [], weights: [])
        }
        XCTAssertNoThrow(try hub.removeCachedRepo(repo, revision: nil))
    }
}
