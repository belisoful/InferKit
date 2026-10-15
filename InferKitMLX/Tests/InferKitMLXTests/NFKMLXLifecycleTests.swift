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

final class NFKMLXLifecycleTests: XCTestCase {

    private static let moduleName = "lifecycle-test-module"
    private static let generationName = "lifecycle-test-generation"
    private static let plainName = "lifecycle-test-plain"
    private static let gatedName = "lifecycle-test-gated"

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
        for name in [Self.moduleName, Self.generationName, Self.plainName, Self.gatedName] {
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
}
