//
//  NFKMLXModelBuilderTests.swift
//  InferKitMLXTests
//
//  The MLX build path: the registry's options factories, a reload that keeps the options, the catalog's
//  build entry through the MLX builder with a placeholder location, the property options it sets by
//  key-value coding, and the option registrations every family's descriptor relies on.
//

import XCTest
import InferKit
import MLX
@testable import InferKitMLX

/// A backend outside the lifecycle classes with an MLX enumeration property, which the registry wraps.
private final class NFKMLXBuilderTestBackend: NSObject, NFKInferenceBackend {
    let options: NFKMLXLoadOptions
    @objc var precision: NFKMLXWeightPrecision = .checkpoint
    @objc var pagesExperts = false

    init(options: NFKMLXLoadOptions) {
        self.options = options
        super.init()
    }

    var isReady: Bool { true }

    var backendIdentifier: String { "builder-test" }

    func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        NFKInferenceResult(outputs: [NFKOutputText: options.residency.optionValue])
    }
}

/// The options each build of a registered factory received, from any thread.
private final class NFKMLXBuildLog: @unchecked Sendable {
    private let lock = NSLock()
    private var builds = [NFKMLXLoadOptions]()

    func record(_ options: NFKMLXLoadOptions) {
        lock.withLock { builds.append(options) }
    }

    var all: [NFKMLXLoadOptions] { lock.withLock { builds } }
}

final class NFKMLXModelBuilderTests: XCTestCase {

    private static let plainName = "builder-test-plain"
    private static let moduleName = "builder-test-module"
    private static let stableDiffusionName = "builder-test-stable-diffusion"

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    /// The URL that makes a registry build reloadable; the test factories read nothing from it.
    private var placeholderURL: URL {
        FileManager.default.temporaryDirectory.appendingPathComponent("NFKMLXModelBuilderTests-source")
    }

    override func tearDown() {
        for name in [Self.plainName, Self.moduleName, Self.stableDiffusionName] {
            NFKMLXModelRegistry.unregisterModel(named: name)
        }
        if NFKMLXGPU.metalLibraryURL != nil {
            NFKMLXGPU.synchronize()
            NFKMLXGPU.clearCache()
        }
        super.tearDown()
    }

    private func registerPlain(log: NFKMLXBuildLog) {
        NFKMLXModelRegistry.register(name: Self.plainName,
                                     consumes: [NFKModelLoadOptionResidency, NFKModelLoadOptionQuantizationBits]) {
            _, options in
            log.record(options)
            return NFKMLXBuilderTestBackend(options: options)
        }
    }

    // MARK: Registry

    func testTheOptionsFactoryReceivesTheOptionsItIsGiven() throws {
        let log = NFKMLXBuildLog()
        registerPlain(log: log)
        XCTAssertEqual(NFKMLXModelRegistry.consumedOptionKeys(forModelNamed: Self.plainName),
                       [NFKModelLoadOptionResidency, NFKModelLoadOptionQuantizationBits])
        let options: [String: Any] = [NFKModelLoadOptionResidency: NFKModelResidencyStaged,
                                      NFKModelLoadOptionQuantizationBits: 4]
        let backend = try NFKMLXModelRegistry.backend(named: Self.plainName, weightsURL: nil, options: options)
        XCTAssertTrue(backend is NFKMLXBuilderTestBackend, "a build without a URL is returned as built")
        let received = try XCTUnwrap(log.all.first)
        XCTAssertEqual(received.residency, .staged)
        XCTAssertEqual(received.quantizationBits, 4)
        XCTAssertNil(received.draftURL)
    }

    func testAFactoryWithoutOptionsConsumesNoKeyAndBuildsWithAny() throws {
        NFKMLXModelRegistry.register(name: Self.plainName) { _ in
            NFKMLXBuilderTestBackend(options: NFKMLXLoadOptions())
        }
        XCTAssertEqual(NFKMLXModelRegistry.consumedOptionKeys(forModelNamed: Self.plainName), [])
        XCTAssertEqual(NFKMLXModelRegistry.consumedOptionKeys(forModelNamed: "no-such-model"), [])
        let backend = try NFKMLXModelRegistry.backend(named: Self.plainName, weightsURL: nil,
                                                      options: [NFKModelLoadOptionResidency: NFKModelResidencyPaged])
        XCTAssertEqual((backend as? NFKMLXBuilderTestBackend)?.options.residency, .automatic)
    }

    func testTheObjectiveCEntryPassesADraftPathAsTheDraftsLocation() throws {
        let log = NFKMLXBuildLog()
        registerPlain(log: log)
        _ = try NFKMLXModelRegistry.backend(named: Self.plainName, weightsURL: nil,
                                            options: [NFKModelLoadOptionDraftModel: "/tmp/builder-test-draft"])
        XCTAssertEqual(log.all.first?.draftURL, URL(fileURLWithPath: "/tmp/builder-test-draft"))
    }

    func testABuildFromAURLReportsItsLoadOptions() throws {
        let log = NFKMLXBuildLog()
        registerPlain(log: log)
        let backend = try NFKMLXModelRegistry.backend(named: Self.plainName, weightsURL: placeholderURL,
                                                      options: [NFKModelLoadOptionResidency: NFKModelResidencyStaged])
        let wrapper = try XCTUnwrap(backend as? NFKMLXReloadableBackend)
        XCTAssertTrue(wrapper.wrappedBackend is NFKMLXBuilderTestBackend)
        XCTAssertEqual(NFKInferenceLoadOptionValues(backend)[NFKModelLoadOptionResidency] as? String,
                       NFKModelResidencyStaged)
    }

    func testAReloadBuildsAgainWithTheSameOptions() throws {
        try requireMLXRuntime()
        let log = NFKMLXBuildLog()
        NFKMLXModelRegistry.register(name: Self.moduleName, consumes: [NFKModelLoadOptionPrecision]) { _, options in
            log.record(options)
            let offset: Float = options.precision == .float32 ? 0.25 : 0.5
            return NFKMLXModuleBackend(identifier: Self.moduleName) { image in clip(image * 0.5 + offset, min: 0, max: 1) }
        }
        let backend = try NFKMLXModelRegistry.backend(named: Self.moduleName, weightsURL: placeholderURL,
                                                      options: [NFKModelLoadOptionPrecision: NFKModelPrecisionFloat32])
        XCTAssertTrue(backend is NFKMLXModuleBackend, "a lifecycle backend keeps its class")
        backend.unload?()
        XCTAssertEqual(backend.isLoaded, false)
        var error: NSError?
        XCTAssertTrue(NFKInferencePrepare(backend, &error), "\(String(describing: error))")
        XCTAssertEqual(backend.isLoaded, true)
        XCTAssertEqual(log.all.count, 2, "the reload ran the factory again")
        XCTAssertEqual(log.all.map(\.precision), [.float32, .float32], "with the options of the first build")
        XCTAssertEqual(NFKInferenceLoadOptionValues(backend)[NFKModelLoadOptionPrecision] as? String,
                       NFKModelPrecisionFloat32)
    }

    func testLoadOptionValuesAreAsRequestedAndModelInfoNamesTheResolvedResidency() throws {
        NFKMLXModelRegistry.register(name: Self.plainName, consumes: [NFKModelLoadOptionResidency]) { _, options in
            options.reportResidency(.staged)
            return NFKMLXGenerationBackend(identifier: Self.plainName, inputKeys: [NFKInputPrompt], parameterKeys: [],
                                           outputKeys: [NFKOutputText]) { _, _ in NFKInferenceResult(outputs: [:]) }
        }
        let requested = [NFKModelLoadOptionResidency: NFKModelResidencyAutomatic]
        let generator = try NFKMLXModelRegistry.backend(named: Self.plainName, weightsURL: nil, options: requested)
        XCTAssertEqual(NFKInferenceLoadOptionValues(generator)[NFKModelLoadOptionResidency] as? String,
                       NFKModelResidencyAutomatic, "the values are the options as requested")
        XCTAssertEqual(generator.modelInfo?[NFKModelInfoResidency] as? String, NFKModelResidencyStaged,
                       "the factory reported the residency the load chose")

        NFKMLXModelRegistry.register(name: Self.plainName, consumes: [NFKModelLoadOptionResidency]) { _, options in
            let backend = NFKMLXBuilderTestBackend(options: options)
            backend.pagesExperts = true
            return backend
        }
        let mixture = try NFKMLXModelRegistry.backend(named: Self.plainName, weightsURL: placeholderURL, options: requested)
        XCTAssertEqual(NFKInferenceLoadOptionValues(mixture)[NFKModelLoadOptionResidency] as? String,
                       NFKModelResidencyAutomatic)
        XCTAssertEqual(mixture.modelInfo?[NFKModelInfoResidency] as? String, NFKModelResidencyPaged,
                       "a backend that pages its experts shows the residency")
    }

    // MARK: The catalog's build entry

    /// A catalog holding one MLX descriptor named `registryName`, whose options are a residency choice, a
    /// weight quantization, and a precision set on the backend's `precision` property.
    private func catalog(registryName: String) -> NFKModelCatalog {
        let residency = NFKModelOption(key: NFKModelLoadOptionResidency, kind: NFKModelOptionKindChoice, title: "Residency",
                                       help: nil, defaultValue: NFKModelResidencyAutomatic, minimum: nil, maximum: nil,
                                       step: nil,
                                       choices: [NFKModelResidencyAutomatic, NFKModelResidencyResident,
                                                 NFKModelResidencyStaged].map {
                                           NFKModelOptionChoice(value: $0, title: nil, extensions: nil)
                                       },
                                       stage: NFKModelOptionStageLoad, group: nil, extensions: nil)
        let bits = NFKModelOption(key: NFKModelLoadOptionQuantizationBits, kind: NFKModelOptionKindChoice,
                                  title: "Weight quantization", help: nil, defaultValue: 0, minimum: nil, maximum: nil,
                                  step: nil,
                                  choices: [0, 8, 4].map { NFKModelOptionChoice(value: $0, title: nil, extensions: nil) },
                                  stage: NFKModelOptionStageLoad, group: nil, extensions: nil)
        let precision = NFKModelOption(key: NFKModelLoadOptionPrecision, kind: NFKModelOptionKindChoice,
                                       title: "Precision", help: nil, defaultValue: NFKModelPrecisionCheckpoint,
                                       minimum: nil, maximum: nil, step: nil,
                                       choices: [NFKModelPrecisionCheckpoint, NFKModelPrecisionFloat32].map {
                                           NFKModelOptionChoice(value: $0, title: nil, extensions: nil)
                                       },
                                       stage: NFKModelOptionStageLoad, group: nil, requirements: nil,
                                       propertyKeyPath: "precision", memoryBytesPerUnit: nil, extensions: nil)
        let variant = NFKModelVariant(identifier: "default", displayName: nil, isDefault: true, registryName: nil,
                                      source: nil, downloadByteCount: nil, residentByteCount: nil, peakByteCount: nil,
                                      parameterCount: nil, precision: nil, minimumMemoryByteCount: nil, extensions: nil)
        let descriptor = NFKModelDescriptor(identifier: "builder-test-model", displayName: "Builder test", family: "test",
                                            summary: nil, engine: NFKModelEngineMLX, registryName: registryName,
                                            providerClassName: nil, tasks: [NFKModelTaskTextGeneration],
                                            inputModalities: nil, outputModalities: nil, inputs: nil, outputs: nil,
                                            options: [residency, bits, precision], variants: [variant], license: nil,
                                            requirements: nil, residencies: nil, standIn: false, introducedVersion: nil,
                                            extensions: nil)
        let catalog = NFKModelCatalog()
        catalog.addDescriptor(descriptor)
        return catalog
    }

    func testTheCatalogBuildsAnMLXDescriptorThroughTheMLXBuilder() throws {
        NFKMLXModelDescriptors.registerBackendBuilder()
        XCTAssertNotNil(NFKModelCatalog.backendBuilder(forEngine: NFKModelEngineMLX))
        let log = NFKMLXBuildLog()
        registerPlain(log: log)
        let catalog = catalog(registryName: Self.plainName)
        let backend = try catalog.backend(forModelIdentifier: "builder-test-model",
                                          options: [NFKModelLoadOptionResidency: NFKModelResidencyStaged,
                                                    NFKModelLoadOptionQuantizationBits: "4",
                                                    NFKModelLoadOptionPrecision: NFKModelPrecisionFloat32,
                                                    NFKModelLoadOptionIgnoreMemoryBudget: true],
                                          location: placeholderURL)
        let received = try XCTUnwrap(log.all.first)
        XCTAssertEqual(received.residency, .staged)
        XCTAssertEqual(received.quantizationBits, 4, "the catalog resolved the text to the choice")
        let built = try XCTUnwrap((backend as? NFKMLXReloadableBackend)?.wrappedBackend as? NFKMLXBuilderTestBackend)
        XCTAssertEqual(built.precision, .float32, "the property option set the enumeration by its case")
        let values = NFKInferenceLoadOptionValues(backend)
        XCTAssertEqual(values[NFKModelLoadOptionResidency] as? String, NFKModelResidencyStaged)
        XCTAssertEqual(values[NFKModelLoadOptionPrecision] as? String, NFKModelPrecisionFloat32)
    }

    // A measured build that reports no residency of its own read its weights in: it is resident, and the
    // model information says so beside the bytes it measured.
    func testAMeasuredBuildReportsItsResidency() throws {
        NFKMLXModelDescriptors.registerBackendBuilder()
        NFKMLXModelRegistry.register(name: Self.plainName, consumes: [NFKModelLoadOptionPrecision]) { _, _ in
            NFKMLXGenerationBackend(identifier: Self.plainName, inputKeys: [NFKInputPrompt], parameterKeys: [],
                                    outputKeys: [NFKOutputText]) { _, _ in NFKInferenceResult(outputs: [:]) }
        }
        let catalog = catalog(registryName: Self.plainName)
        let backend = try catalog.backend(forModelIdentifier: "builder-test-model",
                                          options: [NFKModelLoadOptionIgnoreMemoryBudget: true],
                                          location: placeholderURL)
        let info = try XCTUnwrap(backend.modelInfo)
        XCTAssertEqual(info[NFKModelInfoResidency] as? String, NFKModelResidencyResident)
        XCTAssertNotNil(info[NFKModelInfoResidentBytes], "beside the measured bytes")
    }

    func testAPropertyOptionSetsTheStableDiffusionBackendsPrecisionWithoutADownload() throws {
        NFKMLXModelDescriptors.registerBackendBuilder()
        NFKMLXModelRegistry.register(name: Self.stableDiffusionName) { _ in
            NFKMLXBackend(model: .sdxlTurbo)
        }
        let catalog = catalog(registryName: Self.stableDiffusionName)
        let backend = try catalog.backend(forModelIdentifier: "builder-test-model",
                                          options: [NFKModelLoadOptionPrecision: NFKModelPrecisionFloat32,
                                                    NFKModelLoadOptionIgnoreMemoryBudget: true],
                                          location: nil)
        let stableDiffusion = try XCTUnwrap(backend as? NFKMLXBackend)
        XCTAssertEqual(stableDiffusion.precision, .float32)
        XCTAssertFalse(stableDiffusion.isReady, "setting the precision downloads nothing")
    }

    func testAPropertyOptionTheFactoryConsumesIsNotSetAgain() throws {
        NFKMLXModelDescriptors.registerBackendBuilder()
        let log = NFKMLXBuildLog()
        NFKMLXModelRegistry.register(name: Self.plainName, consumes: [NFKModelLoadOptionPrecision]) { _, options in
            log.record(options)
            return NFKMLXGenerationBackend(identifier: Self.plainName, inputKeys: [NFKInputPrompt], parameterKeys: [],
                                           outputKeys: [NFKOutputText]) { _, _ in NFKInferenceResult(outputs: [:]) }
        }
        let catalog = catalog(registryName: Self.plainName)
        XCTAssertNoThrow(try catalog.backend(forModelIdentifier: "builder-test-model",
                                             options: [NFKModelLoadOptionPrecision: NFKModelPrecisionFloat32,
                                                       NFKModelLoadOptionIgnoreMemoryBudget: true],
                                             location: nil),
                         "a backend without the property builds when its factory reads the option")
        XCTAssertEqual(log.all.first?.precision, .float32)
    }

    func testAnUnregisteredNameIsRefusedWithNoBuilder() throws {
        NFKMLXModelDescriptors.registerBackendBuilder()
        let catalog = catalog(registryName: "builder-test-unregistered")
        XCTAssertThrowsError(try catalog.backend(forModelIdentifier: "builder-test-model",
                                                 options: [NFKModelLoadOptionIgnoreMemoryBudget: true],
                                                 location: placeholderURL)) { error in
            XCTAssertEqual((error as NSError).domain, NFKModelCatalogErrorDomain)
            XCTAssertEqual((error as NSError).code, NFKModelCatalogError.Code.noBuilder.rawValue)
        }
    }

    // MARK: Option registrations

    func testRegisterAllRegistersTheFamiliesThatReadLoadOptions() {
        NFKMLXReferenceModels.registerAll()
        XCTAssertNotNil(NFKModelCatalog.backendBuilder(forEngine: NFKModelEngineMLX))
        let decoder = NFKMLXModelRegistry.consumedOptionKeys(forModelNamed: "qwen3-0.6b")
        XCTAssertTrue(decoder.isSuperset(of: [NFKModelLoadOptionResidency, NFKModelLoadOptionPrecision,
                                              NFKModelLoadOptionQuantizationBits, NFKModelLoadOptionQuantizationGroupSize,
                                              NFKModelLoadOptionContextLength, NFKModelLoadOptionCacheBits,
                                              NFKModelLoadOptionCacheGroupSize, NFKModelLoadOptionDraftModel,
                                              NFKModelLoadOptionDraftBits]), "\(decoder)")
        XCTAssertTrue(NFKMLXModelRegistry.consumedOptionKeys(forModelNamed: "gpt-oss-20b")
            .contains(NFKModelLoadOptionQuantizationBits))
        XCTAssertEqual(NFKMLXModelRegistry.consumedOptionKeys(forModelNamed: "flux.1-dev"), [NFKModelLoadOptionResidency])
        XCTAssertEqual(NFKMLXModelRegistry.consumedOptionKeys(forModelNamed: "wan"),
                       [NFKModelLoadOptionPrecision, NFKModelLoadOptionResidency])
        XCTAssertEqual(NFKMLXModelRegistry.consumedOptionKeys(forModelNamed: "whisper-base"), [NFKModelLoadOptionPrecision])
        XCTAssertEqual(NFKMLXModelRegistry.consumedOptionKeys(forModelNamed: "stable-diffusion-v1-5"),
                       [NFKModelLoadOptionPrecision])
        XCTAssertEqual(NFKMLXModelRegistry.consumedOptionKeys(forModelNamed: "lcm-lora-sdv1-5"),
                       [NFKModelLoadOptionPrecision])
        XCTAssertTrue(NFKMLXModelRegistry.consumedOptionKeys(forModelNamed: "gemma-3-1b-it")
            .isSuperset(of: [NFKModelLoadOptionResidency, NFKModelLoadOptionPrecision]))
        XCTAssertTrue(NFKMLXModelRegistry.consumedOptionKeys(forModelNamed: "deepseek-v4.1-flash")
            .isSuperset(of: [NFKModelLoadOptionResidency, "paging", "expertCacheBytes"]))
        XCTAssertEqual(NFKMLXModelRegistry.consumedOptionKeys(forModelNamed: "real-esrgan-x4"), [],
                       "a family without load options keeps its plain factory")
    }

    func testEveryWhisperReleaseHasASize() {
        for name in NFKMLXOptionRegistrations.registryNames(families: ["whisper"]) {
            XCTAssertNotNil(NFKMLXOptionRegistrations.whisperVariants[name], name)
        }
    }

    func testTheDeepSeekOptionsTakeTheValuesTheirPropertiesName() {
        let options = NFKMLXOptionRegistrations.deepSeekLoadOptions(NFKMLXLoadOptions([
            NFKModelLoadOptionResidency: NFKModelResidencyPaged, "speculates": true, "expertCacheBytes": 1 << 30,
            NFKModelLoadOptionContextLength: 4096,
        ]))
        XCTAssertEqual(options.residency, .paged)
        XCTAssertTrue(options.speculates)
        XCTAssertEqual(options.expertCacheBytes, 1 << 30)
        XCTAssertFalse(options.quantizesActivations)
    }

    // Each argument load option an MLX descriptor declares is one its registered factory reads, so no
    // declared choice is silently ignored.
    func testEveryDeclaredArgumentLoadOptionIsConsumed() {
        NFKMLXReferenceModels.registerAll()
        let skipped: Set<String> = [NFKModelLoadOptionVariant, NFKModelLoadOptionIgnoreMemoryBudget]
        for descriptor in NFKMLXModelDescriptors.modelDescriptors() {
            let declared = Set(descriptor.loadOptions.filter { $0.propertyKeyPath == nil }.map(\.key)).subtracting(skipped)
            guard !declared.isEmpty else {
                continue
            }
            let names = descriptor.variants.isEmpty ? [descriptor.registryName ?? descriptor.identifier]
                : descriptor.variants.map { $0.registryName ?? descriptor.registryName ?? descriptor.identifier }
            for name in Set(names) {
                let consumed = NFKMLXModelRegistry.consumedOptionKeys(forModelNamed: name)
                XCTAssertTrue(declared.isSubset(of: consumed),
                              "\(descriptor.identifier) declares \(declared.subtracting(consumed).sorted()) that \(name) does not read")
            }
        }
    }
}
