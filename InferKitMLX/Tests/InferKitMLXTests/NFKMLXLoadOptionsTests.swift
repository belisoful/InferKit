//
//  NFKMLXLoadOptionsTests.swift
//  InferKitMLXTests
//
//  The load options a registered factory reads: each reading of the resolved values, the generation
//  defaults, the draft's location, the residency a load reports, and the runtime weight quantization on a
//  tiny random-weight decoder.
//

import XCTest
import InferKit
import MLX
import MLXNN
@testable import InferKitMLX

final class NFKMLXLoadOptionsTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    private static let quantizedName = "load-options-test-quantized-decoder"

    override func tearDown() {
        NFKMLXModelRegistry.unregisterModel(named: Self.quantizedName)
        if NFKMLXGPU.metalLibraryURL != nil {
            NFKMLXGPU.synchronize()
            NFKMLXGPU.clearCache()
        }
        super.tearDown()
    }

    // MARK: Readings

    func testEachReadingFollowsItsKey() {
        let options = NFKMLXLoadOptions([
            NFKModelLoadOptionResidency: NFKModelResidencyPaged,
            NFKModelLoadOptionPrecision: NFKModelPrecisionFloat32,
            NFKModelLoadOptionQuantizationBits: 4,
            NFKModelLoadOptionQuantizationGroupSize: 32,
            NFKModelLoadOptionContextLength: 8192,
            NFKModelLoadOptionCacheBits: 8,
            NFKModelLoadOptionCacheGroupSize: 128,
            NFKModelLoadOptionDraftModel: "qwen3-0.6b",
            NFKModelLoadOptionDraftBits: 8,
        ])
        XCTAssertEqual(options.residency, .paged)
        XCTAssertEqual(options.precisionName, NFKModelPrecisionFloat32)
        XCTAssertEqual(options.precision, .float32)
        XCTAssertEqual(options.halfPrecision, false)
        XCTAssertEqual(options.quantizationBits, 4)
        XCTAssertEqual(options.quantizationGroupSize, 32)
        XCTAssertEqual(options.contextLength, 8192)
        XCTAssertEqual(options.cacheBits, 8)
        XCTAssertEqual(options.cacheGroupSize, 128)
        XCTAssertEqual(options.draftModel, "qwen3-0.6b")
        XCTAssertEqual(options.draftBits, 8)
    }

    func testAbsentKeysLeaveTheFactoryItsDefaults() {
        let options = NFKMLXLoadOptions()
        XCTAssertEqual(options.residency, .automatic)
        XCTAssertNil(options.precision, "a factory keeps its own precision")
        XCTAssertNil(options.halfPrecision)
        XCTAssertNil(options.quantizationBits)
        XCTAssertEqual(options.quantizationGroupSize, NFKMLXLoadOptions.defaultQuantizationGroupSize)
        XCTAssertEqual(NFKMLXLoadOptions.defaultQuantizationGroupSize, 64)
        XCTAssertNil(options.contextLength)
        XCTAssertNil(options.cacheBits)
        XCTAssertEqual(options.cacheGroupSize, 64)
        XCTAssertNil(options.draftModel)
        XCTAssertNil(options.draftBits)
        XCTAssertNil(options.draftURL)
    }

    func testZeroBitsAndNoneMeanNothing() {
        let options = NFKMLXLoadOptions([NFKModelLoadOptionQuantizationBits: 0, NFKModelLoadOptionCacheBits: 0,
                                         NFKModelLoadOptionDraftBits: 0, NFKModelLoadOptionDraftModel: "none"])
        XCTAssertNil(options.quantizationBits)
        XCTAssertNil(options.cacheBits)
        XCTAssertNil(options.draftBits)
        XCTAssertNil(options.draftModel)
        XCTAssertNil(NFKMLXLoadOptions([NFKModelLoadOptionDraftModel: ""]).draftModel)
    }

    func testNumbersGivenAsTextAreRead() {
        let options = NFKMLXLoadOptions([NFKModelLoadOptionContextLength: "2048", NFKModelLoadOptionQuantizationBits: "8"])
        XCTAssertEqual(options.contextLength, 2048)
        XCTAssertEqual(options.quantizationBits, 8)
    }

    func testThePrecisionValuesMapToTheMLXPrecisions() {
        XCTAssertEqual(NFKMLXLoadOptions.weightPrecision(named: NFKModelPrecisionFloat32), .float32)
        XCTAssertEqual(NFKMLXLoadOptions.weightPrecision(named: NFKModelPrecisionCheckpoint), .checkpoint)
        XCTAssertEqual(NFKMLXLoadOptions.weightPrecision(named: NFKModelPrecisionBFloat16), .checkpoint)
        XCTAssertEqual(NFKMLXLoadOptions.weightPrecision(named: NFKModelPrecisionFloat16), .checkpoint)
        XCTAssertNil(NFKMLXLoadOptions.weightPrecision(named: "int8"))
        XCTAssertEqual(NFKMLXLoadOptions([NFKModelLoadOptionPrecision: NFKModelPrecisionBFloat16]).halfPrecision, true)
    }

    func testEachResidencyRoundTripsThroughItsOptionValue() {
        let values = [NFKModelResidencyAutomatic, NFKModelResidencyResident, NFKModelResidencyStaged,
                      NFKModelResidencyPaged, NFKModelResidencyStreamed]
        let residencies: [NFKMLXResidency] = [.automatic, .resident, .staged, .paged, .streamed]
        for (value, residency) in zip(values, residencies) {
            XCTAssertEqual(NFKMLXResidency(optionValue: value), residency, value)
            XCTAssertEqual(residency.optionValue, value)
        }
        XCTAssertNil(NFKMLXResidency(optionValue: "swapped"))
        XCTAssertEqual(NFKMLXLoadOptions([NFKModelLoadOptionResidency: "swapped"]).residency, .automatic)
    }

    // A factory offering bfloat16 beside the published precision converts for bfloat16, keeps the release
    // as stored for none or checkpoint, and refuses a precision it has no parity check for.
    func testTheBFloat16CastRefusesWhatItHasNoParityCheckFor() throws {
        XCTAssertNil(try NFKMLXLoadOptions().bfloat16Cast(of: "m"))
        XCTAssertNil(try NFKMLXLoadOptions([NFKModelLoadOptionPrecision: NFKModelPrecisionCheckpoint]).bfloat16Cast(of: "m"))
        XCTAssertEqual(try NFKMLXLoadOptions([NFKModelLoadOptionPrecision: NFKModelPrecisionBFloat16]).bfloat16Cast(of: "m"),
                       .bfloat16)
        for precision in [NFKModelPrecisionFloat16, NFKModelPrecisionFloat32] {
            XCTAssertThrowsError(try NFKMLXLoadOptions([NFKModelLoadOptionPrecision: precision]).bfloat16Cast(of: "m"), precision)
        }
        XCTAssertThrowsError(try NFKMLXLoadOptions([NFKModelLoadOptionPrecision: NFKModelPrecisionBFloat16])
            .bfloat16Cast(of: "m", offersBFloat16: false))
    }

    // A factory that reads float32 itself and offers float16 converts for float16, leaves the published
    // precision and float32 to its own reading, and refuses bfloat16, naming what it loads at.
    func testTheCastConvertsAnOfferedPrecisionAndRefusesAnUncheckedOne() throws {
        func cast(_ precision: String?) throws -> DType? {
            try NFKMLXLoadOptions(precision.map { [NFKModelLoadOptionPrecision: $0] } ?? [:])
                .cast(of: "m", offering: [NFKModelPrecisionFloat16: .float16], reading: [NFKModelPrecisionFloat32])
        }
        XCTAssertNil(try cast(nil))
        XCTAssertNil(try cast(NFKModelPrecisionCheckpoint))
        XCTAssertNil(try cast(NFKModelPrecisionFloat32))
        XCTAssertEqual(try cast(NFKModelPrecisionFloat16), .float16)
        XCTAssertThrowsError(try cast(NFKModelPrecisionBFloat16)) { error in
            XCTAssertTrue("\(error)".contains("loads as published or at float16 or at float32"), "\(error)")
        }
    }

    func testTheDraftBitsKeepAFactoryDefaultOnlyWhenAbsent() {
        XCTAssertEqual(NFKMLXLoadOptions().draftBits(orDefault: 4), 4)
        XCTAssertNil(NFKMLXLoadOptions([NFKModelLoadOptionDraftBits: 0]).draftBits(orDefault: 4))
        XCTAssertEqual(NFKMLXLoadOptions([NFKModelLoadOptionDraftBits: 8]).draftBits(orDefault: 4), 8)
    }

    // MARK: Generation defaults

    func testTheGenerationOptionsCarryTheLoadDefaults() {
        var base = NFKMLXGenerationOptions()
        base.maxTokens = 99
        base.cacheQuantization = .init(bits: 4, groupSize: 64)
        let options = NFKMLXLoadOptions([NFKModelLoadOptionContextLength: 4096, NFKModelLoadOptionCacheBits: 8,
                                         NFKModelLoadOptionCacheGroupSize: 32]).generationOptions(base: base)
        XCTAssertEqual(options.maxTokens, 99, "the base's other options stay")
        XCTAssertEqual(options.contextWindow, 4096)
        XCTAssertEqual(options.cacheQuantization?.bits, 8)
        XCTAssertEqual(options.cacheQuantization?.groupSize, 32)
        let unquantized = NFKMLXLoadOptions([NFKModelLoadOptionCacheBits: 0]).generationOptions(base: base)
        XCTAssertNil(unquantized.cacheQuantization, "cache bits 0 is an unquantized cache")
        XCTAssertEqual(NFKMLXLoadOptions().generationOptions(base: base).cacheQuantization?.bits, 4,
                       "options without cache keys keep the base's cache")
    }

    func testTheLanguageBackendReadsTheLoadDefaultsARequestCarries() {
        var options = NFKMLXGenerationOptions()
        let defaulted = NFKInferenceRequest(inputs: [NFKInputPrompt: "hi"],
                                            parameters: [NFKModelLoadOptionContextLength: 2048,
                                                         NFKModelLoadOptionCacheBits: 8])
        NFKMLXLanguageBackend.applyMLXParameters(from: defaulted, to: &options)
        XCTAssertEqual(options.contextWindow, 2048)
        XCTAssertEqual(options.cacheQuantization?.bits, 8)

        var overridden = NFKMLXGenerationOptions()
        let both = NFKInferenceRequest(inputs: [NFKInputPrompt: "hi"],
                                       parameters: [NFKModelLoadOptionContextLength: 2048,
                                                    NFKMLXGenerationParameterKey.contextWindow: 512])
        NFKMLXLanguageBackend.applyMLXParameters(from: both, to: &overridden)
        XCTAssertEqual(overridden.contextWindow, 512, "the MLX key wins over the load default")
    }

    // MARK: Draft location

    func testTheDraftLocationIsAURLAPathOrACatalogModel() throws {
        let url = URL(fileURLWithPath: "/tmp/draft-release")
        XCTAssertEqual(try NFKMLXLoadOptions.draftLocation(url, in: NFKModelCatalog()), url)
        XCTAssertEqual(try NFKMLXLoadOptions.draftLocation("/tmp/draft-release", in: NFKModelCatalog()), url)
        XCTAssertNil(try NFKMLXLoadOptions.draftLocation(nil, in: NFKModelCatalog()))
        XCTAssertNil(try NFKMLXLoadOptions.draftLocation("none", in: NFKModelCatalog()))

        let catalog = NFKModelCatalog()
        catalog.addDescriptor(NFKModelDescriptor(identifier: "draft-test", displayName: nil, family: "test", summary: nil,
                                                 engine: NFKModelEngineMLX, registryName: nil, providerClassName: nil,
                                                 tasks: [NFKModelTaskTextGeneration], inputModalities: nil,
                                                 outputModalities: nil, inputs: nil, outputs: nil, options: nil,
                                                 variants: nil, license: nil, requirements: nil, residencies: nil,
                                                 standIn: false, introducedVersion: nil, extensions: nil))
        let located = URL(fileURLWithPath: "/tmp/located-draft")
        catalog.locator = { descriptor, _, _ in descriptor.identifier == "draft-test" ? located : nil }
        XCTAssertEqual(try NFKMLXLoadOptions.draftLocation("draft-test", in: catalog), located)

        XCTAssertThrowsError(try NFKMLXLoadOptions.draftLocation("no-such-model", in: catalog)) { error in
            XCTAssertEqual((error as NSError).domain, NFKModelCatalogErrorDomain)
            XCTAssertEqual((error as NSError).code, NFKModelCatalogError.Code.unknownModel.rawValue)
        }
        catalog.locator = { _, _, _ in nil }
        XCTAssertThrowsError(try NFKMLXLoadOptions.draftLocation("draft-test", in: catalog)) { error in
            XCTAssertEqual((error as NSError).code, NFKModelCatalogError.Code.invalidOptions.rawValue,
                           "a draft without files is refused")
        }
    }

    // MARK: Reporting

    func testAReportedResidencyLeavesTheRequestedValues() {
        let options = NFKMLXLoadOptions([NFKModelLoadOptionResidency: NFKModelResidencyAutomatic,
                                         NFKModelLoadOptionPrecision: NFKModelPrecisionFloat32])
        XCTAssertNil(options.resolvedResidency)
        let copy = options
        copy.reportResidency(.staged)
        XCTAssertEqual(options.resolvedResidency, .staged, "every copy of the options shares the report")
        XCTAssertEqual(options.values[NFKModelLoadOptionResidency] as? String, NFKModelResidencyAutomatic,
                       "the values keep the residency as requested")
    }

    // MARK: Runtime weight quantization

    /// A tiny random-weight decoder, the same on every build.
    private static func tinyDecoder() -> NFKMLXLanguageNet {
        MLXRandom.seed(11)
        let net = NFKMLXLanguage.makeNet(.tiny)
        eval(net)
        return net
    }

    private func weightBytes(_ net: NFKMLXLanguageNet) -> Int {
        NFKMLXModelDescription.info(of: [net])[NFKModelInfoWeightBytes] as? Int ?? 0
    }

    func testTheQuantizationOptionPacksTheDecodersLinearLayers() throws {
        try requireMLXRuntime()
        let net = Self.tinyDecoder()
        let linearCount = net.leafModules().flattened().filter { $0.1 is Linear }.count
        let before = weightBytes(net)
        try NFKMLXLoadOptions([NFKModelLoadOptionQuantizationBits: 0]).applyWeightQuantization(to: net)
        XCTAssertEqual(weightBytes(net), before, "no quantization leaves the decoder as loaded")
        XCTAssertFalse(net.leafModules().flattened().contains { $0.1 is QuantizedLinear })

        try NFKMLXLoadOptions([NFKModelLoadOptionQuantizationBits: 4]).applyWeightQuantization(to: net)
        let layers = net.leafModules().flattened().map(\.1)
        XCTAssertEqual(layers.filter { $0 is QuantizedLinear }.count, linearCount,
                       "every Linear layer of the tiny decoder becomes a QuantizedLinear")
        XCTAssertTrue(layers.compactMap { $0 as? QuantizedLinear }.allSatisfy { $0.bits == 4 && $0.groupSize == 64 })
        XCTAssertLessThan(weightBytes(net), before, "the packed decoder holds fewer bytes")
    }

    func testARegistryBuildWithTheQuantizationOptionHoldsAQuantizedDecoder() throws {
        try requireMLXRuntime()
        NFKMLXModelRegistry.register(name: Self.quantizedName, consumes: [NFKModelLoadOptionQuantizationBits]) { _, options in
            let net = Self.tinyDecoder()
            try options.applyWeightQuantization(to: net)
            return NFKMLXLanguageBackend(net: net, tokenizer: nil, identifier: Self.quantizedName,
                                         options: options.generationOptions())
        }
        let plain = try NFKMLXModelRegistry.backend(named: Self.quantizedName, weightsURL: nil, options: nil)
        let packed = try NFKMLXModelRegistry.backend(named: Self.quantizedName, weightsURL: nil,
                                                     options: [NFKModelLoadOptionQuantizationBits: 8])
        let plainInfo = try XCTUnwrap(plain.modelInfo)
        let packedInfo = try XCTUnwrap(packed.modelInfo)
        XCTAssertNil(plainInfo[NFKModelInfoQuantizationBits])
        XCTAssertEqual(packedInfo[NFKModelInfoQuantizationBits] as? Int, 8)
        XCTAssertEqual(packedInfo[NFKModelInfoQuantizationGroupSize] as? Int, 64)
        let plainBytes = try XCTUnwrap(plainInfo[NFKModelInfoWeightBytes] as? Int)
        let packedBytes = try XCTUnwrap(packedInfo[NFKModelInfoWeightBytes] as? Int)
        XCTAssertLessThan(packedBytes, plainBytes)
        XCTAssertEqual(NFKInferenceLoadOptionValues(packed)[NFKModelLoadOptionQuantizationBits] as? Int, 8)
    }
}
