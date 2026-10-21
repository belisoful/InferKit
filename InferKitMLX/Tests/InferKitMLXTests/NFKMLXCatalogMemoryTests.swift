//
//  NFKMLXCatalogMemoryTests.swift
//  InferKitMLXTests
//
//  The memory facts, load options, draft choices, and presets the catalog declares. Building descriptors
//  and estimating their memory touches no MLX, so every test runs under `swift test` without a Metal
//  library.
//

import XCTest
import InferKit
@testable import InferKitMLX

final class NFKMLXCatalogMemoryTests: XCTestCase {

    private let descriptors = NFKMLXModelDescriptors.modelDescriptors()

    private func descriptor(_ identifier: String) throws -> NFKModelDescriptor {
        try XCTUnwrap(descriptors.first { $0.identifier == identifier }, "no descriptor \(identifier)")
    }

    private func measurement(tensorBytes: [String: [String: Int64]], opaqueBytes: [String: Int64] = [:],
                             layers: [String: NFKMLXMeasuredLayers] = [:], expertBytes: [String: [String: Int64]] = [:],
                             quantizableElements: [String: Int64] = [:],
                             geometry: NFKMLXMeasuredGeometry? = nil) -> NFKMLXMeasuredVariant {
        NFKMLXMeasuredVariant(model: "m", variant: "v", status: .measured, note: nil, commit: nil, downloadBytes: 100,
                              parameterCount: 10, tensorBytes: tensorBytes, opaqueBytes: opaqueBytes, layers: layers,
                              expertBytes: expertBytes, quantizableElements: quantizableElements, geometry: geometry)
    }

    // MARK: Facts

    func testEveryMeasuredVariantCarriesMemoryFactsAndItsDerivedFigures() throws {
        for row in NFKMLXCatalogMeasurements.measuredVariants where row.downloadBytes != nil
            && !(row.tensorBytes.isEmpty && row.opaqueBytes.isEmpty) {
            let key = "\(row.model)/\(row.variant)"
            let variant = try XCTUnwrap(try descriptor(row.model).variant(withIdentifier: row.variant), key)
            let facts = try XCTUnwrap(variant.memoryFacts, "\(key) has no memory facts")
            XCTAssertFalse(facts.stages.isEmpty, key)
            XCTAssertNotNil(facts.stages.first?.byteCountByPrecision[NFKModelPrecisionCheckpoint], key)
            XCTAssertNotNil(variant.residentByteCount, "\(key) has no derived resident figure")
            XCTAssertEqual(variant.minimumMemoryByteCount?.intValue,
                           NFKModelMemoryBudget.leastWorkingSetHoldingByteCount(variant.peakByteCount?.intValue ?? -1), key)
        }
    }

    func testAStageThatMixesPrecisionsStatesItsLoadedBytes() throws {
        let plan = NFKMLXLoadPlan(precision: .stored, directories: ["text_encoder": .float32, "transformer": .bfloat16],
                                  stages: [["text_encoder"], ["transformer", "vae"]])
        let row = measurement(tensorBytes: ["text_encoder": ["BF16": 1000], "transformer": ["F32": 4000],
                                            "vae": ["F32": 300], "scheduler": ["F32": 8]])
        let facts = try XCTUnwrap(NFKMLXVariantFacts(row, plan: plan, converted: false, recipe: NFKMLXFactsRecipe(),
                                                     estimatesStoredBytes: false))
        XCTAssertEqual(facts.facts.stages.map(\.identifier), ["text_encoder", "transformer"])
        XCTAssertEqual(facts.stagePrecisions, ["text_encoder": NFKModelPrecisionFloat32,
                                               "transformer": NFKMLXVariantFacts.loadedPrecision])
        let second = facts.facts.stages[1]
        // The transformer at bfloat16 and the VAE and the left-over scheduler as stored.
        XCTAssertEqual(second.byteCountByPrecision[NFKMLXVariantFacts.loadedPrecision]?.intValue, 2000 + 300 + 8)
        XCTAssertEqual(second.byteCountByPrecision[NFKModelPrecisionCheckpoint]?.intValue, 4308)
        XCTAssertEqual(second.title, "transformer, vae, scheduler")
        XCTAssertEqual(facts.facts.stages[0].byteCountByPrecision[NFKModelPrecisionFloat32]?.intValue, 2000)
    }

    func testFactsCarryOnlyWhatTheLoaderActsOn() throws {
        let geometry = NFKMLXMeasuredGeometry(layerCount: 2, attentionHeads: 4, keyValueHeads: 2, headDimension: 8,
                                              maximumPositions: 1024, vocabularySize: 50)
        let row = measurement(tensorBytes: ["": ["BF16": 1000]],
                              layers: ["": NFKMLXMeasuredLayers(count: 2, largestBytes: ["BF16": 400])],
                              expertBytes: ["": ["BF16": 600]], quantizableElements: ["": 300], geometry: geometry)
        let bare = try XCTUnwrap(NFKMLXVariantFacts(row, plan: .stored, converted: false, recipe: NFKMLXFactsRecipe(),
                                                    estimatesStoredBytes: false)).facts
        XCTAssertEqual(bare.stages[0].quantizableElementCount, 0)
        XCTAssertEqual(bare.stages[0].layerCount, 0)
        XCTAssertTrue(bare.stages[0].pageableByteCountByPrecision.isEmpty)
        XCTAssertNil(bare.keyValueByteCountPerToken)
        XCTAssertEqual(bare.vocabularySize?.intValue, 50)

        let recipe = NFKMLXFactsRecipe(stageIdentifier: "decoder", quantizes: true, pagesExperts: true, streamsLayers: true,
                                       keyValue: .dense)
        let full = try XCTUnwrap(NFKMLXVariantFacts(row, plan: .float32, converted: false, recipe: recipe,
                                                    estimatesStoredBytes: false)).facts
        let stage = full.stages[0]
        XCTAssertEqual(stage.identifier, "decoder")
        XCTAssertEqual(stage.quantizableElementCount, 300)
        XCTAssertEqual(stage.layerCount, 2)
        XCTAssertEqual(stage.largestLayerByteCountByPrecision[NFKModelPrecisionFloat32]?.intValue, 800)
        XCTAssertEqual(stage.pageableByteCountByPrecision[NFKModelPrecisionCheckpoint]?.intValue, 600)
        // Two layers, two key-value heads of eight elements, keys and values at 16 bits.
        XCTAssertEqual(full.keyValueByteCountPerToken?.intValue, 2 * 2 * 2 * 8 * 2)
        XCTAssertEqual(full.maximumContextLength?.intValue, 1024)
    }

    // MARK: Key-value geometry

    func testTheDenseRuleIsTheSizingOfTheBuiltConfigurations() {
        let configurations: [NFKMLXLanguageConfiguration] = [.qwen3_0_6B, .qwen3_1_7B, .qwen3_4B, .qwen3_8B, .qwen3_14B,
                                                             .qwen3_32B, .mistralSmall3]
        for configuration in configurations {
            let geometry = NFKMLXMeasuredGeometry(layerCount: configuration.layerCount,
                                                  attentionHeads: configuration.headCount,
                                                  keyValueHeads: configuration.keyValueHeadCount,
                                                  headDimension: configuration.headDimensions)
            let expected = NFKMLXModelSizing.keyValueBytesPerToken(of: configuration, precision: .checkpoint)
            XCTAssertEqual(NFKMLXKeyValueRule.dense.keyValueBytes(of: geometry)?.perToken, expected)
            var attention = geometry
            attention.layerTypes = Array(repeating: "full_attention", count: configuration.layerCount)
            XCTAssertEqual(NFKMLXKeyValueRule.attentionLayers.keyValueBytes(of: attention)?.perToken, expected)
        }
    }

    func testTheMeasuredQwen3GeometryIsTheBuiltConfigurations() throws {
        let pairs: [(String, NFKMLXLanguageConfiguration)] = [("qwen3-0.6b", .qwen3_0_6B), ("qwen3-1.7b", .qwen3_1_7B),
                                                              ("qwen3-4b", .qwen3_4B), ("qwen3-14b", .qwen3_14B),
                                                              ("qwen3-32b", .qwen3_32B)]
        for (identifier, configuration) in pairs {
            let facts = try XCTUnwrap(try descriptor(identifier).defaultVariant?.memoryFacts, identifier)
            XCTAssertEqual(facts.keyValueByteCountPerToken?.intValue,
                           NFKMLXModelSizing.keyValueBytesPerToken(of: configuration, precision: .checkpoint), identifier)
        }
    }

    func testAHybridCachesOnlyItsAttentionLayers() {
        let geometry = NFKMLXMeasuredGeometry(layerCount: 4, attentionHeads: 8, keyValueHeads: 2, headDimension: 64,
                                              layerTypes: ["mamba", "attention", "mamba", "mamba"])
        XCTAssertEqual(NFKMLXKeyValueRule.attentionLayers.keyValueBytes(of: geometry)?.perToken, 2 * 2 * 64 * 2)
        var linear = geometry
        linear.layerTypes = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
        XCTAssertEqual(NFKMLXKeyValueRule.attentionLayers.keyValueBytes(of: linear)?.perToken, 2 * 2 * 64 * 2)
    }

    func testGemma3BoundsItsSlidingLayersToTheWindow() {
        let configuration = NFKMLXGemma3Configuration.gemma3_4B
        let geometry = NFKMLXMeasuredGeometry(layerCount: configuration.layerCount, attentionHeads: configuration.headCount,
                                              keyValueHeads: configuration.keyValueHeadCount,
                                              headDimension: configuration.headDimensions,
                                              slidingWindow: configuration.slidingWindow)
        let full = configuration.layerTypes.filter { $0 == .full }.count
        let perLayer = 2 * configuration.keyValueHeadCount * configuration.headDimensions * 2
        let bytes = NFKMLXKeyValueRule.slidingWindow.keyValueBytes(of: geometry)
        XCTAssertEqual(bytes?.perToken, full * perLayer)
        XCTAssertEqual(bytes?.bounded, (configuration.layerCount - full) * perLayer * configuration.slidingWindow)
    }

    func testSharedLayersCacheNothingAndGlobalLayersTakeTheirOwnWidth() {
        // Gemma 4 E2B: 35 layers, the last 20 shared; of the first 15, every fifth is full attention at a
        // 512-wide head, the rest sliding at 256, one key-value head throughout.
        let types = (0 ..< 35).map { ($0 + 1) % 5 == 0 ? "full_attention" : "sliding_attention" }
        let geometry = NFKMLXMeasuredGeometry(layerCount: 35, attentionHeads: 8, keyValueHeads: 1, headDimension: 256,
                                              slidingWindow: 512, sharedKeyValueLayers: 20, globalHeadDimension: 512,
                                              layerTypes: types)
        XCTAssertEqual(NFKMLXKeyValueRule.sharedLayers.keyValueBytes(of: geometry)?.perToken, 12 * 1024 + 3 * 2048)
    }

    // MARK: Options

    func testEveryMemoryChoiceCarriesAFigureAndAutomaticNone() {
        for descriptor in descriptors {
            guard let model = descriptor.memoryModel else { continue }
            for key in model.optionKeys {
                guard let option = descriptor.option(withKey: key) else {
                    XCTFail("\(descriptor.identifier) names the undeclared option \(key)")
                    continue
                }
                for choice in option.choices {
                    let name = "\(descriptor.identifier) \(key)=\(choice.value)"
                    if key == model.residencyOptionKey && (choice.value as? String) == NFKModelResidencyAutomatic {
                        XCTAssertNil(choice.memory, name)
                        continue
                    }
                    XCTAssertNotNil(choice.memory, "\(name) has no figure")
                    if descriptor.variants.count > 1 {
                        XCTAssertEqual(Set(choice.memoryByVariant.keys), Set(descriptor.variants.map(\.identifier)), name)
                    }
                }
            }
        }
    }

    func testEachFamilyProfileValidatesItsDefaults() throws {
        var checked = Set<String>()
        for descriptor in descriptors {
            guard let entry = NFKMLXModelDescriptors.entries.first(where: { $0.identifier == descriptor.identifier }),
                  let profile = NFKMLXLoadProfile.profile(for: entry) else { continue }
            checked.insert(descriptor.family)
            let name = descriptor.identifier
            XCTAssertEqual(descriptor.problems(options: nil, stage: NFKModelOptionStageLoad).count, 0, name)
            let resolved = try descriptor.resolvedOptions(nil, stage: NFKModelOptionStageLoad)
            XCTAssertNotNil(try? descriptor.estimatedMemory(forOptions: resolved), name)
            if profile.precisions.count > 1 {
                XCTAssertEqual(resolved[NFKModelLoadOptionPrecision] as? String, profile.defaultPrecision, name)
            }
            if entry.residencies.count > 1 {
                XCTAssertEqual(resolved[NFKModelLoadOptionResidency] as? String, NFKModelResidencyAutomatic, name)
            }
            if profile.quantizes {
                XCTAssertEqual(resolved[NFKModelLoadOptionQuantizationBits] as? Int, 0, name)
            }
            let memoryKeys = Set(descriptor.memoryModel?.optionKeys ?? [])
            for option in descriptor.loadOptions where option.key != NFKModelLoadOptionVariant {
                XCTAssertTrue(memoryKeys.contains(option.key), "\(name) declares \(option.key) outside its memory model")
            }
        }
        XCTAssertEqual(checked.subtracting(NFKMLXLoadProfile.familyProfiles.keys).sorted(), [])
        XCTAssertEqual(Set(NFKMLXLoadProfile.familyProfiles.keys).subtracting(checked).sorted(), [],
                       "a family profile no descriptor takes")
    }

    func testTheQuantizationHelpSaysItApproximatesThePublishedWeights() throws {
        let option = try XCTUnwrap(try descriptor("qwen3-4b").option(withKey: NFKModelLoadOptionQuantizationBits))
        XCTAssertEqual(option.stage, NFKModelOptionStageLoad)
        XCTAssertEqual(option.defaultValue as? Int, 0)
        XCTAssertEqual(option.choices.map { $0.value as? Int }, [0, 8, 6, 5, 4, 3, 2])
        XCTAssertTrue(option.help?.contains("approximates the published weights") == true)
        for key in [NFKModelLoadOptionCacheBits, NFKModelLoadOptionDraftBits] {
            let widths = try XCTUnwrap(try descriptor("qwen3-4b").option(withKey: key)).choices.map { $0.value as? Int }
            XCTAssertEqual(widths, [0, 8, 6, 5, 4, 3, 2], key)
        }
    }

    func testADecoderHeldToItsBFloat16ReferenceLoadsAsPublished() throws {
        for identifier in ["qwen3-4b", "qwen3-30b-a3b", "gemma-3n-e2b-it"] {
            let decoder = try descriptor(identifier)
            let precision = try XCTUnwrap(decoder.option(withKey: NFKModelLoadOptionPrecision), identifier)
            XCTAssertEqual(precision.defaultValue as? String, NFKModelPrecisionCheckpoint, identifier)
            let precise = try XCTUnwrap(decoder.preset(withIdentifier: "precise"), identifier)
            XCTAssertEqual(precise.loadOptions[NFKModelLoadOptionPrecision] as? String, NFKModelPrecisionFloat32)
            XCTAssertGreaterThan(try decoder.estimatedMemory(forOptions: precise.loadOptions).residentByteCount,
                                 try decoder.estimatedMemory(forOptions: nil).residentByteCount, identifier)
        }
        let gemma3 = try descriptor("gemma-3-1b-it")
        XCTAssertEqual(gemma3.option(withKey: NFKModelLoadOptionPrecision)?.defaultValue as? String,
                       NFKModelPrecisionFloat32, "Gemma 3's text decoder has no bfloat16 record")
        XCTAssertNil(gemma3.preset(withIdentifier: "precise"))
    }

    func testEachDraftSharesTheFamilyAndVocabularyAndIsSmaller() throws {
        var drafting = 0
        for descriptor in descriptors {
            guard let key = descriptor.memoryModel?.draftOptionKey,
                  let option = descriptor.option(withKey: key) else { continue }
            drafting += 1
            let drafted = try XCTUnwrap(descriptor.defaultVariant)
            for choice in option.choices {
                let draft = try self.descriptor(try XCTUnwrap(choice.value as? String))
                let variant = try XCTUnwrap(draft.defaultVariant)
                XCTAssertEqual(draft.family, descriptor.family)
                XCTAssertEqual(variant.memoryFacts?.vocabularySize, drafted.memoryFacts?.vocabularySize)
                XCTAssertLessThan(variant.parameterCount?.int64Value ?? .max, drafted.parameterCount?.int64Value ?? 0)
                XCTAssertGreaterThan(choice.memory?.residentByteCount ?? 0, drafted.residentByteCount?.intValue ?? .max,
                                     "\(descriptor.identifier) with \(draft.identifier)")
            }
        }
        XCTAssertGreaterThan(drafting, 0, "no decoder takes a draft model")
        let qwen = try descriptor("qwen3-4b").option(withKey: NFKModelLoadOptionDraftModel)
        XCTAssertEqual(qwen?.choices.compactMap { $0.value as? String }, ["qwen3-0.6b", "qwen3-1.7b"])
    }

    // MARK: Presets

    func testEverySuggestedPresetValidates() {
        var count = 0
        for descriptor in descriptors {
            for preset in descriptor.presets {
                count += 1
                let name = "\(descriptor.identifier)@\(preset.identifier)"
                XCTAssertTrue(preset.isSuggested, name)
                XCTAssertEqual(descriptor.problems(options: preset.loadOptions, stage: NFKModelOptionStageLoad)
                                   .map(\.message), [], name)
                XCTAssertEqual(descriptor.problems(options: preset.requestDefaults, stage: NFKModelOptionStageRequest)
                                   .map(\.message), [], name)
            }
        }
        XCTAssertGreaterThan(count, 0)
    }

    func testTheFastDecoderPresetQuantizesTheWeightsAndTheCache() throws {
        let decoder = try descriptor("qwen3-4b")
        let fast = try XCTUnwrap(decoder.preset(withIdentifier: "fast"))
        XCTAssertTrue(fast.changesLoadOptions)
        let quick = try decoder.estimatedMemory(forOptions: fast.loadOptions)
        let published = try decoder.estimatedMemory(forOptions: nil)
        XCTAssertLessThan(quick.residentByteCount, published.residentByteCount / 2)
    }
}
