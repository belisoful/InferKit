//
//  NFKMLXCatalogLanguageTests.swift
//  InferKitMLXTests
//
//  The language models' catalog entries. Building descriptors and the closure backends touches no MLX;
//  the two comparisons against built networks use tiny geometries and skip without a Metal library.
//

import XCTest
import InferKit
@testable import InferKitMLX

final class NFKMLXCatalogLanguageTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    override func tearDown() {
        NFKMLXGPU.clearCache()
        super.tearDown()
    }

    private var descriptors: [NFKModelDescriptor] {
        NFKMLXCatalogLanguage.entries.compactMap { try? $0.descriptor() }
    }

    private func descriptor(_ name: String) throws -> NFKModelDescriptor {
        try XCTUnwrap(descriptors.first { $0.registryName == name }, "no language entry registers \(name)")
    }

    /// Asserts that the entry registered as `name` declares exactly `backend`'s keys.
    private func assertKeys(of name: String, inputs: Set<String>, parameters: Set<String>, outputs: Set<String>,
                            file: StaticString = #filePath, line: UInt = #line) throws {
        let descriptor = try descriptor(name)
        XCTAssertEqual(Set(descriptor.inputs.map(\.key)), inputs, "\(name) inputs", file: file, line: line)
        // The entries here carry no load profile, and a decoder also takes the catalog's context and cache
        // load options as request defaults, so those keys are left out of the backend's side.
        let loadDefaults: Set<String> = [NFKModelLoadOptionContextLength, NFKModelLoadOptionCacheBits,
                                         NFKModelLoadOptionCacheGroupSize]
        XCTAssertEqual(Set(descriptor.requestOptions.map(\.key)), parameters.subtracting(loadDefaults),
                       "\(name) request options", file: file, line: line)
        XCTAssertEqual(Set(descriptor.outputs.map(\.key)), outputs, "\(name) outputs", file: file, line: line)
    }

    func testEveryLanguageEntryBuildsADescriptor() {
        for entry in NFKMLXCatalogLanguage.entries {
            XCTAssertNoThrow(try entry.descriptor(), entry.identifier)
        }
        XCTAssertEqual(descriptors.count, NFKMLXCatalogLanguage.entries.count)
    }

    func testEachIdentifierIsUniqueAndIsItsRegistryName() {
        let identifiers = descriptors.map(\.identifier)
        XCTAssertEqual(Set(identifiers).count, identifiers.count, "duplicate identifiers in \(identifiers)")
        for descriptor in descriptors {
            XCTAssertEqual(descriptor.registryName, descriptor.identifier)
            XCTAssertEqual(descriptor.engine, NFKModelEngineMLX, descriptor.identifier)
            XCTAssertFalse(descriptor.isStandIn, descriptor.identifier)
            XCTAssertNotNil(descriptor.license, descriptor.identifier)
            XCTAssertFalse(descriptor.variants.isEmpty, descriptor.identifier)
            XCTAssertEqual(descriptor.variants.filter(\.isDefault).count, 1, descriptor.identifier)
            for variant in descriptor.variants {
                XCTAssertEqual(variant.registryName, descriptor.registryName, "\(descriptor.identifier) \(variant.identifier)")
                XCTAssertEqual(variant.source?.kind, NFKModelSourceKindHuggingFace, descriptor.identifier)
            }
        }
    }

    func testEveryLanguageRegistryNameIsRegisteredAfterRegisterAll() {
        NFKMLXReferenceModels.registerAll()
        let registered = Set(NFKMLXModelRegistry.registeredModelNames)
        for descriptor in descriptors {
            let names = [descriptor.registryName].compactMap { $0 } + descriptor.variants.compactMap(\.registryName)
            for name in names {
                XCTAssertTrue(registered.contains(name), "\(descriptor.identifier) names unregistered \(name)")
            }
        }
    }

    func testTheGemma4MultimodalEntryDeclaresTheBackendsKeys() throws {
        try assertKeys(of: NFKMLXGemmaLanguage.gemmaMultimodalModelName,
                       inputs: NFKMLXGemmaLanguage.multimodalInputKeys,
                       parameters: NFKMLXGemmaLanguage.multimodalParameterKeys,
                       outputs: [NFKOutputText, NFKOutputUsage])
    }

    func testTheTimesFMEntryDeclaresTheForecastKeys() throws {
        try assertKeys(of: NFKMLXTimesFM.modelName, inputs: [NFKMLXForecastKey.series],
                       parameters: NFKMLXTimesFMParameterKey.all.union([NFKMLXForecastKey.horizon]),
                       outputs: [NFKOutputStructured])
        try assertKeys(of: NFKMLXChronos.modelName, inputs: [NFKMLXForecastKey.series],
                       parameters: [NFKMLXForecastKey.horizon], outputs: [NFKOutputStructured])
    }

    func testTheImageQuestionEntriesDeclareTheBackendsKeys() throws {
        let backend = NFKMLXGenerationBackend.imageQuestion(identifier: "image-question") { _, _, _ in "" }
        let names = ["smolvlm2-256m", NFKMLXSmolVLM.modelName, "smolvlm2-2.2b", NFKMLXQwen3VL.modelName, "qwen3-vl-8b",
                     "qwen3-vl-32b", "qwen3-vl-30b-a3b", NFKMLXPixtral.modelName]
        for name in names {
            try assertKeys(of: name, inputs: backend.supportedInputKeys, parameters: backend.supportedParameterKeys,
                           outputs: backend.supportedOutputKeys)
        }
    }

    func testTheDecisionEntriesDeclareTheBackendsKeys() throws {
        let backend = NFKMLXDecisionBackend(model: NFKFixedDecisionModel())
        let names = ["laya", "laya-typed-decisions", "laya-multilingual", "open-jev-2b", "open-jev-9b", "open-jev-27b-v1.1",
                     "open-jev-deberta-v3-large"]
        for name in names {
            try assertKeys(of: name, inputs: backend.supportedInputKeys, parameters: backend.supportedParameterKeys,
                           outputs: backend.supportedOutputKeys)
        }
    }

    func testTheMADLADReleasesFetchTheShardIndex() throws {
        XCTAssertEqual(NFKMLXMADLAD.weightFiles, ["model.safetensors", "model.safetensors.index.json"])
        for name in [NFKMLXMADLAD.modelName, "madlad400-7b-mt"] {
            let files = try XCTUnwrap(try descriptor(name).defaultVariant?.source?.files)
            XCTAssertTrue(files.contains { $0.path == "model.safetensors.index.json" && $0.isOptional }, name)
        }
    }

    func testGraniteTinyIsRegisteredAndPagesItsExperts() throws {
        NFKMLXReferenceModels.registerAll()
        XCTAssertTrue(NFKMLXModelRegistry.isModelRegistered(NFKMLXGraniteHybrid.graniteTinyModelName))
        let tiny = try descriptor(NFKMLXGraniteHybrid.graniteTinyModelName)
        XCTAssertEqual(tiny.residencies, [NFKModelResidencyResident, NFKModelResidencyPaged])
        XCTAssertEqual(tiny.defaultVariant?.source?.repository, "ibm-granite/granite-4.0-h-tiny")
        let dense = try descriptor(NFKMLXGraniteHybrid.graniteModelName)
        XCTAssertEqual(dense.residencies, [NFKModelResidencyResident])
        XCTAssertEqual(dense.defaultVariant?.source?.repository, "ibm-granite/granite-4.0-h-1b")
    }

    func testTheDecisionModelsAnswerTheDecisionTask() throws {
        for name in ["laya", "laya-typed-decisions", "laya-multilingual", "open-jev-2b", "open-jev-9b",
                     "open-jev-27b-v1.1", "open-jev-deberta-v3-large"] {
            XCTAssertEqual(try descriptor(name).tasks, [NFKModelTaskDecision, NFKModelTaskClassification], name)
        }
    }

    func testOpenJevNamesItsBaseRepositoryAtTheRevisionItWasTrainedOver() throws {
        for variant in NFKMLXOpenJevVariant.allCases {
            let files = try XCTUnwrap(descriptors.first { $0.identifier.hasPrefix("open-jev-")
                && $0.defaultVariant?.source?.repository == variant.repository }?.defaultVariant?.source?.files)
            let base = files.filter { $0.repository == variant.baseRepository }
            XCTAssertEqual(base.map(\.path), NFKMLXOpenJev.baseFiles, variant.repository)
            XCTAssertTrue(base.allSatisfy { $0.revision == variant.baseRevision }, variant.repository)
        }
    }

    func testEachOPUSMTPairCarriesTheLicenseItsModelCardStates() throws {
        let opus = try descriptor(NFKMLXMarian.modelName)
        XCTAssertEqual(opus.license?.identifier, "Apache-2.0")
        for variant in opus.variants {
            let creativeCommons = ["opus-mt-en-de", "opus-mt-zh-en"].contains(variant.identifier)
            XCTAssertEqual(opus.license(for: variant)?.identifier, creativeCommons ? "CC-BY-4.0" : "Apache-2.0",
                           variant.identifier)
            XCTAssertEqual(variant.license != nil, creativeCommons, variant.identifier)
        }
    }

    func testEveryEntryThatTakesToolsListsTheToolCallingOutputs() {
        for descriptor in descriptors where descriptor.options.contains(where: { $0.key == NFKParameterTools }) {
            XCTAssertTrue(NFKMLXToolCalling.outputKeys.isSubset(of: Set(descriptor.outputs.map(\.key))),
                          descriptor.identifier)
        }
    }

    func testTheDenseDecoderEntriesDeclareTheBackendsKeys() throws {
        try requireMLXRuntime()
        let backend = try NFKMLXLanguage.backend(weightsURL: nil, tokenizer: nil, configuration: .tiny)
        let names = ["qwen3-0.6b", "qwen3-1.7b", "qwen3-4b", "qwen3-14b", "qwen3-32b", "qwen3-30b-a3b", "qwen1.5-moe-a2.7b",
                     "mixtral-8x7b-instruct", "gpt-oss-20b", "mistral-small-24b-instruct-2501", "smollm2-135m-instruct-gguf"]
        for name in names {
            try assertKeys(of: name, inputs: backend.supportedInputKeys ?? [],
                           parameters: backend.supportedParameterKeys ?? [], outputs: backend.supportedOutputKeys ?? [])
        }
    }

    func testTheTextEmbeddingEntriesDeclareTheBackendsKeys() throws {
        try requireMLXRuntime()
        let backend = try NFKMLXQwen3Embedding.backend(weightsURL: nil, tokenizer: nil, configuration: .tiny)
        for name in [NFKMLXQwen3Embedding.modelName, NFKMLXEmbeddingGemma.modelName, NFKMLXQwen3VLEmbedder.modelName] {
            try assertKeys(of: name, inputs: backend.supportedInputKeys ?? [],
                           parameters: backend.supportedParameterKeys ?? [], outputs: backend.supportedOutputKeys ?? [])
        }
    }
}

/// A decision model that answers nothing, which builds a decision backend without weights.
private final class NFKFixedDecisionModel: NFKMLXDecisionModel {
    var decisionModelName: String { "fixed" }

    func decide(state: Any, identifiedQuestions: [(identifier: String, question: NFKDecisionQuestion)])
        throws -> (answers: [String: NFKDecisionAnswer], inputTokens: Int) {
        ([:], 0)
    }
}
