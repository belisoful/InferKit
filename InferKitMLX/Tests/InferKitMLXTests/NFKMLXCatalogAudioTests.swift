//
//  NFKMLXCatalogAudioTests.swift
//  InferKitMLXTests
//
//  The audio models' catalog entries. Building and reading descriptors touches no MLX, so every test but
//  the random-weight backend comparison runs under `swift test` without a Metal library.
//

import XCTest
import InferKit
@testable import InferKitMLX

final class NFKMLXCatalogAudioTests: XCTestCase {

    /// Small models whose registered factory builds random weights from a nil URL, cheap enough to build
    /// in a test and compare against their entries.
    private static let randomWeightNames = [
        "whisper-tiny", "silero-vad", "vad-marblenet", "gtcrn", "metricgan-plus", "cmgan", "mpsenet", "basic-pitch",
        "hft-transformer", "conv-tasnet", "denoiser", "deepfilternet3", "nuwave2", "apollo", "snac",
    ]

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    override func tearDown() {
        NFKMLXGPU.clearCache()
        super.tearDown()
    }

    private var descriptors: [NFKModelDescriptor] {
        NFKMLXCatalogAudio.entries.compactMap { try? $0.descriptor() }
    }

    private func entry(_ name: String) throws -> NFKMLXCatalogEntry {
        try XCTUnwrap(NFKMLXCatalogAudio.entries.first { $0.identifier == name }, "no audio entry named \(name)")
    }

    private func assertPorts(_ entry: NFKMLXCatalogEntry, inputs: Set<String>? = nil, outputs: Set<String>? = nil,
                             parameters: Set<String>? = nil, file: StaticString = #filePath, line: UInt = #line) {
        if let inputs {
            XCTAssertEqual(Set(entry.inputs.map(\.key)), inputs, entry.identifier, file: file, line: line)
        }
        if let outputs {
            XCTAssertEqual(Set(entry.outputs.map(\.key)), outputs, entry.identifier, file: file, line: line)
        }
        if let parameters {
            XCTAssertEqual(Set(entry.parameters.map(\.key)), parameters, entry.identifier, file: file, line: line)
        }
    }

    func testEveryAudioEntryBuildsADescriptor() throws {
        for entry in NFKMLXCatalogAudio.entries {
            XCTAssertNoThrow(try entry.descriptor(), entry.identifier)
        }
        XCTAssertEqual(descriptors.count, NFKMLXCatalogAudio.entries.count)
    }

    func testEveryAudioEntryIsNamedByItsRegistryNameAndIsUnique() {
        let identifiers = NFKMLXCatalogAudio.entries.map(\.identifier)
        XCTAssertEqual(Set(identifiers).count, identifiers.count, "duplicate identifiers in \(identifiers)")
        for descriptor in descriptors {
            XCTAssertEqual(descriptor.registryName, descriptor.identifier)
            XCTAssertEqual(descriptor.engine, NFKModelEngineMLX, descriptor.identifier)
            XCTAssertFalse(descriptor.isStandIn, descriptor.identifier)
            XCTAssertFalse(descriptor.variants.isEmpty, descriptor.identifier)
            XCTAssertNotNil(descriptor.license, descriptor.identifier)
            XCTAssertFalse(descriptor.tasks.isEmpty, descriptor.identifier)
            for variant in descriptor.variants {
                XCTAssertEqual(variant.registryName, descriptor.registryName, descriptor.identifier)
                XCTAssertNotNil(variant.source, "\(descriptor.identifier) \(variant.identifier)")
            }
            XCTAssertEqual(descriptor.variants.filter(\.isDefault).count, 1, descriptor.identifier)
        }
    }

    func testEveryAudioRegistryNameIsRegisteredAfterRegisterAll() {
        NFKMLXReferenceModels.registerAll()
        let registered = Set(NFKMLXModelRegistry.registeredModelNames)
        for descriptor in descriptors {
            let names = [descriptor.registryName].compactMap { $0 } + descriptor.variants.compactMap(\.registryName)
            for name in names {
                XCTAssertTrue(registered.contains(name), "\(descriptor.identifier) names unregistered \(name)")
            }
        }
        for name in NFKMLXCatalogAudio.additionalModelNames {
            XCTAssertTrue(registered.contains(name), "\(name) is not registered")
            XCTAssertTrue(descriptors.contains { $0.identifier == name }, "\(name) has no entry")
        }
    }

    func testEveryWhisperReleaseHasItsOwnName() {
        let names = NFKMLXCatalogAudio.whisperReleases.map(\.name)
        XCTAssertEqual(Set(names).count, names.count)
        XCTAssertTrue(names.contains(NFKMLXWhisper.modelName))
        for release in ["base", "small", "medium", "large-v2", "large-v3", "large-v3-turbo"] {
            XCTAssertTrue(names.contains("whisper-\(release)"), release)
        }
    }

    func testEachWhisperReleaseDownloadsItsTokenizerBesideItsCheckpoint() throws {
        for release in NFKMLXCatalogAudio.whisperReleases {
            let source = try XCTUnwrap(try entry(release.name).descriptor().defaultVariant?.source)
            XCTAssertEqual(source.kind, NFKModelSourceKindHuggingFace)
            XCTAssertEqual(source.repository, release.repository)
            XCTAssertEqual(source.revision?.count, 40, "the tokenizer is read at a pinned commit")
            XCTAssertEqual(source.files.map(\.path), NFKMLXWhisper.tokenizerFiles + [URL(string: release.url)!.lastPathComponent])
            XCTAssertEqual(Set(source.files.prefix(3).map(\.role)), [NFKModelFileRoleTokenizer])
            let checkpoint = try XCTUnwrap(source.files.last)
            XCTAssertEqual(checkpoint.url?.absoluteString, release.url, "the weights stay OpenAI's release")
            XCTAssertEqual(checkpoint.role, NFKModelFileRoleWeights)
            XCTAssertTrue(checkpoint.overridesSource)
        }
    }

    func testEachWhisperReleaseOffersTheLanguagesItsTokensName() throws {
        XCTAssertEqual(NFKMLXWhisper.languageCodes(for: .base).count, 99)
        XCTAssertEqual(NFKMLXWhisper.languageCodes(for: .largeV3).count, 100)
        XCTAssertEqual(NFKMLXWhisper.languageCodes(for: .largeV3).last, "yue")
        let options = try entry("whisper-large-v3").descriptor().options
        let language = try XCTUnwrap(options.first { $0.key == NFKParameterSourceLanguage })
        XCTAssertEqual(language.choices.count, 100)
        XCTAssertNil(language.defaultValue, "absent, the model detects the language")
        let task = try XCTUnwrap(options.first { $0.key == NFKMLXGenerationParameterKey.speechTask })
        XCTAssertEqual(task.choices.map { $0.value as? String }, ["transcribe", "translate"])
    }

    func testTheDirectoryModelsRefuseANilURL() {
        NFKMLXReferenceModels.registerAll()
        for name in NFKMLXCatalogAudio.wav2Vec2Releases.map(\.name)
            + [NFKMLXWav2Vec2Bert.modelName, NFKMLXCatalogAudio.kokoroName, NFKMLXCatalogAudio.chatterboxName] {
            XCTAssertThrowsError(try NFKMLXModelRegistry.backend(named: name, weightsURL: nil), name)
        }
    }

    func testTheFileRolesFollowTheFileNames() {
        XCTAssertEqual(NFKMLXCatalogAudio.role(of: "config.json"), NFKModelFileRoleConfig)
        XCTAssertEqual(NFKMLXCatalogAudio.role(of: "preprocessor_config.json"), NFKModelFileRoleConfig)
        XCTAssertEqual(NFKMLXCatalogAudio.role(of: "tokenizer_config.json"), NFKModelFileRoleTokenizer)
        XCTAssertEqual(NFKMLXCatalogAudio.role(of: "qwen_7B/qwen3-8B-tokenizer-music/vocab.json"), NFKModelFileRoleTokenizer)
        XCTAssertEqual(NFKMLXCatalogAudio.role(of: "vocoder/diffusion_pytorch_model.safetensors"), NFKModelFileRoleWeights)
        XCTAssertEqual(NFKMLXCatalogAudio.role(of: "voices/af_heart.pt"), NFKModelFileRoleOther)
    }

    func testTheEntriesFollowTheStaticKeySets() throws {
        for release in NFKMLXCatalogAudio.whisperReleases {
            assertPorts(try entry(release.name), inputs: [NFKInputAudio], outputs: [NFKOutputText, NFKOutputSegments, NFKOutputLanguage],
                        parameters: [NFKMLXGenerationParameterKey.timestamps, NFKParameterSourceLanguage,
                                     NFKMLXGenerationParameterKey.speechTask])
        }
        assertPorts(try entry(NFKMLXFastSpeech2.voiceModelName), inputs: NFKMLXFastSpeech2.voiceInputKeys,
                    outputs: [NFKOutputAudio], parameters: [])
        assertPorts(try entry(NFKMLXCatalogAudio.chatterboxName), inputs: [NFKInputPrompt, NFKInputMessages],
                    outputs: [NFKOutputAudio],
                    parameters: NFKMLXChatterbox.requestParameterKeys.union([NFKParameterSampleRate]))
        assertPorts(try entry(NFKMLXCatalogAudio.kokoroName), inputs: [NFKInputPrompt, NFKInputMessages],
                    outputs: [NFKOutputAudio], parameters: [NFKParameterSampleRate])
        assertPorts(try entry(NFKMLXConvTasNet.modelName),
                    outputs: Set(NFKMLXConvTasNetConfiguration.libri2Mix16k.speakerNames))
        let demucs = NFKMLXDemucsConfiguration()
        assertPorts(try entry(NFKMLXDemucs.modelName), outputs: Set(demucs.stemNames.prefix(demucs.stems)))
        for variant in [NFKMLXHTDemucsVariant.fourStem, .sixStem] {
            let spec = NFKMLXHTDemucs.specs(for: variant)
            assertPorts(try entry(spec.name),
                        outputs: Set(NFKMLXHTDemucsConfiguration.stemNames.prefix(spec.configuration.sources)))
        }
        for variant in [NFKMLXSNACVariant.speech24kHz, .music32kHz, .music44kHz] {
            XCTAssertNoThrow(try entry(NFKMLXSNAC.specs(for: variant).name))
        }
        for variant in [NFKMLXMuScriptorVariant.small, .medium, .large] {
            XCTAssertNoThrow(try entry(NFKMLXMuScriptor.registeredName(for: variant)))
        }
    }

    func testEachWhisperSizeReportsItsRegistryName() {
        for release in NFKMLXCatalogAudio.whisperReleases where release.variant != .large {
            XCTAssertEqual(NFKMLXWhisper.name(for: release.variant), release.name)
        }
        XCTAssertEqual(NFKMLXWhisper.name(for: .large), "whisper-large")
    }

    func testARegisteredWhisperSizeReportsItsRegistryName() throws {
        try requireMLXRuntime()
        NFKMLXReferenceModels.registerAll()
        for name in [NFKMLXWhisper.modelName, "whisper-base"] {
            XCTAssertEqual(try NFKMLXModelRegistry.backend(named: name, weightsURL: nil).backendIdentifier, name)
        }
        XCTAssertEqual(try NFKMLXWhisper.backend(variant: .small, weightsURL: nil).backendIdentifier, "whisper-small")
    }

    func testARegisteredWav2Vec2ReleaseReportsItsRegistryName() throws {
        try requireMLXRuntime()
        NFKMLXReferenceModels.registerAll()
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        var headless = NFKMLXWav2Vec2TrainingTests.tiny
        headless.vocabularySize = nil
        try NFKMLXWav2Vec2.save(NFKMLXWav2Vec2Net(headless), tokenizer: nil, toDirectoryURL: directory)
        for name in ["wav2vec2-base", "hubert-base-ls960"] {
            XCTAssertEqual(try NFKMLXModelRegistry.backend(named: name, weightsURL: directory).backendIdentifier, name)
        }
        XCTAssertEqual(try NFKMLXWav2Vec2.backend(directoryURL: directory).backendIdentifier, "wav2vec2")
    }

    func testCanaryReadsTheArchiveTheRepositoryServes() throws {
        let source = try XCTUnwrap(try entry(NFKMLXCanary.modelName).descriptor().defaultVariant?.source)
        XCTAssertFalse(source.requiresConversion)
        XCTAssertEqual(source.files.map(\.path), NFKMLXCanary.requiredFiles + NFKMLXCanary.weightFiles)

        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        XCTAssertThrowsError(try NFKMLXCanary.releaseCheckpoint(inDirectory: directory)) { error in
            XCTAssertTrue("\(error)".contains("canary-1b-v2.nemo"), "\(error)")
        }
    }

    func testVoiceRestoreNamesTheVocoderFromItsOwnRepository() throws {
        let source = try XCTUnwrap(try entry(NFKMLXVoiceRestoreFactory.modelName).descriptor().defaultVariant?.source)
        let vocoder = try XCTUnwrap(source.files.first { $0.path == NFKMLXVoiceRestoreFactory.vocoderFile })
        XCTAssertEqual(vocoder.repository, NFKMLXVoiceRestoreFactory.vocoderRepo)
        XCTAssertTrue(vocoder.overridesSource)
        XCTAssertEqual(vocoder.role, NFKModelFileRoleWeights)
    }

    func testTheMusicBackendDeclaresItsEntrysPortsWithoutWeights() throws {
        let backend = try NFKMLXMusic3.backend(directoryURL: FileManager.default.temporaryDirectory)
        assertPorts(try entry(NFKMLXMusic3.modelName), inputs: backend.supportedInputKeys ?? [],
                    outputs: backend.supportedOutputKeys ?? [], parameters: backend.supportedParameterKeys ?? [])
    }

    func testEachSmallRandomWeightBackendDeclaresItsEntrysPortsAndOptions() throws {
        try requireMLXRuntime()
        NFKMLXReferenceModels.registerAll()
        for name in Self.randomWeightNames {
            let backend = try NFKMLXModelRegistry.backend(named: name, weightsURL: nil)
            assertPorts(try entry(name), inputs: backend.supportedInputKeys ?? [],
                        outputs: backend.supportedOutputKeys ?? [], parameters: backend.supportedParameterKeys ?? [])
        }
    }
}
