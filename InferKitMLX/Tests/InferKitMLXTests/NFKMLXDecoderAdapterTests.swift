//
//  NFKMLXDecoderAdapterTests.swift
//  InferKitMLXTests
//
//  Decoder adapters held beside the weights: a PEFT or mlx-lm file adds its scaled low-rank product to each
//  layer it names, adapters stack at the strengths a run gives them, and a quantized layer holds the detour
//  beside its packed weight. Measured against a decoder whose weights carry the merged product.
//

import XCTest
import InferKit
import MLX
import MLXNN
@testable import InferKitMLX

final class NFKMLXDecoderAdapterTests: XCTestCase {

    private var directories: [URL] = []

    override func tearDown() {
        for directory in directories {
            try? FileManager.default.removeItem(at: directory)
        }
        directories = []
        super.tearDown()
    }

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    private enum Layout {
        case peft
        case mlxLM
    }

    /// `[rank, inputs]` and `[outputs, rank]` halves drawn from `seed`.
    private func halves(_ seed: UInt64, rank: Int = 4, inputs: Int, outputs: Int) -> (down: MLXArray, up: MLXArray) {
        (MLXRandom.normal([rank, inputs], key: MLXRandom.key(seed)) * 0.2,
         MLXRandom.normal([outputs, rank], key: MLXRandom.key(seed + 100)) * 0.2)
    }

    /// An adapter file in `layout` holding `layers`' halves, with `configuration` written beside it.
    private func adapter(_ layers: [String: (down: MLXArray, up: MLXArray)], layout: Layout,
                         configuration: [String: Any]?) throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        directories.append(directory)
        var arrays = [String: MLXArray]()
        for (path, halves) in layers {
            switch layout {
            case .peft:
                arrays["base_model.model.\(path).lora_A.weight"] = halves.down
                arrays["base_model.model.\(path).lora_B.weight"] = halves.up
            case .mlxLM:
                arrays["\(path).lora_a"] = halves.down.transposed(1, 0)
                arrays["\(path).lora_b"] = halves.up.transposed(1, 0)
            }
        }
        let file = directory.appendingPathComponent(layout == .peft ? "adapter_model.safetensors" : "adapters.safetensors")
        try MLX.save(arrays: arrays, url: file)
        if let configuration {
            try JSONSerialization.data(withJSONObject: configuration)
                .write(to: directory.appendingPathComponent("adapter_config.json"))
        }
        return file
    }

    private func tinyNet() -> NFKMLXLanguageNet {
        let net = NFKMLXLanguage.makeNet(.tiny)
        eval(net)
        return net
    }

    /// A copy of `net` whose weights at each path carry `delta` added.
    private func merged(_ net: NFKMLXLanguageNet, _ deltas: [String: MLXArray]) -> NFKMLXLanguageNet {
        let copy = NFKMLXLanguage.makeNet(net.configuration)
        copy.update(parameters: net.parameters())
        let weights = Dictionary(uniqueKeysWithValues: net.parameters().flattened())
        copy.update(parameters: ModuleParameters.unflattened(deltas.map { path, delta in
            ("\(path).weight", weights["\(path).weight"]! + delta)
        }))
        eval(copy)
        return copy
    }

    private let tokens = MLXArray([3, 17, 42, 99, 5, 260, 7].map(Int32.init)).reshaped([1, 7])

    private func worst(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a - b).max().item(Float.self)
    }

    func testAPEFTAdapterAddsItsScaledProductToEachLayerItNames() throws {
        try requireMLXRuntime()
        let net = tinyNet()
        let query = halves(1, inputs: 64, outputs: 64)
        let value = halves(3, inputs: 64, outputs: 32)
        let file = try adapter(["model.layers.0.self_attn.q_proj": query, "model.layers.1.self_attn.v_proj": value],
                               layout: .peft, configuration: ["r": 4, "lora_alpha": 8])
        let reference = merged(net, ["model.layers.0.self_attn.q_proj": matmul(query.up, query.down) * 2,
                                     "model.layers.1.self_attn.v_proj": matmul(value.up, value.down) * 2])
        let strengths = try NFKMLXDecoderAdapters.hold([(file, 1)], in: net)
        XCTAssertEqual(strengths, [1])
        XCTAssertLessThan(worst(net(tokens), reference(tokens)), 1e-4)
    }

    func testTheMLXLMLayoutGivesThePEFTLayoutsDetours() throws {
        try requireMLXRuntime()
        let peftNet = tinyNet()
        let mlxNet = NFKMLXLanguage.makeNet(.tiny)
        mlxNet.update(parameters: peftNet.parameters())
        let query = halves(5, rank: 8, inputs: 64, outputs: 64)
        let peft = try adapter(["model.layers.1.self_attn.q_proj": query], layout: .peft,
                               configuration: ["r": 8, "lora_alpha": 16, "use_rslora": false])
        let mlx = try adapter(["model.layers.1.self_attn.q_proj": query], layout: .mlxLM,
                              configuration: ["fine_tune_type": "lora", "lora_parameters": ["rank": 8, "scale": 2.0]])
        try NFKMLXDecoderAdapters.hold([(peft, 1)], in: peftNet)
        try NFKMLXDecoderAdapters.hold([(mlx, 1)], in: mlxNet)
        XCTAssertLessThan(worst(peftNet(tokens), mlxNet(tokens)), 1e-6)
    }

    func testHeldAdaptersStackAtTheStrengthsARunGivesThem() throws {
        try requireMLXRuntime()
        let net = tinyNet()
        let first = halves(7, inputs: 64, outputs: 64)
        let second = halves(9, inputs: 64, outputs: 64)
        let path = "model.layers.0.self_attn.q_proj"
        let configuration: [String: Any] = ["r": 4, "lora_alpha": 4]
        let files = [try adapter([path: first], layout: .peft, configuration: configuration),
                     try adapter([path: second], layout: .peft, configuration: configuration)]
        let base = net(tokens)
        eval(base)
        let loaded = try NFKMLXDecoderAdapters.hold([(files[0], 1), (files[1], 0.25)], in: net)
        XCTAssertEqual(loaded, [1, 0.25])

        func expected(_ a: Float, _ b: Float) -> MLXArray {
            merged(net, [path: matmul(first.up, first.down) * a + matmul(second.up, second.down) * b])(tokens)
        }
        // `merged` copies the adapted layer's base weight, which the adapted layer reports as its own.
        XCTAssertLessThan(worst(net(tokens), expected(1, 0.25)), 1e-4, "a run without strengths uses the loaded ones")
        let reversed = NFKMLXAdapterRun.$strengths.withValue([0.5, -1]) { net(tokens) }
        XCTAssertLessThan(worst(reversed, expected(0.5, -1)), 1e-4)
        let off = NFKMLXAdapterRun.$strengths.withValue([0, 0]) { net(tokens) }
        XCTAssertLessThan(worst(off, base), 1e-6, "strength 0 skips the detour")
    }

    func testAQuantizedLayerHoldsTheDetourBesideItsPackedWeight() throws {
        try requireMLXRuntime()
        let net = tinyNet()
        try NFKMLXQuantization.quantize(module: net, bits: 4, groupSize: 32)
        eval(net)
        let layer = { net.model.layers[0].attention.queryProjection }
        let x = MLXRandom.normal([1, 3, 64], key: MLXRandom.key(11))
        let before = layer()(x)
        eval(before)
        let query = halves(13, inputs: 64, outputs: 64)
        let file = try adapter(["model.layers.0.self_attn.q_proj": query], layout: .peft,
                               configuration: ["r": 4, "lora_alpha": 12])
        try NFKMLXDecoderAdapters.hold([(file, 0.5)], in: net)
        XCTAssertTrue(layer() is NFKMLXAdaptedQuantizedLinear)
        let detour = matmul(matmul(x, query.down.transposed(1, 0)), query.up.transposed(1, 0)) * (0.5 * 3)
        XCTAssertLessThan(worst(layer()(x), before + detour), 1e-4)
    }

    func testAnAdapterThatDoesNotFitIsRefusedBeforeAnyLayerChanges() throws {
        try requireMLXRuntime()
        let net = tinyNet()
        let path = "model.layers.0.self_attn.q_proj"
        let good = try adapter([path: halves(15, inputs: 64, outputs: 64)], layout: .peft,
                               configuration: ["r": 4, "lora_alpha": 8])
        let refused: [(String, URL)] = [
            ("no configuration", try adapter([path: halves(17, inputs: 64, outputs: 64)], layout: .peft, configuration: nil)),
            ("a DoRA", try adapter([path: halves(19, inputs: 64, outputs: 64)], layout: .peft,
                                   configuration: ["r": 4, "lora_alpha": 8, "use_dora": true])),
            ("whole layers", try adapter([path: halves(21, inputs: 64, outputs: 64)], layout: .peft,
                                         configuration: ["r": 4, "lora_alpha": 8, "modules_to_save": ["lm_head"]])),
            ("a layer the model lacks", try adapter(["model.layers.9.self_attn.q_proj": halves(23, inputs: 64, outputs: 64)],
                                                    layout: .peft, configuration: ["r": 4, "lora_alpha": 8])),
            ("the wrong shape", try adapter([path: halves(25, inputs: 32, outputs: 64)], layout: .peft,
                                            configuration: ["r": 4, "lora_alpha": 8])),
            ("an mlx-lm DoRA", try adapter([path: halves(27, inputs: 64, outputs: 64)], layout: .mlxLM,
                                           configuration: ["fine_tune_type": "dora", "lora_parameters": ["scale": 1.0]])),
            ("an AdaLoRA", try adapter([path: halves(43, inputs: 64, outputs: 64)], layout: .peft,
                                       configuration: ["peft_type": "ADALORA", "r": 4, "lora_alpha": 8])),
        ]
        for (reason, file) in refused {
            XCTAssertThrowsError(try NFKMLXDecoderAdapters.hold([(good, 1), (file, 1)], in: net), reason)
            XCTAssertFalse(net.model.layers[0].attention.queryProjection is NFKMLXAdaptedLinear,
                           "\(reason) leaves every layer as it was")
        }
    }

    func testTheDecoderEntriesTakeAdaptersTrainedOnTheirRepository() throws {
        NFKMLXReferenceModels.registerAll()
        let descriptors = Dictionary(uniqueKeysWithValues: NFKMLXModelDescriptors.modelDescriptors().map { ($0.identifier, $0) })
        let expected = ["qwen3-0.6b": "Qwen/Qwen3-0.6B", "qwen3-30b-a3b": "Qwen/Qwen3-30B-A3B",
                        "qwen3.5-2b": NFKMLXOpenJevVariant.twoB.baseRepository, "qwen3.5": "Qwen/Qwen3.5-4B"]
        for (identifier, repository) in expected {
            let descriptor = try XCTUnwrap(descriptors[identifier], identifier)
            XCTAssertEqual(descriptor.adapterBases, [repository], identifier)
            XCTAssertEqual(descriptor.option(withKey: NFKModelLoadOptionAdapters)?.stage, NFKModelOptionStageLoad, identifier)
            XCTAssertNotNil(descriptor.option(withKey: NFKParameterAdapterStrengths), identifier)
            XCTAssertEqual(descriptor.memoryModel?.adaptersOptionKey, NFKModelLoadOptionAdapters, identifier)
            XCTAssertTrue(NFKMLXModelRegistry.consumedOptionKeys(forModelNamed: identifier)
                .contains(NFKModelLoadOptionAdapters), identifier)
        }
        XCTAssertEqual(try XCTUnwrap(descriptors["smollm2-135m-instruct-gguf"]).adapterBases, [],
                       "a GGUF decoder reads no adapter file")
    }

    // A transformers Whisper adapter names transformers' modules; the port names them as OpenAI does.
    func testAWhisperAdapterReachesTheLayersItsTransformersNamesMean() throws {
        try requireMLXRuntime()
        var configuration = NFKMLXWhisperConfiguration()
        configuration.nAudioState = 32
        configuration.nAudioHead = 2
        configuration.nAudioLayer = 1
        configuration.nVocab = 50
        configuration.nTextState = 32
        configuration.nTextHead = 2
        configuration.nTextLayer = 1
        configuration.nTextCtx = 16
        let net = NFKMLXWhisperNet(configuration)
        eval(net)
        let query = halves(33, inputs: 32, outputs: 32)
        let file = try adapter(["model.encoder.layers.0.self_attn.q_proj": query,
                                "model.decoder.layers.0.encoder_attn.v_proj": halves(35, inputs: 32, outputs: 32),
                                "model.decoder.layers.0.fc1": halves(37, inputs: 32, outputs: 128)],
                               layout: .peft, configuration: ["r": 4, "lora_alpha": 8])
        let layers = { Dictionary(net.leafModules().flattened(), uniquingKeysWith: { first, _ in first }) }
        let x = MLXRandom.normal([1, 3, 32], key: MLXRandom.key(39))
        let before = (layers()["encoder.blocks.0.attn.query"] as! Linear)(x)
        eval(before)
        try NFKMLXDecoderAdapters.hold([(file, 1)], in: net, modulePath: NFKMLXWhisper.adapterModulePath)
        for path in ["encoder.blocks.0.attn.query", "decoder.blocks.0.cross_attn.value", "decoder.blocks.0.mlp.0"] {
            XCTAssertTrue(layers()[path] is NFKMLXAdaptedLinear, path)
        }
        let detour = matmul(matmul(x, query.down.transposed(1, 0)), query.up.transposed(1, 0)) * 2
        XCTAssertLessThan(worst((layers()["encoder.blocks.0.attn.query"] as! Linear)(x), before + detour), 1e-5)

        let tied = try adapter(["proj_out": halves(41, inputs: 32, outputs: 50)], layout: .peft,
                               configuration: ["r": 4, "lora_alpha": 8])
        XCTAssertThrowsError(try NFKMLXDecoderAdapters.hold([(tied, 1)], in: NFKMLXWhisperNet(configuration),
                                                            modulePath: NFKMLXWhisper.adapterModulePath),
                             "the output projection is the tied embedding, which the port holds once")
    }

    func testAKeptPromptCacheServesOnlyTheStrengthsItWasComputedAt() {
        let cache = NFKMLXPromptCache(layerCount: 2)
        cache.adapterStrengths = [1, 0.5]
        var options = NFKMLXGenerationOptions()
        options.adapterStrengths = [1, 0.5]
        XCTAssertTrue(cache.matches(layerCount: 2, options: options))
        options.adapterStrengths = [1, 0]
        XCTAssertFalse(cache.matches(layerCount: 2, options: options))
        options.adapterStrengths = nil
        XCTAssertFalse(cache.matches(layerCount: 2, options: options))
    }

    // A compiled single-position step would keep the strength it was traced at; with adapters held, the cached
    // pass follows each run's strength.
    func testTheHybridDecodersCachedPassFollowsEachRunsStrength() throws {
        try requireMLXRuntime()
        let configuration = NFKMLXHybridConfiguration(
            hiddenSize: 64, layerCount: 8, intermediateSize: 128, vocabularySize: 256, headCount: 4,
            keyValueHeadCount: 2, headDimensions: 16, ropeTheta: 10_000, partialRotaryFactor: 0.25,
            gatesAttentionOutput: true, linearKeyHeadCount: 2, linearKeyHeadDimensions: 8, linearValueHeadCount: 4,
            linearValueHeadDimensions: 8, linearConvolutionKernel: 4, fullAttentionInterval: 4)
        let net = NFKMLXHybridLanguage.makeNet(configuration)
        eval(net)
        let file = try adapter(["model.layers.0.linear_attn.in_proj_qkv": halves(29, inputs: 64, outputs: 64),
                                "model.layers.3.self_attn.q_proj": halves(31, inputs: 64, outputs: 128)],
                               layout: .peft, configuration: ["r": 4, "lora_alpha": 8])
        try NFKMLXDecoderAdapters.hold([(file, 1)], in: net)
        net.holdsAdapters = true
        let sequence = (0 ..< 12).map { ($0 * 37 + 5) % 256 }
        for strength: Float in [1, 0.3] {
            NFKMLXAdapterRun.$strengths.withValue([strength]) {
                let cache = NFKMLXHybridCache(configuration: configuration)
                _ = net.prefill(Array(sequence.dropLast()), cache: cache)
                let cached = net.lastLogits(MLXArray([Int32(sequence.last!)]).reshaped([1, 1]), cache: cache)[0]
                let full = net(MLXArray(sequence.map(Int32.init)).reshaped([1, sequence.count]))[0, sequence.count - 1]
                XCTAssertLessThan(worst(cached, full), 1e-4, "strength \(strength)")
            }
        }
    }
}
