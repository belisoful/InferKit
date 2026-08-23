//
//  NFKMLXHybridLanguageTests.swift
//  InferKitMLXTests
//
//  The hybrid decoder (Qwen3.5 / 3.6 / 3.8). The smallest release is 27B, about 54 GB at the precision
//  it ships in, so nothing here loads weights or measures numerics. What these tests establish is that
//  the module the code builds IS the model the release describes: every parameter is checked against
//  the released checkpoint's own safetensors headers, name by name and shape by shape.
//
//  A structural match is not a numeric one. Nothing here says the arithmetic is right.
//

import XCTest
import MLX
import MLXNN
@testable import InferKitMLX

final class NFKMLXHybridLanguageTests: XCTestCase {

    private lazy var config: [String: String] = {
        var merged = ProcessInfo.processInfo.environment
        let url = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".inferkit-validation.json")
        if let data = try? Data(contentsOf: url),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: String] {
            json.forEach { merged[$0.key] = $0.value }
        }
        return merged
    }()

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test`; run via xcodebuild")
    }

    /// A configuration small enough to build and run, keeping every structural ratio of the release.
    private var small: NFKMLXHybridConfiguration {
        NFKMLXHybridConfiguration(hiddenSize: 64, layerCount: 4, intermediateSize: 128,
                                  vocabularySize: 256, headCount: 4, keyValueHeadCount: 2,
                                  headDimensions: 16, ropeTheta: 10_000, partialRotaryFactor: 0.25,
                                  gatesAttentionOutput: true,
                                  linearKeyHeadCount: 2, linearKeyHeadDimensions: 8,
                                  linearValueHeadCount: 4, linearValueHeadDimensions: 8,
                                  linearConvolutionKernel: 4, fullAttentionInterval: 4)
    }

    // MARK: The layer pattern

    func testEveryFourthLayerAttendsFully() {
        let configuration = NFKMLXHybridConfiguration.qwen3_8_27B
        XCTAssertEqual(configuration.layerTypes.count, 64)
        for (index, kind) in configuration.layerTypes.enumerated() {
            let expected: NFKMLXHybridLayerKind = (index + 1) % 4 == 0 ? .fullAttention : .linearAttention
            XCTAssertEqual(kind, expected, "layer \(index)")
        }
        XCTAssertEqual(configuration.layerTypes.filter { $0 == .fullAttention }.count, 16,
                       "a quarter of the layers attend fully")
    }

    // Each layer carries exactly one branch: a linear layer has no `self_attn`, a full one has no
    // `linear_attn`. Building both would leave weights the checkpoint does not carry.
    func testALayerCarriesOnlyItsOwnBranch() throws {
        try requireMLXRuntime()
        let net = NFKMLXHybridLanguage.makeNet(small)
        let names = net.parameters().flattened().map(\.0)
        XCTAssertTrue(names.contains { $0.hasPrefix("model.layers.0.linear_attn.") })
        XCTAssertFalse(names.contains { $0.hasPrefix("model.layers.0.self_attn.") })
        XCTAssertTrue(names.contains { $0.hasPrefix("model.layers.3.self_attn.") })
        XCTAssertFalse(names.contains { $0.hasPrefix("model.layers.3.linear_attn.") })
    }

    // MARK: Derived widths

    // The fused projection carries queries and keys for the key heads and values for the value heads;
    // the released 27B numbers are what make it 10240 wide.
    func testTheFusedProjectionWidthFollowsTheHeadCounts() {
        let c = NFKMLXHybridConfiguration.qwen3_8_27B
        XCTAssertEqual(c.linearQKVWidth, 2 * 16 * 128 + 48 * 128)
        XCTAssertEqual(c.linearQKVWidth, 10240)
        XCTAssertEqual(c.linearValueWidth, 6144)
        XCTAssertEqual(c.rotaryDimensions, 64, "a quarter of a 256-wide head turns")
    }

    // MARK: Structure against the released checkpoint

    // The decisive test: the module's parameters, name by name and shape by shape, against the
    // headers of the released 27B checkpoint. No weights are read — only the safetensors headers,
    // which were captured with range requests.
    func testEveryParameterMatchesTheReleasedCheckpoint() throws {
        try requireMLXRuntime()
        guard let shapesPath = config["IK_SHAPES_QWEN3_8"],
              let configPath = config["IK_CONFIG_QWEN3_8"],
              let data = FileManager.default.contents(atPath: shapesPath),
              let reference = try JSONSerialization.jsonObject(with: data) as? [String: [Int]]
        else { throw XCTSkip("set IK_SHAPES_QWEN3_8 and IK_CONFIG_QWEN3_8") }

        let geometry = try NFKMLXHybridLanguage.configuration(
            fromHuggingFace: URL(fileURLWithPath: configPath))
        XCTAssertEqual(geometry.hiddenSize, 5120)
        XCTAssertEqual(geometry.layerCount, 64)
        XCTAssertEqual(geometry.linearValueHeadCount, 48)

        let net = NFKMLXHybridLanguage.makeNet(geometry)
        var checked = 0
        var missing = [String]()
        var mismatched = [String]()

        for (name, value) in net.parameters().flattened() {
            let key = NFKMLXHybridLanguage.referenceKey(for: name)
            guard let expected = reference[key] else { missing.append(key); continue }
            checked += 1
            // A depthwise convolution is the one place the layouts differ: PyTorch stores
            // [channels, 1, kernel] and MLX [channels, kernel, 1]. Compare it as the loader would.
            let ours = key.hasSuffix("conv1d.weight") ? [value.shape[0], value.shape[2], value.shape[1]]
                                                      : value.shape
            if ours != expected {
                mismatched.append("\(key): built \(value.shape), released \(expected)")
            }
        }

        print("VALIDATION structure qwen3.8-27B: \(checked) parameters checked, "
              + "\(missing.count) missing, \(mismatched.count) mismatched")
        XCTAssertTrue(mismatched.isEmpty, "shape mismatches:\n" + mismatched.prefix(8).joined(separator: "\n"))
        XCTAssertTrue(missing.isEmpty, "parameters absent from the release:\n"
                      + missing.prefix(8).joined(separator: "\n"))
        XCTAssertGreaterThan(checked, 800, "the decoder's parameters are all accounted for")
    }

    // The converse: what the release carries that this module does not. The vision tower and the
    // multi-token-prediction head are separate features, and this records that they are known rather
    // than overlooked.
    func testWhatIsDeliberatelyNotImplementedIsNamed() throws {
        guard let shapesPath = config["IK_SHAPES_QWEN3_8"],
              let data = FileManager.default.contents(atPath: shapesPath),
              let reference = try JSONSerialization.jsonObject(with: data) as? [String: [Int]]
        else { throw XCTSkip("set IK_SHAPES_QWEN3_8") }

        let decoder = reference.keys.filter { $0.hasPrefix("model.language_model.") || $0 == "lm_head.weight" }
        let vision = reference.keys.filter { $0.hasPrefix("model.visual.") }
        let prediction = reference.keys.filter { $0.hasPrefix("mtp.") }
        print("VALIDATION structure qwen3.8-27B: decoder \(decoder.count), "
              + "vision tower \(vision.count), multi-token head \(prediction.count)")

        XCTAssertGreaterThan(vision.count, 0, "the release is multimodal")
        XCTAssertGreaterThan(prediction.count, 0, "and carries a multi-token-prediction head")
        XCTAssertEqual(decoder.count + vision.count + prediction.count, reference.count,
                       "every tensor is either the decoder's or a named unimplemented part")
    }

    // MARK: It runs

    // A small configuration executes end to end, which proves the recurrence and the gated attention
    // are wired to each other. It says nothing about whether the arithmetic matches the reference.
    /// The bf16 guard the dense and Gemma stacks carry too: a `.checkpoint` load makes the module
    /// bf16, and the fused attention refuses a float32 mask against it.
    func testABF16ModuleStillForwards() throws {
        try requireMLXRuntime()
        let net = NFKMLXHybridLanguage.makeNet(small)
        let halved = Dictionary(uniqueKeysWithValues: net.parameters().flattened().map {
            ($0.0, $0.1.asType(.bfloat16))
        })
        net.update(parameters: ModuleParameters.unflattened(halved))
        let logits = net(MLXArray([Int32(1), 2, 3, 4, 5]).reshaped([1, 5]))
        eval(logits)
        XCTAssertTrue(logits.asType(.float32).sum().item(Float.self).isFinite)
    }

    func testASmallConfigurationRunsEndToEnd() throws {
        try requireMLXRuntime()
        let net = NFKMLXHybridLanguage.makeNet(small)
        let logits = net(MLXArray([Int32(1), 2, 3, 4, 5]).reshaped([1, 5]))
        eval(logits)
        XCTAssertEqual(logits.shape, [1, 5, small.vocabularySize])
        XCTAssertTrue(logits.sum().item(Float.self).isFinite, "the stack produces finite logits")
    }

    // The recurrence must be causal: a later token cannot change an earlier token's output.
    func testTheRecurrenceIsCausal() throws {
        try requireMLXRuntime()
        let layer = NFKHybridLinearAttention(small)
        let prefix = MLXArray.zeros([1, 3, small.hiddenSize]) + 0.1
        let extended = concatenated([prefix, MLXArray.zeros([1, 2, small.hiddenSize]) + 0.7], axis: 1)

        let short = layer(prefix)
        let long = layer(extended)
        eval(short, long)
        let worst = (short - long[0..., 0 ..< 3]).abs().max().item(Float.self)
        XCTAssertEqual(worst, 0, accuracy: 1e-5, "appending tokens cannot change earlier outputs")
    }
}
