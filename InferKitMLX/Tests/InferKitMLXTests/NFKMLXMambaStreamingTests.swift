//
//  NFKMLXMambaStreamingTests.swift
//  InferKitMLXTests
//
//  Streaming the Mamba-2 decoder's blocks. A random decoder is written out as a sharded bfloat16 release in
//  the released layout (`backbone.*`, the convolution stored `[C, 1, K]`), and the same release loaded with
//  every block held and with its blocks streamed must compute the same values, element for element. On
//  released weights, Codestral Mamba 7B streamed at float32 matches `run_reference.py mamba2_layerwise`
//  (`IK_VAL_MAMBA2`, `IK_PARITY_MAMBA2_LAYERWISE`).
//

import XCTest
import InferKit
import MLX
import MLXNN
import MLXRandom
@testable import InferKitMLX

final class NFKMLXMambaStreamingTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    override func tearDown() {
        if NFKMLXGPU.metalLibraryURL != nil {
            NFKMLXGPU.clearCache()
        }
        super.tearDown()
    }

    private let configuration = NFKMLXMamba2Configuration(
        hiddenSize: 64, layerCount: 6, vocabularySize: 128, rmsEpsilon: 1e-5,
        intermediateSize: 128, headCount: 8, headDimensions: 16, stateSize: 16,
        groupCount: 2, convolutionKernel: 4, useConvolutionBias: true,
        useProjectionBias: false, tiesWordEmbeddings: false)

    private let tokens: [Int32] = [1, 17, 42, 99, 7, 61, 12, 5, 88]

    /// A random decoder written as a two-shard bfloat16 release in the released layout.
    private func release() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("mamba-stream-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        MLXRandom.seed(21)
        let net = NFKMLXMamba.makeNet(configuration)
        let stored = net.parameters().flattened().map { key, value -> (String, MLXArray) in
            let varied = key.hasSuffix("norm.weight") ? 1 + MLXRandom.normal(value.shape) * 0.1 : value
            let shaped = key.hasSuffix("conv1d.weight") ? varied.reshaped([value.dim(0), 1, value.dim(1)]) : varied
            return (key, shaped.asType(.bfloat16))
        }.sorted { $0.0 < $1.0 }
        var weightMap = [String: String]()
        for (index, half) in [stored.prefix(stored.count / 2), stored.dropFirst(stored.count / 2)].enumerated() {
            let name = "model-0000\(index + 1)-of-00002.safetensors"
            try save(arrays: Dictionary(half.map { $0 }, uniquingKeysWith: { first, _ in first }),
                     url: directory.appendingPathComponent(name))
            half.forEach { weightMap[$0.0] = name }
        }
        try JSONSerialization.data(withJSONObject: ["weight_map": weightMap])
            .write(to: directory.appendingPathComponent("model.safetensors.index.json"))
        return directory
    }

    /// The least budget that holds `directory`'s decoder with only its first `held` blocks.
    private func budget(_ directory: URL, precision: NFKMLXWeightPrecision, holding held: Int) throws -> Int {
        let stage = try NFKMLXLayerStream.footprint(directory: directory, precision: precision, includes: { _ in true },
                                                    layerOf: { NFKMLXMamba.block(ofKey: $0)?.layer })
        return NFKMLXResidencyBudget.leastBudget(holding: stage.streamedMinimumBytes + stage.layerBytes.prefix(held).reduce(0, +))
    }

    private func net(_ directory: URL, precision: NFKMLXWeightPrecision, residency: NFKMLXResidency,
                     holding held: Int? = nil) throws -> NFKMLXMamba2Net {
        let net = NFKMLXMamba.makeNet(configuration)
        try NFKMLXMamba.loadWeights(into: net, fromDirectory: directory, precision: precision, residency: residency,
                                    budget: try held.map { try budget(directory, precision: precision, holding: $0) }
                                        ?? NFKMLXResidencyBudget.current())
        return net
    }

    private func maxDifference(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }

    private var input: MLXArray { MLXArray(tokens).reshaped([1, tokens.count]) }

    func testAStreamedMambaDecoderComputesWhatAHeldOneDoes() throws {
        try requireMLXRuntime()
        let directory = try release()
        for precision in [NFKMLXWeightPrecision.float32, .checkpoint] {
            let label = "\(precision)"
            let held = try net(directory, precision: precision, residency: .resident)
            let streamed = try net(directory, precision: precision, residency: .streamed, holding: 2)
            XCTAssertNil(held.layerStream, label)
            XCTAssertEqual(streamed.layerStream?.layers, [2, 3, 4, 5], label)
            let expected = held.blockStates(input)
            for pass in 0 ..< 2 {
                let states = streamed.blockStates(input)
                XCTAssertEqual(states.count, expected.count, label)
                for (index, state) in states.enumerated() {
                    XCTAssertEqual(maxDifference(state, expected[index]), 0, "\(label) pass \(pass) state \(index)")
                }
                XCTAssertEqual(maxDifference(streamed(input), held(input)), 0, "\(label) pass \(pass) logits")
                for index in try XCTUnwrap(streamed.layerStream).layers {
                    XCTAssertTrue(streamed.backbone.layers[index].parameters().flattened().allSatisfy { $0.1.size == 0 },
                                  "\(label): block \(index) holds nothing between its turns")
                }
            }
            try streamed.verifyStream()
        }
    }

    func testAnAutomaticLoadHoldsAReleaseThatFitsAsAWholeLoadDoes() throws {
        try requireMLXRuntime()
        let directory = try release()
        let automatic = try net(directory, precision: .checkpoint, residency: .automatic)
        XCTAssertNil(automatic.layerStream)
        let whole = NFKMLXMamba.makeNet(configuration)
        try NFKMLXMamba.loadWeights(into: whole, fromDirectory: directory, precision: .checkpoint)
        XCTAssertEqual(maxDifference(automatic(input), whole(input)), 0)
    }

    /// A byte-fallback vocabulary as large as the decoder's, so every id it produces decodes.
    private func tokenizer() throws -> NFKMLXMistralTokenizer {
        var vocabulary: [String: Int] = ["<unk>": 0, "<s>": 1, "</s>": 2, "\u{2581}": 3]
        for (index, letter) in "abcdefghijklmnopqrstuvwxyz".enumerated() {
            vocabulary[String(letter)] = 4 + index
        }
        for id in 30 ..< 128 {
            vocabulary["[\(id)]"] = id
        }
        let json: [String: Any] = ["model": ["type": "BPE", "byte_fallback": true, "merges": [String](), "vocab": vocabulary]]
        return try XCTUnwrap(NFKMLXMistralTokenizer(tokenizerJSON: json, prependsEveryRun: true))
    }

    private func reply(_ backend: NFKMLXMambaBackend) throws -> String? {
        let request = NFKInferenceRequest(inputs: [NFKInputPrompt: "abc"],
                                          parameters: [NFKParameterTemperature: 0, NFKParameterMaxTokens: 6])
        return try backend.runInference(for: request).output(forKey: NFKOutputText) as? String
    }

    private func backend(_ net: NFKMLXMamba2Net) throws -> NFKMLXMambaBackend {
        NFKMLXMambaBackend(logits: { net($0) }, tokenizer: try tokenizer(), identifier: "codestral-mamba",
                           modules: [net], layerStream: net.layerStream)
    }

    func testAFailedReadFailsTheRequestAndTheNextRequestReadsTheReleaseAgain() throws {
        try requireMLXRuntime()
        let directory = try release()
        let held = try backend(try net(directory, precision: .checkpoint, residency: .resident))
        let streamedNet = try net(directory, precision: .checkpoint, residency: .streamed, holding: 1)
        let streamed = try backend(streamedNet)
        XCTAssertEqual(held.streamedLayerCount, 0)
        XCTAssertEqual(held.modelInfo[NFKModelInfoResidency] as? String, NFKModelResidencyResident)
        XCTAssertEqual(streamed.streamedLayerCount, 5)
        XCTAssertEqual(streamed.streamedBytesPerPass, streamedNet.layerStream?.bytesPerPass)
        XCTAssertEqual(streamed.modelInfo[NFKModelInfoResidency] as? String, NFKModelResidencyStreamed)
        XCTAssertEqual(try reply(streamed), try reply(held), "the streamed reply")

        let shards = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
            .map { directory.appendingPathComponent($0) }
        let originals = try shards.map { try Data(contentsOf: $0) }
        for file in shards {
            let start = try XCTUnwrap(NFKMLXSafetensors.entries(inFile: file).values.map(\.start).min())
            let handle = try FileHandle(forWritingTo: file)
            try handle.truncate(atOffset: UInt64(start))
            try handle.close()
        }
        XCTAssertThrowsError(try reply(streamed), "the truncated release cannot be read")
        for (file, original) in zip(shards, originals) {
            try original.write(to: file)
        }
        XCTAssertEqual(try reply(streamed), try reply(held), "the next request reads every block again")
        try streamedNet.verifyStream()
    }

    // MARK: Released weights

    func testTheStreamedCodestralMambaMatchesItsReferenceAtFloat32() throws {
        try requireMLXRuntime()
        let environment = NFKMLXValidationConfig.environment
        guard let directoryPath = environment["IK_VAL_MAMBA2"], let recordPath = environment["IK_PARITY_MAMBA2_LAYERWISE"],
              FileManager.default.fileExists(atPath: recordPath) else {
            throw XCTSkip("set IK_VAL_MAMBA2 and IK_PARITY_MAMBA2_LAYERWISE (run_reference.py mamba2_layerwise)")
        }
        let directory = URL(fileURLWithPath: directoryPath)
        let record = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: recordPath)).arrays
        let configuration = try NFKMLXMamba.configuration(fromDirectory: directory)
        let net = NFKMLXMamba.makeNet(configuration)
        try NFKMLXMamba.loadWeights(into: net, fromDirectory: directory, precision: .float32, residency: .automatic)
        let stream = try XCTUnwrap(net.layerStream, "the float32 release does not fit whole, so it streams")

        let tokens = try XCTUnwrap(record["tokens"]).asType(.int32).reshaped([1, -1])
        let started = Date()
        let states = net.blockStates(tokens)
        try net.verifyStream()
        let seconds = Date().timeIntervalSince(started)
        var worst = (index: -1, similarity: 1.0)
        for index in 0 ..< configuration.layerCount {
            let similarity = NFKMLXWav2Vec2Tests.cosine(states[index][0], try XCTUnwrap(record["hidden.\(index)"]))
            if similarity < worst.similarity { worst = (index, similarity) }
        }
        let logits = NFKMLXWav2Vec2Tests.cosine(net(tokens)[0], try XCTUnwrap(record["output"]))
        try net.verifyStream()

        let reference = try XCTUnwrap(record["continuation"]).asType(.int32).asArray(Int32.self)
        var sequence = tokens
        var produced = [Int32]()
        for _ in reference {
            let next = argMax(net(sequence)[0, -1, 0...], axis: -1)
            eval(next)
            try net.verifyStream()
            produced.append(next.item(Int32.self))
            sequence = concatenated([sequence, MLXArray([produced.last!]).reshaped([1, 1])], axis: 1)
        }
        let (bytes, readSeconds) = stream.readStatistics
        print("VALIDATION PARITY codestral-mamba float32: \(stream.layers.count) of \(configuration.layerCount) blocks "
              + "streamed, \(stream.bytesPerPass) bytes per pass, first pass \(Int(seconds)) s, read \(bytes) bytes in "
              + "\(readSeconds) s; worst block state \(worst.similarity) (state \(worst.index)), logits \(logits), "
              + "greedy \(zip(produced, reference).filter { $0 == $1 }.count)/\(reference.count)")
        XCTAssertGreaterThan(worst.similarity, 0.99999, "block state \(worst.index) diverges")
        XCTAssertGreaterThan(logits, 0.99999, "the logits diverge")
        XCTAssertEqual(produced, reference, "the greedy continuation")
    }
}
