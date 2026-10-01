//
//  NFKMLXLoRATests.swift
//  InferKitMLXTests
//
//  Low-rank adaptation. Three properties carry the design: an adapted model starts out identical to the
//  one it wrapped, only the adapters train, and merging reproduces the adapted forward exactly — which
//  is what lets a customized model ship as one ordinary checkpoint.
//

import XCTest
import MLX
import MLXNN
import MLXOptimizers
@testable import InferKitMLX

final class NFKMLXLoRATests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    override func setUp() {
        super.setUp()
        // Seeding initializes MLX's runtime, which needs a Metal library it can find; the methods
        // skip without one.
        guard NFKMLXGPU.metalLibraryURL != nil else { return }
        NFKMLXRandom.seed(20_260_814)
    }

    /// A small stack with named layers, standing in for an attention block.
    private final class Block: Module {
        @ModuleInfo(key: "q") var q: Linear
        @ModuleInfo(key: "v") var v: Linear
        @ModuleInfo(key: "mlp") var mlp: Linear

        override init() {
            _q.wrappedValue = Linear(8, 8)
            _v.wrappedValue = Linear(8, 8)
            _mlp.wrappedValue = Linear(8, 8)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            mlp(q(x) + v(x))
        }
    }

    private func input() -> MLXArray {
        MLXArray((0 ..< 24).map { Float($0 % 7) / 7.0 }).reshaped([3, 8])
    }

    // MARK: - Adapting

    func testAnAdaptedModelStartsIdenticalToTheOneItWrapped() throws {
        try requireMLXRuntime()
        let block = Block()
        let x = input()
        let before = block(x)
        eval(before)

        try NFKMLXLoRA.apply(to: block, rank: 4)

        let after = block(x)
        eval(after)
        // B starts at zero, so the detour contributes nothing until it is trained. A fine-tune's
        // starting point is a checkpoint worth preserving.
        XCTAssertEqual(after.asArray(Float.self), before.asArray(Float.self))
    }

    func testThePredicateSelectsWhichLayersAreAdapted() throws {
        try requireMLXRuntime()
        let block = Block()
        let adapted = try NFKMLXLoRA.apply(to: block, rank: 4) { path, _ in
            path.hasSuffix("q") || path.hasSuffix("v")
        }
        XCTAssertEqual(adapted, 2, "targeting the attention projections is the usual choice")
        XCTAssertTrue(block.q is NFKMLXLoRALinear)
        XCTAssertTrue(block.v is NFKMLXLoRALinear)
        XCTAssertFalse(block.mlp is NFKMLXLoRALinear)
    }

    func testApplyingTwiceDoesNotStackDetours() throws {
        try requireMLXRuntime()
        let block = Block()
        XCTAssertEqual(try NFKMLXLoRA.apply(to: block, rank: 4), 3)
        XCTAssertEqual(try NFKMLXLoRA.apply(to: block, rank: 4), 0, "already-adapted layers are skipped")
    }

    // MARK: - What trains

    func testOnlyTheAdaptersAreTrainable() throws {
        try requireMLXRuntime()
        let block = Block()
        try NFKMLXLoRA.apply(to: block, rank: 4)

        let trainable = block.trainableParameters().flattened().map(\.0)
        XCTAssertFalse(trainable.isEmpty)
        XCTAssertTrue(trainable.allSatisfy { $0.hasSuffix("lora_a") || $0.hasSuffix("lora_b") },
                      "a frozen base costs no gradient and no optimizer state: \(trainable)")
    }

    func testAdaptingCutsTheTrainableParameterCount() throws {
        try requireMLXRuntime()
        let full = Block()
        let fullCount = NFKMLXLoRA.trainableParameterCount(of: full)

        let adapted = Block()
        try NFKMLXLoRA.apply(to: adapted, rank: 2)
        let adaptedCount = NFKMLXLoRA.trainableParameterCount(of: adapted)

        XCTAssertLessThan(adaptedCount, fullCount,
                          "rank 2 over 8-wide layers trains less than the layers themselves: "
                          + "\(adaptedCount) vs \(fullCount)")
    }

    func testTrainingMovesOnlyTheAdapters() throws {
        try requireMLXRuntime()
        let block = Block()
        try NFKMLXLoRA.apply(to: block, rank: 4)
        let baseWeight = block.q.weight.asArray(Float.self)

        let x = input()
        let target = MLXArray.zeros([3, 8])
        try NFKMLXTrainer.train(block, optimizer: Adam(learningRate: 1e-2), steps: 20,
                                batch: { _ in (x, target) },
                                loss: { model, input, target in (model(input) - target).square().mean() })

        XCTAssertEqual(block.q.weight.asArray(Float.self), baseWeight, "the frozen base did not move")
        let adapter = try XCTUnwrap(block.q as? NFKMLXLoRALinear)
        XCTAssertNotEqual(adapter.loraB.asArray(Float.self),
                          [Float](repeating: 0, count: adapter.loraB.size),
                          "the detour did")
    }

    func testTrainingAnAdaptedModelReducesTheLoss() throws {
        try requireMLXRuntime()
        let block = Block()
        try NFKMLXLoRA.apply(to: block, rank: 4)
        let x = input()
        let target = MLXArray.zeros([3, 8])

        let history = try NFKMLXTrainer.train(block, optimizer: Adam(learningRate: 1e-2), steps: 40,
                                              batch: { _ in (x, target) },
                                              loss: { model, input, target in
                                                  (model(input) - target).square().mean()
                                              })
        XCTAssertLessThan(history.last!, history.first! * 0.5,
                          "a rank-4 detour is enough to move the output: "
                          + "\(history.first!) -> \(history.last!)")
    }

    // MARK: - Merging

    func testMergingReproducesTheAdaptedForward() throws {
        try requireMLXRuntime()
        let block = Block()
        try NFKMLXLoRA.apply(to: block, rank: 4)
        let x = input()
        let target = MLXArray.zeros([3, 8])
        try NFKMLXTrainer.train(block, optimizer: Adam(learningRate: 1e-2), steps: 20,
                                batch: { _ in (x, target) },
                                loss: { model, input, target in (model(input) - target).square().mean() })

        let adaptedOutput = block(x)
        eval(adaptedOutput)

        XCTAssertEqual(try NFKMLXLoRA.merge(into: block), 3)
        XCTAssertFalse(block.q is NFKMLXLoRALinear, "plain layers again")

        let mergedOutput = block(x)
        eval(mergedOutput)
        for (merged, adapted) in zip(mergedOutput.asArray(Float.self), adaptedOutput.asArray(Float.self)) {
            XCTAssertEqual(merged, adapted, accuracy: 1e-5,
                           "the fold into the base weights is exact")
        }
    }

    func testAMergedModelSavesAndReloadsAsAnOrdinaryCheckpoint() throws {
        try requireMLXRuntime()
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("lora-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }

        let block = Block()
        try NFKMLXLoRA.apply(to: block, rank: 4)
        let x = input()
        try NFKMLXTrainer.train(block, optimizer: Adam(learningRate: 1e-2), steps: 20,
                                batch: { _ in (x, MLXArray.zeros([3, 8])) },
                                loss: { model, input, target in (model(input) - target).square().mean() })
        try NFKMLXLoRA.merge(into: block)
        try NFKMLXWeights.save(block, to: url)

        // The point of merging: what comes out carries no adapter keys, so an unmodified model loads it.
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        XCTAssertFalse(checkpoint.arrays.keys.contains { $0.contains("lora_") },
                       "no adapter format to carry around: \(checkpoint.arrays.keys.sorted())")

        let plain = Block()
        try NFKMLXWeights.apply(Array(checkpoint.arrays), to: plain)
        let expected = block(x)
        let actual = plain(x)
        eval(expected, actual)
        XCTAssertEqual(actual.asArray(Float.self), expected.asArray(Float.self))
    }

    /// A model that stores a layer in a plain property cannot receive a replacement. MLX's own path
    /// aborts the process there, so the library has to report it instead.
    private final class Unwrapped: Module {
        let projection = Linear(8, 8)
    }

    func testAdaptingALayerThatCannotBeReplacedThrows() throws {
        try requireMLXRuntime()
        XCTAssertThrowsError(try NFKMLXLoRA.apply(to: Unwrapped(), rank: 4)) { error in
            guard case NFKMLXError.loRANotApplicable(let detail) = error else {
                return XCTFail("expected loRANotApplicable, got \(error)")
            }
            XCTAssertTrue(detail.contains("@ModuleInfo"), "names the fix: \(detail)")
        }
    }

    func testMergingUnfreezesTheModel() throws {
        try requireMLXRuntime()
        let block = Block()
        try NFKMLXLoRA.apply(to: block, rank: 4)
        try NFKMLXLoRA.merge(into: block)
        XCTAssertEqual(block.trainableParameters().flattened().count,
                       block.parameters().flattened().count,
                       "a merged model is an ordinary model again")
    }

    // MARK: Quantized bases

    // `QuantizedLinear` is a subclass of `Linear`, so it satisfies the predicate's type without
    // announcing itself, and its `weight` holds packed integers rather than the values the layer
    // computes with. Adapting one would build a detour around something that is not a weight.
    func testAdaptingAQuantizedLayerIsRefusedRatherThanSilentlyWrong() throws {
        try requireMLXRuntime()
        final class Quantized: Module {
            @ModuleInfo(key: "q") var q: QuantizedLinear
            override init() {
                _q.wrappedValue = QuantizedLinear(Linear(64, 32, bias: false), groupSize: 32, bits: 4)
            }
        }
        let model = Quantized()
        XCTAssertThrowsError(try NFKMLXLoRA.apply(to: model)) { error in
            guard case NFKMLXError.loRANotApplicable(let message) = error else {
                return XCTFail("expected loRANotApplicable, got \(error)")
            }
            XCTAssertTrue(message.contains("quantized"), "the message names the reason: \(message)")
        }
    }

    // Merging a low-rank delta into a quantized weight and requantizing rounds the delta away, so a
    // merge that finds a quantized layer refuses rather than writing a checkpoint that silently holds
    // the original model.
    func testMergingIntoAQuantizedModelIsRefused() throws {
        try requireMLXRuntime()
        final class Mixed: Module {
            @ModuleInfo(key: "plain") var plain: Linear
            @ModuleInfo(key: "q") var q: QuantizedLinear
            override init() {
                _plain.wrappedValue = Linear(8, 8, bias: false)
                _q.wrappedValue = QuantizedLinear(Linear(64, 32, bias: false), groupSize: 32, bits: 4)
            }
        }
        let model = Mixed()
        // Adapt the plain layer only, which is what a caller quantizing afterward would have done.
        XCTAssertEqual(try NFKMLXLoRA.apply(to: model, where: { path, _ in path.contains("plain") }), 1)
        XCTAssertThrowsError(try NFKMLXLoRA.merge(into: model))
    }

    // A predicate that excludes the quantized layers still works, so a partly quantized model is not
    // shut out entirely.
    func testExcludingTheQuantizedLayersAllowsTheRestToBeAdapted() throws {
        try requireMLXRuntime()
        final class Mixed: Module {
            @ModuleInfo(key: "plain") var plain: Linear
            @ModuleInfo(key: "q") var q: QuantizedLinear
            override init() {
                _plain.wrappedValue = Linear(8, 8, bias: false)
                _q.wrappedValue = QuantizedLinear(Linear(64, 32, bias: false), groupSize: 32, bits: 4)
            }
        }
        let model = Mixed()
        let adapted = try NFKMLXLoRA.apply(to: model, where: { path, _ in !path.contains("q") })
        XCTAssertEqual(adapted, 1)
    }

    // MARK: - Layers inside containers

    /// A feed-forward held as a module array, the shape Whisper's `mlp` and the diffusion
    /// transformers' `net` arrays take. MLX rebuilds an array from the entries an update names, so an
    /// adapter set that misses the array's tail would drop it.
    private final class ArrayHeld: Module {
        @ModuleInfo(key: "net") var net: [Module]

        override init() {
            _net.wrappedValue = [Linear(8, 16), GELU(), Linear(16, 8), Dropout(p: 0)]
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            net.reduce(x) { ($1 as! UnaryLayer)($0) }
        }
    }

    private final class DictionaryHeld: Module {
        @ModuleInfo(key: "heads") var heads: [String: Linear]

        override init() {
            _heads.wrappedValue = ["a": Linear(8, 8), "b": Linear(8, 8)]
            super.init()
        }
    }

    private final class Stack: Module {
        @ModuleInfo(key: "blocks") var blocks: [Block]

        override init() {
            _blocks.wrappedValue = [Block(), Block(), Block()]
            super.init()
        }
    }

    func testAdaptingTheLinearsOfAnArrayKeepsItsOtherEntries() throws {
        try requireMLXRuntime()
        let model = ArrayHeld()
        let before = model(input())
        XCTAssertEqual(try NFKMLXLoRA.apply(to: model), 2)
        XCTAssertEqual(model.net.count, 4, "the activation and the dropout after the last adapter stay")
        XCTAssertTrue(model.net[0] is NFKMLXLoRALinear)
        XCTAssertTrue(model.net[1] is GELU)
        XCTAssertTrue(model.net[2] is NFKMLXLoRALinear)
        XCTAssertTrue(model.net[3] is Dropout)
        XCTAssertEqual(abs(model(input()) - before).max().item(Float.self), 0, accuracy: 1e-6)
    }

    func testAdaptingOnlyTheFirstLinearOfAnArrayKeepsTheRest() throws {
        try requireMLXRuntime()
        let model = ArrayHeld()
        XCTAssertEqual(try NFKMLXLoRA.apply(to: model) { path, _ in path == "net.0" }, 1)
        XCTAssertEqual(model.net.count, 4)
        XCTAssertTrue(model.net[0] is NFKMLXLoRALinear)
        XCTAssertTrue(type(of: model.net[2]) == Linear.self)
    }

    func testAdaptingOnlyTheLastLinearOfAnArray() throws {
        try requireMLXRuntime()
        let model = ArrayHeld()
        XCTAssertEqual(try NFKMLXLoRA.apply(to: model) { path, _ in path == "net.2" }, 1)
        XCTAssertEqual(model.net.count, 4)
        XCTAssertTrue(type(of: model.net[0]) == Linear.self)
        XCTAssertTrue(model.net[2] is NFKMLXLoRALinear)
    }

    func testAdaptingOneEntryOfADictionaryKeepsTheOthers() throws {
        try requireMLXRuntime()
        let model = DictionaryHeld()
        XCTAssertEqual(try NFKMLXLoRA.apply(to: model) { path, _ in path == "heads.a" }, 1)
        XCTAssertEqual(Set(model.heads.keys), ["a", "b"])
        XCTAssertTrue(model.heads["a"] is NFKMLXLoRALinear)
        XCTAssertTrue(type(of: model.heads["b"]!) == Linear.self)
    }

    func testAdaptingOneBlockOfAStackLeavesTheOthers() throws {
        try requireMLXRuntime()
        let model = Stack()
        XCTAssertEqual(try NFKMLXLoRA.apply(to: model) { path, _ in path.hasPrefix("blocks.1.") }, 3)
        XCTAssertEqual(model.blocks.count, 3)
        XCTAssertTrue(model.blocks[1].q is NFKMLXLoRALinear)
        XCTAssertTrue(type(of: model.blocks[0].q) == Linear.self)
        XCTAssertTrue(type(of: model.blocks[2].mlp) == Linear.self)
    }

    func testMergingAnArrayHeldAdapterKeepsTheArrayAndTheForward() throws {
        try requireMLXRuntime()
        let model = ArrayHeld()
        try NFKMLXLoRA.apply(to: model) { path, _ in path == "net.0" }
        let adapter = model.net[0] as! NFKMLXLoRALinear
        adapter.update(parameters: ModuleParameters.unflattened([("lora_b", MLXArray.ones(adapter.loraB.shape) * 0.1)]))
        let adapted = model(input())
        XCTAssertEqual(try NFKMLXLoRA.merge(into: model), 1)
        XCTAssertEqual(model.net.count, 4)
        XCTAssertTrue(type(of: model.net[0]) == Linear.self)
        XCTAssertLessThan(abs(model(input()) - adapted).max().item(Float.self), 1e-5)
    }

    // MARK: - State no run trains

    private final class Normalized: Module {
        @ModuleInfo(key: "proj") var proj: Linear
        @ModuleInfo(key: "norm") var norm: BatchNorm
        @ModuleInfo(key: "table") var table: Embedding

        override init() {
            _proj.wrappedValue = Linear(8, 8)
            _norm.wrappedValue = BatchNorm(featureCount: 8)
            _table.wrappedValue = QuantizedEmbedding(Embedding(embeddingCount: 4, dimensions: 64), groupSize: 32, bits: 4)
            super.init()
        }
    }

    func testMergingLeavesRunningStatisticsAndQuantizedArraysFrozen() throws {
        try requireMLXRuntime()
        let model = Normalized()
        try NFKMLXLoRA.apply(to: model) { path, _ in path == "proj" }
        try NFKMLXLoRA.merge(into: model)
        let trainable = Set(model.trainableParameters().flattened().map(\.0))
        XCTAssertTrue(trainable.contains("proj.weight"), "merging unfreezes the model")
        XCTAssertTrue(trainable.contains("norm.weight"))
        XCTAssertFalse(trainable.contains("norm.running_mean"))
        XCTAssertFalse(trainable.contains("norm.running_var"))
        XCTAssertFalse(trainable.contains("table.weight"), "a packed table has no gradient")
        XCTAssertFalse(trainable.contains("table.scales"))
    }

    func testTheTrainableCountLeavesOutWhatARunFreezesAgain() throws {
        try requireMLXRuntime()
        let model = Normalized()
        model.unfreeze()
        XCTAssertEqual(NFKMLXLoRA.trainableParameterCount(of: model), 8 * 8 + 8 + 8 + 8,
                       "the projection and the normalization's affine pair")
    }
}
