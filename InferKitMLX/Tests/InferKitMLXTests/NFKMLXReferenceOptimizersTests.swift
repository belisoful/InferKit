//
//  NFKMLXReferenceOptimizersTests.swift
//  InferKitMLXTests
//
//  The recipes' default optimizers against PyTorch's arithmetic, one step from a fresh state, where a
//  missing bias correction shows most: the first update of a bias-corrected Adam moves each parameter by
//  the learning rate times the gradient's sign, and an uncorrected one by 3.2 times that.
//

import XCTest
import MLX
import MLXNN
import MLXOptimizers
@testable import InferKitMLX

final class NFKMLXReferenceOptimizersTests: XCTestCase {

    private final class Pair: Module {
        @ModuleInfo(key: "linear") var linear = Linear(1, 1)
        @ModuleInfo(key: "norm") var norm = LayerNorm(dimensions: 1)
        @ModuleInfo(key: "upscale_norm") var upscaleNorm = LayerNorm(dimensions: 1)
    }

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    /// Every parameter set to 1, every gradient 0.5, one update.
    private func step(_ optimizer: Optimizer, on module: Module) -> [String: Float] {
        module.update(parameters: module.parameters().mapValues { MLXArray.ones(like: $0) })
        let gradients = module.parameters().mapValues { MLXArray.ones(like: $0) * 0.5 }
        optimizer.update(model: module, gradients: gradients)
        eval(module)
        return Dictionary(uniqueKeysWithValues: module.parameters().flattened().map { ($0.0, $0.1.item(Float.self)) })
    }

    func testTheReferenceAdamWTakesPyTorchsFirstStep() throws {
        try requireMLXRuntime()
        let updated = step(NFKMLXReferenceOptimizers.adamW(learningRate: 0.01, weightDecay: 0.1), on: Linear(1, 1))
        // torch.optim.AdamW: p ← p(1 − lr·wd) − lr·m̂/(√v̂ + ε), and m̂/√v̂ is the gradient's sign at step one.
        let expected: Float = 1 * (1 - 0.01 * 0.1) - 0.01 * 0.5 / (0.5 + 1e-8)
        XCTAssertEqual(updated["weight"]!, expected, accuracy: 1e-6)
        let uncorrected = step(AdamW(learningRate: 0.01, weightDecay: 0.1), on: Linear(1, 1))
        XCTAssertEqual((1 * (1 - 0.01 * 0.1) - uncorrected["weight"]!) / 0.01, 0.1 / Float(0.001).squareRoot(),
                       accuracy: 1e-3, "mlx-swift's default AdamW takes a √10-times step here")
    }

    func testTheL2AdamAddsItsDecayToTheGradient() throws {
        try requireMLXRuntime()
        let updated = step(NFKMLXL2Adam(learningRate: 0.01, weightDecay: 0.5), on: Linear(1, 1))
        // torch.optim.Adam(weight_decay=0.5): g ← 0.5 + 0.5·1 = 1, and the first step is lr·sign(g).
        XCTAssertEqual(updated["weight"]!, 1 - 0.01, accuracy: 1e-6)
    }

    func testTheGroupedAdamWExemptsBiasesAndLayerNormsButNotTheExcludedOnes() throws {
        try requireMLXRuntime()
        let module = Pair()
        let layerNorms = NFKMLXReferenceOptimizers.layerNormPrefixes(in: module, excluding: ["upscale_norm"])
        XCTAssertEqual(layerNorms, ["norm."])
        let updated = step(NFKMLXReferenceOptimizers.adamW(
            learningRate: 0.01, weightDecay: 0.1,
            exempting: NFKMLXReferenceOptimizers.biasOrLayerNorm(layerNorms)), on: module)
        let decayed: Float = 1 * (1 - 0.01 * 0.1) - 0.01
        let undecayed: Float = 1 - 0.01
        XCTAssertEqual(updated["linear.weight"]!, decayed, accuracy: 1e-6)
        XCTAssertEqual(updated["linear.bias"]!, undecayed, accuracy: 1e-6)
        XCTAssertEqual(updated["norm.weight"]!, undecayed, accuracy: 1e-6)
        XCTAssertEqual(updated["norm.bias"]!, undecayed, accuracy: 1e-6)
        XCTAssertEqual(updated["upscale_norm.weight"]!, decayed, accuracy: 1e-6, "a LayerNorm2d in the reference decays")
        XCTAssertEqual(updated["upscale_norm.bias"]!, undecayed, accuracy: 1e-6)
    }

    /// Values from the references' own code: SAM 3's `InverseSquareRootParamScheduler` as fetched, and
    /// the fvcore, mmcv, and cosmos-predict1 formulas, over a 100-step run.
    func testTheSchedulesMatchTheReferences() {
        let inverse = NFKMLXLearningRateSchedule.inverseSquareRoot(steps: 100, timescale: 20, warmupSteps: 20, cooldownSteps: 20)
        for (step, expected) in [(0, 0.0), (5, 0.25), (20, 1.0), (21, 0.9759000729485331), (50, 0.6324555320336759),
                                 (85, 0.36380343755449945), (99, 0.022473328748774737)] {
            XCTAssertEqual(Double(inverse.multiplier(step)), expected, accuracy: 1e-6, "inverse square root at \(step)")
        }
        let cosine = NFKMLXLearningRateSchedule.cosine(steps: 100, endScale: 0.1)
        for (step, expected) in [(0, 1.0), (25, 0.8681980515339464), (50, 0.55), (99, 0.10022204783542078)] {
            XCTAssertEqual(Double(cosine.multiplier(step)), expected, accuracy: 1e-6, "cosine at \(step)")
        }
        let poly = NFKMLXLearningRateSchedule.poly(steps: 100, power: 1, warmupSteps: 30, warmupRatio: 1e-6)
        for (step, expected) in [(0, 1.0000000000287557e-06), (10, 0.30000059999999995), (29, 0.686333357),
                                 (30, 0.7), (99, 0.010000000000000009)] {
            XCTAssertEqual(Double(poly.multiplier(step)), expected, accuracy: 1e-6, "poly at \(step)")
        }
        let warmup = NFKMLXLearningRateSchedule.linearWarmup(steps: 5000)
        XCTAssertEqual(warmup.multiplier(0), 0.0002, accuracy: 1e-9)
        XCTAssertEqual(warmup.multiplier(4999), 1)
        XCTAssertEqual(warmup.multiplier(6000), 1)
    }

    func testARecipeFollowsTheReferenceScheduleOnlyWithTheReferenceOptimizer() {
        let reference = NFKMLXLearningRateSchedule.linearWarmup(steps: 100)
        XCTAssertEqual(NFKMLXLearningRateSchedule.resolved(nil, optimizer: nil) { reference }.multiplier(0), 0.01)
        XCTAssertEqual(NFKMLXLearningRateSchedule.resolved(nil, optimizer: SGD(learningRate: 0.1)) { reference }.multiplier(0), 1,
                       "a caller's optimizer keeps the rate it was given")
        let given = NFKMLXLearningRateSchedule { _ in 0.5 }
        XCTAssertEqual(NFKMLXLearningRateSchedule.resolved(given, optimizer: SGD(learningRate: 0.1)) { reference }.multiplier(0), 0.5)
    }

    func testTheRateGroupedAdamWScalesEachGroupsStep() throws {
        try requireMLXRuntime()
        let module = Pair()
        let optimizer = NFKMLXReferenceOptimizers.adamW(learningRate: 0.01, over: module) { key in
            key.hasPrefix("linear") ? (10, 0.1) : (1, 0)
        }
        let updated = step(optimizer, on: module)
        XCTAssertEqual(updated["linear.weight"]!, 1 * (1 - 0.1 * 0.1) - 0.1, accuracy: 1e-6)
        XCTAssertEqual(updated["norm.weight"]!, 1 - 0.01, accuracy: 1e-6)
        XCTAssertEqual(NFKMLXLearningRateSchedule.scheduledGroups(of: optimizer)?.map(\.1), [Float(0.01) * 10, 0.01])
    }

    func testTheTrainerScalesEveryGroupAndRestoresTheRates() throws {
        try requireMLXRuntime()
        let module = Pair()
        module.update(parameters: module.parameters().mapValues { MLXArray.ones(like: $0) })
        let optimizer = NFKMLXReferenceOptimizers.adamW(learningRate: 0.01, over: module) { key in
            key.hasPrefix("linear") ? (10, 0) : (1, 0)
        }
        var afterFirst = [String: Float]()
        try NFKMLXTrainer.train(module, optimizer: optimizer, steps: 2,
                                sample: { _ in MLXArray.ones([1, 1]) },
                                loss: { module, x in module.upscaleNorm(module.norm(module.linear(x))).sum() + module.linear(x).sum() },
                                learningRateSchedule: NFKMLXLearningRateSchedule { $0 == 0 ? 0 : 1 },
                                observer: { step in
                                    if step.index == 0 {
                                        afterFirst = Dictionary(uniqueKeysWithValues: module.parameters().flattened()
                                            .map { ($0.0, $0.1.item(Float.self)) })
                                    }
                                    return true
                                })
        XCTAssertEqual(afterFirst["linear.weight"], 1, "a zero multiplier leaves the first step's parameters")
        XCTAssertNotEqual(module.linear.weight.item(Float.self), 1, "the second step moves them")
        XCTAssertEqual(NFKMLXLearningRateSchedule.scheduledGroups(of: optimizer)?.map(\.1), [Float(0.01) * 10, 0.01],
                       "the base rates are restored")
    }

    func testTheSAM2TrunkLayerDecayFollowsHierasLayerIds() {
        let blocks = 12
        XCTAssertEqual(NFKMLXSAM2.trunkLayerDecay("image_encoder.trunk.blocks.11.attn.qkv.weight", blocks: blocks), 0.9, accuracy: 1e-12)
        XCTAssertEqual(NFKMLXSAM2.trunkLayerDecay("image_encoder.trunk.blocks.0.mlp.layers.0.bias", blocks: blocks), pow(0.9, 12), accuracy: 1e-12)
        XCTAssertEqual(NFKMLXSAM2.trunkLayerDecay("image_encoder.trunk.patch_embed.weight", blocks: blocks), pow(0.9, 13), accuracy: 1e-12)
        XCTAssertEqual(NFKMLXSAM2.trunkLayerDecay("image_encoder.trunk.pos_embed_window", blocks: blocks), 1)
        XCTAssertEqual(NFKMLXSAM2.trunkLayerDecay("image_encoder.neck.convs.0.conv.weight", blocks: blocks), 1)
        XCTAssertEqual(NFKMLXSAM2.trunkLayerDecay("sam_mask_decoder.iou_prediction_head.layers.0.weight", blocks: blocks), 1)
    }
}
