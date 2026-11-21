//
//  NFKMLXBlockedAttentionTests.swift
//  InferKitMLXTests
//

import XCTest
import MLX
import MLXFast
import MLXNN
import MLXRandom
@testable import InferKitMLX

final class NFKMLXBlockedAttentionTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    private func relative(_ a: MLXArray, _ b: MLXArray) -> Float {
        (abs(a - b).max() / abs(b).max()).item(Float.self)
    }

    func testTheBlockedAttentionAndItsGradientsAreMLXs() throws {
        try requireMLXRuntime()
        let shape = [1, 3, 37, 16]
        let q = MLXRandom.normal(shape, key: MLXRandom.key(1)), k = MLXRandom.normal([1, 3, 29, 16], key: MLXRandom.key(2))
        let v = MLXRandom.normal([1, 3, 29, 16], key: MLXRandom.key(3)), w = MLXRandom.normal(shape, key: MLXRandom.key(4))
        let scale: Float = 0.25
        func mlx(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray) -> MLXArray {
            MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: nil)
        }
        // A 10-query block puts 37 queries in four blocks, the last one short.
        func blocked(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray) -> MLXArray {
            NFKMLXBlockedAttention.attention(queries: q, keys: k, values: v, scale: scale, queryBlock: 10)
        }
        XCTAssertLessThan(relative(blocked(q, k, v), mlx(q, k, v)), 1e-6)
        let reference = [grad({ (x: MLXArray) in (mlx(x, k, v) * w).sum() })(q),
                         grad({ (x: MLXArray) in (mlx(q, x, v) * w).sum() })(k),
                         grad({ (x: MLXArray) in (mlx(q, k, x) * w).sum() })(v)]
        let ours = [grad({ (x: MLXArray) in (blocked(x, k, v) * w).sum() })(q),
                    grad({ (x: MLXArray) in (blocked(q, x, v) * w).sum() })(k),
                    grad({ (x: MLXArray) in (blocked(q, k, x) * w).sum() })(v)]
        for (name, (mine, theirs)) in zip(["queries", "keys", "values"], zip(ours, reference)) {
            XCTAssertEqual(mine.shape, theirs.shape)
            XCTAssertLessThan(relative(mine, theirs), 1e-5, "the \(name)' gradient")
        }
    }

    func testAMaskedBlockedAttentionAndItsGradientsAreMLXs() throws {
        try requireMLXRuntime()
        let shape = [2, 2, 23, 8]
        let q = MLXRandom.normal(shape, key: MLXRandom.key(11)), k = MLXRandom.normal(shape, key: MLXRandom.key(12))
        let v = MLXRandom.normal(shape, key: MLXRandom.key(13)), w = MLXRandom.normal(shape, key: MLXRandom.key(14))
        // A per-row additive mask, sliced with its block, and a per-key one broadcast over the rows.
        let rowMask = MLX.where(MLXRandom.uniform(0 ..< 1, [2, 1, 23, 23], key: MLXRandom.key(15)) .< 0.3,
                                MLXArray(Float(-1e9)), MLXArray(Float(0)))
        let keyMask = MLX.where(MLXArray(0 ..< 23).reshaped([1, 1, 1, 23]) .>= 20, MLXArray(Float(-1e9)), MLXArray(Float(0)))
        for mask in [rowMask, keyMask] {
            func mlx(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray) -> MLXArray {
                MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: 0.3, mask: mask)
            }
            func blocked(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray) -> MLXArray {
                NFKMLXBlockedAttention.attention(queries: q, keys: k, values: v, scale: 0.3, mask: mask, queryBlock: 7)
            }
            XCTAssertLessThan(relative(blocked(q, k, v), mlx(q, k, v)), 1e-6)
            let reference = [grad({ (x: MLXArray) in (mlx(x, k, v) * w).sum() })(q),
                             grad({ (x: MLXArray) in (mlx(q, x, v) * w).sum() })(k),
                             grad({ (x: MLXArray) in (mlx(q, k, x) * w).sum() })(v)]
            let ours = [grad({ (x: MLXArray) in (blocked(x, k, v) * w).sum() })(q),
                        grad({ (x: MLXArray) in (blocked(q, x, v) * w).sum() })(k),
                        grad({ (x: MLXArray) in (blocked(q, k, x) * w).sum() })(v)]
            for (name, (mine, theirs)) in zip(["queries", "keys", "values"], zip(ours, reference)) {
                XCTAssertLessThan(relative(mine, theirs), 1e-5, "the \(name)' gradient under a \(mask.shape) mask")
            }
        }
    }

    // Only a training run on float32 past two query blocks takes the blocked path.
    func testOnlyALongFloat32TrainingAttentionIsBlocked() throws {
        try requireMLXRuntime()
        let long = MLXArray.zeros([1, 2, 2 * NFKMLXBlockedAttention.queryBlock + 1, 8])
        XCTAssertFalse(NFKMLXBlockedAttention.applies(to: long, mask: nil))
        NFKMLXBlockedAttention.training {
            XCTAssertTrue(NFKMLXBlockedAttention.applies(to: long, mask: nil))
            XCTAssertFalse(NFKMLXBlockedAttention.applies(to: long.asType(.bfloat16), mask: nil))
            XCTAssertFalse(NFKMLXBlockedAttention.applies(to: long[.ellipsis, 0 ..< 1024, 0...], mask: nil))
            XCTAssertFalse(NFKMLXBlockedAttention.applies(to: long, mask: MLXArray([true])))
        }
        XCTAssertFalse(NFKMLXBlockedAttention.isTraining)
    }

    // A tiny Wan transformer over 1,296 tokens, more than two query blocks: its adapters' gradients through the
    // blocked attention are those through MLX's.
    func testWanTrainsThroughTheBlockedAttentionAsThroughMLXs() throws {
        try requireMLXRuntime()
        let config = NFKMLXWanConfiguration.tiny
        MLXRandom.seed(3)
        let net = NFKMLXWanTransformerNet(config)
        try NFKMLXLoRA.apply(to: net, rank: 4, alpha: 4) { path, _ in NFKMLXWanTrainingPreset.finetrainers.targets.contains(path) }
        let latent = MLXRandom.normal([config.inChannels, 1, 72, 72], key: MLXRandom.key(5))
        let text = MLXRandom.normal([8, config.textDim], key: MLXRandom.key(6))
        let target = MLXRandom.normal(latent.shape, key: MLXRandom.key(7))
        net.recomputesBlocksInBackward = true
        func gradients(blocked: Bool) -> [String: MLXArray] {
            NFKMLXBlockedAttention.$isTraining.withValue(blocked) { gradients() }
        }
        func gradients() -> [String: MLXArray] {
            let step = valueAndGrad(model: net) { (model: NFKMLXWanTransformerNet, arrays: [MLXArray]) -> [MLXArray] in
                [(model(arrays[0], text: text, t: MLXArray([Float(500)])) - target).square().mean()]
            }
            let (_, gradients) = step(net, [latent])
            return Dictionary(uniqueKeysWithValues: gradients.flattened())
        }
        let plain = gradients(blocked: false), blocked = gradients(blocked: true)
        XCTAssertEqual(Set(plain.keys), Set(blocked.keys))
        XCTAssertFalse(plain.isEmpty)
        for (key, value) in plain where abs(value).max().item(Float.self) > 0 {
            XCTAssertLessThan(relative(try XCTUnwrap(blocked[key]), value), 1e-5, key)
        }
    }
}
