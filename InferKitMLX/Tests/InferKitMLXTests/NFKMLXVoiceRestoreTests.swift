//
//  NFKMLXVoiceRestoreTests.swift
//  InferKitMLXTests
//
//  VoiceRestore (skirdey/voicerestore, MIT) flow-matching universal speech restorer. The module forward
//  evaluates MLX arrays, so it skips under `swift test` and runs under `xcodebuild test`. The parity test
//  is gated on the released weights + the recorded oracle (`IK_VAL_VOICERESTORE` + `IK_PARITY_VOICERESTORE`,
//  from `run_reference.py voicerestore`).
//

import XCTest
import InferKit
import MLX
import MLXNN
@testable import InferKitMLX

final class NFKMLXVoiceRestoreTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test` (no bundled metallib); run via xcodebuild")
    }

    private func cosine(_ mine: MLXArray, _ reference: MLXArray) -> Float {
        eval(mine)
        let a = mine.asArray(Float.self), b = reference.asArray(Float.self)
        let n = min(a.count, b.count)
        var dot: Float = 0, na: Float = 0, nb: Float = 0
        for i in 0 ..< n { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
        return dot / (sqrtf(na) * sqrtf(nb) + 1e-20)
    }

    /// The module keys the loader targets, after the block-slot and Sequential-wrapper remaps.
    func testParameterNamesFollowTheModuleLayout() throws {
        try requireMLXRuntime()
        let net = NFKMLXVoiceRestoreFactory.makeNet()
        let names = Set(net.parameters().flattened().map(\.0))
        for expected in ["proj_in.weight", "cond_proj.weight", "to_pred.weight",
                         "transformer.registers", "transformer.abs_pos_emb.weight",
                         "transformer.time_cond_mlp.weight", "transformer.final_norm.weight",
                         "transformer.layers.0.gateloop.norm.gamma",
                         "transformer.layers.0.gateloop.to_qkva.weight",
                         "transformer.layers.0.attn.to_q.weight",
                         "transformer.layers.0.attn.to_v_head_gate.weight",
                         "transformer.layers.0.attn_norm.to_gamma.weight",
                         "transformer.layers.0.ff.proj_in.weight",
                         "transformer.layers.0.ff.proj_out.weight",
                         "transformer.layers.10.skip_proj.weight"] {
            XCTAssertTrue(names.contains(expected), "missing \(expected)")
        }
    }

    /// PARITY: the released VoiceRestore transformer against the recorded oracle, seam by seam over the
    /// packed sequence (32 registers + mel frames), then the final velocity.
    func testTransformerParityOnTheReleasedWeights() throws {
        try requireMLXRuntime()
        let env = ProcessInfo.processInfo.environment
        guard let weightsPath = env["IK_VAL_VOICERESTORE"], let recordPath = env["IK_PARITY_VOICERESTORE"] else {
            throw XCTSkip("set IK_VAL_VOICERESTORE (weights) and IK_PARITY_VOICERESTORE (oracle record)")
        }
        let net = NFKMLXVoiceRestoreFactory.makeNet()
        try NFKMLXVoiceRestoreFactory.loadWeights(into: net, from: URL(fileURLWithPath: weightsPath))
        let r = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: recordPath)).arrays

        let n = r["x_t"]!.dim(0)
        let xT = r["x_t"]!.reshaped([1, n, 100])
        let cond = r["cond"]!.reshaped([1, n, 100])
        let t = r["t"]!                                                    // [1]

        func report(_ name: String, _ value: Float) {
            XCTAssertGreaterThan(value, 0.9999, "seam \(name) diverges (cosine \(value))")
        }

        // Pre-transformer additive condition.
        let xIn = net.projIn(xT) + net.condProj(cond)
        report("x_in", cosine(xIn[0], r["x_in"]!))

        // Replicate the transformer forward with per-seam capture over the packed sequence
        // (32 register tokens + the mel frames).
        let tf = net.transformer
        let condEmb = silu(tf.timeLinear(t.reshaped([1, 1])))              // [1, dim]
        let positions = MLXArray((0 ..< n).map { Int32($0) })
        let rr = tf.config.numRegisters
        let regs = broadcast(tf.registers.expandedDimensions(axis: 0), to: [1, rr, tf.config.dim])
        var x = concatenated([regs, xIn + tf.absPosEmb(positions)], axis: 1)   // [1, R+N, dim]

        var skips = [MLXArray]()
        for (i, block) in tf.layers.enumerated() {
            if i < tf.half {
                skips.append(x)
            } else {
                x = block.skipProj!(concatenated([x, skips.removeLast()], axis: -1))
            }
            let glOut = block.gateloop(x)
            if i == 0 || i == 19 { report("gl\(i)", cosine(glOut[0], r["gl\(i)"]!)) }
            x = glOut + x
            let attnOut = block.attn(block.attnNorm(x, cond: condEmb))
            report("attn\(i)", cosine(attnOut[0], r["attn\(i)"]!))
            x = x + block.attnGate(attnOut, cond: condEmb)
            let ffOut = block.ff(block.ffNorm(x, cond: condEmb))
            if i == 0 || i == 19 { report("ff\(i)", cosine(ffOut[0], r["ff\(i)"]!)) }
            x = x + block.ffGate(ffOut, cond: condEmb)
        }
        let velocity = net.toPred(tf.finalNorm(x)[0..., tf.config.numRegisters..., 0...])
        report("velocity", cosine(velocity[0], r["output"]!))

        // The whole net (public API) reproduces the same velocity.
        let full = net(xT, times: t, cond: cond)
        report("velocity_full", cosine(full[0], r["output"]!))
    }

    /// PARITY: the released BigVGAN vocoder against the recorded oracle (mel → waveform), including the
    /// anti-aliased SnakeBeta activations and the kaiser-sinc resamplings this port recomputes.
    func testBigVGANParityOnTheReleasedWeights() throws {
        try requireMLXRuntime()
        let env = ProcessInfo.processInfo.environment
        guard let weightsPath = env["IK_VAL_BIGVGAN"], let recordPath = env["IK_PARITY_BIGVGAN"] else {
            throw XCTSkip("set IK_VAL_BIGVGAN (weights) and IK_PARITY_BIGVGAN (oracle record)")
        }
        let net = NFKMLXBigVGANFactory.makeNet()
        try net.loadWeights(from: URL(fileURLWithPath: weightsPath))
        let r = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: recordPath)).arrays

        let melNCL = r["mel"]!                                             // [100, T]
        let mel = melNCL.transposed(1, 0).expandedDimensions(axis: 0)      // [1, T, 100]
        func report(_ name: String, _ value: Float) {
            XCTAssertGreaterThan(value, 0.9999, "BigVGAN \(name) diverges (cosine \(value))")
        }
        let wav = net(mel)                                                 // [1, T·256, 1]
        report("waveform", cosine(wav[0][0..., 0], r["output"]!))
    }

    /// PARITY end to end: the BigVGAN mel front end, the CFM midpoint sampler seeded with the reference's
    /// `y0`, and the vocoder — the restored mel and the restored waveform against the recorded oracle.
    func testEndToEndParityOnTheReleasedWeights() throws {
        try requireMLXRuntime()
        let env = ProcessInfo.processInfo.environment
        guard let vrPath = env["IK_VAL_VOICERESTORE"], let bvPath = env["IK_VAL_BIGVGAN"],
              let recordPath = env["IK_PARITY_VOICERESTORE_E2E"] else {
            throw XCTSkip("set IK_VAL_VOICERESTORE, IK_VAL_BIGVGAN and IK_PARITY_VOICERESTORE_E2E")
        }
        let net = NFKMLXVoiceRestoreFactory.makeNet()
        try NFKMLXVoiceRestoreFactory.loadWeights(into: net, from: URL(fileURLWithPath: vrPath))
        let vocoder = NFKMLXBigVGANFactory.makeNet()
        try vocoder.loadWeights(from: URL(fileURLWithPath: bvPath))
        let r = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: recordPath)).arrays
        func report(_ name: String, _ value: Float) {
            XCTAssertGreaterThan(value, 0.9999, "e2e \(name) diverges (cosine \(value))")
        }

        // Mel front end.
        let audio = r["audio"]!.asArray(Float.self)
        let mel = NFKMLXVoiceRestoreMel()(audio)                          // [1, T, 100]
        report("mel", cosine(mel[0], r["mel"]!.transposed(1, 0)))

        // CFM sampler seeded with the reference y0 (8 steps, cfg 0.5), then the vocoder.
        let processed = r["processed"]!.expandedDimensions(axis: 0)       // [1, T, 100]
        let y0 = r["y0"]!.expandedDimensions(axis: 0)
        let restored = NFKMLXVoiceRestoreSampler.sample(net, processed: processed, steps: 8, cfgStrength: 0.5, noise: y0)
        report("restored", cosine(restored[0], r["restored"]!))
        let wav = vocoder(restored)                                       // [1, T·256, 1]
        report("waveform", cosine(wav[0][0..., 0], r["output"]!))
    }
}
