//
//  NFKMLXResembleEnhanceTests.swift
//  InferKitMLXTests
//
//  Resemble Enhance (resemble-ai, MIT) general speech restorer. The five networks evaluate MLX
//  arrays, so these tests skip without a Metal library for MLX (see Tools/mlx-metallib.sh). Parity
//  is gated on the released enhancer_stage2 checkpoint + the recorded oracles: set IK_VAL_REENHANCE
//  to the checkpoint (`mp_rank_00_model_states.pt`) and IK_PARITY_REENHANCE to the directory
//  holding the `reenhance_*.safetensors` records (`run_reference.py reenhance_*`).
//

import XCTest
import InferKit
import MLX
import MLXNN
@testable import InferKitMLX

final class NFKMLXResembleEnhanceTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    private func cosine(_ mine: MLXArray, _ reference: MLXArray) -> Float {
        eval(mine)
        let a = mine.asArray(Float.self), b = reference.asArray(Float.self)
        let n = min(a.count, b.count)
        var dot: Float = 0, na: Float = 0, nb: Float = 0
        for i in 0 ..< n { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
        return dot / (sqrtf(na) * sqrtf(nb) + 1e-20)
    }

    /// `[C, T]` record tensor → `[1, T, C]` NLC.
    private func nlc(_ r: MLXArray) -> MLXArray { r.transposed(1, 0).expandedDimensions(axis: 0) }

    private func load() throws -> (NFKMLXResembleEnhance, String) {
        try requireMLXRuntime()
        let env = NFKMLXValidationConfig.environment
        guard let weights = env["IK_VAL_REENHANCE"], let records = env["IK_PARITY_REENHANCE"] else {
            throw XCTSkip("set IK_VAL_REENHANCE (checkpoint) and IK_PARITY_REENHANCE (records dir)")
        }
        let net = NFKMLXResembleEnhanceFactory.makeNet()
        try NFKMLXResembleEnhanceFactory.loadWeights(into: net, from: URL(fileURLWithPath: weights))
        return (net, records)
    }

    private func record(_ dir: String, _ mode: String) throws -> [String: MLXArray] {
        try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: "\(dir)/reenhance_\(mode).safetensors")).arrays
    }

    /// PARITY: the mel front end (torchaudio magnitude mel + preemphasis + amp-to-db + headroom normalize).
    func testMelParity() throws {
        let (net, dir) = try load()
        let r = try record(dir, "mel")
        let mel = net.mel(r["wav"]!.asArray(Float.self))                    // [1, frames, 128]
        XCTAssertGreaterThan(cosine(mel[0], r["output"]!.transposed(1, 0)), 0.9999)
    }

    /// PARITY: the IRMAE autoencoder, encode then decode.
    func testIRMAEParity() throws {
        let (net, dir) = try load()
        let r = try record(dir, "irmae")
        let z = net.irmae.encode(nlc(r["x"]!))                             // [1, t, 64]
        XCTAssertGreaterThan(cosine(z[0], r["z"]!.transposed(1, 0)), 0.9999)
        let h = net.irmae.decode(z)
        XCTAssertGreaterThan(cosine(h[0], r["output"]!.transposed(1, 0)), 0.9999)
    }

    /// PARITY: the CFM velocity net (WaveNet) at a fixed seam, and the exponential-decay midpoint sampler.
    func testCFMParity() throws {
        let (net, dir) = try load()
        let r = try record(dir, "cfm")
        let v = net.cfm.velocity(psiT: nlc(r["psit"]!), t: 0.5, condition: nlc(r["x"]!))
        XCTAssertGreaterThan(cosine(v[0], r["v"]!.transposed(1, 0)), 0.9999)
        let psi1 = net.cfm.sample(condition: nlc(r["x"]!), psi0: nlc(r["psi0"]!), nfe: 32)
        XCTAssertGreaterThan(cosine(psi1[0], r["output"]!.transposed(1, 0)), 0.9999)
    }

    /// PARITY: the UnivNet location-variable-convolution vocoder, features + a recorded noise → waveform.
    func testUnivNetParity() throws {
        let (net, dir) = try load()
        let r = try record(dir, "univnet")
        let wav = net.vocoder(nlc(r["x"]!), noise: nlc(r["z"]!), npad: 10)  // [1, samples, 1]
        XCTAssertGreaterThan(cosine(wav[0][0..., 0], r["output"]!), 0.999)
    }

    /// PARITY: the stage-1 STFT-mask denoiser, a mixed waveform → a cleaned waveform.
    func testDenoiserParity() throws {
        let (net, dir) = try load()
        let r = try record(dir, "denoiser")
        let out = net.denoiser(r["wav"]!)                                  // [samples]
        XCTAssertGreaterThan(cosine(out, r["output"]!), 0.999)
    }

    /// PARITY end to end: the full enhance path (denoiser blend, LCFM prior + sampler, vocoder) with the
    /// recorded noises, and the final waveform.
    func testEndToEndParity() throws {
        let (net, dir) = try load()
        let r = try record(dir, "e2e")
        let out = net.enhance(r["wav"]!.asArray(Float.self), lambd: 0.5, tau: 0.5, nfe: 32,
                              tauNoise: nlc(r["tau_noise"]!), vocoderNoise: nlc(r["voc_z"]!))
        XCTAssertGreaterThan(cosine(out[0][0..., 0], r["output"]!), 0.999)
    }
}
