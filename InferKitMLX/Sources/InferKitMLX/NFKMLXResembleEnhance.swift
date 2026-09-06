// Resemble Enhance (resemble-ai, MIT), a general speech restorer that fixes noise, reverberation,
// clipping, and band-limiting together. A five-network system: a stage-1 STFT-mask denoiser (`NFKMLXResembleDenoiser`),
// a Latent Conditional Flow Matching stage (an IRMAE autoencoder + a WaveNet CFM velocity net,
// `NFKMLXResembleLCFM`), and a UnivNet location-variable-convolution vocoder (`NFKMLXResembleUnivNet`),
// plus this file's mel front end and orchestration. Grounded on the released source read 2026-09-06
// (`resemble_enhance/`), at reference parity against it seam by seam.
//
// The mel front end reproduces `resemble_enhance.melspec.MelSpectrogram`: preemphasis, a torchaudio
// magnitude mel (Slaney scale, Slaney-normalized, zero-centered STFT), amp-to-db, and a headroom
// normalization. Tensors flow NLC `[B, T, C]` (MLX's Conv1d layout); a mel is `[1, frames, 128]`.

import Foundation
import InferKit
import MLX
import MLXFFT
import MLXNN

/// The Resemble Enhance hyperparameters (`hparams.yaml` of `enhancer_stage2`).
public struct NFKMLXResembleConfiguration: Sendable {
    public var wavRate: Int = 44_100
    public var nFFT: Int = 2048
    public var winSize: Int = 2048
    public var hopSize: Int = 420
    public var numMels: Int = 128
    public var stftMagnitudeMin: Float = 1e-4
    public var preemphasis: Float = 0.97
    public var vocoderExtraDim: Int = 32
    public var latentDim: Int = 64
    public var zScale: Float = 6
    public var univnetNC: Int = 96
    public var cfmSolverNFE: Int = 64
    public var cfmTimeMappingDivisor: Int = 4

    public init() {}

    var vocoderInputDim: Int { numMels + vocoderExtraDim }
}

/// The mel front end (`resemble_enhance.melspec.MelSpectrogram`). Held as a value type outside any module
/// graph, so its filterbank and window stay out of `parameters()`.
struct NFKMLXResembleMel {
    let config: NFKMLXResembleConfiguration
    let window: [Float]
    let filters: MLXArray                                   // [bins, numMels]

    init(_ config: NFKMLXResembleConfiguration) {
        self.config = config
        window = nfkPeriodicHann(config.winSize)
        filters = NFKMLXMel.melFilters(sampleRate: config.wavRate, bins: config.nFFT / 2 + 1,
                                       nMels: config.numMels, fMinimum: 0, fMaximum: Float(config.wavRate) / 2)
    }

    /// `min_level_db = 20·log10(stft_magnitude_min)`; the headroom normalization divides by `-min + 15`.
    private var minLevelDB: Float { 20 * log10f(config.stftMagnitudeMin) }

    /// A magnitude mel from raw samples → `[1, frames, numMels]`. `dropsLast` follows `to_mel`'s
    /// `mel_fn(x)[..., :-1]`.
    func callAsFunction(_ samples: [Float], dropsLast: Bool = true) -> MLXArray {
        let nFFT = config.nFFT, hop = config.hopSize, pad = config.nFFT / 2
        // Preemphasis: pad one zero at the front, then x[i] - 0.97·x[i-1].
        var pre = [Float](repeating: 0, count: samples.count)
        let p = config.preemphasis
        pre[0] = samples[0]
        for i in 1 ..< samples.count { pre[i] = samples[i] - p * samples[i - 1] }
        // Zero (constant) center padding, `torch.stft(center: true, pad_mode: "constant")`.
        var padded = [Float](repeating: 0, count: pre.count + 2 * pad)
        for i in 0 ..< pre.count { padded[i + pad] = pre[i] }
        let frames = 1 + (padded.count - nFFT) / hop
        var frameData = [Float](repeating: 0, count: frames * nFFT)
        for f in 0 ..< frames {
            let start = f * hop
            for j in 0 ..< nFFT { frameData[f * nFFT + j] = padded[start + j] * window[j] }
        }
        let framed = frameData.withUnsafeBufferPointer { MLXArray($0, [frames, nFFT]) }
        let spectrum = rfft(framed, axis: 1)                                  // [frames, bins] complex
        let magnitude = sqrt(spectrum.realPart() * spectrum.realPart()
                             + spectrum.imaginaryPart() * spectrum.imaginaryPart())   // power=1
        var mel = magnitude.matmul(filters)                                  // [frames, numMels]
        // amp_to_db: clamp_min(min).log10()·20, then headroom normalize.
        mel = maximum(mel, MLXArray(config.stftMagnitudeMin))
        mel = log(mel) / logf(10) * 20
        mel = (mel - minLevelDB) / (-minLevelDB + 15)
        let count = dropsLast ? frames - 1 : frames
        return mel[0 ..< count, 0...].reshaped([1, count, config.numMels])
    }
}

/// The global scalar `Normalizer` (`resemble_enhance.common.Normalizer`), loaded from the checkpoint. At
/// inference it applies `(x - mean) / std` with the running statistics; `std = sqrt(var + 1e-9)`.
struct NFKMLXResembleNormalizer {
    let mean: Float
    let std: Float

    init(mean: Float, variance: Float) {
        self.mean = mean
        self.std = (variance + 1e-9).squareRoot()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { (x - mean) / std }
}

// MARK: - Orchestrator

/// The full Resemble Enhance model: the stage-1 denoiser, the LCFM stage (IRMAE + CFM), and the UnivNet
/// vocoder, plus the mel front end and the global normalizer. `enhance` runs the `enhance()` inference
/// path (`resemble_enhance/enhancer/{enhancer,inference}.py`).
public final class NFKMLXResembleEnhance: Module {
    @ModuleInfo(key: "denoiser") var denoiser: NFKMLXResembleDenoiser
    @ModuleInfo(key: "irmae") var irmae: NFKMLXResembleIRMAE
    @ModuleInfo(key: "cfm") var cfm: NFKMLXResembleCFM
    @ModuleInfo(key: "vocoder") var vocoder: NFKMLXResembleUnivNet
    let config: NFKMLXResembleConfiguration
    let mel: NFKMLXResembleMel
    var normalizer: NFKMLXResembleNormalizer

    public init(_ config: NFKMLXResembleConfiguration = .init()) {
        self.config = config
        _denoiser.wrappedValue = NFKMLXResembleDenoiser(hopSize: config.hopSize)
        _irmae.wrappedValue = NFKMLXResembleIRMAE(inputDim: config.numMels, outputDim: config.vocoderInputDim, latentDim: config.latentDim)
        _cfm.wrappedValue = NFKMLXResembleCFM(condDim: config.numMels, outputDim: config.latentDim, timeMappingDivisor: config.cfmTimeMappingDivisor)
        _vocoder.wrappedValue = NFKMLXResembleUnivNet(config)
        mel = NFKMLXResembleMel(config)
        normalizer = NFKMLXResembleNormalizer(mean: 0, variance: 1)
        super.init()
    }

    private func peakNormalize(_ x: [Float]) -> [Float] {
        let m = max(x.map { abs($0) }.max() ?? 0, 1e-7)
        return x.map { $0 / m }
    }

    /// The `enhance()` inference path. `lambd` blends the denoised mel into the mix mel; `tau` mixes noise
    /// into the encoded prior; `nfe` is the CFM function-evaluation budget. `tauNoise` / `vocoderNoise`
    /// override the two random draws (for reproducible parity); when nil they are drawn from MLX's RNG.
    public func enhance(_ samples: [Float], lambd: Float = 0.5, tau: Float = 0.5, nfe: Int = 32,
                        tauNoise: MLXArray? = nil, vocoderNoise: MLXArray? = nil) -> MLXArray {
        var x = peakNormalize(samples)
        x += [Float](repeating: 0, count: 441)                 // inference_chunk npad
        x = peakNormalize(x)                                   // enhancer.forward
        let xMelOriginal = normalizer(mel(x))                  // [1, frames, numMels]
        var xMelDenoised = xMelOriginal
        if lambd > 0 {
            let denWav = denoiser(MLXArray(x)).asArray(Float.self)
            let xMelDen = normalizer(mel(denWav))
            xMelDenoised = lambd * xMelDen + (1 - lambd) * xMelOriginal
        }
        let psi0Enc = config.zScale * irmae.encode(xMelOriginal)   // [1, frames, latentDim]
        let noise = tauNoise ?? MLXRandom.normal(psi0Enc.shape)
        let psi0 = tau * noise + (1 - tau) * psi0Enc
        let z = cfm.sample(condition: xMelDenoised, psi0: psi0, nfe: nfe) / config.zScale
        let h = irmae.decode(z)                                // [1, frames, vocoderInputDim]
        let zn = vocoderNoise ?? MLXRandom.normal([1, h.dim(1) + 10, vocoder.dNoise])
        return vocoder(h, noise: zn)                           // [1, samples, 1]
    }
}

// MARK: - Weight loading

/// Registration and weight loading for Resemble Enhance.
@objc(NFKMLXResembleEnhance_Factory)
public final class NFKMLXResembleEnhanceFactory: NSObject {
    public static let modelName = "resemble-enhance"

    static func makeNet(_ config: NFKMLXResembleConfiguration = .init()) -> NFKMLXResembleEnhance {
        NFKMLXResembleEnhance(config)
    }

    /// Loads the released `enhancer_stage2` checkpoint (`ds/G/default/mp_rank_00_model_states.pt`, key
    /// `module`): fuses the weight-norm pairs, remaps the reference's Sequential slots onto the module
    /// layout, transposes the convolution weights, and loads each sub-network.
    static func loadWeights(into enhance: NFKMLXResembleEnhance, from url: URL) throws {
        let ckpt = try NFKMLXWeights.loadCheckpoint(url: url)
        // The DeepSpeed shard nests everything under `module`, which the reader does not unwrap.
        var arrays = [String: MLXArray]()
        for (key, value) in ckpt.arrays { arrays[key.hasPrefix("module.") ? String(key.dropFirst("module.".count)) : key] = value }
        let fused = NFKMLXMusic3.fusedWeightNorm(arrays)
        var groups: [String: [(String, MLXArray)]] = ["ae": [], "cfm": [], "voc": [], "den": []]
        for (key, value) in fused {
            guard let (component, name) = remapKey(key) else { continue }
            groups[component, default: []].append((name, transpose(name, value, ckpt.needsConvTranspose)))
        }
        try NFKMLXWeights.apply(groups["ae"]!, to: enhance.irmae)
        try NFKMLXWeights.apply(groups["cfm"]!, to: enhance.cfm)
        try NFKMLXWeights.apply(groups["voc"]!, to: enhance.vocoder)
        try NFKMLXWeights.apply(groups["den"]!, to: enhance.denoiser)
        if let mean = arrays["normalizer.running_mean_unsafe"], let varc = arrays["normalizer.running_var_unsafe"] {
            enhance.normalizer = NFKMLXResembleNormalizer(mean: mean.item(Float.self), variance: varc.item(Float.self))
        }
    }

    private static func transpose(_ name: String, _ v: MLXArray, _ needsConvTranspose: Bool) -> MLXArray {
        guard needsConvTranspose else { return v }
        if v.ndim == 4 { return v.transposed(0, 2, 3, 1) }               // Conv2d
        if v.ndim == 3 {
            return name.contains("convt_pre") ? v.transposed(1, 2, 0) : v.transposed(0, 2, 1)
        }
        return v
    }

    private static func resBlockSlot(_ slot: Int) -> String? {
        if slot % 3 == 0 { return "norms.\(slot / 3)" }
        if slot % 3 == 2 { return "convs.\(slot / 3)" }
        return nil
    }

    /// Maps a checkpoint key to `(component, module-relative key)`, or nil to drop it.
    static func remapKey(_ key: String) -> (component: String, name: String)? {
        if key.contains(".filter") || key.hasPrefix("mel_fn.") || key.hasPrefix("dummy") || key.hasPrefix("normalizer.") { return nil }
        if key.hasPrefix("lcfm.ae.") {
            if key.hasPrefix("lcfm.ae.head") || key.hasPrefix("lcfm.ae.estimator") { return nil }
            return ("ae", remapResBlockPath(String(key.dropFirst("lcfm.ae.".count)), containers: ["encoder", "decoder"], indices: 1 ... 4))
        }
        if key.hasPrefix("lcfm.cfm.") { return ("cfm", String(key.dropFirst("lcfm.cfm.".count))) }
        if key.hasPrefix("vocoder.") {
            guard let name = remapUnivNet(String(key.dropFirst("vocoder.".count))) else { return nil }
            return ("voc", name)
        }
        if key.hasPrefix("denoiser.") {
            if key.hasPrefix("denoiser.mel_fn") { return nil }
            return ("den", remapDenoiser(String(key.dropFirst("denoiser.".count))))
        }
        return nil
    }

    /// Rewrites a `<container>.<index>.<slot>...` ResBlock path so the group-norm/convolution slots land
    /// on the named `norms`/`convs` arrays.
    private static func remapResBlockPath(_ rest: String, containers: [String], indices: ClosedRange<Int>) -> String {
        var segs = rest.split(separator: ".").map(String.init)
        if segs.count >= 3, containers.contains(segs[0]), let mid = Int(segs[1]), indices.contains(mid),
           let slot = Int(segs[2]), let mapped = resBlockSlot(slot) {
            segs.replaceSubrange(2 ... 2, with: mapped.split(separator: ".").map(String.init))
        }
        return segs.joined(separator: ".")
    }

    private static func remapUnivNet(_ rest: String) -> String? {
        var segs = rest.split(separator: ".").map(String.init)
        if segs[0] == "conv_pre" { return rest }
        if segs[0] == "conv_post" {                                       // conv_post.1.X → conv_post.X
            if segs.count >= 2, segs[1] == "1" { segs.remove(at: 1) }
            return segs.joined(separator: ".")
        }
        guard segs[0] == "blocks", segs.count >= 3 else { return rest }
        switch segs[2] {
        case "amp_block":                                                // blocks.N.amp_block.J.<slot>.X
            let slotMap = ["0": "conv1", "1": "act", "2": "conv2"]
            if segs.count >= 5, let m = slotMap[segs[4]] { segs[4] = m }
            return segs.joined(separator: ".")
        case "conv_blocks":                                              // blocks.N.conv_blocks.J.1.X → …J.X
            if segs.count >= 5, segs[4] == "1" { segs.remove(at: 4) }
            return segs.joined(separator: ".")
        case "convt_pre":                                               // blocks.N.convt_pre.1.X → …convt_pre.X
            if segs.count >= 4, segs[3] == "1" { segs.remove(at: 3) }
            return segs.joined(separator: ".")
        case "kernel_predictor":
            guard segs.count >= 4 else { return rest }
            if segs[3] == "input_conv" {                                // …input_conv.0.X → …input_conv.X
                if segs.count >= 5, segs[4] == "0" { segs.remove(at: 4) }
                return segs.joined(separator: ".")
            }
            if segs[3] == "residual_convs" {                           // …residual_convs.J.<1|3>.X → …J.conv{1|2}.X
                if segs.count >= 6 { segs[5] = segs[5] == "1" ? "conv1" : "conv2" }
                return segs.joined(separator: ".")
            }
            return rest                                                 // kernel_conv, bias_conv
        default:
            return rest
        }
    }

    private static func remapDenoiser(_ rest: String) -> String {
        var segs = rest.split(separator: ".").map(String.init)
        for i in segs.indices where segs[i] == "res_block1" || segs[i] == "res_block2" {
            if i + 1 < segs.count, let slot = Int(segs[i + 1]), let mapped = resBlockSlot(slot) {
                segs.replaceSubrange((i + 1) ... (i + 1), with: mapped.split(separator: ".").map(String.init))
            }
            break
        }
        return segs.joined(separator: ".")
    }
}
