// The Latent Conditional Flow Matching stage of Resemble Enhance: an IRMAE autoencoder that compresses
// the mel to a 64-channel latent, and a WaveNet CFM velocity net that samples that latent conditioned on
// the (denoised) mel through an exponential-decay midpoint ODE. Grounded on the released
// `resemble_enhance/enhancer/lcfm/{irmae,cfm,wn,lcfm}.py`.
//
// Tensors flow NLC `[B, T, C]`. Sequential-of-Sequential modules are `[Module]` arrays of `UnaryLayer`s
// with marker entries for the parameter-free activations, so the checkpoint's numeric keys line up; the
// ResBlock's group-norm and convolution slots are remapped to named arrays in the loader.

import Foundation
import MLX
import MLXNN

// MARK: - Shared marker activations (occupy a parameter-free Sequential slot)

/// A GELU marker (exact erf GELU, `nn.GELU()`), holding no parameters.
final class NFKResembleGELU: Module, UnaryLayer {
    func callAsFunction(_ x: MLXArray) -> MLXArray { gelu(x) }
}

/// A Tanh marker.
final class NFKResembleTanh: Module, UnaryLayer {
    func callAsFunction(_ x: MLXArray) -> MLXArray { tanh(x) }
}

// MARK: - IRMAE

/// `ResBlock`: four (GroupNorm → GELU → dilated Conv1d) stages, added back to the input. The group norms
/// and convolutions are named arrays (`norms`/`convs`); the loader maps the reference's Sequential slots
/// (0,3,6,9 → norms; 2,5,8,11 → convs).
final class NFKResembleResBlock: Module, UnaryLayer {
    @ModuleInfo(key: "norms") var norms: [GroupNorm]
    @ModuleInfo(key: "convs") var convs: [Conv1d]

    init(channels: Int, dilations: [Int] = [1, 2, 4, 8]) {
        _norms.wrappedValue = (0 ..< 4).map { _ in GroupNorm(groupCount: 32, dimensions: channels, pytorchCompatible: true) }
        _convs.wrappedValue = dilations.map {
            Conv1d(inputChannels: channels, outputChannels: channels, kernelSize: 3, padding: $0, dilation: $0)
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for i in 0 ..< 4 {
            h = norms[i](h)
            h = gelu(h)
            h = convs[i](h)
        }
        return x + h
    }
}

/// The IRMAE autoencoder. `encode` runs the encoder to the 64-channel Tanh latent; `decode` runs the
/// decoder to the 160-channel vocoder input. The training-only `head` and `estimator` are not built.
public final class NFKMLXResembleIRMAE: Module {
    @ModuleInfo(key: "encoder") var encoder: [Module]
    @ModuleInfo(key: "decoder") var decoder: [Module]

    init(inputDim: Int, outputDim: Int, latentDim: Int, hiddenDim: Int = 1024, numIRMs: Int = 4) {
        var enc: [Module] = [Conv1d(inputChannels: inputDim, outputChannels: hiddenDim, kernelSize: 3, padding: 1)]
        enc.append(contentsOf: (0 ..< 4).map { _ in NFKResembleResBlock(channels: hiddenDim) })
        for i in 0 ..< numIRMs {
            enc.append(Conv1d(inputChannels: i == 0 ? hiddenDim : latentDim, outputChannels: latentDim, kernelSize: 1, bias: false))
        }
        enc.append(NFKResembleTanh())
        _encoder.wrappedValue = enc

        var dec: [Module] = [Conv1d(inputChannels: latentDim, outputChannels: hiddenDim, kernelSize: 3, padding: 1)]
        dec.append(contentsOf: (0 ..< 4).map { _ in NFKResembleResBlock(channels: hiddenDim) })
        dec.append(Conv1d(inputChannels: hiddenDim, outputChannels: outputDim, kernelSize: 1))
        _decoder.wrappedValue = dec
    }

    private func run(_ layers: [Module], _ x: MLXArray) -> MLXArray {
        var h = x
        for layer in layers { h = (layer as! UnaryLayer)(h) }
        return h
    }

    /// mel `[1, t, inputDim]` → latent `[1, t, latentDim]`.
    public func encode(_ x: MLXArray) -> MLXArray { run(encoder, x) }
    /// latent `[1, t, latentDim]` → `[1, t, outputDim]`.
    public func decode(_ z: MLXArray) -> MLXArray { run(decoder, z) }
}

// MARK: - CFM velocity net (WaveNet)

/// `SinusodialTimeEmbedding`: `t` → `[cat(sin(t·10^p), cos(t·10^p))]`, `p = linspace(0, 4, dEmbed/2)`.
struct NFKResembleTimeEmbedding {
    let dEmbed: Int
    let powers: MLXArray                                    // [dEmbed/2]

    init(dEmbed: Int) {
        self.dEmbed = dEmbed
        let half = dEmbed / 2
        powers = MLXArray((0 ..< half).map { 4 * Float($0) / Float(half - 1) })
    }

    /// scalar `t` (broadcast per batch) → `[1, dEmbed]`.
    func callAsFunction(_ t: Float) -> MLXArray {
        let scaled = MLXArray(t) * pow(MLXArray(10, dtype: .float32), powers)   // [half]
        return concatenated([sin(scaled), cos(scaled)], axis: 0).reshaped([1, dEmbed])
    }
}

/// A DiffWave-style `WNLayer`: a global 1×1 conv adds the time embedding, a dilated conv mixes time, a
/// local 1×1 conv adds the condition, a gated tanh/sigmoid, then a residual output and a skip.
final class NFKResembleWNLayer: Module {
    @ModuleInfo(key: "gconv") var gconv: Conv1d
    @ModuleInfo(key: "dconv") var dconv: Conv1d
    @ModuleInfo(key: "lconv") var lconv: Conv1d
    @ModuleInfo(key: "out") var out: Conv1d
    let hidden: Int

    init(hidden: Int, localDim: Int, globalDim: Int, kernelSize: Int, dilation: Int) {
        self.hidden = hidden
        _gconv.wrappedValue = Conv1d(inputChannels: globalDim, outputChannels: hidden, kernelSize: 1)
        _lconv.wrappedValue = Conv1d(inputChannels: localDim, outputChannels: hidden * 2, kernelSize: 1)
        _dconv.wrappedValue = Conv1d(inputChannels: hidden, outputChannels: hidden * 2, kernelSize: kernelSize,
                                     padding: dilation * (kernelSize - 1) / 2, dilation: dilation)
        _out.wrappedValue = Conv1d(inputChannels: hidden, outputChannels: hidden * 2, kernelSize: 1)
    }

    /// `z [1, t, hidden]`, `l [1, t, 2·hidden precomputed]` — actually `l` is the raw condition, and this
    /// layer's `lconv` widens it. `g [1, 1, globalDim]`. Returns `(output, skip)`.
    func callAsFunction(_ z0: MLXArray, l: MLXArray, g: MLXArray) -> (MLXArray, MLXArray) {
        var z = z0 + gconv(g)                                // broadcast over time
        z = dconv(z)
        z = z + lconv(l)
        let parts = split(z, parts: 2, axis: 2)              // gated tanh/sigmoid over channels
        z = tanh(parts[0]) * sigmoid(parts[1])
        let h = out(z)
        let hs = split(h, parts: 2, axis: 2)
        let o = (hs[0] + z0) / sqrtf(2)
        return (o, hs[1])
    }
}

/// The `WN` velocity net: `start` 1×1, 30 `WNLayer`s (dilation cycle 5), skips summed and `end` 1×1. The
/// local condition is instance-normalized per channel over time before the layers read it.
public final class NFKMLXResembleWN: Module {
    @ModuleInfo(key: "start") var start: Conv1d
    @ModuleInfo(key: "end") var end: Conv1d
    @ModuleInfo(key: "layers") var layers: [NFKResembleWNLayer]

    init(inputDim: Int, outputDim: Int, localDim: Int, globalDim: Int,
         nLayers: Int = 30, kernelSize: Int = 3, dilationCycle: Int = 5, hidden: Int = 512) {
        _start.wrappedValue = Conv1d(inputChannels: inputDim, outputChannels: hidden, kernelSize: 1)
        _end.wrappedValue = Conv1d(inputChannels: hidden, outputChannels: outputDim, kernelSize: 1)
        _layers.wrappedValue = (0 ..< nLayers).map {
            NFKResembleWNLayer(hidden: hidden, localDim: localDim, globalDim: globalDim,
                               kernelSize: kernelSize, dilation: 1 << ($0 % dilationCycle))
        }
    }

    /// `z [1, t, inputDim]`, `l [1, t, localDim]`, `g [1, 1, globalDim]` → `[1, t, outputDim]`.
    func callAsFunction(_ z0: MLXArray, l l0: MLXArray, g: MLXArray) -> MLXArray {
        var z = start(z0)
        // InstanceNorm1d(affine: false) on the local condition: per-channel over time.
        let mean = l0.mean(axis: 1, keepDims: true)
        let varc = l0.variance(axis: 1, keepDims: true)
        let l = (l0 - mean) * rsqrt(varc + 1e-5)
        var skipSum: MLXArray?
        for layer in layers {
            let (o, s) = layer(z, l: l, g: g)
            z = o
            skipSum = skipSum == nil ? s : skipSum! + s
        }
        let skips = skipSum! / sqrtf(Float(layers.count))
        return end(skips)
    }
}

// MARK: - CFM sampler

/// The Conditional Flow Matching sampler: the WaveNet velocity net and the exponential-decay midpoint
/// ODE (`Solver` in `cfm.py`). The generation runs `t: 0 → 1` from a starting `psi0`.
public final class NFKMLXResembleCFM: Module {
    @ModuleInfo(key: "net") var net: NFKMLXResembleWN
    let timeEmbedding: NFKResembleTimeEmbedding
    let timeMappingDivisor: Int

    init(condDim: Int, outputDim: Int, timeEmbDim: Int = 128, timeMappingDivisor: Int = 4) {
        _net.wrappedValue = NFKMLXResembleWN(inputDim: outputDim, outputDim: outputDim, localDim: condDim, globalDim: timeEmbDim)
        timeEmbedding = NFKResembleTimeEmbedding(dEmbed: timeEmbDim)
        self.timeMappingDivisor = timeMappingDivisor
    }

    /// The velocity `v(psi_t, t | x)`: the time-embedded WaveNet, `t` clamped to `[0, 1]`.
    func velocity(psiT: MLXArray, t: Float, condition: MLXArray) -> MLXArray {
        let tc = max(0, min(1, t))
        let g = timeEmbedding(tc)                            // [1, timeEmbDim]
        return net(psiT, l: condition, g: g.reshaped([1, 1, timeEmbDim]))
    }

    private var timeEmbDim: Int { timeEmbedding.dEmbed }

    /// `exponential_decay_mapping(t, n)`: `h(t) = (a^t − 1)/(a − 1)` with `a` solving `h(1/n) = 0.5`.
    /// The constant `a` is found by Newton's method (scipy `fsolve` in the reference).
    static func exponentialDecay(_ ts: [Float], n: Int) -> [Float] {
        let target: Float = 1 / Float(n)
        func h(_ t: Float, _ a: Double) -> Double { (pow(a, Double(t)) - 1) / (a - 1) }
        // Solve h(1/n, a) = 0.5 for a by Newton on g(a) = h(target, a) - 0.5.
        var a = 0.1
        for _ in 0 ..< 100 {
            let g = h(target, a) - 0.5
            let d = (h(target, a + 1e-6) - h(target, a - 1e-6)) / 2e-6
            let step = g / d
            a -= step
            if abs(step) < 1e-14 { break }
            if a <= 0 { a = 1e-6 }
        }
        return ts.map { Float(h($0, a)) }
    }

    /// Sample from a starting `psi0 [1, t, outputDim]` conditioned on `condition [1, t, condDim]`. `nfe`
    /// is the number of function evaluations; midpoint uses `nfe/2` steps.
    public func sample(condition: MLXArray, psi0: MLXArray, nfe: Int = 32) -> MLXArray {
        let nSteps = nfe / 2                                 // midpoint
        let linear = (0 ... nSteps).map { Float($0) / Float(nSteps) }
        let ts = Self.exponentialDecay(linear, n: timeMappingDivisor)
        var psi = psi0
        for i in 0 ..< nSteps {
            let t = ts[i], dt = ts[i + 1] - ts[i]
            let k1 = velocity(psiT: psi, t: t, condition: condition)
            let mid = psi + (dt / 2) * k1
            let k2 = velocity(psiT: mid, t: t + dt / 2, condition: condition)
            psi = psi + dt * k2
            eval(psi)
        }
        return psi
    }
}
