// Shared recurrent primitives for the speech-restoration ports. A GRU cell that owns its biases as
// parameters (MLXNN's own `GRU` keeps `b`/`bhn` as fixed non-parameters, so a released bias cannot be
// loaded into it), a bidirectional wrapper, and a checkpoint fold that turns PyTorch `nn.GRU` weights
// into these cells' parameters. MP-SENet's FFN and GTCRN's TRA / DPGRNN all read them.

import Foundation
import MLX
import MLXNN

/// A single-direction GRU cell mirroring MLXNN's own GRU math, owning `Wx`/`Wh`/`b`/`bhn` as parameters.
/// Gate order is PyTorch's `(reset, update, new)`; `b` is the input bias plus the reset/update hidden
/// bias, and `bhn` is the new-gate hidden bias applied after the reset gate.
final class NFKMLXGRUCell: Module {
    @ParameterInfo(key: "Wx") var wx: MLXArray                              // [3H, in]
    @ParameterInfo(key: "Wh") var wh: MLXArray                              // [3H, H]
    @ParameterInfo(key: "b") var b: MLXArray                               // [3H]
    @ParameterInfo(key: "bhn") var bhn: MLXArray                           // [H]
    let hiddenSize: Int

    init(inputSize: Int, hiddenSize: Int) {
        self.hiddenSize = hiddenSize
        _wx.wrappedValue = MLXArray.zeros([3 * hiddenSize, inputSize])
        _wh.wrappedValue = MLXArray.zeros([3 * hiddenSize, hiddenSize])
        _b.wrappedValue = MLXArray.zeros([3 * hiddenSize])
        _bhn.wrappedValue = MLXArray.zeros([hiddenSize])
    }

    /// `x` `[B, L, in]` → `[B, L, H]`.
    func callAsFunction(_ x0: MLXArray) -> MLXArray {
        let h = hiddenSize
        let x = addMM(b, x0, wx.transposed(1, 0))                          // [B, L, 3H]
        let xrz = x[.ellipsis, .stride(to: -h)]
        let xn = x[.ellipsis, .stride(from: -h)]
        var hidden: MLXArray?
        var all = [MLXArray]()
        for index in 0 ..< x.dim(-2) {
            var rz = xrz[.ellipsis, index, 0...]
            // The new-gate hidden bias `bhn` applies at EVERY step, including the first, where the
            // hidden state is zero but the bias is not (`n = tanh(W_in x + b_in + r·(W_hn h + b_hn))`).
            // MLX's own GRU drops it at step 0; PyTorch keeps it, so this port keeps it too.
            var hProjN = bhn
            if let previous = hidden {
                let hProj = matmul(previous, wh.transposed(1, 0))
                rz = rz + hProj[.ellipsis, .stride(to: -h)]
                hProjN = hProj[.ellipsis, .stride(from: -h)] + bhn
            }
            rz = sigmoid(rz)
            let parts = split(rz, parts: 2, axis: -1)
            let (r, z) = (parts[0], parts[1])
            let n = tanh(xn[.ellipsis, index, 0...] + r * hProjN)
            hidden = hidden == nil ? (1 - z) * n : (1 - z) * n + z * hidden!
            all.append(hidden!)
        }
        return stacked(all, axis: -2)
    }
}

/// A bidirectional GRU: a forward pass and a backward pass over the reversed sequence, concatenated on
/// the feature axis (`nn.GRU(bidirectional: true)`). Keyed `forward` / `backward` to match the fold.
final class NFKMLXBiGRU: Module {
    @ModuleInfo(key: "forward") var forwardGRU: NFKMLXGRUCell
    @ModuleInfo(key: "backward") var backwardGRU: NFKMLXGRUCell

    init(inputSize: Int, hiddenSize: Int) {
        _forwardGRU.wrappedValue = NFKMLXGRUCell(inputSize: inputSize, hiddenSize: hiddenSize)
        _backwardGRU.wrappedValue = NFKMLXGRUCell(inputSize: inputSize, hiddenSize: hiddenSize)
    }

    static func reversedSequence(_ x: MLXArray) -> MLXArray {
        let length = x.dim(-2)
        let indices = MLXArray((0 ..< length).reversed().map { Int32($0) })
        return take(x, indices, axis: x.ndim - 2)
    }

    /// `[B, L, in]` → `[B, L, 2H]`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let forward = forwardGRU(x)
        let backward = Self.reversedSequence(backwardGRU(Self.reversedSequence(x)))
        return concatenated([forward, backward], axis: -1)
    }
}

/// Folds every PyTorch `nn.GRU` in a checkpoint into `NFKMLXGRUCell` parameters. A bidirectional GRU
/// (its `weight_ih_l0_reverse` is present) becomes `<base>forward.*` and `<base>backward.*`; a
/// unidirectional one becomes `<base>*` directly. Weights map 1:1 to `Wx`/`Wh`; the bias combines the
/// two PyTorch biases the way MLXNN's GRU expects — `b` = `bias_ih` plus the reset/update part of
/// `bias_hh`, and `bhn` = the new-gate part of `bias_hh`.
enum NFKMLXRecurrentFold {
    static func fold(_ arrays: [String: MLXArray]) -> [String: MLXArray] {
        var out = [String: MLXArray]()
        var handled = Set<String>()
        for key in arrays.keys where key.hasSuffix("weight_ih_l0") {
            let base = String(key.dropLast("weight_ih_l0".count))         // "...att_gru." / "...gru."
            let bidirectional = arrays["\(base)weight_ih_l0_reverse"] != nil
            fold(base: base, tag: "", direction: bidirectional ? "forward." : "", from: arrays, into: &out, handled: &handled)
            if bidirectional {
                fold(base: base, tag: "_reverse", direction: "backward.", from: arrays, into: &out, handled: &handled)
            }
        }
        for (key, value) in arrays where !handled.contains(key) { out[key] = value }
        return out
    }

    private static func fold(base: String, tag: String, direction: String,
                             from arrays: [String: MLXArray], into out: inout [String: MLXArray],
                             handled: inout Set<String>) {
        guard let weightIH = arrays["\(base)weight_ih_l0\(tag)"],
              let weightHH = arrays["\(base)weight_hh_l0\(tag)"],
              let biasIH = arrays["\(base)bias_ih_l0\(tag)"],
              let biasHH = arrays["\(base)bias_hh_l0\(tag)"] else { return }
        let hidden = weightIH.dim(0) / 3
        let hiddenRZ = MLX.padded(biasHH[0 ..< 2 * hidden], widths: [IntOrPair((0, hidden))], mode: .constant)
        out["\(base)\(direction)Wx"] = weightIH
        out["\(base)\(direction)Wh"] = weightHH
        out["\(base)\(direction)b"] = biasIH + hiddenRZ
        out["\(base)\(direction)bhn"] = biasHH[2 * hidden ..< 3 * hidden]
        for name in ["weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0"] {
            handled.insert("\(base)\(name)\(tag)")
        }
    }
}
