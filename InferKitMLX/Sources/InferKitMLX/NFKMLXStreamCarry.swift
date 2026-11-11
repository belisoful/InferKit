//
//  NFKMLXStreamCarry.swift
//  InferKitMLX
//
//  The state a causal network carries from one unit of a live stream to the next.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import MLX

/// The state a causal network carries between the units of a live stream: each causal convolution's last
/// input frames and each time-recurrent cell's last hidden state, keyed by the layer that owns it.
///
/// @discussion A streamed forward runs inside `streaming(_:)`, and the layers read the carry through
/// `current`. Outside a stream `current` is nil and each layer pads with zeros and starts its recurrence from
/// zero, which is the clip forward. The first unit of a stream finds no carried state and runs exactly as
/// the start of a clip, so a stream answers each frame from the same inputs a clip of the same samples
/// reads.
final class NFKMLXStreamCarry: @unchecked Sendable {

    /// The carry of the streamed forward running on this task, or nil for a clip forward.
    @TaskLocal static var current: NFKMLXStreamCarry?

    private var arrays: [ObjectIdentifier: MLXArray] = [:]

    /// Runs `body` with this carry as `current`.
    func streaming<T>(_ body: () throws -> T) rethrows -> T {
        try Self.$current.withValue(self, operation: body)
    }

    /// Evaluates `outputs` with every carried array, so the next unit's graph starts from values rather
    /// than from the graphs of every unit before it.
    func evaluate(_ outputs: [MLXArray]) {
        eval(outputs + Array(arrays.values))
    }

    /// Forgets every carried array.
    func clear() {
        arrays.removeAll()
    }

    /// `x` with `frames` frames before it on `axis`: zeros in a clip forward and on a stream's first unit,
    /// else the last `frames` frames the owner's input ended with on the unit before. A causal convolution
    /// that pads its time axis at the front reads its input through this.
    static func causallyPadded(_ x: MLXArray, frames: Int, axis: Int, owner: AnyObject) -> MLXArray {
        guard frames > 0 else {
            return x
        }
        guard let carry = current else {
            let widths = (0 ..< x.ndim).map { $0 == axis ? IntOrPair((frames, 0)) : IntOrPair(0) }
            return MLX.padded(x, widths: widths, mode: .constant)
        }
        var shape = x.shape
        shape[axis] = frames
        let key = ObjectIdentifier(owner)
        let history = carry.arrays[key] ?? MLXArray.zeros(shape, dtype: x.dtype)
        let joined = concatenated([history, x], axis: axis)
        let length = joined.dim(axis)
        let kept = MLXArray(((length - frames) ..< length).map { Int32($0) })
        carry.arrays[key] = take(joined, kept, axis: axis)
        return joined
    }

    /// `x` after the frames its owner kept on `axis` from the unit before, and how many there were: none
    /// in a clip forward or on a stream's first unit. The last `keep` frames of the result are kept for
    /// the next unit. A cache-aware attention keeps its keys and values this way.
    static func history(_ x: MLXArray, keep: Int, axis: Int, owner: AnyObject) -> (joined: MLXArray, previous: Int) {
        guard let carry = current else {
            return (x, 0)
        }
        let key = ObjectIdentifier(owner)
        let previous = carry.arrays[key]
        let joined = previous.map { concatenated([$0, x], axis: axis) } ?? x
        let length = joined.dim(axis)
        if length > keep {
            carry.arrays[key] = take(joined, MLXArray(((length - keep) ..< length).map { Int32($0) }), axis: axis)
        } else {
            carry.arrays[key] = joined
        }
        return (joined, previous?.dim(axis) ?? 0)
    }

    /// `cell` over the sequence `x` `[B, L, in]`: from zero in a clip forward, from the hidden state the
    /// unit before ended with in a stream.
    static func recurrent(_ cell: NFKMLXGRUCell, _ x: MLXArray) -> MLXArray {
        guard let carry = current else {
            return cell(x)
        }
        let key = ObjectIdentifier(cell)
        let output = cell(x, initialHidden: carry.arrays[key])
        carry.arrays[key] = output[.ellipsis, output.dim(-2) - 1, 0...]
        return output
    }
}
