//
//  NFKMLXGatedDeltaKernel.swift
//  InferKitMLX
//
//  The gated delta-rule recurrence of the hybrid decoder's linear-attention layers as one Metal kernel. The
//  recurrence is sequential in time, and written from MLX operations each position is a handful of small
//  kernels; here each thread owns one value channel's column of a head's state and runs every position.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import MLX
import MLXFast

/// The gated delta rule over float32 queries and keys (unit-normed and repeated to the value heads), values,
/// per-position decay `exp(g)`, and write strength, as ``NFKHybridLinearAttention/recurrence(queries:keys:values:decay:write:initialState:)``
/// computes it.
///
/// @discussion Each position decays the state, reads what it holds for the key, writes the correction toward the
/// value at the write strength, and reads the state for the scaled query. A value channel's column of the state
/// depends on that channel alone, so one thread holds it, `keyWidth` floats, and no thread waits on another. The
/// kernel runs on the GPU stream only.
enum NFKMLXGatedDeltaKernel {

    /// The widest key a thread's column holds.
    static let maximumKeyWidth = 256

    /// Whether the kernel runs here: on the GPU stream, for keys no wider than ``maximumKeyWidth``.
    static func runs(keyWidth: Int) -> Bool {
        NFKMLXDevice.currentType == .gpu && keyWidth <= maximumKeyWidth
    }

    /// The read `[batch, length, heads, valueWidth]` and the state after the last position
    /// `[batch, heads, keyWidth, valueWidth]`, from `initialState` (zeros where nil). Not differentiable.
    static func scan(queries: MLXArray, keys: MLXArray, values: MLXArray, decay: MLXArray, write: MLXArray,
                     initialState: MLXArray?) -> (read: MLXArray, state: MLXArray) {
        let (batch, length, heads, keyWidth) = (queries.dim(0), queries.dim(1), queries.dim(2), queries.dim(3))
        let valueWidth = values.dim(3)
        let state = initialState?.asType(.float32) ?? MLXArray.zeros([batch, heads, keyWidth, valueWidth])
        let dims = MLXArray([Int32(batch), Int32(length), Int32(heads), Int32(keyWidth), Int32(valueWidth)])
        let outputs = kernel(
            [queries.asType(.float32), keys.asType(.float32), values.asType(.float32), decay.asType(.float32),
             write.asType(.float32), state, dims],
            grid: (valueWidth, heads, batch), threadGroup: (min(valueWidth, 256), 1, 1),
            outputShapes: [[batch, length, heads, valueWidth], [batch, heads, keyWidth, valueWidth]],
            outputDTypes: [.float32, .float32])
        return (outputs[0], outputs[1])
    }

    /// ``scan(queries:keys:values:decay:write:initialState:)``'s read from a zero state, differentiable: its
    /// gradient is the sequential recurrence's, written from MLX operations.
    static func read(queries: MLXArray, keys: MLXArray, values: MLXArray, decay: MLXArray, write: MLXArray) -> MLXArray {
        let function = NFKMLXCustomFunction.make {
            Forward { inputs in
                [scan(queries: inputs[0], keys: inputs[1], values: inputs[2], decay: inputs[3], write: inputs[4],
                      initialState: nil).read]
            }
            VJP { primals, cotangents in vjp(sequentialRead, primals: primals, cotangents: cotangents).1 }
        }
        return function([queries, keys, values, decay, write])[0]
    }

    /// The recurrence from a zero state in MLX operations, `[queries, keys, values, decay, write]` → `[read]`.
    private static func sequentialRead(_ inputs: [MLXArray]) -> [MLXArray] {
        let (queries, keys, values, decay, write) = (inputs[0], inputs[1], inputs[2], inputs[3], inputs[4])
        let (batch, length, heads) = (queries.dim(0), queries.dim(1), queries.dim(2))
        let scale = 1 / sqrt(Float(queries.dim(3)))
        var state = MLXArray.zeros([batch, heads, queries.dim(3), values.dim(3)])
        var outputs = [MLXArray]()
        for step in 0 ..< length {
            let k = keys[0..., step].expandedDimensions(axis: -1)
            state = state * decay[0..., step].reshaped([batch, heads, 1, 1])
            let correction = (values[0..., step] - (state * k).sum(axis: 2)) * write[0..., step].reshaped([batch, heads, 1])
            state = state + k * correction.expandedDimensions(axis: 2)
            outputs.append((state * (queries[0..., step] * scale).expandedDimensions(axis: -1)).sum(axis: 2))
        }
        return [stacked(outputs, axis: 1)]
    }

    // One thread per (value channel, head, batch row), its state column in registers; the query is scaled by
    // 1/√keyWidth at each read, as the recurrence scales it before reading.
    private static let kernel = MLXFast.metalKernel(
        name: "nfk_gated_delta_scan",
        inputNames: ["q", "k", "v", "g", "beta", "s0", "dims"],
        outputNames: ["y", "s"],
        source: """
            const int column = int(thread_position_in_grid.x);
            const int head = int(thread_position_in_grid.y);
            const int row = int(thread_position_in_grid.z);
            const int batches = dims[0], length = dims[1], heads = dims[2], kw = dims[3], vw = dims[4];
            if (column >= vw || head >= heads || row >= batches) { return; }
            float state[256];
            const int stateBase = (row * heads + head) * kw * vw;
            for (int i = 0; i < kw; ++i) { state[i] = s0[stateBase + i * vw + column]; }
            const float scale = rsqrt(float(kw));
            for (int t = 0; t < length; ++t) {
                const int position = (row * length + t) * heads + head;
                const device float* kt = k + position * kw;
                const device float* qt = q + position * kw;
                const float decay = g[position];
                float held = 0.0f;
                for (int i = 0; i < kw; ++i) {
                    state[i] = state[i] * decay;
                    held += state[i] * kt[i];
                }
                const float correction = (v[position * vw + column] - held) * beta[position];
                float out = 0.0f;
                for (int i = 0; i < kw; ++i) {
                    state[i] = state[i] + kt[i] * correction;
                    out += state[i] * (qt[i] * scale);
                }
                y[position * vw + column] = out;
            }
            for (int i = 0; i < kw; ++i) { s[stateBase + i * vw + column] = state[i]; }
        """)
}
