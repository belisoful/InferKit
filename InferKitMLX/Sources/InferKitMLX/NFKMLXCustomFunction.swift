//
//  NFKMLXCustomFunction.swift
//  InferKitMLX
//
//  Custom functions whose gradient nodes run on the caller's stream, and modules recomputed in the backward
//  pass through them.
//
//  Introduced in InferKit 0.4.0.
//

import Cmlx
import MLX
import MLXNN

/// mlx-swift's `CustomFunction`, applied on the stream the caller's operations run on.
///
/// @discussion A custom function with a gradient places the nodes that carry the gradient on MLX's per-thread
/// C++ default stream, while mlx-swift runs every operation on a stream of its own. Each call then crosses
/// streams twice and waits on a fence each way: about half a millisecond on an M1 Max, whatever the input's
/// size, which a decoder's hundred-odd kernel calls a token turn into tens of milliseconds. A function made here
/// sets the C++ default stream to the caller's stream for the call and restores the previous one after it.
enum NFKMLXCustomFunction {

    static func make(@MLXCustomFunctionBuilder.Builder _ build: () -> ([MLXArray]) -> [MLXArray])
        -> ([MLXArray]) -> [MLXArray] {
        let function = build()
        return { arrays in
            let current = StreamOrDevice.default.ctx
            var device = mlx_device_new()
            var previous = mlx_stream_new()
            defer {
                mlx_stream_free(previous)
                mlx_device_free(device)
            }
            guard mlx_stream_get_device(&device, current) == 0, mlx_get_default_stream(&previous, device) == 0 else {
                return function(arrays)
            }
            mlx_set_default_stream(current)
            defer { mlx_set_default_stream(previous) }
            return function(arrays)
        }
    }

    /// `body`, which runs `module`, with its activations recomputed from its inputs in the backward pass (MLX's
    /// `checkpoint`).
    ///
    /// @discussion The function's primals are `module`'s trainable parameters followed by the inputs, so every
    /// trainable parameter and every input receives its gradient; anything else `body` reads enters as a constant.
    /// The backward places the primals in `module`, reruns `body`, and puts back the arrays `module` held, which
    /// leaves the forward's values and the gradients those of `body` run plainly. Two orderings keep one block's
    /// backward in memory at a time; MLX's scheduler is otherwise free to break either:
    /// - The rerun reads its primals through `depends` on the output cotangents, so it waits for the backward of
    ///   the block after it. Without the wait the reruns run ahead and their attention is held together.
    /// - The input cotangents it returns depend on its parameter gradients, so the block before it waits for
    ///   them. Without that order MLX leaves the parameter gradients, side branches the chain never reads, until
    ///   the chain ends, and every block's backward intermediates stay held for them.
    static func recomputed(_ module: Module, _ body: @escaping ([MLXArray]) -> [MLXArray]) -> ([MLXArray]) -> [MLXArray] {
        let trainable = module.trainableParameters().flattened()
        let keys = trainable.map(\.0)
        let forward: ([MLXArray]) -> [MLXArray] = { arrays in
            // `update(parameters:)` writes into the arrays the module holds, so the held ones are kept as
            // separate handles.
            let held = module.trainableParameters().flattened().map { ($0.0, sharing($0.1)) }
            module.update(parameters: ModuleParameters.unflattened(Array(zip(keys, arrays.prefix(keys.count)))))
            defer { module.update(parameters: ModuleParameters.unflattened(held)) }
            return body(Array(arrays.dropFirst(keys.count)))
        }
        let function = make {
            Forward(forward)
            VJP { primals, cotangents in
                let gradients = vjp(forward, primals: MLX.depends(inputs: primals, dependencies: cotangents),
                                    cotangents: cotangents).1
                guard !keys.isEmpty else {
                    return gradients
                }
                let parameters = Array(gradients.prefix(keys.count))
                return parameters + MLX.depends(inputs: Array(gradients.dropFirst(keys.count)), dependencies: parameters)
            }
        }
        return { inputs in function(trainable.map(\.1) + inputs) }
    }

    /// A second handle on the array `array` refers to.
    private static func sharing(_ array: MLXArray) -> MLXArray {
        var handle = mlx_array_new()
        mlx_array_set(&handle, array.ctx)
        return MLXArray(handle)
    }
}
