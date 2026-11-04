//
//  NFKMLXCustomFunction.swift
//  InferKitMLX
//
//  Custom functions whose gradient nodes run on the caller's stream.
//
//  Introduced in InferKit 0.4.0.
//

import Cmlx
import MLX

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
}
