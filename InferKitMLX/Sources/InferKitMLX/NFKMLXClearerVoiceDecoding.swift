//
//  NFKMLXClearerVoiceDecoding.swift
//  InferKitMLX
//
//  The window grid ClearerVoice-Studio's decoders share (`utils/decode.py`): the FRCRN SE 16K and
//  MossFormer2 SE 48K decoders pad a clip onto it, and decode a clip past their one-pass limit window by
//  window.
//

import Foundation

enum NFKMLXClearerVoiceDecoding {

    /// The decoders' zero padding onto the grid: up to the window, up to window + stride, or (past that)
    /// by `t − ⌊(t − window) / stride⌋ · stride` whenever the clip is off the stride grid.
    static func padding(count t: Int, window: Int, stride: Int) -> Int {
        if t < window {
            return window - t
        }
        if t < window + stride {
            return window + stride - t
        }
        return (t - window) % stride != 0 ? t - (t - window) / stride * stride : 0
    }

    /// The decoders' windowed path. Zero-pads the clip onto the grid, enhances each window at a `stride`
    /// hop, and keeps each window's output less `give_up_length = (window − stride) / 2` samples at every
    /// inner edge. The first window keeps its leading edge. A clip already on the grid keeps the
    /// decoders' zeros over its last `give_up_length` samples, which no window writes. The result is
    /// trimmed to the input length, as ClearerVoice's caller trims it.
    static func stitched(_ samples: [Float], window: Int, stride: Int,
                         segment enhance: ([Float]) -> [Float]) -> [Float] {
        let giveUp = (window - stride) / 2
        let padded = samples + [Float](repeating: 0, count: padding(count: samples.count, window: window, stride: stride))
        var output = [Float](repeating: 0, count: padded.count)
        var start = 0
        while start + window <= padded.count {
            let enhanced = enhance(Array(padded[start ..< start + window]))
            let kept = start == 0 ? 0 ..< window - giveUp : giveUp ..< window - giveUp
            output.replaceSubrange(start + kept.lowerBound ..< start + kept.upperBound, with: enhanced[kept])
            start += stride
        }
        return Array(output.prefix(samples.count))
    }
}
