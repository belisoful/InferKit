import MLX
import MLXFast

/// Attention whose forward and backward each hold one block of queries' scores at a time.
///
/// Under a gradient, MLX's `scaledDotProductAttention` runs unfused on Metal in both directions, so a
/// training step holds whole `[B, H, N, N]` score matrices: 2.9 GB each at float32 for Wan 2.1 1.3B's
/// 7,800 tokens of a 17-frame 480 × 832 clip. This function is a custom function with the same value:
/// its forward attends one query block at a time, and its backward recomputes each block's softmax over
/// the full key row and forms that block's query, key, and value gradients from it, so no step holds more
/// than one `[B, H, queryBlock, N]` block.
enum NFKMLXBlockedAttention {

    /// The queries a block holds: a `[1, 12, 512, 7800]` float32 block of Wan 2.1 1.3B's scores is 192 MB.
    static let queryBlock = 512

    /// `queries` `[B, H, N, D]`, `keys` and `values` `[B, H, M, D]` → `[B, H, N, D]`, softmax attention at
    /// `scale` with no mask, in blocks of `queryBlock` queries.
    static func attention(queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float,
                          queryBlock: Int = queryBlock) -> MLXArray {
        let blocks = stride(from: 0, to: queries.dim(-2), by: queryBlock).map { $0 ..< min($0 + queryBlock, queries.dim(-2)) }
        let function = NFKMLXCustomFunction.make {
            Forward { arrays in
                let (q, k, v) = (arrays[0], arrays[1], arrays[2])
                let attended = blocks.map { block in
                    MLXFast.scaledDotProductAttention(queries: q[.ellipsis, block, 0...], keys: k, values: v,
                                                      scale: scale, mask: nil)
                }
                return [blocks.count == 1 ? attended[0] : concatenated(attended, axis: -2)]
            }
            VJP { primals, cotangents in
                let (q, k, v, upstream) = (primals[0], primals[1], primals[2], cotangents[0])
                let wideKeys = k.asType(.float32), wideValues = v.asType(.float32)
                var queryGradients = [MLXArray]()
                var keyGradient = MLXArray.zeros(k.shape, dtype: .float32)
                var valueGradient = MLXArray.zeros(v.shape, dtype: .float32)
                for block in blocks {
                    // Each block reads the sums the one before it wrote, so MLX forms one block's
                    // probabilities at a time rather than every block's before the first sum.
                    let ordered = depends(inputs: [q[.ellipsis, block, 0...], upstream[.ellipsis, block, 0...]],
                                          dependencies: [keyGradient, valueGradient])
                    let blockQueries = ordered[0].asType(.float32)
                    let blockUpstream = ordered[1].asType(.float32)
                    let probabilities = softmax(matmul(blockQueries, wideKeys.swappedAxes(-1, -2)) * scale,
                                                axis: -1, precise: true)
                    let probabilityGradient = matmul(blockUpstream, wideValues.swappedAxes(-1, -2))
                    let rowTerm = (probabilityGradient * probabilities).sum(axis: -1, keepDims: true)
                    let scoreGradient = probabilities * (probabilityGradient - rowTerm) * scale
                    queryGradients.append(matmul(scoreGradient, wideKeys).asType(q.dtype))
                    keyGradient = keyGradient + matmul(scoreGradient.swappedAxes(-1, -2), blockQueries)
                    valueGradient = valueGradient + matmul(probabilities.swappedAxes(-1, -2), blockUpstream)
                }
                return [blocks.count == 1 ? queryGradients[0] : concatenated(queryGradients, axis: -2),
                        keyGradient.asType(k.dtype), valueGradient.asType(v.dtype)]
            }
        }
        return function([queries, keys, values])[0]
    }
}
