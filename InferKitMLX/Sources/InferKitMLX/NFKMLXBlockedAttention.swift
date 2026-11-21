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

    /// Whether the float32 attentions ``NFKReferenceRounding/flashAttention(queries:keys:values:scale:mask:keyBlock:queryBlock:)``
    /// runs, and the ones a model routes here itself, take this function when they are longer than two
    /// blocks. A training recipe sets it for its run with ``training(_:)``; it is scoped to the task, so an
    /// inference on another task keeps MLX's attention.
    @TaskLocal static var isTraining = false

    /// Runs `body` with ``isTraining`` set.
    static func training<T>(_ body: () throws -> T) rethrows -> T {
        try $isTraining.withValue(true, operation: body)
    }

    /// Whether a call over `queries` takes this function: training, float32, and longer than two blocks.
    /// Over two blocks the split costs more than it saves: Stable Audio's 1,024 tokens at 24 heads peak at
    /// 0.45 GB blocked against MLX's 0.32 GB.
    static func applies(to queries: MLXArray, mask: MLXArray?) -> Bool {
        isTraining && queries.dtype == .float32 && queries.dim(-2) > 2 * queryBlock && mask?.dtype != .bool
    }

    /// `queries` `[B, H, N, D]`, `keys` and `values` `[B, H, M, D]` → `[B, H, N, D]`, softmax attention at
    /// `scale` in blocks of `queryBlock` queries. `mask`, additive and broadcast against `[B, H, N, M]`,
    /// enters as a constant: it takes no gradient.
    static func attention(queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float,
                          mask: MLXArray? = nil, queryBlock: Int = queryBlock) -> MLXArray {
        let rows = queries.dim(-2)
        let blocks = stride(from: 0, to: rows, by: queryBlock).map { $0 ..< min($0 + queryBlock, rows) }
        func blockMask(_ block: Range<Int>) -> MLXArray? {
            guard let mask else { return nil }
            return mask.ndim >= 2 && mask.dim(-2) == rows ? mask[.ellipsis, block, 0...] : mask
        }
        let function = NFKMLXCustomFunction.make {
            Forward { arrays in
                let (q, k, v) = (arrays[0], arrays[1], arrays[2])
                var attended = [MLXArray]()
                for block in blocks {
                    // Under a gradient MLX attends unfused; each block waits on the one before it, so only one
                    // block's scores exist at a time.
                    let queries = attended.last.map { depends(inputs: [q[.ellipsis, block, 0...]], dependencies: [$0])[0] }
                        ?? q[.ellipsis, block, 0...]
                    attended.append(MLXFast.scaledDotProductAttention(queries: queries, keys: k, values: v,
                                                                      scale: scale, mask: blockMask(block)))
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
                    // Each block waits on the one before it, its key and value sums and its query gradient, so
                    // MLX forms one block's probabilities and score gradients at a time and frees them with it.
                    let ordered = depends(inputs: [q[.ellipsis, block, 0...], upstream[.ellipsis, block, 0...]],
                                          dependencies: [keyGradient, valueGradient] + queryGradients.suffix(1))
                    let blockQueries = ordered[0].asType(.float32)
                    let blockUpstream = ordered[1].asType(.float32)
                    var scores = matmul(blockQueries, wideKeys.swappedAxes(-1, -2)) * scale
                    if let added = blockMask(block) {
                        scores = scores + added.asType(.float32)
                    }
                    let probabilities = softmax(scores, axis: -1, precise: true)
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
