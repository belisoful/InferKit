//
//  NFKMLXDeepSeekVision.swift
//  InferKitMLX
//
//  DeepSeek V4.1's image tower and the projection that joins it to the text stream.
//
//  The tower is a plain bidirectional ViT with two-dimensional rotary positions, and it is the one
//  part of the release that ships unquantized: every tensor here is bfloat16 with no block scale
//  beside it, where the decoder's attention is fp8 and its experts are packed four bits to a nibble.
//  A patch enters as raw pixels flattened into one vector, so the patch embedding is a Linear rather
//  than the strided convolution most towers use.
//
//  The aligner turns a grid of patch features into the text decoder's width, pooling each
//  `downsampleRatio` x `downsampleRatio` block of patches into one token.
//

import Foundation
import MLX
import MLXFast
import MLXNN

/// The geometry of DeepSeek V4.1's image tower.
public struct NFKMLXDeepSeekVisionConfiguration: Sendable {
    public var layerCount: Int
    public var hiddenSize: Int
    public var headCount: Int
    public var intermediateSize: Int
    public var patchSize: Int
    public var ropeTheta: Float
    /// How many patches on a side pool into one text token.
    public var downsampleRatio: Int
    /// The text decoder's width, which the aligner projects into.
    public var outputSize: Int
    public var rmsEpsilon: Float
    /// The most positions one picture may occupy in the prompt, which bounds the resize.
    public var maximumTokenCount: Int
    /// The fewest pixels a picture is scaled up to before its grid is planned.
    public var minimumPixels: Int
    /// A picture wider than this many times its height is narrowed to the ratio and then resized
    /// rather than padded. Absent where a release sets none, which the released V4.1 Flash does.
    public var maximumWidthHeightRatio: Int?
    /// The id every position of an image span carries. Only the span's layout tells them apart.
    public var imageTokenID: Int

    public init(layerCount: Int = 32, hiddenSize: Int = 1024, headCount: Int = 16,
                intermediateSize: Int = 2816, patchSize: Int = 14, ropeTheta: Float = 10_000,
                downsampleRatio: Int = 3, outputSize: Int = 5120, rmsEpsilon: Float = 1e-6,
                maximumTokenCount: Int = 1024, minimumPixels: Int = 544 * 544,
                maximumWidthHeightRatio: Int? = nil, imageTokenID: Int = 129_264) {
        self.maximumTokenCount = maximumTokenCount
        self.minimumPixels = minimumPixels
        self.maximumWidthHeightRatio = maximumWidthHeightRatio
        self.imageTokenID = imageTokenID
        self.layerCount = layerCount
        self.hiddenSize = hiddenSize
        self.headCount = headCount
        self.intermediateSize = intermediateSize
        self.patchSize = patchSize
        self.ropeTheta = ropeTheta
        self.downsampleRatio = downsampleRatio
        self.outputSize = outputSize
        self.rmsEpsilon = rmsEpsilon
    }

    var headDimensions: Int { hiddenSize / headCount }
    /// Half of a head's channels turn: one half indexes rows, the other columns.
    var rotaryDimensions: Int { hiddenSize / headCount / 2 }

    /// The released `deepseek-ai/DeepSeek-V4.1-Flash` tower.
    public static let v41Flash = NFKMLXDeepSeekVisionConfiguration()
}

/// The two-dimensional rotary table for an `n_h` by `n_w` grid of patches.
///
/// @discussion A patch's row and column each drive half of the turned channels, interleaved as
/// `[row, column]` pairs before flattening, so a channel alternates between the two axes rather than
/// taking the first half for one and the second half for the other.
func nfkDeepSeekVisionRotary(rows: Int, columns: Int, dimensions: Int, theta: Float)
    -> (cos: MLXArray, sin: MLXArray) {
    let half = dimensions / 2
    let inverse = MLXArray((0 ..< half).map { 1 / pow(theta, Float($0 * 2) / Float(dimensions)) })
    var positions = [Float]()
    positions.reserveCapacity(rows * columns * 2)
    for row in 0 ..< rows {
        for column in 0 ..< columns {
            positions.append(Float(row))
            positions.append(Float(column))
        }
    }
    let grid = MLXArray(positions).reshaped([rows * columns, 2, 1])
    let angles = (grid * inverse).reshaped([rows * columns, 1, dimensions])
    return (cos(angles), sin(angles))
}

/// Rotates the leading `dimensions` channels as two halves, the convention this tower uses.
func nfkDeepSeekVisionRotated(_ x: MLXArray, cos table: MLXArray, sin sine: MLXArray) -> MLXArray {
    let half = x.dim(-1) / 2
    let first = x[.ellipsis, 0 ..< half]
    let second = x[.ellipsis, half...]
    return concatenated([first * table - second * sine, second * table + first * sine], axis: -1)
}

final class NFKDeepSeekVisionAttention: Module {
    @ModuleInfo(key: "wqkv") var fused: Linear
    @ModuleInfo(key: "wo") var output: Linear

    let heads: Int
    let headDimensions: Int

    init(_ c: NFKMLXDeepSeekVisionConfiguration) {
        heads = c.headCount
        headDimensions = c.headDimensions
        _fused.wrappedValue = Linear(c.hiddenSize, 3 * c.hiddenSize, bias: true)
        _output.wrappedValue = Linear(c.hiddenSize, c.hiddenSize, bias: true)
        super.init()
    }

    /// `x` is `[patches, hidden]`: one image, attended over in full.
    func callAsFunction(_ x: MLXArray, cos table: MLXArray, sin sine: MLXArray) -> MLXArray {
        let count = x.dim(0)
        let projected = fused(x).reshaped([count, 3, heads, headDimensions])
        var queries = projected[0..., 0]
        var keys = projected[0..., 1]
        let values = projected[0..., 2]
        // Rotated in float32 against float32 tables and rounded once, as the release rotates.
        queries = nfkDeepSeekVisionRotated(queries.asType(.float32), cos: table, sin: sine)
            .asType(queries.dtype)
        keys = nfkDeepSeekVisionRotated(keys.asType(.float32), cos: table, sin: sine)
            .asType(keys.dtype)

        let scale = 1 / sqrt(Float(headDimensions))
        func headMajor(_ t: MLXArray) -> MLXArray { t.transposed(1, 0, 2).expandedDimensions(axis: 0) }
        let attended: MLXArray
        if x.dtype == .float32 {
            attended = MLXFast.scaledDotProductAttention(
                queries: headMajor(queries), keys: headMajor(keys), values: headMajor(values),
                scale: scale, mask: nil)
        } else {
            // The release calls `F.scaled_dot_product_attention`, whose narrow arithmetic is the
            // backend's. This computes torch's own definition, its MATH backend: widened to
            // float32, the scale split as its square root on queries and keys, rounded once.
            let root = scale.squareRoot()
            attended = MLXFast.scaledDotProductAttention(
                queries: headMajor(queries.asType(.float32) * root),
                keys: headMajor(keys.asType(.float32) * root),
                values: headMajor(values.asType(.float32)), scale: 1, mask: nil).asType(x.dtype)
        }
        return output(attended[0].transposed(1, 0, 2).reshaped([count, heads * headDimensions]))
    }
}

/// The tower's feed-forward: one projection emitting the gate and the lift together.
final class NFKDeepSeekVisionMLP: Module {
    @ModuleInfo(key: "w1") var gateUp: Linear
    @ModuleInfo(key: "w2") var down: Linear

    let intermediateSize: Int

    init(_ c: NFKMLXDeepSeekVisionConfiguration) {
        intermediateSize = c.intermediateSize
        _gateUp.wrappedValue = Linear(c.hiddenSize, 2 * c.intermediateSize, bias: false)
        _down.wrappedValue = Linear(c.intermediateSize, c.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let fused = gateUp(x)
        // MLX composes silu from narrow operations, each rounding; torch rounds it once.
        let gate = fused[.ellipsis, 0 ..< intermediateSize]
        return down(silu(gate.asType(.float32)).asType(gate.dtype)
                    * fused[.ellipsis, intermediateSize...])
    }
}

final class NFKDeepSeekVisionBlock: Module {
    @ModuleInfo(key: "norm1") var attentionNorm: NFKDeepSeekRMSNorm
    @ModuleInfo(key: "attn") var attention: NFKDeepSeekVisionAttention
    @ModuleInfo(key: "norm2") var feedForwardNorm: NFKDeepSeekRMSNorm
    @ModuleInfo(key: "mlp") var feedForward: NFKDeepSeekVisionMLP

    init(_ c: NFKMLXDeepSeekVisionConfiguration) {
        _attentionNorm.wrappedValue = NFKDeepSeekRMSNorm(dimensions: c.hiddenSize, eps: c.rmsEpsilon)
        _attention.wrappedValue = NFKDeepSeekVisionAttention(c)
        _feedForwardNorm.wrappedValue = NFKDeepSeekRMSNorm(dimensions: c.hiddenSize, eps: c.rmsEpsilon)
        _feedForward.wrappedValue = NFKDeepSeekVisionMLP(c)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cos table: MLXArray, sin sine: MLXArray) -> MLXArray {
        let attended = x + attention(attentionNorm(x), cos: table, sin: sine)
        return attended + feedForward(feedForwardNorm(attended))
    }
}

/// DeepSeek V4.1's image tower.
public final class NFKMLXDeepSeekVisionNet: Module {
    @ModuleInfo(key: "patch_embed") var patchEmbed: NFKDeepSeekVisionPatchEmbed
    @ModuleInfo(key: "blocks") var blocks: [NFKDeepSeekVisionBlock]
    @ModuleInfo(key: "norm") var norm: NFKDeepSeekRMSNorm

    let configuration: NFKMLXDeepSeekVisionConfiguration

    init(_ c: NFKMLXDeepSeekVisionConfiguration) {
        configuration = c
        _patchEmbed.wrappedValue = NFKDeepSeekVisionPatchEmbed(c)
        _blocks.wrappedValue = (0 ..< c.layerCount).map { _ in NFKDeepSeekVisionBlock(c) }
        _norm.wrappedValue = NFKDeepSeekRMSNorm(dimensions: c.hiddenSize, eps: c.rmsEpsilon)
        super.init()
    }

    /// `patches` is `[rows · columns, 3, patch, patch]` in row-major grid order.
    public func callAsFunction(_ patches: MLXArray, rows: Int, columns: Int) -> MLXArray {
        // The tower computes in the dtype its parameters hold, which the loader sets.
        var x = patchEmbed(patches.asType(patchEmbed.projection.weight.dtype))
        let (table, sine) = nfkDeepSeekVisionRotary(rows: rows, columns: columns,
                                                    dimensions: configuration.rotaryDimensions,
                                                    theta: configuration.ropeTheta)
        for block in blocks {
            x = block(x, cos: table, sin: sine)
        }
        return norm(x)
    }
}

/// A patch enters as raw pixels, so the embedding is a Linear over the flattened patch.
final class NFKDeepSeekVisionPatchEmbed: Module {
    @ModuleInfo(key: "proj") var projection: Linear

    init(_ c: NFKMLXDeepSeekVisionConfiguration) {
        _projection.wrappedValue = Linear(3 * c.patchSize * c.patchSize, c.hiddenSize, bias: true)
        super.init()
    }

    func callAsFunction(_ patches: MLXArray) -> MLXArray {
        projection(patches.reshaped([patches.dim(0), -1]))
    }
}

/// The projection from patch features to the text decoder's width.
///
/// @discussion Each `downsampleRatio` by `downsampleRatio` block of patches becomes one token, so a
/// grid is padded up to whole blocks and then folded. The fold takes a block's features in
/// channel-major order, which is what the reference's `unfold` produces and what the weight expects.
public final class NFKMLXDeepSeekAligner: Module {
    @ModuleInfo(key: "w1") var input: Linear
    @ModuleInfo(key: "w2") var output: Linear

    let ratio: Int
    let hiddenSize: Int

    init(_ c: NFKMLXDeepSeekVisionConfiguration) {
        ratio = c.downsampleRatio
        hiddenSize = c.hiddenSize
        _input.wrappedValue = Linear(c.hiddenSize * c.downsampleRatio * c.downsampleRatio,
                                     c.outputSize, bias: true)
        _output.wrappedValue = Linear(c.outputSize, c.outputSize, bias: true)
        super.init()
    }

    /// `x` is `[rows · columns, hidden]` → `[tokens, outputSize]`.
    public func callAsFunction(_ x: MLXArray, rows: Int, columns: Int) -> MLXArray {
        let paddedRows = (rows + ratio - 1) / ratio * ratio
        let paddedColumns = (columns + ratio - 1) / ratio * ratio
        var grid = x.reshaped([rows, columns, hiddenSize])
        if paddedRows != rows || paddedColumns != columns {
            grid = padded(grid, widths: [IntOrPair((0, paddedRows - rows)),
                                         IntOrPair((0, paddedColumns - columns)),
                                         IntOrPair((0, 0))])
        }
        // [blockRow, ratio, blockColumn, ratio, hidden] → one token per block, features ordered
        // channel first, then the block's rows and columns.
        let folded = grid
            .reshaped([paddedRows / ratio, ratio, paddedColumns / ratio, ratio, hiddenSize])
            .transposed(0, 2, 4, 1, 3)
            .reshaped([(paddedRows / ratio) * (paddedColumns / ratio), hiddenSize * ratio * ratio])
        // MLX composes gelu from narrow operations, each rounding; torch rounds it once.
        let widened = input(folded)
        return output(gelu(widened.asType(.float32)).asType(widened.dtype))
    }
}

/// An image span's learned delimiters, and the merge that writes a picture into the text stream.
///
/// @discussion The release stores three vectors at the top level of the checkpoint rather than
/// inside the tower, because they live in the DECODER's width: a span opens with `image_start`,
/// closes with `image_end`, and carries `image_newline` at the end of each row of pooled tokens.
/// The aligner's rows fill the slots between them in row-major order.
public final class NFKMLXDeepSeekImageDelimiters: Module {
    @ParameterInfo(key: "image_start") var start: MLXArray
    @ParameterInfo(key: "image_end") var end: MLXArray
    @ParameterInfo(key: "image_newline") var newline: MLXArray

    /// What a position in an image span holds.
    public enum Slot: Sendable {
        case start
        case end
        case newline
        /// A pooled image token, at this row of the aligner's output.
        case token(Int)
    }

    init(outputSize: Int) {
        _start.wrappedValue = MLXArray.zeros([outputSize])
        _end.wrappedValue = MLXArray.zeros([outputSize])
        _newline.wrappedValue = MLXArray.zeros([outputSize])
        super.init()
    }

    /// Overwrites `positions` of `embedded` `[batch, length, hidden]` with a span's contents.
    public func merged(into embedded: MLXArray, aligned: MLXArray,
                       slots: [(position: Int, slot: Slot)], batch: Int = 0) -> MLXArray {
        var rows = [MLXArray]()
        var indices = [Int32]()
        for (position, slot) in slots {
            indices.append(Int32(position))
            switch slot {
            case .start: rows.append(start)
            case .end: rows.append(end)
            case .newline: rows.append(newline)
            case .token(let row): rows.append(aligned[row])
            }
        }
        guard !rows.isEmpty else { return embedded }
        // Scattered into a COPY. `MLXArray` is a class, and its subscript setter writes through
        // `_updateInternal` rather than rebinding, so assigning into the parameter would modify the
        // caller's embedding and hand the same object back. Slicing yields a separate array to
        // scatter into; the compiler cannot warn about this, because from Swift's side nothing is
        // mutated at all.
        let updated = embedded[0...]
        updated[batch, MLXArray(indices)] = stacked(rows, axis: 0).asType(embedded.dtype)
        return updated
    }
}
