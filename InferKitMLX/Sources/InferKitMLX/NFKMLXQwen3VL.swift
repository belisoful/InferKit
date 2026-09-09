//
//  NFKMLXQwen3VL.swift
//  InferKitMLX
//
//  Qwen3-VL's vision tower: the second vision-language model's image encoder, and a second vision
//  architecture beside SmolVLM's SigLIP. Where SigLIP is a plain ViT with a learned position embedding
//  and a pixel-shuffle connector, Qwen3-VL's encoder is a 2D-rotary ViT whose patches are laid out in
//  2×2 merge blocks, a bilinearly interpolated position embedding, a merger that folds each block to the
//  decoder width, and a "deepstack" of three feature maps taken from intermediate layers. The decoder is
//  the Qwen3 dense stack `NFKMLXLanguageNet` already runs.
//

import Foundation
import CoreGraphics
import InferKit
import MLX
import MLXFast
import MLXNN

/// Turns a `CGImage` into the flattened patches and grid the Qwen3-VL vision tower reads. The image is
/// smart-resized so both sides are a multiple of `patchSize · mergeSize` and the pixel budget stays in
/// range, rescaled and normalized to `-1 … 1`, the single frame doubled to the temporal patch, and
/// patchified in the reference's `(t, h, w)` order.
///
/// The resize is CoreGraphics rather than the reference's bicubic, so the patch pixels are a close
/// approximation, the documented difference the SmolVLM processor also carries; the grid and the patch
/// layout are the reference's exactly.
public struct NFKMLXQwen3VLImageProcessor {
    public let patchSize = 16
    public let temporalPatchSize = 2
    public let mergeSize = 2
    public let minPixels = 65_536
    public let maxPixels = 16_777_216

    public init() {}

    /// The reference `smart_resize`: both sides a multiple of `patchSize · mergeSize`, the pixel count
    /// held within `[minPixels, maxPixels]`, the aspect ratio kept as closely as possible.
    public func smartResize(height: Int, width: Int) -> (height: Int, width: Int) {
        let factor = patchSize * mergeSize
        var barHeight = Int((Double(height) / Double(factor)).rounded()) * factor
        var barWidth = Int((Double(width) / Double(factor)).rounded()) * factor
        if barHeight * barWidth > maxPixels {
            let beta = (Double(height * width) / Double(maxPixels)).squareRoot()
            barHeight = Swift.max(factor, Int((Double(height) / beta / Double(factor)).rounded(.down)) * factor)
            barWidth = Swift.max(factor, Int((Double(width) / beta / Double(factor)).rounded(.down)) * factor)
        } else if barHeight * barWidth < minPixels {
            let beta = (Double(minPixels) / Double(height * width)).squareRoot()
            barHeight = Int((Double(height) * beta / Double(factor)).rounded(.up)) * factor
            barWidth = Int((Double(width) * beta / Double(factor)).rounded(.up)) * factor
        }
        return (barHeight, barWidth)
    }

    /// The flattened patches `[grid_t·grid_h·grid_w, 3·temporalPatch·patch²]` and the `(t, h, w)` grid.
    public func process(_ image: CGImage) -> (pixelValues: MLXArray, grid: (t: Int, h: Int, w: Int)) {
        let (height, width) = smartResize(height: image.height, width: image.width)
        var bytes = [UInt8](repeating: 0, count: width * height * 4)
        let context = CGContext(data: &bytes, width: width, height: height, bitsPerComponent: 8,
                                bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
                                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        context?.interpolationQuality = .high
        context?.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

        // Channel-first pixels normalized to -1...1 (mean/std 0.5).
        var planar = [Float](repeating: 0, count: 3 * height * width)
        for y in 0 ..< height {
            for x in 0 ..< width {
                let base = (y * width + x) * 4
                for channel in 0 ..< 3 {
                    planar[channel * height * width + y * width + x] = Float(bytes[base + channel]) / 255 * 2 - 1
                }
            }
        }
        let gridT = 1, gridH = height / patchSize, gridW = width / patchSize
        let frame = MLXArray(planar).reshaped([1, 3, height, width])
        let temporal = concatenated([frame, frame], axis: 0)                // the temporal patch
        let reshaped = temporal.reshaped([gridT, temporalPatchSize, 3,
                                          gridH / mergeSize, mergeSize, patchSize,
                                          gridW / mergeSize, mergeSize, patchSize])
        let ordered = reshaped.transposed(0, 3, 6, 4, 7, 2, 1, 5, 8)
        let flattened = ordered.reshaped([gridT * gridH * gridW,
                                          3 * temporalPatchSize * patchSize * patchSize])
        return (flattened, (gridT, gridH, gridW))
    }
}

/// The geometry of the Qwen3-VL vision encoder.
public struct NFKMLXQwen3VLVisionConfiguration: Sendable {
    public var hiddenSize: Int
    public var depth: Int
    public var headCount: Int
    public var intermediateSize: Int
    public var patchSize: Int
    public var temporalPatchSize: Int
    public var spatialMergeSize: Int
    public var outHiddenSize: Int
    public var positionGridSide: Int
    public var deepstackLayers: [Int]
    public var layerNormEpsilon: Float

    public init(hiddenSize: Int = 1024, depth: Int = 24, headCount: Int = 16, intermediateSize: Int = 4096,
                patchSize: Int = 16, temporalPatchSize: Int = 2, spatialMergeSize: Int = 2,
                outHiddenSize: Int = 2048, positionGridSide: Int = 48,
                deepstackLayers: [Int] = [5, 11, 17], layerNormEpsilon: Float = 1e-6) {
        self.hiddenSize = hiddenSize
        self.depth = depth
        self.headCount = headCount
        self.intermediateSize = intermediateSize
        self.patchSize = patchSize
        self.temporalPatchSize = temporalPatchSize
        self.spatialMergeSize = spatialMergeSize
        self.outHiddenSize = outHiddenSize
        self.positionGridSide = positionGridSide
        self.deepstackLayers = deepstackLayers
        self.layerNormEpsilon = layerNormEpsilon
    }

    public static let qwen3VL2B = NFKMLXQwen3VLVisionConfiguration()

    /// Reads a release's `vision_config` from its `config.json`. The 2B and 4B share one tower; the 8B,
    /// 32B, and 30B-A3B run a deeper, wider one (27 blocks of 1152, hooked at 8/16/24), and every size
    /// projects to its own decoder width.
    public static func configuration(fromHuggingFace url: URL) throws -> NFKMLXQwen3VLVisionConfiguration {
        let data = try Data(contentsOf: url)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw NFKMLXError.unsupportedConfiguration("\(url.lastPathComponent) is not a JSON object")
        }
        return try configuration(fromJSON: json)
    }

    static func configuration(fromJSON json: [String: Any]) throws -> NFKMLXQwen3VLVisionConfiguration {
        guard let vision = json["vision_config"] as? [String: Any] else {
            throw NFKMLXError.unsupportedConfiguration("the config carries no vision_config")
        }
        let kind = (vision["model_type"] as? String) ?? ""
        guard kind == "qwen3_vl" || kind == "qwen3_vl_moe" else {
            throw NFKMLXError.unsupportedConfiguration("this reads a Qwen3-VL vision tower, not \(kind)")
        }
        func integer(_ key: String, _ fallback: Int) -> Int { (vision[key] as? NSNumber)?.intValue ?? fallback }
        let positions = integer("num_position_embeddings", 2304)
        let side = Int(Double(positions).squareRoot().rounded())
        guard side * side == positions else {
            throw NFKMLXError.unsupportedConfiguration("num_position_embeddings \(positions) is not a square grid")
        }
        let deepstack = (vision["deepstack_visual_indexes"] as? [NSNumber])?.map(\.intValue) ?? [5, 11, 17]
        return NFKMLXQwen3VLVisionConfiguration(
            hiddenSize: integer("hidden_size", 1024), depth: integer("depth", 24),
            headCount: integer("num_heads", 16), intermediateSize: integer("intermediate_size", 4096),
            patchSize: integer("patch_size", 16), temporalPatchSize: integer("temporal_patch_size", 2),
            spatialMergeSize: integer("spatial_merge_size", 2), outHiddenSize: integer("out_hidden_size", 2048),
            positionGridSide: side, deepstackLayers: deepstack)
    }

    var headDimensions: Int { hiddenSize / headCount }
    var patchInputSize: Int { 3 * temporalPatchSize * patchSize * patchSize }
    var mergedSize: Int { hiddenSize * spatialMergeSize * spatialMergeSize }
}

/// The patch embedding. The reference convolves each `[temporal, patch, patch]` patch with a full-size
/// kernel, which for a kernel equal to the patch is one linear projection over the flattened patch.
final class NFKQwen3VLPatchEmbed: Module {
    @ModuleInfo(key: "proj") var proj: Linear

    init(_ c: NFKMLXQwen3VLVisionConfiguration) {
        _proj.wrappedValue = Linear(c.patchInputSize, c.hiddenSize, bias: true)
        super.init()
    }

    func callAsFunction(_ pixelValues: MLXArray) -> MLXArray { proj(pixelValues) }
}

/// Qwen3-VL vision attention: one fused QKV projection with a bias, 2D rotary on the queries and keys,
/// and full attention over an image's patches.
final class NFKQwen3VLVisionAttention: Module {
    @ModuleInfo(key: "qkv") var qkv: Linear
    @ModuleInfo(key: "proj") var proj: Linear

    let heads: Int
    let headDimensions: Int
    let scale: Float

    init(_ c: NFKMLXQwen3VLVisionConfiguration) {
        heads = c.headCount
        headDimensions = c.headDimensions
        scale = 1 / sqrt(Float(c.headDimensions))
        _qkv.wrappedValue = Linear(c.hiddenSize, 3 * c.hiddenSize, bias: true)
        _proj.wrappedValue = Linear(c.hiddenSize, c.hiddenSize, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let length = x.dim(0)
        let projected = qkv(x).reshaped([length, 3, heads, headDimensions])
        var queries = projected[0..., 0].transposed(1, 0, 2)         // [heads, length, headDim]
        var keys = projected[0..., 1].transposed(1, 0, 2)
        let values = projected[0..., 2].transposed(1, 0, 2)

        queries = applyRotary(queries, cos: cos, sin: sin)
        keys = applyRotary(keys, cos: cos, sin: sin)

        let attention = MLXFast.scaledDotProductAttention(
            queries: queries.expandedDimensions(axis: 0), keys: keys.expandedDimensions(axis: 0),
            values: values.expandedDimensions(axis: 0), scale: scale, mask: nil)[0]
        return proj(attention.transposed(1, 0, 2).reshaped([length, heads * headDimensions]))
    }

    /// The 2D rotary embedding: `x·cos + rotateHalf(x)·sin`, `cos`/`sin` shared across heads.
    private func applyRotary(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let half = x.dim(2) / 2
        let rotated = concatenated([-x[0..., 0..., half...], x[0..., 0..., 0 ..< half]], axis: -1)
        return x * cos + rotated * sin
    }
}

/// Qwen3-VL vision feed-forward: a projection up, the tanh-approximate GELU, a projection back.
final class NFKQwen3VLVisionMLP: Module {
    @ModuleInfo(key: "linear_fc1") var fc1: Linear
    @ModuleInfo(key: "linear_fc2") var fc2: Linear

    init(_ c: NFKMLXQwen3VLVisionConfiguration) {
        _fc1.wrappedValue = Linear(c.hiddenSize, c.intermediateSize, bias: true)
        _fc2.wrappedValue = Linear(c.intermediateSize, c.hiddenSize, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(geluApproximate(fc1(x))) }
}

/// One Qwen3-VL vision block: pre-normalized attention and feed-forward, each added back.
final class NFKQwen3VLVisionBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "attn") var attention: NFKQwen3VLVisionAttention
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: NFKQwen3VLVisionMLP

    init(_ c: NFKMLXQwen3VLVisionConfiguration) {
        _norm1.wrappedValue = LayerNorm(dimensions: c.hiddenSize, eps: c.layerNormEpsilon)
        _attention.wrappedValue = NFKQwen3VLVisionAttention(c)
        _norm2.wrappedValue = LayerNorm(dimensions: c.hiddenSize, eps: c.layerNormEpsilon)
        _mlp.wrappedValue = NFKQwen3VLVisionMLP(c)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let attended = x + attention(norm1(x), cos: cos, sin: sin)
        return attended + mlp(norm2(attended))
    }
}

/// The patch merger: a normalization, then a 2×2 block folded to `spatialMergeSize²` times the channels
/// and projected to the decoder width. The main merger normalizes before folding; a deepstack merger
/// normalizes after (the `postShuffle` flag).
final class NFKQwen3VLMerger: Module {
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "linear_fc1") var fc1: Linear
    @ModuleInfo(key: "linear_fc2") var fc2: Linear

    let mergedSize: Int
    let postShuffle: Bool

    init(_ c: NFKMLXQwen3VLVisionConfiguration, postShuffle: Bool) {
        mergedSize = c.mergedSize
        self.postShuffle = postShuffle
        _norm.wrappedValue = LayerNorm(dimensions: postShuffle ? c.mergedSize : c.hiddenSize,
                                       eps: c.layerNormEpsilon)
        _fc1.wrappedValue = Linear(c.mergedSize, c.mergedSize, bias: true)
        _fc2.wrappedValue = Linear(c.mergedSize, c.outHiddenSize, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let folded = postShuffle ? norm(x.reshaped([-1, mergedSize])) : norm(x).reshaped([-1, mergedSize])
        return fc2(gelu(fc1(folded)))
    }
}

/// The Qwen3-VL vision encoder. It returns the merged image features and the three deepstack feature
/// maps the decoder injects at its first layers.
public final class NFKMLXQwen3VLVisionNet: Module {
    @ModuleInfo(key: "patch_embed") var patchEmbed: NFKQwen3VLPatchEmbed
    @ModuleInfo(key: "pos_embed") var positionEmbedding: Embedding
    @ModuleInfo(key: "blocks") var blocks: [NFKQwen3VLVisionBlock]
    @ModuleInfo(key: "merger") var merger: NFKQwen3VLMerger
    @ModuleInfo(key: "deepstack_merger_list") var deepstackMergers: [NFKQwen3VLMerger]

    let configuration: NFKMLXQwen3VLVisionConfiguration

    init(_ c: NFKMLXQwen3VLVisionConfiguration) {
        configuration = c
        _patchEmbed.wrappedValue = NFKQwen3VLPatchEmbed(c)
        _positionEmbedding.wrappedValue = Embedding(embeddingCount: c.positionGridSide * c.positionGridSide,
                                                    dimensions: c.hiddenSize)
        _blocks.wrappedValue = (0 ..< c.depth).map { _ in NFKQwen3VLVisionBlock(c) }
        _merger.wrappedValue = NFKQwen3VLMerger(c, postShuffle: false)
        _deepstackMergers.wrappedValue = c.deepstackLayers.map { _ in NFKQwen3VLMerger(c, postShuffle: true) }
        super.init()
    }

    /// `pixelValues` is `[patches, patchInputSize]` in the processor's 2×2-merge-block order, `grid` a
    /// `[temporal, height, width]` patch grid. Returns the merged features `[patches / 4, outHidden]` and
    /// the three deepstack feature maps.
    public func callAsFunction(_ pixelValues: MLXArray, grid: (t: Int, h: Int, w: Int))
        -> (output: MLXArray, deepstack: [MLXArray]) {
        var hidden = patchEmbed(pixelValues) + interpolatedPositionEmbedding(grid: grid)
        let (cos, sin) = rotaryEmbedding(grid: grid)

        var deepstack = [MLXArray]()
        for (index, block) in blocks.enumerated() {
            hidden = block(hidden, cos: cos, sin: sin)
            if let stack = configuration.deepstackLayers.firstIndex(of: index) {
                deepstack.append(deepstackMergers[stack](hidden))
            }
        }
        return (merger(hidden), deepstack)
    }

    /// The learned 48×48 position embedding bilinearly interpolated to the image's grid, then reordered
    /// into 2×2 merge-block order to line up with the patches.
    func interpolatedPositionEmbedding(grid: (t: Int, h: Int, w: Int)) -> MLXArray {
        let side = configuration.positionGridSide
        let (h, w) = (grid.h, grid.w)
        func samples(_ count: Int) -> (floor: [Int], ceil: [Int], fraction: [Float]) {
            var floors = [Int](), ceils = [Int](), fractions = [Float]()
            for index in 0 ..< count {
                let coordinate = count == 1 ? 0 : Float(index) * Float(side - 1) / Float(count - 1)
                let floor = Int(coordinate)
                floors.append(floor)
                ceils.append(Swift.min(floor + 1, side - 1))
                fractions.append(coordinate - Float(floor))
            }
            return (floors, ceils, fractions)
        }
        let rows = samples(h), columns = samples(w)

        var indices = [[Int32]](repeating: [], count: 4)
        var weights = [[Float]](repeating: [], count: 4)
        let merge = configuration.spatialMergeSize
        for blockRow in 0 ..< (h / merge) {
            for blockColumn in 0 ..< (w / merge) {
                for intraRow in 0 ..< merge {
                    for intraColumn in 0 ..< merge {
                        let i = blockRow * merge + intraRow, j = blockColumn * merge + intraColumn
                        let corners = [(rows.floor[i], columns.floor[j], (1 - rows.fraction[i]) * (1 - columns.fraction[j])),
                                       (rows.floor[i], columns.ceil[j], (1 - rows.fraction[i]) * columns.fraction[j]),
                                       (rows.ceil[i], columns.floor[j], rows.fraction[i] * (1 - columns.fraction[j])),
                                       (rows.ceil[i], columns.ceil[j], rows.fraction[i] * columns.fraction[j])]
                        for (corner, value) in corners.enumerated() {
                            indices[corner].append(Int32(value.0 * side + value.1))
                            weights[corner].append(value.2)
                        }
                    }
                }
            }
        }
        let count = h * w
        var result = MLXArray.zeros([count, configuration.hiddenSize])
        for corner in 0 ..< 4 {
            let gathered = positionEmbedding(MLXArray(indices[corner]))
            result = result + gathered * MLXArray(weights[corner]).reshaped([count, 1])
        }
        return grid.t == 1 ? result : concatenated(Array(repeating: result, count: grid.t), axis: 0)
    }

    /// The 2D rotary cosine and sine tables for the patch grid, in merge-block order.
    func rotaryEmbedding(grid: (t: Int, h: Int, w: Int)) -> (cos: MLXArray, sin: MLXArray) {
        let rotaryDimensions = configuration.headDimensions / 2                 // 32
        let frequencies = (0 ..< rotaryDimensions / 2).map {
            1 / pow(Float(10_000), Float(2 * $0) / Float(rotaryDimensions))
        }
        let merge = configuration.spatialMergeSize
        var table = [Float]()
        for _ in 0 ..< grid.t {
            for blockRow in 0 ..< (grid.h / merge) {
                for blockColumn in 0 ..< (grid.w / merge) {
                    for intraRow in 0 ..< merge {
                        for intraColumn in 0 ..< merge {
                            let row = Float(blockRow * merge + intraRow)
                            let column = Float(blockColumn * merge + intraColumn)
                            table.append(contentsOf: frequencies.map { row * $0 })
                            table.append(contentsOf: frequencies.map { column * $0 })
                        }
                    }
                }
            }
        }
        let count = grid.t * grid.h * grid.w
        let rotary = MLXArray(table).reshaped([count, rotaryDimensions])
        let doubled = concatenated([rotary, rotary], axis: -1)                  // [count, headDim]
        return (cos(doubled).expandedDimensions(axis: 0), sin(doubled).expandedDimensions(axis: 0))
    }
}

/// Building and loading Qwen3-VL.
///
/// `NFKMLXQwen3VL` is the released `Qwen/Qwen3-VL-2B-Instruct`. The vision tower is ``NFKMLXQwen3VLVisionNet``;
/// the decoder is the Qwen3 dense stack `NFKMLXLanguage` already runs, loaded from the checkpoint's
/// `model.language_model.` subtree.
@objc(NFKMLXQwen3VL)
public final class NFKMLXQwen3VL: NSObject {

    /// A name for the model the factories produce.
    @objc public static let modelName = "qwen3-vl-2b"

    private let visionNet: NFKMLXQwen3VLVisionNet
    private let textDecoder: NFKMLXLanguageNet
    private let tokenizer: NFKTokenizer
    private let processor = NFKMLXQwen3VLImageProcessor()
    private let endTokens: Set<Int>

    init(visionNet: NFKMLXQwen3VLVisionNet, decoder: NFKMLXLanguageNet, tokenizer: NFKTokenizer,
         endTokens: Set<Int>) {
        self.visionNet = visionNet
        self.textDecoder = decoder
        self.tokenizer = tokenizer
        self.endTokens = endTokens
        super.init()
    }

    /// Loads the whole model — the vision tower, the text decoder, and the release tokenizer — from a
    /// downloaded release directory, ready to answer a question about an image.
    @objc(modelWithDirectoryURL:error:)
    public static func model(directoryURL: URL) throws -> NFKMLXQwen3VL {
        let vision = try visionNet(directoryURL: directoryURL)
        let decoder = try decoder(directoryURL: directoryURL)
        guard let tokenizer = NFKMLXLanguage.releaseTokenizer(inDirectory: directoryURL) else {
            throw NFKMLXError.weightsMismatch("the release carries no tokenizer files")
        }
        // Stop at the end of the assistant turn or the end of text.
        let (specials, endToken) = NFKMLXLanguage.specialTokens(inDirectory: directoryURL)
        var stops = Set([specials["<|im_end|>"], endToken].compactMap { $0 })
        if stops.isEmpty { stops = [151_645] }
        return NFKMLXQwen3VL(visionNet: vision, decoder: decoder, tokenizer: tokenizer, endTokens: stops)
    }

    /// Answers `question` about `image`: the image processor tiles the image, the vision tower and its
    /// deepstack encode it, the tokens splice into the prompt, and the decoder generates a reply with
    /// the cached M-RoPE. The CoreGraphics resize differs from the reference's, so the answer is a close
    /// approximation of the reference pipeline's rather than token-identical.
    @objc(answerForImage:question:maxTokens:)
    public func answer(image: CGImage, question: String, maxTokens: Int) -> String {
        let (pixelValues, grid) = processor.process(image)
        let (visionOutput, deepstack) = visionNet(pixelValues, grid: grid)
        let merged = (grid.h / processor.mergeSize) * (grid.w / processor.mergeSize)
        let prompt = "<|im_start|>user\n<|vision_start|>"
            + String(repeating: "<|image_pad|>", count: merged)
            + "<|vision_end|>" + question + "<|im_end|>\n<|im_start|>assistant\n"
        let inputIds = tokenizer.encode(prompt).map(\.intValue)
        let generated = NFKMLXQwen3VL.generate(
            decoder: textDecoder, inputIds: inputIds, visionFeatures: visionOutput, deepstack: deepstack,
            gridT: grid.t, gridH: grid.h, gridW: grid.w, maxTokens: maxTokens, endTokens: endTokens)
        return tokenizer.decode(generated.map { NSNumber(value: $0) })
    }

    /// Builds the vision tower from a downloaded release directory, at the geometry its `config.json`
    /// describes.
    public static func visionNet(directoryURL: URL) throws -> NFKMLXQwen3VLVisionNet {
        let configuration = try NFKMLXQwen3VLVisionConfiguration.configuration(
            fromHuggingFace: directoryURL.appendingPathComponent("config.json"))
        let net = NFKMLXQwen3VLVisionNet(configuration)
        try loadVisionWeights(into: net, directoryURL: directoryURL)
        return net
    }

    /// Loads the `model.visual.` subtree, single-file or sharded. The patch-embedding convolution weight
    /// is stored 5-D (`[out, channels, temporal, patch, patch]`) and flattens to a linear weight
    /// `[out, patchInput]`.
    static func loadVisionWeights(into net: NFKMLXQwen3VLVisionNet, directoryURL: URL) throws {
        let arrays = try NFKMLXReleaseWeights.arrays(inDirectory: directoryURL) { key in
            key.hasPrefix("model.visual.") ? String(key.dropFirst("model.visual.".count)) : nil
        }
        let mapped = arrays.map { key, value -> (String, MLXArray) in
            key == "patch_embed.proj.weight" && value.ndim == 5 ? (key, value.reshaped([value.dim(0), -1])) : (key, value)
        }
        try NFKMLXWeights.apply(mapped, to: net, verifyShapes: true)
    }

    /// The decoder configuration a release's `config.json` describes: its `text_config` read through the
    /// dense reader (`qwen3_vl_text`, or `qwen3_vl_moe_text` for the 30B-A3B, whose routed experts the
    /// same stack runs). Whether the head is tied is settled by the weights rather than the config,
    /// which the 8B and 32B leave unstated: a release that ships `lm_head.weight` is untied.
    public static func decoderConfiguration(directoryURL: URL) throws -> NFKMLXLanguageConfiguration {
        let data = try Data(contentsOf: directoryURL.appendingPathComponent("config.json"))
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let text = json["text_config"] as? [String: Any] else {
            throw NFKMLXError.unsupportedConfiguration("the config carries no text_config")
        }
        var configuration = try NFKMLXLanguage.configuration(fromJSON: text)
        configuration.tiesWordEmbeddings = !(try releaseShipsHead(directoryURL: directoryURL))
        return configuration
    }

    /// Whether the release's weight files carry `lm_head.weight`, read from the shard index or the
    /// single file's header without materializing anything.
    static func releaseShipsHead(directoryURL: URL) throws -> Bool {
        let index = directoryURL.appendingPathComponent("model.safetensors.index.json")
        if let data = try? Data(contentsOf: index),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let map = json["weight_map"] as? [String: String] {
            return map["lm_head.weight"] != nil
        }
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: directoryURL.appendingPathComponent("model.safetensors"))
        return checkpoint.arrays["lm_head.weight"] != nil
    }

    // The token ids and geometry the decoder integration needs. The text decoder is the Qwen3 1.7B
    // dense stack at the release's rotary base.
    static let imageTokenId = 151_655
    static let spatialMergeSize = 2
    /// The interleaved M-RoPE section widths (`mrope_section`): 24 temporal, 20 height, 20 width
    /// channels of the 64 rotary frequency pairs.
    static let mropeSection = [24, 20, 20]
    static let deepstackLayerCount = 3

    /// The text decoder configuration: the Qwen3 1.7B geometry the release's `text_config` uses, at its
    /// own rotary base (5e6 rather than the dense 1.7B's 1e6).
    public static let decoderConfiguration: NFKMLXLanguageConfiguration = {
        var configuration = NFKMLXLanguageConfiguration.qwen3_1_7B
        configuration.ropeTheta = 5_000_000
        return configuration
    }()

    /// Builds the text decoder at the release's own geometry and loads the `model.language_model.`
    /// subtree, which is the Qwen3 dense stack under a different prefix — single-file or sharded, plus
    /// the top-level `lm_head.weight` an untied release ships.
    ///
    /// The 30B-A3B stores each layer's experts FUSED, as the `x @ W` layouts `gate_up_proj`
    /// `[experts, hidden, 2·width]` (the gate half first, then the up half) and `down_proj`
    /// `[experts, width, hidden]`; those split and transpose into the stacked `[experts, out, in]`
    /// projections the routed feed-forward holds.
    public static func decoder(directoryURL: URL) throws -> NFKMLXLanguageNet {
        let configuration = try decoderConfiguration(directoryURL: directoryURL)
        let net = NFKMLXLanguageNet(configuration)
        let prefix = "model.language_model."
        let arrays = try NFKMLXReleaseWeights.arrays(inDirectory: directoryURL) { key in
            if key.hasPrefix(prefix) { return "model." + String(key.dropFirst(prefix.count)) }
            return key == "lm_head.weight" && !configuration.tiesWordEmbeddings ? key : nil
        }
        try NFKMLXWeights.apply(releaseExperts(arrays), to: net)
        return net
    }

    /// Splits fused expert tensors into the module's stacked projections; every other pair passes through.
    static func releaseExperts(_ pairs: [(String, MLXArray)]) -> [(String, MLXArray)] {
        var mapped = [(String, MLXArray)]()
        mapped.reserveCapacity(pairs.count)
        for (key, value) in pairs {
            if key.hasSuffix(".mlp.experts.gate_up_proj"), value.ndim == 3 {
                let base = String(key.dropLast("gate_up_proj".count))
                let width = value.dim(2) / 2
                mapped.append((base + "gate_proj.weight", value[0..., 0..., 0 ..< width].swappedAxes(1, 2)))
                mapped.append((base + "up_proj.weight", value[0..., 0..., width...].swappedAxes(1, 2)))
            } else if key.hasSuffix(".mlp.experts.down_proj"), value.ndim == 3 {
                mapped.append((key + ".weight", value.swappedAxes(1, 2)))
            } else {
                mapped.append((key, value))
            }
        }
        return mapped
    }

    /// The 3-D M-RoPE positions `[3][sequence]` (temporal, height, width), the reference's
    /// `get_rope_index` for the single-image case: a text run advances all three axes together, and an
    /// image block holds the temporal axis constant while the height and width axes range over the
    /// merged patch grid, both offset past the text that precedes it.
    static func ropePositionIds(inputIds: [Int], gridT: Int, gridH: Int, gridW: Int) -> [[Int]] {
        let sequence = inputIds.count
        let llmH = gridH / spatialMergeSize, llmW = gridW / spatialMergeSize
        var temporal = [Int](), height = [Int](), width = [Int]()
        var start = 0, nextStart = 0
        while start < sequence {
            let imageStart = inputIds[start...].firstIndex(of: imageTokenId)
            let end = imageStart ?? sequence
            let textLength = end - start
            for i in 0 ..< textLength {
                temporal.append(nextStart + i); height.append(nextStart + i); width.append(nextStart + i)
            }
            if imageStart == nil {
                nextStart += textLength
                start = sequence
                continue
            }
            // The image block's positions start past the text, the reference's `text_len + st_idx`.
            let base = textLength + nextStart
            for t in 0 ..< gridT {
                for h in 0 ..< llmH {
                    for w in 0 ..< llmW {
                        temporal.append(base + t); height.append(base + h); width.append(base + w)
                    }
                }
            }
            nextStart = base + Swift.max(gridT - 1, Swift.max(llmH - 1, llmW - 1)) + 1
            start = end + gridT * llmH * llmW
        }
        return [temporal, height, width]
    }

    /// The interleaved M-RoPE cosine and sine tables `[1, sequence, headDim]`. The 64 frequency pairs
    /// are assigned to the three axes in an interleaved layout — channel `c` takes height when
    /// `c % 3 == 1`, width when `c % 3 == 2`, and temporal otherwise — matching the reference's
    /// `apply_interleaved_mrope`, and the rotation is the ordinary rotate-half over the doubled table.
    static func mropeCosSin(positionIds: [[Int]], headDimensions: Int,
                            theta: Float) -> (cos: MLXArray, sin: MLXArray) {
        let half = headDimensions / 2
        let sequence = positionIds[0].count
        let inverseFrequencies = (0 ..< half).map { 1 / powf(theta, Float(2 * $0) / Float(headDimensions)) }
        let lengthHeight = mropeSection[1] * 3, lengthWidth = mropeSection[2] * 3
        var frequencies = [Float](repeating: 0, count: sequence * half)
        for position in 0 ..< sequence {
            for channel in 0 ..< half {
                let axis: Int
                if channel < lengthHeight && channel % 3 == 1 {
                    axis = 1
                } else if channel < lengthWidth && channel % 3 == 2 {
                    axis = 2
                } else {
                    axis = 0
                }
                frequencies[position * half + channel] =
                    Float(positionIds[axis][position]) * inverseFrequencies[channel]
            }
        }
        let table = MLXArray(frequencies).reshaped([sequence, half])
        let doubled = concatenated([table, table], axis: -1)     // [sequence, headDim]
        return (cos(doubled).reshaped([1, sequence, headDimensions]),
                sin(doubled).reshaped([1, sequence, headDimensions]))
    }

    /// The decoder logits over a fused image-and-text sequence, `[1, sequence, vocabulary]`. The vision
    /// tokens splice into the decoder's input embeddings at the image-token positions, the M-RoPE
    /// positions drive the rotary, and the deepstack features add to the first three layers.
    public static func logits(decoder: NFKMLXLanguageNet, inputIds: [Int], visionFeatures: MLXArray,
                              deepstack: [MLXArray], gridT: Int, gridH: Int, gridW: Int) -> MLXArray {
        let sequence = inputIds.count
        let width = decoder.configuration.hiddenSize
        var embeddings = decoder.embed(MLXArray(inputIds.map(Int32.init)).reshaped([1, sequence]))[0]

        var featureIndex = [Int32](repeating: 0, count: sequence)
        var isImage = [Float](repeating: 0, count: sequence)
        var counter: Int32 = 0
        for position in 0 ..< sequence where inputIds[position] == imageTokenId {
            featureIndex[position] = counter
            counter += 1
            isImage[position] = 1
        }
        let indexArray = MLXArray(featureIndex)
        let flatMask = MLXArray(isImage).reshaped([sequence, 1]) .> 0
        let gathered = visionFeatures.reshaped([-1, width]).take(indexArray, axis: 0)
        embeddings = MLX.where(flatMask, gathered, embeddings).reshaped([1, sequence, width])

        let positions = ropePositionIds(inputIds: inputIds, gridT: gridT, gridH: gridH, gridW: gridW)
        let rope = mropeCosSin(positionIds: positions,
                               headDimensions: decoder.configuration.headDimensions,
                               theta: decoder.configuration.ropeTheta)
        let multimodal = NFKLMMultimodal(rope: rope, features: deepstack, featureIndex: indexArray,
                                         mask: flatMask.reshaped([1, sequence, 1]))
        let hidden = decoder.hiddenStates(fromEmbeddings: embeddings, multimodal: multimodal)
        return decoder.logits(fromHidden: hidden)
    }

    /// Greedy continuation from a fused image-and-text prompt, cached: an M-RoPE prefill with the
    /// deepstack, then one token at a time. A generated text token advances all three M-RoPE axes
    /// together, so its rotary reduces to the ordinary 1-D one at the continuing position — the
    /// deepstack applies only to the image tokens of the prompt.
    public static func generate(decoder: NFKMLXLanguageNet, inputIds: [Int], visionFeatures: MLXArray,
                                deepstack: [MLXArray], gridT: Int, gridH: Int, gridW: Int,
                                maxTokens: Int, endTokens: Set<Int>) -> [Int] {
        let sequence = inputIds.count
        let width = decoder.configuration.hiddenSize
        let headDimensions = decoder.configuration.headDimensions
        let theta = decoder.configuration.ropeTheta

        var embeddings = decoder.embed(MLXArray(inputIds.map(Int32.init)).reshaped([1, sequence]))[0]
        var featureIndex = [Int32](repeating: 0, count: sequence)
        var isImage = [Float](repeating: 0, count: sequence)
        var counter: Int32 = 0
        for position in 0 ..< sequence where inputIds[position] == imageTokenId {
            featureIndex[position] = counter
            counter += 1
            isImage[position] = 1
        }
        let indexArray = MLXArray(featureIndex)
        let flatMask = MLXArray(isImage).reshaped([sequence, 1]) .> 0
        let gathered = visionFeatures.reshaped([-1, width]).take(indexArray, axis: 0)
        embeddings = MLX.where(flatMask, gathered, embeddings).reshaped([1, sequence, width])

        let positions = ropePositionIds(inputIds: inputIds, gridT: gridT, gridH: gridH, gridW: gridW)
        var nextPosition = (positions.flatMap { $0 }.max() ?? (sequence - 1)) + 1
        let prefillRope = mropeCosSin(positionIds: positions, headDimensions: headDimensions, theta: theta)
        let multimodal = NFKLMMultimodal(rope: prefillRope, features: deepstack, featureIndex: indexArray,
                                         mask: flatMask.reshaped([1, sequence, 1]))

        let cache = NFKMLXKeyValueCache(layerCount: decoder.configuration.layerCount)
        var hidden = decoder.hiddenStates(fromEmbeddings: embeddings, cache: cache, multimodal: multimodal)

        let dummyIndex = MLXArray([Int32(0)])
        let dummyMask = MLXArray([Float(0)]).reshaped([1, 1, 1]) .> 0
        var produced = [Int]()
        for _ in 0 ..< maxTokens {
            let lastHidden = hidden[0..., (hidden.dim(1) - 1)...]
            let next = decoder.logits(fromHidden: lastHidden).reshaped([-1]).argMax().item(Int.self)
            if endTokens.contains(next) { break }
            produced.append(next)
            let rope = mropeCosSin(positionIds: [[nextPosition], [nextPosition], [nextPosition]],
                                   headDimensions: headDimensions, theta: theta)
            nextPosition += 1
            let decodeMultimodal = NFKLMMultimodal(rope: rope, features: [], featureIndex: dummyIndex,
                                                   mask: dummyMask)
            hidden = decoder.hiddenStates(
                fromEmbeddings: decoder.embed(MLXArray([Int32(next)]).reshaped([1, 1])),
                cache: cache, multimodal: decodeMultimodal)
        }
        return produced
    }
}
