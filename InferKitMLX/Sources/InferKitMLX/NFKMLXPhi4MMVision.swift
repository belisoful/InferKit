//
//  NFKMLXPhi4MMVision.swift
//  InferKitMLX
//
//  Phi-4-multimodal's image tower: a SigLIP-so400m vision transformer, an average-pool compression, the
//  Phi-3.5 HD transform that lays out the global and sub-image crops with learned row and image
//  separators, and a two-layer projector into the decoder's embedding space.
//
//  The image features are the SigLIP encoder's SECOND-TO-LAST hidden state (the patch tokens after 26 of
//  its 27 layers). The last layer, the post-layer-norm, and the attention-pooling head are present in the
//  checkpoint but take no part in this path, so the tower runs 26 layers and stops.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import MLX
import MLXNN

/// The SigLIP-so400m geometry Phi-4-multimodal's image tower runs, truncated to the layer the feature is
/// read from. The full tower has 27 layers; the feature is `hidden_states[-2]`, so 26 layers run.
extension NFKMLXSigLIPConfiguration {
    public static let phi4mm = NFKMLXSigLIPConfiguration(
        hiddenSize: 1152, layerCount: 26, headCount: 16, intermediateSize: 4304,
        patchSize: 14, imageSize: 448, layerNormEpsilon: 1e-6)
}

/// SigLIP patch embedding with NaViT position ids. Each crop's valid patches (the top-left region its
/// patch mask marks) take ids that stretch that region over the full `side × side` position grid: a
/// valid row or column index `k` of `n` maps to the number of boundaries `1/side … (side-1)/side` at or
/// below `k/n`. A full-resolution crop reduces to the plain `0 … side²-1` grid; padded patches keep id 0.
/// SmolVLM's shared embedding shifts its ids by a `1 - 1e-6` coordinate factor, so this tower keeps its own.
final class NFKPhi4MMSigLIPEmbeddings: Module {
    @ModuleInfo(key: "patch_embedding") var patchEmbedding: Conv2d
    @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding
    let side: Int

    init(_ c: NFKMLXSigLIPConfiguration) {
        side = c.grid
        _patchEmbedding.wrappedValue = NFKConv2d(inputChannels: 3, outputChannels: c.hiddenSize,
                                              kernelSize: IntOrPair(c.patchSize),
                                              stride: IntOrPair(c.patchSize), bias: true)
        _positionEmbedding.wrappedValue = Embedding(embeddingCount: c.positionCount, dimensions: c.hiddenSize)
        super.init()
    }

    /// The position id of every patch of one crop, row-major, from its valid `rows × columns` region.
    static func positionIds(side: Int, validRows: Int, validColumns: Int) -> [Int32] {
        func buckets(_ count: Int) -> [Int32] {
            (0 ..< count).map { k in
                let coordinate = Double(k) / Double(count)
                return Int32((1 ..< side).filter { Double($0) / Double(side) <= coordinate }.count)
            }
        }
        let rows = buckets(validRows), columns = buckets(validColumns)
        var ids = [Int32](repeating: 0, count: side * side)
        for r in 0 ..< validRows {
            for c in 0 ..< validColumns { ids[r * side + c] = rows[r] * Int32(side) + columns[c] }
        }
        return ids
    }

    /// `pixelValues` is `[crops, height, width, 3]`; `validRegions` gives each crop's valid `(rows,
    /// columns)` in patches, or nil for all-valid crops. Returns `[crops, patches, hidden]`.
    func callAsFunction(_ pixelValues: MLXArray, validRegions: [(rows: Int, columns: Int)]? = nil) -> MLXArray {
        let patches = patchEmbedding(pixelValues)                 // [crops, grid, grid, hidden]
        let crops = patches.dim(0)
        let embedded = patches.reshaped([crops, side * side, patches.dim(3)])
        let regions = validRegions ?? Array(repeating: (rows: side, columns: side), count: crops)
        let ids = regions.flatMap { Self.positionIds(side: side, validRows: $0.rows, validColumns: $0.columns) }
        return embedded + positionEmbedding(MLXArray(ids).reshaped([crops, side * side]))
    }
}

/// The image tower up to the penultimate hidden state: the SigLIP patch embedding and 26 encoder layers,
/// without the final layer, the post-layer-norm, or the pooling head. Reuses the shared SigLIP encoder
/// blocks with Phi's NaViT patch embedding; padded patches are masked out as attention keys.
final class NFKMLXPhi4MMVisionNet: Module {
    @ModuleInfo(key: "embeddings") var embeddings: NFKPhi4MMSigLIPEmbeddings
    @ModuleInfo(key: "encoder") var encoder: NFKSigLIPEncoder

    init(_ c: NFKMLXSigLIPConfiguration) {
        _embeddings.wrappedValue = NFKPhi4MMSigLIPEmbeddings(c)
        _encoder.wrappedValue = NFKSigLIPEncoder(c)
        super.init()
    }

    /// `pixelValues` is `[crops, height, width, 3]`; `validRegions` gives each crop's valid patch region,
    /// or nil when every patch is valid. Returns each crop's patch features `[crops, patches, hidden]` from
    /// the penultimate layer (no post-layer-norm).
    func callAsFunction(_ pixelValues: MLXArray, validRegions: [(rows: Int, columns: Int)]? = nil) -> MLXArray {
        var hidden = embeddings(pixelValues, validRegions: validRegions)
        let side = embeddings.side
        // The reference adds no mask when every patch of every crop is valid.
        var mask: MLXArray?
        if let validRegions, validRegions.contains(where: { $0.rows < side || $0.columns < side }) {
            var additive = [Float](repeating: 0, count: validRegions.count * side * side)
            for (crop, region) in validRegions.enumerated() {
                for r in 0 ..< side {
                    for c in 0 ..< side where r >= region.rows || c >= region.columns {
                        additive[(crop * side + r) * side + c] = -Float.greatestFiniteMagnitude
                    }
                }
            }
            mask = MLXArray(additive).reshaped([validRegions.count, 1, 1, side * side])
        }
        for layer in encoder.layers { hidden = layer(hidden, mask: mask) }
        return hidden
    }
}

/// The complete image tower: the SigLIP encoder, the learned separators, and the projector, under the
/// release's `model.embed_tokens_extend.image_embed.` subtree.
public final class NFKMLXPhi4MMImageNet: Module {
    @ModuleInfo(key: "img_processor") var processor: NFKMLXPhi4MMVisionNet
    @ModuleInfo(key: "img_projection") var projection: [Module]
    @ParameterInfo(key: "glb_GN") var globalSeparator: MLXArray
    @ParameterInfo(key: "sub_GN") var subSeparator: MLXArray

    let hidden: Int
    let cropSize = 448
    let pooledGrid = 16                                  // 32×32 patches average-pooled 2× to 16×16

    init(_ c: NFKMLXSigLIPConfiguration = .phi4mm, decoderHidden: Int = 3072) {
        hidden = c.hiddenSize
        _processor.wrappedValue = NFKMLXPhi4MMVisionNet(c)
        _projection.wrappedValue = [
            Linear(c.hiddenSize, decoderHidden), Module(), Linear(decoderHidden, decoderHidden),
        ]
        _globalSeparator.wrappedValue = MLXArray.zeros([1, 1, c.hiddenSize])
        _subSeparator.wrappedValue = MLXArray.zeros([1, 1, 1, c.hiddenSize])
        super.init()
    }

    /// Average-pools each crop's 32×32 patch features 2× to 16×16: `[crops, 256, hidden]`.
    func pooled(_ features: MLXArray) -> MLXArray {
        let crops = features.dim(0), grid = pooledGrid
        return NFKReferenceRounding.wide(features.reshaped([crops, grid, 2, grid, 2, hidden])) { $0.mean(axes: [2, 4]) }
            .reshaped([crops, grid * grid, hidden])
    }

    /// The projected image embeddings for one image, `[tokens, decoderHidden]`.
    ///
    /// @discussion `pixels` holds the global view and then the `h × w` sub-image crops, row-major;
    /// `imageSize` is the padded HD image size, a multiple of the 448-pixel crop. `validPatches` is the
    /// HD image's unpadded region in 14-pixel patches (nil when nothing is padded): it sets each crop's
    /// NaViT position ids and key mask, and trims the pooled sub-image grid to its useful rows and
    /// columns. The layout is `[sub-image grid with a row separator per row, an image separator, the
    /// global 16×16 grid with a row separator per row]` (the `avg_pool_2d` / `sub_glb` HD transform,
    /// whose pooling leaves the patch channels unmerged), then projected.
    func projected(pixels: MLXArray, imageSize: (height: Int, width: Int),
                   validPatches: (rows: Int, columns: Int)? = nil) -> MLXArray {
        let grid = pooledGrid, side = 2 * pooledGrid
        let h = imageSize.height / cropSize, w = imageSize.width / cropSize
        let valid = validPatches ?? (rows: h * side, columns: w * side)
        var regions: [(rows: Int, columns: Int)] = [(rows: side, columns: side)]
        for tileRow in 0 ..< h {
            for tileColumn in 0 ..< w {
                regions.append((rows: Swift.min(Swift.max(valid.rows - tileRow * side, 0), side),
                                columns: Swift.min(Swift.max(valid.columns - tileColumn * side, 0), side)))
            }
        }
        let features = pooled(processor(pixels, validRegions: regions))   // [crops, 256, hidden]

        var global = features[0 ..< 1].reshaped([1, grid, grid, hidden])
        global = concatenated([global, broadcast(subSeparator, to: [1, grid, 1, hidden])], axis: 2)
            .reshaped([1, -1, hidden])

        // The pooled mask keeps every other patch row and column, so the useful pooled extent is the
        // valid patch extent halved and rounded up.
        let usefulRows = (valid.rows + 1) / 2, usefulColumns = (valid.columns + 1) / 2
        var sub = features[1 ..< (1 + h * w)]
            .reshaped([1, h, w, grid, grid, hidden])
            .transposed(0, 1, 3, 2, 4, 5)
            .reshaped([1, h * grid, w * grid, hidden])
        sub = sub[0..., 0 ..< usefulRows, 0 ..< usefulColumns, 0...]
        sub = concatenated([sub, broadcast(subSeparator, to: [1, usefulRows, 1, hidden])], axis: 2)
            .reshaped([1, -1, hidden])

        let merged = concatenated([sub, globalSeparator, global], axis: 1)   // sub_glb order
        return project(merged)[0]
    }

    /// The image-token count the prompt reserves for an image, which equals the rows of `projected`.
    static func tokenCount(validPatches: (rows: Int, columns: Int)) -> Int {
        let usefulRows = (validPatches.rows + 1) / 2, usefulColumns = (validPatches.columns + 1) / 2
        return 256 + 1 + usefulRows * usefulColumns + usefulRows + 16
    }

    private func project(_ x: MLXArray) -> MLXArray {
        (projection[2] as! Linear)(NFKReferenceRounding.wide((projection[0] as! Linear)(x)) { gelu($0) })
    }
}
