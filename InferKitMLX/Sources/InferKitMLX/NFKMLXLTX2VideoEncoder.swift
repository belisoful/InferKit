//
//  NFKMLXLTX2VideoEncoder.swift
//  InferKitMLX
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import InferKit
import MLX
import MLXNN

// The LTX-2 video autoencoder's encoder (`LTX2VideoEncoder3d`, Lightricks), which a fine-tune's data
// preparation runs over each clip. A 4×4 patch shuffle folds space into 48 channels (the channel index
// runs `(c, pw, ph)`: width before height), a causal 3-D convolution widens them to the latent width, and
// four down blocks of resnets each end in a downsampler that halves space, time, or both: a padded
// convolution to a fraction of the block's output width whose result is folded space-to-depth, plus the
// input folded the same way and averaged over channel groups to that width. The released encoder's blocks
// halve space, then time, then both twice, so a clip of `8k + 1` frames at `32h × 32w` encodes to
// `k + 1` latent frames of `h × w`. A temporal downsampler first repeats the leading frame, which is why
// the first latent frame holds a single pixel frame. A mid block, the per-pixel RMS norm, SiLU, and a
// causal convolution to `latentChannels + 1` channels close it: the posterior's means and one shared log
// variance. Lightricks' trainer keeps the means and normalizes them by the latents' per-channel
// statistics; ``NFKMLXLTX2VideoEncoderNet/latents(_:)`` returns the same.
//
// Tensors flow as `[batch, frames, height, width, channels]`; the reference is `[B, C, T, H, W]`.

/// The geometry of an LTX-2 video encoder, read from a release's `vae/config.json`.
public struct NFKMLXLTX2VideoEncoderConfiguration: Sendable {
    public var inChannels: Int
    public var latentChannels: Int
    public var blockOutChannels: [Int]
    /// The resnets of each down block and then of the mid block.
    public var layersPerBlock: [Int]
    public var spatioTemporalScaling: [Bool]
    /// Each down block's downsampler: `spatial`, `temporal`, `spatiotemporal`, or `conv` (a stride-2
    /// convolution).
    public var downsampleTypes: [String]
    public var patchSize: Int
    public var resnetEps: Float
    public var isCausal: Bool
    public var reflectsSpatialPadding: Bool

    public init(inChannels: Int = 3, latentChannels: Int = 128, blockOutChannels: [Int] = [256, 512, 1024, 1024],
                layersPerBlock: [Int] = [4, 6, 4, 2, 2], spatioTemporalScaling: [Bool] = [true, true, true, true],
                downsampleTypes: [String] = ["spatial", "temporal", "spatiotemporal", "spatiotemporal"],
                patchSize: Int = 4, resnetEps: Float = 1e-6, isCausal: Bool = true,
                reflectsSpatialPadding: Bool = false) {
        self.inChannels = inChannels
        self.latentChannels = latentChannels
        self.blockOutChannels = blockOutChannels
        self.layersPerBlock = layersPerBlock
        self.spatioTemporalScaling = spatioTemporalScaling
        self.downsampleTypes = downsampleTypes
        self.patchSize = patchSize
        self.resnetEps = resnetEps
        self.isCausal = isCausal
        self.reflectsSpatialPadding = reflectsSpatialPadding
    }

    /// The encoder LTX-2.3 and LTX-2.5 share (`vae/config.json`).
    public static let ltx23 = NFKMLXLTX2VideoEncoderConfiguration()

    /// The release's arrangement at a thirty-second of its width, for weight-free tests.
    public static let tiny = NFKMLXLTX2VideoEncoderConfiguration(
        latentChannels: 8, blockOutChannels: [8, 16, 32, 32], layersPerBlock: [1, 1, 1, 1, 1])

    /// The space-to-depth stride of a downsampler type `(time, height, width)`.
    static func stride(of type: String) -> (Int, Int, Int)? {
        switch type {
        case "spatial": return (1, 2, 2)
        case "temporal": return (2, 1, 1)
        case "spatiotemporal": return (2, 2, 2)
        default: return nil
        }
    }
}

/// The fold of `x` `[B, F, H, W, C]` by `stride` into `[B, F/st, H/sh, W/sw, C·st·sh·sw]`, the channel index
/// running `(c, dt, dh, dw)`: the reference's `unflatten` of each axis and `permute(0, 1, 3, 5, 7, 2, 4, 6)`.
func ltx2SpaceToDepth(_ x: MLXArray, stride: (Int, Int, Int)) -> MLXArray {
    let (b, f, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3), x.dim(4))
    let (st, sh, sw) = stride
    return x.reshaped([b, f / st, st, h / sh, sh, w / sw, sw, c])
        .transposed(0, 1, 3, 5, 7, 2, 4, 6)
        .reshaped([b, f / st, h / sh, w / sw, c * st * sh * sw])
}

/// An LTX-2 downsampler: a padded convolution folded space-to-depth, plus the input folded the same way
/// and averaged over channel groups to the same width. A temporal stride first repeats the leading frame
/// `stride − 1` times.
final class NFKLTX2VideoDownsampler: Module {
    @ModuleInfo(key: "conv") var conv: NFKLTX2CausalConv3d
    let stride: (Int, Int, Int)
    let groupSize: Int

    init(_ inChannels: Int, _ outChannels: Int, stride: (Int, Int, Int), reflects: Bool) {
        self.stride = stride
        let volume = stride.0 * stride.1 * stride.2
        groupSize = inChannels * volume / outChannels
        _conv.wrappedValue = NFKLTX2CausalConv3d(inChannels, outChannels / volume, reflects: reflects)
    }

    func callAsFunction(_ x: MLXArray, causal: Bool) -> MLXArray {
        let padded = stride.0 > 1 ? concatenated([x[0..., 0 ..< (stride.0 - 1)], x], axis: 1) : x
        let folded = ltx2SpaceToDepth(padded, stride: stride)
        let shape = folded.shape
        let residual = folded.reshaped(Array(shape.dropLast()) + [shape[4] / groupSize, groupSize]).mean(axis: -1)
        return ltx2SpaceToDepth(conv(padded, causal: causal), stride: stride) + residual
    }
}

/// An LTX-2 down block: resnets at the input width, then the downsampler to the output width.
final class NFKLTX2VideoDownBlock: Module {
    @ModuleInfo(key: "resnets") var resnets: [NFKLTX2VideoResnet]
    @ModuleInfo(key: "downsamplers") var downsamplers: [Module]

    init(inChannels: Int, outChannels: Int, layers: Int, scales: Bool, downsampleType: String, eps: Float,
         reflects: Bool) throws {
        _resnets.wrappedValue = (0 ..< layers).map { _ in
            NFKLTX2VideoResnet(inChannels, inChannels, eps: eps, reflects: reflects)
        }
        if !scales {
            _downsamplers.wrappedValue = []
        } else if downsampleType == "conv" {
            _downsamplers.wrappedValue = [NFKLTX2CausalConv3d(inChannels, inChannels, stride: 2, reflects: reflects)]
        } else if let stride = NFKMLXLTX2VideoEncoderConfiguration.stride(of: downsampleType) {
            _downsamplers.wrappedValue = [NFKLTX2VideoDownsampler(inChannels, outChannels, stride: stride,
                                                                  reflects: reflects)]
        } else {
            throw NFKMLXError.unsupportedConfiguration("this encoder has no \(downsampleType) downsampler")
        }
    }

    func callAsFunction(_ x: MLXArray, causal: Bool) -> MLXArray {
        let h = resnets.reduce(x) { $1($0, causal: causal) }
        return downsamplers.reduce(h) { h, layer in
            if let downsampler = layer as? NFKLTX2VideoDownsampler { return downsampler(h, causal: causal) }
            return (layer as! NFKLTX2CausalConv3d)(h, causal: causal)
        }
    }
}

/// The LTX-2 video encoder: a clip `[B, 8k + 1, 32h, 32w, 3]` in `−1…1` → the posterior's moments
/// `[B, k + 1, h, w, latentChannels + 1]`, the means and one log variance.
public final class NFKMLXLTX2VideoEncoderNet: Module {
    @ModuleInfo(key: "conv_in") var convIn: NFKLTX2CausalConv3d
    @ModuleInfo(key: "down_blocks") var downBlocks: [NFKLTX2VideoDownBlock]
    @ModuleInfo(key: "mid_block") var midBlock: NFKLTX2VideoMidBlock
    @ModuleInfo(key: "conv_out") var convOut: NFKLTX2CausalConv3d

    public let configuration: NFKMLXLTX2VideoEncoderConfiguration

    public init(_ c: NFKMLXLTX2VideoEncoderConfiguration) throws {
        configuration = c
        let reflects = c.reflectsSpatialPadding
        _convIn.wrappedValue = NFKLTX2CausalConv3d(c.inChannels * c.patchSize * c.patchSize, c.latentChannels,
                                                   reflects: reflects)
        var output = c.latentChannels
        var blocks = [NFKLTX2VideoDownBlock]()
        for (index, width) in c.blockOutChannels.enumerated() {
            blocks.append(try NFKLTX2VideoDownBlock(
                inChannels: output, outChannels: width, layers: c.layersPerBlock[index],
                scales: c.spatioTemporalScaling[index], downsampleType: c.downsampleTypes[index], eps: c.resnetEps,
                reflects: reflects))
            output = width
        }
        _downBlocks.wrappedValue = blocks
        _midBlock.wrappedValue = NFKLTX2VideoMidBlock(output, layers: c.layersPerBlock.last ?? 0, eps: c.resnetEps,
                                                      reflects: reflects)
        _convOut.wrappedValue = NFKLTX2CausalConv3d(output, c.latentChannels + 1, reflects: reflects)
    }

    /// The posterior's moments for `video`: `latentChannels` means, then the shared log variance.
    public func callAsFunction(_ video: MLXArray) -> MLXArray {
        let p = configuration.patchSize, causal = configuration.isCausal
        let (b, f, height, width, c) = (video.dim(0), video.dim(1), video.dim(2), video.dim(3), video.dim(4))
        let patched = video.reshaped([b, f, height / p, p, width / p, p, c])
            .transposed(0, 1, 2, 4, 6, 5, 3)
            .reshaped([b, f, height / p, width / p, c * p * p])
        var h = convIn(patched, causal: causal)
        for block in downBlocks { h = block(h, causal: causal) }
        h = midBlock(h, causal: causal)
        return convOut(NFKReferenceRounding.silu(ltx2PixelNorm(h, eps: 1e-8)), causal: causal)
    }

    /// The normalized latents of `video` `[B, 8k + 1, 32h, 32w, 3]` in `−1…1`: the posterior's means less
    /// the latents' per-channel mean, over their per-channel standard deviation, as the trainer stores them.
    public func latents(_ video: MLXArray, statistics: (mean: MLXArray, std: MLXArray)) -> MLXArray {
        let means = self(video)[.ellipsis, 0 ..< configuration.latentChannels]
        return (means - statistics.mean.asType(means.dtype)) / statistics.std.asType(means.dtype)
    }
}

extension NFKMLXLTX2VideoVAE {

    /// The encoder geometry a `vae/config.json` describes.
    ///
    /// Introduced in InferKit 0.4.0.
    public static func encoderConfiguration(fromHuggingFace url: URL) throws -> NFKMLXLTX2VideoEncoderConfiguration {
        guard let json = try JSONSerialization.jsonObject(with: try Data(contentsOf: url)) as? [String: Any] else {
            throw NFKMLXError.unsupportedConfiguration("\(url.lastPathComponent) is not a JSON object")
        }
        if let kind = json["_class_name"] as? String, kind != "AutoencoderKLLTX2Video" {
            throw NFKMLXError.unsupportedConfiguration("this reads an LTX-2 video autoencoder, not \(kind)")
        }
        let base = NFKMLXLTX2VideoEncoderConfiguration.ltx23
        let widths = (json["block_out_channels"] as? [NSNumber])?.map(\.intValue) ?? base.blockOutChannels
        let scaling: [Bool]
        if let single = json["spatio_temporal_scaling"] as? Bool {
            scaling = [Bool](repeating: single, count: widths.count)
        } else {
            scaling = (json["spatio_temporal_scaling"] as? [NSNumber])?.map(\.boolValue) ?? base.spatioTemporalScaling
        }
        let padding = json["encoder_spatial_padding_mode"] as? String ?? "zeros"
        guard padding == "zeros" || padding == "reflect" else {
            throw NFKMLXError.unsupportedConfiguration(
                "this encoder pads space with zeros or by reflection; the release asks for \(padding)")
        }
        if let types = json["down_block_types"] as? [String], types.contains(where: { $0 != "LTX2VideoDownBlock3D" }) {
            throw NFKMLXError.unsupportedConfiguration("this encoder builds LTX2VideoDownBlock3D only, not \(types)")
        }
        return NFKMLXLTX2VideoEncoderConfiguration(
            inChannels: (json["in_channels"] as? NSNumber)?.intValue ?? 3,
            latentChannels: (json["latent_channels"] as? NSNumber)?.intValue ?? 128,
            blockOutChannels: widths,
            layersPerBlock: (json["layers_per_block"] as? [NSNumber])?.map(\.intValue) ?? base.layersPerBlock,
            spatioTemporalScaling: scaling,
            downsampleTypes: json["downsample_type"] as? [String] ?? base.downsampleTypes,
            patchSize: (json["patch_size"] as? NSNumber)?.intValue ?? 4,
            resnetEps: (json["resnet_norm_eps"] as? NSNumber)?.floatValue ?? 1e-6,
            isCausal: (json["encoder_causal"] as? NSNumber)?.boolValue ?? true,
            reflectsSpatialPadding: padding == "reflect")
    }

    /// Loads a release's `vae/` directory into `encoder`: the `encoder.` tensors, 5-D convolutions transposed
    /// to MLX's layout. Returns the latents' per-channel mean and standard deviation, which
    /// ``NFKMLXLTX2VideoEncoderNet/latents(_:statistics:)`` normalizes by.
    ///
    /// Introduced in InferKit 0.4.0.
    @discardableResult
    public static func loadEncoderWeights(into encoder: NFKMLXLTX2VideoEncoderNet, from directory: URL,
                                          precision: NFKMLXWeightPrecision = .float32) throws
        -> (mean: MLXArray, std: MLXArray) {
        let arrays = try NFKMLXReleaseWeights.materializedArrays(
            inDirectory: directory, precision: precision,
            remap: { $0.hasPrefix("encoder.") || $0.hasPrefix("latents_") ? $0 : nil },
            transform: { key, value in (key, value.ndim == 5 ? value.transposed(0, 2, 3, 4, 1) : value) })
        var statistics = [String: MLXArray]()
        var mapped = [(String, MLXArray)]()
        for (key, value) in arrays {
            if key == "latents_mean" || key == "latents_std" {
                statistics[key] = value.asType(.float32)
            } else {
                mapped.append((String(key.dropFirst("encoder.".count)), value))
            }
        }
        try NFKMLXWeights.apply(mapped, to: encoder, verifyShapes: true)
        let channels = encoder.configuration.latentChannels
        return (statistics["latents_mean"] ?? MLXArray.zeros([channels]),
                statistics["latents_std"] ?? MLXArray.ones([channels]))
    }
}
