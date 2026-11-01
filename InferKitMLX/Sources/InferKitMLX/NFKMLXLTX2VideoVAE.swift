//
//  NFKMLXLTX2VideoVAE.swift
//  InferKitMLX
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import InferKit
import MLX
import MLXNN

// The LTX-2 video autoencoder's decoder (`AutoencoderKLLTX2Video`, Lightricks). It keeps LTX-Video
// 0.9.5's shape (a 3-D convolutional decoder whose upsamplers are a convolution and a depth-to-space
// shuffle over time, height and width) and changes three things: every normalization is an
// affine-free per-pixel RMS norm over the channel axis computed with a square root (eps 1e-8), a resnet
// whose width changes normalizes its shortcut with an AFFINE layer norm before a plain 1×1×1
// convolution, and the decoder pads time by replicating the edge frames on both sides when it decodes
// non-causally, which the release does. Each up block's channel factor divides the width after its
// shuffle, so the released decoder's four blocks halve, keep, halve and halve the width while they
// double time, time, time, and space twice.
//
// Tensors flow as `[batch, frames, height, width, channels]` (MLX's NDHWC); the reference is
// `[B, C, T, H, W]`. A latent of `F` frames decodes to `8 (F − 1) + 1`: each temporal shuffle drops
// the `stride − 1` frames before the first, the causal encoder's single leading frame.

/// The geometry of an LTX-2 video decoder, read from a release's `vae/config.json`.
public struct NFKMLXLTX2VideoVAEConfiguration: Sendable {
    public var outChannels: Int
    public var latentChannels: Int
    /// The decoder's block widths, in the release's order (narrowest first); the decoder runs them
    /// reversed.
    public var blockOutChannels: [Int]
    /// The resnets of the mid block and then of each up block, in the release's order.
    public var layersPerBlock: [Int]
    public var spatioTemporalScaling: [Bool]
    /// Each up block's shuffle, `spatiotemporal`, `temporal` or `spatial`, in the order the decoder runs
    /// them.
    public var upsampleTypes: [String]
    public var upsampleResidual: [Bool]
    public var upsampleFactors: [Int]
    public var patchSize: Int
    public var patchSizeT: Int
    public var resnetEps: Float
    public var isCausal: Bool
    /// Whether the spatial padding reflects (`reflect`) rather than zero-fills (`zeros`).
    public var reflectsSpatialPadding: Bool
    public var spatialCompression: Int
    public var temporalCompression: Int

    public init(outChannels: Int = 3, latentChannels: Int = 128,
                blockOutChannels: [Int] = [256, 512, 512, 1024], layersPerBlock: [Int] = [4, 6, 4, 2, 2],
                spatioTemporalScaling: [Bool] = [true, true, true, true],
                upsampleTypes: [String] = ["spatiotemporal", "spatiotemporal", "temporal", "spatial"],
                upsampleResidual: [Bool] = [false, false, false, false], upsampleFactors: [Int] = [2, 2, 1, 2],
                patchSize: Int = 4, patchSizeT: Int = 1, resnetEps: Float = 1e-6, isCausal: Bool = false,
                reflectsSpatialPadding: Bool = false, spatialCompression: Int = 32,
                temporalCompression: Int = 8) {
        self.outChannels = outChannels
        self.latentChannels = latentChannels
        self.blockOutChannels = blockOutChannels
        self.layersPerBlock = layersPerBlock
        self.spatioTemporalScaling = spatioTemporalScaling
        self.upsampleTypes = upsampleTypes
        self.upsampleResidual = upsampleResidual
        self.upsampleFactors = upsampleFactors
        self.patchSize = patchSize
        self.patchSizeT = patchSizeT
        self.resnetEps = resnetEps
        self.isCausal = isCausal
        self.reflectsSpatialPadding = reflectsSpatialPadding
        self.spatialCompression = spatialCompression
        self.temporalCompression = temporalCompression
    }

    /// The LTX-2.3 release's decoder (`diffusers/LTX-2.3-Diffusers`, `vae/config.json`).
    public static let ltx23 = NFKMLXLTX2VideoVAEConfiguration()

    /// The release's arrangement at a thirty-second of its width, for weight-free tests.
    public static let tiny = NFKMLXLTX2VideoVAEConfiguration(
        latentChannels: 8, blockOutChannels: [8, 16, 16, 32], layersPerBlock: [1, 2, 1, 1, 1])
}

/// The affine-free per-pixel RMS norm over the channel axis, `x / sqrt(mean(x²) + eps)`.
func ltx2PixelNorm(_ x: MLXArray, eps: Float) -> MLXArray {
    let wide = x.asType(.float32)
    return (wide / sqrt(wide.square().mean(axis: -1, keepDims: true) + eps)).asType(x.dtype)
}

/// Pads `x` by reflection along `axis`, `count` on each side, the reference's `padding_mode="reflect"`.
func ltx2ReflectPad(_ x: MLXArray, axis: Int, count: Int) -> MLXArray {
    guard count > 0 else { return x }
    let length = x.dim(axis)
    let left = (1 ... count).reversed().map { x.take(MLXArray(Int32($0)), axis: axis).expandedDimensions(axis: axis) }
    let right = (1 ... count).map {
        x.take(MLXArray(Int32(length - 1 - $0)), axis: axis).expandedDimensions(axis: axis)
    }
    return concatenated(left + [x] + right, axis: axis)
}

/// A 3-D convolution whose time axis is padded by replicating edge frames, all past frames when causal
/// and half on each side when not; its space axes are padded by zeros or by reflection.
final class NFKLTX2CausalConv3d: Module {
    @ModuleInfo(key: "conv") var conv: Conv3d
    let kernel: Int
    let reflects: Bool

    init(_ inChannels: Int, _ outChannels: Int, kernel: Int = 3, stride: Int = 1, reflects: Bool) {
        self.kernel = kernel
        self.reflects = reflects
        let spatial = reflects ? 0 : kernel / 2
        _conv.wrappedValue = Conv3d(inputChannels: inChannels, outputChannels: outChannels,
                                    kernelSize: IntOrTriple(kernel), stride: IntOrTriple(stride),
                                    padding: IntOrTriple((0, spatial, spatial)))
    }

    func callAsFunction(_ x: MLXArray, causal: Bool) -> MLXArray {
        var padded = x
        if kernel > 1 {
            let frames = x.dim(1)
            let first = x[0..., 0 ..< 1]
            let last = x[0..., (frames - 1) ..< frames]
            func repeated(_ frame: MLXArray, _ count: Int) -> MLXArray {
                broadcast(frame, to: [frame.dim(0), count, frame.dim(2), frame.dim(3), frame.dim(4)])
            }
            if causal {
                padded = concatenated([repeated(first, kernel - 1), x], axis: 1)
            } else {
                let half = (kernel - 1) / 2
                padded = concatenated([repeated(first, half), x, repeated(last, half)], axis: 1)
            }
            if reflects {
                padded = ltx2ReflectPad(ltx2ReflectPad(padded, axis: 2, count: kernel / 2), axis: 3, count: kernel / 2)
            }
        }
        return conv(padded)
    }
}

/// An LTX-2 resnet: per-pixel RMS norm, SiLU and a padded convolution, twice, added to the input, whose
/// width a layer-normed 1×1×1 convolution changes where the block widens or narrows.
final class NFKLTX2VideoResnet: Module {
    @ModuleInfo(key: "conv1") var conv1: NFKLTX2CausalConv3d
    @ModuleInfo(key: "conv2") var conv2: NFKLTX2CausalConv3d
    @ModuleInfo(key: "norm3") var norm3: LayerNorm?
    @ModuleInfo(key: "conv_shortcut") var convShortcut: Conv3d?

    init(_ inChannels: Int, _ outChannels: Int, eps: Float, reflects: Bool) {
        _conv1.wrappedValue = NFKLTX2CausalConv3d(inChannels, outChannels, reflects: reflects)
        _conv2.wrappedValue = NFKLTX2CausalConv3d(outChannels, outChannels, reflects: reflects)
        if inChannels != outChannels {
            _norm3.wrappedValue = LayerNorm(dimensions: inChannels, eps: eps)
            _convShortcut.wrappedValue = Conv3d(inputChannels: inChannels, outputChannels: outChannels,
                                                kernelSize: IntOrTriple(1))
        }
    }

    func callAsFunction(_ x: MLXArray, causal: Bool) -> MLXArray {
        var h = conv1(NFKReferenceRounding.silu(ltx2PixelNorm(x, eps: 1e-8)), causal: causal)
        h = conv2(NFKReferenceRounding.silu(ltx2PixelNorm(h, eps: 1e-8)), causal: causal)
        var shortcut = x
        if let norm3 { shortcut = norm3(shortcut) }
        if let convShortcut { shortcut = convShortcut(shortcut) }
        return h + shortcut
    }
}

/// A depth-to-space shuffle of the trailing channel axis into (time, height, width) by `stride`, then
/// the `stride.t − 1` frames before the first dropped. The channel index splits as
/// `(c, dt, dh, dw)`, the reference's `reshape(B, −1, st, sh, sw, F, H, W)`.
func ltx2DepthToSpace(_ x: MLXArray, stride: (Int, Int, Int)) -> MLXArray {
    let (b, t, h, w) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
    let c = x.dim(4) / (stride.0 * stride.1 * stride.2)
    let shuffled = x.reshaped([b, t, h, w, c, stride.0, stride.1, stride.2])
        .transposed(0, 1, 5, 2, 6, 3, 7, 4)
        .reshaped([b, t * stride.0, h * stride.1, w * stride.2, c])
    return shuffled[0..., (stride.0 - 1)...]
}

/// An LTX-2 upsampler: a convolution to `out · stride³ / factor` channels and a depth-to-space shuffle,
/// with the shuffled input tiled to the output width added back where the release asks for it.
final class NFKLTX2VideoUpsampler: Module {
    @ModuleInfo(key: "conv") var conv: NFKLTX2CausalConv3d
    let stride: (Int, Int, Int)
    let residual: Bool
    let factor: Int

    init(_ channels: Int, stride: (Int, Int, Int), residual: Bool, factor: Int, reflects: Bool) {
        self.stride = stride
        self.residual = residual
        self.factor = factor
        let outChannels = channels * stride.0 * stride.1 * stride.2 / factor
        _conv.wrappedValue = NFKLTX2CausalConv3d(channels, outChannels, reflects: reflects)
    }

    func callAsFunction(_ x: MLXArray, causal: Bool) -> MLXArray {
        var out = ltx2DepthToSpace(conv(x, causal: causal), stride: stride)
        if residual {
            let shuffled = ltx2DepthToSpace(x, stride: stride)
            let repeats = stride.0 * stride.1 * stride.2 / factor
            out = out + tiled(shuffled, repetitions: [1, 1, 1, 1, repeats])
        }
        return out
    }
}

/// A decoder up block: a resnet where the width changes, the upsampler, then the resnets.
final class NFKLTX2VideoUpBlock: Module {
    @ModuleInfo(key: "conv_in") var convIn: NFKLTX2VideoResnet?
    @ModuleInfo(key: "upsamplers") var upsamplers: [NFKLTX2VideoUpsampler]?
    @ModuleInfo(key: "resnets") var resnets: [NFKLTX2VideoResnet]

    init(inChannels: Int, outChannels: Int, layers: Int, scales: Bool, upsampleType: String, residual: Bool,
         factor: Int, eps: Float, reflects: Bool) {
        _convIn.wrappedValue = inChannels != outChannels
            ? NFKLTX2VideoResnet(inChannels, outChannels, eps: eps, reflects: reflects) : nil
        if scales {
            let stride: (Int, Int, Int)
            switch upsampleType {
            case "spatial": stride = (1, 2, 2)
            case "temporal": stride = (2, 1, 1)
            default: stride = (2, 2, 2)
            }
            _upsamplers.wrappedValue = [NFKLTX2VideoUpsampler(outChannels * factor, stride: stride,
                                                              residual: residual, factor: factor,
                                                              reflects: reflects)]
        }
        _resnets.wrappedValue = (0 ..< layers).map { _ in
            NFKLTX2VideoResnet(outChannels, outChannels, eps: eps, reflects: reflects)
        }
    }

    func callAsFunction(_ x: MLXArray, causal: Bool) -> MLXArray {
        var h = x
        if let convIn { h = convIn(h, causal: causal) }
        if let upsamplers { h = upsamplers[0](h, causal: causal) }
        for resnet in resnets { h = resnet(h, causal: causal) }
        return h
    }
}

/// The resnets of the decoder's mid block.
final class NFKLTX2VideoMidBlock: Module {
    @ModuleInfo(key: "resnets") var resnets: [NFKLTX2VideoResnet]

    init(_ channels: Int, layers: Int, eps: Float, reflects: Bool) {
        _resnets.wrappedValue = (0 ..< layers).map { _ in
            NFKLTX2VideoResnet(channels, channels, eps: eps, reflects: reflects)
        }
    }

    func callAsFunction(_ x: MLXArray, causal: Bool) -> MLXArray {
        resnets.reduce(x) { $1($0, causal: causal) }
    }
}

/// The LTX-2 video decoder: a latent `[B, F, H, W, latentChannels]` → a video
/// `[B, 8 (F − 1) + 1, 32 H, 32 W, 3]` in about `−1…1`.
public final class NFKMLXLTX2VideoDecoderNet: Module {
    @ModuleInfo(key: "conv_in") var convIn: NFKLTX2CausalConv3d
    @ModuleInfo(key: "mid_block") var midBlock: NFKLTX2VideoMidBlock
    @ModuleInfo(key: "up_blocks") var upBlocks: [NFKLTX2VideoUpBlock]
    @ModuleInfo(key: "conv_out") var convOut: NFKLTX2CausalConv3d

    public let configuration: NFKMLXLTX2VideoVAEConfiguration

    public init(_ c: NFKMLXLTX2VideoVAEConfiguration) {
        configuration = c
        let widths = Array(c.blockOutChannels.reversed())
        let layers = Array(c.layersPerBlock.reversed())
        let scaling = Array(c.spatioTemporalScaling.reversed())
        let residual = Array(c.upsampleResidual.reversed())
        let factors = Array(c.upsampleFactors.reversed())
        let reflects = c.reflectsSpatialPadding
        var output = widths[0]
        _convIn.wrappedValue = NFKLTX2CausalConv3d(c.latentChannels, output, reflects: reflects)
        _midBlock.wrappedValue = NFKLTX2VideoMidBlock(output, layers: layers[0], eps: c.resnetEps, reflects: reflects)
        var blocks = [NFKLTX2VideoUpBlock]()
        for index in widths.indices {
            let input = output / factors[index]
            output = widths[index] / factors[index]
            blocks.append(NFKLTX2VideoUpBlock(
                inChannels: input, outChannels: output, layers: layers[index + 1], scales: scaling[index],
                upsampleType: c.upsampleTypes[index], residual: residual[index], factor: factors[index],
                eps: c.resnetEps, reflects: reflects))
        }
        _upBlocks.wrappedValue = blocks
        _convOut.wrappedValue = NFKLTX2CausalConv3d(output, c.outChannels * c.patchSize * c.patchSize,
                                                    reflects: reflects)
    }

    /// The decoder's input through its output norm, the seam the reference's `norm_out` records.
    func features(_ latent: MLXArray) -> MLXArray {
        let causal = configuration.isCausal
        var h = midBlock(convIn(latent, causal: causal), causal: causal)
        for block in upBlocks { h = block(h, causal: causal) }
        return ltx2PixelNorm(h, eps: 1e-8)
    }

    public func callAsFunction(_ latent: MLXArray) -> MLXArray {
        let h = convOut(NFKReferenceRounding.silu(features(latent)), causal: configuration.isCausal)
        // The patch unshuffle: the reference's `reshape(B, −1, pt, p, p, F, H, W)` pairs the LAST patch
        // axis with height and the one before it with width.
        let p = configuration.patchSize, pt = configuration.patchSizeT
        let (b, f, height, width) = (h.dim(0), h.dim(1), h.dim(2), h.dim(3))
        let c = h.dim(4) / (pt * p * p)
        return h.reshaped([b, f, height, width, c, pt, p, p])
            .transposed(0, 1, 5, 2, 7, 3, 6, 4)
            .reshaped([b, f * pt, height * p, width * p, c])
    }
}

/// Reading, building, and loading the LTX-2 video decoder.
public enum NFKMLXLTX2VideoVAE {

    /// The decoder geometry and latent statistics a `vae/config.json` describes. Refuses by name what this
    /// decoder does not build: a timestep-conditioned decode and noise injection.
    public static func configuration(fromHuggingFace url: URL) throws -> NFKMLXLTX2VideoVAEConfiguration {
        guard let json = try JSONSerialization.jsonObject(with: try Data(contentsOf: url)) as? [String: Any] else {
            throw NFKMLXError.unsupportedConfiguration("\(url.lastPathComponent) is not a JSON object")
        }
        if let kind = json["_class_name"] as? String, kind != "AutoencoderKLLTX2Video" {
            throw NFKMLXError.unsupportedConfiguration("this reads an LTX-2 video autoencoder, not \(kind)")
        }
        if (json["timestep_conditioning"] as? NSNumber)?.boolValue == true {
            throw NFKMLXError.unsupportedConfiguration(
                "this decoder does not build the timestep-conditioned decode the release asks for")
        }
        if let noise = json["decoder_inject_noise"] as? [NSNumber], noise.contains(where: \.boolValue) {
            throw NFKMLXError.unsupportedConfiguration(
                "this decoder does not build the noise injection the release asks for")
        }
        let base = NFKMLXLTX2VideoVAEConfiguration.ltx23
        func integers(_ key: String, _ fallback: [Int]) -> [Int] {
            (json[key] as? [NSNumber])?.map(\.intValue) ?? fallback
        }
        func flags(_ key: String, _ fallback: [Bool], count: Int) -> [Bool] {
            if let single = json[key] as? Bool { return [Bool](repeating: single, count: count) }
            return (json[key] as? [NSNumber])?.map(\.boolValue) ?? fallback
        }
        let widths = integers("decoder_block_out_channels", base.blockOutChannels)
        let padding = json["decoder_spatial_padding_mode"] as? String ?? "zeros"
        guard padding == "zeros" || padding == "reflect" else {
            throw NFKMLXError.unsupportedConfiguration(
                "this decoder pads space with zeros or by reflection; the release asks for \(padding)")
        }
        return NFKMLXLTX2VideoVAEConfiguration(
            outChannels: (json["out_channels"] as? NSNumber)?.intValue ?? 3,
            latentChannels: (json["latent_channels"] as? NSNumber)?.intValue ?? 128,
            blockOutChannels: widths,
            layersPerBlock: integers("decoder_layers_per_block", base.layersPerBlock),
            spatioTemporalScaling: flags("decoder_spatio_temporal_scaling", base.spatioTemporalScaling,
                                         count: widths.count),
            upsampleTypes: json["upsample_type"] as? [String] ?? base.upsampleTypes,
            upsampleResidual: flags("upsample_residual", base.upsampleResidual, count: widths.count),
            upsampleFactors: integers("upsample_factor", base.upsampleFactors),
            patchSize: (json["patch_size"] as? NSNumber)?.intValue ?? 4,
            patchSizeT: (json["patch_size_t"] as? NSNumber)?.intValue ?? 1,
            resnetEps: (json["resnet_norm_eps"] as? NSNumber)?.floatValue ?? 1e-6,
            isCausal: (json["decoder_causal"] as? NSNumber)?.boolValue ?? false,
            reflectsSpatialPadding: padding == "reflect",
            spatialCompression: (json["spatial_compression_ratio"] as? NSNumber)?.intValue ?? 32,
            temporalCompression: (json["temporal_compression_ratio"] as? NSNumber)?.intValue ?? 8)
    }

    /// Loads a release's `vae/` directory into `decoder`: the `decoder.` tensors, 5-D convolutions
    /// transposed to MLX's `[out, kT, kH, kW, in]`; the encoder is not read. Returns the latents'
    /// per-channel mean and standard deviation, which the pipeline denormalizes by before decoding.
    @discardableResult
    public static func loadWeights(into decoder: NFKMLXLTX2VideoDecoderNet, from directory: URL,
                                   precision: NFKMLXWeightPrecision = .float32) throws
        -> (mean: MLXArray, std: MLXArray) {
        let arrays = try NFKMLXReleaseWeights.materializedArrays(
            inDirectory: directory, precision: precision,
            remap: { $0.hasPrefix("decoder.") || $0.hasPrefix("latents_") ? $0 : nil },
            transform: { key, value in (key, value.ndim == 5 ? value.transposed(0, 2, 3, 4, 1) : value) })
        return try apply(arrays, to: decoder)
    }

    /// Applies `decoder.`-prefixed tensors (already in MLX's layout) and returns the latent statistics.
    @discardableResult
    static func apply(_ arrays: [(String, MLXArray)], to decoder: NFKMLXLTX2VideoDecoderNet) throws
        -> (mean: MLXArray, std: MLXArray) {
        var statistics = [String: MLXArray]()
        var mapped = [(String, MLXArray)]()
        for (key, value) in arrays {
            if key == "latents_mean" || key == "latents_std" {
                statistics[key] = value.asType(.float32)
            } else if key.hasPrefix("decoder.") {
                mapped.append((String(key.dropFirst("decoder.".count)), value))
            }
        }
        try NFKMLXWeights.apply(mapped, to: decoder)
        let channels = decoder.configuration.latentChannels
        return (statistics["latents_mean"] ?? MLXArray.zeros([channels]),
                statistics["latents_std"] ?? MLXArray.ones([channels]))
    }
}
