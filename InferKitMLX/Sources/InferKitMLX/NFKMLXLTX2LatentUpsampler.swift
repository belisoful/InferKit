//
//  NFKMLXLTX2LatentUpsampler.swift
//  InferKitMLX
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import InferKit
import MLX
import MLXNN

// The LTX-2 latent upsamplers (`LTX2LatentUpsamplerModel`, Lightricks): a 3-D convolutional network that
// doubles a denormalized video latent in space or in time. An input convolution, group norm and SiLU, four
// residual blocks, the upsampler, four more residual blocks, and an output convolution back to the latent
// width. The spatial upsampler convolves each frame to four times its width and shuffles the channels into
// a 2×2 neighbourhood (or, through the rational resampler, to `num²` times, shuffles by `num`, and blurs
// down by `den`); the temporal upsampler convolves in 3-D to twice its width, shuffles the channels into
// time, and drops the first frame, so `F` latent frames become `2 F − 1`.
//
// Tensors flow channels-last, `[batch, frames, height, width, channels]`.

/// The geometry of an LTX-2 latent upsampler, read from a release's `config.json`.
public struct NFKMLXLTX2LatentUpsamplerConfiguration: Sendable, Equatable {
    public var inChannels: Int
    public var midChannels: Int
    public var blocksPerStage: Int
    public var upsamplesSpace: Bool
    public var upsamplesTime: Bool
    /// The spatial scale the rational resampler applies (0.75, 1.5, 2, or 4); unread unless
    /// ``usesRationalResampler`` and the network upsamples space alone.
    public var rationalScale: Double
    public var usesRationalResampler: Bool

    public init(inChannels: Int = 128, midChannels: Int = 1024, blocksPerStage: Int = 4, upsamplesSpace: Bool = true,
                upsamplesTime: Bool = false, rationalScale: Double = 2, usesRationalResampler: Bool = false) {
        self.inChannels = inChannels
        self.midChannels = midChannels
        self.blocksPerStage = blocksPerStage
        self.upsamplesSpace = upsamplesSpace
        self.upsamplesTime = upsamplesTime
        self.rationalScale = rationalScale
        self.usesRationalResampler = usesRationalResampler
    }

    /// LTX-2.5's spatial ×2 upsampler (`latent_upsampler/`).
    public static let ltx25Spatial = NFKMLXLTX2LatentUpsamplerConfiguration()

    /// LTX-2.5's temporal ×2 upsampler (`temporal_latent_upsampler/`).
    public static let ltx25Temporal = NFKMLXLTX2LatentUpsamplerConfiguration(
        midChannels: 512, upsamplesSpace: false, upsamplesTime: true, usesRationalResampler: true)

    /// The numerator and denominator the rational resampler scales by.
    var rationalFactors: (num: Int, den: Int)? {
        [0.75: (3, 4), 1.5: (3, 2), 2.0: (2, 1), 4.0: (4, 1)][rationalScale]
    }

    /// How many times the network multiplies each spatial side and the frame count (`2 F − 1` for a
    /// temporal upsampler).
    public var spatialScale: Double {
        guard upsamplesSpace else { return 1 }
        if !upsamplesTime, usesRationalResampler { return rationalScale }
        return 2
    }
}

/// A residual block: convolution, group norm, SiLU, convolution, group norm, then SiLU of the sum.
final class NFKLTX2UpsamplerResBlock: Module {
    @ModuleInfo(key: "conv1") var conv1: Conv3d
    @ModuleInfo(key: "norm1") var norm1: GroupNorm
    @ModuleInfo(key: "conv2") var conv2: Conv3d
    @ModuleInfo(key: "norm2") var norm2: GroupNorm

    init(_ channels: Int) {
        _conv1.wrappedValue = Conv3d(inputChannels: channels, outputChannels: channels, kernelSize: 3, padding: 1)
        _norm1.wrappedValue = GroupNorm(groupCount: 32, dimensions: channels, pytorchCompatible: true)
        _conv2.wrappedValue = Conv3d(inputChannels: channels, outputChannels: channels, kernelSize: 3, padding: 1)
        _norm2.wrappedValue = GroupNorm(groupCount: 32, dimensions: channels, pytorchCompatible: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = norm2(conv2(NFKReferenceRounding.silu(norm1(conv1(x)))))
        return NFKReferenceRounding.silu(h + x)
    }
}

/// The rational resampler: a per-frame convolution to `num²` times the width, a `num × num` shuffle, and
/// an anti-aliased stride-`den` downsample through the normalized 5×5 binomial kernel.
final class NFKLTX2RationalResampler: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d
    let num: Int
    let den: Int

    init(_ channels: Int, num: Int, den: Int) {
        self.num = num
        self.den = den
        _conv.wrappedValue = Conv2d(inputChannels: channels, outputChannels: num * num * channels, kernelSize: 3,
                                    padding: 1)
    }

    /// `[N, H, W, C]` → `[N, H num / den, W num / den, C]`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let shuffled = ltx2SpatialShuffle(conv(x), factor: num)
        guard den > 1 else { return shuffled }
        let channels = shuffled.dim(-1)
        let row: [Float] = [1, 4, 6, 4, 1]
        let kernel = row.flatMap { a in row.map { a * $0 / 256 } }
        let weight = broadcast(MLXArray(kernel).reshaped([1, 5, 5, 1]), to: [channels, 5, 5, 1]).asType(shuffled.dtype)
        return conv2d(shuffled, weight, stride: IntOrPair(den), padding: IntOrPair(2), groups: channels)
    }
}

/// A per-frame `factor × factor` depth-to-space shuffle, the channel split `(channel, row, column)`:
/// `[N, H, W, C f²]` → `[N, H f, W f, C]`.
func ltx2SpatialShuffle(_ x: MLXArray, factor f: Int) -> MLXArray {
    let (n, h, w) = (x.dim(0), x.dim(1), x.dim(2))
    let c = x.dim(3) / (f * f)
    return x.reshaped([n, h, w, c, f, f]).transposed(0, 1, 4, 2, 5, 3).reshaped([n, h * f, w * f, c])
}

/// An LTX-2 latent upsampler: a denormalized latent `[B, F, H, W, inChannels]` → the upsampled latent.
public final class NFKMLXLTX2LatentUpsamplerNet: Module {
    @ModuleInfo(key: "initial_conv") var initialConv: Conv3d
    @ModuleInfo(key: "initial_norm") var initialNorm: GroupNorm
    @ModuleInfo(key: "res_blocks") var resBlocks: [NFKLTX2UpsamplerResBlock]
    /// The plain convolutional upsampler, the release's `upsampler.0` (a 2-D convolution for space alone,
    /// 3-D otherwise).
    @ModuleInfo(key: "upsampler") var upsampler: [Module]?
    @ModuleInfo(key: "post_upsample_res_blocks") var postBlocks: [NFKLTX2UpsamplerResBlock]
    @ModuleInfo(key: "final_conv") var finalConv: Conv3d

    public let configuration: NFKMLXLTX2LatentUpsamplerConfiguration
    let rational: Bool

    public init(_ c: NFKMLXLTX2LatentUpsamplerConfiguration) {
        configuration = c
        let mid = c.midChannels
        _initialConv.wrappedValue = Conv3d(inputChannels: c.inChannels, outputChannels: mid, kernelSize: 3, padding: 1)
        _initialNorm.wrappedValue = GroupNorm(groupCount: 32, dimensions: mid, pytorchCompatible: true)
        _resBlocks.wrappedValue = (0 ..< c.blocksPerStage).map { _ in NFKLTX2UpsamplerResBlock(mid) }
        rational = c.upsamplesSpace && !c.upsamplesTime && c.usesRationalResampler
        if rational {
            _upsampler.wrappedValue = nil
        } else if c.upsamplesSpace && c.upsamplesTime {
            _upsampler.wrappedValue = [Conv3d(inputChannels: mid, outputChannels: 8 * mid, kernelSize: 3, padding: 1)]
        } else if c.upsamplesTime {
            _upsampler.wrappedValue = [Conv3d(inputChannels: mid, outputChannels: 2 * mid, kernelSize: 3, padding: 1)]
        } else {
            _upsampler.wrappedValue = [Conv2d(inputChannels: mid, outputChannels: 4 * mid, kernelSize: 3, padding: 1)]
        }
        _postBlocks.wrappedValue = (0 ..< c.blocksPerStage).map { _ in NFKLTX2UpsamplerResBlock(mid) }
        _finalConv.wrappedValue = Conv3d(inputChannels: mid, outputChannels: c.inChannels, kernelSize: 3, padding: 1)
        _rationalResampler.wrappedValue = rational
            ? NFKLTX2RationalResampler(mid, num: c.rationalFactors?.num ?? 2, den: c.rationalFactors?.den ?? 1) : nil
    }

    /// The rational resampler, the release's `upsampler` where it scales space alone through one.
    @ModuleInfo(key: "rational_upsampler") var rationalResampler: NFKLTX2RationalResampler?

    /// The upsampler step alone: `[B, F, H, W, mid]` before it → after it.
    func upsample(_ x: MLXArray) -> MLXArray {
        let (b, f, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3), x.dim(4))
        let cfg = configuration
        if cfg.upsamplesTime {
            let y = (upsampler![0] as! UnaryLayer)(x)
            let shuffled: MLXArray
            if cfg.upsamplesSpace {
                shuffled = y.reshaped([b, f, h, w, c, 2, 2, 2]).transposed(0, 1, 5, 2, 6, 3, 7, 4)
                    .reshaped([b, f * 2, h * 2, w * 2, c])
            } else {
                shuffled = y.reshaped([b, f, h, w, c, 2]).transposed(0, 1, 5, 2, 3, 4).reshaped([b, f * 2, h, w, c])
            }
            return shuffled[0..., 1...]
        }
        let frames = x.reshaped([b * f, h, w, c])
        let up: MLXArray
        if let rationalResampler {
            up = rationalResampler(frames)
        } else {
            up = ltx2SpatialShuffle((upsampler![0] as! UnaryLayer)(frames), factor: 2)
        }
        return up.reshaped([b, f, up.dim(1), up.dim(2), c])
    }

    /// The pre-upsampler half: input convolution, norm, SiLU, and the first residual blocks.
    func head(_ latent: MLXArray) -> MLXArray {
        let h = NFKReferenceRounding.silu(initialNorm(initialConv(latent)))
        return resBlocks.reduce(h) { $1($0) }
    }

    public func callAsFunction(_ latent: MLXArray) -> MLXArray {
        finalConv(postBlocks.reduce(upsample(head(latent))) { $1($0) })
    }
}

/// Reading, building, and loading the LTX-2 latent upsamplers.
public enum NFKMLXLTX2LatentUpsamplers {

    /// The geometry a latent upsampler's `config.json` describes. Refuses by name the per-frame 2-D
    /// arrangement (`dims` 2), which no release ships.
    public static func configuration(fromHuggingFace url: URL) throws -> NFKMLXLTX2LatentUpsamplerConfiguration {
        guard let json = try JSONSerialization.jsonObject(with: try Data(contentsOf: url)) as? [String: Any] else {
            throw NFKMLXError.unsupportedConfiguration("\(url.lastPathComponent) is not a JSON object")
        }
        if let kind = json["_class_name"] as? String, kind != "LTX2LatentUpsamplerModel" {
            throw NFKMLXError.unsupportedConfiguration("this reads an LTX-2 latent upsampler, not \(kind)")
        }
        let dims = (json["dims"] as? NSNumber)?.intValue ?? 3
        guard dims == 3 else {
            throw NFKMLXError.unsupportedConfiguration("this builds the 3-D latent upsampler; the release asks for dims \(dims)")
        }
        let base = NFKMLXLTX2LatentUpsamplerConfiguration()
        let configuration = NFKMLXLTX2LatentUpsamplerConfiguration(
            inChannels: (json["in_channels"] as? NSNumber)?.intValue ?? base.inChannels,
            midChannels: (json["mid_channels"] as? NSNumber)?.intValue ?? base.midChannels,
            blocksPerStage: (json["num_blocks_per_stage"] as? NSNumber)?.intValue ?? base.blocksPerStage,
            upsamplesSpace: (json["spatial_upsample"] as? NSNumber)?.boolValue ?? true,
            upsamplesTime: (json["temporal_upsample"] as? NSNumber)?.boolValue ?? false,
            rationalScale: (json["rational_spatial_scale"] as? NSNumber)?.doubleValue ?? 2,
            usesRationalResampler: (json["use_rational_resampler"] as? NSNumber)?.boolValue ?? true)
        guard configuration.upsamplesSpace || configuration.upsamplesTime else {
            throw NFKMLXError.unsupportedConfiguration("a latent upsampler upsamples space, time, or both")
        }
        if configuration.upsamplesSpace, !configuration.upsamplesTime, configuration.usesRationalResampler,
           configuration.rationalFactors == nil {
            throw NFKMLXError.unsupportedConfiguration(
                "the rational resampler scales by 0.75, 1.5, 2, or 4, not \(configuration.rationalScale)")
        }
        return configuration
    }

    /// Release keys → the module's: the rational resampler's `upsampler.conv` is held as
    /// `rational_upsampler.conv`, and the blur kernel, a fixed buffer, is computed.
    static func moduleKey(_ key: String, rational: Bool) -> String? {
        if key == "upsampler.blur_down.kernel" { return nil }
        if rational, key.hasPrefix("upsampler.conv.") { return "rational_upsampler." + key.dropFirst("upsampler.".count) }
        return key
    }

    /// Release tensors in MLX's layout: 5-D convolutions `[out, kT, kH, kW, in]`, 4-D `[out, kH, kW, in]`.
    static func laid(_ value: MLXArray) -> MLXArray {
        switch value.ndim {
        case 5: return value.transposed(0, 2, 3, 4, 1)
        case 4: return value.transposed(0, 2, 3, 1)
        default: return value
        }
    }

    /// Applies release-named tensors already in MLX's layout.
    static func apply(_ arrays: [(String, MLXArray)], to net: NFKMLXLTX2LatentUpsamplerNet) throws {
        let mapped = arrays.compactMap { key, value in moduleKey(key, rational: net.rational).map { ($0, value) } }
        try NFKMLXWeights.apply(mapped, to: net, verifyShapes: true)
    }

    /// Loads a release's upsampler directory (`latent_upsampler/` or `temporal_latent_upsampler/`) at
    /// `precision`.
    public static func load(from directory: URL, precision: NFKMLXWeightPrecision = .float32) throws
        -> NFKMLXLTX2LatentUpsamplerNet {
        let net = NFKMLXLTX2LatentUpsamplerNet(
            try configuration(fromHuggingFace: directory.appendingPathComponent("config.json")))
        let arrays = try NFKMLXReleaseWeights.materializedArrays(
            inDirectory: directory, precision: precision, remap: { $0 },
            transform: { key, value in (key, laid(value)) })
        try apply(arrays, to: net)
        return net
    }
}
