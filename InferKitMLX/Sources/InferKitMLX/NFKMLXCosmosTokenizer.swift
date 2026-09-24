import Foundation
import MLX
import MLXFast
import MLXNN

/*!
 @abstract Geometry of a Cosmos Tokenizer (`nvidia/Cosmos-0.1-Tokenizer-*`).
 @discussion NVIDIA's Cosmos Tokenizer compresses an image or a video into a continuous latent or a
 grid of discrete tokens and reconstructs it again. Every variant shares one design: a Haar wavelet
 patcher, a convolutional encoder, a 1×1 projection to the latent, and the mirror-image decoder and
 inverse wavelet. The image variants are 2D (group-normalized residual blocks, one self-attention in
 the middle). The video variants are causal in time: every convolution is factorized into a spatial
 `1×3×3` and a temporal `3×1×1` kernel, padded only toward the past, and the middle block attends
 over space within a frame and causally over time at each position, so frame `t` never depends on a
 later frame. The discrete variants quantize the latent with finite scalar quantization (six scalars
 at levels 8, 8, 8, 5, 5, 5, a 64,000-entry implicit codebook, no learned embedding).

 The releases ship an empty `config.json`, so the geometry is fixed per variant here, as NVIDIA's
 own code fixes it from the variant's name, and checked against the release's tensors at load.
 Introduced in InferKit 0.4.0.
 */
public struct NFKMLXCosmosTokenizerConfiguration: Sendable, Equatable {
    /// Whether the tokenizer is the causal video network (true) or the image network.
    public var isVideo: Bool
    /// Whether the latent is quantized to discrete tokens (FSQ) rather than kept continuous.
    public var isDiscrete: Bool
    public var inChannels: Int = 3
    public var channels: Int = 128
    /// Each encoder level's width as a multiple of `channels`.
    public var encoderChannelMultipliers: [Int] = [2, 4, 4]
    /// Each decoder level's width as a multiple of `channels`, indexed by level as the encoder's are.
    public var decoderChannelMultipliers: [Int] = [2, 4, 4]
    public var residualBlocks: Int = 2
    /// The feature resolutions that carry a self-attention block, measured against `resolution`.
    public var attentionResolutions: [Int] = [32]
    public var resolution: Int = 1024
    /// The wavelet patch size. Each factor of two is one Haar level.
    public var patchSize: Int = 4
    public var spatialCompression: Int
    /// The temporal compression. 1 for an image tokenizer.
    public var temporalCompression: Int = 1
    /// The encoder's output width and the decoder's input width.
    public var zChannels: Int = 16
    /// The latent's width: the continuous latent's channels, or the FSQ embedding's (one per level).
    public var latentChannels: Int = 16
    /// The FSQ levels for a discrete tokenizer.
    public var levels: [Int] = [8, 8, 8, 5, 5, 5]

    public init(isVideo: Bool, isDiscrete: Bool, spatialCompression: Int, temporalCompression: Int = 1) {
        self.isVideo = isVideo
        self.isDiscrete = isDiscrete
        self.spatialCompression = spatialCompression
        self.temporalCompression = temporalCompression
        if isDiscrete {
            latentChannels = levels.count
        }
    }

    /// The number of distinct discrete tokens, the product of the FSQ levels.
    public var codebookSize: Int { levels.reduce(1, *) }

    var patchLevels: Int { Int(log2(Double(patchSize))) }
    var spatialResamples: Int { Int(log2(Double(spatialCompression))) - patchLevels }
    var temporalResamples: Int { Int(log2(Double(temporalCompression))) - patchLevels }
}

// MARK: - Shared pieces

private func cosmosSwish(_ x: MLXArray) -> MLXArray { x * sigmoid(x) }

/// The Haar analysis and synthesis steps along one axis. The forward filter pair is
/// `[1/√2, 1/√2]` and `[1/√2, -1/√2]` at stride two; the synthesis is its transpose.
enum NFKCosmosHaar {
    static let tap = Float(0.7071067811865476)

    /// Splits `x` into its low and high bands along `axis`, which must have even length.
    static func split(_ x: MLXArray, axis: Int) -> (MLXArray, MLXArray) {
        var shape = x.shape
        let length = shape[axis]
        shape[axis] = length / 2
        shape.insert(2, at: axis + 1)
        let pairs = x.reshaped(shape)
        let even = pairs.take(MLXArray(Int32(0)), axis: axis + 1)
        let odd = pairs.take(MLXArray(Int32(1)), axis: axis + 1)
        return (tap * (even + odd), tap * (even - odd))
    }

    /// Interleaves a low and a high band back into one signal twice as long along `axis`.
    static func merge(_ low: MLXArray, _ high: MLXArray, axis: Int) -> MLXArray {
        let even = tap * (low + high)
        let odd = tap * (low - high)
        var shape = low.shape
        shape[axis] *= 2
        return stacked([even, odd], axis: axis + 1).reshaped(shape)
    }
}

// MARK: - The image network (2D)

/// The image tokenizer's wavelet patcher: `log2(patch)` Haar levels over `[B, H, W, C]`, each splitting
/// along the width then the height and concatenating the four bands on channels in the order
/// `LL, LH, HL, HH`, with every band averaged (divided by two).
enum NFKCosmosPatcher2D {
    static func patch(_ x: MLXArray, levels: Int) -> MLXArray {
        var x = x
        for _ in 0 ..< levels {
            let (lowW, highW) = NFKCosmosHaar.split(x, axis: 2)
            let (ll, lh) = NFKCosmosHaar.split(lowW, axis: 1)
            let (hl, hh) = NFKCosmosHaar.split(highW, axis: 1)
            x = concatenated([ll, lh, hl, hh], axis: -1) / 2
        }
        return x
    }

    static func unpatch(_ x: MLXArray, levels: Int) -> MLXArray {
        var x = x
        for _ in 0 ..< levels {
            let bands = split(x, parts: 4, axis: -1)
            let low = NFKCosmosHaar.merge(bands[0], bands[1], axis: 1)
            let high = NFKCosmosHaar.merge(bands[2], bands[3], axis: 1)
            x = NFKCosmosHaar.merge(low, high, axis: 2) * 2
        }
        return x
    }
}

/// A residual block: group norm, SiLU, 3×3 convolution, twice, with a 1×1 shortcut where the width
/// changes.
final class NFKCosmosResnet2D: Module {
    @ModuleInfo(key: "norm1") var norm1: GroupNorm
    @ModuleInfo(key: "conv1") var conv1: Conv2d
    @ModuleInfo(key: "norm2") var norm2: GroupNorm
    @ModuleInfo(key: "conv2") var conv2: Conv2d
    @ModuleInfo(key: "nin_shortcut") var shortcut: Conv2d?

    init(_ inChannels: Int, _ outChannels: Int) {
        _norm1.wrappedValue = GroupNorm(groupCount: 32, dimensions: inChannels, eps: 1e-6, pytorchCompatible: true)
        _conv1.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: 3, padding: 1)
        _norm2.wrappedValue = GroupNorm(groupCount: 32, dimensions: outChannels, eps: 1e-6, pytorchCompatible: true)
        _conv2.wrappedValue = Conv2d(inputChannels: outChannels, outputChannels: outChannels, kernelSize: 3, padding: 1)
        if inChannels != outChannels {
            _shortcut.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: 1)
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = conv1(cosmosSwish(norm1(x)))
        h = conv2(cosmosSwish(norm2(h)))
        return (shortcut.map { $0(x) } ?? x) + h
    }
}

/// Single-head self-attention over a frame's positions, through 1×1 convolutions, added back.
final class NFKCosmosAttention2D: Module {
    @ModuleInfo(key: "norm") var norm: GroupNorm
    @ModuleInfo(key: "q") var q: Conv2d
    @ModuleInfo(key: "k") var k: Conv2d
    @ModuleInfo(key: "v") var v: Conv2d
    @ModuleInfo(key: "proj_out") var projOut: Conv2d

    init(_ channels: Int) {
        _norm.wrappedValue = GroupNorm(groupCount: 32, dimensions: channels, eps: 1e-6, pytorchCompatible: true)
        _q.wrappedValue = Conv2d(inputChannels: channels, outputChannels: channels, kernelSize: 1)
        _k.wrappedValue = Conv2d(inputChannels: channels, outputChannels: channels, kernelSize: 1)
        _v.wrappedValue = Conv2d(inputChannels: channels, outputChannels: channels, kernelSize: 1)
        _projOut.wrappedValue = Conv2d(inputChannels: channels, outputChannels: channels, kernelSize: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, h, w, c) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        let normed = norm(x)
        func heads(_ t: MLXArray) -> MLXArray { t.reshaped([b, 1, h * w, c]) }
        let attended = MLXFast.scaledDotProductAttention(
            queries: heads(q(normed)), keys: heads(k(normed)), values: heads(v(normed)),
            scale: 1 / Float(c).squareRoot(), mask: .none)
        return x + projOut(attended.reshaped([b, h, w, c]))
    }
}

/// A stride-2 3×3 convolution after one zero row and column on the bottom and right.
final class NFKCosmosDownsample2D: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d

    init(_ channels: Int) {
        _conv.wrappedValue = Conv2d(inputChannels: channels, outputChannels: channels, kernelSize: 3, stride: 2)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        conv(padded(x, widths: [IntOrPair((0, 0)), IntOrPair((0, 1)), IntOrPair((0, 1)), IntOrPair((0, 0))]))
    }
}

/// A nearest-neighbor 2× enlargement followed by a 3×3 convolution.
final class NFKCosmosUpsample2D: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d

    init(_ channels: Int) {
        _conv.wrappedValue = Conv2d(inputChannels: channels, outputChannels: channels, kernelSize: 3, padding: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        conv(repeated(repeated(x, count: 2, axis: 1), count: 2, axis: 2))
    }
}

/// One resolution level: residual blocks, the attention after each block where the level has it, and
/// the resampler that ends the level.
final class NFKCosmosLevel2D: Module {
    @ModuleInfo(key: "block") var block: [NFKCosmosResnet2D]
    @ModuleInfo(key: "attn") var attn: [NFKCosmosAttention2D]
    @ModuleInfo(key: "downsample") var downsample: NFKCosmosDownsample2D?
    @ModuleInfo(key: "upsample") var upsample: NFKCosmosUpsample2D?

    init(block: [NFKCosmosResnet2D], attn: [NFKCosmosAttention2D],
         downsample: NFKCosmosDownsample2D? = nil, upsample: NFKCosmosUpsample2D? = nil) {
        _block.wrappedValue = block
        _attn.wrappedValue = attn
        _downsample.wrappedValue = downsample
        _upsample.wrappedValue = upsample
    }

    func blocks(_ x: MLXArray) -> MLXArray {
        var h = x
        for (index, resnet) in block.enumerated() {
            h = resnet(h)
            if !attn.isEmpty {
                h = attn[index](h)
            }
        }
        return h
    }
}

/// The middle of either half: a residual block, self-attention, a residual block.
final class NFKCosmosMiddle2D: Module {
    @ModuleInfo(key: "block_1") var block1: NFKCosmosResnet2D
    @ModuleInfo(key: "attn_1") var attn1: NFKCosmosAttention2D
    @ModuleInfo(key: "block_2") var block2: NFKCosmosResnet2D

    init(_ channels: Int) {
        _block1.wrappedValue = NFKCosmosResnet2D(channels, channels)
        _attn1.wrappedValue = NFKCosmosAttention2D(channels)
        _block2.wrappedValue = NFKCosmosResnet2D(channels, channels)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { block2(attn1(block1(x))) }
}

final class NFKCosmosEncoder2D: Module {
    @ModuleInfo(key: "conv_in") var convIn: Conv2d
    @ModuleInfo(key: "down") var down: [NFKCosmosLevel2D]
    @ModuleInfo(key: "mid") var mid: NFKCosmosMiddle2D
    @ModuleInfo(key: "norm_out") var normOut: GroupNorm
    @ModuleInfo(key: "conv_out") var convOut: Conv2d
    let patchLevels: Int

    init(_ c: NFKMLXCosmosTokenizerConfiguration) {
        patchLevels = c.patchLevels
        let patched = c.inChannels * c.patchSize * c.patchSize
        _convIn.wrappedValue = Conv2d(inputChannels: patched, outputChannels: c.channels, kernelSize: 3, padding: 1)
        let multipliers = c.encoderChannelMultipliers
        var resolution = c.resolution / c.patchSize
        var blockIn = c.channels
        var levels: [NFKCosmosLevel2D] = []
        for (level, multiplier) in multipliers.enumerated() {
            let blockOut = c.channels * multiplier
            var blocks: [NFKCosmosResnet2D] = []
            var attentions: [NFKCosmosAttention2D] = []
            for _ in 0 ..< c.residualBlocks {
                blocks.append(NFKCosmosResnet2D(blockIn, blockOut))
                blockIn = blockOut
                if c.attentionResolutions.contains(resolution) {
                    attentions.append(NFKCosmosAttention2D(blockIn))
                }
            }
            var downsample: NFKCosmosDownsample2D?
            if level < c.spatialResamples {
                downsample = NFKCosmosDownsample2D(blockIn)
                resolution /= 2
            }
            levels.append(NFKCosmosLevel2D(block: blocks, attn: attentions, downsample: downsample))
        }
        _down.wrappedValue = levels
        _mid.wrappedValue = NFKCosmosMiddle2D(blockIn)
        _normOut.wrappedValue = GroupNorm(groupCount: 32, dimensions: blockIn, eps: 1e-6, pytorchCompatible: true)
        _convOut.wrappedValue = Conv2d(inputChannels: blockIn, outputChannels: c.zChannels, kernelSize: 3, padding: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = convIn(NFKCosmosPatcher2D.patch(x, levels: patchLevels))
        for level in down {
            h = level.blocks(h)
            if let downsample = level.downsample {
                h = downsample(h)
            }
        }
        return convOut(cosmosSwish(normOut(mid(h))))
    }
}

final class NFKCosmosDecoder2D: Module {
    @ModuleInfo(key: "conv_in") var convIn: Conv2d
    @ModuleInfo(key: "mid") var mid: NFKCosmosMiddle2D
    @ModuleInfo(key: "up") var up: [NFKCosmosLevel2D]
    @ModuleInfo(key: "norm_out") var normOut: GroupNorm
    @ModuleInfo(key: "conv_out") var convOut: Conv2d
    let patchLevels: Int

    init(_ c: NFKMLXCosmosTokenizerConfiguration) {
        patchLevels = c.patchLevels
        let multipliers = c.decoderChannelMultipliers
        let count = multipliers.count
        var blockIn = c.channels * multipliers[count - 1]
        var resolution = (c.resolution / c.patchSize) >> (count - 1)
        _convIn.wrappedValue = Conv2d(inputChannels: c.zChannels, outputChannels: blockIn, kernelSize: 3, padding: 1)
        _mid.wrappedValue = NFKCosmosMiddle2D(blockIn)
        var levels = [NFKCosmosLevel2D?](repeating: nil, count: count)
        for level in (0 ..< count).reversed() {
            let blockOut = c.channels * multipliers[level]
            var blocks: [NFKCosmosResnet2D] = []
            var attentions: [NFKCosmosAttention2D] = []
            for _ in 0 ... c.residualBlocks {
                blocks.append(NFKCosmosResnet2D(blockIn, blockOut))
                blockIn = blockOut
                if c.attentionResolutions.contains(resolution) {
                    attentions.append(NFKCosmosAttention2D(blockIn))
                }
            }
            var upsample: NFKCosmosUpsample2D?
            if level >= count - c.spatialResamples {
                upsample = NFKCosmosUpsample2D(blockIn)
                resolution *= 2
            }
            levels[level] = NFKCosmosLevel2D(block: blocks, attn: attentions, upsample: upsample)
        }
        _up.wrappedValue = levels.compactMap { $0 }
        _normOut.wrappedValue = GroupNorm(groupCount: 32, dimensions: blockIn, eps: 1e-6, pytorchCompatible: true)
        let patched = c.inChannels * c.patchSize * c.patchSize
        _convOut.wrappedValue = Conv2d(inputChannels: blockIn, outputChannels: patched, kernelSize: 3, padding: 1)
    }

    /// The decoder before the inverse wavelet, `[B, h, w, 3·patch²]`.
    func features(_ z: MLXArray) -> MLXArray {
        var h = mid(convIn(z))
        for level in up.reversed() {
            h = level.blocks(h)
            if let upsample = level.upsample {
                h = upsample(h)
            }
        }
        return convOut(cosmosSwish(normOut(h)))
    }

    func callAsFunction(_ z: MLXArray) -> MLXArray {
        NFKCosmosPatcher2D.unpatch(features(z), levels: patchLevels)
    }
}

// MARK: - The video network (causal 3D)

/// The video tokenizer's wavelet patcher over `[B, T, H, W, C]`. The first frame is repeated `patch`
/// times so the clip's first frame fills a whole temporal patch, then each Haar level splits time,
/// height, and width in that order into eight bands (`LLL … HHH`), averaged by `2√2`. The inverse
/// drops the `patch - 1` repeated frames again.
enum NFKCosmosPatcher3D {
    static let rescale = 2 * Float(2).squareRoot()

    static func patch(_ x: MLXArray, patchSize: Int, levels: Int) -> MLXArray {
        let first = x[0..., 0 ..< 1]
        var x = concatenated([repeated(first, count: patchSize, axis: 1), x[0..., 1...]], axis: 1)
        for _ in 0 ..< levels {
            let (low, high) = NFKCosmosHaar.split(x, axis: 1)
            let (ll, lh) = NFKCosmosHaar.split(low, axis: 2)
            let (hl, hh) = NFKCosmosHaar.split(high, axis: 2)
            var bands: [MLXArray] = []
            for band in [ll, lh, hl, hh] {
                let (bandLow, bandHigh) = NFKCosmosHaar.split(band, axis: 3)
                bands.append(bandLow)
                bands.append(bandHigh)
            }
            x = concatenated(bands, axis: -1) / rescale
        }
        return x
    }

    static func unpatch(_ x: MLXArray, patchSize: Int, levels: Int) -> MLXArray {
        var x = x
        for _ in 0 ..< levels {
            let bands = split(x, parts: 8, axis: -1)
            let ll = NFKCosmosHaar.merge(bands[0], bands[1], axis: 3)
            let lh = NFKCosmosHaar.merge(bands[2], bands[3], axis: 3)
            let hl = NFKCosmosHaar.merge(bands[4], bands[5], axis: 3)
            let hh = NFKCosmosHaar.merge(bands[6], bands[7], axis: 3)
            let low = NFKCosmosHaar.merge(ll, lh, axis: 2)
            let high = NFKCosmosHaar.merge(hl, hh, axis: 2)
            x = NFKCosmosHaar.merge(low, high, axis: 1) * rescale
        }
        return x[0..., (patchSize - 1)...]
    }
}

/// A causal 3D convolution over `[B, T, H, W, C]`: the first frame is repeated
/// `(kT - 1) + (1 - strideT)` times in front, the spatial axes are zero-padded on both sides, and the
/// convolution itself pads nothing.
final class NFKCosmosCausalConv3D: Module, UnaryLayer {
    @ModuleInfo(key: "conv3d") var conv3d: Conv3d
    let timePad: Int
    let spatialPad: Int

    init(_ inChannels: Int, _ outChannels: Int, kernel: (Int, Int, Int), stride: Int = 1,
         timeStride: Int = 1, padding: Int = 0) {
        timePad = (kernel.0 - 1) + (1 - timeStride)
        spatialPad = padding
        _conv3d.wrappedValue = Conv3d(inputChannels: inChannels, outputChannels: outChannels,
                                      kernelSize: IntOrTriple(kernel),
                                      stride: IntOrTriple((timeStride, stride, stride)))
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x
        if timePad > 0 {
            x = concatenated([repeated(x[0..., 0 ..< 1], count: timePad, axis: 1), x], axis: 1)
        }
        if spatialPad > 0 {
            x = padded(x, widths: [IntOrPair((0, 0)), IntOrPair((0, 0)), IntOrPair((spatialPad, spatialPad)),
                                   IntOrPair((spatialPad, spatialPad)), IntOrPair((0, 0))])
        }
        return conv3d(x)
    }
}

/// A factorized causal convolution: a spatial `1×3×3` then a temporal `3×1×1`, as NVIDIA's
/// `nn.Sequential` pair (`.0`, `.1`).
private func factorizedConvolution(_ inChannels: Int, _ outChannels: Int) -> [NFKCosmosCausalConv3D] {
    [NFKCosmosCausalConv3D(inChannels, outChannels, kernel: (1, 3, 3), padding: 1),
     NFKCosmosCausalConv3D(outChannels, outChannels, kernel: (3, 1, 1))]
}

private func applied(_ layers: [NFKCosmosCausalConv3D], _ x: MLXArray) -> MLXArray {
    layers.reduce(x) { $1($0) }
}

/// A one-group group norm applied frame by frame (a layer norm over each frame's positions and
/// channels), which keeps normalization causal.
final class NFKCosmosCausalNorm: Module {
    @ModuleInfo(key: "norm") var norm: GroupNorm

    init(_ channels: Int) {
        _norm.wrappedValue = GroupNorm(groupCount: 1, dimensions: channels, eps: 1e-6, pytorchCompatible: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let s = x.shape
        return norm(x.reshaped([s[0] * s[1], s[2], s[3], s[4]])).reshaped(s)
    }
}

final class NFKCosmosResnet3D: Module {
    @ModuleInfo(key: "norm1") var norm1: NFKCosmosCausalNorm
    @ModuleInfo(key: "conv1") var conv1: [NFKCosmosCausalConv3D]
    @ModuleInfo(key: "norm2") var norm2: NFKCosmosCausalNorm
    @ModuleInfo(key: "conv2") var conv2: [NFKCosmosCausalConv3D]
    @ModuleInfo(key: "nin_shortcut") var shortcut: NFKCosmosCausalConv3D?

    init(_ inChannels: Int, _ outChannels: Int) {
        _norm1.wrappedValue = NFKCosmosCausalNorm(inChannels)
        _conv1.wrappedValue = factorizedConvolution(inChannels, outChannels)
        _norm2.wrappedValue = NFKCosmosCausalNorm(outChannels)
        _conv2.wrappedValue = factorizedConvolution(outChannels, outChannels)
        if inChannels != outChannels {
            _shortcut.wrappedValue = NFKCosmosCausalConv3D(inChannels, outChannels, kernel: (1, 1, 1))
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = applied(conv1, cosmosSwish(norm1(x)))
        h = applied(conv2, cosmosSwish(norm2(h)))
        return (shortcut.map { $0(x) } ?? x) + h
    }
}

/// Single-head attention through pointwise causal convolutions. The spatial form attends over one
/// frame's positions; the temporal form attends over one position's frames, each frame seeing only
/// itself and earlier ones.
final class NFKCosmosAttention3D: Module {
    @ModuleInfo(key: "norm") var norm: NFKCosmosCausalNorm
    @ModuleInfo(key: "q") var q: NFKCosmosCausalConv3D
    @ModuleInfo(key: "k") var k: NFKCosmosCausalConv3D
    @ModuleInfo(key: "v") var v: NFKCosmosCausalConv3D
    @ModuleInfo(key: "proj_out") var projOut: NFKCosmosCausalConv3D
    let isTemporal: Bool

    init(_ channels: Int, temporal: Bool) {
        isTemporal = temporal
        _norm.wrappedValue = NFKCosmosCausalNorm(channels)
        _q.wrappedValue = NFKCosmosCausalConv3D(channels, channels, kernel: (1, 1, 1))
        _k.wrappedValue = NFKCosmosCausalConv3D(channels, channels, kernel: (1, 1, 1))
        _v.wrappedValue = NFKCosmosCausalConv3D(channels, channels, kernel: (1, 1, 1))
        _projOut.wrappedValue = NFKCosmosCausalConv3D(channels, channels, kernel: (1, 1, 1))
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, t, h, w, c) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4])
        let normed = norm(x)
        let scale = 1 / Float(c).squareRoot()
        let attended: MLXArray
        if isTemporal {
            func sequences(_ y: MLXArray) -> MLXArray {
                y.transposed(0, 2, 3, 1, 4).reshaped([b * h * w, 1, t, c])
            }
            attended = MLXFast.scaledDotProductAttention(
                queries: sequences(q(normed)), keys: sequences(k(normed)), values: sequences(v(normed)),
                scale: scale, mask: .causal)
                .reshaped([b, h, w, t, c]).transposed(0, 3, 1, 2, 4)
        } else {
            func frames(_ y: MLXArray) -> MLXArray { y.reshaped([b * t, 1, h * w, c]) }
            attended = MLXFast.scaledDotProductAttention(
                queries: frames(q(normed)), keys: frames(k(normed)), values: frames(v(normed)),
                scale: scale, mask: .none)
                .reshaped([b, t, h, w, c])
        }
        return x + projOut(attended)
    }
}

/// Spatial then temporal attention, as NVIDIA's `nn.Sequential` pair.
private func attentionPair(_ channels: Int) -> [NFKCosmosAttention3D] {
    [NFKCosmosAttention3D(channels, temporal: false), NFKCosmosAttention3D(channels, temporal: true)]
}

/// Halves space and/or time. Each reduction adds a strided convolution to an average pool of the same
/// window, and a pointwise convolution follows. A convolution the level does not use is absent from
/// the release, so it is absent here.
final class NFKCosmosDownsample3D: Module {
    @ModuleInfo(key: "conv1") var spatial: NFKCosmosCausalConv3D?
    @ModuleInfo(key: "conv2") var temporal: NFKCosmosCausalConv3D?
    @ModuleInfo(key: "conv3") var pointwise: NFKCosmosCausalConv3D

    init(_ channels: Int, spatial: Bool, temporal: Bool) {
        if spatial {
            _spatial.wrappedValue = NFKCosmosCausalConv3D(channels, channels, kernel: (1, 3, 3), stride: 2)
        }
        if temporal {
            _temporal.wrappedValue = NFKCosmosCausalConv3D(channels, channels, kernel: (3, 1, 1), timeStride: 2)
        }
        _pointwise.wrappedValue = NFKCosmosCausalConv3D(channels, channels, kernel: (1, 1, 1))
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x
        if let spatial {
            x = padded(x, widths: [IntOrPair((0, 0)), IntOrPair((0, 0)), IntOrPair((0, 1)), IntOrPair((0, 1)),
                                   IntOrPair((0, 0))])
            let s = x.shape
            let pooled = x[0..., 0..., 0 ..< (s[2] / 2 * 2), 0 ..< (s[3] / 2 * 2)]
                .reshaped([s[0], s[1], s[2] / 2, 2, s[3] / 2, 2, s[4]]).mean(axes: [3, 5])
            x = spatial(x) + pooled
        }
        if let temporal {
            x = concatenated([x[0..., 0 ..< 1], x], axis: 1)
            let s = x.shape
            let pooled = x[0..., 0 ..< (s[1] / 2 * 2)]
                .reshaped([s[0], s[1] / 2, 2, s[2], s[3], s[4]]).mean(axis: 2)
            x = temporal(x) + pooled
        }
        return pointwise(x)
    }
}

/// Doubles time and/or space, each by repetition refined by a residual convolution, then a pointwise
/// convolution. Doubling time keeps the clip causal: every frame is repeated and the leading copy of
/// the first dropped, so `t` frames become `2t - 1`.
final class NFKCosmosUpsample3D: Module {
    @ModuleInfo(key: "conv1") var temporal: NFKCosmosCausalConv3D?
    @ModuleInfo(key: "conv2") var spatial: NFKCosmosCausalConv3D?
    @ModuleInfo(key: "conv3") var pointwise: NFKCosmosCausalConv3D

    init(_ channels: Int, spatial: Bool, temporal: Bool) {
        if temporal {
            _temporal.wrappedValue = NFKCosmosCausalConv3D(channels, channels, kernel: (3, 1, 1))
        }
        if spatial {
            _spatial.wrappedValue = NFKCosmosCausalConv3D(channels, channels, kernel: (1, 3, 3), padding: 1)
        }
        _pointwise.wrappedValue = NFKCosmosCausalConv3D(channels, channels, kernel: (1, 1, 1))
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x
        if let temporal {
            if x.shape[1] > 1 {
                x = repeated(x, count: 2, axis: 1)[0..., 1...]
            }
            x = temporal(x) + x
        }
        if let spatial {
            x = repeated(repeated(x, count: 2, axis: 2), count: 2, axis: 3)
            x = spatial(x) + x
        }
        return pointwise(x)
    }
}

final class NFKCosmosLevel3D: Module {
    @ModuleInfo(key: "block") var block: [NFKCosmosResnet3D]
    @ModuleInfo(key: "attn") var attn: [[NFKCosmosAttention3D]]
    @ModuleInfo(key: "downsample") var downsample: NFKCosmosDownsample3D?
    @ModuleInfo(key: "upsample") var upsample: NFKCosmosUpsample3D?

    init(block: [NFKCosmosResnet3D], attn: [[NFKCosmosAttention3D]],
         downsample: NFKCosmosDownsample3D? = nil, upsample: NFKCosmosUpsample3D? = nil) {
        _block.wrappedValue = block
        _attn.wrappedValue = attn
        _downsample.wrappedValue = downsample
        _upsample.wrappedValue = upsample
    }

    func blocks(_ x: MLXArray) -> MLXArray {
        var h = x
        for (index, resnet) in block.enumerated() {
            h = resnet(h)
            if !attn.isEmpty {
                h = attn[index].reduce(h) { $1($0) }
            }
        }
        return h
    }
}

final class NFKCosmosMiddle3D: Module {
    @ModuleInfo(key: "block_1") var block1: NFKCosmosResnet3D
    @ModuleInfo(key: "attn_1") var attn1: [NFKCosmosAttention3D]
    @ModuleInfo(key: "block_2") var block2: NFKCosmosResnet3D

    init(_ channels: Int) {
        _block1.wrappedValue = NFKCosmosResnet3D(channels, channels)
        _attn1.wrappedValue = attentionPair(channels)
        _block2.wrappedValue = NFKCosmosResnet3D(channels, channels)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        block2(attn1.reduce(block1(x)) { $1($0) })
    }
}

final class NFKCosmosEncoder3D: Module {
    @ModuleInfo(key: "conv_in") var convIn: [NFKCosmosCausalConv3D]
    @ModuleInfo(key: "down") var down: [NFKCosmosLevel3D]
    @ModuleInfo(key: "mid") var mid: NFKCosmosMiddle3D
    @ModuleInfo(key: "norm_out") var normOut: NFKCosmosCausalNorm
    @ModuleInfo(key: "conv_out") var convOut: [NFKCosmosCausalConv3D]
    let patchSize: Int
    let patchLevels: Int

    init(_ c: NFKMLXCosmosTokenizerConfiguration) {
        patchSize = c.patchSize
        patchLevels = c.patchLevels
        let patched = c.inChannels * c.patchSize * c.patchSize * c.patchSize
        _convIn.wrappedValue = factorizedConvolution(patched, c.channels)
        let multipliers = c.encoderChannelMultipliers
        var resolution = c.resolution / c.patchSize
        var blockIn = c.channels
        var levels: [NFKCosmosLevel3D] = []
        for (level, multiplier) in multipliers.enumerated() {
            let blockOut = c.channels * multiplier
            var blocks: [NFKCosmosResnet3D] = []
            var attentions: [[NFKCosmosAttention3D]] = []
            for _ in 0 ..< c.residualBlocks {
                blocks.append(NFKCosmosResnet3D(blockIn, blockOut))
                blockIn = blockOut
                if c.attentionResolutions.contains(resolution) {
                    attentions.append(attentionPair(blockIn))
                }
            }
            var downsample: NFKCosmosDownsample3D?
            if level != multipliers.count - 1 {
                let spatial = level < c.spatialResamples
                let temporal = level < c.temporalResamples
                if spatial || temporal {
                    downsample = NFKCosmosDownsample3D(blockIn, spatial: spatial, temporal: temporal)
                }
                resolution /= 2
            }
            levels.append(NFKCosmosLevel3D(block: blocks, attn: attentions, downsample: downsample))
        }
        _down.wrappedValue = levels
        _mid.wrappedValue = NFKCosmosMiddle3D(blockIn)
        _normOut.wrappedValue = NFKCosmosCausalNorm(blockIn)
        _convOut.wrappedValue = factorizedConvolution(blockIn, c.zChannels)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = applied(convIn, NFKCosmosPatcher3D.patch(x, patchSize: patchSize, levels: patchLevels))
        for level in down {
            h = level.blocks(h)
            if let downsample = level.downsample {
                h = downsample(h)
            }
        }
        return applied(convOut, cosmosSwish(normOut(mid(h))))
    }
}

final class NFKCosmosDecoder3D: Module {
    @ModuleInfo(key: "conv_in") var convIn: [NFKCosmosCausalConv3D]
    @ModuleInfo(key: "mid") var mid: NFKCosmosMiddle3D
    @ModuleInfo(key: "up") var up: [NFKCosmosLevel3D]
    @ModuleInfo(key: "norm_out") var normOut: NFKCosmosCausalNorm
    @ModuleInfo(key: "conv_out") var convOut: [NFKCosmosCausalConv3D]
    let patchSize: Int
    let patchLevels: Int

    init(_ c: NFKMLXCosmosTokenizerConfiguration) {
        patchSize = c.patchSize
        patchLevels = c.patchLevels
        let multipliers = c.decoderChannelMultipliers
        let count = multipliers.count
        var blockIn = c.channels * multipliers[count - 1]
        var resolution = (c.resolution / c.patchSize) >> (count - 1)
        _convIn.wrappedValue = factorizedConvolution(c.zChannels, blockIn)
        _mid.wrappedValue = NFKCosmosMiddle3D(blockIn)
        var levels = [NFKCosmosLevel3D?](repeating: nil, count: count)
        for level in (0 ..< count).reversed() {
            let blockOut = c.channels * multipliers[level]
            var blocks: [NFKCosmosResnet3D] = []
            var attentions: [[NFKCosmosAttention3D]] = []
            for _ in 0 ... c.residualBlocks {
                blocks.append(NFKCosmosResnet3D(blockIn, blockOut))
                blockIn = blockOut
                if c.attentionResolutions.contains(resolution) {
                    attentions.append(attentionPair(blockIn))
                }
            }
            var upsample: NFKCosmosUpsample3D?
            if level != 0 {
                // The level that undoes encoder level `i` is `count - 1 - i`; the first upsampling
                // level (reversed index 0) is spatial only, which is how the releases were trained.
                let reversedIndex = count - level - 1
                let temporal = 0 < reversedIndex && reversedIndex < c.temporalResamples + 1
                let spatial = temporal
                    || (reversedIndex < c.spatialResamples && c.spatialResamples > c.temporalResamples)
                if spatial || temporal {
                    upsample = NFKCosmosUpsample3D(blockIn, spatial: spatial, temporal: temporal)
                }
                resolution *= 2
            }
            levels[level] = NFKCosmosLevel3D(block: blocks, attn: attentions, upsample: upsample)
        }
        _up.wrappedValue = levels.compactMap { $0 }
        _normOut.wrappedValue = NFKCosmosCausalNorm(blockIn)
        let patched = c.inChannels * c.patchSize * c.patchSize * c.patchSize
        _convOut.wrappedValue = factorizedConvolution(blockIn, patched)
    }

    /// The decoder before the inverse wavelet, `[B, t, h, w, 3·patch³]`.
    func features(_ z: MLXArray) -> MLXArray {
        var h = mid(applied(convIn, z))
        for level in up.reversed() {
            h = level.blocks(h)
            if let upsample = level.upsample {
                h = upsample(h)
            }
        }
        return applied(convOut, cosmosSwish(normOut(h)))
    }

    func callAsFunction(_ z: MLXArray) -> MLXArray {
        NFKCosmosPatcher3D.unpatch(features(z), patchSize: patchSize, levels: patchLevels)
    }
}

// MARK: - Finite scalar quantization

/*!
 @abstract The Cosmos discrete tokenizers' finite scalar quantizer.
 @discussion Each of the latent's channels is bounded by `tanh` to its level's range, rounded to an
 integer, and renormalized to `[-1, 1]`. A token index is the mixed-radix number the six rounded values
 spell, least significant channel first. There is no learned codebook, so the quantizer has no weights.
 Introduced in InferKit 0.4.0.
 */
public struct NFKMLXCosmosFSQ: Sendable {
    public let levels: [Int]

    public init(levels: [Int]) { self.levels = levels }

    private var levelArray: MLXArray { MLXArray(levels.map { Float($0) }) }
    private var halfWidth: MLXArray { MLXArray(levels.map { Float($0 / 2) }) }
    private var basis: MLXArray {
        var running = 1
        var result: [Int32] = []
        for level in levels {
            result.append(Int32(running))
            running *= level
        }
        return MLXArray(result)
    }

    /// Bounds a latent `[..., levels.count]` to each level's range, before rounding: `tanh`, scaled to
    /// half the level count (widened by 0.1%) and shifted so an even level count rounds to its own
    /// integers.
    public func bounded(_ z: MLXArray) -> MLXArray {
        let halfLevel = (levelArray - 1) * (1 + 1e-3) / 2
        let offset = MLXArray(levels.map { Float($0 % 2 == 0 ? 0.5 : 0) })
        let shift = atanh(offset / halfLevel)
        return tanh(z + shift) * halfLevel - offset
    }

    /// Bounds and rounds a latent `[..., levels.count]` to its codes in `[-1, 1]`. The rounding passes
    /// its gradient straight through, as the reference's does.
    public func codes(_ z: MLXArray) -> MLXArray {
        let bounded = bounded(z)
        let rounded = bounded + stopGradient(MLX.round(bounded) - bounded)
        return rounded / halfWidth
    }

    /// The token index of each code vector, `[...]` as int32.
    public func indices(codes: MLXArray) -> MLXArray {
        let digits = MLX.round(codes * halfWidth + halfWidth)
        return (digits * basis.asType(.float32)).sum(axis: -1).asType(.int32)
    }

    /// The code vectors `[..., levels.count]` for token indices.
    public func codes(indices: MLXArray) -> MLXArray {
        let expanded = expandedDimensions(indices.asType(.int32), axis: -1)
        let digits = floorDivide(expanded, basis) % MLXArray(levels.map { Int32($0) })
        return (digits.asType(.float32) - halfWidth) / halfWidth
    }
}

// MARK: - The network

/*!
 @abstract A Cosmos Tokenizer network, image or video, continuous or discrete.
 @discussion Channels-last throughout: an image is `[B, H, W, 3]` and a clip `[B, T, H, W, 3]`, in
 `[-1, 1]`. `encode` returns the latent (continuous variants) or the FSQ codes (discrete variants);
 `decode` takes either back to pixels. The module keys match NVIDIA's released TorchScript state dicts,
 so a release loads without a rename.
 Introduced in InferKit 0.4.0.
 */
public final class NFKMLXCosmosTokenizerNet: Module {
    public let configuration: NFKMLXCosmosTokenizerConfiguration
    @ModuleInfo(key: "encoder") var encoder: Module
    @ModuleInfo(key: "decoder") var decoder: Module
    @ModuleInfo(key: "quant_conv") var quantConv: UnaryLayer
    @ModuleInfo(key: "post_quant_conv") var postQuantConv: UnaryLayer

    public init(configuration c: NFKMLXCosmosTokenizerConfiguration) {
        configuration = c
        if c.isVideo {
            _encoder.wrappedValue = NFKCosmosEncoder3D(c)
            _decoder.wrappedValue = NFKCosmosDecoder3D(c)
            _quantConv.wrappedValue = NFKCosmosCausalConv3D(c.zChannels, c.latentChannels, kernel: (1, 1, 1))
            _postQuantConv.wrappedValue = NFKCosmosCausalConv3D(c.latentChannels, c.zChannels, kernel: (1, 1, 1))
        } else {
            _encoder.wrappedValue = NFKCosmosEncoder2D(c)
            _decoder.wrappedValue = NFKCosmosDecoder2D(c)
            _quantConv.wrappedValue = Conv2d(inputChannels: c.zChannels, outputChannels: c.latentChannels, kernelSize: 1)
            _postQuantConv.wrappedValue = Conv2d(inputChannels: c.latentChannels, outputChannels: c.zChannels, kernelSize: 1)
        }
    }

    /// The quantizer a discrete variant uses.
    public var quantizer: NFKMLXCosmosFSQ { NFKMLXCosmosFSQ(levels: configuration.levels) }

    /// The latent before quantization: the continuous latent, or a discrete variant's input to FSQ.
    func quantizerInput(_ pixels: MLXArray) -> MLXArray { quantConv(encoderOutput(pixels)) }

    /// The encoder's output before the latent projection.
    func encoderOutput(_ pixels: MLXArray) -> MLXArray {
        if let encoder = encoder as? NFKCosmosEncoder3D {
            return encoder(pixels)
        }
        return (encoder as! NFKCosmosEncoder2D)(pixels)
    }

    /// The decoder's output before the inverse wavelet.
    func decoderFeatures(_ latent: MLXArray) -> MLXArray {
        let z = postQuantConv(latent)
        if let decoder = decoder as? NFKCosmosDecoder3D {
            return decoder.features(z)
        }
        return (decoder as! NFKCosmosDecoder2D).features(z)
    }

    /// The continuous latent, or for a discrete variant the FSQ codes, of pixels in `[-1, 1]`.
    public func encode(_ pixels: MLXArray) -> MLXArray {
        let latent = quantConv(encoderOutput(pixels))
        return configuration.isDiscrete ? quantizer.codes(latent) : latent
    }

    /// Pixels in `[-1, 1]` from a continuous latent or a discrete variant's codes.
    public func decode(_ latent: MLXArray) -> MLXArray {
        let z = postQuantConv(latent)
        if let decoder = decoder as? NFKCosmosDecoder3D {
            return decoder(z)
        }
        return (decoder as! NFKCosmosDecoder2D)(z)
    }

    /// The discrete token indices of pixels in `[-1, 1]` (discrete variants).
    public func tokens(_ pixels: MLXArray) -> MLXArray {
        quantizer.indices(codes: encode(pixels))
    }

    /// Pixels in `[-1, 1]` from discrete token indices (discrete variants).
    public func decode(tokens: MLXArray) -> MLXArray {
        decode(quantizer.codes(indices: tokens))
    }

    /// The reconstruction of pixels in `[-1, 1]`: encode, then decode.
    public func callAsFunction(_ pixels: MLXArray) -> MLXArray { decode(encode(pixels)) }
}
