//
//  NFKMLXCodeFormer.swift
//  InferKitMLX
//

import Foundation
import CoreGraphics
import InferKit
import MLX
import MLXNN

// CodeFormer restores a degraded face by treating restoration as code prediction over a learned
// discrete codebook. The reference VQGAN encoder (sczhou/CodeFormer `vqgan_arch.py`) maps the face to
// a 16×16 feature grid; a Transformer predicts, for each grid position, the index of a codebook
// entry; the generator decodes the looked-up entries back to a clean face. Predicting indices rather
// than nearest-neighbor quantization is what makes CodeFormer robust to heavy degradation. The
// controllable feature transformation (the fidelity weight `w`) fuses encoder features into the
// generator at four scales: 0 is full generative quality, 1 leans on the degraded input's detail.
// A caller aligns and crops a face to the model resolution before restoration. Tensors flow in NHWC.

/// CodeFormer dimensions. `base` is the released 512-face model; `tiny` keeps tests fast.
public struct NFKMLXCodeFormerConfiguration: Sendable {
    public var resolution: Int
    /// The encoder's base width (`nf`); each stage's width is `nf` times its multiplier.
    public var baseChannels: Int
    public var channelMultipliers: [Int]
    public var resBlocks: Int
    /// Spatial resolutions at which the coders attend.
    public var attentionResolutions: [Int]
    public var embeddingDimensions: Int
    public var codebookSize: Int
    public var transformerDimensions: Int
    public var codePredictorLayers: Int
    public var heads: Int
    public var normGroups: Int
    /// Resolutions at which the controllable feature transformation fuses encoder detail into the
    /// generator.
    public var connectResolutions: [Int]
    /// The fidelity weight `w`: 0 is full generative quality, 1 keeps the degraded input's detail.
    public var fidelity: Float

    public init(resolution: Int = 512, baseChannels: Int = 64, channelMultipliers: [Int] = [1, 2, 2, 4, 4, 8],
                resBlocks: Int = 2, attentionResolutions: [Int] = [16], embeddingDimensions: Int = 256,
                codebookSize: Int = 1024, transformerDimensions: Int = 512, codePredictorLayers: Int = 9,
                heads: Int = 8, normGroups: Int = 32, connectResolutions: [Int] = [32, 64, 128, 256],
                fidelity: Float = 0.5) {
        self.resolution = resolution
        self.baseChannels = baseChannels
        self.channelMultipliers = channelMultipliers
        self.resBlocks = resBlocks
        self.attentionResolutions = attentionResolutions
        self.embeddingDimensions = embeddingDimensions
        self.codebookSize = codebookSize
        self.transformerDimensions = transformerDimensions
        self.codePredictorLayers = codePredictorLayers
        self.heads = heads
        self.normGroups = normGroups
        self.connectResolutions = connectResolutions
        self.fidelity = fidelity
    }

    public static let base = NFKMLXCodeFormerConfiguration()

    public static let tiny = NFKMLXCodeFormerConfiguration(resolution: 32, baseChannels: 8,
                                                           channelMultipliers: [1, 2], resBlocks: 1,
                                                           attentionResolutions: [16], embeddingDimensions: 8,
                                                           codebookSize: 16, transformerDimensions: 16,
                                                           codePredictorLayers: 1, heads: 2, normGroups: 4,
                                                           connectResolutions: [32])

    /// The side length of the latent grid, and the number of code tokens.
    var latentSide: Int { resolution / (1 << (channelMultipliers.count - 1)) }
    var latentTokens: Int { latentSide * latentSide }
}

private func nfkSwish(_ x: MLXArray) -> MLXArray { x * sigmoid(x) }

/// Shared deterministic small-magnitude initialization for standalone parameters (SwinIR's relative
/// position bias table also draws from it), so a random-weights model runs repeatably. A loaded
/// checkpoint overwrites it.
enum NFKCodeFormerOps {
    static func parameter(_ shape: [Int]) -> MLXArray {
        let count = shape.reduce(1, *)
        var values = [Float](repeating: 0, count: count)
        for i in 0 ..< count {
            values[i] = (Float((i * 733) % 2003) / 2003.0 - 0.5) * 0.04
        }
        return values.withUnsafeBufferPointer { MLXArray($0, shape) }
    }
}

/// The reference VQGAN residual block: two group-normalized swish convolutions, with a 1×1 skip
/// projection when the channel count changes. GroupNorm runs at the reference's epsilon 1e-6.
final class NFKCFResBlock: Module, UnaryLayer {
    @ModuleInfo(key: "norm1") var norm1: GroupNorm
    @ModuleInfo(key: "conv1") var conv1: Conv2d
    @ModuleInfo(key: "norm2") var norm2: GroupNorm
    @ModuleInfo(key: "conv2") var conv2: Conv2d
    @ModuleInfo(key: "conv_out") var convOut: Conv2d?

    init(inChannels: Int, outChannels: Int, groups: Int) {
        _norm1.wrappedValue = GroupNorm(groupCount: groups, dimensions: inChannels, eps: 1e-6,
                                        affine: true, pytorchCompatible: true)
        _conv1.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: 3, padding: 1)
        _norm2.wrappedValue = GroupNorm(groupCount: groups, dimensions: outChannels, eps: 1e-6,
                                        affine: true, pytorchCompatible: true)
        _conv2.wrappedValue = Conv2d(inputChannels: outChannels, outputChannels: outChannels, kernelSize: 3, padding: 1)
        if inChannels != outChannels {
            _convOut.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: 1)
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = conv1(nfkSwish(norm1(x)))
        h = conv2(nfkSwish(norm2(h)))
        let identity = convOut.map { $0(x) } ?? x
        return identity + h
    }
}

/// The reference VQGAN attention block: single-head attention over spatial positions, through 1×1
/// convolutions, added to the identity.
final class NFKCFAttnBlock: Module, UnaryLayer {
    @ModuleInfo(key: "norm") var norm: GroupNorm
    @ModuleInfo(key: "q") var q: Conv2d
    @ModuleInfo(key: "k") var k: Conv2d
    @ModuleInfo(key: "v") var v: Conv2d
    @ModuleInfo(key: "proj_out") var projOut: Conv2d

    init(channels: Int, groups: Int) {
        _norm.wrappedValue = GroupNorm(groupCount: groups, dimensions: channels, eps: 1e-6,
                                       affine: true, pytorchCompatible: true)
        _q.wrappedValue = Conv2d(inputChannels: channels, outputChannels: channels, kernelSize: 1)
        _k.wrappedValue = Conv2d(inputChannels: channels, outputChannels: channels, kernelSize: 1)
        _v.wrappedValue = Conv2d(inputChannels: channels, outputChannels: channels, kernelSize: 1)
        _projOut.wrappedValue = Conv2d(inputChannels: channels, outputChannels: channels, kernelSize: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (h, w, c) = (x.shape[1], x.shape[2], x.shape[3])
        let normalized = norm(x)
        let queries = q(normalized).reshaped([h * w, c])
        let keys = k(normalized).reshaped([h * w, c])
        let values = v(normalized).reshaped([h * w, c])
        let attention = softmax(matmul(queries, keys.transposed(1, 0)) * pow(Float(c), -0.5), axis: -1)
        let out = matmul(attention, values).reshaped([1, h, w, c])
        return x + projOut(out)
    }
}

/// The reference `Downsample`: an asymmetric right/bottom pad, then a stride-2 valid convolution.
final class NFKCFDownsample: Module, UnaryLayer {
    @ModuleInfo(key: "conv") var conv: Conv2d

    init(channels: Int) {
        _conv.wrappedValue = Conv2d(inputChannels: channels, outputChannels: channels, kernelSize: 3, stride: 2)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let padded = MLX.padded(x, widths: [IntOrPair(0), IntOrPair((0, 1)), IntOrPair((0, 1)), IntOrPair(0)],
                                mode: .constant, value: MLXArray(Float(0)))
        return conv(padded)
    }
}

/// The reference `Upsample`: nearest ×2, then a convolution.
final class NFKCFUpsample: Module, UnaryLayer {
    @ModuleInfo(key: "conv") var conv: Conv2d

    init(channels: Int) {
        _conv.wrappedValue = Conv2d(inputChannels: channels, outputChannels: channels, kernelSize: 3, padding: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        conv(NFKMLXResample.upsampleNearest(x, scale: 2))
    }
}

/// A VQGAN coder half: the reference stores encoder and generator as one flat, heterogeneous `blocks`
/// list, so the checkpoint's numeric indices map onto a real array here.
final class NFKCFCoder: Module {
    @ModuleInfo(key: "blocks") var blocks: [any UnaryLayer]

    init(blocks: [any UnaryLayer]) {
        _blocks.wrappedValue = blocks
    }
}

/// The controllable feature transformation (`Fuse_sft_block`): encoder detail and generator features
/// concatenate through a residual block, and a learned scale and shift modulate the generator
/// features, weighted by the fidelity `w`.
final class NFKCFFuseBlock: Module {
    @ModuleInfo(key: "encode_enc") var encodeEnc: NFKCFResBlock
    @ModuleInfo(key: "scale1") var scale1: Conv2d
    @ModuleInfo(key: "scale2") var scale2: Conv2d
    @ModuleInfo(key: "shift1") var shift1: Conv2d
    @ModuleInfo(key: "shift2") var shift2: Conv2d

    init(channels: Int, groups: Int) {
        _encodeEnc.wrappedValue = NFKCFResBlock(inChannels: channels * 2, outChannels: channels, groups: groups)
        _scale1.wrappedValue = Conv2d(inputChannels: channels, outputChannels: channels, kernelSize: 3, padding: 1)
        _scale2.wrappedValue = Conv2d(inputChannels: channels, outputChannels: channels, kernelSize: 3, padding: 1)
        _shift1.wrappedValue = Conv2d(inputChannels: channels, outputChannels: channels, kernelSize: 3, padding: 1)
        _shift2.wrappedValue = Conv2d(inputChannels: channels, outputChannels: channels, kernelSize: 3, padding: 1)
    }

    func callAsFunction(encoded: MLXArray, decoded: MLXArray, weight: Float) -> MLXArray {
        let fused = encodeEnc(concatenated([encoded, decoded], axis: 3))
        let scale = scale2(leakyRelu(scale1(fused), negativeSlope: 0.2))
        let shift = shift2(leakyRelu(shift1(fused), negativeSlope: 0.2))
        return decoded + weight * (decoded * scale + shift)
    }
}

/// The code predictor's Transformer layer (`TransformerSALayer`): pre-norm multi-head self-attention
/// whose queries and keys carry the position embedding (values do not), then a pre-norm GELU MLP.
/// The attention keeps the reference's fused `in_proj_weight` / `out_proj` layout.
final class NFKCFTransformerLayer: Module {
    @ParameterInfo(key: "in_proj_weight") var inProjWeight: MLXArray
    @ParameterInfo(key: "in_proj_bias") var inProjBias: MLXArray
    @ModuleInfo(key: "out_proj") var outProj: Linear
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "linear2") var linear2: Linear
    let heads: Int

    init(dimensions: Int, heads: Int) {
        _inProjWeight.wrappedValue = NFKCodeFormerOps.parameter([dimensions * 3, dimensions])
        _inProjBias.wrappedValue = MLXArray.zeros([dimensions * 3])
        _outProj.wrappedValue = Linear(dimensions, dimensions)
        _norm1.wrappedValue = LayerNorm(dimensions: dimensions)
        _norm2.wrappedValue = LayerNorm(dimensions: dimensions)
        _linear1.wrappedValue = Linear(dimensions, dimensions * 2)
        _linear2.wrappedValue = Linear(dimensions * 2, dimensions)
        self.heads = heads
    }

    func callAsFunction(_ x: MLXArray, positions: MLXArray) -> MLXArray {
        let dimensions = x.shape[1]
        let headDimensions = dimensions / heads
        let tokens = x.shape[0]

        let normalized = norm1(x)
        let withPositions = normalized + positions
        func project(_ input: MLXArray, _ slot: Int) -> MLXArray {
            let weight = inProjWeight[slot * dimensions ..< (slot + 1) * dimensions]
            let bias = inProjBias[slot * dimensions ..< (slot + 1) * dimensions]
            return (matmul(input, weight.transposed(1, 0)) + bias)
                .reshaped([tokens, heads, headDimensions]).transposed(1, 0, 2)
        }
        let queries = project(withPositions, 0)
        let keys = project(withPositions, 1)
        let values = project(normalized, 2)
        let attention = softmax(matmul(queries, keys.transposed(0, 2, 1)) * pow(Float(headDimensions), -0.5), axis: -1)
        let attended = matmul(attention, values).transposed(1, 0, 2).reshaped([tokens, dimensions])
        let afterAttention = x + outProj(attended)

        return afterAttention + linear2(gelu(linear1(norm2(afterAttention))))
    }
}

/// The code-to-index head (`idx_pred_layer`): a LayerNorm and a bias-free projection to the codebook.
/// The reference packs them positionally; the remap translates.
final class NFKCFIndexPredictor: Module {
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "proj") var proj: Linear

    init(dimensions: Int, codebookSize: Int) {
        _norm.wrappedValue = LayerNorm(dimensions: dimensions)
        _proj.wrappedValue = Linear(dimensions, codebookSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        proj(norm(x))
    }
}

/// The codebook, under the reference's `quantize.embedding` name.
final class NFKCFQuantizer: Module {
    @ModuleInfo(key: "embedding") var embedding: Embedding

    init(codebookSize: Int, dimensions: Int) {
        _embedding.wrappedValue = Embedding(embeddingCount: codebookSize, dimensions: dimensions)
    }
}

/// The CodeFormer network: VQGAN encoder, Transformer code predictor, codebook, generator, and the
/// per-scale controllable feature transformation.
final class NFKMLXCodeFormerNet: Module {
    @ModuleInfo(key: "encoder") var encoder: NFKCFCoder
    @ModuleInfo(key: "generator") var generator: NFKCFCoder
    @ModuleInfo(key: "quantize") var quantize: NFKCFQuantizer
    @ModuleInfo(key: "feat_emb") var featEmb: Linear
    @ParameterInfo(key: "position_emb") var positionEmb: MLXArray
    @ModuleInfo(key: "ft_layers") var ftLayers: [NFKCFTransformerLayer]
    @ModuleInfo(key: "idx_pred_layer") var idxPredLayer: NFKCFIndexPredictor
    @ModuleInfo(key: "fuse_blocks") var fuseBlocks: [NFKCFFuseBlock]

    let configuration: NFKMLXCodeFormerConfiguration
    /// Encoder block indices whose output is captured, per connect resolution (parallel to
    /// `fuseBlocks` and `configuration.connectResolutions`).
    let captureIndices: [Int]
    /// Generator block indices after which the corresponding fuse block applies.
    let fuseIndices: [Int]

    init(_ c: NFKMLXCodeFormerConfiguration) {
        configuration = c
        let nf = c.baseChannels
        let stages = c.channelMultipliers.count

        // Encoder, exactly as the reference builds it: conv-in; per stage `resBlocks` residual blocks
        // (attention after each at the configured resolutions) and a downsample between stages; then a
        // residual-attention-residual tail, a norm, and the projection to the embedding width.
        var encoderBlocks: [any UnaryLayer] = [Conv2d(inputChannels: 3, outputChannels: nf, kernelSize: 3, padding: 1)]
        var captures: [Int: Int] = [:]
        var resolution = c.resolution
        var channels = nf
        for stage in 0 ..< stages {
            let outChannels = nf * c.channelMultipliers[stage]
            for _ in 0 ..< c.resBlocks {
                encoderBlocks.append(NFKCFResBlock(inChannels: channels, outChannels: outChannels, groups: c.normGroups))
                channels = outChannels
                if c.attentionResolutions.contains(resolution) {
                    encoderBlocks.append(NFKCFAttnBlock(channels: channels, groups: c.normGroups))
                }
            }
            captures[resolution] = encoderBlocks.count - 1
                - (c.attentionResolutions.contains(resolution) ? 1 : 0)
            if stage != stages - 1 {
                encoderBlocks.append(NFKCFDownsample(channels: channels))
                resolution /= 2
            }
        }
        encoderBlocks.append(NFKCFResBlock(inChannels: channels, outChannels: channels, groups: c.normGroups))
        encoderBlocks.append(NFKCFAttnBlock(channels: channels, groups: c.normGroups))
        encoderBlocks.append(NFKCFResBlock(inChannels: channels, outChannels: channels, groups: c.normGroups))
        encoderBlocks.append(GroupNorm(groupCount: c.normGroups, dimensions: channels, eps: 1e-6,
                                       affine: true, pytorchCompatible: true))
        encoderBlocks.append(Conv2d(inputChannels: channels, outputChannels: c.embeddingDimensions,
                                    kernelSize: 3, padding: 1))
        _encoder.wrappedValue = NFKCFCoder(blocks: encoderBlocks)

        // Generator, mirrored: conv-in; residual-attention-residual head; per stage (coarse to fine)
        // the residual blocks and an upsample between stages; then a norm and the RGB projection.
        var generatorBlocks: [any UnaryLayer] = [Conv2d(inputChannels: c.embeddingDimensions,
                                                        outputChannels: channels, kernelSize: 3, padding: 1)]
        var fuses: [Int: Int] = [:]
        generatorBlocks.append(NFKCFResBlock(inChannels: channels, outChannels: channels, groups: c.normGroups))
        generatorBlocks.append(NFKCFAttnBlock(channels: channels, groups: c.normGroups))
        generatorBlocks.append(NFKCFResBlock(inChannels: channels, outChannels: channels, groups: c.normGroups))
        for stage in stride(from: stages - 1, through: 0, by: -1) {
            let outChannels = nf * c.channelMultipliers[stage]
            for block in 0 ..< c.resBlocks {
                generatorBlocks.append(NFKCFResBlock(inChannels: channels, outChannels: outChannels, groups: c.normGroups))
                channels = outChannels
                if block == 0 {
                    fuses[resolution] = generatorBlocks.count - 1
                }
                if c.attentionResolutions.contains(resolution) {
                    generatorBlocks.append(NFKCFAttnBlock(channels: channels, groups: c.normGroups))
                }
            }
            if stage != 0 {
                generatorBlocks.append(NFKCFUpsample(channels: channels))
                resolution *= 2
            }
        }
        generatorBlocks.append(GroupNorm(groupCount: c.normGroups, dimensions: channels, eps: 1e-6,
                                         affine: true, pytorchCompatible: true))
        generatorBlocks.append(Conv2d(inputChannels: channels, outputChannels: 3, kernelSize: 3, padding: 1))
        _generator.wrappedValue = NFKCFCoder(blocks: generatorBlocks)

        _quantize.wrappedValue = NFKCFQuantizer(codebookSize: c.codebookSize, dimensions: c.embeddingDimensions)
        _featEmb.wrappedValue = Linear(c.embeddingDimensions, c.transformerDimensions)
        _positionEmb.wrappedValue = NFKCodeFormerOps.parameter([c.latentTokens, c.transformerDimensions])
        _ftLayers.wrappedValue = (0 ..< c.codePredictorLayers).map { _ in
            NFKCFTransformerLayer(dimensions: c.transformerDimensions, heads: c.heads)
        }
        _idxPredLayer.wrappedValue = NFKCFIndexPredictor(dimensions: c.transformerDimensions,
                                                         codebookSize: c.codebookSize)

        // One fuse block per connect resolution; each stage's width is what the generator carries
        // there (the encoder capture has the same width by construction).
        var fuseChannels: [Int] = []
        var stageResolution = c.resolution
        var widthAt: [Int: Int] = [:]
        for stage in 0 ..< stages {
            widthAt[stageResolution] = nf * c.channelMultipliers[stage]
            if stage != stages - 1 {
                stageResolution /= 2
            }
        }
        for connect in c.connectResolutions {
            fuseChannels.append(widthAt[connect]!)
        }
        _fuseBlocks.wrappedValue = fuseChannels.map { NFKCFFuseBlock(channels: $0, groups: c.normGroups) }
        captureIndices = c.connectResolutions.map { captures[$0]! }
        fuseIndices = c.connectResolutions.map { fuses[$0]! }
    }

    /// The reference's `adaptive_instance_normalization`: `content` adopts `style`'s per-channel
    /// mean and standard deviation (unbiased variance over the spatial positions, epsilon inside the
    /// square root, as `calc_mean_std` computes them).
    static func adaptiveInstanceNormalized(_ content: MLXArray, style: MLXArray) -> MLXArray {
        func statistics(_ x: MLXArray) -> (mean: MLXArray, deviation: MLXArray) {
            let count = Float(x.shape[1] * x.shape[2])
            let mean = MLX.mean(x, axes: [1, 2], keepDims: true)
            let variance = MLX.sum((x - mean) * (x - mean), axes: [1, 2], keepDims: true) / (count - 1)
            return (mean, sqrt(variance + 1e-5))
        }
        let (contentMean, contentDeviation) = statistics(content)
        let (styleMean, styleDeviation) = statistics(style)
        return (content - contentMean) / contentDeviation * styleDeviation + styleMean
    }

    /// Predicts the code logits `[tokens, codebookSize]` from the encoder's latent `[1, s, s, emb]`.
    func codeLogits(_ latent: MLXArray) -> MLXArray {
        let side = latent.shape[1]
        let tokens = latent.reshaped([side * side, latent.shape[3]])
        var t = featEmb(tokens)
        for layer in ftLayers {
            t = layer(t, positions: positionEmb)
        }
        return idxPredLayer(t)
    }

    /// Restores a bridged face `[H, W, 3]` (`0...1`) at the given fidelity weight. The face resizes to
    /// the model resolution, runs encode → code prediction → codebook lookup → decode with the
    /// controllable feature transformation, and returns `[res, res, 3]` in `0...1`.
    func restore(_ image: MLXArray, fidelity: Float? = nil) -> MLXArray {
        let weight = fidelity ?? configuration.fidelity
        let batched = image.reshaped([1, image.shape[0], image.shape[1], image.shape[2]])
        var x = batched
        if batched.shape[1] != configuration.resolution || batched.shape[2] != configuration.resolution {
            x = NFKMLXResample.resizeBilinear(batched, height: configuration.resolution,
                                              width: configuration.resolution)
        }
        x = x * 2 - 1                                           // the coders work in -1...1

        var captured: [Int: MLXArray] = [:]
        for (index, block) in encoder.blocks.enumerated() {
            x = block(x)
            if let slot = captureIndices.firstIndex(of: index) {
                captured[slot] = x
            }
        }

        let side = x.shape[1]
        let logits = codeLogits(x)
        let indices = logits.argMax(axis: -1)                   // [tokens]
        var out = quantize.embedding(indices).reshaped([1, side, side, configuration.embeddingDimensions])
        // The reference restores with `adain=True`: the quantized features adopt the degraded
        // latent's per-channel statistics, which keeps the decode consistent with this input.
        out = Self.adaptiveInstanceNormalized(out, style: x)

        for (index, block) in generator.blocks.enumerated() {
            out = block(out)
            if weight > 0, let slot = fuseIndices.firstIndex(of: index) {
                out = fuseBlocks[slot](encoded: captured[slot]!, decoded: out, weight: weight)
            }
        }
        return clip((out + 1) * 0.5, min: 0, max: 1).reshaped([configuration.resolution, configuration.resolution, 3])
    }
}

/// Holds the network for capture in the backend's `@Sendable` closure.
private final class NFKCodeFormerHolder: @unchecked Sendable {
    let net: NFKMLXCodeFormerNet
    init(_ net: NFKMLXCodeFormerNet) { self.net = net }
}

/// CodeFormer face restoration as an InferKit backend, and its registration for the Objective-C path.
///
/// `NFKMLXCodeFormerNet` is the reference code-prediction restorer. Random weights run (proving the
/// pipeline); the released `codeformer.pth`, converted to **safetensors**, restores faces. The input
/// face is aligned and cropped by the caller before restoration.
@objc(NFKMLXCodeFormer)
public final class NFKMLXCodeFormer: NSObject {

    /// The registry name the model builds under.
    @objc public static let modelName = "codeformer"

    /// Builds a face-restoration backend directly from optional local weights — no registry required.
    /// A nil `weightsURL` builds random weights (`isReady` is true). Restores at the configuration's
    /// default fidelity; `backend(fidelity:weightsURL:)` chooses it. Run
    /// inference off the render thread.
    @objc(backendWithWeightsURL:error:)
    public static func backend(weightsURL: URL?) throws -> any NFKInferenceBackend {
        try backend(fidelity: NFKMLXCodeFormerConfiguration.base.fidelity, weightsURL: weightsURL)
    }

    /// Builds a face-restoration backend at a chosen fidelity weight. `fidelity` runs from 0 (full
    /// generative quality, the codebook decides every detail) to 1 (the degraded input's own detail is
    /// carried through the controllable feature transformation). One backend restores at one fidelity;
    /// build a second backend to offer the consumer a different tradeoff.
    @objc(backendWithFidelity:weightsURL:error:)
    public static func backend(fidelity: Float, weightsURL: URL?) throws -> any NFKInferenceBackend {
        var configuration = NFKMLXCodeFormerConfiguration.base
        configuration.fidelity = max(0, min(1, fidelity))
        let net = NFKMLXCodeFormerNet(configuration)
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        let holder = NFKCodeFormerHolder(net)
        return NFKMLXModuleBackend(identifier: modelName, isReady: true) { image in holder.net.restore(image) }
    }

    /// Builds a backend that restores every face in a PHOTOGRAPH, rather than an aligned crop.
    ///
    /// @discussion The model itself takes an aligned 512×512 face; this finds the faces, aligns each
    /// one, restores it, and composites the results back into the original frame. Detection and
    /// alignment default to `NFKMLXRetinaFace`, the detector the reference pipeline runs, so the crop
    /// this produces is the crop facexlib produces. Pass `detectorWeightsURL` for it; a nil URL builds
    /// random detector weights and finds nothing useful. The overload taking a detector accepts
    /// `NFKMLXVisionFaceDetector` instead when a download-free path matters more.
    ///
    /// An image with no detectable face passes through unmodified. Run inference off the render
    /// thread: this is one forward pass per face plus detection.
    @objc(photoBackendWithFidelity:weightsURL:detectorWeightsURL:error:)
    public static func photoBackend(fidelity: Float, weightsURL: URL?,
                                    detectorWeightsURL: URL?) throws -> any NFKInferenceBackend {
        try photoBackend(fidelity: fidelity, weightsURL: weightsURL,
                         detector: try NFKMLXRetinaFace.detector(weightsURL: detectorWeightsURL))
    }

    /// The same, with a chosen detector.
    ///
    /// @discussion RetinaFace is the default because it is the detector the reference pipeline runs,
    /// so the crop it produces is the reference's own; it costs a 1.7 MB checkpoint beside
    /// CodeFormer's. `NFKMLXVisionFaceDetector` is the alternative when a download-free path matters
    /// more than matching the reference — the two disagree enough to move the crop, which
    /// `NFKMLXFaceAlignmentTests` measures rather than describes.
    public static func photoBackend(fidelity: Float, weightsURL: URL?,
                                    detector: any NFKMLXFaceDetecting) throws -> any NFKInferenceBackend {
        var configuration = NFKMLXCodeFormerConfiguration.base
        configuration.fidelity = max(0, min(1, fidelity))
        let net = NFKMLXCodeFormerNet(configuration)
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        let holder = NFKCodeFormerHolder(net)
        return NFKMLXPhotoFaceBackend(identifier: "\(modelName)-photo") { image in
            try restoreFaces(in: image, using: holder.net, detector: detector)
        }
    }

    /// Restores every detected face in `image`, compositing each back where it was found.
    static func restoreFaces(in image: CGImage, using net: NFKMLXCodeFormerNet,
                             detector: any NFKMLXFaceDetecting) throws -> CGImage {
        let faces = try detector.faces(in: image)
        guard !faces.isEmpty else { return image }

        var options = NFKMLXImageOptions()
        options.colorSpace = CGColorSpaceCreateDeviceRGB()
        var canvas = image
        for face in faces {
            guard let (crop, transform) = NFKMLXFaceAlignment.alignedCrop(from: image, face: face) else {
                continue
            }
            let (bytes, width, height) = NFKMLXImageBridge.rgbaBytes(from: crop, colorSpace: options.colorSpace)
            let tensor = NFKMLXImageBridge.tensor(rgba: bytes, width: width, height: height, channels: 3)
            let restored = clip(net.restore(tensor), min: 0, max: 1)
            eval(restored)
            let restoredImage = try NFKMLXImageBridge.cgImage(from: restored, options: options)
            if let composited = NFKMLXFaceAlignment.pasteBack(restoredImage, into: canvas, transform: transform) {
                canvas = composited
            }
        }
        return canvas
    }

    /// Downloads the checkpoint from Hugging Face, then builds the backend — no registry required.
    /// Blocking on the network; run off the render thread.
    @objc(backendWithRepo:weightsPath:revision:cacheDirectoryURL:error:)
    public static func backend(repo: String, weightsPath: String, revision: String?, cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        try backend(fidelity: NFKMLXCodeFormerConfiguration.base.fidelity, repo: repo, weightsPath: weightsPath,
                    revision: revision, cacheDirectoryURL: cacheDirectoryURL)
    }

    /// The download factory at a chosen fidelity weight. Blocking on the
    /// network; run off the render thread.
    @objc(backendWithFidelity:repo:weightsPath:revision:cacheDirectoryURL:error:)
    public static func backend(fidelity: Float, repo: String, weightsPath: String, revision: String?,
                               cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        let url = try NFKMLXDownload.weightsURL(repo: repo, weightsPath: weightsPath, revision: revision, cacheDirectoryURL: cacheDirectoryURL)
        return try backend(fidelity: fidelity, weightsURL: url)
    }

    /// The asynchronous form of the download factory: downloads on a background queue, then builds and
    /// delivers the backend (or an error) to `completionHandler`.
    @objc(backendWithRepo:weightsPath:revision:cacheDirectoryURL:completionHandler:)
    public static func backend(repo: String, weightsPath: String, revision: String?, cacheDirectoryURL: URL?,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        backend(fidelity: NFKMLXCodeFormerConfiguration.base.fidelity, repo: repo, weightsPath: weightsPath,
                revision: revision, cacheDirectoryURL: cacheDirectoryURL, completionHandler: completionHandler)
    }

    /// The asynchronous download factory at a chosen fidelity weight.
    @objc(backendWithFidelity:repo:weightsPath:revision:cacheDirectoryURL:completionHandler:)
    public static func backend(fidelity: Float, repo: String, weightsPath: String, revision: String?,
                               cacheDirectoryURL: URL?,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXDownload.backend(repo: repo, weightsPath: weightsPath, revision: revision,
                               cacheDirectoryURL: cacheDirectoryURL,
                               build: { try backend(fidelity: fidelity, weightsURL: $0) },
                               completionHandler: completionHandler)
    }

    /// Registers CodeFormer (`codeformer`) with `NFKMLXModelRegistry`, delegating to `backend(weightsURL:)`.
    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName) { weightsURL in
            try backend(weightsURL: weightsURL)
        }
    }

    /// Maps the reference's names onto the module's: the fuse dictionary's resolution keys and the
    /// positional Sequentials become semantic (MLX's `update(parameters:)` parses a numeric key as an
    /// array index, so `fuse_convs_dict.32` or `scale.0` cannot be kept literally). The coders' flat
    /// `blocks.N` indices land on real arrays and pass through untouched.
    static func remapReferenceKey(_ key: String, connectResolutions: [Int]) -> String {
        var key = key
        if key.hasPrefix("fuse_convs_dict.") {
            let rest = key.dropFirst("fuse_convs_dict.".count)
            guard let dot = rest.firstIndex(of: "."), let size = Int(rest[..<dot]),
                  let slot = connectResolutions.firstIndex(of: size) else { return key }
            key = "fuse_blocks.\(slot)" + String(rest[dot...])
            for branch in ["scale", "shift"] {
                key = key.replacingOccurrences(of: ".\(branch).0.", with: ".\(branch)1.")
                key = key.replacingOccurrences(of: ".\(branch).2.", with: ".\(branch)2.")
            }
            return key
        }
        if key.hasPrefix("idx_pred_layer.0.") {
            return "idx_pred_layer.norm." + key.dropFirst("idx_pred_layer.0.".count)
        }
        if key.hasPrefix("idx_pred_layer.1.") {
            return "idx_pred_layer.proj." + key.dropFirst("idx_pred_layer.1.".count)
        }
        return key.replacingOccurrences(of: ".self_attn.", with: ".")
    }

    /// Loads a safetensors checkpoint into `net`, remapping the reference's names and transposing 4-D
    /// convolution weights from PyTorch's `[out, in, kH, kW]` to MLX's channels-last
    /// `[out, kH, kW, in]`.
    static func loadWeights(into net: NFKMLXCodeFormerNet, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let raw = checkpoint.arrays
        let mapped = raw.map { key, value in
            (remapReferenceKey(key, connectResolutions: net.configuration.connectResolutions),
             checkpoint.needsConvTranspose && value.ndim == 4 ? value.transposed(0, 2, 3, 1) : value)
        }
        try NFKMLXWeights.apply(mapped, to: net)
    }
}
