// Mimi (`kyutai/mimi`, Kyutai, CC-BY-4.0), the transformer-in-codec neural audio codec: a SEANet
// encoder/decoder with a RoPE Transformer on each side and a SPLIT residual vector quantizer (one
// semantic codebook beside 31 acoustic ones). 24 kHz audio in, 12.5 Hz discrete codes, audio back.
// Ported from transformers' `MimiModel` (modeling_mimi.py), the reference the parity test measures
// against (`run_reference.py mimi`).
//
// Everything works in MLX's NLC layout `[B, T, C]` (what Conv1d uses), so the reference's [B, C, T]
// tensors are transposed only at the seams the reference transposes. The k1 `input_proj`/`output_proj`
// convolutions load as `Linear` (the k1 axis squeezed).

import Foundation
import InferKit
import MLX
import MLXNN

/// The Mimi configuration (`kyutai/mimi` `config.json`).
public struct NFKMLXMimiConfiguration: Sendable {
    public var hiddenSize = 512
    public var numFilters = 64
    public var numResidualLayers = 1
    public var upsampleRatios = [8, 6, 5, 4]
    public var kernelSize = 7
    public var lastKernelSize = 3
    public var residualKernelSize = 3
    public var dilationGrowthRate = 2
    public var compress = 2
    public var numQuantizers = 32
    public var numSemanticQuantizers = 1
    public var codebookSize = 2048
    public var codebookDim = 256
    public var numHiddenLayers = 8
    public var numHeads = 8
    public var headDim = 64
    public var intermediateSize = 2048
    public var ropeTheta: Float = 10000
    public var normEps: Float = 1e-5
    public var slidingWindow = 250
    public var layerScaleInit: Float = 0.01
    public var sampleRate = 24000
    public var frameRate: Double = 12.5

    public init() {}
}

// MARK: - Causal convolutions

/// An empty module occupying an `nn.ModuleList` slot that holds an activation in the reference (ELU),
/// so the numeric layer indices match the checkpoint's names.
final class NFKMimiActivation: Module {}

/// `MimiConv1d`: a Conv1d with causal padding. The left pad is `kernelEff - stride`, the right pad is
/// the reference's stride-completion `extra_padding`. `pad_mode` is constant (zero) for the SEANet
/// convolutions and replicate (edge) for the downsample.
final class NFKMimiConv1d: Module {
    @ModuleInfo(key: "conv") var conv: Conv1d
    let stride: Int
    let kernelEff: Int
    let paddingTotal: Int
    let edgePad: Bool

    init(_ inChannels: Int, _ outChannels: Int, kernelSize: Int, stride: Int = 1, dilation: Int = 1,
         bias: Bool = true, edgePad: Bool = false) {
        self.stride = stride
        kernelEff = (kernelSize - 1) * dilation + 1
        paddingTotal = kernelEff - stride
        self.edgePad = edgePad
        _conv.wrappedValue = Conv1d(inputChannels: inChannels, outputChannels: outChannels,
                                    kernelSize: kernelSize, stride: stride, padding: 0, dilation: dilation, bias: bias)
    }

    private func extraPadding(_ length: Int) -> Int {
        let nFrames = Int(ceil(Double(length - kernelEff + paddingTotal) / Double(stride))) + 1 - 1
        let idealLength = nFrames * stride + kernelEff - paddingTotal
        return idealLength - length
    }

    /// `x [B, T, C]` → `[B, T', C']`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let extra = extraPadding(x.dim(1))
        let padded = MLX.padded(x, widths: [IntOrPair(0), IntOrPair((paddingTotal, extra)), IntOrPair(0)],
                                mode: edgePad ? .edge : .constant)
        return conv(padded)
    }
}

/// `MimiConvTranspose1d`: a causal transposed convolution built as zero-insertion (dilate by stride) +
/// a `kernel-1` pad + a plain (or depthwise) convolution over the FLIPPED kernel, then a right trim of
/// `kernel - stride`. MLX's grouped `ConvTransposed1d` disagrees with PyTorch (the GTCRN finding), so
/// the transposed convolution is assembled from a forward convolution, which is exact for both the
/// depthwise upsample and the grouped-1 decoder stages.
final class NFKMimiConvTranspose1d: Module {
    @ModuleInfo(key: "conv") var conv: Conv1d       // weight holds the FLIPPED transposed kernel
    let stride: Int
    let kernel: Int
    let trimRight: Int

    init(_ inChannels: Int, _ outChannels: Int, kernelSize: Int, stride: Int, groups: Int = 1, bias: Bool = true) {
        self.stride = stride
        kernel = kernelSize
        trimRight = kernelSize - stride
        _conv.wrappedValue = Conv1d(inputChannels: inChannels, outputChannels: outChannels,
                                    kernelSize: kernelSize, stride: 1, padding: 0, groups: groups, bias: bias)
    }

    /// `x [B, T, C]` → `[B, (T-1)·stride + kernel - trimRight, C']`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, t, c) = (x.dim(0), x.dim(1), x.dim(2))
        var xd = x
        if stride > 1 {
            xd = x.reshaped([b, t, 1, c])
            xd = MLX.padded(xd, widths: [IntOrPair(0), IntOrPair(0), IntOrPair((0, stride - 1)), IntOrPair(0)], mode: .constant)
            xd = xd.reshaped([b, t * stride, c])[0..., 0 ..< ((t - 1) * stride + 1), 0...]
        }
        xd = MLX.padded(xd, widths: [IntOrPair(0), IntOrPair((kernel - 1, kernel - 1)), IntOrPair(0)], mode: .constant)
        let out = conv(xd)                                          // [B, (T-1)·s + kernel, C']
        return out[0..., 0 ..< (out.dim(1) - trimRight), 0...]
    }
}

/// `MimiResnetBlock`: `block = [ELU, Conv(dim→dim/compress, k), ELU, Conv(dim/compress→dim, 1)]`, added
/// back to the input (identity shortcut).
final class NFKMimiResnetBlock: Module {
    @ModuleInfo(key: "block") var block: [Module]

    init(_ config: NFKMLXMimiConfiguration, dim: Int, dilation: Int) {
        let hidden = dim / config.compress
        _block.wrappedValue = [NFKMimiActivation(),
                               NFKMimiConv1d(dim, hidden, kernelSize: config.residualKernelSize, dilation: dilation),
                               NFKMimiActivation(),
                               NFKMimiConv1d(hidden, dim, kernelSize: 1)]
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for layer in block {
            if let c = layer as? NFKMimiConv1d { h = c(h) } else { h = elu(h) }
        }
        return x + h
    }
}

/// The SEANet encoder: a stack whose numeric indices match the reference's `nn.ModuleList`.
final class NFKMimiEncoder: Module {
    @ModuleInfo(key: "layers") var layers: [Module]

    init(_ config: NFKMLXMimiConfiguration) {
        var model: [Module] = [NFKMimiConv1d(1, config.numFilters, kernelSize: config.kernelSize)]
        var scaling = 1
        for ratio in config.upsampleRatios.reversed() {
            let current = scaling * config.numFilters
            for j in 0 ..< config.numResidualLayers {
                model.append(NFKMimiResnetBlock(config, dim: current, dilation: Int(pow(Double(config.dilationGrowthRate), Double(j)))))
            }
            model.append(NFKMimiActivation())
            model.append(NFKMimiConv1d(current, current * 2, kernelSize: ratio * 2, stride: ratio))
            scaling *= 2
        }
        model.append(NFKMimiActivation())
        model.append(NFKMimiConv1d(scaling * config.numFilters, config.hiddenSize, kernelSize: config.lastKernelSize))
        _layers.wrappedValue = model
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for layer in layers {
            if let c = layer as? NFKMimiConv1d { h = c(h) }
            else if let r = layer as? NFKMimiResnetBlock { h = r(h) }
            else { h = elu(h) }
        }
        return h
    }
}

/// The SEANet decoder: the encoder mirrored, upsampling through transposed convolutions.
final class NFKMimiDecoder: Module {
    @ModuleInfo(key: "layers") var layers: [Module]

    init(_ config: NFKMLXMimiConfiguration) {
        var scaling = Int(pow(2.0, Double(config.upsampleRatios.count)))
        var model: [Module] = [NFKMimiConv1d(config.hiddenSize, scaling * config.numFilters, kernelSize: config.kernelSize)]
        for ratio in config.upsampleRatios {
            let current = scaling * config.numFilters
            model.append(NFKMimiActivation())
            model.append(NFKMimiConvTranspose1d(current, current / 2, kernelSize: ratio * 2, stride: ratio))
            for j in 0 ..< config.numResidualLayers {
                model.append(NFKMimiResnetBlock(config, dim: current / 2, dilation: Int(pow(Double(config.dilationGrowthRate), Double(j)))))
            }
            scaling /= 2
        }
        model.append(NFKMimiActivation())
        model.append(NFKMimiConv1d(config.numFilters, 1, kernelSize: config.lastKernelSize))
        _layers.wrappedValue = model
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for layer in layers {
            if let c = layer as? NFKMimiConv1d { h = c(h) }
            else if let t = layer as? NFKMimiConvTranspose1d { h = t(h) }
            else if let r = layer as? NFKMimiResnetBlock { h = r(h) }
            else { h = elu(h) }
        }
        return h
    }
}

// MARK: - Transformer

/// `MimiLayerScale`: a learned per-channel scale on a residual branch (init 0.01).
final class NFKMimiLayerScale: Module {
    @ParameterInfo(key: "scale") var scale: MLXArray

    init(_ channels: Int, initialScale: Float) {
        _scale.wrappedValue = MLXArray.ones([channels]) * initialScale
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { scale * x }
}

private func nfkMimiRotateHalf(_ x: MLXArray) -> MLXArray {
    let half = x.dim(-1) / 2
    let x1 = x[.ellipsis, 0 ..< half]
    let x2 = x[.ellipsis, half...]
    return concatenated([-x2, x1], axis: -1)
}

/// `MimiAttention`: RoPE multi-head attention, no biases, causal with a sliding window. The mask is
/// additive `[T, T]`; on the short sequences a codec produces (T ≈ 25) the window is inert.
final class NFKMimiAttention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    let numHeads: Int
    let headDim: Int
    let scale: Float

    init(_ config: NFKMLXMimiConfiguration) {
        numHeads = config.numHeads
        headDim = config.headDim
        scale = 1 / sqrtf(Float(config.headDim))
        let inner = config.numHeads * config.headDim
        _qProj.wrappedValue = Linear(config.hiddenSize, inner, bias: false)
        _kProj.wrappedValue = Linear(config.hiddenSize, inner, bias: false)
        _vProj.wrappedValue = Linear(config.hiddenSize, inner, bias: false)
        _oProj.wrappedValue = Linear(inner, config.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray, mask: MLXArray) -> MLXArray {
        let (b, t) = (x.dim(0), x.dim(1))
        func heads(_ v: MLXArray) -> MLXArray { v.reshaped([b, t, numHeads, headDim]).transposed(0, 2, 1, 3) }
        var q = heads(qProj(x)), k = heads(kProj(x))
        let v = heads(vProj(x))
        q = q * cos + nfkMimiRotateHalf(q) * sin
        k = k * cos + nfkMimiRotateHalf(k) * sin
        var scores = matmul(q, k.transposed(0, 1, 3, 2)) * scale + mask
        scores = softmax(scores, axis: -1)
        let out = matmul(scores, v).transposed(0, 2, 1, 3).reshaped([b, t, numHeads * headDim])
        return oProj(out)
    }
}

/// A pre-norm transformer layer with layer scale on each residual branch (gelu MLP).
final class NFKMimiTransformerLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: NFKMimiAttention
    @ModuleInfo(key: "input_layernorm") var inputNorm: LayerNorm
    @ModuleInfo(key: "post_attention_layernorm") var postNorm: LayerNorm
    @ModuleInfo(key: "self_attn_layer_scale") var attnScale: NFKMimiLayerScale
    @ModuleInfo(key: "mlp_layer_scale") var mlpScale: NFKMimiLayerScale
    @ModuleInfo(key: "mlp") var mlp: NFKMimiMLP

    init(_ config: NFKMLXMimiConfiguration) {
        _selfAttn.wrappedValue = NFKMimiAttention(config)
        _inputNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.normEps)
        _postNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.normEps)
        _attnScale.wrappedValue = NFKMimiLayerScale(config.hiddenSize, initialScale: config.layerScaleInit)
        _mlpScale.wrappedValue = NFKMimiLayerScale(config.hiddenSize, initialScale: config.layerScaleInit)
        _mlp.wrappedValue = NFKMimiMLP(config)
    }

    func callAsFunction(_ x0: MLXArray, cos: MLXArray, sin: MLXArray, mask: MLXArray) -> MLXArray {
        var x = x0 + attnScale(selfAttn(inputNorm(x0), cos: cos, sin: sin, mask: mask))
        x = x + mlpScale(mlp(postNorm(x)))
        return x
    }
}

/// `MimiMLP`: `fc2(gelu(fc1(x)))`, no biases (exact-erf gelu).
final class NFKMimiMLP: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear

    init(_ config: NFKMLXMimiConfiguration) {
        _fc1.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _fc2.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(gelu(fc1(x))) }
}

/// The Transformer that contextualizes the SEANet latents (encoder and decoder sides share this class).
final class NFKMimiTransformer: Module {
    @ModuleInfo(key: "layers") var layers: [NFKMimiTransformerLayer]
    let headDim: Int
    let ropeTheta: Float
    let slidingWindow: Int

    init(_ config: NFKMLXMimiConfiguration) {
        _layers.wrappedValue = (0 ..< config.numHiddenLayers).map { _ in NFKMimiTransformerLayer(config) }
        headDim = config.headDim
        ropeTheta = config.ropeTheta
        slidingWindow = config.slidingWindow
    }

    /// `x [B, T, C]` → `[B, T, C]`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let t = x.dim(1)
        // RoPE cos/sin over positions 0..<t, shaped [1, 1, T, headDim] to broadcast over [B, H, T, hd].
        let half = headDim / 2
        let invFreq = MLXArray((0 ..< half).map { powf(ropeTheta, -Float(2 * $0) / Float(headDim)) })
        let positions = MLXArray((0 ..< t).map { Float($0) })
        let freqs = positions.reshaped([t, 1]) * invFreq.reshaped([1, half])   // [T, half]
        let emb = concatenated([freqs, freqs], axis: -1)                       // [T, headDim]
        let cos = MLX.cos(emb).reshaped([1, 1, t, headDim])
        let sin = MLX.sin(emb).reshaped([1, 1, t, headDim])
        // Causal + sliding-window additive mask [1, 1, T, T].
        let rows = MLXArray((0 ..< t).map { Int32($0) }).reshaped([t, 1])
        let cols = MLXArray((0 ..< t).map { Int32($0) }).reshaped([1, t])
        var mask = MLX.where(cols .<= rows, MLXArray(Float(0)), MLXArray(-Float.infinity))
        mask = MLX.where((rows - cols) .< MLXArray(Int32(slidingWindow)), mask, MLXArray(-Float.infinity))
        mask = mask.reshaped([1, 1, t, t])
        var h = x
        for layer in layers { h = layer(h, cos: cos, sin: sin, mask: mask) }
        return h
    }
}

// MARK: - Split residual vector quantizer

/// A Euclidean codebook. The reference stores the EMA state `embed_sum` / `cluster_usage`; the loader
/// folds them into `embed = embed_sum / max(cluster_usage, 1e-5)`, which the nearest-neighbor search and
/// the lookup use directly.
final class NFKMimiCodebook: Module {
    @ParameterInfo(key: "embed") var embed: MLXArray            // [codebookSize, codebookDim]

    init(size: Int, dim: Int) { _embed.wrappedValue = MLXArray.zeros([size, dim]) }

    /// `x [N, D]` → nearest indices `[N]` by Euclidean distance (argmin ‖x−e‖² = argmax(x·e − ½‖e‖²)).
    func encode(_ x: MLXArray) -> MLXArray {
        let half = (embed * embed).sum(axis: -1) * 0.5                          // [K]
        let scores = matmul(x, embed.transposed(1, 0)) - half.reshaped([1, embed.dim(0)])
        return scores.argMax(axis: -1)
    }

    /// `indices [...]` → `[..., D]`.
    func decode(_ indices: MLXArray) -> MLXArray {
        embed[indices]
    }
}

final class NFKMimiVectorQuantization: Module {
    @ModuleInfo(key: "codebook") var codebook: NFKMimiCodebook
    init(size: Int, dim: Int) { _codebook.wrappedValue = NFKMimiCodebook(size: size, dim: dim) }
}

/// A residual vector quantizer: `input_proj` (k1, as a `Linear`), a stack of codebooks over the residual,
/// then `output_proj`.
final class NFKMimiResidualVectorQuantizer: Module {
    @ModuleInfo(key: "layers") var layers: [NFKMimiVectorQuantization]
    @ModuleInfo(key: "input_proj") var inputProj: Linear
    @ModuleInfo(key: "output_proj") var outputProj: Linear

    init(_ config: NFKMLXMimiConfiguration, count: Int) {
        _layers.wrappedValue = (0 ..< count).map { _ in NFKMimiVectorQuantization(size: config.codebookSize, dim: config.codebookDim) }
        _inputProj.wrappedValue = Linear(config.hiddenSize, config.codebookDim, bias: false)
        _outputProj.wrappedValue = Linear(config.codebookDim, config.hiddenSize, bias: false)
    }

    /// `x [B, T, hidden]` → codes `[count, B, T]`.
    func encode(_ x: MLXArray) -> MLXArray {
        let projected = inputProj(x)                                            // [B, T, dim]
        let (b, t, d) = (projected.dim(0), projected.dim(1), projected.dim(2))
        var residual = projected.reshaped([b * t, d])
        var codes = [MLXArray]()
        for layer in layers {
            let idx = layer.codebook.encode(residual)                          // [B·T]
            residual = residual - layer.codebook.decode(idx)
            codes.append(idx.reshaped([b, t]))
        }
        return stacked(codes, axis: 0)                                          // [count, B, T]
    }

    /// `codes [count, B, T]` → `[B, T, hidden]`.
    func decode(_ codes: MLXArray) -> MLXArray {
        var quantized: MLXArray?
        for i in 0 ..< codes.dim(0) {
            let part = layers[i].codebook.decode(codes[i])                     // [B, T, dim]
            quantized = quantized == nil ? part : quantized! + part
        }
        return outputProj(quantized!)
    }
}

/// The split RVQ: a semantic codebook beside the acoustic ones, each quantizing the SAME latent through
/// its own projections; the codes are concatenated and the decodes summed.
final class NFKMimiSplitResidualVectorQuantizer: Module {
    @ModuleInfo(key: "semantic_residual_vector_quantizer") var semantic: NFKMimiResidualVectorQuantizer
    @ModuleInfo(key: "acoustic_residual_vector_quantizer") var acoustic: NFKMimiResidualVectorQuantizer
    let numSemantic: Int

    init(_ config: NFKMLXMimiConfiguration) {
        numSemantic = config.numSemanticQuantizers
        _semantic.wrappedValue = NFKMimiResidualVectorQuantizer(config, count: config.numSemanticQuantizers)
        _acoustic.wrappedValue = NFKMimiResidualVectorQuantizer(config, count: config.numQuantizers - config.numSemanticQuantizers)
    }

    /// `x [B, T, hidden]` → codes `[B, numQuantizers, T]`.
    func encode(_ x: MLXArray) -> MLXArray {
        let sem = semantic.encode(x)                                            // [numSemantic, B, T]
        let aco = acoustic.encode(x)                                            // [numAcoustic, B, T]
        return concatenated([sem, aco], axis: 0).transposed(1, 0, 2)            // [B, K, T]
    }

    /// `codes [B, numQuantizers, T]` → `[B, T, hidden]`.
    func decode(_ codes: MLXArray) -> MLXArray {
        let byLayer = codes.transposed(1, 0, 2)                                 // [K, B, T]
        let sem = semantic.decode(byLayer[0 ..< numSemantic])
        let aco = acoustic.decode(byLayer[numSemantic...])
        return sem + aco
    }
}

// MARK: - The model

/// The Mimi codec network: SEANet encoder + transformer + downsample + split RVQ + upsample + transformer
/// + SEANet decoder.
public final class NFKMLXMimiNet: Module {
    @ModuleInfo(key: "encoder") var encoder: NFKMimiEncoder
    @ModuleInfo(key: "encoder_transformer") var encoderTransformer: NFKMimiTransformer
    @ModuleInfo(key: "downsample") var downsample: NFKMimiConv1d
    @ModuleInfo(key: "upsample") var upsample: NFKMimiConvTranspose1d
    @ModuleInfo(key: "decoder_transformer") var decoderTransformer: NFKMimiTransformer
    @ModuleInfo(key: "decoder") var decoder: NFKMimiDecoder
    @ModuleInfo(key: "quantizer") var quantizer: NFKMimiSplitResidualVectorQuantizer
    let config: NFKMLXMimiConfiguration

    public init(_ config: NFKMLXMimiConfiguration = .init()) {
        self.config = config
        _encoder.wrappedValue = NFKMimiEncoder(config)
        _encoderTransformer.wrappedValue = NFKMimiTransformer(config)
        _downsample.wrappedValue = NFKMimiConv1d(config.hiddenSize, config.hiddenSize, kernelSize: 4, stride: 2,
                                                 bias: false, edgePad: true)
        _upsample.wrappedValue = NFKMimiConvTranspose1d(config.hiddenSize, config.hiddenSize, kernelSize: 4,
                                                        stride: 2, groups: config.hiddenSize, bias: false)
        _decoderTransformer.wrappedValue = NFKMimiTransformer(config)
        _decoder.wrappedValue = NFKMimiDecoder(config)
        _quantizer.wrappedValue = NFKMimiSplitResidualVectorQuantizer(config)
    }

    /// The latent an audio clip encodes to, before quantization: `samples [B, N]` → `[B, T, hidden]`.
    func embed(_ samples: MLXArray) -> MLXArray {
        let x = samples.reshaped([samples.dim(0), samples.dim(1), 1])           // [B, N, 1]
        let e = encoder(x)                                                      // [B, T@25, hidden]
        let et = encoderTransformer(e)
        return downsample(et)                                                   // [B, T@12.5, hidden]
    }

    /// `samples [B, N]` → codes `[B, numQuantizers, T]`.
    public func encode(_ samples: MLXArray) -> MLXArray {
        quantizer.encode(embed(samples))
    }

    /// `codes [B, numQuantizers, T]` → waveform `[B, N']`.
    public func decode(_ codes: MLXArray) -> MLXArray {
        let dq = quantizer.decode(codes)                                        // [B, T@12.5, hidden]
        let us = upsample(dq)                                                   // [B, T@25, hidden]
        let dt = decoderTransformer(us)
        return decoder(dt)[0..., 0..., 0]                                       // [B, N']
    }

    public func callAsFunction(_ samples: MLXArray) -> MLXArray { decode(encode(samples)) }
}

// MARK: - Weight loading

private func nfkMimiFlipKernel(_ w: MLXArray) -> MLXArray {
    // Reverse the kernel axis (axis 1 of MLX's `[out, kernel, in]`) via a gather with reversed indices.
    let k = w.dim(1)
    return MLX.take(w, MLXArray((0 ..< k).reversed().map { Int32($0) }), axis: 1)
}

private func nfkMimiIsDecoderConvTranspose(_ key: String) -> Bool {
    for i in [2, 5, 8, 11] where key == "decoder.layers.\(i).conv.weight" { return true }
    return false
}

extension NFKMLXMimiNet {
    /// Loads the released `model.safetensors` (`kyutai/mimi`): folds each codebook's EMA state into
    /// `embed`, squeezes the k1 projection convolutions to `Linear`, transposes the Conv1d weights to
    /// MLX's NLC layout, and assembles each transposed convolution's forward kernel (transpose + flip).
    public func loadWeights(from url: URL) throws {
        let raw = try NFKMLXWeights.loadCheckpoint(url: url).arrays
        var clusterUsage = [String: MLXArray]()
        for (key, value) in raw where key.hasSuffix(".cluster_usage") { clusterUsage[key] = value }

        var mapped = [(String, MLXArray)]()
        for (key, value) in raw {
            if key.hasSuffix(".initialized") || key.hasSuffix(".cluster_usage") { continue }
            if key.hasSuffix(".embed_sum") {
                let base = String(key.dropLast(".embed_sum".count))
                let usage = maximum(clusterUsage[base + ".cluster_usage"]!, MLXArray(Float(1e-5)))
                mapped.append((base + ".embed", value / usage.reshaped([usage.dim(0), 1])))
            } else if key.hasSuffix("input_proj.weight") || key.hasSuffix("output_proj.weight") {
                mapped.append((key, value.reshaped([value.dim(0), value.dim(1)])))   // squeeze the k1 axis
            } else if key == "upsample.conv.weight" {
                mapped.append((key, nfkMimiFlipKernel(value.transposed(0, 2, 1))))    // depthwise transposed conv
            } else if nfkMimiIsDecoderConvTranspose(key) {
                mapped.append((key, nfkMimiFlipKernel(value.transposed(1, 2, 0))))    // grouped-1 transposed conv
            } else if value.ndim == 3 {
                mapped.append((key, value.transposed(0, 2, 1)))                       // Conv1d [out,in,k] → [out,k,in]
            } else {
                mapped.append((key, value))
            }
        }
        try NFKMLXWeights.apply(mapped, to: self)
    }
}

// MARK: - Backend

private final class NFKMimiHolder: @unchecked Sendable {
    let net: NFKMLXMimiNet
    init(_ net: NFKMLXMimiNet) { self.net = net }
}

/// Mimi reconstruction (audio → codes → audio) as an InferKit backend, reading `NFKInputAudio` and
/// returning the reconstructed clip under `NFKOutputAudio`. The codes themselves are reached through
/// `NFKMLXMimi.encode` / `decode`, which is what a codec-token consumer wants.
@objc(NFKMLXMimiBackend)
public final class NFKMLXMimiBackend: NSObject, NFKInferenceBackend {
    private let holder: NFKMimiHolder
    private let identifier: String
    private let sampleRate: Int

    init(net: NFKMLXMimiNet, identifier: String) {
        holder = NFKMimiHolder(net)
        self.identifier = identifier
        sampleRate = net.config.sampleRate
        super.init()
    }

    @objc public var isReady: Bool { true }
    @objc public var backendIdentifier: String { identifier }

    /// The request parameters the backend reads. Introduced in InferKit 0.4.0.
    @objc public var supportedParameterKeys: Set<String> { [] }

    /// The request inputs the backend reads. Introduced in InferKit 0.4.0.
    @objc public var supportedInputKeys: Set<String> { [NFKInputAudio] }

    @objc(runInferenceForRequest:error:)
    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        guard let (samples, inputRate) = Self.audio(from: request) else { throw NFKMLXError.unsupportedInput }
        let input = inputRate == sampleRate ? samples : NFKMLXAudioRate.matched(samples, from: inputRate, to: sampleRate)
        let x = input.withUnsafeBufferPointer { MLXArray($0, [1, input.count]) }
        let waveform = holder.net(x)[0].asArray(Float.self)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("mimi-\(UUID().uuidString).wav")
        try NFKMLXWaveFile.write(samples: waveform, sampleRate: sampleRate, to: url)
        let asset = NFKAudioAsset(fileURL: url, durationSeconds: Double(waveform.count) / Double(sampleRate),
                                  sampleRate: Double(sampleRate), channelCount: 1)
        return NFKInferenceResult(outputs: [NFKOutputAudio: asset])
    }

    @objc(submitInferenceJobForRequest:)
    public func submitInferenceJob(for request: NFKInferenceRequest) -> NFKInferenceJob {
        let job = NFKInferenceJob()
        Task.detached(priority: .userInitiated) {
            do { job.finish(with: try self.runInference(for: request)) }
            catch { job.finish(withError: error as NSError) }
        }
        return job
    }

    private static func audio(from request: NFKInferenceRequest) -> (samples: [Float], sampleRate: Int)? {
        guard let value = request.input(forKey: NFKInputAudio) else { return nil }
        if let asset = value as? NFKAudioAsset, let url = asset.fileURL, let data = try? Data(contentsOf: url) {
            return NFKMLXWaveFile.read(data)
        }
        if let data = value as? Data { return NFKMLXWaveFile.read(data) }
        return nil
    }
}

// MARK: - Facade, factories, registration

/// Construction, encode/decode, and registration for the Mimi codec.
@objc(NFKMLXMimi)
public final class NFKMLXMimi: NSObject {

    /// The registry name the model builds under.
    @objc public static let modelName = "mimi"

    private let holder: NFKMimiHolder
    init(net: NFKMLXMimiNet) { holder = NFKMimiHolder(net) }

    static func makeNet(_ config: NFKMLXMimiConfiguration = .init()) -> NFKMLXMimiNet { NFKMLXMimiNet(config) }

    /// Encodes a mono 24 kHz waveform to its codebook token streams: `codes[c][t]` is codebook `c`'s
    /// token at frame `t` (codebook 0 is the semantic one).
    public func encode(_ samples: [Float]) -> [[Int]] {
        let x = samples.withUnsafeBufferPointer { MLXArray($0, [1, samples.count]) }
        let codes = holder.net.encode(x)                                        // [1, K, T]
        eval(codes)
        let (books, frames) = (codes.dim(1), codes.dim(2))
        let flat = codes.reshaped([books, frames]).asType(.int32).asArray(Int32.self)
        return (0 ..< books).map { book in (0 ..< frames).map { Int(flat[book * frames + $0]) } }
    }

    /// Reconstructs a mono waveform from codebook token streams.
    public func decode(_ codes: [[Int]]) -> [Float] {
        let books = codes.count, frames = codes.first?.count ?? 0
        let flat = codes.flatMap { $0.map(Int32.init) }
        let array = flat.withUnsafeBufferPointer { MLXArray($0, [1, books, frames]) }
        return holder.net.decode(array)[0].asArray(Float.self)
    }

    private static func loadedNet(_ config: NFKMLXMimiConfiguration, weightsURL: URL?) throws -> NFKMLXMimiNet {
        let net = makeNet(config)
        if let weightsURL { try net.loadWeights(from: weightsURL) }
        return net
    }

    /// Builds a Mimi reconstruction backend from optional local weights — no registry required. A nil
    /// `weightsURL` builds random weights (`isReady` is true). Run off the render thread.
    ///
    /// - Since: InferKit 0.4.0
    @objc(backendWithWeightsURL:error:)
    public static func backend(weightsURL: URL?) throws -> any NFKInferenceBackend {
        NFKMLXMimiBackend(net: try loadedNet(.init(), weightsURL: weightsURL), identifier: modelName)
    }

    /// Builds a Mimi codec object (with `encode` / `decode`) directly from optional local weights.
    public static func codec(weightsURL: URL?) throws -> NFKMLXMimi {
        NFKMLXMimi(net: try loadedNet(.init(), weightsURL: weightsURL))
    }

    /// Downloads the checkpoint from Hugging Face, then builds — no registry required. Blocking on the
    /// network; run off the render thread.
    @objc(backendWithRepo:weightsPath:revision:cacheDirectoryURL:error:)
    public static func backend(repo: String, weightsPath: String, revision: String?, cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        let url = try NFKMLXDownload.weightsURL(repo: repo, weightsPath: weightsPath, revision: revision, cacheDirectoryURL: cacheDirectoryURL)
        return try backend(weightsURL: url)
    }

    /// The asynchronous form of the download factory.
    @objc(backendWithRepo:weightsPath:revision:cacheDirectoryURL:completionHandler:)
    public static func backend(repo: String, weightsPath: String, revision: String?, cacheDirectoryURL: URL?,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXDownload.backend(repo: repo, weightsPath: weightsPath, revision: revision,
                               cacheDirectoryURL: cacheDirectoryURL,
                               build: { try backend(weightsURL: $0) },
                               completionHandler: completionHandler)
    }

    /// Registers Mimi (`mimi`) with `NFKMLXModelRegistry`, delegating to `backend(weightsURL:)`.
    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName) { weightsURL in try backend(weightsURL: weightsURL) }
    }
}
