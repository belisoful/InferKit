//
//  NFKMLXWanS2V.swift
//  InferKitMLX
//
//  The Wan 2.2 S2V-14B speech-to-video transformer (Wan-Video/Wan2.2 `WanModel_S2V`, Apache-2.0): the Wan
//  block over the clip's tokens, a reference image's tokens, and the previous clip's motion frames packed at
//  three scales, driven by a speech track through a causal audio encoder and per-frame audio cross-attention.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import MLX
import MLXNN

/// The S2V transformer's geometry beyond the Wan block's.
public struct NFKMLXWanS2VConfiguration: Sendable {
    /// The Wan block's geometry (`dim`, heads, layers, the patch).
    public var base: NFKMLXWanConfiguration
    /// The pose condition's latent width (`cond_dim`).
    public var conditionChannels: Int
    /// The audio encoder's input width, the wav2vec2 hidden size (`audio_dim`).
    public var audioDim: Int
    /// The wav2vec2 hidden states the encoder weighs, its layers plus its embedding output.
    public var audioLayers: Int
    /// The audio tokens each latent frame reads (`num_audio_token`), before the padding token.
    public var audioTokens: Int
    /// The blocks after which the audio cross-attention runs (`audio_inject_layers`).
    public var audioInjectLayers: [Int]
    /// The frame packer's buckets, nearest first (`zip_frame_buckets`, `[1, 2, 16]`).
    public var framePackBuckets: [Int]
    /// The rotary frame the reference image's tokens take.
    public var referenceFrame: Int

    public init(base: NFKMLXWanConfiguration, conditionChannels: Int = 16, audioDim: Int = 1024, audioLayers: Int = 25,
                audioTokens: Int = 4, audioInjectLayers: [Int] = [0, 4, 8, 12, 16, 20, 24, 27, 30, 33, 36, 39],
                framePackBuckets: [Int] = [1, 2, 16], referenceFrame: Int = 30) {
        self.base = base
        self.conditionChannels = conditionChannels
        self.audioDim = audioDim
        self.audioLayers = audioLayers
        self.audioTokens = audioTokens
        self.audioInjectLayers = audioInjectLayers
        self.framePackBuckets = framePackBuckets
        self.referenceFrame = referenceFrame
    }

    /// The configuration a release's `config.json` states (`WanModel_S2V`). A configuration that turns on a
    /// branch the released model leaves off (the transformer motioner, AdaIN off or in another mode, no frame
    /// packing, no zero timestep for the reference and motion tokens) is refused.
    static func reading(_ url: URL) throws -> NFKMLXWanS2VConfiguration {
        let json = try NFKMLXWanRelease.json(url)
        guard json["model_type"] as? String == "s2v" else {
            throw NFKMLXError.unsupportedConfiguration("\(url.lastPathComponent) is not a Wan S2V configuration")
        }
        let flags: [(String, Bool)] = [("enable_framepack", true), ("enable_adain", true), ("zero_timestep", true),
                                      ("add_last_motion", true), ("enable_motioner", false), ("enable_tsm", false),
                                      ("trainable_token_pos_emb", false)]
        for (key, expected) in flags where ((json[key] as? NSNumber)?.boolValue ?? !expected) != expected {
            throw NFKMLXError.unsupportedConfiguration("a Wan S2V transformer with \(key) \(!expected) is not implemented")
        }
        guard json["adain_mode"] as? String ?? "attn_norm" == "attn_norm",
              json["framepack_drop_mode"] as? String ?? "padd" == "padd" else {
            throw NFKMLXError.unsupportedConfiguration("only the released AdaIN and frame-pack modes are implemented")
        }
        var base = try NFKMLXWanOriginalRelease.transformerConfiguration(fromConfigURL: url)
        base.outChannels = json["out_dim"] as? Int ?? base.inChannels
        return NFKMLXWanS2VConfiguration(base: base, conditionChannels: json["cond_dim"] as? Int ?? 16,
                                         audioDim: json["audio_dim"] as? Int ?? 1024,
                                         audioTokens: json["num_audio_token"] as? Int ?? 4,
                                         audioInjectLayers: json["audio_inject_layers"] as? [Int] ?? [0, 27])
    }
}

/// A 1-D convolution padded causally by repeating the first frame (`CausalConv1d`, `pad_mode="replicate"`).
final class NFKWanS2VCausalConv: Module {
    @ModuleInfo(key: "conv") var conv: Conv1d
    let kernel: Int

    init(_ input: Int, _ output: Int, kernel: Int = 3, stride: Int = 1) {
        self.kernel = kernel
        _conv.wrappedValue = Conv1d(inputChannels: input, outputChannels: output, kernelSize: kernel, stride: stride)
    }

    /// `x` `[B, T, C]`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let first = x[0..., 0 ..< 1, 0...]
        return conv(concatenated([repeated(first, count: kernel - 1, axis: 1), x], axis: 1))
    }
}

/// The audio encoder's convolution stack (`MotionEncoder_tc`): a local branch of `audioTokens` heads and a
/// global branch, sharing the two strided convolutions.
final class NFKWanS2VMotionEncoder: Module {
    @ModuleInfo(key: "conv1_local") var conv1Local: NFKWanS2VCausalConv
    @ModuleInfo(key: "conv1_global") var conv1Global: NFKWanS2VCausalConv
    @ModuleInfo(key: "conv2") var conv2: NFKWanS2VCausalConv
    @ModuleInfo(key: "conv3") var conv3: NFKWanS2VCausalConv
    @ModuleInfo(key: "final_linear") var finalLinear: Linear
    @ParameterInfo(key: "padding_tokens") var paddingTokens: MLXArray

    let heads: Int
    let hidden: Int

    init(inDim: Int, hidden: Int, heads: Int) {
        self.heads = heads
        self.hidden = hidden
        _conv1Local.wrappedValue = NFKWanS2VCausalConv(inDim, hidden / 4 * heads)
        _conv1Global.wrappedValue = NFKWanS2VCausalConv(inDim, hidden / 4)
        _conv2.wrappedValue = NFKWanS2VCausalConv(hidden / 4, hidden / 2, stride: 2)
        _conv3.wrappedValue = NFKWanS2VCausalConv(hidden / 2, hidden, stride: 2)
        _finalLinear.wrappedValue = Linear(hidden, hidden)
        _paddingTokens.wrappedValue = MLXArray.zeros([1, 1, 1, hidden])
    }

    private func norm(_ x: MLXArray) -> MLXArray {
        silu(NFKReferenceRounding.layerNorm(x, weight: nil, bias: nil, eps: 1e-6))
    }

    private func strided(_ x: MLXArray) -> MLXArray { norm(conv3(norm(conv2(x)))) }

    /// `x` `[T, inDim]` → the local tokens `[T', heads + 1, hidden]` and the global token `[T', hidden]`.
    func callAsFunction(_ x: MLXArray) -> (local: MLXArray, global: MLXArray) {
        let t = x.dim(0)
        let input = x.expandedDimensions(axis: 0)
        // `b (n c) t -> (b n) t c`: the local convolution's channels are `heads` groups.
        let local = conv1Local(input).reshaped([t, heads, hidden / 4]).transposed(1, 0, 2)
        let encoded = strided(norm(local))                                       // [heads, T', hidden]
        let tokens = concatenated([encoded.transposed(1, 0, 2),
                                   broadcast(paddingTokens[0], to: [encoded.dim(1), 1, hidden])], axis: 1)
        let global = finalLinear(strided(norm(conv1Global(input))))[0]           // [T', hidden]
        return (tokens, global)
    }
}

/// The weighted sum over wav2vec2's hidden states and the convolution stack (`CausalAudioEncoder`).
final class NFKWanS2VAudioEncoder: Module {
    @ParameterInfo(key: "weights") var weights: MLXArray
    @ModuleInfo(key: "encoder") var encoder: NFKWanS2VMotionEncoder

    init(_ configuration: NFKMLXWanS2VConfiguration) {
        _weights.wrappedValue = MLXArray.ones([1, configuration.audioLayers, 1, 1]) * 0.01
        _encoder.wrappedValue = NFKWanS2VMotionEncoder(inDim: configuration.audioDim, hidden: configuration.base.innerDim,
                                                       heads: configuration.audioTokens)
    }

    /// `features` `[layers, audioDim, T]` → the local tokens `[T', tokens + 1, dim]` and the global `[T', dim]`.
    func callAsFunction(_ features: MLXArray) -> (local: MLXArray, global: MLXArray) {
        let w = silu(weights[0].asType(.float32))                               // [layers, 1, 1]
        let weighted = ((features.asType(.float32) * w) / w.sum(axis: 0, keepDims: true)).sum(axis: 0)  // [audioDim, T]
        return encoder(weighted.transposed(1, 0))
    }
}

/// The audio cross-attention (`AudioCrossAttention`, Wan's cross-attention under the original names).
final class NFKWanS2VAudioAttention: Module {
    @ModuleInfo(key: "q") var q: Linear
    @ModuleInfo(key: "k") var k: Linear
    @ModuleInfo(key: "v") var v: Linear
    @ModuleInfo(key: "o") var o: Linear
    @ModuleInfo(key: "norm_q") var normQ: RMSNorm
    @ModuleInfo(key: "norm_k") var normK: RMSNorm
    let heads: Int

    init(dim: Int, heads: Int, eps: Float) {
        self.heads = heads
        _q.wrappedValue = Linear(dim, dim)
        _k.wrappedValue = Linear(dim, dim)
        _v.wrappedValue = Linear(dim, dim)
        _o.wrappedValue = Linear(dim, dim)
        _normQ.wrappedValue = RMSNorm(dimensions: dim, eps: eps)
        _normK.wrappedValue = RMSNorm(dimensions: dim, eps: eps)
    }

    /// `x` `[F, N, dim]` attends over `context` `[F, M, dim]`, frame by frame.
    func callAsFunction(_ x: MLXArray, context: MLXArray) -> MLXArray {
        let f = x.dim(0), n = x.dim(1), m = context.dim(1), dim = x.dim(2), d = dim / heads
        let qh = NFKWanAttention.normalized(q(x), normQ).reshaped([f, n, heads, d]).transposed(0, 2, 1, 3)
        let kh = NFKWanAttention.normalized(k(context), normK).reshaped([f, m, heads, d]).transposed(0, 2, 1, 3)
        let vh = v(context).reshaped([f, m, heads, d]).transposed(0, 2, 1, 3)
        let out = NFKReferenceRounding.flashAttention(queries: qh, keys: kh, values: vh, scale: 1 / sqrt(Float(d)), mask: nil)
        return o(out.transposed(0, 2, 1, 3).reshaped([f, n, dim]))
    }
}

/// diffusers' `AdaLayerNorm` (`chunk_dim=1`): a non-affine LayerNorm (eps 1e-5) shifted and scaled by a linear map of
/// the SiLU of an embedding, whose first half is the shift.
final class NFKWanS2VAdaLayerNorm: Module {
    @ModuleInfo(key: "linear") var linear: Linear

    init(dim: Int) { _linear.wrappedValue = Linear(dim, 2 * dim) }

    /// `x` `[F, N, dim]`, `embedding` `[F, dim]`.
    func callAsFunction(_ x: MLXArray, embedding: MLXArray) -> MLXArray {
        let projected = linear(silu(embedding))
        let dim = x.dim(2)
        let shift = projected[0..., 0 ..< dim].expandedDimensions(axis: 1)
        let scale = projected[0..., dim...].expandedDimensions(axis: 1)
        return NFKReferenceRounding.layerNorm(x, weight: nil, bias: nil, eps: 1e-5) * (1 + scale) + shift
    }
}

/// The audio injector (`AudioInjector_WAN`): one cross-attention and one AdaIN per injected block.
final class NFKWanS2VAudioInjector: Module {
    @ModuleInfo(key: "injector") var injector: [NFKWanS2VAudioAttention]
    @ModuleInfo(key: "injector_adain_layers") var adain: [NFKWanS2VAdaLayerNorm]

    init(_ configuration: NFKMLXWanS2VConfiguration) {
        let base = configuration.base
        _injector.wrappedValue = configuration.audioInjectLayers.map { _ in
            NFKWanS2VAudioAttention(dim: base.innerDim, heads: base.heads, eps: base.eps)
        }
        _adain.wrappedValue = configuration.audioInjectLayers.map { _ in NFKWanS2VAdaLayerNorm(dim: base.innerDim) }
    }
}

/// The frame packer (`FramePackMotioner`): the motion latents' last 1, 2, and 16 frames patchified at 1×, 2×, and 4×.
final class NFKWanS2VFramePacker: Module {
    @ModuleInfo(key: "proj") var proj: Conv3d
    @ModuleInfo(key: "proj_2x") var proj2x: Conv3d
    @ModuleInfo(key: "proj_4x") var proj4x: Conv3d

    init(channels: Int, dim: Int) {
        _proj.wrappedValue = Conv3d(inputChannels: channels, outputChannels: dim, kernelSize: IntOrTriple([1, 2, 2]),
                                    stride: IntOrTriple([1, 2, 2]))
        _proj2x.wrappedValue = Conv3d(inputChannels: channels, outputChannels: dim, kernelSize: IntOrTriple([2, 4, 4]),
                                      stride: IntOrTriple([2, 4, 4]))
        _proj4x.wrappedValue = Conv3d(inputChannels: channels, outputChannels: dim, kernelSize: IntOrTriple([4, 8, 8]),
                                      stride: IntOrTriple([4, 8, 8]))
    }
}

/// The inputs of one S2V pass beyond the noised latent and the text.
struct NFKWanS2VInputs {
    /// The reference image's latent `[16, 1, h, w]`.
    var reference: MLXArray
    /// The previous clip's motion latents `[16, frames, h, w]`, nil where the first clip drops them.
    var motion: MLXArray?
    /// The pose latents `[16, F', h, w]` (zeros without a pose video).
    var pose: MLXArray
    /// The audio tokens of the clip's latent frames, `[F', tokens + 1, dim]`, and their global tokens `[F', dim]`.
    var audio: (local: MLXArray, global: MLXArray)
}

/// The Wan 2.2 S2V transformer.
public final class NFKMLXWanS2VNet: Module {
    @ModuleInfo(key: "patch_embedding") var patchEmbedding: Conv3d
    @ModuleInfo(key: "condition_embedder") var conditionEmbedder: NFKWanConditionEmbedder
    @ModuleInfo(key: "blocks") var blocks: [NFKWanBlock]
    @ParameterInfo(key: "scale_shift_table") var scaleShiftTable: MLXArray
    @ModuleInfo(key: "proj_out") var projOut: Linear
    @ModuleInfo(key: "cond_encoder") var condEncoder: Conv3d
    @ModuleInfo(key: "casual_audio_encoder") var audioEncoder: NFKWanS2VAudioEncoder
    @ModuleInfo(key: "audio_injector") var audioInjector: NFKWanS2VAudioInjector
    @ModuleInfo(key: "trainable_cond_mask") var tokenType: Embedding
    @ModuleInfo(key: "frame_packer") var framePacker: NFKWanS2VFramePacker

    public let configuration: NFKMLXWanS2VConfiguration
    /// The blocks left in the release and read in their turn on every pass, where the load streams them.
    var layerStream: NFKMLXLayerStream?
    /// Whether a block's activations are recomputed in the backward pass. Introduced in InferKit 0.4.0.
    public var recomputesBlocksInBackward = false

    public init(_ configuration: NFKMLXWanS2VConfiguration) {
        self.configuration = configuration
        let base = configuration.base
        let patch = IntOrTriple(base.patchSize)
        _patchEmbedding.wrappedValue = Conv3d(inputChannels: base.inChannels, outputChannels: base.innerDim,
                                              kernelSize: patch, stride: patch)
        _conditionEmbedder.wrappedValue = NFKWanConditionEmbedder(base)
        _blocks.wrappedValue = (0 ..< base.layers).map { _ in NFKWanBlock(base) }
        _scaleShiftTable.wrappedValue = MLXArray.zeros([1, 2, base.innerDim])
        _projOut.wrappedValue = Linear(base.innerDim, base.outChannels * base.patchSize.reduce(1, *))
        _condEncoder.wrappedValue = Conv3d(inputChannels: configuration.conditionChannels, outputChannels: base.innerDim,
                                           kernelSize: patch, stride: patch)
        _audioEncoder.wrappedValue = NFKWanS2VAudioEncoder(configuration)
        _audioInjector.wrappedValue = NFKWanS2VAudioInjector(configuration)
        _tokenType.wrappedValue = Embedding(embeddingCount: 3, dimensions: base.innerDim)
        _framePacker.wrappedValue = NFKWanS2VFramePacker(channels: base.inChannels, dim: base.innerDim)
    }

    /// The audio tokens of a clip: `features` `[layers, audioDim, frames]` (wav2vec2's hidden states at the clip's
    /// frame rate) with `motionFrames` copies of the first frame ahead of them, encoded, and the first
    /// `motionLatentFrames` latent frames dropped.
    func audioTokens(_ features: MLXArray, motionFrames: Int, motionLatentFrames: Int) -> (local: MLXArray, global: MLXArray) {
        let padded = concatenated([repeated(features[0..., 0..., 0 ..< 1], count: motionFrames, axis: 2), features], axis: 2)
        let encoded = audioEncoder(padded)
        return (encoded.local[motionLatentFrames...], encoded.global[motionLatentFrames...])
    }

    // MARK: Rotary positions

    /// One `rope_precompute` grid: the tokens of a `count` × `height` × `width` patch grid whose frames sit at
    /// `frames` (negative for the motion frames, whose rotation the reference conjugates) and whose rows and columns
    /// are spread over `rangeHeight` × `rangeWidth` positions by `np.linspace(...).astype(int)`.
    struct RopeGrid {
        var frames: [Int]
        var height: Int
        var width: Int
        var rangeHeight: Int
        var rangeWidth: Int

        /// NumPy's `linspace(start, stop, count).astype(int)`: `start + i · step` in float64 with the last entry
        /// exactly `stop`, truncated toward zero.
        static func linspace(_ start: Int, _ stop: Int, _ count: Int) -> [Int] {
            guard count > 1 else { return count == 1 ? [start] : [] }
            let step = Double(stop - start) / Double(count - 1)
            return (0 ..< count).map { $0 == count - 1 ? stop : Int(Double($0) * step + Double(start)) }
        }

        var positions: [(Int, Int, Int)] {
            let rows = Self.linspace(0, rangeHeight - 1, height), columns = Self.linspace(0, rangeWidth - 1, width)
            return frames.flatMap { f in rows.flatMap { h in columns.map { w in (f, h, w) } } }
        }
    }

    /// The rotary tables `[N, headDim / 2]` of `grids`' tokens in order, the angles formed in float64.
    func ropeTables(_ grids: [RopeGrid]) -> (cos: MLXArray, sin: MLXArray) {
        let base = configuration.base
        let inverse = base.ropeAxes.map { d in
            stride(from: 0, to: d, by: 2).map { 1.0 / Foundation.pow(Double(base.ropeTheta), Double($0) / Double(d)) }
        }
        var cosines = [Float](), sines = [Float]()
        var count = 0
        for grid in grids {
            for (f, h, w) in grid.positions {
                count += 1
                for (axis, position) in [f, h, w].enumerated() {
                    for frequency in inverse[axis] {
                        let angle = Double(position) * frequency
                        cosines.append(Float(Foundation.cos(angle)))
                        sines.append(Float(Foundation.sin(angle)))
                    }
                }
            }
        }
        let width = cosines.count / max(count, 1)
        return (MLXArray(cosines).reshaped([count, width]), MLXArray(sines).reshaped([count, width]))
    }

    // MARK: The pass

    /// The flattened patch embedding `[N, dim]` of `x` `[C, F, H, W]` through `conv`.
    private func patches(_ x: MLXArray, _ conv: Conv3d) -> MLXArray {
        conv(x.transposed(1, 2, 3, 0).expandedDimensions(axis: 0)).reshaped([-1, configuration.base.innerDim])
    }

    /// The velocity of the clip's latent `x` `[16, F', h, w]` at timestep `t` (`WanModel_S2V.forward`).
    ///
    /// @discussion The tokens are the clip's (with the pose condition's patches added), the reference image's at
    /// rotary frame `referenceFrame`, and the motion latents' frame-packed tokens at negative frames; a learned
    /// embedding marks each kind (0, 1, 2). The clip's tokens modulate at `t`, the others at timestep 0. After each
    /// block `audioInjectLayers` names, the clip's tokens of each latent frame pass an AdaIN on that frame's global
    /// audio token and attend over its audio tokens, and the result adds to them. The head reads the clip's tokens.
    func callAsFunction(_ x: MLXArray, text: MLXArray, t: Double, inputs: NFKWanS2VInputs) -> MLXArray {
        let base = configuration.base
        let c = base.outChannels, f = x.dim(1), h = x.dim(2), w = x.dim(3)
        let ph = base.patchSize[1], pw = base.patchSize[2]
        let hh = h / ph, ww = w / pw, inner = base.innerDim
        let videoTokens = f * hh * ww

        var parts = [patches(x, patchEmbedding) + patches(inputs.pose, condEncoder), patches(inputs.reference, patchEmbedding)]
        var grids = [RopeGrid(frames: Array(0 ..< f), height: hh, width: ww, rangeHeight: hh, rangeWidth: ww),
                     RopeGrid(frames: [configuration.referenceFrame], height: hh, width: ww, rangeHeight: hh, rangeWidth: ww)]
        if let motion = inputs.motion {
            let buckets = configuration.framePackBuckets, total = buckets.reduce(0, +)
            var packed = MLXArray.zeros([motion.dim(0), total, h, w], dtype: motion.dtype)
            let overlap = min(total, motion.dim(1))
            if overlap > 0 {
                packed = concatenated([packed[0..., 0 ..< (total - overlap)], motion[0..., (motion.dim(1) - overlap)...]], axis: 1)
            }
            let far = buckets[2], middle = buckets[1]
            let clean4x = packed[0..., 0 ..< far], clean2x = packed[0..., far ..< (far + middle)]
            let post = packed[0..., (far + middle)...]
            parts += [patches(post, framePacker.proj), patches(clean2x, framePacker.proj2x), patches(clean4x, framePacker.proj4x)]
            // Each bucket's frames as `rope_precompute` places them: its span of the frames before the clip, read
            // by `np.linspace(...).astype(int)` and negated (the reference conjugates the rotation), at 1×, ½×, and ¼×
            // the clip's patch grid.
            var start = 0
            for (scale, bucket) in zip([1, 2, 4], buckets) {
                start += bucket
                let count = bucket / scale
                grids.append(RopeGrid(frames: RopeGrid.linspace(start, start - bucket + 1, count).map { -$0 },
                                      height: hh / scale, width: ww / scale, rangeHeight: hh, rangeWidth: ww))
            }
        }
        var hidden = concatenated(parts, axis: 0)
        let total = hidden.dim(0), referenceTokens = hh * ww
        var kinds = [Int32](repeating: 0, count: total)
        for index in videoTokens ..< total {
            kinds[index] = index < videoTokens + referenceTokens ? 1 : 2
        }
        hidden = hidden + tokenType(MLXArray(kinds)).asType(hidden.dtype)
        let (cosT, sinT) = ropeTables(grids)

        let (temb, proj) = conditionEmbedder.timesteps(float64: [t, 0], dtype: .float32, freqDim: base.freqDim)
        var index = [Int32](repeating: 1, count: total)
        for token in 0 ..< videoTokens { index[token] = 0 }
        let tokens = NFKWanTokenModulation(proj: proj.reshaped([2, 6, inner]), index: MLXArray(index))
        let context = conditionEmbedder.textEmbedder(text)
        let injected = Dictionary(uniqueKeysWithValues: configuration.audioInjectLayers.enumerated().map { ($1, $0) })

        for (blockIndex, block) in blocks.enumerated() {
            let input = hidden
            let audioIndex = injected[blockIndex]
            let run = { (x: MLXArray) -> MLXArray in
                var y = block(x, context: context, temb: proj, cos: cosT, sin: sinT, tokens: tokens)
                if let audioIndex {
                    y = self.injectingAudio(y, index: audioIndex, frames: f, audio: inputs.audio, videoTokens: videoTokens)
                }
                return y
            }
            guard let output = NFKMLXLayerStream.run(layerStream, layer: block, index: blockIndex, {
                recomputesBlocksInBackward && layerStream == nil
                    ? NFKMLXCustomFunction.recomputed(block) { [run($0[0])] }([input])[0] : run(input)
            }, keeping: { [$0] }) else {
                continue
            }
            hidden = output
        }

        let video = hidden[0 ..< videoTokens].asType(.float32)
        let table = scaleShiftTable[0].asType(.float32)
        let embedding = temb[0 ..< 1].asType(.float32)
        let shift = table[0 ..< 1] + embedding, scale = table[1 ..< 2] + embedding
        let normed = NFKReferenceRounding.layerNorm(video, weight: nil, bias: nil, eps: base.eps) * (1 + scale) + shift
        let out = projOut(normed.asType(projOut.weight.dtype))
        let pt = base.patchSize[0]
        return out.reshaped([f / pt, hh, ww, pt, ph, pw, c]).transposed(6, 0, 3, 1, 4, 2, 5).reshaped([c, f, h, w])
    }

    /// `hidden` with the audio cross-attention of injector `index` added to the clip's tokens.
    func injectingAudio(_ hidden: MLXArray, index: Int, frames: Int, audio: (local: MLXArray, global: MLXArray),
                        videoTokens: Int) -> MLXArray {
        let inner = configuration.base.innerDim
        let video = hidden[0 ..< videoTokens].reshaped([frames, videoTokens / frames, inner])
        let normed = audioInjector.adain[index](video, embedding: audio.global.asType(video.dtype))
        let attended = audioInjector.injector[index](normed, context: audio.local.asType(video.dtype))
        return concatenated([hidden[0 ..< videoTokens] + attended.reshaped([videoTokens, inner]), hidden[videoTokens...]], axis: 0)
    }
}

extension NFKMLXWanS2VNet {
    /// The module key a released tensor name maps to: the Wan block's and embedders' names as
    /// ``NFKMLXWanOriginalRelease/transformerModuleKey(forRelease:)`` maps them, the S2V modules' as released.
    static func moduleKey(forRelease name: String) -> String {
        for prefix in ["casual_audio_encoder.", "audio_injector.", "frame_packer.", "cond_encoder.", "trainable_cond_mask."]
            where name.hasPrefix(prefix) {
            return name
        }
        return NFKMLXWanOriginalRelease.transformerModuleKey(forRelease: name)
    }

    /// A tensor in the module's layout: a 5-D convolution to `[out, t, h, w, in]` and a 3-D one to `[out, k, in]`.
    static func adapted(key: String, value: MLXArray) -> (String, MLXArray) {
        switch value.ndim {
        case 5: return (key, value.transposed(0, 2, 3, 4, 1))
        case 3 where !key.hasSuffix("modulation") && !key.hasSuffix("scale_shift_table"): return (key, value.transposed(0, 2, 1))
        default: return (key, value)
        }
    }

    /// Loads a release's transformer (a `config.json` and its safetensors shards) into `net`.
    static func loadWeights(into net: NFKMLXWanS2VNet, fromDirectory directory: URL,
                            precision: NFKMLXWeightPrecision = .float32) throws {
        let arrays = try NFKMLXReleaseWeights.materializedArrays(inDirectory: directory, precision: precision,
                                                                 remap: moduleKey(forRelease:), transform: adapted)
        try NFKMLXWeights.apply(arrays, to: net, verifyShapes: true)
    }
}
