//
//  NFKMLXWanDancerPipeline.swift
//  InferKitMLX
//
//  Wan-Dancer's sampler: the fork's `WanVideoPipeline.__call__` with the units the released scripts enable. The
//  keyframes condition the clip through their mask and latents, the reference image and the first keyframe
//  through CLIP, the music through the transformer's encoder; DiffSynth-Studio's flow-match schedule at shift 5 runs
//  classifier-free guidance whose unconditional pass skips block 9; the autoencoder encodes and decodes in
//  DiffSynth-Studio's blended spatial tiles, and the global stage decodes each latent frame alone.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import MLX
import MLXNN

/// One Wan-Dancer clip's conditions.
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXWanDancerClip {
    /// The keyframes `[F, H, W, 3]` in `0...1`, black where `keyframeMask` is false.
    public var keyframes: MLXArray
    /// Which of the clip's frames are keyframes.
    public var keyframeMask: [Bool]
    /// The reference image `[H, W, 3]` in `0...1`, at the clip's size.
    public var reference: MLXArray
    /// The music features `[T, 35]`.
    public var music: MLXArray
    /// The clip's frame rate: 30 for the local stage, `30 / k` for the global one.
    public var fps: Double

    public init(keyframes: MLXArray, keyframeMask: [Bool], reference: MLXArray, music: MLXArray, fps: Double) {
        self.keyframes = keyframes
        self.keyframeMask = keyframeMask
        self.reference = reference
        self.music = music
        self.fps = fps
    }
}

/// The Wan-Dancer sampler over one transformer, the Wan 2.1 autoencoder, and the CLIP image encoder.
final class NFKMLXWanDancerPipeline {
    let net: NFKMLXWanDancerNet
    let vae: NFKMLXWanVideoVAENet
    let latentsMean: MLXArray
    let latentsStd: MLXArray
    let imageEncoder: NFKMLXWanAnimateImageEncoderNet
    var steps = 48
    var shift: Float = 5
    var guidance: Float = 5
    /// Whether the unconditional pass skips block 9 (`enable_skip_layer`).
    var skipsBlock = true
    /// DiffSynth-Studio's tile `(height, width)` and stride in latent units.
    var tile = (30, 52)
    var tileStride = (15, 26)
    /// Called after each step with the step index and the latents it leaves.
    var stepObserver: ((Int, MLXArray) -> Void)?

    static let spatialCompression = 8

    init(net: NFKMLXWanDancerNet, vae: NFKMLXWanVideoVAENet, latentsMean: MLXArray, latentsStd: MLXArray,
         imageEncoder: NFKMLXWanAnimateImageEncoderNet) {
        self.net = net
        self.vae = vae
        self.latentsMean = latentsMean
        self.latentsStd = latentsStd
        self.imageEncoder = imageEncoder
    }

    var dtype: DType { NFKReferenceRounding.parameterType(of: net) }

    // MARK: Tiles

    /// DiffSynth-Studio's tile origins along one axis of `length` for tiles of `size` every `stride`: a tile
    /// whose predecessor already reaches the end is dropped.
    static func tileStarts(length: Int, size: Int, stride: Int) -> [Int] {
        Swift.stride(from: 0, to: length, by: stride).filter { !($0 - stride >= 0 && $0 - stride + size >= length) }
    }

    /// `build_1d_mask`: ones, ramped over `border` samples at each side that is not the edge of the whole.
    static func ramp(_ length: Int, leftEdge: Bool, rightEdge: Bool, border: Int) -> [Float] {
        var x = [Float](repeating: 1, count: length)
        if !leftEdge {
            for i in 0 ..< Swift.min(border, length) { x[i] = Float(i + 1) / Float(border) }
        }
        if !rightEdge {
            for i in 0 ..< Swift.min(border, length) { x[length - 1 - i] = Float(i + 1) / Float(border) }
        }
        return x
    }

    /// A tile's blending mask `[h, w]`: the smaller of its row and column ramps.
    static func tileMask(height: Int, width: Int, edges: (Bool, Bool, Bool, Bool), border: (Int, Int)) -> MLXArray {
        let rows = ramp(height, leftEdge: edges.0, rightEdge: edges.1, border: border.0)
        let columns = ramp(width, leftEdge: edges.2, rightEdge: edges.3, border: border.1)
        return minimum(MLXArray(rows).reshaped([height, 1]), MLXArray(columns).reshaped([1, width]))
    }

    /// `body` over DiffSynth-Studio's tiles of a `[H, W]` input, each output scaled `factor` times (or divided,
    /// for `factor < 1`), blended by the tiles' masks at `dtype`.
    func tiled(height: Int, width: Int, size: (Int, Int), stride: (Int, Int), outputScale: (Int, Int) -> (Int, Int),
               border: (Int, Int), dtype: DType, _ body: (Range<Int>, Range<Int>) -> MLXArray) -> MLXArray {
        var values: MLXArray?, weights: MLXArray?
        for h in Self.tileStarts(length: height, size: size.0, stride: stride.0) {
            for w in Self.tileStarts(length: width, size: size.1, stride: stride.1) {
                let rows = h ..< Swift.min(h + size.0, height), columns = w ..< Swift.min(w + size.1, width)
                let out = body(rows, columns)                                   // [C, F, h', w']
                let (th, tw) = (out.dim(2), out.dim(3))
                let mask = Self.tileMask(height: th, width: tw, edges: (h == 0, h + size.0 >= height, w == 0, w + size.1 >= width),
                                         border: border).asType(dtype)
                let (top, left) = outputScale(h, w)
                let (fullH, fullW) = outputScale(height, width)
                if values == nil {
                    values = MLXArray.zeros([out.dim(0), out.dim(1), fullH, fullW], dtype: dtype)
                    weights = MLXArray.zeros([1, 1, fullH, fullW], dtype: dtype)
                }
                values![0..., 0..., top ..< (top + th), left ..< (left + tw)] =
                    values![0..., 0..., top ..< (top + th), left ..< (left + tw)] + out.asType(dtype) * mask
                weights![0..., 0..., top ..< (top + th), left ..< (left + tw)] =
                    weights![0..., 0..., top ..< (top + th), left ..< (left + tw)] + mask
                eval(values!, weights!)
            }
        }
        return values! / weights!
    }

    /// The standardized latents `[16, F', h, w]` of a clip `[F, H, W, 3]` in `-1...1`, encoded in tiles.
    func encoded(_ clip: MLXArray) -> MLXArray {
        let s = Self.spatialCompression
        let vaeType = NFKReferenceRounding.parameterType(of: vae)
        let streamType = dtype
        return tiled(height: clip.dim(1), width: clip.dim(2), size: (tile.0 * s, tile.1 * s),
                     stride: (tileStride.0 * s, tileStride.1 * s), outputScale: { ($0 / s, $1 / s) },
                     border: ((tile.0 - tileStride.0), (tile.1 - tileStride.1)), dtype: streamType) { rows, columns in
            let latent = vae.encode(clip[0..., rows, columns].expandedDimensions(axis: 0).asType(vaeType))[0].asType(.float32)
            return ((latent - latentsMean) * (1 / latentsStd)).transposed(3, 0, 1, 2).asType(streamType)
        }
    }

    /// The clip `[F, H, W, 3]` in `-1...1` that standardized latents `[16, F', h, w]` decode to, decoded in tiles
    /// and clamped.
    func decoded(_ latents: MLXArray) -> MLXArray {
        let s = Self.spatialCompression
        let vaeType = NFKReferenceRounding.parameterType(of: vae)
        let blended = tiled(height: latents.dim(2), width: latents.dim(3), size: tile, stride: tileStride,
                            outputScale: { ($0 * s, $1 * s) }, border: ((tile.0 - tileStride.0) * s, (tile.1 - tileStride.1) * s),
                            dtype: latents.dtype) { rows, columns in
            let part = latents[0..., 0..., rows, columns].transposed(1, 2, 3, 0).expandedDimensions(axis: 0).asType(.float32)
            let unstandardized = part * latentsStd.reshaped([1, 1, 1, 1, -1]) + latentsMean.reshaped([1, 1, 1, 1, -1])
            return vae.decode(unstandardized.asType(vaeType), clamped: false)[0].asType(.float32).transposed(3, 0, 1, 2)
        }
        return clip(blended, min: -1, max: 1).transposed(1, 2, 3, 0)
    }

    /// The global stage's decode: each latent frame decoded alone (`enable_vae_decode_framewise`), one frame each.
    func decodedFramewise(_ latents: MLXArray) -> MLXArray {
        concatenated((0 ..< latents.dim(1)).map { decoded(latents[0..., $0 ..< ($0 + 1)]) }, axis: 0)
    }

    // MARK: Conditions

    /// The condition `[20, F', h, w]` (`WanVideoUnit_ImageKeyframesEmbedder`): the keyframe mask, its first frame
    /// repeated to four, folded four frames a channel, ahead of the keyframes' latents.
    func condition(keyframes: MLXArray, mask: [Bool]) -> MLXArray {
        let s = Self.spatialCompression
        let frames = keyframes.dim(0), h = keyframes.dim(1) / s, w = keyframes.dim(2) / s
        let flags = [mask[0], mask[0], mask[0]] + mask
        let latentFrames = flags.count / 4
        let folded = (0 ..< 4).map { channel in
            MLXArray((0 ..< latentFrames).map { flags[$0 * 4 + channel] ? Float(1) : 0 })
        }
        precondition(frames % 4 == 1, "a clip is 4k + 1 frames")
        let maskChannels = broadcast(stacked(folded).reshaped([4, latentFrames, 1, 1]), to: [4, latentFrames, h, w])
        let latents = encoded(keyframes * 2 - 1)
        return concatenated([maskChannels.asType(latents.dtype), latents], axis: 0)
    }

    /// The pass inputs of a clip, its CLIP features read from the first keyframe and the reference image.
    func inputs(_ clip: NFKMLXWanDancerClip) -> NFKMLXWanDancerInputs {
        let condition = self.condition(keyframes: clip.keyframes, mask: clip.keyframeMask)
        return NFKMLXWanDancerInputs(condition: condition, keyframeFeatures: imageEncoder.features(clip.keyframes[0] * 2 - 1),
                                     referenceFeatures: imageEncoder.features(clip.reference * 2 - 1),
                                     music: clip.music, fps: clip.fps)
    }

    // MARK: Sampling

    /// DiffSynth-Studio's `FlowMatchScheduler(shift:, sigma_min: 0, extra_one_step: true)` sigmas for `steps`:
    /// torch's float32 `linspace(1, 0, steps + 1)` without its last value, shifted.
    static func sigmas(steps: Int, shift: Float) -> [Float] {
        let count = steps + 1
        let step = Float(0 - 1) / Float(count - 1)
        let halfway = count / 2
        let linear = (0 ..< count).map { i -> Float in
            i < halfway ? 1 + step * Float(i) : 0 - step * Float(count - i - 1)
        }
        return linear.prefix(steps).map { shift * $0 / (1 + (shift - 1) * $0) }
    }

    /// Denoises `noise` `[16, F', h, w]` under the positive and negative text features with guidance.
    func denoise(_ noise: MLXArray, positive: MLXArray, negative: MLXArray, inputs: NFKMLXWanDancerInputs) -> MLXArray {
        let sigmas = Self.sigmas(steps: steps, shift: shift)
        var latents = noise.asType(dtype)
        for (index, sigma) in sigmas.enumerated() {
            let timestep = Double(sigma * 1000)
            var prediction = net(latents, text: positive, t: timestep, inputs: inputs)
            if guidance != 1 {
                let unconditional = net(latents, text: negative, t: timestep, inputs: inputs, skipsBlock: skipsBlock)
                prediction = unconditional + guidance * (prediction - unconditional)
            }
            let next: Float = index + 1 < sigmas.count ? sigmas[index + 1] : 0
            latents = latents + prediction * (next - sigma)
            eval(latents)
            stepObserver?(index, latents)
        }
        return latents
    }
}

extension NFKMLXWanAnimateImageEncoderNet {
    /// Builds and loads the image encoder from Wan's open-clip checkpoint
    /// (`models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth`): the vision tower's tensors renamed to
    /// transformers' `CLIPVisionModel` layout, the fused `to_qkv` split into the query, key, and value projections.
    /// The text tower, the projection head, and the logit scale stay unread.
    static func fromOpenCLIP(_ url: URL, precision: NFKMLXWeightPrecision = .float32) throws -> NFKMLXWanAnimateImageEncoderNet {
        let stored = try NFKMLXWeights.materializedCheckpoint(url: url, reading: { $0.hasPrefix("visual.") }).arrays
        let blocks = Set(stored.keys.compactMap { key -> Int? in
            let parts = key.split(separator: ".")
            return parts.count > 3 && parts[1] == "transformer" ? Int(parts[2]) : nil
        })
        guard let qkv = stored["visual.transformer.0.attn.to_qkv.weight"], let fc1 = stored["visual.transformer.0.mlp.0.weight"],
              let positions = stored["visual.pos_embedding"], let patch = stored["visual.patch_embedding.weight"] else {
            throw NFKMLXError.unsupportedConfiguration("\(url.lastPathComponent) holds no open-clip vision tower")
        }
        let width = qkv.dim(1), patchSize = patch.dim(2)
        let grid = Int(Double(positions.dim(1) - 1).squareRoot())
        let configuration = NFKMLXWanAnimateImageEncoderConfiguration(
            hiddenSize: width, layers: blocks.count, heads: width / 80, intermediateSize: fc1.dim(0),
            imageSize: grid * patchSize, patchSize: patchSize)
        var arrays = [(String, MLXArray)]()
        func add(_ name: String, _ value: MLXArray) {
            arrays.append((name, precision == .checkpoint || !value.dtype.isFloatingPoint ? value : value.asType(.float32)))
        }
        let tower = "vision_model."
        for (key, value) in stored where key.hasPrefix("visual.") {
            let parts = key.split(separator: ".").map(String.init)
            guard parts.count >= 2 else { continue }
            switch key {
            case "visual.patch_embedding.weight": add(tower + "embeddings.patch_embedding.weight", value.transposed(0, 2, 3, 1))
            case "visual.cls_embedding": add(tower + "embeddings.class_embedding", value.reshaped([width]))
            case "visual.pos_embedding": add(tower + "embeddings.position_embedding.weight", value[0])
            default:
                if (parts[1] == "pre_norm" || parts[1] == "post_norm") && parts.count == 3 {
                    add(tower + (parts[1] == "pre_norm" ? "pre_layrnorm." : "post_layernorm.") + parts[2], value)
                } else if parts[1] == "transformer" && parts.count > 3 {
                    let layer = tower + "encoder.layers.\(parts[2])."
                    let rest = parts[3...].joined(separator: ".")
                    switch rest {
                    case "attn.to_qkv.weight", "attn.to_qkv.bias":
                        let suffix = parts.last!
                        for (index, name) in ["q_proj", "k_proj", "v_proj"].enumerated() {
                            add(layer + "self_attn.\(name).\(suffix)", value[(index * width) ..< ((index + 1) * width)])
                        }
                    default:
                        let renamed = rest.replacingOccurrences(of: "norm1.", with: "layer_norm1.")
                            .replacingOccurrences(of: "norm2.", with: "layer_norm2.")
                            .replacingOccurrences(of: "attn.proj.", with: "self_attn.out_proj.")
                            .replacingOccurrences(of: "mlp.0.", with: "mlp.fc1.")
                            .replacingOccurrences(of: "mlp.2.", with: "mlp.fc2.")
                        add(layer + renamed, value)
                    }
                }
            }
        }
        let net = NFKMLXWanAnimateImageEncoderNet(configuration)
        try NFKMLXWeights.apply(arrays, to: net, verifyShapes: true)
        return net
    }
}
