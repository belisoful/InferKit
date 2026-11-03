//
//  NFKMLXLTX2DFRRecipe.swift
//  InferKitMLX
//
//  Introduced in InferKit 0.4.0.
//

import CoreGraphics
import Foundation
import InferKit
import MLX

// The stages of LTX-2.5's DFR recipe over an `NFKMLXLTX2Pipeline`: one DFR pass (`LTX2DFRPipeline`), one
// temporal refine round (`LTX2DFRTemporalRefinePipeline`), and the epilogue's keyframe rebuild. A caller
// composes them as the reference's 1080p recipe does: a pass at half the size generating slots, the spatial
// upsampler over the video and the slots, a pass at the full size with the IC-LoRA and the half-size
// reference, then one temporal round per frame-rate doubling, and an optional third spatial stage.
//
// Between stages latents travel RAW (denormalized), as the reference's `output_type="latent"` returns
// them: video `[1, F, H, W, C]`, slots `[1, K, H, W, C]`, audio PACKED `[1, L, C·M]`. A DFR canvas is the
// request padded to whole keyframe segments; ``NFKMLXLTX2DFRLayout/trimmedLatentFrames(frames:temporalCompression:)``
// drops the pad before a decode.

/// What a DFR stage returns.
public struct NFKMLXLTX2DFRResult {
    /// The raw video latent of the padded canvas `[1, F, H, W, C]`.
    public var video: MLXArray
    /// The raw keyframes the next stage carries `[1, K, H, W, C]`: the generated slots, or a round's merged bag.
    public var keyframes: MLXArray?
    /// The pixel frames of ``keyframes`` on this canvas.
    public var positions: [Int]
    /// The raw packed audio `[1, L, C·M]`.
    public var audio: MLXArray
    /// The canvas's pixel frames.
    public var canvasFrames: Int
}

extension NFKMLXLTX2Pipeline {

    var temporalCompression: Int { videoDecoder.configuration.temporalCompression }

    /// The audio latent frames a second: `sampleRate / hop / 4`.
    var audioLatentsPerSecond: Double {
        let c = audioDecoder.configuration
        return Double(c.sampleRate) / Double(c.melHopLength) / Double(NFKMLXLTX2AudioVAEConfiguration.compression)
    }

    /// A raw video latent normalized by the autoencoder's statistics, the reference's `(x − μ) · 1 / σ`.
    func normalizedVideo(_ raw: MLXArray) -> MLXArray { (raw.asType(.float32) - videoStatistics.mean) / videoStatistics.std }

    /// A normalized video latent returned to raw, `x · σ + μ`.
    func rawVideo(_ normalized: MLXArray) -> MLXArray { normalized * videoStatistics.std + videoStatistics.mean }

    /// Packed normalized audio returned to raw.
    func rawAudio(_ normalized: MLXArray) -> MLXArray { normalized * audioStatistics.std + audioStatistics.mean }

    /// One DFR pass, the reference's `LTX2DFRPipeline.__call__` with `output_type="latent"`.
    ///
    /// @discussion `frames` is the request; the pass pads it to whole keyframe segments and, where
    /// `generatesSlots`, appends a slot at each segment end (`slotPositions` where given), seeded from
    /// `slotInitials`. `latents` seeds the grid (a later stage's upsampled video). `audio` is renoised at
    /// `noiseScale` unless `freezesAudio`; nil draws it whole. `reference` is the IC-LoRA's half-size latent.
    /// `guidanceKeyframes` pins given keyframes (the epilogue's). `tiles` runs each step tile by tile.
    /// `sequenceNoise` and `audioNoise` draw the starting noise, in the reference's order.
    public func dfrPass(text: MLXArray, audioText: MLXArray, frames: Int, latentHeight: Int, latentWidth: Int,
                        frameRate: Float, sigmas: [Float], noiseScale given: Float? = nil, latents: MLXArray? = nil,
                        audio: MLXArray? = nil, freezesAudio: Bool = false, generatesSlots: Bool = true,
                        slotInitials: MLXArray? = nil, slotPositions: [Int]? = nil, reference: MLXArray? = nil,
                        referenceScale: Float = 2,
                        guidanceKeyframes: (latents: MLXArray, positions: [Int], strength: Float)? = nil,
                        conditions: [NFKMLXLTX2DFRCondition] = [], tiles: [NFKMLXLTX2DFRLayout.EpilogueTile]? = nil,
                        schedule: NFKMLXFlowMatchConfiguration, sequenceNoise: ([Int]) -> MLXArray,
                        audioNoise: ([Int]) -> MLXArray, onStep: ((Int) -> Void)? = nil) throws -> NFKMLXLTX2DFRResult {
        guard transformer.config.usesKeyframeEmbedding || !generatesSlots else {
            throw NFKMLXError.unsupportedConfiguration("DFR slots need a transformer with the keyframe embedding (LTX-2.5)")
        }
        let r = temporalCompression
        let noiseScale = given ?? sigmas[0]
        let canvas = try NFKMLXLTX2DFRLayout.canvas(frames: frames, temporalCompression: r)
        let slots = generatesSlots ? (slotPositions ?? canvas.positions) : []
        let pinned = guidanceKeyframes.map { guidance in
            guidance.positions.enumerated().map { index, position in
                (position, normalizedVideo(guidance.latents[0..., index ..< index + 1]), guidance.strength)
            }
        } ?? []
        let latentFrames = (canvas.frames - 1) / r + 1
        let audioFrames = Int((Double(canvas.frames) / Double(frameRate) * audioLatentsPerSecond).rounded(.toNearestOrEven))
        let fps = NFKMLXLTX2DFR.conditioningFPS(frameRate)

        let sequence = NFKMLXLTX2DFR.sequence(
            configuration: transformer.config, latentFrames: latentFrames, latentHeight: latentHeight,
            latentWidth: latentWidth, channels: transformer.config.inChannels, latents: latents.map(normalizedVideo),
            conditions: conditions, keyframes: pinned, slots: slots, slotInitials: slotInitials.map(normalizedVideo),
            reference: reference.map(normalizedVideo), referenceScale: referenceScale, fps: fps, noiseScale: noiseScale,
            noise: sequenceNoise)
        let plan = try tiles.map {
            try NFKMLXLTX2DFRLayout.tokenPlan($0, coordinates: sequence.coordinates, latentFrames: latentFrames,
                                              latentHeight: latentHeight, latentWidth: latentWidth)
        }
        let startAudio: MLXArray
        if let audio {
            let normalized = normalizedAudio(audio.asType(.float32))
            let scale = freezesAudio ? Float(0) : noiseScale
            startAudio = scale * audioNoise(normalized.shape).asType(.float32) + (1 - scale) * normalized
        } else {
            startAudio = audioNoise([1, audioFrames, transformer.config.audioInChannels]).asType(.float32)
        }
        let (packed, audioOut) = try NFKMLXLTX2DFR.denoise(
            sequence, audio: startAudio, transformer: transformer, text: text, audioText: audioText, sigmas: sigmas,
            sampler: .euler(schedule), freezesAudio: freezesAudio, fps: fps, tilePlan: plan, onStep: onStep)

        let video = rawVideo(packed[0..., 0 ..< sequence.gridTokens]
            .reshaped([1, latentFrames, latentHeight, latentWidth, -1]))
        var keyframes = sequence.slotTokens.map {
            rawVideo(packed[0..., $0].reshaped([1, slots.count, latentHeight, latentWidth, -1]))
        }
        var positions = slots
        if keyframes == nil, let guidanceKeyframes {
            keyframes = rawVideo(normalizedVideo(guidanceKeyframes.latents))
            positions = guidanceKeyframes.positions
        }
        return NFKMLXLTX2DFRResult(video: video, keyframes: keyframes, positions: positions, audio: rawAudio(audioOut),
                                   canvasFrames: canvas.frames)
    }

    /// The frozen stage-1 audio cut to one temporal tile's playback window and resampled to its token count,
    /// the reference's `_audio_window_for_tile`. `audio` is normalized and packed.
    static func audioWindow(_ audio: MLXArray, pixelStart: Int, tileFrames: Int, playbackFPS: Float, sourceSeconds: Double,
                            conditioningFPS: Float, audioLatentsPerSecond: Double) -> MLXArray {
        let sourceFrames = audio.dim(1)
        let count = Int((Double(tileFrames) / Double(conditioningFPS) * audioLatentsPerSecond).rounded(.toNearestOrEven))
        let start = Double(pixelStart) / Double(playbackFPS) / sourceSeconds * Double(sourceFrames)
        let span = Double(tileFrames) / Double(playbackFPS) / sourceSeconds * Double(sourceFrames)
        let stepSize = Float(span / Double(count)), origin = Float(start)
        var low = [Int32](), high = [Int32](), weight = [Float]()
        for index in 0 ..< count {
            let position = min(max(origin + stepSize * Float(index), 0), Float(sourceFrames - 1))
            let floor = Int(position.rounded(.down))
            low.append(Int32(floor))
            high.append(Int32(min(floor + 1, sourceFrames - 1)))
            weight.append(position - Float(floor))
        }
        let w = MLXArray(weight).reshaped([1, -1, 1])
        return audio.take(MLXArray(low), axis: 1) * (1 - w) + audio.take(MLXArray(high), axis: 1) * w
    }

    /// One temporal refine round, the reference's `LTX2DFRTemporalRefinePipeline.__call__` with
    /// `output_type="latent"`: the canvas doubled in time by `upsampler`, tiled on its keyframe seams, each tile
    /// denoised by the ancestral sampler over `sigmas` with its carried seams pinned at 0.95, a slot invented
    /// mid-segment, and the frozen stage-1 audio of its playback window; the tiles stitched by dropping each
    /// later tile's run-up, and the carried keyframes merged with the new slots.
    ///
    /// @discussion `video` and `keyframes` are the previous stage's raw output on a canvas of `frames` pixel
    /// frames playing at `frameRate`; `audio` is stage 1's raw packed audio and `sourceSeconds` its duration,
    /// fixed across rounds. `round` is 1-based. `conditions` are encoded at this round's size on the original
    /// request's frame grid; a round places each at `2^round` times its pixel frame, rebased per tile.
    /// `tileNoise(tile, shape)` draws a tile's starting noise and
    /// `ancestralNoise(tile)` the closure its sampler renoises with.
    public func temporalRound(video: MLXArray, keyframes: MLXArray, positions: [Int], audio: MLXArray, frames: Int,
                              frameRate: Float, sourceSeconds: Double? = nil, round: Int,
                              sigmas: [Float] = NFKMLXLTX2DFR.temporalSigmas, noiseScale given: Float? = nil,
                              upsampler: NFKMLXLTX2LatentUpsamplerNet, text: MLXArray, audioText: MLXArray,
                              conditions: [NFKMLXLTX2DFRCondition] = [], tileNoise: (Int, [Int]) -> MLXArray, ancestralNoise: (Int) -> ([Int]) -> MLXArray,
                              onStep: ((Int) -> Void)? = nil) throws -> NFKMLXLTX2DFRResult {
        guard transformer.config.usesKeyframeEmbedding else {
            throw NFKMLXError.unsupportedConfiguration("a temporal round invents keyframe slots, which needs LTX-2.5's transformer")
        }
        guard round >= 1 else { throw NFKMLXError.unsupportedConfiguration("temporal rounds count from 1") }
        let r = temporalCompression
        let noiseScale = given ?? sigmas[0]
        let source = sourceSeconds ?? Double(frames) / Double(frameRate)
        let (height, width, channels) = (video.dim(2), video.dim(3), video.dim(4))

        var canvasLatents = normalizedVideo(video)
        let carry = normalizedVideo(keyframes)
        let stageOneAudio = normalizedAudio(audio.asType(.float32))
        canvasLatents = normalizedVideo(Self.upsampled(rawVideo(canvasLatents), by: upsampler))
        let canvasFrames = 2 * (frames - 1) + 1
        let playback = 2 * frameRate
        let fps = NFKMLXLTX2DFR.conditioningFPS(playback)
        let seams = positions.map { $0 * 2 }
        let tiles = try NFKMLXLTX2DFRLayout.temporalTiles(seamPositions: seams, frames: canvasFrames,
                                                          tiles: 1 << round, temporalCompression: r)
        let seamIndex = Dictionary(uniqueKeysWithValues: seams.enumerated().map { ($1, $0) })

        var tileLatents = [MLXArray](), slotPositions = [Int](), slotLatents = [MLXArray]()
        var stepOffset = 0
        for (tileIndex, tile) in tiles.enumerated() {
            let latentCount = tile.interval.end - tile.interval.start
            let tileFrames = (latentCount - 1) * r + 1
            let tileVideo = canvasLatents[0..., tile.interval.start ..< tile.interval.end]
            let anchors = tile.anchors.map { position -> (Int, MLXArray, Float) in
                let index = seamIndex[position]!
                return (position - tile.pixelStart, carry[0..., index ..< index + 1], NFKMLXLTX2DFR.anchorStrength)
            }
            let localSlots = tile.slots.map { $0 - tile.pixelStart }
            let seeds = localSlots.map { min(Int((Double($0) / Double(r)).rounded(.toNearestOrEven)), latentCount - 1) }
            let initials = seeds.isEmpty ? nil : concatenated(seeds.map { tileVideo[0..., $0 ..< $0 + 1] }, axis: 1)
            let scale = 1 << round
            let tileConditions = conditions.compactMap { condition -> NFKMLXLTX2DFRCondition? in
                let pixel = condition.pixelFrame * scale
                guard tile.pixelStart <= pixel, pixel <= tile.pixelEnd else { return nil }
                var placed = condition
                placed.pixelFrame = pixel - tile.pixelStart
                return placed
            }
            let sequence = NFKMLXLTX2DFR.sequence(
                configuration: transformer.config, latentFrames: latentCount, latentHeight: height, latentWidth: width,
                channels: channels, latents: tileVideo, conditions: tileConditions, keyframes: anchors, slots: localSlots, slotInitials: initials,
                fps: fps, noiseScale: noiseScale, noise: { tileNoise(tileIndex, $0) })
            let window = Self.audioWindow(stageOneAudio, pixelStart: tile.pixelStart, tileFrames: tileFrames,
                                          playbackFPS: playback, sourceSeconds: source, conditioningFPS: fps,
                                          audioLatentsPerSecond: audioLatentsPerSecond)
            let offset = stepOffset
            let (packed, _) = try NFKMLXLTX2DFR.denoise(
                sequence, audio: window, transformer: transformer, text: text, audioText: audioText, sigmas: sigmas,
                sampler: .ancestral(eta: NFKMLXLTX2DFR.temporalEta), freezesAudio: true, fps: fps,
                noise: ancestralNoise(tileIndex), onStep: onStep.map { report in { report(offset + $0) } })
            stepOffset += sigmas.count
            tileLatents.append(packed[0..., 0 ..< sequence.gridTokens].reshaped([1, latentCount, height, width, -1]))
            if let slotRange = sequence.slotTokens {
                slotPositions += tile.slots
                slotLatents.append(packed[0..., slotRange].reshaped([1, localSlots.count, height, width, -1]))
            }
            eval(tileLatents.last!)
        }

        let stitched = concatenated(zip(tileLatents, tiles).map { $0[0..., $1.interval.leftRamp...] }, axis: 1)
        guard stitched.dim(1) == (canvasFrames - 1) / r + 1 else {
            throw NFKMLXError.unsupportedConfiguration("round \(round) stitched \(stitched.dim(1)) latent frames")
        }
        var bag = [Int: MLXArray]()
        for (index, position) in seams.enumerated() { bag[position] = carry[0..., index ..< index + 1] }
        let allSlots = slotLatents.isEmpty ? nil : concatenated(slotLatents, axis: 1)
        var firstSlot = [Int: Int]()
        for (index, position) in slotPositions.enumerated() where firstSlot[position] == nil { firstSlot[position] = index }
        for (position, index) in firstSlot { bag[position] = allSlots![0..., index ..< index + 1] }
        let carried = bag.keys.sorted()
        return NFKMLXLTX2DFRResult(
            video: rawVideo(stitched), keyframes: rawVideo(concatenated(carried.map { bag[$0]! }, axis: 1)),
            positions: carried, audio: rawAudio(stageOneAudio), canvasFrames: canvasFrames)
    }

    /// The carried keyframes rebuilt at twice their size by way of RGB, the reference's
    /// `rebuild_epilogue_keyframes`: each plane decoded alone, rounded to 8 bits, stretched ×2 by Pillow's
    /// Lanczos, and encoded again (the posterior's mean). `keyframes` is raw `[1, K, H, W, C]`; the result is raw
    /// `[1, K, 2 H, 2 W, C]`.
    public func rebuiltKeyframes(_ keyframes: MLXArray, encoder: NFKMLXLTX2VideoEncoderNet) -> MLXArray {
        var planes = [MLXArray]()
        for index in 0 ..< keyframes.dim(1) {
            let (rgb, width, height) = keyframeBytes(keyframes[0..., index ..< index + 1])
            let stretched = NFKMLXPillowLanczos.resize(rgb, width: width, height: height, outWidth: width * 2,
                                                       outHeight: height * 2)
            planes.append(encodedBytes(stretched, width: width * 2, height: height * 2, encoder: encoder))
            eval(planes.last!)
        }
        return concatenated(planes, axis: 1)
    }

    /// One raw keyframe `[1, 1, H, W, C]` decoded alone and rounded to 8-bit RGB, row-major.
    func keyframeBytes(_ plane: MLXArray) -> (rgb: [UInt8], width: Int, height: Int) {
        let frame = decodeLatent(rawVideo(normalizedVideo(plane)))[0, 0]
        let rgb = clip(frame, min: -1, max: 1).asArray(Float.self).map { UInt8((($0 + 1) * 127.5).rounded(.toNearestOrEven)) }
        return (rgb, frame.dim(1), frame.dim(0))
    }

    /// 8-bit RGB encoded to the posterior's mean, raw `[1, 1, h, w, C]`.
    func encodedBytes(_ rgb: [UInt8], width: Int, height: Int, encoder: NFKMLXLTX2VideoEncoderNet) -> MLXArray {
        let pixels = MLXArray(rgb.map { Float($0) / 127.5 - 1 }).reshaped([1, 1, height, width, 3])
        let moments = encoder(pixels.asType(encoder.convIn.conv.weight.dtype))
        return moments[.ellipsis, 0 ..< encoder.configuration.latentChannels].asType(.float32)
    }
}

/// A still the DFR passes condition on, the reference's `LTX2VideoCondition` with a single frame and no H.264
/// re-compression (`crf=0`).
public struct NFKMLXLTX2ImageCondition {
    /// The picture `[H, W, 3]`, RGB in `0…255`.
    public var pixels: MLXArray
    /// The latent frame it conditions; a negative index counts from the clip's end, and 0 replaces the first
    /// frame in place.
    public var latentIndex: Int
    public var strength: Float

    public init(pixels: MLXArray, latentIndex: Int = 0, strength: Float = 1) {
        self.pixels = pixels
        self.latentIndex = latentIndex
        self.strength = strength
    }

    /// The condition from a picture's sRGB pixels, its alpha dropped.
    public init(image: CGImage, latentIndex: Int = 0, strength: Float = 1) {
        let (bytes, width, height) = NFKMLXImageBridge.rgbaBytes(from: image, colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!)
        let rgba = MLXArray(bytes).reshaped([height, width, 4])
        self.init(pixels: rgba[0..., 0..., 0 ..< 3].asType(.float32), latentIndex: latentIndex, strength: strength)
    }

    /// The picture resized so it covers `width × height` and center-cropped to it, in `−1…1`, the reference's
    /// `preprocess_conditions`: torch's bilinear interpolation without antialiasing (`align_corners=False`),
    /// the row pass weighted after the column pass, in float32.
    func prepared(width: Int, height: Int) -> MLXArray {
        let source = pixels.asType(.float32)
        let (sourceHeight, sourceWidth) = (source.dim(0), source.dim(1))
        let scale = max(Double(height) / Double(sourceHeight), Double(width) / Double(sourceWidth))
        let (newHeight, newWidth) = (Int((Double(sourceHeight) * scale).rounded(.up)), Int((Double(sourceWidth) * scale).rounded(.up)))
        let (top, left) = ((newHeight - height) / 2, (newWidth - width) / 2)
        func indices(inSize: Int, outSize: Int, from start: Int, count: Int) -> (low: [Int32], high: [Int32], lowWeight: [Float], highWeight: [Float]) {
            let ratio = Float(inSize) / Float(outSize)
            var low = [Int32](), high = [Int32](), w0 = [Float](), w1 = [Float]()
            for dst in start ..< start + count {
                let real = max(ratio * (Float(dst) + 0.5) - 0.5, 0)
                let index = min(Int(real.rounded(.down)), inSize - 1)
                let lambda = min(max(real - Float(index), 0), 1)
                low.append(Int32(index))
                high.append(Int32(index + (index < inSize - 1 ? 1 : 0)))
                w1.append(lambda)
                w0.append(1 - lambda)
            }
            return (low, high, w0, w1)
        }
        let rows = indices(inSize: sourceHeight, outSize: newHeight, from: top, count: height)
        let cols = indices(inSize: sourceWidth, outSize: newWidth, from: left, count: width)
        func across(_ image: MLXArray) -> MLXArray {
            image.take(MLXArray(cols.low), axis: 1) * MLXArray(cols.lowWeight).reshaped([1, -1, 1])
                + image.take(MLXArray(cols.high), axis: 1) * MLXArray(cols.highWeight).reshaped([1, -1, 1])
        }
        let upper = across(source.take(MLXArray(rows.low), axis: 0))
        let lower = across(source.take(MLXArray(rows.high), axis: 0))
        let resized = upper * MLXArray(rows.lowWeight).reshaped([-1, 1, 1]) + lower * MLXArray(rows.highWeight).reshaped([-1, 1, 1])
        return (resized / 127.5 - 1).reshaped([1, 1, height, width, 3])
    }
}

extension NFKMLXLTX2Pipeline {

    /// `conditions` encoded for a pass of `frames` pixel frames at `width × height` pixels, the reference's
    /// `encode_conditions`: each still prepared, encoded to the posterior's mean, and normalized, at the pixel
    /// frame its latent index names on that pass (0 for latent 0, `8 (i − 1) + 1` otherwise). A latent index
    /// past the pass is skipped, as the reference skips it.
    public func encodedConditions(_ conditions: [NFKMLXLTX2ImageCondition], width: Int, height: Int, frames: Int,
                                  encoder: NFKMLXLTX2VideoEncoderNet) -> [NFKMLXLTX2DFRCondition] {
        let r = temporalCompression
        let latentFrames = (frames - 1) / r + 1
        return conditions.compactMap { condition in
            var index = condition.latentIndex
            if index < 0 { index = ((index % latentFrames) + latentFrames) % latentFrames }
            guard index < latentFrames else { return nil }
            let pixels = condition.prepared(width: width, height: height)
            let moments = encoder(pixels.asType(encoder.convIn.conv.weight.dtype))
            let latent = normalizedVideo(moments[.ellipsis, 0 ..< encoder.configuration.latentChannels])
            eval(latent)
            return NFKMLXLTX2DFRCondition(pixelFrame: index == 0 ? 0 : (index - 1) * r + 1, latent: latent,
                                          strength: condition.strength, pixelFrames: 1)
        }
    }
}
