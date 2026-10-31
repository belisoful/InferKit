//
//  NFKMLXWanAnimatePipeline.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXRandom

// The Wan 2.2 Animate 2 glue between the encoders and the clip: a driving video cut into segments of a
// fixed length that overlap by a few carried frames, each segment denoised on its own. The glue is
// measured against diffusers' modular `WanAnimate2Blocks` (`run_reference.py wan_animate_pipeline`) over a
// release directory of small random weights in the release's own layout.
//
// Per segment the reference runs, in order:
// - the driving slice VAE-encoded on its own (the autoencoder is causal in time, so encoding the whole
//   video once and slicing the latents is a different computation) under an all-ones mask;
// - the previous segment's last carried frames, zeros after them (all zeros for the first segment),
//   VAE-encoded under a mask that marks the carried frames, behind the reference image's latents;
// - fresh noise, a reset schedule, and a fresh key/value cache;
// - the extract pass over the driving latents, conditioned on the reference prompt and the CLIP features
//   of the driving video's first frame;
// - the guided denoising loop over the cache, its unconditional pass skipping one block;
// - the decode of every latent frame after the reference's, the first segment kept whole and every later
//   one without its carried frames.
//
// Motion continuity crosses a segment boundary in pixel space, which is why each segment decodes inside
// the loop.

/// How a driving video divides into segments.
struct NFKWanAnimateSegments: Equatable {
    /// The driving video's frames before padding.
    let realFrames: Int
    /// The frames each segment denoises.
    let segmentLength: Int
    /// The frames each segment after the first carries over from the one before it.
    let carriedFrames: Int
    /// The frames the zigzag padding appends: the last real frames, reversed.
    let paddingFrames: Int
    /// The number of segments.
    let count: Int

    /// The frames each segment advances by.
    var advance: Int { segmentLength - carriedFrames }

    /// The geometry the reference's video preprocess derives. Its frame count is floor-divided, so a video
    /// no longer than the carried frames has no segment.
    init(realFrames: Int, segmentLength: Int, carriedFrames: Int) throws {
        guard segmentLength > carriedFrames, carriedFrames >= 0 else {
            throw NFKMLXError.unsupportedConfiguration(
                "a segment of \(segmentLength) frames cannot carry \(carriedFrames) of them")
        }
        let advance = segmentLength - carriedFrames
        let leftover = realFrames > carriedFrames ? (realFrames - carriedFrames) % advance : 0
        let padding = leftover > 0 ? advance - leftover : 0
        let numerator = realFrames + padding - carriedFrames + advance - 1
        let count = numerator >= 0 ? numerator / advance : -((-numerator + advance - 1) / advance)
        guard count > 0 else {
            throw NFKMLXError.unsupportedConfiguration(
                "a driving video of \(realFrames) frames is no longer than the \(carriedFrames) carried frames")
        }
        // The reference mirrors the tail with a slice that wraps below zero; a video shorter than its own
        // padding has no frames to mirror.
        guard padding <= realFrames else {
            throw NFKMLXError.unsupportedConfiguration(
                "a driving video of \(realFrames) frames is shorter than the \(padding) frames its last segment mirrors")
        }
        self.realFrames = realFrames
        self.segmentLength = segmentLength
        self.carriedFrames = carriedFrames
        self.paddingFrames = padding
        self.count = count
    }

    /// The index of each frame of the padded video in the real one: the frames in order, then the last
    /// `paddingFrames` of them reversed.
    var paddedIndices: [Int] {
        Array(0 ..< realFrames) + (realFrames - paddingFrames ..< realFrames).reversed()
    }
}

/// The reference image and driving video brought to one frame size.
enum NFKWanAnimatePreprocess {

    /// The frame `(height, width)` a reference image of `imageWidth` × `imageHeight` resolves to for a
    /// target area of `width` × `height`: the aspect ratio of the image's own size rounded down to
    /// multiples of `modulus`, applied to the target area, and each side rounded down to that modulus.
    static func frameSize(imageWidth: Int, imageHeight: Int, width: Int, height: Int,
                          modulus: Int = 16) -> (height: Int, width: Int) {
        let area = Double(imageWidth * imageHeight)
        let imageAspect = Double(imageHeight) / Double(imageWidth)
        let roundedHeight = Int((area * imageAspect).squareRoot().rounded(.toNearestOrEven)) / modulus * modulus
        let roundedWidth = Int((area / imageAspect).squareRoot().rounded(.toNearestOrEven)) / modulus * modulus
        let aspect = Double(roundedHeight) / Double(roundedWidth)
        let target = Double(height * width)
        return (Int((target * aspect).squareRoot()) / modulus * modulus,
                Int((target / aspect).squareRoot()) / modulus * modulus)
    }

    /// The `(top, left, height, width)` box of the image's content inside a letterboxed frame, from the
    /// image's size rounded down to multiples of `modulus` as ``frameSize(imageWidth:imageHeight:width:height:modulus:)``
    /// rounds it. The final clip is cropped to it.
    static func cropRegion(imageWidth: Int, imageHeight: Int, frameHeight: Int, frameWidth: Int,
                           modulus: Int = 16) -> (top: Int, left: Int, height: Int, width: Int) {
        let area = Double(imageWidth * imageHeight)
        let imageAspect = Double(imageHeight) / Double(imageWidth)
        let ih = Int((area * imageAspect).squareRoot().rounded(.toNearestOrEven)) / modulus * modulus
        let iw = Int((area / imageAspect).squareRoot().rounded(.toNearestOrEven)) / modulus * modulus
        let wider = Double(frameWidth) / Double(frameHeight) < Double(iw) / Double(ih)
        let cropWidth = wider ? frameWidth : iw * frameHeight / ih
        let cropHeight = wider ? ih * frameWidth / iw : frameHeight
        return ((frameHeight - cropHeight) / 2, (frameWidth - cropWidth) / 2, cropHeight, cropWidth)
    }

    /// An RGB image resized to fit `width` × `height` with its aspect ratio kept, by PIL's 8-bit `filter`,
    /// and pasted on black at `((width − w) / 2, (height − h) / 2)`; `[height, width, 3]` in −1…1.
    static func letterboxed(_ rgb: [UInt8], width imageWidth: Int, height imageHeight: Int,
                            toWidth width: Int, toHeight height: Int,
                            filter: NFKMLXPILResample.Filter) -> MLXArray {
        let wider = Double(width) / Double(height) < Double(imageWidth) / Double(imageHeight)
        let contentWidth = wider ? width : imageWidth * height / imageHeight
        let contentHeight = wider ? imageHeight * width / imageWidth : height
        let resized = (contentWidth == imageWidth && contentHeight == imageHeight) ? rgb
            : NFKMLXPILResample.resampled(rgb, width: imageWidth, height: imageHeight, toWidth: contentWidth,
                                          toHeight: contentHeight, filter: filter)
        let top = (height - contentHeight) / 2, left = (width - contentWidth) / 2
        var canvas = [Float](repeating: -1, count: width * height * 3)
        for row in 0 ..< contentHeight {
            for column in 0 ..< contentWidth {
                let from = (row * contentWidth + column) * 3, to = ((row + top) * width + column + left) * 3
                for channel in 0 ..< 3 {
                    canvas[to + channel] = 2 * (Float(resized[from + channel]) / 255) - 1
                }
            }
        }
        return MLXArray(canvas).reshaped([height, width, 3])
    }

    /// The source frames a clip of `count` frames at `sourceRate` keeps when resampled to `targetRate`:
    /// the nearest frame to each target time, ties to even as NumPy rounds.
    static func frameIndices(count: Int, sourceRate: Double, targetRate: Double) -> [Int] {
        let kept = Int(Double(count) / sourceRate * targetRate)
        return (0 ..< max(kept, 0)).map { index in
            let nearest = Int((Double(index) / targetRate * sourceRate).rounded(.toNearestOrEven))
            return min(max(nearest, 0), count - 1)
        }
    }
}

/// The sampler a Wan Animate release names in its `scheduler/scheduler_config.json`.
public enum NFKMLXWanAnimateSchedule: Sendable {
    /// `DPMSolverMultistepScheduler` in its flow-prediction configuration, the standard release's.
    case dpmSolver(NFKMLXDPMSolverConfiguration)
    /// `FlowMatchEulerDiscreteScheduler` at a static shift, the distilled release's.
    case flowMatchEuler(NFKMLXFlowMatchConfiguration)

    /// The standard release's sampler: DPM-Solver++ at a flow shift of 5.
    public static let standard = NFKMLXWanAnimateSchedule.dpmSolver(NFKMLXDPMSolverConfiguration(flowShift: 5))

    /// The sampler a `scheduler_config.json` states. A DPM-Solver++ configuration other than the
    /// second-order midpoint flow one the port implements is refused, as is a dynamic Euler shift.
    static func reading(_ url: URL) throws -> NFKMLXWanAnimateSchedule {
        let json = try configuration(url)
        let train = json["num_train_timesteps"] as? Int ?? 1000
        switch json["_class_name"] as? String {
        case "DPMSolverMultistepScheduler":
            guard json["algorithm_type"] as? String ?? "dpmsolver++" == "dpmsolver++",
                  json["solver_type"] as? String ?? "midpoint" == "midpoint",
                  json["prediction_type"] as? String == "flow_prediction",
                  (json["use_flow_sigmas"] as? NSNumber)?.boolValue == true,
                  json["final_sigmas_type"] as? String ?? "zero" == "zero" else {
                throw NFKMLXError.unsupportedConfiguration(
                    "the release's DPM-Solver++ is not the second-order midpoint flow configuration")
            }
            return .dpmSolver(NFKMLXDPMSolverConfiguration(
                flowShift: (json["flow_shift"] as? NSNumber)?.floatValue ?? 1,
                solverOrder: json["solver_order"] as? Int ?? 2, trainTimesteps: train))
        case "FlowMatchEulerDiscreteScheduler":
            guard (json["use_dynamic_shifting"] as? NSNumber)?.boolValue != true,
                  json["shift_terminal"] == nil || json["shift_terminal"] is NSNull else {
                throw NFKMLXError.unsupportedConfiguration("the release's Euler sampler shifts dynamically")
            }
            return .flowMatchEuler(NFKMLXFlowMatchConfiguration(
                trainTimesteps: train, baseShift: (json["shift"] as? NSNumber)?.floatValue ?? 1,
                shiftTerminal: nil, useDynamicShifting: false))
        case let other:
            throw NFKMLXError.unsupportedConfiguration("the release samples with \(other ?? "an unnamed scheduler")")
        }
    }
}

extension NFKMLXWanAnimateSchedule {
    /// A `scheduler_config.json` as an object. The released configuration writes `-Infinity`
    /// (`lambda_min_clipped`), which only a JSON5 read takes.
    static func configuration(_ url: URL) throws -> [String: Any] {
        guard let json = try JSONSerialization.jsonObject(with: try Data(contentsOf: url),
                                                          options: [.json5Allowed]) as? [String: Any] else {
            throw NFKMLXError.unsupportedConfiguration("\(url.lastPathComponent) is not an object")
        }
        return json
    }
}

/// One segment's sampler state.
private enum NFKWanAnimateSampler {
    case dpmSolver(NFKMLXDPMSolverScheduler)
    case flowMatchEuler(NFKMLXFlowMatchScheduler)

    init(_ schedule: NFKMLXWanAnimateSchedule, steps: Int) {
        switch schedule {
        case .dpmSolver(let configuration):
            var scheduler = NFKMLXDPMSolverScheduler(configuration)
            scheduler.setTimesteps(steps)
            self = .dpmSolver(scheduler)
        case .flowMatchEuler(let configuration):
            var scheduler = NFKMLXFlowMatchScheduler(configuration)
            scheduler.setTimesteps(steps, sequenceLength: 0)
            self = .flowMatchEuler(scheduler)
        }
    }

    var timesteps: [Float] {
        switch self {
        case .dpmSolver(let scheduler): return scheduler.timesteps
        case .flowMatchEuler(let scheduler): return scheduler.timesteps
        }
    }

    mutating func step(velocity: MLXArray, sample: MLXArray, index: Int) -> MLXArray {
        switch self {
        case .dpmSolver(var scheduler):
            let next = scheduler.step(velocity: velocity, sample: sample, index: index)
            self = .dpmSolver(scheduler)
            return next
        case .flowMatchEuler(let scheduler):
            return scheduler.step(velocity: velocity, sample: sample, index: index)
        }
    }
}

/// Holds the pipeline's modules for capture across an async boundary.
private final class NFKWanAnimatePipelineHolder: @unchecked Sendable {
    let transformer: NFKMLXWanAnimateNet
    let vae: NFKMLXWanVideoVAENet
    init(_ transformer: NFKMLXWanAnimateNet, _ vae: NFKMLXWanVideoVAENet) {
        self.transformer = transformer
        self.vae = vae
    }
}

/// The Wan 2.2 Animate pipeline after the encoders: the segment loop from a letterboxed reference image
/// and driving video to a clip.
///
/// @discussion The caller supplies the umT5 features of the prompt, the negative prompt, and the
/// reference prompt, and the CLIP features of the reference image and of the driving video's first frame;
/// ``NFKMLXWanAnimateGenerator`` runs the whole release. Introduced in InferKit 0.4.0.
public final class NFKMLXWanAnimatePipeline {

    private let holder: NFKWanAnimatePipelineHolder
    private let latentsMean: MLXArray
    private let latentsReciprocalStd: MLXArray
    private let schedule: NFKMLXWanAnimateSchedule

    /// The Wan 2.1 autoencoder's per-channel latent statistics, which the original Wan code states and
    /// the release's `vae.pth` does not carry.
    static let wan21LatentsMean: [Float] = [
        -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
        0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
    ]
    static let wan21LatentsStd: [Float] = [
        2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
        3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
    ]

    /// Builds a pipeline from the transformer and the autoencoder.
    ///
    /// `latentsMean` and `latentsStd` are the autoencoder's per-channel latent statistics, the Wan 2.1
    /// values by default. An encode standardizes by them and a decode undoes it.
    public init(transformer: NFKMLXWanAnimateNet, vae: NFKMLXWanVideoVAENet,
                latentsMean: MLXArray? = nil, latentsStd: MLXArray? = nil,
                schedule: NFKMLXWanAnimateSchedule = .standard) {
        holder = NFKWanAnimatePipelineHolder(transformer, vae)
        self.latentsMean = latentsMean ?? MLXArray(Self.wan21LatentsMean)
        // The reference multiplies by the reciprocal standard deviation and divides by it to undo.
        self.latentsReciprocalStd = 1 / (latentsStd ?? MLXArray(Self.wan21LatentsStd))
        self.schedule = schedule
    }

    var transformer: NFKMLXWanAnimateNet { holder.transformer }
    var vae: NFKMLXWanVideoVAENet { holder.vae }

    /// The autoencoder's spatial and temporal compression.
    var compression: (spatial: Int, temporal: Int) {
        NFKMLXWanVideoGenerator.compression(of: vae.configuration)
    }

    // MARK: Latents

    /// The standardized latent `[zDim, T', H', W']` of a clip `[T, H, W, 3]` in −1…1: the autoencoder's
    /// deterministic mean.
    public func encode(_ clip: MLXArray) -> MLXArray {
        let dtype = NFKReferenceRounding.parameterType(of: vae)
        let latent = vae.encode(clip.expandedDimensions(axis: 0).asType(dtype))[0].asType(.float32)
        let standardized = (latent - latentsMean) * latentsReciprocalStd
        eval(standardized)
        return standardized.transposed(3, 0, 1, 2)
    }

    /// The clip `[T, H, W, 3]` in −1…1 a standardized latent `[zDim, T', H', W']` decodes to.
    public func decode(_ latent: MLXArray) -> MLXArray {
        let dtype = NFKReferenceRounding.parameterType(of: vae)
        let channelsLast = latent.asType(.float32).transposed(1, 2, 3, 0)
        let destandardized = (channelsLast / latentsReciprocalStd + latentsMean).asType(dtype)
        let clip = vae.decode(destandardized.expandedDimensions(axis: 0))[0].asType(.float32)
        eval(clip)
        return clip
    }

    /// The image-to-video mask `[4, latentFrames, height, width]` marking the first `frames` pixel frames:
    /// the first latent frame stands for four copies of pixel frame 0, and each later one for four pixel
    /// frames in turn.
    static func conditionMask(latentFrames: Int, height: Int, width: Int, frames: Int) -> MLXArray {
        var values = [Float](repeating: 0, count: 4 * latentFrames)
        for channel in 0 ..< 4 {
            for frame in 0 ..< latentFrames {
                let expanded = 4 * frame + channel
                let pixel = expanded < 4 ? 0 : expanded - 3
                values[channel * latentFrames + frame] = pixel < frames ? 1 : 0
            }
        }
        return broadcast(MLXArray(values).reshaped([4, latentFrames, 1, 1]), to: [4, latentFrames, height, width])
    }

    // MARK: Segment loop

    /// Animates a letterboxed reference image with a letterboxed, zigzag-padded driving video.
    ///
    /// - Parameters:
    ///   - reference: the reference image `[H, W, 3]` in −1…1.
    ///   - driving: the padded driving video `[segments.realFrames + segments.paddingFrames, H, W, 3]` in −1…1.
    ///   - segments: how the driving video divides.
    ///   - prompt: the prompt's umT5 features `[tokens, textDim]`.
    ///   - negativePrompt: the negative prompt's features, read where `guidance` is not 1.
    ///   - referencePrompt: the reference prompt's features, which condition the extract pass.
    ///   - imageEmbeddings: the CLIP features of the reference image `[tokens, 1280]`.
    ///   - drivingImageEmbeddings: the CLIP features of the driving video's first frame.
    ///   - steps: the denoising steps of each segment.
    ///   - guidance: the classifier-free guidance scale; 1 runs the conditional pass alone.
    ///   - noise: each segment's starting noise for its index and shape.
    ///   - progress: called with the fraction of steps done.
    /// - Returns: the clip `[segments.realFrames, H, W, 3]` in −1…1.
    func animate(reference: MLXArray, driving: MLXArray, segments: NFKWanAnimateSegments,
                 prompt: MLXArray, negativePrompt: MLXArray?, referencePrompt: MLXArray,
                 imageEmbeddings: MLXArray, drivingImageEmbeddings: MLXArray, steps: Int, guidance: Float,
                 noise: (Int, [Int]) -> MLXArray, progress: (Double) -> Void = { _ in }) throws -> MLXArray {
        let (spatial, temporal) = compression
        let height = reference.dim(0), width = reference.dim(1)
        guard driving.dim(1) == height, driving.dim(2) == width else {
            throw NFKMLXError.unsupportedConfiguration(
                "the driving video is \(driving.dim(1))×\(driving.dim(2)), the reference image \(height)×\(width)")
        }
        let latentHeight = height / spatial, latentWidth = width / spatial
        let length = segments.segmentLength
        let latentFrames = (length - 1) / temporal + 1
        let patch = transformer.config.patchSize
        let referenceGrid = (latentFrames / patch[0], latentHeight / patch[1], latentWidth / patch[2])
        let videoFrames = length / 4 + 1, videoArea = height * width / 256

        let dtype = NFKReferenceRounding.parameterType(of: transformer)
        let guides = guidance != 1
        guard !guides || negativePrompt != nil else {
            throw NFKMLXError.unsupportedConfiguration("a guided clip needs the negative prompt's features")
        }
        let text = prompt.asType(dtype), negative = negativePrompt?.asType(dtype)
        let referenceText = referencePrompt.asType(dtype)
        let image = imageEmbeddings.asType(dtype), drivingImage = drivingImageEmbeddings.asType(dtype)

        let referenceLatent = encode(reference.expandedDimensions(axis: 0))  // [z, 1, h, w]
        let referenceCondition = concatenated(
            [Self.conditionMask(latentFrames: 1, height: latentHeight, width: latentWidth, frames: 1),
             referenceLatent], axis: 0)
        eval(referenceCondition)

        var pieces = [MLXArray]()
        var previous: MLXArray?
        for segment in 0 ..< segments.count {
            let start = segment * segments.advance
            let drivingLatent = encode(driving[start ..< (start + length)])
            let drivingCondition = concatenated(
                [Self.conditionMask(latentFrames: drivingLatent.dim(1), height: latentHeight, width: latentWidth,
                                    frames: length), drivingLatent], axis: 0)

            let carried = segment > 0 ? segments.carriedFrames : 0
            var conditionPixels = MLXArray.zeros([length, height, width, 3])
            if carried > 0, let previous {
                let tail = previous[(previous.dim(0) - carried)...]
                conditionPixels = concatenated([tail, MLXArray.zeros([length - carried, height, width, 3])], axis: 0)
            }
            let carriedLatent = encode(conditionPixels)
            let generationCondition = concatenated(
                [referenceCondition,
                 concatenated([Self.conditionMask(latentFrames: carriedLatent.dim(1), height: latentHeight,
                                                  width: latentWidth, frames: carried), carriedLatent], axis: 0)],
                axis: 1)
            eval(drivingCondition, generationCondition)

            var latent = noise(segment, [vae.configuration.zDim, generationCondition.dim(1), latentHeight, latentWidth])
                .asType(.float32)
            var scheduler = NFKWanAnimateSampler(schedule, steps: steps)

            let cache = NFKMLXWanAnimateKVCache(layerCount: transformer.config.layers)
            let extracted = try transformer.extractReference(
                latent: drivingLatent.asType(dtype), condition: drivingCondition.asType(dtype), text: referenceText,
                imageEmbeddings: drivingImage, into: cache)
            eval(extracted)
            try verifyStream()

            for index in 0 ..< scheduler.timesteps.count {
                let step = MLXArray([scheduler.timesteps[index]])
                let input = latent.asType(dtype), condition = generationCondition.asType(dtype)
                let conditioned = try transformer.generate(
                    latent: input, condition: condition, text: text, imageEmbeddings: image, timestep: step,
                    cache: cache, referenceGrid: referenceGrid, videoFrames: videoFrames, videoArea: videoArea)
                var velocity = conditioned.asType(.float32)
                if guides, let negative {
                    let unconditioned = try transformer.generate(
                        latent: input, condition: condition, text: negative, imageEmbeddings: image, timestep: step,
                        cache: cache, referenceGrid: referenceGrid, videoFrames: videoFrames, videoArea: videoArea,
                        unconditional: true).asType(.float32)
                    velocity = NFKReferenceRounding.guided(unconditioned, velocity, scale: guidance)
                }
                latent = scheduler.step(velocity: velocity, sample: latent, index: index)
                eval(latent)
                try verifyStream()
                progress(Double(segment * steps + index + 1) / Double(segments.count * steps))
            }
            cache.clear()

            var decoded = decode(latent[0..., 1...])
            if segment > 0 {
                decoded = decoded[segments.carriedFrames...]
            }
            eval(decoded)
            pieces.append(decoded)
            previous = decoded
            NFKMLXGPU.clearCache()
        }
        return concatenated(pieces, axis: 0)[0 ..< segments.realFrames]
    }

    /// Throws the first failed read of a streamed transformer block since the last check, and clears it.
    func verifyStream() throws {
        try transformer.verifyStream()
    }
}
