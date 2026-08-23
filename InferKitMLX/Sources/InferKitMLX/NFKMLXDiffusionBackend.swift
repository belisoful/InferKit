//
//  NFKMLXDiffusionBackend.swift
//  InferKitMLX
//

import Foundation
import CoreGraphics
import Metal
import InferKit
import MLX

/// How the diffusion backend samples, reads its input image, and writes its output.
public struct NFKDiffusionConfiguration: @unchecked Sendable {
    /// Number of denoising steps. Few-step distilled models use 1–8; base SD uses 20–50.
    public var steps: Int
    /// Classifier-free guidance weight, handed to the denoise closure to apply.
    public var guidanceScale: Float
    /// Image-to-image / inpaint source retention, in `0...1`. `1` starts from full noise; a lower
    /// value skips the early (high-noise) steps so more of the source survives.
    public var strength: Float
    /// The latent's channel count. A pixel-space reference model uses 3 or 4; a VAE-latent model uses 4.
    public var latentChannels: Int
    /// The channel count `NFKInputImage` is bridged into (3 for RGB, 4 to carry alpha).
    public var plateChannels: Int
    /// Seed for the initial noise. A fixed value gives a repeatable result; `nil` uses a constant
    /// default so a run without a seed is still deterministic.
    public var seed: UInt64?
    /// Color space and premultiplication for the written image.
    public var imageOptions: NFKMLXImageOptions
    /// Return an `MTLTexture` instead of a `CGImage`.
    public var outputsTexture: Bool
    /// The device for texture output. Defaults to the system default device.
    public var device: MTLDevice?
    /// The result key the decoded image is stored under. Defaults to `NFKOutputImage`.
    public var outputKey: String

    public init(steps: Int = 20,
                guidanceScale: Float = 1,
                strength: Float = 1,
                latentChannels: Int = 3,
                plateChannels: Int = 3,
                seed: UInt64? = nil,
                imageOptions: NFKMLXImageOptions = NFKMLXImageOptions(),
                outputsTexture: Bool = false,
                device: MTLDevice? = nil,
                outputKey: String = NFKOutputImage) {
        self.steps = steps
        self.guidanceScale = guidanceScale
        self.strength = strength
        self.latentChannels = latentChannels
        self.plateChannels = plateChannels
        self.seed = seed
        self.imageOptions = imageOptions
        self.outputsTexture = outputsTexture
        self.device = device
        self.outputKey = outputKey
    }
}

/// What `encode` hands the loop: the conditioning the denoise closure reads, the latent size, and —
/// for image-to-image and inpainting — a clean source latent and a mask.
public struct NFKDiffusionContext {
    /// Conditioning tensors the denoise closure consumes (text embeddings, a control image, a target).
    public var conditioning: [String: MLXArray]
    /// The latent (and output) width in pixels.
    public var width: Int
    /// The latent (and output) height in pixels.
    public var height: Int
    /// A clean latent to noise from for image-to-image / inpainting. `nil` runs text-to-image.
    public var sourceLatent: MLXArray?
    /// An inpaint mask `[H, W, 1]`, `1` where the model regenerates and `0` where the source is kept.
    public var mask: MLXArray?
    /// A starting latent `[H, W, C]` for text-to-image, in place of the loop's own seeded noise. A
    /// caller reproducing another implementation's run supplies that implementation's initial latent
    /// here, which compares the sampler and the networks without also having to match a random source.
    public var initialLatent: MLXArray?

    public init(conditioning: [String: MLXArray] = [:],
                width: Int,
                height: Int,
                sourceLatent: MLXArray? = nil,
                mask: MLXArray? = nil,
                initialLatent: MLXArray? = nil) {
        self.conditioning = conditioning
        self.width = width
        self.height = height
        self.sourceLatent = sourceLatent
        self.mask = mask
        self.initialLatent = initialLatent
    }
}

/// An InferKit backend for a bring-your-own MLX diffusion model.
///
/// Where `NFKMLXModuleBackend` and `NFKMLXMattingBackend` run a single forward, a diffusion model
/// runs an iterative sampler: encode conditioning, start from noise, denoise over N steps through a
/// scheduler, then decode. This backend owns that loop, the InferKit contract (progress per step,
/// cancellation between steps), and the image bridge; the consumer supplies three closures and a
/// scheduler:
///
/// - `encode`: request + bridged input image `[H, W, plateChannels]` + optional mask `[H, W, 1]` →
///   an `NFKDiffusionContext` (conditioning, output size, optional source latent and mask).
/// - `denoise`: latent + timestep + context + guidance → the model's prediction for that step.
/// - `decode`: the final latent → an image tensor `[H, W, C]` in `0...1`.
/// - `scheduler`: the sampler (see `NFKDDIMScheduler`).
///
/// Text-to-image runs when `encode` returns no source latent; a source latent starts image-to-image
/// (with `NFKParameterStrength`); a source latent and a mask run inpainting, compositing the kept
/// region back each step. The result carries the decoded image under `configuration.outputKey`.
///
/// Because the closures are Swift over `MLXArray`, the backend is constructed from Swift (directly or
/// through `NFKMLXModelRegistry`); an Objective-C consumer drives it through `NFKInferenceBackend`.
@objc(NFKMLXDiffusionBackend)
public final class NFKMLXDiffusionBackend: NSObject, NFKInferenceBackend {

    /// Maps the request and bridged inputs to a diffusion context. Runs on the inference thread.
    public typealias Encode = @Sendable (_ request: NFKInferenceRequest, _ image: MLXArray?, _ mask: MLXArray?) throws -> NFKDiffusionContext
    /// The model forward: predicts (per the scheduler's prediction type) for one step.
    public typealias Denoise = @Sendable (_ latent: MLXArray, _ timestep: NFKDiffusionTimestep, _ context: NFKDiffusionContext, _ guidanceScale: Float) -> MLXArray
    /// Maps the final latent to an image tensor `[H, W, C]` in `0...1` (identity, or a VAE decode).
    public typealias Decode = @Sendable (_ latent: MLXArray) -> MLXArray

    private let identifier: String
    private let ready: Bool
    private let configuration: NFKDiffusionConfiguration
    private let scheduler: any NFKDiffusionScheduler
    private let encode: Encode
    private let denoise: Denoise
    private let decode: Decode

    public init(identifier: String = "mlx-diffusion",
                isReady: Bool = true,
                configuration: NFKDiffusionConfiguration = NFKDiffusionConfiguration(),
                scheduler: any NFKDiffusionScheduler = NFKDDIMScheduler(),
                encode: @escaping Encode,
                denoise: @escaping Denoise,
                decode: @escaping Decode = { $0 }) {
        self.identifier = identifier
        self.ready = isReady
        self.configuration = configuration
        self.scheduler = scheduler
        self.encode = encode
        self.denoise = denoise
        self.decode = decode
        super.init()
    }

    // MARK: NFKInferenceBackend

    @objc public var isReady: Bool { ready }

    @objc public var backendIdentifier: String { identifier }

    @objc(runInferenceForRequest:error:)
    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        let job = submitInferenceJob(for: request)
        let semaphore = DispatchSemaphore(value: 0)
        job.completionHandler = { _ in semaphore.signal() }
        semaphore.wait()
        if let result = job.result {
            return result
        }
        if let error = job.error {
            throw error
        }
        throw NFKMLXError.noOutput
    }

    @objc(submitInferenceJobForRequest:)
    public func submitInferenceJob(for request: NFKInferenceRequest) -> NFKInferenceJob {
        let job = NFKInferenceJob()
        let configuration = self.configuration
        let scheduler = self.scheduler
        let encode = self.encode
        let denoise = self.denoise
        let decode = self.decode
        Task.detached {
            do {
                let outputs = try NFKMLXDiffusionBackend.run(request, job: job, configuration: configuration,
                                                             scheduler: scheduler, encode: encode,
                                                             denoise: denoise, decode: decode)
                if job.status == .cancelled { return }
                job.finish(with: NFKInferenceResult(outputs: outputs))
            } catch {
                job.finish(withError: error as NSError)
            }
        }
        return job
    }

    private static func run(_ request: NFKInferenceRequest,
                            job: NFKInferenceJob,
                            configuration base: NFKDiffusionConfiguration,
                            scheduler: any NFKDiffusionScheduler,
                            encode: Encode,
                            denoise: Denoise,
                            decode: Decode) throws -> [String: Any] {
        let configuration = resolved(base, from: request)
        let colorSpace = configuration.imageOptions.colorSpace
        let image = try request.input(forKey: NFKInputImage).map {
            try NFKMLXImageBridge.tensor(from: $0, channels: configuration.plateChannels, colorSpace: colorSpace)
        }
        let mask = try request.input(forKey: NFKInputMask).map {
            try NFKMLXImageBridge.tensor(from: $0, channels: 1, colorSpace: colorSpace)
        }

        let context = try encode(request, image, mask)
        let schedule = scheduler.steps(configuration.steps)
        guard !schedule.isEmpty else { throw NFKMLXError.noOutput }
        let noise = gaussianNoise(height: context.height, width: context.width,
                                  channels: configuration.latentChannels, seed: configuration.seed)

        var latent: MLXArray
        let startIndex: Int
        if let source = context.sourceLatent {
            startIndex = self.startIndex(forStrength: configuration.strength, count: schedule.count)
            latent = scheduler.addNoise(clean: source, noise: noise, timestep: schedule[startIndex])
        } else {
            startIndex = 0
            latent = scheduler.initialLatent(noise: context.initialLatent ?? noise, first: schedule[0])
        }

        for i in startIndex ..< schedule.count {
            if job.status == .cancelled { return [:] }
            let prediction = denoise(latent, schedule[i], context, configuration.guidanceScale)
            latent = scheduler.step(prediction: prediction, timestep: schedule[i], latent: latent)
            if let mask = context.mask, let source = context.sourceLatent {
                // Hold the kept region to the source, noised to the next step's level, so only the
                // masked region is generated.
                let known = i + 1 < schedule.count
                    ? scheduler.addNoise(clean: source, noise: noise, timestep: schedule[i + 1])
                    : source
                latent = mask * latent + (1 - mask) * known
            }
            eval(latent)
            job.reportProgress(Double(i - startIndex + 1) / Double(schedule.count - startIndex))
        }

        let decoded = decode(latent)
        eval(decoded)
        return [configuration.outputKey: try makeOutput(from: decoded, configuration: configuration)]
    }

    private static func makeOutput(from array: MLXArray, configuration: NFKDiffusionConfiguration) throws -> Any {
        if configuration.outputsTexture {
            guard let device = configuration.device ?? MTLCreateSystemDefaultDevice() else {
                throw NFKMLXImageBridge.BridgeError.noMetalDevice
            }
            return try NFKMLXImageBridge.texture(from: array, device: device, options: configuration.imageOptions)
        }
        return try NFKMLXImageBridge.cgImage(from: array, options: configuration.imageOptions)
    }

    /// The configuration a request runs under: the model's defaults, with the generation parameters a
    /// caller sets taking precedence. Sampling is what an InferKit request carries these keys for, so a
    /// caller changes the step count or the guidance per request rather than per backend.
    static func resolved(_ configuration: NFKDiffusionConfiguration,
                         from request: NFKInferenceRequest) -> NFKDiffusionConfiguration {
        var resolved = configuration
        if let steps = request.parameter(forKey: NFKParameterSteps) as? NSNumber {
            resolved.steps = max(steps.intValue, 1)
        }
        if let guidance = request.parameter(forKey: NFKParameterGuidanceScale) as? NSNumber {
            resolved.guidanceScale = guidance.floatValue
        }
        if let strength = request.parameter(forKey: NFKParameterStrength) as? NSNumber {
            resolved.strength = strength.floatValue
        }
        if let seed = request.parameter(forKey: NFKParameterSeed) as? NSNumber {
            resolved.seed = seed.uint64Value
        }
        return resolved
    }

    /// The first schedule index for an image-to-image `strength`. `1` runs every step from full noise;
    /// a lower value skips the early high-noise steps so more of the source survives.
    static func startIndex(forStrength strength: Float, count: Int) -> Int {
        let clamped = min(max(strength, 0), 1)
        let skipped = Int((1 - clamped) * Float(count))
        return min(max(skipped, 0), max(count - 1, 0))
    }

    /// Deterministic unit-variance Gaussian noise `[H, W, C]` from a SplitMix64 stream shaped by
    /// Box–Muller, so a run is repeatable without depending on the MLX random state.
    static func gaussianNoise(height: Int, width: Int, channels: Int, seed: UInt64?) -> MLXArray {
        var state = seed ?? 0x9E37_79B9_7F4A_7C15
        func nextUInt() -> UInt64 {
            state = state &+ 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }
        func nextUnit() -> Float { Float(nextUInt() >> 40) / Float(1 << 24) }   // [0, 1)

        let count = max(height * width * channels, 0)
        var values = [Float](repeating: 0, count: count)
        var index = 0
        while index < count {
            let u1 = max(nextUnit(), 1e-7)
            let u2 = nextUnit()
            let radius = sqrtf(-2 * logf(u1))
            let angle = 2 * Float.pi * u2
            values[index] = radius * cosf(angle)
            if index + 1 < count {
                values[index + 1] = radius * sinf(angle)
            }
            index += 2
        }
        return values.withUnsafeBufferPointer { MLXArray($0, [height, width, channels]) }
    }
}
