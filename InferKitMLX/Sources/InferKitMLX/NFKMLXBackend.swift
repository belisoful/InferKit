//
//  NFKMLXBackend.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX

/// A bundled Stable Diffusion release.
@objc(NFKMLXStableDiffusionModel)
public enum NFKMLXStableDiffusionModel: Int {
    case sdxlTurbo
    case stableDiffusion21Base
    case stableDiffusion15
}

// InferKit's request and job are immutable or internally locked, so they are safe to hand to the
// generation task.
extension NFKInferenceRequest: @retroactive @unchecked Sendable {}
extension NFKInferenceJob: @retroactive @unchecked Sendable {}

/// An InferKit backend that runs a bundled Stable Diffusion release on Apple Silicon.
///
/// It adopts the Objective-C `NFKInferenceBackend` protocol, so an InferKit consumer swaps it in like
/// any other backend. A request with no image input runs text-to-image; a request carrying a `CGImage`
/// under `NFKInputImage` runs image-to-image, with `NFKParameterStrength` controlling how much of the
/// source survives. The result carries the generated image (a `CGImage`) under `NFKOutputImage`.
///
/// The release's files download from Hugging Face on first use and the run is on the GPU, so a caller
/// prefers `submitInferenceJob(for:)` for its progress and cancellation.
///
/// Every part of the run is this package's own — ``NFKMLXSDTextEncoderNet`` for the prompt,
/// ``NFKMLXSDUNet`` for the denoiser, ``NFKDDIMScheduler`` for the sampler, ``NFKMLXSDAutoencoder``
/// for the picture — each measured against its reference implementation.
@objc(NFKMLXBackend)
public final class NFKMLXBackend: NSObject, NFKInferenceBackend {

    @objc public let model: NFKMLXStableDiffusionModel

    /// Where downloaded weights are cached. Nil uses `NFKHFHub.defaultCacheDirectoryURL()`.
    @objc public var cacheDirectoryURL: URL?

    /// The precision the release loads at. The default runs it as published, which for a release that
    /// ships half-precision weights is half the memory and a few decimal digits; `.float32` is the
    /// precision the parity records were measured at.
    @objc public var precision: NFKMLXWeightPrecision = .checkpoint

    private let lock = NSLock()
    private var backend: (any NFKInferenceBackend)?

    @objc public init(model: NFKMLXStableDiffusionModel) {
        self.model = model
        super.init()
    }

    /// A backend around a model that is already built, which skips the download. The release still
    /// names the identity this reports.
    init(model: NFKMLXStableDiffusionModel, loaded: any NFKInferenceBackend) {
        self.model = model
        self.backend = loaded
        super.init()
    }

    // MARK: NFKInferenceBackend

    @objc public var isReady: Bool {
        lock.lock()
        defer { lock.unlock() }
        return backend != nil
    }

    @objc public var backendIdentifier: String { "mlx-stable-diffusion" }

    @objc(prepareWithError:)
    public func prepare() throws {
        _ = try loaded()
    }

    @objc(runInferenceForRequest:error:)
    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        try loaded().runInference(for: request)
    }

    @objc(submitInferenceJobForRequest:)
    public func submitInferenceJob(for request: NFKInferenceRequest) -> NFKInferenceJob {
        // The release downloads and loads on first use, which is seconds to minutes; doing it inside
        // the job keeps that off the caller's thread and reports its failure through the job. The
        // model's own job carries the per-step progress and takes the cancellation, so this one
        // forwards both rather than standing in front of them.
        let job = NFKInferenceJob()
        Task.detached { [self] in
            do {
                let model = try self.loaded()
                // Every backend this builds implements the job form; the protocol makes it optional.
                guard let inner = model.submitInferenceJob?(for: request) else {
                    job.finish(with: try model.runInference(for: request))
                    return
                }
                job.cancellationHandler = { inner.cancel() }
                if job.status == .cancelled {
                    inner.cancel()
                }
                inner.progressHandler = { [weak job] step in
                    job?.reportProgress(step.progress, partialResult: step.partialResult)
                }
                inner.completionHandler = { [weak job] step in
                    guard let job, job.status != .cancelled else { return }
                    if let result = step.result {
                        job.finish(with: result)
                    } else {
                        job.finish(withError: step.error ?? NFKMLXError.noOutput as NSError)
                    }
                }
            } catch {
                job.finish(withError: error as NSError)
            }
        }
        return job
    }

    // MARK: Loading

    private func loaded() throws -> any NFKInferenceBackend {
        lock.lock()
        defer { lock.unlock() }
        if let backend {
            return backend
        }
        let release = NFKMLXStableDiffusionRelease(model)
        let directory = try release.download(cacheDirectoryURL: cacheDirectoryURL)
        let built = try NFKMLXTextToImage.backend(configuration: release.configuration,
                                                  directoryURL: directory, precision: precision)
        backend = built
        return built
    }
}

/// Where a bundled release lives on the hub, and which files it takes.
struct NFKMLXStableDiffusionRelease {

    let repository: String
    let configuration: NFKMLXSDTextToImageConfiguration
    /// Paths within the repository, tried in order at each slot so a release published only in half
    /// precision resolves too.
    private let weightFiles: [[String]]

    init(_ model: NFKMLXStableDiffusionModel) {
        switch model {
        case .stableDiffusion15:
            repository = "stable-diffusion-v1-5/stable-diffusion-v1-5"
            configuration = .stableDiffusion15
            weightFiles = Self.singleTowerFiles
        case .stableDiffusion21Base:
            // A gated repository: set `NFKHFHub.accessToken`, or HF_TOKEN in the environment.
            repository = "stabilityai/stable-diffusion-2-1-base"
            configuration = .stableDiffusion21
            weightFiles = Self.singleTowerFiles
        case .sdxlTurbo:
            repository = "stabilityai/sdxl-turbo"
            configuration = .sdxlTurbo
            weightFiles = Self.singleTowerFiles + [
                ["text_encoder_2/model.safetensors", "text_encoder_2/model.fp16.safetensors"],
            ]
        }
    }

    private static let singleTowerFiles = [
        ["unet/diffusion_pytorch_model.safetensors", "unet/diffusion_pytorch_model.fp16.safetensors"],
        ["vae/diffusion_pytorch_model.safetensors", "vae/diffusion_pytorch_model.fp16.safetensors"],
        ["text_encoder/model.safetensors", "text_encoder/model.fp16.safetensors"],
    ]

    /// The vocabulary files, each with whether the release has to carry it. They are small, and a
    /// missing one is the difference between a prompt and a different prompt, so they are fetched with
    /// the weights rather than lazily. The two that name the markers are optional because not every
    /// release publishes both, and the tokenizer falls back to the end marker for padding.
    private var vocabularyFiles: [(path: String, required: Bool)] {
        var names = [("tokenizer/vocab.json", true), ("tokenizer/merges.txt", true),
                     ("tokenizer/tokenizer_config.json", false),
                     ("tokenizer/special_tokens_map.json", false)]
        if configuration.secondaryTextEncoder != nil {
            names += [("tokenizer_2/vocab.json", true), ("tokenizer_2/merges.txt", true),
                      ("tokenizer_2/tokenizer_config.json", false),
                      ("tokenizer_2/special_tokens_map.json", false)]
        }
        return names
    }

    /// Downloads what the release needs and returns the directory holding it. The hub caches under
    /// `<cache>/<repository>/<revision>/<path>`, so the cache reproduces the release's own tree and
    /// ``NFKMLXSDReleaseFiles`` reads it directly.
    ///
    /// Blocking on the network; the caller runs it off the render thread.
    func download(cacheDirectoryURL: URL?) throws -> URL {
        let cache = cacheDirectoryURL ?? NFKHFHub.defaultCacheDirectoryURL()
        let hub = NFKHFHub(cacheDirectoryURL: cache)
        var root: URL?
        for file in vocabularyFiles {
            do {
                root = try fetch(file.path, from: hub).deletingLastPathComponent()
                    .deletingLastPathComponent()
            } catch {
                if file.required { throw error }
            }
        }
        for alternatives in weightFiles {
            var lastError: Error?
            for path in alternatives {
                do {
                    root = try fetch(path, from: hub).deletingLastPathComponent().deletingLastPathComponent()
                    lastError = nil
                    break
                } catch {
                    lastError = error
                }
            }
            if let lastError { throw lastError }
        }
        guard let root else { throw NFKMLXError.weightsMismatch("the release names no files") }
        return root
    }

    private func fetch(_ path: String, from hub: NFKHFHub) throws -> URL {
        try hub.downloadRepo(repository, revision: nil, path: path, sha256: nil)
    }
}
