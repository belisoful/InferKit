//
//  NFKMLXWav2Vec2Backend.swift
//  InferKitMLX
//
//  The Wav2Vec2 / HuBERT weight loader, the @objc directory and download factories, and an
//  NFKInferenceBackend that transcribes speech with a CTC release or embeds it with a pretraining one.
//

import Foundation
import InferKit
import MLX
import MLXNN

extension NFKMLXWav2Vec2Net {
    /// The weight file a release directory holds: `model.safetensors` when it ships one, else the
    /// `pytorch_model.bin` most releases carry, which the native PyTorch reader loads directly.
    static func weightsURL(inDirectory directory: URL) throws -> URL {
        for name in NFKMLXWav2Vec2.weightFiles {
            let url = directory.appendingPathComponent(name)
            if FileManager.default.fileExists(atPath: url.path) { return url }
        }
        throw NFKMLXError.unsupportedConfiguration(
            "\(directory.lastPathComponent) holds neither model.safetensors nor pytorch_model.bin")
    }

    /// Loads a release's weights. The checkpoint's `wav2vec2.` or `hubert.` prefix is stripped; the
    /// pretraining quantizer and projections are dropped; the position convolution's weight-norm gain and
    /// direction load under `weight_g` and `weight_v` from either spelling the releases use (the
    /// `parametrizations.weight.original0/1` form included); convolutions move to MLX's channels-last
    /// layout.
    ///
    /// Introduced in InferKit 0.4.0.
    public func loadWeights(fromDirectory directory: URL) throws {
        try loadWeights(url: try Self.weightsURL(inDirectory: directory), leavingFresh: [])
    }

    /// Loads every parameter but those under `fresh` (module-name prefixes such as `lm_head.` for a head
    /// retargeted to a new vocabulary), which keep their initialization. Every other parameter must be
    /// supplied, except `masked_spec_embed`, which a release trained without SpecAugment does not carry.
    func loadWeights(url: URL, leavingFresh fresh: [String]) throws {
        let checkpoint = try NFKMLXWeights.materializedCheckpoint(url: url)
        var mapped = [String: MLXArray]()
        let renamed = ["encoder.pos_conv_embed.conv.parametrizations.weight.original0": "encoder.pos_conv_embed.conv.weight_g",
                       "encoder.pos_conv_embed.conv.parametrizations.weight.original1": "encoder.pos_conv_embed.conv.weight_v"]
        for (rawKey, value) in checkpoint.arrays {
            var key = rawKey
            for prefix in ["wav2vec2.", "hubert."] where key.hasPrefix(prefix) {
                key = String(key.dropFirst(prefix.count))
            }
            key = renamed[key] ?? key
            if ["quantizer.", "project_q.", "project_hid.", "label_embs_concat", "final_proj."]
                .contains(where: { key.hasPrefix($0) }) {
                continue
            }
            if fresh.contains(where: { key.hasPrefix($0) }) { continue }
            if value.ndim == 3, checkpoint.needsConvTranspose {
                mapped[key] = value.transposed(0, 2, 1)
            } else {
                mapped[key] = value
            }
        }
        let owned = Set(parameters().flattened().map(\.0))
        let optional: Set<String> = ["masked_spec_embed"]
        let missing = owned.filter { name in
            mapped[name] == nil && !optional.contains(name) && !fresh.contains { name.hasPrefix($0) }
        }
        guard missing.isEmpty else {
            throw NFKMLXError.weightsMismatch("the Wav2Vec2 checkpoint lacks \(missing.count) parameters, starting with "
                                              + missing.sorted().prefix(3).joined(separator: ", "))
        }
        let unexpected = mapped.keys.filter { !owned.contains($0) }
        guard unexpected.isEmpty else {
            throw NFKMLXError.weightsMismatch("the Wav2Vec2 checkpoint carries \(unexpected.count) unread tensors, "
                                              + "starting with " + unexpected.sorted().prefix(3).joined(separator: ", "))
        }
        try NFKMLXWeights.apply(mapped.map { ($0.key, $0.value) }, to: self, strict: false)
    }
}

/// Wav2Vec2 and HuBERT (`facebook/wav2vec2-*`, `facebook/hubert-*`, Apache-2.0), self-supervised speech
/// encoders ported into `MLXNN` at reference parity. A CTC release (`…-960h`, `…-ls960-ft`) transcribes
/// English speech to text; a pretraining release (`wav2vec2-base`, `xls-r-*`, `hubert-*-ll60k`) produces
/// contextual frame features and a mean-pooled utterance embedding.
///
/// Introduced in InferKit 0.4.0.
@objc(NFKMLXWav2Vec2)
public final class NFKMLXWav2Vec2: NSObject {
    @objc public static let modelName = "wav2vec2"
    static let requiredFiles = ["config.json"]
    static let optionalFiles = ["preprocessor_config.json", "vocab.json", "tokenizer_config.json"]
    static let weightFiles = ["model.safetensors", "pytorch_model.bin"]

    /// Builds a backend from a release directory: `config.json`, the weights (`model.safetensors` or
    /// `pytorch_model.bin`), `preprocessor_config.json`, and for a CTC release `vocab.json`. Run
    /// inference off the render thread.
    @objc(backendWithDirectoryURL:error:)
    public static func backend(directoryURL: URL) throws -> NFKMLXWav2Vec2Backend {
        let net = try NFKMLXWav2Vec2Net(configurationURL: directoryURL.appendingPathComponent("config.json"))
        try net.loadWeights(fromDirectory: directoryURL)
        let tokenizer = net.transcribes ? try NFKMLXWav2Vec2Tokenizer(directoryURL: directoryURL) : nil
        return NFKMLXWav2Vec2Backend(net: net, tokenizer: tokenizer)
    }

    /// The asynchronous form of the directory factory. Blocking work runs at user-initiated quality of
    /// service off the calling thread.
    @objc(backendWithDirectoryURL:completionHandler:)
    public static func backend(directoryURL: URL,
                               completionHandler: @escaping (NFKMLXWav2Vec2Backend?, Error?) -> Void) {
        Task.detached(priority: .userInitiated) {
            do { completionHandler(try backend(directoryURL: directoryURL), nil) }
            catch { completionHandler(nil, error) }
        }
    }

    /// Downloads a release into the hub cache and builds the backend.
    ///
    /// @discussion The download fetches `config.json`, `preprocessor_config.json`, `vocab.json`, and
    /// `tokenizer_config.json` where the repo serves them, and `model.safetensors`, or
    /// `pytorch_model.bin` when the repo holds no safetensors. A file the cache already holds is not
    /// fetched again. The call blocks on the network; call it off the render thread. The CTC releases
    /// are `facebook/wav2vec2-base-960h`, `-large-960h`, `-large-960h-lv60-self`, and
    /// `facebook/hubert-large-ls960-ft` and `-xlarge-ls960-ft`; the pretraining releases are
    /// `facebook/wav2vec2-base`, `-large-lv60`, `-large-xlsr-53`, `-xls-r-300m`, `-xls-r-1b`, `-xls-r-2b`,
    /// and `facebook/hubert-base-ls960`, `-large-ll60k`, and `-xlarge-ll60k`. None is gated.
    ///
    /// Introduced in InferKit 0.4.0.
    @objc(backendWithRepo:revision:cacheDirectoryURL:error:)
    public static func backend(repo: String, revision: String?, cacheDirectoryURL: URL?) throws -> NFKMLXWav2Vec2Backend {
        try backend(directoryURL: try NFKMLXReleaseDownload.directory(
            repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL,
            required: requiredFiles, optional: optionalFiles, weights: weightFiles))
    }

    /// The asynchronous form of ``backend(repo:revision:cacheDirectoryURL:)``. The download and the
    /// build run at user-initiated quality of service off the calling thread.
    ///
    /// Introduced in InferKit 0.4.0.
    @objc(backendWithRepo:revision:cacheDirectoryURL:completionHandler:)
    public static func backend(repo: String, revision: String?, cacheDirectoryURL: URL?,
                               completionHandler: @escaping (NFKMLXWav2Vec2Backend?, Error?) -> Void) {
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                completionHandler(try backend(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL), nil)
            } catch {
                completionHandler(nil, error)
            }
        }
    }
}

// MARK: - Backend

/// The Wav2Vec2 / HuBERT inference backend. Audio under `NFKInputAudio` (an `NFKAudioAsset` or WAV
/// `Data`, any sample rate, resampled to 16 kHz) comes back as `NFKOutputText` from a CTC release, and
/// as a mean-pooled `NFKOutputEmbedding` (`[NSNumber]`, the encoder width) from every release.
///
/// Introduced in InferKit 0.4.0.
@objc(NFKMLXWav2Vec2Backend)
public final class NFKMLXWav2Vec2Backend: NSObject, NFKInferenceBackend {
    final class Holder: @unchecked Sendable {
        let net: NFKMLXWav2Vec2Net
        let tokenizer: NFKMLXWav2Vec2Tokenizer?
        init(net: NFKMLXWav2Vec2Net, tokenizer: NFKMLXWav2Vec2Tokenizer?) {
            self.net = net
            self.tokenizer = tokenizer
        }
    }

    private let holder: Holder

    init(net: NFKMLXWav2Vec2Net, tokenizer: NFKMLXWav2Vec2Tokenizer?) {
        holder = Holder(net: net, tokenizer: tokenizer)
        super.init()
    }

    /// Whether the backend transcribes (a CTC release) as well as embeds.
    @objc public var transcribes: Bool { holder.tokenizer != nil }

    @objc public var isReady: Bool { true }
    @objc public var backendIdentifier: String { NFKMLXWav2Vec2.modelName }
    @objc public var supportedParameterKeys: Set<String> { [] }
    @objc public var supportedInputKeys: Set<String> { [NFKInputAudio] }

    @objc(runInferenceForRequest:error:)
    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        let job = submitInferenceJob(for: request)
        let semaphore = DispatchSemaphore(value: 0)
        job.completionHandler = { _ in semaphore.signal() }
        semaphore.wait()
        if let result = job.result { return result }
        if let error = job.error { throw error }
        throw NFKMLXError.noOutput
    }

    @objc(submitInferenceJobForRequest:)
    public func submitInferenceJob(for request: NFKInferenceRequest) -> NFKInferenceJob {
        let job = NFKInferenceJob()
        let holder = self.holder
        Task.detached(priority: .userInitiated) {
            do {
                guard let (samples, rate) = Self.audio(from: request) else { throw NFKMLXError.unsupportedInput }
                let matched = NFKMLXAudioRate.matched(samples, from: rate, to: NFKMLXWav2Vec2Processor.sampleRate)
                let minimum = holder.net.configuration.convKernels.first ?? 400
                guard holder.net.configuration.frameCount(samples: matched.count) > 0, matched.count >= minimum else {
                    throw NFKMLXError.unsupportedInput
                }
                let input = NFKMLXWav2Vec2Processor.inputValues(matched, normalize: holder.net.configuration.normalizesInput)
                let hidden = holder.net(input)
                var outputs: [String: Any] = [:]
                let pooled = hidden.mean(axis: 1)[0].asType(.float32)
                if let head = holder.net.head, let tokenizer = holder.tokenizer {
                    let frameTokens = argMax(head(hidden)[0], axis: -1).asType(.int32)
                    eval(frameTokens, pooled)
                    outputs[NFKOutputText] = tokenizer.text(forFrameTokens: frameTokens.asArray(Int32.self).map(Int.init))
                } else {
                    eval(pooled)
                }
                outputs[NFKOutputEmbedding] = pooled.asArray(Float.self).map { NSNumber(value: $0) }
                job.finish(with: NFKInferenceResult(outputs: outputs))
            } catch {
                job.finish(withError: error as NSError)
            }
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
