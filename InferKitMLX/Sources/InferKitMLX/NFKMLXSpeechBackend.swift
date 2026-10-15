//
//  NFKMLXSpeechBackend.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX

/// How the speech backend renders and writes audio.
public struct NFKMLXSpeechConfiguration: @unchecked Sendable {
    /// The output sample rate in hertz.
    public var sampleRate: Int
    /// The directory the WAV file is written to. Defaults to the temporary directory.
    public var outputDirectory: URL

    public init(sampleRate: Int = 24000, outputDirectory: URL = FileManager.default.temporaryDirectory) {
        self.sampleRate = sampleRate
        self.outputDirectory = outputDirectory
    }
}

/// An InferKit backend for a bring-your-own MLX text-to-speech model.
///
/// Where the image backends map an `MLXArray` to an `MLXArray`, a speech model maps text to a mono
/// waveform: supply a `@Sendable (String) -> MLXArray` closure returning samples in `-1...1`, and the
/// backend reads the prompt (`NFKInputPrompt`, or the concatenated user content of `NFKInputMessages`),
/// runs the closure, writes a 16-bit PCM WAV file, and returns an `NFKAudioAsset` under
/// `NFKOutputAudio`. Because the closure is Swift over `MLXArray`, the backend is constructed from
/// Swift (directly or through `NFKMLXModelRegistry`); an Objective-C consumer drives it through the
/// `NFKInferenceBackend` protocol.
@objc(NFKMLXSpeechBackend)
public final class NFKMLXSpeechBackend: NSObject, NFKInferenceBackend {

    /// Maps input text to a mono waveform `[N]` in `-1...1`, generated at the given sample rate.
    public typealias Synthesize = @Sendable (_ text: String, _ sampleRate: Int) -> MLXArray

    /// Maps input text to a mono waveform as ``Synthesize`` does, reading the request's own parameters.
    /// Introduced in InferKit 0.4.0.
    public typealias RequestSynthesize = @Sendable (_ text: String, _ sampleRate: Int,
                                                    _ request: NFKInferenceRequest) throws -> MLXArray

    private let lifecycle: NFKMLXLifecycle<RequestSynthesize>
    private let identifier: String
    private let ready: Bool
    private let configuration: NFKMLXSpeechConfiguration
    private let requestParameterKeys: Set<String>

    public convenience init(identifier: String = "mlx-speech",
                            isReady: Bool = true,
                            configuration: NFKMLXSpeechConfiguration = NFKMLXSpeechConfiguration(),
                            synthesize: @escaping Synthesize) {
        self.init(identifier: identifier, isReady: isReady, configuration: configuration, parameterKeys: [],
                  requestSynthesize: { text, sampleRate, _ in synthesize(text, sampleRate) })
    }

    /// A backend whose closure reads the request's parameters beside the text.
    ///
    /// - Parameters:
    ///   - identifier: the backend's `backendIdentifier`.
    ///   - isReady: whether the backend reports itself ready to run.
    ///   - configuration: the output sample rate and file settings.
    ///   - parameterKeys: the request parameters `requestSynthesize` reads, which the backend declares beside
    ///     `NFKParameterSampleRate`.
    ///   - requestSynthesize: maps the text, the sample rate, and the request to a mono waveform.
    ///
    /// Introduced in InferKit 0.4.0.
    public init(identifier: String = "mlx-speech",
                isReady: Bool = true,
                configuration: NFKMLXSpeechConfiguration = NFKMLXSpeechConfiguration(),
                parameterKeys: Set<String>,
                requestSynthesize: @escaping RequestSynthesize) {
        self.identifier = identifier
        self.ready = isReady
        self.configuration = configuration
        self.requestParameterKeys = parameterKeys
        self.lifecycle = NFKMLXLifecycle(requestSynthesize)
        super.init()
    }

    // MARK: NFKInferenceBackend

    @objc public var isReady: Bool { ready }

    @objc public var backendIdentifier: String { identifier }

    /// The request parameters the backend reads. Introduced in InferKit 0.4.0.
    @objc public var supportedParameterKeys: Set<String> { requestParameterKeys.union([NFKParameterSampleRate]) }

    /// The request inputs the backend reads. Introduced in InferKit 0.4.0.
    @objc public var supportedInputKeys: Set<String> { [NFKInputPrompt, NFKInputMessages] }

    /// The result keys the backend writes. Introduced in InferKit 0.4.0.
    @objc public var supportedOutputKeys: Set<String> { [NFKOutputAudio] }

    /// `NFKModelInfoResidentBytes` and `NFKModelInfoPeakBytes` from the last measured load; empty before
    /// one. Introduced in InferKit 0.4.0.
    @objc public var modelInfo: [String: Any] { lifecycle.measurementInfo }

    // MARK: Unloading

    /// Frees the synthesis closure and the model it holds, then MLX's cache; the next run or ``prepare()`` loads the
    /// model again. It does nothing for a backend built from a closure alone, which has no source to load
    /// from: only one that `NFKMLXModelRegistry` or a download factory built from weights can unload. With
    /// runs in flight, the model is freed when the last of them ends. Introduced in InferKit 0.4.0.
    @objc public func unload() {
        lifecycle.unload()
    }

    /// Whether the backend holds its model. Introduced in InferKit 0.4.0.
    @objc public var isLoaded: Bool { lifecycle.isLoaded }

    /// The rise in MLX's active memory over the model's last load, or 0 while it is unloaded or before a
    /// measured load. Introduced in InferKit 0.4.0.
    @objc public var residentByteCount: Int { lifecycle.residentByteCount }

    /// The peak of MLX's active memory during the model's last measured load, above its level before it.
    /// Introduced in InferKit 0.4.0.
    @objc public var peakByteCount: Int { lifecycle.peakByteCount }

    /// Loads the model when it is unloaded. Introduced in InferKit 0.4.0.
    @objc(prepareWithError:)
    public func prepare() throws {
        try lifecycle.prepare()
    }

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
        let lifecycle = self.lifecycle
        let configuration = self.configuration
        Task.detached(priority: .userInitiated) {
            do {
                let result = try lifecycle.withState {
                    try NFKMLXSpeechBackend.run(request, configuration: configuration, synthesize: $0)
                }
                job.finish(with: result)
            } catch {
                job.finish(withError: error as NSError)
            }
        }
        return job
    }

    private static func run(_ request: NFKInferenceRequest,
                            configuration: NFKMLXSpeechConfiguration,
                            synthesize: RequestSynthesize) throws -> NFKInferenceResult {
        let text = self.text(from: request)
        guard !text.isEmpty else { throw NFKMLXError.unsupportedInput }

        // A request may override the configured rate; the closure generates at that rate so the pitch
        // stays correct (tagging samples with a different rate would only shift them).
        let requested = (request.parameter(forKey: NFKParameterSampleRate) as? NSNumber)?.intValue
        let sampleRate = max(requested ?? configuration.sampleRate, 1)

        let waveform = try synthesize(text, sampleRate, request)
        eval(waveform)
        let samples = waveform.asType(Float.self).asArray(Float.self)

        let url = configuration.outputDirectory.appendingPathComponent("mlx-speech-\(UUID().uuidString).wav")
        try NFKMLXWaveFile.write(samples: samples, sampleRate: sampleRate, to: url)

        let asset = NFKAudioAsset(fileURL: url,
                                  durationSeconds: Double(samples.count) / Double(sampleRate),
                                  sampleRate: Double(sampleRate),
                                  channelCount: 1)
        return NFKInferenceResult(outputs: [NFKOutputAudio: asset])
    }

    /// The prompt string: `NFKInputPrompt`, or the joined user content of `NFKInputMessages`.
    static func text(from request: NFKInferenceRequest) -> String {
        if let prompt = request.input(forKey: NFKInputPrompt) as? String, !prompt.isEmpty {
            return prompt
        }
        guard let messages = request.input(forKey: NFKInputMessages) as? [[String: Any]] else {
            return ""
        }
        return messages.compactMap { message in
            (message["role"] as? String) == "user" ? message["content"] as? String : nil
        }.joined(separator: " ")
    }
}

extension NFKMLXSpeechBackend: NFKMLXReloading {

    var canReload: Bool { lifecycle.canReload }

    func enableReload(_ build: @escaping () throws -> any NFKInferenceBackend, measurement: NFKMLXLoadMeasurement?) {
        lifecycle.enableReload(building: build, as: NFKMLXSpeechBackend.self, measurement: measurement) {
            try $0.lifecycle.loadedState()
        }
    }
}
