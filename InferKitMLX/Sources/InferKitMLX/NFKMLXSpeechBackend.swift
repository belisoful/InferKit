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

    private let synthesize: Synthesize
    private let identifier: String
    private let ready: Bool
    private let configuration: NFKMLXSpeechConfiguration

    public init(identifier: String = "mlx-speech",
                isReady: Bool = true,
                configuration: NFKMLXSpeechConfiguration = NFKMLXSpeechConfiguration(),
                synthesize: @escaping Synthesize) {
        self.identifier = identifier
        self.ready = isReady
        self.configuration = configuration
        self.synthesize = synthesize
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
        let synthesize = self.synthesize
        let configuration = self.configuration
        Task.detached {
            do {
                let result = try NFKMLXSpeechBackend.run(request, configuration: configuration, synthesize: synthesize)
                job.finish(with: result)
            } catch {
                job.finish(withError: error as NSError)
            }
        }
        return job
    }

    private static func run(_ request: NFKInferenceRequest,
                            configuration: NFKMLXSpeechConfiguration,
                            synthesize: Synthesize) throws -> NFKInferenceResult {
        let text = self.text(from: request)
        guard !text.isEmpty else { throw NFKMLXError.unsupportedInput }

        // A request may override the configured rate; the closure generates at that rate so the pitch
        // stays correct (tagging samples with a different rate would only shift them).
        let requested = (request.parameter(forKey: NFKParameterSampleRate) as? NSNumber)?.intValue
        let sampleRate = max(requested ?? configuration.sampleRate, 1)

        let waveform = synthesize(text, sampleRate)
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
