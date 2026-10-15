//
//  NFKMLXGenerationBackend.swift
//  InferKitMLX
//
//  The InferKit contract over a model object whose own API is a set of Swift or Objective-C methods: a
//  generator, a vision-language model, a reranker, a forecaster. Each model's `backend…` factories build
//  one around a closure that reads the request and calls the object.
//
//  Introduced in InferKit 0.4.0.
//

import CoreGraphics
import Foundation
import InferKit
import MLX

/// A backend that runs one closure per request, one request at a time.
///
/// @discussion The closure reads the request, runs the model, and returns the result, or nil when the job
/// was cancelled. Runs are serialized, so a closure may adjust its model's settings for one request and
/// restore them afterward. ``runInference(for:)`` waits for the job ``submitInferenceJob(for:)`` starts.
/// Objective-C reaches every instance through the model factories that return one.
///
/// Introduced in InferKit 0.4.0.
@objc(NFKMLXGenerationBackend)
public final class NFKMLXGenerationBackend: NSObject, NFKInferenceBackend {

    /// Reads `request`, runs the model, and returns the result, or nil when `job` was cancelled. It may
    /// report progress through `job`. Introduced in InferKit 0.4.0.
    public typealias Run = (_ request: NFKInferenceRequest, _ job: NFKInferenceJob) throws -> NFKInferenceResult?

    private let identifier: String
    private let inputKeys: Set<String>
    private let parameterKeys: Set<String>
    private let outputKeys: Set<String>
    private let ready: () -> Bool
    private let run: Run
    private let information: [String: Any]?
    private let runLock = NSLock()

    /// Builds a backend that runs `run` for each request.
    ///
    /// - Parameters:
    ///   - identifier: the backend's `backendIdentifier`.
    ///   - inputKeys: the request inputs `run` reads.
    ///   - parameterKeys: the request parameters `run` reads.
    ///   - outputKeys: the result keys `run` writes, every key it can write for some request.
    ///   - modelInfo: the `NFKModelInfo*` description of the model, when known.
    ///   - isReady: whether the model can run now.
    ///   - run: the per-request work.
    ///
    /// Introduced in InferKit 0.4.0.
    public init(identifier: String, inputKeys: Set<String>, parameterKeys: Set<String>,
                outputKeys: Set<String>, modelInfo: [String: Any]? = nil, isReady: @escaping () -> Bool = { true },
                run: @escaping Run) {
        self.identifier = identifier
        self.inputKeys = inputKeys
        self.parameterKeys = parameterKeys
        self.outputKeys = outputKeys
        self.information = modelInfo
        self.ready = isReady
        self.run = run
        super.init()
    }

    @objc public var isReady: Bool { ready() }

    @objc public var backendIdentifier: String { identifier }

    /// The request inputs the backend reads. Introduced in InferKit 0.4.0.
    @objc public var supportedInputKeys: Set<String> { inputKeys }

    /// The request parameters the backend reads. Introduced in InferKit 0.4.0.
    @objc public var supportedParameterKeys: Set<String> { parameterKeys }

    /// The result keys the backend writes. Introduced in InferKit 0.4.0.
    @objc public var supportedOutputKeys: Set<String> { outputKeys }

    /// The model's description under the `NFKModelInfo*` keys; empty where none was given. Introduced in
    /// InferKit 0.4.0.
    @objc public var modelInfo: [String: Any] { information ?? [:] }

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
        let run = self.run
        let runLock = self.runLock
        Task.detached(priority: .userInitiated) {
            runLock.withLock {
                if job.status == .cancelled { return }
                do {
                    if let result = try run(request, job) {
                        job.finish(with: result)
                    }
                } catch {
                    job.finish(withError: error as NSError)
                }
            }
        }
        return job
    }
}

// MARK: - Request and result helpers

extension NFKMLXGenerationBackend {

    /// The non-empty string under `key`, or nil.
    static func text(_ request: NFKInferenceRequest, _ key: String) -> String? {
        guard let value = request.input(forKey: key) as? String, !value.isEmpty else { return nil }
        return value
    }

    /// The non-empty string under `key`; a missing or empty one is an unsupported input.
    static func requiredText(_ request: NFKInferenceRequest, _ key: String) throws -> String {
        guard let value = text(request, key) else { throw NFKMLXError.unsupportedInput }
        return value
    }

    /// The non-empty string under the parameter `key`, or nil.
    static func parameterText(_ request: NFKInferenceRequest, _ key: String) -> String? {
        guard let value = request.parameter(forKey: key) as? String, !value.isEmpty else { return nil }
        return value
    }

    /// The number under the parameter `key`, or nil.
    static func number(_ request: NFKInferenceRequest, _ key: String) -> NSNumber? {
        request.parameter(forKey: key) as? NSNumber
    }

    /// The seed under `NFKParameterSeed`, 0 where none is given.
    static func seed(_ request: NFKInferenceRequest) -> UInt64 {
        number(request, NFKParameterSeed)?.uint64Value ?? 0
    }

    /// `value` as a `CGImage`: a `CGImage` as given, an `MTLTexture` read back as 8-bit RGBA.
    static func cgImage(fromInput value: Any) throws -> CGImage {
        let cf = value as CFTypeRef
        if CFGetTypeID(cf) == CGImage.typeID {
            return cf as! CGImage
        }
        let (bytes, width, height) = try NFKMLXImageBridge.rgbaBytes(from: value, colorSpace: CGColorSpaceCreateDeviceRGB())
        return try NFKMLXImageBridge.cgImage(rgba: bytes, width: width, height: height, options: NFKMLXImageOptions())
    }

    /// The image under `key` (a `CGImage` or an `MTLTexture`) as a `CGImage`, or nil.
    static func image(_ request: NFKInferenceRequest, _ key: String) throws -> CGImage? {
        guard let value = request.input(forKey: key) else { return nil }
        return try cgImage(fromInput: value)
    }

    /// The image under `key` as a `CGImage`; a missing one is an unsupported input.
    static func requiredImage(_ request: NFKInferenceRequest, _ key: String) throws -> CGImage {
        guard let image = try image(request, key) else { throw NFKMLXError.unsupportedInput }
        return image
    }

    /// The images under `key`: an array of anything ``image(_:_:)`` accepts, or a single one.
    static func images(_ request: NFKInferenceRequest, _ key: String) throws -> [CGImage] {
        guard let value = request.input(forKey: key) else { return [] }
        let values = (value as? [Any]) ?? [value]
        return try values.map { try cgImage(fromInput: $0) }
    }

    /// `[H, W, 3]` in `[0, 1]` as an 8-bit sRGB `CGImage`.
    static func cgImage(_ pixels: MLXArray) throws -> CGImage {
        try NFKMLXImageBridge.cgImage(from: pixels, options: NFKMLXImageOptions())
    }

    /// A result carrying `image` under `NFKOutputImage`.
    static func imageResult(_ image: CGImage) -> NFKInferenceResult {
        NFKInferenceResult(outputs: [NFKOutputImage: image])
    }

    /// Writes `clip` (`[frames, H, W, 3]` in `[0, 1]`) as a QuickTime movie in the temporary directory and
    /// returns a result carrying it under `NFKOutputVideo`.
    static func videoResult(_ clip: MLXArray, framesPerSecond: Double, identifier: String) throws -> NFKInferenceResult {
        let frames = try (0 ..< clip.dim(0)).map { try cgImage(clip[$0]) }
        guard let first = frames.first else { throw NFKMLXError.noOutput }
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("\(identifier)-\(UUID().uuidString).mov")
        try NFKMLXVideoFile.write(frames, framesPerSecond: framesPerSecond, to: url)
        let asset = NFKVideoAsset(fileURL: url,
                                  durationSeconds: Double(frames.count) / framesPerSecond,
                                  framesPerSecond: framesPerSecond,
                                  dimensions: CGSize(width: first.width, height: first.height))
        return NFKInferenceResult(outputs: [NFKOutputVideo: asset])
    }

    /// The answer length an image-question backend uses where the request names none.
    static let defaultAnswerTokens = 128

    /// A backend that answers a question about an image: it reads `NFKInputImage` and `NFKInputPrompt`,
    /// both required, bounds the answer by `NFKParameterMaxTokens` (``defaultAnswerTokens``), and returns
    /// the text `answer` produces under `NFKOutputText`.
    static func imageQuestion(identifier: String,
                              answer: @escaping (_ image: CGImage, _ question: String, _ maxTokens: Int) -> String)
        -> NFKMLXGenerationBackend {
        NFKMLXGenerationBackend(identifier: identifier, inputKeys: [NFKInputImage, NFKInputPrompt],
                                parameterKeys: [NFKParameterMaxTokens],
                                outputKeys: [NFKOutputText]) { request, job in
            let image = try requiredImage(request, NFKInputImage)
            let question = try requiredText(request, NFKInputPrompt)
            let maxTokens = max(number(request, NFKParameterMaxTokens)?.intValue ?? defaultAnswerTokens, 0)
            job.reportProgress(0)
            let text = answer(image, question, maxTokens)
            if job.status == .cancelled { return nil }
            job.reportProgress(1)
            return NFKInferenceResult(outputs: [NFKOutputText: text])
        }
    }

    /// Registers `factory` under `name`, reading the registry's URL as the release directory. A nil URL is
    /// refused: these models have no random-weights form.
    static func registerDirectoryModel(_ name: String,
                                       factory: @escaping (URL) throws -> any NFKInferenceBackend) {
        NFKMLXModelRegistry.register(name: name) { weightsURL in
            guard let weightsURL else {
                throw NFKMLXError.unsupportedConfiguration("\(name) needs the release directory; there is no random-weights form")
            }
            return try factory(weightsURL)
        }
    }
}
