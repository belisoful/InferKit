//
//  NFKFoundationModelsBackend.swift
//  InferKitFoundationModels
//

import Foundation
import FoundationModels
import InferKit

enum NFKFoundationModelsError: Error {
    case noOutput
}

// InferKit's request and job are immutable or internally locked, so they are safe to hand to the
// generation task.
extension NFKInferenceRequest: @retroactive @unchecked Sendable {}
extension NFKInferenceJob: @retroactive @unchecked Sendable {}

/// An InferKit backend that runs Apple's on-device system language model through the Foundation
/// Models framework.
///
/// It adopts the Objective-C `NFKInferenceBackend` protocol, so an InferKit consumer swaps it in
/// like any other engine: the same request that runs against `NFKCoreMLLanguageBackend` or
/// `NFKRemoteBackend` runs here. A request supplies `NFKInputPrompt` (a string) or
/// `NFKInputMessages` (an OpenAI-style array); a system message becomes the session's instructions.
/// `NFKParameterTemperature` and `NFKParameterMaxTokens` map to `GenerationOptions`. The result
/// carries the text under `NFKOutputText`, and `submitInferenceJob(for:)` streams partial text
/// through the job's `partialResult`.
///
/// `isReady` reflects `SystemLanguageModel.default.availability`: the model needs Apple
/// Intelligence enabled on supported hardware, and `prepare()` reports the reason when it is not
/// available.
@objc(NFKFoundationModelsBackend)
public final class NFKFoundationModelsBackend: NSObject, NFKInferenceBackend {

    /// Tools the on-device model may call during generation. The model decides when to call one from
    /// its name, description, and parameters. Empty by default.
    @objc public var tools: [NFKFoundationTool] = []

    /// When set to a non-empty list of typed fields, the model generates a structured result matching
    /// them instead of free text. The result carries the parsed fields under `NFKOutputStructured`
    /// (a dictionary) and their JSON under `NFKOutputText`. Nil by default (free text).
    @objc public var responseSchema: [NFKFoundationToolParameter]?

    @objc public override init() {
        super.init()
    }

    // MARK: NFKInferenceBackend

    @objc public var isReady: Bool {
        SystemLanguageModel.default.availability == .available
    }

    @objc public var backendIdentifier: String { "foundation-models" }

    @objc(prepareWithError:)
    public func prepare() throws {
        switch SystemLanguageModel.default.availability {
        case .available:
            return
        case .unavailable(let reason):
            throw NSError(domain: NFKInferenceErrorDomain,
                          code: NFKInferenceError.error_InferenceNotReady.rawValue,
                          userInfo: [NSLocalizedDescriptionKey: "the system language model is unavailable: \(reason)"])
        @unknown default:
            throw NSError(domain: NFKInferenceErrorDomain,
                          code: NFKInferenceError.error_InferenceNotReady.rawValue,
                          userInfo: [NSLocalizedDescriptionKey: "the system language model is unavailable"])
        }
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
        throw NFKFoundationModelsError.noOutput
    }

    @objc(submitInferenceJobForRequest:)
    public func submitInferenceJob(for request: NFKInferenceRequest) -> NFKInferenceJob {
        let job = NFKInferenceJob()
        let tools = self.tools
        let responseSchema = self.responseSchema
        let task = Task.detached(priority: .userInitiated) {
            do {
                try self.prepare()
                let plan = Self.plan(for: request)
                let session = Self.makeSession(for: plan, tools: tools)
                let options = Self.generationOptions(for: request)

                if let fields = responseSchema, !fields.isEmpty {
                    let schema = try NFKSchema.generationSchema(name: "Response",
                                                               description: "the structured response",
                                                               properties: fields)
                    let response = try await session.respond(to: plan.prompt, schema: schema, options: options)
                    if Task.isCancelled {
                        job.cancel()
                        return
                    }
                    job.finish(with: NFKInferenceResult(outputs: [
                        NFKOutputStructured: NFKSchema.dictionary(from: response.content, properties: fields),
                        NFKOutputText: response.content.jsonString,
                    ]))
                    return
                }

                let stream = session.streamResponse(to: plan.prompt, options: options)
                var latest = ""
                for try await partial in stream {
                    if Task.isCancelled {
                        job.cancel()
                        return
                    }
                    latest = partial.content
                    job.reportProgress(-1, partialResult: NFKInferenceResult(outputs: [NFKOutputText: latest]))
                }
                job.finish(with: NFKInferenceResult(outputs: [NFKOutputText: latest]))
            } catch {
                if Task.isCancelled {
                    job.cancel()
                } else {
                    job.finish(withError: error)
                }
            }
        }
        job.cancellationHandler = { task.cancel() }
        return job
    }

    // MARK: Request mapping

    struct Turn: Equatable {
        var role: String
        var content: String
    }

    /// The instructions, prior conversation, and current prompt a request describes.
    struct RequestPlan: Equatable {
        var instructions: String?
        var history: [Turn]
        var prompt: String
    }

    /// Maps a request to a plan. System messages become instructions, earlier turns become history,
    /// and the final turn is the prompt. A plain `NFKInputPrompt` is a single-prompt plan.
    static func plan(for request: NFKInferenceRequest) -> RequestPlan {
        if let messages = request.input(forKey: NFKInputMessages) as? [[String: Any]] {
            var instructions: String?
            var turns: [Turn] = []
            for message in messages {
                let role = message["role"] as? String ?? "user"
                let content = message["content"] as? String ?? ""
                if role == "system" {
                    instructions = instructions.map { "\($0)\n\(content)" } ?? content
                } else {
                    turns.append(Turn(role: role, content: content))
                }
            }
            let prompt = turns.last?.content ?? ""
            return RequestPlan(instructions: instructions, history: Array(turns.dropLast()), prompt: prompt)
        }
        let prompt = request.input(forKey: NFKInputPrompt) as? String ?? ""
        return RequestPlan(instructions: nil, history: [], prompt: prompt)
    }

    /// Builds a session seeded with the plan's instructions and prior turns through the Foundation
    /// Models transcript, so multi-turn history is real conversation rather than a flattened string.
    /// Registered tools are adapted to Apple's `Tool` protocol and offered to the session; a tool
    /// whose schema cannot be built is skipped rather than failing the run.
    static func makeSession(for plan: RequestPlan, tools: [NFKFoundationTool]) -> LanguageModelSession {
        var entries: [Transcript.Entry] = []
        if let instructions = plan.instructions {
            entries.append(.instructions(Transcript.Instructions(
                segments: [.text(Transcript.TextSegment(content: instructions))],
                toolDefinitions: [])))
        }
        for turn in plan.history {
            let segment = Transcript.Segment.text(Transcript.TextSegment(content: turn.content))
            if turn.role == "assistant" {
                entries.append(.response(Transcript.Response(assetIDs: [], segments: [segment])))
            } else {
                entries.append(.prompt(Transcript.Prompt(segments: [segment])))
            }
        }

        let adapted: [any Tool] = tools.compactMap { try? NFKToolAdapter(tool: $0) }
        if adapted.isEmpty {
            return entries.isEmpty ? LanguageModelSession() : LanguageModelSession(transcript: Transcript(entries: entries))
        }
        return LanguageModelSession(tools: adapted, transcript: Transcript(entries: entries))
    }

    static func generationOptions(for request: NFKInferenceRequest) -> GenerationOptions {
        var temperature: Double?
        if let value = request.parameter(forKey: NFKParameterTemperature) as? NSNumber {
            temperature = value.doubleValue
        }
        var maximumResponseTokens: Int?
        if let value = request.parameter(forKey: NFKParameterMaxTokens) as? NSNumber {
            maximumResponseTokens = value.intValue
        }
        return GenerationOptions(temperature: temperature, maximumResponseTokens: maximumResponseTokens)
    }
}
