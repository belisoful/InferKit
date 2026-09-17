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

/// `userInfo` keys on the error thrown when a request does not fit the model's context.
/// Introduced in InferKit 0.4.0.
@objc public class NFKFoundationModelsErrorKey: NSObject {
    /// The tokens the request needs (`NSNumber`).
    @objc public static let tokenCount = "NFKFoundationModelsTokenCount"
    /// The tokens the model's context holds (`NSNumber`).
    @objc public static let contextSize = "NFKFoundationModelsContextSize"
}

// InferKit's request and job are immutable or internally locked, so they are safe to hand to the
// generation task.
extension NFKInferenceRequest: @retroactive @unchecked Sendable {}
extension NFKInferenceJob: @retroactive @unchecked Sendable {}

/// An InferKit backend that runs Apple's on-device system language model through the Foundation
/// Models framework.
///
/// It adopts the Objective-C `NFKInferenceBackend` protocol, so an InferKit consumer swaps it in
/// like any other engine: the request that runs against `NFKCoreMLLanguageBackend`, the MLX language
/// backend, or `NFKRemoteBackend` runs here with the same keys.
///
/// - Input: `NFKInputPrompt` (a string) or `NFKInputMessages` (an OpenAI-style array). A system
///   message becomes the session's instructions; earlier turns seed the transcript, including
///   assistant `tool_calls` messages and `tool` results; the last user turn is the prompt.
/// - Sampling: `NFKParameterTemperature` and `NFKParameterMaxTokens` map to `GenerationOptions`;
///   `NFKParameterTopK`, `NFKParameterTopP`, and `NFKParameterSeed` choose the sampling mode, and a
///   temperature of zero is greedy decoding.
/// - Structured output: `NFKParameterJSONSchema` (a JSON Schema object) constrains generation to the
///   schema; the parsed object rides under `NFKOutputStructured` and its JSON under `NFKOutputText`.
///   `NFKParameterChoices` constrains the reply to exactly one of the strings.
/// - Tools: `NFKParameterTools` declares the tools a request offers; a registered `NFKFoundationTool`
///   of the same name supplies the handler. Without the key, every registered tool is offered. A
///   declared tool with no handler ends the turn with the call under `NFKOutputToolCalls`, and the
///   caller replies with a `tool` message.
/// - `submitInferenceJob(for:)` streams partial text through the job's `partialResult`.
///
/// `isReady` reflects `SystemLanguageModel.default.availability`: the model needs Apple
/// Intelligence enabled on supported hardware, and `prepare()` reports the reason when it is not
/// available.
@objc(NFKFoundationModelsBackend)
public final class NFKFoundationModelsBackend: NSObject, NFKInferenceBackend {

    /// The tools with handlers. A request without `NFKParameterTools` offers all of them; a request
    /// with the key offers its own declarations and takes handlers from here by name. Empty by default.
    @objc public var tools: [NFKFoundationTool] = []

    /// The tokens the model's context holds. 4096 below macOS 26.4 / iOS 26.4; the model's own reading
    /// from there on. Introduced in InferKit 0.4.0.
    @objc public var contextSize: Int {
        SystemLanguageModel.default.contextSize
    }

    private let prewarmed = NFKOnce()

    @objc public override init() {
        super.init()
    }

    // MARK: NFKInferenceBackend

    @objc public var isReady: Bool {
        SystemLanguageModel.default.availability == .available
    }

    @objc public var backendIdentifier: String { "foundation-models" }

    /// Checks availability and loads the model's resources once, so the first request does not pay
    /// the warm-up.
    @objc(prepareWithError:)
    public func prepare() throws {
        try Self.checkAvailability()
        prewarmed.run {
            LanguageModelSession().prewarm()
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
        let registered = self.tools
        let task = Task.detached(priority: .userInitiated) {
            do {
                try Self.checkAvailability()
                let plan = Self.plan(for: request)
                let recorder = NFKToolCallRecorder()
                let adapted = try Self.toolAdapters(for: request, registered: registered, recorder: recorder)
                let session = Self.makeSession(for: plan, tools: adapted)
                let options = try Self.generationOptions(for: request)
                let format = try Self.outputFormat(for: request)
                try await Self.checkContext(plan: plan, tools: adapted, format: format)

                var latestText = ""
                do {
                    switch format {
                    case .text:
                        let stream = session.streamResponse(to: plan.prompt, options: options)
                        for try await partial in stream {
                            if Task.isCancelled {
                                job.cancel()
                                return
                            }
                            latestText = partial.content
                            job.reportProgress(-1, partialResult: NFKInferenceResult(outputs: [NFKOutputText: latestText]))
                        }
                        job.finish(with: NFKInferenceResult(outputs: [NFKOutputText: latestText]))

                    case .schema(let schema):
                        let stream = session.streamResponse(to: plan.prompt, schema: schema, options: options)
                        var latest: GeneratedContent?
                        for try await partial in stream {
                            if Task.isCancelled {
                                job.cancel()
                                return
                            }
                            latest = partial.content
                            latestText = partial.content.jsonString
                            job.reportProgress(-1, partialResult: NFKInferenceResult(outputs: [NFKOutputText: latestText]))
                        }
                        guard let content = latest else {
                            throw NFKFoundationModelsError.noOutput
                        }
                        var outputs: [String: Any] = [NFKOutputText: content.jsonString]
                        if let parsed = NFKSchema.jsonObject(from: content) {
                            outputs[NFKOutputStructured] = parsed
                        }
                        job.finish(with: NFKInferenceResult(outputs: outputs))

                    case .choice(let schema):
                        let response = try await session.respond(to: plan.prompt, schema: schema, options: options)
                        if Task.isCancelled {
                            job.cancel()
                            return
                        }
                        let chosen = (try? response.content.value(String.self)) ?? response.content.jsonString
                        job.finish(with: NFKInferenceResult(outputs: [NFKOutputText: chosen]))
                    }
                } catch let error as LanguageModelSession.ToolCallError where error.underlyingError is NFKUnhandledToolCall {
                    // The model called a tool the caller runs: the turn ends here, the way a remote
                    // backend returns a tool-call turn, and the caller answers with a `tool` message.
                    var outputs: [String: Any] = [NFKOutputToolCalls: recorder.recorded.map(\.dictionary)]
                    if !latestText.isEmpty {
                        outputs[NFKOutputText] = latestText
                    }
                    job.finish(with: NFKInferenceResult(outputs: outputs))
                }
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

    // MARK: Availability

    static func checkAvailability() throws {
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

    /// Counts the request's tokens against the context before the session does, so an oversized
    /// request fails with the core's error and both numbers, on an OS whose model counts tokens.
    static func checkContext(plan: RequestPlan, tools: [any Tool], format: OutputFormat) async throws {
        guard #available(macOS 26.4, iOS 26.4, *) else { return }
        let model = SystemLanguageModel.default
        var count = try await model.tokenCount(for: plan.prompt)
        count += try await model.tokenCount(for: transcriptEntries(for: plan))
        if !tools.isEmpty {
            count += try await model.tokenCount(for: tools)
        }
        switch format {
        case .text: break
        case .schema(let schema), .choice(let schema):
            count += try await model.tokenCount(for: schema)
        }
        let contextSize = model.contextSize
        if count > contextSize {
            throw NSError(domain: NFKInferenceErrorDomain,
                          code: NFKInferenceError.error_InferenceUnsupported.rawValue,
                          userInfo: [NSLocalizedDescriptionKey: "the request needs \(count) tokens; the model's context holds \(contextSize)",
                                     NFKFoundationModelsErrorKey.tokenCount: count,
                                     NFKFoundationModelsErrorKey.contextSize: contextSize])
        }
    }

    // MARK: Request mapping

    /// One tool call in a prior assistant turn.
    struct ToolCallTurn: Equatable {
        var id: String
        var name: String
        var argumentsJSON: String
    }

    /// One prior turn of the conversation.
    enum Turn: Equatable {
        case user(String)
        case assistant(String)
        case toolCalls([ToolCallTurn])
        case toolOutput(id: String, name: String, content: String)
    }

    /// The instructions, prior conversation, and current prompt a request describes.
    struct RequestPlan: Equatable {
        var instructions: String?
        var history: [Turn]
        var prompt: String
    }

    /// Maps a request to a plan. System messages become instructions, the last user message is the
    /// prompt, and every other message is history in order: assistant text, assistant `tool_calls`
    /// (the OpenAI wire shape, or the core's `NFKOutputToolCalls` entries handed back as they came),
    /// and `tool` results keyed by `tool_call_id`. Tool turns after the last user message stay in the
    /// history, so a tool result reaches the model and the question is asked again over it. A plain
    /// `NFKInputPrompt` is a single-prompt plan.
    static func plan(for request: NFKInferenceRequest) -> RequestPlan {
        guard let messages = request.input(forKey: NFKInputMessages) as? [[String: Any]] else {
            let prompt = request.input(forKey: NFKInputPrompt) as? String ?? ""
            return RequestPlan(instructions: nil, history: [], prompt: prompt)
        }
        var instructions: String?
        var turns: [Turn] = []
        var toolNamesByID: [String: String] = [:]
        for message in messages {
            let role = message["role"] as? String ?? "user"
            let content = message["content"] as? String ?? ""
            switch role {
            case "system":
                instructions = instructions.map { "\($0)\n\(content)" } ?? content
            case "assistant":
                if let calls = toolCallTurns(in: message), !calls.isEmpty {
                    calls.forEach { toolNamesByID[$0.id] = $0.name }
                    turns.append(.toolCalls(calls))
                }
                if !content.isEmpty {
                    turns.append(.assistant(content))
                }
            case "tool":
                let id = message["tool_call_id"] as? String ?? ""
                let name = message["name"] as? String ?? toolNamesByID[id] ?? ""
                turns.append(.toolOutput(id: id, name: name, content: content))
            default:
                turns.append(.user(content))
            }
        }
        guard let promptIndex = turns.lastIndex(where: { if case .user = $0 { return true } else { return false } }),
              case .user(let prompt) = turns[promptIndex] else {
            return RequestPlan(instructions: instructions, history: turns, prompt: "")
        }
        turns.remove(at: promptIndex)
        return RequestPlan(instructions: instructions, history: turns, prompt: prompt)
    }

    /// The tool calls an assistant message carries, in either the OpenAI wire shape
    /// (`{id, type, function: {name, arguments}}`) or the core's result shape (`{id, name, arguments,
    /// argumentsJSON}`).
    static func toolCallTurns(in message: [String: Any]) -> [ToolCallTurn]? {
        guard let calls = message["tool_calls"] as? [[String: Any]] else { return nil }
        return calls.compactMap { call in
            let function = call["function"] as? [String: Any]
            guard let name = (function?["name"] ?? call["name"]) as? String else { return nil }
            let id = call["id"] as? String ?? UUID().uuidString
            let argumentsJSON: String
            if let json = (function?["arguments"] ?? call["argumentsJSON"]) as? String {
                argumentsJSON = json
            } else if let arguments = call["arguments"],
                      let data = try? JSONSerialization.data(withJSONObject: arguments),
                      let json = String(data: data, encoding: .utf8) {
                argumentsJSON = json
            } else {
                argumentsJSON = "{}"
            }
            return ToolCallTurn(id: id, name: name, argumentsJSON: argumentsJSON)
        }
    }

    /// The transcript entries for a plan's instructions and history.
    static func transcriptEntries(for plan: RequestPlan) -> [Transcript.Entry] {
        var entries: [Transcript.Entry] = []
        if let instructions = plan.instructions {
            entries.append(.instructions(Transcript.Instructions(
                segments: [.text(Transcript.TextSegment(content: instructions))],
                toolDefinitions: [])))
        }
        for turn in plan.history {
            switch turn {
            case .user(let content):
                entries.append(.prompt(Transcript.Prompt(segments: [.text(Transcript.TextSegment(content: content))])))
            case .assistant(let content):
                entries.append(.response(Transcript.Response(assetIDs: [],
                                                             segments: [.text(Transcript.TextSegment(content: content))])))
            case .toolCalls(let calls):
                let transcriptCalls = calls.map { call in
                    let arguments = (try? GeneratedContent(json: call.argumentsJSON)) ?? GeneratedContent(properties: [:])
                    return Transcript.ToolCall(id: call.id, toolName: call.name, arguments: arguments)
                }
                entries.append(.toolCalls(Transcript.ToolCalls(transcriptCalls)))
            case .toolOutput(let id, let name, let content):
                entries.append(.toolOutput(Transcript.ToolOutput(id: id, toolName: name,
                                                                 segments: [.text(Transcript.TextSegment(content: content))])))
            }
        }
        return entries
    }

    /// Builds a session seeded with the plan's instructions and prior turns through the Foundation
    /// Models transcript, offering the adapted tools.
    static func makeSession(for plan: RequestPlan, tools: [any Tool]) -> LanguageModelSession {
        let entries = transcriptEntries(for: plan)
        if tools.isEmpty {
            return entries.isEmpty ? LanguageModelSession() : LanguageModelSession(transcript: Transcript(entries: entries))
        }
        return LanguageModelSession(tools: tools, transcript: Transcript(entries: entries))
    }

    /// The tools a request offers. `NFKParameterTools` entries (`{name, description, parameters}`)
    /// take their handlers from the registered tools by name; without the key, the registered tools
    /// are offered as declared.
    static func toolAdapters(for request: NFKInferenceRequest,
                             registered: [NFKFoundationTool],
                             recorder: NFKToolCallRecorder) throws -> [any Tool] {
        guard let declared = request.parameter(forKey: NFKParameterTools) else {
            return try registered.map { try NFKToolAdapter(tool: $0, recorder: recorder) }
        }
        guard let declarations = declared as? [[String: Any]] else {
            throw NSError(domain: NFKInferenceErrorDomain,
                          code: NFKInferenceError.error_InferenceUnsupported.rawValue,
                          userInfo: [NSLocalizedDescriptionKey: "NFKParameterTools is an array of {name, description, parameters}"])
        }
        let handlers = Dictionary(registered.map { ($0.name, $0.handler) }, uniquingKeysWith: { _, last in last })
        return try declarations.map { declaration in
            guard let name = declaration["name"] as? String else {
                throw NSError(domain: NFKInferenceErrorDomain,
                              code: NFKInferenceError.error_InferenceUnsupported.rawValue,
                              userInfo: [NSLocalizedDescriptionKey: "a tool declaration names its tool"])
            }
            return try NFKToolAdapter(name: name,
                                      description: declaration["description"] as? String ?? "",
                                      parameters: declaration["parameters"] as? [String: Any] ?? ["type": "object"],
                                      handler: handlers[name],
                                      recorder: recorder)
        }
    }

    /// What the reply is constrained to.
    enum OutputFormat {
        case text
        case schema(GenerationSchema)
        case choice(GenerationSchema)
    }

    /// `NFKParameterJSONSchema` and `NFKParameterChoices` become schemas. `NFKParameterOutputFormat`
    /// asks for JSON of no particular shape, which the framework's guided generation cannot express,
    /// so it is refused rather than ignored.
    static func outputFormat(for request: NFKInferenceRequest) throws -> OutputFormat {
        if let schema = request.parameter(forKey: NFKParameterJSONSchema) {
            guard let json = schema as? [String: Any] else {
                throw NSError(domain: NFKInferenceErrorDomain,
                              code: NFKInferenceError.error_InferenceUnsupported.rawValue,
                              userInfo: [NSLocalizedDescriptionKey: "NFKParameterJSONSchema is a JSON Schema object"])
            }
            return .schema(try NFKSchema.generationSchema(name: "Response", json: json))
        }
        if let choices = request.parameter(forKey: NFKParameterChoices) as? [String], !choices.isEmpty {
            return .choice(try NFKSchema.choiceSchema(choices))
        }
        if request.parameter(forKey: NFKParameterOutputFormat) != nil {
            throw NSError(domain: NFKInferenceErrorDomain,
                          code: NFKInferenceError.error_InferenceUnsupported.rawValue,
                          userInfo: [NSLocalizedDescriptionKey: "the system language model constrains JSON through NFKParameterJSONSchema, not NFKParameterOutputFormat"])
        }
        return .text
    }

    /// Temperature and the token limit map directly. The sampling mode follows the core keys: a
    /// temperature of zero is greedy, `NFKParameterTopK` samples among the top k, `NFKParameterTopP`
    /// samples within the probability mass, and `NFKParameterSeed` seeds either; a seed alone seeds
    /// sampling over the whole distribution.
    static func generationOptions(for request: NFKInferenceRequest) throws -> GenerationOptions {
        let temperature = (request.parameter(forKey: NFKParameterTemperature) as? NSNumber)?.doubleValue
        let maximumResponseTokens = (request.parameter(forKey: NFKParameterMaxTokens) as? NSNumber)?.intValue
        let topK = (request.parameter(forKey: NFKParameterTopK) as? NSNumber)?.intValue
        let topP = (request.parameter(forKey: NFKParameterTopP) as? NSNumber)?.doubleValue
        let seed = (request.parameter(forKey: NFKParameterSeed) as? NSNumber)?.uint64Value

        var samplingMode: GenerationOptions.SamplingMode?
        if let temperature, temperature <= 0 {
            samplingMode = .greedy
        } else if let topK {
            samplingMode = .random(top: topK, seed: seed)
        } else if let topP {
            samplingMode = .random(probabilityThreshold: topP, seed: seed)
        } else if let seed {
            samplingMode = .random(probabilityThreshold: 1, seed: seed)
        }
        return GenerationOptions(samplingMode: samplingMode,
                                 temperature: temperature,
                                 maximumResponseTokens: maximumResponseTokens)
    }
}

/// Runs a block once, from whichever thread reaches it first.
final class NFKOnce: @unchecked Sendable {
    private let lock = NSLock()
    private var done = false

    func run(_ block: () -> Void) {
        lock.lock()
        let first = !done
        done = true
        lock.unlock()
        if first {
            block()
        }
    }
}
