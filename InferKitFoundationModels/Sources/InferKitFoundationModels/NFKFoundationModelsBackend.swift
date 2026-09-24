//
//  NFKFoundationModelsBackend.swift
//  InferKitFoundationModels
//

import CoreGraphics
import Foundation
import FoundationModels
import InferKit

enum NFKFoundationModelsError: Error {
    case noOutput
}

/// `userInfo` keys on the errors the backend throws. Introduced in InferKit 0.4.0.
@objc public class NFKFoundationModelsErrorKey: NSObject {
    /// The tokens the request needs (`NSNumber`), on the error for a request that does not fit the
    /// model's context.
    @objc public static let tokenCount = "NFKFoundationModelsTokenCount"
    /// The tokens the model's context holds (`NSNumber`), on the same error.
    @objc public static let contextSize = "NFKFoundationModelsContextSize"
    /// When the Private Cloud Compute quota resets (`NSDate`), on the error for a reached quota,
    /// where the service reports it.
    @objc public static let resetDate = "NFKFoundationModelsResetDate"
}

// InferKit's request and job are immutable or internally locked, so they are safe to hand to the
// generation task.
extension NFKInferenceRequest: @retroactive @unchecked Sendable {}
extension NFKInferenceJob: @retroactive @unchecked Sendable {}

/// An InferKit backend that runs Apple's language models through the Foundation Models framework:
/// the on-device system model, or Apple's larger model on Private Cloud Compute.
///
/// It adopts the Objective-C `NFKInferenceBackend` protocol, so an InferKit consumer swaps it in
/// like any other engine: the request that runs against `NFKCoreMLLanguageBackend`, the MLX language
/// backend, or `NFKRemoteBackend` runs here with the same keys.
///
/// - Model: `model` chooses the on-device system model (the default) or Private Cloud Compute
///   (macOS 27 / iOS 27). `useCase` and `guardrails` specialize the on-device model.
/// - Input: `NFKInputPrompt` (a string) or `NFKInputMessages` (an OpenAI-style array). A system
///   message becomes the session's instructions; earlier turns seed the transcript, including
///   assistant `tool_calls` messages and `tool` results; the last user turn is the prompt.
///   `NFKInputImage` and `NFKInputImages` attach to the prompt on macOS 27 / iOS 27.
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
/// - Reasoning: `NFKParameterReasoningEffort` (light, moderate, or deep) becomes the context's
///   reasoning level on macOS 27 / iOS 27, and what the model showed comes back under
///   `NFKOutputReasoning`.
/// - Usage: `NFKOutputUsage` carries what the turn cost, on macOS 27 / iOS 27, which is where the
///   framework reports token counts.
/// - `submitInferenceJob(for:)` streams partial text through the job's `partialResult`.
///
/// `isReady` reflects the chosen model's availability: the on-device model needs Apple Intelligence
/// enabled on supported hardware; Private Cloud Compute needs macOS 27 / iOS 27, an eligible device,
/// and quota left. `prepare()` reports the reason when the model cannot take a request.
@objc(NFKFoundationModelsBackend)
public final class NFKFoundationModelsBackend: NSObject, NFKInferenceBackend {

    /// The tools with handlers. A request without `NFKParameterTools` offers all of them; a request
    /// with the key offers its own declarations and takes handlers from here by name. Empty by default.
    @objc public var tools: [NFKFoundationTool] = []

    /// The model requests run on. `.onDevice` by default. `.privateCloudCompute` needs macOS 27 /
    /// iOS 27: below that `isReady` is false and a request fails with
    /// `kNFKError_InferenceUnsupported`. A request captures the model when it is submitted, so a
    /// change here does not move a running request. Introduced in InferKit 0.4.0.
    @objc public var model: NFKFoundationModel {
        get { lock.withLock { configuration.model } }
        set { lock.withLock { configuration.model = newValue } }
    }

    /// The on-device model's specialization. `.general` by default. Private Cloud Compute ignores it.
    /// Introduced in InferKit 0.4.0.
    @objc public var useCase: NFKFoundationModelUseCase {
        get { lock.withLock { configuration.useCase } }
        set { lock.withLock { configuration.useCase = newValue } }
    }

    /// The on-device model's content guardrails. `.default` by default. Private Cloud Compute ignores
    /// it. Introduced in InferKit 0.4.0.
    @objc public var guardrails: NFKFoundationModelGuardrails {
        get { lock.withLock { configuration.guardrails } }
        set { lock.withLock { configuration.guardrails = newValue } }
    }

    /// The tokens the model's context holds. For the on-device model, 4096 below macOS 26.4 /
    /// iOS 26.4 and the model's own reading from there on. For Private Cloud Compute the service
    /// reports the size; `prepare()` reads it, and this is 0 until then. Introduced in InferKit 0.4.0.
    @objc public var contextSize: Int {
        let (configuration, cloudContextSize) = lock.withLock { (self.configuration, self.cloudContextSize) }
        switch configuration.model {
        case .onDevice:
            return configuration.systemModel.contextSize
        case .privateCloudCompute:
            return cloudContextSize
        }
    }

    /// The Private Cloud Compute quota, whatever `model` is set to, so an app decides before switching
    /// to it. Nil when the package was built with an SDK before macOS 27, which has no Private Cloud
    /// Compute. Introduced in InferKit 0.4.0.
    @available(macOS 27, iOS 27, *)
    @objc public var privateCloudComputeQuota: NFKFoundationModelQuota? {
        #if compiler(>=6.4)
        return NFKFoundationModelQuota(usage: PrivateCloudComputeLanguageModel().quotaUsage)
        #else
        return nil
        #endif
    }

    /// The on-device model's variant name (`SystemLanguageModel.Variant.displayName`). Nil when
    /// `model` is Private Cloud Compute, which reports no variant, and when the package was built
    /// with an SDK before macOS 27. Introduced in InferKit 0.4.0.
    @available(macOS 27, iOS 27, *)
    @objc public var variantDisplayName: String? {
        #if compiler(>=6.4)
        let configuration = lock.withLock { self.configuration }
        guard configuration.model == .onDevice else { return nil }
        return configuration.systemModel.variant.displayName
        #else
        return nil
        #endif
    }

    private let lock = NSLock()
    private var configuration = NFKFoundationModelConfiguration()
    private var cloudContextSize = 0
    private var prewarmedConfiguration: NFKFoundationModelConfiguration?

    @objc public override init() {
        super.init()
    }

    // MARK: NFKInferenceBackend

    @objc public var isReady: Bool {
        lock.withLock { configuration }.isReady
    }

    @objc public var backendIdentifier: String { "foundation-models" }

    /// The request parameters the backend reads. `NFKParameterReasoningEffort` is among them on
    /// macOS 27 / iOS 27, where the model takes a reasoning level. Introduced in InferKit 0.4.0.
    @objc public var supportedParameterKeys: Set<String> {
        var keys: Set<String> = [NFKParameterTemperature, NFKParameterMaxTokens, NFKParameterTopK,
                                 NFKParameterTopP, NFKParameterSeed, NFKParameterJSONSchema,
                                 NFKParameterChoices, NFKParameterTools]
        #if compiler(>=6.4)
        if #available(macOS 27, iOS 27, *) {
            keys.insert(NFKParameterReasoningEffort)
        }
        #endif
        return keys
    }

    /// The request inputs the backend reads. The image inputs are among them on macOS 27 / iOS 27,
    /// where a prompt takes an image attachment. Introduced in InferKit 0.4.0.
    @objc public var supportedInputKeys: Set<String> {
        var keys: Set<String> = [NFKInputPrompt, NFKInputMessages]
        #if compiler(>=6.4)
        if #available(macOS 27, iOS 27, *) {
            keys.formUnion([NFKInputImage, NFKInputImages])
        }
        #endif
        return keys
    }

    /// Checks the chosen model's availability and loads its resources once, so the first request
    /// does not pay the warm-up. For Private Cloud Compute it also reads `contextSize`.
    @objc(prepareWithError:)
    public func prepare() throws {
        let configuration = lock.withLock { self.configuration }
        try configuration.checkAvailability()
        #if compiler(>=6.4)
        if configuration.model == .privateCloudCompute, #available(macOS 27, iOS 27, *) {
            do {
                let size = try Self.readCloudContextSize()
                lock.withLock { cloudContextSize = size }
            } catch {
                throw NFKFoundationModelsFailure.coreError(for: error)
            }
        }
        #endif
        let warm = lock.withLock { prewarmedConfiguration == configuration }
        if !warm {
            try configuration.makeSession(tools: [], entries: []).prewarm()
            lock.withLock { prewarmedConfiguration = configuration }
        }
    }

    #if compiler(>=6.4)
    /// The service reports the context size asynchronously; `prepare()` is the synchronous seam that
    /// is allowed to wait for it.
    @available(macOS 27, iOS 27, *)
    private static func readCloudContextSize() throws -> Int {
        let outcome = NFKOutcome<Int>()
        let semaphore = DispatchSemaphore(value: 0)
        Task.detached(priority: .userInitiated) {
            do {
                outcome.succeed(try await PrivateCloudComputeLanguageModel().contextSize)
            } catch {
                outcome.fail(error)
            }
            semaphore.signal()
        }
        semaphore.wait()
        return try outcome.value()
    }
    #endif

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
        let configuration = lock.withLock { self.configuration }
        let task = Task.detached(priority: .userInitiated) {
            do {
                try configuration.checkAvailability()
                let plan = Self.plan(for: request)
                let prompt = Self.prompt(text: plan.prompt, images: try Self.images(in: request))
                let effort = try Self.reasoningEffort(for: request)
                let recorder = NFKToolCallRecorder()
                let adapted = try Self.toolAdapters(for: request, registered: registered, recorder: recorder)
                let session = try Self.makeSession(for: plan, tools: adapted, configuration: configuration)
                let options = try Self.generationOptions(for: request)
                let format = try Self.outputFormat(for: request)
                try await Self.checkContext(plan: plan, prompt: prompt, tools: adapted, format: format,
                                            configuration: configuration)

                var latestText = ""
                var reported: [String: Any] = [:]
                do {
                    switch format {
                    case .text:
                        let stream = Self.textStream(from: session, prompt: prompt, options: options,
                                                     reasoningEffort: effort)
                        for try await partial in stream {
                            if Task.isCancelled {
                                job.cancel()
                                return
                            }
                            latestText = partial.content
                            reported = Self.runOutputs(of: partial)
                            job.reportProgress(-1, partialResult: NFKInferenceResult(outputs: Self.outputs(text: latestText, reported: reported)))
                        }
                        job.finish(with: NFKInferenceResult(outputs: Self.outputs(text: latestText, reported: reported)))

                    case .schema(let schema):
                        let stream = Self.schemaStream(from: session, prompt: prompt, schema: schema,
                                                       options: options, reasoningEffort: effort)
                        var latest: GeneratedContent?
                        for try await partial in stream {
                            if Task.isCancelled {
                                job.cancel()
                                return
                            }
                            latest = partial.content
                            latestText = partial.content.jsonString
                            reported = Self.runOutputs(of: partial)
                            job.reportProgress(-1, partialResult: NFKInferenceResult(outputs: Self.outputs(text: latestText, reported: reported)))
                        }
                        guard let content = latest else {
                            throw NFKFoundationModelsError.noOutput
                        }
                        var outputs = Self.outputs(text: content.jsonString, reported: reported)
                        if let parsed = NFKSchema.jsonObject(from: content) {
                            outputs[NFKOutputStructured] = parsed
                        }
                        job.finish(with: NFKInferenceResult(outputs: outputs))

                    case .choice(let schema):
                        let response = try await Self.choiceResponse(from: session, prompt: prompt, schema: schema,
                                                                     options: options, reasoningEffort: effort)
                        if Task.isCancelled {
                            job.cancel()
                            return
                        }
                        let chosen = (try? response.content.value(String.self)) ?? response.content.jsonString
                        job.finish(with: NFKInferenceResult(outputs: Self.outputs(text: chosen, reported: Self.runOutputs(of: response))))
                    }
                } catch let error as LanguageModelSession.ToolCallError where error.underlyingError is NFKUnhandledToolCall {
                    // The model called a tool the caller runs: the turn ends here, the way a remote
                    // backend returns a tool-call turn, and the caller answers with a `tool` message.
                    var outputs = reported
                    outputs[NFKOutputToolCalls] = recorder.recorded.map(\.dictionary)
                    if !latestText.isEmpty {
                        outputs[NFKOutputText] = latestText
                    }
                    job.finish(with: NFKInferenceResult(outputs: outputs))
                }
            } catch {
                if Task.isCancelled {
                    job.cancel()
                } else {
                    job.finish(withError: NFKFoundationModelsFailure.coreError(for: error))
                }
            }
        }
        job.cancellationHandler = { task.cancel() }
        return job
    }

    // MARK: Context

    /// Counts the request's tokens against the context before the session does, so an oversized
    /// request fails with the core's error and both numbers, on an OS whose model counts tokens.
    /// Only the on-device model counts tokens; a Private Cloud Compute request is checked by the
    /// service.
    static func checkContext(plan: RequestPlan, prompt: Prompt? = nil, tools: [any Tool], format: OutputFormat,
                             configuration: NFKFoundationModelConfiguration = .init()) async throws {
        guard configuration.model == .onDevice, #available(macOS 26.4, iOS 26.4, *) else { return }
        let model = configuration.systemModel
        var count = try await model.tokenCount(for: prompt ?? Prompt(plan.prompt))
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

    // MARK: Images, reasoning, and usage

    /// The images a request carries, `NFKInputImage` first and then `NFKInputImages`. An image needs
    /// macOS 27 / iOS 27, where a prompt takes an attachment, so a request that carries one is
    /// refused below that rather than answered without it.
    static func images(in request: NFKInferenceRequest) throws -> [CGImage] {
        var sources: [Any] = []
        if let image = request.input(forKey: NFKInputImage) {
            sources.append(image)
        }
        if let images = request.input(forKey: NFKInputImages) as? [Any] {
            sources.append(contentsOf: images)
        }
        if sources.isEmpty {
            return []
        }
        #if compiler(>=6.4)
        if #available(macOS 27, iOS 27, *) {
            return try sources.map { source in
                guard let image = NFKImageCoding.cgImage(forImage: source) else {
                    throw NSError(domain: NFKInferenceErrorDomain,
                                  code: NFKInferenceError.error_InferenceMissingInput.rawValue,
                                  userInfo: [NSLocalizedDescriptionKey: "an image is not a CGImage, CVPixelBuffer, or BGRA/RGBA texture"])
                }
                return image
            }
        }
        #endif
        throw NSError(domain: NFKInferenceErrorDomain,
                      code: NFKInferenceError.error_InferenceUnsupported.rawValue,
                      userInfo: [NSLocalizedDescriptionKey: "an image input needs macOS 27 / iOS 27"])
    }

    /// The prompt one turn sends: the text, and the images the request attached after it. The images
    /// come from `images(in:)`, which refuses them below macOS 27 / iOS 27, so the list is empty on
    /// an OS whose prompt takes no attachment.
    static func prompt(text: String, images: [CGImage]) -> Prompt {
        #if compiler(>=6.4)
        if #available(macOS 27, iOS 27, *), !images.isEmpty {
            return Prompt {
                text
                for image in images {
                    Attachment(image)
                }
            }
        }
        #endif
        return Prompt(text)
    }

    /// The reasoning level a request asks for, read here so a request that asks for reasoning the OS
    /// cannot give fails before generation starts. Nil when the request asks for none.
    static func reasoningEffort(for request: NFKInferenceRequest) throws -> String? {
        guard let effort = request.parameter(forKey: NFKParameterReasoningEffort) else {
            return nil
        }
        guard let name = effort as? String, !name.isEmpty else {
            throw NSError(domain: NFKInferenceErrorDomain,
                          code: NFKInferenceError.error_InferenceUnsupported.rawValue,
                          userInfo: [NSLocalizedDescriptionKey: "NFKParameterReasoningEffort is a level name: light, moderate, or deep"])
        }
        #if compiler(>=6.4)
        if #available(macOS 27, iOS 27, *) {
            return name
        }
        #endif
        throw NSError(domain: NFKInferenceErrorDomain,
                      code: NFKInferenceError.error_InferenceUnsupported.rawValue,
                      userInfo: [NSLocalizedDescriptionKey: "a reasoning level needs macOS 27 / iOS 27"])
    }

    /// The outputs a reply carries: its text, and whatever the run reported beside it.
    static func outputs(text: String, reported: [String: Any]) -> [String: Any] {
        var outputs = reported
        outputs[NFKOutputText] = text
        return outputs
    }

    /// What the framework reports beside the reply: the reasoning the model showed under
    /// `NFKOutputReasoning`, and what the turn cost under `NFKOutputUsage`. Both arrive on macOS 27 /
    /// iOS 27; below it a run reports neither.
    static func runOutputs<Content>(of snapshot: LanguageModelSession.ResponseStream<Content>.Snapshot) -> [String: Any] {
        #if compiler(>=6.4)
        if #available(macOS 27, iOS 27, *) {
            return runOutputs(entries: snapshot.transcriptEntries, usage: snapshot.usage)
        }
        #endif
        return [:]
    }

    static func runOutputs<Content>(of response: LanguageModelSession.Response<Content>) -> [String: Any] {
        #if compiler(>=6.4)
        if #available(macOS 27, iOS 27, *) {
            return runOutputs(entries: response.transcriptEntries, usage: response.usage)
        }
        #endif
        return [:]
    }

    #if compiler(>=6.4)

    /// The framework's level for each level the contract names. Another string becomes the
    /// framework's custom level, which reaches a model that names one of its own.
    @available(macOS 27, iOS 27, *)
    static func reasoningLevel(named name: String?) -> ContextOptions.ReasoningLevel? {
        switch name {
        case nil: return nil
        case NFKReasoningEffortLight: return .light
        case NFKReasoningEffortModerate: return .moderate
        case NFKReasoningEffortDeep: return .deep
        case let other?: return .custom(other)
        }
    }

    @available(macOS 27, iOS 27, *)
    static func runOutputs(entries: ArraySlice<Transcript.Entry>,
                           usage: LanguageModelSession.Usage) -> [String: Any] {
        var outputs: [String: Any] = [
            NFKOutputUsage: [NFKUsageInputTokens: usage.input.totalTokenCount,
                             NFKUsageCachedTokens: usage.input.cachedTokenCount,
                             NFKUsageOutputTokens: usage.output.totalTokenCount,
                             NFKUsageReasoningTokens: usage.output.reasoningTokenCount],
        ]
        let reasoning = entries.compactMap { entry -> String? in
            guard case .reasoning(let reasoning) = entry else {
                return nil
            }
            return NFKInferKitLanguageModelRequest.text(of: reasoning.segments)
        }.joined(separator: "\n")
        if !reasoning.isEmpty {
            outputs[NFKOutputReasoning] = reasoning
        }
        return outputs
    }

    #endif

    // MARK: Generation calls

    // The macOS 27 SDK adds `contextOptions` to every respond and stream call, which is where the
    // reasoning level rides. The 26 SDKs have neither the parameter nor the type, so the compiler
    // version selects the call that the SDK at hand offers.

    static func textStream(from session: LanguageModelSession, prompt: Prompt, options: GenerationOptions,
                           reasoningEffort: String?) -> LanguageModelSession.ResponseStream<String> {
        #if compiler(>=6.4)
        if #available(macOS 27, iOS 27, *) {
            return session.streamResponse(to: prompt, options: options,
                                          contextOptions: ContextOptions(reasoningLevel: reasoningLevel(named: reasoningEffort)))
        }
        #endif
        return session.streamResponse(to: prompt, options: options)
    }

    static func schemaStream(from session: LanguageModelSession, prompt: Prompt, schema: GenerationSchema,
                             options: GenerationOptions,
                             reasoningEffort: String?) -> LanguageModelSession.ResponseStream<GeneratedContent> {
        #if compiler(>=6.4)
        if #available(macOS 27, iOS 27, *) {
            return session.streamResponse(to: prompt, schema: schema, options: options,
                                          contextOptions: ContextOptions(includeSchemaInPrompt: true,
                                                                         reasoningLevel: reasoningLevel(named: reasoningEffort)))
        }
        #endif
        return session.streamResponse(to: prompt, schema: schema, options: options)
    }

    static func choiceResponse(from session: LanguageModelSession, prompt: Prompt, schema: GenerationSchema,
                               options: GenerationOptions,
                               reasoningEffort: String?) async throws -> LanguageModelSession.Response<GeneratedContent> {
        #if compiler(>=6.4)
        if #available(macOS 27, iOS 27, *) {
            return try await session.respond(to: prompt, schema: schema, options: options,
                                             contextOptions: ContextOptions(includeSchemaInPrompt: true,
                                                                            reasoningLevel: reasoningLevel(named: reasoningEffort)))
        }
        #endif
        return try await session.respond(to: prompt, schema: schema, options: options)
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

    /// Builds a session over the configured model, seeded with the plan's instructions and prior
    /// turns through the Foundation Models transcript and offering the adapted tools.
    static func makeSession(for plan: RequestPlan, tools: [any Tool],
                            configuration: NFKFoundationModelConfiguration = .init()) throws -> LanguageModelSession {
        try configuration.makeSession(tools: tools, entries: transcriptEntries(for: plan))
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
        var options = GenerationOptions(temperature: temperature, maximumResponseTokens: maximumResponseTokens)
        setSamplingMode(samplingMode, on: &options)
        return options
    }

    // The macOS 27 SDK renames `GenerationOptions.sampling` to `samplingMode` and deprecates the old
    // spelling; the 26 SDKs have only the old one. Xcode 27 is the first toolchain with Swift 6.4, so
    // the compiler version selects the spelling the SDK at hand accepts without a warning.
    static func setSamplingMode(_ mode: GenerationOptions.SamplingMode?, on options: inout GenerationOptions) {
        #if compiler(>=6.4)
        options.samplingMode = mode
        #else
        options.sampling = mode
        #endif
    }

    static func samplingMode(of options: GenerationOptions) -> GenerationOptions.SamplingMode? {
        #if compiler(>=6.4)
        return options.samplingMode
        #else
        return options.sampling
        #endif
    }
}

/// Carries one asynchronous result across a semaphore to a synchronous caller.
final class NFKOutcome<Value: Sendable>: @unchecked Sendable {
    private let lock = NSLock()
    private var result: Result<Value, Error>?

    func succeed(_ value: Value) {
        lock.withLock { result = .success(value) }
    }

    func fail(_ error: Error) {
        lock.withLock { result = .failure(error) }
    }

    func value() throws -> Value {
        guard let result = lock.withLock({ result }) else {
            throw NFKFoundationModelsError.noOutput
        }
        return try result.get()
    }
}
