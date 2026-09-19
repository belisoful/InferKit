//
//  NFKInferKitLanguageModel.swift
//  InferKitFoundationModels
//

import CoreGraphics
import Foundation
import FoundationModels
import InferKit

/// The Foundation Models capabilities an InferKit backend reports. Introduced in InferKit 0.4.0.
///
/// The capabilities come from the keys the backend declares through `supportedParameterKeys` and
/// `supportedInputKeys`: a JSON Schema for guided generation, tool declarations for tool calling,
/// an image input for vision. A backend that declares no keys reports no capabilities, and the
/// consumer states them instead.
@objc(NFKInferKitLanguageModelCapabilities)
public final class NFKInferKitLanguageModelCapabilities: NSObject {

    /// The backend constrains a reply to a JSON Schema (`NFKParameterJSONSchema`).
    @objc public let guidedGeneration: Bool

    /// The backend takes tool declarations (`NFKParameterTools`).
    @objc public let toolCalling: Bool

    /// The backend takes an image beside the text (`NFKInputImage` or `NFKInputImages`).
    @objc public let vision: Bool

    /// Reads what the backend declares.
    @objc public init(backend: any NFKInferenceBackend) {
        let parameters = backend.supportedParameterKeys ?? []
        let inputs = backend.supportedInputKeys ?? []
        guidedGeneration = parameters.contains(NFKParameterJSONSchema)
        toolCalling = parameters.contains(NFKParameterTools)
        vision = inputs.contains(NFKInputImage) || inputs.contains(NFKInputImages)
        super.init()
    }

    /// States the capabilities directly, for a backend that declares no keys.
    @objc public init(guidedGeneration: Bool, toolCalling: Bool, vision: Bool) {
        self.guidedGeneration = guidedGeneration
        self.toolCalling = toolCalling
        self.vision = vision
        super.init()
    }
}

#if compiler(>=6.4)

/// An InferKit backend presented to Foundation Models as a language model, so an app runs it
/// through `LanguageModelSession` the way it runs Apple's own models. Needs macOS 27 / iOS 27, and
/// a build with the macOS 27 SDK. Introduced in InferKit 0.4.0.
///
/// ```swift
/// let backend = NFKRemoteBackend(endpointURL: url)
/// let session = LanguageModelSession(model: NFKInferKitLanguageModel(backend: backend))
/// let reply = try await session.respond(to: "Name three sea birds.")
/// ```
///
/// This is the reverse of ``NFKFoundationModelsBackend``, which presents Apple's models to an
/// InferKit consumer. The session's transcript becomes `NFKInputMessages`, its tool definitions
/// become `NFKParameterTools`, its response schema becomes `NFKParameterJSONSchema`, and its
/// generation options become the core's sampling parameters. The reply streams back through the
/// job's partial results.
///
/// The model reports the capabilities the backend declares, so a session refuses guided
/// generation, tool calling, or an image on a backend that takes none.
@available(macOS 27, iOS 27, *)
public struct NFKInferKitLanguageModel: LanguageModel {

    public typealias Executor = NFKInferKitLanguageModelExecutor

    /// What the framework hands the executor it builds for this model. The framework keeps one
    /// executor per distinct configuration, so the backend's identity is the configuration's.
    public struct Configuration: Hashable, @unchecked Sendable {

        /// The backend the model runs.
        public let backend: any NFKInferenceBackend

        public init(backend: any NFKInferenceBackend) {
            self.backend = backend
        }

        public static func == (lhs: Configuration, rhs: Configuration) -> Bool {
            lhs.backend === rhs.backend
        }

        public func hash(into hasher: inout Hasher) {
            hasher.combine(ObjectIdentifier(backend))
        }
    }

    public let capabilities: LanguageModelCapabilities

    public let executorConfiguration: Configuration

    /// Wraps a backend, taking its capabilities from the keys it declares.
    public init(backend: any NFKInferenceBackend) {
        self.init(backend: backend, capabilities: NFKInferKitLanguageModelCapabilities(backend: backend))
    }

    /// Wraps a backend with stated capabilities, for a backend that declares no keys.
    public init(backend: any NFKInferenceBackend, capabilities declared: NFKInferKitLanguageModelCapabilities) {
        var list: [LanguageModelCapabilities.Capability] = []
        if declared.guidedGeneration {
            list.append(.guidedGeneration)
        }
        if declared.toolCalling {
            list.append(.toolCalling)
        }
        if declared.vision {
            list.append(.vision)
        }
        self.capabilities = LanguageModelCapabilities(list)
        self.executorConfiguration = Configuration(backend: backend)
    }
}

/// Runs one Foundation Models generation request against an InferKit backend. The framework builds
/// it from a ``NFKInferKitLanguageModel``'s configuration. Introduced in InferKit 0.4.0.
@available(macOS 27, iOS 27, *)
public struct NFKInferKitLanguageModelExecutor: LanguageModelExecutor {

    public typealias Model = NFKInferKitLanguageModel
    public typealias Configuration = NFKInferKitLanguageModel.Configuration

    private let configuration: Configuration

    public init(configuration: Configuration) throws {
        self.configuration = configuration
    }

    /// Loads the backend's resources when it has any. A failure here is not fatal: the request
    /// reports it.
    public func prewarm(model: Model, transcript: Transcript) {
        NFKInferencePrepare(configuration.backend, nil)
    }

    public func respond(to request: LanguageModelExecutorGenerationRequest,
                        model: Model,
                        streamingInto channel: LanguageModelExecutorGenerationChannel) async throws {
        let inferenceRequest = try NFKInferKitLanguageModelRequest.inferenceRequest(for: request)
        let job = NFKInferenceSubmit(configuration.backend, inferenceRequest, nil)

        // The job reports the reply so far on its own thread; the stream carries those readings
        // into this task, and finishes when the job reaches a terminal state.
        let readings = AsyncStream<Reading> { continuation in
            job.progressHandler = { job in
                if let partial = job.partialResult {
                    continuation.yield(Reading(of: partial))
                }
            }
            job.completionHandler = { _ in continuation.finish() }
        }

        var sent = Reading()
        await withTaskCancellationHandler {
            for await reading in readings {
                sent = await Self.send(reading, after: sent, to: channel)
            }
        } onCancel: {
            job.cancel()
        }

        if let error = job.error {
            throw error
        }
        guard let result = job.result else {
            throw CancellationError()
        }

        _ = await Self.send(Reading(of: result), after: sent, to: channel)

        // The counts are the turn's totals, so they arrive once the turn is over; every append
        // above carries a count of zero because the core reports no per-token reading.
        if let usage = Self.usage(in: result) {
            await channel.send(.response(action: .updateUsage(input: usage.input, output: usage.output)))
        }
        for call in result.toolCalls ?? [] {
            guard let name = call["name"] as? String else {
                continue
            }
            let id = call["id"] as? String ?? UUID().uuidString
            let arguments = call["argumentsJSON"] as? String ?? "{}"
            await channel.send(.toolCalls(action: .toolCall(id: id, name: name,
                                                            action: .appendArguments(arguments, tokenCount: 0))))
        }
    }

    /// One reading of the reply so far: the answer, and the reasoning the model showed before it.
    struct Reading: Sendable {
        var text = ""
        var reasoning = ""

        init() {}

        init(of result: NFKInferenceResult) {
            text = result.text ?? ""
            reasoning = result.output(forKey: NFKOutputReasoning) as? String ?? ""
        }
    }

    /// Appends what a reading adds to each channel of the turn, and answers the reading the channel
    /// now carries.
    static func send(_ reading: Reading, after sent: Reading,
                     to channel: LanguageModelExecutorGenerationChannel) async -> Reading {
        var carried = sent
        let reasoning = appendix(sent: sent.reasoning, text: reading.reasoning)
        if !reasoning.isEmpty {
            carried.reasoning += reasoning
            await channel.send(.reasoning(action: .appendText(reasoning, tokenCount: 0)))
        }
        let text = appendix(sent: sent.text, text: reading.text)
        if !text.isEmpty {
            carried.text += text
            await channel.send(.response(action: .appendText(text, tokenCount: 0)))
        }
        return carried
    }

    /// The text to append to what the channel already carries. A backend reports the reply so far,
    /// so a longer reading contributes its new suffix. A reading that does not extend what was
    /// sent is a rewrite, which an append cannot express, so it contributes nothing.
    static func appendix(sent: String, text: String) -> String {
        guard text.hasPrefix(sent) else {
            return ""
        }
        return String(text.dropFirst(sent.count))
    }

    /// The token counts a result reports, in the shape the channel takes. A backend that reports no
    /// counts leaves them out rather than sending zeros.
    static func usage(in result: NFKInferenceResult)
        -> (input: LanguageModelExecutorGenerationChannel.Usage.Input,
            output: LanguageModelExecutorGenerationChannel.Usage.Output)? {
        guard let counts = result.output(forKey: NFKOutputUsage) as? [String: NSNumber] else {
            return nil
        }
        let count = { (key: String) in counts[key]?.intValue ?? 0 }
        return (.init(totalTokenCount: count(NFKUsageInputTokens),
                      cachedTokenCount: count(NFKUsageCachedTokens)),
                .init(totalTokenCount: count(NFKUsageOutputTokens),
                      reasoningTokenCount: count(NFKUsageReasoningTokens)))
    }
}

#endif

/// Maps a Foundation Models generation request onto the core's request keys.
enum NFKInferKitLanguageModelRequest {

    /// The conversation a transcript describes, in the OpenAI-style shape `NFKInputMessages` takes:
    /// instructions become a system message, a prompt a user message, a response an assistant
    /// message, tool calls an assistant message carrying `tool_calls`, and a tool output a `tool`
    /// message keyed by its call. An attachment rides on the image input instead.
    static func messages(for transcript: Transcript) -> [[String: Any]] {
        var messages: [[String: Any]] = []
        for entry in transcript {
            switch entry {
            case .instructions(let instructions):
                messages.append(["role": "system", "content": text(of: instructions.segments)])
            case .prompt(let prompt):
                messages.append(["role": "user", "content": text(of: prompt.segments)])
            case .response(let response):
                messages.append(["role": "assistant", "content": text(of: response.segments)])
            case .toolCalls(let calls):
                let wire = calls.map { call in
                    ["id": call.id,
                     "type": "function",
                     "function": ["name": call.toolName, "arguments": call.arguments.jsonString]] as [String: Any]
                }
                messages.append(["role": "assistant", "tool_calls": wire])
            case .toolOutput(let output):
                messages.append(["role": "tool",
                                 "tool_call_id": output.id,
                                 "name": output.toolName,
                                 "content": text(of: output.segments)])
            default:
                continue
            }
        }
        return messages
    }

    /// The text a segment list carries. A structured segment carries its JSON.
    static func text(of segments: [Transcript.Segment]) -> String {
        segments.compactMap { segment in
            switch segment {
            case .text(let text):
                return text.content
            case .structure(let structure):
                return structure.content.jsonString
            default:
                return nil
            }
        }.joined(separator: "\n")
    }

    /// Temperature and the token limit map directly. The sampling mode maps to the core's keys the
    /// way ``NFKFoundationModelsBackend`` reads them, so a round trip through both directions keeps
    /// its meaning: greedy decoding is a temperature of zero, and a seed rides beside top-k or
    /// top-p. A build against an SDK before macOS 27 cannot read the mode, which is opaque there,
    /// and drops it.
    static func parameters(for options: GenerationOptions) -> [String: Any] {
        var parameters: [String: Any] = [:]
        if let temperature = options.temperature {
            parameters[NFKParameterTemperature] = temperature
        }
        if let maximumResponseTokens = options.maximumResponseTokens {
            parameters[NFKParameterMaxTokens] = maximumResponseTokens
        }
        #if compiler(>=6.4)
        if #available(macOS 27, iOS 27, *),
           let mode = NFKFoundationModelsBackend.samplingMode(of: options) {
            switch mode.kind {
            case .greedy:
                parameters[NFKParameterTemperature] = 0
            case .randomTopK(let k, let seed):
                parameters[NFKParameterTopK] = k
                if let seed {
                    parameters[NFKParameterSeed] = seed
                }
            case .randomProbabilityThreshold(let threshold, let seed):
                parameters[NFKParameterTopP] = threshold
                if let seed {
                    parameters[NFKParameterSeed] = seed
                }
            @unknown default:
                break
            }
        }
        #endif
        return parameters
    }

    /// The JSON Schema a generation schema describes. The framework encodes a schema as JSON
    /// Schema, which is the shape `NFKParameterJSONSchema` and `NFKParameterTools` take.
    static func schemaJSON(for schema: GenerationSchema) throws -> [String: Any] {
        let data = try JSONEncoder().encode(schema)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw NSError(domain: NFKInferenceErrorDomain,
                          code: NFKInferenceError.error_InferenceUnsupported.rawValue,
                          userInfo: [NSLocalizedDescriptionKey: "a generation schema did not encode to a JSON Schema object"])
        }
        return json
    }

    #if compiler(>=6.4)

    /// The images a transcript attaches, in order, for the core's image input.
    @available(macOS 27, iOS 27, *)
    static func images(in transcript: Transcript) -> [CGImage] {
        var images: [CGImage] = []
        for entry in transcript {
            let segments: [Transcript.Segment]
            switch entry {
            case .instructions(let instructions): segments = instructions.segments
            case .prompt(let prompt): segments = prompt.segments
            case .response(let response): segments = response.segments
            case .toolOutput(let output): segments = output.segments
            case .toolCalls: continue
            default: continue
            }
            for segment in segments {
                guard case .attachment(let attachment) = segment,
                      case .image(let image) = attachment.content else {
                    continue
                }
                images.append(image.cgImage)
            }
        }
        return images
    }

    /// The tool declarations `NFKParameterTools` takes: a name, a description, and the JSON Schema
    /// of the arguments.
    @available(macOS 27, iOS 27, *)
    static func toolDeclarations(for definitions: [Transcript.ToolDefinition]) throws -> [[String: Any]] {
        try definitions.map { definition in
            ["name": definition.name,
             "description": definition.description,
             "parameters": try schemaJSON(for: definition.parameters)]
        }
    }

    /// The name `NFKParameterReasoningEffort` takes for a reasoning level. The framework's three
    /// named levels are the contract's three; a custom level goes out under its own name, which
    /// reaches a backend that names it.
    @available(macOS 27, iOS 27, *)
    static func reasoningEffort(for level: ContextOptions.ReasoningLevel?) -> String? {
        switch level {
        case nil: return nil
        case .light: return NFKReasoningEffortLight
        case .moderate: return NFKReasoningEffortModerate
        case .deep: return NFKReasoningEffortDeep
        case .custom(let name): return name
        @unknown default: return nil
        }
    }

    /// The core request one generation request describes.
    ///
    /// `.disallowed` tool calling drops the declarations. `.required` has no core key, so a
    /// request that demands a call is sent with the declarations and the backend decides.
    @available(macOS 27, iOS 27, *)
    static func inferenceRequest(for request: LanguageModelExecutorGenerationRequest) throws -> NFKInferenceRequest {
        var inputs: [String: Any] = [NFKInputMessages: messages(for: request.transcript)]
        let images = self.images(in: request.transcript)
        if images.count == 1 {
            inputs[NFKInputImage] = images[0]
        } else if images.count > 1 {
            inputs[NFKInputImages] = images
        }

        var parameters = self.parameters(for: request.generationOptions)
        if let schema = request.schema {
            parameters[NFKParameterJSONSchema] = try schemaJSON(for: schema)
        }
        if !request.enabledToolDefinitions.isEmpty,
           request.generationOptions.toolCallingMode?.kind != .disallowed {
            parameters[NFKParameterTools] = try toolDeclarations(for: request.enabledToolDefinitions)
        }
        if let effort = reasoningEffort(for: request.contextOptions.reasoningLevel) {
            parameters[NFKParameterReasoningEffort] = effort
        }
        return NFKInferenceRequest(inputs: inputs, parameters: parameters, outputModality: .text)
    }

    #endif
}
