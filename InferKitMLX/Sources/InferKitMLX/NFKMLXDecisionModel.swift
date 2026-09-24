//
//  NFKMLXDecisionModel.swift
//  InferKitMLX
//
//  The inference contract for the typed-decision models the package ports beyond Laya: the request
//  `NFKTypeSafeBackend` reads, answered on device by whichever model the backend wraps.
//

import Foundation
import InferKit

/// A model that answers typed decisions about a state.
public protocol NFKMLXDecisionModel: AnyObject {
    /// The name the backend reports under `NFKOutputStructured`'s `model`.
    var decisionModelName: String { get }

    /// Answers the questions in the order given, with the input tokens the answers cost.
    func decide(state: Any, identifiedQuestions: [(identifier: String, question: NFKDecisionQuestion)])
        throws -> (answers: [String: NFKDecisionAnswer], inputTokens: Int)
}

/// A typed-decision model behind the inference contract.
///
/// @discussion `NFKInputState` (a string, or a JSON-serializable dictionary or array; `NFKInputPrompt`
/// and then `NFKInputMessages` stand in for it) and `NFKInputQuestions` (a dictionary of
/// `NFKDecisionQuestion`s, or of dictionaries in the wire shape) in; `NFKDecisionAnswer`s under
/// `NFKOutputAnswers`, the reply in the hosted service's shape under `NFKOutputStructured`, and the
/// token count under `NFKOutputUsage` out. A dictionary carries no order, and a model that reads every
/// question in one sequence answers by position, so the questions are read in sorted identifier order.
/// Blocks for the passes; run it off the render thread.
@objc(NFKMLXDecisionBackend)
public final class NFKMLXDecisionBackend: NSObject, NFKInferenceBackend {

    public let model: any NFKMLXDecisionModel

    public init(model: any NFKMLXDecisionModel) {
        self.model = model
        super.init()
    }

    /// The wrapped model's name.
    @objc public var modelName: String { model.decisionModelName }

    @objc public var isReady: Bool { true }
    @objc public var backendIdentifier: String { "mlx-\(model.decisionModelName)" }
    @objc public var supportedParameterKeys: Set<String> { [] }
    @objc public var supportedInputKeys: Set<String> { [NFKInputState, NFKInputQuestions, NFKInputPrompt, NFKInputMessages] }

    @objc(runInferenceForRequest:error:)
    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        guard let state = NFKMLXLayaBackend.state(in: request) else {
            throw NSError(domain: NFKInferenceErrorDomain, code: NFKInferenceError.error_InferenceMissingInput.rawValue,
                          userInfo: [NSLocalizedDescriptionKey: "the request carries no state under NFKInputState, NFKInputPrompt, or NFKInputMessages"])
        }
        let questions = NFKMLXLayaBackend.questions(in: request)
        guard !questions.isEmpty else {
            throw NSError(domain: NFKInferenceErrorDomain, code: NFKInferenceError.error_InferenceMissingInput.rawValue,
                          userInfo: [NSLocalizedDescriptionKey: "the request carries no questions under NFKInputQuestions"])
        }
        let ordered = questions.keys.sorted().map { (identifier: $0, question: questions[$0]!) }
        let (answers, tokens) = try model.decide(state: state, identifiedQuestions: ordered)
        let wire = answers.mapValues { $0.raw }
        let reply: [String: Any] = ["model": model.decisionModelName, "answers": wire,
                                    "usage": ["input_tokens": tokens, "output_tokens": 0]]
        return NFKInferenceResult(outputs: [NFKOutputAnswers: answers, NFKOutputStructured: reply,
                                            NFKOutputUsage: [NFKUsageInputTokens: tokens, NFKUsageOutputTokens: 0]])
    }
}

/// Encodes text to token ids for a decision model: a closure, so a release's own tokenizer and a
/// test's stand-in plug in alike.
public final class NFKMLXDecisionTokenizer: @unchecked Sendable {
    private let encoder: (String) -> [Int]

    public init(_ encode: @escaping (String) -> [Int]) {
        encoder = encode
    }

    /// A SentencePiece model read from its `.model` file, ids as the model's piece indices.
    public convenience init(sentencePiece segmenter: NFKMLXSentencePieceSegmenter) {
        self.init { segmenter.encode($0) }
    }

    /// A core tokenizer, which splits out its special tokens before encoding the text between them.
    public convenience init(tokenizer: NFKTokenizer) {
        self.init { tokenizer.encode($0).map(\.intValue) }
    }

    public func encode(_ text: String) -> [Int] { encoder(text) }
}

extension NFKDecisionAnswer {
    /// An answer in the wire shape, which the initializer always reads for these three types.
    static func decided(_ wire: [String: Any]) -> NFKDecisionAnswer {
        NFKDecisionAnswer(dictionary: wire)!
    }
}

extension NFKDecisionQuestion {
    /// The options a candidate-scoring model reads, in order: a choice's names (with the description
    /// after a colon where one is given), a score's levels, or a noul's `no` then `yes` (each with its
    /// meaning after a colon where one is given).
    var renderedOptions: [String] {
        switch type {
        case .choice:
            return options.map { name in descriptions[name].map { "\(name): \($0)" } ?? name }
        case .score:
            return options
        case .noul:
            return [("no", "false"), ("yes", "true")].map { word, key in
                descriptions[key].map { "\(word): \($0)" } ?? word
            }
        @unknown default:
            return options
        }
    }

    /// The names the answer keys its probabilities by: a choice's option names, a score's level
    /// indices, a noul's `false` and `true`.
    var answerKeys: [String] {
        switch type {
        case .choice: return options
        case .score: return options.indices.map(String.init)
        default: return ["false", "true"]
        }
    }
}
