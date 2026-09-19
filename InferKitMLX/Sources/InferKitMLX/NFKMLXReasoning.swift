//
//  NFKMLXReasoning.swift
//  InferKitMLX
//
//  Splitting a reasoning model's chain out of its answer, and the token counts a run reports.
//

import Foundation
import InferKit

/// The markers a reasoning model wraps its chain in, so the backend hands the chain back under
/// `NFKOutputReasoning` and leaves `NFKOutputText` holding the answer alone.
///
/// @discussion The shipped families disagree on the markers, so this is a value rather than a
/// constant. Qwen3 writes `<think>` and `</think>` around the chain and continues with the answer.
/// gpt-oss writes the harmony channels, opening an `analysis` message and then a `final` one. A
/// backend takes the format from the release's own chat template through ``detected(inChatTemplate:)``,
/// so the markers come from the same template that produces them, and a caller that renders its own
/// prompt names the format on the request instead.
///
/// Introduced in InferKit 0.4.0.
@objc(NFKMLXReasoningFormat)
public final class NFKMLXReasoningFormat: NSObject, @unchecked Sendable {

    /// What opens the chain.
    @objc public let opening: String

    /// What ends the chain.
    @objc public let closing: String

    /// What stands between the end of the chain and the start of the answer, empty where the answer
    /// follows the closing marker directly.
    @objc public let answerPrefix: String

    @objc public init(opening: String, closing: String, answerPrefix: String) {
        self.opening = opening
        self.closing = closing
        self.answerPrefix = answerPrefix
        super.init()
    }

    /// A format whose answer follows the closing marker with nothing in between.
    @objc public convenience init(opening: String, closing: String) {
        self.init(opening: opening, closing: closing, answerPrefix: "")
    }

    /// `<think>` … `</think>`, which Qwen3 and the releases that follow it write.
    @objc public static let thinkTags = NFKMLXReasoningFormat(opening: "<think>", closing: "</think>")

    /// The harmony channels gpt-oss writes: an `analysis` message holds the chain and a `final`
    /// message holds the answer.
    @objc public static let harmonyChannels = NFKMLXReasoningFormat(
        opening: "<|channel|>analysis<|message|>",
        closing: "<|end|>",
        answerPrefix: "<|start|>assistant<|channel|>final<|message|>")

    /// The format a release's chat template produces, or nil where the template shows no chain.
    ///
    /// @discussion The template is read rather than the model named, because the template is what
    /// renders a prior chain back into the prompt and so states the markers the release uses. Either
    /// marker is enough: a template that opens the block names the format as surely as one that
    /// closes it.
    @objc public static func detected(inChatTemplate template: String) -> NFKMLXReasoningFormat? {
        for format in [harmonyChannels, thinkTags]
        where template.contains(format.opening) || template.contains(format.closing) {
            return format
        }
        return nil
    }

    /// The format a request names: one of `think`, `harmony`, or `none`, or the markers themselves
    /// as two or three strings. Nil where the value names nothing this understands.
    static func named(_ value: Any) -> NFKMLXReasoningFormat?? {
        if let name = value as? String {
            switch name.lowercased() {
            case "think": return .some(thinkTags)
            case "harmony": return .some(harmonyChannels)
            case "none": return .some(nil)
            default: return nil
            }
        }
        guard let markers = value as? [String], markers.count == 2 || markers.count == 3 else {
            return nil
        }
        return .some(NFKMLXReasoningFormat(opening: markers[0], closing: markers[1],
                                           answerPrefix: markers.count == 3 ? markers[2] : ""))
    }

    public override func isEqual(_ object: Any?) -> Bool {
        guard let other = object as? NFKMLXReasoningFormat else { return false }
        return opening == other.opening && closing == other.closing && answerPrefix == other.answerPrefix
    }

    public override var hash: Int {
        var hasher = Hasher()
        hasher.combine(opening)
        hasher.combine(closing)
        hasher.combine(answerPrefix)
        return hasher.finalize()
    }

    /// Splits generated text into the chain the model showed and the answer it gave.
    ///
    /// @discussion A run that never closed its chain was cut off mid-thought, so all of it is
    /// reasoning and the answer is empty. Text with no chain at all is the answer, and `reasoning`
    /// is empty. The opening marker is optional on the way in: a template that pre-closes the block
    /// leaves the model nothing to open, and a prompt that already opened one leaves the model
    /// writing the chain straight away.
    func split(_ text: String) -> (reasoning: String, answer: String) {
        guard let end = text.range(of: closing) else {
            guard let start = text.range(of: opening) else {
                return ("", text)
            }
            return (String(text[start.upperBound...]).trimmingCharacters(in: .whitespacesAndNewlines), "")
        }
        var chain = String(text[..<end.lowerBound])
        if let start = chain.range(of: opening) {
            chain = String(chain[start.upperBound...])
        }
        var answer = String(text[end.upperBound...])
        if !answerPrefix.isEmpty, let prefix = answer.range(of: answerPrefix) {
            answer = String(answer[prefix.upperBound...])
        }
        return (chain.trimmingCharacters(in: .whitespacesAndNewlines),
                answer.trimmingCharacters(in: .whitespacesAndNewlines))
    }
}

/// The release's own Jinja `chat_template`: `chat_template.jinja` beside the weights, else the
/// `chat_template` string in `tokenizer_config.json`, else nil. Both spellings are in use, and a
/// release picks one.
func NFKMLXReleaseChatTemplate(inDirectory directory: URL) -> String? {
    if let text = try? String(contentsOf: directory.appendingPathComponent("chat_template.jinja"), encoding: .utf8),
       !text.isEmpty {
        return text
    }
    guard let data = try? Data(contentsOf: directory.appendingPathComponent("tokenizer_config.json")),
          let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
        return nil
    }
    if let template = json["chat_template"] as? String {
        return template
    }
    // Transformers 5.x writes a list of named templates; the default one is the chat template.
    guard let named = json["chat_template"] as? [[String: Any]] else { return nil }
    return (named.first { $0["name"] as? String == "default" } ?? named.first)?["template"] as? String
}

/// The token counts one generation run reports, in the shape `NFKOutputUsage` takes.
enum NFKMLXUsage {

    /// The usage dictionary for a run. A count the runtime cannot know is left out, so a caller
    /// reads a key that is present rather than trusting a zero.
    ///
    /// @discussion The runtime always knows the prompt and the reply, and knows what a retained
    /// prompt cache served. It knows the reasoning share only where a reasoning format applied,
    /// since without one it cannot say whether the text holds a chain at all.
    static func outputs(inputTokens: Int, cachedTokens: Int, outputTokens: Int,
                        reasoningTokens: Int?) -> [String: Int] {
        var usage = [NFKUsageInputTokens: inputTokens,
                     NFKUsageCachedTokens: cachedTokens,
                     NFKUsageOutputTokens: outputTokens]
        if let reasoningTokens {
            usage[NFKUsageReasoningTokens] = reasoningTokens
        }
        return usage
    }

    /// Refuses a request that asks for a reasoning effort of a model that does not reason, so the
    /// caller learns the level went nowhere rather than reading an answer that ignored it.
    static func refuseReasoningEffort(in request: NFKInferenceRequest, model: String) throws {
        guard request.parameter(forKey: NFKParameterReasoningEffort) != nil else { return }
        throw NFKMLXError.unsupportedConfiguration("\(model) is not a reasoning model")
    }

    /// How many of `produced` decode to the first `characters` characters of the reply.
    ///
    /// @discussion A detokenizer's output grows with its input, so the count is found by bisection
    /// rather than by decoding every prefix. It is the reasoning share of a reply whose chain runs
    /// to `characters`.
    static func tokenCount(inPrefixOf produced: [Int], characters: Int,
                           decode: ([Int]) -> String) -> Int {
        guard characters > 0 else { return 0 }
        var low = 0
        var high = produced.count
        while low < high {
            let middle = (low + high) / 2
            if decode(Array(produced[0 ... middle])).count >= characters {
                high = middle
            } else {
                low = middle + 1
            }
        }
        return Swift.min(low + 1, produced.count)
    }
}
