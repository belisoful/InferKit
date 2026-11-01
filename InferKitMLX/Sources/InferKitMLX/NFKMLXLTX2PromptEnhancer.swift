//
//  NFKMLXLTX2PromptEnhancer.swift
//  InferKitMLX
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import MLX

// LTX-2.5's prompt enhancer: a dedicated Gemma 4 E2B (`prompt_enhancer/`, read by `NFKMLXGemmaLanguage`) that
// rewrites a short request into the long audio-visual caption the video model was trained on, as diffusers'
// `LTX2Pipeline.enhance_prompt` runs it (matching `ltx-core`'s `enhance_t2v`). The request becomes the user
// turn `user prompt: <request>` after the system prompt, rendered by the `processor/` chat template with the
// generation prompt; the reply decodes greedily for at most 600 tokens with no repeated 5-gram, stops at the
// release's end tokens (1, 106, and 50), drops special tokens, and is cleaned as the reference cleans it:
// curly quotes, dashes, the no-break space, the prime, and the minus sign become their ASCII forms, and
// everything before the first letter goes.

/// LTX-2.5's prompt enhancer.
///
/// Introduced in InferKit 0.4.0.
@objc(NFKMLXLTX2PromptEnhancer)
public final class NFKMLXLTX2PromptEnhancer: NSObject {

    /// The reference's text-to-video system prompt (`LTX2_5_T2V_DEFAULT_SYSTEM_PROMPT` in diffusers'
    /// `pipelines/ltx2/utils.py`, Apache-2.0, from Lightricks' `gemma_t2v_system_prompt.txt`), byte for byte.
    @objc public static let defaultSystemPrompt = #"""
You are given a user's short text-to-video request. Write a single, highly detailed audio-visual caption describing the video that best fulfills that request, in the EXACT style of the training captions used for this video model. The generated video is scored against the user's ORIGINAL request, so preserve every element the user stated; expand faithfully into the full caption style without contradicting or dropping anything they asked for.

Match this captioning style precisely:

1. Begin immediately with the action or visual detail. Do NOT use "The scene opens…", "We see…", "There is…".

2. Objective, observable description only. Do not infer emotions or intentions — describe what is visible and audible (e.g. not "he looks sad" but "his eyebrows angle downward and his lips are pressed together").

3. Full visual detail: environment (materials, textures, lighting, colors), character appearance (clothing, posture, facial details), and the spatial positioning of all elements. When a human appears, identify them specifically (gendered terms when clearly implied; differentiate multiple people consistently) and describe visible physical attributes — apparent gender presentation, skin tone, estimated age group, hair color/length/style, build, clothing and accessories. Do not infer ethnicity, nationality, religion, or culture.

4. Precise motion and cinematic description. For every shot you MUST include, woven naturally into the prose (never as tags or labels):
   - Shot type (exactly one: extreme wide shot / wide shot / medium shot / medium close-up / close-up / extreme close-up)
   - Camera motion (always stated; if none, explicitly say the camera remains static). Camera movement is expected and good — match the user if they specified it, otherwise choose the treatment that best presents the requested scene.
   - Camera viewpoint relative to subject (front-facing / back-facing / side view / over-the-shoulder / top-down / low-angle / high-angle).
   Express these as flowing prose: "a medium shot frames…, captured from a front-facing angle as the camera slowly pans…". Never as "medium shot, static camera —".

5. Complete soundscape, integrated naturally: any dialogue (quote it exactly, in the original language), tone of voice, background music (type, mood, volume changes), and environmental sounds (footsteps, wind, traffic, animals). If the request implies sound, describe it plausibly.

6. Strict chronological, real-time flow using transitions like "Initially…", "A moment later…", "Simultaneously…". Keep every stated action in motion.

7. One single continuous paragraph. No bullet points, no section headers, no labels like "Audio:" or "Visual:". Exhaustive and lossless — include background elements, subtle movements, lighting, secondary sounds — detailed enough to reconstruct the scene. Aim for a rich, complete paragraph (roughly 150–220 words).

If the user wrote in another language, produce the English caption of the same content. Output ONLY the caption text — no JSON, no preamble.

AESTHETIC QUALITY (in addition to the above, without breaking the objective caption style): render the described scene with strong visual production value — cinematic, film-grade color and contrast, beautiful natural lighting, crisp fine detail and texture, pleasing composition and depth. Weave these quality descriptors naturally into the same observable prose (e.g. "warm cinematic lighting", "richly saturated film-grade color", "crisp high-resolution detail") — describe how the exact requested scene LOOKS at its most visually striking, never adding new objects or actions. Keep everything else (framing triple, soundscape, chronological single paragraph, faithfulness) exactly as specified.

"""#

    /// The system turn, ``defaultSystemPrompt`` unless set.
    @objc public var systemPrompt: String = NFKMLXLTX2PromptEnhancer.defaultSystemPrompt
    /// The longest reply, 600 tokens by the reference's default.
    @objc public var maximumNewTokens: Int = 600

    static let repeatedNgramSize = 5
    static let endTokens: Set<Int> = [1, 106, 50]

    /// The last position's logits `[1, vocabulary]` for token ids `[1, length]`.
    let logits: (MLXArray) -> MLXArray
    let tokenizer: NFKMLXGemmaTokenizer
    let template: String

    init(logits: @escaping (MLXArray) -> MLXArray, tokenizer: NFKMLXGemmaTokenizer, template: String) {
        self.logits = logits
        self.tokenizer = tokenizer
        self.template = template
    }

    /// Assembles the enhancer from a directory holding the release's `prompt_enhancer/` and `processor/`
    /// folders (an LTX-2.5 release directory). The decoder loads at its stored bfloat16.
    @objc(enhancerWithDirectoryURL:error:)
    public static func enhancer(directoryURL: URL) throws -> NFKMLXLTX2PromptEnhancer {
        let processor = directoryURL.appendingPathComponent("processor")
        guard let tokenizer = NFKMLXGemmaTokenizer(directoryURL: processor) else {
            throw NFKMLXError.unsupportedConfiguration("the enhancer's processor/ has no readable tokenizer.json")
        }
        let template = try String(contentsOf: processor.appendingPathComponent("chat_template.jinja"), encoding: .utf8)
        let decoder = try NFKMLXGemmaLanguage.textDecoder(
            directoryURL: directoryURL.appendingPathComponent("prompt_enhancer"), precision: .checkpoint,
            residency: .resident, streams: false)
        guard let net = decoder.module as? NFKMLXGemmaNet else {
            throw NFKMLXError.unsupportedConfiguration("the enhancer's prompt_enhancer/ is not a Gemma 4 E-series decoder")
        }
        return NFKMLXLTX2PromptEnhancer(logits: { net.lastLogits($0) }, tokenizer: tokenizer, template: template)
    }

    /// The prompt's token ids: the system prompt and `user prompt: <prompt>` through the chat template with
    /// the generation prompt.
    func tokens(for prompt: String) throws -> [Int] {
        let text = try NFKMLXChatTemplateRenderer.render(
            template, messages: [["role": "system", "content": systemPrompt],
                                 ["role": "user", "content": "user prompt: \(prompt)"]],
            addGenerationPrompt: true, bosToken: "<bos>", eosToken: "<eos>")
        return tokenizer.encode(text)
    }

    /// The reply's token ids for `prompt` ids: greedy, never completing a 5-gram the sequence already holds.
    func reply(to prompt: [Int]) -> [Int] {
        var sequence = prompt
        var produced = [Int]()
        for _ in 0 ..< maximumNewTokens {
            let input = MLXArray(sequence.map(Int32.init)).reshaped([1, sequence.count])
            var scores = logits(input)[0].asType(.float32)
            let banned = NFKMLXSeq2SeqDecoder.repeatedNgramCompletions(sequence, size: Self.repeatedNgramSize)
            if !banned.isEmpty {
                var mask = [Float](repeating: 0, count: scores.dim(0))
                for token in banned { mask[token] = -Float.infinity }
                scores = scores + MLXArray(mask)
            }
            let next = scores.argMax(axis: -1).item(Int.self)
            if Self.endTokens.contains(next) { break }
            produced.append(next)
            sequence.append(next)
        }
        return produced
    }

    /// `prompt` rewritten as the long audio-visual caption the video model was trained on.
    @objc(enhancedPrompt:error:)
    public func enhancedPrompt(_ prompt: String) throws -> String {
        Self.cleaned(tokenizer.decode(reply(to: try tokens(for: prompt)), skipSpecial: true))
    }

    /// The reference's `clean_response`: the typographic quotes, dashes, spaces, and signs it maps become
    /// ASCII, and everything before the first letter goes.
    static func cleaned(_ text: String) -> String {
        let replacements: [Character: String] = [
            "\u{2018}": "'", "\u{2019}": "'", "\u{201C}": "\"", "\u{201D}": "\"", "\u{2014}": "-",
            "\u{2013}": "-", "\u{00A0}": " ", "\u{2032}": "'", "\u{2212}": "-",
        ]
        let mapped = String(text.flatMap { replacements[$0] ?? String($0) })
        guard let first = mapped.firstIndex(where: \.isLetter) else { return mapped }
        return String(mapped[first...])
    }
}
