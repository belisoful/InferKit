//
//  NFKMLXChatterboxMultilingual.swift
//  InferKit
//
//  The text layer of the multilingual Chatterbox release: the language tag the model is conditioned
//  on, and the per-language rewriting the reference applies before the byte-pair encoder sees the
//  text. The acoustic stack is shared with the English release; only the text embedding's width and
//  this layer differ.
//

import Foundation

/// Converts Chinese glyphs to the Cangjie codes the multilingual vocabulary spells them with.
///
/// The released `Cangjie5_TC.json` is a list of tab-separated rows whose first two fields are the
/// glyph and its code. Several glyphs share a code, and the reference disambiguates them by the
/// glyph's position among the rows carrying that code, so the table is read both ways.
public final class NFKMLXChatterboxCangjie {
    private let codeForGlyph: [String: String]
    private let glyphsForCode: [String: [String]]

    public init(url: URL) throws {
        let data = try Data(contentsOf: url)
        guard let rows = try JSONSerialization.jsonObject(with: data) as? [String] else {
            throw NFKMLXError.unsupportedConfiguration("\(url.lastPathComponent) is not a Cangjie table")
        }
        var codeForGlyph = [String: String]()
        var glyphsForCode = [String: [String]]()
        for row in rows {
            let fields = row.split(separator: "\t", omittingEmptySubsequences: false)
            guard fields.count >= 2 else { continue }
            let (glyph, code) = (String(fields[0]), String(fields[1]))
            codeForGlyph[glyph] = code
            glyphsForCode[code, default: []].append(glyph)
        }
        self.codeForGlyph = codeForGlyph
        self.glyphsForCode = glyphsForCode
    }

    /// The code for one glyph, carrying the index that separates glyphs sharing it. Nil for a
    /// character the table does not hold, which is how kana reach the vocabulary unchanged.
    func code(for glyph: String) -> String? {
        guard let code = codeForGlyph[glyph],
              let index = glyphsForCode[code]?.firstIndex(of: glyph) else {
            return nil
        }
        return index > 0 ? code + String(index) : code
    }

    /// Rewrites every "other letter" the table holds as its bracketed run, terminated by `[cj_.]`,
    /// and passes every other character through.
    ///
    /// The reference segments Chinese with pkuseg first and joins the words with spaces. That package
    /// is optional there, and the reference skips the step when it is absent, which is the path this
    /// matches.
    public func callAsFunction(_ text: String) -> String {
        var output = String.UnicodeScalarView()
        for scalar in text.unicodeScalars {
            guard scalar.properties.generalCategory == .otherLetter,
                  let code = code(for: String(scalar)) else {
                output.append(scalar)
                continue
            }
            for character in code {
                output.append(contentsOf: "[cj_\(character)]".unicodeScalars)
            }
            output.append(contentsOf: "[cj_.]".unicodeScalars)
        }
        return String(output)
    }
}

extension NFKMLXChatterbox {
    /// The language ids the multilingual release accepts, as the reference lists them.
    @objc public static let supportedLanguages = [
        "ar", "da", "de", "el", "en", "es", "fi", "fr", "he", "hi", "it", "ja",
        "ko", "ms", "nl", "no", "pl", "pt", "ru", "sv", "sw", "tr", "zh"
    ]
}

extension NFKMLXChatterboxTextTokenizer {
    /// Decomposes Hangul syllables into the Jamo the multilingual vocabulary spells Korean with, and
    /// passes any character outside the syllable block through.
    public static func koreanDecomposed(_ text: String) -> String {
        var output = String.UnicodeScalarView()
        for scalar in text.unicodeScalars {
            let value = Int(scalar.value)
            guard (0xAC00 ... 0xD7AF).contains(value) else {
                output.append(scalar)
                continue
            }
            let base = value - 0xAC00
            output.append(Unicode.Scalar(0x1100 + base / (21 * 28))!)
            output.append(Unicode.Scalar(0x1161 + (base % (21 * 28)) / 28)!)
            if base % 28 > 0 {
                output.append(Unicode.Scalar(0x11A7 + base % 28)!)
            }
        }
        return String(output).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// The ids for `text` in `language`, following the reference's order: lowercase, NFKD, the
    /// language's own rewriting, the bracketed language tag, and the byte-pair encoder.
    ///
    /// Chinese and Korean are rewritten here. The reference reaches for an optional package for
    /// Japanese (`pykakasi`), Hebrew (`dicta_onnx`) and Russian (`russian_text_stresser`), and passes
    /// the text through unchanged when one is absent; those three take that path here, so their text
    /// reaches the model as the reference sends it without its optional packages installed.
    public func encode(_ text: String, language: String?, lowercase: Bool = true,
                       nfkdNormalize: Bool = true) -> [Int] {
        var prepared = lowercase ? text.lowercased() : text
        if nfkdNormalize {
            prepared = prepared.decomposedStringWithCompatibilityMapping
        }
        switch language?.lowercased() {
        case "zh":
            if let cangjie { prepared = cangjie(prepared) }
        case "ko":
            prepared = Self.koreanDecomposed(prepared)
        default:
            break
        }
        if let language {
            prepared = "[\(language.lowercased())]" + prepared
        }
        return encode(prepared)
    }

    /// The text in `language` with the model's start and stop tokens, ready for T3.
    public func encodeForSynthesis(_ text: String, language: String?) -> [Int] {
        [startToken] + encode(text, language: language) + [stopToken]
    }
}
