//
//  NFKMLXWanPromptCleaning.swift
//  InferKitMLX
//
//  The prompt cleanup of Wan-Video/Wan2.2's own tokenizer (`HuggingfaceTokenizer(clean="whitespace")`,
//  `whitespace_clean(basic_clean(text))`), which the official pipelines (S2V, and the releases built on them)
//  run before umT5 reads a prompt.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation

/// The official Wan tokenizer's prompt cleanup.
///
/// @discussion `basic_clean` runs `ftfy.fix_text` (ftfy 6.3), unescapes HTML twice (`html.unescape`), and
/// trims; `whitespace_clean` replaces each run of whitespace with one space and trims. Of `fix_text`'s default
/// fixes, every one that acts on valid Unicode applies, in its order and to a fixed point: HTML entities (where
/// the text holds no `<`), C1 controls read as Windows-1252, Latin ligatures expanded, fullwidth and halfwidth
/// forms brought to their standard width, curly quotes straightened, line breaks unified, terminal escapes and
/// control characters removed, and NFC. `fix_text`'s mojibake repair, which re-decodes text mis-read under
/// another encoding, is not applied, and of the named HTML references only `&amp;`, `&lt;`, `&gt;`, `&quot;`,
/// `&apos;`, and `&nbsp;` are read; numeric references are read as `html.unescape` reads them.
enum NFKMLXWanPromptCleaning {

    /// A prompt as the official tokenizer cleans it.
    static func cleaned(_ text: String) -> String {
        var fixed = text
        let unescapes = !text.contains("<")
        while true {
            let original = fixed
            if unescapes {
                fixed = entitiesUnescaped(fixed, numericOnlyWhenComplete: true)
            }
            fixed = mapped(fixed) { scalar in c1Controls[scalar.value].map { [$0] } }
            fixed = mapped(fixed) { scalar in ligatures[scalar.value].map { Array($0.unicodeScalars) } }
            fixed = mapped(fixed) { scalar in widthMap[scalar.value].map { [$0] } }
            fixed = mapped(fixed) { scalar in
                if singleQuotes.contains(scalar.value) { return ["'"] }
                if doubleQuotes.contains(scalar.value) { return ["\""] }
                return nil
            }
            fixed = lineBreaksUnified(fixed)
            fixed = fixed.replacingOccurrences(of: "\u{1B}\\[[0-9;]*[a-zA-Z]", with: "", options: .regularExpression)
            fixed = mapped(fixed) { scalar in controlCharacters.contains(scalar.value) ? [] : nil }
            fixed = fixed.precomposedStringWithCanonicalMapping
            if fixed == original {
                break
            }
        }
        fixed = entitiesUnescaped(entitiesUnescaped(fixed, numericOnlyWhenComplete: false), numericOnlyWhenComplete: false)
        return collapsedWhitespace(fixed)
    }

    // MARK: Tables

    /// Windows-1252's reading of the C1 control bytes it defines; the five it leaves undefined stay.
    static let c1Controls: [UInt32: Unicode.Scalar] = [
        0x80: "\u{20AC}", 0x82: "\u{201A}", 0x83: "\u{0192}", 0x84: "\u{201E}", 0x85: "\u{2026}", 0x86: "\u{2020}",
        0x87: "\u{2021}", 0x88: "\u{02C6}", 0x89: "\u{2030}", 0x8A: "\u{0160}", 0x8B: "\u{2039}", 0x8C: "\u{0152}",
        0x8E: "\u{017D}", 0x91: "\u{2018}", 0x92: "\u{2019}", 0x93: "\u{201C}", 0x94: "\u{201D}", 0x95: "\u{2022}",
        0x96: "\u{2013}", 0x97: "\u{2014}", 0x98: "\u{02DC}", 0x99: "\u{2122}", 0x9A: "\u{0161}", 0x9B: "\u{203A}",
        0x9C: "\u{0153}", 0x9E: "\u{017E}", 0x9F: "\u{0178}",
    ]

    /// ftfy's `LIGATURES`.
    static let ligatures: [UInt32: String] = [
        0x0132: "IJ", 0x0133: "ij", 0x0149: "\u{02BC}n", 0x01F1: "DZ", 0x01F2: "Dz", 0x01F3: "dz", 0x01C4: "D\u{017D}",
        0x01C5: "D\u{017E}", 0x01C6: "d\u{017E}", 0x01C7: "LJ", 0x01C8: "Lj", 0x01C9: "lj", 0x01CA: "NJ", 0x01CB: "Nj",
        0x01CC: "nj", 0xFB00: "ff", 0xFB01: "fi", 0xFB02: "fl", 0xFB03: "ffi", 0xFB04: "ffl", 0xFB05: "\u{017F}t",
        0xFB06: "st",
    ]

    /// ftfy's `WIDTH_MAP`: the ideographic space, and each fullwidth or halfwidth form whose NFKC form differs.
    static let widthMap: [UInt32: Unicode.Scalar] = {
        var map: [UInt32: Unicode.Scalar] = [0x3000: " "]
        for value in UInt32(0xFF01) ..< 0xFFF0 {
            guard let scalar = Unicode.Scalar(value) else { continue }
            let normalized = String(Character(scalar)).precomposedStringWithCompatibilityMapping.unicodeScalars
            if normalized.count == 1, let first = normalized.first, first != scalar {
                map[value] = first
            }
        }
        return map
    }()

    static let singleQuotes: Set<UInt32> = [0x02BC, 0x2018, 0x2019, 0x201A, 0x201B]
    static let doubleQuotes: Set<UInt32> = [0x201C, 0x201D, 0x201E, 0x201F]

    /// ftfy's `CONTROL_CHARS`, which `remove_control_chars` deletes.
    static let controlCharacters: Set<UInt32> = Set(Array(0x00 ... 0x08) + [0x0B] + Array(0x0E ... 0x1F) + [0x7F]
        + Array(0x206A ... 0x206F) + [0xFEFF] + Array(0xFFF9 ... 0xFFFC))

    static let namedEntities: [String: String] = ["amp": "&", "lt": "<", "gt": ">", "quot": "\"", "apos": "'",
                                                  "nbsp": "\u{00A0}"]
    /// The names `html.unescape` also reads without their semicolon.
    static let legacyEntities: Set<String> = ["amp", "lt", "gt", "quot", "nbsp"]

    // MARK: Steps

    static func mapped(_ text: String, _ replacement: (Unicode.Scalar) -> [Unicode.Scalar]?) -> String {
        var scalars = String.UnicodeScalarView()
        for scalar in text.unicodeScalars {
            if let replaced = replacement(scalar) {
                scalars.append(contentsOf: replaced)
            } else {
                scalars.append(scalar)
            }
        }
        return String(scalars)
    }

    /// CRLF, CR, the line and paragraph separators, and NEXT LINE, each as a line feed.
    static func lineBreaksUnified(_ text: String) -> String {
        var scalars = String.UnicodeScalarView()
        var previousCarriageReturn = false
        for scalar in text.unicodeScalars {
            if scalar == "\n", previousCarriageReturn {
                previousCarriageReturn = false
                continue
            }
            previousCarriageReturn = scalar == "\r"
            scalars.append(["\r", "\u{2028}", "\u{2029}", "\u{85}"].contains(scalar) ? "\n" : scalar)
        }
        return String(scalars)
    }

    /// HTML character references unescaped. `numericOnlyWhenComplete` reads them as ftfy's `unescape_html` does
    /// (`&#?[0-9A-Za-z]{1,24};` only, a numeric reference kept where it does not resolve); otherwise as
    /// `html.unescape` does (a numeric reference's semicolon optional, Python's replacements for invalid
    /// code points).
    static func entitiesUnescaped(_ text: String, numericOnlyWhenComplete: Bool) -> String {
        let scalars = Array(text.unicodeScalars)
        var output = String.UnicodeScalarView()
        var index = 0
        while index < scalars.count {
            guard scalars[index] == "&", let (replacement, length) = reference(scalars, at: index,
                                                                               complete: numericOnlyWhenComplete) else {
                output.append(scalars[index])
                index += 1
                continue
            }
            output.append(contentsOf: replacement.unicodeScalars)
            index += length
        }
        return String(output)
    }

    /// The reference starting at `start` (an ampersand) and the scalars it spans.
    private static func reference(_ s: [Unicode.Scalar], at start: Int, complete: Bool) -> (String, Int)? {
        var end = start + 1
        let isNumeric = end < s.count && s[end] == "#"
        if isNumeric {
            end += 1
            let hex = end < s.count && (s[end] == "x" || s[end] == "X")
            if hex { end += 1 }
            let digitsStart = end
            while end < s.count, hex ? s[end].properties.isASCIIHexDigit : ("0" ... "9").contains(s[end]) {
                end += 1
            }
            guard end > digitsStart else { return nil }
            let terminated = end < s.count && s[end] == ";"
            if complete && !terminated { return nil }
            let digits = String(String.UnicodeScalarView(s[digitsStart ..< end]))
            let length = end - start + (terminated ? 1 : 0)
            guard let value = UInt64(digits, radix: hex ? 16 : 10).map({ min($0, 0x110000) }) else { return nil }
            return (characterReference(UInt32(value)), length)
        }
        while end < s.count, end - start <= 24, s[end].properties.isAlphabetic || ("0" ... "9").contains(s[end]) {
            end += 1
        }
        let name = String(String.UnicodeScalarView(s[(start + 1) ..< end]))
        if end < s.count, s[end] == ";", let value = namedEntities[name] {
            return (value, end - start + 1)
        }
        if !complete {
            for legacy in legacyEntities where name.hasPrefix(legacy) {
                return (namedEntities[legacy]!, legacy.unicodeScalars.count + 1)
            }
        }
        return nil
    }

    /// `html.unescape`'s reading of a numeric reference: Windows-1252 for the C1 range, U+FFFD for NUL, surrogates,
    /// and values past Unicode, and nothing for the code points HTML calls invalid.
    private static func characterReference(_ value: UInt32) -> String {
        if value == 0x0D { return "\r" }
        if value == 0 || (0xD800 ... 0xDFFF).contains(value) || value > 0x10FFFF { return "\u{FFFD}" }
        if (0x80 ... 0x9F).contains(value) {
            return c1Controls[value].map { String($0) } ?? String(Unicode.Scalar(value)!)
        }
        let invalid = (0x01 ... 0x08).contains(value) || (0x0E ... 0x1F).contains(value) || (0x7F ... 0x9F).contains(value)
            || (0xFDD0 ... 0xFDEF).contains(value) || value == 0x0B || (value & 0xFFFE) == 0xFFFE
        return invalid ? "" : String(Unicode.Scalar(value)!)
    }

    /// Each run of whitespace as one space, trimmed (`re.sub(r"\s+", " ", text).strip()`).
    static func collapsedWhitespace(_ text: String) -> String {
        var scalars = String.UnicodeScalarView()
        var pendingSpace = false
        for scalar in text.unicodeScalars {
            if scalar.properties.isWhitespace || (0x1C ... 0x1F).contains(scalar.value) {
                pendingSpace = !scalars.isEmpty
                continue
            }
            if pendingSpace {
                scalars.append(" ")
                pendingSpace = false
            }
            scalars.append(scalar)
        }
        return String(scalars)
    }
}
