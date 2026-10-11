//
//  NFKMLXOrderedJSON.swift
//  InferKitMLX
//
//  A JSON reader that keeps each object's key order, which `JSONSerialization` discards. A chat
//  template writes a tool call's arguments back in the order the model wrote them, so the rendered
//  turn is the model's own text.
//

import Foundation

enum NFKJinjaOrderedJSON {
    /// The value `text` holds: `String`, `NSNumber` (a boolean as one), `NSNull`, `[Any]`, and
    /// `NFKJinjaOrderedObject` for an object. Nil when `text` is not exactly one JSON value.
    static func parse(_ text: String) -> Any? {
        var reader = Reader(Array(text.unicodeScalars))
        guard let value = reader.value() else { return nil }
        reader.skipSpace()
        return reader.atEnd ? value : nil
    }

    private struct Reader {
        let scalars: [Unicode.Scalar]
        var i = 0
        init(_ scalars: [Unicode.Scalar]) { self.scalars = scalars }

        var atEnd: Bool { i >= scalars.count }

        mutating func skipSpace() {
            while i < scalars.count, [" ", "\n", "\r", "\t"].contains(scalars[i]) { i += 1 }
        }

        mutating func value() -> Any? {
            skipSpace()
            guard i < scalars.count else { return nil }
            switch scalars[i] {
            case "{": return object()
            case "[": return array()
            case "\"": return string()
            case "t": return literal("true", NSNumber(value: true))
            case "f": return literal("false", NSNumber(value: false))
            case "n": return literal("null", NSNull())
            default: return number()
            }
        }

        mutating func literal(_ word: String, _ result: Any) -> Any? {
            let letters = Array(word.unicodeScalars)
            guard i + letters.count <= scalars.count, Array(scalars[i..<i + letters.count]) == letters else { return nil }
            i += letters.count
            return result
        }

        mutating func object() -> Any? {
            i += 1
            var pairs: [(String, Any)] = []
            skipSpace()
            if i < scalars.count, scalars[i] == "}" { i += 1; return NFKJinjaOrderedObject(pairs: pairs) }
            while true {
                skipSpace()
                guard i < scalars.count, scalars[i] == "\"", let key = string() else { return nil }
                skipSpace()
                guard i < scalars.count, scalars[i] == ":" else { return nil }
                i += 1
                guard let member = value() else { return nil }
                pairs.append((key, member))
                skipSpace()
                guard i < scalars.count else { return nil }
                if scalars[i] == "," { i += 1; continue }
                if scalars[i] == "}" { i += 1; return NFKJinjaOrderedObject(pairs: pairs) }
                return nil
            }
        }

        mutating func array() -> Any? {
            i += 1
            var items: [Any] = []
            skipSpace()
            if i < scalars.count, scalars[i] == "]" { i += 1; return items }
            while true {
                guard let item = value() else { return nil }
                items.append(item)
                skipSpace()
                guard i < scalars.count else { return nil }
                if scalars[i] == "," { i += 1; continue }
                if scalars[i] == "]" { i += 1; return items }
                return nil
            }
        }

        mutating func string() -> String? {
            i += 1
            var out = String.UnicodeScalarView()
            while i < scalars.count {
                let c = scalars[i]
                i += 1
                if c == "\"" { return String(out) }
                guard c == "\\" else { out.append(c); continue }
                guard i < scalars.count else { return nil }
                let e = scalars[i]
                i += 1
                switch e {
                case "\"": out.append("\"")
                case "\\": out.append("\\")
                case "/": out.append("/")
                case "b": out.append("\u{08}")
                case "f": out.append("\u{0C}")
                case "n": out.append("\n")
                case "r": out.append("\r")
                case "t": out.append("\t")
                case "u":
                    guard var unit = hex4() else { return nil }
                    // A surrogate pair spells one scalar beyond the basic plane.
                    if (0xD800...0xDBFF).contains(unit), i + 1 < scalars.count, scalars[i] == "\\", scalars[i + 1] == "u" {
                        i += 2
                        guard let low = hex4(), (0xDC00...0xDFFF).contains(low) else { return nil }
                        unit = 0x10000 + ((unit - 0xD800) << 10) + (low - 0xDC00)
                    }
                    guard let scalar = Unicode.Scalar(unit) else { return nil }
                    out.append(scalar)
                default: return nil
                }
            }
            return nil
        }

        mutating func hex4() -> UInt32? {
            guard i + 4 <= scalars.count, let value = UInt32(String(String.UnicodeScalarView(scalars[i..<i + 4])), radix: 16) else { return nil }
            i += 4
            return value
        }

        mutating func number() -> Any? {
            let start = i
            if i < scalars.count, scalars[i] == "-" { i += 1 }
            var floating = false
            while i < scalars.count {
                let c = scalars[i]
                if c >= "0" && c <= "9" {
                    i += 1
                } else if c == "." || c == "e" || c == "E" || ((c == "+" || c == "-") && floating) {
                    floating = true
                    i += 1
                } else {
                    break
                }
            }
            let text = String(String.UnicodeScalarView(scalars[start..<i]))
            if !floating, let integer = Int(text) { return NSNumber(value: integer) }
            return Double(text).map { NSNumber(value: $0) }
        }
    }
}
