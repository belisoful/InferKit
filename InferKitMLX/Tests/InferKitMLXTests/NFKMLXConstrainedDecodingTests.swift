//
//  NFKMLXConstrainedDecodingTests.swift
//  InferKitMLXTests
//

import XCTest
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXConstrainedDecodingTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test`; run via xcodebuild")
    }

    /// A vocabulary sized to the tiny decoder: single bytes at their own ids, JSON-shaped pieces
    /// above them, and the last id as the end token.
    private func toyVocabulary() -> NFKMLXVocabulary {
        let size = NFKMLXLanguageConfiguration.tiny.vocabularySize
        var tokens = (0 ..< 256).map { [UInt8($0)] }
        let pieces = ["{\"", "\":", ", \"", "\"}", "true", "false", "null", "123", "\"name\"", "\"id\"",
                      "[", "]", "{", "}", " ", "\"", "42", "0.5", "-7", "1e3", "yes", "no", "maybe", "may",
                      "\"a\"", "\"b\"", "\\n", "\\u00", "ab", "cd", "hello", "world"]
        for piece in pieces { tokens.append(Array(piece.utf8)) }
        while tokens.count < size - 1 { tokens.append(Array("x\(tokens.count)".utf8)) }
        tokens.append([])                                                     // the end token
        return NFKMLXVocabulary(tokens: tokens, endToken: size - 1)
    }

    private func text(_ tokens: [Int], _ vocabulary: NFKMLXVocabulary) -> String {
        String(decoding: tokens.flatMap { vocabulary.tokens[$0] }, as: UTF8.self)
    }

    // MARK: The JSON grammar

    func testTheJSONGrammarAcceptsValidPrefixesAndRejectsInvalidOnes() {
        let json = NFKMLXJSONConstraint(vocabulary: toyVocabulary())
        for prefix in ["{", "{\"a\": 1", "{\"a\": [1, 2, {\"b\": null}], \"c\": \"x\\\"y\"", "[",
                       "  {\n\"k\"\t:\ttrue", "{\"n\": -0.5e+3", "{\"s\": \"\\u00e9", "[\"multi byte é\""] {
            XCTAssertTrue(json.accepts(prefix), "\(prefix) is a valid prefix")
        }
        for invalid in ["}", "{a", "{\"a\" 1", "{\"a\": 01", "{\"a\": 1.}", "{\"a\": tru3", "{\"a\": \"\\x\"",
                        "[1,]", "{,}", "\"scalar root\"", "{\"a\":1}}", "{\"a\":1} 2"] {
            XCTAssertFalse(json.accepts(invalid), "\(invalid) is rejected")
        }
        for complete in ["{}", "[]", "{\"a\": 1}", "[1, 2.5, \"x\", true, null, {\"b\": []}]", " {} \n"] {
            XCTAssertTrue(json.isComplete(complete), "\(complete) is complete")
        }
        XCTAssertFalse(json.isComplete("{\"a\": 1"))
        XCTAssertFalse(json.isComplete("[1, 2"))

        // Whitespace is capped between tokens and free inside a string.
        XCTAssertTrue(json.accepts("{" + String(repeating: " ", count: 8) + "\"a\""))
        XCTAssertFalse(json.accepts("{" + String(repeating: " ", count: 9) + "\"a\""))
        XCTAssertFalse(json.accepts(String(repeating: "\n", count: 9)))
        XCTAssertTrue(json.accepts("{\"" + String(repeating: " ", count: 40)))
        XCTAssertTrue(json.accepts("{\"a\":\n    1,\n    \"b\": 2}"))

        let scalar = NFKMLXJSONConstraint(vocabulary: toyVocabulary(), root: .any)
        XCTAssertTrue(scalar.isComplete("\"text\""))
        XCTAssertTrue(scalar.isComplete("-12.5"))
        XCTAssertTrue(scalar.accepts("12."))
        XCTAssertFalse(scalar.isComplete("12."))

        let object = NFKMLXJSONConstraint(vocabulary: toyVocabulary(), root: .object)
        XCTAssertTrue(object.accepts("{\"a\": [1]}"))
        XCTAssertFalse(object.accepts("["))
        XCTAssertFalse(NFKMLXJSONConstraint(vocabulary: toyVocabulary(), root: .array).accepts("{"))
    }

    func testTheAdmissibleTokensFollowTheGrammar() {
        let vocabulary = toyVocabulary()
        let json = NFKMLXJSONConstraint(vocabulary: vocabulary)
        func allowed(after text: String) -> Set<String> {
            let state = json.advance(json.initialState(), bytes: Array(text.utf8))!
            return Set(json.allowedTokens(from: state).map { String(decoding: vocabulary.tokens[$0], as: UTF8.self) })
        }
        let opening = allowed(after: "")
        XCTAssertTrue(opening.isSuperset(of: ["{", "[", " ", "{\""]))
        XCTAssertTrue(opening.isDisjoint(with: ["}", "\"", "true", "123", ""]), "no end token before a document")

        let afterBrace = allowed(after: "{")
        XCTAssertTrue(afterBrace.isSuperset(of: ["\"", "}", "\"name\"", " "]))
        XCTAssertFalse(afterBrace.contains("123"), "an object wants a key")

        let afterColon = allowed(after: "{\"a\":")
        XCTAssertTrue(afterColon.isSuperset(of: ["123", "true", "null", "[", "{", "\"", "-7"]))
        XCTAssertFalse(afterColon.contains("}"))

        let closed = allowed(after: "{\"a\": 1}")
        XCTAssertTrue(closed.contains(""), "the end token is admitted once the document closes")
        XCTAssertTrue(closed.isSubset(of: ["", " ", "\n", "\t", "\r"]))
    }

    // MARK: Constrained generation

    func testConstrainedGenerationProducesWellFormedJSON() throws {
        try requireMLXRuntime()
        let vocabulary = toyVocabulary()
        let json = NFKMLXJSONConstraint(vocabulary: vocabulary)
        let net = NFKMLXLanguage.makeNet(.tiny)
        for seed in [UInt64(1), 2, 3] {
            var options = NFKMLXGenerationOptions()
            options.maxTokens = 80
            options.temperature = 1
            options.seed = seed
            options.constraint = json
            let produced = net.generate(prompt: [3, 17, 42], options: options)
            let output = text(produced, vocabulary)
            XCTAssertTrue(json.accepts(output), "every prefix stays inside the grammar: \(output)")
            if produced.count < options.maxTokens {
                // The run ended at the end token, which the grammar admits only for a complete
                // document — and a complete document parses.
                XCTAssertTrue(json.isComplete(output), output)
                XCTAssertNoThrow(try JSONSerialization.jsonObject(with: Data(output.utf8),
                                                                  options: .fragmentsAllowed), output)
            }
        }
    }

    func testAChoiceConstraintYieldsExactlyOneChoice() throws {
        try requireMLXRuntime()
        let vocabulary = toyVocabulary()
        let choices = ["yes", "no", "maybe"]
        let constraint = NFKMLXChoiceConstraint(choices: choices, vocabulary: vocabulary)
        let net = NFKMLXLanguage.makeNet(.tiny)
        for seed in [UInt64(1), 2, 3, 4] {
            var options = NFKMLXGenerationOptions()
            options.maxTokens = 20
            options.temperature = 1
            options.seed = seed
            options.constraint = constraint
            let output = text(net.generate(prompt: [7, 9], options: options), vocabulary)
            XCTAssertTrue(choices.contains(output), "\(output) is one of the choices")
        }
        // "may" is a prefix of "maybe": the run may end at either, never between.
        let prefixed = NFKMLXChoiceConstraint(choices: ["may", "maybe"], vocabulary: vocabulary)
        XCTAssertTrue(prefixed.isComplete("may"))
        XCTAssertTrue(prefixed.accepts("mayb"))
        XCTAssertFalse(prefixed.isComplete("mayb"))
        XCTAssertFalse(prefixed.accepts("mayx"))
    }

    func testTheRequestKeysSelectTheConstraint() {
        var options = NFKMLXGenerationOptions()
        NFKMLXLanguageBackend.applyMLXParameters(
            from: NFKInferenceRequest(inputs: [NFKInputPrompt: "hi"],
                                      parameters: [NFKMLXGenerationParameterKey.outputFormat: "json"]),
            to: &options)
        XCTAssertTrue(options.jsonOutput)
        XCTAssertEqual(options.jsonRoot, .container)
        XCTAssertNil(options.choices)
        NFKMLXLanguageBackend.applyMLXParameters(
            from: NFKInferenceRequest(inputs: [NFKInputPrompt: "hi"],
                                      parameters: [NFKMLXGenerationParameterKey.outputFormat: "json-object"]),
            to: &options)
        XCTAssertEqual(options.jsonRoot, .object)

        var picking = NFKMLXGenerationOptions()
        NFKMLXLanguageBackend.applyMLXParameters(
            from: NFKInferenceRequest(inputs: [NFKInputPrompt: "hi"],
                                      parameters: [NFKMLXGenerationParameterKey.choices: ["yes", "no"]]),
            to: &picking)
        XCTAssertEqual(picking.choices, ["yes", "no"])
        XCTAssertFalse(picking.jsonOutput)
    }

    func testTheVocabularyReadsEveryTokensBytesFromTheTokenizer() throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        // The GPT-2 byte-to-unicode map spells a space as "Ġ" and a newline as "Ċ".
        let vocab: [String: Int] = ["h": 0, "e": 1, "l": 2, "o": 3, "he": 4, "ll": 5, "hello": 6, "Ġ": 7, "Ċ": 8]
        try JSONSerialization.data(withJSONObject: vocab).write(to: directory.appendingPathComponent("vocab.json"))
        try "#version: 0.2\nh e\nl l\nhe ll\nhell o\n".write(to: directory.appendingPathComponent("merges.txt"),
                                                            atomically: true, encoding: .utf8)
        let tokenizer = try NFKTokenizer(forManifest: ["tokenizer": ["type": "bpe-bytelevel",
                                                                     "specialTokens": ["<eos>": 9]],
                                                       "eosTokenId": 9],
                                         directory: directory)
        let vocabulary = NFKMLXVocabulary(tokenizer: tokenizer, size: 12)
        XCTAssertEqual(vocabulary.tokens[6], Array("hello".utf8))
        XCTAssertEqual(vocabulary.tokens[7], [0x20])
        XCTAssertEqual(vocabulary.tokens[8], [0x0A])
        XCTAssertEqual(vocabulary.tokens[9], Array("<eos>".utf8))
        XCTAssertEqual(vocabulary.tokens[11], [], "an id past the tokenizer has no bytes")
        XCTAssertEqual(vocabulary.endToken, 9)
    }
}
