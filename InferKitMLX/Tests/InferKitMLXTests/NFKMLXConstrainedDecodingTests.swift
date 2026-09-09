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
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
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

    // MARK: The schema grammar

    private func schema(_ text: String) throws -> NFKMLXJSONSchemaConstraint {
        NFKMLXJSONSchemaConstraint(schema: try NFKMLXJSONSchema(jsonText: text), vocabulary: toyVocabulary())
    }

    func testTheSchemaGrammarEnforcesKeysTypesAndBounds() throws {
        let grammar = try schema("""
            {"type": "object",
             "properties": {"name": {"type": "string"}, "age": {"type": "integer"},
                            "tags": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 2},
                            "mood": {"enum": ["happy", "sad"]}, "score": {"type": ["number", "null"]}},
             "required": ["name", "age"], "additionalProperties": false}
            """)
        for prefix in ["{", "{\"name\"", "{\"age\": 4", "{\"tags\": [\"a\"", "{\"mood\": \"ha", "{\"score\": null",
                       "{\"score\": -1.5e2", "{ \"name\" : \"x\" , \"age\" : 3 }", "{\"age\": 3, \"name\": \"x\""] {
            XCTAssertTrue(grammar.accepts(prefix), "\(prefix) is a valid prefix")
        }
        for invalid in ["[", "{\"nam3\"", "{\"age\": 4.", "{\"age\": 4e", "{\"age\": \"4\"", "{\"tags\": []",
                        "{\"tags\": [\"a\", \"b\", \"c\"", "{\"tags\": [1", "{\"mood\": \"angry\"", "{\"mood\": \"happ\"}",
                        "{\"name\": \"x\", \"name\": \"y\"", "{\"age\": 3}", "{\"extra\": 1", "{\"score\": true",
                        "{\"name\": \"x\", \"age\": 3, \"tags\": [\"a\"], \"mood\": \"sad\", \"score\": 1,"] {
            XCTAssertFalse(grammar.accepts(invalid), "\(invalid) is rejected")
        }
        for complete in ["{\"name\": \"x\", \"age\": 3}",
                         "{\"age\": 3, \"name\": \"x\", \"tags\": [\"a\", \"b\"], \"mood\": \"sad\", \"score\": null}",
                         "{\"mood\": \"happy\", \"name\": \"\", \"age\": -0}"] {
            XCTAssertTrue(grammar.isComplete(complete), "\(complete) is complete")
        }
        XCTAssertFalse(grammar.isComplete("{\"name\": \"x\""), "a required key is missing")
        XCTAssertFalse(grammar.isComplete("{\"name\": \"x\", \"age\": 3"))
    }

    func testTheSchemaGrammarAdmitsUnlistedKeysAnyValuesAndAlternatives() throws {
        let unlisted = try schema("""
            {"type": "object", "properties": {"id": {"type": "integer"}}, "additionalProperties": {"type": "boolean"}}
            """)
        XCTAssertTrue(unlisted.accepts("{\"note\": true"))
        XCTAssertTrue(unlisted.accepts("{\"idx\": false"), "a key that outgrows a property becomes an unlisted one")
        XCTAssertFalse(unlisted.accepts("{\"note\": 1"), "an unlisted key takes the additional schema")
        XCTAssertTrue(unlisted.isComplete("{\"id\": 1, \"idx\": false}"))
        XCTAssertFalse(unlisted.accepts("{\"id\": true"), "a declared key keeps its own type")

        let any = try schema("{\"properties\": {\"meta\": {}}, \"required\": [\"meta\"]}")
        XCTAssertTrue(any.isComplete("{\"meta\": {\"a\": [1, 2, {\"b\": null}]}}"))
        XCTAssertTrue(any.isComplete("{\"meta\": 5}"))
        XCTAssertTrue(any.isComplete("{\"meta\": \"s\", \"other\": [true]}"), "unlisted keys default to allowed")
        XCTAssertFalse(any.accepts("{\"meta\": [1,]"))
        XCTAssertFalse(any.isComplete("{\"meta\": 5"))

        let tagged = try schema("""
            {"anyOf": [
              {"type": "object", "properties": {"kind": {"const": "a"}, "x": {"type": "integer"}},
               "required": ["kind", "x"], "additionalProperties": false},
              {"type": "object", "properties": {"kind": {"const": "b"}, "y": {"type": "string"}},
               "required": ["kind", "y"], "additionalProperties": false}]}
            """)
        XCTAssertTrue(tagged.accepts("{\"kind\": \""), "both alternatives stay alive until the bytes decide")
        XCTAssertTrue(tagged.isComplete("{\"kind\": \"a\", \"x\": 1}"))
        XCTAssertTrue(tagged.isComplete("{\"y\": \"s\", \"kind\": \"b\"}"))
        XCTAssertFalse(tagged.accepts("{\"kind\": \"a\", \"y\""))
        XCTAssertFalse(tagged.accepts("{\"kind\": \"c"))
        XCTAssertFalse(tagged.isComplete("{\"kind\": \"a\"}"))

        let recursive = try schema("""
            {"$defs": {"node": {"type": "object",
                                "properties": {"value": {"type": "integer"},
                                               "children": {"type": "array", "items": {"$ref": "#/$defs/node"}}},
                                "required": ["value"], "additionalProperties": false}},
             "$ref": "#/$defs/node"}
            """)
        XCTAssertTrue(recursive.isComplete("{\"value\": 1, \"children\": [{\"value\": 2, \"children\": []}, {\"value\": 3}]}"))
        XCTAssertFalse(recursive.accepts("{\"value\": 1, \"children\": [3"))

        let integer = try schema("{\"type\": \"integer\"}")
        XCTAssertTrue(integer.isComplete("42"))
        XCTAssertTrue(integer.accepts("-"))
        XCTAssertFalse(integer.accepts("4."))
        XCTAssertFalse(integer.accepts("\"4\""))
        let choice = try schema("{\"enum\": [\"yes\", \"no\", 3, null]}")
        XCTAssertTrue(choice.isComplete("\"yes\""))
        XCTAssertTrue(choice.isComplete("3"))
        XCTAssertTrue(choice.isComplete("null"))
        XCTAssertFalse(choice.accepts("\"maybe"))
        XCTAssertFalse(choice.isComplete("\"ye"))
    }

    func testTheSchemaCompilerRefusesWhatItCannotEnforce() {
        XCTAssertThrowsError(try NFKMLXJSONSchema(jsonText: "{\"allOf\": [{}]}"))
        XCTAssertThrowsError(try NFKMLXJSONSchema(jsonText: "{\"not\": {\"type\": \"string\"}}"))
        XCTAssertThrowsError(try NFKMLXJSONSchema(jsonText: "{\"type\": \"object\", \"required\": [\"a\"]}"),
                             "a required key must be declared")
        XCTAssertThrowsError(try NFKMLXJSONSchema(jsonText: "{\"$ref\": \"#/$defs/missing\"}"))
        XCTAssertThrowsError(try NFKMLXJSONSchema(jsonText: "{\"type\": \"date\"}"))
        XCTAssertThrowsError(try NFKMLXJSONSchema(jsonText: "false"), "a schema that admits nothing")
        XCTAssertThrowsError(try NFKMLXJSONSchema(jsonText: "{\"type\": \"array\", \"minItems\": 3, \"maxItems\": 2}"))
        // Content-narrowing keywords the grammar cannot check are ignored, so the schema still compiles.
        XCTAssertNoThrow(try NFKMLXJSONSchema(jsonText: "{\"type\": \"string\", \"pattern\": \"^a\", \"minLength\": 2}"))
        XCTAssertNoThrow(try NFKMLXJSONSchema(jsonText: "{\"type\": \"integer\", \"minimum\": 0, \"maximum\": 10}"))
        XCTAssertNoThrow(try NFKMLXJSONSchema(jsonText: "true"))
    }

    func testTheAdmissibleTokensFollowTheSchema() throws {
        let vocabulary = toyVocabulary()
        let grammar = NFKMLXJSONSchemaConstraint(schema: try NFKMLXJSONSchema(jsonText: """
            {"type": "object", "properties": {"name": {"type": "string"}, "id": {"type": "integer"}},
             "required": ["name", "id"], "additionalProperties": false}
            """), vocabulary: vocabulary)
        func allowed(after text: String) -> Set<String> {
            let state = grammar.advance(grammar.initialState(), bytes: Array(text.utf8))!
            return Set(grammar.allowedTokens(from: state).map { String(decoding: vocabulary.tokens[$0], as: UTF8.self) })
        }
        let opened = allowed(after: "{")
        XCTAssertTrue(opened.isSuperset(of: ["\"", "\"name\"", "\"id\"", " "]))
        XCTAssertTrue(opened.isDisjoint(with: ["}", "\"a\"", "\"b\"", "123", "[", "\"}"]), "no close before the required keys, no undeclared key")

        let idValue = allowed(after: "{\"id\":")
        XCTAssertTrue(idValue.isSuperset(of: ["123", "42", "-7", " "]))
        XCTAssertTrue(idValue.isDisjoint(with: ["0.5", "1e3", "\"", "true", "null", "[", "{"]), "an integer slot takes integers")

        let afterId = allowed(after: "{\"id\": 1")
        XCTAssertTrue(afterId.isSuperset(of: [",", ", \""]))
        XCTAssertFalse(afterId.contains("}"), "name is still required")

        let both = allowed(after: "{\"id\": 1, \"name\": \"x\"")
        XCTAssertTrue(both.isSuperset(of: ["}"]))
        XCTAssertTrue(both.isDisjoint(with: [",", ", \""]), "every key is written and unlisted ones are forbidden")

        let closed = allowed(after: "{\"id\": 1, \"name\": \"x\"}")
        XCTAssertTrue(closed.contains(""), "the end token is admitted once the document closes")
        XCTAssertTrue(closed.isSubset(of: ["", " ", "\n", "\t", "\r"]))
    }

    func testSchemaConstrainedGenerationConformsToTheSchema() throws {
        try requireMLXRuntime()
        let vocabulary = toyVocabulary()
        let grammar = NFKMLXJSONSchemaConstraint(schema: try NFKMLXJSONSchema(jsonText: """
            {"type": "object",
             "properties": {"name": {"type": "string"}, "id": {"type": "integer"}, "ok": {"type": "boolean"}},
             "required": ["name", "id"], "additionalProperties": false}
            """), vocabulary: vocabulary)
        let net = NFKMLXLanguage.makeNet(.tiny)
        for seed in [UInt64(1), 2, 3] {
            var options = NFKMLXGenerationOptions()
            options.maxTokens = 80
            options.temperature = 1
            options.seed = seed
            options.constraint = grammar
            let produced = net.generate(prompt: [3, 17, 42], options: options)
            let output = text(produced, vocabulary)
            XCTAssertTrue(grammar.accepts(output), "every prefix stays inside the schema: \(output)")
            if produced.count < options.maxTokens {
                XCTAssertTrue(grammar.isComplete(output), output)
                let object = try XCTUnwrap(try JSONSerialization.jsonObject(with: Data(output.utf8)) as? [String: Any], output)
                XCTAssertTrue(Set(object.keys).isSubset(of: ["name", "id", "ok"]), output)
                XCTAssertTrue(object["name"] is String, output)
                let id = try XCTUnwrap(object["id"] as? NSNumber, output)
                XCTAssertEqual(id.doubleValue, id.doubleValue.rounded(), "id is an integer: \(output)")
            }
        }
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
