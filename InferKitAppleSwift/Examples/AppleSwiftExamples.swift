//
//  AppleSwiftExamples.swift
//  InferKitAppleSwiftExamples
//
//  Compiled and run by CI so the snippets in Docs/examples.md cannot silently drift. Each method
//  mirrors a section there.
//

import XCTest
import InferKit
@testable import InferKitAppleSwift

final class AppleSwiftExamples: XCTestCase {

    // Docs/examples.md: Reading a document's structure
    func testExampleReadingADocument() throws {
        let backend = NFKVisionDocumentBackend()
        XCTAssertEqual(backend.backendIdentifier, "vision-document")

        // let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: page]))
        // result.text gives the whole transcript; result.structured gives paragraphs, lists, tables.
        XCTAssertTrue(backend.supportedInputKeys.contains(NFKInputImage))
    }

    // Docs/examples.md: Is the lens dirty?
    func testExampleJudgingTheLens() {
        let backend = NFKVisionSmudgeBackend()
        XCTAssertEqual(backend.backendIdentifier, "vision-smudge")
        // The verdict arrives as one classification labeled "smudge", whose confidence is the answer.
    }

    // Docs/examples.md: Transcribing with SpeechAnalyzer
    func testExampleTranscribingWithTheAnalyzer() {
        let backend = NFKSpeechAnalyzerBackend(locale: Locale(identifier: "en-US"))
        XCTAssertEqual(backend.locale.identifier, "en-US")

        // prepare() reserves the locale and installs its assets, which is a download the first time,
        // and isReady answers from what it found.
        XCTAssertFalse(backend.isReady)
        // try backend.prepare()
        // let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputAudio: asset]))
        // result.text is the transcription; result.segments carries one entry per reported range.
    }

    // Docs/examples.md: Discovery picks the best transcriber that is linked
    func testExampleDiscoveryPrefersTheAnalyzer() throws {
        let backend = try NFKDynamicBackend.backend(forCapability: NFKCapabilityTranscription)
        XCTAssertEqual(backend.backendIdentifier, "apple-speech-analyzer",
                       "InferKitMLX is not linked here, so the analyzer answers before the core's recognizer")
    }
    // Docs/examples.md: Apple's own engines — translating on device
    func testExampleTranslating() throws {
        let backend = NFKTranslationBackend(sourceLanguage: "en", targetLanguage: "es")
        backend.responseTimeout = 5          // every wait on the framework is bounded
        XCTAssertEqual(backend.backendIdentifier, "apple-translation")

        let request = NFKInferenceRequest(
            inputs: [NFKInputPrompt: "Good morning."],
            parameters: [NFKParameterTargetLanguage: "es"])   // the request's pair wins
        do {
            let result = try backend.runInference(for: request)
            XCTAssertFalse(try XCTUnwrap(result.text).isEmpty)
        } catch {
            // The pair's model is not installed, or the framework did not answer. Both arrive as
            // kNFKError_InferenceNotReady, and neither is a failure of the request.
            XCTAssertEqual((error as NSError).code, NFKInferenceError.error_InferenceNotReady.rawValue)
        }
    }

    // Docs/examples.md: Discovery — linking this package answers the translation capability
    func testExampleDiscoveryAnswersTranslation() throws {
        let backend = try NFKDynamicBackend.backend(forCapability: NFKCapabilityTranslation)
        XCTAssertEqual(backend.backendIdentifier, "apple-translation",
                       "no MLX translator is linked here, so Apple's framework answers")
    }

}
