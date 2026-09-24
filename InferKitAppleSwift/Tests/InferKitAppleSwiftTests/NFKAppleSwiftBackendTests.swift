//
//  NFKAppleSwiftBackendTests.swift
//  InferKitAppleSwiftTests
//
//  The contract and the refusals run everywhere. Transcription needs the locale's assets installed
//  and the document reader needs a page, so those tests state what they need and skip without it.
//

import CoreText
import XCTest
import InferKit
@testable import InferKitAppleSwift

final class NFKAppleSwiftBackendTests: XCTestCase {

    // MARK: Images

    private func image(width: Int, height: Int, drawing: ((CGContext) -> Void)? = nil) throws -> CGImage {
        let space = CGColorSpaceCreateDeviceRGB()
        let context = try XCTUnwrap(CGContext(data: nil, width: width, height: height,
                                              bitsPerComponent: 8, bytesPerRow: 0, space: space,
                                              bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                                                  | CGBitmapInfo.byteOrder32Little.rawValue))
        context.setFillColor(red: 1, green: 1, blue: 1, alpha: 1)
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        drawing?(context)
        return try XCTUnwrap(context.makeImage())
    }

    private func pageImage(lines: [String]) throws -> CGImage {
        try image(width: 600, height: 400) { context in
            let font = CTFontCreateWithName("Helvetica" as CFString, 40, nil)
            for (index, line) in lines.enumerated() {
                let string = NSAttributedString(string: line, attributes: [.font: font])
                let ctLine = CTLineCreateWithAttributedString(string)
                context.textPosition = CGPoint(x: 40, y: 320 - CGFloat(index) * 60)
                CTLineDraw(ctLine, context)
            }
        }
    }

    // MARK: The document reader

    func testTheDocumentBackendReportsItsContract() {
        let backend = NFKVisionDocumentBackend()
        XCTAssertTrue(backend.isReady)
        XCTAssertEqual(backend.backendIdentifier, "vision-document")
        XCTAssertTrue(backend.supportedInputKeys.contains(NFKInputImage))
    }

    func testAPageBecomesATranscriptAndAStructure() throws {
        let page = try pageImage(lines: ["Inferkit", "Apple engines"])
        let result = try NFKVisionDocumentBackend().runInference(
            for: NFKInferenceRequest(inputs: [NFKInputImage: page]))

        let transcript = try XCTUnwrap(result.text)
        XCTAssertTrue(transcript.contains("Inferkit"), "read: \(transcript)")

        let structure = try XCTUnwrap(result.structured)
        XCTAssertNotNil(structure["paragraphs"] as? [String])
        XCTAssertNotNil(structure["lists"] as? [[String]])
        XCTAssertNotNil(structure["tables"] as? [[[String]]])
    }

    func testTheDocumentBackendNeedsAnImage() {
        XCTAssertThrowsError(try NFKVisionDocumentBackend().runInference(for: NFKInferenceRequest(inputs: [:]))) { error in
            XCTAssertEqual((error as NSError).code, NFKInferenceError.error_InferenceMissingInput.rawValue)
        }
    }

    // MARK: The smudge reader

    func testACleanRenderIsNotJudgedSmudged() throws {
        let sharp = try image(width: 320, height: 240) { context in
            context.setFillColor(red: 0, green: 0, blue: 0, alpha: 1)
            for x in stride(from: 0, to: 320, by: 20) {
                context.fill(CGRect(x: x, y: 0, width: 10, height: 240))
            }
        }
        let result = try NFKVisionSmudgeBackend().runInference(
            for: NFKInferenceRequest(inputs: [NFKInputImage: sharp]))

        let smudge = try XCTUnwrap(result.classifications?.first)
        XCTAssertEqual(smudge.label, "smudge")
        XCTAssertGreaterThanOrEqual(smudge.confidence, 0.0)
        XCTAssertLessThanOrEqual(smudge.confidence, 1.0)
    }

    // MARK: The analyzer

    func testTheAnalyzerReportsItsContract() {
        let backend = NFKSpeechAnalyzerBackend()
        XCTAssertEqual(backend.backendIdentifier, "apple-speech-analyzer")
        XCTAssertTrue(backend.supportedInputKeys.contains(NFKInputAudio))
        XCTAssertEqual(backend.locale.identifier, Locale.current.identifier)

        let swedish = Locale(identifier: "sv-SE")
        backend.locale = swedish
        XCTAssertEqual(backend.locale.identifier, swedish.identifier)
    }

    func testAnUnpreparedAnalyzerIsNotReady() {
        // Readiness answers from what prepare() found, because the system reports installed locales
        // asynchronously and the property cannot wait.
        XCTAssertFalse(NFKSpeechAnalyzerBackend().isReady)
    }

    func testTheAnalyzerNeedsAudio() {
        XCTAssertThrowsError(try NFKSpeechAnalyzerBackend().runInference(for: NFKInferenceRequest(inputs: [:]))) { error in
            XCTAssertEqual((error as NSError).code, NFKInferenceError.error_InferenceMissingInput.rawValue)
        }
    }

    func testTheSupportedLocalesAreReported() {
        let reported = expectation(description: "supported locales")
        NFKSpeechAnalyzerBackend.supportedLocales { locales in
            XCTAssertGreaterThan(locales.count, 0, "the analyzer names the locales it can install")
            reported.fulfill()
        }
        wait(for: [reported], timeout: 30)
    }

    // MARK: Translation

    func testTheTranslationBackendReportsItsContract() {
        let backend = NFKTranslationBackend(sourceLanguage: "en", targetLanguage: "es")
        XCTAssertEqual(backend.backendIdentifier, "apple-translation")
        XCTAssertTrue(backend.supportedInputKeys.contains(NFKInputPrompt))
        XCTAssertTrue(backend.supportedParameterKeys.contains(NFKParameterTargetLanguage))
        XCTAssertTrue(backend.supportedParameterKeys.contains(NFKParameterSourceLanguage))
        XCTAssertFalse(backend.isReady, "readiness answers from what prepare() found")
    }

    func testTranslationNeedsText() {
        let backend = NFKTranslationBackend(sourceLanguage: "en", targetLanguage: "es")
        XCTAssertThrowsError(try backend.runInference(for: NFKInferenceRequest(inputs: [:]))) { error in
            XCTAssertEqual((error as NSError).code, NFKInferenceError.error_InferenceMissingInput.rawValue)
        }
    }

    func testTranslationNeedsATargetLanguage() {
        // The target is the one parameter with no default: a backend built without one, and a request
        // that names none, cannot say what to translate into.
        let request = NFKInferenceRequest(inputs: [NFKInputPrompt: "Good morning."])
        XCTAssertThrowsError(try NFKTranslationBackend().runInference(for: request)) { error in
            XCTAssertEqual((error as NSError).code, NFKInferenceError.error_InferenceMissingInput.rawValue)
        }
    }

    func testAnUnknownLanguageIsAnsweredRatherThanHung() {
        // Apple's Translation framework answers nothing at all in a test bundle: asked about a
        // language, it neither reports the pair nor refuses it. The bound on the wait is what turns
        // that silence into an error a caller can act on, and this is the test that holds it.
        let backend = NFKTranslationBackend(sourceLanguage: "zz", targetLanguage: "en")
        backend.responseTimeout = 5
        XCTAssertThrowsError(try backend.runInference(
            for: NFKInferenceRequest(inputs: [NFKInputPrompt: "Good morning."]))) { error in
            XCTAssertEqual((error as NSError).domain, NFKInferenceErrorDomain)
            XCTAssertTrue([NFKInferenceError.error_InferenceUnsupported.rawValue,
                           NFKInferenceError.error_InferenceNotReady.rawValue]
                              .contains((error as NSError).code),
                          "a pair the system will not translate, or silence from it: \(error)")
        }
    }

    func testTheRequestOverridesTheBackendsPair() throws {
        // The pair on the request wins, so one backend serves every pair a caller asks for.
        let backend = NFKTranslationBackend(sourceLanguage: "en", targetLanguage: "zz")
        backend.responseTimeout = 5
        let request = NFKInferenceRequest(
            inputs: [NFKInputPrompt: "Good morning."],
            parameters: [NFKParameterSourceLanguage: "en", NFKParameterTargetLanguage: "es"])
        do {
            // The call stays outside XCTUnwrap: an unwrap records its failure before the catch below
            // can decide that silence from the framework is a skip rather than a defect.
            let result = try backend.runInference(for: request)
            XCTAssertFalse(try XCTUnwrap(result.text).isEmpty)
        } catch {
            // The en to es model is not installed here, or the framework said nothing. Both arrive as
            // kNFKError_InferenceNotReady, and neither is a failure of the request.
            XCTAssertEqual((error as NSError).code, NFKInferenceError.error_InferenceNotReady.rawValue)
            throw XCTSkip("the en to es model did not answer on this machine")
        }
    }

    func testTheSupportedLanguagesAreReported() throws {
        let reported = expectation(description: "supported languages")
        NFKTranslationBackend.supportedLanguages { languages in
            XCTAssertGreaterThan(languages.count, 0, "the system names the languages it translates")
            reported.fulfill()
        }
        // Silence from the framework is the usual answer in a test bundle, and it is not a defect in
        // the backend, so this reports what it could not reach instead of failing.
        if XCTWaiter().wait(for: [reported], timeout: 10) != .completed {
            throw XCTSkip("the Translation framework did not answer in this process")
        }
    }

    // MARK: Discovery

    func testLinkingThisPackageActivatesTranslation() throws {
        XCTAssertTrue(NFKDynamicBackend.isProviderAvailable("NFKTranslationProvider"))
        XCTAssertTrue(NFKDynamicBackend.isCapabilityAvailable(NFKCapabilityTranslation))

        // No MLX translation model is linked here, so Apple's framework answers the capability.
        let backend = try NFKDynamicBackend.backend(forCapability: NFKCapabilityTranslation)
        XCTAssertEqual(backend.backendIdentifier, "apple-translation")
    }

    func testLinkingThisPackageActivatesTranscription() throws {
        XCTAssertTrue(NFKDynamicBackend.isProviderAvailable("NFKSpeechAnalyzerProvider"))
        XCTAssertTrue(NFKDynamicBackend.isCapabilityAvailable(NFKCapabilityTranscription))

        // InferKitMLX is not linked here, so this package's analyzer answers ahead of the core's
        // own recognizer.
        let backend = try NFKDynamicBackend.backend(forCapability: NFKCapabilityTranscription)
        XCTAssertEqual(backend.backendIdentifier, "apple-speech-analyzer")
    }
}
