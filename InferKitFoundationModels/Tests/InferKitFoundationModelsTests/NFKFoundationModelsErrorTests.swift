//
//  NFKFoundationModelsErrorTests.swift
//  InferKitFoundationModelsTests
//
//  GenerationError's cases carry a context a caller can build, so the macOS 26 mapping runs here in
//  full. The macOS 27 errors have no public initializer for their payloads and their types need that
//  OS, so those tests skip and the mapping is compile-verified.
//

import XCTest
import FoundationModels
import InferKit
@testable import InferKitFoundationModels

final class NFKFoundationModelsErrorTests: XCTestCase {

    private func context(_ description: String = "measured") -> LanguageModelSession.GenerationError.Context {
        LanguageModelSession.GenerationError.Context(debugDescription: description)
    }

    private func code(of error: Error) -> Int {
        NFKFoundationModelsFailure.coreError(for: error).code
    }

    // MARK: What an app has to decide

    func testARefusalIsItsOwnCodeSoAnAppDoesNotRetryIt() {
        XCTAssertEqual(code(of: LanguageModelSession.GenerationError.guardrailViolation(context())),
                       NFKInferenceError.error_InferenceRefused.rawValue)
        XCTAssertEqual(code(of: LanguageModelSession.GenerationError.refusal(.init(transcriptEntries: []), context())),
                       NFKInferenceError.error_InferenceRefused.rawValue)
    }

    func testARateLimitIsItsOwnCodeSoAnAppBacksOff() {
        XCTAssertEqual(code(of: LanguageModelSession.GenerationError.rateLimited(context())),
                       NFKInferenceError.error_InferenceRateLimited.rawValue)
    }

    func testARequestTheModelCannotTakeIsUnsupported() {
        XCTAssertEqual(code(of: LanguageModelSession.GenerationError.exceededContextWindowSize(context())),
                       NFKInferenceError.error_InferenceUnsupported.rawValue)
        XCTAssertEqual(code(of: LanguageModelSession.GenerationError.unsupportedGuide(context())),
                       NFKInferenceError.error_InferenceUnsupported.rawValue)
        XCTAssertEqual(code(of: LanguageModelSession.GenerationError.unsupportedLanguageOrLocale(context())),
                       NFKInferenceError.error_InferenceUnsupported.rawValue)
    }

    func testMissingAssetsAreNotReadyRatherThanAFailure() {
        XCTAssertEqual(code(of: LanguageModelSession.GenerationError.assetsUnavailable(context())),
                       NFKInferenceError.error_InferenceNotReady.rawValue)
    }

    func testTheModelsOwnTroubleIsABackendFailure() {
        XCTAssertEqual(code(of: LanguageModelSession.GenerationError.decodingFailure(context())),
                       NFKInferenceError.error_InferenceBackendFailure.rawValue)
        XCTAssertEqual(code(of: LanguageModelSession.GenerationError.concurrentRequests(context())),
                       NFKInferenceError.error_InferenceBackendFailure.rawValue)
    }

    // MARK: What the error carries

    func testAMappedErrorKeepsTheOriginalAndItsReason() throws {
        let generation = LanguageModelSession.GenerationError.guardrailViolation(context("the guardrail fired"))
        let error = NFKFoundationModelsFailure.coreError(for: generation)
        XCTAssertEqual(error.domain, NFKInferenceErrorDomain)
        XCTAssertFalse(error.localizedDescription.isEmpty)
        let underlying = try XCTUnwrap(error.userInfo[NSUnderlyingErrorKey] as? NSError)
        XCTAssertTrue(underlying.localizedDescription.contains("guardrail")
                      || String(describing: generation).contains("guardrail"))
    }

    func testAnErrorTheBackendRaisedPassesThroughUnchanged() {
        let raised = NSError(domain: NFKInferenceErrorDomain,
                             code: NFKInferenceError.error_InferenceMissingInput.rawValue,
                             userInfo: [NSLocalizedDescriptionKey: "no prompt"])
        let mapped = NFKFoundationModelsFailure.coreError(for: raised)
        XCTAssertEqual(mapped, raised, "the backend's own errors are already in the contract's domain")
        XCTAssertNil(mapped.userInfo[NSUnderlyingErrorKey])
    }

    func testAnUnrelatedErrorBecomesABackendFailure() {
        struct Trouble: Error {}
        let error = NFKFoundationModelsFailure.coreError(for: Trouble())
        XCTAssertEqual(error.code, NFKInferenceError.error_InferenceBackendFailure.rawValue)
        XCTAssertNotNil(error.userInfo[NSUnderlyingErrorKey])
    }

    // MARK: macOS 27

    func testTheContextCountsRideOnTheError() throws {
        #if compiler(>=6.4)
        guard #available(macOS 27, iOS 27, *) else {
            throw XCTSkip("LanguageModelError needs macOS 27")
        }
        // The 27 error carries the two counts the core keys were made for; a 26 error carries neither,
        // which is why the preflight in the backend exists.
        let exceeded = LanguageModelError.ContextSizeExceeded(contextSize: 4096, tokenCount: 5000,
                                                              debugDescription: "too long")
        let error = NFKFoundationModelsFailure.coreError(for: LanguageModelError.contextSizeExceeded(exceeded))
        XCTAssertEqual(error.code, NFKInferenceError.error_InferenceUnsupported.rawValue)
        XCTAssertEqual(error.userInfo[NFKFoundationModelsErrorKey.tokenCount] as? Int, 5000)
        XCTAssertEqual(error.userInfo[NFKFoundationModelsErrorKey.contextSize] as? Int, 4096)
        #else
        throw XCTSkip("built with an SDK before macOS 27")
        #endif
    }

    func testARateLimitCarriesItsResetDate() throws {
        #if compiler(>=6.4)
        guard #available(macOS 27, iOS 27, *) else {
            throw XCTSkip("LanguageModelError needs macOS 27")
        }
        let reset = Date(timeIntervalSince1970: 1_800_000_000)
        let limited = LanguageModelError.RateLimited(resetDate: reset, debugDescription: "slow down")
        let error = NFKFoundationModelsFailure.coreError(for: LanguageModelError.rateLimited(limited))
        XCTAssertEqual(error.code, NFKInferenceError.error_InferenceRateLimited.rawValue)
        XCTAssertEqual(error.userInfo[NFKFoundationModelsErrorKey.resetDate] as? Date, reset)
        #else
        throw XCTSkip("built with an SDK before macOS 27")
        #endif
    }
}
