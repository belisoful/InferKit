//
//  NFKMLXDeclaredKeysTests.swift
//  InferKitMLXTests
//
//  What each backend declares through the protocol's `supportedParameterKeys` and
//  `supportedInputKeys`. The closure backends carry the keys their closures were built to read, so
//  these cover the union as well as the fixed sets. Building a backend over a net reaches MLX and
//  skips without a Metal library.
//

import XCTest
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXDeclaredKeysTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    // MARK: Fixed sets

    func testTheLanguageBackendDeclaresTheCoreAndPackageKeys() throws {
        try requireMLXRuntime()
        let backend = NFKMLXLanguageBackend(net: NFKMLXLanguage.makeNet(.tiny), tokenizer: nil,
                                            identifier: "tiny")
        XCTAssertEqual(backend.supportedInputKeys, [NFKInputPrompt, NFKInputMessages])
        for key in [NFKParameterTemperature, NFKParameterTopP, NFKParameterMaxTokens, NFKParameterSeed,
                    NFKParameterJSONSchema, NFKParameterOutputFormat, NFKParameterChoices,
                    NFKMLXGenerationParameterKey.draftTokens, NFKMLXGenerationParameterKey.chatTemplate] {
            XCTAssertTrue(backend.supportedParameterKeys.contains(key), key)
        }
        XCTAssertFalse(backend.supportedParameterKeys.contains(NFKParameterTools),
                       "the backend runs no tools")
        XCTAssertFalse(backend.supportedParameterKeys.contains(NFKParameterTopK),
                       "sampling takes top-p, not top-k")
    }

    func testTheStableDiffusionBackendAnswersForTheModelItBuilds() {
        let backend = NFKMLXBackend(model: .sdxlTurbo)
        XCTAssertTrue(backend.supportedInputKeys.isSuperset(of: [NFKInputPrompt, NFKInputNegativePrompt,
                                                                NFKInputImage]))
        XCTAssertTrue(backend.supportedParameterKeys.isSuperset(of: [NFKParameterSteps, NFKParameterSeed,
                                                                     NFKParameterWidth, NFKParameterHeight]))
    }

    // MARK: Closure backends

    func testTheModuleBackendAddsWhatItsForwardReads() {
        let plain = NFKMLXModuleBackend(identifier: "plain") { image in image }
        XCTAssertEqual(plain.supportedInputKeys, [NFKInputImage])
        XCTAssertEqual(plain.supportedParameterKeys, [])

        let conditioned = NFKMLXModuleBackend(identifier: "conditioned",
                                              forwardInputKeys: [NFKInputControl],
                                              forwardParameterKeys: [NFKParameterStrength]) { image, _ in image }
        XCTAssertEqual(conditioned.supportedInputKeys, [NFKInputImage, NFKInputControl])
        XCTAssertEqual(conditioned.supportedParameterKeys, [NFKParameterStrength])
    }

    func testTheMattingBackendAddsWhatItsForwardReads() {
        let plain = NFKMLXMattingBackend(identifier: "plain") { plate, _ in plate }
        XCTAssertEqual(plain.supportedInputKeys, [NFKInputImage, NFKInputMask])

        let pointed = NFKMLXMattingBackend(identifier: "pointed",
                                           forwardParameterKeys: [NFKSAMPointKey]) { plate, _, _ in plate }
        XCTAssertEqual(pointed.supportedParameterKeys, [NFKSAMPointKey])
    }

    func testTheTensorBackendDeclaresItsPorts() {
        let configuration = NFKMLXTensorConfiguration(
            inputs: [NFKMLXTensorPort(key: NFKInputImage, tensorName: "plate"),
                     NFKMLXTensorPort(key: NFKInputControl, tensorName: "guide")],
            outputs: [NFKMLXTensorPort(key: NFKOutputImage, tensorName: "out")])
        let backend = NFKMLXTensorBackend(identifier: "ports", configuration: configuration) { $0 }
        XCTAssertEqual(backend.supportedInputKeys, [NFKInputImage, NFKInputControl])
        XCTAssertEqual(backend.supportedParameterKeys, [])
    }

    func testTheDiffusionBackendDeclaresTheSamplingKeysItResolves() {
        let backend = NFKMLXDiffusionBackend(
            identifier: "sampler",
            encodedInputKeys: [NFKInputPrompt],
            encode: { _, _, _ in NFKDiffusionContext(width: 8, height: 8) },
            denoise: { latent, _, _, _ in latent })
        XCTAssertEqual(backend.supportedInputKeys, [NFKInputImage, NFKInputMask, NFKInputPrompt])
        XCTAssertEqual(backend.supportedParameterKeys, [NFKParameterSteps, NFKParameterGuidanceScale,
                                                        NFKParameterStrength, NFKParameterSeed])
    }

    // MARK: Shipped models

    func testStyleTransferDeclaresTheStyleImageAndTheBlend() throws {
        try requireMLXRuntime()
        let backend = try NFKMLXAdaIN.backend(encoderURL: nil, decoderURL: nil)
        XCTAssertEqual(backend.supportedInputKeys, [NFKInputImage, NFKInputControl])
        XCTAssertEqual(backend.supportedParameterKeys, [NFKParameterStrength])
    }

    func testSegmentAnythingDeclaresItsClickPoint() throws {
        try requireMLXRuntime()
        let backend = try NFKMLXSAM.backend(weightsURL: nil)
        XCTAssertEqual(backend.supportedParameterKeys, [NFKSAMPointKey])
        XCTAssertEqual(backend.supportedInputKeys?.contains(NFKInputImage), true)
    }

    // A backend that reads one input and no parameters says so, which is a different answer from
    // declaring nothing at all.
    func testAnAudioBackendDeclaresItsAudioInputAndNoParameters() throws {
        try requireMLXRuntime()
        let backend = try NFKMLXSileroVAD.backend(weightsURL: nil)
        XCTAssertEqual(backend.supportedInputKeys, [NFKInputAudio])
        XCTAssertEqual(backend.supportedParameterKeys, [])
    }
}
