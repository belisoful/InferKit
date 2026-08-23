//
//  NFKMLXBackendTests.swift
//  InferKitMLXTests
//
//  The backend's contract and the release each model resolves to, which need no GPU or weights.
//  Generation itself is measured against the reference pipelines in `NFKMLXReferenceParityTests`.
//

import XCTest
import InferKit
@testable import InferKitMLX

final class NFKMLXBackendTests: XCTestCase {

    func testTheBackendReportsItsIdentityAndIsNotReadyBeforeLoading() {
        let backend = NFKMLXBackend(model: .sdxlTurbo)
        XCTAssertEqual(backend.backendIdentifier, "mlx-stable-diffusion")
        XCTAssertFalse(backend.isReady)
    }

    func testEachModelResolvesToItsRelease() {
        let releases: [(NFKMLXStableDiffusionModel, String)] = [
            (.stableDiffusion15, "stable-diffusion-v1-5/stable-diffusion-v1-5"),
            (.stableDiffusion21Base, "stabilityai/stable-diffusion-2-1-base"),
            (.sdxlTurbo, "stabilityai/sdxl-turbo"),
        ]
        for (model, repository) in releases {
            XCTAssertEqual(NFKMLXStableDiffusionRelease(model).repository, repository)
        }
    }

    // The three releases differ in the conditioning they take, which is what the geometry has to
    // match: one text tower or two, and how wide the UNet's cross-attention is.
    func testEachReleaseCarriesItsOwnGeometry() {
        let sd15 = NFKMLXStableDiffusionRelease(.stableDiffusion15).configuration
        XCTAssertEqual(sd15.unet.crossAttentionDimensions, 768)
        XCTAssertNil(sd15.secondaryTextEncoder)

        let sd21 = NFKMLXStableDiffusionRelease(.stableDiffusion21Base).configuration
        XCTAssertEqual(sd21.unet.crossAttentionDimensions, 1024)
        XCTAssertEqual(sd21.textEncoder.activation, .gelu)

        let sdxl = NFKMLXStableDiffusionRelease(.sdxlTurbo).configuration
        XCTAssertEqual(sdxl.unet.crossAttentionDimensions, 2048)
        XCTAssertEqual(sdxl.secondaryTextEncoder?.width, 1280)
        XCTAssertNotNil(sdxl.unet.additionEmbedding)
        XCTAssertEqual(sdxl.timestepSpacing, .trailing)
    }

    // A request's generation parameters override the release's defaults, which is what an InferKit
    // request carries these keys for.
    func testARequestOverridesTheSamplingDefaults() {
        var configuration = NFKDiffusionConfiguration(steps: 20, guidanceScale: 7.5)
        configuration.strength = 1
        let request = NFKInferenceRequest(
            inputs: [NFKInputPrompt: "a red car"],
            parameters: [NFKParameterSteps: 4, NFKParameterGuidanceScale: 2.5,
                         NFKParameterStrength: 0.6, NFKParameterSeed: 42])

        let resolved = NFKMLXDiffusionBackend.resolved(configuration, from: request)
        XCTAssertEqual(resolved.steps, 4)
        XCTAssertEqual(resolved.guidanceScale, 2.5)
        XCTAssertEqual(resolved.strength, 0.6)
        XCTAssertEqual(resolved.seed, 42)
    }

    // The model's own job carries the per-step progress and takes the cancellation. Wrapping it
    // without forwarding both would leave a caller watching a job that never moves.
    func testTheBackendForwardsTheModelsProgressAndResult() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test`; run via xcodebuild")
        let backend = NFKMLXBackend(model: .stableDiffusion15,
                                    loaded: NFKMLXTextToImage.backend(configuration: .tiny))
        XCTAssertTrue(backend.isReady)

        let finished = expectation(description: "the job finishes")
        var reported = [Double]()
        let job = backend.submitInferenceJob(for: NFKInferenceRequest(
            inputs: [NFKInputPrompt: "a lighthouse"], parameters: [NFKParameterSteps: 3]))
        job.progressHandler = { reported.append($0.progress) }
        job.completionHandler = { _ in finished.fulfill() }
        wait(for: [finished], timeout: 120)

        XCTAssertNotNil(job.result?.output(forKey: NFKOutputImage))
        XCTAssertEqual(reported.count, 3, "one report per denoising step")
        XCTAssertEqual(try XCTUnwrap(reported.last), 1, accuracy: 1e-6, "the last step completes the run")
    }

    func testAnEmptyRequestKeepsTheReleaseDefaults() {
        let configuration = NFKDiffusionConfiguration(steps: 20, guidanceScale: 7.5)
        let resolved = NFKMLXDiffusionBackend.resolved(configuration, from: NFKInferenceRequest(inputs: [:]))
        XCTAssertEqual(resolved.steps, 20)
        XCTAssertEqual(resolved.guidanceScale, 7.5)
        XCTAssertNil(resolved.seed)
    }
}
