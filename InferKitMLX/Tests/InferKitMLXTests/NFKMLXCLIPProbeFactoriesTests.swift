//
//  NFKMLXCLIPProbeFactoriesTests.swift
//  InferKitMLXTests
//
//  A saved CLIP probe reached the way Objective-C reaches it: the variant, the CLIP weights, the probe
//  file, and the class names become a classifier backend; a probe of another width is refused; the
//  registry name refuses a missing directory.
//

import XCTest
import CoreGraphics
import InferKit
import MLX
import MLXRandom
@testable import InferKitMLX

final class NFKMLXCLIPProbeFactoriesTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    private func savedProbe(width: Int, classes: Int) throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("clip-probe-\(UUID().uuidString).safetensors")
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        try NFKMLXWeights.save(NFKMLXCLIPProbe(embedDimensions: width, classCount: classes), to: url)
        return url
    }

    func testASavedProbeOverTheVariantClassifiesAnImage() throws {
        try requireMLXRuntime()
        MLXRandom.seed(53)
        let backend = try NFKMLXCLIP.probeBackend(variant: .vitB32, weightsURL: nil,
                                                  probeURL: try savedProbe(width: 512, classes: 3),
                                                  labels: ["cat", "dog", "fox"])
        XCTAssertEqual(backend.backendIdentifier, NFKMLXCLIP.probeModelName)
        let image = NFKMLXFluxControlNetBackendTests.plate(width: 224, height: 224)
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: image], parameters: [:]))
        let classes = try XCTUnwrap(result.output(forKey: NFKOutputClassifications) as? [NFKClassification])
        XCTAssertEqual(Set(classes.compactMap(\.label)), ["cat", "dog", "fox"])
    }

    func testAProbeOfAnotherWidthIsRefused() throws {
        try requireMLXRuntime()
        XCTAssertThrowsError(try NFKMLXCLIP.probeBackend(variant: .vitL14, weightsURL: nil,
                                                         probeURL: try savedProbe(width: 512, classes: 2), labels: nil))
    }

    func testTheRegistryRefusesAMissingProbeDirectory() throws {
        NFKMLXCLIP.registerProbe()
        XCTAssertThrowsError(try NFKMLXModelRegistry.backend(named: NFKMLXCLIP.probeModelName, weightsURL: nil))
    }
}
