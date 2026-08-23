//
//  NFKMLXReferenceModelsTests.swift
//  InferKitMLXTests
//
//  Registration stores factory closures and touches no MLX, so these run under `swift test`.
//

import XCTest
import InferKit
@testable import InferKitMLX

final class NFKMLXReferenceModelsTests: XCTestCase {

    func testRegisterAllPopulatesEveryBundledModel() {
        NFKMLXReferenceModels.registerAll()
        let names = Set(NFKMLXModelRegistry.registeredModelNames)
        for expected in ["green-screen-keyer", "tone-speech", "whisper-tiny", "demucs",
                         "diffusion-upscaler", "diffusion-depth", "diffusion-inpaint",
                         "real-esrgan-x4", "real-esrgan-x4-anime", "real-esrgan-x2",
                         "depth-anything-v2-small", "depth-anything-v2-base", "depth-anything-v2-large",
                         "u2net", "u2netp", "nafnet", "sam", "rife", "raft", "lama-inpaint", "sd-inpaint",
                         "marigold-depth", "sd-x4-upscaler"] {
            XCTAssertTrue(names.contains(expected), "registerAll did not register \(expected)")
        }
    }
}
