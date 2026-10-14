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
                         "marigold-depth", "sd-x4-upscaler", "basic-pitch", "allin1", "muscriptor", "muscriptor-small", "muscriptor-large", "hft-transformer",
                         "parakeet-tdt", "canary-1b-v2", "voxtral-mini-3b", "granite-speech-3.3-2b", "phi-4-multimodal",
                         "codestral-mamba", "granite-4.0-h", "nemotron-nano-2", "qwen4-exp", "qwen3.5",
                         "chronos-bolt-base", "sam2-video", "rife-clip", "video-super-resolution-clip",
                         "robust-video-matting-clip", "robust-video-matting-resnet50-clip",
                         "ltx-video-0.9.0", "wan", "z-image", "qwen-image-2.1", "sd3", "flux.1-schnell",
                         "flux.2-klein-4b", "smolvlm2-500m", "qwen3-vl-2b", "pixtral-12b", "gte-reranker-modernbert-base",
                         "qwen3-vl-reranker-2b", "timesfm-2.5-200m", "sana", "sam3", "flux.1-controlnet", "sd3-controlnet",
                         "fastspeech2-voice", "clip-probe", "gemma4-multimodal"] {
            XCTAssertTrue(names.contains(expected), "registerAll did not register \(expected)")
        }
        for variant in NFKMLXCosmosTokenizerVariant.allCases {
            XCTAssertTrue(names.contains(variant.modelName), "registerAll did not register \(variant.modelName)")
        }
    }

    func testADirectoryModelRefusesToBuildWithoutWeights() {
        NFKMLXReferenceModels.registerAll()
        for name in ["parakeet-tdt", "canary-1b-v2", "voxtral-mini-3b", "granite-speech-3.3-2b", "phi-4-multimodal",
                     "codestral-mamba", "granite-4.0-h", "nemotron-nano-2", "qwen4-exp", "qwen3.5",
                     "ltx-video-0.9.0", "wan", "z-image", "qwen-image-2.1", "sd3", "flux.1-schnell",
                     "flux.2-klein-4b", "smolvlm2-500m", "qwen3-vl-2b", "pixtral-12b", "gte-reranker-modernbert-base",
                     "qwen3-vl-reranker-2b", "timesfm-2.5-200m", "sana", "sam3", "flux.1-controlnet", "sd3-controlnet",
                     "fastspeech2-voice", "clip-probe", "gemma4-multimodal"] {
            XCTAssertThrowsError(try NFKMLXModelRegistry.backend(named: name, weightsURL: nil), name)
        }
    }
}
