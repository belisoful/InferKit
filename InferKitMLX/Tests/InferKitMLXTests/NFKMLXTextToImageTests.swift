//
//  NFKMLXTextToImageTests.swift
//  InferKitMLXTests
//
//  Structure, the model input a prompt becomes, and a round trip through the public backend. The
//  numbers are measured against diffusers' own pipelines in `NFKMLXReferenceParityTests`.
//

import XCTest
import CoreGraphics
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXTextToImageTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test`; run via xcodebuild")
    }

    // MARK: The model input a prompt becomes

    /// A vocabulary small enough to write, with the two markers and a padding token of its own.
    private func makeTokenizerDirectory(padToken: String) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)

        let vocabulary: [String: Int] = ["a</w>": 1, "b</w>": 2, "c</w>": 3, "!": 4,
                                         "<|startoftext|>": 10, "<|endoftext|>": 11]
        try JSONSerialization.data(withJSONObject: vocabulary)
            .write(to: directory.appendingPathComponent("vocab.json"))
        try "#version: 0.2\n".write(to: directory.appendingPathComponent("merges.txt"),
                                   atomically: true, encoding: .utf8)
        try JSONSerialization.data(withJSONObject: ["pad_token": padToken])
            .write(to: directory.appendingPathComponent("special_tokens_map.json"))
        return directory
    }

    func testAPromptBecomesTheMarkersAroundItPaddedToTheContextLength() throws {
        let directory = try makeTokenizerDirectory(padToken: "<|endoftext|>")
        defer { try? FileManager.default.removeItem(at: directory) }

        let tokenizer = try NFKMLXSDPromptTokenizer(directoryURL: directory)
        XCTAssertEqual(tokenizer.tokens(for: "a b", contextLength: 6), [10, 1, 2, 11, 11, 11])
    }

    // Stable Diffusion 2.x and SDXL's second tower pad with `!`, which `special_tokens_map.json` names
    // and `tokenizer_config.json` does not — padding with the wrong token makes the model read a
    // different sentence.
    func testTheReleaseNamesItsOwnPaddingToken() throws {
        let directory = try makeTokenizerDirectory(padToken: "!")
        defer { try? FileManager.default.removeItem(at: directory) }

        let tokenizer = try NFKMLXSDPromptTokenizer(directoryURL: directory)
        XCTAssertEqual(tokenizer.paddingTokenId, 4)
        XCTAssertEqual(tokenizer.tokens(for: "a", contextLength: 5), [10, 1, 11, 4, 4])
    }

    // A prompt longer than the context is cut with the end marker kept last, as the reference's
    // truncation does.
    func testAnOverlongPromptKeepsItsEndMarker() throws {
        let directory = try makeTokenizerDirectory(padToken: "<|endoftext|>")
        defer { try? FileManager.default.removeItem(at: directory) }

        let tokenizer = try NFKMLXSDPromptTokenizer(directoryURL: directory)
        let tokens = tokenizer.tokens(for: "a b c a b c", contextLength: 4)
        XCTAssertEqual(tokens, [10, 1, 2, 11])
    }

    // MARK: The released directory layout

    func testTheReleaseLayoutResolvesHalfPrecisionFilenames() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        for (folder, name) in [("unet", "diffusion_pytorch_model.fp16.safetensors"),
                               ("vae", "diffusion_pytorch_model.safetensors"),
                               ("text_encoder", "model.fp16.safetensors")] {
            let subdirectory = directory.appendingPathComponent(folder)
            try FileManager.default.createDirectory(at: subdirectory, withIntermediateDirectories: true)
            try Data().write(to: subdirectory.appendingPathComponent(name))
        }

        let files = try NFKMLXSDReleaseFiles(directoryURL: directory)
        XCTAssertEqual(files.unet.lastPathComponent, "diffusion_pytorch_model.fp16.safetensors")
        XCTAssertEqual(files.vae.lastPathComponent, "diffusion_pytorch_model.safetensors")
        XCTAssertNil(files.secondaryTextEncoder, "this release carries one text tower")
    }

    func testAReleaseMissingAFileIsRejected() {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        XCTAssertThrowsError(try NFKMLXSDReleaseFiles(directoryURL: directory))
    }

    // MARK: Structure

    func testTheTextEncoderRemapCoversEveryParameter() throws {
        try requireMLXRuntime()
        let configuration = NFKMLXSDTextEncoderConfiguration.tiny
        let net = NFKMLXSDTextEncoderNet(configuration: configuration)

        // The reference's own key layout, which the remap has to fold onto the module's.
        var reference: [String: MLXArray] = [
            "text_model.embeddings.token_embedding.weight":
                MLXArray.zeros([configuration.vocabularySize, configuration.width]),
            "text_model.embeddings.position_embedding.weight":
                MLXArray.zeros([configuration.contextLength, configuration.width]),
            "text_model.embeddings.position_ids": MLXArray.zeros([1, configuration.contextLength]),
            "text_model.final_layer_norm.weight": MLXArray.ones([configuration.width]),
            "text_model.final_layer_norm.bias": MLXArray.zeros([configuration.width]),
        ]
        for layer in 0 ..< configuration.layers {
            let prefix = "text_model.encoder.layers.\(layer)."
            for part in ["q_proj", "k_proj", "v_proj", "out_proj"] {
                reference[prefix + "self_attn.\(part).weight"] =
                    MLXArray.zeros([configuration.width, configuration.width])
                reference[prefix + "self_attn.\(part).bias"] = MLXArray.zeros([configuration.width])
            }
            for part in ["layer_norm1", "layer_norm2"] {
                reference[prefix + "\(part).weight"] = MLXArray.ones([configuration.width])
                reference[prefix + "\(part).bias"] = MLXArray.zeros([configuration.width])
            }
            reference[prefix + "mlp.fc1.weight"] =
                MLXArray.zeros([configuration.intermediate, configuration.width])
            reference[prefix + "mlp.fc1.bias"] = MLXArray.zeros([configuration.intermediate])
            reference[prefix + "mlp.fc2.weight"] =
                MLXArray.zeros([configuration.width, configuration.intermediate])
            reference[prefix + "mlp.fc2.bias"] = MLXArray.zeros([configuration.width])
        }

        // Throws unless every parameter of the module is covered.
        try NFKMLXWeights.apply(NFKMLXSDTextEncoder.remap(reference), to: net)
    }

    func testTheTextEncoderProducesTheContextTheUNetTakes() throws {
        try requireMLXRuntime()
        let configuration = NFKMLXSDTextEncoderConfiguration.tiny
        let net = NFKMLXSDTextEncoderNet(configuration: configuration)
        let embedding = net.encode(Array(repeating: 1, count: configuration.contextLength))
        eval(embedding.hidden)
        XCTAssertEqual(embedding.hidden.shape, [1, configuration.contextLength, configuration.width])
        XCTAssertNil(embedding.pooled, "only a tower carrying a projection pools")
    }

    // A tower with a projection also supplies the pooled embedding SDXL conditions on.
    func testAProjectedTowerPools() throws {
        try requireMLXRuntime()
        var configuration = NFKMLXSDTextEncoderConfiguration.tiny
        configuration.projectionDimensions = 6
        let net = NFKMLXSDTextEncoderNet(configuration: configuration)
        let embedding = net.encode(Array(repeating: 1, count: configuration.contextLength))
        let pooled = try XCTUnwrap(embedding.pooled)
        eval(pooled)
        XCTAssertEqual(pooled.shape, [1, 6])
    }

    // MARK: Round trip

    // Random weights make noise rather than a picture; what this proves is that the whole chain runs
    // and returns an image of the requested size through the public backend.
    func testTheBackendReturnsAnImageOfTheRequestedSize() throws {
        try requireMLXRuntime()
        let backend = NFKMLXTextToImage.backend(configuration: .tiny)
        let request = NFKInferenceRequest(inputs: [NFKInputPrompt: "a lighthouse"],
                                          parameters: [NFKParameterSteps: 2,
                                                       NFKParameterWidth: 64, NFKParameterHeight: 64])
        let result = try backend.runInference(for: request)
        let image = try XCTUnwrap(result.output(forKey: NFKOutputImage)) as! CGImage
        XCTAssertEqual(image.width, 64)
        XCTAssertEqual(image.height, 64)
    }
}
