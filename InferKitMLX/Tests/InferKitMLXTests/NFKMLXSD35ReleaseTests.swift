//
//  NFKMLXSD35ReleaseTests.swift
//  InferKitMLXTests
//
//  The released SD 3.5 transformers and VAE at float32 against diffusers, `run_reference.py sd35_release`.
//  A release runs when its diffusers directory (`IK_VAL_SD35_<SIZE>`) and record (`IK_PARITY_SD35_<SIZE>`)
//  are both present; the 8B releases are measured on four-block cuts (the first three blocks and the final
//  `context_pre_only` block). The text encoders are not re-measured here: SD 3.5's CLIP-G is byte-identical to
//  SDXL's second tower, its CLIP-L to SDXL's first tower with a projection, and both are at parity there.
//

import XCTest
import MLX
import MLXNN
@testable import InferKitMLX

final class NFKMLXSD35ReleaseTests: XCTestCase {

    override func tearDown() {
        Memory.clearCache()
        super.tearDown()
    }

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    /// PARITY: each release's transformer seam by seam, its velocity, and its VAE's decode and encode.
    func testEveryReleaseIsAtParity() throws {
        try requireMLXRuntime()
        let env = NFKMLXValidationConfig.environment
        var measured = 0
        for size in ["MEDIUM", "LARGE", "LARGE_TURBO"] {
            guard let directory = env["IK_VAL_SD35_\(size)"], let path = env["IK_PARITY_SD35_\(size)"],
                  FileManager.default.fileExists(atPath: path) else { continue }
            try autoreleasepool { try measure(size, directory: URL(fileURLWithPath: directory), recordPath: path) }
            Memory.clearCache()
            measured += 1
        }
        if measured == 0 { throw XCTSkip("set IK_VAL_SD35_<SIZE> and IK_PARITY_SD35_<SIZE>") }
    }

    private func measure(_ size: String, directory: URL, recordPath: String) throws {
        let record = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: recordPath)).arrays
        var readings = [(String, Double)]()
        try autoreleasepool {
            let configuration = try NFKMLXSD3TransformerNet.configuration(
                fromHuggingFace: directory.appendingPathComponent("transformer/config.json"))
            let net = NFKMLXSD3TransformerNet(configuration)
            try NFKMLXSD3TransformerNet.loadWeights(into: net, from: directory.appendingPathComponent("transformer"),
                                                    precision: .float32)
            let latent = try XCTUnwrap(record["latent"]), encoder = try XCTUnwrap(record["encoder"])
            let pooled = try XCTUnwrap(record["pooled"]), timestep = try XCTUnwrap(record["timestep"])
            let picks = Set(try XCTUnwrap(record["block_indices"]).asArray(Int64.self).map(Int.init))

            var image = net.posEmbed(latent)
            readings.append(("patch", NFKMLXWav2Vec2Tests.cosine(image[0], record["patch"]!)))
            let temb = net.timeTextEmbed(timestep: timestep, pooled: pooled)
            var context: MLXArray? = net.contextEmbedder(encoder)
            for (index, block) in net.transformerBlocks.enumerated() {
                let (newContext, newImage) = block(image, encoder: context!, temb: temb)
                image = newImage
                context = newContext
                guard picks.contains(index) else { continue }
                readings.append(("block\(index) image", NFKMLXWav2Vec2Tests.cosine(image[0], try XCTUnwrap(record["block\(index).image"]))))
                if let context, let reference = record["block\(index).text"] {
                    readings.append(("block\(index) text", NFKMLXWav2Vec2Tests.cosine(context[0], reference)))
                }
            }
            let velocity = net(latent, encoderHidden: encoder, pooled: pooled, timestep: timestep)
            readings.append(("velocity", NFKMLXWav2Vec2Tests.cosine(velocity[0], record["output"]!)))
        }
        Memory.clearCache()

        // A transformer cut (`truncate.py --diffusers-transformer`) carries no VAE; the three releases share
        // one, measured through medium.
        guard FileManager.default.fileExists(atPath: directory.appendingPathComponent("vae/config.json").path) else {
            print("VALIDATION PARITY sd3.5 \(size): " + readings.map { "\($0.0) \($0.1)" }.joined(separator: ", "))
            for (seam, similarity) in readings { XCTAssertGreaterThan(similarity, 0.99999, "\(size) \(seam) diverges") }
            return
        }
        let vaeConfiguration = try NFKMLXStableDiffusionModels.vaeConfiguration(
            fromHuggingFace: directory.appendingPathComponent("vae/config.json"))
        let vae = NFKMLXSDAutoencoder(configuration: vaeConfiguration)
        try NFKMLXStableDiffusionModels.loadVAEWeights(
            into: vae, from: directory.appendingPathComponent("vae/diffusion_pytorch_model.safetensors"), precision: .float32)
        let json = try JSONSerialization.jsonObject(with: Data(contentsOf: directory.appendingPathComponent("vae/config.json"))) as? [String: Any]
        let scale = (json?["scaling_factor"] as? NSNumber)?.floatValue ?? 1
        let shift = (json?["shift_factor"] as? NSNumber)?.floatValue ?? 0
        let latent = try XCTUnwrap(record["latent"])
        let decoded = vae.decode((latent / scale + shift).transposed(0, 2, 3, 1))
        let referenceImage = try XCTUnwrap(record["vae_image"]).transposed(0, 2, 3, 1)
        readings.append(("vae decode", NFKMLXWav2Vec2Tests.cosine(decoded, referenceImage)))
        let mean = vae.encode(clip(referenceImage, min: -1, max: 1)).mean
        readings.append(("vae encode mean", NFKMLXWav2Vec2Tests.cosine(mean, try XCTUnwrap(record["vae_mean"]).transposed(0, 2, 3, 1))))

        print("VALIDATION PARITY sd3.5 \(size): " + readings.map { "\($0.0) \($0.1)" }.joined(separator: ", "))
        for (seam, similarity) in readings {
            XCTAssertGreaterThan(similarity, 0.99999, "\(size) \(seam) diverges")
        }
    }
}
