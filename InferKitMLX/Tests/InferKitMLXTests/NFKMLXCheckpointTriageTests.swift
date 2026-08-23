//
//  NFKMLXCheckpointTriageTests.swift
//  InferKitMLXTests
//
//  A triage sweep over the models whose converters still carry `--list-keys` (their reference key remap
//  was never worked out). For each, it attempts the real public factory against a converted checkpoint
//  and reports what the coverage guard says, so the whole group can be classified in one run:
//  loads cleanly, needs a name remap, or needs architecture work.
//
//  Reports rather than fails — the point is the survey. Configure paths in ~/.inferkit-validation.json
//  under IK_TRIAGE_<NAME>.
//

import XCTest
import Foundation
import InferKit
@testable import InferKitMLX

final class NFKMLXCheckpointTriageTests: XCTestCase {

    private lazy var config: [String: String] = {
        var merged = ProcessInfo.processInfo.environment
        let url = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".inferkit-validation.json")
        if let data = try? Data(contentsOf: url),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: String] {
            json.forEach { merged[$0.key] = $0.value }
        }
        return merged
    }()

    func testTriageEveryUnverifiedCheckpoint() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test`; run via xcodebuild")

        let models: [(String, (URL) throws -> any NFKInferenceBackend)] = [
            ("YOLO",        { try NFKMLXYOLO.backend(weightsURL: $0, labels: nil) }),
            ("CLIP",        { try NFKMLXCLIP.backend(weightsURL: $0) }),
            ("SAM",         { try NFKMLXSAM.backend(weightsURL: $0) }),
            ("DEEPLAB",     { try NFKMLXDeepLab.backend(weightsURL: $0) }),
            ("SWINIR",      { try NFKMLXSwinIR.backend(weightsURL: $0) }),
            ("CODEFORMER",  { try NFKMLXCodeFormer.backend(weightsURL: $0) }),
            ("RVM",         { try NFKMLXRVM.backend(weightsURL: $0) }),
            ("DENOISER",    { try NFKMLXDenoiser.backend(weightsURL: $0) }),
            ("SEGFORMER",   { try NFKMLXSegFormer.backend(weightsURL: $0) }),
            ("CONVTASNET",  { let net = NFKMLXConvTasNetNet(.libri2Mix16k)
                              try NFKMLXConvTasNet.loadWeights(into: net, from: $0)
                              return NFKMLXConvTasNetBackend(net: net, identifier: "conv-tasnet") }),
            ("AUDIOTAGGER", { try NFKMLXAudioTagger.backend(weightsURL: $0, labels: nil) }),
            ("VIDEOSR",     { try NFKMLXVideoSR.backend(weightsURL: $0) }),
            ("LAMA",        { try NFKMLXLaMa.backend(weightsURL: $0) }),
            ("DEMUCS",      { try NFKMLXDemucs.backend(weightsURL: $0) }),
            ("VAD",         { try NFKMLXVAD.backend(weightsURL: $0) }),
            ("NAFNET",      { try NFKMLXNAFNet.backend(weightsURL: $0) }),
            ("MODNET",      { try NFKMLXMODNet.backend(weightsURL: $0) }),
            ("BISENET",     { try NFKMLXBiSeNet.backend(weightsURL: $0) }),
            ("SIGGRAPH17",  { try NFKMLXSiggraphColorizer.backend(weightsURL: $0) }),
            ("BISENETV2",   { try NFKMLXBiSeNetV2.backend(weightsURL: $0) }),
            ("RIFEV4",      { try NFKMLXRIFEv4.backend(weightsURL: $0) }),
            ("SAM2MEMORY",  { let attention = NFKMLXSAM2.makeMemoryAttention()
                              let encoder = NFKMLXSAM2.makeMemoryEncoder()
                              try NFKMLXSAM2.loadMemoryWeights(into: attention, encoder: encoder, from: $0)
                              return NFKMLXModuleBackend(identifier: "sam2-memory", isReady: true) { $0 } }),
            ("SAM2DECODER", { let decoder = NFKMLXSAM2.makeDecoder()
                              let prompt = NFKMLXSAM2.makePromptEncoder()
                              try NFKMLXSAM2.loadDecoderWeights(into: decoder, prompt: prompt, from: $0)
                              return NFKMLXModuleBackend(identifier: "sam2-decoder", isReady: true) { $0 } }),
            ("SAM2",        { let net = NFKMLXSAM2.makeEncoder()
                              try NFKMLXSAM2.loadWeights(into: net, from: $0)
                              return NFKMLXModuleBackend(identifier: "sam2-encoder", isReady: true) { $0 } }),
            ("RIFE",        { try NFKMLXRIFE.backend(weightsURL: $0) }),
            ("HTDEMUCS",    { try NFKMLXHTDemucs.backend(weightsURL: $0) }),
            ("SDUNET",      { let net = NFKMLXSDUNet(configuration: .inpainting)
                              try NFKMLXStableDiffusionModels.loadUNetWeights(into: net, from: $0)
                              return NFKMLXModuleBackend(identifier: "sd-unet", isReady: true) { $0 } }),
            ("MARIGOLDUNET",{ let net = NFKMLXSDUNet(configuration: .marigold)
                              try NFKMLXStableDiffusionModels.loadUNetWeights(into: net, from: $0)
                              return NFKMLXModuleBackend(identifier: "marigold-unet", isReady: true) { $0 } }),
            ("UPSCALERUNET",{ let net = NFKMLXSDUNet(configuration: .upscaler)
                              try NFKMLXStableDiffusionModels.loadUNetWeights(into: net, from: $0)
                              return NFKMLXModuleBackend(identifier: "upscaler-unet", isReady: true) { $0 } }),
            ("UPSCALERVAE", { let net = NFKMLXSDAutoencoder(configuration: .upscaler)
                              try NFKMLXStableDiffusionModels.loadVAEWeights(into: net, from: $0)
                              return NFKMLXModuleBackend(identifier: "upscaler-vae", isReady: true) { $0 } }),
            ("SD15TEXT",    { _ = try NFKMLXSDTextEncoder.net(configuration: .stableDiffusion15, weightsURL: $0)
                              return NFKMLXModuleBackend(identifier: "sd15-text", isReady: true) { $0 } }),
            ("SD21TEXT",    { _ = try NFKMLXSDTextEncoder.net(configuration: .stableDiffusion21, weightsURL: $0)
                              return NFKMLXModuleBackend(identifier: "sd21-text", isReady: true) { $0 } }),
            ("SDXLTEXT",    { _ = try NFKMLXSDTextEncoder.net(configuration: .sdxlPrimary, weightsURL: $0)
                              return NFKMLXModuleBackend(identifier: "sdxl-text", isReady: true) { $0 } }),
            ("SDXLTEXT2",   { _ = try NFKMLXSDTextEncoder.net(configuration: .sdxlSecondary, weightsURL: $0)
                              return NFKMLXModuleBackend(identifier: "sdxl-text-2", isReady: true) { $0 } }),
            ("SD21UNET",    { var configuration = NFKMLXSDTextToImageConfiguration.stableDiffusion21.unet
                              configuration.inputChannels = 4
                              let net = NFKMLXSDUNet(configuration: configuration)
                              try NFKMLXStableDiffusionModels.loadUNetWeights(into: net, from: $0)
                              return NFKMLXModuleBackend(identifier: "sd21-unet", isReady: true) { $0 } }),
            ("SDXLUNET",    { let net = NFKMLXSDUNet(configuration: .sdxl)
                              try NFKMLXStableDiffusionModels.loadUNetWeights(into: net, from: $0)
                              return NFKMLXModuleBackend(identifier: "sdxl-unet", isReady: true) { $0 } }),
            ("SDVAE",       { let net = NFKMLXSDAutoencoder(configuration: .stableDiffusion)
                              try NFKMLXStableDiffusionModels.loadVAEWeights(into: net, from: $0)
                              return NFKMLXModuleBackend(identifier: "sd-vae", isReady: true) { $0 } }),
        ]

        let dumpDirectory = config["IK_VAL_OUTDIR"].map { URL(fileURLWithPath: $0, isDirectory: true) }
        defer { NFKMLXWeights.diagnosticsHandler = nil }

        var report = [String]()
        for (name, build) in models {
            guard let path = config["IK_TRIAGE_\(name)"], FileManager.default.fileExists(atPath: path) else {
                report.append("TRIAGE \(name): SKIP (no checkpoint configured)")
                continue
            }
            // Record both key sets so a remap can be worked out from the actual names on each side.
            NFKMLXWeights.diagnosticsHandler = { expected, provided in
                guard let dumpDirectory else { return }
                let text = "EXPECTED (\(expected.count))\n" + expected.sorted().joined(separator: "\n")
                    + "\n\nPROVIDED (\(provided.count))\n" + provided.sorted().joined(separator: "\n")
                try? text.write(to: dumpDirectory.appendingPathComponent("keys-\(name).txt"),
                                atomically: true, encoding: .utf8)
            }
            do {
                let backend = try build(URL(fileURLWithPath: path))
                report.append("TRIAGE \(name): LOADS — every parameter covered (identifier \(backend.backendIdentifier))")
            } catch NFKMLXError.weightsMismatch(let detail) {
                report.append("TRIAGE \(name): MISMATCH — \(detail)")
            } catch {
                report.append("TRIAGE \(name): ERROR — \(error)")
            }
        }
        print(report.joined(separator: "\n"))
    }
}
