//
//  NFKMLXFluxControlNetBackendTests.swift
//  InferKitMLXTests
//
//  FLUX.1 ControlNet behind the InferKit contract, at a tiny random geometry: a prompt and a control
//  image become an image at the requested size, a union ControlNet takes its control type as a request
//  key, and a request's settings do not outlive it.
//

import XCTest
import CoreGraphics
import InferKit
import MLX
import MLXRandom
@testable import InferKitMLX

final class NFKMLXFluxControlNetBackendTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    /// A CLIP vocabulary small enough to write, with its two markers.
    static func clipTokenizer(_ test: XCTestCase) throws -> NFKMLXSDPromptTokenizer {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("controlnet-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        test.addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        let vocabulary: [String: Int] = ["a</w>": 1, "red</w>": 2, "fox</w>": 3,
                                         "<|startoftext|>": 10, "<|endoftext|>": 11]
        try JSONSerialization.data(withJSONObject: vocabulary).write(to: directory.appendingPathComponent("vocab.json"))
        try "#version: 0.2\n".write(to: directory.appendingPathComponent("merges.txt"), atomically: true, encoding: .utf8)
        return try NFKMLXSDPromptTokenizer(directoryURL: directory)
    }

    /// A unigram SentencePiece model assembled as its protobuf: the unknown marker and three pieces.
    static func t5Segmenter() throws -> NFKMLXSentencePieceSegmenter {
        func varint(_ value: Int) -> [UInt8] {
            var v = value, out = [UInt8]()
            repeat {
                var byte = UInt8(v & 0x7f)
                v >>= 7
                if v != 0 { byte |= 0x80 }
                out.append(byte)
            } while v != 0
            return out
        }
        func lengthDelimited(field: Int, _ bytes: [UInt8]) -> [UInt8] { varint(field << 3 | 2) + varint(bytes.count) + bytes }
        var body = [UInt8]()
        let pieces: [(String, Float, Int)] = [("<unk>", 0, 2), ("\u{2581}a", -1, 1), ("\u{2581}red", -1, 1),
                                              ("\u{2581}fox", -1, 1)]
        for (text, score, type) in pieces {
            var piece = lengthDelimited(field: 1, Array(text.utf8))
            piece += varint(2 << 3 | 5) + withUnsafeBytes(of: score.bitPattern.littleEndian) { Array($0) }
            piece += varint(3 << 3 | 0) + varint(type)
            body += lengthDelimited(field: 1, piece)
        }
        body += lengthDelimited(field: 2, varint(3 << 3 | 0) + varint(1))
        body += lengthDelimited(field: 3, lengthDelimited(field: 1, Array("nmt_nfkc".utf8)))
        return NFKMLXSentencePieceSegmenter(model: try NFKMLXSentencePieceModel(data: Data(body)))
    }

    /// A small RGBA control image.
    static func plate(width: Int, height: Int) -> CGImage {
        var bytes = [UInt8](repeating: 255, count: width * height * 4)
        for index in stride(from: 0, to: bytes.count, by: 4) {
            bytes[index] = UInt8((index / 4) % 251)
            bytes[index + 1] = UInt8((index / 8) % 241)
        }
        let provider = CGDataProvider(data: Data(bytes) as CFData)!
        return CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4,
                       space: CGColorSpaceCreateDeviceRGB(),
                       bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
    }

    static func cgImage(_ value: Any?) throws -> CGImage {
        guard let value, CFGetTypeID(value as CFTypeRef) == CGImage.typeID else {
            throw NFKMLXError.noOutput
        }
        return (value as! CGImage)
    }

    /// A tiny [dev]-shaped base and a three-mode union ControlNet.
    private func tinyGenerator() throws -> NFKMLXFluxControlNetGenerator {
        MLXRandom.seed(23)
        var clip = NFKMLXSDTextEncoderConfiguration.tiny
        clip.contextLength = 77
        let t5 = NFKMLXT5Configuration.tiny
        let base = NFKMLXFluxConfiguration(
            inChannels: 8, outChannels: 8, numLayers: 3, numSingleLayers: 3, attentionHeadDim: 6,
            numAttentionHeads: 2, jointAttentionDim: t5.dModel, pooledProjectionDim: clip.width,
            guidanceEmbeds: true, axesDimsRope: [2, 2, 2])
        let control = NFKMLXFluxControlNetConfiguration(
            inChannels: 8, numLayers: 2, numSingleLayers: 2, attentionHeadDim: 6, numAttentionHeads: 2,
            jointAttentionDim: t5.dModel, pooledProjectionDim: clip.width, guidanceEmbeds: true,
            axesDimsRope: [2, 2, 2], numMode: 3)
        var vae = NFKMLXSDVAEConfiguration()
        vae.latentChannels = 2
        vae.blockChannels = [8, 16]
        vae.layersPerBlock = 1
        vae.normalizationGroups = 4
        vae.useQuantConv = false
        let tokenizer = try Self.clipTokenizer(self)
        let segmenter = try Self.t5Segmenter()
        let generator = NFKMLXFluxControlNetGenerator(
            resident: true, controlModeCount: 3, steps: 28,
            pixelsPerLatent: 1 << (vae.blockChannels.count - 1),
            loadTextEncoder: {
                NFKMLXFluxTextEncoder(clip: NFKMLXSDTextEncoderNet(configuration: clip), clipTokenizer: tokenizer,
                                      t5: NFKMLXT5Encoder(net: NFKMLXT5Encoder.makeNet(t5)), t5Segmenter: segmenter,
                                      t5Context: 8)
            },
            loadPipeline: {
                NFKMLXFluxControlNetPipeline(transformer: NFKMLXFluxTransformerNet(base),
                                             controlnet: NFKMLXFluxControlNetNet(control),
                                             vae: NFKMLXSDAutoencoder(configuration: vae), schedule: .flux)
            })
        try generator.loadResident()
        return generator
    }

    func testAPromptAndAControlImageBecomeAnImageAtTheRequestedSize() throws {
        try requireMLXRuntime()
        let generator = try tinyGenerator()
        let backend = generator.inferenceBackend()
        XCTAssertEqual(backend.backendIdentifier, NFKMLXFluxControlNetGenerator.modelName)
        XCTAssertEqual(backend.supportedInputKeys, [NFKInputPrompt, NFKInputControl])

        let request = NFKInferenceRequest(
            inputs: [NFKInputPrompt: "a red fox", NFKInputControl: Self.plate(width: 20, height: 12)],
            parameters: [NFKParameterWidth: 8, NFKParameterHeight: 12, NFKParameterSteps: 2, NFKParameterSeed: 1,
                         NFKMLXControlNetParameterKey.controlMode: 2,
                         NFKMLXControlNetParameterKey.conditioningScale: 0.5])
        let image = try Self.cgImage(backend.runInference(for: request).output(forKey: NFKOutputImage))
        XCTAssertEqual(image.width, 8)
        XCTAssertEqual(image.height, 12)
        XCTAssertEqual(generator.steps, 28, "the request's steps apply to that request only")
        XCTAssertEqual(generator.conditioningScale, 1, "and so does its conditioning scale")
    }

    func testAnUnsizedRequestTakesTheControlImagesSizeOnTheGrid() throws {
        try requireMLXRuntime()
        let request = NFKInferenceRequest(
            inputs: [NFKInputPrompt: "a fox", NFKInputControl: Self.plate(width: 13, height: 9)],
            parameters: [NFKParameterSteps: 1, NFKMLXControlNetParameterKey.controlMode: 0])
        let image = try Self.cgImage(try tinyGenerator().inferenceBackend().runInference(for: request)
            .output(forKey: NFKOutputImage))
        XCTAssertEqual(image.width, 12)
        XCTAssertEqual(image.height, 8)
    }

    func testAUnionControlNetRefusesARequestWithoutAValidMode() throws {
        try requireMLXRuntime()
        let backend = try tinyGenerator().inferenceBackend()
        let inputs: [String: Any] = [NFKInputPrompt: "a fox", NFKInputControl: Self.plate(width: 8, height: 8)]
        XCTAssertThrowsError(try backend.runInference(for: NFKInferenceRequest(
            inputs: inputs, parameters: [NFKParameterSteps: 1])))
        XCTAssertThrowsError(try backend.runInference(for: NFKInferenceRequest(
            inputs: inputs, parameters: [NFKParameterSteps: 1, NFKMLXControlNetParameterKey.controlMode: 3])))
    }

    func testARequestWithoutAControlImageIsRefused() throws {
        try requireMLXRuntime()
        let backend = try tinyGenerator().inferenceBackend()
        XCTAssertThrowsError(try backend.runInference(for: NFKInferenceRequest(
            inputs: [NFKInputPrompt: "a fox"], parameters: [:])))
    }
}
