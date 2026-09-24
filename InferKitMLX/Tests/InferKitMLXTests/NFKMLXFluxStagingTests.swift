//
//  NFKMLXFluxStagingTests.swift
//  InferKitMLXTests
//
//  FLUX.1's staging: a resident facade and a staged one over the same weights produce the same image,
//  and the staged one holds neither stage between images. The arithmetic of each stage is measured in
//  the parity tests.
//

import XCTest
import InferKit
import MLX
import MLXNN
@testable import InferKitMLX

final class NFKMLXFluxStagingTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    /// A CLIP vocabulary small enough to write, with its two markers.
    private func clipTokenizer() throws -> NFKMLXSDPromptTokenizer {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("flux-staging-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        let vocabulary: [String: Int] = ["a</w>": 1, "red</w>": 2, "fox</w>": 3,
                                         "<|startoftext|>": 10, "<|endoftext|>": 11]
        try JSONSerialization.data(withJSONObject: vocabulary)
            .write(to: directory.appendingPathComponent("vocab.json"))
        try "#version: 0.2\n".write(to: directory.appendingPathComponent("merges.txt"),
                                    atomically: true, encoding: .utf8)
        return try NFKMLXSDPromptTokenizer(directoryURL: directory)
    }

    /// A unigram SentencePiece model assembled as its protobuf: the unknown marker and three pieces.
    private func t5Segmenter() throws -> NFKMLXSentencePieceSegmenter {
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

    func testFluxStagingReleasesEachStageAndChangesNothing() throws {
        try requireMLXRuntime()
        MLXRandom.seed(31)
        var clip = NFKMLXSDTextEncoderConfiguration.tiny
        clip.contextLength = 77
        let t5 = NFKMLXT5Configuration.tiny
        let geometry = NFKMLXFluxConfiguration(
            inChannels: 8, outChannels: 8, numLayers: 1, numSingleLayers: 1, attentionHeadDim: 6,
            numAttentionHeads: 2, jointAttentionDim: t5.dModel, pooledProjectionDim: clip.width,
            guidanceEmbeds: false, axesDimsRope: [2, 2, 2])
        var vae = NFKMLXSDVAEConfiguration()
        vae.latentChannels = 2
        vae.blockChannels = [8, 16]
        vae.layersPerBlock = 1
        vae.normalizationGroups = 4
        vae.useQuantConv = false

        // One set of weights, applied to a fresh module on every load, so a staged reload reads the
        // same network a resident load holds.
        func captured(_ module: Module) -> [(String, MLXArray)] {
            let parameters = module.parameters().flattened()
            eval(parameters.map(\.1))
            return parameters
        }
        let clipWeights = captured(NFKMLXSDTextEncoderNet(configuration: clip))
        let t5Weights = captured(NFKMLXT5Encoder.makeNet(t5))
        let transformerWeights = captured(NFKMLXFluxTransformerNet(geometry))
        let autoencoderWeights = captured(NFKMLXSDAutoencoder(configuration: vae))
        let tokenizer = try clipTokenizer()
        let segmenter = try t5Segmenter()

        var loads = (encoder: 0, pipeline: 0)
        func facade(resident: Bool) throws -> NFKMLXFlux {
            let flux = NFKMLXFlux(
                resident: resident,
                loadTextEncoder: {
                    loads.encoder += 1
                    let clipNet = NFKMLXSDTextEncoderNet(configuration: clip)
                    try NFKMLXWeights.apply(clipWeights, to: clipNet)
                    let t5Net = NFKMLXT5Encoder.makeNet(t5)
                    try NFKMLXWeights.apply(t5Weights, to: t5Net)
                    return NFKMLXFluxTextEncoder(clip: clipNet, clipTokenizer: tokenizer, t5: NFKMLXT5Encoder(net: t5Net),
                                                 t5Segmenter: segmenter, t5Context: 8)
                },
                loadPipeline: {
                    loads.pipeline += 1
                    let transformer = NFKMLXFluxTransformerNet(geometry)
                    try NFKMLXWeights.apply(transformerWeights, to: transformer)
                    let autoencoder = NFKMLXSDAutoencoder(configuration: vae)
                    try NFKMLXWeights.apply(autoencoderWeights, to: autoencoder)
                    return NFKMLXFluxPipeline(transformer: transformer, vae: autoencoder)
                })
            if resident {
                try flux.loadResident()
            }
            flux.steps = 2
            return flux
        }

        let resident = try facade(resident: true)
        let residentImages = try (0 ..< 2).map { _ in
            try resident.image(forPrompt: "a red fox", width: 32, height: 32, seed: 5)
        }
        XCTAssertEqual(loads.encoder, 1, "a resident facade loads the encoders once")
        XCTAssertEqual(loads.pipeline, 1, "and the pipeline once")
        XCTAssertTrue(resident.holdsStagesResident)
        XCTAssertTrue(resident.isHoldingTextEncoder && resident.isHoldingPipeline)

        loads = (0, 0)
        let staged = try facade(resident: false)
        XCTAssertFalse(staged.holdsStagesResident)
        XCTAssertFalse(staged.isHoldingTextEncoder || staged.isHoldingPipeline, "a staged facade loads nothing up front")
        let stagedImages = try (0 ..< 2).map { _ in
            try staged.image(forPrompt: "a red fox", width: 32, height: 32, seed: 5)
        }
        XCTAssertEqual(loads.encoder, 2, "a staged facade loads the encoders once per image")
        XCTAssertEqual(loads.pipeline, 2, "and the pipeline once per image")
        XCTAssertFalse(staged.isHoldingTextEncoder || staged.isHoldingPipeline, "and holds neither after")

        for (residentImage, stagedImage) in zip(residentImages, stagedImages) {
            XCTAssertEqual(stagedImage.reshaped([-1]).asArray(Float.self),
                           residentImage.reshaped([-1]).asArray(Float.self),
                           "staging does not change the image")
        }
    }
}
