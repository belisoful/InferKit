//
//  NFKMLXCatalogLiveTests.swift
//  InferKitMLXTests
//
//  The live profiles in the package's catalog. Building descriptors touches no MLX.
//

import XCTest
import InferKit
@testable import InferKitMLX

final class NFKMLXCatalogLiveTests: XCTestCase {

    private lazy var descriptors = NFKMLXModelDescriptors.modelDescriptors()

    func testEveryLiveModelIsAShippedEntryWhosePortsCarryTheFrame() throws {
        for (identifier, outputs) in NFKMLXCatalogLive.videoFrameOutputs {
            let descriptor = try XCTUnwrap(descriptors.first { $0.identifier == identifier }, "\(identifier) has no catalog entry")
            let live = try XCTUnwrap(descriptor.liveProfile, identifier)
            XCTAssertEqual(live.unit, NFKModelLiveUnitVideoFrame, identifier)
            XCTAssertEqual(live.inputKeys, [NFKInputImage], identifier)
            XCTAssertEqual(live.outputKeys, outputs, identifier)
            XCTAssertFalse(live.isStateful, identifier)
            XCTAssertTrue(descriptor.inputs.contains { $0.key == NFKInputImage && $0.isRequired }, identifier)
        }
    }

    func testEveryLiveModelCarriesItsMeasuredTimings() throws {
        let live = descriptors.filter { $0.liveProfile != nil }
        XCTAssertEqual(Set(live.map(\.identifier)), Set(NFKMLXCatalogLive.timings.keys), "every live model is timed, and only those")
        for descriptor in live {
            let profile = try XCTUnwrap(descriptor.liveProfile)
            let sizes = profile.timings.map { "\($0.width)x\($0.height)" }
            if profile.unit == NFKModelLiveUnitVideoFrame {
                XCTAssertEqual(Set(sizes), ["1280x720", "1920x1080"], descriptor.identifier)
            } else {
                XCTAssertEqual(sizes, ["0x0"], descriptor.identifier)
            }
            for timing in profile.timings {
                XCTAssertNil(timing.variantIdentifier, "\(descriptor.identifier): the default variant was measured")
                XCTAssertEqual(timing.chip, NFKMLXCatalogLive.measuredChip, descriptor.identifier)
                XCTAssertEqual(timing.measuredDate, NFKMLXCatalogLive.measuredDate, descriptor.identifier)
                XCTAssertGreaterThan(timing.millisecondsPerUnit, 0, descriptor.identifier)
            }
        }
    }

    func testTheStatefulModelsDeclareTheirState() throws {
        for (identifier, outputs) in NFKMLXCatalogLive.statefulVideoFrameOutputs {
            let live = try XCTUnwrap(descriptors.first { $0.identifier == identifier }?.liveProfile, identifier)
            XCTAssertTrue(live.isStateful, identifier)
            XCTAssertEqual(live.outputKeys, outputs, identifier)
        }
    }

    func testTheAudioModelsDeclareTheirChunk() throws {
        let silero = try XCTUnwrap(descriptors.first { $0.identifier == "silero-vad" }?.liveProfile)
        XCTAssertEqual(silero.unit, NFKModelLiveUnitAudioChunk)
        XCTAssertEqual(silero.chunkDuration, Double(NFKMLXSileroVADConfiguration.v6.numSamples) / 16000)
        XCTAssertEqual(silero.sampleRate, Double(NFKMLXSileroVADConfiguration.v6.sampleRate))
        XCTAssertEqual(silero.channelCount, 1)
        XCTAssertTrue(silero.isStateful)
    }

    func testTheSpeechEnhancersDeclareTheirStep() throws {
        let steps: [(identifier: String, chunk: Double, rate: Double, lookahead: Double)] = [
            ("gtcrn", Double(NFKMLXGTCRNConfiguration().hopSize) / 16000, 16000, 32),
            ("deepfilternet3", Double(NFKMLXDeepFilterNetConfiguration().hopSize) / 48000, 48000, 40),
            ("frcrn", Double(NFKMLXFRCRNConfiguration().decodeStride) / 16000, 16000, 1000),
        ]
        for step in steps {
            let descriptor = try XCTUnwrap(descriptors.first { $0.identifier == step.identifier }, step.identifier)
            let live = try XCTUnwrap(descriptor.liveProfile, step.identifier)
            XCTAssertEqual(live.unit, NFKModelLiveUnitAudioChunk, step.identifier)
            XCTAssertEqual(live.inputKeys, [NFKInputAudio], step.identifier)
            XCTAssertEqual(live.outputKeys, [NFKOutputAudio], step.identifier)
            XCTAssertTrue(live.isStateful, step.identifier)
            XCTAssertEqual(live.chunkDuration, step.chunk, accuracy: 1e-12, step.identifier)
            XCTAssertEqual(live.sampleRate, step.rate, step.identifier)
            XCTAssertEqual(live.channelCount, 1, step.identifier)
            XCTAssertEqual(live.lookaheadMilliseconds, step.lookahead, step.identifier)
            XCTAssertTrue(descriptor.outputs.contains { $0.key == NFKOutputAudio }, step.identifier)
        }
    }

    func testTheStreamingRecognizerDeclaresItsChunk() throws {
        let configuration = NFKMLXNemotronASRConfiguration.english06B
        let descriptor = try XCTUnwrap(descriptors.first { $0.identifier == NFKMLXNemotronASR.modelName })
        let live = try XCTUnwrap(descriptor.liveProfile)
        let frames = configuration.chunkFrames(lookahead: configuration.defaultLookahead)
        XCTAssertEqual(live.unit, NFKModelLiveUnitAudioChunk)
        XCTAssertEqual(live.outputKeys, [NFKOutputText, NFKOutputSegments])
        XCTAssertTrue(live.isStateful)
        XCTAssertEqual(live.chunkDuration, Double(frames * configuration.hopSamples) / 16000, accuracy: 1e-12)
        // A sample just inside a chunk's first window waits for the chunk's last frame to be complete.
        let waited = configuration.hopSamples * (frames - 1) + configuration.fftSize / 2 - 1 + configuration.windowSamples / 2 - 1
        XCTAssertEqual(live.lookaheadMilliseconds, (Double(waited) / 16).rounded(.up))
        XCTAssertEqual(descriptor.option(withKey: NFKMLXGenerationParameterKey.lookaheadFrames)?.defaultValue as? Int,
                       configuration.defaultLookahead)
    }

    func testOnlyTheDeclaredModelsAreLive() {
        let live = Set(descriptors.filter { $0.liveProfile != nil }.map(\.identifier))
        XCTAssertEqual(live, Set(NFKMLXCatalogLive.videoFrameOutputs.keys).union(NFKMLXCatalogLive.statefulVideoFrameOutputs.keys)
            .union(NFKMLXCatalogLive.audioChunks.keys))
        XCTAssertNil(descriptors.first { $0.identifier == "birefnet" }?.liveProfile, "a seconds-per-frame matter is not live")
    }
}
