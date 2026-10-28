//
//  NFKMLXDetectionParameterKeyTests.swift
//  InferKitMLXTests
//
//  A request's thresholds and output length reach the detectors, the voice activity detectors, TrOCR,
//  and Sa2VA, and the catalog declares each as an option. The detectors run on random weights, where a
//  threshold of 0 passes every candidate and one above 1 passes none.
//

import XCTest
import CoreGraphics
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXDetectionParameterKeyTests: XCTestCase {

    override func tearDown() {
        if NFKMLXGPU.metalLibraryURL != nil {
            NFKMLXGPU.clearCache()
        }
        super.tearDown()
    }

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil, "no Metal library for MLX; run Tools/mlx-metallib.sh")
    }

    private static let confidence = NFKMLXDetectionParameterKey.confidenceThreshold
    private static let overlap = NFKMLXDetectionParameterKey.overlapThreshold

    /// A small gray plate with a lighter square, enough for a random network to score.
    private static func plate(side: Int = 96) -> CGImage {
        var rgba = [UInt8](repeating: 90, count: side * side * 4)
        for y in side / 4 ..< side / 2 {
            for x in side / 4 ..< side / 2 {
                for channel in 0 ..< 3 { rgba[(y * side + x) * 4 + channel] = 200 }
            }
        }
        let provider = CGDataProvider(data: Data(rgba) as CFData)!
        return CGImage(width: side, height: side, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: side * 4,
                       space: CGColorSpaceCreateDeviceRGB(),
                       bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
    }

    private func detections(_ backend: any NFKInferenceBackend, _ parameters: [String: Any]) throws -> [NFKDetection] {
        let request = NFKInferenceRequest(inputs: [NFKInputImage: Self.plate()], parameters: parameters)
        return try XCTUnwrap(try backend.runInference(for: request).output(forKey: NFKOutputDetections) as? [NFKDetection])
    }

    /// Nothing passes a threshold above 1, and a threshold at the median score of what passes 0 keeps the
    /// more confident part, each at or above it. Random weights put the best of 80 class scores above 0.5
    /// almost everywhere, so a fixed 0.5 would keep everything.
    private func assertTheConfidenceThresholdFilters(_ backend: any NFKInferenceBackend,
                                                     file: StaticString = #filePath, line: UInt = #line) throws {
        let all = try detections(backend, [Self.confidence: 0])
        let scores = all.map(\.confidence).sorted()
        let median = scores.isEmpty ? 0.5 : scores[scores.count / 2]
        let confident = try detections(backend, [Self.confidence: median])
        XCTAssertGreaterThan(all.count, confident.count, "\(backend.backendIdentifier) at \(median)", file: file, line: line)
        XCTAssertTrue(confident.allSatisfy { $0.confidence >= Double(Float(median)) - 1e-6 }, backend.backendIdentifier,
                      file: file, line: line)
        XCTAssertEqual(try detections(backend, [Self.confidence: 1.01]).count, 0, backend.backendIdentifier,
                       file: file, line: line)
    }

    // MARK: Detectors

    func testTheYOLOThresholdsChangeWhatIsReturned() throws {
        try requireMLXRuntime()
        NFKMLXRandom.seed(31)
        let backend = try NFKMLXYOLO.backend(weightsURL: nil, labels: nil)
        XCTAssertEqual(backend.supportedParameterKeys, [Self.confidence, Self.overlap])
        try assertTheConfidenceThresholdFilters(backend)
        let unsuppressed = try detections(backend, [Self.confidence: 0, Self.overlap: 1])
        let suppressed = try detections(backend, [Self.confidence: 0, Self.overlap: 0])
        XCTAssertGreaterThan(unsuppressed.count, suppressed.count, "a lower overlap threshold suppresses more boxes")
    }

    func testTheGenerationsDeclareTheOverlapOnlyWhereTheySuppress() {
        XCTAssertEqual(NFKMLXYOLOGenerationBackend.parameterKeys(endToEnd: false), [Self.confidence, Self.overlap])
        XCTAssertEqual(NFKMLXYOLOGenerationBackend.parameterKeys(endToEnd: true), [Self.confidence])
    }

    func testTheRetinaFaceThresholdsChangeWhatIsReturned() throws {
        try requireMLXRuntime()
        NFKMLXRandom.seed(32)
        let backend = try NFKMLXRetinaFace.backend(weightsURL: nil)
        XCTAssertEqual(backend.supportedParameterKeys, [Self.confidence, Self.overlap])
        try assertTheConfidenceThresholdFilters(backend)
        let unsuppressed = try detections(backend, [Self.confidence: 0, Self.overlap: 1])
        let suppressed = try detections(backend, [Self.confidence: 0, Self.overlap: 0])
        XCTAssertGreaterThan(unsuppressed.count, suppressed.count, "a lower overlap threshold suppresses more boxes")
    }

    func testTheDETRConfidenceThresholdChangesWhatIsReturned() throws {
        try requireMLXRuntime()
        NFKMLXRandom.seed(33)
        let backends: [any NFKInferenceBackend] = [
            try NFKMLXRTDetr.backend(variant: .r18vd, weightsURL: nil, labels: nil),
            try NFKMLXRFDetr.backend(variant: .nano, weightsURL: nil, labels: nil),
            try NFKMLXRFDetrSegmentation.backend(variant: .nano, weightsURL: nil, labels: nil),
        ]
        for backend in backends {
            XCTAssertEqual(backend.supportedParameterKeys, [Self.confidence], backend.backendIdentifier)
            try assertTheConfidenceThresholdFilters(backend)
        }
    }

    // MARK: Voice activity

    /// A one-second 16 kHz WAV of a tone, written to a temporary file.
    private func toneClip() throws -> NFKAudioAsset {
        let samples = (0 ..< 16000).map { Float(sin(Double($0) * 2 * .pi * 220 / 16000) * 0.3) }
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("vad-\(UUID().uuidString).wav")
        try NFKMLXWaveFile.write(samples: samples, sampleRate: 16000, to: url)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return NFKAudioAsset(fileURL: url, durationSeconds: 1, sampleRate: 16000, channelCount: 1)
    }

    func testTheSpeechThresholdChangesWhatCountsAsSpeech() throws {
        try requireMLXRuntime()
        NFKMLXRandom.seed(34)
        let clip = try toneClip()
        let key = NFKMLXVoiceActivityParameterKey.speechThreshold
        for backend in [try NFKMLXSileroVAD.backend(weightsURL: nil), try NFKMLXVAD.backend(weightsURL: nil)] {
            XCTAssertEqual(backend.supportedParameterKeys, [key], backend.backendIdentifier)
            func spans(_ threshold: Double) throws -> [NFKAudioSegment] {
                let request = NFKInferenceRequest(inputs: [NFKInputAudio: clip], parameters: [key: threshold])
                return try XCTUnwrap(try backend.runInference(for: request).output(forKey: NFKOutputSegments) as? [NFKAudioSegment])
            }
            let everything = try spans(0)
            XCTAssertEqual(everything.count, 1, "every frame is speech at 0: \(backend.backendIdentifier)")
            XCTAssertEqual(everything.first?.endSeconds ?? 0, 1, accuracy: 0.05, backend.backendIdentifier)
            XCTAssertEqual(try spans(1.01).count, 0, "no frame is speech above 1: \(backend.backendIdentifier)")
        }
    }

    // MARK: Output length

    func testARequestSetsTheOutputLength() {
        func request(_ parameters: [String: Any]) -> NFKInferenceRequest {
            NFKInferenceRequest(inputs: [NFKInputPrompt: "hello"], parameters: parameters)
        }
        XCTAssertEqual(NFKMLXTrOCR.maximumTokens(for: request([:])), NFKMLXTrOCR.maximumTokens)
        XCTAssertEqual(NFKMLXTrOCR.maximumTokens(for: request([NFKParameterMaxTokens: 200])), 200)
        XCTAssertEqual(NFKMLXTrOCR.maximumTokens(for: request([NFKParameterMaxTokens: 0])), 1)
        XCTAssertEqual(NFKMLXSa2VA.maximumTokens(for: request([:])), NFKMLXSa2VA.maximumTokens)
        XCTAssertEqual(NFKMLXSa2VA.maximumTokens(for: request([NFKParameterMaxTokens: 1024])), 1024)
    }

    // MARK: Catalog

    private func parameterKeys(_ name: String) throws -> Set<String> {
        let descriptor = try XCTUnwrap(NFKModelCatalog.shared.descriptor(withIdentifier: name), name)
        return Set(descriptor.options.filter { $0.stage == NFKModelOptionStageRequest }.map(\.key))
    }

    func testTheCatalogDeclaresEachThresholdAtItsModelsDefault() throws {
        NFKMLXReferenceModels.registerAll()
        for name in ["yolo", "yolov8x", "yolov9c", "yolo11n", "yolo12x", NFKMLXRetinaFace.modelName] {
            XCTAssertTrue(try parameterKeys(name).isSuperset(of: [Self.confidence, Self.overlap]), name)
        }
        for name in ["yolov10n", "yolo26n", "rtdetr", "rtdetr-v2-r18vd", "rf-detr", "rf-detr-seg",
                     "table-transformer-detection"] {
            let keys = try parameterKeys(name)
            XCTAssertTrue(keys.contains(Self.confidence), name)
            XCTAssertFalse(keys.contains(Self.overlap), "\(name) suppresses nothing")
        }
        for name in [NFKMLXSileroVAD.modelName, NFKMLXVAD.modelName] {
            XCTAssertTrue(try parameterKeys(name).contains(NFKMLXVoiceActivityParameterKey.speechThreshold), name)
        }
        for name in ["trocr-base-printed", "sa2va-1b"] {
            XCTAssertTrue(try parameterKeys(name).contains(NFKParameterMaxTokens), name)
        }
        let yolo = try XCTUnwrap(NFKModelCatalog.shared.descriptor(withIdentifier: "yolo"))
        XCTAssertEqual(yolo.option(withKey: Self.confidence)?.defaultValue as? Double, 0.25)
        XCTAssertEqual(yolo.option(withKey: Self.overlap)?.defaultValue as? Double, 0.7)
        let face = try XCTUnwrap(NFKModelCatalog.shared.descriptor(withIdentifier: NFKMLXRetinaFace.modelName))
        XCTAssertEqual(face.option(withKey: Self.confidence)?.defaultValue as? Double, 0.8)
        let trocr = try XCTUnwrap(NFKModelCatalog.shared.descriptor(withIdentifier: "trocr-base-printed"))
        XCTAssertEqual(trocr.option(withKey: NFKParameterMaxTokens)?.defaultValue as? Int, 64)
    }
}
