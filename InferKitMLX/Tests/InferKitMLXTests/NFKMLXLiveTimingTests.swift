//
//  NFKMLXLiveTimingTests.swift
//  InferKitMLXTests
//
//  The measured time per unit of every live model in the package's catalog, through the live stream a
//  client opens: a video frame at 1280 × 720 and 1920 × 1080, an audio chunk of the profile's duration.
//  Set IK_TIME_LIVE to the path of the JSON report to write; each model reads its released weights from
//  its validation key. Runs only in a release build, under the Large Model Coordination lock, on a quiet
//  machine.
//

import AVFoundation
import CoreVideo
import XCTest
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXLiveTimingTests: XCTestCase {

    /// The validation key of each live model's released weights, by catalog identifier.
    static let weightKeys: [String: String] = [
        "modnet": "IK_VAL_MODNET",
        "u2netp": "IK_VAL_U2NETP",
        "bisenet": "IK_VAL_BISENET",
        "bisenet-v2": "IK_VAL_BISENETV2",
        "segformer-b0": "IK_VAL_SEGFORMER",
        "deeplabv3": "IK_VAL_DEEPLAB",
        "depth-anything-v2-small": "IK_VAL_DEPTH",
        "depth-anything-3-small": "IK_VAL_DEPTH3_WEIGHTS",
        "pose-simplebaseline": "IK_VAL_POSE",
        "retinaface-mobile025": "IK_VAL_RETINAFACE",
        "rf-detr-nano": "IK_VAL_RFDETR_NANO",
        "rf-detr-small": "IK_VAL_RFDETR_SMALL",
        "rf-detr-seg-small": "IK_VAL_RF_DETR_SEG_SMALL",
        "rtdetr-r18vd": "IK_VAL_RTDETR_R18VD",
        "real-esrgan-general-x4v3": "IK_VAL_REALESRGAN_GENERAL_V3",
        "swinir-lightweight-x2": "IK_VAL_SWINIR_LIGHT_X2",
        "swinir-lightweight-x3": "IK_VAL_SWINIR_LIGHT_X3",
        "swinir-lightweight-x4": "IK_VAL_SWINIR_LIGHT_X4",
        "robust-video-matting": "IK_VAL_RVM",
        "robust-video-matting-resnet50": "IK_VAL_RVM_RESNET50",
        "silero-vad": "IK_VAL_SILERO_VAD",
        "gtcrn": "IK_VAL_GTCRN",
        "deepfilternet3": "IK_VAL_DEEPFILTERNET",
        "frcrn": "IK_VAL_FRCRN",
        "nemotron-asr-streaming": "IK_VAL_NEMOTRON_ASR_STREAMING",
    ]

    /// The file a model's registry factory reads inside the folder its validation key names, where the
    /// factory takes a weights file rather than the folder.
    static let weightFiles: [String: String] = [
        "rf-detr-nano": "model.safetensors",
        "rf-detr-small": "model.safetensors",
        "rtdetr-r18vd": "model.safetensors",
    ]

    /// The frame sizes a video model is timed at.
    static let frameSizes = [(1280, 720), (1920, 1080)]

    /// The units timed after the warm-up units, and the most time one model's timed units may take at one
    /// size. A unit slower than `slowUnitSeconds` warms up once and is timed until the budget passes,
    /// at least once.
    static let warmUpUnits = 3
    static let timedUnits = 30
    static let timeBudgetSeconds = 60.0
    static let slowUnitSeconds = 5.0

    override func setUp() {
        super.setUp()
        NFKMLXReferenceModels.registerAll()
    }

    override func tearDown() {
        NFKMLXGPU.clearCache()
        super.tearDown()
    }

    #if DEBUG
    static let isDebugBuild = true
    #else
    static let isDebugBuild = false
    #endif

    private static func setting(_ key: String) -> String? {
        NFKMLXValidationConfig.environment[key].flatMap { $0.isEmpty ? nil : $0 }
    }

    func testEveryLiveModelIsTimedThroughItsStream() throws {
        guard let reportPath = Self.setting("IK_TIME_LIVE") else {
            throw XCTSkip("set IK_TIME_LIVE to the report's path to time the live models")
        }
        try XCTSkipIf(Self.isDebugBuild, "time the live models in a release build (swift test -c release -Xswiftc "
                      + "-enable-testing); a debug build's frame conversion alone takes seconds")
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil, "no Metal library for MLX; run Tools/mlx-metallib.sh")
        let chip = NFKHardwareProfile.current.chipName
        let day = ISO8601DateFormatter.string(from: Date(), timeZone: .current, formatOptions: [.withFullDate])
        var rows: [[String: Any]] = []
        var failures: [String] = []
        let descriptors = NFKMLXModelDescriptors.modelDescriptors()
            .filter { $0.liveProfile != nil }
            .sorted { $0.identifier < $1.identifier }
        XCTAssertEqual(Set(descriptors.map(\.identifier)), Set(Self.weightKeys.keys), "every live model has a weight key")
        for descriptor in descriptors {
            let profile = try XCTUnwrap(descriptor.liveProfile)
            let variant = descriptor.defaultVariant
            guard let key = Self.weightKeys[descriptor.identifier], let path = Self.setting(key) else {
                rows.append(["model": descriptor.identifier, "skipped": "no local weights"])
                print("LIVE-TIMING \(descriptor.identifier): no local weights")
                continue
            }
            let weights = Self.weightFiles[descriptor.identifier].map { (path as NSString).appendingPathComponent($0) } ?? path
            guard FileManager.default.fileExists(atPath: weights) else {
                rows.append(["model": descriptor.identifier, "skipped": "\(key) names a path that does not exist"])
                print("LIVE-TIMING \(descriptor.identifier): \(key) names a path that does not exist")
                continue
            }
            do {
                try Self.time(descriptor, profile: profile, variant: variant, weights: weights, chip: chip, day: day, into: &rows)
            } catch {
                rows.append(["model": descriptor.identifier, "failed": "\(error)"])
                failures.append("\(descriptor.identifier): \(error)")
                print("LIVE-TIMING \(descriptor.identifier): failed: \(error)")
            }
            NFKMLXGPU.clearCache()
        }
        let report = try JSONSerialization.data(withJSONObject: rows, options: [.prettyPrinted, .sortedKeys])
        try report.write(to: URL(fileURLWithPath: reportPath))
        XCTAssertTrue(failures.isEmpty, "every live model with local weights is timed: \(failures)")
    }

    /// Times one model through its stream: a video model at each frame size, an audio model per chunk.
    private static func time(_ descriptor: NFKModelDescriptor, profile: NFKModelLiveProfile, variant: NFKModelVariant?,
                             weights path: String, chip: String, day: String, into rows: inout [[String: Any]]) throws {
        let backend = try NFKMLXModelRegistry.backend(named: variant?.registryName ?? descriptor.identifier,
                                                      weightsURL: URL(fileURLWithPath: path))
        if profile.unit == NFKModelLiveUnitVideoFrame {
            for (width, height) in Self.frameSizes {
                let frame = try Self.frame(width: width, height: height)
                let (milliseconds, units) = try Self.median(backend: backend, unit: { _ in
                    NFKInferenceRequest(inputs: [NFKInputImage: frame])
                })
                rows.append(["model": descriptor.identifier, "variant": variant?.identifier ?? NSNull(), "chip": chip,
                             "width": width, "height": height, "milliseconds": milliseconds, "units": units, "measured": day])
                print("LIVE-TIMING \(descriptor.identifier) \(width)x\(height): \(milliseconds) ms per frame")
            }
        } else {
            let sampleRate = Int(profile.sampleRate)
            let count = Int((profile.chunkDuration * profile.sampleRate).rounded())
            let signal = NFKMLXSpeechEnhancerLiveStreamTests.signal(
                count: count * (Self.warmUpUnits + Self.timedUnits + 4), sampleRate: sampleRate)
            let (milliseconds, units) = try Self.median(backend: backend, unit: { index in
                let start = (index * count) % (signal.count - count)
                return NFKInferenceRequest(inputs: [NFKInputAudio: Self.buffer(signal[start ..< start + count],
                                                                                sampleRate: sampleRate)])
            })
            let realTimeFactor = milliseconds / (profile.chunkDuration * 1000)
            rows.append(["model": descriptor.identifier, "variant": variant?.identifier ?? NSNull(), "chip": chip,
                         "milliseconds": milliseconds, "realTimeFactor": realTimeFactor, "units": units, "measured": day])
            print("LIVE-TIMING \(descriptor.identifier): \(milliseconds) ms per \(profile.chunkDuration) s chunk, "
                  + "real-time factor \(realTimeFactor)")
        }
    }

    /// The median wall time of `resultForUnit:` over the timed units, after the warm-up units, through the
    /// stream `NFKInferenceOpenLiveStream` opens, and the count of units timed. Units stop once they pass
    /// the time budget; a unit slower than `slowUnitSeconds` warms up once and is timed at least once.
    private static func median(backend: any NFKInferenceBackend,
                               unit: (Int) -> NFKInferenceRequest) throws -> (milliseconds: Double, units: Int) {
        var error: NSError?
        guard let stream = NFKInferenceOpenLiveStream(backend, nil, &error) else {
            throw error ?? NSError(domain: NFKInferenceErrorDomain, code: NFKInferenceError.error_InferenceBackendFailure.rawValue)
        }
        defer { stream.close() }
        func timed(_ index: Int) throws -> Double {
            let request = unit(index)
            let before = DispatchTime.now().uptimeNanoseconds
            _ = try stream.result(for: request)
            return Double(DispatchTime.now().uptimeNanoseconds - before) / 1e6
        }
        let slow = try timed(0) > slowUnitSeconds * 1000
        let warmUps = slow ? 1 : warmUpUnits
        for index in 1 ..< warmUps {
            _ = try timed(index)
        }
        var times: [Double] = []
        let started = Date()
        for index in 0 ..< timedUnits {
            times.append(try timed(warmUps + index))
            if Date().timeIntervalSince(started) > timeBudgetSeconds, times.count >= (slow ? 1 : 5) {
                break
            }
        }
        times.sort()
        let median = times.count % 2 == 1 ? times[times.count / 2] : (times[times.count / 2 - 1] + times[times.count / 2]) / 2
        return (median, times.count)
    }

    /// A BGRA camera-sized frame: a gradient with a moving bright square, as a capture device hands one.
    private static func frame(width: Int, height: Int) throws -> CVPixelBuffer {
        var buffer: CVPixelBuffer?
        let attributes = [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary
        guard CVPixelBufferCreate(nil, width, height, kCVPixelFormatType_32BGRA, attributes, &buffer) == kCVReturnSuccess,
              let buffer else {
            throw XCTSkip("no \(width)x\(height) pixel buffer")
        }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        let rowBytes = CVPixelBufferGetBytesPerRow(buffer)
        let pixels = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: UInt8.self)
        for y in 0 ..< height {
            for x in 0 ..< width {
                let inSquare = abs(x - width / 2) < width / 6 && abs(y - height / 2) < height / 4
                let offset = y * rowBytes + x * 4
                pixels[offset] = UInt8((x * 255) / width)
                pixels[offset + 1] = inSquare ? 230 : UInt8((y * 255) / height)
                pixels[offset + 2] = inSquare ? 210 : 90
                pixels[offset + 3] = 255
            }
        }
        return buffer
    }

    private static func buffer(_ samples: ArraySlice<Float>, sampleRate: Int) -> AVAudioPCMBuffer {
        NFKMLXLiveAudio.buffer(Array(samples), sampleRate: sampleRate)
    }
}
