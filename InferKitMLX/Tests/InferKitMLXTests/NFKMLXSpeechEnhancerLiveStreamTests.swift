//
//  NFKMLXSpeechEnhancerLiveStreamTests.swift
//  InferKitMLXTests
//
//  The speech enhancers on a live source: a stream answers each sample as the clip forward of every sample
//  so far answers it.
//

import AVFoundation
import XCTest
import InferKit
import MLX
import MLXNN
@testable import InferKitMLX

final class NFKMLXSpeechEnhancerLiveStreamTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    /// Every trainable parameter drawn uniformly from ±`scale`, so each recurrence and convolution history
    /// moves the output.
    static func randomize(_ module: Module, scale: Float) {
        let drawn = module.trainableParameters().mapValues { MLXRandom.uniform(low: -scale, high: scale, $0.shape) }
        module.update(parameters: drawn)
        eval(module)
    }

    /// A voiced signal with noise whose level steps, `count` samples at `sampleRate`.
    static func signal(count: Int, sampleRate: Int) -> [Float] {
        var seed: UInt64 = 0x9E3779B97F4A7C15
        return (0 ..< count).map { index in
            seed = seed &* 6364136223846793005 &+ 1442695040888963407
            let noise = Float(Int64(bitPattern: seed >> 11) % 2000) / 1000 - 1
            let time = Float(index) / Float(sampleRate)
            let voiced = 0.4 * sinf(2 * .pi * 180 * time) + 0.2 * sinf(2 * .pi * 720 * time + 0.3)
            let level: Float = (index / (sampleRate / 4)) % 2 == 0 ? 0.05 : 0.3
            return voiced * (0.5 + 0.5 * sinf(2 * .pi * 3 * time)) + level * noise
        }
    }

    /// `samples` cut at uneven unit lengths, some shorter than a hop and some spanning several frames.
    static func units(_ samples: [Float], lengths: [Int]) -> [[Float]] {
        var units: [[Float]] = []
        var start = 0
        var index = 0
        while start < samples.count {
            let end = min(start + lengths[index % lengths.count], samples.count)
            units.append(Array(samples[start ..< end]))
            start = end
            index += 1
        }
        return units
    }

    /// The clip forward's own summation-order spread, the floor a stream is held to. A stream runs the
    /// clip's operations on the frames each unit completes, and a matrix product's or an FFT's rounding
    /// depends on how many rows it runs (MLX's single-row CPU `addmm` takes another path, and the GPU's
    /// kernels tile by row count), so a stream and its clip differ by summation order alone. The floor is
    /// the largest gap the clip itself shows between two orders of the same sums: the clip on the other
    /// device, and the clip of each prefix, over the samples a stream of that prefix has answered.
    static func clipFloor(_ samples: [Float], prefixes: [Int], answered: (Int) -> Int,
                          clip: ([Float]) -> [Float]) -> Float {
        let whole = clip(samples)
        let otherDevice: Device = Device.defaultDevice().deviceType == .cpu ? .gpu : .cpu
        let reordered = Device.withDefaultDevice(otherDevice) { clip(samples) }
        var floor = gap(Array(reordered.prefix(answered(samples.count))), whole).difference
        for prefix in prefixes {
            let count = answered(prefix)
            guard count > 0 else {
                continue
            }
            floor = max(floor, gap(Array(clip(Array(samples[..<prefix])).prefix(count)), whole).difference)
        }
        return floor
    }

    /// Every `stride`-th cumulative unit boundary of `units`.
    static func boundaries(_ units: [[Float]], every stride: Int) -> [Int] {
        var total = 0
        var result: [Int] = []
        for (index, unit) in units.enumerated() {
            total += unit.count
            if index % stride == stride - 1 {
                result.append(total)
            }
        }
        return result
    }

    /// The largest absolute difference over the shared prefix, and the largest absolute value of `reference`.
    static func gap(_ streamed: [Float], _ reference: [Float]) -> (difference: Float, peak: Float) {
        let count = min(streamed.count, reference.count)
        var difference: Float = 0
        var peak: Float = 0
        for index in 0 ..< count {
            difference = max(difference, abs(streamed[index] - reference[index]))
            peak = max(peak, abs(reference[index]))
        }
        return (difference, peak)
    }

    // MARK: GTCRN

    private static let gtcrnUnits = [160, 333, 512, 1000, 7, 2048, 256]

    /// GTCRN's stream against its clip forward: the samples it answers, a whole clip pushed as one unit,
    /// uneven units, and a control that forgets the state at every unit.
    private func assertGTCRNStreams(_ net: NFKMLXGTCRN, samples: [Float], label: String) {
        let config = NFKMLXGTCRNConfiguration()
        let clip: ([Float]) -> [Float] = { input in
            let enhanced = NFKMLXGTCRNBackend.enhance(input, net: net, config: config)
            eval(enhanced)
            return enhanced.reshaped([enhanced.size]).asArray(Float.self)
        }
        let reference = clip(samples)
        let units = Self.units(samples, lengths: Self.gtcrnUnits)
        let answered = { (count: Int) in max(0, (count / config.hopSize - 1) * config.hopSize) }
        let floor = Self.clipFloor(samples, prefixes: Self.boundaries(units, every: 6), answered: answered, clip: clip)

        let whole = NFKMLXGTCRNBackend.stream(samples, net: net, state: NFKGTCRNStreamState(config))
        let state = NFKGTCRNStreamState(config)
        let streamed = units.flatMap { NFKMLXGTCRNBackend.stream($0, net: net, state: state) }
        let forgetful = units.flatMap { NFKMLXGTCRNBackend.stream($0, net: net, state: NFKGTCRNStreamState(config)) }
        let (wholeGap, peak) = Self.gap(whole, reference)
        let streamGap = Self.gap(streamed, reference).difference
        print("VALIDATION GTCRN stream \(label): clip floor \(floor), whole-clip unit \(wholeGap), uneven units \(streamGap), "
              + "forgetful control \(Self.gap(forgetful, reference).difference), peak \(peak)")
        XCTAssertEqual(streamed.count, answered(samples.count), "the stream answers one hop behind")
        XCTAssertEqual(whole.count, answered(samples.count))
        XCTAssertLessThanOrEqual(wholeGap, floor, "one unit answers the clip within the clip's own order spread")
        XCTAssertLessThanOrEqual(streamGap, floor, "uneven units answer the clip within the clip's own order spread")
        XCTAssertGreaterThan(Self.gap(forgetful, reference).difference, 1000 * max(floor, 1e-7),
                             "the control discriminates: a stream that forgets its state misses the clip")
    }

    func testAGTCRNStreamAnswersEachSampleAsTheClipDoes() throws {
        try requireMLXRuntime()
        NFKMLXRandom.seed(11)
        let net = NFKMLXGTCRNFactory.makeNet()
        Self.randomize(net, scale: 0.3)
        assertGTCRNStreams(net, samples: Self.signal(count: 16000 * 2 + 77, sampleRate: 16000), label: "random")
    }

    func testAGTCRNStreamOnTheCPU() throws {
        try requireMLXRuntime()
        NFKMLXRandom.seed(11)
        let net = NFKMLXGTCRNFactory.makeNet()
        Self.randomize(net, scale: 0.3)
        Device.withDefaultDevice(.cpu) {
            assertGTCRNStreams(net, samples: Self.signal(count: 16000 + 77, sampleRate: 16000), label: "random-cpu")
        }
    }

    func testADeepFilterNetStreamOnTheCPU() throws {
        try requireMLXRuntime()
        NFKMLXRandom.seed(13)
        let net = NFKMLXDeepFilterNetFactory.makeNet()
        Self.randomize(net, scale: 0.2)
        net.update(parameters: ModuleParameters.unflattened(["erb_fb": abs(net.erbFB), "erb_inv_fb": abs(net.erbInvFB)]))
        Device.withDefaultDevice(.cpu) {
            assertDeepFilterNetStreams(net, samples: Self.signal(count: 24000 + 333, sampleRate: 48000), label: "random-cpu")
        }
    }

    func testTheReleasedGTCRNStreamsAsItsClip() throws {
        try requireMLXRuntime()
        let environment = NFKMLXValidationConfig.environment
        guard let weightsPath = environment["IK_VAL_GTCRN"], let recordPath = environment["IK_PARITY_GTCRN"] else {
            throw XCTSkip("set IK_VAL_GTCRN and IK_PARITY_GTCRN to stream the released weights")
        }
        let net = NFKMLXGTCRNFactory.makeNet()
        try NFKMLXGTCRNFactory.loadWeights(into: net, from: URL(fileURLWithPath: weightsPath))
        let record = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: recordPath)).arrays
        assertGTCRNStreams(net, samples: try XCTUnwrap(record["waveform"]).asArray(Float.self), label: "released")
    }

    func testAGTCRNStreamAnswersBuffersAndRefusesAnotherRate() throws {
        try requireMLXRuntime()
        NFKMLXRandom.seed(11)
        let net = NFKMLXGTCRNFactory.makeNet()
        Self.randomize(net, scale: 0.3)
        let backend = NFKMLXGTCRNBackend(net: net, config: NFKMLXGTCRNConfiguration(), identifier: "gtcrn-live-test")
        let stream = try backend.openLiveStream(parameters: nil)
        let samples = Self.signal(count: 4096, sampleRate: 16000)
        let first = try XCTUnwrap(stream.result(for: Self.request(Array(samples[..<200]), sampleRate: 16000))
            .output(forKey: NFKOutputAudio) as? AVAudioPCMBuffer)
        XCTAssertEqual(first.frameLength, 0, "a unit short of the first frames answers an empty buffer")
        let rest = try XCTUnwrap(stream.result(for: Self.request(Array(samples[200...]), sampleRate: 16000))
            .output(forKey: NFKOutputAudio) as? AVAudioPCMBuffer)
        XCTAssertEqual(Int(rest.frameLength), 4096 - 256)
        XCTAssertEqual(rest.format.sampleRate, 16000)
        XCTAssertThrowsError(try stream.result(for: Self.request([0, 0], sampleRate: 48000))) {
            XCTAssertEqual(($0 as NSError).code, NFKInferenceError.error_InferenceUnsupported.rawValue)
        }
        stream.reset()
        let again = try XCTUnwrap(stream.result(for: Self.request(samples, sampleRate: 16000))
            .output(forKey: NFKOutputAudio) as? AVAudioPCMBuffer)
        XCTAssertEqual(Int(again.frameLength), 4096 - 256, "reset starts the next unit as a new stream")
        let restored = Array(UnsafeBufferPointer(start: rest.floatChannelData![0], count: Int(rest.frameLength)))
        let replayed = Array(UnsafeBufferPointer(start: again.floatChannelData![0], count: Int(again.frameLength)))
        XCTAssertLessThan(Self.gap(replayed, restored).difference, 1e-5)
    }

    static func request(_ samples: [Float], sampleRate: Int) -> NFKInferenceRequest {
        NFKInferenceRequest(inputs: [NFKInputAudio: NFKMLXLiveAudio.buffer(samples, sampleRate: sampleRate)])
    }

    // MARK: DeepFilterNet3

    private static let deepFilterNetUnits = [480, 1000, 7, 4800, 960, 333, 2400]

    private func assertDeepFilterNetStreams(_ net: NFKMLXDeepFilterNet, samples: [Float], label: String) {
        let dsp = NFKMLXDeepFilterNetDSP(config: net.config, erbFB: net.erbFB)
        let hop = net.config.hopSize
        let clip: ([Float]) -> [Float] = { NFKMLXDeepFilterNetBackend.enhance($0, net: net, dsp: dsp) }
        let reference = clip(samples)
        let units = Self.units(samples, lengths: Self.deepFilterNetUnits)
        let answered = { (count: Int) in max(0, (count / hop - 3) * hop) }
        let floor = Self.clipFloor(samples, prefixes: Self.boundaries(units, every: 4), answered: answered, clip: clip)

        let whole = NFKMLXDeepFilterNetBackend.stream(samples, net: net, dsp: dsp, state: NFKDFStreamState(dsp: dsp, net: net))
        let state = NFKDFStreamState(dsp: dsp, net: net)
        let streamed = units.flatMap { NFKMLXDeepFilterNetBackend.stream($0, net: net, dsp: dsp, state: state) }
        var forgetful: [Float] = []
        for unit in units {
            forgetful += NFKMLXDeepFilterNetBackend.stream(unit, net: net, dsp: dsp, state: NFKDFStreamState(dsp: dsp, net: net))
        }
        let (wholeGap, peak) = Self.gap(whole, reference)
        let streamGap = Self.gap(streamed, reference).difference
        let controlGap = Self.gap(forgetful, reference).difference
        print("VALIDATION DeepFilterNet3 stream \(label): clip floor \(floor), whole-clip unit \(wholeGap), uneven units \(streamGap), "
              + "forgetful control \(controlGap), peak \(peak)")
        XCTAssertEqual(streamed.count, answered(samples.count), "the stream answers three hops behind")
        XCTAssertEqual(wholeGap, 0, "one unit runs the clip's operations on the clip's frames")
        XCTAssertLessThanOrEqual(streamGap, floor, "uneven units answer the clip within the clip's own order spread")
        XCTAssertGreaterThan(controlGap, 1000 * max(floor, 1e-7),
                             "the control discriminates: a stream that forgets its state misses the clip")
    }

    func testADeepFilterNetStreamAnswersEachSampleAsTheClipDoes() throws {
        try requireMLXRuntime()
        NFKMLXRandom.seed(13)
        let net = NFKMLXDeepFilterNetFactory.makeNet()
        Self.randomize(net, scale: 0.2)
        net.update(parameters: ModuleParameters.unflattened(["erb_fb": abs(net.erbFB), "erb_inv_fb": abs(net.erbInvFB)]))
        assertDeepFilterNetStreams(net, samples: Self.signal(count: 48000 + 333, sampleRate: 48000), label: "random")
    }

    func testTheReleasedDeepFilterNetStreamsAsItsClip() throws {
        try requireMLXRuntime()
        let environment = NFKMLXValidationConfig.environment
        guard let weightsPath = environment["IK_VAL_DEEPFILTERNET"], let recordPath = environment["IK_PARITY_DEEPFILTERNET"] else {
            throw XCTSkip("set IK_VAL_DEEPFILTERNET and IK_PARITY_DEEPFILTERNET to stream the released weights")
        }
        let net = NFKMLXDeepFilterNetFactory.makeNet()
        try NFKMLXDeepFilterNetFactory.loadWeights(into: net, from: URL(fileURLWithPath: weightsPath))
        let record = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: recordPath)).arrays
        assertDeepFilterNetStreams(net, samples: try XCTUnwrap(record["waveform"]).asArray(Float.self), label: "released")
    }

    // MARK: FRCRN

    private func assertFRCRNStreams(_ net: NFKMLXFRCRNNet, samples: [Float], label: String) {
        let configuration = net.configuration
        let reference = NFKMLXClearerVoiceDecoding.stitched(samples, window: configuration.decodeWindow,
                                                            stride: configuration.decodeStride) { window in
            let waveform = NFKMLXFRCRNBackend.inference(window, net: net)
            eval(waveform)
            return waveform.reshaped([-1]).asArray(Float.self)
        }
        let state = NFKFRCRNStreamState()
        let streamed = Self.units(samples, lengths: [160, 12000, 7, 4000, 30000, 1600])
            .flatMap { NFKMLXFRCRNBackend.stream($0, net: net, state: state) }
        let giveUp = (configuration.decodeWindow - configuration.decodeStride) / 2
        let windows = (samples.count - configuration.decodeWindow) / configuration.decodeStride + 1
        let (streamGap, peak) = Self.gap(streamed, reference)
        print("VALIDATION FRCRN stream \(label): windowed decode gap \(streamGap), peak \(peak), samples \(streamed.count)")
        XCTAssertEqual(streamed.count, (windows - 1) * configuration.decodeStride + configuration.decodeWindow - giveUp)
        XCTAssertEqual(streamGap, 0, "each window runs the windowed decode's operations on the same samples")
    }

    func testAnFRCRNStreamAnswersEachWindowAsTheWindowedDecodeDoes() throws {
        try requireMLXRuntime()
        NFKMLXRandom.seed(17)
        let net = NFKMLXFRCRN.makeNet()
        Self.randomize(net, scale: 0.1)
        assertFRCRNStreams(net, samples: Self.signal(count: 16000 * 4 + 1234, sampleRate: 16000), label: "random")
    }

    func testTheReleasedFRCRNStreamsAsItsWindowedDecode() throws {
        try requireMLXRuntime()
        let environment = NFKMLXValidationConfig.environment
        guard let checkpoint = environment["IK_VAL_FRCRN"], let recordPath = environment["IK_PARITY_FRCRN"] else {
            throw XCTSkip("set IK_VAL_FRCRN and IK_PARITY_FRCRN to stream the released weights")
        }
        let net = NFKMLXFRCRN.makeNet()
        try NFKMLXFRCRN.loadWeights(into: net, from: URL(fileURLWithPath: checkpoint))
        let record = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: recordPath)).arrays
        let samples = try XCTUnwrap(record["long_waveform"] ?? record["waveform"]).asArray(Float.self)
        assertFRCRNStreams(net, samples: samples, label: "released")
    }
}
