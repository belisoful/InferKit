//
//  NFKMLXNemotronASRTests.swift
//  InferKitMLXTests
//
//  Nemotron ASR Streaming: the cache-aware recognizer's stream against its clip forward under the
//  chunked mask, and the released weights against the reference.
//

import AVFoundation
import XCTest
import InferKit
import MLX
import MLXNN
import MLXOptimizers
@testable import InferKitMLX

final class NFKMLXNemotronASRTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil, "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    private func maxGap(_ a: MLXArray, _ b: MLXArray) -> Float { abs(a - b).max().item(Float.self) }

    /// A tiny random recognizer whose every parameter moves its output.
    private func tinyNet() -> NFKMLXNemotronASRNet {
        NFKMLXRandom.seed(21)
        let net = NFKMLXNemotronASRNet(.tiny)
        NFKMLXSpeechEnhancerLiveStreamTests.randomize(net, scale: 0.3)
        return net
    }

    func testTheGeometryFollowsTheRelease() {
        let configuration = NFKMLXNemotronASRConfiguration.english06B
        XCTAssertEqual(configuration.lookaheads.map { configuration.firstChunkFrames(lookahead: $0) }, [105, 49, 9, 1])
        XCTAssertEqual(configuration.lookaheads.map { configuration.chunkFrames(lookahead: $0) }, [112, 56, 16, 8])
        XCTAssertEqual(configuration.encoderFrameSeconds, 0.08, accuracy: 1e-12)
        let net = NFKMLXNemotronASRNet(.english06B)
        XCTAssertEqual(net.encoder.subsampling.linear.weight.shape, [1024, 256 * 17], "the causal subsampler keeps 17 of 128 bins")
    }

    func testTheChunkedMaskShowsEachQueryItsChunkAndTheLeftContext() {
        let encoder = NFKNemotronASREncoder(.tiny)                       // left context 6
        let mask = encoder.chunkedMask(frames: 9, lookahead: 2).asArray(Bool.self)
        func visible(_ query: Int, _ key: Int) -> Bool { mask[query * 9 + key] }
        XCTAssertTrue(visible(0, 2), "a query sees the rest of its own chunk")
        XCTAssertFalse(visible(0, 3), "and nothing after it")
        XCTAssertTrue(visible(8, 0), "two chunks of three back are the six left frames")
        XCTAssertFalse(visible(5, 6), "nor the next chunk")
    }

    /// The front end, the encoder, and the transducer, streamed in uneven units, against the clip forward
    /// over the frames the stream completed, held to the clip's own CPU/GPU summation-order spread.
    func testAStreamAnswersAsTheClipForwardUnderTheChunkedMask() throws {
        try requireMLXRuntime()
        let net = tinyNet()
        let configuration = net.configuration
        let samples = NFKMLXSpeechEnhancerLiveStreamTests.signal(count: 16000 * 2, sampleRate: 16000)
        for lookahead in configuration.lookaheads {
            let state = NFKNemotronASRStreamState(configuration, lookahead: lookahead)
            var chunks: [MLXArray] = []
            var encoded: [MLXArray] = []
            var tokens: [NFKMLXNemotronASRNet.Token] = []
            for unit in NFKMLXSpeechEnhancerLiveStreamTests.units(samples, lengths: [160, 1000, 7, 2400, 333, 4000]) {
                state.append(unit, frontEnd: net.frontEnd)
                while let chunk = state.takeChunk() {
                    let frames = state.carry.streaming { net.encoder(chunk, lookahead: nil) }
                    state.carry.evaluate([frames])
                    chunks.append(chunk)
                    encoded.append(frames)
                    tokens += net.decode(encoded: frames, state: state.decoding)
                }
            }
            let streamedFeatures = concatenated(chunks, axis: 1)
            let streamedFrames = concatenated(encoded, axis: 1)
            let count = streamedFeatures.dim(1)
            XCTAssertEqual(count, configuration.firstChunkFrames(lookahead: lookahead)
                           + (encoded.count - 1) * configuration.chunkFrames(lookahead: lookahead))
            XCTAssertEqual(streamedFrames.dim(1), encoded.count * (lookahead + 1))

            let clip: () -> (MLXArray, MLXArray) = {
                let features = net.frontEnd.features(samples)[0..., 0 ..< count, 0...]
                let frames = net.encode(features: features, lookahead: lookahead)
                eval(features, frames)
                return (features, frames)
            }
            let (features, frames) = clip()
            let (otherFeatures, otherFrames) = Device.withDefaultDevice(.cpu) { clip() }
            // The clip's own summation-order spread, the widest gap between orders of the same sums: the
            // clip on the other device, and on each device the clip of every whole-chunk prefix and of the
            // features with a chunk more, each over the frames it shares, which runs the same operations on
            // other row counts. The chunked mask makes the shared frames the same sums in each.
            var featureFloor = maxGap(features, otherFeatures)
            var frameFloor = maxGap(frames, otherFrames)
            let first = configuration.firstChunkFrames(lookahead: lookahead), later = configuration.chunkFrames(lookahead: lookahead)
            let longer = net.frontEnd.features(samples)
            for device in [Device.gpu, Device.cpu] {
                Device.withDefaultDevice(device) {
                    var lengths = (1 ..< encoded.count).map { first + ($0 - 1) * later }
                    if longer.dim(1) >= count + later {
                        lengths.append(count + later)
                    }
                    for length in lengths {
                        let variant = net.encode(features: longer[0..., 0 ..< length, 0...], lookahead: lookahead)
                        let shared = min(variant.dim(1), frames.dim(1))
                        frameFloor = max(frameFloor, maxGap(variant[0..., 0 ..< shared, 0...], frames[0..., 0 ..< shared, 0...]))
                    }
                    let batched = net.encode(features: concatenated([features, features], axis: 0), lookahead: lookahead)
                    frameFloor = max(frameFloor, maxGap(batched[0 ..< 1], frames))
                    for prefix in [samples.count / 3, samples.count / 2, 2 * samples.count / 3] {
                        let variant = net.frontEnd.features(Array(samples[..<prefix]))
                        let shared = min(variant.dim(1) - 2, count)
                        featureFloor = max(featureFloor, maxGap(variant[0..., 0 ..< shared, 0...], features[0..., 0 ..< shared, 0...]))
                    }
                }
            }
            let featureGap = maxGap(streamedFeatures, features), frameGap = maxGap(streamedFrames, frames)
            let clipTokens = net.decode(encoded: frames, state: NFKMLXNemotronASRDecodingState())
            print("VALIDATION nemotron-asr stream lookahead \(lookahead): features \(featureGap) (floor \(featureFloor)), "
                  + "encoder \(frameGap) (floor \(frameFloor)), tokens \(tokens.count) vs clip \(clipTokens.count)")
            XCTAssertLessThanOrEqual(featureGap, featureFloor, "lookahead \(lookahead): the stream's frames are the clip's")
            XCTAssertLessThanOrEqual(frameGap, frameFloor, "lookahead \(lookahead): the cached encoder answers as the masked clip")
            XCTAssertEqual(tokens, clipTokens, "lookahead \(lookahead): the carried transducer state decodes the clip's tokens")

            let wholeMaskless = net.encoder(net.frontEnd.features(samples)[0..., 0 ..< count, 0...], lookahead: 1_000_000)
            XCTAssertGreaterThan(maxGap(wholeMaskless, frames), 1000 * max(frameFloor, 1e-7),
                                 "the control discriminates: an unlimited context misses the chunked forward")
        }
    }

    func testTheBackendsStreamReturnsTextAndRefusesAnotherRate() throws {
        try requireMLXRuntime()
        let net = tinyNet()
        let pieces = (0 ..< net.configuration.vocabulary).map { "\u{2581}t\($0)" }
        let backend = NFKMLXNemotronASRBackend(net: net, tokenizer: NFKMLXNemotronASRTokenizer(pieces: pieces, specialIds: [net.configuration.blank]),
                                               identifier: "nemotron-live-test")
        let stream = try backend.openLiveStream(parameters: [NFKMLXGenerationParameterKey.lookaheadFrames: 1])
        let samples = NFKMLXSpeechEnhancerLiveStreamTests.signal(count: 16000, sampleRate: 16000)
        var text = ""
        var segments: [NFKAudioSegment] = []
        for unit in NFKMLXSpeechEnhancerLiveStreamTests.units(samples, lengths: [1600]) {
            let result = try stream.result(for: NFKMLXSpeechEnhancerLiveStreamTests.request(unit, sampleRate: 16000))
            text += try XCTUnwrap(result.text)
            segments += try XCTUnwrap(result.segments)
        }
        XCTAssertEqual(segments.compactMap(\.label).joined(), text, "the units' texts join into the transcript")
        XCTAssertEqual(segments.map(\.startSeconds), segments.map(\.startSeconds).sorted())
        XCTAssertThrowsError(try stream.result(for: NFKMLXSpeechEnhancerLiveStreamTests.request([0, 0], sampleRate: 48000))) {
            XCTAssertEqual(($0 as NSError).code, NFKInferenceError.error_InferenceUnsupported.rawValue)
        }
        XCTAssertThrowsError(try backend.openLiveStream(parameters: [NFKMLXGenerationParameterKey.lookaheadFrames: 5])) {
            XCTAssertEqual(($0 as NSError).code, NFKInferenceError.error_InferenceUnsupported.rawValue)
        }
    }

    func testTheTokenizerDecodesPiecesAndByteFallbacks() {
        let tokenizer = NFKMLXNemotronASRTokenizer(pieces: ["<unk>", "\u{2581}Hello", ",", "\u{2581}w", "orld", "<0xC3>", "<0xA9>", "<blank>"],
                                                   specialIds: [7])
        XCTAssertEqual(tokenizer.text(for: [1, 2, 3, 4, 7]), "Hello, world")
        XCTAssertEqual(tokenizer.text(for: [3, 5, 6]), "wé")
        XCTAssertEqual(tokenizer.text(for: [3, 4], atStart: false), " world", "a later unit keeps its word's space")
    }

    // MARK: Training

    /// The objective against NeMo's `RNNTLoss` and `RNNTLossPytorch` on fixed logits
    /// (`run_reference.py nemotron_asr_streaming_loss`): each utterance's plain negative log-likelihood,
    /// and the release's loss (FastEmit at 0.005, `mean_volume`) and its gradient with respect to the logits.
    func testTheTransducerLossMatchesTheReference() throws {
        try requireMLXRuntime()
        guard let recordPath = NFKMLXValidationConfig.environment["IK_PARITY_NEMOTRON_ASR_STREAMING_LOSS"] else {
            throw XCTSkip("set IK_PARITY_NEMOTRON_ASR_STREAMING_LOSS (run_reference.py nemotron_asr_streaming_loss)")
        }
        let record = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: recordPath)).arrays
        let logits = try XCTUnwrap(record["logits"]), labels = try XCTUnwrap(record["labels"])
        let frames = try XCTUnwrap(record["frames"]).asArray(Int.self)
        let counts = try XCTUnwrap(record["label_lengths"]).asArray(Int.self)
        let blank = logits.dim(3) - 1
        func utterances(_ batch: MLXArray) -> [(logits: MLXArray, labels: MLXArray)] {
            (0 ..< frames.count).map { (batch[$0, 0 ..< frames[$0], 0 ..< counts[$0] + 1], labels[$0, 0 ..< counts[$0]]) }
        }
        let plain = NFKMLXNemotronASRObjective(blank: blank, fastEmitLambda: 0, reduction: .meanBatch)
        let nll = utterances(logits).map { plain.negativeLogLikelihood(logits: $0.logits, labels: $0.labels).item(Float.self) }
        let reference = try XCTUnwrap(record["nll_pytorch"]).asArray(Float.self)
        let release = NFKMLXNemotronASRObjective(blank: blank)
        let loss = release.loss(utterances(logits)).item(Float.self)
        let referenceLoss = try XCTUnwrap(record["loss"]).item(Float.self)
        let gradient = grad { (batch: MLXArray) in release.loss(utterances(batch)) }(logits)
        let referenceGradient = try XCTUnwrap(record["grad"])
        let gradientGap = maxGap(gradient, referenceGradient)
        print("VALIDATION PARITY nemotron-asr loss: plain \(nll) vs \(reference); release \(loss) vs \(referenceLoss); "
              + "gradient max |d| \(gradientGap), peak \(abs(referenceGradient).max().item(Float.self))")
        for (ours, theirs) in zip(nll, reference) {
            XCTAssertEqual(ours, theirs, accuracy: 4e-5 * theirs)
        }
        XCTAssertEqual(loss, referenceLoss, accuracy: 4e-5 * referenceLoss)
        XCTAssertLessThan(gradientGap, 1e-5)
    }

    func testTheReleaseScheduleWarmsUpAsNoamAnnealingDoes() {
        let schedule = NFKMLXNemotronASR.referenceSchedule
        let scale = 1 / sqrtf(1024)
        XCTAssertEqual(schedule.multiplier(0), scale * powf(8000, -1.5), accuracy: 1e-12, "the counter reads at least 1")
        XCTAssertEqual(schedule.multiplier(1), scale * powf(8000, -1.5), accuracy: 1e-12)
        XCTAssertEqual(schedule.multiplier(8000), scale / sqrtf(8000), accuracy: 1e-9)
        XCTAssertEqual(schedule.multiplier(100_000_000), scale / sqrtf(100_000_000), accuracy: 1e-12)
        XCTAssertEqual(schedule.multiplier(100_000_000_000_000), 1e-6 / 3, accuracy: 1e-12, "the floor holds once the decay passes it")
    }

    func testAFineTuneLowersTheLossAndMovesOnlyTheTransducer() throws {
        try requireMLXRuntime()
        let net = tinyNet()
        let samples = NFKMLXSpeechEnhancerLiveStreamTests.signal(count: 16000, sampleRate: 16000)
        let labels = [3, 7, 1, 9, 4]
        // Copies: an optimizer update replaces a parameter's contents in place.
        let encoderBefore = net.encoder.parameters().flattened().map { $0.1 + 0 }
        let headBefore = net.joint.head.weight + 0
        eval(encoderBefore + [headBefore])
        let losses = try NFKMLXNemotronASR.fineTune(net, examples: { _ in [(samples, labels)] }, trainable: .transducer,
                                                     optimizer: Adam(learningRate: 3e-3), steps: 12)
        XCTAssertLessThan(losses.last!, losses.first!, "the loss falls")
        let encoderAfter = net.encoder.parameters().flattened().map { $0.1 }
        XCTAssertTrue(zip(encoderBefore, encoderAfter).allSatisfy { maxGap($0, $1) == 0 }, "the frozen encoder does not move")
        XCTAssertGreaterThan(maxGap(headBefore, net.joint.head.weight), 0, "the joint head moves")
    }

    func testAFineTunedCheckpointReloadsThroughTheNetworkBuilder() throws {
        try requireMLXRuntime()
        let net = tinyNet()
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("nemotron-asr-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let checkpoint = directory.appendingPathComponent("fine-tuned.safetensors")
        try NFKMLXWeights.save(net, to: checkpoint)
        let reloaded = NFKMLXNemotronASRNet(.tiny)
        try NFKMLXNemotronASR.loadWeights(into: reloaded, from: checkpoint)
        let features = net.frontEnd.features(NFKMLXSpeechEnhancerLiveStreamTests.signal(count: 8000, sampleRate: 16000))
        let ours = net.encode(features: features, lookahead: 1), theirs = reloaded.encode(features: features, lookahead: 1)
        XCTAssertEqual(maxGap(ours, theirs), 0)
        XCTAssertEqual(net.recognize(Array(repeating: 0.1, count: 8000), lookahead: 1),
                       reloaded.recognize(Array(repeating: 0.1, count: 8000), lookahead: 1))
    }

    // MARK: Released weights

    private func cosine(_ a: MLXArray, _ b: MLXArray) -> Double {
        let x = a.asType(.float32).reshaped([-1]), y = b.asType(.float32).reshaped([-1])
        let dot = (x * y).sum().item(Double.self)
        return dot / (sqrt((x * x).sum().item(Double.self)) * sqrt((y * y).sum().item(Double.self)))
    }

    /// The released recognizer against transformers' `NemotronAsrStreamingForRNNT` on the validation
    /// clip (`run_reference.py nemotron_asr_streaming`): the features, the subsampler, the first layer,
    /// the chunk-masked encoder and the greedy tokens at every trained lookahead, and the cached stream
    /// over the reference's own chunks, held to the reference's end-to-end order floor and its float64 runs.
    func testTheReleasedRecognizerMatchesTheReference() throws {
        try requireMLXRuntime()
        let environment = NFKMLXValidationConfig.environment
        guard let directory = environment["IK_VAL_NEMOTRON_ASR_STREAMING"],
              let recordPath = environment["IK_PARITY_NEMOTRON_ASR_STREAMING_ORDERS"] else {
            throw XCTSkip("set IK_VAL_NEMOTRON_ASR_STREAMING and IK_PARITY_NEMOTRON_ASR_STREAMING_ORDERS "
                          + "(IK_PROBE_OUTPUT_ORDER_FLOOR=1 run_reference.py nemotron_asr_streaming)")
        }
        let release = URL(fileURLWithPath: directory)
        let configuration = try NFKMLXNemotronASR.configuration(directoryURL: release)
        let net = NFKMLXNemotronASRNet(configuration)
        try NFKMLXNemotronASR.loadWeights(into: net, from: release.appendingPathComponent("model.safetensors"))
        let tokenizer = try NFKMLXNemotronASRTokenizer(tokenizerURL: release.appendingPathComponent("tokenizer.json"))
        let record = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: recordPath)).arrays
        let waveform = try XCTUnwrap(record["waveform"]).asArray(Float.self)

        let features = net.frontEnd.features(waveform)
        let referenceFeatures = try XCTUnwrap(record["features"])
        XCTAssertEqual(features.dim(1), referenceFeatures.dim(0))
        let featureCosine = cosine(features[0], referenceFeatures)
        print("VALIDATION PARITY nemotron-asr features cosine \(featureCosine), max |d| \(maxGap(features[0], referenceFeatures))")
        XCTAssertGreaterThan(featureCosine, 0.9999999)

        let defaultLookahead = configuration.defaultLookahead
        let defaultCount = try XCTUnwrap(record["count_R\(defaultLookahead)"]).item(Int.self)
        let subsampled = net.encoder.subsampling(referenceFeatures[0 ..< defaultCount].expandedDimensions(axis: 0))
        print("VALIDATION PARITY nemotron-asr subsampled cosine \(cosine(subsampled[0], try XCTUnwrap(record["subsampled"])))")
        XCTAssertGreaterThan(cosine(subsampled[0], try XCTUnwrap(record["subsampled"])), 0.9999999)

        for lookahead in configuration.lookaheads {
            let count = try XCTUnwrap(record["count_R\(lookahead)"]).item(Int.self)
            let input = referenceFeatures[0 ..< count].expandedDimensions(axis: 0)
            let encoded = net.encode(features: input, lookahead: lookahead)
            let reference = try XCTUnwrap(record["encoded_R\(lookahead)"])
            let referenceStream = try XCTUnwrap(record["streamed_R\(lookahead)"])
            let encoderCosine = cosine(encoded[0], reference)

            let first = configuration.firstChunkFrames(lookahead: lookahead), later = configuration.chunkFrames(lookahead: lookahead)
            func stream() -> MLXArray {
                let carry = NFKMLXStreamCarry()
                var streamed: [MLXArray] = []
                var start = 0
                while start < count {
                    let size = start == 0 ? first : later
                    let frames = carry.streaming { net.encoder(input[0..., start ..< start + size, 0...], lookahead: nil) }
                    carry.evaluate([frames])
                    streamed.append(frames)
                    start += size
                }
                return concatenated(streamed, axis: 1)
            }
            let ours = stream()
            // MLX's CPU device, printed and not asserted: at R = 13 it sits past the reference's farthest run
            // from float64, an open item the model's entry records.
            let (cpuStream, cpuClip) = Device.withDefaultDevice(.cpu) { () -> (MLXArray, MLXArray) in
                let clip = net.encode(features: input, lookahead: lookahead)
                let streamed = stream()
                eval(clip, streamed)
                return (streamed, clip)
            }
            let referenceGap = maxGap(referenceStream, reference)
            let streamGap = maxGap(ours[0], encoded[0])

            // Every float32 run is held to the reference's farthest float32 run from its float64 answer, and
            // the stream-to-clip gap, two orders of the same sums, to the farthest of the reference's clip
            // reruns in the fixed summation orders. At float64 the reference's stream and clip are one
            // computation. Every bar is read from the record.
            let orderRuns = record.filter { $0.key.hasPrefix("encoded_R\(lookahead).order.") }
            XCTAssertFalse(orderRuns.isEmpty, "R=\(lookahead): the record carries the reference's order reruns")
            let orderDistances = orderRuns.map { (name: $0.key, gap: maxGap($0.value, reference)) }.sorted { $0.gap > $1.gap }
            let orderFloor = orderDistances.first?.gap ?? 0
            let truth = try XCTUnwrap(record["truth_R\(lookahead)"])
            let truthStream = try XCTUnwrap(record["truth_streamed_R\(lookahead)"])
            let truthGap = try XCTUnwrap(record["truth_gap_R\(lookahead)"]).item(Float.self)
            let truthFloor = (Array(orderRuns.values) + [reference]).map { maxGap($0, truth) }.max().map { max($0, maxGap(referenceStream, truthStream)) } ?? 0
            let clipFromTruth = maxGap(encoded[0], truth), streamFromTruth = maxGap(ours[0], truthStream)
            print("VALIDATION PARITY nemotron-asr R=\(lookahead) floor: stream vs clip \(streamGap) against order floor \(orderFloor) "
                  + "(\(orderDistances.first?.name ?? "none")), reference's own \(referenceGap); float64 stream vs clip \(truthGap); "
                  + "from float64: clip \(clipFromTruth), stream \(streamFromTruth), the reference's farthest run \(truthFloor), "
                  + "reference clip \(maxGap(reference, truth)), reference stream \(maxGap(referenceStream, truthStream)); "
                  + "CPU: stream vs clip \(maxGap(cpuStream[0], cpuClip[0])), from float64 clip \(maxGap(cpuClip[0], truth)) "
                  + "(\(maxGap(cpuClip[0], truth) / truthFloor)x), stream \(maxGap(cpuStream[0], truthStream)) "
                  + "(\(maxGap(cpuStream[0], truthStream) / truthFloor)x); "
                  + "the reference stream from ours \(maxGap(ours[0], referenceStream)); "
                  + "every order: \(orderDistances.map { "\($0.name.split(separator: ".").dropFirst(2).joined(separator: ".")) \($0.gap)" }.joined(separator: ", "))")
            XCTAssertLessThanOrEqual(clipFromTruth, truthFloor, "R=\(lookahead): the clip sits as near float64 as the reference's own runs")
            XCTAssertLessThanOrEqual(streamFromTruth, truthFloor, "R=\(lookahead): the stream sits as near float64 as the reference's own runs")
            XCTAssertLessThanOrEqual(streamGap, orderFloor, "R=\(lookahead): the stream sits within the reference's end-to-end order floor")
            XCTAssertLessThan(truthGap, 1e-12, "R=\(lookahead): at float64 the reference's stream and clip are one computation")

            let tokens = net.decode(encoded: encoded, state: NFKMLXNemotronASRDecodingState())
            let referenceTokens = try XCTUnwrap(record["tokens_R\(lookahead)"]).asArray(Int.self)
            let referenceFrames = try XCTUnwrap(record["frames_R\(lookahead)"]).asArray(Int.self)
            let text = String(decoding: try XCTUnwrap(record["text_R\(lookahead)"]).asArray(Int.self).map(UInt8.init), as: UTF8.self)
            print("VALIDATION PARITY nemotron-asr R=\(lookahead): encoder cosine \(encoderCosine), max |d| \(maxGap(encoded[0], reference)); "
                  + "stream cosine \(cosine(ours[0], referenceStream)); "
                  + "tokens \(tokens.count) vs \(referenceTokens.count): \(tokenizer.text(for: tokens.map(\.id)))")
            XCTAssertGreaterThan(encoderCosine, 0.9999999, "R=\(lookahead)")
            XCTAssertEqual(tokens.map(\.id), referenceTokens, "R=\(lookahead)")
            XCTAssertEqual(tokens.map(\.frame), referenceFrames, "R=\(lookahead)")
            XCTAssertEqual(tokenizer.text(for: tokens.map(\.id)), text, "R=\(lookahead)")
            if lookahead == defaultLookahead {
                let pooler = net.encoderProjector(encoded)[0]
                XCTAssertGreaterThan(cosine(pooler, try XCTUnwrap(record["pooler"])), 0.9999999)
                let layer0 = net.encoder.layers[0](subsampled, positions: { net.encoder.positionTable(keys: $0) },
                                                   mask: net.encoder.chunkedMask(frames: subsampled.dim(1), lookahead: lookahead))
                let layer0Cosine = cosine(layer0[0], try XCTUnwrap(record["layer0"]))
                print("VALIDATION PARITY nemotron-asr layer0 cosine \(layer0Cosine)")
                XCTAssertGreaterThan(layer0Cosine, 0.9999999)
            }
        }

        let (prediction, _) = net.decoder.step(token: configuration.blank, state: nil)
        XCTAssertGreaterThan(cosine(prediction[0], try XCTUnwrap(record["prediction0"])), 0.9999999)
        let joint = net.joint(encoder: net.encoderProjector(net.encode(features: referenceFeatures[0 ..< 1 + 8 * defaultLookahead]
            .expandedDimensions(axis: 0), lookahead: defaultLookahead))[0, 0 ..< 1], prediction: prediction)
        print("VALIDATION PARITY nemotron-asr joint0 cosine \(cosine(joint[0], try XCTUnwrap(record["joint0"])))")
    }
}
