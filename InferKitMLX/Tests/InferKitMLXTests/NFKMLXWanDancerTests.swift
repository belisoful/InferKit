//
//  NFKMLXWanDancerTests.swift
//  InferKitMLXTests
//
//  Wan-Dancer-14B: the music features held to librosa 0.10.2 as `get_music_base_feature` calls it, seam by
//  seam, where the validation store holds the record.
//

import XCTest
import InferKit
import MLX
import MLXNN
@testable import InferKitMLX

final class NFKMLXWanDancerTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    private func cosine(_ x: [Double], _ y: [Double]) -> Double {
        precondition(x.count == y.count, "lengths \(x.count) and \(y.count)")
        var dot = 0.0, xx = 0.0, yy = 0.0
        for i in x.indices {
            dot += x[i] * y[i]
            xx += x[i] * x[i]
            yy += y[i] * y[i]
        }
        return dot / (xx.squareRoot() * yy.squareRoot())
    }

    private func doubles(_ array: MLXArray) -> [Double] { array.asType(.float32).asArray(Float.self).map(Double.init) }
    private func integers(_ array: MLXArray) -> [Int] { array.asType(.int32).asArray(Int32.self).map(Int.init) }

    private func musicRecord() throws -> [String: MLXArray] {
        guard let path = NFKMLXValidationConfig.environment["IK_PARITY_WAN_DANCER_MUSIC"] else {
            throw XCTSkip("set IK_PARITY_WAN_DANCER_MUSIC (run_reference.py wan_dancer_music)")
        }
        return try MLX.loadArrays(url: URL(fileURLWithPath: path))
    }

    func testTheResamplerFollowsSciPy() {
        let x = (0 ..< 1000).map { sin(Double($0) * 0.37) + 0.3 * cos(Double($0) * 1.9) }
        let halved = NFKMLXWanDancerMusic.resamplePoly(x, up: 1, down: 2)
        XCTAssertEqual(halved.count, 500)
        let raised = NFKMLXWanDancerMusic.resamplePoly(x, up: 735, down: 512)
        XCTAssertEqual(raised.count, 1000 * 735 / 512 + 1)
    }

    func testTheMusicFeaturesFollowLibrosa() throws {
        try requireMLXRuntime()
        let record = try musicRecord()
        let samples = record["samples"]!.asArray(Float.self)
        let data = NFKMLXWanDancerMusic.resampled(samples.map(Double.init), from: 15360, to: 15360)
        let music = NFKMLXWanDancerMusic.self
        let power = music.powerSpectrogram(data, fftSize: 2048, hop: 512)
        let decibels = music.powerToDecibels(music.melSpectrogram(power, rate: 22050, fftSize: 2048, bands: 128))
        let envelope = music.onsetStrength(decibels)
        let mfcc = music.cepstrum(decibels, count: 20)
        let tuning = music.estimatedTuning(data, rate: 22050, binsPerOctave: 36)
        let cqt = music.constantQ(data, rate: 22050, hop: 512, bins: 252, binsPerOctave: 36,
                                  tuning: doubles(record["poly.tuning"]!)[0])
        let chroma = music.chromaCENS(data, rate: 22050, hop: 512)
        let peaks = music.onsetPeaks(envelope, rate: 22050, hop: 512)
        let tempoData = music.resampled(samples.map(Double.init), from: 15360, to: 22050)
        let startTempo = music.tempo(music.onsetStrength(music.powerToDecibels(music.melSpectrogram(
            music.powerSpectrogram(tempoData, fftSize: 2048, hop: 512), rate: 22050, fftSize: 2048, bands: 128))),
                                     rate: 22050, hop: 512, startTempo: 120)
        let beats = music.beatTrack(envelope, rate: 22050, hop: 512, startTempo: startTempo, tightness: 100)

        let figures: [(String, Double)] = [
            ("mel dB", cosine(decibels.flatMap { $0 }, doubles(record["poly.mel_db"]!))),
            ("envelope", cosine(envelope, doubles(record["poly.envelope"]!))),
            ("mfcc", cosine(mfcc.flatMap { $0 }, doubles(record["poly.mfcc"]!))),
            ("cqt", cosine((0 ..< cqt[0].count).flatMap { t in cqt.map { $0[t] } },
                           doubles(record["poly.cqt"]!.transposed(1, 0)))),
            ("chroma", cosine(chroma.flatMap { $0 }, doubles(record["poly.chroma"]!))),
            ("tempo samples", cosine(tempoData, doubles(record["poly.tempo_data"]!))),
        ]
        for (name, figure) in figures {
            print(String(format: "wan dancer music %@: %.16f", name, figure))
        }
        print("wan dancer music tuning \(tuning) vs \(doubles(record["poly.tuning"]!)[0]), start tempo \(startTempo) vs \(doubles(record["poly.start_bpm"]!)[0])")
        let recordedPeaks = integers(record["poly.peaks"]!), recordedBeats = integers(record["poly.beats"]!)
        print("wan dancer music peaks \(Set(peaks).symmetricDifference(recordedPeaks).count)/\(recordedPeaks.count) differ, beats \(Set(beats).symmetricDifference(recordedBeats).count)/\(recordedBeats.count) differ")
        for (name, figure) in figures {
            XCTAssertGreaterThan(figure, 0.99999, name)
        }
        XCTAssertEqual(tuning, doubles(record["poly.tuning"]!)[0], accuracy: 1e-6)
        XCTAssertEqual(peaks, recordedPeaks)
        XCTAssertEqual(beats, recordedBeats)

        let features = music.features(samples: samples, sampleRate: 15360)
        let reference = record["ref.features"]!
        XCTAssertEqual(features.shape, reference.shape)
        let rows = features.dim(0)
        for (name, columns) in [("envelope", 0 ..< 1), ("mfcc", 1 ..< 21), ("chroma", 21 ..< 33), ("onsets", 33 ..< 34),
                                ("beats", 34 ..< 35)] {
            let figure = cosine(doubles(features[0..., columns].reshaped([rows * columns.count])),
                                doubles(reference[0..., columns].reshaped([rows * columns.count])))
            print(String(format: "wan dancer music reference %@: %.16f", name, figure))
        }
    }

    // MARK: The pipeline

    private func cosine(_ a: MLXArray, _ b: MLXArray) -> Double {
        cosine(doubles(a.reshaped([-1])), doubles(b.reshaped([-1])))
    }

    private func testDev() throws -> URL {
        guard let path = NFKMLXValidationConfig.environment["IK_VAL_WAN_DANCER_TESTDEV"] else {
            throw XCTSkip("set IK_VAL_WAN_DANCER_TESTDEV (Tools/wan-dancer-testdev/make_testdev.py)")
        }
        return URL(fileURLWithPath: path)
    }

    private func pipelineRecord() throws -> (URL, [String: MLXArray]) {
        let directory = try testDev()
        guard let path = NFKMLXValidationConfig.environment["IK_PARITY_WAN_DANCER_PIPELINE"] else {
            throw XCTSkip("set IK_PARITY_WAN_DANCER_PIPELINE (run_reference.py wan_dancer_pipeline --checkpoint <testdev>)")
        }
        return (directory, try MLX.loadArrays(url: URL(fileURLWithPath: path)))
    }

    /// The oracle's prompt, and each stage's as the released scripts complete it.
    private let prompt = "A dancer in a red dress turns on a bright stage, full body, steady camera."

    private func stagePrompt(_ stage: String, _ record: [String: MLXArray]) -> String {
        stage == "global" ? prompt + String(format: "帧率是%.4f", Double(record["global.fps"]!.item(Float.self)))
            : prompt + ", 帧率是30fps。"
    }

    private func pipeline(_ directory: URL, stage: String) throws -> NFKMLXWanDancerPipeline {
        let net = NFKMLXWanDancerNet(try NFKMLXWanDancerConfiguration.reading(directory.appendingPathComponent("config.json")))
        try NFKMLXWanDancerNet.loadWeights(into: net, file: directory.appendingPathComponent("\(stage)_model.safetensors"))
        let autoencoder = try NFKMLXWanOriginalRelease.autoencoder(
            from: directory.appendingPathComponent(NFKMLXWanOriginalRelease.wan21AutoencoderFile))
        let clip = try NFKMLXWanAnimateImageEncoderNet.fromOpenCLIP(
            directory.appendingPathComponent("models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"))
        let pipeline = NFKMLXWanDancerPipeline(net: net, vae: autoencoder.net, latentsMean: autoencoder.mean,
                                               latentsStd: autoencoder.std, imageEncoder: clip)
        pipeline.steps = 3
        return pipeline
    }

    func testThePipelineFollowsTheReference() throws {
        try requireMLXRuntime()
        let (directory, record) = try pipelineRecord()
        let encoder = try NFKMLXWanOriginalRelease.textEncoder(
            from: directory.appendingPathComponent(NFKMLXWanOriginalRelease.textEncoderFile))
        let segmenter = try NFKMLXWanFunRelease.segmenter(directory)
        func features(_ text: String) -> MLXArray {
            NFKMLXWanFunRelease.features(NFKMLXWanPromptCleaning.cleaned(text), encoder: encoder, segmenter: segmenter)
        }
        let negative = features(NFKMLXWanDancerGenerator.defaultNegativePrompt)
        let reference = record["reference_image"]!.asType(.float32) / 255
        for stage in ["global", "local"] {
            let pipeline = try self.pipeline(directory, stage: stage)
            let positive = features(stagePrompt(stage, record))
            let mask = record["\(stage).mask"]!.asArray(Int32.self).map { $0 == 1 }
            let fps = stage == "global" ? Double(record["global.fps"]!.item(Float.self)) : 30
            let clip = NFKMLXWanDancerClip(keyframes: record["\(stage).keyframes"]!.asType(.float32) / 255, keyframeMask: mask,
                                           reference: reference, music: record["\(stage).music"]!, fps: fps)
            let inputs = pipeline.inputs(clip)
            var figures: [(String, Double)] = [
                ("prompt features", cosine(positive, record["\(stage).context"]!)),
                ("keyframe clip", cosine(inputs.keyframeFeatures, record["\(stage).clip_feature"]!)),
                ("reference clip", cosine(inputs.referenceFeatures, record["\(stage).refimage_feature"]!)),
                ("condition", cosine(inputs.condition, record["\(stage).y"]!)),
            ]
            let noise = record["\(stage).noise"]!
            let frames = noise.dim(1)
            figures.append(("music tokens", cosine(pipeline.net.musicTokens(inputs.music, frames: frames, dtype: .float32),
                                                   record["\(stage).music_tokens"]!)))
            let t = Double(NFKMLXWanDancerPipeline.sigmas(steps: 3, shift: 5)[0] * 1000)
            figures.append(("call 0", cosine(pipeline.net(noise, text: positive, t: t, inputs: inputs), record["\(stage).call0"]!)))
            figures.append(("call 1", cosine(pipeline.net(noise, text: negative, t: t, inputs: inputs, skipsBlock: true),
                                             record["\(stage).call1"]!)))
            var steps = [MLXArray]()
            pipeline.stepObserver = { steps.append($1) }
            let latents = pipeline.denoise(noise, positive: positive, negative: negative, inputs: inputs)
            for (index, step) in steps.enumerated() {
                figures.append(("step \(index)", cosine(step, record["\(stage).steps"]![index])))
            }
            let video = stage == "global" ? pipeline.decodedFramewise(latents) : pipeline.decoded(latents)
            figures.append(("decoded", cosine(video.transposed(3, 0, 1, 2), record["\(stage).decoded"]!)))
            for (name, figure) in figures {
                print(String(format: "wan dancer %@ %@: %.16f", stage, name, figure))
            }
            for (name, figure) in figures {
                XCTAssertGreaterThan(figure, 0.99999, "\(stage) \(name)")
            }
        }
    }

    // MARK: Planning

    func testTheLocalStagePlansAsTheScriptDoes() throws {
        try requireMLXRuntime()
        guard let path = NFKMLXValidationConfig.environment["IK_PARITY_WAN_DANCER_PLANNING"] else {
            throw XCTSkip("set IK_PARITY_WAN_DANCER_PLANNING (run_reference.py wan_dancer_planning)")
        }
        let record = try MLX.loadArrays(url: URL(fileURLWithPath: path))
        for total in [149, 300, 447, 600, 1000, 5400] {
            let expected = record["masks.\(total)"]!
            let masks = NFKMLXWanDancerGenerator.keyframeMasks(totalFrames: total, keyframeCount: 38)
            var shared = masks
            for segment in 0 ..< Swift.max(0, shared.count - 1) {
                shared[segment][148] = true
            }
            XCTAssertEqual(shared.count, expected.dim(0), "\(total) frames")
            XCTAssertEqual(shared.flatMap { $0.map { $0 ? Int32(1) : 0 } }, expected.asArray(Int32.self), "\(total) frames")
        }
        for rate in [44100, 48000, 22050] {
            let samples = record["music.\(rate).samples"]!.asArray(Float.self)
            XCTAssertEqual(NFKMLXWanDancerGenerator.containerDuration(samples: samples.count, sampleRate: rate),
                           Double(record["music.\(rate).duration"]!.item(Float.self)), accuracy: 1e-4)
            let clips = NFKMLXWanDancerGenerator.musicClips(samples, sampleRate: rate)
            XCTAssertEqual(clips.map(\.count), record["music.\(rate).lengths"]!.asArray(Int32.self).map(Int.init))
            XCTAssertEqual(clips.last!, record["music.\(rate).last"]!.asArray(Float.self), "\(rate) Hz")
        }
    }

    // MARK: Training

    private func example(_ record: [String: MLXArray], _ prefix: String) -> NFKMLXWanDancerTrainingExample {
        let inputs = NFKMLXWanDancerInputs(condition: record["\(prefix).condition"]!, keyframeFeatures: record["\(prefix).keyframe"]!,
                                           referenceFeatures: record["\(prefix).reference"]!, music: record["\(prefix).music"]!,
                                           fps: Double(record["\(prefix).fps"]!.item(Float.self)), musicEncoderSpans: 1)
        return NFKMLXWanDancerTrainingExample(latent: record["\(prefix).latent"]!, text: record["\(prefix).text"]!, inputs: inputs)
    }

    func testTheLoRAMatchesDiffSynth() throws {
        try requireMLXRuntime()
        let directory = try testDev()
        guard let path = NFKMLXValidationConfig.environment["IK_PARITY_WAN_DANCER_LORA_DIFFSYNTH"] else {
            throw XCTSkip("set IK_PARITY_WAN_DANCER_LORA_DIFFSYNTH (run_reference.py wan_dancer_lora_diffsynth --checkpoint <testdev>)")
        }
        let record = try MLX.loadArrays(url: URL(fileURLWithPath: path))
        let preset = NFKMLXWanDancerGenerator.diffSynthPreset
        for stage in ["local", "global"] {
            let draw = preset.objective.sampler.draw(record["\(stage).index"]!.item(Float.self), weighting: preset.objective.weighting)
            XCTAssertEqual(draw.timestep, record["\(stage).timestep"]!.item(Float.self))
            XCTAssertEqual(draw.sigma, record["\(stage).sigma"]!.item(Float.self))
            XCTAssertEqual(draw.weight, record["\(stage).weight"]!.item(Float.self), accuracy: 1e-6)
            let net = try NFKMLXWanDancerGenerator.network(directoryURL: directory, stage: stage, precision: .float32)
            let adapted = try NFKMLXLoRA.apply(to: net, rank: 4, alpha: 8) { path, _ in NFKMLXWanDancerGenerator.adapts(path) }
            let adapters = record.compactMap { key, value -> (String, MLXArray)? in
                key.hasPrefix("\(stage).lora::") ? (String(key.dropFirst("\(stage).lora::".count)), value) : nil
            }
            XCTAssertEqual(adapted * 2, adapters.count, "the recipe adapts peft's targets")
            try net.update(parameters: ModuleParameters.unflattened(adapters), verify: .noUnusedKeys)
            let example = self.example(record, stage)
            let noise = record["\(stage).noise"]!
            let predictionCosine = cosine(NFKMLXWanDancerGenerator.prediction(net, example: example, draw: draw, noise: noise),
                                          record["\(stage).prediction"]!)
            let loss = NFKMLXWanDancerGenerator.loss(net, example: example, draw: draw, noise: noise).item(Float.self)
            let reference = record["\(stage).loss"]!.item(Float.self)
            XCTAssertEqual(loss, reference, accuracy: 1e-5 * reference)
            net.freeze()
            for (_, layer) in net.leafModules().flattened() where layer is NFKMLXLoRALinear {
                layer.unfreeze(recursive: false, keys: ["lora_a", "lora_b"])
            }
            let gradients = Dictionary(uniqueKeysWithValues: valueAndGrad(model: net) { net, _ in
                [NFKMLXWanDancerGenerator.loss(net, example: example, draw: draw, noise: noise)]
            }(net, []).1.flattened())
            var lowest = 1.0, matched = 0
            for (key, value) in record where key.hasPrefix("\(stage).grad::") {
                guard let ours = gradients[String(key.dropFirst("\(stage).grad::".count))] else {
                    XCTFail("no gradient for \(key)")
                    continue
                }
                matched += 1
                lowest = Swift.min(lowest, cosine(ours, value))
            }
            XCTAssertEqual(matched, gradients.count, "a gradient for every adapter")
            print("wan dancer \(stage) lora diffsynth: prediction \(String(format: "%.16f", predictionCosine)), loss \(loss) vs \(reference), "
                  + "lowest gradient cosine \(String(format: "%.12f", lowest)), \(adapted) adapters")
            XCTAssertGreaterThan(predictionCosine, 0.999999)
            XCTAssertGreaterThan(lowest, 0.999999)
        }
    }

    func testTheTransformerTrainsSavesAndReloadsThroughItsFactory() throws {
        try requireMLXRuntime()
        let directory = try testDev()
        let generator = try NFKMLXWanDancerGenerator.generator(directoryURL: directory, precision: .float32)
        MLXRandom.seed(3)
        let video = MLXRandom.uniform(0 ..< 1, [5, 64, 128, 3])
        let keyframes = video * MLXArray([Float(1), 0, 0, 0, 1]).reshaped([5, 1, 1, 1])
        let clip = NFKMLXWanDancerClip(keyframes: keyframes, keyframeMask: [true, false, false, false, true], reference: video[0],
                                       music: MLXRandom.normal([160, 35]), fps: 30)
        let example = try generator.trainingExample(video: video, prompt: "a dancer turns, 帧率是30fps。", clip: clip)
        XCTAssertEqual(example.latent.shape, [16, 2, 8, 16])
        XCTAssertEqual(example.inputs.condition.shape, [20, 2, 8, 16])
        var preset = NFKMLXWanDancerGenerator.diffSynthPreset
        preset.learningRate = 1e-2
        preset.warmupSteps = 0
        let net = try NFKMLXWanDancerGenerator.network(directoryURL: directory, precision: .float32)
        let losses = try NFKMLXWanDancerGenerator.fineTune(net, examples: { _ in [example] }, preset: preset, steps: 3, seed: 7)
        XCTAssertEqual(losses.count, 3)

        let url = FileManager.default.temporaryDirectory.appendingPathComponent("wan-dancer-lora-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try NFKMLXWanDancerGenerator.saveAdapters(of: net, to: url, preset: preset)
        let fresh = try NFKMLXWanDancerGenerator.network(directoryURL: directory, precision: .float32)
        let merged = try NFKMLXWanLoRA.apply(NFKMLXWanLoRA.deltas(adapterURL: url, naming: { $0 }, scale: 1).deltas, to: fresh)
        XCTAssertEqual(merged, 28 * 10 + 8 * 4)
        let draw = NFKMLXWanTrainingDraw(sigma: 0.5, timestep: 500, weight: 1)
        let noise = MLXRandom.normal(example.latent.shape, key: MLXRandom.key(9))
        let trained = NFKMLXWanDancerGenerator.prediction(net, example: example, draw: draw, noise: noise)
        let reloaded = NFKMLXWanDancerGenerator.prediction(fresh, example: example, draw: draw, noise: noise)
        XCTAssertLessThan((trained - reloaded).abs().max().item(Float.self), 1e-3, "the saved adapters merge to the trained network")

        let before = try generator.withPipeline(global: false) {
            NFKMLXWanDancerGenerator.prediction($0.net, example: example, draw: draw, noise: noise)
        }
        try generator.addAdapter(at: url, scale: 1, global: false)
        XCTAssertEqual(generator.adapterCount, 1)
        let after = try generator.withPipeline(global: false) {
            NFKMLXWanDancerGenerator.prediction($0.net, example: example, draw: draw, noise: noise)
        }
        XCTAssertGreaterThan((after - before).abs().max().item(Float.self), 0, "the adapter changes the stage")
        XCTAssertLessThan((after - trained).abs().max().item(Float.self), 1e-3, "the generator's stage reads the adapter")
    }

    // MARK: Structure

    func testBothReleasedTransformersMatchTheLoader() throws {
        try requireMLXRuntime()
        guard let root = NFKMLXValidationConfig.environment["IK_SHAPES_ROOT"] else { throw XCTSkip("set IK_SHAPES_ROOT") }
        for name in ["wan-dancer-14b-global", "wan-dancer-14b-local"] {
            let directory = URL(fileURLWithPath: root).appendingPathComponent(name)
            guard let data = try? Data(contentsOf: directory.appendingPathComponent("shapes.json")) else {
                throw XCTSkip("fetch \(name) with Tools/validation-assets/shapes.py")
            }
            let released = try XCTUnwrap(try JSONSerialization.jsonObject(with: data) as? [String: [Int]])
            let net = NFKMLXWanDancerNet(try NFKMLXWanDancerConfiguration.reading(directory.appendingPathComponent("config.json")))
            let built = Dictionary(net.parameters().flattened().map { ($0.0, $0.1.shape) }, uniquingKeysWith: { a, _ in a })
            let mapped = released.mapValues { shape in shape.count >= 4 ? [shape[0]] + shape.dropFirst(2) + [shape[1]] : shape }
            let missing = built.keys.filter { mapped[$0] == nil }.sorted()
            let unconsumed = mapped.keys.filter { built[$0] == nil }.sorted()
            let mismatched = built.compactMap { key, shape in mapped[key].flatMap { $0 == shape ? nil : "\(key) \(shape) vs \($0)" } }
            print("VALIDATION structure \(name): \(built.count) built, \(released.count) released, \(missing.count) missing, "
                  + "\(unconsumed.count) unconsumed, \(mismatched.count) mismatched")
            XCTAssertTrue(missing.isEmpty, "\(name) lacks \(missing.prefix(5))")
            XCTAssertTrue(unconsumed.isEmpty, "\(name) ships \(unconsumed.prefix(5)) nothing reads")
            XCTAssertTrue(mismatched.isEmpty, "\(name): \(mismatched.prefix(5))")
        }
    }

    // MARK: Residencies

    func testTheGeneratorFollowsTheReferenceThroughEveryResidency() throws {
        try requireMLXRuntime()
        let (directory, record) = try pipelineRecord()
        let reference = record["reference_image"]!.asType(.float32) / 255
        let mask = record["local.mask"]!.asArray(Int32.self).map { $0 == 1 }
        let clip = NFKMLXWanDancerClip(keyframes: record["local.keyframes"]!.asType(.float32) / 255, keyframeMask: mask,
                                       reference: reference, music: record["local.music"]!, fps: 30)
        let cases: [(String, NFKMLXResidency, Set<Int>?, Int?)] = [
            ("resident", .resident, nil, nil), ("staged", .staged, nil, nil), ("automatic", .automatic, nil, nil),
            ("streamed", .streamed, [1, 9, 27], nil), ("packed to 8 bits", .staged, nil, 8)]
        // At the test-dev's 48 width a 32-wide group packs each block's feed-forward output (96 wide in).
        for (name, residency, streamed, bits) in cases {
            let generator = try NFKMLXWanDancerGenerator.generator(directoryURL: directory, residency: residency, precision: .float32,
                                                                   bits: bits, groupSize: 32, streamedBlocks: streamed)
            XCTAssertEqual(generator.streamedLayerCount, streamed?.count ?? 0, name)
            let positive = try generator.features(stagePrompt("local", record))
            let negative = try generator.features(NFKMLXWanDancerGenerator.defaultNegativePrompt)
            let latents = try generator.withPipeline(global: false) { pipeline -> MLXArray in
                if bits != nil {
                    let packed = pipeline.net.leafModules().flattened().filter { $0.1 is QuantizedLinear }.count
                    XCTAssertEqual(packed, 28, "every block's feed-forward output packs")
                }
                pipeline.steps = 3
                return pipeline.denoise(record["local.noise"]!, positive: positive, negative: negative, inputs: pipeline.inputs(clip))
            }
            let figure = cosine(latents, record["local.steps"]![2])
            print(String(format: "wan dancer residency %@: %.16f", name, figure))
            XCTAssertGreaterThan(figure, bits == nil ? 0.99999 : 0.999, name)
        }
    }
}
