//
//  NFKMLXWanS2VTests.swift
//  InferKitMLXTests
//
//  Wan 2.2 S2V-14B (Wan-Video/Wan2.2): the official tokenizer's prompt cleanup, the rotary grids, the original
//  UniPC schedule, and, where the validation store holds the test-dev release and its record, the whole sampler
//  against the official `WanS2V.generate` (`run_reference.py wan_s2v_pipeline`). Runs where MLX has a Metal
//  library (see Tools/mlx-metallib.sh).
//

import XCTest
import InferKit
import MLX
import MLXNN
@testable import InferKitMLX

final class NFKMLXWanS2VTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    /// The cosine of two arrays, summed in double precision on the host.
    private func cosine(_ a: MLXArray, _ b: MLXArray) -> Double {
        let x = a.asType(.float32).asArray(Float.self), y = b.asType(.float32).asArray(Float.self)
        precondition(x.count == y.count, "shapes \(a.shape) and \(b.shape)")
        var dot = 0.0, xx = 0.0, yy = 0.0
        for i in x.indices {
            dot += Double(x[i]) * Double(y[i])
            xx += Double(x[i]) * Double(x[i])
            yy += Double(y[i]) * Double(y[i])
        }
        return dot / (xx.squareRoot() * yy.squareRoot())
    }

    // MARK: Weight-free

    func testThePromptCleanupFollowsTheOfficialTokenizer() {
        // Each expected value is `whitespace_clean(basic_clean(text))` under ftfy 6.3.1.
        let cases: [(String, String)] = [
            ("A woman  singing on a stage &amp; smiling, cinematic", "A woman singing on a stage & smiling, cinematic"),
            ("\u{201C}quoted\u{201D} \u{2018}x\u{2019} \u{FB01} \u{FF21}\u{FF22}\u{FF23}", "\"quoted\" 'x' fi ABC"),
            ("\u{FF25}\u{FF4E}\u{FF47}\u{FF4C}\u{FF49}\u{FF53}\u{FF48}\u{FF0C}", "English,"),
            ("  tabs\tand\r\nbreaks \u{3000}here  ", "tabs and breaks here"),
            ("&amp;amp; &#65;&#x42; &lt;b&gt;", "& AB <b>"),
            ("\u{1B}[36mblue\u{1B}[0m \u{0080}5", "blue \u{20AC}5"),
            ("<b>&amp;</b>", "<b>&</b>"),
        ]
        for (text, expected) in cases {
            XCTAssertEqual(NFKMLXWanPromptCleaning.cleaned(text), expected, text)
        }
        let negative = "画面模糊，最差质量，画面模糊"
        XCTAssertEqual(NFKMLXWanPromptCleaning.cleaned(negative), "画面模糊,最差质量,画面模糊")
    }

    func testTheRotaryGridsSpreadAsNumPyDoes() {
        typealias Grid = NFKMLXWanS2VNet.RopeGrid
        XCTAssertEqual(Grid.linspace(19, 4, 4), [19, 14, 9, 4])
        XCTAssertEqual(Grid.linspace(3, 2, 1), [3])
        XCTAssertEqual(Grid.linspace(0, 7, 2), [0, 7])
        XCTAssertEqual(Grid.linspace(0, 7, 4), [0, 2, 4, 7])
        XCTAssertEqual(Grid.linspace(0, 9, 3), [0, 4, 9])
    }

    func testTheOriginalUniPCScheduleFollowsTheOfficialSolver() {
        var scheduler = NFKMLXUniPCScheduler(NFKMLXUniPCConfiguration(flowShift: 1))
        scheduler.setOriginalTimesteps(3, shift: 3)
        XCTAssertEqual(scheduler.timesteps, [999, 856, 599])
        XCTAssertEqual(scheduler.sigmas, [0.9996664524078369, 0.8567752838134766, 0.5996398329734802, 0])
    }

    func testTheAudioBucketsFollowTheOfficialSampler() throws {
        try requireMLXRuntime()
        let frames = 23, layers = 2, width = 3
        let features = MLXArray((0 ..< (layers * frames * width)).map(Float.init)).reshaped([layers, frames, width])
        let audio = NFKMLXWanS2VPipeline.buckets(features, fps: 16, framesPerSegment: 8)
        XCTAssertEqual(audio.segments, 2)
        XCTAssertEqual(audio.buckets.shape, [layers, width, 16])
        // `np.round(linspace(0, 1, 16, endpoint=False) · 30)`: frame 13 reads index 24, past the 23 frames.
        let expected: [Int] = [0, 2, 4, 6, 8, 9, 11, 13, 15, 17, 19, 21, 22, 24, 26, 28]
        let column = audio.buckets[0, 0].asArray(Float.self)
        for (frame, index) in expected.enumerated() {
            XCTAssertEqual(column[frame], index < frames ? Float(index * width) : 0, "frame \(frame)")
        }
    }

    func testTheReleasedTransformerMatchesItsLoader() throws {
        try requireMLXRuntime()
        guard let root = NFKMLXValidationConfig.environment["IK_SHAPES_ROOT"] else { throw XCTSkip("set IK_SHAPES_ROOT") }
        let directory = URL(fileURLWithPath: root).appendingPathComponent("wan22-s2v-14b")
        guard let data = try? Data(contentsOf: directory.appendingPathComponent("shapes.json")) else {
            throw XCTSkip("fetch wan22-s2v-14b with Tools/validation-assets/shapes.py")
        }
        let released = try XCTUnwrap(try JSONSerialization.jsonObject(with: data) as? [String: [Int]])
        let net = NFKMLXWanS2VNet(try NFKMLXWanS2VConfiguration.reading(directory.appendingPathComponent("config.json")))
        let built = Dictionary(net.parameters().flattened().map { ($0.0, $0.1.shape) }, uniquingKeysWith: { a, _ in a })
        let mapped = Dictionary(released.map { key, shape -> (String, [Int]) in
            let name = NFKMLXWanS2VNet.moduleKey(forRelease: key)
            return (name, NFKMLXWanS2VNet.adapted(key: name, value: MLXArray.zeros(shape)).1.shape)
        }, uniquingKeysWith: { a, _ in a })
        let missing = built.keys.filter { mapped[$0] == nil }.sorted()
        let unconsumed = mapped.keys.filter { built[$0] == nil }.sorted()
        let mismatched = built.compactMap { key, shape in mapped[key].flatMap { $0 == shape ? nil : "\(key) \(shape) vs \($0)" } }
        print("VALIDATION structure wan22-s2v-14b: \(built.count) built, \(released.count) released, \(missing.count) missing, "
              + "\(unconsumed.count) unconsumed, \(mismatched.count) mismatched")
        XCTAssertTrue(missing.isEmpty, "the release lacks \(missing.prefix(5))")
        XCTAssertTrue(unconsumed.isEmpty, "the release ships \(unconsumed.prefix(5)) nothing reads")
        XCTAssertTrue(mismatched.isEmpty, "\(mismatched.prefix(5))")
    }

    // MARK: Against the official pipeline

    private let prompt = "A woman  singing on a stage &amp; smiling, cinematic"
    private let negative = "画面模糊，最差质量，画面模糊，细节模糊不清，情绪激动剧烈，手快速抖动，字幕，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

    func testS2VMatchesTheOfficialPipeline() throws {
        try requireMLXRuntime()
        guard let path = NFKMLXValidationConfig.environment["IK_VAL_WAN22_S2V_TESTDEV"] else {
            throw XCTSkip("set IK_VAL_WAN22_S2V_TESTDEV (Tools/wan-s2v-testdev/make_testdev.py)")
        }
        guard let recordPath = NFKMLXValidationConfig.environment["IK_PARITY_WAN22_S2V_PIPELINE"] else {
            throw XCTSkip("set IK_PARITY_WAN22_S2V_PIPELINE (run_reference.py wan_s2v_pipeline --checkpoint <testdev>)")
        }
        let directory = URL(fileURLWithPath: path)
        let record = try MLX.loadArrays(url: URL(fileURLWithPath: recordPath))
        let configuration = try NFKMLXWanS2VConfiguration.reading(directory.appendingPathComponent("config.json"))
        let net = NFKMLXWanS2VNet(configuration)
        try NFKMLXWanS2VNet.loadWeights(into: net, fromDirectory: directory)
        let autoencoder = try NFKMLXWanOriginalRelease.autoencoder(
            from: directory.appendingPathComponent(NFKMLXWanOriginalRelease.wan21AutoencoderFile))
        let pipeline = NFKMLXWanS2VPipeline(net: net, vae: autoencoder.net, latentsMean: autoencoder.mean,
                                            latentsStd: autoencoder.std)
        let encoder = try NFKMLXWanOriginalRelease.textEncoder(
            from: directory.appendingPathComponent(NFKMLXWanOriginalRelease.textEncoderFile))
        let segmenter = try NFKMLXWanFunRelease.segmenter(directory)
        var measured = [String: Double]()
        var features = [String: MLXArray]()
        for (name, text) in [("prompt", prompt), ("negative", negative)] {
            let cleaned = NFKMLXWanPromptCleaning.cleaned(text)
            let ids = NFKMLXWanFunRelease.prompt(cleaned, segmenter: segmenter).ids
            XCTAssertEqual(ids.map(Int32.init), record["\(name)_ids"]!.asArray(Int32.self), "\(name) token ids")
            features[name] = NFKMLXWanFunRelease.features(cleaned, encoder: encoder, segmenter: segmenter)
            let length = record["\(name)_embeds"]!.dim(0)
            measured["\(name) features"] = cosine(features[name]![0 ..< length], record["\(name)_embeds"]!)
        }

        let wav2vec2 = try NFKMLXWav2Vec2Net(configurationURL: directory
            .appendingPathComponent("wav2vec2-large-xlsr-53-english/config.json"))
        try wav2vec2.loadWeights(fromDirectory: directory.appendingPathComponent("wav2vec2-large-xlsr-53-english"))
        let samples = record["audio"]!.asArray(Float.self)
        let input = NFKMLXWav2Vec2Processor.inputValues(samples, normalize: wav2vec2.configuration.normalizesInput)
        measured["audio input values"] = cosine(input, record["audio.input_values"]!)
        let states = NFKMLXWanS2VPipeline.hiddenStates(input, wav2vec2: wav2vec2)
        let audioFeatures = NFKMLXWanS2VPipeline.interpolated(states, inputRate: 50, outputRate: 30)
        measured["audio features"] = cosine(audioFeatures, record["audio.features"]!)
        let geometry = record["geometry"]!.asArray(Int32.self)
        let frames = Int(geometry[2]), steps = Int(geometry[3])
        let audio = NFKMLXWanS2VPipeline.audio(samples, wav2vec2: wav2vec2, fps: 16, framesPerSegment: frames)
        XCTAssertEqual(audio.segments, Int(record["audio.repeats"]!.item(Int32.self)), "segments")
        measured["audio buckets"] = cosine(audio.buckets, record["audio.buckets"]![0])

        let reference = record["ref_image"]!
        measured["reference latent"] = cosine(pipeline.encoded((reference * 2 - 1).expandedDimensions(axis: 0)),
                                              record["ref_latents"]!)
        measured["motion latents 0"] = cosine(pipeline.encoded(MLXArray.zeros([73, reference.dim(0), reference.dim(1), 3])),
                                              record["motion_latents.0"]!)
        let tokens = net.audioTokens(record["0.input.audio"]![0], motionFrames: 73, motionLatentFrames: 19)
        let referenceGlobal = record["seam.0.audio.0"]![0, 19...], referenceLocal = record["seam.0.audio.1"]![0, 19...]
        measured["audio global tokens"] = cosine(tokens.global, referenceGlobal.squeezed(axis: 1))
        measured["audio local tokens"] = cosine(tokens.local, referenceLocal)

        // The plain run, then a pose video driving the segments from the reference image.
        for prefix in ["", "pose."] {
            let pose = prefix.isEmpty ? nil : record["pose_video"]!
            if let pose {
                let latents = pipeline.poseLatents(pose, segments: audio.segments, frames: frames, latentShape: [])
                for (segment, latent) in latents.enumerated() {
                    measured["\(prefix)segment \(segment) pose latents"] = cosine(latent, record["\(prefix)\(segment).input.cond"]!)
                }
            }
            let video = try pipeline.generate(
                text: features["prompt"]!, negative: features["negative"]!, reference: reference, audio: audio,
                pose: pose, startsFromReference: pose != nil, noise: { record["\(prefix)\($0).noise"]! }, steps: steps,
                guidance: 4.5, shift: 3,
                trace: { segment, trace in
                    let run = "\(prefix)\(segment)"
                    for (index, call) in trace.calls.enumerated() {
                        measured["\(prefix)segment \(segment) call \(index)"] = self.cosine(call, record["\(run).call.\(index)"]!)
                    }
                    for (index, latents) in trace.steps.enumerated() {
                        measured["\(prefix)segment \(segment) step \(index)"] = self.cosine(latents, record["\(run).step.\(index)"]!)
                    }
                })
            XCTAssertEqual(video.shape, record[prefix + "video"]!.shape)
            measured["\(prefix)clip"] = cosine(video, record[prefix + "video"]!)
        }
        for (name, value) in measured.sorted(by: { $0.key < $1.key }) {
            print("wan s2v \(name): \(String(format: "%.16f", value))")
            XCTAssertGreaterThan(value, 0.9999999, name)
        }
    }

    // MARK: The generator

    private func pipelineRecord() throws -> (URL, [String: MLXArray]) {
        let directory = try testDev()
        guard let path = NFKMLXValidationConfig.environment["IK_PARITY_WAN22_S2V_PIPELINE"] else {
            throw XCTSkip("set IK_PARITY_WAN22_S2V_PIPELINE (run_reference.py wan_s2v_pipeline --checkpoint <testdev>)")
        }
        return (directory, try MLX.loadArrays(url: URL(fileURLWithPath: path)))
    }

    /// The generator's clip from the record's reference, speech, and noise, as the official run made it.
    private func generatorRun(_ generator: NFKMLXWanS2VGenerator, _ record: [String: MLXArray]) throws -> MLXArray {
        generator.steps = 3
        generator.framesPerSegment = 8
        return try generator.generate(prompt: prompt, negativePrompt: nil, reference: record["ref_image"]!,
                                      audio: record["audio"]!.asArray(Float.self), pose: nil, startsFromReference: false,
                                      segments: nil, seed: 0, noise: { record["\($0).noise"]! })
    }

    func testTheGeneratorMatchesTheOfficialPipelineThroughEveryResidency() throws {
        try requireMLXRuntime()
        let (directory, record) = try pipelineRecord()
        let resident = try NFKMLXWanS2VGenerator.generator(directoryURL: directory, residency: .resident,
                                                           precision: .float32, quantizationBits: 0)
        XCTAssertTrue(resident.holdsStagesResident)
        let held = try generatorRun(resident, record)
        let measured = cosine(held, record["video"]!)
        print("wan s2v generator: \(String(format: "%.16f", measured))")
        XCTAssertGreaterThan(measured, 0.9999999)

        let staged = try NFKMLXWanS2VGenerator.generator(directoryURL: directory, residency: .staged,
                                                         precision: .float32, quantizationBits: 0)
        XCTAssertFalse(staged.holdsStagesResident)
        XCTAssertEqual((try generatorRun(staged, record) - held).abs().max().item(Float.self), 0, "staged")

        let streamed = try NFKMLXWanS2VGenerator.generator(directoryURL: directory, residency: .resident,
                                                           precision: .float32, bits: nil, streamedBlocks: [1])
        XCTAssertEqual(streamed.streamedLayerCount, 1)
        XCTAssertEqual((try generatorRun(streamed, record) - held).abs().max().item(Float.self), 0, "streamed")

        let packed = try NFKMLXWanS2VGenerator.generator(directoryURL: directory, residency: .resident,
                                                         precision: .float32, quantizationBits: 8)
        XCTAssertEqual(packed.quantizationBits, 8)
        XCTAssertGreaterThan(cosine(try generatorRun(packed, record), held), 0.999, "8-bit")
    }

    func testTheClipSizeFollowsTheOfficialSearch() {
        // `get_size_less_than_area(height, width, target_area)` for each case, as (height, width).
        let cases: [(height: Int, width: Int, area: Int, expected: (Int, Int))] = [
            (720, 1280, 1280 * 720, (704, 1280)), (1080, 1920, 1280 * 720, (704, 1280)), (1024, 768, 832 * 480, (704, 512)),
            (500, 333, 1280 * 720, (512, 384)), (64, 64, 64 * 64, (64, 64)), (3000, 200, 832 * 480, (2048, 192)),
        ]
        for c in cases {
            let size = NFKMLXWanS2VGenerator.clipSize(imageWidth: c.width, imageHeight: c.height, maximumArea: c.area)
            XCTAssertEqual(size.height, c.expected.0, "\(c.height) × \(c.width)")
            XCTAssertEqual(size.width, c.expected.1, "\(c.height) × \(c.width)")
        }
    }

    func testTheBackendReturnsAClipAndTheRegistryReachesIt() throws {
        try requireMLXRuntime()
        let (directory, record) = try pipelineRecord()
        let generator = try NFKMLXWanS2VGenerator.generator(directoryURL: directory)
        let reference = try NFKMLXImageBridge.cgImage(from: record["ref_image"]!, options: NFKMLXImageOptions())
        let speech = NFKMLXWaveFile.data(samples: record["audio"]!.asArray(Float.self), sampleRate: 16000)
        let request = NFKInferenceRequest(
            inputs: [NFKInputPrompt: prompt, NFKInputImage: reference, NFKInputAudio: speech],
            parameters: [NFKParameterWidth: 128, NFKParameterHeight: 64, NFKParameterSteps: 2,
                         NFKMLXGenerationParameterKey.segmentLength: 8, NFKMLXGenerationParameterKey.segmentCount: 1])
        let result = try generator.inferenceBackend().runInference(for: request)
        let asset = try XCTUnwrap(result.output(forKey: NFKOutputVideo) as? NFKVideoAsset)
        XCTAssertEqual(asset.dimensions, CGSize(width: 128, height: 64))
        NFKMLXWanS2VGenerator.register()
        XCTAssertNoThrow(try NFKMLXModelRegistry.backend(named: NFKMLXWanS2VGenerator.modelName, weightsURL: directory))
    }

    // MARK: Training

    private func testDev() throws -> URL {
        guard let path = NFKMLXValidationConfig.environment["IK_VAL_WAN22_S2V_TESTDEV"] else {
            throw XCTSkip("set IK_VAL_WAN22_S2V_TESTDEV (Tools/wan-s2v-testdev/make_testdev.py)")
        }
        return URL(fileURLWithPath: path)
    }

    func testS2VLoRAMatchesDiffSynth() throws {
        try requireMLXRuntime()
        let directory = try testDev()
        guard let path = NFKMLXValidationConfig.environment["IK_PARITY_WAN22_S2V_LORA_DIFFSYNTH"] else {
            throw XCTSkip("set IK_PARITY_WAN22_S2V_LORA_DIFFSYNTH (run_reference.py wan_s2v_lora_diffsynth --checkpoint <testdev>)")
        }
        let record = try MLX.loadArrays(url: URL(fileURLWithPath: path))
        let preset = NFKMLXWanS2VGenerator.diffSynthPreset
        let draw = preset.objective.sampler.draw(record["index"]!.item(Float.self), weighting: preset.objective.weighting)
        XCTAssertEqual(draw.timestep, record["timestep"]!.item(Float.self))
        XCTAssertEqual(draw.sigma, record["sigma"]!.item(Float.self))
        XCTAssertEqual(draw.weight, record["weight"]!.item(Float.self), accuracy: 1e-6)
        let net = try NFKMLXWanS2VGenerator.network(directoryURL: directory, precision: .float32)
        let adapted = try NFKMLXLoRA.apply(to: net, rank: 4, alpha: 8) { path, _ in
            NFKMLXWanS2VGenerator.adapts(path, targets: preset.targets)
        }
        let adapters = record.compactMap { key, value -> (String, MLXArray)? in
            guard key.hasPrefix("lora::") else { return nil }
            let name = String(key.dropFirst("lora::".count)), layer = String(name.split(separator: ".").dropLast().joined(separator: "."))
            return NFKMLXWanLoRA.speechToVideoPath(original: layer).map { ($0 + "." + name.split(separator: ".").last!, value) }
        }
        XCTAssertEqual(adapted * 2, adapters.count, "the recipe adapts DiffSynth's targets")
        try net.update(parameters: ModuleParameters.unflattened(adapters), verify: .noUnusedKeys)
        let example = NFKMLXWanS2VTrainingExample(latent: record["latent"]!, text: record["text"]!, audio: record["audio"]!,
                                                  pose: record["pose"]!)
        let noise = record["noise"]!
        let predictionCosine = cosine(NFKMLXWanS2VGenerator.prediction(net, example: example, draw: draw, noise: noise),
                                      record["prediction"]!)
        let loss = NFKMLXWanS2VGenerator.loss(net, example: example, draw: draw, noise: noise).item(Float.self)
        let reference = record["loss"]!.item(Float.self)
        XCTAssertEqual(loss, reference, accuracy: 1e-5 * reference)
        net.freeze()
        for (_, layer) in net.leafModules().flattened() where layer is NFKMLXLoRALinear {
            layer.unfreeze(recursive: false, keys: ["lora_a", "lora_b"])
        }
        let gradients = Dictionary(uniqueKeysWithValues: valueAndGrad(model: net) { net, _ in
            [NFKMLXWanS2VGenerator.loss(net, example: example, draw: draw, noise: noise)]
        }(net, []).1.flattened())
        var lowest = 1.0
        var matched = 0
        for (key, value) in record where key.hasPrefix("grad::") {
            let name = String(key.dropFirst("grad::".count))
            let layer = String(name.split(separator: ".").dropLast().joined(separator: "."))
            guard let path = NFKMLXWanLoRA.speechToVideoPath(original: layer),
                  let ours = gradients[path + "." + name.split(separator: ".").last!] else {
                XCTFail("no gradient for \(name)")
                continue
            }
            matched += 1
            lowest = min(lowest, cosine(ours, value))
        }
        XCTAssertEqual(matched, gradients.count, "a gradient for every adapter")
        print("wan s2v lora diffsynth: prediction \(String(format: "%.16f", predictionCosine)), loss \(loss) vs \(reference), "
              + "lowest gradient cosine \(String(format: "%.12f", lowest))")
        XCTAssertGreaterThan(predictionCosine, 0.999999)
        XCTAssertGreaterThan(lowest, 0.999999)
    }

    func testS2VTrainsSavesAndReloadsThroughItsFactory() throws {
        try requireMLXRuntime()
        let directory = try testDev()
        let generator = try NFKMLXWanS2VGenerator.generator(directoryURL: directory, residency: .resident,
                                                            precision: .float32, quantizationBits: 0)
        MLXRandom.seed(3)
        let video = MLXRandom.uniform(0 ..< 1, [9, 64, 128, 3])
        let speech = (0 ..< 12800).map { Float(sin(Double($0) * 0.05)) * 0.3 }
        let example = try generator.trainingExample(video: video, prompt: prompt, audio: speech,
                                                    pose: MLXRandom.uniform(0 ..< 1, [8, 64, 128, 3]))
        XCTAssertEqual(example.latent.shape, [16, 3, 8, 16])
        XCTAssertEqual(example.audio.shape, [25, 32, 8])
        XCTAssertEqual(example.pose?.shape, [16, 2, 8, 16])
        var preset = NFKMLXWanS2VGenerator.diffSynthPreset
        preset.learningRate = 1e-2
        preset.warmupSteps = 0
        let net = try NFKMLXWanS2VGenerator.network(directoryURL: directory, precision: .float32)
        let losses = try NFKMLXWanS2VGenerator.fineTune(net, examples: { _ in [example] }, preset: preset, steps: 3, seed: 7)
        XCTAssertEqual(losses.count, 3)

        let url = FileManager.default.temporaryDirectory.appendingPathComponent("wan-s2v-lora-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try NFKMLXWanS2VGenerator.saveAdapters(of: net, to: url, preset: preset)
        let fresh = try NFKMLXWanS2VGenerator.network(directoryURL: directory, precision: .float32)
        let report = try NFKMLXWanLoRA.merge(adapterURL: url, into: fresh)
        XCTAssertEqual(report.mergedLayers, 2 * 10 + 2 * 4)
        let draw = NFKMLXWanTrainingDraw(sigma: 0.5, timestep: 500, weight: 1)
        let noise = MLXRandom.normal(example.latent.shape, key: MLXRandom.key(9))
        let trained = NFKMLXWanS2VGenerator.prediction(net, example: example, draw: draw, noise: noise)
        let merged = NFKMLXWanS2VGenerator.prediction(fresh, example: example, draw: draw, noise: noise)
        XCTAssertLessThan((trained - merged).abs().max().item(Float.self), 1e-3, "the saved adapters merge to the trained network")

        generator.steps = 2
        generator.framesPerSegment = 8
        let reference = video[0]
        let before = try generator.generate(prompt: prompt, negativePrompt: nil, reference: reference, audio: speech, pose: nil,
                                            startsFromReference: false, segments: 1, seed: 1)
        try generator.addAdapter(at: url, scale: 1)
        XCTAssertEqual(generator.adapterCount, 1)
        let after = try generator.generate(prompt: prompt, negativePrompt: nil, reference: reference, audio: speech, pose: nil,
                                           startsFromReference: false, segments: 1, seed: 1)
        XCTAssertEqual(before.shape, [5, 64, 128, 3])
        XCTAssertGreaterThan((after - before).abs().max().item(Float.self), 0, "the adapter changes the clip")
    }
}
