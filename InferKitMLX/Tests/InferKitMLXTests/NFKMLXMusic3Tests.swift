//
//  NFKMLXMusic3Tests.swift
//  InferKitMLXTests
//
//  Stage 1 of the MiniMax Music 3 port: the Flow-VAE vocoder. The numeric comparison against
//  diffusers' own MiniMaxMusic3Vocoder lives in NFKMLXReferenceParityTests; these cover the
//  structure — the activation, the exact upsampling, the stereo fold, and that the released
//  checkpoint and the module account for each other tensor for tensor.
//

import XCTest
import InferKit
import MLX
import MLXNN
@testable import InferKitMLX

final class NFKMLXMusic3Tests: XCTestCase {

    private lazy var config: [String: String] = {
        var merged = ProcessInfo.processInfo.environment
        let url = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".inferkit-validation.json")
        if let data = try? Data(contentsOf: url),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: String] {
            json.forEach { merged[$0.key] = $0.value }
        }
        return merged
    }()

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test`; run via xcodebuild")
    }

    override func tearDown() {
        NFKMLXGPU.clearCache()
        super.tearDown()
    }

    private func cosine(_ a: [Double], _ b: [Double]) -> Double {
        var dot = 0.0, na = 0.0, nb = 0.0
        for i in 0 ..< min(a.count, b.count) {
            dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]
        }
        return (na > 0 && nb > 0) ? dot / (na.squareRoot() * nb.squareRoot()) : 0
    }

    func testTheSnakeActivationMatchesTheReferenceFormula() throws {
        try requireMLXRuntime()
        let snake = NFKMusic3Snake(channels: 2)
        snake.update(parameters: ModuleParameters.unflattened([("alpha", MLXArray([Float(0.5), 2.0]).reshaped([1, 1, 2]))]))
        let x: [Float] = [0.3, -1.1]
        let y = snake(MLXArray(x).reshaped([1, 1, 2]))
        eval(y)
        let values = y.reshaped([-1]).asArray(Float.self)
        for channel in 0 ..< 2 {
            let alpha: Float = channel == 0 ? 0.5 : 2.0
            let expected = x[channel] + (1.0 / (alpha + 1e-9)) * pow(Foundation.sin(alpha * x[channel]), 2)
            XCTAssertEqual(values[channel], expected, accuracy: 1e-6)
        }
    }

    // Kernel 2·stride with padding ceil(stride/2) makes each transposed convolution multiply the
    // length by exactly its stride, so the waveform is exactly hop × latent frames.
    func testTheVocoderUpsamplesByExactlyTheHop() throws {
        try requireMLXRuntime()
        let configuration = NFKMLXMusic3VocoderConfiguration.tiny
        let net = NFKMLXMusic3.makeVocoder(configuration)
        let latents = MLXArray.zeros([1, 5, configuration.latentChannels])
        let wave = net.waveform(latents)
        eval(wave)
        XCTAssertEqual(wave.shape, [1, 5 * configuration.hop, 2])
        let extremes = wave.abs().max().item(Float.self)
        XCTAssertLessThanOrEqual(extremes, 1.0, "tanh bounds the waveform")
    }

    // Stereo is the latent's two channel halves decoded as two mono streams by ONE shared decoder.
    // Swapping the halves must therefore swap the output channels exactly — which pins the fold's
    // ordering without any weights being right.
    func testSwappingTheLatentHalvesSwapsTheStereoChannels() throws {
        try requireMLXRuntime()
        let configuration = NFKMLXMusic3VocoderConfiguration.tiny
        let net = NFKMLXMusic3.makeVocoder(configuration)
        let half = configuration.latentChannels / 2

        let key = MLXRandom.key(3)
        let latents = MLXRandom.normal([1, 4, configuration.latentChannels], key: key)
        let swapped = concatenated([latents[.ellipsis, half...], latents[.ellipsis, ..<half]], axis: -1)

        let wave = net.waveform(latents)
        let waveSwapped = net.waveform(swapped)
        eval(wave, waveSwapped)

        let leftThenRight = wave.asArray(Float.self)
        let rightThenLeft = waveSwapped[.ellipsis, MLXArray([1, 0])].asArray(Float.self)
        for (a, b) in zip(leftThenRight, rightThenLeft) {
            XCTAssertEqual(a, b, accuracy: 1e-5)
        }
    }

    // The released checkpoint and the module account for each other exactly: after the weight-norm
    // fusion, every checkpoint tensor is a module parameter and every module parameter comes from the
    // checkpoint — the converse the DeepSeek experience showed a one-directional check misses.
    func testEveryReleasedTensorIsDeclaredAndEveryParameterCovered() throws {
        try requireMLXRuntime()
        guard let path = config["IK_VAL_MUSIC3_VOCODER"], FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_VAL_MUSIC3_VOCODER (Tools/validation-assets/fetch.py --only MUSIC3)")
        }
        let url = URL(fileURLWithPath: path)
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        XCTAssertEqual(checkpoint.arrays.count, 121, "the release ships 121 tensors")

        let fused = NFKMLXMusic3.fusedWeightNorm(checkpoint.arrays)
        let provided = Set(fused.keys.map(NFKMLXMusic3.remapVocoderKey))

        let net = NFKMLXMusic3.makeVocoder()
        let expected = Set(net.parameters().flattened().map(\.0))

        XCTAssertEqual(provided.subtracting(expected).sorted(), [],
                       "every released tensor lands on a module parameter")
        XCTAssertEqual(expected.subtracting(provided).sorted(), [],
                       "every module parameter comes from the release")

        try NFKMLXMusic3.loadVocoderWeights(into: net, from: url)
        for (key, value) in net.parameters().flattened() where key.hasSuffix("alpha") {
            XCTAssertEqual(value.shape.count, 3)
            XCTAssertEqual(Array(value.shape[0 ... 1]), [1, 1], "\(key) is transposed to broadcast over NLC")
        }
    }

    // The depth sequence is decoded one codebook at a time, so a step's hidden state must not change
    // when later steps are appended — which is exactly what the causal mask provides.
    func testTheDepthDecoderIsCausal() throws {
        try requireMLXRuntime()
        let net = NFKMLXMusic3.makeDepthDecoder(.tiny)
        let key = MLXRandom.key(7)
        let sequence = MLXRandom.normal([1, 6, 64], key: key)

        let full = net.hiddenStates(sequence)
        let prefix = net.hiddenStates(sequence[0..., ..<4])
        eval(full, prefix)

        let fullValues = full[0..., ..<4].asArray(Float.self)
        let prefixValues = prefix.asArray(Float.self)
        for (a, b) in zip(fullValues, prefixValues) {
            XCTAssertEqual(a, b, accuracy: 1e-5)
        }
    }

    // Nearest-neighbor resampling at the released rates maps 13 frames onto 44 latents, and the
    // scale 13/44 puts source frame 0 under the first four latents — pinned without weights.
    func testTheConditionEncoderResamplesToTheLatentRate() throws {
        try requireMLXRuntime()
        let configuration = NFKMLXMusic3ConditionConfiguration.tiny
        XCTAssertEqual(configuration.latentLength(forFrames: 13), 44)
        XCTAssertEqual(configuration.latentLength(forFrames: 200), 689)

        let net = NFKMLXMusic3.makeConditionEncoder(configuration)
        let key = MLXRandom.key(9)
        let hidden = MLXRandom.normal([1, 13, configuration.conditionLayers * configuration.hiddenDimensions],
                                      key: key)
        let condition = net.condition(hidden)
        eval(condition)
        XCTAssertEqual(condition.shape, [1, 44, configuration.outputDimensions])

        let first = condition[0, 0].asArray(Float.self)
        let third = condition[0, 3].asArray(Float.self)
        let fifth = condition[0, 4].asArray(Float.self)
        for (a, b) in zip(first, third) {
            XCTAssertEqual(a, b, accuracy: 0, "latents 0...3 all read source frame 0")
        }
        XCTAssertNotEqual(first, fifth, "latent 4 reads source frame 1")
    }

    func testTheDepthAndConditionReleasesAreAccountedBothDirections() throws {
        try requireMLXRuntime()
        guard let depthPath = config["IK_VAL_MUSIC3_DEPTH"],
              let conditionPath = config["IK_VAL_MUSIC3_CONDITION"],
              FileManager.default.fileExists(atPath: depthPath),
              FileManager.default.fileExists(atPath: conditionPath) else {
            throw XCTSkip("set IK_VAL_MUSIC3_DEPTH / IK_VAL_MUSIC3_CONDITION (fetch.py --only MUSIC3)")
        }

        let depth = NFKMLXMusic3.makeDepthDecoder()
        let depthArrays = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: depthPath)).arrays
        XCTAssertEqual(depthArrays.count, 47, "the depth release ships 47 tensors")
        let depthExpected = Set(depth.parameters().flattened().map(\.0))
        XCTAssertEqual(Set(depthArrays.keys).subtracting(depthExpected).sorted(), [])
        XCTAssertEqual(depthExpected.subtracting(depthArrays.keys).sorted(), [])

        try NFKMLXMusic3.loadDepthWeights(into: depth, from: URL(fileURLWithPath: depthPath))
        // The release is bf16; the default load converts, so the module still computes in float32.
        XCTAssertEqual(depth.projection.weight.dtype, .float32)

        let condition = NFKMLXMusic3.makeConditionEncoder()
        let conditionArrays = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: conditionPath)).arrays
        XCTAssertEqual(conditionArrays.count, 4, "the condition release ships 4 tensors")
        let conditionExpected = Set(condition.parameters().flattened().map(\.0))
        XCTAssertEqual(Set(conditionArrays.keys).subtracting(conditionExpected).sorted(), [])
        XCTAssertEqual(conditionExpected.subtracting(conditionArrays.keys).sorted(), [])
        try NFKMLXMusic3.loadConditionWeights(into: condition, from: URL(fileURLWithPath: conditionPath))
    }

    // The DiT's timestep is a sequence token, not an adaLN: prepended before the blocks and stripped
    // after, so the velocity keeps the latent's shape — and a different timestep must change it.
    func testTheDiTVelocityKeepsTheLatentShapeAndReadsTheTimestep() throws {
        try requireMLXRuntime()
        let configuration = NFKMLXMusic3DiTConfiguration.tiny
        let net = NFKMLXMusic3.makeDiT(configuration)
        let key = MLXRandom.key(21)
        let latents = MLXRandom.normal([1, 6, configuration.latentChannels], key: key)
        let condition = MLXRandom.normal([1, 6, configuration.conditionDimensions],
                                         key: MLXRandom.key(22))

        let early = net.velocity(latents: latents, timestep: MLXArray([Float(0)]), condition: condition)
        let late = net.velocity(latents: latents, timestep: MLXArray([Float(0.9)]), condition: condition)
        eval(early, late)
        XCTAssertEqual(early.shape, latents.shape)
        let difference = (early - late).abs().max().item(Float.self)
        XCTAssertGreaterThan(difference, 0, "the timestep conditions the velocity")
    }

    // The schedule's arithmetic, checked without the record: σ runs 0 → 1 − 1/N with a terminal 1,
    // a step moves by (σ_next − σ)·v, and the overlap blend is pure noise at σ 0 and (almost) the
    // previous window's latent at σ 1.
    func testTheFlowScheduleArithmetic() throws {
        try requireMLXRuntime()
        let sigmas = NFKMusic3FlowSchedule.sigmas(steps: 30)
        XCTAssertEqual(sigmas.count, 31)
        XCTAssertEqual(sigmas.first, 0)
        XCTAssertEqual(sigmas[29], 1 - 1.0 / 30, accuracy: 1e-6)
        XCTAssertEqual(sigmas.last, 1)

        let latent = MLXArray([Float(2)])
        let stepped = NFKMusic3FlowSchedule.step(latent: latent, velocity: MLXArray([Float(0.5)]),
                                                 sigma: 0.2, nextSigma: 0.6)
        eval(stepped)
        XCTAssertEqual(stepped.item(Float.self), 2 + 0.4 * 0.5, accuracy: 1e-6)

        let noise = MLXArray([Float(3)]), previous = MLXArray([Float(-1)])
        let atNoise = NFKMusic3FlowSchedule.blendedOverlap(noise: noise, previous: previous, sigma: 0)
        let atData = NFKMusic3FlowSchedule.blendedOverlap(noise: noise, previous: previous, sigma: 1)
        eval(atNoise, atData)
        XCTAssertEqual(atNoise.item(Float.self), 3, accuracy: 1e-6)
        XCTAssertEqual(atData.item(Float.self), -1 + 3e-6, accuracy: 1e-6)
    }

    // The measured half of the schedule: diffusers' own FlowMatchEulerDiscreteScheduler, configured
    // from the release's scheduler config and driven with the pipeline's linspace, produces exactly
    // this σ sequence — and the timesteps the transformer consumes are the σ values themselves.
    func testTheFlowScheduleMatchesTheReferenceScheduler() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_MUSIC3_DIT"], FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_PARITY_MUSIC3_DIT (run_reference.py music_dit)")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: path))
        let referenceSigmas = try XCTUnwrap(arrays["sigmas"]).asArray(Float.self)
        let referenceTimesteps = try XCTUnwrap(arrays["timesteps"]).asArray(Float.self)

        let sigmas = NFKMusic3FlowSchedule.sigmas(steps: 30)
        XCTAssertEqual(sigmas.count, referenceSigmas.count)
        for (mine, theirs) in zip(sigmas, referenceSigmas) {
            XCTAssertEqual(mine, theirs, accuracy: 1e-7)
        }
        for (mine, theirs) in zip(sigmas, referenceTimesteps) {
            XCTAssertEqual(mine, theirs, accuracy: 1e-7, "the model's timestep is σ itself")
        }
    }

    // The DiT is 9.7 GB, so the structural check enumerates rather than builds: the tiny module's
    // key template, expanded to 36 layers, must equal the shard index's own key set exactly.
    func testTheDiTReleaseIsAccountedBothDirections() throws {
        try requireMLXRuntime()
        guard let directory = config["IK_VAL_MUSIC3_DIT"],
              FileManager.default.fileExists(atPath: directory) else {
            throw XCTSkip("set IK_VAL_MUSIC3_DIT (fetch.py --only MUSIC3)")
        }
        let indexURL = URL(fileURLWithPath: directory)
            .appendingPathComponent("diffusion_pytorch_model.safetensors.index.json")
        let json = try JSONSerialization.jsonObject(with: Data(contentsOf: indexURL)) as? [String: Any]
        let weightMap = try XCTUnwrap(json?["weight_map"] as? [String: String])
        let released = Set(weightMap.keys)

        let template = NFKMLXMusic3.makeDiT(.tiny).parameters().flattened().map(\.0)
        var expected = Set<String>()
        for key in template {
            guard key.hasPrefix("transformer_blocks.") else {
                expected.insert(key)
                continue
            }
            let suffix = key.split(separator: ".", maxSplits: 2)[2]
            for layer in 0 ..< 36 {
                expected.insert("transformer_blocks.\(layer).\(suffix)")
            }
        }
        XCTAssertEqual(released.subtracting(expected).sorted(), [],
                       "every released tensor lands on a module parameter")
        XCTAssertEqual(expected.subtracting(released).sorted(), [],
                       "every module parameter comes from the release")
    }

    // The window bookkeeping, checked against the reference's own arithmetic: a fitting timeline is
    // one window, and a longer one steps by the hop while stopping before `frames − hop`, so the
    // last window still spans a full 200 frames.
    func testTheFlowMatcherWindowsAndCropsLikeTheReference() throws {
        XCTAssertEqual(NFKMusic3FlowMatcher.chunkStarts(frames: 150), [0])
        XCTAssertEqual(NFKMusic3FlowMatcher.chunkStarts(frames: 200), [0])
        XCTAssertEqual(NFKMusic3FlowMatcher.chunkStarts(frames: 201), [0, 100])
        XCTAssertEqual(NFKMusic3FlowMatcher.chunkStarts(frames: 500), [0, 100, 200, 300])

        let samples = 689 * 512
        XCTAssertEqual(NFKMusic3FlowMatcher.keptRange(chunkIndex: 0, chunkCount: 3,
                                                      samples: samples, hop: 512),
                       0 ..< samples - 258 * 512)
        XCTAssertEqual(NFKMusic3FlowMatcher.keptRange(chunkIndex: 1, chunkCount: 3,
                                                      samples: samples, hop: 512),
                       86 * 512 ..< samples - 258 * 512)
        XCTAssertEqual(NFKMusic3FlowMatcher.keptRange(chunkIndex: 2, chunkCount: 3,
                                                      samples: samples, hop: 512),
                       86 * 512 ..< samples)
    }

    // Continuity is enforced inside the sampler and locked after it: the second window's leading
    // overlap must come back EQUAL to the first window's carried span, whatever the weights.
    func testNeighboringWindowsShareTheirBoundaryLatents() throws {
        try requireMLXRuntime()
        var transformerConfiguration = NFKMLXMusic3DiTConfiguration.tiny
        transformerConfiguration.conditionDimensions = NFKMLXMusic3ConditionConfiguration.tiny.outputDimensions
        let matcher = NFKMusic3FlowMatcher(
            transformer: NFKMLXMusic3.makeDiT(transformerConfiguration),
            conditionEncoder: NFKMLXMusic3.makeConditionEncoder(.tiny))

        let hiddenWidth = NFKMLXMusic3ConditionConfiguration.tiny.conditionLayers
            * NFKMLXMusic3ConditionConfiguration.tiny.hiddenDimensions
        let frameHiddens = MLXRandom.normal([1, 250, hiddenWidth], key: MLXRandom.key(31))
        let chunks = try XCTUnwrap(matcher.latentChunks(frameHiddens: frameHiddens, steps: 2))
        XCTAssertEqual(chunks.count, 2)

        let first = chunks[0], second = chunks[1]
        let length = first.shape[1]
        let overlapStart = max(0, length - 2 * 172)
        let overlapEnd = max(overlapStart, length - 172)
        let carried = first[0..., overlapStart ..< overlapEnd]
        let leading = second[0..., ..<carried.shape[1]]
        eval(carried, leading)
        let worst = (carried - leading).abs().max().item(Float.self)
        XCTAssertEqual(worst, 0, "the overlap is locked to the previous window's carry")
    }

    // The cleaners against the reference's own intermediate strings (captured from
    // `_clean_caption` / `_normalize_lyrics` on the shared MUSIC_PROMPTS cases), so a cleaning bug
    // reads as a string diff here before it reads as a token mismatch in the parity record.
    func testTheCaptionAndLyricsCleanersMatchTheReferenceStrings() throws {
        XCTAssertEqual(NFKMusic3Prompt.cleanedCaption("Dreamy synth-pop, female vocals"),
                       "Dreamy synth-pop, female vocals")
        XCTAssertEqual(
            NFKMusic3Prompt.cleanedCaption(
                "# Epic Rock\n- **loud** guitars\n* driving *rhythm*\n---\n• four    spaces"),
            "Epic Rock\nloud guitars\ndriving rhythm\nfourspaces")
        XCTAssertEqual(NFKMusic3Prompt.cleanedCaption("<|bpm 128|> J-ポップ 🎵 vocals"),
                       "bpm is 128 J-ポップ 🎵 vocals")
        XCTAssertEqual(NFKMusic3Prompt.cleanedCaption("  jazz,  with\n\n\nswing!  "),
                       "  jazz,  with\nswing!")

        XCTAssertEqual(NFKMusic3Prompt.normalizedLyrics("[verse]\nHello world"),
                       "[start]\n[verse]\nHello world")
        XCTAssertEqual(
            NFKMusic3Prompt.normalizedLyrics("[Verse] ignored text\nFirst line [Chorus] second"),
            "[start]\n[verse]\nFirst line\n[chorus]\nsecond")
        XCTAssertEqual(NFKMusic3Prompt.normalizedLyrics("[intro]\nこんにちは ^ 世界"),
                       "[start]\n[intro]\nこんにちは\n世界")
        XCTAssertEqual(NFKMusic3Prompt.normalizedLyrics("[verse]\nline one  \n[bridge]\nend"),
                       "[start]\n[verse]\nline one  \n[bridge]\nend")
    }

    // The whole prompt contract, token for token against the reference tokenizer: the cleaners, the
    // special-token template, the byte-level BPE with added-token splitting, and the CFG-row
    // substitution. The cases are run_reference.py's MUSIC_PROMPTS, verbatim.
    func testThePromptContractMatchesTheReferenceTokenizer() throws {
        try requireMLXRuntime()
        guard let recordPath = config["IK_PARITY_MUSIC3_TOKENIZER"],
              let bpePath = config["IK_VAL_MUSIC3_BPE"],
              FileManager.default.fileExists(atPath: bpePath) else {
            throw XCTSkip("set IK_PARITY_MUSIC3_TOKENIZER and IK_VAL_MUSIC3_BPE")
        }
        let arrays = try loadArrays(url: URL(fileURLWithPath: recordPath))
        let tokenizer = try NFKMusic3Prompt.tokenizer(directory: URL(fileURLWithPath: bpePath))

        let cases: [(String, String)] = [
            ("Dreamy synth-pop, female vocals", "[verse]\nHello world"),
            ("# Epic Rock\n- **loud** guitars\n* driving *rhythm*\n---\n• four    spaces",
             "[Verse] ignored text\nFirst line [Chorus] second"),
            ("<|bpm 128|> J-ポップ 🎵 vocals", "[intro]\nこんにちは ^ 世界"),
            ("  jazz,  with\n\n\nswing!  ", "[verse]\nline one  \n[bridge]\nend"),
        ]
        XCTAssertEqual(try XCTUnwrap(arrays["case_count"]).item(Int32.self), Int32(cases.count))
        for (index, (caption, lyrics)) in cases.enumerated() {
            let reference = try XCTUnwrap(arrays["case\(index)_ids"])
            let mine = try NFKMusic3Prompt.textIDs(caption: caption, lyrics: lyrics,
                                                   tokenizer: tokenizer)
            eval(mine)
            XCTAssertEqual(mine.shape, reference.shape, "case \(index) length")
            XCTAssertEqual(mine.asArray(Int32.self), reference.asArray(Int32.self),
                           "case \(index) matches token for token, both CFG rows")
        }
    }

    // The sampler's threshold must be the k-th LARGEST logit. Read through `MLX.top`'s unsorted
    // result it silently becomes argmax — so 80 seeded draws must show more than one distinct
    // choice, and every choice must come from the true top-k.
    func testTheSamplerDrawsFromTheTopKAndOnlyTheTopK() throws {
        let logits: [Float] = (0 ..< 100).map { Float($0) / 10 }        // top-5 = indices 95...99
        var support = Set<Int>()
        for seed in 0 ..< 80 {
            var sampler = NFKMusic3Sampler(seed: UInt64(seed))
            support.insert(sampler.sample(logits: logits, topK: 5))
        }
        XCTAssertGreaterThan(support.count, 1, "sampling, not argmax")
        XCTAssertTrue(support.isSubset(of: Set(95 ..< 100)), "only the top-k are reachable")
    }

    // The reference declares its prompt and frame limits and never applies the position budget;
    // here the sum is enforced before any forward runs, so the guard needs no real weights.
    func testTheAutoregressiveStageRejectsAnOverlongRequest() throws {
        try requireMLXRuntime()
        var languageConfiguration = NFKMLXLanguageConfiguration(
            hiddenSize: 64, layerCount: 1, headCount: 4, keyValueHeadCount: 2, headDimensions: 16,
            intermediateSize: 64, vocabularySize: 200_000, tiesWordEmbeddings: true)
        languageConfiguration.normalizesQueryAndKey = true
        var depthConfiguration = NFKMLXMusic3DepthConfiguration.tiny
        depthConfiguration.hiddenSize = 64
        let stage = NFKMusic3AutoregressiveStage(
            languageModel: NFKMLXLanguageNet(languageConfiguration),
            depthDecoder: NFKMLXMusic3.makeDepthDecoder(depthConfiguration))

        let overlongPrompt = MLXArray.zeros([2, 5_001], type: Int32.self)
        XCTAssertThrowsError(try stage.generate(textIDs: overlongPrompt, maxFrames: 10))

        let prompt = MLXArray.zeros([2, 2_000], type: Int32.self)
        XCTAssertThrowsError(try stage.generate(textIDs: prompt, maxFrames: 9_000),
                             "2000 prompt tokens + 9000 frames exceed the 10240-position budget")
    }

    // A quantized save records its geometry in the checkpoint metadata, and the loaders quantize a
    // fresh module's STRUCTURE to match before applying — without that, a packed uint32 weight
    // loaded into a plain Linear adopts the wrong shape and dtype silently. The round trip must
    // reproduce the quantized forward exactly, packed values and scales alike.
    func testAQuantizedCheckpointRoundTripsThroughTheLoaders() throws {
        try requireMLXRuntime()
        let scratch = FileManager.default.temporaryDirectory
        defer { NFKMLXGPU.clearCache() }

        let depth = NFKMLXMusic3.makeDepthDecoder(.tiny)
        NFKMLXQuantization.quantize(module: depth, bits: 4, groupSize: 32)
        let depthInput = MLXRandom.normal([1, 5, 64], key: MLXRandom.key(41))
        let depthBefore = depth.hiddenStates(depthInput)
        let depthURL = scratch.appendingPathComponent("\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: depthURL) }
        try NFKMLXWeights.save(depth, to: depthURL)
        let depthReloaded = NFKMLXMusic3.makeDepthDecoder(.tiny)
        try NFKMLXMusic3.loadDepthWeights(into: depthReloaded, from: depthURL)
        let depthAfter = depthReloaded.hiddenStates(depthInput)
        eval(depthBefore, depthAfter)
        XCTAssertEqual((depthBefore - depthAfter).abs().max().item(Float.self), 0,
                       "the quantized depth decoder reloads exactly")

        let transformer = NFKMLXMusic3.makeDiT(.tiny)
        NFKMLXQuantization.quantize(module: transformer, bits: 8, groupSize: 32)
        let latents = MLXRandom.normal([1, 6, 8], key: MLXRandom.key(42))
        let condition = MLXRandom.normal([1, 6, 16], key: MLXRandom.key(43))
        let timestep = MLXArray([Float(0.5)])
        let ditBefore = transformer.velocity(latents: latents, timestep: timestep, condition: condition)
        let ditURL = scratch.appendingPathComponent("\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: ditURL) }
        try NFKMLXWeights.save(transformer, to: ditURL)
        let ditReloaded = NFKMLXMusic3.makeDiT(.tiny)
        try NFKMLXMusic3.loadDiTWeights(into: ditReloaded, from: ditURL)
        let ditAfter = ditReloaded.velocity(latents: latents, timestep: timestep, condition: condition)
        eval(ditBefore, ditAfter)
        XCTAssertEqual((ditBefore - ditAfter).abs().max().item(Float.self), 0,
                       "the quantized DiT reloads exactly")

        var languageConfiguration = NFKMLXLanguageConfiguration(
            hiddenSize: 64, layerCount: 2, headCount: 4, keyValueHeadCount: 2, headDimensions: 16,
            intermediateSize: 96, vocabularySize: 128, tiesWordEmbeddings: false)
        languageConfiguration.normalizesQueryAndKey = true
        let language = NFKMLXLanguage.makeNet(languageConfiguration)
        NFKMLXQuantization.quantize(module: language, bits: 4, groupSize: 32)
        let tokens = MLXArray([Int32(3), 17, 42, 99]).reshaped([1, 4])
        let logitsBefore = language(tokens)
        let languageURL = scratch.appendingPathComponent("\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: languageURL) }
        try NFKMLXWeights.save(language, to: languageURL)
        let languageReloaded = NFKMLXLanguage.makeNet(languageConfiguration)
        try NFKMLXLanguage.loadWeights(into: languageReloaded, from: languageURL)
        let logitsAfter = languageReloaded(tokens)
        eval(logitsBefore, logitsAfter)
        XCTAssertEqual((logitsBefore - logitsAfter).abs().max().item(Float.self), 0,
                       "the quantized language model reloads exactly")

        // With the input embedding quantized too (the music recipe, since the LM is untied), the
        // packed embedding must survive the round trip. The loader records only bits:groupSize, so it
        // reconstructs the QuantizedEmbedding by seeing the saved weight is uint32 — a saved file
        // without a quantized embedding (the case above) reloads through the same path unchanged.
        let embeddedLanguage = NFKMLXLanguage.makeNet(languageConfiguration)
        NFKMLXQuantization.quantize(module: embeddedLanguage, bits: 4, groupSize: 32,
                                    includeEmbeddings: true)
        XCTAssertTrue(embeddedLanguage.model.embedTokens is QuantizedEmbedding,
                      "includeEmbeddings packs the input embedding")
        let embeddedBefore = embeddedLanguage(tokens)
        let embeddedURL = scratch.appendingPathComponent("\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: embeddedURL) }
        try NFKMLXWeights.save(embeddedLanguage, to: embeddedURL)
        let embeddedReloaded = NFKMLXLanguage.makeNet(languageConfiguration)
        try NFKMLXLanguage.loadWeights(into: embeddedReloaded, from: embeddedURL)
        XCTAssertTrue(embeddedReloaded.model.embedTokens is QuantizedEmbedding,
                      "the loader reconstructs the quantized embedding from the packed weight")
        let embeddedAfter = embeddedReloaded(tokens)
        eval(embeddedBefore, embeddedAfter)
        XCTAssertEqual((embeddedBefore - embeddedAfter).abs().max().item(Float.self), 0,
                       "the embedding-quantized language model reloads exactly")
    }

    // The residency decision is arithmetic over the weights and the working set, so it is asserted
    // at fixed numbers: a quantized stack fits with the 4 GB activation/cache reserve, the
    // full-precision stack does not, and the budget is the working set rather than live free memory.
    func testTheResidencyPolicyIsDecidedFromTheWeights() throws {
        let budget = Int(25.0 * 0.85 * Double(1 << 30))          // an M1 Max's working-set budget
        XCTAssertTrue(NFKMLXMusicBackend.keepsStagesResident(weightBytes: 9 << 30,
                                                             workingSetBudget: budget))
        XCTAssertFalse(NFKMLXMusicBackend.keepsStagesResident(weightBytes: 28 << 30,
                                                              workingSetBudget: budget))
        XCTAssertFalse(NFKMLXMusicBackend.keepsStagesResident(weightBytes: 18 << 30,
                                                              workingSetBudget: budget),
                       "the reserve for activations and the CFG cache is part of the arithmetic")
        XCTAssertFalse(NFKMLXMusicBackend.keepsStagesResident(weightBytes: 1 << 30,
                                                              workingSetBudget: 0),
                       "an unreadable budget stages rather than gambling")
    }

    // The chained pipeline on the real release: prompt + lyrics in, a stereo 44.1 kHz WAV out. Each
    // stage's arithmetic is measured by its own parity record; a sampled song cannot be compared to
    // the reference bitwise (the random streams differ by construction), so this asserts what a
    // sampled run CAN promise — the stages connect, the clip respects the requested duration, and
    // the audio is real signal rather than silence or clipping.
    func testTheMusicBackendGeneratesAClipEndToEnd() throws {
        try requireMLXRuntime()
        guard let directory = config["IK_VAL_MUSIC3_DIR"],
              FileManager.default.fileExists(
                atPath: directory + "/language_model/model.safetensors.index.json") else {
            throw XCTSkip("set IK_VAL_MUSIC3_DIR to the full release (fetch.py --only MUSIC3)")
        }

        let backend = try NFKMLXMusic3.backend(directoryURL: URL(fileURLWithPath: directory))
        XCTAssertTrue(backend.isReady)
        let request = NFKInferenceRequest(
            inputs: [NFKInputPrompt: "Gentle ambient pad, warm and slow",
                     NFKInputLyrics: "[verse]\nHello world"],
            parameters: [NFKParameterDurationSeconds: 2.0,
                         NFKParameterSteps: 8,
                         NFKParameterSeed: 7])
        let result = try backend.runInference(for: request)

        let asset = try XCTUnwrap(result.output(forKey: NFKOutputAudio) as? NFKAudioAsset)
        XCTAssertEqual(asset.channelCount, 2)
        XCTAssertEqual(asset.sampleRate, 44_100)
        XCTAssertGreaterThan(asset.durationSeconds, 0)
        XCTAssertLessThanOrEqual(asset.durationSeconds, 2.1, "the duration bound holds")

        let fileURL = try XCTUnwrap(asset.fileURL)
        let (samples, sampleRate) = try XCTUnwrap(NFKMLXWaveFile.read(Data(contentsOf: fileURL)))
        XCTAssertEqual(sampleRate, 44_100)
        let squares = samples.reduce(Float(0)) { $0 + $1 * $1 }
        let rms = (squares / Float(max(samples.count, 1))).squareRoot()
        print("MUSIC3 END TO END: \(asset.durationSeconds)s, rms \(rms)")
        XCTAssertGreaterThan(rms, 1e-4, "the clip is signal, not silence")
        XCTAssertLessThan(rms, 0.9, "the clip is signal, not full-scale clipping")
        // Numbers cannot say whether it sounds like music; IK_MUSIC3_KEEP_CLIP names a path to keep
        // the WAV at for listening.
        if let keep = config["IK_MUSIC3_KEEP_CLIP"] {
            try? FileManager.default.removeItem(atPath: keep)
            try FileManager.default.copyItem(at: fileURL, to: URL(fileURLWithPath: keep))
        }
        try? FileManager.default.removeItem(at: fileURL)
    }

    // Numbers prove the stages are correct; only an ear proves the whole chain sounds like music.
    // This generates a full-length take that spans several DiT windows (so the multi-window stitch is
    // exercised on real content) and keeps the WAV for listening. It is opt-in — CI skips it — because
    // it is minutes long and the parity records are the ground truth for correctness. Overrides come
    // through the same config channel the validation suites use; the defaults are a real prompt.
    func testGenerateAListeningClip() throws {
        try requireMLXRuntime()
        try XCTSkipUnless(config["IK_MUSIC3_LISTEN"] == "1",
                          "set IK_MUSIC3_LISTEN=1 to generate a full-length clip for listening")

        // Prefer the resident quantized stack (near-transparent numerically, far faster); fall back to
        // an explicit dir or the full-precision release.
        let releaseURL: URL
        if let listen = config["IK_VAL_MUSIC3_LISTEN_DIR"] {
            releaseURL = URL(fileURLWithPath: listen)
        } else if FileManager.default.fileExists(
                    atPath: quantizedReleaseURL.appendingPathComponent("language_model/model.safetensors").path)
                    || FileManager.default.fileExists(
                    atPath: quantizedReleaseURL.appendingPathComponent("language_model/model.safetensors.index.json").path) {
            releaseURL = quantizedReleaseURL
        } else if let full = config["IK_VAL_MUSIC3_DIR"] {
            releaseURL = URL(fileURLWithPath: full)
        } else {
            throw XCTSkip("no music release found (quantize one, or set IK_VAL_MUSIC3_LISTEN_DIR)")
        }

        let duration = config["IK_MUSIC3_DURATION"].flatMap(Double.init) ?? 15.0
        let steps = config["IK_MUSIC3_STEPS"].flatMap(Int.init) ?? 30
        let seed = config["IK_MUSIC3_SEED"].flatMap(Int.init) ?? 7
        let prompt = config["IK_MUSIC3_PROMPT"]
            ?? "warm indie folk, fingerpicked acoustic guitar, soft brushed drums, gentle upright bass, "
             + "intimate female vocal, nostalgic and hopeful, 90 bpm"
        let lyrics = config["IK_MUSIC3_LYRICS"]
            ?? "[verse]\nMorning light across the kitchen floor\nCoffee cooling by the open door\n"
             + "[chorus]\nAnd we'll go slow, and we'll go far\nWherever the quiet roads are\n"

        let backend = try NFKMLXMusic3.backend(directoryURL: releaseURL)
        try XCTSkipUnless(backend.isReady, "the release at \(releaseURL.path) is incomplete")

        let request = NFKInferenceRequest(
            inputs: [NFKInputPrompt: prompt, NFKInputLyrics: lyrics],
            parameters: [NFKParameterDurationSeconds: duration,
                         NFKParameterSteps: steps,
                         NFKParameterSeed: seed])
        let started = Date()
        let result = try backend.runInference(for: request)
        let wallClock = Date().timeIntervalSince(started)

        let asset = try XCTUnwrap(result.output(forKey: NFKOutputAudio) as? NFKAudioAsset)
        XCTAssertEqual(asset.channelCount, 2)
        XCTAssertEqual(asset.sampleRate, 44_100)
        XCTAssertGreaterThan(asset.durationSeconds, 0)
        XCTAssertLessThanOrEqual(asset.durationSeconds, duration + 0.2, "the duration bound holds")

        let sourceURL = try XCTUnwrap(asset.fileURL)
        let (samples, _) = try XCTUnwrap(NFKMLXWaveFile.read(Data(contentsOf: sourceURL)))
        let squares = samples.reduce(Float(0)) { $0 + $1 * $1 }
        let rms = (squares / Float(max(samples.count, 1))).squareRoot()
        print("MUSIC3 LISTEN: \(asset.durationSeconds)s, rms \(rms), \(steps) steps, "
              + "seed \(seed), \(String(format: "%.1f", wallClock))s wall, release \(releaseURL.lastPathComponent)")
        XCTAssertGreaterThan(rms, 1e-4, "the clip is signal, not silence")
        XCTAssertLessThan(rms, 0.9, "the clip is signal, not full-scale clipping")

        let keepPath = config["IK_MUSIC3_KEEP_CLIP"]
            ?? FileManager.default.homeDirectoryForCurrentUser
                .appendingPathComponent(".inferkit-validation/music3-listen.wav").path
        try? FileManager.default.removeItem(atPath: keepPath)
        try FileManager.default.copyItem(at: sourceURL, to: URL(fileURLWithPath: keepPath))
        try? FileManager.default.removeItem(at: sourceURL)
    }

    /// The cached quantized release. The directory name carries the recipe, so changing the
    /// defaults regenerates rather than silently reusing a stale packing.
    private var quantizedReleaseURL: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".inferkit-validation/minimax-music3-q4lm-q4emb-q8dit")
    }

    /// Quantizes the release once (4-bit language model incl. its input embedding, 4-bit depth
    /// decoder, 8-bit DiT, group 64) into the durable validation store; later runs reuse it. Skips
    /// when the full release is absent.
    private func quantizedRelease() throws -> URL {
        guard let directory = config["IK_VAL_MUSIC3_DIR"],
              FileManager.default.fileExists(
                atPath: directory + "/language_model/model.safetensors.index.json") else {
            throw XCTSkip("set IK_VAL_MUSIC3_DIR to the full release (fetch.py --only MUSIC3)")
        }
        let destination = quantizedReleaseURL
        if !FileManager.default.fileExists(
            atPath: destination.appendingPathComponent("language_model/model.safetensors").path) {
            try NFKMLXMusic3.quantizeRelease(at: URL(fileURLWithPath: directory), to: destination)
            NFKMLXGPU.clearCache()
        }
        return destination
    }

    // The quantization cost is MEASURED, not assumed, against the same records the full-precision
    // parity runs use: the 4-bit DiT's velocity and the 4-bit language model's prefill and logits.
    // The floors are set from the first measured run; the printed numbers are the record.
    func testTheQuantizedReleaseMeasuresItsCostAgainstTheRecords() throws {
        try requireMLXRuntime()
        let release = try quantizedRelease()
        guard let ditRecord = config["IK_PARITY_MUSIC3_DIT"],
              let arRecord = config["IK_PARITY_MUSIC3_AR"] else {
            throw XCTSkip("set IK_PARITY_MUSIC3_DIT and IK_PARITY_MUSIC3_AR")
        }
        func flatCosine(_ a: MLXArray, _ b: MLXArray) -> Double {
            cosine(a.reshaped([-1]).asType(.float32).asArray(Float.self).map(Double.init),
                   b.reshaped([-1]).asType(.float32).asArray(Float.self).map(Double.init))
        }

        let ditArrays = try loadArrays(url: URL(fileURLWithPath: ditRecord))
        let latents = try XCTUnwrap(ditArrays["latents"]).transposed(1, 0).expandedDimensions(axis: 0)
        let condition = try XCTUnwrap(ditArrays["condition"]).expandedDimensions(axis: 0)
        let transformer = NFKMLXMusic3.makeDiT()
        try NFKMLXMusic3.loadDiTWeights(into: transformer,
                                        from: release.appendingPathComponent("transformer"))
        let velocity = transformer.velocity(latents: latents, timestep: MLXArray([Float(0.5)]),
                                            condition: condition)
        eval(velocity)
        let ditSimilarity = flatCosine(velocity[0].transposed(1, 0),
                                       try XCTUnwrap(ditArrays["velocity_tmid"]))
        print("QUANTIZED music3-dit 8-bit velocity: cosine \(ditSimilarity)")
        // The DiT is the quantization-sensitive stage — 4-bit measured 0.9775, which is why the
        // recipe holds it at 8-bit.
        XCTAssertGreaterThan(ditSimilarity, 0.99, "the 8-bit DiT stays close to the reference")
        NFKMLXGPU.clearCache()

        let arArrays = try loadArrays(url: URL(fileURLWithPath: arRecord))
        let textIDs = try XCTUnwrap(arArrays["text_ids"])
        let languageDirectory = release.appendingPathComponent("language_model")
        let configuration = try NFKMLXLanguage.configuration(
            fromHuggingFace: languageDirectory.appendingPathComponent("config.json"))
        let language = NFKMLXLanguage.makeNet(configuration)
        try NFKMLXLanguage.loadWeights(into: language, fromDirectory: languageDirectory,
                                       precision: .checkpoint)
        let cache = NFKMLXKeyValueCache(layerCount: configuration.layerCount)
        var lastHidden = language.hiddenStates(fromEmbeddings: language.embed(textIDs), cache: cache)
        lastHidden = lastHidden[0..., lastHidden.shape[1] - 1]
        let logits = language.logits(fromHidden: lastHidden).asType(.float32)
        eval(logits)
        let prefillSimilarity = flatCosine(lastHidden, try XCTUnwrap(arArrays["prefill_hidden"]))
        let logitsSimilarity = flatCosine(logits, try XCTUnwrap(arArrays["first_logits"]))
        print("QUANTIZED music3-lm 4-bit prefill: cosine \(prefillSimilarity)")
        print("QUANTIZED music3-lm 4-bit first logits: cosine \(logitsSimilarity)")
        XCTAssertGreaterThan(prefillSimilarity, 0.98)
        XCTAssertGreaterThan(logitsSimilarity, 0.999)
    }

    // A measurement, not a recipe change. The quantized DiT defaults to 8-bit because the flow field
    // is the quantization-sensitive stage (4-bit measured 0.9775 against the record). This sweeps the
    // DiT alone at 4/6/8 bits against the same velocity record, so the trade between stack size and
    // the flow field's fidelity is on numbers before any default moves. Opt-in — it reloads the
    // 9.7 GB float DiT once per bit width — so CI skips it.
    func testTheDiTQuantizationBitWidthSweep() throws {
        try requireMLXRuntime()
        try XCTSkipUnless(config["IK_MUSIC3_DIT_SWEEP"] == "1",
                          "set IK_MUSIC3_DIT_SWEEP=1 to sweep the DiT quantization bit width")
        guard let full = config["IK_VAL_MUSIC3_DIR"],
              let ditRecord = config["IK_PARITY_MUSIC3_DIT"],
              FileManager.default.fileExists(atPath: ditRecord),
              FileManager.default.fileExists(atPath: full + "/transformer") else {
            throw XCTSkip("set IK_VAL_MUSIC3_DIR (full release) and IK_PARITY_MUSIC3_DIT")
        }
        let transformerURL = URL(fileURLWithPath: full).appendingPathComponent("transformer")
        func flatCosine(_ a: MLXArray, _ b: MLXArray) -> Double {
            cosine(a.reshaped([-1]).asType(.float32).asArray(Float.self).map(Double.init),
                   b.reshaped([-1]).asType(.float32).asArray(Float.self).map(Double.init))
        }
        let ditArrays = try loadArrays(url: URL(fileURLWithPath: ditRecord))
        let latents = try XCTUnwrap(ditArrays["latents"]).transposed(1, 0).expandedDimensions(axis: 0)
        let condition = try XCTUnwrap(ditArrays["condition"]).expandedDimensions(axis: 0)
        let reference = try XCTUnwrap(ditArrays["velocity_tmid"])

        func measure(bits: Int?) throws -> Double {
            let transformer = NFKMLXMusic3.makeDiT()
            try NFKMLXMusic3.loadDiTWeights(into: transformer, from: transformerURL)
            if let bits {
                NFKMLXQuantization.quantize(module: transformer, bits: bits, groupSize: 64)
            }
            let velocity = transformer.velocity(latents: latents, timestep: MLXArray([Float(0.5)]),
                                                condition: condition)
            eval(velocity)
            let similarity = flatCosine(velocity[0].transposed(1, 0), reference)
            NFKMLXGPU.clearCache()
            return similarity
        }

        let baseline = try measure(bits: nil)
        print("DIT SWEEP float32 baseline velocity: cosine \(baseline)")
        XCTAssertGreaterThan(baseline, 0.9999, "the float DiT reproduces the record")

        // Floors are set from the first measured run; the printed numbers are the record. 6-bit sits
        // an order of magnitude below 8-bit and saves ~0.6 GB on the DiT, which does not justify
        // moving the default off 8-bit for a stage whose error compounds over the sampling loop.
        let floors: [Int: Double] = [8: 0.9995, 6: 0.995, 4: 0.97]
        for bits in [8, 6, 4] {
            let similarity = try measure(bits: bits)
            print("DIT SWEEP \(bits)-bit velocity: cosine \(similarity)")
            XCTAssertGreaterThan(similarity, floors[bits]!, "\(bits)-bit DiT held its measured floor")
        }
    }

    // The largest tensor left unquantized in the music stack is the LM's input embedding
    // (embed_tokens, 200000 x 4096 bf16 = 1.6 GiB). The LM is UNTIED — its lm_head is a separate
    // Linear already packed to 4-bit — so quantizing the embedding here affects only the input
    // representation, not the output projection, and is measurable against the AR record. This probes
    // 4-bit and 8-bit embedding against the same prefill/logits record the recipe already uses, so the
    // stack-size saving is weighed against the LM's fidelity before the embedding joins the recipe.
    // Opt-in: it loads the 16 GiB bf16 LM once per variant.
    func testTheEmbeddingQuantizationCostAgainstTheARRecord() throws {
        try requireMLXRuntime()
        try XCTSkipUnless(config["IK_MUSIC3_EMB_PROBE"] == "1",
                          "set IK_MUSIC3_EMB_PROBE=1 to probe quantizing the LM input embedding")
        guard let full = config["IK_VAL_MUSIC3_DIR"], let arRecord = config["IK_PARITY_MUSIC3_AR"],
              FileManager.default.fileExists(atPath: arRecord),
              FileManager.default.fileExists(
                atPath: full + "/language_model/model.safetensors.index.json") else {
            throw XCTSkip("set IK_VAL_MUSIC3_DIR (full release) and IK_PARITY_MUSIC3_AR")
        }
        func flatCosine(_ a: MLXArray, _ b: MLXArray) -> Double {
            cosine(a.reshaped([-1]).asType(.float32).asArray(Float.self).map(Double.init),
                   b.reshaped([-1]).asType(.float32).asArray(Float.self).map(Double.init))
        }
        let arArrays = try loadArrays(url: URL(fileURLWithPath: arRecord))
        let textIDs = try XCTUnwrap(arArrays["text_ids"])
        let languageDirectory = URL(fileURLWithPath: full).appendingPathComponent("language_model")
        let configuration = try NFKMLXLanguage.configuration(
            fromHuggingFace: languageDirectory.appendingPathComponent("config.json"))

        // linearBits fixed at the recipe's 4; embeddingBits nil = leave the embedding unquantized
        // (the shipped recipe), else quantize embed_tokens at that width in a second pass.
        func measure(embeddingBits: Int?) throws -> (prefill: Double, logits: Double) {
            let language = NFKMLXLanguage.makeNet(configuration)
            try NFKMLXLanguage.loadWeights(into: language, fromDirectory: languageDirectory,
                                           precision: .checkpoint)
            MLXNN.quantize(model: language, groupSize: 64, bits: 4) { _, layer in
                guard let linear = layer as? Linear, !(linear is QuantizedLinear) else { return false }
                return linear.weight.shape[1] % 64 == 0
            }
            if let embeddingBits {
                MLXNN.quantize(model: language, groupSize: 64, bits: embeddingBits) { _, layer in
                    guard let embedding = layer as? Embedding,
                          !(embedding is QuantizedEmbedding) else { return false }
                    return embedding.weight.shape[1] % 64 == 0
                }
            }
            let cache = NFKMLXKeyValueCache(layerCount: configuration.layerCount)
            var lastHidden = language.hiddenStates(fromEmbeddings: language.embed(textIDs), cache: cache)
            lastHidden = lastHidden[0..., lastHidden.shape[1] - 1]
            let logits = language.logits(fromHidden: lastHidden).asType(.float32)
            eval(logits)
            let result = (flatCosine(lastHidden, try XCTUnwrap(arArrays["prefill_hidden"])),
                          flatCosine(logits, try XCTUnwrap(arArrays["first_logits"])))
            NFKMLXGPU.clearCache()
            return result
        }

        let base = try measure(embeddingBits: nil)
        print("EMB PROBE embedding bf16 (shipped): prefill \(base.prefill), logits \(base.logits)")
        for bits in [8, 4] {
            let measured = try measure(embeddingBits: bits)
            // A [200000, 4096] embedding at `bits`, group 64: packed weight + bf16 scales/biases.
            let packed = 200_000.0 * 4096.0 * Double(bits) / 8.0
            let scales = 200_000.0 * (4096.0 / 64.0) * 2.0 * 2.0
            let savedGiB = (1_638_400_000.0 - (packed + scales)) / Double(1 << 30)
            print(String(format: "EMB PROBE embedding %d-bit: prefill %.7f, logits %.7f, saves %.2f GiB",
                         bits, measured.prefill, measured.logits, savedGiB))
        }
        // The shipped recipe holds a 0.9995 logits floor at 4-bit Linear; the embedding must not drop
        // it materially to be worth the space.
        XCTAssertGreaterThan(base.logits, 0.999, "the baseline reproduces the AR record")
    }

    // The quantized stack is what makes residency possible: the whole release fits the working set,
    // so the second run skips every load. Both runs must still produce real audio.
    func testTheMusicBackendKeepsAQuantizedStackResident() throws {
        try requireMLXRuntime()
        let release = try quantizedRelease()
        let weightBytes = NFKMLXMusicBackend.stackWeightBytes(in: release)
        print("QUANTIZED music3 stack on disk: \(Double(weightBytes) / Double(1 << 30)) GiB")

        let backend = NFKMLXMusicBackend(directoryURL: release)
        let request = NFKInferenceRequest(
            inputs: [NFKInputPrompt: "Gentle ambient pad, warm and slow",
                     NFKInputLyrics: "[verse]\nHello world"],
            parameters: [NFKParameterDurationSeconds: 2.0,
                         NFKParameterSteps: 8,
                         NFKParameterSeed: 7])

        let firstStart = Date()
        let first = try backend.runInference(for: request)
        let firstSeconds = Date().timeIntervalSince(firstStart)
        let secondStart = Date()
        let second = try backend.runInference(for: request)
        let secondSeconds = Date().timeIntervalSince(secondStart)
        print("QUANTIZED music3 e2e: first \(firstSeconds)s, second \(secondSeconds)s, "
              + "resident \(backend.isHoldingStagesResident)")

        for result in [first, second] {
            let asset = try XCTUnwrap(result.output(forKey: NFKOutputAudio) as? NFKAudioAsset)
            XCTAssertEqual(asset.channelCount, 2)
            XCTAssertGreaterThan(asset.durationSeconds, 0)
            if let fileURL = asset.fileURL { try? FileManager.default.removeItem(at: fileURL) }
        }
        XCTAssertTrue(backend.isHoldingStagesResident,
                      "a 4-bit stack fits the working set, so the stages stay loaded")
    }

    // The AR stage is a Qwen3-8B over a 200000-token vocabulary with an untied head. The count is
    // checked at a fixed synthetic budget so the arithmetic is asserted machine-independently, and
    // the real machine's verdicts are reported for the precision decision the port must make.
    func testTheMusicLanguageStageIsSizedBeforeLoading() throws {
        let geometry = NFKMLXLanguageConfiguration(
            hiddenSize: 4096, layerCount: 36, headCount: 32, keyValueHeadCount: 8,
            headDimensions: 128, intermediateSize: 12288, vocabularySize: 200_000,
            ropeTheta: 1_000_000, tiesWordEmbeddings: false, normalizesQueryAndKey: true)

        let parameters = NFKMLXModelSizing.parameterCount(of: geometry)
        XCTAssertEqual(parameters, 8_584_475_648, "the geometry counts to ~8.6B parameters")

        // 24 GiB reference budget: 16-bit weights (16.0 GiB) fit, float32 (32.0 GiB) cannot.
        let referenceBudget = 24 << 30
        XCTAssertFalse(NFKMLXModelSizing.fit(of: geometry, precision: .float32,
                                             budget: referenceBudget).fits)
        XCTAssertTrue(NFKMLXModelSizing.fit(of: geometry, tokens: 1,
                                            precision: .checkpoint, budget: referenceBudget).fits)

        // The model's own position budget is 10240, and the CFG pair doubles the cache: every step
        // runs a conditional and an unconditional row, so the plan asks for 2 × 10240 positions.
        for (name, precision) in [("float32", NFKMLXWeightPrecision.float32), ("16-bit", .checkpoint)] {
            let fit = NFKMLXModelSizing.fit(of: geometry, tokens: 2 * 10_240, precision: precision)
            print("MUSIC3 SIZING language stage at \(name): \(fit.describedFit)")
        }
    }
}
