//
//  MLXCustomizationExamples.swift
//  InferKitMLXExamples
//
//  The customization snippets from Docs/examples.md, compiled against the package's PUBLIC surface.
//
//  This file imports InferKitMLX without `@testable` deliberately. A fine-tuning recipe a consumer
//  cannot call is not shipped, and the whole path here — build a network, train it, write a
//  checkpoint, reload it through the model's own factory — has to hold together for an app that
//  links the package like any other dependency. Compiling it that way is what keeps a customization
//  API from drifting back behind `internal`.
//

import XCTest
import CoreGraphics
import InferKit
import MLX
import MLXOptimizers
import InferKitMLX

final class MLXCustomizationExamples: XCTestCase {

    // Docs/examples.md: Retargeting SAM 2 onto a consumer's own subject
    func testExampleRetargetingSAM2OnOwnMasks() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
        let tuned = FileManager.default.temporaryDirectory
            .appendingPathComponent("sam2-example-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: tuned) }

        // Nil weights builds the released geometry at its random initialization; a real run passes a
        // released checkpoint here. The example uses a small one so it stays a test.
        let net = NFKMLXSAM2TrackerNet(
            NFKMLXSAM2TrackerConfiguration(
                encoder: NFKMLXSAM2Configuration(embedDimensions: 12, heads: 1, stages: [1, 1, 1, 1],
                                                 windowSpec: [2, 2, 2, 2], globalAttentionBlocks: [3],
                                                 backgroundWindow: 2),
                imageSize: 64, occlusionSpatialEmbedding: true,
                temporalPositionEncodingForObjectPointers: true))

        let frames = [Self.clickedFrame()]
        let history = try NFKMLXSAM2.fineTune(net, examples: { frames[$0 % frames.count] },
                                              trainable: .maskDecoder, steps: 4)
        XCTAssertEqual(history.count, 4)

        try NFKMLXWeights.save(net, to: tuned)
        let backend = try NFKMLXSAM2.backend(variant: .tiny, release: .sam21, weightsURL: nil)
        XCTAssertTrue(backend.isReady, "the fine-tuned checkpoint loads through the shipped factory")
    }

    // Docs/examples.md: Adapting Qwen3-VL retrieval to a consumer's own corpus
    func testExampleAdaptingQwen3VLRetrievalOnOwnPairs() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
        let tuned = FileManager.default.temporaryDirectory
            .appendingPathComponent("qwen3vl-adapter-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: tuned) }

        // What the released embedder produced for a batch of query and document pairs. A real run
        // computes these once with embedding(forText:) or embedding(forImage:text:instruction:) and
        // reuses them at every step: the 2B backbone never enters the training graph.
        let width = 16
        let queries = MLXArray((0 ..< 4 * width).map { Float(sin(Double($0) * 0.7)) })
            .reshaped([4, width])
        let documents = MLXArray((0 ..< 4 * width).map { Float(cos(Double($0) * 0.7)) })
            .reshaped([4, width])

        let adapter = NFKMLXQwen3VLEmbeddingAdapter(dimensions: width)
        let objective = NFKMLXQwen3VLEmbeddingObjective()
        let history = try NFKMLXTrainer.train(
            adapter, optimizer: Adam(learningRate: 1e-2), steps: 8,
            batch: { _ in (queries, documents.reshaped([1, 4, width])) },
            loss: { model, q, d in objective(model, queries: q, documents: d) },
            clipGradientNorm: 1)
        XCTAssertEqual(history.count, 8)

        try NFKMLXWeights.save(adapter, to: tuned)

        // With a release directory the whole run is one call, and the saved adapter installs on the
        // model so every later embedding is the adapted one.
        if let directory = ProcessInfo.processInfo.environment["IK_VAL_QWEN3_VL_EMBEDDING"] {
            let embedder = try NFKMLXQwen3VLEmbedder.embedder(
                directoryURL: URL(fileURLWithPath: directory))
            let releaseAdapter = try embedder.makeAdapter()
            try embedder.fineTune(adapter: releaseAdapter, queries: queries, documents: [documents],
                                  steps: 2)
            try NFKMLXWeights.save(releaseAdapter, to: tuned)
            try embedder.loadAdapter(from: tuned)
            XCTAssertEqual(embedder.embedding(forText: "a query").count, embedder.embeddingDimensions)
        }
    }

    // Docs/examples.md: Retargeting the Qwen3-VL reranker onto a consumer's own relevance
    func testExampleRetargetingQwen3VLRerankerOnOwnLabels() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
        let width = 12
        // The last-position hidden state of each labeled pair, which a real run reads once from the
        // frozen backbone. A head built from a release starts at makeHead(), which is the release's
        // own scoring direction rather than a random one.
        let hidden = MLXArray((0 ..< 6 * width).map { Float(sin(Double($0) * 0.3)) })
            .reshaped([6, width])
        let labels = MLXArray([Float(1), 0, 1, 0, 1, 0])

        let head = NFKMLXQwen3VLRerankerHead(dimensions: width)
        let objective = NFKMLXQwen3VLRerankerObjective()
        let history = try NFKMLXTrainer.train(
            head, optimizer: Adam(learningRate: 5e-2), steps: 12,
            batch: { _ in (hidden, labels) },
            loss: { model, states, targets in objective(model, hidden: states, labels: targets) },
            clipGradientNorm: 1)
        XCTAssertLessThan(history[history.count - 1], history[0], "the binary loss falls")
    }

    // Docs/examples.md: Retargeting SAM 3's detector onto a consumer's own instances
    func testExampleRetargetingSAM3OnOwnBoxes() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
        let detector = NFKMLXSAM3DetectorNet(
            NFKMLXSAM3DetectorConfiguration(hiddenSize: 32, intermediateSize: 48, heads: 2,
                                            encoderLayers: 1, decoderLayers: 2, queryCount: 8))

        // What the frozen encoders produced for one image, which a real run gets from
        // NFKMLXSAM3.encode(image:tokens:valid:using:) once and reuses at every step.
        let sides = [16, 8, 4]
        let levels = sides.map { MLXArray.zeros([1, $0, $0, 32]) + 0.1 }
        let positions = NFKMLXSAM3.positionEncodings(for: levels, width: 32)
        let prompt = MLXArray.zeros([1, 4, 32]) + 0.05
        let valid = MLXArray([Float(1), 1, 0, 0]).reshaped([1, 4])
        let boxes = MLXArray([Float(0.3), 0.3, 0.2, 0.2]).reshaped([1, 4])

        let history = try NFKMLXSAM3.fineTune(
            detector, examples: { _ in (levels, positions, prompt, valid, boxes) },
            trainable: .decoder, steps: 3)
        XCTAssertEqual(history.count, 3)

        let tuned = FileManager.default.temporaryDirectory
            .appendingPathComponent("sam3-example-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: tuned) }
        try NFKMLXWeights.save(detector, to: tuned)
        XCTAssertTrue(FileManager.default.fileExists(atPath: tuned.path))
    }

    // Docs/examples.md: Customizing a model on a consumer's own data
    func testExampleFineTuningRoundTripsThroughTheShippedFactory() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
        let tuned = FileManager.default.temporaryDirectory
            .appendingPathComponent("zerodce-example-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: tuned) }

        let net = try NFKMLXZeroDCE.network(weightsURL: nil)
        var objective = NFKMLXZeroDCEObjective()
        objective.wellExposedLevel = 0.65

        let myPhotos = [Self.darkPhotos()]
        let history = try NFKMLXZeroDCE.fineTune(net, photos: { myPhotos[$0 % myPhotos.count] },
                                                 objective: objective, steps: 4,
                                                 checkpoint: NFKMLXTrainingCheckpoint(url: tuned,
                                                                                      everySteps: 2)) { _ in
            true                                            // return false to end the run early
        }
        XCTAssertEqual(history.count, 4)

        try NFKMLXWeights.save(net, to: tuned)
        let backend = try NFKMLXZeroDCE.backend(weightsURL: tuned)
        XCTAssertEqual(backend.backendIdentifier, NFKMLXZeroDCE.modelName)
    }

    // Docs/examples.md: Adapting a Cosmos Tokenizer to your own footage
    func testExampleAdaptingACosmosTokenizerToOwnFootage() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
        let tuned = FileManager.default.temporaryDirectory
            .appendingPathComponent("cosmos-example-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: tuned) }

        // A real run passes the release's autoencoder.jit and the VGG-16 ImageNet weights
        // (timm/vgg16.tv_in1k); nil builds both at random initialization so the example needs no download.
        let net = try NFKMLXCosmosTokenizer.network(variant: .continuousVideo4x8x8, weightsURL: nil)
        let objective = try NFKMLXCosmosTokenizerObjective(vggWeightsURL: nil)

        let myClip = MLX.clip(MLXRandom.normal([1, 5, 32, 32, 3]) * 0.3, min: -1, max: 1)  // [B, T, H, W, 3]
        let history = try NFKMLXCosmosTokenizer.fineTune(net, examples: { _ in myClip }, trainable: .decoder,
                                                         objective: objective, steps: 2)
        XCTAssertEqual(history.count, 2)

        try NFKMLXWeights.save(net, to: tuned)
        let backend = try NFKMLXCosmosTokenizer.backend(variant: .continuousVideo4x8x8, weightsURL: tuned)
        XCTAssertEqual(backend.backendIdentifier, NFKMLXCosmosTokenizerVariant.continuousVideo4x8x8.modelName)
    }

    // Docs/examples.md: Teaching Sa2VA your own referring segmentation
    func testExampleTeachingSa2VAOwnSegmentation() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
        // The mask terms score each [SEG]'s low-resolution mask logits against its mask, on points drawn
        // where the prediction is least certain.
        let objective = NFKMLXSa2VAObjective()
        let logits = MLXRandom.normal([5, 256, 256])
        let targets = NFKMLXSa2VAObjective.resizedTargets(MLXArray.zeros([5, 448, 448]), side: 256)
        let terms = objective.maskTerms(masks: logits, targets: targets, points: objective.uncertainPoints(masks: logits))
        XCTAssertGreaterThan(terms.mask.item(Float.self), 0)
        XCTAssertEqual(NFKMLXLearningRateSchedule.mmengineWarmupCosine(steps: 100).multiplier(4), 1)

        // With a release directory: tile the image, write the turn in the release's template, mask the
        // prompt out of the labels, and give one mask per [SEG].
        if let directory = ProcessInfo.processInfo.environment["IK_VAL_SA2VA_1B"] {
            let release = URL(fileURLWithPath: directory)
            let tuned = FileManager.default.temporaryDirectory.appendingPathComponent("sa2va-\(UUID().uuidString)")
            defer { try? FileManager.default.removeItem(at: tuned) }
            let net = try NFKMLXSa2VA.network(directoryURL: release)
            let tokenizer = try XCTUnwrap(NFKMLXSa2VA.tokenizer(inDirectory: release))
            let photo = Self.gray(448, level: 90)
            let tiles = try NFKMLXSa2VAProcessor.dynamicTiles(photo, side: net.configuration.imageSize)
            let prompt = NFKMLXSa2VAProcessor.promptText("<image>Please segment the gray square.",
                                                         imageTokens: tiles.count * net.configuration.tokensPerTile,
                                                         template: net.configuration.template)
            let promptIds = tokenizer.encode(prompt).map(\.int32Value)
            let answerIds = tokenizer.encode("Sure, [SEG].<|im_end|>").map(\.int32Value)
            let example = NFKMLXSa2VAExample(
                pixelValues: NFKMLXSa2VAProcessor.tilePixels(tiles, side: net.configuration.imageSize),
                inputIds: MLXArray(promptIds + answerIds),
                labels: MLXArray([Int32](repeating: -100, count: promptIds.count) + answerIds),
                groundingImage: try NFKMLXSa2VAProcessor.groundingPixels(photo).transposed(0, 2, 3, 1),
                masks: MLXArray.ones([1, 448, 448]))
            try NFKMLXSa2VA.fineTune(net, examples: { _ in example }, rank: 8, alpha: 16, steps: 1)
            try NFKMLXLoRA.merge(into: net)
            try NFKMLXSa2VA.save(net, toDirectoryURL: tuned, release: release)
            XCTAssertNoThrow(try NFKMLXSa2VA.backend(directoryURL: tuned))
        }
    }

    // Docs/examples.md: Adapting Florence-2 to your own task
    func testExampleAdaptingFlorence2OnOwnAnswers() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
        // The objective scores teacher-forced logits against the answer ids, `<s> … </s>`.
        let answer = MLXArray([Int32(0), 250, 7, 2])
        let logits = MLXRandom.normal([1, 4, 51289])
        XCTAssertGreaterThan(NFKMLXFlorence2Objective().loss(logits: logits, target: answer).item(Float.self), 0)

        // With a release directory: adapt, merge, save beside the release's tokenizer, and reload.
        if let directory = ProcessInfo.processInfo.environment["IK_VAL_FLORENCE2_BASE_FT"] {
            let release = URL(fileURLWithPath: directory)
            let tuned = FileManager.default.temporaryDirectory.appendingPathComponent("florence2-\(UUID().uuidString)")
            defer { try? FileManager.default.removeItem(at: tuned) }
            let net = try NFKMLXFlorence2.network(directoryURL: release)
            let tokenizer = try XCTUnwrap(NFKMLXFlorence2Processor.tokenizer(inDirectory: release))
            let pixels = try NFKMLXFlorence2Processor.pixelValues(Self.gray(96, level: 120))
            let prompt = NFKMLXFlorence2Processor.encodePrompt(NFKMLXFlorence2Processor.expandPrompt("<CAPTION>"),
                                                               tokenizer: tokenizer, eosTokenId: 2).reshaped([-1])
            let target = NFKMLXFlorence2Processor.encodePrompt("A gray square.", tokenizer: tokenizer, eosTokenId: 2).reshaped([-1])
            try NFKMLXFlorence2.fineTune(net, examples: { _ in (pixels, prompt, target) }, steps: 1)
            try NFKMLXLoRA.merge(into: net)
            try NFKMLXFlorence2.save(net, toDirectoryURL: tuned, release: release)
            XCTAssertNoThrow(try NFKMLXFlorence2.backend(directoryURL: tuned))
        }
    }

    // Docs/examples.md: Retargeting a table detector to your own document classes
    func testExampleRetargetingTableTransformerOnOwnClasses() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
        // A small transformer over the ResNet-18 backbone keeps the example free of downloads; a real run
        // builds it with NFKMLXTableTransformer.network(directoryURL:labels:) from a release.
        var configuration = NFKMLXTableTransformerConfiguration()
        configuration.dModel = 32
        configuration.encoderLayers = 1
        configuration.decoderLayers = 1
        configuration.encoderAttentionHeads = 2
        configuration.decoderAttentionHeads = 2
        configuration.encoderFFNDim = 64
        configuration.decoderFFNDim = 64
        configuration.numQueries = 6
        configuration.numLabels = 3
        configuration.labels = ["table", "figure", "chart"]
        let net = NFKMLXTableTransformerNet(configuration)

        let page = try NFKMLXTableTransformerProcessor.pixelValues(Self.gray(96, level: 230), sizing: .longestEdge(64))
        let boxes = MLXArray([Float(0), 0.5, 0.4, 0.8, 0.5, 2, 0.5, 0.85, 0.6, 0.2]).reshaped([2, 5])
        let history = try NFKMLXTableTransformer.fineTune(net, examples: { _ in (page, boxes) }, steps: 2,
                                                          stepsPerEpoch: 1)
        XCTAssertEqual(history.count, 2)

        if let directory = ProcessInfo.processInfo.environment["IK_VAL_TABLE_TRANSFORMER_DETECTION"] {
            let release = URL(fileURLWithPath: directory)
            let tuned = FileManager.default.temporaryDirectory.appendingPathComponent("tatr-\(UUID().uuidString)")
            defer { try? FileManager.default.removeItem(at: tuned) }
            let retargeted = try NFKMLXTableTransformer.network(directoryURL: release, labels: configuration.labels)
            try NFKMLXTableTransformer.fineTune(retargeted, examples: { _ in (page, boxes) }, trainable: .heads, steps: 1)
            try NFKMLXTableTransformer.save(retargeted, toDirectoryURL: tuned, release: release)
            XCTAssertNoThrow(try NFKMLXTableTransformer.backend(directoryURL: tuned))
        }
    }

    // Docs/examples.md: Adapting a handwriting reader to your own writing
    func testExampleAdaptingTrOCROnOwnLines() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
        // A tiny network in the released layout keeps the example free of downloads; a real run builds it
        // with NFKMLXTrOCR.network(directoryURL:) from a release such as microsoft/trocr-small-handwritten.
        let vision = NFKMLXTrOCRVisionConfiguration(imageSize: 32, hiddenSize: 32, layers: 1, heads: 2,
                                                    intermediateSize: 64, qkvBias: true)
        let language = NFKMLXSeq2SeqConfiguration(
            vocabularySize: 40, dModel: 32, encoderLayers: 0, decoderLayers: 1, heads: 2, encoderFFDim: 64,
            decoderFFDim: 64, maxPositions: 64, activation: .relu, positions: .learned, normalizeBefore: false,
            finalLayerNorm: false, layerNormEmbedding: true, scaleEmbedding: true, finalLogitsBias: false,
            padTokenId: 1, eosTokenId: 2, decoderStartTokenId: 2, crossAttentionWidth: 32)
        let net = NFKMLXTrOCRNet(vision: vision, language: language)

        let line = try NFKMLXTrOCRProcessor.pixelValues(Self.gray(64, level: 30), side: 32)
        let transcription = MLXArray([Int32(7), 12, 19, 2])       // targetIds(for:tokenizer:endToken:)
        let history = try NFKMLXTrOCR.fineTune(net, examples: { _ in (line, transcription) }, steps: 3)
        XCTAssertEqual(history.count, 3)

        // With a release directory, the tuned network saves beside the release's own tokenizer and
        // configuration, and the ordinary factory reads it.
        if let directory = ProcessInfo.processInfo.environment["IK_VAL_TROCR_SMALL_HANDWRITTEN"] {
            let release = URL(fileURLWithPath: directory)
            let tuned = FileManager.default.temporaryDirectory.appendingPathComponent("trocr-\(UUID().uuidString)")
            defer { try? FileManager.default.removeItem(at: tuned) }
            let released = try NFKMLXTrOCR.network(directoryURL: release)
            let tokenizer = try XCTUnwrap(NFKMLXTrOCRProcessor.tokenizer(inDirectory: release))
            let pixels = try NFKMLXTrOCRProcessor.pixelValues(Self.gray(96, level: 200), bicubic: true)
            let target = NFKMLXTrOCRProcessor.targetIds(for: "hello", tokenizer: tokenizer, endToken: 2)
            try NFKMLXTrOCR.fineTune(released, examples: { _ in (pixels, target) }, trainable: .decoder, steps: 1)
            try NFKMLXTrOCR.save(released, toDirectoryURL: tuned, release: release)
            XCTAssertEqual(try NFKMLXTrOCR.backend(directoryURL: tuned).backendIdentifier, NFKMLXTrOCR.modelName)
        }
    }

    // Docs/examples.md: Training a V-JEPA 2 probe on your own clips
    func testExampleTrainingAVJEPA2ProbeOnOwnClips() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
        let release = FileManager.default.temporaryDirectory.appendingPathComponent("vjepa2-release-\(UUID().uuidString)")
        let tuned = FileManager.default.temporaryDirectory.appendingPathComponent("vjepa2-probe-\(UUID().uuidString)")
        defer {
            try? FileManager.default.removeItem(at: release)
            try? FileManager.default.removeItem(at: tuned)
        }
        // A real run passes a downloaded release such as facebook/vjepa2-vitl-fpc64-256; a tiny encoder
        // written in the same layout keeps the example free of downloads.
        var tiny = NFKMLXVJEPA2Configuration()
        tiny.hiddenSize = 64
        tiny.numHiddenLayers = 2
        tiny.numAttentionHeads = 4
        tiny.framesPerClip = 2
        tiny.cropSize = 32
        try NFKMLXVJEPA2.save(NFKMLXVJEPA2Net(tiny), toDirectoryURL: release)

        let net = try NFKMLXVJEPA2.network(directoryURL: release, labels: ["pour", "stir"])
        let pouring = try NFKMLXVJEPA2Processor.clip(frames: [Self.gray(48, level: 40), Self.gray(48, level: 90)],
                                                     configuration: net.configuration)
        let stirring = try NFKMLXVJEPA2Processor.clip(frames: [Self.gray(48, level: 200), Self.gray(48, level: 150)],
                                                      configuration: net.configuration)
        let myClips = [(pouring, 0), (stirring, 1)]
        let history = try NFKMLXVJEPA2.fineTune(net, examples: { myClips[$0 % myClips.count] }, steps: 4)
        XCTAssertEqual(history.count, 4)

        try NFKMLXVJEPA2.save(net, toDirectoryURL: tuned)
        let backend = try NFKMLXVJEPA2.backend(directoryURL: tuned)
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: Self.gray(48, level: 60)]))
        let classes = result.output(forKey: NFKOutputClassifications) as? [NFKClassification]
        XCTAssertEqual(Set(classes?.map(\.label) ?? []), ["pour", "stir"])
    }

    // Docs/examples.md: Adapting a decision model to your own decisions
    func testExampleAdaptingLayaOnOwnDecisions() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
        let tuned = FileManager.default.temporaryDirectory
            .appendingPathComponent("laya-example-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: tuned) }

        // A real run builds from the release directory (NFKMLXLaya.laya(directoryURL:)); the example
        // uses the tiny geometry and a stand-in tokenizer so it stays a test.
        let tokenizer = NFKMLXLayaTokenizer { text in text.unicodeScalars.map { Int($0.value % 500) + 8 } }
        let laya = try NFKMLXLaya.laya(weightsURL: nil, tokenizer: tokenizer, configuration: .tiny)

        // The consumer's own decisions: a state, the question, and the right answer.
        let department = NFKDecisionQuestion.choiceQuestion(withInstructions: "Which team?", options: ["billing", "technical"])
        let examples = [
            NFKMLXLayaExample(state: "My invoice is wrong.", question: department, label: 0),
            NFKMLXLayaExample(state: "The app crashes on launch.", question: department, label: 1),
            NFKMLXLayaExample(state: "I need this today.",
                              question: NFKDecisionQuestion.noulQuestion(withInstructions: "Urgent?"), holds: true),
        ]
        // The head alone trains by default (the encoder stays as released); .all trains the encoder too.
        let history = try laya.fineTune(examples: examples, steps: 4, learningRate: 1e-3, trainable: .head)
        XCTAssertEqual(history.count, 4)

        try NFKMLXWeights.save(laya.net, to: tuned)
        // The fine-tuned file reloads through the same factory (a release directory plus the file from
        // Objective-C: layaWithDirectoryURL:weightsURL:error:).
        let reloaded = try NFKMLXLaya.laya(weightsURL: tuned, tokenizer: tokenizer, configuration: .tiny)
        XCTAssertEqual(reloaded.answer(state: "My invoice is wrong.", question: department).type, .choice)
    }

    // Docs/examples.md: Adapting a decision model to your own conversations
    func testExampleAdaptingLayaOnOwnConversations() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
        let tokenizer = NFKMLXLayaTokenizer { text in text.unicodeScalars.map { Int($0.value % 500) + 8 } }
        let laya = try NFKMLXLaya.laya(weightsURL: nil, tokenizer: tokenizer, configuration: .tiny)

        // A conversation is trained as its prefixes, so the model learns to judge the outcome early.
        // The context rides with every prefix; the turns are strings or {role, text} records.
        let resolved = NFKDecisionQuestion.noulQuestion(withInstructions: "The problem will be resolved by the end of the conversation.")
        let episodes = [
            NFKMLXLayaEpisode(context: ["channel": "chat"],
                              turns: [["role": "customer", "text": "My payouts keep failing."],
                                      ["role": "agent", "text": "Escalating to payments now."],
                                      ["role": "customer", "text": "Thank you."]],
                              question: resolved, holds: true),
            NFKMLXLayaEpisode(context: ["channel": "email"],
                              turns: ["It is still broken.", "We cannot help with that."],
                              question: resolved, holds: false),
        ]
        // λ = 1 trains every prefix toward the outcome, the release's own setting; a lower λ leans on
        // the model's own prediction for the next prefix.
        let history = try laya.fineTune(episodes: episodes, steps: 4, learningRate: 1e-3, lambda: 1)
        XCTAssertEqual(history.count, 4)
        XCTAssertEqual(laya.prefixes(of: episodes[0]).map(\.length), [1, 2, 3])
    }

    // Docs/examples.md: Adapting the community Jev reproductions (open-jev-deberta)
    func testExampleAdaptingOpenJevDeBERTaOnOwnDecisions() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
        let tuned = FileManager.default.temporaryDirectory
            .appendingPathComponent("open-jev-deberta-example-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: tuned) }

        // A real run builds from the release directory (NFKMLXOpenJevDeBERTa.openJev(directoryURL:)); the
        // example uses the tiny geometry and a stand-in tokenizer so it stays a test.
        let tokenizer = NFKMLXDecisionTokenizer { text in text.unicodeScalars.map { Int($0.value % 400) + 10 } }
        let model = NFKMLXOpenJevDeBERTa(net: NFKMLXOpenJevDeBERTaNet(.tiny), tokenizer: tokenizer)

        // One labeled state can carry several questions; each label indexes the right option.
        let team = NFKDecisionQuestion.choiceQuestion(withInstructions: "Which team?", options: ["billing", "technical"])
        let refund = NFKDecisionQuestion.noulQuestion(withInstructions: "Asks for a refund.")
        let examples = [
            NFKMLXOpenJevDeBERTaExample(state: "Charged twice, refund me.", questions: [team, refund], labels: [0, 1]),
            NFKMLXOpenJevDeBERTaExample(state: "The app crashes.", question: team, label: 1),
        ]
        // The head alone by default at the release's head rate; .all trains the encoder too at 3e-5.
        let history = try model.fineTune(examples: examples, steps: 3, batchSize: 2)
        XCTAssertEqual(history.count, 3)

        try NFKMLXWeights.save(model.net, to: tuned)
        // The tuned file reloads with the release directory: openJev(directoryURL:weightsURL:), from
        // Objective-C openJevWithDirectoryURL:weightsURL:error:.
        let reloaded = NFKMLXOpenJevDeBERTa(net: try NFKMLXOpenJevDeBERTa.network(weightsURL: tuned, configuration: .tiny),
                                            tokenizer: tokenizer)
        XCTAssertEqual(try reloaded.decide(state: "Charged twice.", questions: [team]).first?.type, .choice)
    }

    // Docs/examples.md: Adapting the community Jev reproductions (Open-Jev)
    func testExampleAdaptingOpenJevOnOwnDecisions() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
        let tuned = FileManager.default.temporaryDirectory.appendingPathComponent("open-jev-example-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: tuned) }

        // A real run loads the release over its base at .float32:
        //   NFKMLXOpenJev.openJev(checkpointDirectoryURL: release.checkpointDirectoryURL,
        //                         baseDirectoryURL: release.baseDirectoryURL, precision: .float32)
        // The example adapts a tiny random decoder with a word-level stand-in tokenizer.
        let net = NFKMLXOpenJevNet(.tiny)
        try net.adapt()
        let tokenizer = NFKMLXDecisionTokenizer { text in
            text.split(whereSeparator: \.isWhitespace).map { $0.unicodeScalars.reduce(7) { ($0 * 31 + Int($1.value)) % 500 } + 5 }
        }
        let model = NFKMLXOpenJev(net: net, tokenizer: tokenizer)

        // A target has one entry per candidate: a choice's options, or a noul's [no, yes].
        let team = NFKDecisionQuestion.choiceQuestion(withInstructions: "Which team?", options: ["billing", "technical"])
        let examples = [
            NFKMLXOpenJevExample(state: "Charged twice.", question: team, label: 0),
            NFKMLXOpenJevExample(state: "Refund me.", question: .noulQuestion(withInstructions: "Asks for a refund."), holds: true),
        ]
        // The adapter and the head train, at the release's rates; the base stays as released.
        let history = try model.fineTune(examples: examples, steps: 2, batchSize: 2)
        XCTAssertEqual(history.count, 2)

        // The tuned model saves as a release checkpoint directory, which the same factory reloads over
        // the same base: openJev(checkpointDirectoryURL: tuned, baseDirectoryURL: base).
        try model.save(to: tuned)
        XCTAssertTrue(FileManager.default.fileExists(atPath: tuned.appendingPathComponent("adapter/adapter_model.safetensors").path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: tuned.appendingPathComponent("head.safetensors").path))
    }

    // Docs/examples.md: Adapting a translator to your own sentence pairs
    func testExampleAdaptingATranslatorOnOwnSentencePairs() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
        let tuned = FileManager.default.temporaryDirectory
            .appendingPathComponent("opus-mt-example-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tuned, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tuned) }

        // A release directory gives NFKMLXMarian.network(directoryURL:) the real weights and
        // NFKMLXMarian.translator(directoryURL:) the tokenizers whose sourceIds(for:target:) and
        // targetTokenizer.encode(_:dummyPrefix:) + [eos] produce each pair's ids. A tiny random
        // network stands in here.
        let net = try NFKMLXMarian.network(directoryURL: nil, configuration: .tinyMarian)
        let pairs: [(source: MLXArray, target: MLXArray)] = [
            (MLXArray([Int32(5), 6, 7, 0]), MLXArray([Int32(9), 10, 11, 0])),
            (MLXArray([Int32(8), 6, 0]), MLXArray([Int32(12), 10, 0])),
        ]
        let history = try NFKMLXMarian.fineTune(net, examples: { pairs[$0 % pairs.count] }, rank: 4, steps: 4)
        XCTAssertEqual(history.count, 4)

        // Merge the adapters, save one ordinary checkpoint, and rebuild through the factory.
        try NFKMLXLoRA.merge(into: net)
        try NFKMLXWeights.save(net, to: tuned.appendingPathComponent("model.safetensors"))
        let reloaded = try NFKMLXMarian.network(directoryURL: tuned, configuration: .tinyMarian)
        XCTAssertEqual(reloaded.configuration.vocabularySize, net.configuration.vocabularySize)
    }

    // Docs/examples.md: Adapting a language model to your own text
    func testExampleAdaptingALanguageModelOnOwnText() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("granite-example-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }

        // A release directory gives NFKMLXGraniteHybrid.configuration(fromDirectory:) the geometry and
        // network(weightsURL:configuration:) the real weights; a tiny random net stands in here. Each
        // example is one token sequence of the consumer's own text as ids.
        let config = NFKMLXGraniteHybridConfiguration(
            hiddenSize: 64, layerCount: 4, vocabularySize: 128, tiesWordEmbeddings: false,
            headCount: 4, keyValueHeadCount: 2, headDimensions: 16, mambaHeadCount: 8,
            mambaHeadDimensions: 16, mambaGroupCount: 1, mambaStateSize: 16, sharedIntermediateSize: 96,
            layerTypes: [.mamba, .attention, .mamba, .attention])
        let net = try NFKMLXGraniteHybrid.network(weightsURL: nil, configuration: config)
        let sequences = [MLXArray([Int32(3), 17, 42, 99, 7, 61]), MLXArray([Int32(8), 6, 5, 4, 3, 2])]
        let history = try NFKMLXGraniteHybrid.fineTune(net, examples: { sequences[$0 % sequences.count] },
                                                       rank: 4, steps: 4)
        XCTAssertEqual(history.count, 4)

        try NFKMLXLoRA.merge(into: net)
        try NFKMLXWeights.save(net, to: url)
        let reloaded = try NFKMLXGraniteHybrid.network(weightsURL: url, configuration: config)
        let a = net(sequences[0].reshaped([1, 6])), b = reloaded(sequences[0].reshaped([1, 6]))
        XCTAssertLessThan(abs(a - b).max().item(Float.self), 1e-4, "the merged checkpoint reloads")
    }

    // Docs/examples.md: Retargeting a segmentation model to your own classes
    func testExampleSegmentationDataAndSamplerFeedTheTrainer() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
        let myFrames = [Self.gray(8, level: 60), Self.gray(8, level: 200)]
        let myMasks = [Self.gray(8, level: 0), Self.gray(8, level: 255)]
        let sampler = NFKMLXBatchSampler(count: myFrames.count, seed: 7)

        let index = sampler.indices(forStep: 0)[0]
        let image = try NFKMLXTrainingData.tensor(myFrames[index])
        let labels = try NFKMLXTrainingData.labels(myMasks[index], classCount: 3)

        XCTAssertEqual(image.shape, [8, 8, 3])
        XCTAssertEqual(labels.shape, [8, 8])
        XCTAssertEqual(labels.dtype, .int32, "cross-entropy takes class indices")
    }

    // Docs/examples.md: LoRA, for models with no small head to train
    func testExampleLoRAAdaptsMergesAndLeavesAnOrdinaryCheckpoint() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
        let net = NFKMLXZeroDCENet(filters: 4)
        let adapted = try NFKMLXLoRA.apply(to: net, rank: 8, alpha: 16) { path, _ in
            path.hasSuffix("q") || path.hasSuffix("v")
        }
        // Zero-DCE is all convolution, so the attention predicate matches nothing — which `apply`
        // reports rather than hiding.
        XCTAssertEqual(adapted, 0)
        XCTAssertEqual(try NFKMLXLoRA.merge(into: net), 0)
    }

    // Docs/examples.md: A custom image classifier from a handful of photos
    func testExampleCLIPProbeClassifiesThroughABackend() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
        var configuration = NFKMLXCLIPConfiguration()
        configuration.imageResolution = 32
        configuration.patchSize = 16
        configuration.visionWidth = 32
        configuration.visionLayers = 1
        configuration.visionHeads = 2
        configuration.embedDimensions = 16
        configuration.textWidth = 32
        configuration.textLayers = 1
        configuration.textHeads = 2
        configuration.contextLength = 16
        configuration.vocabularySize = 64
        let clip = try NFKMLXCLIP.network(weightsURL: nil, configuration: configuration)

        let myPhotos = [Self.gray(32, level: 40), Self.gray(32, level: 220)]
        let cached = try NFKMLXCLIP.embeddings(for: myPhotos, using: clip)      // run once
        let probe = NFKMLXCLIPProbe(embedDimensions: 16, classCount: 2)
        try NFKMLXCLIP.trainProbe(probe, embeddings: cached, labels: MLXArray([Int32(0), 1]), steps: 20)

        let classifier = NFKMLXCLIP.probeBackend(net: clip, probe: probe, labels: ["dark", "bright"])
        let result = try classifier.runInference(
            for: NFKInferenceRequest(inputs: [NFKInputImage: myPhotos[0]]))
        XCTAssertEqual(result.classifications?.count, 2)
    }

    // MARK: The public surface these recipes rest on

    /// The generic trainer entry, an optimizer chosen by the caller, and both ends of the checkpoint
    /// API, driven the way an app drives them. `Docs/examples.md` shows this shape in the LoRA
    /// section, where the caller supplies the optimizer and the loss.
    func testTheTrainerAndTheCheckpointAPIAreCallableFromOutsideThePackage() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
        let tuned = FileManager.default.temporaryDirectory
            .appendingPathComponent("trainer-example-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: tuned) }

        let net = NFKMLXZeroDCENet(filters: 4)
        let photos = Self.darkPhotos()
        let objective = NFKMLXZeroDCEObjective()
        let history = try NFKMLXTrainer.train(net, optimizer: AdamW(learningRate: 1e-4), steps: 3,
                                              sample: { _ in photos },
                                              loss: objective.callAsFunction,
                                              clipGradientNorm: 0.1)
        XCTAssertEqual(history.count, 3)

        try NFKMLXWeights.save(net, to: tuned)
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: tuned)
        XCTAssertFalse(checkpoint.needsConvTranspose,
                       "a checkpoint written by save is already in the module's own layout")
    }

    /// A call that documents what it throws has to let a caller act on it, so the error type is part
    /// of the customization surface.
    func testAFullyFrozenModelReportsThatNothingWouldTrain() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
        let net = NFKMLXZeroDCENet(filters: 4)
        net.freeze()
        let photos = Self.darkPhotos()
        let objective = NFKMLXZeroDCEObjective()

        XCTAssertThrowsError(try NFKMLXTrainer.train(net, optimizer: AdamW(learningRate: 1e-4),
                                                     steps: 1, sample: { _ in photos },
                                                     loss: objective.callAsFunction)) { error in
            guard case NFKMLXError.nothingToTrain = error else {
                return XCTFail("expected nothingToTrain, got \(error)")
            }
        }
    }

    private static func gray(_ side: Int, level: UInt8) -> CGImage {
        var pixels = [UInt8](repeating: 255, count: side * side * 4)
        for pixel in 0 ..< (side * side) {
            pixels[pixel * 4] = level
            pixels[pixel * 4 + 1] = level
            pixels[pixel * 4 + 2] = level
        }
        let provider = CGDataProvider(data: Data(pixels) as CFData)!
        return CGImage(width: side, height: side, bitsPerComponent: 8, bitsPerPixel: 32,
                       bytesPerRow: side * 4, space: CGColorSpaceCreateDeviceRGB(),
                       bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
                       provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
    }

    private static func darkPhotos() -> MLXArray {
        var values = [Float](repeating: 0, count: 16 * 16 * 3)
        for i in 0 ..< values.count {
            values[i] = Float((i * 37) % 60) / 255.0
        }
        return values.withUnsafeBufferPointer { MLXArray($0, [1, 16, 16, 3]) }
    }

    /// One annotated frame for the SAM 2 example: a plate, a click on the subject, and its mask.
    private static func clickedFrame(size: Int = 64)
        -> (image: MLXArray, points: [(x: Float, y: Float, label: Int)], target: MLXArray) {
        var pixels = [Float](repeating: 0.2, count: size * size * 3)
        var mask = [Float](repeating: 0, count: size * size)
        for row in (size / 4) ..< (3 * size / 4) {
            for column in (size / 4) ..< (3 * size / 4) {
                mask[row * size + column] = 1
                pixels[(row * size + column) * 3] = 0.9
            }
        }
        let image = pixels.withUnsafeBufferPointer { MLXArray($0, [1, size, size, 3]) }
        let target = mask.withUnsafeBufferPointer { MLXArray($0, [1, size, size]) }
        return (image, [(Float(size / 2), Float(size / 2), 1)], target)
    }
}
