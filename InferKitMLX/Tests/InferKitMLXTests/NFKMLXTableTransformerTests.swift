//
//  NFKMLXTableTransformerTests.swift
//  InferKitMLXTests
//
//  Table Transformer (microsoft/table-transformer-*, MIT) — a vanilla DETR with a ResNet-18 backbone:
//  table detection and the structure-recognition v1.0 and v1.1 releases. The network evaluates MLX arrays, so these skip without a Metal library for MLX
//  (see Tools/mlx-metallib.sh). The parity tests are gated on the released directory
//  (`IK_VAL_TABLE_TRANSFORMER`) + the recorded oracle (`IK_PARITY_TABLE_TRANSFORMER`, from
//  `run_reference.py table_transformer`), compared seam by seam.
//

import XCTest
import InferKit
import CoreGraphics
import ImageIO
import MLX
import MLXNN
import MLXRandom
@testable import InferKitMLX

final class NFKMLXTableTransformerTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    private func cosine(_ mine: MLXArray, _ reference: MLXArray) -> Float {
        eval(mine)
        let a = mine.reshaped([-1]).asArray(Float.self), b = reference.reshaped([-1]).asArray(Float.self)
        let n = min(a.count, b.count)
        var dot: Float = 0, na: Float = 0, nb: Float = 0
        for i in 0 ..< n { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
        return dot / (sqrtf(na) * sqrtf(nb) + 1e-20)
    }

    private func loadedNet(_ directory: String) throws -> NFKMLXTableTransformerNet {
        let url = URL(fileURLWithPath: directory)
        let net = try NFKMLXTableTransformerNet(configurationURL: url.appendingPathComponent("config.json"))
        try net.loadWeights(fromDirectory: url)
        return net
    }

    /// The module keys the loader targets mirror the checkpoint (timm ResNet naming under the backbone,
    /// the pre-norm encoder / decoder with their final layer norms, and the heads).
    func testParameterNamesFollowTheModuleLayout() throws {
        try requireMLXRuntime()
        let net = NFKMLXTableTransformerNet(.structureRecognition)
        let names = Set(net.parameters().flattened().map(\.0))
        for expected in ["model.backbone.conv_encoder.model.conv1.weight",
                         "model.backbone.conv_encoder.model.bn1.running_var",
                         "model.backbone.conv_encoder.model.layer2.0.downsample.0.weight",
                         "model.backbone.conv_encoder.model.layer2.0.downsample.1.running_mean",
                         "model.input_projection.weight",
                         "model.query_position_embeddings.weight",
                         "model.encoder.layernorm.weight",
                         "model.encoder.layers.0.self_attn.q_proj.weight",
                         "model.decoder.layernorm.weight",
                         "model.decoder.layers.0.encoder_attn.k_proj.weight",
                         "class_labels_classifier.weight",
                         "bbox_predictor.layers.2.weight"] {
            XCTAssertTrue(names.contains(expected), "missing \(expected)")
        }
    }

    /// PARITY: the whole forward against the recorded oracle, seam by seam (the ResNet-18 feature map,
    /// the encoder output, the decoder output, the class logits, and the predicted boxes).
    func testSeamParityOnTheReleasedWeights() throws {
        try requireMLXRuntime()
        let env = NFKMLXValidationConfig.environment
        guard let directory = env["IK_VAL_TABLE_TRANSFORMER"], let recordPath = env["IK_PARITY_TABLE_TRANSFORMER"] else {
            throw XCTSkip("set IK_VAL_TABLE_TRANSFORMER (release directory) and IK_PARITY_TABLE_TRANSFORMER (oracle record)")
        }
        let net = try loadedNet(directory)
        let rec = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: recordPath)).arrays
        let pixels = rec["pixels"]!                                                 // [H, W, 3] normalized
        let x = pixels.reshaped([1, pixels.dim(0), pixels.dim(1), 3]).asType(.float32)

        let detection = net(x)
        XCTAssertGreaterThan(cosine(detection.backboneFeatures[0], rec["backbone"]!), 0.999, "backbone diverges")
        XCTAssertGreaterThan(cosine(detection.encoderLast[0], rec["enc_last"]!), 0.999, "encoder output diverges")
        XCTAssertGreaterThan(cosine(detection.decoderLast[0], rec["dec_last"]!), 0.999, "decoder output diverges")
        XCTAssertGreaterThan(cosine(detection.logits, rec["output"]!), 0.999, "class logits diverge")
        XCTAssertGreaterThan(cosine(detection.boxes, rec["pred_boxes"]!), 0.999, "predicted boxes diverge")
    }

    /// PARITY on every other release: table detection (15 queries, two classes) and the three v1.1
    /// structure releases, whose transformers ResNet backbone names load through the timm layout.
    func testEveryOtherReleaseIsAtParity() throws {
        try requireMLXRuntime()
        let env = NFKMLXValidationConfig.environment
        var measured = [String]()
        for suffix in ["DETECTION", "V11_ALL", "V11_FIN", "V11_PUB"] {
            guard let directory = env["IK_VAL_TABLE_TRANSFORMER_\(suffix)"],
                  let recordPath = env["IK_PARITY_TABLE_TRANSFORMER_\(suffix)"],
                  FileManager.default.fileExists(atPath: recordPath) else { continue }
            let net = try loadedNet(directory)
            let rec = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: recordPath)).arrays
            let pixels = rec["pixels"]!
            let detection = net(pixels.reshaped([1, pixels.dim(0), pixels.dim(1), 3]).asType(.float32))
            let seams = [("backbone", cosine(detection.backboneFeatures[0], rec["backbone"]!)),
                         ("encoder", cosine(detection.encoderLast[0], rec["enc_last"]!)),
                         ("decoder", cosine(detection.decoderLast[0], rec["dec_last"]!)),
                         ("logits", cosine(detection.logits, rec["output"]!)),
                         ("boxes", cosine(detection.boxes, rec["pred_boxes"]!))]
            print("VALIDATION PARITY table-transformer \(suffix): " + seams.map { "\($0.0) \($0.1)" }.joined(separator: ", "))
            for (name, similarity) in seams { XCTAssertGreaterThan(similarity, 0.999, "\(suffix) \(name) diverges") }
            XCTAssertEqual(NFKMLXTableTransformerSizing.release(at: URL(fileURLWithPath: directory)),
                           suffix == "DETECTION" ? .shortestEdge(800, longestEdge: 800) : .longestEdge(800))
            measured.append(suffix)
        }
        try XCTSkipIf(measured.isEmpty, "set IK_VAL_TABLE_TRANSFORMER_<DETECTION|V11_ALL|V11_FIN|V11_PUB> and their IK_PARITY_ records")
    }

    /// The three sizing rules against transformers' own `get_resize_output_image_size` and
    /// `get_image_size_for_max_height_width`.
    func testTheSizingRulesMatchTheReferenceProcessor() {
        let cases: [((Int, Int), NFKMLXTableTransformerSizing, (Int, Int))] = [
            ((600, 1000), .structureRecognition, (600, 1000)), ((1234, 567), .structureRecognition, (1000, 459)),
            ((600, 1000), .shortestEdge(800, longestEdge: 800), (480, 800)),
            ((1234, 567), .shortestEdge(800, longestEdge: 800), (800, 368)),
            ((600, 1000), .longestEdge(800), (480, 800)), ((1234, 567), .longestEdge(800), (800, 367)),
            ((333, 333), .longestEdge(800), (800, 800))]
        for ((height, width), sizing, expected) in cases {
            let target = NFKMLXTableTransformerProcessor.targetSize(height: height, width: width, sizing: sizing)
            XCTAssertEqual([target.height, target.width], [expected.0, expected.1], "\(height)x\(width) under \(sizing)")
        }
    }

    func testTheTransformersResNetNamesMapToTheTimmLayout() {
        let prefix = "model.backbone.conv_encoder.model."
        let cases = ["embedder.embedder.convolution.weight": "conv1.weight",
                     "embedder.embedder.normalization.running_var": "bn1.running_var",
                     "encoder.stages.0.layers.1.layer.0.convolution.weight": "layer1.1.conv1.weight",
                     "encoder.stages.3.layers.0.layer.1.normalization.bias": "layer4.0.bn2.bias",
                     "encoder.stages.1.layers.0.shortcut.convolution.weight": "layer2.0.downsample.0.weight",
                     "encoder.stages.1.layers.0.shortcut.normalization.running_mean": "layer2.0.downsample.1.running_mean",
                     "layer1.0.conv1.weight": "layer1.0.conv1.weight"]
        for (hf, timm) in cases {
            XCTAssertEqual(NFKMLXTableTransformerNet.timmBackboneKey(prefix + hf), prefix + timm)
        }
        XCTAssertEqual(NFKMLXTableTransformerNet.timmBackboneKey("model.encoder.layers.0.fc1.weight"),
                       "model.encoder.layers.0.fc1.weight")
    }

    /// The backend recognizes a table's structure end to end (image processor + backbone + DETR
    /// transformer + post-processing): a clean grid yields a table with its rows and columns.
    func testTheBackendDetectsTableStructure() throws {
        try requireMLXRuntime()
        let env = NFKMLXValidationConfig.environment
        guard let directory = env["IK_VAL_TABLE_TRANSFORMER"] else { throw XCTSkip("set IK_VAL_TABLE_TRANSFORMER") }
        guard let imagePath = env["IK_VAL_TABLE_TRANSFORMER_IMAGE"], let image = Self.loadImage(imagePath) else {
            throw XCTSkip("set IK_VAL_TABLE_TRANSFORMER_IMAGE to a table image")
        }
        let backend = try NFKMLXTableTransformer.backend(directoryURL: URL(fileURLWithPath: directory))
        let request = NFKInferenceRequest(inputs: [NFKInputImage: image])
        let result = try backend.runInference(for: request)
        guard let detections = result.output(forKey: NFKOutputDetections) as? [NFKDetection] else {
            return XCTFail("the backend produced no detections")
        }
        let labels = detections.map { $0.label ?? "" }
        XCTAssertTrue(labels.contains("table"), "the table itself is detected; got \(labels)")
        XCTAssertGreaterThanOrEqual(labels.filter { $0 == "table row" }.count, 3, "several rows are detected")
        XCTAssertGreaterThanOrEqual(labels.filter { $0 == "table column" }.count, 3, "several columns are detected")
    }

    static func loadImage(_ path: String) -> CGImage? {
        guard FileManager.default.fileExists(atPath: path),
              let source = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil) else { return nil }
        return CGImageSourceCreateImageAtIndex(source, 0, nil)
    }
}

/// Table Transformer's fine-tuning recipe: the set objective against microsoft/table-transformer's own
/// DETR criterion (`run_reference.py table_transformer_loss`, `IK_PARITY_TABLE_TRANSFORMER_<RELEASE>_LOSS`),
/// the freezing policies, and a retargeted release's save and factory reload.
final class NFKMLXTableTransformerTrainingTests: XCTestCase {

    override func tearDown() {
        if NFKMLXGPU.metalLibraryURL != nil {
            NFKMLXGPU.clearCache()
        }
        super.tearDown()
    }

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil, "no Metal library for MLX; run Tools/mlx-metallib.sh")
    }

    private func release(_ name: String) throws -> (directory: URL, record: [String: MLXArray]) {
        try requireMLXRuntime()
        let env = NFKMLXValidationConfig.environment
        guard let directory = env["IK_VAL_TABLE_TRANSFORMER_\(name)"],
              let path = env["IK_PARITY_TABLE_TRANSFORMER_\(name)_LOSS"], FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_VAL_TABLE_TRANSFORMER_\(name) and IK_PARITY_TABLE_TRANSFORMER_\(name)_LOSS")
        }
        return (URL(fileURLWithPath: directory), try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: path)).arrays)
    }

    private static func targets(_ record: [String: MLXArray]) -> MLXArray {
        concatenated([record["target_classes"]!.asType(.float32).reshaped([-1, 1]), record["target_boxes"]!], axis: 1)
    }

    /// PARITY: on the release's own outputs, the matching and each term equal DETR's `HungarianMatcher`
    /// and `SetCriterion`; the port's own forward reproduces the weighted total.
    func testTheObjectiveIsDETRsSetCriterionOnTheReleases() throws {
        var measured = 0
        for name in ["DETECTION", "V11_ALL"] {
            guard let (directory, record) = try? release(name) else { continue }
            let objective = NFKMLXTableTransformerObjective()
            let targets = Self.targets(record)
            let logits = record["logits"]!, boxes = record["pred_boxes"]!
            XCTAssertEqual(objective.match(logits: logits, boxes: boxes, targets: targets),
                           record["matched_queries"]!.asArray(Int32.self).map(Int.init), "\(name) matching")
            let parts = objective.terms(logits: logits, boxes: boxes, targets: targets)
            let total = objective.loss(logits: logits, boxes: boxes, targets: targets).item(Float.self)
            let net = try NFKMLXTableTransformer.network(directoryURL: directory)
            let forward = objective(net, record["pixels"]!.expandedDimensions(axis: 0), targets).item(Float.self)
            print("VALIDATION PARITY table_transformer \(name) loss: ce \(parts.classification.item(Float.self)) vs "
                  + "\(record["loss_ce"]!.item(Float.self)), bbox \(parts.box.item(Float.self)) vs "
                  + "\(record["loss_bbox"]!.item(Float.self)), giou \(parts.generalizedIoU.item(Float.self)) vs "
                  + "\(record["loss_giou"]!.item(Float.self)), total \(total) vs \(record["output"]!.item(Float.self)), "
                  + "port forward \(forward)")
            for (mine, key) in [(parts.classification, "loss_ce"), (parts.box, "loss_bbox"), (parts.generalizedIoU, "loss_giou")] {
                XCTAssertEqual(mine.item(Float.self), record[key]!.item(Float.self), accuracy: 1e-5, "\(name) \(key)")
            }
            XCTAssertEqual(total, record["output"]!.item(Float.self), accuracy: 1e-5)
            XCTAssertEqual(forward, record["output"]!.item(Float.self), accuracy: 1e-3)
            NFKMLXGPU.clearCache()
            measured += 1
        }
        try XCTSkipIf(measured == 0, "no Table Transformer loss record is set")
    }

    private static func tinyNet() -> NFKMLXTableTransformerNet {
        var configuration = NFKMLXTableTransformerConfiguration()
        configuration.dModel = 32
        configuration.encoderLayers = 1
        configuration.decoderLayers = 1
        configuration.encoderAttentionHeads = 2
        configuration.decoderAttentionHeads = 2
        configuration.encoderFFNDim = 64
        configuration.decoderFFNDim = 64
        configuration.numQueries = 6
        configuration.numLabels = 2
        configuration.labels = ["table", "figure"]
        return NFKMLXTableTransformerNet(configuration)
    }

    /// Each policy moves what it names and nothing else; the stem, the first stage, and every frozen
    /// batch norm never move, and the loss falls under the reference policy.
    func testEachPolicyMovesWhatItNames() throws {
        try requireMLXRuntime()
        let pixels = MLXRandom.uniform(low: -1, high: 1, [1, 64, 64, 3], key: MLXRandom.key(1))
        let targets = MLXArray([Float(0), 0.5, 0.5, 0.6, 0.4, 1, 0.3, 0.7, 0.2, 0.2]).reshaped([2, 5])
        for trainable in [NFKMLXTableTransformerTrainable.heads, .transformer, .everything] {
            MLXRandom.seed(2)
            let net = Self.tinyNet()
            let before = Dictionary(uniqueKeysWithValues: net.parameters().flattened().map { ($0.0, $0.1 + 0) })
            eval(Array(before.values))
            let losses = try NFKMLXTableTransformer.fineTune(
                net, examples: { _ in (pixels, targets) }, trainable: trainable,
                optimizer: NFKMLXReferenceOptimizers.adamW(learningRate: 1e-3, weightDecay: 0), steps: 12,
                clipGradientNorm: nil)
            XCTAssertLessThan(losses.suffix(3).reduce(0, +), losses.prefix(3).reduce(0, +), "\(trainable): the loss falls")
            for (name, value) in net.parameters().flattened() {
                let moved = abs(value - before[name]!).max().item(Float.self) > 0
                let isHead = name.hasPrefix("class_labels_classifier.") || name.hasPrefix("bbox_predictor.")
                let isBackbone = name.hasPrefix("model.backbone.")
                let isFrozenNorm = name.contains(".bn") || name.contains("downsample.1")
                let isLateStage = ["layer2.", "layer3.", "layer4."].contains { name.contains($0) }
                let shouldMove: Bool
                switch trainable {
                case .heads: shouldMove = isHead
                case .transformer: shouldMove = !isBackbone
                case .everything: shouldMove = !isBackbone || (isLateStage && !isFrozenNorm)
                }
                if !shouldMove {
                    XCTAssertFalse(moved, "\(trainable): \(name) stays frozen")
                }
            }
        }
    }

    /// A released detector retargeted to three classes, fine-tuned, saved, and reloaded through the
    /// factory: the same logits, and the backend reports the new labels.
    func testARetargetedReleaseReloadsThroughTheFactory() throws {
        let (directory, record) = try release("DETECTION")
        let labels = ["table", "figure", "chart"]
        let net = try NFKMLXTableTransformer.network(directoryURL: directory, labels: labels)
        XCTAssertEqual(net.config.numLabels, 3)
        let pixels = record["pixels"]!.expandedDimensions(axis: 0)
        let targets = Self.targets(record)
        try NFKMLXTableTransformer.fineTune(net, examples: { _ in (pixels, targets) }, trainable: .heads, steps: 2)
        let saved = FileManager.default.temporaryDirectory.appendingPathComponent("tatr-tuned-\(UUID().uuidString)")
        addTeardownBlock { try? FileManager.default.removeItem(at: saved) }
        try NFKMLXTableTransformer.save(net, toDirectoryURL: saved, release: directory)

        let reloaded = try NFKMLXTableTransformer.network(directoryURL: saved)
        XCTAssertEqual(reloaded.config.labels, labels)
        XCTAssertLessThan(abs(reloaded(pixels).logits - net(pixels).logits).max().item(Float.self), 1e-5)
        let backend = try NFKMLXTableTransformer.backend(directoryURL: saved)
        let image = try XCTUnwrap(NFKMLXTableTransformerTests.loadImage(
            NFKMLXValidationConfig.environment["IK_VAL_TABLE_TRANSFORMER_IMAGE"] ?? ""))
        XCTAssertNoThrow(try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: image])))
    }
}
