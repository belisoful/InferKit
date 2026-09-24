//
//  NFKMLXVJEPA2Tests.swift
//  InferKitMLXTests
//
//  V-JEPA 2 vision encoder (facebook/vjepa2-vitl-fpc64-256, MIT) — a self-supervised video ViT-L with a
//  3D tubelet patch embedding and 3D rotary attention. The network evaluates MLX arrays, so these skip
//  without a Metal library for MLX (see Tools/mlx-metallib.sh). The parity test is gated on the released
//  directory (`IK_VAL_VJEPA2`) + the recorded oracle (`IK_PARITY_VJEPA2`, from `run_reference.py vjepa2`),
//  compared seam by seam.
//

import XCTest
import InferKit
import CoreGraphics
import MLX
import MLXNN
import MLXRandom
@testable import InferKitMLX

final class NFKMLXVJEPA2Tests: XCTestCase {

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

    private func loadedNet(_ directory: String) throws -> NFKMLXVJEPA2Net {
        let url = URL(fileURLWithPath: directory)
        let net = try NFKMLXVJEPA2Net(configurationURL: url.appendingPathComponent("config.json"))
        try net.loadWeights(fromDirectory: url)
        return net
    }

    /// The module keys the loader targets mirror the checkpoint (`encoder.*`: the 3D patch embedding,
    /// the per-block rotary attention and MLP, and the final layer norm).
    func testParameterNamesFollowTheModuleLayout() throws {
        try requireMLXRuntime()
        let net = NFKMLXVJEPA2Net(.vitLarge)
        let names = Set(net.parameters().flattened().map(\.0))
        for expected in ["encoder.embeddings.patch_embeddings.proj.weight",
                         "encoder.embeddings.patch_embeddings.proj.bias",
                         "encoder.layer.0.norm1.weight",
                         "encoder.layer.0.attention.query.weight",
                         "encoder.layer.0.attention.key.weight",
                         "encoder.layer.0.attention.value.weight",
                         "encoder.layer.0.attention.proj.weight",
                         "encoder.layer.0.mlp.fc1.weight",
                         "encoder.layer.0.mlp.fc2.weight",
                         "encoder.layer.23.norm2.weight",
                         "encoder.layernorm.weight"] {
            XCTAssertTrue(names.contains(expected), "missing \(expected)")
        }
    }

    /// PARITY: the encoder forward against the recorded oracle, seam by seam (the tubelet patch
    /// embedding, a middle block's hidden state, and the final normed feature sequence).
    func testSeamParityOnTheReleasedWeights() throws {
        try requireMLXRuntime()
        let env = NFKMLXValidationConfig.environment
        guard let directory = env["IK_VAL_VJEPA2"], let recordPath = env["IK_PARITY_VJEPA2"] else {
            throw XCTSkip("set IK_VAL_VJEPA2 (release directory) and IK_PARITY_VJEPA2 (oracle record)")
        }
        let net = try loadedNet(directory)
        let rec = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: recordPath)).arrays
        let clip = rec["clip"]!                                                      // [T, H, W, 3]
        let x = clip.reshaped([1, clip.dim(0), clip.dim(1), clip.dim(2), 3]).asType(.float32)

        let states = net.hiddenStates(x)
        XCTAssertGreaterThan(cosine(states[0][0], rec["patch"]!), 0.999, "patch embedding diverges")
        XCTAssertGreaterThan(cosine(states[12][0], rec["mid"]!), 0.999, "middle block diverges")
        XCTAssertGreaterThan(cosine(states[states.count - 1][0], rec["output"]!), 0.999, "final features diverge")
    }

    /// PARITY on every other release, each against its own record:
    /// - the ViT-H and ViT-g encoders at 256 and 384;
    /// - the four video-classification releases, whose pooler output and class logits are measured too.
    /// ViT-g runs in about 4 GB of float32 weights; the suite clears MLX's cache between releases.
    func testEveryOtherReleaseIsAtParity() throws {
        try requireMLXRuntime()
        let env = NFKMLXValidationConfig.environment
        var measured = [String]()
        for name in ["VITH_FPC64_256", "VITG_FPC64_256", "VITG_FPC64_384", "VITL_FPC16_256_SSV2",
                     "VITL_FPC32_256_DIVING48", "VITG_FPC64_384_SSV2", "VITG_FPC32_384_DIVING48"] {
            guard let directory = env["IK_VAL_VJEPA2_\(name)"], let recordPath = env["IK_PARITY_VJEPA2_\(name)"],
                  FileManager.default.fileExists(atPath: recordPath) else { continue }
            try autoreleasepool {
                let net = try loadedNet(directory)
                let rec = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: recordPath)).arrays
                let clip = rec["clip"]!
                let x = clip.reshaped([1, clip.dim(0), clip.dim(1), clip.dim(2), 3]).asType(.float32)
                let states = net.hiddenStates(x)
                let features = states[states.count - 1]
                var seams = [("patch", cosine(states[0][0], rec["patch"]!)), ("mid", cosine(states[12][0], rec["mid"]!)),
                             ("final", cosine(features[0], rec["output"]!))]
                if let pooledReference = rec["pooled"], let logitsReference = rec["logits"] {
                    XCTAssertTrue(net.classifies, "\(name) builds its classifier")
                    let pooled = try XCTUnwrap(net.pooled(features))
                    seams.append(("pooled", cosine(pooled[0], pooledReference)))
                    let logits = try XCTUnwrap(net.classifier)(pooled)[0]
                    seams.append(("logits", cosine(logits, logitsReference)))
                    XCTAssertEqual(logits.argMax().item(Int.self), logitsReference.argMax().item(Int.self), "\(name) top class")
                    XCTAssertEqual(net.configuration.labels.count, logitsReference.dim(0))
                }
                print("VALIDATION PARITY vjepa2 \(name): " + seams.map { "\($0.0) \($0.1)" }.joined(separator: ", "))
                for (seam, similarity) in seams { XCTAssertGreaterThan(similarity, 0.999, "\(name) \(seam) diverges") }
                XCTAssertEqual(net.configuration.shortestEdge, net.configuration.cropSize == 384 ? 438 : 292)

                // End to end through the public factory: an image in, the embedding and, from a
                // classification release, every class ranked with its label.
                let backend = try NFKMLXVJEPA2.backend(directoryURL: URL(fileURLWithPath: directory))
                let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputImage: Self.patternImage(seed: 1)]))
                XCTAssertEqual((result.output(forKey: NFKOutputEmbedding) as? [NSNumber])?.count, net.configuration.hiddenSize)
                if net.classifies {
                    let classes = try XCTUnwrap(result.output(forKey: NFKOutputClassifications) as? [NFKClassification])
                    XCTAssertEqual(classes.count, net.configuration.labels.count, "\(name) ranks every class")
                    XCTAssertEqual(classes.map(\.confidence), classes.map(\.confidence).sorted(by: >), "\(name) most confident first")
                    XCTAssertEqual(classes.first?.label, net.configuration.labels[classes.first!.classIndex])
                }
            }
            NFKMLXGPU.clearCache()
            measured.append(name)
        }
        try XCTSkipIf(measured.isEmpty, "set IK_VAL_VJEPA2_<RELEASE> and IK_PARITY_VJEPA2_<RELEASE>")
    }

    /// The backend embeds an image end to end (video processor + tubelet embedding + rotary encoder +
    /// mean pooling): a released-length feature vector, finite, deterministic, and input-sensitive.
    func testTheBackendEmbedsAnImage() throws {
        try requireMLXRuntime()
        let env = NFKMLXValidationConfig.environment
        guard let directory = env["IK_VAL_VJEPA2"] else { throw XCTSkip("set IK_VAL_VJEPA2") }
        let backend = try NFKMLXVJEPA2.backend(directoryURL: URL(fileURLWithPath: directory))

        func embed(_ image: CGImage) throws -> [NSNumber] {
            let request = NFKInferenceRequest(inputs: [NFKInputImage: image])
            let result = try backend.runInference(for: request)
            guard let embedding = result.output(forKey: NFKOutputEmbedding) as? [NSNumber] else {
                XCTFail("the backend produced no embedding"); return []
            }
            return embedding
        }

        let config = try NFKMLXVJEPA2Configuration(
            configurationURL: URL(fileURLWithPath: directory).appendingPathComponent("config.json"))
        let first = try embed(Self.patternImage(seed: 1))
        let second = try embed(Self.patternImage(seed: 2))
        let firstAgain = try embed(Self.patternImage(seed: 1))

        XCTAssertEqual(first.count, config.hiddenSize, "embedding width is the hidden size")
        XCTAssertTrue(first.allSatisfy { $0.floatValue.isFinite }, "the embedding is finite")
        XCTAssertEqual(first, firstAgain, "the same image embeds deterministically")
        XCTAssertNotEqual(first, second, "distinct images embed differently")
    }

    /// A deterministic RGB test image with a seed-dependent pattern, written straight to a byte buffer.
    static func patternImage(seed: Int) -> CGImage {
        let size = 320
        var bytes = [UInt8](repeating: 0, count: size * size * 4)
        for y in 0 ..< size {
            for x in 0 ..< size {
                let o = (y * size + x) * 4
                bytes[o] = UInt8((x * seed) % 256)
                bytes[o + 1] = UInt8((y * (seed + 1)) % 256)
                bytes[o + 2] = UInt8(((x + y) * seed) % 256)
                bytes[o + 3] = 255
            }
        }
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let context = CGContext(data: &bytes, width: size, height: size, bitsPerComponent: 8,
                                bytesPerRow: size * 4, space: colorSpace,
                                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        return context.makeImage()!
    }
}

/// The V-JEPA 2 probe recipe: the objective, schedule, and initialization against the reference's own
/// code (`run_reference.py vjepa2_probe`, `IK_PARITY_VJEPA2_PROBE`), and a fine-tune on a tiny release
/// written to a temporary directory, through the save and the factory reload.
final class NFKMLXVJEPA2TrainingTests: XCTestCase {

    override func tearDown() {
        if NFKMLXGPU.metalLibraryURL != nil {
            NFKMLXGPU.clearCache()
        }
        super.tearDown()
    }

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil, "no Metal library for MLX; run Tools/mlx-metallib.sh")
    }

    private func probeRecord() throws -> [String: MLXArray] {
        try requireMLXRuntime()
        guard let path = NFKMLXValidationConfig.environment["IK_PARITY_VJEPA2_PROBE"],
              FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_PARITY_VJEPA2_PROBE (run_reference.py vjepa2_probe)")
        }
        return try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: path)).arrays
    }

    private static var tiny: NFKMLXVJEPA2Configuration {
        var c = NFKMLXVJEPA2Configuration()
        c.hiddenSize = 64
        c.numHiddenLayers = 2
        c.numAttentionHeads = 4
        c.framesPerClip = 2
        c.cropSize = 32
        c.shortestEdge = 36
        return c
    }

    /// A tiny encoder release (no head) written the way a released directory is laid out.
    private func tinyRelease() throws -> URL {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("vjepa2-tiny-\(UUID().uuidString)")
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        MLXRandom.seed(3)
        let encoder = NFKMLXVJEPA2Net(Self.tiny)
        try NFKMLXVJEPA2.save(encoder, toDirectoryURL: directory)
        return directory
    }

    /// PARITY: the objective is the reference's `CrossEntropyLoss` on identical logits and labels.
    func testTheObjectiveMatchesTheReference() throws {
        let record = try probeRecord()
        let loss = NFKMLXVJEPA2Objective().loss(logits: record["logits"]!, labels: record["labels"]!)
        let reference = record["output"]!.item(Float.self)
        print("VALIDATION PARITY vjepa2 probe loss: \(loss.item(Float.self)) vs \(reference)")
        XCTAssertEqual(loss.item(Float.self), reference, accuracy: 1e-6)
    }

    /// PARITY: `warmupCosine` is the reference's `WarmupCosineLRSchedule`, stepped before each update.
    func testTheScheduleMatchesTheReference() throws {
        let record = try probeRecord()
        let recipe = NFKMLXLearningRateSchedule.warmupCosine(steps: 12)
        let warm = NFKMLXLearningRateSchedule.warmupCosine(steps: 12, warmupSteps: 3, startScale: 0.2, endScale: 0.02)
        let defaults = record["lr_default"]!.asArray(Float.self), warmed = record["lr_warmup"]!.asArray(Float.self)
        for step in 0 ..< 12 {
            XCTAssertEqual(5e-3 * recipe.multiplier(step), defaults[step], accuracy: 1e-9, "default schedule step \(step)")
            XCTAssertEqual(5e-3 * warm.multiplier(step), warmed[step], accuracy: 1e-9, "warm-up schedule step \(step)")
        }
    }

    /// PARITY: a fresh probe draws each tensor at the standard deviation the reference's
    /// `AttentivePooler` gives it, the residual rescaling included.
    func testAFreshProbeIsInitializedAsTheReference() throws {
        let record = try probeRecord()
        let geometry = record["width"]!.asArray(Int32.self)
        var configuration = NFKMLXVJEPA2Configuration()
        configuration.hiddenSize = Int(geometry[0])
        configuration.numAttentionHeads = 8
        configuration.poolerLayers = Int(geometry[1]) - 1
        configuration.labels = (0 ..< 16).map(String.init)
        let net = NFKMLXVJEPA2Net(configuration)
        let pooler = try XCTUnwrap(net.pooler)
        MLXRandom.seed(11)
        NFKMLXVJEPA2.initializeAsReference(pooler, layers: configuration.poolerLayers)
        let parameters = Dictionary(uniqueKeysWithValues: pooler.parameters().flattened())
        var compared = 0
        for (key, value) in record where key.hasPrefix("std/") {
            let name = String(key.dropFirst("std/".count))
            let mine = try XCTUnwrap(parameters[name], name)
            let std = sqrt(((mine - mine.mean()) * (mine - mine.mean())).mean()).item(Float.self)
            let reference = value.item(Float.self)
            // The query token has 512 draws, the matrices a quarter million or more.
            let tolerance: Float = name == "query_tokens" ? 0.12 : 0.02
            XCTAssertEqual(std / reference, 1, accuracy: tolerance, "\(name): \(std) vs \(reference)")
            compared += 1
        }
        for (name, value) in parameters where name.hasSuffix(".bias") {
            XCTAssertEqual(abs(value).max().item(Float.self), 0, "\(name) starts at zero")
        }
        XCTAssertEqual(compared, 1 + 3 * 6 + 5, "every recorded tensor is compared")
    }

    /// A probe fine-tune on a tiny release: the loss falls, the probe moves, the encoder does not, and
    /// the classifier-only policy leaves the pooler where it was.
    func testAProbeFineTuneMovesOnlyTheProbe() throws {
        try requireMLXRuntime()
        let release = try tinyRelease()
        for trainable in [NFKMLXVJEPA2Trainable.probe, .classifier] {
            let net = try NFKMLXVJEPA2.network(directoryURL: release, labels: ["a", "b", "c"])
            XCTAssertTrue(net.classifies)
            let before = Dictionary(uniqueKeysWithValues: net.parameters().flattened().map { ($0.0, $0.1 + 0) })
            eval(Array(before.values))
            MLXRandom.seed(5)
            let clips = (0 ..< 3).map { _ in MLXRandom.normal([1, 2, 32, 32, 3]) }
            let losses = try NFKMLXVJEPA2.fineTune(net, examples: { (clips[$0 % 3], $0 % 3) }, trainable: trainable,
                                                  learningRate: 1e-3, steps: 30)
            let first = losses.prefix(3).reduce(0, +), last = losses.suffix(3).reduce(0, +)
            XCTAssertLessThan(last, first, "\(trainable): the loss falls")
            for (name, value) in net.parameters().flattened() {
                let moved = abs(value - before[name]!).max().item(Float.self) > 0
                let shouldMove = name.hasPrefix("classifier.") || (trainable == .probe && name.hasPrefix("pooler."))
                // A LayerNorm scale or a bias can stay at its start when its gradient happens to vanish, so
                // only the linear weights are required to move.
                if (shouldMove && name.hasSuffix("proj.weight")) || name == "classifier.weight" {
                    XCTAssertTrue(moved, "\(trainable): \(name) moves")
                }
                if !shouldMove {
                    XCTAssertFalse(moved, "\(trainable): \(name) stays frozen")
                }
            }
        }
    }

    /// The round trip: a fine-tuned probe saved as a directory reloads through the factory with the
    /// same logits, and the backend ranks the consumer's own labels.
    func testAFineTunedProbeReloadsThroughTheFactory() throws {
        try requireMLXRuntime()
        let net = try NFKMLXVJEPA2.network(directoryURL: try tinyRelease(), labels: ["cat", "dog"])
        MLXRandom.seed(9)
        let clip = MLXRandom.normal([1, 2, 32, 32, 3])
        try NFKMLXVJEPA2.fineTune(net, examples: { _ in (clip, 1) }, steps: 3)
        let saved = FileManager.default.temporaryDirectory.appendingPathComponent("vjepa2-probe-\(UUID().uuidString)")
        addTeardownBlock { try? FileManager.default.removeItem(at: saved) }
        try NFKMLXVJEPA2.save(net, toDirectoryURL: saved)

        let reloaded = try NFKMLXVJEPA2.network(directoryURL: saved)
        XCTAssertEqual(reloaded.configuration.labels, ["cat", "dog"])
        let difference = abs(reloaded.classLogits(clip)! - net.classLogits(clip)!).max().item(Float.self)
        XCTAssertLessThan(difference, 1e-6, "the reloaded probe reproduces the logits")

        let backend = try NFKMLXVJEPA2.backend(directoryURL: saved)
        let result = try backend.runInference(
            for: NFKInferenceRequest(inputs: [NFKInputImage: NFKMLXVJEPA2Tests.patternImage(seed: 2)]))
        let classes = try XCTUnwrap(result.output(forKey: NFKOutputClassifications) as? [NFKClassification])
        XCTAssertEqual(Set(classes.map(\.label)), ["cat", "dog"])
    }

    /// Retargeting keeps a release's classifier when the class count matches and replaces it when not.
    func testRetargetingKeepsAMatchingClassifier() throws {
        try requireMLXRuntime()
        let trained = try NFKMLXVJEPA2.network(directoryURL: try tinyRelease(), labels: ["x", "y"])
        let saved = FileManager.default.temporaryDirectory.appendingPathComponent("vjepa2-head-\(UUID().uuidString)")
        addTeardownBlock { try? FileManager.default.removeItem(at: saved) }
        try NFKMLXVJEPA2.save(trained, toDirectoryURL: saved)
        let renamed = try NFKMLXVJEPA2.network(directoryURL: saved, labels: ["left", "right"])
        XCTAssertEqual(abs(renamed.classifier!.weight - trained.classifier!.weight).max().item(Float.self), 0)
        XCTAssertEqual(abs(renamed.pooler!.queryTokens - trained.pooler!.queryTokens).max().item(Float.self), 0)
        let widened = try NFKMLXVJEPA2.network(directoryURL: saved, labels: ["a", "b", "c"])
        XCTAssertEqual(widened.classifier!.weight.dim(0), 3)
        XCTAssertEqual(abs(widened.pooler!.queryTokens - trained.pooler!.queryTokens).max().item(Float.self), 0,
                       "a release's pooler loads under a new class set")
    }
}
