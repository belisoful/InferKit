//
//  NFKMLXBasicPitchTrainingTests.swift
//  InferKitMLXTests
//
//  Basic Pitch's full fine-tune: the trainable layout and its Keras batch normalization, the Keras
//  optimizer, the unit-norm constraint, the objective, the recipe, and the round trip through the
//  model's own factory. The parity tests read the released weights (`IK_VAL_BASIC_PITCH`, for the
//  constant-Q front end) and a record from `run_reference.py basic_pitch_training`
//  (`IK_PARITY_BASIC_PITCH_TRAINING`), and skip without them.
//

import XCTest
import InferKit
import MLX
import MLXNN
@testable import InferKitMLX

final class NFKMLXBasicPitchTrainingTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    override func setUp() {
        super.setUp()
        guard NFKMLXGPU.metalLibraryURL != nil else { return }
        NFKMLXRandom.seed(20_260_925)
    }

    override func tearDown() {
        if NFKMLXGPU.metalLibraryURL != nil {
            NFKMLXGPU.clearCache()
        }
        super.tearDown()
    }

    private let windowSamples = 43844

    /// A trainable network with a random constant-Q front end, which a fresh one leaves at zero.
    private func trainableNet() -> NFKMLXBasicPitchNet {
        let net = NFKMLXBasicPitch.makeNet(.icassp2022Trainable)
        net.update(parameters: ModuleParameters.unflattened([
            ("cqt.kernel_a", MLXRandom.normal(net.cqt.kernelA.shape) * 0.05),
            ("cqt.kernel_b", MLXRandom.normal(net.cqt.kernelB.shape) * 0.05),
            ("cqt.lowpass", MLXRandom.normal(net.cqt.lowpass.shape) * 0.05)]))
        net.resetToReferenceInitialization()
        return net
    }

    /// One window of a two-note tone and targets that light those notes.
    private func example() -> NFKMLXBasicPitchExample {
        let samples = (0 ..< windowSamples).map { index -> Float in
            let t = Float(index) / 22050
            return 0.3 * sinf(2 * .pi * 261.63 * t) + 0.2 * sinf(2 * .pi * 392.0 * t)
        }
        var note = [Float](repeating: 0, count: 172 * 88)
        var contour = [Float](repeating: 0, count: 172 * 264)
        var onset = [Float](repeating: 0, count: 172 * 88)
        for key in [39, 46] {
            for frame in 0 ..< 172 {
                note[frame * 88 + key] = 1
                contour[frame * 264 + 3 * key + 1] = 1
            }
            onset[key] = 1
        }
        return NFKMLXBasicPitchExample(audio: MLXArray(samples, [1, windowSamples, 1]),
                                       contour: MLXArray(contour, [1, 172, 264]),
                                       note: MLXArray(note, [1, 172, 88]),
                                       onset: MLXArray(onset, [1, 172, 88]))
    }

    /// The network's parameters as evaluated copies. `update(parameters:)` writes new values into the
    /// arrays a module already holds, so a dictionary of the live arrays taken before a run reads the
    /// values after it.
    private func parameters(_ net: Module) -> [String: MLXArray] {
        let copies = net.parameters().flattened().map { ($0.0, $0.1 + 0) }
        eval(copies.map { $0.1 })
        return Dictionary(uniqueKeysWithValues: copies)
    }

    // MARK: The trainable layout

    func testTheSeparateLayoutCarriesThreeNormalizationsAndNoFoldedScale() throws {
        try requireMLXRuntime()
        let names = Set(NFKMLXBasicPitch.makeNet(.icassp2022Trainable).parameters().flattened().map(\.0))
        for layer in ["log_norm", "contour_norm", "onset_norm"] {
            for part in ["weight", "bias", "running_mean", "running_var"] {
                XCTAssertTrue(names.contains("\(layer).\(part)"), "missing \(layer).\(part)")
            }
        }
        XCTAssertFalse(names.contains("norm_scale"))
        XCTAssertFalse(names.contains("norm_bias"))
        let folded = Set(NFKMLXBasicPitch.makeNet().parameters().flattened().map(\.0))
        XCTAssertFalse(folded.contains { $0.hasPrefix("log_norm.") })
    }

    func testANewNetworkStartsInEvaluationMode() throws {
        try requireMLXRuntime()
        XCTAssertFalse(NFKMLXBasicPitch.makeNet(.icassp2022Trainable).training)
    }

    func testTheMovingStatisticsAreNotTrainable() throws {
        try requireMLXRuntime()
        let net = NFKMLXBasicPitch.makeNet(.icassp2022Trainable)
        net.unfreeze()
        let trainable = Set(net.trainableParameters().flattened().map(\.0))
        XCTAssertTrue(trainable.contains("contour_norm.weight"))
        XCTAssertFalse(trainable.contains { $0.hasSuffix("running_mean") || $0.hasSuffix("running_var") })
    }

    /// TensorFlow's fused batch normalization folds the unbiased batch variance into its moving
    /// variance. Two elements with variance 1 have unbiased variance 2, so one step from a moving
    /// variance of 1 reaches `0.99 + 0.01 · 2`, where MLXNN's `BatchNorm` would reach 1.
    func testTheNormalizationFoldsTheUnbiasedVarianceIntoItsMovingVariance() throws {
        try requireMLXRuntime()
        let normalization = NFKBasicPitchBatchNorm(channels: 1)
        normalization.train(true)
        let normalized = normalization(MLXArray([Float(1), 3], [1, 1, 2, 1]))
        eval(normalized, normalization)
        XCTAssertEqual(normalization.runningVar.item(Float.self), 1.01, accuracy: 1e-6)
        XCTAssertEqual(normalization.runningMean.item(Float.self), 0.02, accuracy: 1e-6)
        // Normalization itself divides by the biased variance, 1.
        XCTAssertEqual(normalized.asArray(Float.self)[1], 1 / (1 + 1e-3).squareRoot(), accuracy: 1e-5)
    }

    // MARK: The optimizer and the constraint

    /// Keras adds epsilon after the square root of the uncorrected second moment. With a gradient small
    /// enough to meet epsilon, that separates it from PyTorch's placement.
    func testKerasAdamTakesKerasFirstStep() throws {
        try requireMLXRuntime()
        let model = Linear(weight: MLXArray([Float(0.5)], [1, 1]))
        let gradient: Float = 2e-6
        let optimizer = NFKMLXKerasAdam(learningRate: 1e-3)
        optimizer.update(model: model, gradients: ModuleParameters.unflattened([("weight", MLXArray([gradient], [1, 1]))]))
        eval(model)
        let m = 0.1 * Double(gradient), v = 0.001 * Double(gradient) * Double(gradient)
        let alpha = 1e-3 * (1 - 0.999).squareRoot() / (1 - 0.9)
        let expected = 0.5 - alpha * m / (v.squareRoot() + 1e-7)
        XCTAssertEqual(Double(model.weight.item(Float.self)), expected, accuracy: 1e-7)
        let torchPlacement = 0.5 - 1e-3 * (m / 0.1) / ((v / 0.001).squareRoot() + 1e-7)
        XCTAssertGreaterThan(abs(expected - torchPlacement), 1e-5, "the test must separate the two placements")
    }

    func testTheConstraintProjectsEveryTrainableKernelToUnitNorm() throws {
        try requireMLXRuntime()
        let net = trainableNet()
        net.unfreeze()
        net.noteOut.freeze()
        let frozenBefore = net.noteOut.weight
        net.unitNormalizeKernels()
        eval(net)
        for (name, convolution) in net.constrainedConvolutions where name != "note_out" {
            let norms = sqrt(convolution.weight.square().sum(axes: [1, 2, 3])).asArray(Float.self)
            XCTAssertEqual(norms.min() ?? 0, 1, accuracy: 1e-5, name)
            XCTAssertEqual(norms.max() ?? 0, 1, accuracy: 1e-5, name)
        }
        XCTAssertEqual(abs(net.noteOut.weight - frozenBefore).max().item(Float.self), 0, "a frozen kernel is left alone")
    }

    // MARK: The objective

    func testTheObjectiveScoresTheTargetsLowerThanTheirComplement() throws {
        try requireMLXRuntime()
        let batch = example()
        let objective = NFKMLXBasicPitchObjective()
        let close = objective.loss(predicted: (batch.contour * 0.9 + 0.05, batch.note * 0.9 + 0.05, batch.onset * 0.9 + 0.05),
                                   target: (batch.contour, batch.note, batch.onset)).item(Float.self)
        let far = objective.loss(predicted: (0.95 - batch.contour * 0.9, 0.95 - batch.note * 0.9, 0.95 - batch.onset * 0.9),
                                 target: (batch.contour, batch.note, batch.onset)).item(Float.self)
        XCTAssertLessThan(close, far)
    }

    func testTheWeightedOnsetTermStaysFiniteWithNoOnsets() throws {
        try requireMLXRuntime()
        let silent = MLXArray.zeros([1, 172, 88])
        let objective = NFKMLXBasicPitchObjective(weightedOnset: true)
        let term = objective.onsetTerm(target: silent, predicted: MLXArray.ones([1, 172, 88]) * 0.2)
        XCTAssertTrue(term.item(Float.self).isFinite)
    }

    // MARK: The recipe

    func testAFineTuneLowersTheLossTrainsTheHeadsAndKeepsTheFrontEnd() throws {
        try requireMLXRuntime()
        let net = trainableNet()
        let before = parameters(net)
        let batch = example()
        let losses = try NFKMLXBasicPitch.fineTune(net, examples: { _ in batch }, steps: 6)
        XCTAssertLessThan(losses.last!, losses.first!)
        let after = parameters(net)
        for key in NFKMLXBasicPitch.frontEndKeys {
            XCTAssertEqual(abs(after[key]! - before[key]!).max().item(Float.self), 0, "\(key) is a constant")
        }
        XCTAssertGreaterThan(abs(after["contour_conv.weight"]! - before["contour_conv.weight"]!).max().item(Float.self), 0)
        XCTAssertGreaterThan(abs(after["contour_norm.running_mean"]! - before["contour_norm.running_mean"]!).max().item(Float.self), 0,
                             "a training step folds the batch into the moving statistics")
        for (name, convolution) in net.constrainedConvolutions {
            let norms = sqrt(convolution.weight.square().sum(axes: [1, 2, 3])).asArray(Float.self)
            XCTAssertEqual(norms.max() ?? 0, 1, accuracy: 1e-5, "\(name) stays at unit norm")
        }
        XCTAssertFalse(net.training, "the run returns the network ready to infer")
    }

    func testAFoldedNetworkIsRefused() throws {
        try requireMLXRuntime()
        let batch = example()
        XCTAssertThrowsError(try NFKMLXBasicPitch.fineTune(NFKMLXBasicPitch.makeNet(), examples: { _ in batch }, steps: 1)) { error in
            guard case NFKMLXError.unsupportedConfiguration = error else {
                return XCTFail("expected unsupportedConfiguration, got \(error)")
            }
        }
    }

    func testAFineTunedNetworkRoundTripsThroughTheFactory() throws {
        try requireMLXRuntime()
        let net = trainableNet()
        let batch = example()
        try NFKMLXBasicPitch.fineTune(net, examples: { _ in batch }, steps: 2)
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("basic-pitch-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try NFKMLXWeights.save(net, to: url)

        XCTAssertEqual(try NFKMLXBasicPitch.normalization(ofCheckpointAt: url), .separate)
        let reloaded = try NFKMLXBasicPitch.network(weightsURL: url)
        XCTAssertEqual(reloaded.configuration.normalization, .separate)
        let original = net.posteriorgrams(batch.audio)
        let restored = reloaded.posteriorgrams(batch.audio)
        XCTAssertLessThan(abs(original.note - restored.note).max().item(Float.self), 1e-6)
        XCTAssertLessThan(abs(original.contour - restored.contour).max().item(Float.self), 1e-6)

        let backend = try NFKMLXBasicPitch.backend(weightsURL: url)
        XCTAssertTrue(backend.isReady)
    }

    func testReinitializingKeepsTheFrontEndAndResetsTheRest() throws {
        try requireMLXRuntime()
        let source = trainableNet()
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("basic-pitch-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try NFKMLXWeights.save(source, to: url)

        let fresh = try NFKMLXBasicPitch.network(weightsURL: url, reinitializing: true)
        let original = parameters(source), reset = parameters(fresh)
        XCTAssertEqual(abs(reset["cqt.kernel_a"]! - original["cqt.kernel_a"]!).max().item(Float.self), 0)
        XCTAssertGreaterThan(abs(reset["contour_conv.weight"]! - original["contour_conv.weight"]!).max().item(Float.self), 0)
        XCTAssertEqual(abs(reset["contour_conv.bias"]!).max().item(Float.self), 0)
        // The reference's uniform bound for the 3×39 contour kernel over 8 channels in and out.
        let limit = (6.0 / Float(3 * 39 * 8)).squareRoot()
        XCTAssertLessThanOrEqual(abs(reset["contour_conv.weight"]!).max().item(Float.self), limit)
    }

    // MARK: Parity against the reference's own training step

    private struct Reference {
        let record: [String: MLXArray]
        let weightsURL: URL
    }

    /// The released weights in the trainable layout, assembled from the converted release's
    /// constant-Q front end and the record's SavedModel variables, both in PyTorch layout, then loaded
    /// through `network(weightsURL:)` as a converted file would be.
    private func reference() throws -> Reference {
        let environment = NFKMLXValidationConfig.environment
        guard let weightsPath = environment["IK_VAL_BASIC_PITCH"],
              let recordPath = environment["IK_PARITY_BASIC_PITCH_TRAINING"],
              FileManager.default.fileExists(atPath: weightsPath),
              FileManager.default.fileExists(atPath: recordPath) else {
            throw XCTSkip("set IK_VAL_BASIC_PITCH (converted weights) and IK_PARITY_BASIC_PITCH_TRAINING "
                          + "(run_reference.py basic_pitch_training)")
        }
        let record = try loadArrays(url: URL(fileURLWithPath: recordPath))
        let released = try loadArrays(url: URL(fileURLWithPath: weightsPath))
        var trainable = released.filter { NFKMLXBasicPitch.frontEndKeys.contains($0.key) }
        for (key, value) in record where key.hasPrefix("w::") {
            trainable[String(key.dropFirst(3))] = value
        }
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("basic-pitch-trainable-\(UUID().uuidString).safetensors")
        try save(arrays: trainable, url: url)
        return Reference(record: record, weightsURL: url)
    }

    /// A recorded tensor in MLX's layout: a PyTorch-layout convolution kernel is transposed.
    private func layout(_ value: MLXArray) -> MLXArray {
        value.ndim == 4 ? value.transposed(0, 2, 3, 1) : value
    }

    private func recordedExample(_ record: [String: MLXArray]) -> NFKMLXBasicPitchExample {
        NFKMLXBasicPitchExample(audio: record["audio"]!, contour: record["target_contour"]!,
                                note: record["target_note"]!, onset: record["target_onset"]!)
    }

    private func cosine(_ a: MLXArray, _ b: MLXArray) -> Double {
        let x = a.reshaped([-1]).asArray(Float.self).map(Double.init)
        let y = b.reshaped([-1]).asArray(Float.self).map(Double.init)
        let dot = zip(x, y).reduce(0) { $0 + $1.0 * $1.1 }
        let (xx, yy) = (x.reduce(0) { $0 + $1 * $1 }, y.reduce(0) { $0 + $1 * $1 })
        guard xx > 0, yy > 0 else { return xx == yy ? 1 : 0 }
        return dot / (xx * yy).squareRoot()
    }

    func testTheTrainableLayoutMatchesTheReferenceForwardInBothModes() throws {
        try requireMLXRuntime()
        let reference = try reference()
        defer { try? FileManager.default.removeItem(at: reference.weightsURL) }
        let record = reference.record
        let batch = recordedExample(record)

        let net = try NFKMLXBasicPitch.network(weightsURL: reference.weightsURL)
        XCTAssertEqual(net.configuration.normalization, .separate)
        let inference = net.posteriorgrams(batch.audio)
        net.train(true)
        let training = net.posteriorgrams(batch.audio)

        for (name, ours, theirs) in [("eval contour", inference.contour, record["eval_contour"]!),
                                     ("eval note", inference.note, record["eval_note"]!),
                                     ("eval onset", inference.onset, record["eval_onset"]!),
                                     ("train contour", training.contour, record["train_contour"]!),
                                     ("train note", training.note, record["train_note"]!),
                                     ("train onset", training.onset, record["train_onset"]!)] {
            let similarity = cosine(ours, theirs)
            let largest = abs(ours - theirs).max().item(Float.self)
            print("PARITY basic-pitch-training \(name): cosine \(similarity), max |difference| \(largest)")
            XCTAssertGreaterThan(similarity, 0.99999, name)
        }
    }

    func testTheObjectiveMatchesTheReferenceOnItsOwnPredictions() throws {
        try requireMLXRuntime()
        let reference = try reference()
        defer { try? FileManager.default.removeItem(at: reference.weightsURL) }
        let record = reference.record
        let predicted = (record["train_contour"]!, record["train_note"]!, record["train_onset"]!)
        let target = (record["target_contour"]!, record["target_note"]!, record["target_onset"]!)

        let terms = NFKMLXBasicPitchObjective().components(predicted: predicted, target: target)
        let weighted = NFKMLXBasicPitchObjective(weightedOnset: true).onsetTerm(target: target.2, predicted: predicted.2)
        for (name, ours, theirs) in [("contour", terms.contour, record["loss_contour"]!),
                                     ("note", terms.note, record["loss_note"]!),
                                     ("onset", terms.onset, record["loss_onset"]!),
                                     ("weighted onset", weighted, record["loss_onset_weighted"]!)] {
            let expected = theirs.item(Float.self)
            print("PARITY basic-pitch-loss \(name): ours \(ours.item(Float.self)), reference \(expected)")
            XCTAssertEqual(ours.item(Float.self), expected, accuracy: abs(expected) * 1e-5, name)
        }
    }

    func testTheGradientMatchesTheReference() throws {
        try requireMLXRuntime()
        let reference = try reference()
        defer { try? FileManager.default.removeItem(at: reference.weightsURL) }
        let record = reference.record
        let batch = recordedExample(record)
        let net = try NFKMLXBasicPitch.network(weightsURL: reference.weightsURL)
        net.unfreeze()
        net.cqt.freeze()
        net.train(true)
        let objective = NFKMLXBasicPitchObjective()
        let lossAndGradient = valueAndGrad(model: net) { net, _ in [objective(net, batch)] }
        let (_, gradients) = lossAndGradient(net, [])
        let ours = Dictionary(uniqueKeysWithValues: gradients.flattened())
        let recorded = record.compactMap { key, value -> (String, MLXArray)? in
            key.hasPrefix("grad::") ? (String(key.dropFirst(6)), layout(value)) : nil
        }
        let largest = recorded.map { abs($0.1).max().item(Float.self) }.max() ?? 0
        var worst = 1.0
        for (name, reference) in recorded {
            guard let gradient = ours[name] else { return XCTFail("no gradient for \(name)") }
            // A bias feeding a batch normalization has a gradient of exactly zero, because the
            // normalization subtracts the channel mean; both sides hold only rounding there, and a
            // cosine of rounding measures nothing. Those are held to an absolute bound instead.
            if abs(reference).max().item(Float.self) < largest * 1e-5 {
                XCTAssertLessThan(abs(gradient).max().item(Float.self), largest * 1e-5, "\(name) is zero")
                continue
            }
            let similarity = cosine(gradient, reference)
            worst = min(worst, similarity)
            XCTAssertGreaterThan(similarity, 0.9999, name)
        }
        XCTAssertNil(ours["contour_norm.running_mean"], "a moving statistic takes no gradient")
        print("PARITY basic-pitch-gradient: worst cosine \(worst) over \(recorded.count) parameters")
    }

    /// Keras's first step from the reference's own gradients, then the unit-norm constraint, isolated
    /// from any difference in the forward pass.
    func testKerasAdamAndTheConstraintReproduceTheReferencesFirstStep() throws {
        try requireMLXRuntime()
        let reference = try reference()
        defer { try? FileManager.default.removeItem(at: reference.weightsURL) }
        let record = reference.record
        let net = try NFKMLXBasicPitch.network(weightsURL: reference.weightsURL)
        net.unfreeze()
        net.cqt.freeze()
        let gradients = record.compactMap { key, value -> (String, MLXArray)? in
            key.hasPrefix("grad::") ? (String(key.dropFirst(6)), layout(value)) : nil
        }
        NFKMLXKerasAdam(learningRate: 1e-3).update(model: net, gradients: ModuleParameters.unflattened(gradients))
        net.unitNormalizeKernels()
        eval(net)
        let ours = parameters(net)
        var worst: Float = 0
        for (key, value) in record where key.hasPrefix("step1::") {
            let name = String(key.dropFirst(7))
            guard !name.hasSuffix("running_mean"), !name.hasSuffix("running_var") else { continue }
            let difference = abs(ours[name]! - layout(value)).max().item(Float.self)
            worst = max(worst, difference)
            XCTAssertLessThan(difference, 2e-6, name)
        }
        print("PARITY basic-pitch-optimizer: worst |difference| after one step \(worst)")
    }

    func testThreeRecipeStepsFollowTheReferencesTrainOnBatch() throws {
        try requireMLXRuntime()
        let reference = try reference()
        defer { try? FileManager.default.removeItem(at: reference.weightsURL) }
        let record = reference.record
        let batch = recordedExample(record)
        let net = try NFKMLXBasicPitch.network(weightsURL: reference.weightsURL)
        let released = parameters(net)

        let losses = try NFKMLXBasicPitch.fineTune(net, examples: { _ in batch }, steps: 3)
        let expected = record["output"]!.asArray(Float.self)
        print("PARITY basic-pitch-steps: ours \(losses), reference \(expected)")
        for (ours, theirs) in zip(losses, expected) {
            XCTAssertEqual(ours, theirs, accuracy: abs(theirs) * 1e-4)
        }

        let ours = parameters(net)
        // Every parameter and moving statistic is compared by how far it moved from the release, so a
        // value that never moved cannot pass on its resemblance to where it started. A bias feeding a
        // batch normalization has a gradient of exactly zero, and Adam moves it only by rounding: at
        // most 1.1e-5 over the three steps in the reference, against about 2.8e-3 for a parameter that
        // trains. Those are held to an absolute bound 50 times below a real step.
        let gradients = record.compactMap { key, value -> (String, MLXArray)? in
            key.hasPrefix("grad::") ? (String(key.dropFirst(6)), value) : nil
        }
        let largest = gradients.map { abs($0.1).max().item(Float.self) }.max() ?? 0
        let untrained = Set(gradients.filter { abs($0.1).max().item(Float.self) < largest * 1e-5 }.map { $0.0 })
        var worst = 1.0
        for (key, value) in record where key.hasPrefix("step3::") {
            let name = String(key.dropFirst(7))
            if untrained.contains(name) {
                XCTAssertLessThan(abs(ours[name]! - layout(value)).max().item(Float.self), 5e-5, "\(name) holds still")
                continue
            }
            let start = released[name]!
            let similarity = cosine(ours[name]! - start, layout(value) - start)
            worst = min(worst, similarity)
            XCTAssertGreaterThan(similarity, 0.999, name)
        }
        XCTAssertEqual(untrained, ["contour_conv.bias", "onset_conv.bias"])
        print("PARITY basic-pitch-steps: worst cosine of the three steps' movement \(worst)")
    }

    // MARK: The data adapter

    /// `run_reference.py`'s `BASIC_PITCH_TARGET_NOTES`, chosen for the edges of the reference's rules.
    private let edgeNotes: [(Double, Double, Int, Int)] = [
        (0.10, 0.90, 60, 100), (0.30, 1.40, 21, 90), (0.30, 1.40, 108, 90), (0.50, 0.70, 20, 90),
        (0.50, 0.70, 109, 90), (1.0 + 0.5 / 86, 1.60, 64, 80), (1.20, 1.20, 67, 70),
        (1.50, 2.30, 60, 100), (2.00, 2.60, 60, 0), (3.40, 9.00, 72, 110)]

    private var edgeSequence: NFKMIDISequence {
        NFKMIDISequence(notes: edgeNotes.map {
            NFKMIDINote(pitch: $0.2, startSeconds: $0.0, endSeconds: $0.1, velocity: $0.3)
        })
    }

    /// The whole-track targets and three windows against `mirdata`'s `to_sparse_index`, the reference's
    /// `sparse2dense`, and `extract_window`, exactly: every target cell and every audio sample.
    func testTheTargetsAndWindowsMatchTheReferencesConstruction() throws {
        guard let path = NFKMLXValidationConfig.environment["IK_PARITY_BASIC_PITCH_TARGETS"],
              FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_PARITY_BASIC_PITCH_TARGETS (run_reference.py basic_pitch_targets)")
        }
        try requireMLXRuntime()
        let record = try loadArrays(url: URL(fileURLWithPath: path))
        let audio = record["audio"]!.reshaped([-1]).asArray(Float.self)
        let track = NFKBasicPitchTargets.track(notes: edgeSequence.notes, durationSeconds: Double(audio.count) / 22050)
        XCTAssertEqual(track.frames, Int(record["output"]!.item(Float.self)))
        XCTAssertEqual(track.note, record["track_note"]!.reshaped([-1]).asArray(Float.self))
        XCTAssertEqual(track.onset, record["track_onset"]!.reshaped([-1]).asArray(Float.self))
        XCTAssertEqual(track.contour, record["track_contour"]!.reshaped([-1]).asArray(Float.self))

        let starts = record["window_starts"]!.asArray(Float.self)
        for (index, start) in starts.enumerated() {
            let example = try NFKMLXBasicPitch.trainingExample(samples: audio, sampleRate: 22050, notes: edgeSequence,
                                                               startSeconds: Double(start))
            for (name, ours) in [("audio", example.audio), ("note", example.note),
                                 ("onset", example.onset), ("contour", example.contour)] {
                XCTAssertEqual(ours.reshaped([-1]).asArray(Float.self),
                               record["window\(index)_\(name)"]!.reshaped([-1]).asArray(Float.self),
                               "window \(index) \(name)")
            }
        }
        print("PARITY basic-pitch-targets: \(track.frames) frames and \(starts.count) windows identical")
    }

    func testTheWindowDrawSkipsSilenceAndRepeatsFromItsSeed() throws {
        try requireMLXRuntime()
        let samples = [Float](repeating: 0, count: 22050 * 6)
        let notes = NFKMIDISequence(notes: [NFKMIDINote(pitch: 60, startSeconds: 4.0, endSeconds: 5.9, velocity: 90)])
        let first = try NFKMLXBasicPitch.trainingExamples(samples: samples, sampleRate: 22050, notes: notes, count: 3, seed: 7)
        let again = try NFKMLXBasicPitch.trainingExamples(samples: samples, sampleRate: 22050, notes: notes, count: 3, seed: 7)
        XCTAssertEqual(first.count, 3)
        for (a, b) in zip(first, again) {
            XCTAssertGreaterThan(a.note.max().item(Float.self), 0, "a drawn window holds a note")
            XCTAssertEqual(a.note.asArray(Float.self), b.note.asArray(Float.self))
        }
        XCTAssertThrowsError(try NFKMLXBasicPitch.trainingExample(samples: samples, sampleRate: 22050, notes: notes,
                                                                  startSeconds: 5.0))
    }
}
