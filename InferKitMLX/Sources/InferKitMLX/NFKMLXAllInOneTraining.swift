//
//  NFKMLXAllInOneTraining.swift
//  InferKitMLX
//
//  Fine-tuning All-In-One on a consumer's own annotated tracks: their genre's form, their labels'
//  boundaries, their beat. Every weight trains: the network is 383,577 parameters.
//
//  The recipe is the authors' trainer (`training/trainer.py` at mir-aidj/all-in-one 18e7890). Targets
//  follow `DatasetBase`: each annotation time lands on the frame `librosa.time_to_frames` gives it, and
//  the beat, downbeat, and section activations are widened so the neighboring frames score half as
//  much (the section boundaries a second ring at a quarter). The objective is `compute_losses`: binary
//  cross-entropy with logits on the three activations and cross-entropy on the per-frame function
//  label, masked and averaged, weighted 1, 3, 15, and 0.1. The optimizer is timm's `RAdam` at the
//  configuration's rate 0.005 and weight decay 0.00025, split into timm's decay groups. The network
//  runs the reference's dropouts and stochastic depth while it trains.
//

import Foundation
import MLX
import MLXNN
import MLXOptimizers

/*!
 @abstract A track's annotation as the frame targets All-In-One trains against.
 @discussion Built the way the reference's `DatasetBase` builds a training item, over `frameCount`
 frames at the configuration's frame rate. Introduced in InferKit 0.5.0.
 */
public struct NFKMLXAllInOneTargets {

    /// The widened beat activation, `[1, frames]`.
    public let beat: MLXArray
    /// The widened downbeat activation, `[1, frames]`.
    public let downbeat: MLXArray
    /// The widened section-boundary activation, `[1, frames]`.
    public let section: MLXArray
    /// Each frame's functional label as an index into the configuration's labels, `[1, frames]`.
    public let function: MLXArray

    /// - Parameters:
    ///   - beatTimes: every beat, in seconds.
    ///   - downbeatTimes: every downbeat, in seconds.
    ///   - sectionBoundaries: where each section starts, in seconds, in order.
    ///   - sectionLabels: one label more than there are boundaries: the label before the first boundary,
    ///     then the label each boundary opens. Each is one of `configuration.labels`; Harmonix
    ///     annotations open with `start` and close with `end`.
    ///   - frameCount: the frames the spectrograms span, `spectrograms.dim(2)`.
    ///   - configuration: the network's configuration, for its frame rate and labels.
    public init(beatTimes: [Double], downbeatTimes: [Double], sectionBoundaries: [Double],
                sectionLabels: [String], frameCount: Int,
                configuration: NFKMLXAllInOneConfiguration = .harmonix) throws {
        guard sectionLabels.count == sectionBoundaries.count + 1 else {
            throw NFKMLXError.trainingDataMismatch(
                "\(sectionBoundaries.count) section boundaries need \(sectionBoundaries.count + 1) labels, "
                + "and \(sectionLabels.count) were supplied")
        }
        let indices = try sectionLabels.map { label -> Int32 in
            guard let index = configuration.labels.firstIndex(of: label) else {
                throw NFKMLXError.trainingDataMismatch("\"\(label)\" is not one of the model's labels \(configuration.labels)")
            }
            return Int32(index)
        }
        let frame = { (time: Double) in Self.frame(time, configuration: configuration) }
        func activation(_ times: [Double], neighbors: Int) -> MLXArray {
            var events = [Float](repeating: 0, count: frameCount)
            for index in times.map(frame) where index >= 0 && index < frameCount {
                events[index] = 1
            }
            return MLXArray(Self.widened(events, neighbors: neighbors)).reshaped([1, frameCount])
        }
        beat = activation(beatTimes, neighbors: 1)
        downbeat = activation(downbeatTimes, neighbors: 1)
        section = activation(sectionBoundaries, neighbors: 2)

        let boundaryFrames = sectionBoundaries.map(frame)
        let labels = (0 ..< frameCount).map { index in
            indices[boundaryFrames.filter { $0 <= index }.count]
        }
        function = MLXArray(labels).reshaped([1, frameCount])
    }

    /// `librosa.time_to_frames`: the sample the time truncates to, floor-divided by the hop.
    static func frame(_ time: Double, configuration: NFKMLXAllInOneConfiguration) -> Int {
        let sample = Int(time * Double(configuration.sampleRate))
        return Int((Double(sample) / Double(configuration.hopSize)).rounded(.down))
    }

    /// `widen_temporal_events`: each pass takes the three-frame running maximum and halves every
    /// frame that is not itself an event, so after two passes the rings read 1, 0.5, 0.25.
    static func widened(_ events: [Float], neighbors: Int) -> [Float] {
        var widened = events
        for _ in 0 ..< neighbors {
            widened = widened.indices.map { index in
                // scipy's `maximum_filter1d` reflects at the edge, which repeats the edge frame.
                let lower = max(index - 1, 0), upper = min(index + 1, widened.count - 1)
                return max(widened[lower], widened[index], widened[upper])
            }
            for index in widened.indices where events[index] != 1 && widened[index] > 0 {
                widened[index] *= 0.5
            }
        }
        return widened
    }
}

/*!
 @abstract The objective an All-In-One fine-tune minimizes: the reference's `compute_losses`.
 @discussion Binary cross-entropy with logits on the beat, downbeat, and section activations and
 cross-entropy on the functional label, each masked and averaged over the frames, then weighted.
 `learnsRhythm` adds the beat and downbeat terms, and `learnsStructure` adds the function term (when
 `learnsLabels`) and the section term (when `learnsSegments`), as the reference's switches do.
 Measured against the reference by `run_reference.py allin1_training`. Introduced in InferKit 0.5.0.
 */
public struct NFKMLXAllInOneObjective: Sendable {
    public var beatWeight: Float = 1
    public var downbeatWeight: Float = 3
    public var sectionWeight: Float = 15
    public var functionWeight: Float = 0.1
    public var learnsRhythm = true
    public var learnsStructure = true
    public var learnsSegments = true
    public var learnsLabels = true

    public init() {}

    /// The four weighted terms, each a scalar, the way the reference logs them.
    public func components(_ logits: NFKMLXAllInOneLogits, targets: NFKMLXAllInOneTargets, mask: MLXArray? = nil)
        -> (beat: MLXArray, downbeat: MLXArray, section: MLXArray, function: MLXArray) {
        let weights = mask ?? MLXArray.ones(targets.beat.shape)
        func binary(_ x: MLXArray, _ y: MLXArray) -> MLXArray {
            (weights * (maximum(x, 0) - x * y + log1p(exp(-abs(x))))).mean()
        }
        // [1, labels, frames] against [1, frames]: the log-probability of each frame's own label.
        let logProbabilities = logits.function - logSumExp(logits.function, axis: 1, keepDims: true)
        let picked = takeAlong(logProbabilities, targets.function.expandedDimensions(axis: 1).asType(.int32), axis: 1)
            .squeezed(axis: 1)
        let function = (weights * -picked).mean()
        return (beatWeight * binary(logits.beat, targets.beat),
                downbeatWeight * binary(logits.downbeat, targets.downbeat),
                sectionWeight * binary(logits.section, targets.section),
                functionWeight * function)
    }

    /// The loss, a scalar: the terms the switches select, summed.
    public func loss(_ logits: NFKMLXAllInOneLogits, targets: NFKMLXAllInOneTargets, mask: MLXArray? = nil) -> MLXArray {
        let terms = components(logits, targets: targets, mask: mask)
        var total = MLXArray(Float(0))
        if learnsRhythm {
            total = total + terms.beat + terms.downbeat
        }
        if learnsStructure {
            if learnsLabels {
                total = total + terms.function
            }
            if learnsSegments {
                total = total + terms.section
            }
        }
        return total
    }
}

extension NFKMLXAllInOne {

    /// Builds the network itself, ready to fine-tune, from a released `.pth` or a file
    /// `NFKMLXWeights.save` wrote.
    ///
    /// - Since: InferKit 0.5.0
    public static func network(weightsURL: URL?,
                               configuration: NFKMLXAllInOneConfiguration = .harmonix) throws -> NFKMLXAllInOneNet {
        let net = makeNet(configuration)
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        return net
    }

    /// Fine-tunes every weight of `net` on annotated tracks, returning the loss from each step.
    ///
    /// The whole path is three calls: ``network(weightsURL:configuration:)`` to build, this to train, and
    /// `NFKMLXWeights.save` to write a checkpoint that `backendWithWeightsURL:error:` loads.
    ///
    /// - Parameters:
    ///   - net: the network, from ``network(weightsURL:configuration:)``.
    ///   - examples: supplies one track per step: its stems' spectrograms from
    ///     `NFKMLXAllInOneNet.spectrograms(stems:)` and its ``NFKMLXAllInOneTargets`` over the same
    ///     frames. The reference trains on 300-second segments, so a whole song is one example.
    ///   - objective: the reference's `compute_losses`.
    ///   - optimizer: the update rule. Nil uses timm's `RAdam` at the configuration's rate 0.005 and
    ///     weight decay 0.00025, with no decay on biases or one-dimensional parameters.
    ///   - steps: how many tracks to train on.
    ///   - clipGradientNorm: bounds the global gradient norm before the update. The reference does not clip.
    ///   - learningRateSchedule: multiplies the rate at each step. Nil holds it constant; the reference
    ///     lowers it on a validation plateau, which a run without a validation set cannot measure.
    ///   - checkpoint: writes the network periodically, so a suspended run keeps its progress.
    ///   - observer: receives each step and can end the run early.
    ///
    /// The spectrogram front end and the post-processing thresholds stay fixed, as the reference trains
    /// on precomputed spectrograms. A run is minutes; call it off the render thread.
    @discardableResult
    public static func fineTune(
        _ net: NFKMLXAllInOneNet,
        examples: (Int) -> (spectrograms: MLXArray, targets: NFKMLXAllInOneTargets),
        objective: NFKMLXAllInOneObjective = NFKMLXAllInOneObjective(),
        optimizer: Optimizer? = nil,
        steps: Int,
        clipGradientNorm: Float? = nil,
        learningRateSchedule: NFKMLXLearningRateSchedule? = nil,
        checkpoint: NFKMLXTrainingCheckpoint? = nil,
        observer: NFKMLXTrainer.Observer? = nil
    ) throws -> [Float] {
        try NFKMLXFineTune.run(
            net,
            freezing: {
                net.unfreeze()
                net.frontEnd.freeze()
                net.postprocess.freeze()
            },
            optimizer: optimizer,
            reference: { NFKMLXReferenceOptimizers.rAdam(learningRate: 0.005, weightDecay: 0.00025) },
            referenceSchedule: { .constant },
            steps: steps,
            arrays: { step in
                let example = examples(step)
                let targets = example.targets
                return [example.spectrograms, targets.beat, targets.downbeat, targets.section, targets.function]
            },
            loss: { net, arrays in
                let targets = NFKMLXAllInOneTargets(beat: arrays[1], downbeat: arrays[2], section: arrays[3],
                                                    function: arrays[4])
                return objective.loss(net(arrays[0]), targets: targets)
            },
            clipGradientNorm: clipGradientNorm,
            learningRateSchedule: learningRateSchedule,
            checkpoint: checkpoint, observer: observer)
    }
}

extension NFKMLXAllInOneTargets {
    /// Reassembles targets from their arrays, inside a training step.
    init(beat: MLXArray, downbeat: MLXArray, section: MLXArray, function: MLXArray) {
        self.beat = beat
        self.downbeat = downbeat
        self.section = section
        self.function = function
    }
}
