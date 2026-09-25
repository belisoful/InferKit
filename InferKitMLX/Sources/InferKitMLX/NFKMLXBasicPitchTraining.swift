//
//  NFKMLXBasicPitchTraining.swift
//  InferKitMLX
//
//  Fine-tuning Basic Pitch on a consumer's own recordings and their notes: an instrument, a tuning, or a
//  recording chain the released model transcribes poorly. At 16,864 trainable parameters every weight
//  trains on a device, so the level is a full fine-tune.
//
//  Everything here follows the reference's own training code in the `basic-pitch` 0.4.0 distribution.
//  `models.loss()` is three binary cross-entropies, one per posteriorgram, each with label smoothing
//  0.2 and summed with equal weights. `train.py` compiles it under Keras's `Adam` at 1e-3 with no
//  gradient clip. `models.model()` puts a `UnitNorm` constraint on every convolution kernel, which Keras
//  applies after each update, and trains three batch normalizations that the released ONNX graph folds
//  away, so the recipe trains the network's `.separate` layout.
//
//  `train.py` halves the rate when the validation loss stalls for ten epochs of a hundred steps
//  (`ReduceLROnPlateau`). That needs a validation set a device fine-tune usually lacks, and a run of a
//  few hundred steps never reaches it, so the reference schedule here is constant; a caller with a
//  validation set passes their own.
//

import Foundation
import InferKit
import MLX
import MLXNN
import MLXOptimizers

/*!
 @abstract One batch of Basic Pitch training windows and their targets.
 @discussion `audio` is `[B, 43844, 1]` at 22,050 Hz, two seconds less one hop, as the network reads a
 window. The targets are binary `[B, 172, bins]` at the reference's annotation rate of 86 frames a
 second: `contour` over 264 bins, three per semitone, and `note` and `onset` over the 88 piano keys.
 Introduced in InferKit 0.5.0.
 */
public struct NFKMLXBasicPitchExample {
    public var audio: MLXArray
    public var contour: MLXArray
    public var note: MLXArray
    public var onset: MLXArray

    public init(audio: MLXArray, contour: MLXArray, note: MLXArray, onset: MLXArray) {
        self.audio = audio
        self.contour = contour
        self.note = note
        self.onset = onset
    }

    /// Several examples stacked into one batch along the first axis.
    public static func batch(_ examples: [NFKMLXBasicPitchExample]) -> NFKMLXBasicPitchExample {
        NFKMLXBasicPitchExample(audio: concatenated(examples.map(\.audio), axis: 0),
                                contour: concatenated(examples.map(\.contour), axis: 0),
                                note: concatenated(examples.map(\.note), axis: 0),
                                onset: concatenated(examples.map(\.onset), axis: 0))
    }
}

/*!
 @abstract The objective a Basic Pitch fine-tune minimizes: the reference's `models.loss()`.
 @discussion Each posteriorgram is scored by Keras's `binary_crossentropy` on probabilities: the target
 is smoothed toward 0.5 by `labelSmoothing`, the prediction is clipped to `[1e-7, 1 − 1e-7]`, the
 cross-entropy adds 1e-7 inside each logarithm, and the mean runs over every element. The three terms
 add with equal weights. With `weightedOnset` the onset term is `models.weighted_transcription_loss`: the
 cross-entropy over the target's zeros and over its other entries, each averaged on its own, mixed by
 `positiveOnsetWeight`. Measured against the reference by `run_reference.py basic_pitch_training`.
 Introduced in InferKit 0.5.0.
 */
public struct NFKMLXBasicPitchObjective: Sendable {

    /// Squeezes every target toward 0.5. The reference's is 0.2.
    public var labelSmoothing: Float

    /// Balances the onset term between the target's zeros and its onsets. The reference's default is
    /// off.
    public var weightedOnset: Bool

    /// The share the onsets take of the weighted onset term. The reference's is 0.5.
    public var positiveOnsetWeight: Float

    public init(labelSmoothing: Float = 0.2, weightedOnset: Bool = false, positiveOnsetWeight: Float = 0.5) {
        self.labelSmoothing = labelSmoothing
        self.weightedOnset = weightedOnset
        self.positiveOnsetWeight = positiveOnsetWeight
    }

    /// Keras's `binary_crossentropy` on probabilities, element by element.
    func crossEntropy(target: MLXArray, predicted: MLXArray) -> MLXArray {
        let epsilon: Float = 1e-7
        let smoothed = target * (1 - labelSmoothing) + 0.5 * labelSmoothing
        let clipped = clip(predicted, min: epsilon, max: 1 - epsilon)
        return -(smoothed * log(clipped + epsilon) + (1 - smoothed) * log(1 - clipped + epsilon))
    }

    /// The onset term: the plain mean, or with `weightedOnset` the reference's class-balanced mix.
    ///
    /// The reference's weighted term is a mean over an empty set, and so not a number, for a batch
    /// holding no onset at all. Here that half contributes zero instead.
    func onsetTerm(target: MLXArray, predicted: MLXArray) -> MLXArray {
        let entropy = crossEntropy(target: target, predicted: predicted)
        guard weightedOnset else {
            return entropy.mean()
        }
        let negative = (target .== Float(0)).asType(.float32)
        let positive = 1 - negative
        let negativeMean = (entropy * negative).sum() / maximum(negative.sum(), 1)
        let positiveMean = (entropy * positive).sum() / maximum(positive.sum(), 1)
        return (1 - positiveOnsetWeight) * negativeMean + positiveOnsetWeight * positiveMean
    }

    /// The three terms before they add, each a scalar, so a parity run can say which one disagrees.
    public func components(predicted: (contour: MLXArray, note: MLXArray, onset: MLXArray),
                           target: (contour: MLXArray, note: MLXArray, onset: MLXArray))
        -> (contour: MLXArray, note: MLXArray, onset: MLXArray) {
        (crossEntropy(target: target.contour, predicted: predicted.contour).mean(),
         crossEntropy(target: target.note, predicted: predicted.note).mean(),
         onsetTerm(target: target.onset, predicted: predicted.onset))
    }

    /// The summed loss, a scalar.
    public func loss(predicted: (contour: MLXArray, note: MLXArray, onset: MLXArray),
                     target: (contour: MLXArray, note: MLXArray, onset: MLXArray)) -> MLXArray {
        let terms = components(predicted: predicted, target: target)
        return terms.contour + terms.note + terms.onset
    }

    /// Scores `net` on one batch. The network runs in whatever mode it is in: a fine-tune puts it in
    /// training mode, where each batch normalization reads the batch's own statistics.
    public func callAsFunction(_ net: NFKMLXBasicPitchNet, _ example: NFKMLXBasicPitchExample) -> MLXArray {
        loss(predicted: net.posteriorgrams(example.audio),
             target: (example.contour, example.note, example.onset))
    }
}

extension NFKMLXBasicPitchNet {

    /// The six convolutions the reference constrains, by the key each weight saves under.
    var constrainedConvolutions: [(String, Conv2d)] {
        [("contour_conv", contourConv), ("contour_out", contourOut), ("note_conv", noteConv),
         ("note_out", noteOut), ("onset_conv", onsetConv), ("onset_out", onsetOut)]
    }

    /// Projects every trainable convolution kernel back to unit norm, Keras's `UnitNorm(axis=[0, 1, 2])`
    /// on each output filter: `w / (1e-7 + ‖w‖)`. Keras applies it after every update to the trainable
    /// variables that carry it, so a frozen kernel is left as it is.
    func unitNormalizeKernels() {
        let trainable = Set(trainableParameters().flattened().map(\.0))
        let projected = constrainedConvolutions.compactMap { name, convolution -> (String, MLXArray)? in
            let key = "\(name).weight"
            guard trainable.contains(key) else { return nil }
            let norm = sqrt(convolution.weight.square().sum(axes: [1, 2, 3], keepDims: true))
            return (key, convolution.weight / (norm + 1e-7))
        }
        update(parameters: ModuleParameters.unflattened(projected))
    }

    /// Resets the convolutions and batch normalizations to the reference's initialization, leaving the
    /// constant-Q front end as it is. `models.model()` initializes every kernel with
    /// `VarianceScaling(scale: 2, mode: "fan_avg", distribution: "uniform")`, a uniform draw within
    /// `±√(6 / fan_avg)` where `fan_avg` averages the kernel's input and output fans, and every bias at
    /// zero; a Keras batch normalization starts as the identity.
    func resetToReferenceInitialization() {
        var reset = [(String, MLXArray)]()
        for (name, convolution) in constrainedConvolutions {
            let shape = convolution.weight.shape                                  // [out, kH, kW, in]
            let receptive = Float(shape[1] * shape[2])
            let fanAverage = (receptive * Float(shape[3]) + receptive * Float(shape[0])) / 2
            let limit = (6 / fanAverage).squareRoot()
            reset.append(("\(name).weight", MLXRandom.uniform(-limit ..< limit, shape)))
            reset.append(("\(name).bias", MLXArray.zeros([shape[0]])))
        }
        for (name, normalization) in [("log_norm", logNorm), ("contour_norm", contourNorm), ("onset_norm", onsetNorm)] {
            guard let normalization else { continue }
            let channels = normalization.weight.dim(0)
            reset.append(("\(name).weight", MLXArray.ones([channels])))
            reset.append(("\(name).bias", MLXArray.zeros([channels])))
            reset.append(("\(name).running_mean", MLXArray.zeros([channels])))
            reset.append(("\(name).running_var", MLXArray.ones([channels])))
        }
        update(parameters: ModuleParameters.unflattened(reset))
    }
}

/// Builds Basic Pitch training windows from a recording and the notes played in it, the way the
/// reference's data pipeline builds them from a MIDI-annotated track such as MAESTRO.
///
/// The targets follow `mirdata`'s `NoteData.to_sparse_index` (mirdata 1.0) over the reference's grids,
/// and the windows follow `tf_example_deserialization.extract_window`. Measured against both by
/// `run_reference.py basic_pitch_targets`.
enum NFKBasicPitchTargets {

    /// The rate the reference annotates at, `AUDIO_SAMPLE_RATE // FFT_HOP`. It is 86, where the network
    /// produces 86.13 frames a second, and the reference trains with that misalignment.
    static let annotationRate = 86
    static let annotationHop = 1.0 / Double(annotationRate)
    static let windowFrames = 172

    /// The reference's frequency grid: `base · d^k` for `bins` values, `d = 2^(1/(12 · perSemitone))`.
    static func logBins(perSemitone: Int, bins: Int) -> [Double] {
        let ratio = Foundation.pow(2.0, 1.0 / Double(12 * perSemitone))
        return (0 ..< bins).map { Foundation.log(27.5 * Foundation.pow(ratio, Double($0))) }
    }

    static let noteBins = logBins(perSemitone: 1, bins: 88)
    static let contourBins = logBins(perSemitone: 3, bins: 264)

    /// `closest_index`: the nearest value's index, the first on a tie, or -1 outside the range.
    static func closest(_ value: Double, in scale: [Double]) -> Int {
        guard let low = scale.first, let high = scale.last, value >= low, value <= high else {
            return -1
        }
        var best = 0
        for index in 1 ..< scale.count where abs(value - scale[index]) < abs(value - scale[best]) {
            best = index
        }
        return best
    }

    /// The frame nearest `seconds` on `arange(0, duration + hop, hop)`, the first on a tie, or -1
    /// outside it.
    static func frame(_ seconds: Double, frames: Int) -> Int {
        let last = Double(frames - 1) * annotationHop
        guard seconds >= 0, seconds <= last else {
            return -1
        }
        let guess = Int((seconds / annotationHop).rounded(.down))
        var best = max(guess - 1, 0)
        for candidate in best ... min(guess + 1, frames - 1)
            where abs(seconds - Double(candidate) * annotationHop) < abs(seconds - Double(best) * annotationHop) {
            best = candidate
        }
        return best
    }

    /// The three targets over a whole recording, row-major `[frames, bins]`.
    ///
    /// A note's start and end each snap to the nearest frame, and it fills every frame from the one to
    /// the other inclusive, or to the last frame when it ends past the recording. Its pitch snaps to
    /// the nearest bin in log frequency, and a pitch outside the grid drops the note, so the piano's 88
    /// keys are what trains. Its value is its velocity over 127 rounded up: 1 for any sounding note,
    /// 0 for velocity 0. Where notes overlap on one pitch the later note's value stands, as the
    /// reference's dense conversion leaves it. The contour target is the notes on the contour grid,
    /// as the reference builds it for MIDI-annotated data, so a note lights bin `3 · (pitch − 21)`.
    static func track(notes: [NFKMIDINote], durationSeconds: Double)
        -> (frames: Int, note: [Float], onset: [Float], contour: [Float]) {
        let frames = Int(((durationSeconds + annotationHop) / annotationHop).rounded(.up))
        var note = [Float](repeating: 0, count: frames * 88)
        var onset = [Float](repeating: 0, count: frames * 88)
        var contour = [Float](repeating: 0, count: frames * 264)

        func fill(_ target: inout [Float], bins: Int, frequency: Int, from start: Int, to end: Int, value: Float) {
            guard frequency != -1, !(start == -1 && end == -1) else { return }
            let first = max(start, 0)
            let stop = (end != -1 ? end : frames - 1) + 1
            guard stop > first else { return }
            for index in first ..< stop {
                target[index * bins + frequency] = value
            }
        }

        for midi in notes {
            let pitch = Foundation.log(440.0 * Foundation.pow(2.0, (Double(midi.pitch) - 69.0) / 12.0))
            let value = Float((Double(midi.velocity) / 127.0).rounded(.up))
            let start = frame(midi.startSeconds, frames: frames)
            let end = frame(midi.endSeconds, frames: frames)
            let key = closest(pitch, in: noteBins)
            fill(&note, bins: 88, frequency: key, from: start, to: end, value: value)
            fill(&contour, bins: 264, frequency: closest(pitch, in: contourBins), from: start, to: end, value: value)
            if start != -1, key != -1 {
                onset[start * 88 + key] = value
            }
        }
        return (frames, note, onset, contour)
    }

    /// The first sample and first frame of a window starting at `startSeconds`, rounded as
    /// `trim_time` rounds them: a single-precision product, half to even.
    static func windowStart(_ startSeconds: Double, rate: Int) -> Int {
        Int((Float(rate) * Float(startSeconds)).rounded(.toNearestOrEven))
    }
}

extension NFKMLXBasicPitch {

    /// One training window of a recording and its notes: the audio from `startSeconds` for the
    /// network's 43,844 samples, and the three targets from the same moment for 172 frames at the
    /// reference's 86 frames a second.
    ///
    /// - Parameters:
    ///   - samples: the whole recording, mono.
    ///   - sampleRate: its rate. The reference prepares every recording at 22,050 Hz with sox; a
    ///     recording at another rate is resampled here with `NFKMLXAudioRate`, which is not sox's
    ///     resampler.
    ///   - notes: what the recording plays, as `NFKMIDINote`s with times in seconds from its start.
    ///     Pitch bends are not read; the reference builds a MIDI track's contour target from the notes
    ///     alone.
    ///   - startSeconds: where the window starts. It must leave a whole window before the end.
    ///
    /// - Throws: `NFKMLXError.trainingDataMismatch` when the window runs past the recording.
    ///
    /// - Since: InferKit 0.5.0
    public static func trainingExample(samples: [Float], sampleRate: Int, notes: NFKMIDISequence,
                                       startSeconds: Double) throws -> NFKMLXBasicPitchExample {
        try prepared(samples: samples, sampleRate: sampleRate, notes: notes).window(at: startSeconds)
    }

    /// Training windows drawn the way the reference samples them: each start uniform over the
    /// recording, and a window whose notes and contours are all silent drawn again, as
    /// `is_not_all_silent_annotations` drops one. The draw comes from `seed`, so a run is repeatable;
    /// it is not TensorFlow's random stream.
    ///
    /// - Returns: up to `count` windows; fewer when the recording holds too little that sounds.
    ///
    /// - Since: InferKit 0.5.0
    public static func trainingExamples(samples: [Float], sampleRate: Int, notes: NFKMIDISequence,
                                        count: Int, seed: UInt64 = 0) throws -> [NFKMLXBasicPitchExample] {
        let configuration = NFKMLXBasicPitchConfiguration.icassp2022
        let recording = prepared(samples: samples, sampleRate: sampleRate, notes: notes)
        let seconds = Double(recording.audio.count) / Double(configuration.sampleRate)
        let window = Double(configuration.windowSamples) / Double(configuration.sampleRate)
        // One frame of headroom keeps the target slice inside the track after rounding.
        let latest = seconds - window - NFKBasicPitchTargets.annotationHop
        guard latest >= 0 else {
            throw NFKMLXError.trainingDataMismatch(
                "the recording is \(seconds) s long; a Basic Pitch window needs \(window) s")
        }
        var state = seed
        func uniform() -> Double {
            state = state &+ 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return Double((z ^ (z >> 31)) >> 11) / Double(1 << 53)
        }
        var examples = [NFKMLXBasicPitchExample]()
        var attempts = 0
        while examples.count < count, attempts < count * 20 {
            attempts += 1
            let example = try recording.window(at: uniform() * latest)
            if example.note.max().item(Float.self) > 0 || example.contour.max().item(Float.self) > 0 {
                examples.append(example)
            }
        }
        return examples
    }

    /// A recording at the network's rate beside its whole-track targets, ready to cut windows from.
    struct PreparedRecording {
        let audio: [Float]
        let track: (frames: Int, note: [Float], onset: [Float], contour: [Float])

        /// The window from `startSeconds`: `extract_window`'s slice of the audio and of each target.
        func window(at startSeconds: Double) throws -> NFKMLXBasicPitchExample {
            let configuration = NFKMLXBasicPitchConfiguration.icassp2022
            let sample = NFKBasicPitchTargets.windowStart(startSeconds, rate: configuration.sampleRate)
            let frame = NFKBasicPitchTargets.windowStart(startSeconds, rate: NFKBasicPitchTargets.annotationRate)
            let frames = NFKBasicPitchTargets.windowFrames
            guard startSeconds >= 0, sample + configuration.windowSamples <= audio.count,
                  frame + frames <= track.frames else {
                let length = Double(audio.count) / Double(configuration.sampleRate)
                throw NFKMLXError.trainingDataMismatch(
                    "a window from \(startSeconds) s runs past the recording's \(length) s; a Basic Pitch "
                    + "window is \(Double(configuration.windowSamples) / Double(configuration.sampleRate)) s long")
            }
            func rows(_ target: [Float], bins: Int) -> MLXArray {
                MLXArray(Array(target[(frame * bins) ..< ((frame + frames) * bins)]), [1, frames, bins])
            }
            let window = Array(audio[sample ..< (sample + configuration.windowSamples)])
            return NFKMLXBasicPitchExample(audio: MLXArray(window, [1, configuration.windowSamples, 1]),
                                           contour: rows(track.contour, bins: 264),
                                           note: rows(track.note, bins: 88),
                                           onset: rows(track.onset, bins: 88))
        }
    }

    static func prepared(samples: [Float], sampleRate: Int, notes: NFKMIDISequence) -> PreparedRecording {
        let rate = NFKMLXBasicPitchConfiguration.icassp2022.sampleRate
        let audio = NFKMLXAudioRate.matched(samples, from: sampleRate, to: rate)
        return PreparedRecording(audio: audio,
                                 track: NFKBasicPitchTargets.track(notes: notes.notes,
                                                                   durationSeconds: Double(audio.count) / Double(rate)))
    }

    /// The constant-Q front end's tensors, which every checkpoint carries and no run trains.
    static let frontEndKeys = ["cqt.kernel_a", "cqt.kernel_b", "cqt.kernel_bias", "cqt.lowpass",
                               "cqt.lowpass_bias", "cqt.scale"]

    /// Builds the trainable network from a checkpoint, in the layout the checkpoint holds.
    ///
    /// - Parameters:
    ///   - weightsURL: a converted checkpoint or a file `NFKMLXWeights.save` wrote. The trainable layout
    ///     comes from `Tools/basic-pitch-to-safetensors --saved-model`; the plain converter writes the
    ///     folded layout, which runs inference only.
    ///   - reinitializing: starts the convolutions and normalizations from the reference's
    ///     initialization in the trainable layout, which is training from scratch. The constant-Q
    ///     front end is a fixed transform with no initialization of its own, so it still comes from
    ///     the checkpoint, and either layout supplies it.
    ///
    /// - Since: InferKit 0.5.0
    public static func network(weightsURL: URL, reinitializing: Bool = false) throws -> NFKMLXBasicPitchNet {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: weightsURL)
        guard reinitializing else {
            let net = makeNet(NFKMLXBasicPitchConfiguration(normalization: normalization(of: checkpoint)))
            try apply(checkpoint, to: net)
            return net
        }
        let missing = frontEndKeys.filter { checkpoint.arrays[$0] == nil }
        guard missing.isEmpty else {
            throw NFKMLXError.weightsMismatch(
                "the checkpoint has no constant-Q front end (\(missing.joined(separator: ", "))), "
                + "which a network trained from scratch still reads")
        }
        let net = makeNet(.icassp2022Trainable)
        try NFKMLXWeights.apply(mapped(checkpoint).filter { frontEndKeys.contains($0.0) }, to: net, strict: false)
        net.resetToReferenceInitialization()
        return net
    }

    /// Fine-tunes every trained weight of `net` on a consumer's own windows, returning the loss from
    /// each step.
    ///
    /// The whole path is three calls: ``network(weightsURL:reinitializing:)`` to build, this to train,
    /// and `NFKMLXWeights.save` to write a checkpoint that `backendWithWeightsURL:error:` loads.
    ///
    /// - Parameters:
    ///   - net: the network, in the `.separate` layout.
    ///   - examples: supplies one batch per step. `train.py` batches 16 windows.
    ///   - objective: the reference's `models.loss()`.
    ///   - optimizer: the update rule. Nil uses Keras's `Adam` at 1e-3, `train.py`'s.
    ///   - steps: how many batches to train on.
    ///   - clipGradientNorm: bounds the global gradient norm. The reference sets none.
    ///   - learningRateSchedule: multiplies the rate at each step. Nil holds the reference's rate
    ///     constant.
    ///   - checkpoint: writes the network periodically, so a suspended run keeps its progress.
    ///   - observer: receives each step and can end the run early.
    ///
    /// The constant-Q front end stays frozen, as it is a constant in the reference. Every other
    /// parameter trains, the batch normalizations read each batch's own statistics and update their
    /// moving ones, and after every update each convolution kernel is projected back to unit norm,
    /// whichever optimizer runs. A run is seconds to minutes; call it off the render thread.
    ///
    /// - Throws: `NFKMLXError.unsupportedConfiguration` for a network in the `.folded` layout, which
    ///   cannot train as the reference does.
    ///
    /// - Since: InferKit 0.5.0
    @discardableResult
    public static func fineTune(
        _ net: NFKMLXBasicPitchNet,
        examples: (Int) -> NFKMLXBasicPitchExample,
        objective: NFKMLXBasicPitchObjective = NFKMLXBasicPitchObjective(),
        optimizer: Optimizer? = nil,
        steps: Int,
        clipGradientNorm: Float? = nil,
        learningRateSchedule: NFKMLXLearningRateSchedule? = nil,
        checkpoint: NFKMLXTrainingCheckpoint? = nil,
        observer: NFKMLXTrainer.Observer? = nil
    ) throws -> [Float] {
        guard net.configuration.normalization == .separate else {
            throw NFKMLXError.unsupportedConfiguration(
                "this Basic Pitch network has its batch normalizations folded into its convolutions, as "
                + "the released ONNX graph ships them, and cannot train as the reference does. Convert "
                + "the release with `Tools/basic-pitch-to-safetensors --saved-model`, or build with "
                + "`network(weightsURL:reinitializing: true)` to train from scratch.")
        }
        return try NFKMLXFineTune.run(
            net,
            freezing: {
                net.unfreeze()
                net.cqt.freeze()
            },
            optimizer: optimizer,
            reference: { NFKMLXKerasAdam(learningRate: 1e-3) },
            referenceSchedule: { .constant },
            steps: steps,
            arrays: { step in
                let example = examples(step)
                return [example.audio, example.contour, example.note, example.onset]
            },
            loss: { net, arrays in
                objective(net, NFKMLXBasicPitchExample(audio: arrays[0], contour: arrays[1],
                                                       note: arrays[2], onset: arrays[3]))
            },
            clipGradientNorm: clipGradientNorm,
            learningRateSchedule: learningRateSchedule,
            checkpoint: checkpoint,
            constraint: { $0.unitNormalizeKernels() },
            observer: observer)
    }
}
