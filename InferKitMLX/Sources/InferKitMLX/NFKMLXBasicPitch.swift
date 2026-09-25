//
//  NFKMLXBasicPitch.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXNN

// Music transcription turns a recording into notes. This is Basic Pitch (Spotify, ICASSP 2022): a
// constant-Q front end, harmonic stacking, and three small convolutional heads that score, per frame,
// a continuous pitch contour (264 bins, three per semitone), a note activation (88 bins, one per piano
// key), and a note onset (88 bins). Note creation reads the three posteriorgrams and returns notes with
// a pitch-bend curve each, which the backend hands back as an `NFKMIDISequence` under `NFKOutputMIDI`.
//
// The model is instrument-agnostic and tiny — 35,736 parameters, the smallest network in the package.
//
// The released artifact is an ONNX graph holding the whole pipeline, the constant-Q transform included,
// so the CQT's complex kernels, its anti-aliasing filter, and its per-bin scale arrive as checkpoint
// tensors rather than as constants this port derives. `Tools/basic-pitch-to-safetensors` lifts them out.
//
// The CQT is nnAudio's `CQT2010v2`: rather than one transform at a huge window, it computes one octave
// at a time with the same 36 kernels, halving the signal rate between octaves (an anti-aliasing filter
// at stride 2) and halving the hop with it, so every octave lands on the same 172-frame grid. Nine
// octaves stack to 324 bins and the lowest 15, which sit below the requested 27.5 Hz, are dropped.
//
// The released graph folds the Keras batch normalizations into the convolutions they follow; the one
// that survives is the single-channel normalization after the log, kept as a scale and a bias. That is
// the `.folded` layout, and inference needs nothing more. Training needs the normalizations separate,
// because in training mode each one normalizes by its batch's own statistics, which a folded
// convolution cannot express. The `.separate` layout keeps them, as the reference's Keras model does.

/// Where Basic Pitch's batch normalizations live.
///
/// Introduced in InferKit 0.5.0.
public enum NFKMLXBasicPitchNormalization: Sendable {
    /// Folded into the convolutions they follow, with the one after the log kept as a scale and a bias.
    /// This is the released ONNX graph's layout, and it runs inference only.
    case folded
    /// Three batch normalizations separate from the convolutions, as the reference trains them: after
    /// the log, after the contour convolution, and after the onset convolution. A fine-tune needs this
    /// layout.
    case separate
}

/// Basic Pitch's geometry. The defaults are the released `icassp_2022` model.
public struct NFKMLXBasicPitchConfiguration: Sendable {
    /// The rate the model reads. A clip at another rate is resampled to it.
    public var sampleRate: Int
    /// The samples between frames. 256 at 22050 Hz gives 86.13 frames per second.
    public var hopLength: Int
    /// The samples one window feeds the network: two seconds less one hop.
    public var windowSamples: Int
    /// The frames one window produces.
    public var frames: Int
    /// The CQT bins the transform keeps, three per semitone.
    public var cqtBins: Int
    /// The CQT kernel length, which is also the anti-aliasing filter's.
    public var kernelLength: Int
    /// The octaves the transform walks, highest first.
    public var octaves: Int
    /// The harmonics harmonic stacking shifts the CQT by, the subharmonic first.
    public var harmonics: [Double]
    /// The contour bins the network scores, three per semitone over 88 semitones.
    public var contourBins: Int
    /// The note and onset bins the network scores, one per piano key.
    public var noteBins: Int
    /// The MIDI pitch of the lowest note bin. 21 is A0, the lowest key on a piano.
    public var lowestPitch: Int
    /// The frequency of the lowest CQT bin.
    public var baseFrequency: Double
    /// The frames two neighboring windows share, half of them dropped from each side on the seam.
    public var overlappingFrames: Int
    /// Where the batch normalizations live. The released ONNX graph is `.folded`; training needs
    /// `.separate`.
    public var normalization: NFKMLXBasicPitchNormalization

    public init(sampleRate: Int = 22050, hopLength: Int = 256, windowSamples: Int = 43844,
                frames: Int = 172, cqtBins: Int = 309, kernelLength: Int = 256, octaves: Int = 9,
                harmonics: [Double] = [0.5, 1, 2, 3, 4, 5, 6, 7], contourBins: Int = 264,
                noteBins: Int = 88, lowestPitch: Int = 21, baseFrequency: Double = 27.5,
                overlappingFrames: Int = 30, normalization: NFKMLXBasicPitchNormalization = .folded) {
        self.sampleRate = sampleRate
        self.hopLength = hopLength
        self.windowSamples = windowSamples
        self.frames = frames
        self.cqtBins = cqtBins
        self.kernelLength = kernelLength
        self.octaves = octaves
        self.harmonics = harmonics
        self.contourBins = contourBins
        self.noteBins = noteBins
        self.lowestPitch = lowestPitch
        self.baseFrequency = baseFrequency
        self.overlappingFrames = overlappingFrames
        self.normalization = normalization
    }

    /// The released model.
    public static let icassp2022 = NFKMLXBasicPitchConfiguration()

    /// The released model with its batch normalizations separate, the layout a fine-tune trains.
    public static let icassp2022Trainable = NFKMLXBasicPitchConfiguration(normalization: .separate)

    /// The bins per semitone the contour carries.
    var binsPerSemitone: Int { contourBins / noteBins }
    /// The samples a window advances, the overlap withheld.
    var windowHop: Int { windowSamples - overlappingFrames * hopLength }
    /// The seconds one frame spans.
    var frameSeconds: Double { Double(hopLength) / Double(sampleRate) }
    /// The bin harmonic stacking shifts by for a harmonic: `round(3 · 12 · log2(harmonic))`.
    func shift(forHarmonic harmonic: Double) -> Int {
        Int((Double(binsPerSemitone) * 12.0 * log2(harmonic)).rounded())
    }
}

/// The constant-Q transform, octave by octave, from the released kernels.
///
/// `kernelA` and `kernelB` are the transform's two quadrature halves. The reference negates one of
/// them, and the magnitude squares both, so which is the real part does not reach any output.
final class NFKBasicPitchCQT: Module {
    @ParameterInfo(key: "kernel_a") var kernelA: MLXArray            // [36, kernelLength, 1]
    @ParameterInfo(key: "kernel_b") var kernelB: MLXArray
    @ParameterInfo(key: "kernel_bias") var kernelBias: MLXArray      // [36]
    @ParameterInfo(key: "lowpass") var lowpass: MLXArray             // [1, kernelLength, 1]
    @ParameterInfo(key: "lowpass_bias") var lowpassBias: MLXArray    // [1]
    @ParameterInfo(key: "scale") var scale: MLXArray                 // [cqtBins]

    let configuration: NFKMLXBasicPitchConfiguration
    /// The bins one octave carries.
    let binsPerOctave: Int

    init(_ configuration: NFKMLXBasicPitchConfiguration) {
        self.configuration = configuration
        binsPerOctave = configuration.binsPerSemitone * 12
        let length = configuration.kernelLength
        _kernelA.wrappedValue = MLXArray.zeros([binsPerOctave, length, 1])
        _kernelB.wrappedValue = MLXArray.zeros([binsPerOctave, length, 1])
        _kernelBias.wrappedValue = MLXArray.zeros([binsPerOctave])
        _lowpass.wrappedValue = MLXArray.zeros([1, length, 1])
        _lowpassBias.wrappedValue = MLXArray.zeros([1])
        _scale.wrappedValue = MLXArray.ones([configuration.cqtBins])
    }

    /// Reflect-pads `[B, L, 1]` by `pad` on each side without repeating the edge sample, which is what
    /// the reference's `reflection_pad1d` does.
    private func reflectPad(_ signal: MLXArray, _ pad: Int) -> MLXArray {
        let length = signal.dim(1)
        var indices = [Int32]()
        indices.reserveCapacity(length + 2 * pad)
        for i in stride(from: pad, through: 1, by: -1) { indices.append(Int32(i)) }
        indices.append(contentsOf: (0 ..< length).map { Int32($0) })
        for i in stride(from: length - 2, through: length - 1 - pad, by: -1) { indices.append(Int32(i)) }
        return take(signal, MLXArray(indices), axis: 1)
    }

    /// `[B, samples, 1]` → the CQT magnitude `[B, frames, cqtBins]`, the scale applied.
    func callAsFunction(_ audio: MLXArray) -> MLXArray {
        let pad = configuration.kernelLength / 2
        var signal = audio
        var hop = configuration.hopLength
        var octaveA = [MLXArray]()
        var octaveB = [MLXArray]()

        for octave in 0 ..< configuration.octaves {
            let padded = reflectPad(signal, pad)
            octaveA.append(conv1d(padded, kernelA, stride: hop) + kernelBias)
            octaveB.append(conv1d(padded, kernelB, stride: hop) + kernelBias)
            guard octave + 1 < configuration.octaves else { break }
            // The next octave reads the signal at half the rate, so its hop halves with it. The
            // anti-aliasing filter pads with zeros rather than by reflection.
            let filterPad = MLX.padded(signal, widths: [IntOrPair((0, 0)), IntOrPair((pad - 1, pad - 1)), IntOrPair((0, 0))],
                                       mode: .constant)
            signal = conv1d(filterPad, lowpass, stride: 2) + lowpassBias
            hop /= 2
        }

        // Every octave lands on the same frame grid; a window whose length is not the released one can
        // leave the highest octave one frame short, so the stack is cropped to what they all reach.
        let frames = min(octaveA.map { $0.dim(1) }.min() ?? 0, octaveB.map { $0.dim(1) }.min() ?? 0)
        // The lowest octave comes first, and the bins below the requested base frequency fall off the
        // bottom of the stack.
        let lowestFirst = { (parts: [MLXArray]) -> MLXArray in
            let cropped = parts.reversed().map { $0[0..., 0 ..< frames, 0...] }
            let stacked = concatenated(Array(cropped), axis: 2)
            return stacked[0..., 0..., (stacked.dim(2) - self.configuration.cqtBins)...]
        }
        let scaled = scale.reshaped([1, 1, configuration.cqtBins])
        let a = lowestFirst(octaveA) * scaled
        let b = lowestFirst(octaveB) * scaled
        return sqrt(a.square() + b.square())
    }
}

/// A Keras `BatchNormalization` over the last axis, the reference's normalization.
///
/// Two conventions set it apart from MLXNN's `BatchNorm`. The moving averages keep 0.99 of their value
/// each step. The moving variance folds in the unbiased batch variance, as TensorFlow's fused kernel
/// does, where MLXNN's folds in the biased one. Normalization itself divides by the biased variance in
/// both. The moving statistics never take a gradient.
final class NFKBasicPitchBatchNorm: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray
    @ParameterInfo(key: "running_mean") var runningMean: MLXArray
    @ParameterInfo(key: "running_var") var runningVar: MLXArray

    static let epsilon: Float = 1e-3
    static let momentum: Float = 0.99

    init(channels: Int) {
        _weight.wrappedValue = MLXArray.ones([channels])
        _bias.wrappedValue = MLXArray.zeros([channels])
        _runningMean.wrappedValue = MLXArray.zeros([channels])
        _runningVar.wrappedValue = MLXArray.ones([channels])
        super.init()
    }

    /// The moving statistics never take a gradient, however the tree around them was unfrozen. A
    /// parent's recursive `unfreeze()` clears every child's frozen set without calling the child, so
    /// overriding `unfreeze` would not hold; `noGrad()` is what the trainable-parameter filter reads.
    override func noGrad() -> Set<String> {
        super.noGrad().union(["running_mean", "running_var"])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        guard training else {
            return (x - runningMean) * rsqrt(runningVar + Self.epsilon) * weight + bias
        }
        let axes = Array(0 ..< (x.ndim - 1))
        let mean = x.mean(axes: axes)
        let variance = x.variance(axes: axes)
        let count = Float(x.size / x.dim(-1))
        let unbiased = stopGradient(variance) * (count / max(count - 1, 1))
        runningMean._updateInternal(Self.momentum * runningMean + (1 - Self.momentum) * stopGradient(mean))
        runningVar._updateInternal(Self.momentum * runningVar + (1 - Self.momentum) * unbiased)
        return (x - mean) * rsqrt(variance + Self.epsilon) * weight + bias
    }
}

/// Basic Pitch's network: the CQT, the normalized log, harmonic stacking, and the contour, note, and
/// onset heads.
public final class NFKMLXBasicPitchNet: Module {
    @ModuleInfo(key: "cqt") var cqt: NFKBasicPitchCQT
    // The folded layout carries the normalization after the log as a scale and a bias; the separate
    // layout carries three batch normalizations instead. Each layout leaves the other's members nil.
    @ParameterInfo(key: "norm_scale") var normScale: MLXArray?
    @ParameterInfo(key: "norm_bias") var normBias: MLXArray?
    @ModuleInfo(key: "log_norm") var logNorm: NFKBasicPitchBatchNorm?
    @ModuleInfo(key: "contour_norm") var contourNorm: NFKBasicPitchBatchNorm?
    @ModuleInfo(key: "onset_norm") var onsetNorm: NFKBasicPitchBatchNorm?
    @ModuleInfo(key: "contour_conv") var contourConv: Conv2d
    @ModuleInfo(key: "contour_out") var contourOut: Conv2d
    @ModuleInfo(key: "note_conv") var noteConv: Conv2d
    @ModuleInfo(key: "note_out") var noteOut: Conv2d
    @ModuleInfo(key: "onset_conv") var onsetConv: Conv2d
    @ModuleInfo(key: "onset_out") var onsetOut: Conv2d

    public let configuration: NFKMLXBasicPitchConfiguration

    public init(_ configuration: NFKMLXBasicPitchConfiguration = .icassp2022) {
        self.configuration = configuration
        let harmonics = configuration.harmonics.count
        _cqt.wrappedValue = NFKBasicPitchCQT(configuration)
        switch configuration.normalization {
        case .folded:
            _normScale.wrappedValue = MLXArray.ones([1])
            _normBias.wrappedValue = MLXArray.zeros([1])
        case .separate:
            _logNorm.wrappedValue = NFKBasicPitchBatchNorm(channels: 1)
            _contourNorm.wrappedValue = NFKBasicPitchBatchNorm(channels: 8)
            _onsetNorm.wrappedValue = NFKBasicPitchBatchNorm(channels: 32)
        }
        // A convolution's first axis is time and its second is frequency; the strided pair reduces the
        // three contour bins per semitone to one note bin.
        _contourConv.wrappedValue = Conv2d(inputChannels: harmonics, outputChannels: 8,
                                           kernelSize: [3, 39], stride: 1, padding: [1, 19])
        _contourOut.wrappedValue = Conv2d(inputChannels: 8, outputChannels: 1,
                                          kernelSize: [5, 5], stride: 1, padding: [2, 2])
        _noteConv.wrappedValue = Conv2d(inputChannels: 1, outputChannels: 32,
                                        kernelSize: [7, 7], stride: [1, 3], padding: [3, 2])
        _noteOut.wrappedValue = Conv2d(inputChannels: 32, outputChannels: 1,
                                       kernelSize: [7, 3], stride: 1, padding: [3, 1])
        _onsetConv.wrappedValue = Conv2d(inputChannels: harmonics, outputChannels: 32,
                                         kernelSize: [5, 5], stride: [1, 3], padding: [2, 1])
        _onsetOut.wrappedValue = Conv2d(inputChannels: 33, outputChannels: 1,
                                        kernelSize: [3, 3], stride: 1, padding: [1, 1])
        super.init()
        // An MLX module starts in training mode, where the separate layout's normalizations would read
        // each window's own statistics. Inference reads the moving ones; `NFKMLXTrainer` switches a run
        // into training mode and back.
        train(false)
    }

    /// The CQT magnitude in decibels, normalized to 0...1 per window, then through the normalization
    /// after the log, folded or separate. `[B, frames, cqtBins]` → `[B, frames, cqtBins, 1]`.
    func normalizedLog(_ magnitude: MLXArray) -> MLXArray {
        let decibels = log(magnitude.square() + 1e-10) * Float(10.0 / log(10.0))
        let minimum = decibels.min(axes: [1, 2], keepDims: true)
        let shifted = decibels - minimum
        let maximum = shifted.max(axes: [1, 2], keepDims: true)
        // A silent window has no range to normalize by; the reference divides to zero rather than NaN.
        let normalized = MLX.where(maximum .== 0, MLXArray(Float(0)), shifted / maximum)
        if let logNorm {
            return logNorm(normalized.expandedDimensions(axis: 3))
        }
        return (normalized * (normScale ?? MLXArray(Float(1))) + (normBias ?? MLXArray(Float(0))))
            .expandedDimensions(axis: 3)
    }

    /// Stacks the spectrogram against itself shifted to each harmonic, so one bin sees a note's whole
    /// harmonic series. `[B, frames, cqtBins, 1]` → `[B, frames, contourBins, harmonics]`.
    func harmonicStack(_ spectrogram: MLXArray) -> MLXArray {
        let bins = spectrogram.dim(2)
        var stacked = [MLXArray]()
        for harmonic in configuration.harmonics {
            let shift = configuration.shift(forHarmonic: harmonic)
            if shift == 0 {
                stacked.append(spectrogram)
            } else if shift > 0 {
                // A harmonic above the fundamental reads higher bins, so the stack slides down and the
                // top fills with zeros.
                let sliced = spectrogram[0..., 0..., shift...]
                stacked.append(MLX.padded(sliced, widths: [IntOrPair((0, 0)), IntOrPair((0, 0)),
                                                           IntOrPair((0, shift)), IntOrPair((0, 0))], mode: .constant))
            } else {
                let sliced = spectrogram[0..., 0..., 0 ..< (bins + shift)]
                stacked.append(MLX.padded(sliced, widths: [IntOrPair((0, 0)), IntOrPair((0, 0)),
                                                           IntOrPair((-shift, 0)), IntOrPair((0, 0))], mode: .constant))
            }
        }
        return concatenated(stacked, axis: 3)[0..., 0..., 0 ..< configuration.contourBins, 0...]
    }

    /// The three posteriorgrams for one batch of windows `[B, samples, 1]`: the contour
    /// `[B, frames, 264]`, the note `[B, frames, 88]`, and the onset `[B, frames, 88]`.
    public func posteriorgrams(_ audio: MLXArray) -> (contour: MLXArray, note: MLXArray, onset: MLXArray) {
        let stacked = harmonicStack(normalizedLog(cqt(audio)))

        var contourHidden = contourConv(stacked)
        if let contourNorm {
            contourHidden = contourNorm(contourHidden)
        }
        contourHidden = relu(contourHidden)
        let contour = sigmoid(contourOut(contourHidden))                       // [B, frames, 264, 1]

        let notePre = sigmoid(noteOut(relu(noteConv(contour))))                // [B, frames, 88, 1]

        var onsetHidden = onsetConv(stacked)                                   // [B, frames, 88, 32]
        if let onsetNorm {
            onsetHidden = onsetNorm(onsetHidden)
        }
        onsetHidden = relu(onsetHidden)
        let onset = sigmoid(onsetOut(concatenated([notePre, onsetHidden], axis: 3)))

        return (contour.squeezed(axis: 3), notePre.squeezed(axis: 3), onset.squeezed(axis: 3))
    }
}

/// What note creation reads out of the posteriorgrams.
public struct NFKMLXBasicPitchOptions: Sendable {
    /// The onset activation a peak must reach to start a note.
    public var onsetThreshold: Float
    /// The note activation below which a sounding note ends.
    public var frameThreshold: Float
    /// The frames a note must last to be kept.
    public var minimumNoteFrames: Int
    /// Adds onsets where the note activation rises sharply, beside the ones the onset head scores.
    public var infersOnsets: Bool
    /// Reads a pitch-bend curve for each note out of the contour posteriorgram.
    public var includesPitchBends: Bool
    /// Picks up notes the onset head missed by walking outward from what energy the notes left behind.
    public var melodiaTrick: Bool
    /// The lowest frequency to transcribe, in hertz, or nil for no floor.
    public var minimumFrequency: Double?
    /// The highest frequency to transcribe, in hertz, or nil for no ceiling.
    public var maximumFrequency: Double?
    /// The tempo the MIDI file is written at.
    public var tempoBPM: Double
    /// The General MIDI program the notes are written on. 4 is Electric Piano 1, which is what the
    /// reference writes.
    public var program: Int

    public init(onsetThreshold: Float = 0.5, frameThreshold: Float = 0.3, minimumNoteFrames: Int = 11,
                infersOnsets: Bool = true, includesPitchBends: Bool = true, melodiaTrick: Bool = true,
                minimumFrequency: Double? = nil, maximumFrequency: Double? = nil,
                tempoBPM: Double = 120, program: Int = 4) {
        self.onsetThreshold = onsetThreshold
        self.frameThreshold = frameThreshold
        self.minimumNoteFrames = minimumNoteFrames
        self.infersOnsets = infersOnsets
        self.includesPitchBends = includesPitchBends
        self.melodiaTrick = melodiaTrick
        self.minimumFrequency = minimumFrequency
        self.maximumFrequency = maximumFrequency
        self.tempoBPM = tempoBPM
        self.program = program
    }

    public static let `default` = NFKMLXBasicPitchOptions()
}

/// One note as note creation finds it, in frames rather than seconds.
struct NFKBasicPitchNoteEvent {
    var startFrame: Int
    var endFrame: Int
    var pitch: Int
    var amplitude: Float
    var bendBins: [Int]?
}

/// Note creation: the three posteriorgrams in, notes out.
///
/// The reference's `note_creation.py` is sequential array work — peak-picking, tracking a note forward
/// until its energy dies, then reading a pitch-bend curve around its bin — so it ports as plain arrays
/// rather than as tensors.
enum NFKBasicPitchNoteCreation {

    /// Local maxima along time, strictly greater than both neighbors (`scipy.signal.argrelmax`).
    static func onsetPeaks(_ onsets: [[Float]]) -> [[Float]] {
        let frames = onsets.count
        guard frames > 2 else { return onsets.map { row in row.map { _ in Float(0) } } }
        var peaks = onsets.map { row in [Float](repeating: 0, count: row.count) }
        for frame in 1 ..< (frames - 1) {
            for bin in 0 ..< onsets[frame].count where onsets[frame][bin] > onsets[frame - 1][bin]
                && onsets[frame][bin] > onsets[frame + 1][bin] {
                peaks[frame][bin] = onsets[frame][bin]
            }
        }
        return peaks
    }

    /// Onsets inferred from a sharp rise in the note activation, rescaled to the onset head's own
    /// maximum and combined with it by taking the larger of the two.
    static func inferredOnsets(_ onsets: [[Float]], notes: [[Float]], differences: Int = 2) -> [[Float]] {
        let frames = notes.count
        guard frames > differences, let bins = notes.first?.count else { return onsets }

        var difference = [[Float]](repeating: [Float](repeating: 0, count: bins), count: frames)
        for frame in 0 ..< frames {
            for bin in 0 ..< bins {
                var smallest = Float.greatestFiniteMagnitude
                for n in 1 ... differences {
                    // The reference prepends `n` zero frames before differencing, so a frame closer to
                    // the start than `n` differences against zero.
                    let previous = frame - n >= 0 ? notes[frame - n][bin] : 0
                    smallest = min(smallest, notes[frame][bin] - previous)
                }
                difference[frame][bin] = max(smallest, 0)
            }
        }
        for frame in 0 ..< min(differences, frames) {
            for bin in 0 ..< bins { difference[frame][bin] = 0 }
        }

        let onsetMaximum = onsets.flatMap { $0 }.max() ?? 0
        let differenceMaximum = difference.flatMap { $0 }.max() ?? 0
        guard differenceMaximum > 0 else { return onsets }
        let rescale = onsetMaximum / differenceMaximum

        var combined = onsets
        for frame in 0 ..< frames {
            for bin in 0 ..< bins {
                combined[frame][bin] = max(onsets[frame][bin], difference[frame][bin] * rescale)
            }
        }
        return combined
    }

    /// The bins outside the requested frequency range, zeroed in place.
    static func constrain(_ matrix: inout [[Float]], lowestBin: Int, highestBin: Int) {
        for frame in 0 ..< matrix.count {
            for bin in 0 ..< matrix[frame].count where bin < lowestBin || bin >= highestBin {
                matrix[frame][bin] = 0
            }
        }
    }

    /// Decodes the note and onset posteriorgrams into note events.
    static func notes(note: [[Float]], onset: [[Float]], configuration: NFKMLXBasicPitchConfiguration,
                      options: NFKMLXBasicPitchOptions) -> [NFKBasicPitchNoteEvent] {
        let frameCount = note.count
        guard frameCount > 1, let bins = note.first?.count else { return [] }
        // A note may only be extinguished by this many consecutive frames below the threshold, which is
        // also how far the melodia pass walks past the end of a note before giving up.
        let energyTolerance = 11
        let highestBin = bins - 1

        var frames = note
        var onsets = onset
        let lowestAllowed = options.minimumFrequency.map { Int((hzToMIDI($0) - Double(configuration.lowestPitch)).rounded()) } ?? 0
        let highestAllowed = options.maximumFrequency.map { Int((hzToMIDI($0) - Double(configuration.lowestPitch)).rounded()) } ?? bins
        constrain(&frames, lowestBin: lowestAllowed, highestBin: highestAllowed)
        constrain(&onsets, lowestBin: lowestAllowed, highestBin: highestAllowed)

        if options.infersOnsets {
            onsets = inferredOnsets(onsets, notes: frames)
        }
        let peaks = onsetPeaks(onsets)

        var remaining = frames
        var events = [NFKBasicPitchNoteEvent]()

        // The reference walks its onsets backwards in time, so an earlier note claims the energy a
        // later one would otherwise take.
        var candidates = [(frame: Int, bin: Int)]()
        for frame in 0 ..< frameCount {
            for bin in 0 ..< bins where peaks[frame][bin] >= options.onsetThreshold {
                candidates.append((frame, bin))
            }
        }
        for candidate in candidates.reversed() {
            let start = candidate.frame
            let bin = candidate.bin
            if start >= frameCount - 1 { continue }

            var index = start + 1
            var quiet = 0
            while index < frameCount - 1 && quiet < energyTolerance {
                quiet = remaining[index][bin] < options.frameThreshold ? quiet + 1 : 0
                index += 1
            }
            index -= quiet
            if index - start <= options.minimumNoteFrames { continue }

            for frame in start ..< index {
                remaining[frame][bin] = 0
                if bin < highestBin { remaining[frame][bin + 1] = 0 }
                if bin > 0 { remaining[frame][bin - 1] = 0 }
            }
            events.append(NFKBasicPitchNoteEvent(startFrame: start, endFrame: index,
                                                 pitch: bin + configuration.lowestPitch,
                                                 amplitude: mean(frames, from: start, to: index, bin: bin),
                                                 bendBins: nil))
        }

        if options.melodiaTrick {
            while true {
                var best = Float(0)
                var bestFrame = 0
                var bestBin = 0
                for frame in 0 ..< frameCount {
                    for bin in 0 ..< bins where remaining[frame][bin] > best {
                        best = remaining[frame][bin]
                        bestFrame = frame
                        bestBin = bin
                    }
                }
                if best <= options.frameThreshold { break }
                remaining[bestFrame][bestBin] = 0

                var index = bestFrame + 1
                var quiet = 0
                while index < frameCount - 1 && quiet < energyTolerance {
                    quiet = remaining[index][bestBin] < options.frameThreshold ? quiet + 1 : 0
                    clear(&remaining, frame: index, bin: bestBin, highestBin: highestBin)
                    index += 1
                }
                let end = index - 1 - quiet

                index = bestFrame - 1
                quiet = 0
                while index > 0 && quiet < energyTolerance {
                    quiet = remaining[index][bestBin] < options.frameThreshold ? quiet + 1 : 0
                    clear(&remaining, frame: index, bin: bestBin, highestBin: highestBin)
                    index -= 1
                }
                let start = index + 1 + quiet

                if end - start <= options.minimumNoteFrames { continue }
                events.append(NFKBasicPitchNoteEvent(startFrame: start, endFrame: end,
                                                     pitch: bestBin + configuration.lowestPitch,
                                                     amplitude: mean(frames, from: start, to: end, bin: bestBin),
                                                     bendBins: nil))
            }
        }

        return events
    }

    private static func clear(_ matrix: inout [[Float]], frame: Int, bin: Int, highestBin: Int) {
        matrix[frame][bin] = 0
        if bin < highestBin { matrix[frame][bin + 1] = 0 }
        if bin > 0 { matrix[frame][bin - 1] = 0 }
    }

    private static func mean(_ matrix: [[Float]], from start: Int, to end: Int, bin: Int) -> Float {
        guard end > start else { return 0 }
        var total = Float(0)
        for frame in start ..< end { total += matrix[frame][bin] }
        return total / Float(end - start)
    }

    static func hzToMIDI(_ hz: Double) -> Double {
        12.0 * log2(hz / 440.0) + 69.0
    }

    static func midiToHz(_ pitch: Int) -> Double {
        440.0 * pow(2.0, (Double(pitch) - 69.0) / 12.0)
    }

    /// The pitch-bend curve of each note, read out of the contour posteriorgram: within a window of
    /// bins around the note's own, the loudest bin per frame, weighted by a Gaussian so a far-off
    /// harmonic does not win. The result is in contour bins, a third of a semitone each.
    static func pitchBends(_ contour: [[Float]], events: [NFKBasicPitchNoteEvent],
                           configuration: NFKMLXBasicPitchConfiguration,
                           tolerance: Int = 25) -> [NFKBasicPitchNoteEvent] {
        let windowLength = tolerance * 2 + 1
        let gaussian: [Float] = (0 ..< windowLength).map {
            let offset = (Double($0) - Double(windowLength - 1) / 2.0) / 5.0
            return Float(exp(-0.5 * offset * offset))
        }
        let contourBins = configuration.contourBins

        return events.map { event in
            var event = event
            let center = Int((Double(configuration.binsPerSemitone) * 12.0
                              * log2(midiToHz(event.pitch) / configuration.baseFrequency)).rounded())
            let first = max(center - tolerance, 0)
            let last = min(contourBins, center + tolerance + 1)
            let windowStart = max(0, tolerance - center)
            let windowEnd = windowLength - max(0, center - (contourBins - tolerance - 1))
            guard last > first, windowEnd > windowStart else { return event }
            let shift = tolerance - windowStart

            var bends = [Int]()
            for frame in event.startFrame ..< min(event.endFrame, contour.count) {
                var best = -Float.greatestFiniteMagnitude
                var bestOffset = 0
                for (offset, bin) in (first ..< last).enumerated() {
                    let weight = gaussian[windowStart + offset]
                    let value = contour[frame][bin] * weight
                    if value > best {
                        best = value
                        bestOffset = offset
                    }
                }
                bends.append(bestOffset - shift)
            }
            event.bendBins = bends
            return event
        }
    }

    /// Drops the bend curve from every note that overlaps another in time. The pitch wheel is a
    /// property of a MIDI channel, so two notes sounding together cannot bend apart on one.
    static func dropOverlappingBends(_ events: [NFKBasicPitchNoteEvent]) -> [NFKBasicPitchNoteEvent] {
        var sorted = events.sorted {
            ($0.startFrame, $0.endFrame, $0.pitch) < ($1.startFrame, $1.endFrame, $1.pitch)
        }
        guard sorted.count > 1 else { return sorted }
        for i in 0 ..< (sorted.count - 1) {
            for j in (i + 1) ..< sorted.count {
                if sorted[j].startFrame >= sorted[i].endFrame { break }
                sorted[i].bendBins = nil
                sorted[j].bendBins = nil
            }
        }
        return sorted
    }
}

extension NFKMLXBasicPitchNet {

    /// The seconds each frame of an unwrapped posteriorgram falls at.
    ///
    /// The reference corrects for the fraction of a frame each window's samples fall short of its
    /// frame count, accumulated once per window, and adds the alignment offset it measured.
    func frameTimes(count: Int) -> [Double] {
        let hop = Double(configuration.hopLength) / Double(configuration.sampleRate)
        let windowOffset = hop * (Double(configuration.frames)
                                  - Double(configuration.windowSamples) / Double(configuration.hopLength)) + 0.0018
        return (0 ..< count).map { frame in
            Double(frame) * hop - windowOffset * Double(frame / configuration.frames)
        }
    }

    /// Splits a clip into the overlapping windows the network reads. The clip is prepended with half an
    /// overlap of silence so the first window's own seam falls before the audio starts.
    func windows(_ samples: [Float]) -> MLXArray {
        let overlap = configuration.overlappingFrames * configuration.hopLength
        var padded = [Float](repeating: 0, count: overlap / 2)
        padded.append(contentsOf: samples)

        let hop = configuration.windowHop
        var rows = [Float]()
        var count = 0
        var start = 0
        repeat {
            var window = [Float](repeating: 0, count: configuration.windowSamples)
            for i in 0 ..< configuration.windowSamples where start + i < padded.count {
                window[i] = padded[start + i]
            }
            rows.append(contentsOf: window)
            count += 1
            start += hop
        } while start < padded.count

        return rows.withUnsafeBufferPointer { MLXArray($0, [count, configuration.windowSamples, 1]) }
    }

    /// Runs every window and stitches the results: half the overlap comes off each window's ends, and
    /// the tail is cut to the frames the clip's own length reaches.
    func unwrapped(_ windowed: MLXArray, sampleCount: Int) -> (contour: [[Float]], note: [[Float]], onset: [[Float]]) {
        let (contour, note, onset) = posteriorgrams(windowed)
        eval(contour, note, onset)

        let half = configuration.overlappingFrames / 2
        let framesPerWindow = configuration.frames - configuration.overlappingFrames
        // The reference trims to the frames the clip itself reaches, counted at its own integer frame
        // rate (`sampleRate / hop`, 86 rather than 86.13).
        let framesPerSecond = configuration.sampleRate / configuration.hopLength
        let expected = Int((Double(sampleCount) * Double(framesPerSecond) / Double(configuration.sampleRate)).rounded(.down))

        func stitch(_ array: MLXArray) -> [[Float]] {
            let windows = array.dim(0)
            let bins = array.dim(2)
            let values = array[0..., half ..< (array.dim(1) - half), 0...].asArray(Float.self)
            var rows = [[Float]]()
            rows.reserveCapacity(windows * framesPerWindow)
            for window in 0 ..< windows {
                for frame in 0 ..< framesPerWindow {
                    let base = (window * framesPerWindow + frame) * bins
                    rows.append(Array(values[base ..< (base + bins)]))
                }
            }
            return Array(rows.prefix(max(expected, 0)))
        }
        return (stitch(contour), stitch(note), stitch(onset))
    }

    /// Transcribes a mono clip. The network reads one rate, so a clip at another is resampled first.
    public func transcribe(_ samples: [Float], sampleRate: Int,
                           options: NFKMLXBasicPitchOptions = .default) -> NFKMIDISequence {
        let matched = NFKMLXAudioRate.matched(samples, from: sampleRate, to: configuration.sampleRate)
        let (contour, note, onset) = unwrapped(windows(matched), sampleCount: matched.count)

        var events = NFKBasicPitchNoteCreation.notes(note: note, onset: onset,
                                                     configuration: configuration, options: options)
        if options.includesPitchBends {
            events = NFKBasicPitchNoteCreation.pitchBends(contour, events: events, configuration: configuration)
            events = NFKBasicPitchNoteCreation.dropOverlappingBends(events)
        }

        let times = frameTimes(count: max(contour.count, 1))
        let notes: [NFKMIDINote] = events.map { event in
            let start = times.indices.contains(event.startFrame) ? times[event.startFrame] : 0
            let end = times.indices.contains(event.endFrame) ? times[event.endFrame] : start
            let bend = event.bendBins.map { bins in
                bins.map { NSNumber(value: Double($0) / Double(configuration.binsPerSemitone)) }
            }
            return NFKMIDINote(pitch: event.pitch, startSeconds: start, endSeconds: end,
                               velocity: min(127, max(0, Int((127.0 * Double(event.amplitude)).rounded()))),
                               program: options.program, percussion: false, pitchBend: bend)
        }
        return NFKMIDISequence(notes: notes, tempoBPM: options.tempoBPM, beatsPerBar: 4, beatUnit: 4,
                               ticksPerQuarterNote: 480)
    }
}

/// The request parameters a music-transcription backend reads, for a caller that configures note
/// creation the way it sets `NFKParameterTemperature`.
@objc(NFKMLXTranscriptionParameterKey)
public final class NFKMLXTranscriptionParameterKey: NSObject {
    /// The onset activation a peak must reach to start a note (NSNumber, 0...1).
    @objc public static let onsetThreshold = "NFKMLXParameterOnsetThreshold"
    /// The note activation below which a sounding note ends (NSNumber, 0...1).
    @objc public static let frameThreshold = "NFKMLXParameterFrameThreshold"
    /// The frames a note must last to be kept (NSNumber).
    @objc public static let minimumNoteFrames = "NFKMLXParameterMinimumNoteFrames"
    /// Whether notes carry a pitch-bend curve (NSNumber boolean).
    @objc public static let pitchBends = "NFKMLXParameterPitchBends"
    /// Whether note creation picks up notes the onset head missed (NSNumber boolean).
    @objc public static let melodiaTrick = "NFKMLXParameterMelodiaTrick"
    /// The lowest frequency to transcribe, in hertz (NSNumber).
    @objc public static let minimumFrequency = "NFKMLXParameterMinimumFrequency"
    /// The highest frequency to transcribe, in hertz (NSNumber).
    @objc public static let maximumFrequency = "NFKMLXParameterMaximumFrequency"
    /// The tempo the MIDI is written at (NSNumber, beats per minute).
    @objc public static let tempo = "NFKMLXParameterTempo"
    /// The General MIDI program the notes are written on (NSNumber, 0...127).
    @objc public static let program = "NFKMLXParameterProgram"
}

/// Holds the network for capture in the backend's `@Sendable` closure.
private final class NFKBasicPitchHolder: @unchecked Sendable {
    let net: NFKMLXBasicPitchNet
    init(_ net: NFKMLXBasicPitchNet) { self.net = net }
}

/// Basic Pitch as an InferKit backend. Reads `NFKInputAudio`; returns the transcription as an
/// `NFKMIDISequence` under `NFKOutputMIDI`.
@objc(NFKMLXBasicPitchBackend)
public final class NFKMLXBasicPitchBackend: NSObject, NFKInferenceBackend {

    private let holder: NFKBasicPitchHolder
    private let identifier: String

    init(net: NFKMLXBasicPitchNet, identifier: String) {
        holder = NFKBasicPitchHolder(net)
        self.identifier = identifier
        super.init()
    }

    @objc public var isReady: Bool { true }
    @objc public var backendIdentifier: String { identifier }

    @objc public var supportedParameterKeys: Set<String> {
        [NFKMLXTranscriptionParameterKey.onsetThreshold,
         NFKMLXTranscriptionParameterKey.frameThreshold,
         NFKMLXTranscriptionParameterKey.minimumNoteFrames,
         NFKMLXTranscriptionParameterKey.pitchBends,
         NFKMLXTranscriptionParameterKey.melodiaTrick,
         NFKMLXTranscriptionParameterKey.minimumFrequency,
         NFKMLXTranscriptionParameterKey.maximumFrequency,
         NFKMLXTranscriptionParameterKey.tempo,
         NFKMLXTranscriptionParameterKey.program]
    }

    @objc public var supportedInputKeys: Set<String> { [NFKInputAudio] }

    @objc(runInferenceForRequest:error:)
    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        guard let (samples, sampleRate) = Self.audio(from: request) else {
            throw NFKMLXError.unsupportedInput
        }
        let sequence = holder.net.transcribe(samples, sampleRate: sampleRate,
                                             options: Self.options(from: request))
        return NFKInferenceResult(outputs: [NFKOutputMIDI: sequence])
    }

    @objc(submitInferenceJobForRequest:)
    public func submitInferenceJob(for request: NFKInferenceRequest) -> NFKInferenceJob {
        let job = NFKInferenceJob()
        Task.detached(priority: .userInitiated) {
            do {
                job.finish(with: try self.runInference(for: request))
            } catch {
                job.finish(withError: error as NSError)
            }
        }
        return job
    }

    static func options(from request: NFKInferenceRequest) -> NFKMLXBasicPitchOptions {
        var options = NFKMLXBasicPitchOptions.default
        func number(_ key: String) -> NSNumber? { request.parameter(forKey: key) as? NSNumber }
        if let value = number(NFKMLXTranscriptionParameterKey.onsetThreshold) { options.onsetThreshold = value.floatValue }
        if let value = number(NFKMLXTranscriptionParameterKey.frameThreshold) { options.frameThreshold = value.floatValue }
        if let value = number(NFKMLXTranscriptionParameterKey.minimumNoteFrames) { options.minimumNoteFrames = value.intValue }
        if let value = number(NFKMLXTranscriptionParameterKey.pitchBends) { options.includesPitchBends = value.boolValue }
        if let value = number(NFKMLXTranscriptionParameterKey.melodiaTrick) { options.melodiaTrick = value.boolValue }
        if let value = number(NFKMLXTranscriptionParameterKey.minimumFrequency) { options.minimumFrequency = value.doubleValue }
        if let value = number(NFKMLXTranscriptionParameterKey.maximumFrequency) { options.maximumFrequency = value.doubleValue }
        if let value = number(NFKMLXTranscriptionParameterKey.tempo) { options.tempoBPM = value.doubleValue }
        if let value = number(NFKMLXTranscriptionParameterKey.program) { options.program = value.intValue }
        return options
    }

    private static func audio(from request: NFKInferenceRequest) -> (samples: [Float], sampleRate: Int)? {
        guard let value = request.input(forKey: NFKInputAudio) else { return nil }
        if let asset = value as? NFKAudioAsset, let url = asset.fileURL, let data = try? Data(contentsOf: url) {
            return NFKMLXWaveFile.read(data)
        }
        if let data = value as? Data { return NFKMLXWaveFile.read(data) }
        return nil
    }
}

/// Registration and weight loading for Basic Pitch.
@objc(NFKMLXBasicPitch)
public final class NFKMLXBasicPitch: NSObject {

    /// The registry name the model builds under.
    @objc public static let modelName = "basic-pitch"

    public static func makeNet(_ configuration: NFKMLXBasicPitchConfiguration = .icassp2022) -> NFKMLXBasicPitchNet {
        NFKMLXBasicPitchNet(configuration)
    }

    /// Builds a Basic Pitch backend from optional local weights — no registry required. A nil
    /// `weightsURL` builds random weights (`isReady` is true). Run inference off the render thread.
    ///
    /// - Since: InferKit 0.4.0
    @objc(backendWithWeightsURL:error:)
    public static func backend(weightsURL: URL?) throws -> any NFKInferenceBackend {
        let net = try weightsURL.map { try network(weightsURL: $0) } ?? makeNet()
        return NFKMLXBasicPitchBackend(net: net, identifier: modelName)
    }

    /// Downloads the checkpoint from Hugging Face, then builds — no registry required. Blocking on the
    /// network; run off the render thread.
    @objc(backendWithRepo:weightsPath:revision:cacheDirectoryURL:error:)
    public static func backend(repo: String, weightsPath: String, revision: String?, cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        let url = try NFKMLXDownload.weightsURL(repo: repo, weightsPath: weightsPath, revision: revision, cacheDirectoryURL: cacheDirectoryURL)
        return try backend(weightsURL: url)
    }

    /// The asynchronous form of the download factory: downloads on a background queue, then builds and
    /// delivers the backend (or an error) to `completionHandler`.
    @objc(backendWithRepo:weightsPath:revision:cacheDirectoryURL:completionHandler:)
    public static func backend(repo: String, weightsPath: String, revision: String?, cacheDirectoryURL: URL?,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXDownload.backend(repo: repo, weightsPath: weightsPath, revision: revision,
                               cacheDirectoryURL: cacheDirectoryURL,
                               build: { try backend(weightsURL: $0) },
                               completionHandler: completionHandler)
    }

    /// Registers Basic Pitch (`basic-pitch`) with `NFKMLXModelRegistry`.
    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName) { weightsURL in try backend(weightsURL: weightsURL) }
    }

    /// Loads a converted checkpoint, transposing convolution weights into MLX's layout: `[out, in, k]`
    /// → `[out, k, in]` for the CQT's kernels and `[out, in, kH, kW]` → `[out, kH, kW, in]` for the
    /// network's. A file `NFKMLXWeights.save` wrote is already in MLX's layout and loads as it is.
    ///
    /// The checkpoint's layout must match the network's: build the network with ``network(weightsURL:reinitializing:)``,
    /// which reads the layout from the file.
    public static func loadWeights(into net: NFKMLXBasicPitchNet, from url: URL) throws {
        try apply(try NFKMLXWeights.loadCheckpoint(url: url), to: net)
    }

    /// The normalization layout a checkpoint holds: separate when it carries the normalization after
    /// the log as a batch normalization, folded otherwise.
    ///
    /// - Since: InferKit 0.5.0
    public static func normalization(ofCheckpointAt url: URL) throws -> NFKMLXBasicPitchNormalization {
        normalization(of: try NFKMLXWeights.loadCheckpoint(url: url))
    }

    static func normalization(of checkpoint: NFKMLXWeights.Checkpoint) -> NFKMLXBasicPitchNormalization {
        checkpoint.arrays["log_norm.weight"] != nil ? .separate : .folded
    }

    static func mapped(_ checkpoint: NFKMLXWeights.Checkpoint) -> [(String, MLXArray)] {
        checkpoint.arrays.map { key, value in
            guard checkpoint.needsConvTranspose else { return (key, value) }
            switch value.ndim {
            case 3: return (key, value.transposed(0, 2, 1))
            case 4: return (key, value.transposed(0, 2, 3, 1))
            default: return (key, value)
            }
        }
    }

    static func apply(_ checkpoint: NFKMLXWeights.Checkpoint, to net: NFKMLXBasicPitchNet) throws {
        let held = normalization(of: checkpoint)
        guard held == net.configuration.normalization else {
            throw NFKMLXError.weightsMismatch(
                "the checkpoint holds Basic Pitch's \(held) normalization layout and the network was built "
                + "for \(net.configuration.normalization); build it with `network(weightsURL:)`, which "
                + "reads the layout from the file")
        }
        try NFKMLXWeights.apply(mapped(checkpoint), to: net)
    }
}
