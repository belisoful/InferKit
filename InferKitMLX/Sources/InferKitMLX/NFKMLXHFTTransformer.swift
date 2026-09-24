//
//  NFKMLXHFTTransformer.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXNN
import MLXFast

// hFT-Transformer transcribes piano by attending in two directions in turn. A convolutional stem
// reads a 65-frame window around each output frame; a transformer then attends ACROSS FREQUENCY,
// turning 256 mel bins into 88 note queries through cross-attention; and a second transformer
// attends ACROSS TIME, refining each note's own track of 128 frames. Four heads read each level:
// onset, offset, multi-pitch (whether the note is sounding), and velocity.
//
// The two levels are both outputs. The frequency level is the model's first answer and the time
// level is its refinement, and the reference evaluates the second, which is what this port returns.
//
// 5.5M parameters, MIT, from Sony's ISMIR 2023 implementation. It is the accuracy counterpart to the
// instrument-agnostic Basic Pitch: piano only, and much better at it.
//
// Two details of the reference's shape are worth stating because they are unusual. Each layer holds
// ONE LayerNorm instance and applies it after every residual, rather than one norm per residual, so
// a layer has a single `layer_norm.weight`. And the model is post-norm throughout.

/// hFT-Transformer's geometry. The defaults are the released MAESTRO model.
public struct NFKMLXHFTTransformerConfiguration: Sendable {
    public var sampleRate: Int
    public var hopLength: Int
    public var nFFT: Int
    public var melBins: Int
    /// The floor added inside the log. Its logarithm is also what the segment padding is filled with.
    public var logOffset: Float
    /// The frames of context before each segment.
    public var marginBefore: Int
    /// The frames of context after each segment.
    public var marginAfter: Int
    /// The frames one forward scores.
    public var segmentFrames: Int
    public var hidden: Int
    public var heads: Int
    public var feedForward: Int
    public var layers: Int
    public var cnnChannels: Int
    public var cnnKernel: Int
    public var notes: Int
    /// The MIDI pitch of the lowest note. 21 is A0.
    public var lowestPitch: Int
    public var velocities: Int

    public init(sampleRate: Int = 16000, hopLength: Int = 256, nFFT: Int = 2048, melBins: Int = 256,
                logOffset: Float = 1e-8, marginBefore: Int = 32, marginAfter: Int = 32,
                segmentFrames: Int = 128, hidden: Int = 256, heads: Int = 4, feedForward: Int = 512,
                layers: Int = 3, cnnChannels: Int = 4, cnnKernel: Int = 5, notes: Int = 88,
                lowestPitch: Int = 21, velocities: Int = 128) {
        self.sampleRate = sampleRate
        self.hopLength = hopLength
        self.nFFT = nFFT
        self.melBins = melBins
        self.logOffset = logOffset
        self.marginBefore = marginBefore
        self.marginAfter = marginAfter
        self.segmentFrames = segmentFrames
        self.hidden = hidden
        self.heads = heads
        self.feedForward = feedForward
        self.layers = layers
        self.cnnChannels = cnnChannels
        self.cnnKernel = cnnKernel
        self.notes = notes
        self.lowestPitch = lowestPitch
        self.velocities = velocities
    }

    /// The released MAESTRO-V3 model (`model_016_003.pkl`).
    public static let maestro = NFKMLXHFTTransformerConfiguration()

    /// The frames the convolutional stem reads per output frame.
    public var processFrames: Int { marginBefore + marginAfter + 1 }
    /// The width the stem hands the frequency transformer.
    public var cnnDimension: Int { cnnChannels * (processFrames - cnnKernel + 1) }
    /// What the segment padding is filled with: the log-mel value of silence.
    public var padValue: Float { log(logOffset) }
    /// The seconds one frame spans.
    public var frameSeconds: Double { Double(hopLength) / Double(sampleRate) }
    public var headDimension: Int { hidden / heads }
}

/// The log-mel front end. The window and filterbank are the reference's torchaudio ones, shipped in
/// the converted checkpoint rather than re-derived.
final class NFKHFTFrontEnd: Module {
    @ParameterInfo(key: "window") var window: MLXArray
    @ParameterInfo(key: "filterbank") var filterbank: MLXArray          // [nFFT / 2 + 1, melBins]

    let configuration: NFKMLXHFTTransformerConfiguration

    init(_ configuration: NFKMLXHFTTransformerConfiguration) {
        self.configuration = configuration
        _window.wrappedValue = MLXArray.zeros([configuration.nFFT])
        _filterbank.wrappedValue = MLXArray.zeros([configuration.nFFT / 2 + 1, configuration.melBins])
    }

    /// A mono clip at the model's rate → `[frames, melBins]` log-mel.
    func callAsFunction(_ samples: [Float]) -> MLXArray {
        let nFFT = configuration.nFFT
        let hop = configuration.hopLength
        let pad = nFFT / 2

        // torchaudio centers the frames and the reference asks for constant (zero) padding.
        var padded = [Float](repeating: 0, count: samples.count + 2 * pad)
        for index in 0 ..< samples.count { padded[pad + index] = samples[index] }

        let frames = 1 + (padded.count - nFFT) / hop
        var framed = [Float](repeating: 0, count: frames * nFFT)
        for frame in 0 ..< frames {
            let start = frame * hop
            for index in 0 ..< nFFT { framed[frame * nFFT + index] = padded[start + index] }
        }
        let windowed = framed.withUnsafeBufferPointer { MLXArray($0, [frames, nFFT]) } * window
        let spectrum = MLXFFT.rfft(windowed, axis: 1)
        // The reference's MelSpectrogram takes power 2.
        let power = spectrum.realPart().square() + spectrum.imaginaryPart().square()
        return log(power.matmul(filterbank) + configuration.logOffset)
    }
}

/// Multi-head attention with separate projections, as the reference's seq2seq base writes it.
final class NFKHFTAttention: Module {
    @ModuleInfo(key: "fc_q") var query: Linear
    @ModuleInfo(key: "fc_k") var key: Linear
    @ModuleInfo(key: "fc_v") var value: Linear
    @ModuleInfo(key: "fc_o") var output: Linear

    let heads: Int
    let headDimension: Int

    init(hidden: Int, heads: Int) {
        self.heads = heads
        headDimension = hidden / heads
        _query.wrappedValue = Linear(hidden, hidden)
        _key.wrappedValue = Linear(hidden, hidden)
        _value.wrappedValue = Linear(hidden, hidden)
        _output.wrappedValue = Linear(hidden, hidden)
    }

    /// Returns the attended values and the attention itself, which the frequency decoder keeps: it is
    /// the map from note to frequency bin the paper shows.
    func callAsFunction(_ queries: MLXArray, _ keys: MLXArray, _ values: MLXArray)
        -> (output: MLXArray, attention: MLXArray) {
        let batch = queries.dim(0)
        func split(_ array: MLXArray) -> MLXArray {
            array.reshaped([batch, -1, heads, headDimension]).transposed(0, 2, 1, 3)
        }
        let q = split(query(queries))
        let k = split(key(keys))
        let v = split(value(values))

        let energy = q.matmul(k.transposed(0, 1, 3, 2)) / sqrt(Float(headDimension))
        let attention = softmax(energy, axis: -1)
        let context = attention.matmul(v).transposed(0, 2, 1, 3)
        return (output(context.reshaped([batch, -1, heads * headDimension])), attention)
    }
}

/// The position-wise feed-forward: expand, rectify, project back.
final class NFKHFTFeedForward: Module {
    @ModuleInfo(key: "fc_1") var expand: Linear
    @ModuleInfo(key: "fc_2") var project: Linear

    init(hidden: Int, feedForward: Int) {
        _expand.wrappedValue = Linear(hidden, feedForward)
        _project.wrappedValue = Linear(feedForward, hidden)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { project(relu(expand(x))) }
}

/// A post-norm self-attention block. The single `layer_norm` is applied after both residuals, which
/// is the reference's own shape rather than an omission.
final class NFKHFTEncoderLayer: Module {
    @ModuleInfo(key: "layer_norm") var norm: LayerNorm
    @ModuleInfo(key: "self_attention") var attention: NFKHFTAttention
    @ModuleInfo(key: "positionwise_feedforward") var feedForward: NFKHFTFeedForward

    init(_ configuration: NFKMLXHFTTransformerConfiguration) {
        _norm.wrappedValue = LayerNorm(dimensions: configuration.hidden)
        _attention.wrappedValue = NFKHFTAttention(hidden: configuration.hidden, heads: configuration.heads)
        _feedForward.wrappedValue = NFKHFTFeedForward(hidden: configuration.hidden,
                                                      feedForward: configuration.feedForward)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = norm(x + attention(x, x, x).output)
        h = norm(h + feedForward(h))
        return h
    }
}

/// The first decoder block: the note queries read the frequency axis, with no self-attention yet.
final class NFKHFTDecoderZeroLayer: Module {
    @ModuleInfo(key: "layer_norm") var norm: LayerNorm
    @ModuleInfo(key: "encoder_attention") var attention: NFKHFTAttention
    @ModuleInfo(key: "positionwise_feedforward") var feedForward: NFKHFTFeedForward

    init(_ configuration: NFKMLXHFTTransformerConfiguration) {
        _norm.wrappedValue = LayerNorm(dimensions: configuration.hidden)
        _attention.wrappedValue = NFKHFTAttention(hidden: configuration.hidden, heads: configuration.heads)
        _feedForward.wrappedValue = NFKHFTFeedForward(hidden: configuration.hidden,
                                                      feedForward: configuration.feedForward)
    }

    func callAsFunction(_ encoded: MLXArray, _ notes: MLXArray) -> (MLXArray, MLXArray) {
        let (attended, attention) = self.attention(notes, encoded, encoded)
        var h = norm(notes + attended)
        h = norm(h + feedForward(h))
        return (h, attention)
    }
}

/// A later decoder block: the notes attend to each other, then to the frequency axis.
final class NFKHFTDecoderLayer: Module {
    @ModuleInfo(key: "layer_norm") var norm: LayerNorm
    @ModuleInfo(key: "self_attention") var selfAttention: NFKHFTAttention
    @ModuleInfo(key: "encoder_attention") var encoderAttention: NFKHFTAttention
    @ModuleInfo(key: "positionwise_feedforward") var feedForward: NFKHFTFeedForward

    init(_ configuration: NFKMLXHFTTransformerConfiguration) {
        _norm.wrappedValue = LayerNorm(dimensions: configuration.hidden)
        _selfAttention.wrappedValue = NFKHFTAttention(hidden: configuration.hidden, heads: configuration.heads)
        _encoderAttention.wrappedValue = NFKHFTAttention(hidden: configuration.hidden, heads: configuration.heads)
        _feedForward.wrappedValue = NFKHFTFeedForward(hidden: configuration.hidden,
                                                      feedForward: configuration.feedForward)
    }

    func callAsFunction(_ encoded: MLXArray, _ notes: MLXArray) -> (MLXArray, MLXArray) {
        var h = norm(notes + selfAttention(notes, notes, notes).output)
        let (attended, attention) = encoderAttention(h, encoded, encoded)
        h = norm(h + attended)
        h = norm(h + feedForward(h))
        return (h, attention)
    }
}

/// The frequency encoder: a convolutional stem over the context window, then self-attention across
/// the 256 mel bins.
final class NFKHFTEncoder: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d
    @ModuleInfo(key: "tok_embedding_freq") var tokenEmbedding: Linear
    @ModuleInfo(key: "pos_embedding_freq") var positionEmbedding: Embedding
    @ModuleInfo(key: "layers_freq") var layers: [NFKHFTEncoderLayer]

    let configuration: NFKMLXHFTTransformerConfiguration

    init(_ configuration: NFKMLXHFTTransformerConfiguration) {
        self.configuration = configuration
        _conv.wrappedValue = Conv2d(inputChannels: 1, outputChannels: configuration.cnnChannels,
                                    kernelSize: [1, configuration.cnnKernel])
        _tokenEmbedding.wrappedValue = Linear(configuration.cnnDimension, configuration.hidden)
        _positionEmbedding.wrappedValue = Embedding(embeddingCount: configuration.melBins,
                                                    dimensions: configuration.hidden)
        _layers.wrappedValue = (0 ..< configuration.layers).map { _ in NFKHFTEncoderLayer(configuration) }
    }

    /// `[bins, marginBefore + segmentFrames + marginAfter]` → `[segmentFrames, bins, hidden]`.
    func callAsFunction(_ spectrogram: MLXArray) -> MLXArray {
        let bins = configuration.melBins
        let frames = configuration.segmentFrames
        let process = configuration.processFrames

        // One sliding window of `process` frames per output frame.
        var gather = [Int32]()
        gather.reserveCapacity(frames * process)
        for frame in 0 ..< frames {
            for offset in 0 ..< process { gather.append(Int32(frame + offset)) }
        }
        let windows = take(spectrogram, MLXArray(gather), axis: 1)       // [bins, frames · process]
            .reshaped([bins, frames, process]).transposed(1, 0, 2)       // [frames, bins, process]

        // The stem reads each bin's own window. In MLX's layout the channel axis trails, so it is
        // moved back in front of the window axis before the flatten: the reference's width is
        // (channel, position), not (position, channel).
        let convolved = conv(windows.expandedDimensions(axis: 3))        // [frames, bins, process - k + 1, channels]
        let stem = convolved.transposed(0, 1, 3, 2)
            .reshaped([frames, bins, configuration.cnnDimension])

        let positions = positionEmbedding(MLXArray(Array(0 ..< bins).map { Int32($0) }))
        var h = tokenEmbedding(stem) * sqrt(Float(configuration.hidden)) + positions
        for layer in layers { h = layer(h) }
        return h                                                          // [frames, bins, hidden]
    }
}

/// What one segment scores, at both levels.
public struct NFKMLXHFTTransformerOutput {
    /// `[frames, notes]`, the frequency level.
    public var onsetFrequency: MLXArray
    public var offsetFrequency: MLXArray
    public var mpeFrequency: MLXArray
    /// `[frames, notes, velocities]` logits.
    public var velocityFrequency: MLXArray
    /// `[frames, notes]`, the time level, which is the model's final answer.
    public var onset: MLXArray
    public var offset: MLXArray
    public var mpe: MLXArray
    public var velocity: MLXArray
    /// `[frames, heads, notes, bins]`, the note-to-frequency map the first decoder block forms.
    public var attention: MLXArray
}

/// The decoder: 88 note queries read the frequency axis, then each note's own frames read each other.
final class NFKHFTDecoder: Module {
    @ModuleInfo(key: "pos_embedding_freq") var notePositions: Embedding
    @ModuleInfo(key: "layer_zero_freq") var firstLayer: NFKHFTDecoderZeroLayer
    @ModuleInfo(key: "layers_freq") var frequencyLayers: [NFKHFTDecoderLayer]
    @ModuleInfo(key: "fc_onset_freq") var onsetFrequency: Linear
    @ModuleInfo(key: "fc_offset_freq") var offsetFrequency: Linear
    @ModuleInfo(key: "fc_mpe_freq") var mpeFrequency: Linear
    @ModuleInfo(key: "fc_velocity_freq") var velocityFrequency: Linear
    @ModuleInfo(key: "pos_embedding_time") var timePositions: Embedding
    @ModuleInfo(key: "layers_time") var timeLayers: [NFKHFTEncoderLayer]
    @ModuleInfo(key: "fc_onset_time") var onsetTime: Linear
    @ModuleInfo(key: "fc_offset_time") var offsetTime: Linear
    @ModuleInfo(key: "fc_mpe_time") var mpeTime: Linear
    @ModuleInfo(key: "fc_velocity_time") var velocityTime: Linear

    let configuration: NFKMLXHFTTransformerConfiguration

    init(_ configuration: NFKMLXHFTTransformerConfiguration) {
        self.configuration = configuration
        let hidden = configuration.hidden
        _notePositions.wrappedValue = Embedding(embeddingCount: configuration.notes, dimensions: hidden)
        _firstLayer.wrappedValue = NFKHFTDecoderZeroLayer(configuration)
        _frequencyLayers.wrappedValue = (0 ..< (configuration.layers - 1)).map { _ in NFKHFTDecoderLayer(configuration) }
        _onsetFrequency.wrappedValue = Linear(hidden, 1)
        _offsetFrequency.wrappedValue = Linear(hidden, 1)
        _mpeFrequency.wrappedValue = Linear(hidden, 1)
        _velocityFrequency.wrappedValue = Linear(hidden, configuration.velocities)
        _timePositions.wrappedValue = Embedding(embeddingCount: configuration.segmentFrames, dimensions: hidden)
        _timeLayers.wrappedValue = (0 ..< configuration.layers).map { _ in NFKHFTEncoderLayer(configuration) }
        _onsetTime.wrappedValue = Linear(hidden, 1)
        _offsetTime.wrappedValue = Linear(hidden, 1)
        _mpeTime.wrappedValue = Linear(hidden, 1)
        _velocityTime.wrappedValue = Linear(hidden, configuration.velocities)
    }

    /// `[frames, bins, hidden]` → both levels.
    func callAsFunction(_ encoded: MLXArray) -> NFKMLXHFTTransformerOutput {
        let frames = configuration.segmentFrames
        let notes = configuration.notes
        let hidden = configuration.hidden

        // The note queries are the position embedding alone, unscaled — unlike the encoder's tokens
        // and unlike the time branch below, both of which scale by the square root of the width.
        let queries = notePositions(MLXArray(Array(0 ..< notes).map { Int32($0) }))
        var h = broadcast(queries.expandedDimensions(axis: 0), to: [frames, notes, hidden])

        var (state, attention) = firstLayer(encoded, h)
        h = state
        for layer in frequencyLayers {
            (state, attention) = layer(encoded, h)
            h = state
        }

        let onsetFrequencyOut = sigmoid(onsetFrequency(h).reshaped([frames, notes]))
        let offsetFrequencyOut = sigmoid(offsetFrequency(h).reshaped([frames, notes]))
        let mpeFrequencyOut = sigmoid(mpeFrequency(h).reshaped([frames, notes]))
        let velocityFrequencyOut = velocityFrequency(h).reshaped([frames, notes, configuration.velocities])

        // Each note's own track of frames, read as a sequence.
        var time = h.transposed(1, 0, 2)                                  // [notes, frames, hidden]
        let positions = timePositions(MLXArray(Array(0 ..< frames).map { Int32($0) }))
        time = time * sqrt(Float(hidden)) + positions
        for layer in timeLayers { time = layer(time) }

        let onsetOut = sigmoid(onsetTime(time).reshaped([notes, frames])).transposed(1, 0)
        let offsetOut = sigmoid(offsetTime(time).reshaped([notes, frames])).transposed(1, 0)
        let mpeOut = sigmoid(mpeTime(time).reshaped([notes, frames])).transposed(1, 0)
        let velocityOut = velocityTime(time).reshaped([notes, frames, configuration.velocities])
            .transposed(1, 0, 2)

        return NFKMLXHFTTransformerOutput(
            onsetFrequency: onsetFrequencyOut, offsetFrequency: offsetFrequencyOut,
            mpeFrequency: mpeFrequencyOut, velocityFrequency: velocityFrequencyOut,
            onset: onsetOut, offset: offsetOut, mpe: mpeOut, velocity: velocityOut,
            attention: attention)
    }
}

/// hFT-Transformer's network.
public final class NFKMLXHFTTransformerNet: Module {
    @ModuleInfo(key: "frontend") var frontEnd: NFKHFTFrontEnd
    @ModuleInfo(key: "encoder") var encoder: NFKHFTEncoder
    @ModuleInfo(key: "decoder") var decoder: NFKHFTDecoder

    public let configuration: NFKMLXHFTTransformerConfiguration

    public init(_ configuration: NFKMLXHFTTransformerConfiguration = .maestro) {
        self.configuration = configuration
        _frontEnd.wrappedValue = NFKHFTFrontEnd(configuration)
        _encoder.wrappedValue = NFKHFTEncoder(configuration)
        _decoder.wrappedValue = NFKHFTDecoder(configuration)
    }

    /// One segment: `[bins, marginBefore + segmentFrames + marginAfter]` in, both levels out.
    public func callAsFunction(_ segment: MLXArray) -> NFKMLXHFTTransformerOutput {
        decoder(encoder(segment))
    }

    /// The whole clip's posteriorgrams, segment by segment.
    ///
    /// The feature is padded with the log-mel value of silence: `marginBefore` frames in front, and
    /// enough behind to fill the last segment plus `marginAfter`.
    public func posteriorgrams(_ samples: [Float], sampleRate: Int)
        -> (onset: [[Float]], offset: [[Float]], mpe: [[Float]], velocity: [[Int]]) {
        let matched = NFKMLXAudioRate.matched(samples, from: sampleRate, to: configuration.sampleRate)
        let feature = frontEnd(matched)
        eval(feature)
        let frames = feature.dim(0)
        let segment = configuration.segmentFrames
        let segments = max(1, (frames + segment - 1) / segment)
        let tail = segments * segment - frames

        let padded = MLX.padded(feature,
                                widths: [IntOrPair((configuration.marginBefore, tail + configuration.marginAfter)),
                                         IntOrPair((0, 0))],
                                mode: .constant, value: MLXArray(configuration.padValue))

        var onset = [[Float]](), offset = [[Float]](), mpe = [[Float]](), velocity = [[Int]]()
        for index in 0 ..< segments {
            let start = index * segment
            let window = padded[start ..< (start + configuration.processFrames + segment - 1), 0...]
            let output = self(window.transposed(1, 0))
            eval(output.onset, output.offset, output.mpe, output.velocity)

            let notes = configuration.notes
            let onsetValues = output.onset.asArray(Float.self)
            let offsetValues = output.offset.asArray(Float.self)
            let mpeValues = output.mpe.asArray(Float.self)
            let velocityValues = argMax(output.velocity, axis: -1).asArray(Int32.self)
            for frame in 0 ..< segment {
                let base = frame * notes
                onset.append(Array(onsetValues[base ..< (base + notes)]))
                offset.append(Array(offsetValues[base ..< (base + notes)]))
                mpe.append(Array(mpeValues[base ..< (base + notes)]))
                velocity.append(velocityValues[base ..< (base + notes)].map(Int.init))
            }
        }
        return (onset, offset, mpe, velocity)
    }

    /// Transcribes a clip to notes.
    public func transcribe(_ samples: [Float], sampleRate: Int,
                           options: NFKMLXHFTTransformerOptions = .default) -> NFKMIDISequence {
        let grams = posteriorgrams(samples, sampleRate: sampleRate)
        let notes = NFKHFTNoteDetection.notes(onset: grams.onset, offset: grams.offset, mpe: grams.mpe,
                                              velocity: grams.velocity, configuration: configuration,
                                              options: options)
        return NFKMIDISequence(notes: notes, tempoBPM: options.tempoBPM, beatsPerBar: 4, beatUnit: 4,
                               ticksPerQuarterNote: 480)
    }
}

/// What note detection reads out of the four posteriorgrams.
public struct NFKMLXHFTTransformerOptions: Sendable {
    /// How an offset is chosen when both the offset head and the multi-pitch head give one.
    public enum OffsetRule: String, Sendable {
        /// The earlier of the two. The reference's default.
        case shorter
        /// The later of the two.
        case longer
        /// The offset head alone.
        case offset
    }

    public var onsetThreshold: Float
    public var offsetThreshold: Float
    public var multiPitchThreshold: Float
    public var offsetRule: OffsetRule
    /// Drops notes the velocity head scores at zero, which is the reference's default.
    public var dropsSilentNotes: Bool
    public var tempoBPM: Double
    /// The General MIDI program the notes are written on. 0 is Acoustic Grand Piano.
    public var program: Int

    public init(onsetThreshold: Float = 0.5, offsetThreshold: Float = 0.5,
                multiPitchThreshold: Float = 0.5, offsetRule: OffsetRule = .shorter,
                dropsSilentNotes: Bool = true, tempoBPM: Double = 120, program: Int = 0) {
        self.onsetThreshold = onsetThreshold
        self.offsetThreshold = offsetThreshold
        self.multiPitchThreshold = multiPitchThreshold
        self.offsetRule = offsetRule
        self.dropsSilentNotes = dropsSilentNotes
        self.tempoBPM = tempoBPM
        self.program = program
    }

    public static let `default` = NFKMLXHFTTransformerOptions()
}

/// The reference's `mpe2note`: the four posteriorgrams become notes.
enum NFKHFTNoteDetection {

    /// The frames where a note's activation is a local maximum above the threshold, with the peak's
    /// time refined by fitting its two neighbors.
    ///
    /// "Local maximum" here is the reference's own scan: from the frame, walk outward until a
    /// strictly different value is found, and keep the frame only if neither side is larger. A
    /// plateau therefore survives, which a strict neighbor comparison would drop.
    static func peaks(_ activation: [[Float]], note: Int, threshold: Float, hopSeconds: Double)
        -> [(frame: Int, time: Double)] {
        var found = [(frame: Int, time: Double)]()
        let frames = activation.count
        for frame in 0 ..< frames where activation[frame][note] >= threshold {
            let value = activation[frame][note]
            var leftOK = true
            var index = frame - 1
            while index >= 0 {
                if value > activation[index][note] { leftOK = true; break }
                if value < activation[index][note] { leftOK = false; break }
                index -= 1
            }
            var rightOK = true
            index = frame + 1
            while index < frames {
                if value > activation[index][note] { rightOK = true; break }
                if value < activation[index][note] { rightOK = false; break }
                index += 1
            }
            guard leftOK && rightOK else { continue }

            var time = Double(frame) * hopSeconds
            if frame > 0 && frame < frames - 1 {
                let before = activation[frame - 1][note]
                let after = activation[frame + 1][note]
                if before > after {
                    time -= hopSeconds * 0.5 * Double(before - after) / Double(value - after)
                } else if after > before {
                    time += hopSeconds * 0.5 * Double(after - before) / Double(value - before)
                }
            }
            found.append((frame, time))
        }
        return found
    }

    static func notes(onset: [[Float]], offset: [[Float]], mpe: [[Float]], velocity: [[Int]],
                      configuration: NFKMLXHFTTransformerConfiguration,
                      options: NFKMLXHFTTransformerOptions) -> [NFKMIDINote] {
        let hop = configuration.frameSeconds
        var found = [NFKMIDINote]()

        for note in 0 ..< configuration.notes {
            let onsets = peaks(onset, note: note, threshold: options.onsetThreshold, hopSeconds: hop)
            let offsets = peaks(offset, note: note, threshold: options.offsetThreshold, hopSeconds: hop)
            var perNote = [NFKMIDINote]()

            for (index, peak) in onsets.enumerated() {
                let nextFrame: Int
                let nextTime: Double
                if index + 1 < onsets.count {
                    nextFrame = onsets[index + 1].frame
                    nextTime = onsets[index + 1].time
                } else {
                    nextFrame = mpe.count
                    nextTime = Double(nextFrame - 1) * hop
                }

                // The first offset peak after this onset, capped at the next onset.
                var offsetFrame = peak.frame + 1
                var offsetTime = 0.0
                var hasOffset = false
                for candidate in offsets where candidate.frame > peak.frame {
                    offsetFrame = candidate.frame
                    offsetTime = candidate.time
                    hasOffset = true
                    break
                }
                if offsetFrame > nextFrame {
                    offsetFrame = nextFrame
                    offsetTime = nextTime
                }

                // Where the multi-pitch head says the note stops sounding.
                var mpeFrame = peak.frame + 1
                var mpeTime = 0.0
                var hasMPE = false
                var scan = peak.frame + 1
                while scan < nextFrame && scan < mpe.count {
                    if mpe[scan][note] < options.multiPitchThreshold {
                        mpeFrame = scan
                        mpeTime = Double(scan) * hop
                        hasMPE = true
                        break
                    }
                    scan += 1
                }

                let end: Double
                switch (hasOffset, hasMPE) {
                case (false, false): end = nextTime
                case (true, false): end = offsetTime
                case (false, true): end = mpeTime
                case (true, true):
                    switch options.offsetRule {
                    case .offset: end = offsetTime
                    case .longer: end = offsetFrame >= mpeFrame ? offsetTime : mpeTime
                    case .shorter: end = offsetFrame <= mpeFrame ? offsetTime : mpeTime
                    }
                }

                let level = peak.frame < velocity.count ? velocity[peak.frame][note] : 0
                if options.dropsSilentNotes && level <= 0 { continue }

                perNote.append(NFKMIDINote(pitch: note + configuration.lowestPitch,
                                           startSeconds: peak.time, endSeconds: end,
                                           velocity: level, program: options.program,
                                           percussion: false, pitchBend: nil))
                // A repeated note that starts before the previous one ended truncates it.
                if perNote.count > 1 {
                    let last = perNote[perNote.count - 1]
                    let previous = perNote[perNote.count - 2]
                    if last.startSeconds < previous.endSeconds {
                        perNote[perNote.count - 2] = NFKMIDINote(
                            pitch: previous.pitch, startSeconds: previous.startSeconds,
                            endSeconds: last.startSeconds, velocity: previous.velocity,
                            program: previous.program, percussion: previous.isPercussion,
                            pitchBend: previous.pitchBend)
                    }
                }
            }
            found.append(contentsOf: perNote)
        }
        return found.sorted { ($0.startSeconds, Double($0.pitch)) < ($1.startSeconds, Double($1.pitch)) }
    }
}

/// Holds the network for capture in the backend's `@Sendable` closure.
private final class NFKHFTHolder: @unchecked Sendable {
    let net: NFKMLXHFTTransformerNet
    init(_ net: NFKMLXHFTTransformerNet) { self.net = net }
}

/// hFT-Transformer as an InferKit backend. Reads `NFKInputAudio`; returns the transcription as an
/// `NFKMIDISequence` under `NFKOutputMIDI`.
@objc(NFKMLXHFTTransformerBackend)
public final class NFKMLXHFTTransformerBackend: NSObject, NFKInferenceBackend {

    private let holder: NFKHFTHolder
    private let identifier: String

    init(net: NFKMLXHFTTransformerNet, identifier: String) {
        holder = NFKHFTHolder(net)
        self.identifier = identifier
        super.init()
    }

    @objc public var isReady: Bool { true }
    @objc public var backendIdentifier: String { identifier }
    @objc public var supportedInputKeys: Set<String> { [NFKInputAudio] }
    @objc public var supportedParameterKeys: Set<String> {
        [NFKMLXTranscriptionParameterKey.onsetThreshold,
         NFKMLXTranscriptionParameterKey.frameThreshold,
         NFKMLXTranscriptionParameterKey.tempo,
         NFKMLXTranscriptionParameterKey.program]
    }

    @objc(runInferenceForRequest:error:)
    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        guard let value = request.input(forKey: NFKInputAudio) else { throw NFKMLXError.unsupportedInput }
        var clip: (samples: [Float], sampleRate: Int)?
        if let asset = value as? NFKAudioAsset, let url = asset.fileURL, let data = try? Data(contentsOf: url) {
            clip = NFKMLXWaveFile.read(data)
        } else if let data = value as? Data {
            clip = NFKMLXWaveFile.read(data)
        }
        guard let clip else { throw NFKMLXError.unsupportedInput }

        var options = NFKMLXHFTTransformerOptions.default
        func number(_ key: String) -> NSNumber? { request.parameter(forKey: key) as? NSNumber }
        if let value = number(NFKMLXTranscriptionParameterKey.onsetThreshold) { options.onsetThreshold = value.floatValue }
        // The frame threshold is the multi-pitch head's, which is what decides where a note stops.
        if let value = number(NFKMLXTranscriptionParameterKey.frameThreshold) { options.multiPitchThreshold = value.floatValue }
        if let value = number(NFKMLXTranscriptionParameterKey.tempo) { options.tempoBPM = value.doubleValue }
        if let value = number(NFKMLXTranscriptionParameterKey.program) { options.program = value.intValue }

        let sequence = holder.net.transcribe(clip.samples, sampleRate: clip.sampleRate, options: options)
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
}

/// Registration and weight loading for hFT-Transformer.
@objc(NFKMLXHFTTransformer)
public final class NFKMLXHFTTransformer: NSObject {

    /// The registry name the model builds under.
    @objc public static let modelName = "hft-transformer"

    public static func makeNet(_ configuration: NFKMLXHFTTransformerConfiguration = .maestro) -> NFKMLXHFTTransformerNet {
        NFKMLXHFTTransformerNet(configuration)
    }

    /// Builds an hFT-Transformer backend from optional local weights. A nil `weightsURL` builds
    /// random weights. Run inference off the render thread.
    ///
    /// - Since: InferKit 0.4.0
    @objc(backendWithWeightsURL:error:)
    public static func backend(weightsURL: URL?) throws -> any NFKInferenceBackend {
        let net = makeNet()
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        return NFKMLXHFTTransformerBackend(net: net, identifier: modelName)
    }

    /// Downloads the checkpoint from Hugging Face, then builds. Blocking on the network; run off the
    /// render thread.
    @objc(backendWithRepo:weightsPath:revision:cacheDirectoryURL:error:)
    public static func backend(repo: String, weightsPath: String, revision: String?, cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        let url = try NFKMLXDownload.weightsURL(repo: repo, weightsPath: weightsPath, revision: revision,
                                                cacheDirectoryURL: cacheDirectoryURL)
        return try backend(weightsURL: url)
    }

    /// The asynchronous form of the download factory.
    @objc(backendWithRepo:weightsPath:revision:cacheDirectoryURL:completionHandler:)
    public static func backend(repo: String, weightsPath: String, revision: String?, cacheDirectoryURL: URL?,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXDownload.backend(repo: repo, weightsPath: weightsPath, revision: revision,
                               cacheDirectoryURL: cacheDirectoryURL,
                               build: { try backend(weightsURL: $0) },
                               completionHandler: completionHandler)
    }

    /// Registers hFT-Transformer (`hft-transformer`) with `NFKMLXModelRegistry`.
    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName) { weightsURL in try backend(weightsURL: weightsURL) }
    }

    /// Loads a converted checkpoint, transposing the stem's convolution into MLX's layout.
    ///
    /// The converter already shortens the reference's `encoder_spec2midi.*` / `decoder_spec2midi.*`
    /// names, so nothing else is remapped here.
    public static func loadWeights(into net: NFKMLXHFTTransformerNet, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        var mapped = [(String, MLXArray)]()
        for (key, value) in checkpoint.arrays {
            mapped.append((key, value.ndim == 4 ? value.transposed(0, 2, 3, 1) : value))
        }
        try NFKMLXWeights.apply(mapped, to: net)
    }
}
