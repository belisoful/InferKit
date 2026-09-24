//
//  NFKMLXMuScriptor.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXNN
import MLXFast

// MuScriptor transcribes a mixture into MIDI by generating tokens: the mel spectrogram of five
// seconds of audio is projected into the transformer's width and prepended to the sequence, and a
// causal decoder writes out an MT3 event stream (time shifts, programs, pitches, velocities, drums)
// that `NFKMLXMuScriptorEvents` turns back into notes. One track per instrument comes out of the
// program events rather than from a separate model.
//
// The architecture is audiocraft's language model with a single codebook: a pre-norm causal
// transformer, bias-free projections, a fused `in_proj_weight`, GELU feed-forward, sinusoidal
// absolute positions added at the transformer's entry, and a final LayerNorm before the head.
//
// Three conditioners run: the mel spectrogram, an instrument-presence class, and a dataset class.
// Each is PREPENDED in turn, so the last one processed lands first — the sequence the transformer
// sees is mel, then dataset, then instrument, then the generated tokens. That order is not the order
// the conditioners are declared in, and getting it wrong changes every position the model reads.
//
// The mel front end's window and filterbank are checkpoint buffers, so they load rather than being
// re-derived here, the way Basic Pitch's constant-Q kernels do.
//
// The weights are CC BY-NC 4.0 behind a gated repository: non-commercial use, and a Hugging Face
// token that has accepted the license. The code is MIT, and the released geometry is in it, so the
// module is built and tested without the weights.

/// MuScriptor's geometry. The three presets are the released repositories.
public struct NFKMLXMuScriptorConfiguration: Sendable {
    public var dimension: Int
    public var heads: Int
    public var layers: Int
    /// The output head's width. The vocabulary is narrower; the rest is masked.
    public var card: Int
    /// The feed-forward's expansion.
    public var hiddenScale: Int
    /// The sinusoidal position embedding's period.
    public var maxPeriod: Double
    public var sampleRate: Int
    public var nFFT: Int
    /// The mel frames per second, which sets the hop to `sampleRate / frameRate`.
    public var frameRate: Int
    public var melBins: Int
    /// The floor added before the log, so silence does not take the log of zero.
    public var logEpsilon: Float
    /// The audio one forward reads.
    public var segmentSeconds: Double
    /// The tokens one chunk may generate before the decode gives up.
    public var maximumTokens: Int
    /// The first token the model masks out of its own head. The vocabulary ends here; the released
    /// heads are wider, and the tokens past it are never sampled.
    public var maskedFromToken: Int
    public var instrumentClasses: Int
    public var datasetClasses: Int

    public init(dimension: Int, heads: Int, layers: Int, card: Int, hiddenScale: Int = 4,
                maxPeriod: Double = 10000, sampleRate: Int = 16000, nFFT: Int = 2048,
                frameRate: Int = 100, melBins: Int = 512, logEpsilon: Float = 1e-6,
                segmentSeconds: Double = 5.0, maximumTokens: Int = 2000,
                maskedFromToken: Int = 1393, instrumentClasses: Int = 1000, datasetClasses: Int = 4) {
        self.dimension = dimension
        self.heads = heads
        self.layers = layers
        self.card = card
        self.hiddenScale = hiddenScale
        self.maxPeriod = maxPeriod
        self.sampleRate = sampleRate
        self.nFFT = nFFT
        self.frameRate = frameRate
        self.melBins = melBins
        self.logEpsilon = logEpsilon
        self.segmentSeconds = segmentSeconds
        self.maximumTokens = maximumTokens
        self.maskedFromToken = maskedFromToken
        self.instrumentClasses = instrumentClasses
        self.datasetClasses = datasetClasses
    }

    /// `MuScriptor/muscriptor-small`, 103M parameters.
    public static let small = NFKMLXMuScriptorConfiguration(dimension: 768, heads: 12, layers: 14, card: 1393)
    /// `MuScriptor/muscriptor-medium`, 307M parameters. The release's own default.
    public static let medium = NFKMLXMuScriptorConfiguration(dimension: 1024, heads: 16, layers: 24, card: 1395)
    /// `MuScriptor/muscriptor-large`, 1.4B parameters.
    public static let large = NFKMLXMuScriptorConfiguration(dimension: 1536, heads: 24, layers: 48, card: 1395)

    /// The samples between mel frames.
    public var hopLength: Int { sampleRate / frameRate }
    /// The samples one chunk reads.
    public var segmentSamples: Int { Int(segmentSeconds * Double(sampleRate)) }
    public var headDimension: Int { dimension / heads }
}

/// The released variants, for an Objective-C caller.
@objc(NFKMLXMuScriptorVariant)
public enum NFKMLXMuScriptorVariant: Int, Sendable {
    case small, medium, large

    var configuration: NFKMLXMuScriptorConfiguration {
        switch self {
        case .small: return .small
        case .medium: return .medium
        case .large: return .large
        }
    }

    var repository: String {
        switch self {
        case .small: return "MuScriptor/muscriptor-small"
        case .medium: return "MuScriptor/muscriptor-medium"
        case .large: return "MuScriptor/muscriptor-large"
        }
    }
}

/// Holds the analysis window under the name the checkpoint gives it.
final class NFKMuScriptorSpectrogramWindow: Module {
    @ParameterInfo(key: "window") var window: MLXArray
    init(nFFT: Int) { _window.wrappedValue = MLXArray.zeros([nFFT]) }
}

/// Holds the mel filterbank under the name the checkpoint gives it.
final class NFKMuScriptorMelScale: Module {
    @ParameterInfo(key: "fb") var filterbank: MLXArray                  // [nFFT / 2 + 1, melBins]
    init(bins: Int, mels: Int) { _filterbank.wrappedValue = MLXArray.zeros([bins, mels]) }
}

/// torchaudio's `MelSpectrogram` at the settings the conditioner asks for: magnitude rather than
/// power, centered frames with reflection padding, an HTK mel scale with no Slaney normalization.
final class NFKMuScriptorMelSpectrogram: Module {
    @ModuleInfo(key: "spectrogram") var spectrogram: NFKMuScriptorSpectrogramWindow
    @ModuleInfo(key: "mel_scale") var melScale: NFKMuScriptorMelScale

    let configuration: NFKMLXMuScriptorConfiguration

    init(_ configuration: NFKMLXMuScriptorConfiguration) {
        self.configuration = configuration
        _spectrogram.wrappedValue = NFKMuScriptorSpectrogramWindow(nFFT: configuration.nFFT)
        _melScale.wrappedValue = NFKMuScriptorMelScale(bins: configuration.nFFT / 2 + 1,
                                                       mels: configuration.melBins)
    }

    /// A mono clip → `[frames, melBins]` magnitudes.
    func callAsFunction(_ samples: [Float]) -> MLXArray {
        let nFFT = configuration.nFFT
        let hop = configuration.hopLength
        let pad = nFFT / 2

        // `center=true` with reflection padding: the first frame is centered on sample 0.
        var padded = [Float](repeating: 0, count: samples.count + 2 * pad)
        for index in 0 ..< pad { padded[index] = samples[min(pad - index, samples.count - 1)] }
        for index in 0 ..< samples.count { padded[pad + index] = samples[index] }
        for index in 0 ..< pad {
            let mirrored = samples.count - 2 - index
            padded[pad + samples.count + index] = samples[max(0, mirrored)]
        }

        let frames = 1 + (padded.count - nFFT) / hop
        var framed = [Float](repeating: 0, count: frames * nFFT)
        for frame in 0 ..< frames {
            let start = frame * hop
            for index in 0 ..< nFFT { framed[frame * nFFT + index] = padded[start + index] }
        }
        let windowed = framed.withUnsafeBufferPointer { MLXArray($0, [frames, nFFT]) } * spectrogram.window
        let spectrum = MLXFFT.rfft(windowed, axis: 1)
        let magnitude = sqrt(spectrum.realPart().square() + spectrum.imaginaryPart().square())
        return magnitude.matmul(melScale.filterbank)                    // [frames, melBins]
    }
}

/// The mel conditioner: a log mel spectrogram projected into the transformer's width.
final class NFKMuScriptorMelConditioner: Module {
    @ModuleInfo(key: "mel_spec_transform") var melSpectrogram: NFKMuScriptorMelSpectrogram
    @ModuleInfo(key: "output_proj") var outputProjection: Linear

    let configuration: NFKMLXMuScriptorConfiguration

    init(_ configuration: NFKMLXMuScriptorConfiguration) {
        self.configuration = configuration
        _melSpectrogram.wrappedValue = NFKMuScriptorMelSpectrogram(configuration)
        _outputProjection.wrappedValue = Linear(configuration.melBins, configuration.dimension)
    }

    /// The frames a clip of `count` samples fills. Centered framing produces one more than this, and
    /// the reference masks that last frame away: its length mask counts `samples / hop` frames, so a
    /// chunk's final mel frame is always zeroed before the transformer sees it.
    func validFrames(sampleCount: Int) -> Int {
        sampleCount / configuration.hopLength
    }

    /// A mono clip → `[1, frames, dimension]`.
    func callAsFunction(_ samples: [Float]) -> MLXArray {
        let mel = melSpectrogram(samples)
        let logMel = log(mel + configuration.logEpsilon)
        let projected = outputProjection(logMel.expandedDimensions(axis: 0))

        let frames = projected.dim(1)
        let valid = min(validFrames(sampleCount: samples.count), frames)
        guard valid < frames else { return projected }
        let mask = MLXArray((0 ..< frames).map { Float($0 < valid ? 1 : 0) }).reshaped([1, frames, 1])
        return projected * mask
    }
}

/// A class conditioner: one embedding per class, with a slot for "unspecified".
///
/// The reference shifts the index twice, once when it tokenizes and once when it embeds, so an
/// unspecified class reads row 1 rather than row 0. Reproduced rather than tidied.
final class NFKMuScriptorClassConditioner: Module {
    @ModuleInfo(key: "embed") var embed: Embedding

    init(classes: Int, dimension: Int) {
        _embed.wrappedValue = Embedding(embeddingCount: classes + 1, dimensions: dimension)
    }

    /// `nil` for unspecified. Returns `[1, 1, dimension]`.
    func callAsFunction(_ value: Int?) -> MLXArray {
        let row = (value ?? -1) + 2
        return embed(MLXArray([Int32(row)])).expandedDimensions(axis: 0)
    }
}

/// The three conditioners, under the names the checkpoint gives them.
final class NFKMuScriptorConditioners: Module {
    @ModuleInfo(key: "self_wav") var selfWav: NFKMuScriptorMelConditioner
    @ModuleInfo(key: "instrument_group") var instrumentGroup: NFKMuScriptorClassConditioner
    @ModuleInfo(key: "dataset_name") var datasetName: NFKMuScriptorClassConditioner

    init(_ configuration: NFKMLXMuScriptorConfiguration) {
        _selfWav.wrappedValue = NFKMuScriptorMelConditioner(configuration)
        _instrumentGroup.wrappedValue = NFKMuScriptorClassConditioner(classes: configuration.instrumentClasses,
                                                                      dimension: configuration.dimension)
        _datasetName.wrappedValue = NFKMuScriptorClassConditioner(classes: configuration.datasetClasses,
                                                                  dimension: configuration.dimension)
    }
}

final class NFKMuScriptorConditionProvider: Module {
    @ModuleInfo(key: "conditioners") var conditioners: NFKMuScriptorConditioners
    init(_ configuration: NFKMLXMuScriptorConfiguration) {
        _conditioners.wrappedValue = NFKMuScriptorConditioners(configuration)
    }
}

/// The keys and values one layer has seen, grown as the decode advances.
final class NFKMuScriptorLayerCache {
    var keys: MLXArray?
    var values: MLXArray?

    func append(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        let keys = self.keys.map { concatenated([$0, newKeys], axis: 2) } ?? newKeys
        let values = self.values.map { concatenated([$0, newValues], axis: 2) } ?? newValues
        self.keys = keys
        self.values = values
        return (keys, values)
    }
}

/// Causal self-attention with one fused projection for query, key, and value, and no biases.
final class NFKMuScriptorAttention: Module {
    @ParameterInfo(key: "in_proj_weight") var inProjection: MLXArray    // [3 · dim, dim]
    @ModuleInfo(key: "out_proj") var outProjection: Linear

    let heads: Int
    let headDimension: Int

    init(dimension: Int, heads: Int) {
        self.heads = heads
        headDimension = dimension / heads
        _inProjection.wrappedValue = MLXArray.zeros([3 * dimension, dimension])
        _outProjection.wrappedValue = Linear(dimension, dimension, bias: false)
    }

    /// `[B, T, C]` → `[B, T, C]`. `cache` carries the keys and values of everything already read.
    func callAsFunction(_ x: MLXArray, cache: NFKMuScriptorLayerCache?, mask: MLXArray?) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))
        let projected = x.matmul(inProjection.transposed(1, 0))         // [B, T, 3C]
        // The reference packs the projection as (part, head, channel), so the parts split first.
        let parts = projected.reshaped([batch, length, 3, heads, headDimension])
        let q = parts[0..., 0..., 0].transposed(0, 2, 1, 3)             // [B, heads, T, headDim]
        var k = parts[0..., 0..., 1].transposed(0, 2, 1, 3)
        var v = parts[0..., 0..., 2].transposed(0, 2, 1, 3)

        if let cache {
            (k, v) = cache.append(keys: k, values: v)
        }
        let attended = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: 1.0 / sqrt(Float(headDimension)),
            mask: mask.map { $0.asType(q.dtype) })
        return outProjection(attended.transposed(0, 2, 1, 3).reshaped([batch, length, heads * headDimension]))
    }
}

/// A pre-norm block: attention, then a GELU feed-forward, both bias-free.
final class NFKMuScriptorLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: NFKMuScriptorAttention
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "linear2") var linear2: Linear

    init(_ configuration: NFKMLXMuScriptorConfiguration) {
        let dimension = configuration.dimension
        let hidden = configuration.hiddenScale * dimension
        _attention.wrappedValue = NFKMuScriptorAttention(dimension: dimension, heads: configuration.heads)
        _norm1.wrappedValue = LayerNorm(dimensions: dimension, eps: 1e-5)
        _norm2.wrappedValue = LayerNorm(dimensions: dimension, eps: 1e-5)
        _linear1.wrappedValue = Linear(dimension, hidden, bias: false)
        _linear2.wrappedValue = Linear(hidden, dimension, bias: false)
    }

    func callAsFunction(_ x: MLXArray, cache: NFKMuScriptorLayerCache?, mask: MLXArray?) -> MLXArray {
        var x = x + attention(norm1(x), cache: cache, mask: mask)
        x = x + linear2(gelu(linear1(norm2(x))))
        return x
    }
}

/// The stack, with sinusoidal absolute positions added at its entry.
final class NFKMuScriptorTransformer: Module {
    @ModuleInfo(key: "layers") var layers: [NFKMuScriptorLayer]

    let configuration: NFKMLXMuScriptorConfiguration

    init(_ configuration: NFKMLXMuScriptorConfiguration) {
        self.configuration = configuration
        _layers.wrappedValue = (0 ..< configuration.layers).map { _ in NFKMuScriptorLayer(configuration) }
    }

    /// The position embedding of `count` positions starting at `offset`.
    ///
    /// The cosine half comes first, and the exponent divides by `halfDimension − 1` rather than by
    /// the half dimension, which is the reference's own spelling.
    func positionEmbedding(count: Int, offset: Int) -> MLXArray {
        let dimension = configuration.dimension
        let half = dimension / 2
        var values = [Float](repeating: 0, count: count * dimension)
        for index in 0 ..< count {
            let position = Double(offset + index)
            for channel in 0 ..< half {
                let phase = position / pow(configuration.maxPeriod, Double(channel) / Double(half - 1))
                values[index * dimension + channel] = Float(cos(phase))
                values[index * dimension + half + channel] = Float(sin(phase))
            }
        }
        return values.withUnsafeBufferPointer { MLXArray($0, [1, count, dimension]) }
    }

    func callAsFunction(_ x: MLXArray, offset: Int, caches: [NFKMuScriptorLayerCache]?,
                        mask: MLXArray?) -> MLXArray {
        var h = x + positionEmbedding(count: x.dim(1), offset: offset).asType(x.dtype)
        for (index, layer) in layers.enumerated() {
            h = layer(h, cache: caches?[index], mask: mask)
        }
        return h
    }
}

/// MuScriptor's network.
public final class NFKMLXMuScriptorNet: Module {
    @ModuleInfo(key: "emb") var embedding: Embedding
    @ModuleInfo(key: "transformer") var transformer: NFKMuScriptorTransformer
    @ModuleInfo(key: "out_norm") var outNorm: LayerNorm
    @ModuleInfo(key: "linear") var head: Linear
    @ModuleInfo(key: "condition_provider") var conditionProvider: NFKMuScriptorConditionProvider

    public let configuration: NFKMLXMuScriptorConfiguration
    public let vocabulary: NFKMuScriptorVocabulary

    public init(_ configuration: NFKMLXMuScriptorConfiguration = .medium) {
        self.configuration = configuration
        vocabulary = NFKMuScriptorVocabulary()
        _embedding.wrappedValue = Embedding(embeddingCount: configuration.card + 1,
                                            dimensions: configuration.dimension)
        _transformer.wrappedValue = NFKMuScriptorTransformer(configuration)
        _outNorm.wrappedValue = LayerNorm(dimensions: configuration.dimension, eps: 1e-5)
        _head.wrappedValue = Linear(configuration.dimension, configuration.card, bias: false)
        _conditionProvider.wrappedValue = NFKMuScriptorConditionProvider(configuration)
    }

    /// The token the sequence opens with, which is one past the vocabulary the head scores.
    public var initialToken: Int { configuration.card }

    /// The prefix one chunk conditions on: the mel, then the dataset class, then the instrument class.
    ///
    /// The reference prepends each condition in turn, so the order here is the reverse of the order
    /// they are declared in.
    public func conditioningPrefix(samples: [Float], instrument: Int? = nil,
                                   dataset: Int? = nil) -> MLXArray {
        let conditioners = conditionProvider.conditioners
        return concatenated([conditioners.selfWav(samples),
                             conditioners.datasetName(dataset),
                             conditioners.instrumentGroup(instrument)], axis: 1)
    }

    /// The logits of the last position, given a prefix and the tokens so far.
    func logits(prefix: MLXArray, tokens: [Int], caches: [NFKMuScriptorLayerCache]) -> MLXArray {
        let embedded = embedding(MLXArray(tokens.map { Int32($0) })).expandedDimensions(axis: 0)
        let input = concatenated([prefix, embedded], axis: 1)
        let mask = NFKMLXLanguageNet.causalMask(input.dim(1), offset: 0)
        let hidden = outNorm(transformer(input, offset: 0, caches: caches, mask: mask))
        return head(hidden[0..., -1, 0...])
    }

    /// One decode step against the caches the prefill filled.
    func step(token: Int, offset: Int, caches: [NFKMuScriptorLayerCache]) -> MLXArray {
        let embedded = embedding(MLXArray([Int32(token)])).expandedDimensions(axis: 0)
        let hidden = outNorm(transformer(embedded, offset: offset, caches: caches, mask: nil))
        return head(hidden[0..., -1, 0...])
    }

    /// Masks the head down to the vocabulary the tokenizer defines, plus anything the caller forbids.
    func masked(_ logits: MLXArray, forbidden: [Int]) -> MLXArray {
        var scores = logits
        if configuration.maskedFromToken < configuration.card {
            let keep = MLXArray((0 ..< configuration.card).map {
                Float($0 < configuration.maskedFromToken ? 0 : -Float.infinity)
            })
            scores = scores + keep
        }
        for token in forbidden where token >= 0 && token < configuration.card {
            scores[0..., token] = MLXArray(-Float.infinity)
        }
        return scores
    }

    /// Transcribes one five-second chunk, greedily.
    public func generate(chunk samples: [Float], instrument: Int? = nil, dataset: Int? = nil,
                         prelude: [Int] = [], forbidden: [Int] = []) -> [Int] {
        let caches = (0 ..< configuration.layers).map { _ in NFKMuScriptorLayerCache() }
        let prefix = conditioningPrefix(samples: samples, instrument: instrument, dataset: dataset)

        // The sequence opens with the initial token, then whatever tie section the previous chunk
        // left open is teacher-forced before the model writes anything of its own.
        let forced = [initialToken] + prelude
        var tokens = prelude

        var scores = masked(logits(prefix: prefix, tokens: forced, caches: caches), forbidden: forbidden)
        var offset = prefix.dim(1) + forced.count

        while tokens.count < configuration.maximumTokens {
            let next = argMax(scores, axis: -1).item(Int.self)
            if next == vocabulary.endOfSequence { break }
            tokens.append(next)
            scores = masked(step(token: next, offset: offset, caches: caches), forbidden: forbidden)
            offset += 1
        }
        return tokens
    }

    /// Transcribes a clip chunk by chunk, forcing each chunk's tie section from what the previous one
    /// left sounding.
    public func transcribe(_ samples: [Float], sampleRate: Int,
                           instrument: Int? = nil, dataset: Int? = nil) -> NFKMIDISequence {
        let matched = NFKMLXAudioRate.matched(samples, from: sampleRate, to: configuration.sampleRate)
        let segment = configuration.segmentSamples
        let chunkCount = max(1, (matched.count + segment - 1) / segment)

        let tracker = NFKMuScriptorNoteTracker(vocabulary: vocabulary, frameRate: configuration.frameRate)
        var chunks = [(boundary: NFKMuScriptorChunkBoundary, tokens: [Int])]()

        for index in 0 ..< chunkCount {
            let start = index * segment
            let clip = Array(matched[start ..< min(start + segment, matched.count)])
            let seek = Double(index) * configuration.segmentSeconds
            let next = index + 1 < chunkCount ? Double(index + 1) * configuration.segmentSeconds : nil
            let boundary = NFKMuScriptorChunkBoundary(seekSeconds: seek, nextSeekSeconds: next)

            let prelude = index == 0 ? [] : vocabulary.tieSectionTokens(openNotes: tracker.openNotes)
            let tokens = generate(chunk: clip, instrument: instrument, dataset: dataset, prelude: prelude)

            // Replay the chunk through the tracker so the next one's tie section is what is actually
            // sounding. The same machine decodes the notes below.
            _ = tracker.feed(boundary: boundary)
            for token in tokens where token != vocabulary.endOfSequence {
                _ = tracker.feed(token: token)
            }
            chunks.append((boundary, tokens))
        }

        let notes = NFKMuScriptorNotes.notes(chunks: chunks, vocabulary: vocabulary,
                                             frameRate: configuration.frameRate)
        return NFKMIDISequence(notes: notes, tempoBPM: 120, beatsPerBar: 4, beatUnit: 4,
                               ticksPerQuarterNote: 480)
    }
}

/// The request parameters MuScriptor reads beyond the shared transcription keys.
@objc(NFKMLXMuScriptorParameterKey)
public final class NFKMLXMuScriptorParameterKey: NSObject {
    /// Which instrument group the model should expect to hear (NSNumber). Conditioning on it narrows
    /// what the model transcribes; leaving it out lets the model decide.
    @objc public static let instrumentGroup = "NFKMLXParameterInstrumentGroup"
    /// The dataset class the released model was conditioned with during training (NSNumber).
    @objc public static let datasetClass = "NFKMLXParameterDatasetClass"
}

/// Holds the network for capture in the backend's `@Sendable` closure.
private final class NFKMuScriptorHolder: @unchecked Sendable {
    let net: NFKMLXMuScriptorNet
    init(_ net: NFKMLXMuScriptorNet) { self.net = net }
}

/// MuScriptor as an InferKit backend. Reads `NFKInputAudio`; returns the transcription as an
/// `NFKMIDISequence` under `NFKOutputMIDI`.
@objc(NFKMLXMuScriptorBackend)
public final class NFKMLXMuScriptorBackend: NSObject, NFKInferenceBackend {

    private let holder: NFKMuScriptorHolder
    private let identifier: String

    init(net: NFKMLXMuScriptorNet, identifier: String) {
        holder = NFKMuScriptorHolder(net)
        self.identifier = identifier
        super.init()
    }

    @objc public var isReady: Bool { true }
    @objc public var backendIdentifier: String { identifier }
    @objc public var supportedInputKeys: Set<String> { [NFKInputAudio] }
    @objc public var supportedParameterKeys: Set<String> {
        [NFKMLXMuScriptorParameterKey.instrumentGroup, NFKMLXMuScriptorParameterKey.datasetClass]
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

        let instrument = (request.parameter(forKey: NFKMLXMuScriptorParameterKey.instrumentGroup) as? NSNumber)?.intValue
        let sequence = holder.net.transcribe(clip.samples, sampleRate: clip.sampleRate, instrument: instrument)
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

/// Registration and weight loading for MuScriptor.
@objc(NFKMLXMuScriptor)
public final class NFKMLXMuScriptor: NSObject {

    /// The registry name the default variant builds under.
    @objc public static let modelName = "muscriptor"

    public static func makeNet(_ configuration: NFKMLXMuScriptorConfiguration = .medium) -> NFKMLXMuScriptorNet {
        NFKMLXMuScriptorNet(configuration)
    }

    /// Builds a MuScriptor backend from optional local weights. A nil `weightsURL` builds random
    /// weights. Run inference off the render thread.
    ///
    /// - Since: InferKit 0.4.0
    @objc(backendWithVariant:weightsURL:error:)
    public static func backend(variant: NFKMLXMuScriptorVariant, weightsURL: URL?) throws -> any NFKInferenceBackend {
        let net = makeNet(variant.configuration)
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        return NFKMLXMuScriptorBackend(net: net, identifier: registeredName(for: variant))
    }

    /// The medium release, which is the reference's own default.
    @objc(backendWithWeightsURL:error:)
    public static func backend(weightsURL: URL?) throws -> any NFKInferenceBackend {
        try backend(variant: .medium, weightsURL: weightsURL)
    }

    /// Downloads the checkpoint from Hugging Face, then builds. The repository is gated: the account
    /// behind `NFKHFHub.accessToken` (or `HF_TOKEN`) must have accepted the model's license, and the
    /// weights are CC BY-NC 4.0. Blocking on the network; run off the render thread.
    @objc(backendWithVariant:repo:weightsPath:revision:cacheDirectoryURL:error:)
    public static func backend(variant: NFKMLXMuScriptorVariant, repo: String, weightsPath: String,
                               revision: String?, cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        let url = try NFKMLXDownload.weightsURL(repo: repo, weightsPath: weightsPath, revision: revision,
                                                cacheDirectoryURL: cacheDirectoryURL)
        return try backend(variant: variant, weightsURL: url)
    }

    /// The asynchronous form of the download factory.
    @objc(backendWithVariant:repo:weightsPath:revision:cacheDirectoryURL:completionHandler:)
    public static func backend(variant: NFKMLXMuScriptorVariant, repo: String, weightsPath: String,
                               revision: String?, cacheDirectoryURL: URL?,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXDownload.backend(repo: repo, weightsPath: weightsPath, revision: revision,
                               cacheDirectoryURL: cacheDirectoryURL,
                               build: { try backend(variant: variant, weightsURL: $0) },
                               completionHandler: completionHandler)
    }

    /// The registry name of a variant. The medium release is the reference's own default, so it takes
    /// the bare name and a backend built from it reports the same identifier the registry knows it by.
    static func registeredName(for variant: NFKMLXMuScriptorVariant) -> String {
        variant == .medium ? modelName : "\(modelName)-\(variant.name)"
    }

    /// Registers each released variant with `NFKMLXModelRegistry`.
    @objc public static func register() {
        for variant in [NFKMLXMuScriptorVariant.small, .medium, .large] {
            NFKMLXModelRegistry.register(name: registeredName(for: variant)) { weightsURL in
                try backend(variant: variant, weightsURL: weightsURL)
            }
        }
    }

    /// Loads a released checkpoint.
    ///
    /// The released files are already safetensors in the module's own naming, so the only work is the
    /// legacy single-codebook remap the reference also applies: older checkpoints store the embedding
    /// and the head as the first entry of a module list.
    public static func loadWeights(into net: NFKMLXMuScriptorNet, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        var mapped = [(String, MLXArray)]()
        for (key, value) in checkpoint.arrays {
            mapped.append((remapReferenceKey(key), value))
        }
        try NFKMLXWeights.apply(mapped, to: net)
    }

    /// Maps a released key onto the module's names.
    static func remapReferenceKey(_ key: String) -> String {
        if key.hasPrefix("emb.0.") { return "emb." + key.dropFirst("emb.0.".count) }
        if key.hasPrefix("linears.0.") { return "linear." + key.dropFirst("linears.0.".count) }
        return key
    }
}

extension NFKMLXMuScriptorVariant {
    var name: String {
        switch self {
        case .small: return "small"
        case .medium: return "medium"
        case .large: return "large"
        }
    }
}
