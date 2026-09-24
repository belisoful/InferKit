//
//  NFKMLXAllInOne.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXNN

// Music structure analysis divides a track into what a listener hears as its parts. This is
// All-In-One (Kim and Nam, ISMIR 2023): one model that jointly tracks beats and downbeats, finds
// the boundaries between functional sections, and labels each section (intro, verse, chorus, bridge,
// outro, break, solo, instrumental). It reads the four source-separated stems HT Demucs produces, so
// the drums carry the meter and the vocals carry the form rather than both competing in one mixture.
//
// The network is small: a convolutional embedding that collapses 81 frequency bins to one embedding
// per frame, then eleven blocks, each a dilated neighborhood attention across time followed by a
// neighborhood attention across the four instruments. Four linear heads read the last block.
//
// Neighborhood attention is NATTEN's, where each query attends to the kernel positions nearest it at
// the layer's dilation. The neighbor indices depend only on the length, the kernel, and the dilation,
// so they are index tables computed once per shape rather than an attention mask: a three-minute
// track is 18,000 frames, whose full attention matrix would not fit.
//
// Dilation doubles per block (1, 2, 4, ... 1024), and each time layer runs a second attention at
// twice its own dilation, so the eleventh block sees about 40 seconds either side.

/// All-In-One's geometry. The defaults are the released Harmonix models.
public struct NFKMLXAllInOneConfiguration: Sendable {
    /// The rate the front end reads.
    public var sampleRate: Int
    /// The samples between frames. 441 at 44.1 kHz gives exactly 100 frames per second.
    public var hopSize: Int
    /// The transform window.
    public var windowSize: Int
    /// The filterbank's bands per octave.
    public var bandsPerOctave: Int
    /// The filterbank's lowest frequency, in hertz.
    public var minimumFrequency: Double
    /// The filterbank's highest frequency, in hertz.
    public var maximumFrequency: Double
    /// The filterbank's bands, which is what the embedding reads.
    public var inputBins: Int
    /// The embedding width.
    public var embedDimension: Int
    /// The blocks.
    public var depth: Int
    /// The attention heads per layer.
    public var heads: Int
    /// The neighborhood a query attends to.
    public var kernelSize: Int
    /// The factor dilation grows by from block to block.
    public var dilationFactor: Int
    /// The feed-forward expansion.
    public var mlpRatio: Double
    /// The stems the model reads, in the order bass, drums, other, vocals.
    public var instruments: Int
    /// The functional labels the model scores.
    public var labels: [String]
    public var layerNormEpsilon: Float
    /// The dropout after each convolution stage of the embedding, `drop_conv`. Active only in training.
    public var convolutionDropout: Float = 0.2
    /// The dropout on the attention probabilities and the attention output, `drop_attention`.
    public var attentionDropout: Float = 0.2
    /// The dropout on the feed-forward output, `drop_hidden`.
    public var hiddenDropout: Float = 0.2
    /// The largest stochastic-depth rate, `drop_path`, reached at the last block; the rates rise
    /// linearly from 0 at the first.
    public var dropPath: Float = 0.1
    /// The frames per beat at the fastest tempo the model considers, which sets the width of the
    /// local-maximum filter the section post-processing uses.
    public var minimumFramesPerBeat: Int

    public init(sampleRate: Int = 44100, hopSize: Int = 441, windowSize: Int = 2048,
                bandsPerOctave: Int = 12, minimumFrequency: Double = 30, maximumFrequency: Double = 17000,
                inputBins: Int = 81, embedDimension: Int = 24, depth: Int = 11, heads: Int = 2,
                kernelSize: Int = 5, dilationFactor: Int = 2, mlpRatio: Double = 4.0,
                instruments: Int = 4,
                labels: [String] = ["start", "end", "intro", "outro", "break", "bridge", "inst",
                                    "solo", "verse", "chorus"],
                layerNormEpsilon: Float = 1e-5, minimumFramesPerBeat: Int = 24) {
        self.sampleRate = sampleRate
        self.hopSize = hopSize
        self.windowSize = windowSize
        self.bandsPerOctave = bandsPerOctave
        self.minimumFrequency = minimumFrequency
        self.maximumFrequency = maximumFrequency
        self.inputBins = inputBins
        self.embedDimension = embedDimension
        self.depth = depth
        self.heads = heads
        self.kernelSize = kernelSize
        self.dilationFactor = dilationFactor
        self.mlpRatio = mlpRatio
        self.instruments = instruments
        self.labels = labels
        self.layerNormEpsilon = layerNormEpsilon
        self.minimumFramesPerBeat = minimumFramesPerBeat
    }

    /// The released Harmonix models (`harmonix-fold0` … `harmonix-fold7`).
    public static let harmonix = NFKMLXAllInOneConfiguration()

    /// The frames per second the front end produces.
    public var framesPerSecond: Int { sampleRate / hopSize }
    /// The dilation of block `index`.
    func dilation(atBlock index: Int) -> Int {
        Int(pow(Double(dilationFactor), Double(index)))
    }
}

/// The neighborhood a query attends to, as index tables.
///
/// NATTEN centers a query's window on itself and slides the window inward at the edges rather than
/// masking it, so every query reads exactly `kernel` positions. The relative-position bias is
/// re-indexed by the same shift, which is why the two tables are built together.
struct NFKNeighborhoodWindow {
    /// The first neighbor of each position.
    let start: [Int]
    /// The first bias index of each position.
    let biasStart: [Int]
    let kernel: Int
    let dilation: Int

    init(length: Int, kernel: Int, dilation: Int) {
        self.kernel = kernel
        self.dilation = dilation
        let half = kernel / 2
        var start = [Int](repeating: 0, count: length)
        var biasStart = [Int](repeating: 0, count: length)
        for index in 0 ..< length {
            start[index] = Self.windowStart(index, length: length, kernel: kernel, half: half, dilation: dilation)
            biasStart[index] = Self.biasStart(index, length: length, kernel: kernel, half: half, dilation: dilation)
        }
        self.start = start
        self.biasStart = biasStart
    }

    /// `get_window_start` from NATTEN's `natten_cpu_commons.h`.
    private static func windowStart(_ index: Int, length: Int, kernel: Int, half: Int, dilation: Int) -> Int {
        if dilation <= 1 {
            return max(index - half, 0) + (index + half >= length ? (length - index - half - 1) : 0)
        }
        let neighbor = index - half * dilation
        if neighbor < 0 {
            return index % dilation
        }
        if index + half * dilation >= length {
            let remainder = index % dilation
            let whole = (length / dilation) * dilation
            let leftover = length - whole
            if remainder < leftover {
                return length - leftover + remainder - 2 * half * dilation
            }
            return whole + remainder - kernel * dilation
        }
        return neighbor
    }

    /// `get_pb_start` from the same header.
    private static func biasStart(_ index: Int, length: Int, kernel: Int, half: Int, dilation: Int) -> Int {
        if dilation <= 1 {
            return half + (index < half ? (half - index) : 0) + (index + half >= length ? (length - index - 1 - half) : 0)
        }
        if index - half * dilation < 0 {
            return kernel - 1 - (index / dilation)
        }
        if index + half * dilation >= length {
            return (length - index - 1) / dilation
        }
        return half
    }

    /// The positions offset `step` of the neighborhood reads, as an index array.
    func neighbors(at step: Int) -> MLXArray {
        MLXArray(start.map { Int32($0 + step * dilation) })
    }

    /// The bias entries offset `step` reads.
    func biases(at step: Int) -> MLXArray {
        MLXArray(biasStart.map { Int32($0 + step) })
    }
}

/// One neighborhood attention: query, key, value, and the learned relative-position bias.
///
/// The reference computes the whole `[positions, kernel]` score tensor in a fused kernel. Here each
/// of the `kernel` offsets is one gather, which keeps the working set at the size of the input rather
/// than `kernel` times it, and the scores are stacked only after they are all scored.
final class NFKAllInOneAttention: Module {
    @ModuleInfo(key: "query") var query: Linear
    @ModuleInfo(key: "key") var key: Linear
    @ModuleInfo(key: "value") var value: Linear
    /// The relative-position bias, `[heads, 2·kernel − 1]` in one dimension and
    /// `[heads, 2·kernel − 1, 2·kernel − 1]` in two.
    @ParameterInfo(key: "rpb") var rpb: MLXArray

    let heads: Int
    let headDimension: Int
    let kernelSize: Int
    let dilation: Int
    let twoDimensional: Bool
    let probabilityDropout: Dropout

    init(dimension: Int, heads: Int, kernelSize: Int, dilation: Int, twoDimensional: Bool, bias: Bool = true,
         dropout: Float = 0) {
        probabilityDropout = Dropout(p: dropout)
        self.heads = heads
        self.headDimension = dimension / heads
        self.kernelSize = kernelSize
        self.dilation = dilation
        self.twoDimensional = twoDimensional
        _query.wrappedValue = Linear(dimension, dimension, bias: bias)
        _key.wrappedValue = Linear(dimension, dimension, bias: bias)
        _value.wrappedValue = Linear(dimension, dimension, bias: bias)
        let span = 2 * kernelSize - 1
        _rpb.wrappedValue = twoDimensional ? MLXArray.zeros([heads, span, span]) : MLXArray.zeros([heads, span])
    }

    /// `[B, T, C]` → `[B, T, C]`, attending along T.
    func callAsFunction(_ hidden: MLXArray) -> MLXArray {
        let (batch, length) = (hidden.dim(0), hidden.dim(1))
        let scale = 1.0 / sqrt(Float(headDimension))
        // [B, heads, T, headDim]
        func heads(_ array: MLXArray) -> MLXArray {
            array.reshaped([batch, length, self.heads, headDimension]).transposed(0, 2, 1, 3)
        }
        let q = heads(query(hidden)) * scale
        let k = heads(key(hidden))
        let v = heads(value(hidden))

        let window = NFKNeighborhoodWindow(length: length, kernel: kernelSize, dilation: dilation)
        var scores = [MLXArray]()
        for step in 0 ..< kernelSize {
            let gathered = take(k, window.neighbors(at: step), axis: 2)          // [B, heads, T, headDim]
            let bias = take(rpb, window.biases(at: step), axis: 1)               // [heads, T]
            scores.append(((q * gathered).sum(axis: -1) + bias).expandedDimensions(axis: -1))
        }
        let probabilities = probabilityDropout(softmax(concatenated(scores, axis: -1), axis: -1))    // [B, heads, T, kernel]

        var context = MLXArray.zeros(like: v)
        for step in 0 ..< kernelSize {
            let gathered = take(v, window.neighbors(at: step), axis: 2)
            context = context + probabilities[0..., 0..., 0..., step].expandedDimensions(axis: -1) * gathered
        }
        return context.transposed(0, 2, 1, 3).reshaped([batch, length, self.heads * headDimension])
    }

    /// `[B, rows, columns, C]` → the same shape, attending over both axes at once.
    func callAsFunction2D(_ hidden: MLXArray) -> MLXArray {
        let (batch, rows, columns) = (hidden.dim(0), hidden.dim(1), hidden.dim(2))
        let scale = 1.0 / sqrt(Float(headDimension))
        func heads(_ array: MLXArray) -> MLXArray {
            // [B, rows, columns, heads, headDim] → [B, heads, rows, columns, headDim]
            array.reshaped([batch, rows, columns, self.heads, headDimension]).transposed(0, 3, 1, 2, 4)
        }
        let q = heads(query(hidden)) * scale
        let k = heads(key(hidden))
        let v = heads(value(hidden))

        let rowWindow = NFKNeighborhoodWindow(length: rows, kernel: kernelSize, dilation: dilation)
        let columnWindow = NFKNeighborhoodWindow(length: columns, kernel: kernelSize, dilation: dilation)

        func gather(_ array: MLXArray, _ rowStep: Int, _ columnStep: Int) -> MLXArray {
            let byRow = take(array, rowWindow.neighbors(at: rowStep), axis: 2)
            return take(byRow, columnWindow.neighbors(at: columnStep), axis: 3)
        }
        func bias(_ rowStep: Int, _ columnStep: Int) -> MLXArray {
            let byRow = take(rpb, rowWindow.biases(at: rowStep), axis: 1)         // [heads, rows, span]
            return take(byRow, columnWindow.biases(at: columnStep), axis: 2)      // [heads, rows, columns]
        }

        var scores = [MLXArray]()
        for rowStep in 0 ..< kernelSize {
            for columnStep in 0 ..< kernelSize {
                let gathered = gather(k, rowStep, columnStep)
                scores.append(((q * gathered).sum(axis: -1) + bias(rowStep, columnStep)).expandedDimensions(axis: -1))
            }
        }
        let probabilities = probabilityDropout(softmax(concatenated(scores, axis: -1), axis: -1))     // [B, heads, rows, columns, kernel²]

        var context = MLXArray.zeros(like: v)
        var offset = 0
        for rowStep in 0 ..< kernelSize {
            for columnStep in 0 ..< kernelSize {
                let gathered = gather(v, rowStep, columnStep)
                context = context + probabilities[0..., 0..., 0..., 0..., offset].expandedDimensions(axis: -1) * gathered
                offset += 1
            }
        }
        return context.transposed(0, 2, 3, 1, 4).reshaped([batch, rows, columns, self.heads * headDimension])
    }
}

/// A `Linear` behind the name the reference gives it, so the checkpoint's keys land with no remap.
final class NFKAllInOneDense: Module {
    @ModuleInfo(key: "dense") var dense: Linear
    init(_ inputDimension: Int, _ outputDimension: Int) {
        _dense.wrappedValue = Linear(inputDimension, outputDimension)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { dense(x) }
}

/// The attention and its output projection, under the reference's `attention.self` / `attention.output`.
final class NFKAllInOneAttentionModule: Module {
    @ModuleInfo(key: "self") var attention: NFKAllInOneAttention
    @ModuleInfo(key: "output") var output: NFKAllInOneDense
    let outputDropout: Dropout

    init(dimension: Int, heads: Int, kernelSize: Int, dilation: Int, twoDimensional: Bool, dropout: Float = 0) {
        _attention.wrappedValue = NFKAllInOneAttention(dimension: dimension, heads: heads,
                                                       kernelSize: kernelSize, dilation: dilation,
                                                       twoDimensional: twoDimensional, dropout: dropout)
        _output.wrappedValue = NFKAllInOneDense(dimension, dimension)
        outputDropout = Dropout(p: dropout)
    }

    func callAsFunction(_ hidden: MLXArray) -> MLXArray {
        outputDropout(output(attention(hidden)))
    }

    func callAsFunction2D(_ hidden: MLXArray) -> MLXArray {
        outputDropout(output(attention.callAsFunction2D(hidden)))
    }
}

/// Stochastic depth (`DinatDropPath`): in training, each sample's residual branch is dropped whole
/// with probability `rate` and the kept ones scaled by `1 / (1 − rate)`. The identity at inference.
final class NFKAllInOneDropPath: Module {
    let rate: Float

    init(rate: Float) {
        self.rate = rate
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        guard training, rate > 0 else {
            return x
        }
        let keep = 1 - rate
        var shape = [x.dim(0)]
        shape += [Int](repeating: 1, count: x.ndim - 1)
        return x * MLXRandom.bernoulli(MLXArray(keep), shape).asType(x.dtype) / keep
    }
}

/// One transformer layer: neighborhood attention (twice, at two dilations, in the time layers), then
/// a feed-forward.
///
/// The doubled attention is not a residual stack. The two attention outputs are concatenated for the
/// feed-forward, which therefore reads twice the width, while the residual takes their mean. The
/// asymmetry is load-bearing, and it is why `layernorm_after` is `2·dim` wide.
final class NFKAllInOneLayer: Module {
    @ModuleInfo(key: "layernorm_before") var normBefore: LayerNorm
    @ModuleInfo(key: "layernorm_after") var normAfter: LayerNorm
    @ModuleInfo(key: "attention") var attention: NFKAllInOneAttentionModule
    @ModuleInfo(key: "attention2") var attention2: NFKAllInOneAttentionModule?
    @ModuleInfo(key: "intermediate") var intermediate: NFKAllInOneDense
    @ModuleInfo(key: "output") var output: NFKAllInOneDense

    let twoDimensional: Bool
    let doubleAttention: Bool
    /// The positions a layer needs before its neighborhood fits; a shorter input is padded to it.
    let windowSize: Int
    let hiddenDropout: Dropout
    let dropPath: NFKAllInOneDropPath

    init(dimension: Int, heads: Int, kernelSize: Int, dilation: Int, mlpRatio: Double,
         doubleAttention: Bool, twoDimensional: Bool, epsilon: Float,
         attentionDropout: Float = 0, hiddenDropout: Float = 0, dropPath: Float = 0) {
        self.hiddenDropout = Dropout(p: hiddenDropout)
        self.dropPath = NFKAllInOneDropPath(rate: dropPath)
        self.twoDimensional = twoDimensional
        self.doubleAttention = doubleAttention
        windowSize = kernelSize * dilation * (doubleAttention ? 2 : 1)
        let after = doubleAttention ? dimension * 2 : dimension
        _normBefore.wrappedValue = LayerNorm(dimensions: dimension, eps: epsilon)
        _normAfter.wrappedValue = LayerNorm(dimensions: after, eps: epsilon)
        _attention.wrappedValue = NFKAllInOneAttentionModule(dimension: dimension, heads: heads,
                                                             kernelSize: kernelSize, dilation: dilation,
                                                             twoDimensional: twoDimensional,
                                                             dropout: attentionDropout)
        _attention2.wrappedValue = doubleAttention
            ? NFKAllInOneAttentionModule(dimension: dimension, heads: heads, kernelSize: kernelSize,
                                         dilation: dilation * 2, twoDimensional: twoDimensional,
                                         dropout: attentionDropout)
            : nil
        _intermediate.wrappedValue = NFKAllInOneDense(after, Int(Double(after) * mlpRatio))
        _output.wrappedValue = NFKAllInOneDense(Int(Double(after) * mlpRatio), dimension)
    }

    func callAsFunction(_ hidden: MLXArray) -> MLXArray {
        let shortcut = hidden
        var normalized = normBefore(hidden)

        // The neighborhood needs `kernel · dilation` positions on every axis it spans. A shorter axis
        // is padded at the end and cropped back afterwards, which is the only path the
        // four-instrument axis ever takes.
        let rows = normalized.dim(1)
        let columns = twoDimensional ? normalized.dim(2) : 0
        let rowPad = max(0, windowSize - rows)
        let columnPad = twoDimensional ? max(0, windowSize - columns) : 0
        if rowPad > 0 || columnPad > 0 {
            var widths = [IntOrPair](repeating: IntOrPair((0, 0)), count: normalized.ndim)
            widths[1] = IntOrPair((0, rowPad))
            if twoDimensional { widths[2] = IntOrPair((0, columnPad)) }
            normalized = MLX.padded(normalized, widths: widths, mode: .constant)
        }

        var outputs = [MLXArray]()
        for module in [attention, attention2].compactMap({ $0 }) {
            var attended = twoDimensional ? module.callAsFunction2D(normalized) : module(normalized)
            if rowPad > 0 || columnPad > 0 {
                attended = twoDimensional ? attended[0..., 0 ..< rows, 0 ..< columns, 0...]
                                          : attended[0..., 0 ..< rows, 0...]
            }
            outputs.append(shortcut + dropPath(attended))
        }

        let hiddenStates: MLXArray
        let residual: MLXArray
        if doubleAttention, outputs.count == 2 {
            hiddenStates = concatenated(outputs, axis: -1)
            residual = (outputs[0] + outputs[1]) / 2
        } else {
            hiddenStates = outputs[0]
            residual = outputs[0]
        }
        return residual + dropPath(hiddenDropout(output(gelu(intermediate(normAfter(hiddenStates))))))
    }
}

/// One block: attention across time, then attention across the instruments.
final class NFKAllInOneBlock: Module {
    @ModuleInfo(key: "timelayer") var timeLayer: NFKAllInOneLayer
    @ModuleInfo(key: "instlayer") var instrumentLayer: NFKAllInOneLayer

    let instruments: Int

    init(_ configuration: NFKMLXAllInOneConfiguration, dilation: Int, dropPath: Float = 0) {
        instruments = configuration.instruments
        _timeLayer.wrappedValue = NFKAllInOneLayer(dimension: configuration.embedDimension,
                                                   heads: configuration.heads,
                                                   kernelSize: configuration.kernelSize,
                                                   dilation: dilation, mlpRatio: configuration.mlpRatio,
                                                   doubleAttention: true, twoDimensional: false,
                                                   epsilon: configuration.layerNormEpsilon,
                                                   attentionDropout: configuration.attentionDropout,
                                                   hiddenDropout: configuration.hiddenDropout, dropPath: dropPath)
        _instrumentLayer.wrappedValue = NFKAllInOneLayer(dimension: configuration.embedDimension,
                                                         heads: configuration.heads,
                                                         kernelSize: configuration.kernelSize,
                                                         dilation: 1, mlpRatio: configuration.mlpRatio,
                                                         doubleAttention: false, twoDimensional: true,
                                                         epsilon: configuration.layerNormEpsilon,
                                                         attentionDropout: configuration.attentionDropout,
                                                         hiddenDropout: configuration.hiddenDropout,
                                                         dropPath: dropPath)
    }

    /// `[B·instruments, T, C]` → the same shape.
    func callAsFunction(_ hidden: MLXArray) -> MLXArray {
        let (rows, frames, channels) = (hidden.dim(0), hidden.dim(1), hidden.dim(2))
        var states = timeLayer(hidden)
        states = states.reshaped([rows / instruments, instruments, frames, channels])
        states = instrumentLayer(states)
        return states.reshaped([rows, frames, channels])
    }
}

/// The convolutional front end: three convolution and max-pool stages that collapse the 81 filterbank
/// bands to one embedding per frame.
final class NFKAllInOneEmbeddings: Module {
    @ModuleInfo(key: "conv0") var conv0: Conv2d
    @ModuleInfo(key: "conv1") var conv1: Conv2d
    @ModuleInfo(key: "conv2") var conv2: Conv2d
    @ModuleInfo(key: "norm") var norm: LayerNorm

    private let pool = MaxPool2d(kernelSize: [1, 3], stride: [1, 3])
    let drop0: Dropout
    let drop1: Dropout
    let dropout: Dropout

    init(_ configuration: NFKMLXAllInOneConfiguration) {
        drop0 = Dropout(p: configuration.convolutionDropout)
        drop1 = Dropout(p: configuration.convolutionDropout)
        dropout = Dropout(p: configuration.convolutionDropout)
        let width = configuration.embedDimension
        // The convolutions pad in time and not in frequency, so each stage narrows the band axis.
        _conv0.wrappedValue = Conv2d(inputChannels: 1, outputChannels: width / 2,
                                     kernelSize: [3, 3], stride: 1, padding: [1, 0])
        _conv1.wrappedValue = Conv2d(inputChannels: width / 2, outputChannels: width,
                                     kernelSize: [1, 12], stride: 1, padding: 0)
        _conv2.wrappedValue = Conv2d(inputChannels: width, outputChannels: width,
                                     kernelSize: [3, 3], stride: 1, padding: [1, 0])
        _norm.wrappedValue = LayerNorm(dimensions: width, eps: configuration.layerNormEpsilon)
    }

    /// `[B·instruments, T, bands, 1]` → `[B·instruments, T, C]`.
    func callAsFunction(_ spectrogram: MLXArray) -> MLXArray {
        var x = drop0(elu(pool(conv0(spectrogram))))
        x = drop1(elu(pool(conv1(x))))
        x = elu(pool(conv2(x)))
        return dropout(norm(x.squeezed(axis: 2)))
    }
}

/// One head: the instruments' embeddings joined and scored per frame.
final class NFKAllInOneHead: Module {
    @ModuleInfo(key: "classifier") var classifier: Linear

    init(inputDimension: Int, classes: Int) {
        _classifier.wrappedValue = Linear(inputDimension, classes)
    }

    /// `[B, instruments, T, C]` → `[B, classes, T]`.
    func callAsFunction(_ hidden: MLXArray) -> MLXArray {
        let (batch, instruments, frames, channels) = (hidden.dim(0), hidden.dim(1), hidden.dim(2), hidden.dim(3))
        let joined = hidden.transposed(0, 2, 1, 3).reshaped([batch, frames, instruments * channels])
        return classifier(joined).transposed(0, 2, 1)
    }
}

/// The post-processing thresholds the checkpoint tunes: the beat threshold, then the downbeat one.
///
/// The beat decoder trims the track to the span that reaches its threshold, so a different fold's
/// value moves where the beats stop. They are held here rather than defaulted in code because the
/// released folds disagree (0.21 and 0.22).
final class NFKAllInOnePostprocess: Module {
    @ParameterInfo(key: "thresholds") var thresholds: MLXArray

    override init() {
        _thresholds.wrappedValue = MLXArray([Float(0.19), Float(0.22)])
    }

    /// The downbeat threshold the bar tracker trims by.
    var downbeatThreshold: Double { Double(thresholds[1].item(Float.self)) }
}

/// The four logits the model scores per frame.
public struct NFKMLXAllInOneLogits {
    /// `[frames]`.
    public var beat: MLXArray
    /// `[frames]`.
    public var downbeat: MLXArray
    /// `[frames]`.
    public var section: MLXArray
    /// `[labels, frames]`.
    public var function: MLXArray
}

/// All-In-One's network.
public final class NFKMLXAllInOneNet: Module {
    @ModuleInfo(key: "frontend") var frontEnd: NFKAllInOneFrontEnd
    @ModuleInfo(key: "postprocess") var postprocess: NFKAllInOnePostprocess
    @ModuleInfo(key: "embeddings") var embeddings: NFKAllInOneEmbeddings
    @ModuleInfo(key: "encoder") var encoder: NFKAllInOneEncoder
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "beat_classifier") var beatHead: NFKAllInOneHead
    @ModuleInfo(key: "downbeat_classifier") var downbeatHead: NFKAllInOneHead
    @ModuleInfo(key: "section_classifier") var sectionHead: NFKAllInOneHead
    @ModuleInfo(key: "function_classifier") var functionHead: NFKAllInOneHead

    public let configuration: NFKMLXAllInOneConfiguration

    public init(_ configuration: NFKMLXAllInOneConfiguration = .harmonix) {
        self.configuration = configuration
        let joined = configuration.instruments * configuration.embedDimension
        _frontEnd.wrappedValue = NFKAllInOneFrontEnd(configuration)
        _postprocess.wrappedValue = NFKAllInOnePostprocess()
        _embeddings.wrappedValue = NFKAllInOneEmbeddings(configuration)
        _encoder.wrappedValue = NFKAllInOneEncoder(configuration)
        _norm.wrappedValue = LayerNorm(dimensions: configuration.embedDimension, eps: configuration.layerNormEpsilon)
        _beatHead.wrappedValue = NFKAllInOneHead(inputDimension: joined, classes: 1)
        _downbeatHead.wrappedValue = NFKAllInOneHead(inputDimension: joined, classes: 1)
        _sectionHead.wrappedValue = NFKAllInOneHead(inputDimension: joined, classes: 1)
        _functionHead.wrappedValue = NFKAllInOneHead(inputDimension: joined, classes: configuration.labels.count)
        super.init()
        // A module starts in training mode, which would run the dropouts at inference; the trainer
        // switches them on for a run and restores this.
        train(false)
    }

    /// `[B, instruments, T, bands]` → the four sets of logits, for the first item of the batch.
    public func callAsFunction(_ spectrograms: MLXArray) -> NFKMLXAllInOneLogits {
        let (batch, instruments, frames, bands) = (spectrograms.dim(0), spectrograms.dim(1),
                                                   spectrograms.dim(2), spectrograms.dim(3))
        let stacked = spectrograms.reshaped([batch * instruments, frames, bands, 1])
        var hidden = embeddings(stacked)
        hidden = encoder(hidden)
        hidden = norm(hidden.reshaped([batch, instruments, frames, configuration.embedDimension]))

        return NFKMLXAllInOneLogits(beat: beatHead(hidden).squeezed(axis: 1),
                                    downbeat: downbeatHead(hidden).squeezed(axis: 1),
                                    section: sectionHead(hidden).squeezed(axis: 1),
                                    function: functionHead(hidden))
    }
}

/// The eleven blocks, at doubling dilations.
public final class NFKAllInOneEncoder: Module {
    @ModuleInfo(key: "layers") var layers: [NFKAllInOneBlock]

    init(_ configuration: NFKMLXAllInOneConfiguration) {
        let depth = configuration.depth
        _layers.wrappedValue = (0 ..< depth).map {
            NFKAllInOneBlock(configuration, dilation: configuration.dilation(atBlock: $0),
                             dropPath: depth > 1 ? configuration.dropPath * Float($0) / Float(depth - 1) : 0)
        }
    }

    func callAsFunction(_ hidden: MLXArray) -> MLXArray {
        var states = hidden
        for layer in layers {
            states = layer(states)
        }
        return states
    }
}

/// The front end: madmom's filtered logarithmic spectrogram.
///
/// The filterbank is a `[bins, bands]` matrix fixed by the configuration (12 bands per octave from
/// 30 Hz to 17 kHz, each filter normalized to sum to one), so it travels in the checkpoint rather than
/// being re-derived here. `Tools/allin1-to-safetensors` takes it from madmom itself, which is the only
/// way to be sure of the band edges the released model was trained on.
public final class NFKAllInOneFrontEnd: Module {
    @ParameterInfo(key: "filterbank") var filterbank: MLXArray          // [windowSize / 2, bands]

    let configuration: NFKMLXAllInOneConfiguration
    /// The analysis window, held as samples rather than as an `MLXArray`: a module collects its
    /// array properties as parameters, and this one is a constant of the configuration that no
    /// checkpoint carries.
    private let window: [Float]

    init(_ configuration: NFKMLXAllInOneConfiguration) {
        self.configuration = configuration
        let size = configuration.windowSize
        // `np.hanning` is the symmetric window, which divides by `N − 1`.
        window = (0 ..< size).map {
            Float(0.5 - 0.5 * cos(2.0 * Double.pi * Double($0) / Double(size - 1)))
        }
        _filterbank.wrappedValue = MLXArray.zeros([size / 2, configuration.inputBins])
    }

    /// The frames a clip of `samples` produces: `ceil(samples / hop)`, each centered on its own hop.
    public func frameCount(samples: Int) -> Int {
        (samples + configuration.hopSize - 1) / configuration.hopSize
    }

    /// A mono clip at the configured rate → `[frames, bands]`.
    public func callAsFunction(_ samples: [Float]) -> MLXArray {
        let size = configuration.windowSize
        let hop = configuration.hopSize
        let frames = frameCount(samples: samples.count)
        // A frame is centered on its hop, so the signal is padded by half a window at the front and
        // by whatever the last frame reaches past the end.
        let lead = size / 2
        let total = max(samples.count + lead, (frames - 1) * hop + size)
        var padded = [Float](repeating: 0, count: total)
        for index in 0 ..< samples.count { padded[lead + index] = samples[index] }

        var framed = [Float](repeating: 0, count: frames * size)
        for frame in 0 ..< frames {
            let start = frame * hop
            for index in 0 ..< size { framed[frame * size + index] = padded[start + index] * window[index] }
        }
        let signal = framed.withUnsafeBufferPointer { MLXArray($0, [frames, size]) }
        // madmom keeps the bins below Nyquist and drops the Nyquist bin itself.
        let spectrum = MLXFFT.rfft(signal, axis: 1)[0..., 0 ..< (size / 2)]
        let magnitude = sqrt(spectrum.realPart().square() + spectrum.imaginaryPart().square())
        return log(magnitude.matmul(filterbank) + 1) * Float(1.0 / log(10.0))
    }
}

/// What the analyzer found in a track.
public struct NFKMLXAllInOneAnalysis {
    /// The functional sections, in order, labeled from the model's own vocabulary.
    public var sections: [NFKAudioSegment]
    /// The beats, with their position in the bar when the meter is tracked.
    public var beats: [NFKMusicBeat]
    /// The tempo in beats per minute, or nil when the track has fewer than two beats.
    public var tempoBPM: Double?
}

/// The post-processing that turns the section head's frame scores into labeled spans.
///
/// The section head scores a boundary, not a section: a peak in it is a place where the music
/// changes. Peaks are picked against a twelve-second window either side, so a boundary has to stand
/// out over that span rather than merely be a local maximum, and each span is then labeled by the
/// label that wins the average over its own frames.
enum NFKAllInOneStructure {

    /// Keeps a value where it is the maximum of the window centered on it, and zeros the rest.
    static func localMaxima(_ values: [Float], filterSize: Int) -> [Float] {
        let half = filterSize / 2
        var kept = [Float](repeating: 0, count: values.count)
        for index in 0 ..< values.count {
            var maximum = -Float.greatestFiniteMagnitude
            for offset in max(0, index - half) ... min(values.count - 1, index + half) {
                maximum = max(maximum, values[offset])
            }
            if values[index] == maximum { kept[index] = values[index] }
        }
        return kept
    }

    /// The boundary strength at each peak: how far it stands above the mean of the windows before and
    /// after it. Zero everywhere else.
    static func peakPicking(_ activation: [Float], past: Int, future: Int) -> [Float] {
        let count = activation.count
        var strength = [Float](repeating: 0, count: count)
        func value(_ index: Int) -> Float {
            index >= 0 && index < count ? activation[index] : 0
        }
        for index in 0 ..< count {
            guard activation[index] > 0 else { continue }
            var maximum = -Float.greatestFiniteMagnitude
            for offset in (index - past) ... (index + future) { maximum = max(maximum, value(offset)) }
            guard activation[index] == maximum else { continue }

            var pastSum = Float(0), futureSum = Float(0)
            for offset in 1 ... past { pastSum += value(index - offset) }
            for offset in 1 ... future { futureSum += value(index + offset) }
            let pastMean = pastSum / Float(past)
            let futureMean = futureSum / Float(future)
            strength[index] = activation[index] - (pastMean + futureMean) / 2
        }
        return strength
    }

    /// The labeled sections a track divides into.
    static func sections(sectionProbabilities: [Float], functionProbabilities: [[Float]],
                         configuration: NFKMLXAllInOneConfiguration) -> [NFKAudioSegment] {
        let frames = sectionProbabilities.count
        guard frames > 0 else { return [] }
        let fps = configuration.framesPerSecond
        let peaks = localMaxima(sectionProbabilities, filterSize: 4 * configuration.minimumFramesPerBeat + 1)
        let strength = peakPicking(peaks, past: 12 * fps, future: 12 * fps)

        let duration = Double(frames) * Double(configuration.hopSize) / Double(configuration.sampleRate)
        var boundaryFrames = (0 ..< frames).filter { strength[$0] > 0 }
        var times = boundaryFrames.map { Double($0) * Double(configuration.hopSize) / Double(configuration.sampleRate) }
        if times.first != 0 { times.insert(0, at: 0) }
        if times.last != duration { times.append(duration) }

        // The label of a span is the one that wins the average over the span's own frames. The spans
        // are the gaps between boundaries, and a boundary at frame 0 opens no gap of its own.
        boundaryFrames.removeAll { $0 == 0 }
        var spans = [(Int, Int)]()
        var start = 0
        for boundary in boundaryFrames {
            spans.append((start, boundary))
            start = boundary
        }
        spans.append((start, frames))

        var sections = [NFKAudioSegment]()
        for (index, span) in spans.enumerated() where index + 1 < times.count {
            var best = 0
            var bestMean = -Float.greatestFiniteMagnitude
            for label in 0 ..< functionProbabilities.count {
                var total = Float(0)
                for frame in span.0 ..< span.1 { total += functionProbabilities[label][frame] }
                let mean = span.1 > span.0 ? total / Float(span.1 - span.0) : 0
                if mean > bestMean {
                    bestMean = mean
                    best = label
                }
            }
            sections.append(NFKAudioSegment(startSeconds: times[index], endSeconds: times[index + 1],
                                            label: configuration.labels[best],
                                            confidence: Double(min(max(bestMean, 0), 1))))
        }
        return sections
    }

    /// The tempo the beats imply: the most common of their rounded rates.
    static func tempo(beats: [NFKMusicBeat]) -> Double? {
        guard beats.count >= 2 else { return nil }
        var counts = [Int: Int]()
        for index in 1 ..< beats.count {
            let interval = beats[index].timeSeconds - beats[index - 1].timeSeconds
            guard interval > 0 else { continue }
            counts[Int((60.0 / interval).rounded()), default: 0] += 1
        }
        return counts.max { left, right in
            left.value == right.value ? left.key < right.key : left.value < right.value
        }.map { Double($0.key) }
    }
}

extension NFKMLXAllInOneNet {

    /// The spectrograms of the four stems, stacked the way the network reads them:
    /// `[1, instruments, frames, bands]`. The stems are the HT Demucs order, bass first.
    public func spectrograms(stems: [[Float]]) -> MLXArray {
        let frames = stems.map { frontEnd.frameCount(samples: $0.count) }.min() ?? 0
        let planes = stems.map { frontEnd($0)[0 ..< frames, 0...].expandedDimensions(axis: 0) }
        return concatenated(planes, axis: 0).expandedDimensions(axis: 0)
    }

    /// Analyzes a track from its four stems.
    ///
    /// The beats come back unmetered (`positionInBar` 0) unless a bar tracker is given: the released
    /// model scores a beat and a downbeat probability per frame, and turning those into beat times is
    /// a decoding problem of its own.
    public func analyze(stems: [[Float]], barTracker: NFKMLXBarTracker? = nil) -> NFKMLXAllInOneAnalysis {
        let logits = self(spectrograms(stems: stems))
        eval(logits.beat, logits.downbeat, logits.section, logits.function)

        let sectionProbabilities = sigmoid(logits.section)[0].asArray(Float.self)
        let functionProbabilities = softmax(logits.function, axis: 1)[0]
        let labels = configuration.labels.count
        let frames = sectionProbabilities.count
        let flattened = functionProbabilities.asArray(Float.self)
        let perLabel = (0 ..< labels).map { label in
            Array(flattened[(label * frames) ..< ((label + 1) * frames)])
        }

        let sections = NFKAllInOneStructure.sections(sectionProbabilities: sectionProbabilities,
                                                     functionProbabilities: perLabel,
                                                     configuration: configuration)
        var beats = [NFKMusicBeat]()
        if let barTracker {
            let beatProbabilities = sigmoid(logits.beat)[0].asArray(Float.self)
            let downbeatProbabilities = sigmoid(logits.downbeat)[0].asArray(Float.self)
            beats = barTracker.track(beat: beatProbabilities, downbeat: downbeatProbabilities,
                                     threshold: postprocess.downbeatThreshold)
        }
        return NFKMLXAllInOneAnalysis(sections: sections, beats: beats,
                                      tempoBPM: NFKAllInOneStructure.tempo(beats: beats))
    }
}

/// Holds the networks for capture in the backend's `@Sendable` closure.
private final class NFKAllInOneHolder: @unchecked Sendable {
    let net: NFKMLXAllInOneNet
    let separator: NFKMLXHTDemucsNet?
    let barTracker: NFKMLXBarTracker
    init(net: NFKMLXAllInOneNet, separator: NFKMLXHTDemucsNet?, barTracker: NFKMLXBarTracker) {
        self.net = net
        self.separator = separator
        self.barTracker = barTracker
    }
}

/// Music structure analysis as an InferKit backend. Reads `NFKInputAudio`; returns the functional
/// sections under `NFKOutputSegments`, the beats under `NFKOutputBeats`, and the tempo under
/// `NFKOutputTempo`.
///
/// The model reads four stems. `NFKInputAudio` is either the mixture, which the attached HT Demucs
/// separates, or an array of the four stems in the order bass, drums, other, vocals.
@objc(NFKMLXAllInOneBackend)
public final class NFKMLXAllInOneBackend: NSObject, NFKInferenceBackend {

    private let holder: NFKAllInOneHolder
    private let identifier: String

    init(net: NFKMLXAllInOneNet, separator: NFKMLXHTDemucsNet?, identifier: String) {
        holder = NFKAllInOneHolder(net: net, separator: separator,
                                   barTracker: NFKMLXBarTracker(NFKMLXBarTrackerConfiguration(
                                       framesPerSecond: net.configuration.framesPerSecond)))
        self.identifier = identifier
        super.init()
    }

    @objc public var isReady: Bool { true }
    @objc public var backendIdentifier: String { identifier }
    @objc public var supportedParameterKeys: Set<String> { [] }
    @objc public var supportedInputKeys: Set<String> { [NFKInputAudio] }

    /// Whether the backend can take a mixture, which needs the separator, rather than four stems.
    @objc public var separatesMixtures: Bool { holder.separator != nil }

    @objc(runInferenceForRequest:error:)
    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        let rate = holder.net.configuration.sampleRate
        guard let clips = Self.audio(from: request), !clips.isEmpty else {
            throw NFKMLXError.unsupportedInput
        }

        var stems: [[Float]]
        if clips.count == holder.net.configuration.instruments {
            stems = clips.map { NFKMLXAudioRate.matched($0.samples, from: $0.sampleRate, to: rate) }
        } else if clips.count == 1, let separator = holder.separator {
            stems = Self.separate(clips[0], with: separator, to: rate)
        } else {
            throw NFKMLXError.unsupportedInput
        }

        let analysis = holder.net.analyze(stems: stems, barTracker: holder.barTracker)
        var outputs: [String: Any] = [NFKOutputSegments: analysis.sections, NFKOutputBeats: analysis.beats]
        if let tempo = analysis.tempoBPM {
            outputs[NFKOutputTempo] = NSNumber(value: tempo)
        }
        return NFKInferenceResult(outputs: outputs)
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

    /// The four stems of a mixture, in the order the model reads them.
    ///
    /// HT Demucs emits drums, bass, other, vocals; All-In-One was trained on the stems in the order
    /// the reference reads its files, which is alphabetical: bass, drums, other, vocals. Swapping the
    /// first two is silent — the model still runs — so the order is fixed here rather than assumed.
    static func separate(_ clip: (samples: [Float], sampleRate: Int), with separator: NFKMLXHTDemucsNet,
                         to rate: Int) -> [[Float]] {
        let matched = NFKMLXAudioRate.matched(clip.samples, from: clip.sampleRate, to: rate)
        let mono = MLXArray(matched).reshaped([1, matched.count])
        let stereo = concatenated([mono, mono], axis: 0)
        let separated = separator.separate(stereo)                      // [sources, channels, length]
        eval(separated)

        let order = [1, 0, 2, 3]                                        // bass, drums, other, vocals
        return order.map { source in
            separated[source].mean(axis: 0).asArray(Float.self)
        }
    }

    private static func audio(from request: NFKInferenceRequest) -> [(samples: [Float], sampleRate: Int)]? {
        guard let value = request.input(forKey: NFKInputAudio) else { return nil }
        func read(_ item: Any) -> (samples: [Float], sampleRate: Int)? {
            if let asset = item as? NFKAudioAsset, let url = asset.fileURL, let data = try? Data(contentsOf: url) {
                return NFKMLXWaveFile.read(data)
            }
            if let data = item as? Data { return NFKMLXWaveFile.read(data) }
            return nil
        }
        if let list = value as? [Any] { return list.compactMap(read) }
        return read(value).map { [$0] }
    }
}

/// Registration and weight loading for All-In-One music structure analysis.
@objc(NFKMLXAllInOne)
public final class NFKMLXAllInOne: NSObject {

    /// The registry name the model builds under.
    @objc public static let modelName = "allin1"

    public static func makeNet(_ configuration: NFKMLXAllInOneConfiguration = .harmonix) -> NFKMLXAllInOneNet {
        NFKMLXAllInOneNet(configuration)
    }

    /// Builds an All-In-One backend from optional local weights — no registry required. Without a
    /// separator the backend takes the four stems; `backend(weightsURL:demucsWeightsURL:)` adds one so
    /// it can take a mixture. A nil `weightsURL` builds random weights. Run inference off the render
    /// thread.
    ///
    /// - Since: InferKit 0.4.0
    @objc(backendWithWeightsURL:error:)
    public static func backend(weightsURL: URL?) throws -> any NFKInferenceBackend {
        try backend(weightsURL: weightsURL, demucsWeightsURL: nil)
    }

    /// Builds a backend that separates a mixture itself, with HT Demucs behind it.
    @objc(backendWithWeightsURL:demucsWeightsURL:error:)
    public static func backend(weightsURL: URL?, demucsWeightsURL: URL?) throws -> any NFKInferenceBackend {
        let net = makeNet()
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        var separator: NFKMLXHTDemucsNet?
        if let demucsWeightsURL {
            let demucs = NFKMLXHTDemucs.makeNet()
            try NFKMLXHTDemucs.loadWeights(into: demucs, from: demucsWeightsURL)
            separator = demucs
        }
        return NFKMLXAllInOneBackend(net: net, separator: separator, identifier: modelName)
    }

    /// Downloads the checkpoint from Hugging Face, then builds — no registry required. Blocking on the
    /// network; run off the render thread.
    @objc(backendWithRepo:weightsPath:revision:cacheDirectoryURL:error:)
    public static func backend(repo: String, weightsPath: String, revision: String?, cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        let url = try NFKMLXDownload.weightsURL(repo: repo, weightsPath: weightsPath, revision: revision, cacheDirectoryURL: cacheDirectoryURL)
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

    /// Registers All-In-One (`allin1`) with `NFKMLXModelRegistry`. The registered backend takes the
    /// four stems; a mixture needs the separator, which `backend(weightsURL:demucsWeightsURL:)` attaches.
    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName) { weightsURL in try backend(weightsURL: weightsURL) }
    }

    /// Loads a converted checkpoint, transposing the embedding convolutions `[out, in, kH, kW]` →
    /// MLX's `[out, kH, kW, in]`. Every other tensor transfers unchanged, so the module's names are
    /// the reference's and no remap is needed.
    public static func loadWeights(into net: NFKMLXAllInOneNet, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        var mapped = [(String, MLXArray)]()
        for (key, value) in checkpoint.arrays {
            mapped.append((key, value.ndim == 4 && checkpoint.needsConvTranspose ? value.transposed(0, 2, 3, 1) : value))
        }
        try NFKMLXWeights.apply(mapped, to: net)
    }
}
