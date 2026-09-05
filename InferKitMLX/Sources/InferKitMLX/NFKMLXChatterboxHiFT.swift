//
//  NFKMLXChatterboxHiFT.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXFFT
import MLXNN

// Chatterbox stage 5: HiFT, the mel-to-waveform half of S3Gen (a HiFTNet: neural source filter plus
// iSTFTNet). A convolutional F0 predictor reads the mel, a harmonic sine source is built from the F0 at
// the sample rate, and the generator upsamples the mel ×120 through three transposed convolutions and
// Snake-activated residual blocks, adding the source's short-time spectrum at each scale, then emits a
// 16-point spectrum per frame that an inverse STFT at hop 4 turns into 24 kHz audio.

/// The Snake activation with a linear-scale per-channel `alpha`: `x + sin²(αx) / α`.
final class NFKHiFTSnake: Module {
    @ParameterInfo(key: "alpha") var alpha: MLXArray
    init(channels: Int) {
        _alpha.wrappedValue = MLXArray.ones([channels])
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        x + (1 / (alpha + 1e-9)) * sin(x * alpha).square()
    }
}

/// HiFi-GAN's multi-dilation residual block with Snake activations.
final class NFKHiFTResBlock: Module {
    @ModuleInfo(key: "convs1") var convs1: [Conv1d]
    @ModuleInfo(key: "convs2") var convs2: [Conv1d]
    @ModuleInfo(key: "activations1") var activations1: [NFKHiFTSnake]
    @ModuleInfo(key: "activations2") var activations2: [NFKHiFTSnake]

    init(channels: Int, kernel: Int, dilations: [Int] = [1, 3, 5]) {
        _convs1.wrappedValue = dilations.map {
            Conv1d(inputChannels: channels, outputChannels: channels, kernelSize: kernel,
                   padding: (kernel * $0 - $0) / 2, dilation: $0)
        }
        _convs2.wrappedValue = dilations.map { _ in
            Conv1d(inputChannels: channels, outputChannels: channels, kernelSize: kernel, padding: (kernel - 1) / 2)
        }
        _activations1.wrappedValue = dilations.map { _ in NFKHiFTSnake(channels: channels) }
        _activations2.wrappedValue = dilations.map { _ in NFKHiFTSnake(channels: channels) }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x
        for index in convs1.indices {
            var xt = convs1[index](activations1[index](x))
            xt = convs2[index](activations2[index](xt))
            x = xt + x
        }
        return x
    }
}

/// A parameter-free marker holding the Sequential slot of the reference's `ELU`.
final class NFKHiFTELU: Module {}

/// `ConvRNNF0Predictor`: five 3-wide convolutions with ELU, then a linear to one value per frame,
/// reported as its magnitude.
final class NFKHiFTF0Predictor: Module {
    @ModuleInfo(key: "condnet") var condnet: [Module]
    @ModuleInfo(key: "classifier") var classifier: Linear

    init(inChannels: Int = 80, channels: Int = 512) {
        var layers = [Module]()
        for index in 0 ..< 5 {
            layers.append(Conv1d(inputChannels: index == 0 ? inChannels : channels, outputChannels: channels,
                                 kernelSize: 3, padding: 1))
            layers.append(NFKHiFTELU())
        }
        _condnet.wrappedValue = layers
        _classifier.wrappedValue = Linear(channels, 1)
        super.init()
    }

    /// `[batch, frames, 80]` → F0 in Hz `[batch, frames]`.
    func callAsFunction(_ mel: MLXArray) -> MLXArray {
        var x = mel
        for layer in condnet {
            if let conv = layer as? Conv1d { x = elu(conv(x)) }
        }
        return abs(classifier(x)[0..., 0..., 0])
    }
}

/// `SourceModuleHnNSF`: an F0-driven harmonic sine source (nine harmonics), merged to one channel.
final class NFKHiFTSource: Module {
    @ModuleInfo(key: "l_linear") var merge: Linear
    let sampleRate: Float
    let harmonics: Int
    let amplitude: Float = 0.1
    let noiseStd: Float = 0.003
    let voicedThreshold: Float = 10

    init(sampleRate: Int, harmonics: Int = 8) {
        self.sampleRate = Float(sampleRate)
        self.harmonics = harmonics
        _merge.wrappedValue = Linear(harmonics + 1, 1)
        super.init()
    }

    /// - Parameters:
    ///   - f0: `[batch, samples]`, the frame F0 held over each frame's samples.
    ///   - deterministic: true zeroes the random harmonic phases and the additive noise, which is how the
    ///     parity record is made; the reference draws both fresh on every call.
    /// - Returns: the merged source `[batch, samples]`.
    func callAsFunction(_ f0: MLXArray, deterministic: Bool) -> MLXArray {
        let (batch, samples) = (f0.dim(0), f0.dim(1))
        let orders = MLXArray((1 ... Int32(harmonics + 1)).map { Float($0) })
        // The instantaneous frequency per harmonic in cycles per sample, integrated to a phase.
        let cycles = f0.expandedDimensions(axis: 2) * orders / sampleRate                       // [b, L, 9]
        let theta = 2 * Float.pi * (cumsum(cycles, axis: 1) % 1)
        var phase = MLXArray.zeros([batch, 1, harmonics + 1])
        if !deterministic {
            let random = MLXRandom.uniform(low: -Float.pi, high: Float.pi, [batch, 1, harmonics + 1])
            phase = concatenated([MLXArray.zeros([batch, 1, 1]), random[0..., 0..., 1...]], axis: 2)
        }
        let voiced = (f0 .> voicedThreshold).asType(.float32).expandedDimensions(axis: 2)
        var sines = amplitude * sin(theta + phase) * voiced
        if !deterministic {
            let noiseAmplitude = voiced * noiseStd + (1 - voiced) * amplitude / 3
            sines = sines + noiseAmplitude * MLXRandom.normal([batch, samples, harmonics + 1])
        }
        return tanh(merge(sines))[0..., 0..., 0]
    }
}

/// The HiFT generator (`mel2wav`).
public final class NFKHiFTGeneratorNet: Module {
    @ModuleInfo(key: "f0_predictor") var f0Predictor: NFKHiFTF0Predictor
    @ModuleInfo(key: "m_source") var source: NFKHiFTSource
    @ModuleInfo(key: "conv_pre") var convPre: Conv1d
    @ModuleInfo(key: "ups") var ups: [ConvTransposed1d]
    @ModuleInfo(key: "source_downs") var sourceDowns: [Conv1d]
    @ModuleInfo(key: "source_resblocks") var sourceResblocks: [NFKHiFTResBlock]
    @ModuleInfo(key: "resblocks") var resblocks: [NFKHiFTResBlock]
    @ModuleInfo(key: "conv_post") var convPost: Conv1d
    public let sampleRate: Int
    public let samplesPerFrame: Int
    let fftSize = 16
    let hop = 4
    let stft: NFKKokoroSTFT
    let resblockKernels = [3, 7, 11]

    public init(sampleRate: Int = 24000, melChannels: Int = 80, baseChannels: Int = 512,
                upsampleRates: [Int] = [8, 5, 3], upsampleKernels: [Int] = [16, 11, 7],
                sourceKernels: [Int] = [7, 7, 11]) {
        self.sampleRate = sampleRate
        samplesPerFrame = upsampleRates.reduce(1, *) * hop
        stft = NFKKokoroSTFT(nFFT: fftSize, hop: hop)
        _f0Predictor.wrappedValue = NFKHiFTF0Predictor(inChannels: melChannels, channels: baseChannels)
        _source.wrappedValue = NFKHiFTSource(sampleRate: sampleRate)
        _convPre.wrappedValue = Conv1d(inputChannels: melChannels, outputChannels: baseChannels, kernelSize: 7, padding: 3)
        var ups = [ConvTransposed1d]()
        for (index, (rate, kernel)) in zip(upsampleRates, upsampleKernels).enumerated() {
            ups.append(ConvTransposed1d(inputChannels: baseChannels >> index, outputChannels: baseChannels >> (index + 1),
                                        kernelSize: kernel, stride: rate, padding: (kernel - rate) / 2))
        }
        _ups.wrappedValue = ups
        // The source is read at each scale through a strided convolution whose stride is the product of
        // the REMAINING upsample rates, so its resolution matches the generator's at that stage.
        var cumulative = [1]
        for rate in upsampleRates.reversed().dropLast() { cumulative.append(cumulative.last! * rate) }
        var downs = [Conv1d](), sourceBlocks = [NFKHiFTResBlock]()
        for (index, stride) in cumulative.reversed().enumerated() {
            let channels = baseChannels >> (index + 1)
            downs.append(stride == 1
                ? Conv1d(inputChannels: fftSize + 2, outputChannels: channels, kernelSize: 1)
                : Conv1d(inputChannels: fftSize + 2, outputChannels: channels, kernelSize: stride * 2,
                         stride: stride, padding: stride / 2))
            sourceBlocks.append(NFKHiFTResBlock(channels: channels, kernel: sourceKernels[index]))
        }
        _sourceDowns.wrappedValue = downs
        _sourceResblocks.wrappedValue = sourceBlocks
        var blocks = [NFKHiFTResBlock]()
        for index in upsampleRates.indices {
            for kernel in resblockKernels { blocks.append(NFKHiFTResBlock(channels: baseChannels >> (index + 1), kernel: kernel)) }
        }
        _resblocks.wrappedValue = blocks
        _convPost.wrappedValue = Conv1d(inputChannels: baseChannels >> upsampleRates.count, outputChannels: fftSize + 2,
                                        kernelSize: 7, padding: 3)
        super.init()
    }

    /// The predicted F0 `[batch, frames]` of a mel `[batch, frames, 80]`.
    public func f0(mel: MLXArray) -> MLXArray { f0Predictor(mel) }

    /// The harmonic source `[batch, frames · samplesPerFrame]` for a frame F0 `[batch, frames]`.
    public func source(f0: MLXArray, deterministic: Bool) -> MLXArray {
        source(repeated(f0, count: samplesPerFrame, axis: 1), deterministic: deterministic)
    }

    /// `[batch, samples]` audio from a mel `[batch, frames, 80]` and its source `[batch, samples]`.
    public func decode(mel: MLXArray, source s: MLXArray) -> MLXArray {
        // The source's own short-time spectrum, real and imaginary parts as 18 channels.
        let (magnitude, phase) = stft.transform(s)
        let spectrum = concatenated([magnitude * cos(phase), magnitude * sin(phase)], axis: 1).transposed(0, 2, 1)
        var x = convPre(mel)
        for index in ups.indices {
            x = ups[index](leakyRelu(x, negativeSlope: 0.1))
            if index == ups.count - 1 {
                x = concatenated([x[0..., 1 ..< 2], x], axis: 1)                    // ReflectionPad1d((1, 0))
            }
            x = x + sourceResblocks[index](sourceDowns[index](spectrum))
            var sum: MLXArray?
            for kernelIndex in resblockKernels.indices {
                let block = resblocks[index * resblockKernels.count + kernelIndex](x)
                sum = sum.map { $0 + block } ?? block
            }
            x = sum! / Float(resblockKernels.count)
        }
        // The one bare `leaky_relu` before `conv_post` runs at PyTorch's default slope.
        x = convPost(leakyRelu(x, negativeSlope: 0.01))
        let bins = fftSize / 2 + 1
        let outMagnitude = minimum(exp(x[0..., 0..., 0 ..< bins]), MLXArray(Float(100))).transposed(0, 2, 1)
        let outPhase = sin(x[0..., 0..., bins...]).transposed(0, 2, 1)
        let audio = stft.inverse(magnitude: outMagnitude, phase: outPhase)
        return clip(audio, min: -0.99, max: 0.99)
    }

    /// Mel `[frames, 80]` → audio samples, with the reference's 40 ms leading fade-in.
    public func waveform(mel: MLXArray, deterministicSource: Bool = false) -> [Float] {
        let batched = mel.expandedDimensions(axis: 0)
        let s = source(f0: f0(mel: batched), deterministic: deterministicSource)
        var samples = decode(mel: batched, source: s)[0].asArray(Float.self)
        let trim = sampleRate / 50
        for index in 0 ..< min(2 * trim, samples.count) {
            let gain: Float = index < trim ? 0 : (cosf(Float.pi * (1 - Float(index - trim) / Float(trim - 1))) + 1) / 2
            samples[index] *= gain
        }
        return samples
    }
}

// MARK: - S3Gen

/// What S3Gen needs of the reference voice (`embed_ref`): its speech codes, its 24 kHz mel, and its x-vector.
public struct NFKMLXS3GenPrompt {
    public var tokens: [Int]
    public var mel: MLXArray
    public var xVector: MLXArray
    public init(tokens: [Int], mel: MLXArray, xVector: MLXArray) {
        self.tokens = tokens
        self.mel = mel
        self.xVector = xVector
    }
}

/// S3Gen: speech codes and a voice prompt in, 24 kHz audio out.
public final class NFKMLXS3GenNet: Module {
    @ModuleInfo(key: "speaker_encoder") var speakerEncoder: NFKCAMPPlusNet
    @ModuleInfo(key: "flow") var flow: NFKS3FlowNet
    @ModuleInfo(key: "mel2wav") var vocoder: NFKHiFTGeneratorNet
    public let sampleRate = 24000
    /// The reference truncates a voice prompt to ten seconds.
    public let maximumPromptSeconds = 10

    public override init() {
        _speakerEncoder.wrappedValue = NFKCAMPPlusNet()
        _flow.wrappedValue = NFKS3FlowNet()
        _vocoder.wrappedValue = NFKHiFTGeneratorNet()
        super.init()
    }

    /// Builds the voice prompt from the reference audio at 24 kHz and 16 kHz (`embed_ref`): the mel of
    /// the 24 kHz signal, the x-vector and speech codes of the 16 kHz one, the codes trimmed to half the
    /// mel's frame count when the two disagree.
    public func prompt(samples24k: [Float], samples16k: [Float], tokenizer: NFKMLXS3TokenizerNet) -> NFKMLXS3GenPrompt {
        let mel = NFKChatterboxPromptMel.features(Array(samples24k.prefix(maximumPromptSeconds * sampleRate)))
        let xVector = speakerEncoder.embedding(samples: Array(samples16k.prefix(maximumPromptSeconds * 16000)))
        var tokens = tokenizer.tokenize(Array(samples16k.prefix(maximumPromptSeconds * 16000)))
        if mel.dim(0) != 2 * tokens.count { tokens = Array(tokens.prefix(mel.dim(0) / 2)) }
        return NFKMLXS3GenPrompt(tokens: tokens, mel: mel, xVector: xVector)
    }

    /// The mel `[frames, 80]` for `tokens` in the prompt's voice.
    public func mel(tokens: [Int], prompt: NFKMLXS3GenPrompt, noise: MLXArray? = nil,
                    options: NFKS3FlowOptions = NFKS3FlowOptions()) -> MLXArray {
        flow.mel(tokens: tokens, promptTokens: prompt.tokens, promptMel: prompt.mel, xVector: prompt.xVector,
                 noise: noise, options: options)
    }

    /// Speech codes to 24 kHz samples.
    public func synthesize(tokens: [Int], prompt: NFKMLXS3GenPrompt,
                           options: NFKS3FlowOptions = NFKS3FlowOptions()) -> [Float] {
        vocoder.waveform(mel: mel(tokens: tokens, prompt: prompt, options: options))
    }
}

// MARK: - Loading

extension NFKMLXChatterbox {
    /// Loads the released `s3gen.safetensors` (its `tokenizer.` subtree belongs to
    /// ``NFKMLXS3TokenizerNet`` and is skipped). Weight-normalized convolutions fuse to plain weights,
    /// the CAMPPlus `xvector` Sequential and the estimator's positional block lists map onto named
    /// modules, and every convolution transposes to channels-last.
    public static func loadS3GenWeights(into net: NFKMLXS3GenNet, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        var arrays = [String: MLXArray]()
        for (key, value) in checkpoint.arrays where !key.hasPrefix("tokenizer.") && !key.hasSuffix("num_batches_tracked") {
            arrays[key] = value
        }
        arrays = NFKMLXSNAC.fusedWeightNorm(arrays)
        var mapped = [(String, MLXArray)]()
        for (key, value) in arrays {
            let name = remapS3GenKey(key)
            var array = value
            if checkpoint.needsConvTranspose, name.hasSuffix(".weight") {
                if array.ndim == 4 { array = array.transposed(0, 2, 3, 1) }
                else if array.ndim == 3 { array = name.hasPrefix("mel2wav.ups.") ? array.transposed(1, 2, 0) : array.transposed(0, 2, 1) }
            }
            mapped.append((name, array))
        }
        try NFKMLXWeights.apply(mapped, to: net)
        net.train(false)
    }

    static func remapS3GenKey(_ key: String) -> String {
        var name = key
        if name.hasPrefix("speaker_encoder.xvector.") {
            name = "speaker_encoder." + name.dropFirst("speaker_encoder.xvector.".count)
            // `block1.tdnnd7.` → `block1.layers.6.`
            if let range = name.range(of: #"\.tdnnd(\d+)\."#, options: .regularExpression) {
                let number = Int(name[range].filter(\.isNumber))!
                name.replaceSubrange(range, with: ".layers.\(number - 1).")
            }
        }
        if name.hasPrefix("flow.decoder.estimator.") {
            name = "flow.estimator." + name.dropFirst("flow.decoder.estimator.".count)
            name = name.replacingOccurrences(of: "time_mlp.linear_1.", with: "time_mlp_1.")
            name = name.replacingOccurrences(of: "time_mlp.linear_2.", with: "time_mlp_2.")
            // `down_blocks.0.0.` resnet, `.0.1.M.` transformer M, `.0.2.` the resample convolution.
            if let range = name.range(of: #"_blocks\.(\d+)\.(\d)\."#, options: .regularExpression) {
                let parts = name[range].split(separator: ".")
                let stage = parts[1], slot = parts[2]
                let replacement = slot == "0" ? "_blocks.\(stage).resnet." : slot == "1" ? "_blocks.\(stage).transformers." : "_blocks.\(stage).resample.conv."
                name.replaceSubrange(range, with: replacement)
            }
            name = name.replacingOccurrences(of: ".block.0.", with: ".conv.conv.")
            name = name.replacingOccurrences(of: ".block.2.", with: ".norm.")
            name = name.replacingOccurrences(of: ".mlp.1.", with: ".mlp.")
            name = name.replacingOccurrences(of: ".attn1.to_out.0.", with: ".attn1.to_out.")
            name = name.replacingOccurrences(of: ".ff.net.0.proj.", with: ".ff_proj.")
            name = name.replacingOccurrences(of: ".ff.net.2.", with: ".ff_out.")
        }
        if name.hasPrefix("flow.encoder.") {
            name = name.replacingOccurrences(of: "embed.out.0.", with: "embed.linear.")
            name = name.replacingOccurrences(of: "embed.out.1.", with: "embed.norm.")
        }
        return name
    }
}
