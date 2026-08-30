//
//  NFKMLXHiFiGAN.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXNN

// HiFi-GAN is a neural vocoder: a mel-spectrogram in, a raw waveform out. It is the synthesis half of a
// text-to-speech voice (the acoustic model produces the mel, this produces the audio). A convolutional
// generator upsamples the mel by transposed convolutions, each followed by a multi-receptive-field
// fusion of dilated residual blocks. Tensors flow NLC; the 1-D transposed conv reuses Demucs's
// `NFKDemucsConvT1d`.

/// HiFi-GAN generator sizing. Defaults approximate the `v1` configuration (hop 256).
public struct NFKMLXHiFiGANConfiguration: Sendable {
    public var melBins: Int = 80
    public var initialChannels: Int = 512
    public var upsampleRates: [Int] = [8, 8, 2, 2]
    public var upsampleKernels: [Int] = [16, 16, 4, 4]
    public var resblockKernels: [Int] = [3, 7, 11]
    public var resblockDilations: [[Int]] = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    public init() {}
    var hop: Int { upsampleRates.reduce(1, *) }
}

/// A residual block: pairs of dilated then plain convolutions, each a residual (leaky-ReLU gated).
final class NFKHiFiResBlock: Module {
    @ModuleInfo(key: "convs1") var convs1: [Conv1d]
    @ModuleInfo(key: "convs2") var convs2: [Conv1d]

    init(channels: Int, kernel: Int, dilations: [Int]) {
        _convs1.wrappedValue = dilations.map {
            Conv1d(inputChannels: channels, outputChannels: channels, kernelSize: kernel, padding: (kernel - 1) * $0 / 2, dilation: $0)
        }
        _convs2.wrappedValue = dilations.map { _ in
            Conv1d(inputChannels: channels, outputChannels: channels, kernelSize: kernel, padding: (kernel - 1) / 2)
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = x
        for (conv1, conv2) in zip(convs1, convs2) {
            let residual = conv2(leakyRelu(conv1(leakyRelu(y, negativeSlope: 0.1)), negativeSlope: 0.1))
            y = y + residual
        }
        return y
    }
}

/// The HiFi-GAN generator: `[1, T, melBins]` → `[1, T·hop, 1]` in `-1...1`.
final class NFKMLXHiFiGANNet: Module {
    @ModuleInfo(key: "conv_pre") var convPre: Conv1d
    @ModuleInfo(key: "ups") var ups: [NFKDemucsConvT1d]
    @ModuleInfo(key: "resblocks") var resblocks: [NFKHiFiResBlock]
    @ModuleInfo(key: "conv_post") var convPost: Conv1d

    let configuration: NFKMLXHiFiGANConfiguration
    private let kernelsPerStage: Int

    init(_ configuration: NFKMLXHiFiGANConfiguration) {
        self.configuration = configuration
        kernelsPerStage = configuration.resblockKernels.count
        _convPre.wrappedValue = Conv1d(inputChannels: configuration.melBins, outputChannels: configuration.initialChannels, kernelSize: 7, padding: 3)

        var channels = configuration.initialChannels
        var upConvs = [NFKDemucsConvT1d]()
        var blocks = [NFKHiFiResBlock]()
        for (rate, kernel) in zip(configuration.upsampleRates, configuration.upsampleKernels) {
            let outChannels = channels / 2
            upConvs.append(NFKDemucsConvT1d(channels, outChannels, kernel: kernel, stride: rate, padding: (kernel - rate) / 2))
            for (resKernel, dilations) in zip(configuration.resblockKernels, configuration.resblockDilations) {
                blocks.append(NFKHiFiResBlock(channels: outChannels, kernel: resKernel, dilations: dilations))
            }
            channels = outChannels
        }
        _ups.wrappedValue = upConvs
        _resblocks.wrappedValue = blocks
        _convPost.wrappedValue = Conv1d(inputChannels: channels, outputChannels: 1, kernelSize: 7, padding: 3)
    }

    /// `[1, T, melBins]` → `[1, T·hop, 1]`.
    func waveform(_ mel: MLXArray) -> MLXArray {
        var x = convPre(mel)
        for (index, up) in ups.enumerated() {
            x = up(leakyRelu(x, negativeSlope: 0.1))
            var fused = resblocks[index * kernelsPerStage](x)
            for k in 1 ..< kernelsPerStage {
                fused = fused + resblocks[index * kernelsPerStage + k](x)
            }
            x = fused / Float(kernelsPerStage)
        }
        // The reference's ONE bare `F.leaky_relu(x)` — the slope before conv_post is PyTorch's
        // default 0.01, not the 0.1 every other activation in the network uses. Measured: 0.1 here
        // scores 0.99954 against the released weights, 0.01 restores parity.
        return tanh(convPost(leakyRelu(x, negativeSlope: 0.01)))
    }
}

/// Registration and weight loading for the HiFi-GAN vocoder (used inside a TTS pipeline).
@objc(NFKMLXHiFiGAN)
public final class NFKMLXHiFiGAN: NSObject {

    static func makeNet(_ configuration: NFKMLXHiFiGANConfiguration = NFKMLXHiFiGANConfiguration()) -> NFKMLXHiFiGANNet {
        NFKMLXHiFiGANNet(configuration)
    }

    static func loadWeights(into net: NFKMLXHiFiGANNet, from url: URL, remap: (String) -> String = { $0 }) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let raw = referenceRenamed(NFKMLXMusic3.fusedWeightNorm(checkpoint.arrays))
        let mapped = raw.map { key, value -> (String, MLXArray) in
            // The upsampling stages are transposed convolutions, whose PyTorch layout is
            // [in, out, kernel] — not the forward convolutions' [out, in, kernel] — wrapped in a
            // singleton-width ConvTransposed2d, the same treatment Demucs's decoder gets.
            if checkpoint.needsConvTranspose, key.hasPrefix("ups."), key.hasSuffix("conv.weight"),
               value.ndim == 3 {
                return (remap(key), value.transposed(1, 2, 0).expandedDimensions(axis: 2))
            }
            if checkpoint.needsConvTranspose, value.ndim == 4 { return (remap(key), value.transposed(0, 2, 3, 1)) }
            if checkpoint.needsConvTranspose, value.ndim == 3 { return (remap(key), value.transposed(0, 2, 1)) }
            return (remap(key), value)
        }
        try NFKMLXWeights.apply(mapped, to: net)
    }

    /// The renames `Tools/hifigan-to-safetensors` applies offline, so a raw release loads directly
    /// (its weight norm is fused before this): the espnet-paired release's `vocoder.` prefix and
    /// `upsampler.` naming, and the upsampling stages' `ups.N.weight` → `ups.N.conv.weight`. A
    /// converted file's keys pass through unchanged.
    static func referenceRenamed(_ arrays: [String: MLXArray]) -> [String: MLXArray] {
        var renamed = [String: MLXArray]()
        renamed.reserveCapacity(arrays.count)
        for (key, value) in arrays {
            var name = key
            if name.hasPrefix("vocoder.") {
                name = String(name.dropFirst("vocoder.".count))
                    .replacingOccurrences(of: "upsampler.", with: "ups.")
            }
            let parts = name.split(separator: ".").map(String.init)
            if parts.count == 3, parts[0] == "ups" {
                name = ([parts[0], parts[1], "conv"] + parts[2...]).joined(separator: ".")
            }
            renamed[name] = value
        }
        return renamed
    }
}
