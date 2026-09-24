//
//  NFKMLXCosmosTokenizerTraining.swift
//  InferKitMLX
//
//  Fine-tuning a Cosmos Tokenizer on a consumer's own images or clips, the full customization recipe.
//
//  NVIDIA post-trains the released tokenizers to a new domain (their own surgical and lidar
//  tokenizers are such fine-tunes), and the recipe is small enough for a device: the network is
//  about 80M parameters and trains on crops. The objective is the one the reference's post-training
//  experiments run (`cosmos_predict1/tokenizer/training`): a mean absolute pixel error plus 0.1 times
//  a perceptual term, the layer-weighted mean absolute difference of VGG-16 features at its five
//  `relu*_*` taps, over each image or frame. The reference turns its Gram-matrix term and its flow and
//  consistency terms off for post-training; the Gram term ships here too, measured against the
//  reference with it on, and stays off by default. A discrete tokenizer trains through the same
//  objective, with its rounding's gradient passed straight through.
//
//  Training every weight moves the latent space, which breaks compatibility with a world model trained
//  on the released latents. Training the decoder alone keeps the encoder, and so the latents and
//  tokens, exactly as released.
//

import Foundation
import MLX
import MLXNN
import MLXOptimizers

/// Which parameters a Cosmos Tokenizer fine-tune updates.
public enum NFKMLXCosmosTokenizerTrainable: Sendable {

    /// Every parameter, as the reference's post-training does.
    case everything

    /// The decoder and the projection from the latent into it, with the encoder and its latent
    /// projection frozen: reconstructions adapt to the new domain while every latent and token stays
    /// the one the release computes.
    case decoder
}

/// A parameterless stand-in for a ReLU or max-pool slot in VGG's `features` list, so the array indices
/// keep matching the checkpoint's (`features.0`, `features.2`, `features.5`, …).
final class NFKVGGPassThrough: Module {}

/*!
 @abstract The VGG-16 convolutional trunk (torchvision's `features`), the perceptual loss network.
 @discussion Returns the activations after `relu1_2`, `relu2_2`, `relu3_3`, `relu4_3`, and `relu5_3`
 for images `[N, H, W, 3]`. Loads torchvision's ImageNet weights, as `timm/vgg16.tv_in1k` republishes
 them (`model.safetensors`) or as `vgg16-397923af.pth`; the classifier is not used. The trunk is
 frozen when built. Nil weights leave it at its random initialization, which exercises the wiring of an
 objective and nothing more: a perceptual term needs the ImageNet weights to mean anything.
 Introduced in InferKit 0.4.0.
 */
public final class NFKMLXVGG16Features: Module {
    @ModuleInfo(key: "features") var features: [Module]

    /// The trunk's convolution positions in torchvision's `features`, and the positions after which a
    /// max-pool halves the resolution.
    static let convolutions: [(index: Int, input: Int, output: Int)] = [
        (0, 3, 64), (2, 64, 64), (5, 64, 128), (7, 128, 128), (10, 128, 256), (12, 256, 256), (14, 256, 256),
        (17, 256, 512), (19, 512, 512), (21, 512, 512), (24, 512, 512), (26, 512, 512), (28, 512, 512),
    ]
    static let pools: Set<Int> = [4, 9, 16, 23]
    static let taps: Set<Int> = [3, 8, 15, 22, 29]

    public init(weightsURL: URL?) throws {
        var layers: [Module] = (0 ..< 30).map { _ in NFKVGGPassThrough() }
        for convolution in Self.convolutions {
            layers[convolution.index] = Conv2d(inputChannels: convolution.input, outputChannels: convolution.output,
                                               kernelSize: 3, padding: 1)
        }
        _features.wrappedValue = layers
        super.init()
        defer { freeze() }
        guard let weightsURL else { return }
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: weightsURL)
        let mapped = checkpoint.arrays.compactMap { key, value -> (String, MLXArray)? in
            guard key.hasPrefix("features.") else { return nil }
            let array = value.asType(.float32)
            return (key, checkpoint.needsConvTranspose && array.ndim == 4 ? array.transposed(0, 2, 3, 1) : array)
        }
        try NFKMLXWeights.apply(mapped, to: self, verifyShapes: true)
    }

    /// The five tapped activations of images `[N, H, W, 3]` already normalized for VGG.
    public func callAsFunction(_ x: MLXArray) -> [MLXArray] {
        var h = x
        var tapped: [MLXArray] = []
        for index in 0 ..< 30 {
            if let convolution = features[index] as? Conv2d {
                h = convolution(h)
            } else if Self.pools.contains(index) {
                let s = h.shape
                h = h[0..., 0 ..< (s[1] / 2 * 2), 0 ..< (s[2] / 2 * 2)]
                    .reshaped([s[0], s[1] / 2, 2, s[2] / 2, 2, s[3]]).max(axes: [2, 4])
            } else {
                h = relu(h)
            }
            if Self.taps.contains(index) {
                tapped.append(h)
            }
        }
        return tapped
    }
}

/// The reconstruction objective a Cosmos Tokenizer fine-tune minimizes: the reference's `ColorLoss`
/// (L1) and `PerceptualLoss` (VGG-16 feature L1, and its optional Gram-matrix term), each reduced by its
/// mean and summed.
public struct NFKMLXCosmosTokenizerObjective {

    /// The reference's post-training weights: color 1, perceptual 0.1, Gram off.
    public var colorWeight: Float = 1
    public var perceptualWeight: Float = 0.1
    public var gramWeight: Float = 0
    /// The weight of each VGG tap, shallow to deep.
    public var layerWeights: [Float] = [1 / 2.6, 1 / 4.8, 1 / 3.7, 1 / 5.6, 10 / 1.5]

    public let perceptualNetwork: NFKMLXVGG16Features

    /// Builds the objective over VGG-16 ImageNet weights (see ``NFKMLXVGG16Features``).
    public init(vggWeightsURL: URL?) throws {
        perceptualNetwork = try NFKMLXVGG16Features(weightsURL: vggWeightsURL)
    }

    /// The loss of a reconstruction against its target, both images `[B, H, W, 3]` or clips
    /// `[B, T, H, W, 3]` in `[-1, 1]`, as a scalar.
    public func loss(reconstruction: MLXArray, target: MLXArray) -> MLXArray {
        let parts = components(reconstruction: reconstruction, target: target)
        return parts.color * colorWeight + parts.perceptual * perceptualWeight
            + (gramWeight > 0 ? parts.gram * gramWeight : MLXArray(Float(0)))
    }

    /// The three terms before they are weighted, each a scalar, so a parity run can say which one
    /// disagrees. The Gram term is computed only when its weight is nonzero, as the reference skips it.
    public func components(reconstruction: MLXArray, target: MLXArray)
        -> (color: MLXArray, perceptual: MLXArray, gram: MLXArray) {
        let color = abs(target - reconstruction).mean()
        let isVideo = target.ndim == 5
        let batch = target.shape[0]
        func frames(_ x: MLXArray) -> MLXArray {
            isVideo ? x.reshaped([-1, x.shape[2], x.shape[3], x.shape[4]]) : x
        }
        let shift = MLXArray([Float(-0.030), -0.088, -0.188])
        let scale = MLXArray([Float(0.458), 0.448, 0.450])
        let targetFeatures = perceptualNetwork((frames(target) - shift) / scale)
        let reconstructionFeatures = perceptualNetwork((frames(reconstruction) - shift) / scale)

        var perceptual = MLXArray(Float(0))
        var gram = MLXArray(Float(0))
        for (index, (a, b)) in zip(targetFeatures, reconstructionFeatures).enumerated() {
            perceptual = perceptual + abs(a - b).mean() * layerWeights[index]
            if gramWeight > 0 {
                gram = gram + square(gramMatrix(a, batch: batch, isVideo: isVideo)
                    - gramMatrix(b, batch: batch, isVideo: isVideo)).mean() * layerWeights[index]
            }
        }
        return (color, perceptual, gram)
    }

    /// Feature correlations `[N, C, C]`: over each image's positions, or for a clip over every
    /// position of every frame together, divided by the positions counted.
    private func gramMatrix(_ features: MLXArray, batch: Int, isVideo: Bool) -> MLXArray {
        let c = features.shape[3]
        let rows = isVideo ? batch : features.shape[0]
        let flat = features.reshaped([rows, -1, c])
        return matmul(flat.transposed(0, 2, 1), flat) / Float(flat.shape[1])
    }
}

extension NFKMLXCosmosTokenizer {

    /// Fine-tunes a tokenizer network on a consumer's own images or clips and returns the loss from
    /// each step. The whole path is three calls: ``network(variant:weightsURL:)`` to build, this to
    /// train, and `NFKMLXWeights.save` to write a checkpoint the factories load like a release.
    ///
    /// - Parameters:
    ///   - net: the network to train.
    ///   - examples: supplies one batch per step, images `[B, H, W, 3]` or clips `[B, T, H, W, 3]` in
    ///     `[-1, 1]`, with sides a multiple of 16 and a clip's frame count one more than a multiple of
    ///     the temporal compression. `NFKMLXTrainingData.batch` builds images in `0…1`; scale them.
    ///   - trainable: which parameters update. Freezing is applied here and persists on `net`.
    ///   - objective: the reconstruction loss.
    ///   - optimizer: nil uses the reference's AdamW (learning rate 1e-4, betas 0.5 and 0.999, epsilon
    ///     1e-8, weight decay 0.01, bias-corrected as PyTorch's is).
    ///   - steps: how many batches to train on.
    ///   - clipGradientNorm: bounds the global gradient norm. The reference does not clip.
    ///   - learningRateSchedule: multiplies the rate at each step. Nil uses the reference's
    ///     `WarmupLambdaLR`, a linear warm-up over 5,000 steps, when the reference optimizer runs. With a
    ///     caller's optimizer, nil holds that optimizer's rate constant.
    ///   - checkpoint: writes the network periodically, so a suspended run keeps its progress.
    ///   - observer: receives each step and can end the run early.
    ///
    /// A run is multi-second; call it off the render thread.
    @discardableResult
    public static func fineTune(
        _ net: NFKMLXCosmosTokenizerNet,
        examples: (Int) -> MLXArray,
        trainable: NFKMLXCosmosTokenizerTrainable = .everything,
        objective: NFKMLXCosmosTokenizerObjective,
        optimizer: Optimizer? = nil,
        steps: Int,
        clipGradientNorm: Float? = nil,
        learningRateSchedule: NFKMLXLearningRateSchedule? = nil,
        checkpoint: NFKMLXTrainingCheckpoint? = nil,
        observer: NFKMLXTrainer.Observer? = nil
    ) throws -> [Float] {
        try NFKMLXFineTune.run(
            net,
            freezing: { apply(trainable, to: net) },
            optimizer: optimizer,
            reference: {
                NFKMLXReferenceOptimizers.adamW(learningRate: 1e-4, betas: (0.5, 0.999), weightDecay: 0.01)
            },
            referenceSchedule: { .linearWarmup(steps: 5000) },
            steps: steps,
            sample: examples,
            loss: { net, batch in objective.loss(reconstruction: net(batch), target: batch) },
            clipGradientNorm: clipGradientNorm,
            learningRateSchedule: learningRateSchedule,
            checkpoint: checkpoint, observer: observer)
    }

    static func apply(_ trainable: NFKMLXCosmosTokenizerTrainable, to net: NFKMLXCosmosTokenizerNet) {
        net.unfreeze()
        if trainable == .decoder {
            net.encoder.freeze()
            (net.quantConv as Module).freeze()
        }
    }
}
