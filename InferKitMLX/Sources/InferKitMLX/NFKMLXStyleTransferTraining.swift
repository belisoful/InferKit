//
//  NFKMLXStyleTransferTraining.swift
//  InferKitMLX
//
//  Training a fast style transfer network on a consumer's own style image, as the PyTorch examples'
//  `neural_style.py train` trains `TransformerNet`: every parameter from scratch, a content term over
//  VGG-16's relu2_2 and a style term over the Gram matrices of its first four taps, and Adam at 1e-3
//  with no schedule.
//

import Foundation
import MLX
import MLXNN
import MLXOptimizers

/// The initializations `torch.nn` gives a fresh layer where MLX's differ.
enum NFKTorchInitialization {

    /// Draws every `Conv2d` bias in `module` as `torch.nn.Conv2d.reset_parameters` does, uniform within
    /// `±1/√fan_in`. MLX starts a convolution's bias at zero and draws its weight within the bound
    /// torch's `kaiming_uniform_(a=√5)` gives, so the weights stay as built.
    static func drawConvolutionBiases(in module: Module) {
        var biases = [(String, MLXArray)]()
        for (path, leaf) in module.leafModules().flattened() {
            guard let convolution = leaf as? Conv2d, convolution.bias != nil else { continue }
            let weight = convolution.weight
            let bound = (1 / Float(weight.dim(1) * weight.dim(2) * weight.dim(3))).squareRoot()
            biases.append((path + ".bias", MLXRandom.uniform(low: -bound, high: bound, [weight.dim(0)])))
        }
        module.update(parameters: ModuleParameters.unflattened(biases))
    }
}

/// Convolutions a float64 run reaches on MLX's CPU.
enum NFKFloat64Convolution {

    /// `convolution` applied to NHWC `x`. MLX has no float64 convolution, so a float64 `x` is zero-padded by
    /// the layer's padding and convolved as one matrix product per kernel tap, which MLX's CPU computes in
    /// float64. Any other type takes the layer itself.
    static func apply(_ convolution: Conv2d, _ x: MLXArray) -> MLXArray {
        guard x.dtype == .float64 else { return convolution(x) }
        let bordered = NFKMLXResample.spatiallyPadded(x, IntOrPair(convolution.padding), value: 0)
        let weight = convolution.weight
        let stride = convolution.stride.0
        let height = (bordered.dim(1) - weight.dim(1)) / stride + 1
        let width = (bordered.dim(2) - weight.dim(2)) / stride + 1
        var output: MLXArray?
        for ky in 0 ..< weight.dim(1) {
            for kx in 0 ..< weight.dim(2) {
                let tap = bordered[0..., .stride(from: ky, to: ky + stride * (height - 1) + 1, by: stride),
                                   .stride(from: kx, to: kx + stride * (width - 1) + 1, by: stride), 0...]
                let term = matmul(tap, weight[0..., ky, kx, 0...].transposed())
                output = output.map { $0 + term } ?? term
            }
        }
        return convolution.bias.map { output! + $0 } ?? output!
    }
}

/// The loss `neural_style.py train` minimizes: a content term between VGG-16's relu2_2 activations of
/// the stylized and the content image, and a style term between the Gram matrices of the stylized
/// image's relu1_2, relu2_2, relu3_3, and relu4_3 and those of the style image.
///
/// @discussion Images arrive in `0...1`, as `ToTensor` produces them, and are scaled to `0...255` as
/// the reference's `mul(255)` scales them before the network. Each side is then normalized for VGG
/// (divided by 255, less ImageNet's mean, over its deviation). The style image's Gram matrices are
/// computed once, at its own size, when the objective is built. Each term is a mean squared error, and
/// a Gram matrix is divided by its channels times its positions.
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXStyleTransferObjective {
    /// The content term's factor, the reference's `--content-weight`.
    public var contentWeight: Float
    /// The style term's factor, the reference's `--style-weight`.
    public var styleWeight: Float
    /// The VGG-16 trunk the terms read, frozen.
    public let perceptualNetwork: NFKMLXVGG16Features
    /// The style image's Gram matrices at the four taps, each `[1, C, C]`.
    public let styleGrams: [MLXArray]

    /// Builds the objective over VGG-16 ImageNet weights (see ``NFKMLXVGG16Features``) and one style
    /// image `[H, W, 3]` in `0...1`. The reference reads the style at its own size unless
    /// `--style-size` names one.
    public init(vggWeightsURL: URL?, styleImage: MLXArray, contentWeight: Float = 1e5,
                styleWeight: Float = 1e10) throws {
        self.init(perceptualNetwork: try NFKMLXVGG16Features(weightsURL: vggWeightsURL), styleImage: styleImage,
                  contentWeight: contentWeight, styleWeight: styleWeight)
    }

    /// Builds the objective over an already loaded trunk, so several styles share one.
    public init(perceptualNetwork: NFKMLXVGG16Features, styleImage: MLXArray, contentWeight: Float = 1e5,
                styleWeight: Float = 1e10) {
        self.perceptualNetwork = perceptualNetwork
        self.contentWeight = contentWeight
        self.styleWeight = styleWeight
        let style = (styleImage * 255).asType(NFKReferenceRounding.parameterType(of: perceptualNetwork))
            .reshaped([1] + styleImage.shape)
        let grams = perceptualNetwork.activations(Self.normalized(style), taps: 4).map(Self.gram)
        eval(grams)
        styleGrams = grams
    }

    /// Scores the network's stylization of content images `[N, H, W, 3]` in `0...1`.
    public func callAsFunction(_ net: NFKMLXStyleTransferNet, _ content: MLXArray) -> MLXArray {
        let pixels = content * 255
        return loss(stylized: net(pixels), content: pixels)
    }

    /// The weighted sum of the two terms for stylized images and their content images, both
    /// `[N, H, W, 3]` in `0...255`.
    public func loss(stylized: MLXArray, content: MLXArray) -> MLXArray {
        let terms = components(stylized: stylized, content: content)
        return terms.content + terms.style
    }

    /// The two terms, each already multiplied by its factor, as the reference logs them.
    public func components(stylized: MLXArray, content: MLXArray) -> (content: MLXArray, style: MLXArray) {
        let stylizedFeatures = perceptualNetwork.activations(Self.normalized(stylized), taps: 4)
        let contentFeatures = perceptualNetwork.activations(Self.normalized(stopGradient(content)), taps: 4)
        let contentTerm = square(stylizedFeatures[1] - contentFeatures[1]).mean() * contentWeight
        var styleTerm = MLXArray(Float(0))
        for (features, styleGram) in zip(stylizedFeatures, styleGrams) {
            styleTerm = styleTerm + square(Self.gram(features) - styleGram).mean()
        }
        return (contentTerm, styleTerm * styleWeight)
    }

    /// `utils.normalize_batch`: `0...255` pixels divided by 255, then standardized by ImageNet's statistics,
    /// which take the pixels' own type as `batch.new_tensor` gives them.
    static func normalized(_ pixels: MLXArray) -> MLXArray {
        func statistic(_ values: [Double]) -> MLXArray {
            guard pixels.dtype == .float64 else { return MLXArray(values.map { Float($0) }) }
            return stacked(values.map { MLXArray(float64: $0) })
        }
        return (pixels / 255 - statistic([0.485, 0.456, 0.406])) / statistic([0.229, 0.224, 0.225])
    }

    /// `utils.gram_matrix` of activations `[N, H, W, C]`: `[N, C, C]`, divided by `C · H · W`.
    static func gram(_ features: MLXArray) -> MLXArray {
        let (batch, height, width, channels) = (features.dim(0), features.dim(1), features.dim(2), features.dim(3))
        let flattened = features.reshaped([batch, height * width, channels])
        return matmul(flattened.transposed(0, 2, 1), flattened) / Float(channels * height * width)
    }
}

public extension NFKMLXStyleTransfer {

    /// The optimizer `neural_style.py train` builds: `torch.optim.Adam` at `--lr` 1e-3, bias-corrected,
    /// with no weight decay. Introduced in InferKit 0.4.0.
    static func referenceOptimizer() -> Optimizer {
        NFKMLXReferenceOptimizers.adamW(learningRate: 1e-3, weightDecay: 0)
    }

    /// Trains a fast style transfer network to apply the objective's style.
    ///
    /// - Parameters:
    ///   - net: the network, from ``network(weightsURL:)``. Nil weights train from scratch, as the
    ///     reference does for every style.
    ///   - contentImages: supplies one batch per step, `[N, H, W, 3]` in `0...1`, through
    ///     `NFKMLXTrainingData.batch`. The reference reads COCO 2014 in file order, each image resized
    ///     on its shorter side to 256 and center-cropped to 256 × 256, four to a batch, for two epochs.
    ///   - objective: the content and style terms, which hold the style image.
    ///   - optimizer: the update rule. Nil uses ``referenceOptimizer()``.
    ///   - steps: how many updates to train for.
    ///   - accumulationSteps: how many batches each update averages; `steps` counts updates.
    ///   - precision: the precision the passes compute in; float32 by default.
    ///   - learningRateSchedule: the schedule over the run. Nil keeps the rate constant, as the reference does.
    ///   - checkpoint: writes the network periodically.
    ///   - observer: receives each step and can end the run early.
    ///
    /// Every parameter of the network trains; the VGG-16 trunk stays frozen. Save with
    /// `NFKMLXWeights.save`; ``network(weightsURL:)`` and ``backend(weightsURL:)`` read the file back.
    /// A run is minutes; call it off the render thread.
    /// Introduced in InferKit 0.4.0.
    @discardableResult
    static func fineTune(
        _ net: NFKMLXStyleTransferNet,
        contentImages: (Int) -> MLXArray,
        objective: NFKMLXStyleTransferObjective,
        optimizer: Optimizer? = nil,
        steps: Int,
        accumulationSteps: Int = 1,
        precision: NFKMLXTrainingPrecision = .float32,
        learningRateSchedule: NFKMLXLearningRateSchedule? = nil,
        checkpoint: NFKMLXTrainingCheckpoint? = nil,
        observer: NFKMLXTrainer.Observer? = nil
    ) throws -> [Float] {
        try NFKMLXFineTune.run(
            net,
            freezing: {},
            optimizer: optimizer,
            reference: { referenceOptimizer() },
            referenceSchedule: { .constant },
            steps: steps,
            sample: contentImages,
            loss: objective.callAsFunction,
            clipGradientNorm: nil, accumulationSteps: accumulationSteps, precision: precision,
            learningRateSchedule: learningRateSchedule, checkpoint: checkpoint, observer: observer)
    }
}
