//
//  NFKMLXRIFEv4Training.swift
//  InferKitMLX
//
//  RIFE v4's fine-tune, ported from the v4.15 training code hzwer/Practical-RIFE links (`train.py`,
//  `model.py`, `Flownet.py`, `vgg.py`, `ssim.py`): a VGG-19 perceptual term less a tenth of SSIM, the
//  distance of each block's frame and its encoder features from the target's, a teacher blended from the
//  blocks' own flows, a consistency term toward it, AdamW under a warm-up and cosine, and a clip at 1.
//

import Foundation
import MLX
import MLXNN
import MLXOptimizers

/**
 The first five stages of torchvision's VGG-19 `features`, up to `relu5_1`, as RIFE's
 `VGGPerceptualLoss` reads them: the activations after `relu1_1`, `relu2_1`, `relu3_1`, `relu4_1`, and
 `relu5_1`.

 The weights are torchvision's ImageNet release (`vgg19-dcbb9e9d.pth`); the classifier and the layers past
 `relu5_1` are not used. The trunk is frozen when built. Nil weights leave it at its random initialization,
 which exercises the wiring of an objective and nothing more. Introduced in InferKit 0.4.0.
 */
public final class NFKMLXVGG19Features: Module {
    @ModuleInfo(key: "features") var features: [Module]

    /// The convolution positions in torchvision's `features` up to `relu5_1`, and the positions after
    /// which a max-pool halves the resolution.
    static let convolutions: [(index: Int, input: Int, output: Int)] = [
        (0, 3, 64), (2, 64, 64), (5, 64, 128), (7, 128, 128), (10, 128, 256), (12, 256, 256), (14, 256, 256),
        (16, 256, 256), (19, 256, 512), (21, 512, 512), (23, 512, 512), (25, 512, 512), (28, 512, 512),
    ]
    static let pools: Set<Int> = [4, 9, 18, 27]
    static let taps: Set<Int> = [1, 6, 11, 20, 29]
    static let layerCount = 30

    public init(weightsURL: URL?) throws {
        var layers: [Module] = (0 ..< Self.layerCount).map { _ in NFKVGGPassThrough() }
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
            let parts = key.split(separator: ".")
            guard parts.count == 3, parts[0] == "features", let index = Int(parts[1]), index < Self.layerCount else {
                return nil
            }
            let array = value.asType(.float32)
            return (key, checkpoint.needsConvTranspose && array.ndim == 4 ? array.transposed(0, 2, 3, 1) : array)
        }
        try NFKMLXWeights.apply(mapped, to: self, verifyShapes: true)
    }

    /// The five tapped activations of images `[N, H, W, 3]` already normalized for VGG.
    public func callAsFunction(_ x: MLXArray) -> [MLXArray] {
        var h = x
        var tapped: [MLXArray] = []
        for index in 0 ..< Self.layerCount {
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

/// The objective RIFE v4.15's `Model.update` minimizes, and the moving-average copy of the frame encoder it
/// scores features with.
///
/// @discussion With `merged_i` each block's frame, the teacher's frame and flow `merged_T` and `flow_T`, and
/// `decay(a_0…a_3) = ((((a_0)·0.8 + a_1)·0.8 + a_2)·0.8 + a_3)·0.8`:
/// - perceptual: the VGG-19 term on `merged_3` less 0.1 × SSIM (window 3, σ 1.5, stride 3), with each block's
///   `encode_loss` added before each decay step;
/// - L1: `decay(mean|merged_i − target|)`;
/// - teacher: `mean|merged_T − target|` plus 1e-5 × the mean flow magnitude of `flow_T`;
/// - consistency: for each block, 0.001 × the mean distance of its flow from `flow_T` (held constant) over the
///   pixels where its frame is farther from the target than the teacher's by more than 0.01, plus 1e-5 × the
///   mean flow magnitude of the last block.
///
/// The total is `perceptual + 0.1 · teacher + consistency + 0.05 · L1`.
///
/// `encode_loss` sums the mean absolute difference of the four encoder features of a frame and of the target,
/// computed by `targetEncoder`. The reference copies the network's encoder into it at the start and moves it
/// 1% of the way toward the network's after every update (`updateTarget(from:)`).
///
/// Introduced in InferKit 0.4.0.
public final class NFKMLXRIFEv4Objective {
    /// VGG-19 tap weights, `relu1_1` to `relu5_1`.
    public var perceptualWeights: [Float] = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
    public var ssimWeight: Float = 0.1
    public var teacherWeight: Float = 0.1
    public var l1Weight: Float = 0.05
    /// The factor each block's term is multiplied by after it is added.
    public var blockDecay: Float = 0.8
    public var consistencyWeight: Float = 0.001
    /// The weight of the flow-magnitude terms.
    public var flowMagnitudeWeight: Float = 1e-5
    /// How far the target encoder moves toward the network's at each update.
    public var targetRate: Float = 0.01

    public let perceptualNetwork: NFKMLXVGG19Features
    let targetEncoder: NFKRIFEv4Encoder

    /// Builds the objective over VGG-19 ImageNet weights (see ``NFKMLXVGG19Features``), with the target
    /// encoder a copy of `net`'s.
    public init(vggWeightsURL: URL?, net: NFKMLXRIFEv4Net) throws {
        perceptualNetwork = try NFKMLXVGG19Features(weightsURL: vggWeightsURL)
        targetEncoder = NFKRIFEv4Encoder()
        targetEncoder.update(parameters: ModuleParameters.unflattened(net.encode.parameters().flattened().map { ($0.0, $0.1 * 1) }))
        targetEncoder.freeze()
        eval(targetEncoder)
    }

    /// The loss of `net` interpolating `frame0` and `frame1` (`[N, H, W, 3]` in [0, 1], `H` and `W` multiples of
    /// 32) toward `middle` at `timestep` (`[N]`), with the blocks at `scales`.
    public func callAsFunction(_ net: NFKMLXRIFEv4Net, _ frame0: MLXArray, _ frame1: MLXArray, _ middle: MLXArray,
                               timestep: MLXArray, scales: [Int] = [8, 4, 2, 1]) -> MLXArray {
        loss(net.trainingPass(frame0, frame1, timestep: timestep, scales: scales), target: middle)
    }

    /// Moves the target encoder toward `net`'s, as the reference's `soft_update` does after each update.
    public func updateTarget(from net: NFKMLXRIFEv4Net) {
        let current = Dictionary(uniqueKeysWithValues: net.encode.parameters().flattened())
        let moved = targetEncoder.parameters().flattened().map { key, value in
            (key, value * (1 - targetRate) + current[key]! * targetRate)
        }
        targetEncoder.update(parameters: ModuleParameters.unflattened(moved))
        eval(targetEncoder)
    }

    func loss(_ pass: NFKRIFEv4TrainingPass, target: MLXArray) -> MLXArray {
        let terms = components(pass, target: target)
        return terms.perceptual + teacherWeight * terms.teacher + terms.consistency + l1Weight * terms.l1
    }

    func components(_ pass: NFKRIFEv4TrainingPass, target: MLXArray)
        -> (perceptual: MLXArray, teacher: MLXArray, consistency: MLXArray, l1: MLXArray) {
        var l1 = MLXArray(Float(0))
        for merged in pass.merged {
            l1 = (l1 + abs(merged - target).mean()) * blockDecay
        }
        let teacher = abs(pass.teacherMerged - target).mean() + flowMagnitude(pass.teacherFlow) * flowMagnitudeWeight

        let teacherFlow = stopGradient(pass.teacherFlow)
        let teacherError = abs(pass.teacherMerged - target).mean(axis: 3, keepDims: true)
        var consistency = MLXArray(Float(0))
        for (flow, merged) in zip(pass.flows, pass.merged) {
            let worse = stopGradient((abs(merged - target).mean(axis: 3, keepDims: true) .> teacherError + 0.01)
                .asType(flow.dtype))
            let distance = sqrt(square(teacherFlow - flow).sum(axis: 3, keepDims: true))
            consistency = consistency + (distance * worse).mean() * consistencyWeight
        }
        consistency = consistency + flowMagnitude(pass.flows[3]) * flowMagnitudeWeight

        var perceptual = vgg(pass.merged[3], target) - ssim(pass.merged[3], target) * ssimWeight
        for merged in pass.merged {
            perceptual = (perceptual + encoderDistance(merged, target)) * blockDecay
        }
        return (perceptual, teacher, consistency, l1)
    }

    /// The mean over pixels of `√(Σ_c (flow_c² + 1e-6))`.
    private func flowMagnitude(_ flow: MLXArray) -> MLXArray {
        sqrt((square(flow) + 1e-6).sum(axis: 3)).mean()
    }

    func vgg(_ x: MLXArray, _ y: MLXArray) -> MLXArray {
        let mean = MLXArray([Float(0.485), 0.456, 0.406]), deviation = MLXArray([Float(0.229), 0.224, 0.225])
        let ours = perceptualNetwork((x - mean) / deviation)
        let theirs = perceptualNetwork((y - mean) / deviation).map { stopGradient($0) }
        var total = MLXArray(Float(0))
        for (index, weight) in perceptualWeights.enumerated() {
            total = total + weight * abs(ours[index] - theirs[index]).mean()
        }
        return total
    }

    /// `ssim.py`'s `SSIM()`: a 3-tap Gaussian window of σ 1.5 per channel, stride 3, padding 1, constants
    /// 0.01² and 0.03², averaged over the map.
    func ssim(_ x: MLXArray, _ y: MLXArray) -> MLXArray {
        let channels = x.dim(3)
        let taps = (0 ..< 3).map { exp(-Double(($0 - 1) * ($0 - 1)) / (2 * 1.5 * 1.5)) }
        let sum = taps.reduce(0, +)
        let gauss = taps.map { Float($0 / sum) }
        let window2D = (0 ..< 3).flatMap { i in (0 ..< 3).map { j in gauss[i] * gauss[j] } }
        let window = broadcast(MLXArray(window2D, [1, 3, 3, 1]), to: [channels, 3, 3, 1])
        func filtered(_ v: MLXArray) -> MLXArray {
            conv2d(v, window, stride: IntOrPair(3), padding: IntOrPair(1), groups: channels)
        }
        let muX = filtered(x), muY = filtered(y)
        let sigmaX = filtered(x * x) - muX * muX
        let sigmaY = filtered(y * y) - muY * muY
        let sigmaXY = filtered(x * y) - muX * muY
        let c1: Float = 0.01 * 0.01, c2: Float = 0.03 * 0.03
        let map = ((2 * muX * muY + c1) * (2 * sigmaXY + c2)) / ((muX * muX + muY * muY + c1) * (sigmaX + sigmaY + c2))
        return map.mean()
    }

    /// `encode_loss`: the target encoder's four features of `x` against those of `y`, held constant.
    func encoderDistance(_ x: MLXArray, _ y: MLXArray) -> MLXArray {
        let ours = targetEncoder.features(x), theirs = targetEncoder.features(y)
        var total = MLXArray(Float(0))
        for (a, b) in zip(ours, theirs) {
            total = total + abs(a - stopGradient(b)).mean()
        }
        return total
    }
}

public extension NFKMLXRIFEv4 {

    /// `train.py`'s batch: one triplet on each of eight devices, doubled by the mirrored copies. Introduced in
    /// InferKit 0.4.0.
    static let referenceBatchSize = 8

    /// The crop `dataset.py` trains on, 448 pixels square. Introduced in InferKit 0.4.0.
    static let referenceImageSize = 448

    /// `train.py`'s warm-up, in updates. Introduced in InferKit 0.4.0.
    static let referenceWarmupSteps = 2000

    /// Builds RIFE v4 for training, inference, or reloading a trained checkpoint. Nil weights leave the
    /// network at its random initialization. Introduced in InferKit 0.4.0.
    static func network(weightsURL: URL?) throws -> NFKMLXRIFEv4Net {
        let net = makeNet()
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        return net
    }

    /// The block scales `model.py` draws for a training update from `draw`, uniform in [0, 1): `[4, 2, 1, 1]`
    /// below 0.3, `[2, 1, 1, 1]` below 0.6, and `[8, 4, 2, 1]` otherwise. Introduced in InferKit 0.4.0.
    static func referenceScales(draw: Float) -> [Int] {
        draw < 0.3 ? [4, 2, 1, 1] : draw < 0.6 ? [2, 1, 1, 1] : [8, 4, 2, 1]
    }

    /// The optimizer `model.py` builds: `torch.optim.AdamW` with weight decay 0.01 on every parameter, at
    /// `train.py`'s peak rate 1e-4. Introduced in InferKit 0.4.0.
    static func referenceOptimizer() -> Optimizer {
        NFKMLXReferenceOptimizers.adamW(learningRate: 1e-4, weightDecay: 1e-2)
    }

    /// `train.py`'s `get_learning_rate` over a run of `steps`, as a multiple of the peak rate: `k / warmup`
    /// for the first `warmup` updates, then a half cosine to zero at the run's end, plus 1e-7 / 1e-4 throughout.
    /// Introduced in InferKit 0.4.0.
    static func referenceSchedule(steps: Int, warmupSteps: Int = referenceWarmupSteps) -> NFKMLXLearningRateSchedule {
        NFKMLXLearningRateSchedule { step in
            let multiple: Float
            if step < warmupSteps {
                multiple = Float(step) / Float(warmupSteps)
            } else {
                let progress = Float(step - warmupSteps) / Float(max(steps - warmupSteps, 1))
                multiple = cos(progress * Float.pi) * 0.5 + 0.5
            }
            return multiple + 1e-3
        }
    }

    /// Fine-tunes RIFE v4 on frame triplets.
    ///
    /// - Parameters:
    ///   - net: the network, from ``network(weightsURL:)``.
    ///   - examples: supplies one batch per step: the outer frames `[N, H, W, 3]` in [0, 1], the frame between
    ///     them, and each triplet's timestep `[N]`, `H` and `W` multiples of 32. The reference crops 448 pixels
    ///     square (``referenceImageSize``) and flips, mirrors, and swaps the frames at random; those are the
    ///     caller's.
    ///   - objective: the loss and its target encoder, built over `net`.
    ///   - optimizer: the update rule. Nil uses ``referenceOptimizer()``.
    ///   - steps: how many updates to train for.
    ///   - mirrors: appends each batch's left-right mirror to it, as `train.py` doubles every batch.
    ///   - scaleSeed: seeds the block scales each update draws (``referenceScales(draw:)``), so a run repeats.
    ///   - clipGradientNorm: bounds the global gradient norm, 1 as the reference clips.
    ///   - accumulationSteps: how many batches each update averages; `steps` counts updates.
    ///   - precision: the precision the passes compute in; float32 by default.
    ///   - learningRateSchedule: the schedule over the run. Nil is ``referenceSchedule(steps:warmupSteps:)``
    ///     with the reference optimizer and a constant rate with a caller's.
    ///   - checkpoint: writes the network periodically.
    ///   - observer: receives each step and can end the run early.
    ///
    /// Every parameter trains. The objective's target encoder moves toward the network's after each update.
    /// Save with `NFKMLXWeights.save`; ``network(weightsURL:)`` and `backend(weightsURL:)` read the file back. A
    /// run is hours; call it off the render thread. Introduced in InferKit 0.4.0.
    @discardableResult
    static func fineTune(
        _ net: NFKMLXRIFEv4Net,
        examples: (Int) -> (frame0: MLXArray, frame1: MLXArray, middle: MLXArray, timestep: MLXArray),
        objective: NFKMLXRIFEv4Objective,
        optimizer: Optimizer? = nil,
        steps: Int,
        mirrors: Bool = true,
        scaleSeed: UInt64 = 0,
        clipGradientNorm: Float? = 1,
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
            referenceSchedule: { referenceSchedule(steps: steps) },
            steps: steps,
            arrays: { step in
                let example = examples(step)
                func doubled(_ x: MLXArray) -> MLXArray {
                    mirrors ? concatenated([x, x[0..., 0..., .stride(by: -1)]], axis: 0) : x
                }
                let timestep = mirrors ? concatenated([example.timestep, example.timestep], axis: 0) : example.timestep
                let draw = MLXRandom.uniform(low: 0, high: 1, [1], key: MLXRandom.key(scaleSeed &+ UInt64(step)))
                    .item(Float.self)
                let scales = MLXArray(referenceScales(draw: draw).map(Int32.init))
                return [doubled(example.frame0), doubled(example.frame1), doubled(example.middle), timestep, scales]
            },
            loss: { net, arrays in
                objective(net, arrays[0], arrays[1], arrays[2], timestep: arrays[3],
                          scales: arrays[4].asArray(Int32.self).map(Int.init))
            },
            clipGradientNorm: clipGradientNorm, accumulationSteps: accumulationSteps, precision: precision,
            learningRateSchedule: learningRateSchedule, checkpoint: checkpoint,
            constraint: { objective.updateTarget(from: $0) },
            observer: observer)
    }
}
