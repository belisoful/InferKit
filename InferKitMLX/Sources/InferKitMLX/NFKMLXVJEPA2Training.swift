//
//  NFKMLXVJEPA2Training.swift
//  InferKitMLX
//
//  Fine-tuning V-JEPA 2 the way its reference evaluates it: an attentive probe trained on the frozen
//  encoder's features (`evals/video_classification_frozen` in facebookresearch/vjepa2). The reference
//  runs the encoder under `no_grad` and trains an `AttentiveClassifier`, which is the classification
//  releases' `pooler` plus `classifier`.
//

import Foundation
import CoreGraphics
import MLX
import MLXNN
import MLXOptimizers
import MLXRandom

/// Which parameters a V-JEPA 2 fine-tune updates. The encoder stays frozen in both.
///
/// Introduced in InferKit 0.4.0.
public enum NFKMLXVJEPA2Trainable: Sendable {
    /// The attentive pooler and the classifier: the reference's frozen-encoder probe.
    case probe
    /// The linear classifier alone, over the pooler's output.
    case classifier
}

/// The objective a V-JEPA 2 probe minimizes: the reference's `torch.nn.CrossEntropyLoss()`.
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXVJEPA2Objective: Sendable {
    public init() {}

    /// Scores `net` on a normalized clip `[B, frames, crop, crop, 3]` against class indices `[B]`. The
    /// network must carry a classifier (``NFKMLXVJEPA2/network(directoryURL:labels:)``).
    public func callAsFunction(_ net: NFKMLXVJEPA2Net, _ clip: MLXArray, _ labels: MLXArray) -> MLXArray {
        guard let logits = net.classLogits(clip) else {
            preconditionFailure("a V-JEPA 2 probe needs a classifier; build the network with labels")
        }
        return loss(logits: logits, labels: labels)
    }

    /// The mean over the batch of the negative log-softmax at each label, from class logits `[B, classes]`.
    ///
    /// @discussion The reference sums this loss over the temporal segments and spatial views of each
    /// clip before its backward pass; one view per example is the single-term case.
    public func loss(logits: MLXArray, labels: MLXArray) -> MLXArray {
        crossEntropy(logits: logits, targets: labels, reduction: .mean)
    }
}

extension NFKMLXVJEPA2Processor {
    /// A normalized clip `[1, frames, crop, crop, 3]` from video frames, as the backend builds one: the
    /// release's frame count, crop, and shortest edge from `configuration`. A single image becomes a
    /// one-tubelet clip.
    ///
    /// Introduced in InferKit 0.4.0.
    public static func clip(frames: [CGImage], configuration: NFKMLXVJEPA2Configuration) throws -> MLXArray {
        guard !frames.isEmpty else { throw NFKMLXError.unsupportedInput }
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let tensors = frames.map { frame -> MLXArray in
            let (bytes, width, height) = NFKMLXImageBridge.rgbaBytes(from: frame, colorSpace: colorSpace)
            return NFKMLXImageBridge.tensor(rgba: bytes, width: width, height: height, channels: 3)
        }
        return clip(from: tensors, frameCount: configuration.framesPerClip, cropSize: configuration.cropSize,
                    tubeletSize: configuration.tubeletSize, shortestEdge: configuration.shortestEdge)
    }
}

extension NFKMLXVJEPA2 {

    /// Builds the network itself, ready to fine-tune, from a released directory.
    ///
    /// - Parameters:
    ///   - directoryURL: a V-JEPA 2 release, or a directory ``save(_:toDirectoryURL:)`` wrote.
    ///   - labels: the consumer's own classes. Nil keeps the release's (none for an encoder release).
    ///     With labels, a release without an attentive pooler gets a fresh one, initialized as the
    ///     reference's `AttentivePooler` is; a classifier whose class count differs is left at its fresh
    ///     initialization, and everything else loads.
    ///
    /// @discussion Dropping a differently sized classifier is required: MLX's `update(parameters:)`
    /// adopts a checkpoint's shapes rather than validating them, so the release's classifier would
    /// replace the consumer's and the model would rank the old classes.
    ///
    /// Introduced in InferKit 0.4.0.
    public static func network(directoryURL: URL, labels: [String]? = nil) throws -> NFKMLXVJEPA2Net {
        let release = try NFKMLXVJEPA2Configuration(configurationURL: directoryURL.appendingPathComponent("config.json"))
        guard let labels, labels != release.labels else {
            let net = NFKMLXVJEPA2Net(release)
            try net.loadWeights(fromDirectory: directoryURL)
            return net
        }
        guard !labels.isEmpty else {
            throw NFKMLXError.unsupportedConfiguration("a V-JEPA 2 probe needs at least one class")
        }
        var configuration = release
        configuration.labels = labels
        if configuration.poolerLayers == 0 {
            configuration.poolerLayers = 3
        }
        let net = NFKMLXVJEPA2Net(configuration)
        var fresh = ["classifier."]
        if release.labels.isEmpty, let pooler = net.pooler {
            initializeAsReference(pooler, layers: configuration.poolerLayers)
            fresh.append("pooler.")
        }
        let keepsClassifier = release.labels.count == labels.count
        if keepsClassifier {
            fresh.removeAll { $0 == "classifier." }
        }
        try net.loadWeights(fromDirectory: directoryURL, leavingFresh: fresh)
        return net
    }

    /// The reference's `AttentivePooler` initialization: every linear weight and the query token drawn
    /// from a normal of standard deviation 0.02 (timm's `trunc_normal_` truncates at ±2, a hundred
    /// standard deviations out), biases zero, and each residual branch's last projection divided by
    /// `√(2·(i + 1))` for self-attention layer `i`, the cross-attention layer's by the last layer's
    /// factor. Layer norms keep their unit scale and zero shift.
    static func initializeAsReference(_ pooler: NFKVJEPA2AttentivePooler, layers: Int) {
        var parameters = [(String, MLXArray)]()
        for (path, module) in pooler.leafModules().flattened() {
            guard let linear = module as? Linear else { continue }
            parameters.append((path + ".weight", MLXRandom.normal(linear.weight.shape) * 0.02))
            if linear.bias != nil {
                parameters.append((path + ".bias", MLXArray.zeros([linear.weight.dim(0)])))
            }
        }
        parameters.append(("query_tokens", MLXRandom.normal(pooler.queryTokens.shape) * 0.02))
        var rescaled = Dictionary(uniqueKeysWithValues: parameters)
        for layer in 0 ..< layers {
            let scale = Float(2 * (layer + 1)).squareRoot()
            for name in ["self_attention_layers.\(layer).self_attn.out_proj.weight",
                         "self_attention_layers.\(layer).mlp.fc2.weight"] {
                rescaled[name] = rescaled[name]! / scale
            }
        }
        let crossName = "cross_attention_layer.mlp.fc2.weight"
        rescaled[crossName] = rescaled[crossName]! / Float(2 * max(layers, 1)).squareRoot()
        pooler.update(parameters: ModuleParameters.unflattened(Array(rescaled)))
    }

    /// Fine-tunes the probe on a consumer's own labeled clips, returning the loss from each step.
    ///
    /// The whole customization path is three calls: ``network(directoryURL:labels:)`` to build, this to
    /// train, and ``save(_:toDirectoryURL:)`` to write a directory that ``backend(directoryURL:)`` loads
    /// like any release.
    ///
    /// - Parameters:
    ///   - net: the network to train; it must carry a classifier.
    ///   - examples: supplies one labeled clip per step: a clip from
    ///     ``NFKMLXVJEPA2Processor/clip(frames:configuration:)`` and its class index.
    ///   - trainable: which parameters update. Freezing is applied here and persists on `net`.
    ///   - objective: the supervised loss.
    ///   - optimizer: the update rule. Nil uses the reference's `torch.optim.AdamW` (bias-corrected,
    ///     betas 0.9 and 0.999) at `learningRate`, with `weightDecay` on every probe parameter.
    ///   - learningRate: the reference optimizer's peak rate. The reference sweeps twenty heads over
    ///     5e-3, 3e-3, 1e-3, 3e-4, and 1e-4 against weight decays 0.01, 0.1, 0.4, and 0.8, and keeps
    ///     the best on validation; the default is the first of them.
    ///   - weightDecay: the reference optimizer's weight decay, held constant as the reference holds it
    ///     when its start and end values match.
    ///   - steps: how many examples to train on.
    ///   - clipGradientNorm: bounds the global gradient norm before the update. The reference does not clip.
    ///   - learningRateSchedule: multiplies the rate at each step. Nil uses the reference's
    ///     `WarmupCosineLRSchedule` with no warm-up, a cosine to zero over the run, when the reference
    ///     optimizer runs. With a caller's optimizer, nil holds that optimizer's rate constant.
    ///   - checkpoint: writes the network periodically, so a suspended run keeps its progress.
    ///   - observer: receives each step and can end the run early.
    ///
    /// A run is multi-second; call it off the render thread.
    ///
    /// Introduced in InferKit 0.4.0.
    @discardableResult
    public static func fineTune(
        _ net: NFKMLXVJEPA2Net,
        examples: (Int) -> (clip: MLXArray, label: Int),
        trainable: NFKMLXVJEPA2Trainable = .probe,
        objective: NFKMLXVJEPA2Objective = NFKMLXVJEPA2Objective(),
        optimizer: Optimizer? = nil,
        learningRate: Float = 5e-3,
        weightDecay: Float = 0.01,
        steps: Int,
        clipGradientNorm: Float? = nil,
        learningRateSchedule: NFKMLXLearningRateSchedule? = nil,
        checkpoint: NFKMLXTrainingCheckpoint? = nil,
        observer: NFKMLXTrainer.Observer? = nil
    ) throws -> [Float] {
        guard net.classifies else {
            throw NFKMLXError.unsupportedConfiguration(
                "this V-JEPA 2 network has no classifier; build it with network(directoryURL:labels:)")
        }
        return try NFKMLXFineTune.run(
            net,
            freezing: { apply(trainable, to: net) },
            optimizer: optimizer,
            reference: { NFKMLXReferenceOptimizers.adamW(learningRate: learningRate,
                                                         weightDecay: weightDecay) },
            referenceSchedule: { .warmupCosine(steps: steps) },
            steps: steps,
            batch: { step in
                let example = examples(step)
                return (example.clip, MLXArray([Int32(example.label)]))
            },
            loss: objective.callAsFunction,
            clipGradientNorm: clipGradientNorm,
            learningRateSchedule: learningRateSchedule,
            checkpoint: checkpoint, observer: observer)
    }

    /// Writes `net` as a release directory: `model.safetensors` in the module's own layout, and the
    /// `config.json` and `video_preprocessor_config.json` its configuration reads back from, the labels
    /// as `id2label`. ``backend(directoryURL:)`` and ``network(directoryURL:labels:)`` load it.
    ///
    /// Introduced in InferKit 0.4.0.
    public static func save(_ net: NFKMLXVJEPA2Net, toDirectoryURL directory: URL) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try NFKMLXWeights.save(net, to: directory.appendingPathComponent("model.safetensors"))
        let c = net.configuration
        var config: [String: Any] = [
            "model_type": "vjepa2", "hidden_size": c.hiddenSize, "num_hidden_layers": c.numHiddenLayers,
            "num_attention_heads": c.numAttentionHeads, "mlp_ratio": c.mlpRatio, "patch_size": c.patchSize,
            "tubelet_size": c.tubeletSize, "frames_per_clip": c.framesPerClip, "crop_size": c.cropSize,
            "in_chans": c.inChannels, "layer_norm_eps": Double(c.layerNormEps), "qkv_bias": c.qkvBias,
            "architectures": [c.labels.isEmpty ? "VJEPA2Model" : "VJEPA2ForVideoClassification"],
        ]
        if !c.labels.isEmpty {
            config["num_pooler_layers"] = c.poolerLayers
            config["id2label"] = Dictionary(uniqueKeysWithValues: c.labels.enumerated().map { (String($0.offset), $0.element) })
            config["label2id"] = Dictionary(uniqueKeysWithValues: c.labels.enumerated().map { ($0.element, $0.offset) })
        }
        let processor: [String: Any] = ["size": ["shortest_edge": c.shortestEdge],
                                        "crop_size": ["height": c.cropSize, "width": c.cropSize]]
        for (name, object) in [("config.json", config), ("video_preprocessor_config.json", processor)] {
            try JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys])
                .write(to: directory.appendingPathComponent(name))
        }
    }

    /// Freezes everything but the parameters `trainable` names.
    private static func apply(_ trainable: NFKMLXVJEPA2Trainable, to net: NFKMLXVJEPA2Net) {
        net.freeze()
        switch trainable {
        case .probe:
            net.pooler?.unfreeze()
            net.classifier?.unfreeze()
        case .classifier:
            net.classifier?.unfreeze()
        }
    }
}
