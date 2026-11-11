//
//  NFKMLXTrellisTraining.swift
//  InferKitMLX
//
//  TRELLIS customization: the structure encoder, the rectified-flow objective of the release's trainers
//  (`FlowMatchingCFGTrainer`, `SparseFlowMatchingCFGTrainer`), the fine-tuning recipes for both flows, and
//  the data adapter from a mesh to the structure stage's training latent.
//

import Foundation
import InferKit
import MLX
import MLXNN
import MLXOptimizers

// MARK: - Structure encoder

/// A 2³ convolution of stride 2 that halves each side.
final class NFKTrellisDownsample3d: Module {
    @ModuleInfo(key: "conv") var conv: Conv3d
    init(inputChannels: Int, outputChannels: Int) {
        _conv.wrappedValue = Conv3d(inputChannels: inputChannels, outputChannels: outputChannels, kernelSize: 2, stride: 2)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { conv(x) }
}

/// The encoder from a 64³ occupancy to the dense structure latent, the release's
/// `ss_enc_conv3d_16l8_fp16`: an input convolution, residual blocks with a stride-2 convolution between
/// widths, middle blocks, and a normalized, activated output convolution to the posterior's mean and
/// log-variance. Build one with `NFKMLXTrellis.structureEncoder(weightsURL:configuration:)`.
/// Introduced in InferKit 0.4.0.
public final class NFKMLXTrellisStructureEncoder: Module {
    @ModuleInfo(key: "input_layer") var inputLayer: Conv3d
    @ModuleInfo(key: "blocks") var blocks: [Module]
    @ModuleInfo(key: "middle_block") var middle: [NFKTrellisResidualBlock3d]
    @ModuleInfo(key: "out_layer") var outputLayers: [Module]

    public let configuration: NFKMLXTrellisConfiguration

    init(_ configuration: NFKMLXTrellisConfiguration) {
        self.configuration = configuration
        let channels = Array(configuration.decoderChannels.reversed())
        _inputLayer.wrappedValue = Conv3d(inputChannels: 1, outputChannels: channels[0], kernelSize: 3, padding: 1)
        var sequence = [Module]()
        for (index, width) in channels.enumerated() {
            sequence += (0 ..< configuration.decoderResidualBlocks).map { _ in NFKTrellisResidualBlock3d(channels: width) as Module }
            if index + 1 < channels.count {
                sequence.append(NFKTrellisDownsample3d(inputChannels: width, outputChannels: channels[index + 1]))
            }
        }
        _blocks.wrappedValue = sequence
        let last = channels[channels.count - 1]
        _middle.wrappedValue = (0 ..< configuration.decoderMiddleBlocks).map { _ in NFKTrellisResidualBlock3d(channels: last) }
        _outputLayers.wrappedValue = [LayerNorm(dimensions: last), NFKTrellisSlot(),
                                      Conv3d(inputChannels: last, outputChannels: configuration.structureChannels * 2,
                                             kernelSize: 3, padding: 1)]
    }

    /// The posterior's mean and log-variance, each `[N, R, R, R, structureChannels]`, for occupancies
    /// `[N, 4R, 4R, 4R, 1]` (1 occupied, 0 empty). The release encodes training latents as the mean.
    public func callAsFunction(_ occupancy: MLXArray) -> (mean: MLXArray, logVariance: MLXArray) {
        var h = inputLayer(occupancy)
        for block in blocks {
            if let residual = block as? NFKTrellisResidualBlock3d {
                h = residual(h)
            } else {
                h = (block as! NFKTrellisDownsample3d)(h)
            }
        }
        for block in middle {
            h = block(h)
        }
        let norm = outputLayers[0] as! LayerNorm, conv = outputLayers[2] as! Conv3d
        let parts = split(conv(silu(norm(h))), parts: 2, axis: -1)
        return (parts[0], parts[1])
    }
}

extension NFKMLXTrellis {
    /// The structure encoder with `ss_enc_conv3d_16l8_fp16.safetensors` at `weightsURL` loaded, at float32;
    /// nil builds random weights.
    public static func structureEncoder(weightsURL: URL?, configuration: NFKMLXTrellisConfiguration = .released) throws
        -> NFKMLXTrellisStructureEncoder {
        let encoder = NFKMLXTrellisStructureEncoder(configuration)
        if let weightsURL {
            try load(weightsURL, into: encoder) { _ in true }
        }
        return encoder
    }
}

// MARK: - Objective

/// The rectified-flow objective of the release's trainers: x_t = (1 − t) x₀ + (σ + (1 − σ) t) ε with
/// σ = 1e-5, the flow predicting the velocity (1 − σ) ε − x₀ at t · 1000, scored by mean squared error.
/// Introduced in InferKit 0.4.0.
public enum NFKMLXTrellisObjective {
    public static let sigmaMin: Float = 1e-5

    /// The noised sample at t (broadcast over every axis after the first).
    public static func diffuse(_ x0: MLXArray, noise: MLXArray, t: MLXArray) -> MLXArray {
        let shaped = t.reshaped([t.dim(0)] + Array(repeating: 1, count: x0.ndim - 1))
        return (1 - shaped) * x0 + (sigmaMin + (1 - sigmaMin) * shaped) * noise
    }

    /// The velocity the flow is trained to predict.
    public static func target(_ x0: MLXArray, noise: MLXArray) -> MLXArray {
        (1 - sigmaMin) * noise - x0
    }

    /// The mean squared error over every element.
    public static func loss(prediction: MLXArray, target: MLXArray) -> MLXArray {
        mean(square(prediction - target))
    }

    /// The structure flow's loss on structure latents `[N, R, R, R, C]`, conditions `[N, L, conditionWidth]`,
    /// noise like the latents, and times `[N]` in 0...1.
    public static func structureLoss(_ flow: NFKMLXTrellisStructureFlow, x0: MLXArray, condition: MLXArray, noise: MLXArray,
                                     t: MLXArray) -> MLXArray {
        let prediction = flow(diffuse(x0, noise: noise, t: t), t: t * 1000, condition: condition)
        return loss(prediction: prediction, target: target(x0, noise: noise))
    }

    /// The latent flow's loss on one structure's normalized latent features `[M, latentChannels]` on
    /// `voxels` `[M, 3]`, its condition `[1, L, conditionWidth]`, noise like the features, and a time `[1]`.
    public static func latentLoss(_ flow: NFKMLXTrellisLatentFlow, x0: MLXArray, voxels: MLXArray, condition: MLXArray,
                                  noise: MLXArray, t: MLXArray) -> MLXArray {
        let grid = NFKTrellisVoxelGrid(voxels, resolution: flow.configuration.occupancyResolution)
        let x = (1 - t) * x0 + (sigmaMin + (1 - sigmaMin) * t) * noise
        let prediction = flow.velocity(x, grid: grid, t: t * 1000, condition: condition)
        return loss(prediction: prediction, target: target(x0, noise: noise))
    }

    /// The trainers' time draw: σ(z) with z normal at mean 1 and deviation 1 (`logitNormal`).
    public static func sampleTimes(_ count: Int, key: MLXArray? = nil) -> MLXArray {
        sigmoid(MLXRandom.normal([count], key: key) + 1)
    }
}

// MARK: - Recipes

/// What a TRELLIS flow fine-tune trains. Introduced in InferKit 0.4.0.
public enum NFKMLXTrellisTrainable: Sendable, Equatable {
    /// Low-rank detours on every attention projection (self and cross), everything else frozen.
    case attentionLoRA(rank: Int, alpha: Float)
    /// Every weight of the flow, as the release's trainers run.
    case all
}

/// One structure-stage training example: a condition and the structure latent of the object it shows.
/// Introduced in InferKit 0.4.0.
public struct NFKMLXTrellisStructureExample {
    /// `[1, L, conditionWidth]`, from ``NFKMLXTrellisPipeline/condition(prepared:)`` or `condition(prompt:)`.
    public let condition: MLXArray
    /// `[1, R, R, R, structureChannels]`, from ``NFKMLXTrellisTraining/structureLatent(of:encoder:)``.
    public let latent: MLXArray
    public init(condition: MLXArray, latent: MLXArray) {
        self.condition = condition
        self.latent = latent
    }
}

/// One latent-stage training example: a condition and a normalized structured latent on its voxels.
/// Introduced in InferKit 0.4.0.
public struct NFKMLXTrellisLatentExample {
    public let condition: MLXArray
    /// `[M, 3]` on the 64³ grid.
    public let voxels: MLXArray
    /// `[M, latentChannels]`, normalized as the flow samples it (before the release's denormalization).
    public let latent: MLXArray
    public init(condition: MLXArray, voxels: MLXArray, latent: MLXArray) {
        self.condition = condition
        self.voxels = voxels
        self.latent = latent
    }
}

/// TRELLIS's fine-tuning recipes and data adapter. Introduced in InferKit 0.4.0.
public enum NFKMLXTrellisTraining {
    /// The release trainers' optimizer: AdamW at 1e-4 without weight decay, constant rate.
    public static func referenceOptimizer() -> Optimizer {
        NFKMLXAdam(learningRate: 1e-4)
    }

    /// The release trainers' classifier-free dropout: the condition replaced by zeros with this probability.
    public static let unconditionalProbability: Float = 0.1

    /// Applies `trainable` to a flow: LoRA detours on the attention projections, or every weight.
    static func freeze(_ flow: Module, _ trainable: NFKMLXTrellisTrainable) throws {
        switch trainable {
        case .all:
            flow.unfreeze()
        case .attentionLoRA(let rank, let alpha):
            let projections: Set<String> = ["to_qkv", "to_out", "to_q", "to_kv"]
            try NFKMLXLoRA.apply(to: flow, rank: rank, alpha: alpha) { path, _ in
                projections.contains(String(path.split(separator: ".").last ?? ""))
            }
        }
    }

    /// The condition each step trains on: the example's, or zeros with ``unconditionalProbability``.
    static func dropped(_ condition: MLXArray, key: MLXArray) -> MLXArray {
        MLXRandom.uniform(0 ..< 1, [1], key: key).item(Float.self) < unconditionalProbability ? MLXArray.zeros(like: condition) : condition
    }

    /// Fine-tunes the structure flow on examples, one per step (cycling).
    ///
    /// - Parameters:
    ///   - flow: the flow, from ``NFKMLXTrellis/structureFlow(weightsURL:configuration:)``.
    ///   - examples: the conditions and structure latents; ``structureLatent(of:encoder:)`` makes a latent
    ///     from a mesh.
    ///   - trainable: what trains; LoRA on the attention projections by default, which a device holds.
    ///   - optimizer: the update rule. Nil uses ``referenceOptimizer()``.
    ///   - steps: how many updates.
    ///   - seed: seeds the noise, the times, and the condition dropout.
    ///   - clipGradientNorm: bounds the global gradient norm, the trainers' 1.0 by default.
    ///   - observer: receives each step and can end the run early.
    ///
    /// Save with `NFKMLXWeights.save` (after ``NFKMLXLoRA/merge(into:)`` for LoRA, or its adapters alone);
    /// `NFKMLXTrellis.structureFlow(weightsURL:)` and the backend factories' `structureFlowWeightsURL` read
    /// the file back. A run is minutes; call it off the render thread.
    @discardableResult
    public static func fineTuneStructureFlow(_ flow: NFKMLXTrellisStructureFlow, examples: [NFKMLXTrellisStructureExample],
                                             trainable: NFKMLXTrellisTrainable = .attentionLoRA(rank: 16, alpha: 16),
                                             optimizer: Optimizer? = nil, steps: Int, seed: UInt64 = 0, clipGradientNorm: Float? = 1,
                                             observer: NFKMLXTrainer.Observer? = nil) throws -> [Float] {
        guard !examples.isEmpty else { throw NFKMLXError.nothingToTrain("no training examples") }
        return try NFKMLXFineTune.run(
            flow, freezing: { try freeze(flow, trainable) }, optimizer: optimizer, reference: referenceOptimizer,
            referenceSchedule: { .constant }, steps: steps,
            arrays: { step in
                let example = examples[step % examples.count]
                let keys = MLXRandom.split(key: MLXRandom.key(seed &+ UInt64(step)), into: 3)
                return [example.latent, dropped(example.condition, key: keys[0]),
                        MLXRandom.normal(example.latent.shape, key: keys[1]), NFKMLXTrellisObjective.sampleTimes(1, key: keys[2])]
            },
            loss: { flow, arrays in
                NFKMLXTrellisObjective.structureLoss(flow, x0: arrays[0], condition: arrays[1], noise: arrays[2], t: arrays[3])
            },
            clipGradientNorm: clipGradientNorm, observer: observer)
    }

    /// Fine-tunes the latent flow on examples, one structure per step (cycling). The structured latents
    /// come from the release's offline encoder (`dataset_toolkits/encode_latent.py`), which reads DINOv2
    /// features rendered from 150 views; the arguments follow ``fineTuneStructureFlow(_:examples:trainable:optimizer:steps:seed:clipGradientNorm:observer:)``.
    @discardableResult
    public static func fineTuneLatentFlow(_ flow: NFKMLXTrellisLatentFlow, examples: [NFKMLXTrellisLatentExample],
                                          trainable: NFKMLXTrellisTrainable = .attentionLoRA(rank: 16, alpha: 16),
                                          optimizer: Optimizer? = nil, steps: Int, seed: UInt64 = 0, clipGradientNorm: Float? = 1,
                                          observer: NFKMLXTrainer.Observer? = nil) throws -> [Float] {
        guard !examples.isEmpty else { throw NFKMLXError.nothingToTrain("no training examples") }
        return try NFKMLXFineTune.run(
            flow, freezing: { try freeze(flow, trainable) }, optimizer: optimizer, reference: referenceOptimizer,
            referenceSchedule: { .constant }, steps: steps,
            arrays: { step in
                let example = examples[step % examples.count]
                let keys = MLXRandom.split(key: MLXRandom.key(seed &+ UInt64(step)), into: 3)
                return [example.latent, example.voxels, dropped(example.condition, key: keys[0]),
                        MLXRandom.normal(example.latent.shape, key: keys[1]), NFKMLXTrellisObjective.sampleTimes(1, key: keys[2])]
            },
            loss: { flow, arrays in
                NFKMLXTrellisObjective.latentLoss(flow, x0: arrays[0], voxels: arrays[1], condition: arrays[2], noise: arrays[3],
                                                  t: arrays[4])
            },
            clipGradientNorm: clipGradientNorm, observer: observer)
    }

    // MARK: Data

    /// The structure latent `[1, R, R, R, structureChannels]` of a mesh: its occupancy encoded to the
    /// posterior's mean, as the release encodes its training set.
    public static func structureLatent(of mesh: NFKMesh, encoder: NFKMLXTrellisStructureEncoder) -> MLXArray {
        encoder(occupancy(of: mesh, resolution: encoder.configuration.occupancyResolution)).mean
    }

    /// A mesh's surface voxels on a `resolution`³ grid `[1, R, R, R, 1]` (1 where a triangle passes through
    /// the voxel). The mesh is centered and scaled so its longest side spans the grid's cube, then turned
    /// from glTF's frame (y up) to the decoders' (z up), as the release's renders normalize an asset.
    /// Each triangle is sampled densely enough that no voxel it crosses is skipped; a sample's cell is the
    /// floor of its grid coordinate (half-open cells, the far face clamped into the last), and samples
    /// are placed as a + (b − a) u + (c − a) v so a triangle lying in a cell face stays on it.
    public static func occupancy(of mesh: NFKMesh, resolution: Int) -> MLXArray {
        var grid = [Float](repeating: 0, count: resolution * resolution * resolution)
        let count = mesh.vertexCount
        guard count > 0 else { return MLXArray(grid).reshaped([1, resolution, resolution, resolution, 1]) }
        let low = mesh.boundingBoxMinimum, high = mesh.boundingBoxMaximum
        let center = ((low.x + high.x) / 2, (low.y + high.y) / 2, (low.z + high.z) / 2)
        let extent = max(high.x - low.x, max(high.y - low.y, high.z - low.z))
        let scale = extent > 0 ? 1 / extent : 1
        func voxel(_ v: NFKMeshVector) -> (Float, Float, Float) {
            let (x, y, z) = ((v.x - center.0) * scale, (v.y - center.1) * scale, (v.z - center.2) * scale)
            return ((x + 0.5) * Float(resolution), (-z + 0.5) * Float(resolution), (y + 0.5) * Float(resolution))
        }
        func mark(_ p: (Float, Float, Float)) {
            let index = [p.0, p.1, p.2].map { min(resolution - 1, max(0, Int($0.rounded(.down)))) }
            grid[(index[0] * resolution + index[1]) * resolution + index[2]] = 1
        }
        for face in 0 ..< mesh.faceCount {
            let triangle = mesh.face(at: face)
            let a = voxel(mesh.position(at: UInt(triangle.a))), b = voxel(mesh.position(at: UInt(triangle.b)))
            let c = voxel(mesh.position(at: UInt(triangle.c)))
            func length(_ p: (Float, Float, Float), _ q: (Float, Float, Float)) -> Float {
                ((p.0 - q.0) * (p.0 - q.0) + (p.1 - q.1) * (p.1 - q.1) + (p.2 - q.2) * (p.2 - q.2)).squareRoot()
            }
            let divisions = max(1, Int((max(length(a, b), max(length(b, c), length(c, a))) * 2).rounded(.up)))
            for i in 0 ... divisions {
                for j in 0 ... (divisions - i) {
                    let u = Float(i) / Float(divisions), v = Float(j) / Float(divisions)
                    mark((a.0 + (b.0 - a.0) * u + (c.0 - a.0) * v, a.1 + (b.1 - a.1) * u + (c.1 - a.1) * v,
                          a.2 + (b.2 - a.2) * u + (c.2 - a.2) * v))
                }
            }
        }
        return MLXArray(grid).reshaped([1, resolution, resolution, resolution, 1])
    }
}
