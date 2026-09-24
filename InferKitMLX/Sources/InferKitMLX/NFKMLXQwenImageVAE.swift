//
//  NFKMLXQwenImageVAE.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXNN

// Qwen-Image 2.1's autoencoder (`AutoencoderKLQwenImage21`) is the Wan 2.2 residual VAE
// ``NFKMLXWanVideoVAENet`` already runs, SPECIALIZED TO ONE FRAME. In the reference its causal 3-D
// convolution subclasses `nn.Conv2d`: it squeezes the temporal axis away, runs a 2-D convolution,
// unsqueezes, and raises if handed a feature cache. So the checkpoint's convolution weights are 4-D,
// every temporal kernel is 1, and the resamplers' temporal branches never execute — a single frame
// leaves their cache slots empty, which is the branch that skips them.
//
// What this file adds is the configuration, the weight layout, and the latent normalization; the
// network is the shipped one under ``NFKMLXWanVAEConfiguration/imageOnly``.

/// Building and loading the Qwen-Image 2.1 autoencoder.
@objc(NFKMLXQwenImageVAE)
public final class NFKMLXQwenImageVAE: NSObject {

    /// A name for the model the factories produce.
    @objc public static let modelName = "qwen-image-2.1-vae"

    /// Builds the autoencoder at a geometry.
    public static func makeNet(_ configuration: NFKMLXWanVAEConfiguration = .qwenImage21)
        -> NFKMLXWanVideoVAENet {
        NFKMLXWanVideoVAENet(configuration)
    }

    /// The geometry a release's `vae/config.json` describes.
    public static func configuration(fromHuggingFace url: URL) throws -> NFKMLXWanVAEConfiguration {
        let data = try Data(contentsOf: url)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw NFKMLXError.unsupportedConfiguration("the autoencoder config is not an object")
        }
        var configuration = NFKMLXWanVAEConfiguration.qwenImage21
        configuration.baseDim = json["base_dim"] as? Int ?? configuration.baseDim
        configuration.decoderBaseDim = json["decoder_base_dim"] as? Int ?? configuration.decoderBaseDim
        configuration.zDim = json["z_dim"] as? Int ?? configuration.zDim
        if let mult = json["dim_mult"] as? [Int] { configuration.dimMult = mult }
        configuration.numResBlocks = json["num_res_blocks"] as? Int ?? configuration.numResBlocks
        if let temporal = json["temperal_downsample"] as? [Bool] {
            configuration.temporalDownsample = temporal
        }
        // A null `patch_size` is no patchify, which is what this release carries.
        configuration.patchSize = json["patch_size"] as? Int ?? 1
        configuration.inChannels = json["in_channels"] as? Int ?? configuration.inChannels
        configuration.isResidual = (json["is_residual"] as? NSNumber)?.boolValue ?? configuration.isResidual
        return configuration
    }

    /// The per-channel latent mean and standard deviation the release records, shaped to broadcast over
    /// an NDHWC latent.
    ///
    /// @discussion The pipeline works in a normalized latent space: an encoded image is
    /// `(latent - mean) / std` and a sampled latent is `latent * std + mean` before it is decoded.
    /// Skipping this leaves a decode that runs and produces a wrong image.
    public static func latentStatistics(fromHuggingFace url: URL) throws
        -> (mean: MLXArray, standardDeviation: MLXArray) {
        let data = try Data(contentsOf: url)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let mean = json["latents_mean"] as? [Double],
              let standardDeviation = json["latents_std"] as? [Double] else {
            throw NFKMLXError.unsupportedConfiguration("the autoencoder config carries no latent statistics")
        }
        let shape = [1, 1, 1, 1, mean.count]
        return (MLXArray(mean.map { Float($0) }).reshaped(shape),
                MLXArray(standardDeviation.map { Float($0) }).reshaped(shape))
    }

    /// The release's key and layout translated to the module's.
    ///
    /// The convolutions are stored `[out, in, kh, kw]`. A spatial `Conv2d` takes `[out, kh, kw, in]`,
    /// and a causal convolution takes `[out, kt, kh, kw, in]` with a temporal kernel of 1, which is
    /// the same weight with an axis inserted. The RMS normalizations store their scale with trailing
    /// singleton axes, which flatten.
    public static func adapted(key: String, value: MLXArray) -> (String, MLXArray) {
        if key.hasSuffix(".gamma") { return (key, value.reshaped([-1])) }
        guard value.ndim == 4 else { return (key, value) }
        let spatial = key.contains(".resample.1.") || key.contains(".to_qkv.") || key.contains(".proj.")
        let transposed = value.transposed(0, 2, 3, 1)
        return (key, spatial ? transposed : transposed.expandedDimensions(axis: 1))
    }

    /// Loads a release's `vae/` weights.
    public static func loadWeights(into net: NFKMLXWanVideoVAENet, fromDirectory directory: URL,
                                   precision: NFKMLXWeightPrecision = .float32) throws {
        let arrays = try NFKMLXReleaseWeights.arrays(inDirectory: directory, precision: precision)
        try NFKMLXWeights.apply(arrays.map { adapted(key: $0.0, value: $0.1) }, to: net,
                                verifyShapes: true)
    }

    /// Builds the autoencoder from a release's `vae/` directory, reading its own `config.json`.
    public static func net(directoryURL: URL, precision: NFKMLXWeightPrecision = .float32) throws
        -> NFKMLXWanVideoVAENet {
        let configuration = try configuration(
            fromHuggingFace: directoryURL.appendingPathComponent("config.json"))
        let net = makeNet(configuration)
        try loadWeights(into: net, fromDirectory: directoryURL, precision: precision)
        return net
    }
}
