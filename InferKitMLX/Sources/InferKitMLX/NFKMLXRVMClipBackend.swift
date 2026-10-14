//
//  NFKMLXRVMClipBackend.swift
//  InferKitMLX
//
//  Robust Video Matting over whole clips: the recurrent state the decoder's GRUs keep is carried from
//  each frame to the next, which is what the model is trained to read.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import InferKit
import MLX

/// What the Robust Video Matting clip backend writes.
///
/// Introduced in InferKit 0.4.0.
@objc(NFKMLXRVMClipOutput)
public enum NFKMLXRVMClipOutput: Int {
    /// The alpha matte as gray, white on the subject.
    case alpha
    /// The predicted foreground colors.
    case foreground
    /// The foreground over the reference's green background, `(120, 255, 155)`.
    case composition
}

/// Holds the network for capture in the clip backend's transform.
private final class NFKRVMClipHolder: @unchecked Sendable {
    let net: NFKMLXRVMNet
    init(_ net: NFKMLXRVMNet) { self.net = net }
}

extension NFKMLXRVM {

    /// Mattes `frames` (`[H, W, 3]` in `0...1`) in order, carrying the recurrent state from each frame to
    /// the next, and returns one `[H, W, 3]` frame of `output` per input.
    ///
    /// @discussion Each frame runs at the reference's automatic downsample ratio, `min(512 / max(H, W), 1)`,
    /// with the deep guided filter lifting the result back to the frame's size. Each frame's result and
    /// state are evaluated before the next frame reads them, so the graph stays one frame deep.
    static func matteClip(_ frames: [MLXArray], net: NFKMLXRVMNet, output: NFKMLXRVMClipOutput) -> [MLXArray] {
        var state = NFKMLXRVMNet.initialState
        let green = MLXArray([Float(120) / 255, 1, Float(155) / 255])
        return frames.map { frame in
            let (height, width) = (frame.dim(0), frame.dim(1))
            let ratio = Swift.min(512 / Float(Swift.max(height, width)), 1)
            let (foreground, alpha, next) = net.forward(frame.reshaped([1, height, width, 3]), state: state,
                                                        downsampleRatio: ratio)
            state = next
            let written: MLXArray
            switch output {
            case .alpha:
                written = concatenated([alpha, alpha, alpha], axis: 3)
            case .foreground:
                written = foreground
            case .composition:
                written = foreground * alpha + green * (1 - alpha)
            }
            let result = written.reshaped([height, width, 3])
            eval([result] + state.compactMap { $0 })
            return result
        }
    }

    /// The clip backend over `net`, reporting `identifier`.
    static func clipBackend(net: NFKMLXRVMNet, identifier: String, output: NFKMLXRVMClipOutput) -> any NFKInferenceBackend {
        let holder = NFKRVMClipHolder(net)
        return NFKMLXVideoBackend(identifier: identifier) { frames in
            matteClip(frames, net: holder.net, output: output)
        }
    }

    /// The registry name and identifier of a clip backend: the encoder's name with `-clip`
    /// (`robust-video-matting-clip`, `robust-video-matting-resnet50-clip`). Introduced in InferKit 0.4.0.
    @objc(clipModelNameForVariant:)
    public static func clipModelName(variant: NFKMLXRVMVariant) -> String {
        "\(specs(for: variant).name)-clip"
    }

    /// Builds a backend that mattes a whole video clip with one of the released encoders.
    ///
    /// @discussion Reads an `NFKVideoAsset` under `NFKInputVideo` and returns one under `NFKOutputVideo`,
    /// at the source's size and frame rate, writing `output` for every frame. The decoder's recurrent
    /// state carries from each frame to the next, so a frame's matte draws on the frames before it, as
    /// the reference's video inference runs. A nil `weightsURL` builds random weights. Run inference off
    /// the render thread; a clip is one forward pass per frame. Introduced in InferKit 0.4.0.
    @objc(clipBackendWithVariant:output:weightsURL:error:)
    public static func clipBackend(variant: NFKMLXRVMVariant, output: NFKMLXRVMClipOutput,
                                   weightsURL: URL?) throws -> any NFKInferenceBackend {
        let net = NFKMLXRVMNet(specs(for: variant).configuration)
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        net.train(false)
        return clipBackend(net: net, identifier: clipModelName(variant: variant), output: output)
    }

    /// ``clipBackend(variant:output:weightsURL:)`` with the MobileNetV3 encoder, writing the alpha matte.
    /// Introduced in InferKit 0.4.0.
    @objc(clipBackendWithWeightsURL:error:)
    public static func clipBackend(weightsURL: URL?) throws -> any NFKInferenceBackend {
        try clipBackend(variant: .mobileNetV3, output: .alpha, weightsURL: weightsURL)
    }

    /// Downloads the checkpoint from Hugging Face, then builds the clip backend. A file already in the
    /// cache is not fetched again. The call blocks on the network, so run it off the render thread.
    /// Introduced in InferKit 0.4.0.
    @objc(clipBackendWithVariant:output:repo:weightsPath:revision:cacheDirectoryURL:error:)
    public static func clipBackend(variant: NFKMLXRVMVariant, output: NFKMLXRVMClipOutput, repo: String,
                                   weightsPath: String, revision: String?,
                                   cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        let url = try NFKMLXDownload.weightsURL(repo: repo, weightsPath: weightsPath, revision: revision,
                                                cacheDirectoryURL: cacheDirectoryURL)
        return try clipBackend(variant: variant, output: output, weightsURL: url)
    }

    /// The asynchronous form of
    /// ``clipBackend(variant:output:repo:weightsPath:revision:cacheDirectoryURL:)``. The handler runs on the
    /// download's background queue. Introduced in InferKit 0.4.0.
    @objc(clipBackendWithVariant:output:repo:weightsPath:revision:cacheDirectoryURL:completionHandler:)
    public static func clipBackend(variant: NFKMLXRVMVariant, output: NFKMLXRVMClipOutput, repo: String,
                                   weightsPath: String, revision: String?, cacheDirectoryURL: URL?,
                                   completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXDownload.backend(repo: repo, weightsPath: weightsPath, revision: revision,
                               cacheDirectoryURL: cacheDirectoryURL,
                               build: { try clipBackend(variant: variant, output: output, weightsURL: $0) },
                               completionHandler: completionHandler)
    }

    /// Registers both encoders' clip backends (`robust-video-matting-clip`,
    /// `robust-video-matting-resnet50-clip`) with `NFKMLXModelRegistry`, each writing the alpha matte.
    /// Introduced in InferKit 0.4.0.
    @objc public static func registerClip() {
        for variant in [NFKMLXRVMVariant.mobileNetV3, .resNet50] {
            NFKMLXModelRegistry.register(name: clipModelName(variant: variant)) { weightsURL in
                try clipBackend(variant: variant, output: .alpha, weightsURL: weightsURL)
            }
        }
    }
}
