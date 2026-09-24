import CoreGraphics
import Foundation
import InferKit
import MLX
import MLXNN

/*!
 @abstract The ten released Cosmos Tokenizers (`nvidia/Cosmos-0.1-Tokenizer-*`).
 @discussion The name carries the kind and the compression: C/D is continuous or discrete, I/V is
 image or video, and the factors are time × height × width (height × width for an image).
 Introduced in InferKit 0.4.0.
 */
@objc public enum NFKMLXCosmosTokenizerVariant: Int, Sendable, CaseIterable {
    case continuousImage8x8
    case continuousImage16x16
    case discreteImage8x8
    case discreteImage16x16
    case continuousVideo4x8x8
    case continuousVideo8x8x8
    case continuousVideo8x16x16
    case discreteVideo4x8x8
    case discreteVideo8x8x8
    case discreteVideo8x16x16

    /// NVIDIA's short name for the variant (`CI8x8`, `DV8x16x16`, …).
    public var releaseName: String {
        switch self {
        case .continuousImage8x8: return "CI8x8"
        case .continuousImage16x16: return "CI16x16"
        case .discreteImage8x8: return "DI8x8"
        case .discreteImage16x16: return "DI16x16"
        case .continuousVideo4x8x8: return "CV4x8x8"
        case .continuousVideo8x8x8: return "CV8x8x8"
        case .continuousVideo8x16x16: return "CV8x16x16"
        case .discreteVideo4x8x8: return "DV4x8x8"
        case .discreteVideo8x8x8: return "DV8x8x8"
        case .discreteVideo8x16x16: return "DV8x16x16"
        }
    }

    /// The Hugging Face repository the variant is published in.
    public var repository: String { "nvidia/Cosmos-0.1-Tokenizer-\(releaseName)" }

    /// The name the variant registers under with `NFKMLXModelRegistry`.
    public var modelName: String { "cosmos-tokenizer-\(releaseName.lowercased())" }

    public var isVideo: Bool { rawValue >= NFKMLXCosmosTokenizerVariant.continuousVideo4x8x8.rawValue }

    public var isDiscrete: Bool {
        switch self {
        case .discreteImage8x8, .discreteImage16x16, .discreteVideo4x8x8, .discreteVideo8x8x8,
             .discreteVideo8x16x16:
            return true
        default:
            return false
        }
    }
}

extension NFKMLXCosmosTokenizerConfiguration {

    /// The released geometry of `variant`, as NVIDIA's code builds it and as the release's tensors
    /// confirm: every variant patches with a 4× Haar wavelet and runs 128 base channels. The discrete
    /// image tokenizers widen the encoder's output to 256 before the six-channel FSQ projection, and the
    /// 4× continuous video tokenizer has two levels in each half.
    public static func variant(_ variant: NFKMLXCosmosTokenizerVariant) -> NFKMLXCosmosTokenizerConfiguration {
        var c: NFKMLXCosmosTokenizerConfiguration
        switch variant {
        case .continuousImage8x8, .discreteImage8x8:
            c = .init(isVideo: false, isDiscrete: variant.isDiscrete, spatialCompression: 8)
        case .continuousImage16x16, .discreteImage16x16:
            c = .init(isVideo: false, isDiscrete: variant.isDiscrete, spatialCompression: 16)
        case .continuousVideo4x8x8, .discreteVideo4x8x8:
            c = .init(isVideo: true, isDiscrete: variant.isDiscrete, spatialCompression: 8, temporalCompression: 4)
        case .continuousVideo8x8x8, .discreteVideo8x8x8:
            c = .init(isVideo: true, isDiscrete: variant.isDiscrete, spatialCompression: 8, temporalCompression: 8)
        case .continuousVideo8x16x16, .discreteVideo8x16x16:
            c = .init(isVideo: true, isDiscrete: variant.isDiscrete, spatialCompression: 16, temporalCompression: 8)
        }
        if variant == .discreteImage8x8 || variant == .discreteImage16x16 {
            c.zChannels = 256
        }
        if variant == .continuousVideo4x8x8 {
            c.encoderChannelMultipliers = [2, 4]
            c.decoderChannelMultipliers = [2, 4]
        }
        return c
    }
}

/// Holds the network for capture in `@Sendable` work.
private final class NFKCosmosTokenizerHolder: @unchecked Sendable {
    let net: NFKMLXCosmosTokenizerNet
    init(_ net: NFKMLXCosmosTokenizerNet) { self.net = net }
}

/*!
 @abstract A tokenizer's latent or token grid, for a caller in Objective-C.
 @discussion `shape` is channels-last: `[frames, height, width, channels]` for a continuous video
 latent, `[height, width, channels]` for an image latent, and without the channel axis for discrete
 tokens. `data` holds the values row-major, little-endian: float32 for a latent, int32 token indices for
 a discrete tokenizer.
 Introduced in InferKit 0.4.0.
 */
@objc(NFKMLXCosmosTokenizerCode)
public final class NFKMLXCosmosTokenizerCode: NSObject {
    @objc public let shape: [NSNumber]
    @objc public let data: Data
    @objc public let isDiscrete: Bool

    @objc public init(shape: [NSNumber], data: Data, isDiscrete: Bool) {
        self.shape = shape
        self.data = data
        self.isDiscrete = isDiscrete
    }

    convenience init(_ array: MLXArray, isDiscrete: Bool) {
        let values = isDiscrete ? array.asType(.int32) : array.asType(.float32)
        eval(values)
        let data = isDiscrete
            ? values.asArray(Int32.self).withUnsafeBufferPointer { Data(buffer: $0) }
            : values.asArray(Float.self).withUnsafeBufferPointer { Data(buffer: $0) }
        self.init(shape: values.shape.map { NSNumber(value: $0) }, data: data, isDiscrete: isDiscrete)
    }

    var array: MLXArray {
        let dims = shape.map(\.intValue)
        if isDiscrete {
            let values = data.withUnsafeBytes { Array($0.bindMemory(to: Int32.self)) }
            return MLXArray(values, dims)
        }
        let values = data.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        return MLXArray(values, dims)
    }
}

/// The reference's input handling: pixels in `[-1, 1]`, space zero-padded to a multiple of 16 around
/// the center, and a clip's frame count edge-padded to one more than a multiple of 8, split before and
/// after, so every variant's compression divides it. The reconstruction is cropped back.
enum NFKMLXCosmosTokenizerProcessor {
    static let spatialAlignment = 16
    static let temporalAlignment = 8
    /// The frames the reference reconstructs at once when a clip is longer.
    static let temporalWindow = 17

    struct Padding {
        var top = 0, left = 0, front = 0
        var height = 0, width = 0, frames = 1
    }

    static func pad(_ clip: MLXArray) -> (MLXArray, Padding) {
        let (t, h, w) = (clip.shape[0], clip.shape[1], clip.shape[2])
        let padHeight = (spatialAlignment - h % spatialAlignment) % spatialAlignment
        let padWidth = (spatialAlignment - w % spatialAlignment) % spatialAlignment
        let padFrames = (temporalAlignment - (t - 1) % temporalAlignment) % temporalAlignment
        var padding = Padding(top: padHeight / 2, left: padWidth / 2, front: padFrames / 2,
                              height: h, width: w, frames: t)
        var x = MLX.padded(clip, widths: [IntOrPair((0, 0)),
                                          IntOrPair((padding.top, padHeight - padding.top)),
                                          IntOrPair((padding.left, padWidth - padding.left)),
                                          IntOrPair((0, 0))])
        if padFrames > 0 {
            let front = repeated(x[0 ..< 1], count: padding.front, axis: 0)
            let back = repeated(x[(t - 1) ..< t], count: padFrames - padding.front, axis: 0)
            x = concatenated([front, x, back], axis: 0)
        }
        padding.frames = t
        return (x, padding)
    }

    static func crop(_ clip: MLXArray, _ padding: Padding) -> MLXArray {
        clip[padding.front ..< (padding.front + padding.frames),
             padding.top ..< (padding.top + padding.height),
             padding.left ..< (padding.left + padding.width)]
    }

    /// Frames `[H, W, 3]` in `0…1` as one clip `[T, H, W, 3]` in `[-1, 1]`.
    static func signedClip(_ frames: [MLXArray]) -> MLXArray { stacked(frames) * 2 - 1 }

    /// A clip `[T, H, W, 3]` in `[-1, 1]` as frames in `0…1`, clamped.
    static func frames(_ signed: MLXArray) -> [MLXArray] {
        let pixels = MLX.clip((signed + 1) / 2, min: 0, max: 1)
        return (0 ..< pixels.shape[0]).map { pixels[$0] }
    }
}

/*!
 @abstract A Cosmos Tokenizer: image or video to a continuous latent or discrete tokens, and back.
 @discussion Build one with ``tokenizer(variant:weightsURL:)`` (or the `@objc` factories) from a
 release's `autoencoder.jit`, its `encoder.jit` and `decoder.jit` together, or a safetensors file
 ``NFKMLXWeights/save(_:to:)`` wrote after a fine-tune. The Swift surface works on channels-last
 `MLXArray`s in `[-1, 1]`: an image `[B, H, W, 3]`, a clip `[B, T, H, W, 3]` whose frame count is one
 more than a multiple of the temporal compression. The Objective-C surface works on `CGImage`s and
 ``NFKMLXCosmosTokenizerCode``.
 Introduced in InferKit 0.4.0.
 */
@objc(NFKMLXCosmosTokenizer)
public final class NFKMLXCosmosTokenizer: NSObject {

    /// The prefix every variant's registry name starts with.
    @objc public static let modelName = "cosmos-tokenizer"

    private let holder: NFKCosmosTokenizerHolder
    @objc public let variant: NFKMLXCosmosTokenizerVariant

    init(net: NFKMLXCosmosTokenizerNet, variant: NFKMLXCosmosTokenizerVariant) {
        holder = NFKCosmosTokenizerHolder(net)
        self.variant = variant
    }

    /// The network, for a Swift caller composing it into a larger graph or fine-tuning it.
    public var network: NFKMLXCosmosTokenizerNet { holder.net }

    // MARK: Swift surface

    /// The latent, or a discrete variant's FSQ codes.
    public func encode(_ pixels: MLXArray) -> MLXArray {
        let latent = holder.net.encode(pixels)
        eval(latent)
        return latent
    }

    /// Pixels in `[-1, 1]` from a latent or a discrete variant's codes.
    public func decode(_ latent: MLXArray) -> MLXArray {
        let pixels = holder.net.decode(latent)
        eval(pixels)
        return pixels
    }

    /// A discrete variant's token indices, int32.
    public func tokens(_ pixels: MLXArray) -> MLXArray {
        let tokens = holder.net.tokens(pixels)
        eval(tokens)
        return tokens
    }

    /// Pixels in `[-1, 1]` from a discrete variant's token indices.
    public func decode(tokens: MLXArray) -> MLXArray {
        let pixels = holder.net.decode(tokens: tokens)
        eval(pixels)
        return pixels
    }

    /// Reconstructs frames `[H, W, 3]` in `0…1` (one frame for an image), padding and windowing as the
    /// reference does, and returns the same number of frames at the same size.
    public func reconstruct(frames: [MLXArray]) -> [MLXArray] {
        let net = holder.net
        guard net.configuration.isVideo else {
            return frames.map { frame in
                let (padded, padding) = NFKMLXCosmosTokenizerProcessor.pad(NFKMLXCosmosTokenizerProcessor.signedClip([frame]))
                let output = net(padded)
                return NFKMLXCosmosTokenizerProcessor.frames(NFKMLXCosmosTokenizerProcessor.crop(output, padding))[0]
            }
        }
        var result: [MLXArray] = []
        let window = NFKMLXCosmosTokenizerProcessor.temporalWindow
        for start in stride(from: 0, to: frames.count, by: window) {
            let part = Array(frames[start ..< min(start + window, frames.count)])
            let (padded, padding) = NFKMLXCosmosTokenizerProcessor.pad(NFKMLXCosmosTokenizerProcessor.signedClip(part))
            let output = net(padded.expandedDimensions(axis: 0))[0]
            let cropped = NFKMLXCosmosTokenizerProcessor.crop(output, padding)
            eval(cropped)
            result += NFKMLXCosmosTokenizerProcessor.frames(cropped)
        }
        return result
    }

    // MARK: Objective-C surface

    /// Encodes an image (a video variant treats it as a one-frame clip) to its latent, or a discrete
    /// variant's token indices. The image's sides must be multiples of 16.
    @objc(codeForImage:error:)
    public func code(forImage image: CGImage) throws -> NFKMLXCosmosTokenizerCode {
        try code(forFrames: [image])
    }

    /// Encodes frames (`CGImage`s: a clip whose count is one more than a multiple of the temporal
    /// compression, or one image) to the latent or token grid.
    @objc(codeForFrames:error:)
    public func code(forFrames images: [Any]) throws -> NFKMLXCosmosTokenizerCode {
        let net = holder.net
        let pixels = try pixels(images)
        let encoded = net.configuration.isDiscrete ? net.tokens(pixels) : net.encode(pixels)
        return NFKMLXCosmosTokenizerCode(encoded[0], isDiscrete: net.configuration.isDiscrete)
    }

    /// Decodes a latent or token grid back to frames, `CGImage`s (one for an image variant).
    @objc(framesForCode:error:)
    public func frames(for code: NFKMLXCosmosTokenizerCode) throws -> [Any] {
        let net = holder.net
        guard code.isDiscrete == net.configuration.isDiscrete else {
            throw NFKMLXError.unsupportedConfiguration(
                "the code is \(code.isDiscrete ? "discrete" : "continuous") and this tokenizer is not")
        }
        let batched = code.array.expandedDimensions(axis: 0)
        let decoded = (code.isDiscrete ? net.decode(tokens: batched) : net.decode(batched))[0]
        let clip = net.configuration.isVideo ? decoded : decoded.expandedDimensions(axis: 0)
        eval(clip)
        let options = NFKMLXImageOptions()
        return try NFKMLXCosmosTokenizerProcessor.frames(clip).map { try NFKMLXImageBridge.cgImage(from: $0, options: options) }
    }

    private func pixels(_ images: [Any]) throws -> MLXArray {
        guard !images.isEmpty else { throw NFKMLXError.unsupportedInput }
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let frames = try images.map { try NFKMLXImageBridge.tensor(from: $0, channels: 3, colorSpace: colorSpace) }
        let clip = NFKMLXCosmosTokenizerProcessor.signedClip(frames)
        return holder.net.configuration.isVideo ? clip.expandedDimensions(axis: 0) : clip
    }

    // MARK: Factories

    /// Builds the network, loading a released or fine-tuned checkpoint. Nil weights leave the network at
    /// its random initialization, which is training from scratch.
    ///
    /// - Parameters:
    ///   - variant: the release, which fixes the network's geometry.
    ///   - weightsURLs: one file holding both halves (a release's `autoencoder.jit`, or a fine-tuned
    ///     safetensors), or a release's `encoder.jit` and `decoder.jit` together.
    public static func network(variant: NFKMLXCosmosTokenizerVariant,
                               weightsURLs: [URL]) throws -> NFKMLXCosmosTokenizerNet {
        let net = NFKMLXCosmosTokenizerNet(configuration: .variant(variant))
        if !weightsURLs.isEmpty {
            try loadWeights(into: net, from: weightsURLs)
        }
        return net
    }

    /// Builds the network from one checkpoint holding both halves, or randomly initialized when nil.
    public static func network(variant: NFKMLXCosmosTokenizerVariant, weightsURL: URL?) throws -> NFKMLXCosmosTokenizerNet {
        try network(variant: variant, weightsURLs: weightsURL.map { [$0] } ?? [])
    }

    /// Builds a tokenizer from one checkpoint holding both halves: a release's `autoencoder.jit`, or a
    /// safetensors a fine-tune wrote.
    @objc(tokenizerWithVariant:weightsURL:error:)
    public static func tokenizer(variant: NFKMLXCosmosTokenizerVariant, weightsURL: URL) throws -> NFKMLXCosmosTokenizer {
        NFKMLXCosmosTokenizer(net: try network(variant: variant, weightsURL: weightsURL), variant: variant)
    }

    /// Builds a tokenizer from a release's separate `encoder.jit` and `decoder.jit`.
    @objc(tokenizerWithVariant:encoderWeightsURL:decoderWeightsURL:error:)
    public static func tokenizer(variant: NFKMLXCosmosTokenizerVariant, encoderWeightsURL: URL,
                                 decoderWeightsURL: URL) throws -> NFKMLXCosmosTokenizer {
        NFKMLXCosmosTokenizer(net: try network(variant: variant, weightsURLs: [encoderWeightsURL, decoderWeightsURL]),
                              variant: variant)
    }

    /// The reconstruction backend over one checkpoint holding both halves. Nil weights build it randomly
    /// initialized.
    @objc(backendWithVariant:weightsURL:error:)
    public static func backend(variant: NFKMLXCosmosTokenizerVariant, weightsURL: URL?) throws -> any NFKInferenceBackend {
        let net = try network(variant: variant, weightsURL: weightsURL)
        return NFKMLXCosmosTokenizerBackend(tokenizer: NFKMLXCosmosTokenizer(net: net, variant: variant))
    }

    /// Downloads a checkpoint from Hugging Face, then builds the backend. The release's own file is
    /// `autoencoder.jit` in the variant's ``NFKMLXCosmosTokenizerVariant/repository``.
    @objc(backendWithVariant:repo:weightsPath:revision:cacheDirectoryURL:error:)
    public static func backend(variant: NFKMLXCosmosTokenizerVariant, repo: String, weightsPath: String,
                               revision: String?, cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        let url = try NFKMLXDownload.weightsURL(repo: repo, weightsPath: weightsPath, revision: revision,
                                                cacheDirectoryURL: cacheDirectoryURL)
        return try backend(variant: variant, weightsURL: url)
    }

    /// The asynchronous form of the download factory.
    @objc(backendWithVariant:repo:weightsPath:revision:cacheDirectoryURL:completionHandler:)
    public static func backend(variant: NFKMLXCosmosTokenizerVariant, repo: String, weightsPath: String,
                               revision: String?, cacheDirectoryURL: URL?,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXDownload.backend(repo: repo, weightsPath: weightsPath, revision: revision,
                               cacheDirectoryURL: cacheDirectoryURL,
                               build: { try backend(variant: variant, weightsURL: $0) },
                               completionHandler: completionHandler)
    }

    /// Registers every variant with `NFKMLXModelRegistry` under its ``NFKMLXCosmosTokenizerVariant/modelName``
    /// (`cosmos-tokenizer-ci8x8`, …).
    @objc public static func register() {
        for variant in NFKMLXCosmosTokenizerVariant.allCases {
            NFKMLXModelRegistry.register(name: variant.modelName) { weightsURL in
                try backend(variant: variant, weightsURL: weightsURL)
            }
        }
    }

    /// Buffers TorchScript serializes beside the parameters, which the network derives instead.
    private static let derivedBufferSuffixes = ["wavelets", "_arange", "patch_size_buffer", "_levels", "_basis",
                                                "implicit_codebook"]

    /// Loads NVIDIA's TorchScript state dicts (stored bfloat16, read at float32) or a safetensors a
    /// fine-tune wrote. The module keys are the release's own; only the convolution layout changes, and
    /// only for a PyTorch-layout file. Every built shape must equal the release's, which is what
    /// confirms the variant's fixed geometry against the checkpoint.
    static func loadWeights(into net: NFKMLXCosmosTokenizerNet, from urls: [URL]) throws {
        var mapped: [(String, MLXArray)] = []
        for url in urls {
            let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
            for (key, value) in checkpoint.arrays {
                guard !derivedBufferSuffixes.contains(where: { key.hasSuffix($0) }) else { continue }
                var array = value.asType(.float32)
                if checkpoint.needsConvTranspose {
                    if array.ndim == 5 {
                        array = array.transposed(0, 2, 3, 4, 1)
                    } else if array.ndim == 4 {
                        array = array.transposed(0, 2, 3, 1)
                    }
                }
                mapped.append((key, array))
            }
        }
        try NFKMLXWeights.apply(mapped, to: net, verifyShapes: true)
    }
}

/// Cosmos Tokenizer reconstruction as an InferKit backend: an image under `NFKInputImage` comes back
/// under `NFKOutputImage`, and for a video variant a clip under `NFKInputVideo` comes back as a new
/// `NFKVideoAsset` under `NFKOutputVideo`, at the source's frame rate. The latent and tokens are
/// reached through ``NFKMLXCosmosTokenizer``.
@objc(NFKMLXCosmosTokenizerBackend)
public final class NFKMLXCosmosTokenizerBackend: NSObject, NFKInferenceBackend {

    private let tokenizer: NFKMLXCosmosTokenizer

    init(tokenizer: NFKMLXCosmosTokenizer) {
        self.tokenizer = tokenizer
        super.init()
    }

    @objc public var isReady: Bool { true }
    @objc public var backendIdentifier: String { tokenizer.variant.modelName }

    /// The request parameters the backend reads.
    @objc public var supportedParameterKeys: Set<String> { [] }

    /// The request inputs the backend reads.
    @objc public var supportedInputKeys: Set<String> {
        tokenizer.variant.isVideo ? [NFKInputVideo, NFKInputImage] : [NFKInputImage]
    }

    @objc(runInferenceForRequest:error:)
    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        var options = NFKMLXImageOptions()
        options.colorSpace = colorSpace
        if tokenizer.variant.isVideo, let asset = request.input(forKey: NFKInputVideo) as? NFKVideoAsset,
           let sourceURL = asset.fileURL {
            let clip = try NFKMLXVideoFile.read(sourceURL)
            let frames = clip.frames.map { frame -> MLXArray in
                let (bytes, width, height) = NFKMLXImageBridge.rgbaBytes(from: frame, colorSpace: colorSpace)
                return NFKMLXImageBridge.tensor(rgba: bytes, width: width, height: height, channels: 3)
            }
            let written = try tokenizer.reconstruct(frames: frames).map { try NFKMLXImageBridge.cgImage(from: $0, options: options) }
            guard let first = written.first else { throw NFKMLXError.noOutput }
            let outputURL = FileManager.default.temporaryDirectory
                .appendingPathComponent("\(backendIdentifier)-\(UUID().uuidString).mov")
            try NFKMLXVideoFile.write(written, framesPerSecond: clip.framesPerSecond, to: outputURL)
            let result = NFKVideoAsset(fileURL: outputURL,
                                       durationSeconds: Double(written.count) / clip.framesPerSecond,
                                       framesPerSecond: clip.framesPerSecond,
                                       dimensions: CGSize(width: first.width, height: first.height))
            return NFKInferenceResult(outputs: [NFKOutputVideo: result])
        }
        guard let value = request.input(forKey: NFKInputImage) else { throw NFKMLXError.unsupportedInput }
        let image = try NFKMLXImageBridge.tensor(from: value, channels: 3, colorSpace: colorSpace)
        let reconstruction = tokenizer.reconstruct(frames: [image])[0]
        eval(reconstruction)
        return NFKInferenceResult(outputs: [NFKOutputImage: try NFKMLXImageBridge.cgImage(from: reconstruction, options: options)])
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
}
