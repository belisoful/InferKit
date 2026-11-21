//
//  NFKMLXStableNormal.swift
//  InferKitMLX
//
//  StableNormal-turbo (`Stable-X/yoso-normal-v1-8-1`, Apache-2.0), the one-step YOSO normal estimator
//  Hi3DGen runs, as hugoycj/StableNormal's `hubconf.Predictor` runs it with `YOSONormalsPipeline`
//  (`stablenormal/pipeline_yoso_normal.py`, Apache-2.0).
//
//  The step is deterministic: the autoencoder's mean latent, the ControlNet-VAE's residuals over that
//  latent, one UNet forward at timestep 0 on the same latent with those residuals, and the decode
//  renormalized per pixel. Every byte stage around it follows Pillow and NumPy to the integer.
//
//  Introduced in InferKit 0.4.0.
//

import CoreGraphics
import Foundation
import InferKit
import MLX
import MLXNN

// MARK: The network

/// The three networks of `YOSONormalsPipeline`: the UNet and autoencoder (``NFKMLXSDPipeline``) and the
/// ControlNet-VAE (``NFKMLXSDControlNetVAE``). Introduced in InferKit 0.4.0.
public final class NFKMLXStableNormalNet: Module {
    @ModuleInfo(key: "pipeline") var pipeline: NFKMLXSDPipeline
    @ModuleInfo(key: "controlnet") var controlnet: NFKMLXSDControlNetVAE

    /// The token count of the conditioning the pipeline sets in place of a prompt, `zeros(1, 257, 1024)`.
    static let contextTokens = 257

    /// The pipeline's `t_start`: the UNet and the ControlNet-VAE both run at timestep 0.
    static let timestep: Int32 = 0

    public init(unet: NFKMLXSDUNetConfiguration = .yoso, vae: NFKMLXSDVAEConfiguration = .stableDiffusion) {
        self._pipeline.wrappedValue = NFKMLXSDPipeline(unet: unet, vae: vae)
        self._controlnet.wrappedValue = NFKMLXSDControlNetVAE(configuration: unet)
    }

    var scale: Float { pipeline.vae.configuration.scaleFactor }

    var context: MLXArray {
        MLXArray.zeros([1, Self.contextTokens, pipeline.unet.configuration.crossAttentionDimensions])
    }

    /// The autoencoder's mean latent of `pixels` (`[1, h, w, 3]` in −1…1), times the scaling factor.
    func imageLatent(_ pixels: MLXArray) -> MLXArray {
        pipeline.latent(of: pixels, scale: scale)
    }

    func residuals(_ latent: MLXArray) -> NFKSDControlResiduals {
        controlnet(latent, timestep: MLXArray([Self.timestep]), context: context)
    }

    func predictedLatent(_ latent: MLXArray, residuals: NFKSDControlResiduals) -> MLXArray {
        pipeline.unet(latent, timestep: MLXArray([Self.timestep]), context: context, controlResiduals: residuals)
    }

    /// The decode of `latent`, renormalized per pixel: `decode_prediction`.
    func decoded(_ latent: MLXArray) -> MLXArray {
        Self.normalized(pipeline.image(from: latent, scale: scale))
    }

    /// `normalize_normals`: each pixel divided by its norm over the channels, the norm clamped to 1e-6.
    static func normalized(_ normals: MLXArray) -> MLXArray {
        normals / maximum(sqrt((normals * normals).sum(axis: -1, keepDims: true)), MLXArray(Float(1e-6)))
    }

    /// The predicted normals of `pixels` (`[1, h, w, 3]` in −1…1), `[1, h, w, 3]` of unit length. The
    /// pipeline normalizes twice, once in `decode_prediction` and again after its resize to the input
    /// resolution, and the port does both.
    public func callAsFunction(_ pixels: MLXArray) -> MLXArray {
        let latent = imageLatent(pixels)
        return Self.normalized(decoded(predictedLatent(latent, residuals: residuals(latent))))
    }
}

// MARK: The predictor

/// Every stage of one predictor run, for the parity tests and for a caller that reads the mask.
struct NFKMLXStableNormalStages {
    /// The white composite at the input size, RGB8.
    let composite: [UInt8]
    /// The size the predictor's `resize_image` resizes to.
    let processingWidth: Int
    let processingHeight: Int
    /// The composite resized to the processing size with Pillow's Lanczos, RGB8.
    let resized: [UInt8]
    /// The Marigold-preprocessed pixels, `[1, h, w, 3]` in −1…1.
    let pixels: MLXArray
    let latent: MLXArray
    let residuals: NFKSDControlResiduals
    let predictedLatent: MLXArray
    /// The decode, renormalized, `[1, h, w, 3]`.
    let decoded: MLXArray
    /// `decoded` renormalized again, `[h, w, 3]`.
    let prediction: [Float]
    /// `alpha > 128` at the processing size, 0 or 1.
    let mask: [UInt8]
    /// The prediction with every pixel outside the mask set to 1.
    let masked: [Float]
    /// The masked prediction as 8-bit color at the processing size, RGB8.
    let map: [UInt8]
    /// `map` resized back to the input size with Pillow's Lanczos, RGB8.
    let normalImage: [UInt8]
}

/// StableNormal-turbo's predictor over loaded networks: the image path `hubconf.Predictor.__call__` takes
/// with `data_type="object"` on an RGBA image, around the pipeline's one step.
final class NFKMLXStableNormalModel {
    let net: NFKMLXStableNormalNet
    /// The pipeline's `default_processing_resolution`. The step runs only where the predictor's resize
    /// lands the longer side on it, so the pipeline's own resizes are identities.
    let processingResolution: Int
    private let lock = NSLock()

    init(net: NFKMLXStableNormalNet, processingResolution: Int = 768) {
        self.net = net
        self.processingResolution = processingResolution
    }

    /// The networks from a release directory (`unet/`, `controlnet/`, `vae/`, each with its `config.json`
    /// and its `diffusion_pytorch_model.safetensors` or `.fp16.safetensors`), loaded at float32.
    static func load(directoryURL: URL) throws -> NFKMLXStableNormalModel {
        func weights(_ folder: String) throws -> URL {
            try NFKMLXSDReleaseFiles.weightsURL(in: directoryURL, folder: folder, base: "diffusion_pytorch_model")
        }
        let vaeConfiguration = try NFKMLXStableDiffusionModels.vaeConfiguration(
            fromHuggingFace: directoryURL.appendingPathComponent("vae/config.json"))
        let net = NFKMLXStableNormalNet(unet: .yoso, vae: vaeConfiguration)
        net.train(false)
        try net.pipeline.loadWeights(unetURL: try weights("unet"), vaeURL: try weights("vae"), precision: .float32)
        try NFKMLXStableDiffusionModels.loadControlNetVAEWeights(into: net.controlnet, from: try weights("controlnet"),
                                                                 precision: .float32)
        return NFKMLXStableNormalModel(net: net)
    }

    /// The predictor's normal map for straight or premultiplied 8-bit RGBA (`height × width × 4`): the white composite,
    /// the 768 resize, the step, the alpha mask, and the Lanczos resize back to `width × height`. Returns RGB8.
    ///
    /// @discussion The bytes are read as given, as Pillow reads an RGBA image. An opaque image composites to
    /// itself and masks nothing, which is the predictor's path for an RGB image with no mask.
    func normalMap(rgba: [UInt8], width: Int, height: Int, resolution: Int = 768) throws -> [UInt8] {
        try stages(rgba: rgba, width: width, height: height, resolution: resolution).normalImage
    }

    func stages(rgba: [UInt8], width: Int, height: Int, resolution: Int = 768) throws -> NFKMLXStableNormalStages {
        guard width > 0, height > 0, rgba.count == width * height * 4 else {
            throw NFKMLXError.unsupportedConfiguration("an RGBA image of \(width) × \(height) holds \(width * height * 4) bytes")
        }
        let (processingWidth, processingHeight) = Self.processingSize(width: width, height: height, resolution: resolution)
        guard processingWidth > 0, processingHeight > 0 else {
            throw NFKMLXError.unsupportedConfiguration("a \(width) × \(height) image resizes to a side of 0 at \(resolution)")
        }
        guard max(processingWidth, processingHeight) == processingResolution else {
            throw NFKMLXError.unsupportedConfiguration(
                "the predictor resizes to \(processingWidth) × \(processingHeight); the step runs at a longer side of \(processingResolution)")
        }
        let composite = Self.rgb(NFKMLXImageBridge.compositedOnWhite(rgba: rgba))
        let resized = NFKMLXPillowLanczos.resize(composite, width: width, height: height,
                                                 outWidth: processingWidth, outHeight: processingHeight)
        let alphaMask = (0 ..< width * height).map { rgba[$0 * 4 + 3] > 128 ? UInt8(1) : 0 }
        let mask = Self.nearest(alphaMask, width: width, height: height, outWidth: processingWidth, outHeight: processingHeight)

        let pixels = Self.pixels(resized, width: processingWidth, height: processingHeight)
        let (latent, residuals, predicted, decoded, normals) = lock.withLock {
            let latent = net.imageLatent(pixels)
            let residuals = net.residuals(latent)
            let predicted = net.predictedLatent(latent, residuals: residuals)
            let decoded = net.decoded(predicted)
            let normals = NFKMLXStableNormalNet.normalized(decoded)
            eval(normals)
            return (latent, residuals, predicted, decoded, normals)
        }
        let prediction = normals.reshaped([-1]).asArray(Float.self)
        let masked = Self.masked(prediction, mask: mask)
        let map = Self.byteMap(masked)
        let normalImage = NFKMLXPillowLanczos.resize(map, width: processingWidth, height: processingHeight,
                                                     outWidth: width, outHeight: height)
        return NFKMLXStableNormalStages(composite: composite, processingWidth: processingWidth, processingHeight: processingHeight,
                                        resized: resized, pixels: pixels, latent: latent, residuals: residuals,
                                        predictedLatent: predicted, decoded: decoded, prediction: prediction, mask: mask,
                                        masked: masked, map: map, normalImage: normalImage)
    }

    // MARK: The byte stages

    /// `resize_image`: the longer side scaled to `resolution`, each side rounded to a multiple of 64 with
    /// NumPy's half-to-even rounding, in double precision.
    static func processingSize(width: Int, height: Int, resolution: Int) -> (width: Int, height: Int) {
        let k = Double(resolution) / Double(max(width, height))
        func side(_ length: Int) -> Int { Int((Double(length) * k / 64).rounded(.toNearestOrEven)) * 64 }
        return (side(width), side(height))
    }

    static func rgb(_ rgba: [UInt8]) -> [UInt8] {
        var rgb = [UInt8](repeating: 0, count: rgba.count / 4 * 3)
        for pixel in 0 ..< rgba.count / 4 {
            rgb[pixel * 3] = rgba[pixel * 4]
            rgb[pixel * 3 + 1] = rgba[pixel * 4 + 1]
            rgb[pixel * 3 + 2] = rgba[pixel * 4 + 2]
        }
        return rgb
    }

    /// Pillow's `NEAREST` resize of a one-channel image. Each axis's source position starts half a step in
    /// and advances by the scale in double precision, and the running sum truncates to the source pixel. At
    /// 1024 → 768 it picks a different pixel from `(x + 0.5) · scale` at 99 of 768 positions.
    static func nearest(_ pixels: [UInt8], width: Int, height: Int, outWidth: Int, outHeight: Int) -> [UInt8] {
        func indices(_ inSize: Int, _ outSize: Int) -> [Int] {
            let step = Double(inSize) / Double(outSize)
            var position = step * 0.5
            var out = [Int]()
            for _ in 0 ..< outSize {
                out.append(min(Int(position), inSize - 1))
                position += step
            }
            return out
        }
        let columns = indices(width, outWidth), rows = indices(height, outHeight)
        var out = [UInt8](repeating: 0, count: outWidth * outHeight)
        for (y, row) in rows.enumerated() {
            for (x, column) in columns.enumerated() {
                out[y * outWidth + x] = pixels[row * width + column]
            }
        }
        return out
    }

    /// The Marigold image processor's `preprocess` of RGB8: each byte over 255, times 2, less 1, each
    /// operation rounded to float32 as torch rounds it, `[1, h, w, 3]`.
    static func pixels(_ rgb: [UInt8], width: Int, height: Int) -> MLXArray {
        let values = rgb.map { Float($0) / 255 * 2 - 1 }
        return MLXArray(values, [1, height, width, 3])
    }

    /// `_apply_mask`: every channel of a pixel outside the mask set to 1.
    static func masked(_ prediction: [Float], mask: [UInt8]) -> [Float] {
        var masked = prediction
        for (pixel, inside) in mask.enumerated() where inside == 0 {
            masked[pixel * 3] = 1
            masked[pixel * 3 + 1] = 1
            masked[pixel * 3 + 2] = 1
        }
        return masked
    }

    /// `((clip(p, −1, 1) + 1) / 2 · 255).astype(uint8)`: float32 at every operation, truncated.
    static func byteMap(_ normals: [Float]) -> [UInt8] {
        normals.map { UInt8((min(max($0, -1), 1) + 1) / 2 * 255) }
    }
}

// MARK: The backend

/// StableNormal-turbo as an InferKit backend: an image under `NFKInputImage` → its normal map at the
/// image's size under `NFKOutputImage`, an opaque RGB `CGImage`. Introduced in InferKit 0.4.0.
///
/// @discussion An image with alpha composites onto white and its alpha (above 128) is the object mask,
/// outside of which the normals read (1, 1, 1); an opaque image runs with no mask. The input is resized
/// so its longer side is 768.
@objc(NFKMLXStableNormalBackend)
public final class NFKMLXStableNormalBackend: NSObject, NFKInferenceBackend {
    let model: NFKMLXStableNormalModel

    init(model: NFKMLXStableNormalModel) {
        self.model = model
        super.init()
    }

    @objc public var isReady: Bool { true }
    @objc public var backendIdentifier: String { NFKMLXStableNormal.modelName }

    /// The request parameters the backend reads. Introduced in InferKit 0.4.0.
    @objc public var supportedParameterKeys: Set<String> { [] }

    /// The request inputs the backend reads. Introduced in InferKit 0.4.0.
    @objc public var supportedInputKeys: Set<String> { [NFKInputImage] }

    /// The result keys the backend writes. Introduced in InferKit 0.4.0.
    @objc public var supportedOutputKeys: Set<String> { [NFKOutputImage] }

    @objc(runInferenceForRequest:error:)
    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        guard let value = request.input(forKey: NFKInputImage) else { throw NFKMLXError.unsupportedInput }
        let image = try NFKMLXImageBridge.rgbaBytes(from: value, colorSpace: CGColorSpaceCreateDeviceRGB())
        let map = try model.normalMap(rgba: image.bytes, width: image.width, height: image.height)
        var rgba = [UInt8](repeating: 255, count: image.width * image.height * 4)
        for pixel in 0 ..< image.width * image.height {
            rgba[pixel * 4] = map[pixel * 3]
            rgba[pixel * 4 + 1] = map[pixel * 3 + 1]
            rgba[pixel * 4 + 2] = map[pixel * 3 + 2]
        }
        let output = try NFKMLXImageBridge.cgImage(rgba: rgba, width: image.width, height: image.height,
                                                   options: NFKMLXImageOptions())
        return NFKInferenceResult(outputs: [NFKOutputImage: output])
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

// MARK: Factories

/// StableNormal-turbo's factories and registration. Introduced in InferKit 0.4.0.
@objc(NFKMLXStableNormal)
public final class NFKMLXStableNormal: NSObject {

    /// The registry name, `stablenormal-turbo`.
    @objc public static let modelName = "stablenormal-turbo"
    /// The release the parity record measures, `Stable-X/yoso-normal-v1-8-1`.
    @objc public static let repo = "Stable-X/yoso-normal-v1-8-1"
    /// The measured revision of ``repo``.
    @objc public static let revision = "4dd7d000d76febb68254ebe27251c2cb79dc1c2a"

    static let components: [NFKMLXReleaseComponent] = ["unet", "controlnet", "vae"].map { folder in
        NFKMLXReleaseComponent(required: ["\(folder)/config.json"],
                               weights: ["\(folder)/diffusion_pytorch_model.fp16.safetensors",
                                         "\(folder)/diffusion_pytorch_model.safetensors"])
    }

    /// Builds from a release directory: `unet/`, `controlnet/`, and `vae/`, each with its `config.json` and
    /// its `diffusion_pytorch_model.safetensors` or the `.fp16.safetensors` the release publishes. The
    /// networks load at float32. Run inference off the render thread.
    @objc(backendWithDirectoryURL:error:)
    public static func backend(directoryURL: URL) throws -> any NFKInferenceBackend {
        NFKMLXStableNormalBackend(model: try NFKMLXStableNormalModel.load(directoryURL: directoryURL))
    }

    /// Downloads the release's three networks (about 2.6 GB at half precision) into the cache, then builds.
    /// A nil `revision` reads the measured one. Blocking on the network; run off the main and render threads.
    @objc(backendWithRepo:revision:cacheDirectoryURL:error:)
    public static func backend(repo: String, revision: String?, cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        try NFKMLXBuildHolds.keeping {
            try backend(directoryURL: try NFKMLXReleaseDownload.directory(
                repo: repo, revision: revision ?? (repo == Self.repo ? Self.revision : nil),
                cacheDirectoryURL: cacheDirectoryURL, components: components))
        }
    }

    /// The asynchronous form of ``backend(repo:revision:cacheDirectoryURL:)``.
    @objc(backendWithRepo:revision:cacheDirectoryURL:completionHandler:)
    public static func backend(repo: String, revision: String?, cacheDirectoryURL: URL?,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXReleaseDownload.async(completionHandler) {
            try backend(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL)
        }
    }

    /// Registers `stablenormal-turbo` with `NFKMLXModelRegistry`. The location is the release directory a catalog
    /// download fills; without one the measured release downloads into the cache.
    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName) { directoryURL in
            guard let directoryURL else {
                return try backend(repo: repo, revision: nil, cacheDirectoryURL: nil)
            }
            return try backend(directoryURL: directoryURL)
        }
    }
}
