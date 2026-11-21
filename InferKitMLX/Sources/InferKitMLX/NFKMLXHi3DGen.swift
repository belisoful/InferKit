//
//  NFKMLXHi3DGen.swift
//  InferKitMLX
//
//  Hi3DGen (Stable-X, MIT): image to 3D through a normal map. BiRefNet cuts the subject out, the image is
//  cropped and padded to 1024², StableNormal-turbo estimates its normal map, and TRELLIS's two stages run
//  on that map with Hi3DGen's own flows. The mesh decoder's features become a surface by marching cubes over
//  the averaged corner SDF. The path follows the release's `app.py` `generate_3d`.
//
//  Introduced in InferKit 0.4.0.
//

import CoreGraphics
import Foundation
import InferKit
import MLX

/// Hi3DGen's image preparation (`Hi3DGenPipeline.preprocess_image` at 1024): an image with alpha keeps it;
/// an opaque one is cut out by BiRefNet's mask. The alpha's tight box (alpha above 0.8 of full) widened by a
/// fifth, clamped to the image, padded to a square with transparent black, resized with Pillow's Lanczos
/// over premultiplied RGBA, then every channel multiplied as the release multiplies it in float32.
enum NFKHi3DGenImagePreparation {
    /// The longest side the release keeps before masking.
    static let maskingSide = 1024

    /// The prepared RGBA8 image `side × side × 4`, its color premultiplied by its alpha, from straight
    /// RGBA8 pixels. `mask` returns BiRefNet's binary mask (`height × width`, 0 or 1) for opaque RGB8
    /// pixels. An image with no foreground above 0.8 alpha comes back as its own RGB with an opaque alpha,
    /// unresized, as the release returns it.
    static func prepare(rgba: [UInt8], width: Int, height: Int, side: Int = 1024,
                        mask: ([UInt8], Int, Int) throws -> [UInt8]) throws -> (pixels: [UInt8], width: Int, height: Int) {
        var image = rgba, w = width, h = height
        if !image.enumerated().contains(where: { $0.offset % 4 == 3 && $0.element != 255 }) {
            var rgb = stride(from: 0, to: image.count, by: 4).flatMap { image[$0 ..< $0 + 3] }
            let scale = min(1, Double(maskingSide) / Double(max(w, h)))
            if scale < 1 {
                let (scaledWidth, scaledHeight) = (Int(Double(w) * scale), Int(Double(h) * scale))
                rgb = NFKMLXPillowLanczos.resize(rgb, width: w, height: h, outWidth: scaledWidth, outHeight: scaledHeight)
                (w, h) = (scaledWidth, scaledHeight)
            }
            let binary = try mask(rgb, w, h)
            image = [UInt8](repeating: 0, count: w * h * 4)
            for pixel in 0 ..< w * h {
                image[pixel * 4] = rgb[pixel * 3]
                image[pixel * 4 + 1] = rgb[pixel * 3 + 1]
                image[pixel * 4 + 2] = rgb[pixel * 3 + 2]
                image[pixel * 4 + 3] = binary[pixel] * 255
            }
        }
        var minX = w, minY = h, maxX = -1, maxY = -1
        for y in 0 ..< h {
            for x in 0 ..< w where Float(image[(y * w + x) * 4 + 3]) > 0.8 * 255 {
                minX = min(minX, x)
                maxX = max(maxX, x)
                minY = min(minY, y)
                maxY = max(maxY, y)
            }
        }
        guard maxX >= 0 else {
            var opaque = image
            for pixel in 0 ..< w * h { opaque[pixel * 4 + 3] = 255 }
            return (opaque, w, h)
        }
        let centerX = Double(minX + maxX) / 2, centerY = Double(minY + maxY) / 2
        let size = Int(Double(max(maxX - minX, maxY - minY)) * 1.2)
        // Python's int() truncates toward zero, and the box is then clamped to the image.
        let left = max(0, Int(centerX - Double(size / 2))), top = max(0, Int(centerY - Double(size / 2)))
        let right = min(w, Int(centerX + Double(size / 2))), bottom = min(h, Int(centerY + Double(size / 2)))
        let cropWidth = right - left, cropHeight = bottom - top
        let square = max(cropWidth, cropHeight)
        let offsetX = cropWidth > cropHeight ? 0 : (cropHeight - cropWidth) / 2
        let offsetY = cropWidth > cropHeight ? (cropWidth - cropHeight) / 2 : 0
        var padded = [UInt8](repeating: 0, count: square * square * 4)
        for y in 0 ..< cropHeight {
            for x in 0 ..< cropWidth {
                for c in 0 ..< 4 {
                    padded[((y + offsetY) * square + x + offsetX) * 4 + c] = image[((y + top) * w + x + left) * 4 + c]
                }
            }
        }
        let resized = NFKMLXPillowLanczos.resizeRGBA(padded, width: square, height: square, outWidth: side, outHeight: side)
        var prepared = [UInt8](repeating: 0, count: side * side * 4)
        for pixel in 0 ..< side * side {
            let alpha = Float(resized[pixel * 4 + 3]) / 255
            for c in 0 ..< 3 {
                prepared[pixel * 4 + c] = UInt8(Float(resized[pixel * 4 + c]) / 255 * alpha * 255)
            }
            prepared[pixel * 4 + 3] = UInt8(alpha * 255)
        }
        return (prepared, side, side)
    }

    /// BiRefNet's binary mask of opaque RGB8 pixels as the release forms it: Pillow's bilinear resize to
    /// 1024², ImageNet normalization, the sigmoid of the logit scaled to 8 bits by truncation, Pillow's
    /// bicubic resize back, and the pixels above 128.
    static func birefnetMask(rgb: [UInt8], width: Int, height: Int, model: NFKMLXBiRefNetModel) -> [UInt8] {
        let side = NFKMLXBiRefNet.inputSize
        let resized = NFKMLXPillowLanczos.resize(rgb, width: width, height: height, outWidth: side, outHeight: side, filter: .bilinear)
        let mean = MLXArray([Float(0.485), 0.456, 0.406]), deviation = MLXArray([Float(0.229), 0.224, 0.225])
        let pixels = (MLXArray(resized.map { Float($0) / 255 }).reshaped([1, side, side, 3]) - mean) / deviation
        let probability = sigmoid(model.logit(pixels)).reshaped([side * side])
        eval(probability)
        let bytes = (probability.asArray(Float.self)).map { UInt8($0 * 255) }
        let mask = NFKMLXPillowLanczos.resize(bytes, width: side, height: side, outWidth: width, outHeight: height, channels: 1,
                                              filter: .bicubic)
        return mask.map { $0 > 128 ? 1 : 0 }
    }
}

/// Hi3DGen's sampler settings: the release's `app.py` runs the structure stage for 50 steps and the latent
/// stage for 6, both at guidance 3, inside `pipeline.json`'s guidance interval and rescale.
extension NFKMLXTrellisSampling {
    static func hi3dgen(_ release: NFKMLXTrellisSampling, steps: Int) -> NFKMLXTrellisSampling {
        var sampling = release
        sampling.steps = steps
        sampling.guidance = 3
        return sampling
    }
}

/// What a Hi3DGen run produces: the normal map it conditions on and the mesh.
struct NFKMLXHi3DGenResult {
    let normalMap: [UInt8]
    let normalMapSide: Int
    let mesh: NFKMesh?
}

/// Hi3DGen end to end: preparation with BiRefNet, StableNormal-turbo's normal map, and TRELLIS's stages on
/// the normal map with Hi3DGen's flows and marching-cubes extraction. Introduced in InferKit 0.4.0.
///
/// The release's structure and mesh decoders are TRELLIS-image-large's, so its flows fine-tune through
/// `NFKMLXTrellisTraining`: ``condition(rgba:width:height:)`` makes an example's condition from an image, and
/// `NFKMLXTrellisTraining.structureLatent(of:encoder:)` with TRELLIS-image-large's structure encoder makes
/// its structure latent from a mesh.
public final class NFKMLXHi3DGenPipeline: @unchecked Sendable {
    /// The TRELLIS stages: the DINOv2 encoder, Hi3DGen's flows, and the decoders.
    public let trellis: NFKMLXTrellisPipeline
    let normals: NFKTrellisStage<NFKMLXStableNormalModel>
    let birefnet: NFKTrellisStage<NFKMLXBiRefNetModel>
    /// The release app's structure sampling: 50 steps at guidance 3.
    public let structureSampling: NFKMLXTrellisSampling
    /// The release app's latent sampling: 6 steps at guidance 3.
    public let latentSampling: NFKMLXTrellisSampling

    /// The side the prepared image and the normal map are formed at.
    static let preparedSide = 1024
    /// The resolution StableNormal-turbo runs at.
    static let normalResolution = 768

    /// Builds from the Hi3DGen release directory, the StableNormal-turbo release directory, DINOv2 ViT-L/14
    /// with registers (`dinov2_vitl14_reg4_pretrain.pth`), and BiRefNet's `model.safetensors`. `flows`, where
    /// set, replace the release's flow checkpoints with files `NFKMLXWeights.save` writes after
    /// `NFKMLXTrellisTraining`'s recipes.
    public init(directoryURL: URL, normalDirectoryURL: URL, encoderWeightsURL: URL, birefnetWeightsURL: URL,
                residency: NFKMLXResidency = .automatic, flows: NFKMLXTrellisFlowWeights = NFKMLXTrellisFlowWeights()) throws {
        let release = try NFKMLXTrellisRelease(directoryURL: directoryURL, imageReleaseDirectoryURL: directoryURL)
        guard release.isImageConditioned else {
            throw NFKMLXError.unsupportedConfiguration("\(directoryURL.lastPathComponent) is not an image-conditioned release")
        }
        trellis = try NFKMLXTrellisPipeline(release: release, encoderWeightsURL: encoderWeightsURL, textEncoderDirectoryURL: nil,
                                            quantizationBits: nil, residency: residency, overrides: flows,
                                            surfaceExtraction: .marchingCubes)
        let resident = trellis.holdsStagesResident
        normals = NFKTrellisStage(keepsResident: resident) { try NFKMLXStableNormalModel.load(directoryURL: normalDirectoryURL) }
        birefnet = NFKTrellisStage(keepsResident: resident) {
            let model = NFKMLXBiRefNetModel()
            try NFKMLXBiRefNet.loadWeights(into: model, from: birefnetWeightsURL)
            return model
        }
        structureSampling = .hi3dgen(release.structureSampling, steps: 50)
        latentSampling = .hi3dgen(release.latentSampling, steps: 6)
    }

    /// The prepared 1024² RGBA from straight RGBA8 pixels.
    func prepared(rgba: [UInt8], width: Int, height: Int) throws -> (pixels: [UInt8], width: Int, height: Int) {
        try NFKHi3DGenImagePreparation.prepare(rgba: rgba, width: width, height: height, side: Self.preparedSide) { rgb, w, h in
            try birefnet.use { NFKHi3DGenImagePreparation.birefnetMask(rgb: rgb, width: w, height: h, model: $0) }
        }
    }

    /// The normal map (RGB8 at the prepared image's size) of prepared RGBA8 pixels.
    func normalMap(prepared: (pixels: [UInt8], width: Int, height: Int)) throws -> [UInt8] {
        try normals.use { try $0.normalMap(rgba: prepared.pixels, width: prepared.width, height: prepared.height,
                                           resolution: Self.normalResolution) }
    }

    /// The TRELLIS condition of a normal map: the map resized to the encoder's side with Pillow's Lanczos.
    func condition(normalMap: [UInt8], width: Int, height: Int) throws -> (condition: MLXArray, unconditional: MLXArray) {
        let side = trellis.configuration.imageSize
        let resized = NFKMLXPillowLanczos.resize(normalMap, width: width, height: height, outWidth: side, outHeight: side)
        return try trellis.condition(prepared: resized)
    }

    /// The condition and the unconditional condition the flows read for straight RGBA8 pixels: the prepared
    /// image's normal map, encoded by DINOv2.
    public func condition(rgba: [UInt8], width: Int, height: Int) throws -> (condition: MLXArray, unconditional: MLXArray) {
        let prepared = try self.prepared(rgba: rgba, width: width, height: height)
        return try condition(normalMap: try normalMap(prepared: prepared), width: prepared.width, height: prepared.height)
    }

    func generate(rgba: [UInt8], width: Int, height: Int, seed: UInt64, structureSampling: NFKMLXTrellisSampling? = nil,
                  latentSampling: NFKMLXTrellisSampling? = nil) throws -> NFKMLXHi3DGenResult {
        let prepared = try self.prepared(rgba: rgba, width: width, height: height)
        let map = try normalMap(prepared: prepared)
        let condition = try self.condition(normalMap: map, width: prepared.width, height: prepared.height)
        let result = try trellis.generate(condition: condition.condition, unconditional: condition.unconditional, seed: seed,
                                          outputs: .mesh, structureSampling: structureSampling ?? self.structureSampling,
                                          latentSampling: latentSampling ?? self.latentSampling)
        return NFKMLXHi3DGenResult(normalMap: map, normalMapSide: prepared.width, mesh: result.mesh)
    }
}

/// Hi3DGen as a backend: an image under `NFKInputImage`, the mesh under `NFKOutputMesh` and the normal map it
/// conditions on under `NFKOutputImage`. Parameters: `NFKParameterSeed` (default 42) and the TRELLIS step and
/// guidance keys (`NFKMLXTrellisParameterKey`). Introduced in InferKit 0.4.0.
@objc(NFKMLXHi3DGenBackend)
public final class NFKMLXHi3DGenBackend: NSObject, NFKInferenceBackend {
    let pipeline: NFKMLXHi3DGenPipeline
    private let lock = NSLock()

    init(pipeline: NFKMLXHi3DGenPipeline) {
        self.pipeline = pipeline
        super.init()
    }

    @objc public var isReady: Bool { true }
    @objc public var backendIdentifier: String { NFKMLXHi3DGenFactory.modelName }
    /// Whether the networks stay loaded between runs (else each loads for its stage).
    @objc public var holdsStagesResident: Bool { pipeline.trellis.holdsStagesResident }
    @objc public var supportedInputKeys: Set<String> { [NFKInputImage] }
    @objc public var supportedOutputKeys: Set<String> { [NFKOutputMesh, NFKOutputImage] }
    @objc public var supportedParameterKeys: Set<String> {
        [NFKParameterSeed, NFKMLXTrellisParameterKey.structureSteps, NFKMLXTrellisParameterKey.latentSteps,
         NFKMLXTrellisParameterKey.structureGuidance, NFKMLXTrellisParameterKey.latentGuidance]
    }

    @objc(runInferenceForRequest:error:)
    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        guard let value = request.input(forKey: NFKInputImage) else {
            throw NFKMLXError.unsupportedInput
        }
        let image = try NFKMLXImageBridge.rgbaBytes(from: value, colorSpace: CGColorSpaceCreateDeviceRGB())
        func number(_ key: String) -> NSNumber? { request.parameter(forKey: key) as? NSNumber }
        var structure = pipeline.structureSampling, latent = pipeline.latentSampling
        if let steps = number(NFKMLXTrellisParameterKey.structureSteps) { structure.steps = max(1, steps.intValue) }
        if let steps = number(NFKMLXTrellisParameterKey.latentSteps) { latent.steps = max(1, steps.intValue) }
        if let guidance = number(NFKMLXTrellisParameterKey.structureGuidance) { structure.guidance = guidance.floatValue }
        if let guidance = number(NFKMLXTrellisParameterKey.latentGuidance) { latent.guidance = guidance.floatValue }
        let seed = number(NFKParameterSeed)?.uint64Value ?? 42
        let result = try lock.withLock {
            try pipeline.generate(rgba: image.bytes, width: image.width, height: image.height, seed: seed,
                                  structureSampling: structure, latentSampling: latent)
        }
        let side = result.normalMapSide
        var rgba = [UInt8](repeating: 255, count: side * side * 4)
        for pixel in 0 ..< side * side {
            for c in 0 ..< 3 { rgba[pixel * 4 + c] = result.normalMap[pixel * 3 + c] }
        }
        var values = [String: Any]()
        values[NFKOutputMesh] = result.mesh
        values[NFKOutputImage] = try NFKMLXImageBridge.cgImage(rgba: rgba, width: side, height: side, options: NFKMLXImageOptions())
        return NFKInferenceResult(outputs: values)
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

/// Delivers an asynchronous factory's result across the detached task.
private final class NFKHi3DGenCompletion: @unchecked Sendable {
    let handler: ((any NFKInferenceBackend)?, Error?) -> Void
    init(_ handler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) { self.handler = handler }
    func call(_ backend: (any NFKInferenceBackend)?, _ error: Error?) { handler(backend, error) }
}

/// Hi3DGen's factories, registration, and release facts. Introduced in InferKit 0.4.0.
@objc(NFKMLXHi3DGenFactory)
public final class NFKMLXHi3DGenFactory: NSObject {
    /// The registry name.
    @objc public static let modelName = "hi3dgen"
    static let repo = "Stable-X/trellis-normal-v0-1"
    static let revision = "fdfac1d0e0cb3f86adcac69239c1abdd82228e17"
    static let checkpoints = ["ss_dec_conv3d_16l8_fp16", "ss_flow_normal_dit_L_16l8_fp16", "slat_flow_normal_dit_L_64l8p2_fp16",
                              "slat_dec_mesh_swin8_B_64l8m256c_fp16"]
    static let birefnetRepo = "ZhengPeng7/BiRefNet"

    /// Builds from local files: the Hi3DGen release directory, the StableNormal-turbo release directory, DINOv2
    /// ViT-L/14 with registers (`dinov2_vitl14_reg4_pretrain.pth`), and BiRefNet's `model.safetensors`.
    /// `residency` holds the stages resident or loads each for its stage.
    @objc(backendWithDirectoryURL:normalDirectoryURL:encoderWeightsURL:birefnetWeightsURL:residency:error:)
    public static func backend(directoryURL: URL, normalDirectoryURL: URL, encoderWeightsURL: URL, birefnetWeightsURL: URL,
                               residency: NFKMLXResidency) throws -> any NFKInferenceBackend {
        NFKMLXHi3DGenBackend(pipeline: try NFKMLXHi3DGenPipeline(directoryURL: directoryURL, normalDirectoryURL: normalDirectoryURL,
                                                                 encoderWeightsURL: encoderWeightsURL,
                                                                 birefnetWeightsURL: birefnetWeightsURL, residency: residency))
    }

    /// The release with fine-tuned flows: `structureFlowWeightsURL` and `latentFlowWeightsURL`, where set,
    /// replace the release's checkpoints (files `NFKMLXWeights.save` writes after `NFKMLXTrellisTraining`'s
    /// recipes). The other arguments follow
    /// ``backend(directoryURL:normalDirectoryURL:encoderWeightsURL:birefnetWeightsURL:residency:)``.
    @objc(backendWithDirectoryURL:normalDirectoryURL:encoderWeightsURL:birefnetWeightsURL:structureFlowWeightsURL:latentFlowWeightsURL:residency:error:)
    public static func backend(directoryURL: URL, normalDirectoryURL: URL, encoderWeightsURL: URL, birefnetWeightsURL: URL,
                               structureFlowWeightsURL: URL?, latentFlowWeightsURL: URL?,
                               residency: NFKMLXResidency) throws -> any NFKInferenceBackend {
        let flows = NFKMLXTrellisFlowWeights(structure: structureFlowWeightsURL, latent: latentFlowWeightsURL)
        return NFKMLXHi3DGenBackend(pipeline: try NFKMLXHi3DGenPipeline(directoryURL: directoryURL, normalDirectoryURL: normalDirectoryURL,
                                                                        encoderWeightsURL: encoderWeightsURL,
                                                                        birefnetWeightsURL: birefnetWeightsURL, residency: residency,
                                                                        flows: flows))
    }

    /// Downloads the Hi3DGen release at `revision` (nil for the pinned one), StableNormal-turbo, DINOv2, and
    /// BiRefNet into `cacheDirectoryURL` (nil for the default cache), then builds. Blocking on the network;
    /// run off the render thread.
    @objc(backendWithRevision:cacheDirectoryURL:residency:error:)
    public static func backend(revision: String?, cacheDirectoryURL: URL?, residency: NFKMLXResidency) throws -> any NFKInferenceBackend {
        try NFKMLXBuildHolds.keeping {
            let files = try download(revision: revision, cacheDirectoryURL: cacheDirectoryURL)
            return try backend(directoryURL: files.release, normalDirectoryURL: files.normals, encoderWeightsURL: files.encoder,
                               birefnetWeightsURL: files.birefnet, residency: residency)
        }
    }

    /// The asynchronous form of ``backend(revision:cacheDirectoryURL:residency:)``.
    @objc(backendWithRevision:cacheDirectoryURL:residency:completionHandler:)
    public static func backend(revision: String?, cacheDirectoryURL: URL?, residency: NFKMLXResidency,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        let handler = NFKHi3DGenCompletion(completionHandler)
        Task.detached(priority: .userInitiated) {
            do {
                handler.call(try backend(revision: revision, cacheDirectoryURL: cacheDirectoryURL, residency: residency), nil)
            } catch {
                handler.call(nil, error)
            }
        }
    }

    struct Files {
        let release: URL
        let normals: URL
        let encoder: URL
        let birefnet: URL
    }

    static func download(revision: String?, cacheDirectoryURL: URL?) throws -> Files {
        let release = try NFKMLXReleaseDownload.directory(repo: repo, revision: revision ?? self.revision, cacheDirectoryURL: cacheDirectoryURL,
                                                          components: NFKMLXTrellisFactory.components(checkpoints))
        let normals = try NFKMLXReleaseDownload.directory(repo: NFKMLXStableNormal.repo, revision: NFKMLXStableNormal.revision,
                                                          cacheDirectoryURL: cacheDirectoryURL, components: NFKMLXStableNormal.components)
        let encoder = try NFKMLXTrellisFactory.downloadedEncoder(cacheDirectoryURL: cacheDirectoryURL)
        let birefnet = try NFKMLXDownload.weightsURL(repo: birefnetRepo, weightsPath: "model.safetensors", revision: nil,
                                                     cacheDirectoryURL: cacheDirectoryURL)
        return Files(release: release, normals: normals, encoder: encoder, birefnet: birefnet)
    }

    /// Registers Hi3DGen with `NFKMLXModelRegistry` under `hi3dgen`. The location is the release directory a
    /// catalog download fills; StableNormal-turbo, DINOv2, and BiRefNet come from the cache, downloaded when
    /// missing. `NFKModelLoadOptionResidency` holds the stages resident or loads each for its stage.
    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName, consumes: [NFKModelLoadOptionResidency]) { location, options in
            var files = try download(revision: nil, cacheDirectoryURL: nil)
            if let location {
                files = Files(release: location, normals: files.normals, encoder: files.encoder, birefnet: files.birefnet)
            }
            return try backend(directoryURL: files.release, normalDirectoryURL: files.normals, encoderWeightsURL: files.encoder,
                               birefnetWeightsURL: files.birefnet, residency: options.residency)
        }
    }
}
