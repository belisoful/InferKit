//
//  NFKMLXStableFast3DBackend.swift
//  InferKitMLX
//

import CoreGraphics
import Foundation
import ImageIO
import InferKit
import MLX
import MLXNN
import UniformTypeIdentifiers

// MARK: - Request options

/// The request-parameter keys the Stable Fast 3D backend reads. Introduced in InferKit 0.4.0.
@objc(NFKMLXStableFast3DParameterKey)
public final class NFKMLXStableFast3DParameterKey: NSObject {
    /// The side of the baked texture atlas in pixels (NSNumber, default 1024).
    @objc public static let textureResolution = "NFKMLXParameterTextureResolution"
    /// The density the surface is drawn at (NSNumber, default 10).
    @objc public static let threshold = NFKMLXTripoSRParameterKey.threshold
    /// The share of the image's side the object is scaled to fill when the image carries alpha (NSNumber in
    /// 0...1, default 0.85). An image without alpha is used as it is.
    @objc public static let foregroundRatio = NFKMLXTripoSRParameterKey.foregroundRatio
    /// The texture files' image type, `NFKMeshTextureMIMETypePNG` (default) or `NFKMeshTextureMIMETypeJPEG`
    /// (NSString), which the release writes.
    @objc public static let textureFormat = "NFKMLXParameterTextureFormat"
    /// Whether the result also holds the illumination estimate under
    /// `NFKMLXStableFast3DOutputKey.illuminationAmplitudes` (NSNumber boolean, default NO).
    @objc public static let estimateIllumination = "NFKMLXParameterEstimateIllumination"
    /// Whether the network and its field decoder run on the CPU in float32, each operation as the release's
    /// torch 2.14.1 runs it on Apple silicon, which reproduces the release's mesh exactly (NSNumber boolean,
    /// default NO). The run takes several times the GPU's time and needs the backend's checkpoint file.
    @objc public static let referenceArithmetic = "NFKMLXParameterReferenceArithmetic"
}

/// The result keys the Stable Fast 3D backend writes beside `NFKOutputMesh`. Introduced in InferKit 0.4.0.
@objc(NFKMLXStableFast3DOutputKey)
public final class NFKMLXStableFast3DOutputKey: NSObject {
    /// The illumination head's 24 spherical-Gaussian light amplitudes (NSArray of NSNumber), when
    /// `NFKMLXStableFast3DParameterKey.estimateIllumination` asks for them. The release publishes no lobe
    /// axes or sharpnesses to pair them with.
    @objc public static let illuminationAmplitudes = "NFKMLXOutputIlluminationAmplitudes"
}

/// The options a run takes.
public struct NFKMLXStableFast3DOptions: Sendable {
    public var textureResolution: Int = 1024
    public var threshold: Float = 10
    /// A float64, as the reference demo's `--foreground-ratio` is.
    public var foregroundRatio: Double = 0.85
    public var textureFormat: String = NFKMeshTextureMIMETypePNG
    public var estimateIllumination: Bool = false
    /// Runs the network and its field decoder in the reference's arithmetic
    /// (`NFKMLXStableFast3DParameterKey.referenceArithmetic`).
    public var referenceArithmetic: Bool = false
    /// The seed of the texture quantization's dither.
    public var seed: UInt64 = 0
    public init() {}
    public static let `default` = NFKMLXStableFast3DOptions()
}

/// One reconstruction: the textured mesh and, when asked, the illumination estimate.
public struct NFKMLXStableFast3DReconstruction {
    public let mesh: NFKMesh
    public let illuminationAmplitudes: [Float]?
}

// MARK: - Backend

private final class NFKSF3DHolder: @unchecked Sendable {
    let net: NFKMLXStableFast3DNet
    let grid: NFKMLXTetrahedralGrid
    let weightsURL: URL?
    private let lock = NSLock()
    private var reference: NFKMLXStableFast3DReferenceNetwork?

    init(_ net: NFKMLXStableFast3DNet, grid: NFKMLXTetrahedralGrid, weightsURL: URL?) {
        self.net = net
        self.grid = grid
        self.weightsURL = weightsURL
    }

    /// The reference-arithmetic network over the checkpoint file, built on first use.
    func referenceNetwork() throws -> NFKMLXStableFast3DReferenceNetwork {
        lock.lock()
        defer { lock.unlock() }
        if let reference { return reference }
        guard let weightsURL else {
            throw NFKMLXError.unsupportedConfiguration("the reference arithmetic reads the checkpoint file, and this backend was built without one")
        }
        let built = try NFKMLXStableFast3DReferenceNetwork(weightsURL: weightsURL, configuration: net.configuration)
        reference = built
        return built
    }
}

/// Stable Fast 3D as an InferKit backend: an image under `NFKInputImage` in, a textured `NFKMesh` under
/// `NFKOutputMesh` out.
@objc(NFKMLXStableFast3DBackend)
public final class NFKMLXStableFast3DBackend: NSObject, NFKInferenceBackend {

    private let holder: NFKSF3DHolder
    private let identifier: String

    init(net: NFKMLXStableFast3DNet, grid: NFKMLXTetrahedralGrid, weightsURL: URL?, identifier: String) {
        holder = NFKSF3DHolder(net, grid: grid, weightsURL: weightsURL)
        self.identifier = identifier
        super.init()
    }

    @objc public var isReady: Bool { true }
    @objc public var backendIdentifier: String { identifier }

    @objc public var supportedParameterKeys: Set<String> {
        [NFKMLXStableFast3DParameterKey.textureResolution, NFKMLXStableFast3DParameterKey.threshold,
         NFKMLXStableFast3DParameterKey.foregroundRatio, NFKMLXStableFast3DParameterKey.textureFormat,
         NFKMLXStableFast3DParameterKey.estimateIllumination, NFKMLXStableFast3DParameterKey.referenceArithmetic, NFKParameterSeed]
    }
    @objc public var supportedInputKeys: Set<String> { [NFKInputImage] }
    @objc public var supportedOutputKeys: Set<String> { [NFKOutputMesh, NFKMLXStableFast3DOutputKey.illuminationAmplitudes] }

    @objc(runInferenceForRequest:error:)
    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        guard let value = request.input(forKey: NFKInputImage) else {
            throw NFKMLXError.unsupportedInput
        }
        let options = try Self.options(from: request)
        let image = try NFKMLXImageBridge.rgbaBytes(from: value, colorSpace: CGColorSpaceCreateDeviceRGB())
        let reconstruction = options.referenceArithmetic
            ? try holder.referenceNetwork().reconstruct(rgba: image.bytes, width: image.width, height: image.height, grid: holder.grid,
                                                        net: holder.net, options: options)
            : try holder.net.reconstruct(rgba: image.bytes, width: image.width, height: image.height, grid: holder.grid, options: options)
        var outputs: [String: Any] = [NFKOutputMesh: reconstruction.mesh]
        if let amplitudes = reconstruction.illuminationAmplitudes {
            outputs[NFKMLXStableFast3DOutputKey.illuminationAmplitudes] = amplitudes.map { NSNumber(value: $0) }
        }
        return NFKInferenceResult(outputs: outputs)
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

    static func options(from request: NFKInferenceRequest) throws -> NFKMLXStableFast3DOptions {
        var options = NFKMLXStableFast3DOptions.default
        func number(_ key: String) -> NSNumber? { request.parameter(forKey: key) as? NSNumber }
        if let value = number(NFKMLXStableFast3DParameterKey.textureResolution) {
            options.textureResolution = min(8192, max(16, value.intValue))
        }
        if let value = number(NFKMLXStableFast3DParameterKey.threshold) { options.threshold = value.floatValue }
        if let value = number(NFKMLXStableFast3DParameterKey.foregroundRatio) {
            options.foregroundRatio = min(1, max(0.05, value.doubleValue))
        }
        if let value = request.parameter(forKey: NFKMLXStableFast3DParameterKey.textureFormat) as? String {
            guard value == NFKMeshTextureMIMETypePNG || value == NFKMeshTextureMIMETypeJPEG else {
                throw NFKMLXError.unsupportedConfiguration("a texture is written as \(NFKMeshTextureMIMETypePNG) or \(NFKMeshTextureMIMETypeJPEG), not \(value)")
            }
            options.textureFormat = value
        }
        if let value = number(NFKMLXStableFast3DParameterKey.estimateIllumination) { options.estimateIllumination = value.boolValue }
        if let value = number(NFKMLXStableFast3DParameterKey.referenceArithmetic) { options.referenceArithmetic = value.boolValue }
        if let value = number(NFKParameterSeed) { options.seed = value.uint64Value }
        return options
    }
}

extension NFKMLXStableFast3DNet {

    /// The whole run from straight RGBA8 pixels: the demo's framing and the release's `prepare_image`
    /// (`NFKMLXStableFast3D.conditioning(rgba:width:height:side:foregroundRatio:background:)`), the forward,
    /// the material estimate, and the mesh (`NFKMLXStableFast3DGeometry.mesh(sceneCode:net:grid:roughness:metallic:options:)`).
    public func reconstruct(rgba: [UInt8], width: Int, height: Int, grid: NFKMLXTetrahedralGrid,
                            options: NFKMLXStableFast3DOptions) throws -> NFKMLXStableFast3DReconstruction {
        let (color, mask) = NFKMLXStableFast3D.conditioning(rgba: rgba, width: width, height: height, side: configuration.imageSize,
                                                             foregroundRatio: options.foregroundRatio,
                                                             background: configuration.background)
        let images = color.expandedDimensions(axis: 0)
        let tokens = triplaneTokens(imageTokens(images))
        let sceneCode = sceneCodes(tokens)[0]
        let materials = self.materials(images * mask.expandedDimensions(axis: 0))
        var amplitudes: MLXArray? = nil
        if options.estimateIllumination {
            amplitudes = illumination(tokens)[0]
        }
        eval(sceneCode, materials.roughness, materials.metallic)
        if let amplitudes { eval(amplitudes) }
        let mesh = try NFKMLXStableFast3DGeometry.mesh(sceneCode: sceneCode, net: self, grid: grid,
                                                       roughness: materials.roughness[0].item(Float.self),
                                                       metallic: materials.metallic[0].item(Float.self), options: options)
        return NFKMLXStableFast3DReconstruction(mesh: mesh, illuminationAmplitudes: amplitudes?.asArray(Float.self))
    }
}

// MARK: - Factories and loading

/// Stable Fast 3D as an InferKit backend, and its registration for the Objective-C path.
@objc(NFKMLXStableFast3D)
public final class NFKMLXStableFast3D: NSObject {

    /// The registry name the model builds under.
    @objc public static let modelName = "stable-fast-3d"
    /// The release repository and the revision the port is measured at.
    @objc public static let repository = "stabilityai/stable-fast-3d"
    @objc public static let revision = "f0c9a8ffd62cb1bbc8a7a53c9f87a0be1b6be778"
    /// The checkpoint's file name in the release.
    @objc public static let weightsFileName = "model.safetensors"
    /// The tetrahedral grid's file name. The release's code repository holds it; the weights repository does not.
    @objc public static let gridFileName = "160_tets.npz"
    /// The grid's address, at the code repository's commit the port is measured at.
    @objc public static let gridURL = URL(string: "https://raw.githubusercontent.com/Stability-AI/stable-fast-3d/ff21fc491b4dc5314bf6734c7c0dabd86b5f5bb2/load/tets/160_tets.npz")!

    static func makeNet(_ configuration: NFKMLXStableFast3DConfiguration = .released) -> NFKMLXStableFast3DNet {
        NFKMLXStableFast3DNet(configuration)
    }

    /// The network at `configuration`, with the checkpoint at `weightsURL` loaded. Nil weights is the random
    /// initialization. A `dtype` converts every floating tensor as it loads (the release is float32;
    /// `.bfloat16` halves it); `quantizationBits` then packs every linear layer whose input width is a
    /// multiple of `groupSize` (the heads' first layers, 120 wide, stay as loaded). Introduced in InferKit 0.4.0.
    public static func network(weightsURL: URL?, configuration: NFKMLXStableFast3DConfiguration = .released,
                               castingTo dtype: DType? = nil, quantizationBits: Int? = nil,
                               groupSize: Int = NFKMLXLoadOptions.defaultQuantizationGroupSize) throws -> NFKMLXStableFast3DNet {
        let net = makeNet(configuration)
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL, castingTo: dtype)
        } else if let dtype {
            net.update(parameters: net.parameters().mapValues { $0.asType(dtype) })
        }
        if let bits = quantizationBits {
            guard (2 ... 8).contains(bits) else {
                throw NFKMLXError.unsupportedConfiguration("a packed load takes 2 to 8 bits, not \(bits)")
            }
            // MLX's quantizer aborts the process on any other group size.
            guard [32, 64, 128].contains(groupSize) else {
                throw NFKMLXError.unsupportedConfiguration("a packed load takes a group size of 32, 64, or 128, not \(groupSize)")
            }
            try NFKMLXQuantization.quantize(module: net, bits: bits, groupSize: groupSize)
        }
        return net
    }

    /// Builds a Stable Fast 3D backend from the release's checkpoint (`model.safetensors`) and the
    /// tetrahedral grid (`160_tets.npz`). A nil `weightsURL` builds random weights (`isReady` is true). Run
    /// inference off the render thread.
    @objc(backendWithWeightsURL:gridURL:error:)
    public static func backend(weightsURL: URL?, gridURL: URL) throws -> any NFKInferenceBackend {
        NFKMLXStableFast3DBackend(net: try network(weightsURL: weightsURL), grid: try NFKMLXTetrahedralGrid(contentsOf: gridURL),
                                  weightsURL: weightsURL, identifier: modelName)
    }

    /// Builds from a checkpoint and a grid already read, such as `NFKMLXTetrahedralGrid.lattice(divisions:)`.
    @objc(backendWithWeightsURL:grid:error:)
    public static func backend(weightsURL: URL?, grid: NFKMLXTetrahedralGrid) throws -> any NFKInferenceBackend {
        NFKMLXStableFast3DBackend(net: try network(weightsURL: weightsURL), grid: grid, weightsURL: weightsURL, identifier: modelName)
    }

    /// Builds from a directory holding the checkpoint and the grid side by side.
    @objc(backendWithDirectoryURL:error:)
    public static func backend(directoryURL: URL) throws -> any NFKInferenceBackend {
        try backend(weightsURL: directoryURL.appendingPathComponent(weightsFileName),
                    gridURL: directoryURL.appendingPathComponent(gridFileName))
    }

    /// Downloads the checkpoint from Hugging Face and the grid from its code repository into the cache,
    /// then builds. Blocking on the network; run off the render thread. A nil `revision` reads the
    /// measured one.
    @objc(backendWithRepo:revision:cacheDirectoryURL:error:)
    public static func backend(repo: String, revision: String?, cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        try NFKMLXBuildHolds.keeping {
            let weights = try NFKMLXDownload.weightsURL(repo: repo, weightsPath: weightsFileName, revision: revision ?? self.revision,
                                                        cacheDirectoryURL: cacheDirectoryURL)
            return try backend(weightsURL: weights, gridURL: try downloadedGrid(cacheDirectoryURL: cacheDirectoryURL))
        }
    }

    /// The asynchronous form of the download factory: downloads on a background queue, then builds and
    /// delivers the backend (or an error) to `completionHandler`.
    @objc(backendWithRepo:revision:cacheDirectoryURL:completionHandler:)
    public static func backend(repo: String, revision: String?, cacheDirectoryURL: URL?,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        let handler = NFKSF3DCompletion(completionHandler)
        Task.detached(priority: .userInitiated) {
            do {
                handler.call(try backend(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL), nil)
            } catch {
                handler.call(nil, error)
            }
        }
    }

    /// Registers Stable Fast 3D (`stable-fast-3d`) with `NFKMLXModelRegistry`. The location is the release
    /// directory a catalog download fills (the grid comes from the cache's URL downloads when the
    /// directory does not hold it) or the checkpoint itself. `NFKModelLoadOptionPrecision` `bfloat16`
    /// converts the float32 release as it loads and `checkpoint` keeps it; `NFKModelLoadOptionQuantizationBits`
    /// packs the linear layers.
    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName, consumes: [NFKModelLoadOptionPrecision, NFKModelLoadOptionQuantizationBits,
                                                                NFKModelLoadOptionQuantizationGroupSize]) { location, options in
            let files = try releaseFiles(at: location)
            let net = try network(weightsURL: files.weights, castingTo: try options.bfloat16Cast(of: modelName),
                                  quantizationBits: options.quantizationBits, groupSize: options.quantizationGroupSize)
            return NFKMLXStableFast3DBackend(net: net, grid: try NFKMLXTetrahedralGrid(contentsOf: files.grid), weightsURL: files.weights,
                                             identifier: modelName)
        }
    }

    /// The checkpoint and the grid for a registry location: a directory holding the checkpoint (and the
    /// grid, or a cache whose URL downloads hold it), or the checkpoint file.
    static func releaseFiles(at location: URL?) throws -> (weights: URL?, grid: URL) {
        guard let location else {
            throw NFKMLXError.weightsMismatch("Stable Fast 3D needs its release directory, which holds \(weightsFileName)")
        }
        var isDirectory: ObjCBool = false
        FileManager.default.fileExists(atPath: location.path, isDirectory: &isDirectory)
        let directory = isDirectory.boolValue ? location : location.deletingLastPathComponent()
        let weights = isDirectory.boolValue ? directory.appendingPathComponent(weightsFileName) : location
        let beside = directory.appendingPathComponent(gridFileName)
        if FileManager.default.fileExists(atPath: beside.path) {
            return (weights, beside)
        }
        // The cache keeps a repository's snapshot at <cache>/<org>/<name>/<revision> and a URL download
        // under <cache>/@url.
        var roots = [NFKHFHub.defaultCacheDirectoryURL()]
        var root = directory
        for _ in 0 ..< 3 { root = root.deletingLastPathComponent() }
        roots.insert(root, at: 0)
        for root in roots {
            if let cached = NFKHFHub(cacheDirectoryURL: root).localURL(for: gridURL, fileName: nil),
               FileManager.default.fileExists(atPath: cached.path) {
                return (weights, cached)
            }
        }
        throw NFKMLXError.weightsMismatch("the tetrahedral grid \(gridFileName) is not beside \(weightsFileName) or downloaded")
    }

    /// The grid in the cache's URL downloads, fetched first when it is not there.
    static func downloadedGrid(cacheDirectoryURL: URL?) throws -> URL {
        let hub = NFKMLXHub.hub(cacheDirectoryURL: cacheDirectoryURL)
        guard let local = hub.localURL(for: gridURL, fileName: nil) else {
            throw NFKMLXError.weightsMismatch("the cache has no place for \(gridURL.absoluteString)")
        }
        if !FileManager.default.fileExists(atPath: local.path) {
            try FileManager.default.createDirectory(at: local.deletingLastPathComponent(), withIntermediateDirectories: true)
            try hub.fetch(gridURL, toFileURL: local)
        }
        return local
    }

    /// Loads the release checkpoint or a safetensors file into `net`.
    ///
    /// The release stores DINOv2's position embeddings at its 518-pixel training grid (37 × 37 patches plus
    /// the class token); the forward reads the conditioning size, so they are resized here as the
    /// release's `interpolate_pos_encoding` resizes them on every forward. The mask token, the bounding
    /// box, and the material estimator's text tower, which the forward never reaches, are dropped. The
    /// estimator heads' nested lists take their module names (`NFKSF3DBetaHead.moduleKey(_:)`).
    /// Convolution weights move from PyTorch's layout to MLX's.
    public static func loadWeights(into net: NFKMLXStableFast3DNet, from url: URL, castingTo dtype: DType? = nil) throws {
        let checkpoint = try NFKMLXWeights.materializedCheckpoint(url: url) { key in
            !droppedKey(key)
        }
        var arrays = mapped(checkpoint.arrays, needsConvTranspose: checkpoint.needsConvTranspose, for: net)
        if let dtype {
            arrays = arrays.map { key, value in
                ([.float32, .float16, .bfloat16].contains(value.dtype) ? (key, value.asType(dtype)) : (key, value))
            }
        }
        try NFKMLXWeights.apply(arrays, to: net)
    }

    /// Whether the forward never reads a checkpoint tensor.
    static func droppedKey(_ key: String) -> Bool {
        if key == "bbox" || key == "image_tokenizer.model.embeddings.mask_token" {
            return true
        }
        return key.hasPrefix("image_estimator.model.") && !key.hasPrefix("image_estimator.model.visual.")
    }

    /// Checkpoint arrays under the module's keys and layout.
    static func mapped(_ arrays: [String: MLXArray], needsConvTranspose: Bool, for net: NFKMLXStableFast3DNet) -> [(String, MLXArray)] {
        let positionKey = "image_tokenizer.model.embeddings.position_embeddings"
        let configuration = net.configuration
        let expectedPositions = configuration.patchGrid * configuration.patchGrid + 1
        return arrays.compactMap { key, value -> (String, MLXArray)? in
            if droppedKey(key) {
                return nil
            }
            if key == positionKey && value.dim(1) != expectedPositions {
                return (key, resizedPositionEmbeddings(value, grid: configuration.patchGrid))
            }
            let name = NFKSF3DBetaHead.moduleKey(key)
            guard needsConvTranspose && value.ndim == 4 && key != "tokenizer.embeddings" else { return (name, value) }
            return (name, value.transposed(0, 2, 3, 1))
        }
    }

    /// The release's `Dinov2Embeddings.interpolate_pos_encoding`: the patch positions as a square grid,
    /// bicubic to `grid × grid` with `scale_factor` `(grid + 0.1) / side`, the class position kept first.
    /// torch maps each output sample by that factor, not by the size ratio, so the taps sit a little off the
    /// ones a resize to the size alone would read.
    static func resizedPositionEmbeddings(_ embeddings: MLXArray, grid: Int) -> MLXArray {
        let width = embeddings.dim(2)
        let count = embeddings.dim(1) - 1
        let side = Int(Double(count).squareRoot().rounded())
        let classPosition = embeddings[0..., 0 ..< 1, 0...]
        let patches = embeddings[0..., 1..., 0...].reshaped([side, side, width])
        // `area_pixel_compute_scale` with a scale given: the source step is float32(1 / scale_factor).
        let step = Double(Float(1 / ((Double(grid) + 0.1) / Double(side))))
        let matrix = NFKMLXPhi4MMImageProcessor.resampleMatrix(from: side, to: grid, scale: step)
        let vertical = matmul(matrix, patches.transposed(2, 0, 1))                           // [width, grid, side]
        let resized = matmul(vertical, matrix.transposed(1, 0)).transposed(1, 2, 0)          // [grid, grid, width]
        return concatenated([classPosition, resized.reshaped([1, grid * grid, width])], axis: 1)
    }

    // MARK: Preparation

    /// The conditioning image and its mask, `[side, side, 3]` and `[side, side, 1]`, from straight RGBA8
    /// pixels: the demo's `resize_foreground` framing when the image carries alpha, the release's
    /// `prepare_image` (Pillow's default bicubic resize of the RGBA image to `side`, which resamples the
    /// premultiplied colors, then the composite on `background` by `torch.lerp`), and the model's
    /// antialiased resize to `side`.
    public static func conditioning(rgba: [UInt8], width: Int, height: Int, side: Int, foregroundRatio: Double,
                                    background: Float) -> (color: MLXArray, mask: MLXArray) {
        let framed = frame(rgba: rgba, width: width, height: height, foregroundRatio: foregroundRatio)
        let resized = pilResized(framed.pixels, width: framed.width, height: framed.height, to: side)
        let values = MLXArray(resized.map { Float($0) / 255 }).reshaped([side, side, 4])
        let mask = values[0..., 0..., 3 ..< 4]
        let color = lerp(MLXArray(background), values[0..., 0..., 0 ..< 3], weight: mask)
        return (NFKMLXTripoSR.resizeAntialiased(color, side: side), NFKMLXTripoSR.resizeAntialiased(mask, side: side))
    }

    /// The conditioning image `[side, side, 3]` and mask `[side, side]` as the release's CPU run computes them:
    /// the framing and the Pillow resize, the bytes over 255 in float32, and `torch.lerp` onto the background
    /// as its vector kernel fuses it. The model's antialiased resize to the side the image already has
    /// returns each value unchanged.
    static func referenceConditioning(rgba: [UInt8], width: Int, height: Int, side: Int, foregroundRatio: Double,
                                      background: Float) -> (color: [Float], mask: [Float]) {
        let framed = frame(rgba: rgba, width: width, height: height, foregroundRatio: foregroundRatio)
        let resized = pilResized(framed.pixels, width: framed.width, height: framed.height, to: side)
        var color = [Float](repeating: 0, count: side * side * 3), mask = [Float](repeating: 0, count: side * side)
        for pixel in 0 ..< side * side {
            let weight = Float(resized[pixel * 4 + 3]) / 255
            mask[pixel] = weight
            let small = abs(weight) < 0.5
            for channel in 0 ..< 3 {
                let end = Float(resized[pixel * 4 + channel]) / 255
                color[pixel * 3 + channel] = (small ? background : end).addingProduct(small ? weight : weight - 1, end - background)
            }
        }
        return (color, mask)
    }

    /// `torch.lerp`, which reads from the nearer end for precision: `start + w · (end − start)` below a
    /// weight of one half, `end − (end − start) · (1 − w)` from it.
    static func lerp(_ start: MLXArray, _ end: MLXArray, weight: MLXArray) -> MLXArray {
        let difference = end - start
        return which(weight .< Float(0.5), start + weight * difference, end - difference * (1 - weight))
    }

    /// The demo's `resize_foreground`: the box of the pixels with alpha, a square of `max(h, w) / ratio`
    /// pixels around its center, cropped with transparent black outside the image. The box's sides are
    /// its last index minus its first, as `get_bbox_from_mask` measures them, and the crop's corner and side
    /// are the float64 values truncated toward zero. An image whose alpha is everywhere opaque is returned
    /// as it is, and `prepare_image` stretches it to the square.
    static func frame(rgba: [UInt8], width: Int, height: Int, foregroundRatio: Double) -> (pixels: [UInt8], width: Int, height: Int) {
        var minY = height, maxY = -1, minX = width, maxX = -1
        var opaque = true
        for y in 0 ..< height {
            for x in 0 ..< width {
                let alpha = rgba[(y * width + x) * 4 + 3]
                if alpha < 255 { opaque = false }
                if alpha > 0 {
                    minY = min(minY, y); maxY = max(maxY, y); minX = min(minX, x); maxX = max(maxX, x)
                }
            }
        }
        if opaque || maxY < minY {
            return (rgba, width, height)
        }
        let boxHeight = Double(maxY - minY), boxWidth = Double(maxX - minX)
        let centerY = Double(minY + maxY) / 2, centerX = Double(minX + maxX) / 2
        let scale = max(boxHeight, boxWidth) / foregroundRatio
        let top = Int(centerY - scale / 2), left = Int(centerX - scale / 2)
        let side = max(1, Int(scale))
        var out = [UInt8](repeating: 0, count: side * side * 4)
        for y in 0 ..< side {
            let sourceY = top + y
            guard sourceY >= 0 && sourceY < height else { continue }
            for x in 0 ..< side {
                let sourceX = left + x
                guard sourceX >= 0 && sourceX < width else { continue }
                let from = (sourceY * width + sourceX) * 4, to = (y * side + x) * 4
                out[to] = rgba[from]; out[to + 1] = rgba[from + 1]; out[to + 2] = rgba[from + 2]; out[to + 3] = rgba[from + 3]
            }
        }
        return (out, side, side)
    }

    /// Pillow's `Image.resize` of an RGBA image with its default filter, bicubic: the image converted to
    /// premultiplied `RGBa` (each color `MULDIV255(c, a)`), resampled in 8 bits, and converted back (each
    /// color `255 · c / a` truncated and clipped, an alpha of 0 or 255 copying the color).
    static func pilResized(_ rgba: [UInt8], width: Int, height: Int, to side: Int) -> [UInt8] {
        guard width != side || height != side else { return rgba }
        var premultiplied = rgba
        for pixel in stride(from: 0, to: rgba.count, by: 4) {
            let alpha = Int(rgba[pixel + 3])
            for channel in 0 ..< 3 {
                let product = Int(rgba[pixel + channel]) * alpha + 128
                premultiplied[pixel + channel] = UInt8(((product >> 8) + product) >> 8)
            }
        }
        var resized = NFKMLXPILResample.resampled(premultiplied, width: width, height: height, toWidth: side, toHeight: side,
                                                  filter: .bicubic, channels: 4)
        for pixel in stride(from: 0, to: resized.count, by: 4) {
            let alpha = Int(resized[pixel + 3])
            guard alpha != 0 && alpha != 255 else { continue }
            for channel in 0 ..< 3 {
                resized[pixel + channel] = UInt8(min(255, 255 * Int(resized[pixel + channel]) / alpha))
            }
        }
        return resized
    }

    // MARK: Small operations

    /// `F.softplus` at its defaults: `log(1 + eˣ)`, and `x` itself above 20.
    static func softplus(_ x: MLXArray) -> MLXArray {
        which(x .> Float(20), x, log1p(exp(x)))
    }

    /// torch's `Beta.mode`, read from its Dirichlet: `(α − 1)⁺ / ((α − 1)⁺ + (β − 1)⁺)`, and 1 where both
    /// concentrations lie below one (the one-hot of the first of two undefined modes).
    static func betaMode(alpha: MLXArray, beta: MLXArray) -> MLXArray {
        let a = maximum(alpha - 1, Float(0)), b = maximum(beta - 1, Float(0))
        return which((alpha .< Float(1)) .&& (beta .< Float(1)), MLXArray(Float(1)), a / (a + b))
    }

    /// `F.normalize` along the last axis: `x / max(‖x‖, epsilon)`.
    static func normalized(_ x: MLXArray, epsilon: Float) -> MLXArray {
        x / maximum(sqrt((x * x).sum(axis: -1, keepDims: true)), epsilon)
    }

    /// The 1-D operator `[outN, inN]` of `F.interpolate(mode: "bilinear", align_corners: false)` without
    /// antialiasing: half-pixel centers, the source clamped at zero, two taps.
    static func bilinearMatrix(from inN: Int, to outN: Int) -> MLXArray {
        let scale = Float(inN) / Float(outN)
        var rows = [Float](repeating: 0, count: outN * inN)
        for i in 0 ..< outN {
            let source = max(scale * (Float(i) + 0.5) - 0.5, 0)
            let low = Int(source)
            let high = min(low + 1, inN - 1)
            let lambda = source - Float(low)
            rows[i * inN + low] += 1 - lambda
            rows[i * inN + high] += lambda
        }
        return MLXArray(rows).reshaped([outN, inN])
    }

    /// An RGB8 image `[height, width]` as a PNG or JPEG file.
    static func encodedImage(_ rgb: [UInt8], width: Int, height: Int, type: String) throws -> NFKMeshTexture {
        var rgba = [UInt8](repeating: 255, count: width * height * 4)
        for pixel in 0 ..< (width * height) {
            rgba[pixel * 4] = rgb[pixel * 3]
            rgba[pixel * 4 + 1] = rgb[pixel * 3 + 1]
            rgba[pixel * 4 + 2] = rgb[pixel * 3 + 2]
        }
        let options = NFKMLXImageOptions()
        let image = try NFKMLXImageBridge.cgImage(rgba: rgba, width: width, height: height, options: options)
        let data = NSMutableData()
        let uniformType = (type == NFKMeshTextureMIMETypeJPEG ? UTType.jpeg : UTType.png).identifier as CFString
        guard let destination = CGImageDestinationCreateWithData(data, uniformType, 1, nil) else {
            throw NFKMLXError.unsupportedConfiguration("ImageIO writes no \(type)")
        }
        let properties = type == NFKMeshTextureMIMETypeJPEG ? [kCGImageDestinationLossyCompressionQuality: 0.95] as CFDictionary : nil
        CGImageDestinationAddImage(destination, image, properties)
        guard CGImageDestinationFinalize(destination) else {
            throw NFKMLXError.unsupportedConfiguration("ImageIO could not write the \(type) texture")
        }
        return try NFKMeshTexture(data: data as Data, mimeType: type)
    }
}

/// A completion handler carried into the detached task that downloads and builds.
private final class NFKSF3DCompletion: @unchecked Sendable {
    let handler: ((any NFKInferenceBackend)?, Error?) -> Void
    init(_ handler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) { self.handler = handler }
    func call(_ backend: (any NFKInferenceBackend)?, _ error: Error?) { handler(backend, error) }
}
