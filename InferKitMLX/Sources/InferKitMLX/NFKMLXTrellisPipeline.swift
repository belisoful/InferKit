//
//  NFKMLXTrellisPipeline.swift
//  InferKitMLX
//
//  TRELLIS end to end: the release's preprocessing, the rectified-flow sampler with guidance on an
//  interval, the two generation stages, and the decoders to a mesh and Gaussians, as
//  `TrellisImageTo3DPipeline` and `TrellisTextTo3DPipeline` run them.
//

import CoreGraphics
import Foundation
import InferKit
import MLX
import MLXNN

/// The sampler of one stage: Euler steps of the flow from noise at t = 1 to t = 0 over a timetable bent
/// toward t = 1 by `rescale`, with classifier-free guidance where t lies in the guidance interval.
/// Introduced in InferKit 0.4.0.
public struct NFKMLXTrellisSampling: Sendable, Equatable {
    public var steps = 25
    public var guidance: Float = 5
    public var guidanceInterval: ClosedRange<Double> = 0.5 ... 1
    public var rescale: Double = 3

    public init() {}

    /// The image release's setting for both stages.
    public static let image = NFKMLXTrellisSampling()

    /// The text releases' setting for both stages.
    public static let text: NFKMLXTrellisSampling = {
        var sampling = NFKMLXTrellisSampling()
        sampling.guidance = 7.5
        sampling.guidanceInterval = 0.5 ... 0.95
        return sampling
    }()

    /// The (t, next t) pairs: `steps + 1` evenly spaced points from 1 to 0, each mapped to
    /// rescale · t / (1 + (rescale − 1) · t), in double precision as numpy computes them.
    public var timetable: [(Double, Double)] {
        let points = (0 ... steps).map { index -> Double in
            let t = index == steps ? 0 : Double(index) * (-1 / Double(steps)) + 1
            return rescale * t / (1 + (rescale - 1) * t)
        }
        return (0 ..< steps).map { (points[$0], points[$0 + 1]) }
    }

    /// The sample after every step from `noise`, where `velocity(x, t, conditional)` evaluates the flow
    /// at `x` and t · 1000 with the condition (true) or the unconditional one (false).
    func sample(_ noise: MLXArray, velocity: (MLXArray, MLXArray, Bool) -> MLXArray) -> MLXArray {
        var x = noise
        for (t, next) in timetable {
            let time = MLXArray([Float(1000 * t)])
            var v = velocity(x, time, true)
            if guidanceInterval.contains(t) {
                v = (1 + guidance) * v - guidance * velocity(x, time, false)
            }
            x = x - Float(t - next) * v
            eval(x)
        }
        return x
    }
}

/// The release's image preparation: the alpha's tight box (alpha > 0.8 of full) widened by a fifth,
/// cropped (outside the image reads transparent black), resized to 518 × 518 with Pillow's Lanczos over
/// premultiplied RGBA, then the color multiplied by the alpha and truncated to 8 bits.
enum NFKTrellisImagePreparation {
    /// The prepared 8-bit RGB image `side × side × 3` (the release's 518) from straight RGBA8 pixels. An
    /// image with no alpha below 255 is used whole.
    static func prepare(rgba: [UInt8], width: Int, height: Int, side: Int = 518) -> [UInt8] {
        var minX = width, minY = height, maxX = -1, maxY = -1
        for y in 0 ..< height {
            for x in 0 ..< width where Float(rgba[(y * width + x) * 4 + 3]) > 0.8 * 255 {
                minX = min(minX, x)
                maxX = max(maxX, x)
                minY = min(minY, y)
                maxY = max(maxY, y)
            }
        }
        if maxX < 0 {
            (minX, minY, maxX, maxY) = (0, 0, width - 1, height - 1)
        }
        let centerX = Double(minX + maxX) / 2, centerY = Double(minY + maxY) / 2
        let size = Int(Double(max(maxX - minX, maxY - minY)) * 1.2)
        let half = Double(size / 2)
        // PIL's crop rounds a fractional box half to even.
        let box = [centerX - half, centerY - half, centerX + half, centerY + half].map { Int($0.rounded(.toNearestOrEven)) }
        let cropWidth = box[2] - box[0], cropHeight = box[3] - box[1]
        var crop = [UInt8](repeating: 0, count: cropWidth * cropHeight * 4)
        for y in 0 ..< cropHeight {
            let sourceY = y + box[1]
            guard sourceY >= 0 && sourceY < height else { continue }
            for x in 0 ..< cropWidth {
                let sourceX = x + box[0]
                guard sourceX >= 0 && sourceX < width else { continue }
                for c in 0 ..< 4 {
                    crop[(y * cropWidth + x) * 4 + c] = rgba[(sourceY * width + sourceX) * 4 + c]
                }
            }
        }
        let resized = NFKMLXPillowLanczos.resizeRGBA(crop, width: cropWidth, height: cropHeight, outWidth: side, outHeight: side)
        var prepared = [UInt8](repeating: 0, count: side * side * 3)
        for pixel in 0 ..< side * side {
            let alpha = Float(resized[pixel * 4 + 3]) / 255
            for c in 0 ..< 3 {
                prepared[pixel * 3 + c] = UInt8(Float(resized[pixel * 4 + c]) / 255 * alpha * 255)
            }
        }
        return prepared
    }
}

/// Flow checkpoints that replace the release's, such as a fine-tune saved with `NFKMLXWeights.save`.
/// Introduced in InferKit 0.4.0.
public struct NFKMLXTrellisFlowWeights: Sendable {
    public var structure: URL?
    public var latent: URL?
    public init(structure: URL? = nil, latent: URL? = nil) {
        self.structure = structure
        self.latent = latent
    }
}

/// What a TRELLIS run produces. Introduced in InferKit 0.4.0.
public struct NFKMLXTrellisOutputs: OptionSet, Sendable {
    public let rawValue: Int
    public init(rawValue: Int) { self.rawValue = rawValue }
    /// The FlexiCubes mesh with a color per vertex.
    public static let mesh = NFKMLXTrellisOutputs(rawValue: 1)
    /// The Gaussians.
    public static let gaussians = NFKMLXTrellisOutputs(rawValue: 2)
    public static let all: NFKMLXTrellisOutputs = [.mesh, .gaussians]
}

/// A generated object: the sparse structure, the structured latent, and the decoded representations.
/// Introduced in InferKit 0.4.0.
public struct NFKMLXTrellisResult {
    /// The occupied voxels `[N, 3]` of the 64³ grid.
    public let voxels: MLXArray
    /// The denormalized structured latent `[N, latentChannels]`.
    public let latent: MLXArray
    public let mesh: NFKMesh?
    public let gaussians: NFKGaussianSplats?
}

/// One network of a staged pipeline: built from its checkpoint on first use, then kept while the pipeline
/// holds its stages resident, or released when its stage ends.
final class NFKTrellisStage<Network: Module>: @unchecked Sendable {
    private let build: () throws -> Network
    private var network: Network?
    private let lock = NSLock()
    let keepsResident: Bool

    init(keepsResident: Bool, build: @escaping () throws -> Network) {
        self.keepsResident = keepsResident
        self.build = build
    }

    /// Runs `body` on the network, building it first when it is not held.
    func use<T>(_ body: (Network) throws -> T) throws -> T {
        let built: Network = try lock.withLock {
            if let network {
                return network
            }
            let fresh = try build()
            if keepsResident {
                network = fresh
            }
            return fresh
        }
        let result = try body(built)
        if !keepsResident {
            NFKMLXGPU.clearCache()
        }
        return result
    }
}

/// A TRELLIS pipeline: the conditioner (DINOv2 for the image release, CLIP's text tower for the text
/// ones), the two flows, and the three decoders. Build one with `image(directoryURL:encoderWeightsURL:)` or
/// `text(directoryURL:imageReleaseDirectoryURL:textEncoderDirectoryURL:)`. Staged, a pipeline loads each
/// network for its stage and releases it after; resident, it keeps them all. Run it off the render
/// thread. Introduced in InferKit 0.4.0.
public final class NFKMLXTrellisPipeline: @unchecked Sendable {
    public let configuration: NFKMLXTrellisConfiguration
    public let structureSampling: NFKMLXTrellisSampling
    public let latentSampling: NFKMLXTrellisSampling
    /// Whether the networks stay loaded between stages.
    public let holdsStagesResident: Bool
    let imageEncoder: NFKTrellisStage<NFKMLXTrellisImageEncoder>?
    let textEncoder: NFKTrellisStage<NFKMLXSDTextEncoderNet>?
    let tokenizer: NFKMLXSDPromptTokenizer?
    let structureFlow: NFKTrellisStage<NFKMLXTrellisStructureFlow>
    let structureDecoder: NFKTrellisStage<NFKMLXTrellisStructureDecoder>
    let latentFlow: NFKTrellisStage<NFKMLXTrellisLatentFlow>
    let meshDecoder: NFKTrellisStage<NFKMLXTrellisMeshDecoder>
    let gaussianDecoder: NFKTrellisStage<NFKMLXTrellisGaussianDecoder>
    let latentMean: MLXArray
    let latentDeviation: MLXArray

    /// An image pipeline of random networks at `configuration`, which holds its stages resident: a test's
    /// or an example's run of the whole path without a release.
    public static func random(configuration: NFKMLXTrellisConfiguration) -> NFKMLXTrellisPipeline {
        NFKMLXTrellisPipeline(randomAt: configuration)
    }

    private init(randomAt configuration: NFKMLXTrellisConfiguration) {
        self.configuration = configuration
        structureSampling = .image
        latentSampling = .image
        holdsStagesResident = true
        imageEncoder = NFKTrellisStage(keepsResident: true) { try NFKMLXTrellis.imageEncoder(weightsURL: nil, configuration: configuration) }
        textEncoder = nil
        tokenizer = nil
        structureFlow = NFKTrellisStage(keepsResident: true) { try NFKMLXTrellis.structureFlow(weightsURL: nil, configuration: configuration) }
        structureDecoder = NFKTrellisStage(keepsResident: true) {
            try NFKMLXTrellis.structureDecoder(weightsURL: nil, configuration: configuration)
        }
        latentFlow = NFKTrellisStage(keepsResident: true) { try NFKMLXTrellis.latentFlow(weightsURL: nil, configuration: configuration) }
        meshDecoder = NFKTrellisStage(keepsResident: true) { try NFKMLXTrellis.meshDecoder(weightsURL: nil, configuration: configuration) }
        gaussianDecoder = NFKTrellisStage(keepsResident: true) {
            try NFKMLXTrellis.gaussianDecoder(weightsURL: nil, configuration: configuration)
        }
        latentMean = MLXArray.zeros([1, configuration.latentChannels])
        latentDeviation = MLXArray.ones([1, configuration.latentChannels])
    }

    init(release: NFKMLXTrellisRelease, encoderWeightsURL: URL?, textEncoderDirectoryURL: URL?, quantizationBits: Int?,
         residency: NFKMLXResidency, overrides: NFKMLXTrellisFlowWeights = NFKMLXTrellisFlowWeights()) throws {
        let configuration = release.configuration
        self.configuration = configuration
        structureSampling = release.structureSampling
        latentSampling = release.latentSampling
        let resident = try NFKMLXResidencyBudget.holdsResident(release.residentBytes(textEncoderDirectoryURL: textEncoderDirectoryURL,
                                                                                     encoderWeightsURL: encoderWeightsURL),
                                                               residency: residency, budget: NFKMLXResidencyBudget.current())
        holdsStagesResident = resident
        func quantized<Network: Module>(_ network: Network) throws -> Network {
            if let bits = quantizationBits {
                try NFKMLXQuantization.quantize(module: network, bits: bits, groupSize: NFKMLXLoadOptions.defaultQuantizationGroupSize)
            }
            return network
        }
        imageEncoder = encoderWeightsURL.map { url in
            NFKTrellisStage(keepsResident: resident) { try NFKMLXTrellis.imageEncoder(weightsURL: url, configuration: configuration) }
        }
        textEncoder = textEncoderDirectoryURL.map { directory in
            NFKTrellisStage(keepsResident: resident) {
                // The release is the whole CLIPModel, which carries the text projection; the pipeline reads the
                // last hidden states, as CLIPTextModel returns them.
                var tower = NFKMLXSDTextEncoderConfiguration.stableDiffusion15
                tower.projectionDimensions = 768
                return try NFKMLXSDTextEncoder.net(configuration: tower,
                                                   weightsURL: directory.appendingPathComponent("model.safetensors"), precision: .float32)
            }
        }
        tokenizer = try textEncoderDirectoryURL.map { try NFKMLXSDPromptTokenizer(directoryURL: $0) }
        let structureFlowURL = try overrides.structure ?? release.checkpoint("sparse_structure_flow_model")
        let structureDecoderURL = try release.checkpoint("sparse_structure_decoder")
        let latentFlowURL = try overrides.latent ?? release.checkpoint("slat_flow_model")
        let meshDecoderURL = try release.checkpoint("slat_decoder_mesh")
        let gaussianDecoderURL = try release.checkpoint("slat_decoder_gs")
        structureFlow = NFKTrellisStage(keepsResident: resident) {
            try quantized(NFKMLXTrellis.structureFlow(weightsURL: structureFlowURL, configuration: configuration))
        }
        structureDecoder = NFKTrellisStage(keepsResident: resident) {
            try NFKMLXTrellis.structureDecoder(weightsURL: structureDecoderURL, configuration: configuration)
        }
        latentFlow = NFKTrellisStage(keepsResident: resident) {
            try quantized(NFKMLXTrellis.latentFlow(weightsURL: latentFlowURL, configuration: configuration))
        }
        meshDecoder = NFKTrellisStage(keepsResident: resident) {
            try NFKMLXTrellis.meshDecoder(weightsURL: meshDecoderURL, configuration: configuration)
        }
        gaussianDecoder = NFKTrellisStage(keepsResident: resident) {
            try NFKMLXTrellis.gaussianDecoder(weightsURL: gaussianDecoderURL, configuration: configuration)
        }
        latentMean = MLXArray(release.latentMean).reshaped([1, -1])
        latentDeviation = MLXArray(release.latentDeviation).reshaped([1, -1])
    }

    /// The image release (`microsoft/TRELLIS-image-large`) from its directory and DINOv2 ViT-L/14 with
    /// registers (`dinov2_vitl14_reg4_pretrain.pth`, which the release loads through `torch.hub`).
    /// `quantizationBits` packs the two flows' linear layers; `residency` holds the stages resident or
    /// loads each for its stage (`.automatic` holds them where the memory budget allows).
    public static func image(directoryURL: URL, encoderWeightsURL: URL, quantizationBits: Int? = nil,
                             residency: NFKMLXResidency = .automatic,
                             flows: NFKMLXTrellisFlowWeights = NFKMLXTrellisFlowWeights()) throws -> NFKMLXTrellisPipeline {
        let release = try NFKMLXTrellisRelease(directoryURL: directoryURL, imageReleaseDirectoryURL: directoryURL)
        guard release.isImageConditioned else {
            throw NFKMLXError.unsupportedConfiguration("\(directoryURL.lastPathComponent) is a text release; build it with text(…)")
        }
        return try NFKMLXTrellisPipeline(release: release, encoderWeightsURL: encoderWeightsURL, textEncoderDirectoryURL: nil,
                                         quantizationBits: quantizationBits, residency: residency, overrides: flows)
    }

    /// A text release (`microsoft/TRELLIS-text-base`, `-large`, or `-xlarge`) from its directory, the image
    /// release's directory (which holds the decoders the text releases name), and CLIP ViT-L/14
    /// (`openai/clip-vit-large-patch14`: `model.safetensors`, `vocab.json`, `merges.txt`).
    public static func text(directoryURL: URL, imageReleaseDirectoryURL: URL, textEncoderDirectoryURL: URL,
                            quantizationBits: Int? = nil, residency: NFKMLXResidency = .automatic,
                            flows: NFKMLXTrellisFlowWeights = NFKMLXTrellisFlowWeights()) throws -> NFKMLXTrellisPipeline {
        let release = try NFKMLXTrellisRelease(directoryURL: directoryURL, imageReleaseDirectoryURL: imageReleaseDirectoryURL)
        guard !release.isImageConditioned else {
            throw NFKMLXError.unsupportedConfiguration("\(directoryURL.lastPathComponent) is an image release; build it with image(…)")
        }
        return try NFKMLXTrellisPipeline(release: release, encoderWeightsURL: nil, textEncoderDirectoryURL: textEncoderDirectoryURL,
                                         quantizationBits: quantizationBits, residency: residency, overrides: flows)
    }

    /// Whether the pipeline reads an image (else a prompt).
    public var isImageConditioned: Bool { imageEncoder != nil }

    /// The prepared RGB image the encoder reads (`imageSize` square, the release's 518), from straight RGBA8
    /// pixels.
    public func preparedImage(rgba: [UInt8], width: Int, height: Int) -> [UInt8] {
        NFKTrellisImagePreparation.prepare(rgba: rgba, width: width, height: height, side: configuration.imageSize)
    }

    /// The condition and the unconditional condition for prepared pixels (`imageSize² × 3`): DINOv2's
    /// tokens and zeros.
    public func condition(prepared: [UInt8]) throws -> (condition: MLXArray, unconditional: MLXArray) {
        guard let imageEncoder else {
            throw NFKMLXError.unsupportedConfiguration("a text release reads a prompt, not an image")
        }
        let side = configuration.imageSize
        let image = MLXArray(prepared.map { Float($0) / 255 }).reshaped([1, side, side, 3])
        let condition = try imageEncoder.use { encoder -> MLXArray in
            let tokens = encoder(image)
            eval(tokens)
            return tokens
        }
        return (condition, MLXArray.zeros(like: condition))
    }

    /// The condition and the unconditional condition for a prompt: CLIP's last hidden states for the
    /// prompt and for the empty prompt, each padded to 77 tokens.
    public func condition(prompt: String) throws -> (condition: MLXArray, unconditional: MLXArray) {
        guard let textEncoder, let tokenizer else {
            throw NFKMLXError.unsupportedConfiguration("the image release reads an image, not a prompt")
        }
        return try textEncoder.use { net in
            let length = net.configuration.contextLength
            let condition = net.encode(tokenizer.tokens(for: prompt, contextLength: length)).hidden
            let unconditional = net.encode(tokenizer.tokens(for: "", contextLength: length)).hidden
            eval(condition, unconditional)
            return (condition, unconditional)
        }
    }

    /// The structure latent sampled from `noise` `[1, R, R, R, structureChannels]`.
    public func sampleStructure(noise: MLXArray, condition: MLXArray, unconditional: MLXArray,
                                sampling: NFKMLXTrellisSampling? = nil) throws -> MLXArray {
        try structureFlow.use { flow in
            (sampling ?? structureSampling).sample(noise) { x, t, conditional in
                flow(x, t: t, condition: conditional ? condition : unconditional)
            }
        }
    }

    /// The occupancy logits `[1, 4R, 4R, 4R, 1]` of a structure latent.
    public func occupancy(_ structure: MLXArray) throws -> MLXArray {
        try structureDecoder.use { decoder in
            let logits = decoder(structure)
            eval(logits)
            return logits
        }
    }

    /// The normalized structured latent sampled from `noise` `[N, latentChannels]` on `voxels` `[N, 3]`.
    public func sampleLatent(noise: MLXArray, voxels: MLXArray, condition: MLXArray, unconditional: MLXArray,
                             sampling: NFKMLXTrellisSampling? = nil) throws -> MLXArray {
        let grid = NFKTrellisVoxelGrid(voxels, resolution: configuration.occupancyResolution)
        return try latentFlow.use { flow in
            (sampling ?? latentSampling).sample(noise) { x, t, conditional in
                flow.velocity(x, grid: grid, t: t, condition: conditional ? condition : unconditional)
            }
        }
    }

    /// The structured latent denormalized with the release's per-channel mean and deviation.
    public func denormalized(_ latent: MLXArray) -> MLXArray {
        latent * latentDeviation + latentMean
    }

    /// The run after the condition: noise from `seed`, both stages, and the decoders `outputs` names.
    public func generate(condition: MLXArray, unconditional: MLXArray, seed: UInt64, outputs: NFKMLXTrellisOutputs = .all,
                         structureSampling: NFKMLXTrellisSampling? = nil, latentSampling: NFKMLXTrellisSampling? = nil) throws
        -> NFKMLXTrellisResult {
        let keys = MLXRandom.split(key: MLXRandom.key(seed), into: 2)
        let side = configuration.structureResolution
        let structureNoise = MLXRandom.normal([1, side, side, side, configuration.structureChannels], key: keys[0])
        let structure = try sampleStructure(noise: structureNoise, condition: condition, unconditional: unconditional,
                                            sampling: structureSampling)
        let voxels = NFKMLXTrellisStructureDecoder.occupiedVoxels(try occupancy(structure)[0])
        guard voxels.dim(0) > 0 else {
            throw NFKMLXError.unsupportedConfiguration("the structure stage left no voxel occupied")
        }
        let latentNoise = MLXRandom.normal([voxels.dim(0), configuration.latentChannels], key: keys[1])
        let latent = denormalized(try sampleLatent(noise: latentNoise, voxels: voxels, condition: condition,
                                                   unconditional: unconditional, sampling: latentSampling))
        eval(latent)
        return try decode(latent: latent, voxels: voxels, outputs: outputs)
    }

    /// The mesh and the Gaussians of a denormalized structured latent on `voxels`.
    public func decode(latent: MLXArray, voxels: MLXArray, outputs: NFKMLXTrellisOutputs = .all) throws -> NFKMLXTrellisResult {
        let grid = NFKTrellisVoxelGrid(voxels, resolution: configuration.occupancyResolution)
        var gaussians: NFKGaussianSplats?
        var mesh: NFKMesh?
        if outputs.contains(.gaussians) {
            gaussians = try gaussianDecoder.use { decoder in
                try NFKTrellisExport.splats(decoder.gaussians(decoder.features(latent, grid: grid), grid: grid))
            }
        }
        if outputs.contains(.mesh) {
            let surface = try meshDecoder.use { decoder -> NFKTrellisSurface in
                let (features, fine) = decoder.features(latent, grid: grid)
                return NFKTrellisFlexiCubes.surface(coordinates: fine.coordinates, features: features.asArray(Float.self),
                                                   resolution: fine.resolution)
            }
            mesh = try NFKTrellisExport.mesh(surface)
        }
        return NFKMLXTrellisResult(voxels: voxels, latent: latent, mesh: mesh, gaussians: gaussians)
    }
}

/// The decoders' frame (z up) into glTF's (y up, the object facing +z): (x, y, z) → (x, z, −y), a
/// quarter turn about x, as the release's GLB export turns its mesh.
enum NFKTrellisExport {
    static func mesh(_ surface: NFKTrellisSurface) throws -> NFKMesh {
        var positions = [Float](repeating: 0, count: surface.positions.count)
        var colors = [Float](repeating: 0, count: surface.vertexCount * 3)
        for vertex in 0 ..< surface.vertexCount {
            positions[vertex * 3] = surface.positions[vertex * 3]
            positions[vertex * 3 + 1] = surface.positions[vertex * 3 + 2]
            positions[vertex * 3 + 2] = -surface.positions[vertex * 3 + 1]
            for c in 0 ..< 3 {
                colors[vertex * 3 + c] = surface.colors[vertex * 6 + c]
            }
        }
        let data = { (values: [Float]) in values.withUnsafeBufferPointer { Data(buffer: $0) } }
        return try NFKMesh(positions: data(positions), normals: nil, colors: data(colors), textureCoordinates: nil,
                           indices: surface.triangles.withUnsafeBufferPointer { Data(buffer: $0) })
    }

    static func splats(_ gaussians: NFKTrellisGaussians) throws -> NFKGaussianSplats {
        let raw = gaussians.positions.asArray(Float.self)
        let rawRotations = gaussians.rotations.asArray(Float.self)
        let count = raw.count / 3
        var positions = [Float](repeating: 0, count: raw.count)
        var rotations = [Float](repeating: 0, count: rawRotations.count)
        let half = Float(0.5).squareRoot()
        for index in 0 ..< count {
            positions[index * 3] = raw[index * 3]
            positions[index * 3 + 1] = raw[index * 3 + 2]
            positions[index * 3 + 2] = -raw[index * 3 + 1]
            // (√½, −√½, 0, 0) ⊗ q: the same quarter turn applied to the orientation.
            let (w, x, y, z) = (rawRotations[index * 4], rawRotations[index * 4 + 1], rawRotations[index * 4 + 2], rawRotations[index * 4 + 3])
            rotations[index * 4] = half * (w + x)
            rotations[index * 4 + 1] = half * (x - w)
            rotations[index * 4 + 2] = half * (y + z)
            rotations[index * 4 + 3] = half * (z - y)
        }
        let data = { (values: [Float]) in values.withUnsafeBufferPointer { Data(buffer: $0) } }
        return try NFKGaussianSplats(positions: data(positions), scales: data(gaussians.scales.asArray(Float.self)),
                                     rotations: data(rotations), opacities: data(gaussians.opacities.asArray(Float.self)),
                                     colors: data(gaussians.colors.asArray(Float.self)))
    }
}

/// A release directory's `pipeline.json` and checkpoint configurations, checked against what the port
/// reads.
struct NFKMLXTrellisRelease {
    let configuration: NFKMLXTrellisConfiguration
    let structureSampling: NFKMLXTrellisSampling
    let latentSampling: NFKMLXTrellisSampling
    let latentMean: [Float]
    let latentDeviation: [Float]
    let isImageConditioned: Bool
    private let checkpoints: [String: URL]

    static let imageReleasePrefix = "JeffreyXiang/TRELLIS-image-large/"

    init(directoryURL: URL, imageReleaseDirectoryURL: URL) throws {
        let pipeline = try Self.json(directoryURL.appendingPathComponent("pipeline.json"))
        guard let args = pipeline["args"] as? [String: Any], let models = args["models"] as? [String: String],
              let normalization = args["slat_normalization"] as? [String: [Double]],
              let mean = normalization["mean"], let deviation = normalization["std"] else {
            throw NFKMLXError.weightsMismatch("pipeline.json names no models or structured-latent normalization")
        }
        var checkpoints = [String: URL]()
        for (name, path) in models {
            checkpoints[name] = path.hasPrefix(Self.imageReleasePrefix)
                ? imageReleaseDirectoryURL.appendingPathComponent(String(path.dropFirst(Self.imageReleasePrefix.count)))
                : directoryURL.appendingPathComponent(path)
        }
        self.checkpoints = checkpoints
        latentMean = mean.map(Float.init)
        latentDeviation = deviation.map(Float.init)
        isImageConditioned = args["image_cond_model"] != nil
        if isImageConditioned {
            guard args["image_cond_model"] as? String == "dinov2_vitl14_reg" else {
                throw NFKMLXError.unsupportedConfiguration("the image conditioner is \(args["image_cond_model"] ?? "unnamed"), not dinov2_vitl14_reg")
            }
        } else {
            guard args["text_cond_model"] as? String == "openai/clip-vit-large-patch14" else {
                throw NFKMLXError.unsupportedConfiguration("the text conditioner is \(args["text_cond_model"] ?? "unnamed"), not CLIP ViT-L/14")
            }
        }
        structureSampling = try Self.sampling(args["sparse_structure_sampler"])
        latentSampling = try Self.sampling(args["slat_sampler"])

        var configuration = NFKMLXTrellisConfiguration()
        let structure = try Self.arguments(checkpoints, "sparse_structure_flow_model", name: "SparseStructureFlowModel")
        let latent = try Self.arguments(checkpoints, "slat_flow_model", name: "SLatFlowModel")
        for (args, patch) in [(structure, 1), (latent, 2)] {
            guard args["pe_mode"] as? String ?? "ape" == "ape", args["patch_size"] as? Int == patch,
                  !(args["share_mod"] as? Bool ?? false), args["qk_rms_norm"] as? Bool == true,
                  args["in_channels"] as? Int == 8, args["out_channels"] as? Int == 8, args["mlp_ratio"] as? Int ?? 4 == 4 else {
                throw NFKMLXError.unsupportedConfiguration("a flow sets a positional mode, patch, shared modulation, or width the port does not read")
            }
        }
        guard let width = structure["model_channels"] as? Int, latent["model_channels"] as? Int == width,
              let blocks = structure["num_blocks"] as? Int, latent["num_blocks"] as? Int == blocks,
              let condition = structure["cond_channels"] as? Int, latent["cond_channels"] as? Int == condition,
              let io = latent["io_block_channels"] as? [Int], io.count == 1, let ioBlocks = latent["num_io_res_blocks"] as? Int,
              structure["resolution"] as? Int == 16, latent["resolution"] as? Int == 64 else {
            throw NFKMLXError.unsupportedConfiguration("the two flows differ in width, depth, or condition, which the port holds in common, or a flow names no residual-block geometry")
        }
        let heads = structure["num_heads"] as? Int ?? width / 64
        configuration.flowWidth = width
        configuration.flowBlocks = blocks
        configuration.flowHeads = heads
        configuration.flowHidden = width * 4
        configuration.conditionWidth = condition
        configuration.crossAttentionHeadNorm = structure["qk_rms_norm_cross"] as? Bool ?? false
        configuration.latentCrossAttentionHeadNorm = latent["qk_rms_norm_cross"] as? Bool ?? false
        configuration.latentFlowIOWidth = io[0]
        configuration.latentFlowIOBlocks = ioBlocks
        let decoder = try Self.arguments(checkpoints, "slat_decoder_mesh", name: "SLatMeshDecoder")
        guard decoder["model_channels"] as? Int == configuration.decoderWidth, decoder["num_blocks"] as? Int == configuration.decoderBlocks,
              decoder["attn_mode"] as? String == "swin", decoder["window_size"] as? Int == configuration.decoderWindow,
              (decoder["representation_config"] as? [String: Any])?["use_color"] as? Bool == true else {
            throw NFKMLXError.unsupportedConfiguration("the mesh decoder is not the 768-wide swin decoder with colors the port reads")
        }
        self.configuration = configuration
    }

    /// The bytes the networks hold at float32: twice the half-precision checkpoints, plus the conditioner.
    func residentBytes(textEncoderDirectoryURL: URL?, encoderWeightsURL: URL?) -> Int {
        func size(_ url: URL?) -> Int {
            guard let url, let attributes = try? FileManager.default.attributesOfItem(atPath: url.path) else { return 0 }
            return (attributes[.size] as? NSNumber)?.intValue ?? 0
        }
        let networks = ["sparse_structure_flow_model", "sparse_structure_decoder", "slat_flow_model", "slat_decoder_mesh", "slat_decoder_gs"]
        return networks.reduce(0) { $0 + 2 * size(try? checkpoint($1)) } + size(encoderWeightsURL)
            + size(textEncoderDirectoryURL?.appendingPathComponent("model.safetensors")) / 4
    }

    func checkpoint(_ name: String) throws -> URL {
        guard let base = checkpoints[name] else {
            throw NFKMLXError.weightsMismatch("pipeline.json names no \(name)")
        }
        return base.appendingPathExtension("safetensors")
    }

    static func json(_ url: URL) throws -> [String: Any] {
        guard let data = try? Data(contentsOf: url), let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw NFKMLXError.weightsMismatch("\(url.lastPathComponent) is missing or not a JSON object")
        }
        return object
    }

    static func arguments(_ checkpoints: [String: URL], _ model: String, name: String) throws -> [String: Any] {
        guard let base = checkpoints[model] else {
            throw NFKMLXError.weightsMismatch("pipeline.json names no \(model)")
        }
        let config = try json(base.appendingPathExtension("json"))
        guard config["name"] as? String == name, let args = config["args"] as? [String: Any] else {
            throw NFKMLXError.unsupportedConfiguration("\(model) is \(config["name"] ?? "unnamed"), not \(name)")
        }
        return args
    }

    static func sampling(_ value: Any?) throws -> NFKMLXTrellisSampling {
        guard let sampler = value as? [String: Any], sampler["name"] as? String == "FlowEulerGuidanceIntervalSampler",
              let params = sampler["params"] as? [String: Any], let interval = params["cfg_interval"] as? [Double], interval.count == 2 else {
            throw NFKMLXError.unsupportedConfiguration("a sampler is not FlowEulerGuidanceIntervalSampler with a guidance interval")
        }
        if let sigma = (sampler["args"] as? [String: Any])?["sigma_min"] as? Double, sigma != 1e-5 {
            throw NFKMLXError.unsupportedConfiguration("the sampler's sigma_min is \(sigma), not 1e-5")
        }
        var sampling = NFKMLXTrellisSampling()
        sampling.steps = params["steps"] as? Int ?? 25
        sampling.guidance = Float(params["cfg_strength"] as? Double ?? 5)
        sampling.guidanceInterval = interval[0] ... interval[1]
        sampling.rescale = params["rescale_t"] as? Double ?? 1
        return sampling
    }
}
