//
//  NFKMLXSAM2Tracker.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXNN

// SAM 2's tracker (`SAM2Base`): the piece that turns the five ported networks into a video model, and
// the last part of a released checkpoint that had nothing reading it. The image encoder, prompt
// encoder, mask decoder, memory encoder, and memory attention are in `NFKMLXSAM2.swift`; what lives
// here is the orchestration between them and the dozen parameters it owns — the embedding that stands
// in for an empty memory, the temporal encodings that tell one remembered frame from another, the
// projection that turns a decoded mask token into the object pointer carried forward, and the
// fallbacks used when the object leaves the frame.
//
// A frame is tracked in three steps. The encoder produces the feature pyramid. The top level is
// conditioned: on the first frame by adding `no_mem_embed`, and afterwards by attending over the
// memories of earlier frames and their object pointers. The decoder then predicts the mask, and the
// memory encoder folds the frame's features and that mask into the next memory.
//
// SAM 2.1 changes no geometry. It adds two flags, each of which brings one parameter:
// `no_obj_embed_spatial` is added to a memory whose frame the model believes holds no object, and
// `obj_ptr_tpos_proj` projects a sine encoding of how far back a pointer came from. A 2.1 checkpoint
// loaded into a 2.0 configuration leaves both unread, which is why the release is part of the
// configuration rather than inferred.

/// `PositionEmbeddingSine` with `normalize: true`: the row and column index over the grid scaled to
/// `2π`, through the usual sine and cosine ladder, with the row half of the channels leading.
///
/// The reference names this encoding by its TOTAL width and halves it per axis, which is why a neck
/// configured at 256 and a memory encoder configured at 64 both emit exactly that many channels.
enum NFKMLXSAM2PositionEmbedding {
    static func sine(height: Int, width: Int, features: Int, temperature: Float = 10000) -> MLXArray {
        let half = features / 2
        let scale = 2 * Float.pi
        let epsilon: Float = 1e-6
        let rows = (1 ... height).map { (Float($0) / (Float(height) + epsilon)) * scale }
        let columns = (1 ... width).map { (Float($0) / (Float(width) + epsilon)) * scale }
        let dimensions = (0 ..< half).map { powf(temperature, 2 * Float($0 / 2) / Float(half)) }

        var values = [Float](repeating: 0, count: height * width * half * 2)
        for row in 0 ..< height {
            for column in 0 ..< width {
                let base = (row * width + column) * half * 2
                for index in 0 ..< half {
                    let pair = index / 2
                    let y = rows[row] / dimensions[index]
                    let x = columns[column] / dimensions[index]
                    if index % 2 == 0 {
                        values[base + pair * 2] = sinf(y)
                        values[base + half + pair * 2] = sinf(x)
                    } else {
                        values[base + pair * 2 + 1] = cosf(y)
                        values[base + half + pair * 2 + 1] = cosf(x)
                    }
                }
            }
        }
        return MLXArray(values, [1, height, width, half * 2])
    }

    /// The reference's `get_1d_sine_pe`: a sine ladder over one position per row, sines leading, with
    /// the frequency held for two channels at a time as the 2-D grid encoding holds it.
    static func sine(positions: [Float], features: Int, temperature: Float = 10000) -> MLXArray {
        let half = features / 2
        var values = [Float](repeating: 0, count: positions.count * features)
        for (row, position) in positions.enumerated() {
            for index in 0 ..< half {
                let angle = position / powf(temperature, 2 * Float(index / 2) / Float(half))
                values[row * features + index] = sinf(angle)
                values[row * features + half + index] = cosf(angle)
            }
        }
        return MLXArray(values, [positions.count, features])
    }
}

/// The released SAM 2 image-encoder size.
@objc(NFKMLXSAM2Variant)
public enum NFKMLXSAM2Variant: Int {
    case tiny
    case small
    case basePlus
    case large
}

/// Which SAM 2 release a checkpoint comes from. The geometry is the same; 2.1 adds the occlusion
/// embedding and the object pointers' temporal encoding, and a checkpoint carries exactly the
/// parameters its own release declares.
@objc(NFKMLXSAM2Release)
public enum NFKMLXSAM2Release: Int {
    case sam2
    case sam21
}

/// SAM 2 tracker sizing: the image encoder's geometry plus what the tracking loop needs.
public struct NFKMLXSAM2TrackerConfiguration: Sendable {
    public var encoder: NFKMLXSAM2Configuration
    public var imageSize: Int
    public var hiddenDimensions: Int
    public var memoryDimensions: Int
    /// Remembered frames the memory attention reads, the most recent first.
    public var memoryFrames: Int
    /// Object pointers the memory attention reads, counting the current frame.
    public var maximumObjectPointers: Int
    /// Scale and bias applied to the mask probabilities the memory encoder reads.
    public var sigmoidScale: Float
    public var sigmoidBias: Float
    /// 2.1: a memory whose frame holds no object takes a learned spatial embedding.
    public var occlusionSpatialEmbedding: Bool
    /// 2.1: an object pointer's temporal distance is encoded and projected rather than zeroed.
    public var temporalPositionEncodingForObjectPointers: Bool

    public init(encoder: NFKMLXSAM2Configuration = .tiny, imageSize: Int = 1024,
                hiddenDimensions: Int = 256, memoryDimensions: Int = 64, memoryFrames: Int = 7,
                maximumObjectPointers: Int = 16, sigmoidScale: Float = 20, sigmoidBias: Float = -10,
                occlusionSpatialEmbedding: Bool = false,
                temporalPositionEncodingForObjectPointers: Bool = false) {
        self.encoder = encoder
        self.imageSize = imageSize
        self.hiddenDimensions = hiddenDimensions
        self.memoryDimensions = memoryDimensions
        self.memoryFrames = memoryFrames
        self.maximumObjectPointers = maximumObjectPointers
        self.sigmoidScale = sigmoidScale
        self.sigmoidBias = sigmoidBias
        self.occlusionSpatialEmbedding = occlusionSpatialEmbedding
        self.temporalPositionEncodingForObjectPointers = temporalPositionEncodingForObjectPointers
    }

    /// The configuration a released size and release carry.
    public static func geometry(_ variant: NFKMLXSAM2Variant,
                                release: NFKMLXSAM2Release = .sam2) -> NFKMLXSAM2TrackerConfiguration {
        let encoder: NFKMLXSAM2Configuration
        switch variant {
        case .tiny: encoder = .tiny
        case .small: encoder = .small
        case .basePlus: encoder = .basePlus
        case .large: encoder = .large
        }
        return NFKMLXSAM2TrackerConfiguration(
            encoder: encoder,
            occlusionSpatialEmbedding: release == .sam21,
            temporalPositionEncodingForObjectPointers: release == .sam21)
    }

    /// The patch grid the top feature level lands on.
    var featureGrid: Int { imageSize / 16 }
    /// The resolution the decoder's own mask output comes back at.
    var maskGrid: Int { featureGrid * 4 }
}

/// One tracked frame's state: the memory the next frames attend over, and the pointer carried forward.
struct NFKMLXSAM2TrackedFrame {
    var memory: MLXArray                                                   // [1, tokens, memoryDim]
    var memoryPosition: MLXArray                                           // [1, tokens, memoryDim]
    var objectPointer: MLXArray                                            // [1, hidden]
    var isConditioning: Bool
}

/// The state one tracked object accumulates over a clip.
///
/// The reference keeps one of these per object and runs the whole model once per object, so tracking
/// several objects means several sessions over the same frames.
public final class NFKMLXSAM2TrackerSession {
    var frames = [Int: NFKMLXSAM2TrackedFrame]()

    /// The clip's length, when it is known ahead of time. It bounds how many object pointers a frame
    /// reads and, with it, the span their temporal encodings are normalized over, so a clip shorter
    /// than that bound does not encode distances it can never see. A session left at nil is a
    /// streaming one, which takes the configured bound.
    public var frameCount: Int?

    public init(frameCount: Int? = nil) { self.frameCount = frameCount }

    /// Frame indices the session holds, in order.
    public var trackedFrames: [Int] { frames.keys.sorted() }

    public func reset() { frames.removeAll() }
}

/// The SAM 2 tracker: the five networks under their released names, and the parameters the tracking
/// loop owns. One released checkpoint loads into this module and nothing is left over.
public final class NFKMLXSAM2TrackerNet: Module {
    @ModuleInfo(key: "image_encoder") var imageEncoder: NFKMLXSAM2EncoderNet
    @ModuleInfo(key: "sam_prompt_encoder") var promptEncoder: NFKSAMPromptEncoder
    @ModuleInfo(key: "sam_mask_decoder") var maskDecoder: NFKMLXSAM2Decoder
    @ModuleInfo(key: "memory_attention") var memoryAttention: NFKMLXSAM2MemoryAttentionNet
    @ModuleInfo(key: "memory_encoder") var memoryEncoder: NFKMLXSAM2MemoryEncoderNet
    @ModuleInfo(key: "obj_ptr_proj") var objectPointerProjection: NFKSAMMLP
    @ModuleInfo(key: "obj_ptr_tpos_proj") var objectPointerTemporalProjection: Linear?
    @ModuleInfo(key: "mask_downsample") var maskDownsample: Conv2d
    @ParameterInfo(key: "no_mem_embed") var noMemoryEmbedding: MLXArray
    @ParameterInfo(key: "no_mem_pos_enc") var noMemoryPositionEncoding: MLXArray
    @ParameterInfo(key: "maskmem_tpos_enc") var memoryTemporalEncoding: MLXArray
    @ParameterInfo(key: "no_obj_ptr") var noObjectPointer: MLXArray
    @ParameterInfo(key: "no_obj_embed_spatial") var occlusionSpatialEmbedding: MLXArray?

    public let configuration: NFKMLXSAM2TrackerConfiguration

    public init(_ configuration: NFKMLXSAM2TrackerConfiguration = NFKMLXSAM2TrackerConfiguration()) {
        self.configuration = configuration
        let hidden = configuration.hiddenDimensions, memory = configuration.memoryDimensions
        _imageEncoder.wrappedValue = NFKMLXSAM2EncoderNet(configuration.encoder)
        _promptEncoder.wrappedValue = NFKSAMPromptEncoder(NFKMLXSAMConfiguration.vitB)
        _maskDecoder.wrappedValue = NFKMLXSAM2Decoder()
        _memoryAttention.wrappedValue = NFKMLXSAM2MemoryAttentionNet()
        _memoryEncoder.wrappedValue = NFKMLXSAM2MemoryEncoderNet()
        _objectPointerProjection.wrappedValue = NFKSAMMLP(dim: hidden, hidden: hidden, out: hidden, layers: 3)
        _objectPointerTemporalProjection.wrappedValue =
            configuration.temporalPositionEncodingForObjectPointers ? Linear(hidden, memory) : nil
        _maskDownsample.wrappedValue = Conv2d(inputChannels: 1, outputChannels: 1, kernelSize: 4, stride: 4)
        _noMemoryEmbedding.wrappedValue = MLXArray.zeros([1, 1, hidden])
        _noMemoryPositionEncoding.wrappedValue = MLXArray.zeros([1, 1, hidden])
        _memoryTemporalEncoding.wrappedValue = MLXArray.zeros([configuration.memoryFrames, 1, 1, memory])
        _noObjectPointer.wrappedValue = MLXArray.zeros([1, hidden])
        _occlusionSpatialEmbedding.wrappedValue =
            configuration.occlusionSpatialEmbedding ? MLXArray.zeros([1, memory]) : nil
    }

    /// One frame's result.
    public struct Prediction {
        /// Mask logits at a quarter of the frame's resolution, `[1, maskGrid, maskGrid]`.
        public var maskLogits: MLXArray
        /// The same mask at the frame's resolution, `[1, imageSize, imageSize]`.
        public var highResolutionMaskLogits: MLXArray
        /// The decoder's estimate of the mask's quality.
        public var intersectionOverUnion: MLXArray
        /// Above zero where the model believes the object is present.
        public var objectScore: MLXArray
    }

    /// Tracks one frame and folds it into `session`.
    ///
    /// - Parameters:
    ///   - image: the frame `[1, imageSize, imageSize, 3]`, already normalized.
    ///   - frameIndex: where the frame sits in the clip; memories are gathered by distance from it.
    ///   - points: clicks in pixels of the model's input size with their labels, on a frame the caller
    ///     is conditioning. A frame tracked without clicks passes nil and reads the memories instead.
    ///   - session: the clip's memories, which the tracked frame joins.
    public func track(image: MLXArray, frameIndex: Int, points: [(x: Float, y: Float, label: Int)]? = nil,
                      session: NFKMLXSAM2TrackerSession) -> Prediction {
        let levels = imageEncoder.features(image)
        let grid = configuration.featureGrid
        let hidden = configuration.hiddenDimensions
        let visionFeatures = levels[levels.count - 1]                      // [1, grid, grid, hidden]
        let visionPosition = NFKMLXSAM2PositionEmbedding
            .sine(height: grid, width: grid, features: hidden)
            .reshaped([1, grid * grid, hidden])

        let conditioning = points != nil
        let conditioned = conditionedFeatures(visionFeatures, position: visionPosition,
                                              frameIndex: frameIndex, isConditioning: conditioning,
                                              session: session)

        let predicted = predict(conditioned: conditioned, levels: levels, points: points)

        // The tracker takes the multimask slot the decoder scores highest.
        let best = argMax(predicted.intersectionOverUnion, axis: -1).item(Int.self)
        let maskLogits = predicted.masks[0..., best]                       // [1, maskGrid, maskGrid]
        let highResolution = NFKMLXResample.resizeBilinear(
            maskLogits.expandedDimensions(axis: 3),
            height: configuration.imageSize, width: configuration.imageSize)

        let pointer = objectPointer(from: predicted.maskTokens[0..., 1 + best],
                                    present: predicted.objectScore .> 0)
        let memory = encodeMemory(features: visionFeatures, maskLogits: highResolution,
                                  objectScore: predicted.objectScore, fromPoints: conditioning)
        session.frames[frameIndex] = NFKMLXSAM2TrackedFrame(
            memory: memory.memory, memoryPosition: memory.position, objectPointer: pointer,
            isConditioning: conditioning)

        return Prediction(maskLogits: maskLogits,
                          highResolutionMaskLogits: highResolution.squeezed(axis: 3),
                          intersectionOverUnion: predicted.intersectionOverUnion,
                          objectScore: predicted.objectScore)
    }

    /// The prompt encoder and mask decoder over already-conditioned features, and the suppression the
    /// reference applies to a frame it reads as empty.
    ///
    /// `masks` and `intersectionOverUnion` carry the three MULTIMASK slots, not the single-mask one
    /// the first slot holds: those are what the tracker chooses between and what the training
    /// objective scores.
    private func predict(conditioned: MLXArray, levels: [MLXArray],
                         points: [(x: Float, y: Float, label: Int)]?)
        -> (masks: MLXArray, intersectionOverUnion: MLXArray, objectScore: MLXArray,
            maskTokens: MLXArray) {
        // The decoder reads the same prompt shape on every frame: a tracked frame contributes an
        // ignored point where a conditioned one contributes its clicks. A click arrives in pixels of
        // the model's own input; the reference shifts it to the pixel's center before normalizing.
        let size = Float(configuration.imageSize)
        let grid = configuration.featureGrid
        let prompt = (points ?? [(Float(0), Float(0), -1)])
            .map { (x: ($0.x + 0.5) / size, y: ($0.y + 0.5) / size, label: $0.label) }
        let sparse = promptEncoder.sparse(points: prompt)
        let dense = promptEncoder.dense(grid: grid)
        let positional = promptEncoder.positionEncoding.grid(grid, grid)
        let decoded = maskDecoder(features: conditioned, positional: positional, sparse: sparse,
                                  dense: dense, highResolution: Array(levels.dropLast()))

        // A frame the model reads as empty carries no mask at all, not a weak one.
        let present = decoded.objectScore .> 0                             // [1, 1]
        let masks = MLX.where(present.reshaped([1, 1, 1, 1]), decoded.masks, MLXArray(Float(-1024)))
        return (masks[0..., 1...], decoded.iou[0..., 1...], decoded.objectScore, decoded.maskTokens)
    }

    /// Segments one frame from a click, with no memory and nothing stored: the conditioning frame's
    /// path on its own.
    ///
    /// This is what a fine-tune scores. The masks come back at the frame's own resolution, all three
    /// multimask slots, because the training objective picks the best of them itself.
    ///
    /// - Returns: masks `[1, 3, imageSize, imageSize]`, the decoder's quality estimate `[1, 3]`, and
    ///   the object score `[1, 1]`.
    public func segment(image: MLXArray, points: [(x: Float, y: Float, label: Int)])
        -> (masks: MLXArray, intersectionOverUnion: MLXArray, objectScore: MLXArray) {
        let levels = imageEncoder.features(image)
        let hidden = configuration.hiddenDimensions
        let conditioned = levels[levels.count - 1] + noMemoryEmbedding.reshaped([1, 1, 1, hidden])
        let predicted = predict(conditioned: conditioned, levels: levels, points: points)
        let size = configuration.imageSize
        // The masks are `[1, 3, grid, grid]`; the resampler reads a channel axis, so the slots ride
        // there and come back off it.
        let upsampled = NFKMLXResample.resizeBilinear(predicted.masks.transposed(0, 2, 3, 1),
                                                      height: size, width: size)
        return (upsampled.transposed(0, 3, 1, 2), predicted.intersectionOverUnion,
                predicted.objectScore)
    }

    /// The object pointer a decoded mask token becomes. A frame the model reads as empty contributes
    /// the learned `no_obj_ptr` instead, which is what keeps an absent object from dragging the
    /// memory of a present one.
    func objectPointer(from token: MLXArray, present: MLXArray) -> MLXArray {
        let projected = objectPointerProjection(token)
        let weight = present.reshaped([-1, 1]).asType(projected.dtype)
        return weight * projected + (1 - weight) * noObjectPointer
    }

    /// Folds a frame's features and its predicted mask into the memory later frames attend over.
    ///
    /// A mask that came from clicks is binarized; a tracked one keeps its probabilities. The memory
    /// is stored through bfloat16, as the reference stores it — the rounding is part of what the next
    /// frame reads.
    func encodeMemory(features: MLXArray, maskLogits: MLXArray, objectScore: MLXArray,
                      fromPoints: Bool) -> (memory: MLXArray, position: MLXArray) {
        let probabilities = fromPoints ? (maskLogits .> 0).asType(maskLogits.dtype) : sigmoid(maskLogits)
        let scaled = probabilities * configuration.sigmoidScale + configuration.sigmoidBias
        var memory = memoryEncoder(features: features, maskInput: scaled)
        if let occlusionSpatialEmbedding {
            let absent = 1 - (objectScore .> 0).asType(memory.dtype).reshaped([-1, 1, 1, 1])
            memory = memory + absent * occlusionSpatialEmbedding.reshaped([1, 1, 1, -1])
        }
        let width = configuration.memoryDimensions
        let grid = memory.dim(1)
        memory = memory.asType(.bfloat16).asType(.float32).reshaped([1, grid * grid, width])
        let position = NFKMLXSAM2PositionEmbedding.sine(height: grid, width: grid, features: width)
            .reshaped([1, grid * grid, width])
        return (memory, position)
    }

    /// The top feature level, conditioned on what the session remembers.
    private func conditionedFeatures(_ features: MLXArray, position: MLXArray, frameIndex: Int,
                                     isConditioning: Bool, session: NFKMLXSAM2TrackerSession) -> MLXArray {
        let grid = configuration.featureGrid, hidden = configuration.hiddenDimensions
        if isConditioning || session.frames.isEmpty {
            return features + noMemoryEmbedding.reshaped([1, 1, 1, hidden])
        }
        let flat = features.reshaped([1, grid * grid, hidden])
        let (memory, memoryPosition, pointerTokens) = memoryBank(frameIndex: frameIndex, session: session)
        let conditioned = memoryAttention(current: flat, memory: memory, currentPosition: position,
                                          memoryPosition: memoryPosition,
                                          objectPointerTokens: pointerTokens)
        return conditioned.reshaped([1, grid, grid, hidden])
    }

    /// The memories and object pointers a frame attends over, in the reference's order: every
    /// conditioning frame first, then the `memoryFrames - 1` frames immediately before this one, then
    /// the object pointers.
    private func memoryBank(frameIndex: Int, session: NFKMLXSAM2TrackerSession)
        -> (memory: MLXArray, position: MLXArray, pointerTokens: Int) {
        var memories = [MLXArray]()
        var positions = [MLXArray]()
        for index in session.trackedFrames where session.frames[index]!.isConditioning {
            let frame = session.frames[index]!
            memories.append(frame.memory)
            // A conditioning frame takes the temporal encoding of distance zero wherever it sits.
            positions.append(frame.memoryPosition + memoryTemporalEncoding[configuration.memoryFrames - 1])
        }
        for offset in stride(from: configuration.memoryFrames - 1, through: 1, by: -1) {
            guard let frame = session.frames[frameIndex - offset], !frame.isConditioning else { continue }
            memories.append(frame.memory)
            positions.append(frame.memoryPosition + memoryTemporalEncoding[offset - 1])
        }

        let (pointers, pointerPositions) = objectPointerBank(frameIndex: frameIndex, session: session)
        if let pointers, let pointerPositions {
            memories.append(pointers)
            positions.append(pointerPositions)
        }
        return (concatenated(memories, axis: 1), concatenated(positions, axis: 1),
                pointers?.dim(1) ?? 0)
    }

    /// The object pointers of earlier frames, split into memory-width tokens.
    ///
    /// A pointer is as wide as the decoder's tokens and the memory attention reads memory-width keys,
    /// so each pointer becomes four consecutive tokens. Under 2.1 each carries a projected sine
    /// encoding of how many frames back it came from; under 2.0 it carries nothing.
    private func objectPointerBank(frameIndex: Int, session: NFKMLXSAM2TrackerSession)
        -> (MLXArray?, MLXArray?) {
        let hidden = configuration.hiddenDimensions, width = configuration.memoryDimensions
        var offsets = [Float]()
        var tokens = [MLXArray]()
        for index in session.trackedFrames where session.frames[index]!.isConditioning && index <= frameIndex {
            offsets.append(Float(frameIndex - index))
            tokens.append(session.frames[index]!.objectPointer)
        }
        let bound = min(session.frameCount ?? configuration.maximumObjectPointers,
                        configuration.maximumObjectPointers)
        for offset in 1 ..< max(bound, 1) {
            let index = frameIndex - offset
            guard index >= 0 else { break }
            guard let frame = session.frames[index], !frame.isConditioning else { continue }
            offsets.append(Float(offset))
            tokens.append(frame.objectPointer)
        }
        guard !tokens.isEmpty else { return (nil, nil) }

        let splits = hidden / width
        let stacked = concatenated(tokens, axis: 0).reshaped([tokens.count * splits, width])
        let pointers = stacked.reshaped([1, tokens.count * splits, width])

        var positions: MLXArray
        if let objectPointerTemporalProjection {
            let span = Float(bound - 1)
            let encoded = NFKMLXSAM2PositionEmbedding.sine(positions: offsets.map { $0 / span },
                                                           features: hidden)
            positions = objectPointerTemporalProjection(encoded)           // [tokens, width]
        } else {
            positions = MLXArray.zeros([tokens.count, width])
        }
        // Every split of one pointer shares that pointer's position.
        positions = repeated(positions.reshaped([tokens.count, 1, width]), count: splits, axis: 1)
        return (pointers, positions.reshaped([1, tokens.count * splits, width]))
    }
}

/// Holds a tracker for the backend, running one frame at a time with a fresh session.
private final class NFKMLXSAM2Holder: @unchecked Sendable {
    let net: NFKMLXSAM2TrackerNet
    init(_ net: NFKMLXSAM2TrackerNet) { self.net = net }

    /// A plate `[H, W, 3]` in `0...1` and a click in `0...1` → `[H, W, 4]`: the plate carried as
    /// straight foreground with the mask as alpha, which is the matting backend's contract.
    func segment(_ plate: MLXArray, point: (x: Float, y: Float)) -> MLXArray {
        let (height, width) = (plate.shape[0], plate.shape[1])
        let size = net.configuration.imageSize
        let resized = NFKMLXResample.resizeBilinear(plate.reshaped([1, height, width, 3]),
                                                    height: size, width: size)
        // SAM 2 normalizes with ImageNet statistics on the 0...1 scale, where SAM 1 uses the 0...255
        // one. Feeding either model the other's normalization leaves the encoder outside the
        // distribution it was trained on.
        let mean = MLXArray([Float(0.485), 0.456, 0.406])
        let deviation = MLXArray([Float(0.229), 0.224, 0.225])
        let normalized = (resized - mean) / deviation
        let prediction = net.track(image: normalized, frameIndex: 0,
                                   points: [(point.x * Float(size), point.y * Float(size), 1)],
                                   session: NFKMLXSAM2TrackerSession())
        let alpha = NFKMLXResample.resizeBilinear(
            sigmoid(prediction.maskLogits).expandedDimensions(axis: 3), height: height, width: width)
        return concatenated([plate, alpha.reshaped([height, width, 1])], axis: 2)
    }
}

extension NFKMLXSAM2 {

    @objc public static let modelName = "sam2"

    /// The configuration a released size and release carry.
    public static func configuration(for variant: NFKMLXSAM2Variant,
                                     release: NFKMLXSAM2Release = .sam2) -> NFKMLXSAM2TrackerConfiguration {
        NFKMLXSAM2TrackerConfiguration.geometry(variant, release: release)
    }

    /// Builds the tracker at a chosen size and release.
    public static func makeTracker(variant: NFKMLXSAM2Variant = .tiny,
                                   release: NFKMLXSAM2Release = .sam2) -> NFKMLXSAM2TrackerNet {
        NFKMLXSAM2TrackerNet(configuration(for: variant, release: release))
    }

    /// The module key a released tensor name maps to. The five networks keep their released prefixes
    /// so one checkpoint loads into one module; inside each, the existing per-network mapping applies.
    static func remapTrackerKey(_ key: String) -> String? {
        for prefix in ["image_encoder."] where key.hasPrefix(prefix) {
            return remapReferenceKey(key).map { prefix + $0 }
        }
        for prefix in ["sam_prompt_encoder.", "sam_mask_decoder."] where key.hasPrefix(prefix) {
            return remapDecoderKey(key).map { prefix + $0 }
        }
        for prefix in ["memory_attention.", "memory_encoder."] where key.hasPrefix(prefix) {
            return remapMemoryKey(key).map { prefix + $0 }
        }
        // The tracker's own parameters are already named as the module names them.
        return key
    }

    /// Loads a whole released checkpoint into the tracker. Every tensor a release ships is read.
    public static func loadWeights(into tracker: NFKMLXSAM2TrackerNet, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let mapped = checkpoint.arrays.compactMap { key, value -> (String, MLXArray)? in
            guard let name = remapTrackerKey(key) else { return nil }
            guard checkpoint.needsConvTranspose, value.ndim == 4 else { return (name, value) }
            // The temporal encodings are `[frames, 1, 1, width]`, not a convolution weight: the axis
            // move a convolution needs would silently reshape them to `[frames, 1, width, 1]`, which
            // broadcasts against nothing and only fails once a second frame reads the memory.
            if name == "maskmem_tpos_enc" { return (name, value) }
            // The mask decoder upscales with transposed convolutions, stored `[in, out, kH, kW]`;
            // every other 4-D tensor here is a forward convolution or a position grid.
            if name.contains("sam_mask_decoder.upscale") { return (name, value.transposed(1, 2, 3, 0)) }
            return (name, value.transposed(0, 2, 3, 1))
        }
        try NFKMLXWeights.apply(mapped, to: tracker)
    }

    /// Builds a SAM 2 backend at a chosen size and release. A released checkpoint fits only its own
    /// size, and a 2.1 checkpoint carries two parameters a 2.0 configuration does not declare.
    @objc(backendWithVariant:release:weightsURL:error:)
    public static func backend(variant: NFKMLXSAM2Variant, release: NFKMLXSAM2Release,
                               weightsURL: URL?) throws -> any NFKInferenceBackend {
        let net = makeTracker(variant: variant, release: release)
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        let holder = NFKMLXSAM2Holder(net)
        var configuration = NFKMattingConfiguration()
        configuration.emitsMatte = true
        return NFKMLXMattingBackend(identifier: modelName, configuration: configuration,
                                    forwardParameterKeys: [NFKSAMPointKey]) { plate, _, request in
            let point = NFKMLXSAM2.point(from: request, width: plate.shape[1], height: plate.shape[0])
            return holder.segment(plate, point: point)
        }
    }

    /// The click in `0...1`, read from the request's `NFKSAMPointKey` parameter (`[x, y]` in pixels
    /// for a `width` × `height` plate). Defaults to the plate's center when absent.
    static func point(from request: NFKInferenceRequest, width: Int, height: Int) -> (x: Float, y: Float) {
        guard let value = request.parameter(forKey: NFKSAMPointKey) as? [NSNumber], value.count >= 2,
              width > 0, height > 0 else {
            return (0.5, 0.5)
        }
        return (min(max(value[0].floatValue / Float(width), 0), 1),
                min(max(value[1].floatValue / Float(height), 0), 1))
    }

    /// Builds a SAM 2 backend from optional local weights — no registry required. A nil `weightsURL`
    /// builds random weights. The request supplies the plate under `NFKInputImage` and an optional
    /// click under `NFKSAMPointKey`. Run inference off the render thread.
    @objc(backendWithWeightsURL:error:)
    public static func backend(weightsURL: URL?) throws -> any NFKInferenceBackend {
        try backend(variant: .tiny, release: .sam21, weightsURL: weightsURL)
    }

    /// Downloads the checkpoint from Hugging Face, then builds at a chosen size and release.
    /// Blocking on the network; run off the render thread.
    @objc(backendWithVariant:release:repo:weightsPath:revision:cacheDirectoryURL:error:)
    public static func backend(variant: NFKMLXSAM2Variant, release: NFKMLXSAM2Release, repo: String,
                               weightsPath: String, revision: String?,
                               cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        let url = try NFKMLXDownload.weightsURL(repo: repo, weightsPath: weightsPath,
                                                revision: revision, cacheDirectoryURL: cacheDirectoryURL)
        return try backend(variant: variant, release: release, weightsURL: url)
    }

    /// Downloads the checkpoint from Hugging Face, then builds — no registry required.
    /// Blocking on the network; run off the render thread.
    @objc(backendWithRepo:weightsPath:revision:cacheDirectoryURL:error:)
    public static func backend(repo: String, weightsPath: String, revision: String?,
                               cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        try backend(variant: .tiny, release: .sam21, repo: repo, weightsPath: weightsPath,
                    revision: revision, cacheDirectoryURL: cacheDirectoryURL)
    }

    /// The asynchronous form of the variant download factory.
    @objc(backendWithVariant:release:repo:weightsPath:revision:cacheDirectoryURL:completionHandler:)
    public static func backend(variant: NFKMLXSAM2Variant, release: NFKMLXSAM2Release, repo: String,
                               weightsPath: String, revision: String?, cacheDirectoryURL: URL?,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXDownload.backend(repo: repo, weightsPath: weightsPath, revision: revision,
                               cacheDirectoryURL: cacheDirectoryURL,
                               build: { try backend(variant: variant, release: release, weightsURL: $0) },
                               completionHandler: completionHandler)
    }

    /// The asynchronous form of the download factory.
    @objc(backendWithRepo:weightsPath:revision:cacheDirectoryURL:completionHandler:)
    public static func backend(repo: String, weightsPath: String, revision: String?,
                               cacheDirectoryURL: URL?,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        backend(variant: .tiny, release: .sam21, repo: repo, weightsPath: weightsPath,
                revision: revision, cacheDirectoryURL: cacheDirectoryURL,
                completionHandler: completionHandler)
    }

    /// Registers SAM 2 (`sam2`) with `NFKMLXModelRegistry`, delegating to `backend(weightsURL:)`.
    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName) { weightsURL in try backend(weightsURL: weightsURL) }
    }
}

