//
//  NFKMLXRetinaFace.swift
//  InferKitMLX
//
//  RetinaFace face detection with five-point landmarks, in `MLXNN`. This is the detector the CodeFormer
//  reference pipeline uses (through facexlib), so it is what makes a restored photograph comparable to
//  the reference's rather than merely similar to it.
//
//  The mobile0.25 release is the one ported here: a MobileNetV1 backbone at quarter width, a
//  three-level FPN, SSH context modules, and per-level class / box / landmark heads over two anchors a
//  cell. The whole checkpoint is 1.7 MB.
//

import CoreGraphics
import Foundation
import MLX
import InferKit
import MLXNN

/// The RetinaFace geometry. The released mobile0.25 values are the defaults.
public struct NFKMLXRetinaFaceConfiguration: Sendable {
    /// Backbone width at the second stage; the FPN takes 2×, 4×, and 8× of it.
    public var inChannels: Int = 32
    /// The width the FPN and SSH modules run at.
    public var outChannels: Int = 64
    /// Anchor box sizes per pyramid level, in input pixels.
    public var minSizes: [[Float]] = [[16, 32], [64, 128], [256, 512]]
    /// The stride each pyramid level is sampled at.
    public var steps: [Int] = [8, 16, 32]
    /// Box and landmark decoding variances.
    public var variance: [Float] = [0.1, 0.2]
    /// The negative slope every activation uses (`out_channels <= 64` selects 0.1 in the reference).
    public var leakySlope: Float = 0.1

    public init() {}

    /// Anchors per cell, which the heads' channel counts follow.
    var anchorsPerCell: Int { minSizes[0].count }
}

/// A 3×3 or 1×1 convolution with batch normalization, optionally activated.
///
/// The reference builds these as `nn.Sequential`, so its keys are positional (`.0` the convolution,
/// `.1` the normalization); `remapReferenceKey` translates that.
final class NFKRetinaConvBN: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d
    @ModuleInfo(key: "bn") var bn: BatchNorm
    let slope: Float?

    init(_ inChannels: Int, _ outChannels: Int, kernel: Int = 3, stride: Int = 1, slope: Float? = 0.1) {
        _conv.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: outChannels,
                                    kernelSize: IntOrPair(kernel), stride: IntOrPair(stride),
                                    padding: IntOrPair(kernel == 3 ? 1 : 0), bias: false)
        _bn.wrappedValue = BatchNorm(featureCount: outChannels)
        self.slope = slope
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = bn(conv(x))
        guard let slope else { return y }
        return leakyRelu(y, negativeSlope: slope)
    }
}

/// A depthwise-separable block: a grouped 3×3, then a pointwise 1×1, each normalized and activated.
final class NFKRetinaDepthwise: Module {
    @ModuleInfo(key: "dwconv") var dwconv: Conv2d
    @ModuleInfo(key: "dwbn") var dwbn: BatchNorm
    @ModuleInfo(key: "pwconv") var pwconv: Conv2d
    @ModuleInfo(key: "pwbn") var pwbn: BatchNorm
    let slope: Float

    init(_ inChannels: Int, _ outChannels: Int, stride: Int, slope: Float = 0.1) {
        _dwconv.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: inChannels,
                                      kernelSize: 3, stride: IntOrPair(stride), padding: 1,
                                      groups: inChannels, bias: false)
        _dwbn.wrappedValue = BatchNorm(featureCount: inChannels)
        _pwconv.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: outChannels,
                                      kernelSize: 1, bias: false)
        _pwbn.wrappedValue = BatchNorm(featureCount: outChannels)
        self.slope = slope
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let d = leakyRelu(dwbn(dwconv(x)), negativeSlope: slope)
        return leakyRelu(pwbn(pwconv(d)), negativeSlope: slope)
    }
}

/// The SSH context module: a 3×3 branch beside 5×5 and 7×7 receptive fields built from stacked 3×3s,
/// concatenated and activated together.
final class NFKRetinaSSH: Module {
    @ModuleInfo(key: "conv3X3") var conv3x3: NFKRetinaConvBN
    @ModuleInfo(key: "conv5X5_1") var conv5x5a: NFKRetinaConvBN
    @ModuleInfo(key: "conv5X5_2") var conv5x5b: NFKRetinaConvBN
    @ModuleInfo(key: "conv7X7_2") var conv7x7a: NFKRetinaConvBN
    @ModuleInfo(key: "conv7x7_3") var conv7x7b: NFKRetinaConvBN

    init(_ channels: Int, slope: Float) {
        _conv3x3.wrappedValue = NFKRetinaConvBN(channels, channels / 2, slope: nil)
        _conv5x5a.wrappedValue = NFKRetinaConvBN(channels, channels / 4, slope: slope)
        _conv5x5b.wrappedValue = NFKRetinaConvBN(channels / 4, channels / 4, slope: nil)
        _conv7x7a.wrappedValue = NFKRetinaConvBN(channels / 4, channels / 4, slope: slope)
        _conv7x7b.wrappedValue = NFKRetinaConvBN(channels / 4, channels / 4, slope: nil)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let three = conv3x3(x)
        let five1 = conv5x5a(x)
        let five = conv5x5b(five1)
        let seven = conv7x7b(conv7x7a(five1))
        return relu(concatenated([three, five, seven], axis: -1))
    }
}

/// One 1×1 head. Its key matches the reference's (`conv1x1`), so the head weights need no remapping.
final class NFKRetinaHead: Module {
    @ModuleInfo(key: "conv1x1") var conv1x1: Conv2d

    init(_ channels: Int, outputs: Int) {
        _conv1x1.wrappedValue = Conv2d(inputChannels: channels, outputChannels: outputs, kernelSize: 1)
        super.init()
    }

    /// `[1, H, W, anchors·width]` becomes `[predictions, width]`, which is the reference's reshape.
    func callAsFunction(_ x: MLXArray, width: Int) -> MLXArray {
        conv1x1(x).reshaped([-1, width])
    }
}

/// The MobileNetV1 backbone, FPN, SSH modules, and heads.
final class NFKMLXRetinaFaceNet: Module {
    // The reference's `stage1` opens with a plain convolution and continues with depthwise blocks.
    // Naming the stem separately keeps each array homogeneous; the remap accounts for the offset.
    @ModuleInfo(key: "stem") var stem: NFKRetinaConvBN
    @ModuleInfo(key: "stage1") var stage1: [NFKRetinaDepthwise]
    @ModuleInfo(key: "stage2") var stage2: [NFKRetinaDepthwise]
    @ModuleInfo(key: "stage3") var stage3: [NFKRetinaDepthwise]

    @ModuleInfo(key: "output1") var output1: NFKRetinaConvBN
    @ModuleInfo(key: "output2") var output2: NFKRetinaConvBN
    @ModuleInfo(key: "output3") var output3: NFKRetinaConvBN
    @ModuleInfo(key: "merge1") var merge1: NFKRetinaConvBN
    @ModuleInfo(key: "merge2") var merge2: NFKRetinaConvBN

    @ModuleInfo(key: "ssh1") var ssh1: NFKRetinaSSH
    @ModuleInfo(key: "ssh2") var ssh2: NFKRetinaSSH
    @ModuleInfo(key: "ssh3") var ssh3: NFKRetinaSSH

    @ModuleInfo(key: "ClassHead") var classHead: [NFKRetinaHead]
    @ModuleInfo(key: "BboxHead") var bboxHead: [NFKRetinaHead]
    @ModuleInfo(key: "LandmarkHead") var landmarkHead: [NFKRetinaHead]

    let configuration: NFKMLXRetinaFaceConfiguration

    init(_ c: NFKMLXRetinaFaceConfiguration) {
        configuration = c
        let slope = c.leakySlope
        _stem.wrappedValue = NFKRetinaConvBN(3, 8, stride: 2, slope: slope)
        _stage1.wrappedValue = [NFKRetinaDepthwise(8, 16, stride: 1, slope: slope),
                                NFKRetinaDepthwise(16, 32, stride: 2, slope: slope),
                                NFKRetinaDepthwise(32, 32, stride: 1, slope: slope),
                                NFKRetinaDepthwise(32, 64, stride: 2, slope: slope),
                                NFKRetinaDepthwise(64, 64, stride: 1, slope: slope)]
        _stage2.wrappedValue = [NFKRetinaDepthwise(64, 128, stride: 2, slope: slope)]
            + (0 ..< 5).map { _ in NFKRetinaDepthwise(128, 128, stride: 1, slope: slope) }
        _stage3.wrappedValue = [NFKRetinaDepthwise(128, 256, stride: 2, slope: slope),
                                NFKRetinaDepthwise(256, 256, stride: 1, slope: slope)]

        let out = c.outChannels
        _output1.wrappedValue = NFKRetinaConvBN(c.inChannels * 2, out, kernel: 1, slope: slope)
        _output2.wrappedValue = NFKRetinaConvBN(c.inChannels * 4, out, kernel: 1, slope: slope)
        _output3.wrappedValue = NFKRetinaConvBN(c.inChannels * 8, out, kernel: 1, slope: slope)
        _merge1.wrappedValue = NFKRetinaConvBN(out, out, slope: slope)
        _merge2.wrappedValue = NFKRetinaConvBN(out, out, slope: slope)

        _ssh1.wrappedValue = NFKRetinaSSH(out, slope: slope)
        _ssh2.wrappedValue = NFKRetinaSSH(out, slope: slope)
        _ssh3.wrappedValue = NFKRetinaSSH(out, slope: slope)

        let anchors = c.anchorsPerCell
        _classHead.wrappedValue = (0 ..< 3).map { _ in NFKRetinaHead(out, outputs: anchors * 2) }
        _bboxHead.wrappedValue = (0 ..< 3).map { _ in NFKRetinaHead(out, outputs: anchors * 4) }
        _landmarkHead.wrappedValue = (0 ..< 3).map { _ in NFKRetinaHead(out, outputs: anchors * 10) }
        super.init()
    }

    /// Runs the network on a prepared `[1, H, W, 3]` input.
    ///
    /// - Returns: box offsets `[predictions, 4]`, class scores `[predictions, 2]`, and landmark
    ///   offsets `[predictions, 10]`, all in the reference's prediction order.
    func callAsFunction(_ x: MLXArray) -> (boxes: MLXArray, scores: MLXArray, landmarks: MLXArray) {
        var feature = stem(x)
        for block in stage1 { feature = block(feature) }
        let c1 = feature
        for block in stage2 { feature = block(feature) }
        let c2 = feature
        for block in stage3 { feature = block(feature) }
        let c3 = feature

        // The FPN fuses top-down with NEAREST resampling, as `F.interpolate` defaults to.
        let p3 = output3(c3)
        var p2 = output2(c2)
        var p1 = output1(c1)
        // The coarsest level is used as produced; only the finer two are fused.
        p2 = merge2(p2 + NFKMLXResample.resizeNearest(p3, height: p2.shape[1], width: p2.shape[2]))
        p1 = merge1(p1 + NFKMLXResample.resizeNearest(p2, height: p1.shape[1], width: p1.shape[2]))

        let features = [ssh1(p1), ssh2(p2), ssh3(p3)]
        let boxes = concatenated(zip(bboxHead, features).map { $0($1, width: 4) }, axis: 0)
        let scores = concatenated(zip(classHead, features).map { $0($1, width: 2) }, axis: 0)
        let landmarks = concatenated(zip(landmarkHead, features).map { $0($1, width: 10) }, axis: 0)
        return (boxes, softmax(scores, axis: -1), landmarks)
    }
}

/// Holds the network for capture in a `@Sendable` body.
private final class NFKRetinaHolder: @unchecked Sendable {
    let net: NFKMLXRetinaFaceNet
    init(_ net: NFKMLXRetinaFaceNet) { self.net = net }
}

/// RetinaFace detection: anchors, decoding, suppression, and weight loading.
///
/// This is the detector the CodeFormer reference pipeline runs, so a photograph aligned through it is
/// aligned the way the reference aligns it. `NFKMLXFaceAlignment` uses Vision instead when no detector
/// is supplied, which needs no weights but is not the reference's.
@objc(NFKMLXRetinaFace)
public final class NFKMLXRetinaFace: NSObject {

    /// The registry name the model builds under.
    @objc public static let modelName = "retinaface-mobile025"

    /// The per-channel mean the reference subtracts, in the BGR order its OpenCV input arrives in.
    static let channelMean: [Float] = [104, 117, 123]

    static func makeNet(_ configuration: NFKMLXRetinaFaceConfiguration = NFKMLXRetinaFaceConfiguration())
        -> NFKMLXRetinaFaceNet {
        let net = NFKMLXRetinaFaceNet(configuration)
        net.train(false)                    // batch normalization uses the checkpoint's running statistics
        return net
    }

    /// The anchor boxes for an input of `height`×`width`, as `[anchors, 4]` centre-form and normalized.
    ///
    /// One anchor per size per cell, walking the levels coarse-last, which is the order the heads'
    /// predictions are concatenated in.
    static func anchors(height: Int, width: Int,
                        configuration: NFKMLXRetinaFaceConfiguration) -> MLXArray {
        var values = [Float]()
        for (level, step) in configuration.steps.enumerated() {
            let rows = Int(ceil(Double(height) / Double(step)))
            let columns = Int(ceil(Double(width) / Double(step)))
            for row in 0 ..< rows {
                for column in 0 ..< columns {
                    for size in configuration.minSizes[level] {
                        let centreX = (Float(column) + 0.5) * Float(step) / Float(width)
                        let centreY = (Float(row) + 0.5) * Float(step) / Float(height)
                        values += [centreX, centreY, size / Float(width), size / Float(height)]
                    }
                }
            }
        }
        return MLXArray(values).reshaped([values.count / 4, 4])
    }

    /// Turns the network's offsets into corner-form boxes and absolute landmarks.
    ///
    /// The reference's decode: a centre is the anchor's centre plus a variance-scaled offset of the
    /// anchor's size, and a size is the anchor's size times the exponential of its offset.
    static func decode(boxes: MLXArray, landmarks: MLXArray, anchors: MLXArray,
                       variance: [Float]) -> (boxes: MLXArray, landmarks: MLXArray) {
        let centres = anchors[0..., 0 ..< 2]
        let sizes = anchors[0..., 2 ..< 4]

        let decodedCentres = centres + boxes[0..., 0 ..< 2] * variance[0] * sizes
        let decodedSizes = sizes * exp(boxes[0..., 2 ..< 4] * variance[1])
        let topLeft = decodedCentres - decodedSizes / 2
        let bottomRight = topLeft + decodedSizes

        // Each of the five points decodes against the same anchor.
        let points = (0 ..< 5).map { index -> MLXArray in
            centres + landmarks[0..., (index * 2) ..< (index * 2 + 2)] * variance[0] * sizes
        }
        return (concatenated([topLeft, bottomRight], axis: 1), concatenated(points, axis: 1))
    }

    /// Greedy non-maximum suppression over corner-form boxes, highest score first.
    static func suppress(boxes: [[Float]], scores: [Float], threshold: Float) -> [Int] {
        let order = scores.indices.sorted { scores[$0] > scores[$1] }
        var keep = [Int]()
        var removed = Set<Int>()
        for candidate in order where !removed.contains(candidate) {
            keep.append(candidate)
            for other in order where other != candidate && !removed.contains(other) {
                if overlap(boxes[candidate], boxes[other]) > threshold { removed.insert(other) }
            }
        }
        return keep
    }

    private static func overlap(_ a: [Float], _ b: [Float]) -> Float {
        let x0 = max(a[0], b[0]), y0 = max(a[1], b[1])
        let x1 = min(a[2], b[2]), y1 = min(a[3], b[3])
        let intersection = max(0, x1 - x0) * max(0, y1 - y0)
        let union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
        return union > 0 ? intersection / union : 0
    }

    /// Detects faces in `image`, most confident first.
    ///
    /// Coordinates come back in image pixels with a top-left origin, matching `NFKFaceObservation`.
    /// Thresholds are the reference's own defaults.
    static func detect(_ image: CGImage, using net: NFKMLXRetinaFaceNet,
                       confidenceThreshold: Float = 0.8,
                       suppressionThreshold: Float = 0.4) -> [NFKFaceObservation] {
        detect(prepared(image), width: image.width, height: image.height, using: net,
               confidenceThreshold: confidenceThreshold, suppressionThreshold: suppressionThreshold)
    }

    /// Detects from an already-bridged `[H, W, 3]` RGB tensor in `0...1`.
    static func detect(tensor: MLXArray, using net: NFKMLXRetinaFaceNet,
                       confidenceThreshold: Float = 0.8,
                       suppressionThreshold: Float = 0.4) -> [NFKFaceObservation] {
        detect(prepared(tensor), width: tensor.shape[1], height: tensor.shape[0], using: net,
               confidenceThreshold: confidenceThreshold, suppressionThreshold: suppressionThreshold)
    }

    private static func detect(_ input: MLXArray, width: Int, height: Int,
                               using net: NFKMLXRetinaFaceNet,
                               confidenceThreshold: Float,
                               suppressionThreshold: Float) -> [NFKFaceObservation] {
        let (rawBoxes, rawScores, rawLandmarks) = net(input)
        let priors = anchors(height: height, width: width, configuration: net.configuration)
        let (decodedBoxes, decodedLandmarks) = decode(boxes: rawBoxes, landmarks: rawLandmarks,
                                                     anchors: priors, variance: net.configuration.variance)
        eval(decodedBoxes, decodedLandmarks, rawScores)

        // The second class is "face"; the first is background.
        let scores = rawScores[0..., 1].asArray(Float.self)
        let boxValues = decodedBoxes.asArray(Float.self)
        let landmarkValues = decodedLandmarks.asArray(Float.self)

        var candidates = [Int]()
        for (index, score) in scores.enumerated() where score > confidenceThreshold {
            candidates.append(index)
        }
        guard !candidates.isEmpty else { return [] }

        // Decoding is normalized; the reference scales by the frame before suppressing.
        let scaled = candidates.map { index -> [Float] in
            [boxValues[index * 4 + 0] * Float(width), boxValues[index * 4 + 1] * Float(height),
             boxValues[index * 4 + 2] * Float(width), boxValues[index * 4 + 3] * Float(height)]
        }
        let candidateScores = candidates.map { scores[$0] }
        let kept = suppress(boxes: scaled, scores: candidateScores, threshold: suppressionThreshold)

        return kept.map { position -> NFKFaceObservation in
            let index = candidates[position]
            let box = scaled[position]
            let points = (0 ..< 5).map { point in
                CGPoint(x: CGFloat(landmarkValues[index * 10 + point * 2] * Float(width)),
                        y: CGFloat(landmarkValues[index * 10 + point * 2 + 1] * Float(height)))
            }
            return NFKFaceObservation(
                boundingBox: CGRect(x: CGFloat(box[0]), y: CGFloat(box[1]),
                                    width: CGFloat(box[2] - box[0]), height: CGFloat(box[3] - box[1])),
                landmarks: points, confidence: candidateScores[position])
        }
    }

    /// Bridges a frame to the network's input: 0…255 in BGR order, with the reference's mean removed.
    static func prepared(_ image: CGImage) -> MLXArray {
        let (bytes, width, height) = NFKMLXImageBridge.rgbaBytes(from: image,
                                                                 colorSpace: CGColorSpaceCreateDeviceRGB())
        return prepared(NFKMLXImageBridge.tensor(rgba: bytes, width: width, height: height, channels: 3))
    }

    /// The same preparation from an already-bridged `[H, W, 3]` **RGB** tensor in `0...1`.
    ///
    /// Two things are easy to get wrong here and neither raises on its own: the channel order and the
    /// scale. The order cannot be detected from the values, so it is pinned by
    /// `testTheChannelOrderIsLoadBearing` instead — swapping it costs the detection outright. The
    /// scale can be detected, and is: a caller handing over `0...255` has already applied the wrong
    /// units, and the mean subtraction below would be meaningless.
    static func prepared(_ rgb01: MLXArray) -> MLXArray {
        assert({
            let peak = rgb01.max().item(Float.self)
            return peak <= 1.5
        }(), "RetinaFace takes RGB in 0...1; this tensor looks like 0...255")
        let (height, width) = (rgb01.shape[0], rgb01.shape[1])
        let rgb = rgb01 * 255
        // The reference reads its images through OpenCV, so its channel order is BGR.
        let bgr = concatenated([rgb[0..., 0..., 2 ..< 3], rgb[0..., 0..., 1 ..< 2], rgb[0..., 0..., 0 ..< 1]],
                               axis: -1)
        return (bgr - MLXArray(channelMean)).reshaped([1, height, width, 3])
    }

    /// Loads a converted safetensors checkpoint.
    static func loadWeights(into net: NFKMLXRetinaFaceNet, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let mapped = checkpoint.arrays.compactMap { key, value -> (String, MLXArray)? in
            guard let name = remapReferenceKey(key) else { return nil }
            if checkpoint.needsConvTranspose, value.ndim == 4 {
                return (name, value.transposed(0, 2, 3, 1))
            }
            return (name, value)
        }
        try NFKMLXWeights.apply(mapped, to: net)
    }

    /// Translates the reference's positional `nn.Sequential` keys onto the module's names.
    ///
    /// Returns nil for a key the port does not carry: the backbone's classifier (`body.fc`, `body.avg`)
    /// is trained for ImageNet and unused for detection, and `num_batches_tracked` is a counter rather
    /// than a parameter.
    static func remapReferenceKey(_ key: String) -> String? {
        if key.hasSuffix("num_batches_tracked") { return nil }
        if key.hasPrefix("body.fc") || key.hasPrefix("body.avg") { return nil }

        var name = key
        // The heads already match: ClassHead.0.conv1x1.weight and so on.
        if name.hasPrefix("ClassHead") || name.hasPrefix("BboxHead") || name.hasPrefix("LandmarkHead") {
            return name
        }
        if name.hasPrefix("fpn.") {
            name = String(name.dropFirst("fpn.".count))
        }
        name = name.replacingOccurrences(of: "body.", with: "")

        // `stage1.0` is the plain stem convolution; the depthwise blocks after it shift down by one.
        if name.hasPrefix("stage1.") {
            let parts = name.split(separator: ".").map(String.init)
            if let index = Int(parts[1]) {
                name = index == 0 ? (["stem"] + parts.dropFirst(2)).joined(separator: ".")
                                  : (["stage1", String(index - 1)] + parts.dropFirst(2)).joined(separator: ".")
            }
        }
        return positional(name)
    }

    /// Renames a Sequential position to the slot it holds.
    ///
    /// A plain convolution block is `.0` the convolution and `.1` the normalization; a depthwise block
    /// adds `.3` and `.4` for the pointwise pair. Which one a key belongs to is decided by whether the
    /// block is the stem, an FPN or SSH convolution, or a depthwise stage block.
    private static func positional(_ name: String) -> String {
        let depthwise = ["0": "dwconv", "1": "dwbn", "3": "pwconv", "4": "pwbn"]
        let plain = ["0": "conv", "1": "bn"]
        var parts = name.split(separator: ".").map(String.init)
        guard let slotIndex = parts.firstIndex(where: { Int($0) != nil && $0.count == 1 }),
              slotIndex + 1 < parts.count
        else { return name }

        // A stage block is depthwise; everything else (stem, FPN, SSH) is a plain convolution pair.
        let isStageBlock = parts.first.map { $0.hasPrefix("stage") } ?? false
        let table = isStageBlock && slotIndex > 0 ? depthwise : plain
        // For a stage, the numeric block index comes first and the slot second.
        let slot = isStageBlock ? slotIndex + 1 : slotIndex
        guard slot < parts.count, let replacement = table[parts[slot]] else { return name }
        parts[slot] = replacement
        return parts.joined(separator: ".")
    }
}

/// Detection through RetinaFace, which is what the CodeFormer reference pipeline uses.
///
/// Build one with ``NFKMLXRetinaFace/detector(weightsURL:confidenceThreshold:suppressionThreshold:)`` and hand it to
/// `NFKMLXCodeFormer.photoBackend`. It costs a 1.7 MB checkpoint and makes the aligned crop the
/// reference's own.
@objc(NFKMLXRetinaFaceDetector)
public final class NFKMLXRetinaFaceDetector: NSObject, NFKMLXFaceDetecting, @unchecked Sendable {
    private let holder: NFKRetinaHolder
    private let confidenceThreshold: Float
    private let suppressionThreshold: Float

    init(net: NFKMLXRetinaFaceNet, confidenceThreshold: Float, suppressionThreshold: Float) {
        self.holder = NFKRetinaHolder(net)
        self.confidenceThreshold = confidenceThreshold
        self.suppressionThreshold = suppressionThreshold
        super.init()
    }

    public func faces(in image: CGImage) throws -> [NFKFaceObservation] {
        NFKMLXRetinaFace.detect(image, using: holder.net,
                                confidenceThreshold: confidenceThreshold,
                                suppressionThreshold: suppressionThreshold)
    }

    /// The Objective-C entry: the faces in `image`, each with its box, confidence, and five landmarks.
    @objc(facesInImage:error:)
    public func faces(inImage image: CGImage) throws -> [NFKFaceObservation] {
        try faces(in: image)
    }
}

extension NFKMLXRetinaFace {

    /// Builds a detector from a converted checkpoint. Thresholds default to the reference's own. The
    /// returned detector is `@objc`, so an Objective-C caller reuses it across images and reads each
    /// face's five landmarks — the alignment data the detection backend's boxes omit.
    @objc(detectorWithWeightsURL:confidenceThreshold:suppressionThreshold:error:)
    public static func detector(weightsURL: URL?, confidenceThreshold: Float = 0.8,
                                suppressionThreshold: Float = 0.4) throws -> NFKMLXRetinaFaceDetector {
        let net = makeNet()
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        return NFKMLXRetinaFaceDetector(net: net, confidenceThreshold: confidenceThreshold,
                                        suppressionThreshold: suppressionThreshold)
    }
}

extension NFKMLXRetinaFace {

    /// Builds a face-detection backend directly from optional local weights — no registry required.
    ///
    /// Reads `NFKInputImage` and returns `NSArray<NFKDetection *>` under `NFKOutputDetections`, with
    /// boxes normalized 0…1 and the origin top-left, the same convention `NFKMLXYOLO` uses. The five
    /// landmarks are not carried by `NFKDetection`; use
    /// ``detector(weightsURL:confidenceThreshold:suppressionThreshold:)`` when you need them.
    /// A nil `weightsURL` builds random weights. Run inference off the render thread.
    @objc(backendWithWeightsURL:error:)
    public static func backend(weightsURL: URL?) throws -> any NFKInferenceBackend {
        let net = makeNet()
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        let holder = NFKRetinaHolder(net)
        return NFKMLXDetectionBackend(identifier: modelName) { image in
            detect(image, using: holder.net).map { face in
                let box = CGRect(x: face.boundingBox.minX / CGFloat(image.width),
                                 y: face.boundingBox.minY / CGFloat(image.height),
                                 width: face.boundingBox.width / CGFloat(image.width),
                                 height: face.boundingBox.height / CGFloat(image.height))
                return NFKDetection(label: "face", classIndex: 0,
                                    confidence: Double(face.confidence), boundingBox: box)
            }
        }
    }

    /// Downloads the checkpoint from Hugging Face, then builds — no registry required.
    /// Blocking on the network; run off the render thread.
    @objc(backendWithRepo:weightsPath:revision:cacheDirectoryURL:error:)
    public static func backend(repo: String, weightsPath: String, revision: String?,
                               cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        let url = try NFKMLXDownload.weightsURL(repo: repo, weightsPath: weightsPath,
                                                revision: revision, cacheDirectoryURL: cacheDirectoryURL)
        return try backend(weightsURL: url)
    }

    /// The asynchronous form of the download factory: downloads on a background queue, then builds and
    /// delivers the backend (or an error) to `completionHandler`.
    @objc(backendWithRepo:weightsPath:revision:cacheDirectoryURL:completionHandler:)
    public static func backend(repo: String, weightsPath: String, revision: String?, cacheDirectoryURL: URL?,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXDownload.backend(repo: repo, weightsPath: weightsPath, revision: revision,
                               cacheDirectoryURL: cacheDirectoryURL,
                               build: { try backend(weightsURL: $0) },
                               completionHandler: completionHandler)
    }

    /// Registers RetinaFace with `NFKMLXModelRegistry`, delegating to `backend(weightsURL:)`.
    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName) { weightsURL in try backend(weightsURL: weightsURL) }
    }
}

/// An image backend returning detections, for models whose output is a list of boxes.
final class NFKMLXDetectionBackend: NSObject, NFKInferenceBackend {
    private let identifier: String
    private let box: NFKMLXDetectionBox

    init(identifier: String, detect: @escaping (CGImage) -> [NFKDetection]) {
        self.identifier = identifier
        self.box = NFKMLXDetectionBox(detect)
        super.init()
    }

    var isReady: Bool { true }
    var backendIdentifier: String { identifier }

    func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        guard let value = request.input(forKey: NFKInputImage),
              CFGetTypeID(value as CFTypeRef) == CGImage.typeID else {
            throw NFKMLXError.unsupportedInput
        }
        return NFKInferenceResult(outputs: [NFKOutputDetections: box.detect(value as! CGImage)])
    }
}

private final class NFKMLXDetectionBox: @unchecked Sendable {
    let detect: (CGImage) -> [NFKDetection]
    init(_ detect: @escaping (CGImage) -> [NFKDetection]) { self.detect = detect }
}
