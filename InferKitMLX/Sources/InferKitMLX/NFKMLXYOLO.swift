//
//  NFKMLXYOLO.swift
//  InferKitMLX
//

import Foundation
import CoreGraphics
import InferKit
import MLX
import MLXNN

// YOLO detects objects in one forward pass. This is the reference YOLOv8 (ultralytics): a CSPDarknet
// backbone of Conv and C2f stages ending in SPPF, a PAN-FPN neck that fuses three scales in both
// directions, and a decoupled detection head with distribution-focal box regression — each box side is
// a softmax over `regMax` bins whose expectation (the DFL convolution, fixed to 0...regMax-1) is a
// distance from the cell's anchor point. There is no objectness in v8; a candidate's confidence is its
// best class probability. Decoding turns the grid predictions into normalized boxes, and greedy
// per-class non-max suppression removes overlapping duplicates. The result is an
// NSArray<NFKDetection *> under NFKOutputDetections. Tensors flow in NHWC.

/// YOLO dimensions and thresholds. `base` is YOLOv8n (640-input, 80-class COCO); `tiny` keeps tests
/// fast. The thresholds default to the reference's inference settings.
public struct NFKMLXYOLOConfiguration: Sendable {
    public var inputResolution: Int
    /// Channels after each stride stage, P1/2 through P5/32.
    public var widths: [Int]
    /// C2f bottleneck repeats at the P2, P3, P4, and P5 stages (the neck's C2f blocks repeat
    /// `backboneRepeats[0]` times, as the reference scales both from the same depth multiple).
    public var backboneRepeats: [Int]
    public var classCount: Int
    /// Distribution-focal bins per box side.
    public var regMax: Int
    public var confidenceThreshold: Float
    public var iouThreshold: Float

    public init(inputResolution: Int = 640, widths: [Int] = [16, 32, 64, 128, 256],
                backboneRepeats: [Int] = [1, 2, 2, 1], classCount: Int = 80, regMax: Int = 16,
                confidenceThreshold: Float = 0.25, iouThreshold: Float = 0.7) {
        self.inputResolution = inputResolution
        self.widths = widths
        self.backboneRepeats = backboneRepeats
        self.classCount = classCount
        self.regMax = regMax
        self.confidenceThreshold = confidenceThreshold
        self.iouThreshold = iouThreshold
    }

    /// YOLOv8n, the smallest released model.
    public static let base = NFKMLXYOLOConfiguration()

    /// YOLOv8s: the same depth as `base` at twice the width. The releases scale by two independent
    /// multiples — width sets the channel counts, depth the C2f repeats — and n and s share a depth.
    public static let small = NFKMLXYOLOConfiguration(widths: [32, 64, 128, 256, 512],
                                                      backboneRepeats: [1, 2, 2, 1])

    /// YOLOv8m: a deeper multiple as well as a wider one, so its C2f stages repeat `[2, 4, 4, 2]`.
    public static let medium = NFKMLXYOLOConfiguration(widths: [48, 96, 192, 384, 576],
                                                       backboneRepeats: [2, 4, 4, 2])

    /// YOLOv8l: the first size at the full depth multiple, so its C2f stages repeat `[3, 6, 6, 3]`,
    /// and the first whose deepest stage is capped below the width multiple would give.
    public static let large = NFKMLXYOLOConfiguration(widths: [64, 128, 256, 512, 512],
                                                      backboneRepeats: [3, 6, 6, 3])

    /// YOLOv8x: the same depth as `large` at a wider multiple; the deepest stage is capped the same way.
    public static let extraLarge = NFKMLXYOLOConfiguration(widths: [80, 160, 320, 640, 640],
                                                           backboneRepeats: [3, 6, 6, 3])

    public static let tiny = NFKMLXYOLOConfiguration(inputResolution: 64, widths: [4, 8, 8, 8, 16],
                                                     backboneRepeats: [1, 1, 1, 1], classCount: 3,
                                                     regMax: 4, confidenceThreshold: 0, iouThreshold: 0.5)
}

/// The reference `Conv`: convolution (no bias) + BatchNorm + SiLU. Ultralytics resets every
/// BatchNorm to epsilon 1e-3 (`initialize_weights`), so that is built in here.
final class NFKYOLOConv: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d
    @ModuleInfo(key: "bn") var bn: BatchNorm

    init(inChannels: Int, outChannels: Int, kernel: Int = 1, stride: Int = 1) {
        _conv.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: outChannels,
                                    kernelSize: IntOrPair(kernel), stride: IntOrPair(stride),
                                    padding: IntOrPair(kernel / 2), bias: false)
        _bn.wrappedValue = BatchNorm(featureCount: outChannels, eps: 1e-3)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        silu(bn(conv(x)))
    }
}

/// The C2f bottleneck: two 3×3 `Conv`s, with a residual when configured and the channels allow it.
final class NFKYOLOBottleneck: Module {
    @ModuleInfo(key: "cv1") var cv1: NFKYOLOConv
    @ModuleInfo(key: "cv2") var cv2: NFKYOLOConv
    let usesResidual: Bool

    init(channels: Int, shortcut: Bool) {
        _cv1.wrappedValue = NFKYOLOConv(inChannels: channels, outChannels: channels, kernel: 3)
        _cv2.wrappedValue = NFKYOLOConv(inChannels: channels, outChannels: channels, kernel: 3)
        usesResidual = shortcut
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let out = cv2(cv1(x))
        return usesResidual ? x + out : out
    }
}

/// The reference `C2f` stage: a 1×1 split into two halves, bottlenecks chained off the second half
/// with every intermediate kept, and a 1×1 fuse of all of them.
final class NFKYOLOC2f: Module {
    @ModuleInfo(key: "cv1") var cv1: NFKYOLOConv
    @ModuleInfo(key: "cv2") var cv2: NFKYOLOConv
    @ModuleInfo(key: "m") var m: [NFKYOLOBottleneck]
    let hidden: Int

    init(inChannels: Int, outChannels: Int, repeats: Int, shortcut: Bool) {
        let half = outChannels / 2
        hidden = half
        _cv1.wrappedValue = NFKYOLOConv(inChannels: inChannels, outChannels: half * 2)
        _cv2.wrappedValue = NFKYOLOConv(inChannels: half * (2 + repeats), outChannels: outChannels)
        _m.wrappedValue = (0 ..< repeats).map { _ in NFKYOLOBottleneck(channels: half, shortcut: shortcut) }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let split = cv1(x)
        var parts = [split[0..., 0..., 0..., 0 ..< hidden],
                     split[0..., 0..., 0..., hidden ..< hidden * 2]]
        for bottleneck in m {
            parts.append(bottleneck(parts.last!))
        }
        return cv2(concatenated(parts, axis: 3))
    }
}

/// The reference `SPPF`: a 1×1 narrowing, three chained 5×5 stride-1 max pools, and a 1×1 fuse of all
/// four scales.
final class NFKYOLOSPPF: Module {
    @ModuleInfo(key: "cv1") var cv1: NFKYOLOConv
    @ModuleInfo(key: "cv2") var cv2: NFKYOLOConv

    init(channels: Int) {
        _cv1.wrappedValue = NFKYOLOConv(inChannels: channels, outChannels: channels / 2)
        _cv2.wrappedValue = NFKYOLOConv(inChannels: channels * 2, outChannels: channels)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let narrowed = cv1(x)
        let pool1 = NFKMLXResample.maxPooled(narrowed, kernel: 5, stride: 1, padding: 2)
        let pool2 = NFKMLXResample.maxPooled(pool1, kernel: 5, stride: 1, padding: 2)
        let pool3 = NFKMLXResample.maxPooled(pool2, kernel: 5, stride: 1, padding: 2)
        return cv2(concatenated([narrowed, pool1, pool2, pool3], axis: 3))
    }
}

/// One detection-head branch: two 3×3 `Conv`s and a plain 1×1 output convolution. The reference packs
/// these positionally in a Sequential; the names here are semantic, translated by the remap.
final class NFKYOLODetectBranch: Module {
    @ModuleInfo(key: "conv1") var conv1: NFKYOLOConv
    @ModuleInfo(key: "conv2") var conv2: NFKYOLOConv
    @ModuleInfo(key: "out") var out: Conv2d

    init(inChannels: Int, hidden: Int, outChannels: Int) {
        _conv1.wrappedValue = NFKYOLOConv(inChannels: inChannels, outChannels: hidden, kernel: 3)
        _conv2.wrappedValue = NFKYOLOConv(inChannels: hidden, outChannels: hidden, kernel: 3)
        _out.wrappedValue = Conv2d(inputChannels: hidden, outputChannels: outChannels, kernelSize: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        out(conv2(conv1(x)))
    }
}

/// The distribution-focal expectation: a fixed 1×1 convolution whose weights are the bin indices
/// `0...regMax-1`, applied over a softmax. It loads from the checkpoint like any other weight.
final class NFKYOLODFL: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d
    let bins: Int

    init(bins: Int) {
        _conv.wrappedValue = Conv2d(inputChannels: bins, outputChannels: 1, kernelSize: 1, bias: false)
        let indices = MLXArray((0 ..< bins).map { Float($0) }).reshaped([1, 1, 1, bins])
        _conv.wrappedValue.update(parameters: ModuleParameters.unflattened([("weight", indices)]))
        self.bins = bins
    }

    /// `[anchors, 4·bins]` distributions → `[anchors, 4]` expected distances.
    func callAsFunction(_ distribution: MLXArray) -> MLXArray {
        let anchors = distribution.shape[0]
        let perSide = softmax(distribution.reshaped([anchors, 4, bins]), axis: -1)
        let weights = conv.weight.reshaped([bins, 1])
        return matmul(perSide.reshaped([anchors * 4, bins]), weights).reshaped([anchors, 4])
    }
}

/// The v8 detection head: per-scale decoupled box and class branches plus the DFL decode.
final class NFKYOLODetect: Module {
    @ModuleInfo(key: "cv2") var cv2: [NFKYOLODetectBranch]
    @ModuleInfo(key: "cv3") var cv3: [NFKYOLODetectBranch]
    @ModuleInfo(key: "dfl") var dfl: NFKYOLODFL

    init(channels: [Int], classCount: Int, regMax: Int) {
        let boxHidden = max(16, channels[0] / 4, regMax * 4)
        let classHidden = max(channels[0], min(classCount, 100))
        _cv2.wrappedValue = channels.map { NFKYOLODetectBranch(inChannels: $0, hidden: boxHidden,
                                                               outChannels: 4 * regMax) }
        _cv3.wrappedValue = channels.map { NFKYOLODetectBranch(inChannels: $0, hidden: classHidden,
                                                               outChannels: classCount) }
        _dfl.wrappedValue = NFKYOLODFL(bins: regMax)
    }
}

/// The YOLOv8 network: CSPDarknet backbone, PAN-FPN neck, and the decoupled DFL head.
final class NFKMLXYOLONet: Module {
    @ModuleInfo(key: "conv0") var conv0: NFKYOLOConv
    @ModuleInfo(key: "conv1") var conv1: NFKYOLOConv
    @ModuleInfo(key: "c2f2") var c2f2: NFKYOLOC2f
    @ModuleInfo(key: "conv3") var conv3: NFKYOLOConv
    @ModuleInfo(key: "c2f4") var c2f4: NFKYOLOC2f
    @ModuleInfo(key: "conv5") var conv5: NFKYOLOConv
    @ModuleInfo(key: "c2f6") var c2f6: NFKYOLOC2f
    @ModuleInfo(key: "conv7") var conv7: NFKYOLOConv
    @ModuleInfo(key: "c2f8") var c2f8: NFKYOLOC2f
    @ModuleInfo(key: "sppf") var sppf: NFKYOLOSPPF
    @ModuleInfo(key: "c2f12") var c2f12: NFKYOLOC2f
    @ModuleInfo(key: "c2f15") var c2f15: NFKYOLOC2f
    @ModuleInfo(key: "conv16") var conv16: NFKYOLOConv
    @ModuleInfo(key: "c2f18") var c2f18: NFKYOLOC2f
    @ModuleInfo(key: "conv19") var conv19: NFKYOLOConv
    @ModuleInfo(key: "c2f21") var c2f21: NFKYOLOC2f
    @ModuleInfo(key: "detect") var detect: NFKYOLODetect

    let configuration: NFKMLXYOLOConfiguration

    init(_ c: NFKMLXYOLOConfiguration) {
        configuration = c
        let w = c.widths
        let r = c.backboneRepeats
        _conv0.wrappedValue = NFKYOLOConv(inChannels: 3, outChannels: w[0], kernel: 3, stride: 2)
        _conv1.wrappedValue = NFKYOLOConv(inChannels: w[0], outChannels: w[1], kernel: 3, stride: 2)
        _c2f2.wrappedValue = NFKYOLOC2f(inChannels: w[1], outChannels: w[1], repeats: r[0], shortcut: true)
        _conv3.wrappedValue = NFKYOLOConv(inChannels: w[1], outChannels: w[2], kernel: 3, stride: 2)
        _c2f4.wrappedValue = NFKYOLOC2f(inChannels: w[2], outChannels: w[2], repeats: r[1], shortcut: true)
        _conv5.wrappedValue = NFKYOLOConv(inChannels: w[2], outChannels: w[3], kernel: 3, stride: 2)
        _c2f6.wrappedValue = NFKYOLOC2f(inChannels: w[3], outChannels: w[3], repeats: r[2], shortcut: true)
        _conv7.wrappedValue = NFKYOLOConv(inChannels: w[3], outChannels: w[4], kernel: 3, stride: 2)
        _c2f8.wrappedValue = NFKYOLOC2f(inChannels: w[4], outChannels: w[4], repeats: r[3], shortcut: true)
        _sppf.wrappedValue = NFKYOLOSPPF(channels: w[4])
        _c2f12.wrappedValue = NFKYOLOC2f(inChannels: w[4] + w[3], outChannels: w[3], repeats: r[0], shortcut: false)
        _c2f15.wrappedValue = NFKYOLOC2f(inChannels: w[3] + w[2], outChannels: w[2], repeats: r[0], shortcut: false)
        _conv16.wrappedValue = NFKYOLOConv(inChannels: w[2], outChannels: w[2], kernel: 3, stride: 2)
        _c2f18.wrappedValue = NFKYOLOC2f(inChannels: w[2] + w[3], outChannels: w[3], repeats: r[0], shortcut: false)
        _conv19.wrappedValue = NFKYOLOConv(inChannels: w[3], outChannels: w[3], kernel: 3, stride: 2)
        _c2f21.wrappedValue = NFKYOLOC2f(inChannels: w[3] + w[4], outChannels: w[4], repeats: r[0], shortcut: false)
        _detect.wrappedValue = NFKYOLODetect(channels: [w[2], w[3], w[4]],
                                             classCount: c.classCount, regMax: c.regMax)
    }

    /// Backbone + neck: the three detection scales at strides 8, 16, and 32.
    func features(_ image: MLXArray) -> [MLXArray] {
        let p3Backbone = c2f4(conv3(c2f2(conv1(conv0(image)))))
        let p4Backbone = c2f6(conv5(p3Backbone))
        let p5 = sppf(c2f8(conv7(p4Backbone)))

        let p4Mid = c2f12(concatenated([NFKMLXResample.upsampleNearest(p5, scale: 2), p4Backbone], axis: 3))
        let p3Out = c2f15(concatenated([NFKMLXResample.upsampleNearest(p4Mid, scale: 2), p3Backbone], axis: 3))
        let p4Out = c2f18(concatenated([conv16(p3Out), p4Mid], axis: 3))
        let p5Out = c2f21(concatenated([conv19(p4Out), p5], axis: 3))
        return [p3Out, p4Out, p5Out]
    }

    /// How a frame was fitted into the network's input: the aspect-preserving scale and the padding
    /// that centered it. Decoded boxes come back in padded-input pixels, so mapping them to the
    /// original frame needs both.
    struct Letterbox {
        var scale: Float
        var left: Int
        var top: Int
        var originalHeight: Int
        var originalWidth: Int
    }

    /// The reference `LetterBox`: scale the frame to fit `inputResolution` with its aspect preserved,
    /// then pad it to a multiple of the coarsest stride with the reference's gray. Padding to a stride
    /// multiple rather than to a square is the reference's `auto` mode, so a wide frame runs at
    /// something like 640×384 instead of wasting a third of the input on gray. The `±0.1` offsets are
    /// the reference's own: a half-pixel split would otherwise land on a rounding tie.
    static func letterbox(_ image: MLXArray, resolution: Int, stride: Int = 32) -> (input: MLXArray, box: Letterbox) {
        let (height, width) = (image.shape[1], image.shape[2])
        let scale = min(Float(resolution) / Float(height), Float(resolution) / Float(width))
        let scaledHeight = Int((Float(height) * scale).rounded())
        let scaledWidth = Int((Float(width) * scale).rounded())
        let verticalPad = Float((resolution - scaledHeight) % stride) / 2
        let horizontalPad = Float((resolution - scaledWidth) % stride) / 2
        let top = Int((verticalPad - 0.1).rounded()), bottom = Int((verticalPad + 0.1).rounded())
        let left = Int((horizontalPad - 0.1).rounded()), right = Int((horizontalPad + 0.1).rounded())

        var resized = image
        if scaledHeight != height || scaledWidth != width {
            resized = NFKMLXResample.resizeBilinear(image, height: scaledHeight, width: scaledWidth)
        }
        let padded = MLX.padded(resized,
                                widths: [IntOrPair(0), IntOrPair((top, bottom)), IntOrPair((left, right)), IntOrPair(0)],
                                mode: .constant, value: MLXArray(Float(114) / 255))
        return (padded, Letterbox(scale: scale, left: left, top: top,
                                  originalHeight: height, originalWidth: width))
    }

    /// The reference's pre-suppression prediction tensor: `[anchors, 4 + classCount]` — box centers
    /// and sizes in pixels of the letterboxed input, then sigmoid class probabilities.
    func predictions(_ image: MLXArray) -> MLXArray {
        predictionsWithLetterbox(image).predictions
    }

    /// The prediction tensor together with the fitting that produced it, for a caller that has to map
    /// boxes back to the original frame.
    func predictionsWithLetterbox(_ image: MLXArray) -> (predictions: MLXArray, letterbox: Letterbox) {
        let batched = image.ndim == 3 ? image.reshaped([1, image.shape[0], image.shape[1], image.shape[2]]) : image
        let (input, box) = Self.letterbox(batched, resolution: configuration.inputResolution)
        let scales = features(input)

        var boxes = [MLXArray]()
        var classes = [MLXArray]()
        for (index, feature) in scales.enumerated() {
            let (gh, gw) = (feature.shape[1], feature.shape[2])
            let stride = Float(input.shape[1]) / Float(gh)
            let count = gh * gw
            let distribution = detect.cv2[index](feature).reshaped([count, 4 * configuration.regMax])
            let distances = detect.dfl(distribution)                       // [count, 4] ltrb in strides

            var anchorX = [Float](repeating: 0, count: count)
            var anchorY = [Float](repeating: 0, count: count)
            for y in 0 ..< gh {
                for x in 0 ..< gw {
                    anchorX[y * gw + x] = Float(x) + 0.5
                    anchorY[y * gw + x] = Float(y) + 0.5
                }
            }
            let ax = anchorX.withUnsafeBufferPointer { MLXArray($0, [count, 1]) }
            let ay = anchorY.withUnsafeBufferPointer { MLXArray($0, [count, 1]) }
            let left = distances[0..., 0 ..< 1], top = distances[0..., 1 ..< 2]
            let right = distances[0..., 2 ..< 3], bottom = distances[0..., 3 ..< 4]
            let centerX = (ax - left + ax + right) / 2
            let centerY = (ay - top + ay + bottom) / 2
            let width = left + right
            let height = top + bottom
            boxes.append(concatenated([centerX, centerY, width, height], axis: 1) * stride)
            classes.append(sigmoid(detect.cv3[index](feature).reshaped([count, configuration.classCount])))
        }
        let predictions = concatenated([concatenated(boxes, axis: 0), concatenated(classes, axis: 0)], axis: 1)
        return (predictions, box)
    }

    /// Detects objects in a bridged image `[H, W, 3]` (`0...1`): predict, decode to normalized boxes,
    /// and non-max suppress. `labels` names classes when available.
    func detect(_ image: MLXArray, labels: [String]?) -> [NFKDetection] {
        let (predicted, box) = predictionsWithLetterbox(image.ndim == 3
            ? image.reshaped([1, image.shape[0], image.shape[1], image.shape[2]]) : image)
        eval(predicted)
        let values = predicted.asArray(Float.self)
        let classCount = configuration.classCount
        let rowWidth = 4 + classCount

        var candidates: [(classIndex: Int, confidence: Float, rect: CGRect)] = []
        for anchor in 0 ..< predicted.shape[0] {
            let base = anchor * rowWidth
            var bestClass = 0
            var bestScore: Float = 0
            for k in 0 ..< classCount {
                let score = values[base + 4 + k]
                if score > bestScore {
                    bestScore = score
                    bestClass = k
                }
            }
            if bestScore < configuration.confidenceThreshold {
                continue
            }
            // Undo the fitting: drop the padding, undo the scale, and normalize against the frame the
            // caller passed in — not against the padded input it happened to be fitted into.
            let cx = (values[base] - Float(box.left)) / box.scale / Float(box.originalWidth)
            let cy = (values[base + 1] - Float(box.top)) / box.scale / Float(box.originalHeight)
            let w = values[base + 2] / box.scale / Float(box.originalWidth)
            let h = values[base + 3] / box.scale / Float(box.originalHeight)
            // Clip the corners, not the extent: capping the width alone leaves a box that crosses an
            // edge the same size and merely shifts it inward, which the reference never does.
            let minX = min(max(cx - w / 2, 0), 1), maxX = min(max(cx + w / 2, 0), 1)
            let minY = min(max(cy - h / 2, 0), 1), maxY = min(max(cy + h / 2, 0), 1)
            let rect = CGRect(x: CGFloat(minX), y: CGFloat(minY),
                              width: CGFloat(maxX - minX), height: CGFloat(maxY - minY))
            candidates.append((bestClass, bestScore, rect))
        }

        return NFKMLXYOLONet.nonMaxSuppress(candidates, iouThreshold: configuration.iouThreshold).map { candidate in
            let label = labels.flatMap { candidate.classIndex < $0.count ? $0[candidate.classIndex] : nil }
            return NFKDetection(label: label, classIndex: candidate.classIndex,
                                confidence: Double(candidate.confidence), boundingBox: candidate.rect)
        }
    }

    /// Greedy per-class non-max suppression: keep the highest-confidence box, drop others of the same
    /// class that overlap it beyond `iouThreshold`.
    static func nonMaxSuppress(_ candidates: [(classIndex: Int, confidence: Float, rect: CGRect)],
                               iouThreshold: Float) -> [(classIndex: Int, confidence: Float, rect: CGRect)] {
        let sorted = candidates.sorted { $0.confidence > $1.confidence }
        var kept: [(classIndex: Int, confidence: Float, rect: CGRect)] = []
        for candidate in sorted {
            let overlaps = kept.contains { $0.classIndex == candidate.classIndex
                && intersectionOverUnion($0.rect, candidate.rect) > iouThreshold }
            if !overlaps {
                kept.append(candidate)
            }
        }
        return kept
    }

    static func intersectionOverUnion(_ a: CGRect, _ b: CGRect) -> Float {
        let intersection = a.intersection(b)
        if intersection.isNull {
            return 0
        }
        let intersectionArea = intersection.width * intersection.height
        let unionArea = a.width * a.height + b.width * b.height - intersectionArea
        return unionArea > 0 ? Float(intersectionArea / unionArea) : 0
    }
}

/// Holds the network and class labels for capture in the backend's `@Sendable` closure.
private final class NFKYOLOHolder: @unchecked Sendable {
    let net: NFKMLXYOLONet
    let labels: [String]?
    init(_ net: NFKMLXYOLONet, labels: [String]?) {
        self.net = net
        self.labels = labels
    }
}

/// An object-detection backend: an image under `NFKInputImage` produces detections under
/// `NFKOutputDetections`.
@objc(NFKMLXYOLOBackend)
public final class NFKMLXYOLOBackend: NSObject, NFKInferenceBackend {

    private let holder: NFKYOLOHolder
    private let identifier: String

    init(net: NFKMLXYOLONet, identifier: String, labels: [String]?) {
        holder = NFKYOLOHolder(net, labels: labels)
        self.identifier = identifier
        super.init()
    }

    @objc public var isReady: Bool { true }
    @objc public var backendIdentifier: String { identifier }

    @objc(runInferenceForRequest:error:)
    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        let job = submitInferenceJob(for: request)
        let semaphore = DispatchSemaphore(value: 0)
        job.completionHandler = { _ in semaphore.signal() }
        semaphore.wait()
        if let result = job.result {
            return result
        }
        if let error = job.error {
            throw error
        }
        throw NFKMLXError.noOutput
    }

    @objc(submitInferenceJobForRequest:)
    public func submitInferenceJob(for request: NFKInferenceRequest) -> NFKInferenceJob {
        let job = NFKInferenceJob()
        let holder = self.holder
        Task.detached(priority: .userInitiated) {
            do {
                guard let value = request.input(forKey: NFKInputImage) else {
                    throw NFKMLXError.unsupportedInput
                }
                let image = try NFKMLXImageBridge.tensor(from: value, channels: 3, colorSpace: CGColorSpaceCreateDeviceRGB())
                let detections = holder.net.detect(image, labels: holder.labels)
                job.finish(with: NFKInferenceResult(outputs: [NFKOutputDetections: detections]))
            } catch {
                job.finish(withError: error as NSError)
            }
        }
        return job
    }
}

/// YOLOv8 object detection as an InferKit backend, and its registration for the Objective-C path.
///
/// `NFKMLXYOLONet` is the reference detector. Random weights run (proving the pipeline); the released
/// `yolov8n.pt`, converted to **safetensors**, detects accurately.
/// Which released YOLOv8 size a backend is built for.
@objc(NFKMLXYOLOVariant)
public enum NFKMLXYOLOVariant: Int {
    case nano
    case small
    case medium
    case large
    case extraLarge
}

@objc(NFKMLXYOLO)
public final class NFKMLXYOLO: NSObject {

    /// The registry name the model builds under.
    @objc public static let modelName = "yolo"

    /// Builds a detection backend directly from optional local weights — no registry required. A nil
    /// `weightsURL` builds random weights (`isReady` is true). `labels` names classes when available.
    /// Run inference off the render thread.
    @objc(backendWithWeightsURL:labels:error:)
    public static func backend(weightsURL: URL?, labels: [String]?) throws -> any NFKInferenceBackend {
        try backend(variant: .nano, weightsURL: weightsURL, labels: labels)
    }

    /// Builds at one of the released sizes. A checkpoint only fits the size it was trained as.
    @objc(backendWithVariant:weightsURL:labels:error:)
    public static func backend(variant: NFKMLXYOLOVariant, weightsURL: URL?, labels: [String]?) throws -> any NFKInferenceBackend {
        let geometry: NFKMLXYOLOConfiguration
        switch variant {
        case .nano: geometry = .base
        case .small: geometry = .small
        case .medium: geometry = .medium
        case .large: geometry = .large
        case .extraLarge: geometry = .extraLarge
        }
        let net = NFKMLXYOLONet(geometry)
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        net.train(false)
        return NFKMLXYOLOBackend(net: net, identifier: modelName, labels: labels)
    }

    /// Downloads the checkpoint from Hugging Face, then builds the backend — no registry required.
    /// Blocking on the network; run off the render thread.
    @objc(backendWithRepo:weightsPath:revision:cacheDirectoryURL:labels:error:)
    public static func backend(repo: String, weightsPath: String, revision: String?,
                               cacheDirectoryURL: URL?, labels: [String]?) throws -> any NFKInferenceBackend {
        let url = try NFKMLXDownload.weightsURL(repo: repo, weightsPath: weightsPath, revision: revision, cacheDirectoryURL: cacheDirectoryURL)
        return try backend(weightsURL: url, labels: labels)
    }

    /// The asynchronous form of the download factory: downloads on a background queue, then builds and
    /// delivers the backend (or an error) to `completionHandler`.
    @objc(backendWithRepo:weightsPath:revision:cacheDirectoryURL:labels:completionHandler:)
    public static func backend(repo: String, weightsPath: String, revision: String?,
                               cacheDirectoryURL: URL?, labels: [String]?,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXDownload.backend(repo: repo, weightsPath: weightsPath, revision: revision,
                               cacheDirectoryURL: cacheDirectoryURL,
                               build: { try backend(weightsURL: $0, labels: labels) },
                               completionHandler: completionHandler)
    }

    /// The download factory at a chosen size, so small/medium/large/extraLarge are reachable over the
    /// network, not only from a local file. Blocking; run off the render thread.
    @objc(backendWithVariant:repo:weightsPath:revision:cacheDirectoryURL:labels:error:)
    public static func backend(variant: NFKMLXYOLOVariant, repo: String, weightsPath: String,
                               revision: String?, cacheDirectoryURL: URL?, labels: [String]?) throws -> any NFKInferenceBackend {
        let url = try NFKMLXDownload.weightsURL(repo: repo, weightsPath: weightsPath, revision: revision, cacheDirectoryURL: cacheDirectoryURL)
        return try backend(variant: variant, weightsURL: url, labels: labels)
    }

    /// The asynchronous form of the variant download factory.
    @objc(backendWithVariant:repo:weightsPath:revision:cacheDirectoryURL:labels:completionHandler:)
    public static func backend(variant: NFKMLXYOLOVariant, repo: String, weightsPath: String,
                               revision: String?, cacheDirectoryURL: URL?, labels: [String]?,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXDownload.backend(repo: repo, weightsPath: weightsPath, revision: revision,
                               cacheDirectoryURL: cacheDirectoryURL,
                               build: { try backend(variant: variant, weightsURL: $0, labels: labels) },
                               completionHandler: completionHandler)
    }

    /// Registers YOLO (`yolo`) with `NFKMLXModelRegistry`, delegating to `backend(weightsURL:labels:)`.
    /// The registered backend has no class labels; a caller that wants names builds through the factory.
    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName) { weightsURL in
            try backend(weightsURL: weightsURL, labels: nil)
        }
    }

    /// The reference module list's index for each named stage (indices 10/13 are upsamples and
    /// 11/14/17/20 concatenations, which carry no parameters).
    private static let stageNames: [Int: String] = [
        0: "conv0", 1: "conv1", 2: "c2f2", 3: "conv3", 4: "c2f4", 5: "conv5", 6: "c2f6",
        7: "conv7", 8: "c2f8", 9: "sppf", 12: "c2f12", 15: "c2f15", 16: "conv16",
        18: "c2f18", 19: "conv19", 21: "c2f21", 22: "detect"]

    /// Maps the reference's positional names onto the module's semantic ones: `model.N.` becomes the
    /// stage's name, and the head branches' inner Sequential (`cv2.i.0/1/2`) becomes
    /// `conv1`/`conv2`/`out` (MLX's `update(parameters:)` parses a numeric key as an array index, so
    /// module keys cannot stay positional).
    static func remapReferenceKey(_ key: String) -> String {
        var key = key
        if key.hasPrefix("model.") {
            let rest = key.dropFirst("model.".count)
            guard let dot = rest.firstIndex(of: "."), let index = Int(rest[..<dot]),
                  let name = stageNames[index] else { return key }
            key = name + String(rest[dot...])
        }
        for branch in ["cv2", "cv3"] where key.hasPrefix("detect.\(branch).") {
            let prefix = "detect.\(branch)."
            let sub = key.dropFirst(prefix.count)
            guard let dot = sub.firstIndex(of: "."), let scale = Int(sub[..<dot]) else { continue }
            let tail = sub[sub.index(after: dot)...]
            guard let tailDot = tail.firstIndex(of: "."), let position = Int(tail[..<tailDot]) else { continue }
            let names = ["conv1", "conv2", "out"]
            key = prefix + "\(scale)." + names[position] + tail[tailDot...]
        }
        return key
    }

    /// Loads a safetensors checkpoint into `net`, remapping the reference's names and transposing 4-D
    /// convolution weights from PyTorch's `[out, in, kH, kW]` to MLX's channels-last
    /// `[out, kH, kW, in]`.
    static func loadWeights(into net: NFKMLXYOLONet, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let raw = checkpoint.arrays
        let mapped = raw.map { key, value in
            (remapReferenceKey(key), checkpoint.needsConvTranspose && value.ndim == 4 ? value.transposed(0, 2, 3, 1) : value)
        }
        try NFKMLXWeights.apply(mapped, to: net)
    }
}
