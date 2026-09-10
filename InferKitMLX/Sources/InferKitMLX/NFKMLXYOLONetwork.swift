//
//  NFKMLXYOLONetwork.swift
//  InferKitMLX
//

import Foundation
import CoreGraphics
import InferKit
import MLX
import MLXNN

// The network the graph rows build, and the detection head the later generations share. The modules
// live in one `[Module]` array so a released checkpoint's `model.<N>.…` keys land on array index N.

/// The class branch the generations after YOLOv9 use: two depthwise-plus-pointwise pairs and a 1×1
/// output convolution. The reference nests these in Sequentials, so the loader's remap translates
/// `cv3.<scale>.0.0` and its siblings onto these names.
final class NFKYOLOClassBranch: Module {
    @ModuleInfo(key: "dw0") var dw0: NFKYOLOConv
    @ModuleInfo(key: "pw0") var pw0: NFKYOLOConv
    @ModuleInfo(key: "dw1") var dw1: NFKYOLOConv
    @ModuleInfo(key: "pw1") var pw1: NFKYOLOConv
    @ModuleInfo(key: "out") var out: Conv2d

    init(inChannels: Int, hidden: Int, classCount: Int) {
        _dw0.wrappedValue = NFKYOLOConv(inChannels: inChannels, outChannels: inChannels, kernel: 3,
                                        groups: inChannels)
        _pw0.wrappedValue = NFKYOLOConv(inChannels: inChannels, outChannels: hidden)
        _dw1.wrappedValue = NFKYOLOConv(inChannels: hidden, outChannels: hidden, kernel: 3, groups: hidden)
        _pw1.wrappedValue = NFKYOLOConv(inChannels: hidden, outChannels: hidden)
        _out.wrappedValue = Conv2d(inputChannels: hidden, outputChannels: classCount, kernelSize: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { out(pw1(dw1(pw0(dw0(x))))) }
}

/// One decoupled head: a box branch and a class branch per scale, and the distribution-focal decode.
/// `endToEnd` adds the one-to-one branches the NMS-free generations predict from.
final class NFKYOLOGenerationDetect: Module {
    @ModuleInfo(key: "cv2") var cv2: [NFKYOLODetectBranch]
    @ModuleInfo(key: "cv3") var cv3: [Module]
    @ModuleInfo(key: "one2one_cv2") var oneToOneBox: [NFKYOLODetectBranch]?
    @ModuleInfo(key: "one2one_cv3") var oneToOneClass: [Module]?
    @ModuleInfo(key: "dfl") var dfl: NFKYOLODFL?

    let classCount: Int
    let regMax: Int
    let endToEnd: Bool

    init(channels: [Int], classCount: Int, regMax: Int, legacy: Bool, endToEnd: Bool) {
        self.classCount = classCount
        self.regMax = regMax
        self.endToEnd = endToEnd
        let boxHidden = max(16, channels[0] / 4, regMax * 4)
        let classHidden = max(channels[0], min(classCount, 100))

        func boxBranches() -> [NFKYOLODetectBranch] {
            channels.map { NFKYOLODetectBranch(inChannels: $0, hidden: boxHidden, outChannels: 4 * regMax) }
        }
        func classBranches() -> [Module] {
            channels.map { input -> Module in
                legacy ? NFKYOLODetectBranch(inChannels: input, hidden: classHidden, outChannels: classCount)
                       : NFKYOLOClassBranch(inChannels: input, hidden: classHidden, classCount: classCount)
            }
        }
        _cv2.wrappedValue = boxBranches()
        _cv3.wrappedValue = classBranches()
        _oneToOneBox.wrappedValue = endToEnd ? boxBranches() : nil
        _oneToOneClass.wrappedValue = endToEnd ? classBranches() : nil
        // With one bin the expectation is the value itself, so the reference builds no DFL module.
        _dfl.wrappedValue = regMax > 1 ? NFKYOLODFL(bins: regMax) : nil
    }

    static func runClass(_ branch: Module, _ x: MLXArray) -> MLXArray {
        switch branch {
        case let modern as NFKYOLOClassBranch: return modern(x)
        case let legacy as NFKYOLODetectBranch: return legacy(x)
        default: return x
        }
    }

    /// The raw per-scale box and class maps the inference path decodes. `oneToOne` selects the
    /// end-to-end branch, which is what the NMS-free generations predict from.
    func maps(_ features: [MLXArray], oneToOne: Bool) -> (boxes: [MLXArray], scores: [MLXArray]) {
        let boxBranches = oneToOne ? (oneToOneBox ?? cv2) : cv2
        let classBranches = oneToOne ? (oneToOneClass ?? cv3) : cv3
        var boxes = [MLXArray]()
        var scores = [MLXArray]()
        for (index, feature) in features.enumerated() {
            boxes.append(boxBranches[index](feature))
            scores.append(Self.runClass(classBranches[index], feature))
        }
        return (boxes, scores)
    }
}

/// A YOLO generation built from its graph rows. Input `[1, H, W, 3]` in `0...1`; `predictions` returns
/// `[anchors, 4 + classes]` with the boxes in pixels and the class scores already through a sigmoid,
/// which is the reference's own pre-suppression tensor.
final class NFKMLXYOLOGenerationNet: Module {
    @ModuleInfo(key: "model") var model: [Module]

    let nodes: [NFKYOLONode]
    let generation: NFKMLXYOLOGeneration
    let classCount: Int
    let strides: [Int] = [8, 16, 32]

    /// The head is the last row, which every graph here ends with.
    var head: NFKYOLOGenerationDetect { model[model.count - 1] as! NFKYOLOGenerationDetect }

    init(generation: NFKMLXYOLOGeneration, letter: NFKMLXYOLOScaleLetter, classCount: Int = 80) {
        self.generation = generation
        self.classCount = classCount
        let rows: [NFKYOLONode]
        switch generation {
        case .v9: rows = NFKYOLOGraphs.yolo9(letter)
        case .v10: rows = NFKYOLOGraphs.yolo10(letter)
        case .v11: rows = NFKYOLOGraphs.yolo11(letter)
        case .v12: rows = NFKYOLOGraphs.yolo12(letter)
        case .v26: rows = NFKYOLOGraphs.yolo26(letter)
        }
        nodes = rows

        let scale = NFKYOLOGraphs.scale(generation, letter)
        var widths = [Int]()                                    // the channel count each row outputs
        var built = [Module]()
        var input = 3
        for (index, node) in rows.enumerated() {
            let sources = node.from.map { $0 == -1 ? index - 1 : $0 }
            let inChannels = sources.count == 1 ? (sources[0] < 0 ? 3 : widths[sources[0]])
                                                : sources.reduce(0) { $0 + widths[$1] }
            let repeats = node.repeats > 1 ? max(Int((Float(node.repeats) * scale.depth).rounded()), 1)
                                           : node.repeats
            let (module, outChannels) = NFKMLXYOLOGenerationNet.build(
                node.kind, inChannels: inChannels, repeats: repeats, scale: scale, letter: letter,
                generation: generation, classCount: classCount,
                // Only the head reads several earlier rows by absolute index; every other row's
                // `from` may carry the -1 that means "the previous row".
                headChannels: node.from.allSatisfy { $0 >= 0 } ? node.from.map { widths[$0] } : [])
            built.append(module)
            widths.append(outChannels)
            input = outChannels
        }
        _ = input
        _model.wrappedValue = built
    }

    private static func scaled(_ value: Int, _ scale: NFKYOLOScale) -> Int {
        NFKYOLOGraphs.makeDivisible(Float(min(value, scale.maxChannels)) * scale.width)
    }

    private static func build(_ kind: NFKYOLONode.Kind, inChannels: Int, repeats: Int,
                              scale: NFKYOLOScale, letter: NFKMLXYOLOScaleLetter,
                              generation: NFKMLXYOLOGeneration, classCount: Int,
                              headChannels: [Int]) -> (Module, Int) {
        switch kind {
        case let .conv(out, kernel, stride):
            let width = scaled(out, scale)
            return (NFKYOLOConv(inChannels: inChannels, outChannels: width, kernel: kernel, stride: stride), width)
        case let .c2f(out, shortcut):
            let width = scaled(out, scale)
            return (NFKYOLOC2f(inChannels: inChannels, outChannels: width, repeats: repeats,
                               shortcut: shortcut), width)
        case let .c3k2(out, c3k, expansion, shortcut):
            let width = scaled(out, scale)
            let useC3k = c3k || NFKYOLOGraphs.usesC3k(letter)
            return (NFKYOLOC3k2(inChannels: inChannels, outChannels: width, repeats: repeats,
                                c3k: useC3k, expansion: expansion, shortcut: shortcut), width)
        case let .c3k2Attention(out, expansion):
            let width = scaled(out, scale)
            return (NFKYOLOC3k2Attention(inChannels: inChannels, outChannels: width, repeats: repeats,
                                         expansion: expansion), width)
        case let .a2c2f(out, areaAttention, area):
            let width = scaled(out, scale)
            guard areaAttention else {
                return (NFKYOLOA2C2fC3k(inChannels: inChannels, outChannels: width, repeats: repeats), width)
            }
            let residual = NFKYOLOGraphs.usesA2C2fResidual(letter)
            return (NFKYOLOA2C2f(inChannels: inChannels, outChannels: width, repeats: repeats,
                                 areaAttention: areaAttention, area: area, residual: residual,
                                 mlpRatio: residual ? 1.2 : 2.0), width)
        case let .c2fCIB(out, shortcut, largeKernel):
            let width = scaled(out, scale)
            return (NFKYOLOC2fCIB(inChannels: inChannels, outChannels: width, repeats: repeats,
                                  shortcut: shortcut, largeKernel: largeKernel), width)
        case let .sppf(out):
            let width = scaled(out, scale)
            return (NFKYOLOSPPF(channels: width), width)
        case let .sppfShortcut(out):
            let width = scaled(out, scale)
            return (NFKYOLOSPPFShortcut(inChannels: inChannels, outChannels: width), width)
        case let .sppelan(out, mid):
            return (NFKYOLOSPPELAN(inChannels: inChannels, outChannels: out, midChannels: mid), out)
        case let .c2psa(out):
            let width = scaled(out, scale)
            return (NFKYOLOC2PSA(channels: width, repeats: repeats), width)
        case let .psa(out):
            let width = scaled(out, scale)
            return (NFKYOLOPSA(channels: width), width)
        case let .scDown(out, kernel, stride):
            let width = scaled(out, scale)
            return (NFKYOLOSCDown(inChannels: inChannels, outChannels: width, kernel: kernel,
                                  stride: stride), width)
        case let .aConv(out):
            return (NFKYOLOAConv(inChannels: inChannels, outChannels: out), out)
        case let .aDown(out):
            return (NFKYOLOADown(inChannels: inChannels, outChannels: out), out)
        case let .elan1(out, mid, branch):
            return (NFKYOLOELAN1(inChannels: inChannels, outChannels: out, midChannels: mid,
                                 branchChannels: branch), out)
        case let .repNCSPELAN4(out, mid, branch, innerRepeats):
            return (NFKYOLORepNCSPELAN4(inChannels: inChannels, outChannels: out, midChannels: mid,
                                        branchChannels: branch, repeats: innerRepeats), out)
        case .upsample, .concat, .identity:
            return (Module(), inChannels)
        case let .cbLinear(outs):
            return (NFKYOLOCBLinear(inChannels: inChannels, widths: outs), outs.reduce(0, +))
        case .cbFuse:
            // The fused row carries the width of its last source, which the caller states.
            return (Module(), inChannels)
        case .detect:
            let head = NFKYOLOGenerationDetect(
                channels: headChannels, classCount: classCount,
                regMax: NFKYOLOGraphs.regMax(generation),
                legacy: NFKYOLOGraphs.usesLegacyHead(generation),
                endToEnd: NFKYOLOGraphs.isEndToEnd(generation))
            return (head, classCount)
        }
    }
}

/// YOLO26's `SPPF`: the narrowing convolution has no activation and the fused result adds the input.
final class NFKYOLOSPPFShortcut: Module {
    @ModuleInfo(key: "cv1") var cv1: NFKYOLOConv
    @ModuleInfo(key: "cv2") var cv2: NFKYOLOConv
    private let usesResidual: Bool

    init(inChannels: Int, outChannels: Int, pools: Int = 3) {
        let hidden = inChannels / 2
        _cv1.wrappedValue = NFKYOLOConv(inChannels: inChannels, outChannels: hidden, activates: false)
        _cv2.wrappedValue = NFKYOLOConv(inChannels: hidden * (pools + 1), outChannels: outChannels)
        usesResidual = inChannels == outChannels
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var parts = [cv1(x)]
        for _ in 0 ..< 3 {
            parts.append(NFKMLXResample.maxPooled(parts[parts.count - 1], kernel: IntOrPair(5),
                                                  stride: IntOrPair(1), padding: IntOrPair(2)))
        }
        let fused = cv2(concatenated(parts, axis: 3))
        return usesResidual ? fused + x : fused
    }
}

extension NFKMLXYOLOGenerationNet {

    /// Runs the graph and returns the three head inputs, finest scale first.
    ///
    /// A row holds a list rather than one tensor, because `CBLinear` splits its output into several
    /// and the programmable-gradient branch of YOLOv9e reads one split per source.
    func features(_ image: MLXArray) -> [MLXArray] {
        var outputs = [[MLXArray]]()
        for (index, node) in nodes.enumerated() {
            let sources = node.from.map { $0 == -1 ? index - 1 : $0 }
            let value: [MLXArray]
            switch node.kind {
            case .identity:
                value = index == 0 ? [image] : outputs[sources[0]]
            case .upsample:
                value = [NFKMLXResample.upsampleNearest(outputs[sources[0]][0], scale: 2)]
            case .concat:
                value = [concatenated(sources.map { outputs[$0][0] }, axis: 3)]
            case .detect:
                return sources.map { outputs[$0][0] }
            case .cbLinear:
                let input = index == 0 ? image : outputs[sources[0]][0]
                value = (model[index] as? NFKYOLOCBLinear)?(input) ?? [input]
            case let .cbFuse(indices):
                value = [NFKMLXYOLOGenerationNet.fuse(sources.map { outputs[$0] }, indices: indices)]
            default:
                let input = index == 0 ? image : outputs[sources[0]][0]
                value = [NFKMLXYOLOGenerationNet.run(model[index], input)]
            }
            outputs.append(value)
        }
        return []
    }

    /// `CBFuse`: every source but the last contributes the split the row names, resized to the last
    /// source's extent, and the sum of all of them is the row's output.
    static func fuse(_ sources: [[MLXArray]], indices: [Int]) -> MLXArray {
        guard let target = sources.last?.first else { return MLXArray(0) }
        var total = target
        for (position, source) in sources.dropLast().enumerated() {
            let index = position < indices.count ? indices[position] : 0
            guard index < source.count else { continue }
            total = total + NFKMLXResample.resizeNearest(source[index], height: target.shape[1],
                                                          width: target.shape[2])
        }
        return total
    }

    static func run(_ module: Module, _ x: MLXArray) -> MLXArray {
        switch module {
        case let block as NFKYOLOConv: return block(x)
        case let block as NFKYOLOC2f: return block(x)
        case let block as NFKYOLOC3k2: return block(x)
        case let block as NFKYOLOC3k2Attention: return block(x)
        case let block as NFKYOLOA2C2f: return block(x)
        case let block as NFKYOLOA2C2fC3k: return block(x)
        case let block as NFKYOLOC2fCIB: return block(x)
        case let block as NFKYOLOSPPF: return block(x)
        case let block as NFKYOLOSPPFShortcut: return block(x)
        case let block as NFKYOLOSPPELAN: return block(x)
        case let block as NFKYOLOC2PSA: return block(x)
        case let block as NFKYOLOPSA: return block(x)
        case let block as NFKYOLOSCDown: return block(x)
        case let block as NFKYOLOAConv: return block(x)
        case let block as NFKYOLOADown: return block(x)
        case let block as NFKYOLOELAN1: return block(x)
        case let block as NFKYOLORepNCSPELAN4: return block(x)
        default: return x
        }
    }

    /// The reference's pre-suppression tensor: `[anchors, 4 + classes]`, boxes in pixels and the class
    /// scores already through a sigmoid. The end-to-end generations decode corner boxes rather than
    /// center-and-size, and predict from the one-to-one branch.
    func predictions(_ image: MLXArray) -> MLXArray {
        let scales = features(image)
        let endToEnd = head.endToEnd
        let (boxMaps, scoreMaps) = head.maps(scales, oneToOne: endToEnd)

        var distributions = [MLXArray]()
        var scores = [MLXArray]()
        var anchors = [MLXArray]()
        var strideValues = [Float]()
        for (index, map) in boxMaps.enumerated() {
            let (height, width) = (map.shape[1], map.shape[2])
            distributions.append(map.reshaped([height * width, map.shape[3]]))
            scores.append(scoreMaps[index].reshaped([height * width, classCount]))
            anchors.append(NFKMLXYOLOGenerationNet.anchorGrid(height: height, width: width))
            strideValues.append(contentsOf: Array(repeating: Float(strides[index]), count: height * width))
        }
        let distribution = concatenated(distributions, axis: 0)
        let anchorPoints = concatenated(anchors, axis: 0)
        let stride = MLXArray(strideValues).reshaped([strideValues.count, 1])

        // With one bin the distribution is already the distance; otherwise it is the softmax
        // expectation over the bins.
        let distances = head.regMax > 1 ? (head.dfl?(distribution) ?? distribution) : distribution
        let leftTop = distances[0..., 0 ..< 2]
        let rightBottom = distances[0..., 2 ..< 4]
        let topLeft = anchorPoints - leftTop
        let bottomRight = anchorPoints + rightBottom
        let box = endToEnd ? concatenated([topLeft, bottomRight], axis: 1)
                           : concatenated([(topLeft + bottomRight) / 2, bottomRight - topLeft], axis: 1)
        return concatenated([box * stride, sigmoid(concatenated(scores, axis: 0))], axis: 1)
    }

    /// The reference's `make_anchors` at the default half-cell offset: one point per cell, x before y.
    static func anchorGrid(height: Int, width: Int) -> MLXArray {
        var values = [Float]()
        values.reserveCapacity(height * width * 2)
        for y in 0 ..< height {
            for x in 0 ..< width {
                values.append(Float(x) + 0.5)
                values.append(Float(y) + 0.5)
            }
        }
        return MLXArray(values).reshaped([height * width, 2])
    }
}

extension NFKMLXYOLOGenerationNet {

    /// The output of one graph row, for a parity test that localizes a mismatch to a stage.
    func stageOutput(_ image: MLXArray, at layer: Int) -> MLXArray {
        var outputs = [[MLXArray]]()
        for (index, node) in nodes.enumerated() {
            let sources = node.from.map { $0 == -1 ? index - 1 : $0 }
            let value: [MLXArray]
            switch node.kind {
            case .identity:
                value = index == 0 ? [image] : outputs[sources[0]]
            case .upsample:
                value = [NFKMLXResample.upsampleNearest(outputs[sources[0]][0], scale: 2)]
            case .concat:
                value = [concatenated(sources.map { outputs[$0][0] }, axis: 3)]
            case .detect:
                return outputs[layer][0]
            case .cbLinear:
                let input = index == 0 ? image : outputs[sources[0]][0]
                value = (model[index] as? NFKYOLOCBLinear)?(input) ?? [input]
            case let .cbFuse(indices):
                value = [NFKMLXYOLOGenerationNet.fuse(sources.map { outputs[$0] }, indices: indices)]
            default:
                let input = index == 0 ? image : outputs[sources[0]][0]
                value = [NFKMLXYOLOGenerationNet.run(model[index], input)]
            }
            outputs.append(value)
            if index == layer { return value[0] }
        }
        return outputs[layer][0]
    }

    static func runRow(_ module: Module, _ x: MLXArray) -> MLXArray { run(module, x) }
}

extension NFKMLXYOLOGenerationNet {

    /// The consumer path: a frame in, `NFKDetection`s out with boxes normalized to the caller's own
    /// frame. The fitting is the v8 port's, which is the reference's `LetterBox`. The end-to-end
    /// generations skip suppression, because their one-to-one branch is trained to emit one box per
    /// object; the others run the reference's greedy per-class pass.
    func detect(_ image: MLXArray, labels: [String]?, confidenceThreshold: Float = 0.25,
                iouThreshold: Float = 0.7, maxDetections: Int = 300) -> [NFKDetection] {
        let batched = image.ndim == 3
            ? image.reshaped([1, image.shape[0], image.shape[1], image.shape[2]]) : image
        let (input, box) = NFKMLXYOLONet.letterbox(batched, resolution: 640)
        let predicted = predictions(input)
        eval(predicted)
        let values = predicted.asArray(Float.self)
        let rowWidth = 4 + classCount
        let endToEnd = head.endToEnd

        func normalizedX(_ value: Float) -> Float {
            min(max((value - Float(box.left)) / box.scale / Float(box.originalWidth), 0), 1)
        }
        func normalizedY(_ value: Float) -> Float {
            min(max((value - Float(box.top)) / box.scale / Float(box.originalHeight), 0), 1)
        }
        /// The anchor's box, normalized to the caller's frame. The end-to-end head decodes corners;
        /// the others decode a center and an extent.
        func rect(at base: Int) -> CGRect {
            let corners: (Float, Float, Float, Float)
            if endToEnd {
                corners = (values[base], values[base + 1], values[base + 2], values[base + 3])
            } else {
                let (cx, cy, w, h) = (values[base], values[base + 1], values[base + 2], values[base + 3])
                corners = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
            }
            let minX = normalizedX(corners.0), minY = normalizedY(corners.1)
            let maxX = normalizedX(corners.2), maxY = normalizedY(corners.3)
            return CGRect(x: CGFloat(minX), y: CGFloat(minY),
                          width: CGFloat(maxX - minX), height: CGFloat(maxY - minY))
        }

        var candidates: [(classIndex: Int, confidence: Float, rect: CGRect)] = []
        if endToEnd {
            // The end-to-end head selects over anchor-and-class PAIRS: its `postprocess` flattens the
            // scores and takes the highest `maxDetections`, so one anchor can be reported twice under
            // two classes. Taking the best class per anchor would silently drop the second.
            for anchor in 0 ..< predicted.shape[0] {
                let base = anchor * rowWidth
                for k in 0 ..< classCount where values[base + 4 + k] >= confidenceThreshold {
                    candidates.append((k, values[base + 4 + k], rect(at: base)))
                }
            }
        } else {
            for anchor in 0 ..< predicted.shape[0] {
                let base = anchor * rowWidth
                var bestClass = 0
                var bestScore: Float = 0
                for k in 0 ..< classCount where values[base + 4 + k] > bestScore {
                    bestScore = values[base + 4 + k]
                    bestClass = k
                }
                guard bestScore >= confidenceThreshold else { continue }
                candidates.append((bestClass, bestScore, rect(at: base)))
            }
        }

        let kept = endToEnd
            ? Array(candidates.sorted { $0.confidence > $1.confidence }.prefix(maxDetections))
            : NFKMLXYOLONet.nonMaxSuppress(candidates, iouThreshold: iouThreshold)
        return kept.map { candidate in
            let label = labels.flatMap { candidate.classIndex < $0.count ? $0[candidate.classIndex] : nil }
            return NFKDetection(label: label, classIndex: candidate.classIndex,
                                confidence: Double(candidate.confidence), boundingBox: candidate.rect)
        }
    }
}

/// The detection backend for the generations after v8: an image under `NFKInputImage` produces
/// detections under `NFKOutputDetections`.
@objc(NFKMLXYOLOGenerationBackend)
public final class NFKMLXYOLOGenerationBackend: NSObject, NFKInferenceBackend {

    private let holder: NFKYOLOGenerationBackendHolder
    private let identifier: String

    init(net: NFKMLXYOLOGenerationNet, identifier: String, labels: [String]?) {
        holder = NFKYOLOGenerationBackendHolder(net, labels: labels)
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
        if let result = job.result { return result }
        if let error = job.error { throw error }
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
                let image = try NFKMLXImageBridge.tensor(from: value, channels: 3,
                                                         colorSpace: CGColorSpaceCreateDeviceRGB())
                let detections = holder.net.detect(image, labels: holder.labels)
                job.finish(with: NFKInferenceResult(outputs: [NFKOutputDetections: detections]))
            } catch {
                job.finish(withError: error as NSError)
            }
        }
        return job
    }
}

private final class NFKYOLOGenerationBackendHolder: @unchecked Sendable {
    let net: NFKMLXYOLOGenerationNet
    let labels: [String]?
    init(_ net: NFKMLXYOLOGenerationNet, labels: [String]?) {
        self.net = net
        self.labels = labels
    }
}
