//
//  NFKMLXVideoSR.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXNN

// Video super-resolution upscales a clip while using neighboring frames for detail a single frame
// cannot supply. This is the reference BasicVSR (mmediting `BasicVSRNet`, ×4): SPyNet estimates
// optical flow between neighboring frames, and two propagation branches — one running backward
// through time, one forward — warp their hidden features along that flow before a stack of residual
// blocks absorbs each frame. The two directions fuse per frame, and a two-stage pixel-shuffle
// reconstructor upscales, adding the bilinearly upscaled input as a base. Bidirectionality means a
// frame's result draws on frames after it as well as before it, so the whole clip goes through
// `upscaleSequence` together; `upscale` runs the same network over a single frame. Tensors flow in
// NHWC.

/// Video-SR dimensions. `base` is the released BasicVSR (REDS4) geometry; `tiny` keeps tests fast.
/// The upscale is ×4 (two ×2 pixel-shuffle stages), as the reference supports only ×4.
public struct NFKMLXVideoSRConfiguration: Sendable {
    /// The propagation branches' feature width.
    public var midChannels: Int
    /// Residual blocks per propagation branch.
    public var blocks: Int
    /// SPyNet pyramid levels. Input sides must be a multiple of `2^(levels-1)` after SPyNet's own
    /// upsizing, which rounds up to that multiple internally.
    public var spynetLevels: Int
    /// The reconstructor's width after the second pixel shuffle.
    public var hrChannels: Int

    public init(midChannels: Int = 64, blocks: Int = 30, spynetLevels: Int = 6, hrChannels: Int = 64) {
        self.midChannels = midChannels
        self.blocks = blocks
        self.spynetLevels = spynetLevels
        self.hrChannels = hrChannels
    }

    public static let base = NFKMLXVideoSRConfiguration()

    public static let tiny = NFKMLXVideoSRConfiguration(midChannels: 8, blocks: 1, spynetLevels: 3, hrChannels: 8)
}

/// The reference `ResidualBlockNoBN`: convolution → ReLU → convolution, added to the identity.
final class NFKVSRResidualBlock: Module {
    @ModuleInfo(key: "conv1") var conv1: Conv2d
    @ModuleInfo(key: "conv2") var conv2: Conv2d

    init(channels: Int) {
        _conv1.wrappedValue = Conv2d(inputChannels: channels, outputChannels: channels, kernelSize: 3, padding: 1)
        _conv2.wrappedValue = Conv2d(inputChannels: channels, outputChannels: channels, kernelSize: 3, padding: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        x + conv2(relu(conv1(x)))
    }
}

/// A propagation branch (`ResidualBlocksWithInputConv`): a channel-matching convolution with leaky
/// ReLU, then the residual blocks. The reference packs these into one Sequential (`main.0` the
/// convolution, `main.2` the block stack); the names here are semantic, translated by the remap.
final class NFKVSRPropagationBranch: Module {
    @ModuleInfo(key: "conv_in") var convIn: Conv2d
    @ModuleInfo(key: "blocks") var blocks: [NFKVSRResidualBlock]

    init(inChannels: Int, outChannels: Int, blocks blockCount: Int) {
        _convIn.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: 3, padding: 1)
        _blocks.wrappedValue = (0 ..< blockCount).map { _ in NFKVSRResidualBlock(channels: outChannels) }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = leakyRelu(convIn(x), negativeSlope: 0.1)
        for block in blocks {
            out = block(out)
        }
        return out
    }
}

/// One SPyNet pyramid-level module: five 7×7 convolutions (8 → 32 → 64 → 32 → 16 → 2), ReLU after
/// all but the last, refining the upsampled flow from the coarser level.
final class NFKVSRSPyNetModule: Module {
    final class Entry: Module {
        @ModuleInfo(key: "conv") var conv: Conv2d
        init(inChannels: Int, outChannels: Int) {
            _conv.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: outChannels,
                                        kernelSize: 7, padding: 3)
        }
    }

    @ModuleInfo(key: "basic_module") var layers: [Entry]

    override init() {
        let widths = [8, 32, 64, 32, 16, 2]
        _layers.wrappedValue = (0 ..< widths.count - 1).map { Entry(inChannels: widths[$0], outChannels: widths[$0 + 1]) }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = x
        for (index, layer) in layers.enumerated() {
            out = layer.conv(out)
            if index < layers.count - 1 {
                out = relu(out)
            }
        }
        return out
    }
}

/// SPyNet optical flow: a coarse-to-fine pyramid where each level warps the supporting frame by the
/// upsampled coarser flow and predicts a residual. The ImageNet normalization tensors load from the
/// checkpoint (the reference registers them as buffers).
final class NFKVSRSPyNet: Module {
    @ModuleInfo(key: "basic_module") var modules: [NFKVSRSPyNetModule]
    @ParameterInfo(key: "mean") var mean: MLXArray
    @ParameterInfo(key: "std") var std: MLXArray

    init(levels: Int) {
        _modules.wrappedValue = (0 ..< levels).map { _ in NFKVSRSPyNetModule() }
        _mean.wrappedValue = MLXArray([Float(0.485), 0.456, 0.406]).reshaped([1, 1, 1, 3])
        _std.wrappedValue = MLXArray([Float(0.229), 0.224, 0.225]).reshaped([1, 1, 1, 3])
    }

    /// The flow from `reference` to `support`, `[1, H, W, 2]` in pixels (dx, dy), both frames
    /// `[1, H, W, 3]` in `0...1`. SPyNet upsizes internally to a multiple of `2^(levels-1)` and
    /// rescales the flow back, as the reference does.
    func flow(_ reference: MLXArray, _ support: MLXArray) -> MLXArray {
        let (height, width) = (reference.shape[1], reference.shape[2])
        let multiple = 1 << (modules.count - 1)
        let upHeight = height % multiple == 0 ? height : multiple * (height / multiple + 1)
        let upWidth = width % multiple == 0 ? width : multiple * (width / multiple + 1)
        var ref = reference, supp = support
        if upHeight != height || upWidth != width {
            ref = NFKMLXResample.resizeBilinear(ref, height: upHeight, width: upWidth)
            supp = NFKMLXResample.resizeBilinear(supp, height: upHeight, width: upWidth)
        }

        var refPyramid = [(ref - mean) / std]
        var suppPyramid = [(supp - mean) / std]
        for _ in 1 ..< modules.count {
            refPyramid.append(AvgPool2d(kernelSize: 2, stride: 2)(refPyramid.last!))
            suppPyramid.append(AvgPool2d(kernelSize: 2, stride: 2)(suppPyramid.last!))
        }
        refPyramid.reverse()
        suppPyramid.reverse()

        var flow = MLXArray.zeros([1, refPyramid[0].shape[1], refPyramid[0].shape[2], 2])
        for level in 0 ..< modules.count {
            let flowUp: MLXArray
            if level == 0 {
                flowUp = flow
            } else {
                flowUp = NFKMLXVideoSRNet.resizeBilinearAlignCorners(
                    flow, height: refPyramid[level].shape[1], width: refPyramid[level].shape[2]) * 2.0
            }
            let warped = NFKMLXVideoSRNet.flowWarp(suppPyramid[level], flow: flowUp, zeroOutside: false)
            flow = flowUp + modules[level](concatenated([refPyramid[level], warped, flowUp], axis: 3))
        }

        if upHeight != height || upWidth != width {
            flow = NFKMLXResample.resizeBilinear(flow, height: height, width: width)
            let scale = MLXArray([Float(width) / Float(upWidth), Float(height) / Float(upHeight)]).reshaped([1, 1, 1, 2])
            flow = flow * scale
        }
        return flow
    }
}

/// The reference `PixelShufflePack`: a channel-expanding convolution followed by a ×2 pixel shuffle.
final class NFKVSRPixelShufflePack: Module {
    @ModuleInfo(key: "upsample_conv") var conv: Conv2d

    init(inChannels: Int, outChannels: Int) {
        _conv.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: outChannels * 4,
                                    kernelSize: 3, padding: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        NFKMLXPixelShuffle.apply(conv(x), factor: 2)
    }
}

/// The BasicVSR network: SPyNet flow, backward and forward propagation branches, fusion, and the
/// pixel-shuffle reconstructor.
final class NFKMLXVideoSRNet: Module {
    @ModuleInfo(key: "spynet") var spynet: NFKVSRSPyNet
    @ModuleInfo(key: "backward_resblocks") var backwardBranch: NFKVSRPropagationBranch
    @ModuleInfo(key: "forward_resblocks") var forwardBranch: NFKVSRPropagationBranch
    @ModuleInfo(key: "fusion") var fusion: Conv2d
    @ModuleInfo(key: "upsample1") var upsample1: NFKVSRPixelShufflePack
    @ModuleInfo(key: "upsample2") var upsample2: NFKVSRPixelShufflePack
    @ModuleInfo(key: "conv_hr") var convHr: Conv2d
    @ModuleInfo(key: "conv_last") var convLast: Conv2d

    let configuration: NFKMLXVideoSRConfiguration

    init(_ c: NFKMLXVideoSRConfiguration) {
        configuration = c
        _spynet.wrappedValue = NFKVSRSPyNet(levels: c.spynetLevels)
        _backwardBranch.wrappedValue = NFKVSRPropagationBranch(inChannels: c.midChannels + 3,
                                                               outChannels: c.midChannels, blocks: c.blocks)
        _forwardBranch.wrappedValue = NFKVSRPropagationBranch(inChannels: c.midChannels + 3,
                                                              outChannels: c.midChannels, blocks: c.blocks)
        _fusion.wrappedValue = Conv2d(inputChannels: c.midChannels * 2, outputChannels: c.midChannels, kernelSize: 1)
        _upsample1.wrappedValue = NFKVSRPixelShufflePack(inChannels: c.midChannels, outChannels: c.midChannels)
        _upsample2.wrappedValue = NFKVSRPixelShufflePack(inChannels: c.midChannels, outChannels: c.hrChannels)
        _convHr.wrappedValue = Conv2d(inputChannels: c.hrChannels, outputChannels: c.hrChannels, kernelSize: 3, padding: 1)
        _convLast.wrappedValue = Conv2d(inputChannels: c.hrChannels, outputChannels: 3, kernelSize: 3, padding: 1)
    }

    /// Bilinear backward warp at integer pixel coordinates (`grid_sample` with `align_corners=true`):
    /// `output[y, x] = input[y + dy, x + dx]`. `zeroOutside` selects the padding mode — `true` is the
    /// reference's `zeros` (an out-of-range tap contributes nothing; the propagation warp), `false`
    /// its `border` (coordinates clamp to the edge; SPyNet's pyramid warp).
    static func flowWarp(_ x: MLXArray, flow: MLXArray, zeroOutside: Bool) -> MLXArray {
        let (h, w, c) = (x.shape[1], x.shape[2], x.shape[3])
        var baseX = [Float](repeating: 0, count: h * w)
        var baseY = [Float](repeating: 0, count: h * w)
        for y in 0 ..< h {
            for x in 0 ..< w {
                baseX[y * w + x] = Float(x)
                baseY[y * w + x] = Float(y)
            }
        }
        let gridX = baseX.withUnsafeBufferPointer { MLXArray($0, [h, w]) }
        let gridY = baseY.withUnsafeBufferPointer { MLXArray($0, [h, w]) }

        var sampleX = gridX + flow[0, 0..., 0..., 0]
        var sampleY = gridY + flow[0, 0..., 0..., 1]
        if !zeroOutside {
            sampleX = clip(sampleX, min: 0, max: Float(w - 1))
            sampleY = clip(sampleY, min: 0, max: Float(h - 1))
        }
        let x0 = sampleX.floor(), y0 = sampleY.floor()
        let x1 = x0 + 1, y1 = y0 + 1
        let weightX = (sampleX - x0).reshaped([h, w, 1])
        let weightY = (sampleY - y0).reshaped([h, w, 1])

        let flat = x.reshaped([h * w, c])
        func tap(_ yy: MLXArray, _ xx: MLXArray) -> MLXArray {
            let clampedY = clip(yy, min: 0, max: Float(h - 1)).asType(.int32)
            let clampedX = clip(xx, min: 0, max: Float(w - 1)).asType(.int32)
            let gathered = flat.take((clampedY * Int32(w) + clampedX).reshaped([h * w]), axis: 0)
                .reshaped([h, w, c])
            guard zeroOutside else { return gathered }
            let inside = ((yy .>= 0) .&& (yy .<= Float(h - 1)) .&& (xx .>= 0) .&& (xx .<= Float(w - 1)))
            return gathered * inside.asType(x.dtype).reshaped([h, w, 1])
        }
        let top = tap(y0, x0) * (1 - weightX) + tap(y0, x1) * weightX
        let bottom = tap(y1, x0) * (1 - weightX) + tap(y1, x1) * weightX
        return (top * (1 - weightY) + bottom * weightY).reshaped([1, h, w, c])
    }

    /// Bilinear resize with `align_corners=true` (the reference's flow upsampling between pyramid
    /// levels); `NFKMLXResample.resizeBilinear` is the `align_corners=false` grid.
    static func resizeBilinearAlignCorners(_ x: MLXArray, height: Int, width: Int) -> MLXArray {
        let (n, h, w, c) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        func axis(_ outSize: Int, _ inSize: Int) -> (lo: MLXArray, hi: MLXArray, frac: [Float]) {
            let scale = outSize > 1 ? Float(inSize - 1) / Float(outSize - 1) : 0
            var lo = [Int32](), hi = [Int32](), frac = [Float]()
            for o in 0 ..< outSize {
                let src = min(Float(o) * scale, Float(inSize - 1))
                let low = Int(src.rounded(.down))
                lo.append(Int32(low))
                hi.append(Int32(min(low + 1, inSize - 1)))
                frac.append(src - Float(low))
            }
            return (MLXArray(lo), MLXArray(hi), frac)
        }
        let (y0, y1, yf) = axis(height, h)
        let (x0, x1, xf) = axis(width, w)
        let rows0 = x[0..., y0], rows1 = x[0..., y1]
        let c00 = rows0[0..., 0..., x0], c01 = rows0[0..., 0..., x1]
        let c10 = rows1[0..., 0..., x0], c11 = rows1[0..., 0..., x1]
        let wy = MLXArray(yf).reshaped([1, height, 1, 1])
        let wx = MLXArray(xf).reshaped([1, 1, width, 1])
        let top = c00 * (1 - wx) + c01 * wx
        let bottom = c10 * (1 - wx) + c11 * wx
        return (top * (1 - wy) + bottom * wy).reshaped([n, height, width, c])
    }

    /// Upscales a clip ×4, `[H, W, 3]` (`0...1`) per frame. The backward branch runs the whole clip
    /// first, so each output frame draws on the frames after it as well as before it.
    func upscaleSequence(_ frames: [MLXArray]) -> [MLXArray] {
        let batched = frames.map { $0.reshaped([1, $0.shape[0], $0.shape[1], 3]) }
        let count = batched.count
        let (h, w) = (batched[0].shape[1], batched[0].shape[2])

        // flowsBackward[i]: frame i → i+1, used pulling features backward; flowsForward[i]: i+1 → i.
        var flowsBackward = [MLXArray]()
        var flowsForward = [MLXArray]()
        for i in 0 ..< count - 1 {
            flowsBackward.append(spynet.flow(batched[i], batched[i + 1]))
            flowsForward.append(spynet.flow(batched[i + 1], batched[i]))
        }

        var backwardFeatures = [MLXArray?](repeating: nil, count: count)
        var feature = MLXArray.zeros([1, h, w, configuration.midChannels])
        for i in stride(from: count - 1, through: 0, by: -1) {
            if i < count - 1 {
                feature = Self.flowWarp(feature, flow: flowsBackward[i], zeroOutside: true)
            }
            feature = backwardBranch(concatenated([batched[i], feature], axis: 3))
            backwardFeatures[i] = feature
        }

        var outputs = [MLXArray]()
        feature = MLXArray.zeros([1, h, w, configuration.midChannels])
        for i in 0 ..< count {
            if i > 0 {
                feature = Self.flowWarp(feature, flow: flowsForward[i - 1], zeroOutside: true)
            }
            feature = forwardBranch(concatenated([batched[i], feature], axis: 3))

            var out = leakyRelu(fusion(concatenated([backwardFeatures[i]!, feature], axis: 3)), negativeSlope: 0.1)
            out = leakyRelu(upsample1(out), negativeSlope: 0.1)
            out = leakyRelu(upsample2(out), negativeSlope: 0.1)
            out = leakyRelu(convHr(out), negativeSlope: 0.1)
            out = convLast(out)
            out = out + NFKMLXResample.resizeBilinear(batched[i], height: h * 4, width: w * 4)
            outputs.append(out.reshaped([h * 4, w * 4, 3]))
        }
        return outputs
    }

    /// Upscales a single bridged image `[H, W, 3]` with no temporal context.
    func upscale(_ image: MLXArray) -> MLXArray {
        upscaleSequence([image])[0]
    }
}

/// Holds the network for capture in the backend's `@Sendable` forward closure.
private final class NFKVideoSRHolder: @unchecked Sendable {
    let net: NFKMLXVideoSRNet
    init(_ net: NFKMLXVideoSRNet) { self.net = net }
}

/// BasicVSR video super-resolution as an InferKit backend, and its registration for the Objective-C
/// path.
///
/// `NFKMLXVideoSRNet` is the reference BasicVSR. Random weights run (proving the pipeline); the
/// released REDS4 checkpoint, converted to **safetensors**, upscales sharply. The module backend
/// upscales a single frame (no temporal context); a video consumer hands the whole clip to
/// `NFKMLXVideoSRNet.upscaleSequence`, whose propagation is bidirectional.
@objc(NFKMLXVideoSR)
public final class NFKMLXVideoSR: NSObject {

    /// The registry name the model builds under.
    @objc public static let modelName = "video-super-resolution"

    /// Builds a video-SR backend directly from optional local weights — no registry required. A nil
    /// `weightsURL` builds random weights (`isReady` is true). Run
    /// inference off the render thread.
    @objc(backendWithWeightsURL:error:)
    public static func backend(weightsURL: URL?) throws -> any NFKInferenceBackend {
        let net = NFKMLXVideoSRNet(.base)
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        let holder = NFKVideoSRHolder(net)
        return NFKMLXModuleBackend(identifier: modelName, isReady: true) { image in
            clip(holder.net.upscale(image), min: 0, max: 1)
        }
    }

    /// Builds a BasicVSR backend that upscales a whole CLIP.
    ///
    /// @discussion Reads an `NFKVideoAsset` under `NFKInputVideo` and returns one under
    /// `NFKOutputVideo`, at four times the source's dimensions and the same frame rate. This is the
    /// path the model is built for: propagation runs backward AND forward through time, so a frame
    /// draws on frames after it and upscaling a clip is not the same as upscaling its frames
    /// independently. Run inference off the render thread.
    @objc(clipBackendWithWeightsURL:error:)
    public static func clipBackend(weightsURL: URL?) throws -> any NFKInferenceBackend {
        let net = NFKMLXVideoSRNet(.base)
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        let holder = NFKVideoSRHolder(net)
        return NFKMLXVideoBackend(identifier: "\(modelName)-clip") { frames in
            holder.net.upscaleSequence(frames).map { clip($0, min: 0, max: 1) }
        }
    }

    /// Downloads the checkpoint from Hugging Face, then builds the backend — no registry required.
    /// Blocking on the network; run off the render thread.
    @objc(backendWithRepo:weightsPath:revision:cacheDirectoryURL:error:)
    public static func backend(repo: String, weightsPath: String, revision: String?, cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        let url = try NFKMLXDownload.weightsURL(repo: repo, weightsPath: weightsPath, revision: revision, cacheDirectoryURL: cacheDirectoryURL)
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

    /// Registers video SR (`video-super-resolution`) with `NFKMLXModelRegistry`, delegating to
    /// `backend(weightsURL:)`.
    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName) { weightsURL in
            try backend(weightsURL: weightsURL)
        }
    }

    /// Maps the released checkpoint's names onto the module's: the `generator.` wrapper prefix is
    /// stripped, and the propagation branches' positional Sequential (`main.0` the input convolution,
    /// `main.2` the residual-block stack) becomes semantic.
    static func remapReferenceKey(_ key: String) -> String {
        var key = key
        if key.hasPrefix("generator.") {
            key = String(key.dropFirst("generator.".count))
        }
        key = key.replacingOccurrences(of: ".main.0.", with: ".conv_in.")
        key = key.replacingOccurrences(of: ".main.2.", with: ".blocks.")
        return key
    }

    /// Loads a safetensors checkpoint into `net`, remapping the reference's names and transposing 4-D
    /// convolution weights from PyTorch's `[out, in, kH, kW]` to MLX's channels-last
    /// `[out, kH, kW, in]` (SPyNet's `[1, 3, 1, 1]` normalization buffers become `[1, 1, 1, 3]`
    /// through the same transpose).
    static func loadWeights(into net: NFKMLXVideoSRNet, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let raw = checkpoint.arrays
        let mapped = raw.map { key, value in
            (remapReferenceKey(key), checkpoint.needsConvTranspose && value.ndim == 4 ? value.transposed(0, 2, 3, 1) : value)
        }
        try NFKMLXWeights.apply(mapped, to: net)
    }
}
