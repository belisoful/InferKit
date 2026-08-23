//
//  NFKMLXRIFE.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXNN

// RIFE (Real-time Intermediate Flow Estimation) interpolates a frame between two inputs. It runs
// through `NFKMLXTensorBackend`: two frames in (under the keys "frame0" / "frame1"), the middle frame
// out under `NFKOutputImage`. The IFNet estimates bidirectional flow coarse-to-fine, backward-warps
// each frame, and blends them with a learned mask. Tensors flow NHWC; the warp is a bilinear
// grid-sample built from gather (MLX has no grid_sample).
//
// Block channels/input planes follow RIFE HDv3 (`c` 240/150/90). Matching a specific RIFE version's
// checkpoint (the nested `conv0.0.0` names, channel counts) is a `remap` (validation-sweep task).

/// A convolution + per-channel PReLU (RIFE's `conv`).
final class NFKRIFEConv: Module {
    let conv: Conv2d
    let prelu: PReLU

    init(_ inChannels: Int, _ outChannels: Int, kernelSize: Int = 3, stride: Int = 1, padding: Int = 1) {
        conv = Conv2d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: IntOrPair(kernelSize),
                      stride: IntOrPair(stride), padding: IntOrPair(padding))
        prelu = PReLU(count: outChannels)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { prelu(conv(x)) }
}

/// An IFBlock: downsample by `scale`, two stride-2 convs, a residual conv stack, then a transposed
/// convolution producing a flow (4) + mask (1) field, resized back to the input resolution.
/// A transposed-convolution head: the reference stores each as a `Sequential` of transposed
/// convolution, PReLU, transposed convolution, so its keys are `0`, `1`, `2`.
final class NFKRIFEHead: Module {
    @ModuleInfo(key: "up1") var up1: ConvTransposed2d
    @ModuleInfo(key: "prelu") var prelu: PReLU
    @ModuleInfo(key: "up2") var up2: ConvTransposed2d

    init(channels c: Int, outChannels: Int) {
        _up1.wrappedValue = ConvTransposed2d(inputChannels: c, outputChannels: c / 2, kernelSize: 4,
                                             stride: 2, padding: 1)
        _prelu.wrappedValue = PReLU(count: c / 2)
        _up2.wrappedValue = ConvTransposed2d(inputChannels: c / 2, outputChannels: outChannels,
                                             kernelSize: 4, stride: 2, padding: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { up2(prelu(up1(x))) }
}

/// One coarse-to-fine IFBlock. The trunk is four groups of two convolutions, each group added back to
/// its own input — not one residual over all eight — and the flow and mask leave through separate
/// heads, each of which upsamples ×4 to undo `conv0`'s ×4 stride.
final class NFKRIFEBlock: Module {
    @ModuleInfo(key: "conv0") var conv0: [NFKRIFEConv]
    @ModuleInfo(key: "convblock0") var convblock0: [NFKRIFEConv]
    @ModuleInfo(key: "convblock1") var convblock1: [NFKRIFEConv]
    @ModuleInfo(key: "convblock2") var convblock2: [NFKRIFEConv]
    @ModuleInfo(key: "convblock3") var convblock3: [NFKRIFEConv]
    @ModuleInfo(key: "conv1") var conv1: NFKRIFEHead
    @ModuleInfo(key: "conv2") var conv2: NFKRIFEHead

    init(inPlanes: Int, channels c: Int) {
        _conv0.wrappedValue = [NFKRIFEConv(inPlanes, c / 2, stride: 2), NFKRIFEConv(c / 2, c, stride: 2)]
        for group in [_convblock0, _convblock1, _convblock2, _convblock3] {
            group.wrappedValue = [NFKRIFEConv(c, c), NFKRIFEConv(c, c)]
        }
        _conv1.wrappedValue = NFKRIFEHead(channels: c, outChannels: 4)
        _conv2.wrappedValue = NFKRIFEHead(channels: c, outChannels: 1)
    }

    /// `x` is the seven-channel stack (both frames plus the running mask); `flow` is the running
    /// four-channel field. Both are reduced by `scale` on the way in and the results restored on the
    /// way out, bilinearly — the reference interpolates, and nearest neighbour aliases across scales.
    func callAsFunction(_ x: MLXArray, flow: MLXArray, scale: Int) -> (flow: MLXArray, mask: MLXArray) {
        func reduced(_ array: MLXArray) -> MLXArray {
            scale == 1 ? array : NFKMLXResample.resizeBilinear(array, height: array.shape[1] / scale,
                                                               width: array.shape[2] / scale)
        }
        let input = concatenated([reduced(x), reduced(flow) * (1.0 / Float(scale))], axis: 3)

        var feature = input
        for layer in conv0 {
            feature = layer(feature)
        }
        for group in [convblock0, convblock1, convblock2, convblock3] {
            var residual = feature
            for layer in group {
                residual = layer(residual)
            }
            feature = residual + feature
        }

        var predictedFlow = conv1(feature)
        var predictedMask = conv2(feature)
        if scale != 1 {
            let (height, width) = (x.shape[1], x.shape[2])
            predictedFlow = NFKMLXResample.resizeBilinear(predictedFlow, height: height, width: width) * Float(scale)
            predictedMask = NFKMLXResample.resizeBilinear(predictedMask, height: height, width: width)
        }
        return (predictedFlow, predictedMask)
    }
}

/// The RIFE IFNet. Two frames `[1, H, W, 3]` in, the interpolated middle frame out.
final class NFKMLXRIFENet: Module {
    @ModuleInfo(key: "blocks") var blocks: [NFKRIFEBlock]

    private let scales = [4, 2, 1]

    override init() {
        // Every block in the released network has the same shape: seven input channels (both frames
        // plus the mask) plus the four flow channels, at width 90.
        _blocks.wrappedValue = (0 ..< 3).map { _ in NFKRIFEBlock(inPlanes: 11, channels: 90) }
    }

    /// Bilinear backward warp: `output[y, x] = img[y + flow_y, x + flow_x]`, built from gather. `img`
    /// and `flow` are `[1, H, W, C]` / `[1, H, W, 2]`.
    static func warp(_ img: MLXArray, flow: MLXArray) -> MLXArray {
        let (h, w, c) = (img.shape[1], img.shape[2], img.shape[3])
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

        let sampleX = clip(gridX + flow[0, 0..., 0..., 0], min: 0, max: Float(w - 1))
        let sampleY = clip(gridY + flow[0, 0..., 0..., 1], min: 0, max: Float(h - 1))
        let x0 = sampleX.floor(), y0 = sampleY.floor()
        let x1 = clip(x0 + 1, min: 0, max: Float(w - 1)), y1 = clip(y0 + 1, min: 0, max: Float(h - 1))
        let weightX = (sampleX - x0).reshaped([h, w, 1])
        let weightY = (sampleY - y0).reshaped([h, w, 1])

        let flat = img.reshaped([h * w, c])
        func gather(_ yy: MLXArray, _ xx: MLXArray) -> MLXArray {
            let index = (yy.asType(.int32) * Int32(w) + xx.asType(.int32)).reshaped([h * w])
            return flat.take(index, axis: 0).reshaped([h, w, c])
        }
        let top = gather(y0, x0) * (1 - weightX) + gather(y0, x1) * weightX
        let bottom = gather(y1, x0) * (1 - weightX) + gather(y1, x1) * weightX
        return (top * (1 - weightY) + bottom * weightY).reshaped([1, h, w, c])
    }

    func interpolate(_ frame0: MLXArray, _ frame1: MLXArray) -> MLXArray {
        let unbatched = frame0.ndim == 3
        let (originalH, originalW) = (frame0.shape[unbatched ? 0 : 1], frame0.shape[unbatched ? 1 : 2])
        var img0 = unbatched ? frame0.reshaped([1, originalH, originalW, 3]) : frame0
        var img1 = unbatched ? frame1.reshaped([1, originalH, originalW, 3]) : frame1

        // Pad to a multiple of 32 so the coarsest (÷4 by scale, ÷4 by conv0) path stays integer.
        let multiple = 32
        let padH = (multiple - originalH % multiple) % multiple
        let padW = (multiple - originalW % multiple) % multiple
        if padH > 0 || padW > 0 {
            let widths = [IntOrPair((0, 0)), IntOrPair((0, padH)), IntOrPair((0, padW)), IntOrPair((0, 0))]
            img0 = padded(img0, widths: widths, mode: .edge)
            img1 = padded(img1, widths: widths, mode: .edge)
        }

        /// Swaps the two halves of a four-channel flow field, so a result computed with the frames
        /// exchanged can be read in the original order.
        func swapped(_ flow: MLXArray) -> MLXArray {
            concatenated([flow[0..., 0..., 0..., 2 ..< 4], flow[0..., 0..., 0..., 0 ..< 2]], axis: 3)
        }

        var flow = MLXArray.zeros([1, img0.shape[1], img0.shape[2], 4])
        var mask = MLXArray.zeros([1, img0.shape[1], img0.shape[2], 1])
        var warped0 = img0
        var warped1 = img1

        for (index, block) in blocks.enumerated() {
            // Each block is applied twice and averaged: once as given, once with the frames swapped,
            // the mask negated, and the flow halves exchanged. The network is trained to be
            // symmetric in its two inputs, and one pass alone is not.
            let (f0, m0) = block(concatenated([warped0, warped1, mask], axis: 3),
                                 flow: flow, scale: scales[index])
            let (f1, m1) = block(concatenated([warped1, warped0, -mask], axis: 3),
                                 flow: swapped(flow), scale: scales[index])
            flow = flow + (f0 + swapped(f1)) / 2
            mask = mask + (m0 - m1) / 2
            warped0 = Self.warp(img0, flow: flow[0..., 0..., 0..., 0 ..< 2])
            warped1 = Self.warp(img1, flow: flow[0..., 0..., 0..., 2 ..< 4])
        }

        let blend = sigmoid(mask)
        let merged = clip(warped0 * blend + warped1 * (1 - blend), min: 0, max: 1)
        let cropped = merged[0..., 0 ..< originalH, 0 ..< originalW, 0...]
        return unbatched ? cropped.reshaped([originalH, originalW, 3]) : cropped
    }
}

/// RIFE frame interpolation as an InferKit backend, and its registration for the Objective-C path.
@objc(NFKMLXRIFE)
public final class NFKMLXRIFE: NSObject {

    @objc public static let modelName = "rife"
    /// The two frame input keys the backend reads.
    @objc public static let frame0Key = "frame0"
    @objc public static let frame1Key = "frame1"

    static func makeNet() -> NFKMLXRIFENet { NFKMLXRIFENet() }

    /// Builds a RIFE frame-interpolation backend directly from optional local weights — no registry
    /// required. A nil `weightsURL` builds random weights (`isReady` is true). Reads two frames under
    /// `frame0Key` / `frame1Key`, returns the interpolated frame under `NFKOutputImage`.
    /// Run inference off the render thread.
    @objc(backendWithWeightsURL:error:)
    public static func backend(weightsURL: URL?) throws -> any NFKInferenceBackend {
        let net = NFKMLXRIFENet()
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        let holder = NFKMLXRIFEHolder(net)
        let configuration = NFKMLXTensorConfiguration(
            inputs: [NFKMLXTensorPort(key: frame0Key, tensorName: "frame0", channels: 3),
                     NFKMLXTensorPort(key: frame1Key, tensorName: "frame1", channels: 3)],
            outputs: [NFKMLXTensorPort(key: NFKOutputImage, tensorName: "middle")])
        return NFKMLXTensorBackend(identifier: modelName, configuration: configuration) { inputs in
            ["middle": holder.net.interpolate(inputs["frame0"]!, inputs["frame1"]!)]
        }
    }

    /// Builds a RIFE backend that interpolates a whole CLIP, doubling its frame rate.
    ///
    /// @discussion Reads an `NFKVideoAsset` under `NFKInputVideo` and returns one under
    /// `NFKOutputVideo`. Every adjacent pair contributes its own frame and one synthesized between
    /// them, so a clip of `n` frames becomes `2n - 1` and the result is written at twice the source
    /// rate — writing it at the source rate would produce slow motion rather than smoother playback.
    /// Run inference off the render thread; a clip is many forward passes.
    @objc(clipBackendWithWeightsURL:error:)
    public static func clipBackend(weightsURL: URL?) throws -> any NFKInferenceBackend {
        let net = NFKMLXRIFENet()
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        let holder = NFKMLXRIFEHolder(net)
        var configuration = NFKMLXVideoConfiguration()
        configuration.frameRateMultiplier = 2
        return NFKMLXVideoBackend(identifier: "\(modelName)-clip", configuration: configuration) { frames in
            guard frames.count > 1 else { return frames }
            var output = [MLXArray]()
            output.reserveCapacity(frames.count * 2 - 1)
            for index in 0 ..< frames.count - 1 {
                output.append(frames[index])
                output.append(holder.net.interpolate(frames[index], frames[index + 1]))
            }
            output.append(frames[frames.count - 1])
            return output
        }
    }

    /// Downloads the checkpoint from Hugging Face, then builds — no registry required.
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

    /// Registers RIFE (`rife`) with `NFKMLXModelRegistry`, delegating to `backend(weightsURL:)`.
    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName) { weightsURL in try backend(weightsURL: weightsURL) }
    }

    /// Loads a safetensors checkpoint, transposing 4-D convolution weights to MLX's layout. `remap`
    /// maps the reference nested `conv0.0.0` / `block…` names to the module's keys (sweep task).
    /// Maps the released `flownet.pkl` names onto the module's. The checkpoint carries the training
    /// wrapper's `module.` prefix and names its blocks `block0…block2`; every convolution and head is
    /// a `Sequential`, so its entries are numbered. `block_tea` is the training-time teacher and has
    /// no counterpart here — it stays unmapped and is ignored as an extra key.
    static func remapReferenceKey(_ key: String) -> String {
        var key = key
        if key.hasPrefix("module.") {
            key = String(key.dropFirst("module.".count))
        }
        for index in 0 ..< 3 where key.hasPrefix("block\(index).") {
            key = "blocks.\(index)." + key.dropFirst("block\(index).".count)
        }
        // `conv0.i` and `convblockN.i` are (convolution, PReLU) pairs.
        for group in ["conv0", "convblock0", "convblock1", "convblock2", "convblock3"] {
            for slot in 0 ..< 2 {
                key = key.replacingOccurrences(of: ".\(group).\(slot).0.", with: ".\(group).\(slot).conv.")
                key = key.replacingOccurrences(of: ".\(group).\(slot).1.", with: ".\(group).\(slot).prelu.")
            }
        }
        // The flow and mask heads are (transposed convolution, PReLU, transposed convolution).
        for head in ["conv1", "conv2"] {
            key = key.replacingOccurrences(of: ".\(head).0.", with: ".\(head).up1.")
            key = key.replacingOccurrences(of: ".\(head).1.", with: ".\(head).prelu.")
            key = key.replacingOccurrences(of: ".\(head).2.", with: ".\(head).up2.")
        }
        return key
    }

    static func loadWeights(into net: NFKMLXRIFENet, from url: URL, remap: ((String) -> String)? = nil) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let raw = checkpoint.arrays
        let mapped = raw.map { key, value -> (String, MLXArray) in
            let name = remap?(key) ?? remapReferenceKey(key)
            // The heads are transposed convolutions, stored `[in, out, kH, kW]` upstream.
            if checkpoint.needsConvTranspose, value.ndim == 4 {
                let isTransposed = name.contains(".up1.") || name.contains(".up2.")
                return (name, isTransposed ? value.transposed(1, 2, 3, 0) : value.transposed(0, 2, 3, 1))
            }
            return (name, value)
        }
        try NFKMLXWeights.apply(mapped, to: net)
    }
}

private final class NFKMLXRIFEHolder: @unchecked Sendable {
    let net: NFKMLXRIFENet
    init(_ net: NFKMLXRIFENet) { self.net = net }
}

// MARK: - RIFE v4
//
// v4 is a third IFNet generation, not a variant of HDv3: four blocks instead of three, a learned
// frame encoder whose features are warped alongside the frames, residual `ResConv` trunk entries
// carrying a per-channel scale, a transposed convolution plus pixel shuffle in place of a single
// upsampling convolution, and a **timestep** channel that lets it interpolate at any point between
// the two frames rather than only the midpoint.

/// A v4 convolution unit: a strided convolution and a parameter-free leaky ReLU. HDv3 used a PReLU
/// here, which carries a weight per channel — the difference shows up as eight uncovered parameters
/// rather than as a wrong number, so the coverage guard names it precisely.
final class NFKRIFEv4Conv: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d

    init(_ inChannels: Int, _ outChannels: Int, stride: Int = 1) {
        _conv.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: outChannels,
                                    kernelSize: 3, stride: IntOrPair(stride), padding: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { leakyRelu(conv(x), negativeSlope: 0.2) }
}

/// The reference `ResConv`: one convolution scaled by a learned per-channel factor, added back.
final class NFKRIFEResConv: Module {
    @ModuleInfo(key: "conv") var conv: Conv2d
    @ParameterInfo(key: "beta") var beta: MLXArray

    init(channels: Int) {
        _conv.wrappedValue = Conv2d(inputChannels: channels, outputChannels: channels,
                                    kernelSize: 3, padding: 1)
        _beta.wrappedValue = MLXArray.ones([1, 1, 1, channels])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        leakyRelu(conv(x) * beta + x, negativeSlope: 0.2)
    }
}

/// The v4 IFBlock: two strided convolutions, eight residual entries, then a transposed convolution
/// whose output pixel-shuffles ×2 into four flow channels and two mask/feature channels.
final class NFKRIFEv4Block: Module {
    @ModuleInfo(key: "conv0") var conv0: [NFKRIFEv4Conv]
    @ModuleInfo(key: "convblock") var convblock: [NFKRIFEResConv]
    @ModuleInfo(key: "lastconv") var lastconv: ConvTransposed2d

    init(inPlanes: Int, channels c: Int) {
        _conv0.wrappedValue = [NFKRIFEv4Conv(inPlanes, c / 2, stride: 2), NFKRIFEv4Conv(c / 2, c, stride: 2)]
        _convblock.wrappedValue = (0 ..< 8).map { _ in NFKRIFEResConv(channels: c) }
        _lastconv.wrappedValue = ConvTransposed2d(inputChannels: c, outputChannels: 4 * 6,
                                                  kernelSize: 4, stride: 2, padding: 1)
    }

    /// Returns the flow delta `[1, H, W, 4]` and the mask `[1, H, W, 1]` at the input resolution.
    func callAsFunction(_ x: MLXArray, flow: MLXArray?, scale: Int) -> (flow: MLXArray, mask: MLXArray) {
        let (height, width) = (x.shape[1], x.shape[2])
        func reduced(_ array: MLXArray) -> MLXArray {
            scale == 1 ? array : NFKMLXResample.resizeBilinear(array, height: array.shape[1] / scale,
                                                               width: array.shape[2] / scale)
        }
        var input = reduced(x)
        if let flow {
            input = concatenated([input, reduced(flow) * (1.0 / Float(scale))], axis: 3)
        }

        var feature = input
        for layer in conv0 {
            feature = layer(feature)
        }
        for layer in convblock {
            feature = layer(feature)
        }
        // The upsampling convolution emits `4 × 6` channels that shuffle ×2 into six.
        var field = NFKMLXPixelShuffle.apply(lastconv(feature), factor: 2)
        field = NFKMLXResample.resizeBilinear(field, height: height, width: width)
        let flowField = field[0..., 0..., 0..., 0 ..< 4] * Float(scale)
        return (flowField, field[0..., 0..., 0..., 4 ..< 5])
    }
}

/// The learned frame encoder (`Head`): three convolutions and a transposed convolution back to the
/// frame's own resolution, emitting eight feature channels the blocks consume alongside the frames.
final class NFKRIFEv4Encoder: Module {
    @ModuleInfo(key: "cnn0") var cnn0: Conv2d
    @ModuleInfo(key: "cnn1") var cnn1: Conv2d
    @ModuleInfo(key: "cnn2") var cnn2: Conv2d
    @ModuleInfo(key: "cnn3") var cnn3: ConvTransposed2d

    override init() {
        _cnn0.wrappedValue = Conv2d(inputChannels: 3, outputChannels: 32, kernelSize: 3, stride: 2, padding: 1)
        _cnn1.wrappedValue = Conv2d(inputChannels: 32, outputChannels: 32, kernelSize: 3, padding: 1)
        _cnn2.wrappedValue = Conv2d(inputChannels: 32, outputChannels: 32, kernelSize: 3, padding: 1)
        _cnn3.wrappedValue = ConvTransposed2d(inputChannels: 32, outputChannels: 8, kernelSize: 4,
                                              stride: 2, padding: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var out = leakyRelu(cnn0(x), negativeSlope: 0.2)
        out = leakyRelu(cnn1(out), negativeSlope: 0.2)
        out = leakyRelu(cnn2(out), negativeSlope: 0.2)
        return cnn3(out)
    }
}

/// The RIFE v4 IFNet: a frame encoder and four coarse-to-fine blocks, conditioned on a timestep.
final class NFKMLXRIFEv4Net: Module {
    @ModuleInfo(key: "encode") var encode: NFKRIFEv4Encoder
    @ModuleInfo(key: "block0") var block0: NFKRIFEv4Block
    @ModuleInfo(key: "block1") var block1: NFKRIFEv4Block
    @ModuleInfo(key: "block2") var block2: NFKRIFEv4Block
    @ModuleInfo(key: "block3") var block3: NFKRIFEv4Block

    private let scales = [8, 4, 2, 1]

    override init() {
        _encode.wrappedValue = NFKRIFEv4Encoder()
        // Seven channels (both frames plus the timestep) and sixteen encoded ones for the first
        // block; the later blocks see the warped frames and features, the timestep, and the mask.
        _block0.wrappedValue = NFKRIFEv4Block(inPlanes: 7 + 16, channels: 192)
        _block1.wrappedValue = NFKRIFEv4Block(inPlanes: 8 + 4 + 16, channels: 128)
        _block2.wrappedValue = NFKRIFEv4Block(inPlanes: 8 + 4 + 16, channels: 96)
        _block3.wrappedValue = NFKRIFEv4Block(inPlanes: 8 + 4 + 16, channels: 64)
    }

    /// Interpolates between two frames `[1, H, W, 3]` (`0...1`) at `timestep` — 0.5 is the midpoint,
    /// and any value in `0...1` is valid, which is what v4 adds over the earlier generations.
    func interpolate(_ frame0: MLXArray, _ frame1: MLXArray, timestep: Float = 0.5) -> MLXArray {
        let unbatched = frame0.ndim == 3
        let (originalH, originalW) = (frame0.shape[unbatched ? 0 : 1], frame0.shape[unbatched ? 1 : 2])
        var img0 = unbatched ? frame0.reshaped([1, originalH, originalW, 3]) : frame0
        var img1 = unbatched ? frame1.reshaped([1, originalH, originalW, 3]) : frame1
        img0 = clip(img0, min: 0, max: 1)
        img1 = clip(img1, min: 0, max: 1)

        // The reference pads to a multiple of 64 before the network and crops afterwards.
        let multiple = 64
        let padH = (multiple - originalH % multiple) % multiple
        let padW = (multiple - originalW % multiple) % multiple
        if padH > 0 || padW > 0 {
            let widths = [IntOrPair((0, 0)), IntOrPair((0, padH)), IntOrPair((0, padW)), IntOrPair((0, 0))]
            img0 = padded(img0, widths: widths, mode: .constant, value: MLXArray(Float(0)))
            img1 = padded(img1, widths: widths, mode: .constant, value: MLXArray(Float(0)))
        }

        let feature0 = encode(img0)
        let feature1 = encode(img1)
        let time = MLXArray.full([1, img0.shape[1], img0.shape[2], 1], values: MLXArray(timestep))

        var flow: MLXArray?
        var mask = MLXArray.zeros([1, img0.shape[1], img0.shape[2], 1])
        var warped0 = img0
        var warped1 = img1

        for (index, block) in [block0, block1, block2, block3].enumerated() {
            if let current = flow {
                let input = concatenated([warped0, warped1,
                                          NFKMLXRIFENet.warp(feature0, flow: current[0..., 0..., 0..., 0 ..< 2]),
                                          NFKMLXRIFENet.warp(feature1, flow: current[0..., 0..., 0..., 2 ..< 4]),
                                          time, mask], axis: 3)
                let (delta, newMask) = block(input, flow: current, scale: scales[index])
                flow = current + delta
                mask = newMask
            } else {
                let input = concatenated([img0, img1, feature0, feature1, time], axis: 3)
                let (initial, initialMask) = block(input, flow: nil, scale: scales[index])
                flow = initial
                mask = initialMask
            }
            warped0 = NFKMLXRIFENet.warp(img0, flow: flow![0..., 0..., 0..., 0 ..< 2])
            warped1 = NFKMLXRIFENet.warp(img1, flow: flow![0..., 0..., 0..., 2 ..< 4])
        }

        let blend = sigmoid(mask)
        let merged = clip(warped0 * blend + warped1 * (1 - blend), min: 0, max: 1)
        let cropped = merged[0..., 0 ..< originalH, 0 ..< originalW, 0...]
        return unbatched ? cropped.reshaped([originalH, originalW, 3]) : cropped
    }
}

/// Holds the network for capture in the backend's `@Sendable` closure.
private final class NFKMLXRIFEv4Holder: @unchecked Sendable {
    let net: NFKMLXRIFEv4Net
    init(_ net: NFKMLXRIFEv4Net) { self.net = net }
}

/// RIFE v4 frame interpolation as an InferKit backend.
///
/// Two frames under `frame0` / `frame1` produce the interpolated frame under `NFKOutputImage`, as
/// for `NFKMLXRIFE`. v4 conditions on a timestep, so a caller wanting a point other than the midpoint
/// uses `NFKMLXRIFEv4Net.interpolate(_:_:timestep:)` directly.
@objc(NFKMLXRIFEv4)
public final class NFKMLXRIFEv4: NSObject {

    /// The registry name the model builds under.
    @objc public static let modelName = "rife-v4"

    static func makeNet() -> NFKMLXRIFEv4Net { NFKMLXRIFEv4Net() }

    /// Builds an interpolation backend directly from optional local weights — no registry required.
    /// Run inference off the render thread.
    @objc(backendWithWeightsURL:error:)
    public static func backend(weightsURL: URL?) throws -> any NFKInferenceBackend {
        let net = NFKMLXRIFEv4Net()
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        let holder = NFKMLXRIFEv4Holder(net)
        let configuration = NFKMLXTensorConfiguration(
            inputs: [NFKMLXTensorPort(key: NFKMLXRIFE.frame0Key, tensorName: "frame0", channels: 3),
                     NFKMLXTensorPort(key: NFKMLXRIFE.frame1Key, tensorName: "frame1", channels: 3)],
            outputs: [NFKMLXTensorPort(key: NFKOutputImage, tensorName: "middle")])
        return NFKMLXTensorBackend(identifier: modelName, configuration: configuration) { inputs in
            ["middle": holder.net.interpolate(inputs["frame0"]!, inputs["frame1"]!)]
        }
    }

    /// Downloads the checkpoint, then builds the backend. Blocking on the network; run off the render
    /// thread.
    @objc(backendWithRepo:weightsPath:revision:cacheDirectoryURL:error:)
    public static func backend(repo: String, weightsPath: String, revision: String?, cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        let url = try NFKMLXDownload.weightsURL(repo: repo, weightsPath: weightsPath, revision: revision, cacheDirectoryURL: cacheDirectoryURL)
        return try backend(weightsURL: url)
    }

    /// The asynchronous form of the download factory.
    @objc(backendWithRepo:weightsPath:revision:cacheDirectoryURL:completionHandler:)
    public static func backend(repo: String, weightsPath: String, revision: String?, cacheDirectoryURL: URL?,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXDownload.backend(repo: repo, weightsPath: weightsPath, revision: revision,
                               cacheDirectoryURL: cacheDirectoryURL,
                               build: { try backend(weightsURL: $0) },
                               completionHandler: completionHandler)
    }

    /// Registers RIFE v4 (`rife-v4`) with `NFKMLXModelRegistry`.
    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName) { weightsURL in
            try backend(weightsURL: weightsURL)
        }
    }

    /// Maps the released names onto the module's. The v4 releases ship without the training wrapper's
    /// prefix, so only the `Sequential` entries inside each block and the encoder need naming.
    static func remapReferenceKey(_ key: String) -> String {
        var key = key
        if key.hasPrefix("module.") {
            key = String(key.dropFirst("module.".count))
        }
        // `conv0.i` is a (convolution, PReLU) pair; `lastconv.0` is the upsampling convolution.
        for slot in 0 ..< 2 {
            // Only the convolution carries parameters; the leaky ReLU beside it does not.
            key = key.replacingOccurrences(of: ".conv0.\(slot).0.", with: ".conv0.\(slot).conv.")
        }
        key = key.replacingOccurrences(of: ".lastconv.0.", with: ".lastconv.")
        return key
    }

    /// Loads a safetensors checkpoint. The upsampling convolutions and the encoder's last layer are
    /// transposed convolutions and take the other axis order; `beta` is a per-channel vector stored
    /// `[1, C, 1, 1]`, which becomes `[1, 1, 1, C]` in this layout.
    static func loadWeights(into net: NFKMLXRIFEv4Net, from url: URL) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let raw = checkpoint.arrays
        let mapped = raw.map { key, value -> (String, MLXArray) in
            let name = remapReferenceKey(key)
            if name.hasSuffix(".beta") {
                return (name, value.transposed(0, 2, 3, 1))
            }
            if checkpoint.needsConvTranspose, value.ndim == 4 {
                let isTransposed = name.hasSuffix("lastconv.weight") || name.hasSuffix("cnn3.weight")
                return (name, isTransposed ? value.transposed(1, 2, 3, 0) : value.transposed(0, 2, 3, 1))
            }
            return (name, value)
        }
        try NFKMLXWeights.apply(mapped, to: net)
    }
}
