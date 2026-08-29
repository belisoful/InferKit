//
//  NFKMLXRAFT.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXNN

// RAFT (Recurrent All-Pairs Field Transforms) estimates dense optical flow between two frames. It runs
// through `NFKMLXTensorBackend`: two frames in under keys "frame0" / "frame1", the flow field out under
// `NFKOutputImage` as a packed flow map (R = 0.5 + fx/scale, G = 0.5 + fy/scale, mid-gray = no motion);
// the raw flow `[H, W, 2]` is available from `NFKMLXRAFTNet.flow`. The pipeline: a shared feature
// encoder, an all-pairs correlation volume + pyramid, a context encoder, and an iterative ConvGRU that
// refines the flow. Tensors flow NHWC; correlation lookup is bilinear via `take` gather.
//
// Faithful to RAFT-large (feature 256, hidden 128, 4 correlation levels, radius 4) with two documented
// simplifications: bilinear ×8 upsampling instead of the learned convex mask (the mask head is kept so
// its weights still load), and a low default iteration count. Matching a checkpoint's nested names is a
// `remap` (validation-sweep task).

private let kCorrLevels = 4
// Internal rather than private so a test can express the plane layout in terms of it instead
// of repeating the number.
let kCorrRadius = 4
private var kCorrPlanes: Int { kCorrLevels * (2 * kCorrRadius + 1) * (2 * kCorrRadius + 1) }

/// The normalization an encoder is built with. Reference RAFT takes a `norm_fn` per encoder: the
/// feature encoder normalizes with `nn.InstanceNorm2d`, which defaults to `affine=False` and so holds
/// no learnable parameters; the context encoder normalizes with `nn.BatchNorm2d`, which holds a weight,
/// a bias, and running statistics. A checkpoint covers an encoder's parameters only when the encoder
/// normalizes the way it was trained.
enum NFKRAFTNormKind {
    case instance
    case batch

    func makeNorm(channels: Int) -> Module & UnaryLayer {
        switch self {
        case .instance:
            return InstanceNorm(dimensions: channels, affine: false)
        case .batch:
            return BatchNorm(featureCount: channels)
        }
    }
}

/// A residual block normalized by its encoder's norm kind (RAFT's `ResidualBlock`).
final class NFKRAFTResBlock: Module {
    let conv1: Conv2d
    let conv2: Conv2d
    let norm1: Module & UnaryLayer
    let norm2: Module & UnaryLayer
    @ModuleInfo(key: "downsample") var downsample: Conv2d?
    @ModuleInfo(key: "norm3") var norm3: (Module & UnaryLayer)?

    init(_ inChannels: Int, _ outChannels: Int, stride: Int, norm: NFKRAFTNormKind) {
        conv1 = Conv2d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: 3, stride: IntOrPair(stride), padding: 1)
        conv2 = Conv2d(inputChannels: outChannels, outputChannels: outChannels, kernelSize: 3, padding: 1)
        norm1 = norm.makeNorm(channels: outChannels)
        norm2 = norm.makeNorm(channels: outChannels)
        if stride != 1 || inChannels != outChannels {
            _downsample.wrappedValue = Conv2d(inputChannels: inChannels, outputChannels: outChannels, kernelSize: 1, stride: IntOrPair(stride))
            _norm3.wrappedValue = norm.makeNorm(channels: outChannels)
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = relu(norm1(conv1(x)))
        y = relu(norm2(conv2(y)))
        let shortcut = downsample.map { norm3!($0(x)) } ?? x
        return relu(y + shortcut)
    }
}

/// A RAFT encoder (feature or context): downsamples by 8 to `outputDim` channels.
final class NFKRAFTEncoder: Module {
    let conv1: Conv2d
    let norm1: Module & UnaryLayer
    @ModuleInfo(key: "layer1") var layer1: [NFKRAFTResBlock]
    @ModuleInfo(key: "layer2") var layer2: [NFKRAFTResBlock]
    @ModuleInfo(key: "layer3") var layer3: [NFKRAFTResBlock]
    let conv2: Conv2d

    init(outputDim: Int, norm: NFKRAFTNormKind) {
        conv1 = Conv2d(inputChannels: 3, outputChannels: 64, kernelSize: 7, stride: 2, padding: 3)
        norm1 = norm.makeNorm(channels: 64)
        _layer1.wrappedValue = [NFKRAFTResBlock(64, 64, stride: 1, norm: norm), NFKRAFTResBlock(64, 64, stride: 1, norm: norm)]
        _layer2.wrappedValue = [NFKRAFTResBlock(64, 96, stride: 2, norm: norm), NFKRAFTResBlock(96, 96, stride: 1, norm: norm)]
        _layer3.wrappedValue = [NFKRAFTResBlock(96, 128, stride: 2, norm: norm), NFKRAFTResBlock(128, 128, stride: 1, norm: norm)]
        conv2 = Conv2d(inputChannels: 128, outputChannels: outputDim, kernelSize: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = relu(norm1(conv1(x)))
        for block in layer1 { y = block(y) }
        for block in layer2 { y = block(y) }
        for block in layer3 { y = block(y) }
        return conv2(y)
    }
}

/// The all-pairs correlation volume, its pyramid, and the neighborhood lookup.
enum NFKRAFTCorrelation {

    /// Builds the correlation pyramid from feature maps `[1, H, W, C]`.
    static func pyramid(_ fmap1: MLXArray, _ fmap2: MLXArray) -> [MLXArray] {
        let (h, w, c) = (fmap1.shape[1], fmap1.shape[2], fmap1.shape[3])
        let f1 = fmap1.reshaped([h * w, c])
        let f2 = fmap2.reshaped([h * w, c])
        let corr = f1.matmul(f2.transposed(1, 0)) / sqrtf(Float(c))    // [P, P]
        var level = corr.reshaped([h * w, h, w, 1])                    // per source pixel, a target map
        var levels = [level]
        for _ in 1 ..< kCorrLevels {
            let (lh, lw) = (level.shape[1], level.shape[2])
            guard lh >= 2, lw >= 2 else { break }
            // 2×2 mean pooling. An odd extent has no partial window: the reference pools with
            // `avg_pool2d(2, stride 2)`, which drops the trailing row / column, so crop to even first.
            let (ph, pw) = ((lh / 2) * 2, (lw / 2) * 2)
            let cropped = level[0..., 0 ..< ph, 0 ..< pw, 0...]
            level = cropped.reshaped([h * w, ph / 2, 2, pw / 2, 2, 1]).mean(axes: [2, 4])
            levels.append(level)
        }
        return levels
    }

    /// One fused neighborhood lookup, in place of the several thousand gathers the elementwise path
    /// issues.
    ///
    /// @discussion The elementwise path walks `(2r+1)²` planes per level and does four gathers for
    /// each, so a single lookup is over thirteen hundred dispatches — and the update runs it once per
    /// GRU iteration. One thread per (pixel, plane) does the same arithmetic in one dispatch.
    ///
    /// Metal source as a string, compiled and cached by MLX on first use: no `.metal` file, and
    /// nothing for a consumer's build to link. `ensureRowContiguous` is left on, so the inputs arrive
    /// laid out as the indices below assume.
    private static let neighborhoodKernel = MLXFast.metalKernel(
        name: "nfk_raft_correlation_lookup",
        inputNames: ["volume", "coordsX", "coordsY", "extent", "scale"],
        outputNames: ["out"],
        source: """
            uint tid = thread_position_in_grid.x;
            const int mapHeight = extent[0];
            const int mapWidth = extent[1];
            const int radius = extent[2];
            const int planes = extent[3];
            const int pixels = extent[4];
            if (tid >= uint(pixels * planes)) { return; }

            const int pixel = int(tid) / planes;
            const int plane = int(tid) % planes;
            const int side = 2 * radius + 1;
            // The outer index shifts x and the inner shifts y, which is the order the trained 1x1
            // `convc1` reads. Swapping them loads cleanly and is wrong.
            const int dx = plane / side - radius;
            const int dy = plane % side - radius;

            const float sx = coordsX[pixel] / scale[0] + float(dx);
            const float sy = coordsY[pixel] / scale[0] + float(dy);
            const float fx = floor(sx);
            const float fy = floor(sy);
            const float wx = sx - fx;
            const float wy = sy - fy;

            const int base = pixel * mapHeight * mapWidth;
            float accumulated = 0.0f;
            for (int cornerY = 0; cornerY < 2; ++cornerY) {
                for (int cornerX = 0; cornerX < 2; ++cornerX) {
                    const float y = fy + float(cornerY);
                    const float x = fx + float(cornerX);
                    // `grid_sample(padding_mode: "zeros")`: a corner outside the map contributes
                    // nothing rather than repeating the edge.
                    if (y < 0.0f || y > float(mapHeight - 1) || x < 0.0f || x > float(mapWidth - 1)) {
                        continue;
                    }
                    const float weight = (cornerX == 0 ? (1.0f - wx) : wx)
                                       * (cornerY == 0 ? (1.0f - wy) : wy);
                    accumulated += volume[base + int(y) * mapWidth + int(x)] * weight;
                }
            }
            out[tid] = accumulated;
        """)

    /// Looks up a `(2r+1)²` neighborhood at each pyramid level around `coords` `[1, H, W, 2]`, returning
    /// correlation features `[1, H, W, kCorrPlanes]`.
    ///
    /// @discussion Runs the fused kernel on the GPU and the elementwise gathers on the CPU, because a
    /// Metal kernel cannot dispatch on the CPU stream and the package lets a caller select it.
    /// `NFKMLXRAFTTests.testTheFusedLookupMatchesTheGatherPath` holds the two to each other.
    static func lookup(_ levels: [MLXArray], coords: MLXArray, height h: Int, width w: Int) -> MLXArray {
        NFKMLXDevice.currentType == .gpu
            ? fusedLookup(levels, coords: coords, height: h, width: w)
            : gatherLookup(levels, coords: coords, height: h, width: w)
    }

    /// The fused path: one dispatch per pyramid level.
    static func fusedLookup(_ levels: [MLXArray], coords: MLXArray,
                            height h: Int, width w: Int) -> MLXArray {
        let pixels = h * w
        let side = 2 * kCorrRadius + 1
        let planes = side * side
        let coordsX = coords[0, 0..., 0..., 0].reshaped([pixels])
        let coordsY = coords[0, 0..., 0..., 1].reshaped([pixels])

        var perLevel = [MLXArray]()
        for (levelIndex, level) in levels.enumerated() {
            let (lh, lw) = (level.shape[1], level.shape[2])
            let extent = MLXArray([Int32(lh), Int32(lw), Int32(kCorrRadius),
                                   Int32(planes), Int32(pixels)])
            let scale = MLXArray([Float(1 << levelIndex)])
            let threads = pixels * planes
            let group = Swift.min(256, threads)
            let result = neighborhoodKernel(
                [level.reshaped([pixels * lh * lw]), coordsX, coordsY, extent, scale],
                grid: (threads, 1, 1),
                threadGroup: (Swift.max(group, 1), 1, 1),
                outputShapes: [[pixels, planes]],
                outputDTypes: [.float32])
            perLevel.append(result[0])
        }
        return concatenated(perLevel, axis: 1).reshaped([1, h, w, perLevel.count * planes])
    }

    /// The elementwise path, kept for the CPU stream and as the reference the fused one is held to.
    static func gatherLookup(_ levels: [MLXArray], coords: MLXArray,
                             height h: Int, width w: Int) -> MLXArray {
        let pixels = h * w
        let pIndex = MLXArray((0 ..< pixels).map { Int32($0) })
        let coordsX = coords[0, 0..., 0..., 0].reshaped([pixels])
        let coordsY = coords[0, 0..., 0..., 1].reshaped([pixels])

        var features = [MLXArray]()
        for (levelIndex, level) in levels.enumerated() {
            let (lh, lw) = (level.shape[1], level.shape[2])
            let volume = level.reshaped([pixels * lh * lw])
            let scale = Float(1 << levelIndex)
            // Plane order follows the reference, where the neighborhood offsets come from
            // `stack(meshgrid(dy, dx))` added to an `(x, y)` centroid: the outer index shifts x and the
            // inner index shifts y. The 1×1 `convc1` is trained on that order, so it is not arbitrary.
            for dx in -kCorrRadius ... kCorrRadius {
                for dy in -kCorrRadius ... kCorrRadius {
                    let sx = coordsX / scale + Float(dx)
                    let sy = coordsY / scale + Float(dy)
                    features.append(bilinear(volume, pIndex: pIndex, mapHeight: lh, mapWidth: lw, sx: sx, sy: sy))
                }
            }
        }
        // features: kCorrPlanes arrays of [P]; stack to [P, planes] -> [1, H, W, planes].
        let planes = features.count
        let stacked = concatenated(features.map { $0.reshaped([pixels, 1]) }, axis: 1)
        return stacked.reshaped([1, h, w, planes])
    }

    /// Bilinear sample of a `[P, Hi, Wi]`-flattened volume at per-pixel `(sx, sy)` `[P]`.
    ///
    /// The reference samples with `grid_sample(padding_mode: "zeros")`, so a corner outside the map
    /// contributes nothing rather than repeating the edge. The distinction is not marginal here: the
    /// lookup radius is 4 and the coarsest pyramid level is a few cells wide, so most of that
    /// neighborhood lies outside the map.
    private static func bilinear(_ volume: MLXArray, pIndex: MLXArray, mapHeight: Int, mapWidth: Int,
                                 sx: MLXArray, sy: MLXArray) -> MLXArray {
        let x0 = sx.floor(), y0 = sy.floor()
        let x1 = x0 + 1, y1 = y0 + 1
        let wx = sx - x0, wy = sy - y0
        let stride = Int32(mapHeight * mapWidth), rowStride = Int32(mapWidth)
        func inside(_ value: MLXArray, _ size: Int) -> MLXArray {
            let low: MLXArray = (value .>= MLXArray(Float(0))).asType(.float32)
            let high: MLXArray = (value .<= MLXArray(Float(size - 1))).asType(.float32)
            return low * high
        }
        func gather(_ yy: MLXArray, _ xx: MLXArray) -> MLXArray {
            let rows: MLXArray = clip(yy, min: 0, max: Float(mapHeight - 1)).asType(.int32)
            let columns: MLXArray = clip(xx, min: 0, max: Float(mapWidth - 1)).asType(.int32)
            let index: MLXArray = pIndex * stride + rows * rowStride + columns
            let valid: MLXArray = inside(yy, mapHeight) * inside(xx, mapWidth)
            return volume.take(index, axis: 0) * valid
        }
        let top = gather(y0, x0) * (1 - wx) + gather(y0, x1) * wx
        let bottom = gather(y1, x0) * (1 - wx) + gather(y1, x1) * wx
        return top * (1 - wy) + bottom * wy
    }
}

/// Encodes correlation features and the current flow into motion features (RAFT's `BasicMotionEncoder`).
final class NFKRAFTMotionEncoder: Module {
    let convc1: Conv2d
    let convc2: Conv2d
    let convf1: Conv2d
    let convf2: Conv2d
    let conv: Conv2d

    override init() {
        convc1 = Conv2d(inputChannels: kCorrPlanes, outputChannels: 256, kernelSize: 1)
        convc2 = Conv2d(inputChannels: 256, outputChannels: 192, kernelSize: 3, padding: 1)
        convf1 = Conv2d(inputChannels: 2, outputChannels: 128, kernelSize: 7, padding: 3)
        convf2 = Conv2d(inputChannels: 128, outputChannels: 64, kernelSize: 3, padding: 1)
        conv = Conv2d(inputChannels: 192 + 64, outputChannels: 126, kernelSize: 3, padding: 1)
    }

    func callAsFunction(_ flow: MLXArray, _ corr: MLXArray) -> MLXArray {
        let cor = relu(convc2(relu(convc1(corr))))
        let flo = relu(convf2(relu(convf1(flow))))
        let out = relu(conv(concatenated([cor, flo], axis: 3)))
        return concatenated([out, flow], axis: 3)               // 126 + 2 = 128
    }
}

/// The separable ConvGRU (RAFT's `SepConvGRU`): a 1×5 then a 5×1 gated update.
final class NFKRAFTConvGRU: Module {
    let convz1, convr1, convq1: Conv2d
    let convz2, convr2, convq2: Conv2d

    init(hidden: Int, input: Int) {
        let total = hidden + input
        convz1 = Conv2d(inputChannels: total, outputChannels: hidden, kernelSize: IntOrPair((1, 5)), padding: IntOrPair((0, 2)))
        convr1 = Conv2d(inputChannels: total, outputChannels: hidden, kernelSize: IntOrPair((1, 5)), padding: IntOrPair((0, 2)))
        convq1 = Conv2d(inputChannels: total, outputChannels: hidden, kernelSize: IntOrPair((1, 5)), padding: IntOrPair((0, 2)))
        convz2 = Conv2d(inputChannels: total, outputChannels: hidden, kernelSize: IntOrPair((5, 1)), padding: IntOrPair((2, 0)))
        convr2 = Conv2d(inputChannels: total, outputChannels: hidden, kernelSize: IntOrPair((5, 1)), padding: IntOrPair((2, 0)))
        convq2 = Conv2d(inputChannels: total, outputChannels: hidden, kernelSize: IntOrPair((5, 1)), padding: IntOrPair((2, 0)))
    }

    func callAsFunction(_ hidden: MLXArray, _ x: MLXArray) -> MLXArray {
        var h = hidden
        var hx = concatenated([h, x], axis: 3)
        var z = sigmoid(convz1(hx)), r = sigmoid(convr1(hx))
        var q = tanh(convq1(concatenated([r * h, x], axis: 3)))
        h = (1 - z) * h + z * q
        hx = concatenated([h, x], axis: 3)
        z = sigmoid(convz2(hx)); r = sigmoid(convr2(hx))
        q = tanh(convq2(concatenated([r * h, x], axis: 3)))
        h = (1 - z) * h + z * q
        return h
    }
}

/// The update block: motion encoder + ConvGRU + flow head (+ a kept mask head).
final class NFKRAFTUpdateBlock: Module {
    @ModuleInfo(key: "encoder") var encoder: NFKRAFTMotionEncoder
    @ModuleInfo(key: "gru") var gru: NFKRAFTConvGRU
    @ModuleInfo(key: "flow_head") var flowHead: [Conv2d]
    @ModuleInfo(key: "mask") var mask: [Conv2d]

    override init() {
        _encoder.wrappedValue = NFKRAFTMotionEncoder()
        _gru.wrappedValue = NFKRAFTConvGRU(hidden: 128, input: 128 + 128)
        _flowHead.wrappedValue = [Conv2d(inputChannels: 128, outputChannels: 256, kernelSize: 3, padding: 1),
                                  Conv2d(inputChannels: 256, outputChannels: 2, kernelSize: 3, padding: 1)]
        _mask.wrappedValue = [Conv2d(inputChannels: 128, outputChannels: 256, kernelSize: 3, padding: 1),
                              Conv2d(inputChannels: 256, outputChannels: 64 * 9, kernelSize: 1)]
    }

    func callAsFunction(_ net: MLXArray, _ context: MLXArray, _ corr: MLXArray, _ flow: MLXArray)
        -> (net: MLXArray, mask: MLXArray, flow: MLXArray) {
        let motion = encoder(flow, corr)
        let hidden = gru(net, concatenated([context, motion], axis: 3))
        let deltaFlow = flowHead[1](relu(flowHead[0](hidden)))
        // The reference scales the mask head down; the weights are trained with that factor in place.
        let upsamplingMask = 0.25 * mask[1](relu(mask[0](hidden)))
        return (hidden, upsamplingMask, deltaFlow)
    }
}

/// The RAFT network. Two frames `[1, H, W, 3]` in, a dense flow field `[H, W, 2]` out (`flow`).
final class NFKMLXRAFTNet: Module {
    @ModuleInfo(key: "fnet") var fnet: NFKRAFTEncoder
    @ModuleInfo(key: "cnet") var cnet: NFKRAFTEncoder
    @ModuleInfo(key: "update") var update: NFKRAFTUpdateBlock

    let iterations: Int

    init(iterations: Int = 6) {
        self.iterations = iterations
        _fnet.wrappedValue = NFKRAFTEncoder(outputDim: 256, norm: .instance)
        _cnet.wrappedValue = NFKRAFTEncoder(outputDim: 256, norm: .batch)          // 128 hidden + 128 context
        _update.wrappedValue = NFKRAFTUpdateBlock()
    }

    private static func coordinateGrid(height h: Int, width w: Int) -> MLXArray {
        var values = [Float](repeating: 0, count: h * w * 2)
        for y in 0 ..< h {
            for x in 0 ..< w {
                values[(y * w + x) * 2] = Float(x)
                values[(y * w + x) * 2 + 1] = Float(y)
            }
        }
        return values.withUnsafeBufferPointer { MLXArray($0, [1, h, w, 2]) }
    }

    /// The eighth-resolution field the recurrent update converges to, `[H/8, W/8, 2]`. This is the
    /// network's own output; `flow` upsamples it to the input size.
    func flowLow(_ frame0: MLXArray, _ frame1: MLXArray) -> MLXArray {
        let (flow, _) = estimate(frame0, frame1)
        return flow[0]
    }

    /// Estimates flow at full resolution. `frame0`, `frame1` are `[1, H, W, 3]` in `0...1` (the image
    /// bridge's range); returns `[H, W, 2]`.
    func flow(_ frame0: MLXArray, _ frame1: MLXArray) -> MLXArray {
        let unbatched = frame0.ndim == 3
        let (h, w) = (frame0.shape[unbatched ? 0 : 1], frame0.shape[unbatched ? 1 : 2])
        let (flow, mask) = estimate(frame0, frame1)
        return Self.upsampled(flow, mask: mask)[0, 0 ..< h, 0 ..< w, 0...]
    }

    /// Expands the eighth-resolution field to full resolution the way the reference does: each output
    /// pixel is a convex combination of its coarse 3×3 neighborhood, weighted by the mask the final
    /// update predicts. Resampling instead leaves a blocky field, since a flow edge is a discontinuity
    /// the interpolation has no way to place.
    static func upsampled(_ flow: MLXArray, mask: MLXArray) -> MLXArray {
        let (n, h, w) = (flow.shape[0], flow.shape[1], flow.shape[2])
        let scale = 8
        // The mask's channel axis carries the neighborhood outside the subpixel block, so the
        // normalization runs over the nine neighbors of each output pixel.
        let weights = softmax(mask.reshaped([n, h, w, 9, scale * scale]), axis: 3)

        // The 3×3 neighborhood of every coarse cell. The reference unfolds a zero-padded field, so a
        // cell on the border combines with zeros rather than with a repeated edge.
        let bordered = NFKMLXResample.spatiallyPadded(flow, 1, value: 0)
        var planes: [MLXArray] = []
        for dy in 0 ... 2 {
            for dx in 0 ... 2 {
                planes.append(bordered[0..., dy ..< (dy + h), dx ..< (dx + w), 0...]
                    .reshaped([n, h, w, 1, 1, 2]))
            }
        }
        let stacked = concatenated(planes, axis: 3)                     // [n, h, w, 9, 1, 2]
        let combined = (weights.reshaped([n, h, w, 9, scale * scale, 1]) * stacked).sum(axis: 3)
        // The 64 combinations of each cell are its 8×8 block of output pixels.
        return (Float(scale) * combined).reshaped([n, h, w, scale, scale, 2])
            .transposed(0, 1, 3, 2, 4, 5).reshaped([n, h * scale, w * scale, 2])
    }

    /// Runs the recurrent update, returning the eighth-resolution flow `[1, H/8, W/8, 2]` and the
    /// upsampling mask the final iteration predicts.
    private func estimate(_ frame0: MLXArray, _ frame1: MLXArray) -> (flow: MLXArray, mask: MLXArray) {
        let unbatched = frame0.ndim == 3
        let (h, w) = (frame0.shape[unbatched ? 0 : 1], frame0.shape[unbatched ? 1 : 2])
        var img0 = unbatched ? frame0.reshaped([1, h, w, 3]) : frame0
        var img1 = unbatched ? frame1.reshaped([1, h, w, 3]) : frame1

        let multiple = 8
        let padH = (multiple - h % multiple) % multiple
        let padW = (multiple - w % multiple) % multiple
        if padH > 0 || padW > 0 {
            let widths = [IntOrPair((0, 0)), IntOrPair((0, padH)), IntOrPair((0, padW)), IntOrPair((0, 0))]
            img0 = padded(img0, widths: widths, mode: .edge)
            img1 = padded(img1, widths: widths, mode: .edge)
        }

        // The encoders are trained on `2 * (pixel / 255) - 1`; the bridge delivers `0...1`.
        img0 = 2 * img0 - 1
        img1 = 2 * img1 - 1

        let fmap1 = fnet(img0)
        let fmap2 = fnet(img1)
        let pyramid = NFKRAFTCorrelation.pyramid(fmap1, fmap2)
        let (fh, fw) = (fmap1.shape[1], fmap1.shape[2])

        let contextMaps = cnet(img0)
        let split = contextMaps.split(parts: 2, axis: 3)
        var net = tanh(split[0])
        let context = relu(split[1])

        let coords0 = Self.coordinateGrid(height: fh, width: fw)
        var coords1 = coords0
        var mask = MLXArray.zeros([1, fh, fw, 64 * 9])
        for _ in 0 ..< iterations {
            let corr = NFKRAFTCorrelation.lookup(pyramid, coords: coords1, height: fh, width: fw)
            let flowLow = coords1 - coords0
            let (newNet, upsamplingMask, deltaFlow) = update(net, context, corr, flowLow)
            net = newNet
            mask = upsamplingMask
            coords1 = coords1 + deltaFlow
        }
        return (coords1 - coords0, mask)
    }

    /// A packed flow map `[H, W, 3]` (mid-gray is no motion), for the image output.
    func flowMap(_ frame0: MLXArray, _ frame1: MLXArray, scale: Float = 32) -> MLXArray {
        let flow = self.flow(frame0, frame1)
        let (h, w) = (flow.shape[0], flow.shape[1])
        let packed = clip(flow / (2 * scale) + 0.5, min: 0, max: 1)
        let zero = MLXArray.zeros([h, w, 1])
        return concatenated([packed, zero], axis: 2)
    }
}

/// RAFT optical flow as an InferKit backend, and its registration for the Objective-C path.
@objc(NFKMLXRAFT)
public final class NFKMLXRAFT: NSObject {

    @objc public static let modelName = "raft"
    @objc public static let frame0Key = "frame0"
    @objc public static let frame1Key = "frame1"

    static func makeNet(iterations: Int = 6) -> NFKMLXRAFTNet {
        let net = NFKMLXRAFTNet(iterations: iterations)
        net.train(false)                                       // the cnet BatchNorm running statistics
        return net
    }

    /// Builds a RAFT optical-flow backend directly from optional local weights — no registry required.
    /// A nil `weightsURL` builds random weights (`isReady` is true). Two frames under `frame0Key` /
    /// `frame1Key`, a packed flow map under `NFKOutputImage`. Run
    /// inference off the render thread.
    @objc(backendWithWeightsURL:error:)
    public static func backend(weightsURL: URL?) throws -> any NFKInferenceBackend {
        let net = makeNet()
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        let holder = NFKMLXRAFTHolder(net)
        let configuration = NFKMLXTensorConfiguration(
            inputs: [NFKMLXTensorPort(key: frame0Key, tensorName: "frame0", channels: 3),
                     NFKMLXTensorPort(key: frame1Key, tensorName: "frame1", channels: 3)],
            outputs: [NFKMLXTensorPort(key: NFKOutputImage, tensorName: "flow")])
        return NFKMLXTensorBackend(identifier: modelName, configuration: configuration) { inputs in
            ["flow": holder.net.flowMap(inputs["frame0"]!, inputs["frame1"]!)]
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

    /// Registers RAFT (`raft`) with `NFKMLXModelRegistry`, delegating to `backend(weightsURL:)`.
    @objc public static func register() {
        NFKMLXModelRegistry.register(name: modelName) { weightsURL in try backend(weightsURL: weightsURL) }
    }

    /// Loads a safetensors checkpoint, transposing 4-D convolution weights to MLX's layout. `remap`
    /// maps the reference nested names to the module's keys (sweep task).
    static func loadWeights(into net: NFKMLXRAFTNet, from url: URL, remap: (String) -> String = { $0 }) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let raw = checkpoint.arrays
        let mapped = raw.map { key, value in
            (remap(key), checkpoint.needsConvTranspose && value.ndim == 4 ? value.transposed(0, 2, 3, 1) : value)
        }
        try NFKMLXWeights.apply(mapped, to: net)
    }
}

private final class NFKMLXRAFTHolder: @unchecked Sendable {
    let net: NFKMLXRAFTNet
    init(_ net: NFKMLXRAFTNet) { self.net = net }
}
