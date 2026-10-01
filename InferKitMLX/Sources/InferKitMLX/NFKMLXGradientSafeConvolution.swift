//
//  NFKMLXGradientSafeConvolution.swift
//  InferKitMLX
//
//  The input gradient of a wide stride-1 convolution, computed so the GPU gets it right.
//
//  mlx 0.32.2 (the core the package's mlx-swift revision vendors) computes the gradient of `conv1d`,
//  `conv2d`, and `conv3d` with respect to their INPUT wrongly on the GPU when the convolution runs at
//  stride 1 and a kernel axis has more than 16 taps. The forward and the weight gradient stay exact, and
//  the CPU is exact. Measured against the CPU on random tensors: a 17-tap `conv1d` at cosine 0.92, a
//  32-tap one with 32 input channels at 0.22, a 3×39 `conv2d` at 0.80, a 1×1×17 `conv3d` at 0.88. Every
//  stride above 1 on any axis was exact, every transposed convolution was exact, and a kernel of 16 taps
//  or fewer was exact. The failing channel counts follow no rule the measurements support, so the
//  workaround keys on stride and tap count alone.
//
//  The workaround sums convolutions over slices of the kernel at most 16 taps wide, each over the window
//  of the explicitly padded input it reaches. That is the same sum in a different order, and its input
//  gradient matches the CPU's.
//
//  In a training run every parameter after such a convolution reads parity, because its gradient needs
//  only the forward and the output gradient. Only parameters before it drift. `NFKMLXTrainer` therefore
//  swaps every affected convolution for its sliced form for the length of a run and swaps it back
//  afterward, so inference never changes.
//

import Foundation
import MLX
import MLXNN

/// Wide stride-1 convolutions computed over kernel slices of at most 16 taps, the form whose input
/// gradient is exact on the GPU.
///
/// Introduced in InferKit 0.5.0.
public enum NFKMLXGradientSafeConvolution {

    /// The widest kernel axis the GPU differentiates exactly.
    public static let maximumTaps = 16

    /// Whether a convolution with this kernel and stride is one the GPU differentiates wrongly.
    public static func needsSlicing(kernel: [Int], stride: [Int]) -> Bool {
        stride.allSatisfy { $0 == 1 } && kernel.contains { $0 > maximumTaps }
    }

    /// The convolution of `x` (`[B, spatial…, C]`) with `weight` (`[O, kernel…, C / groups]`) at stride 1,
    /// summed over kernel slices of at most 16 taps. Equal to MLX's own convolution up to summation
    /// order.
    public static func convolve(_ x: MLXArray, _ weight: MLXArray, padding: [Int], dilation: [Int],
                                groups: Int) -> MLXArray {
        let axes = weight.ndim - 2
        let kernel = Array(weight.shape[1 ... axes])
        var padded = x
        if padding.contains(where: { $0 > 0 }) {
            let widths = [IntOrPair((0, 0))] + padding.map { IntOrPair(($0, $0)) } + [IntOrPair((0, 0))]
            padded = MLX.padded(x, widths: widths)
        }
        let outputs = (0 ..< axes).map { padded.dim($0 + 1) - dilation[$0] * (kernel[$0] - 1) }
        let slices = kernel.map { taps in
            Swift.stride(from: 0, to: taps, by: maximumTaps).map { ($0, min(maximumTaps, taps - $0)) }
        }

        var total: MLXArray?
        func visit(_ axis: Int, _ input: MLXArray, _ part: MLXArray) {
            guard axis < axes else {
                let term = unsliced(input, part, dilation: dilation, groups: groups)
                total = total.map { $0 + term } ?? term
                return
            }
            for (start, taps) in slices[axis] {
                let whole = taps == kernel[axis]
                let window = outputs[axis] + dilation[axis] * (taps - 1)
                visit(axis + 1,
                      whole ? input : slice(input, axis: axis + 1, from: start * dilation[axis], count: window),
                      whole ? part : slice(part, axis: axis + 1, from: start, count: taps))
            }
        }
        visit(0, padded, weight)
        return total!
    }

    /// One unpadded stride-1 convolution by MLX's own operator.
    private static func unsliced(_ x: MLXArray, _ weight: MLXArray, dilation: [Int], groups: Int) -> MLXArray {
        switch dilation.count {
        case 1:
            return conv1d(x, weight, stride: 1, padding: 0, dilation: dilation[0], groups: groups)
        case 2:
            return conv2d(x, weight, stride: 1, padding: 0, dilation: .init((dilation[0], dilation[1])),
                          groups: groups)
        default:
            return conv3d(x, weight, stride: 1, padding: 0,
                          dilation: .init((dilation[0], dilation[1], dilation[2])), groups: groups)
        }
    }

    private static func slice(_ x: MLXArray, axis: Int, from start: Int, count: Int) -> MLXArray {
        split(x, indices: [start, start + count], axis: axis)[1]
    }

    /// The sliced convolution plus its bias, rounded as the module it replaces rounds a half-precision
    /// input. MLXNN's convolutions round the convolution and then the bias; the package's `NFKConv1d` and
    /// `NFKConv2d` accumulate both in float32 and round once, as torch does. Either way the slices are
    /// summed in float32, so slicing adds no rounding of its own.
    static func apply(_ x: MLXArray, weight: MLXArray, bias: MLXArray?, padding: [Int], dilation: [Int],
                      groups: Int, roundsOnce: Bool) -> MLXArray {
        guard NFKReferenceRounding.isReduced(x) else {
            let y = convolve(x, weight, padding: padding, dilation: dilation, groups: groups)
            return bias.map { y + $0 } ?? y
        }
        let y = convolve(x.asType(.float32), weight.asType(.float32), padding: padding, dilation: dilation,
                         groups: groups)
        guard roundsOnce else {
            return bias.map { y.asType(x.dtype) + $0 } ?? y.asType(x.dtype)
        }
        return (bias.map { y + $0.asType(.float32) } ?? y).asType(x.dtype)
    }

    // MARK: Swapping a model's convolutions for the run

    /// The convolutions swapped out of a model, held so they can go back in.
    public final class Installation {
        fileprivate let model: Module
        fileprivate let swaps: [(path: String, original: Module, replacement: Module)]

        fileprivate init(model: Module, swaps: [(path: String, original: Module, replacement: Module)]) {
            self.model = model
            self.swaps = swaps
        }

        /// The paths of the convolutions that were swapped.
        public var paths: [String] { swaps.map { $0.path } }

        /// Copies what the run trained back into the original convolutions and puts them back.
        public func restore() {
            for swap in swaps.reversed() {
                swap.original.update(parameters: swap.replacement.parameters())
                swap.original.train(swap.replacement.training)
                // The path took a module when the swap went in, so it takes the original back.
                _ = try? NFKMLXModuleReplacement.place(swap.original, at: swap.path, in: model)
            }
        }
    }

    /// Swaps every stride-1 convolution in `model` with a kernel axis wider than 16 taps for its sliced
    /// form. Each replacement carries the original's parameters, frozen state, and mode, and computes the
    /// same function. Call ``Installation/restore()`` when the gradients are done.
    ///
    /// `NFKMLXTrainer` does this for every run. A training loop of the caller's own, built directly on
    /// `valueAndGrad`, calls it around its steps.
    ///
    /// - Throws: `NFKMLXError.unsupportedConfiguration` when MLX cannot replace an affected
    ///   convolution, which is the case for one held in a plain property. Its gradient on the GPU would
    ///   be wrong, so the run is refused. A child declared with `@ModuleInfo`, directly or inside an
    ///   array of modules, swaps.
    public static func install(in model: Module) throws -> Installation {
        var swaps = [(path: String, original: Module, replacement: Module)]()
        var refused = [String]()
        for (path, module) in model.leafModules().flattened() {
            guard let replacement = replacement(for: module) else { continue }
            replacement.update(parameters: module.parameters())
            let frozen = Array(module.noGrad())
            if !frozen.isEmpty {
                replacement.freeze(recursive: false, keys: frozen)
            }
            replacement.train(module.training)
            do {
                try NFKMLXModuleReplacement.place(replacement, at: path, in: model)
                swaps.append((path, module, replacement))
            } catch {
                refused.append(path)
            }
        }
        let installation = Installation(model: model, swaps: swaps)
        guard refused.isEmpty else {
            installation.restore()
            throw NFKMLXError.unsupportedConfiguration(
                "MLX could not replace \(refused.count) stride-1 convolution(s) wider than \(maximumTaps) taps "
                + "(\(refused.prefix(3).joined(separator: ", "))\(refused.count > 3 ? ", …" : "")). The GPU computes "
                + "their input gradient wrongly in this MLX, so the run is refused. MLX replaces a child declared "
                + "with @ModuleInfo, directly or inside an array of modules; a plain property it cannot.")
        }
        return installation
    }

    /// The sliced stand-in for `module`, or nil when it needs none. A replacement's initializer draws a
    /// random weight that its parameters then overwrite; it draws from a private state, so the run's own
    /// random stream stays where the caller left it.
    static func replacement(for module: Module) -> Module? {
        if type(of: module) == Conv1d.self || module is NFKConv1d, let convolution = module as? Conv1d,
           needsSlicing(kernel: [convolution.weight.dim(1)], stride: [convolution.stride]) {
            return withRandomState(MLXRandom.RandomState(seed: 0)) {
                NFKMLXSlicedConv1d(convolution, roundsOnce: module is NFKConv1d)
            }
        }
        if type(of: module) == Conv2d.self || module is NFKConv2d, let convolution = module as? Conv2d,
           needsSlicing(kernel: [convolution.weight.dim(1), convolution.weight.dim(2)],
                        stride: [convolution.stride.0, convolution.stride.1]) {
            return withRandomState(MLXRandom.RandomState(seed: 0)) {
                NFKMLXSlicedConv2d(convolution, roundsOnce: module is NFKConv2d)
            }
        }
        if type(of: module) == Conv3d.self, let convolution = module as? Conv3d,
           needsSlicing(kernel: [convolution.weight.dim(1), convolution.weight.dim(2), convolution.weight.dim(3)],
                        stride: [convolution.stride.0, convolution.stride.1, convolution.stride.2]) {
            return withRandomState(MLXRandom.RandomState(seed: 0)) {
                NFKMLXSlicedConv3d(convolution)
            }
        }
        return nil
    }
}

/// A `Conv1d` computed over kernel slices of at most 16 taps. Its parameters keep the replaced
/// convolution's keys, so a checkpoint written mid-run loads into the original.
final class NFKMLXSlicedConv1d: Conv1d {
    let roundsOnce: Bool

    init(_ base: Conv1d, roundsOnce: Bool) {
        self.roundsOnce = roundsOnce
        let shape = base.weight.shape                                           // [out, k, in / groups]
        super.init(inputChannels: shape[2] * base.groups, outputChannels: shape[0], kernelSize: shape[1],
                   stride: base.stride, padding: base.padding, dilation: base.dilation, groups: base.groups,
                   bias: base.bias != nil)
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        NFKMLXGradientSafeConvolution.apply(x, weight: weight, bias: bias, padding: [padding],
                                            dilation: [dilation], groups: groups, roundsOnce: roundsOnce)
    }
}

/// A `Conv2d` computed over kernel slices of at most 16 taps.
final class NFKMLXSlicedConv2d: Conv2d {
    let roundsOnce: Bool

    init(_ base: Conv2d, roundsOnce: Bool) {
        self.roundsOnce = roundsOnce
        let shape = base.weight.shape                                           // [out, kH, kW, in / groups]
        super.init(inputChannels: shape[3] * base.groups, outputChannels: shape[0],
                   kernelSize: .init((shape[1], shape[2])), stride: .init(base.stride),
                   padding: .init(base.padding), dilation: .init(base.dilation), groups: base.groups,
                   bias: base.bias != nil)
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        NFKMLXGradientSafeConvolution.apply(x, weight: weight, bias: bias, padding: [padding.0, padding.1],
                                            dilation: [dilation.0, dilation.1], groups: groups,
                                            roundsOnce: roundsOnce)
    }
}

/// A `Conv3d` computed over kernel slices of at most 16 taps.
final class NFKMLXSlicedConv3d: Conv3d {
    init(_ base: Conv3d) {
        let shape = base.weight.shape                                           // [out, kD, kH, kW, in / groups]
        super.init(inputChannels: shape[4] * base.groups, outputChannels: shape[0],
                   kernelSize: .init((shape[1], shape[2], shape[3])), stride: .init(base.stride),
                   padding: .init(base.padding), dilation: .init(base.dilation), groups: base.groups,
                   bias: base.bias != nil)
    }

    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        NFKMLXGradientSafeConvolution.apply(x, weight: weight, bias: bias,
                                            padding: [padding.0, padding.1, padding.2],
                                            dilation: [dilation.0, dilation.1, dilation.2], groups: groups,
                                            roundsOnce: false)
    }
}
