//
//  NFKMLXYOLOGraph.swift
//  InferKitMLX
//

import Foundation
import MLX
import MLXNN

// The YOLO generations after v8 are the same interpreter over different layer lists: ultralytics
// describes each release as a YAML graph of `(from, repeats, module, args)` rows and `parse_model`
// scales the widths and depths by the release's letter. This file carries those rows as Swift values
// and builds them the same way, so a released checkpoint's `model.<N>.…` keys land on array index N
// with no remap outside the detection head.

/// One row of an ultralytics model graph.
struct NFKYOLONode {
    /// The layer indices this row reads. `-1` is the previous row.
    let from: [Int]
    /// The row's repeat count before depth scaling.
    let repeats: Int
    let kind: Kind

    init(_ from: [Int], _ repeats: Int, _ kind: Kind) {
        self.from = from
        self.repeats = repeats
        self.kind = kind
    }

    /// The module kinds the shipped graphs use, with the reference's own argument order.
    enum Kind {
        case conv(out: Int, kernel: Int, stride: Int)
        case c2f(out: Int, shortcut: Bool)
        case c3k2(out: Int, c3k: Bool, expansion: Float, shortcut: Bool)
        case c3k2Attention(out: Int, expansion: Float)
        case a2c2f(out: Int, areaAttention: Bool, area: Int)
        case c2fCIB(out: Int, shortcut: Bool, largeKernel: Bool)
        case sppf(out: Int)
        case sppfShortcut(out: Int)
        case sppelan(out: Int, mid: Int)
        case c2psa(out: Int)
        case psa(out: Int)
        case scDown(out: Int, kernel: Int, stride: Int)
        case aConv(out: Int)
        case aDown(out: Int)
        case elan1(out: Int, mid: Int, branch: Int)
        case repNCSPELAN4(out: Int, mid: Int, branch: Int, repeats: Int)
        case upsample
        case concat
        case detect
        case identity
        /// `CBLinear`: one convolution whose output splits into the listed widths, so the row holds a
        /// list of tensors rather than one.
        case cbLinear(outs: [Int])
        /// `CBFuse`: takes the stated split from each source, resizes it to the last source's extent,
        /// and sums.
        case cbFuse(indices: [Int])
    }
}

/// The compound scaling constants a release's letter selects.
struct NFKYOLOScale: Sendable {
    let depth: Float
    let width: Float
    let maxChannels: Int

    init(_ depth: Float, _ width: Float, _ maxChannels: Int) {
        self.depth = depth
        self.width = width
        self.maxChannels = maxChannels
    }
}

/// The released YOLO generations this port covers, and the graph each one is.
public enum NFKMLXYOLOGeneration: Sendable {
    case v9, v10, v11, v12, v26
}

/// The scale letter a release carries.
public enum NFKMLXYOLOScaleLetter: String, Sendable {
    case tiny = "t", nano = "n", small = "s", medium = "m", compact = "c",
         balanced = "b", large = "l", extended = "e", extraLarge = "x"
}

enum NFKYOLOGraphs {

    static func makeDivisible(_ value: Float, _ divisor: Int = 8) -> Int {
        Int((value / Float(divisor)).rounded(.up)) * divisor
    }

    /// The reference's per-letter constants. YOLOv9 states its widths directly rather than scaling,
    /// so its rows carry a unit scale and the graph itself differs per size.
    static func scale(_ generation: NFKMLXYOLOGeneration, _ letter: NFKMLXYOLOScaleLetter) -> NFKYOLOScale {
        switch generation {
        case .v9:
            return NFKYOLOScale(1, 1, 10_000)
        case .v10:
            switch letter {
            case .nano: return NFKYOLOScale(0.33, 0.25, 1024)
            case .small: return NFKYOLOScale(0.33, 0.50, 1024)
            case .medium: return NFKYOLOScale(0.67, 0.75, 768)
            case .balanced: return NFKYOLOScale(0.67, 1.00, 512)
            case .large: return NFKYOLOScale(1.00, 1.00, 512)
            default: return NFKYOLOScale(1.00, 1.25, 512)
            }
        case .v11, .v12, .v26:
            switch letter {
            case .nano: return NFKYOLOScale(0.50, 0.25, 1024)
            case .small: return NFKYOLOScale(0.50, 0.50, 1024)
            case .medium: return NFKYOLOScale(0.50, 1.00, 512)
            case .large: return NFKYOLOScale(1.00, 1.00, 512)
            default: return NFKYOLOScale(1.00, 1.50, 512)
            }
        }
    }

    /// Whether a release's `C3k2` stages use the `C3k` unit, which the reference switches on for the
    /// medium and larger letters.
    static func usesC3k(_ letter: NFKMLXYOLOScaleLetter) -> Bool {
        [.medium, .large, .extraLarge].contains(letter)
    }

    /// Whether a release's `A2C2f` stages carry the residual gate, which the reference switches on for
    /// the large and extra-large letters.
    static func usesA2C2fResidual(_ letter: NFKMLXYOLOScaleLetter) -> Bool {
        [.large, .extraLarge].contains(letter)
    }

    /// The detection head's class-branch shape. The reference tracks a `legacy` flag that any of
    /// `C3k2`, `A2C2f` or `C2fCIB` clears, so it is exactly "not YOLOv9" among these generations.
    static func usesLegacyHead(_ generation: NFKMLXYOLOGeneration) -> Bool { generation == .v9 }

    /// Whether the head runs the one-to-one branch and needs no suppression.
    static func isEndToEnd(_ generation: NFKMLXYOLOGeneration) -> Bool {
        generation == .v10 || generation == .v26
    }

    /// The distribution-focal bin count. YOLO26 regresses the box directly, with one bin.
    static func regMax(_ generation: NFKMLXYOLOGeneration) -> Int { generation == .v26 ? 1 : 16 }
}

extension NFKYOLOGraphs {

    /// YOLO11: a `C3k2` backbone closed by SPPF and `C2PSA`, and a `C3k2` PAN-FPN neck.
    static func yolo11(_ letter: NFKMLXYOLOScaleLetter) -> [NFKYOLONode] {
        let c3k = usesC3k(letter)
        return [
            NFKYOLONode([-1], 1, .conv(out: 64, kernel: 3, stride: 2)),
            NFKYOLONode([-1], 1, .conv(out: 128, kernel: 3, stride: 2)),
            NFKYOLONode([-1], 2, .c3k2(out: 256, c3k: c3k, expansion: 0.25, shortcut: true)),
            NFKYOLONode([-1], 1, .conv(out: 256, kernel: 3, stride: 2)),
            NFKYOLONode([-1], 2, .c3k2(out: 512, c3k: c3k, expansion: 0.25, shortcut: true)),
            NFKYOLONode([-1], 1, .conv(out: 512, kernel: 3, stride: 2)),
            NFKYOLONode([-1], 2, .c3k2(out: 512, c3k: true, expansion: 0.5, shortcut: true)),
            NFKYOLONode([-1], 1, .conv(out: 1024, kernel: 3, stride: 2)),
            NFKYOLONode([-1], 2, .c3k2(out: 1024, c3k: true, expansion: 0.5, shortcut: true)),
            NFKYOLONode([-1], 1, .sppf(out: 1024)),
            NFKYOLONode([-1], 2, .c2psa(out: 1024)),
            NFKYOLONode([-1], 1, .upsample),
            NFKYOLONode([-1, 6], 1, .concat),
            NFKYOLONode([-1], 2, .c3k2(out: 512, c3k: c3k, expansion: 0.5, shortcut: true)),
            NFKYOLONode([-1], 1, .upsample),
            NFKYOLONode([-1, 4], 1, .concat),
            NFKYOLONode([-1], 2, .c3k2(out: 256, c3k: c3k, expansion: 0.5, shortcut: true)),
            NFKYOLONode([-1], 1, .conv(out: 256, kernel: 3, stride: 2)),
            NFKYOLONode([-1, 13], 1, .concat),
            NFKYOLONode([-1], 2, .c3k2(out: 512, c3k: c3k, expansion: 0.5, shortcut: true)),
            NFKYOLONode([-1], 1, .conv(out: 512, kernel: 3, stride: 2)),
            NFKYOLONode([-1, 10], 1, .concat),
            NFKYOLONode([-1], 2, .c3k2(out: 1024, c3k: true, expansion: 0.5, shortcut: true)),
            NFKYOLONode([16, 19, 22], 1, .detect),
        ]
    }

    /// YOLOv12: YOLO11's skeleton with the deeper stages replaced by area attention.
    static func yolo12(_ letter: NFKMLXYOLOScaleLetter) -> [NFKYOLONode] {
        let c3k = usesC3k(letter)
        return [
            NFKYOLONode([-1], 1, .conv(out: 64, kernel: 3, stride: 2)),
            NFKYOLONode([-1], 1, .conv(out: 128, kernel: 3, stride: 2)),
            NFKYOLONode([-1], 2, .c3k2(out: 256, c3k: c3k, expansion: 0.25, shortcut: true)),
            NFKYOLONode([-1], 1, .conv(out: 256, kernel: 3, stride: 2)),
            NFKYOLONode([-1], 2, .c3k2(out: 512, c3k: c3k, expansion: 0.25, shortcut: true)),
            NFKYOLONode([-1], 1, .conv(out: 512, kernel: 3, stride: 2)),
            NFKYOLONode([-1], 4, .a2c2f(out: 512, areaAttention: true, area: 4)),
            NFKYOLONode([-1], 1, .conv(out: 1024, kernel: 3, stride: 2)),
            NFKYOLONode([-1], 4, .a2c2f(out: 1024, areaAttention: true, area: 1)),
            NFKYOLONode([-1], 1, .upsample),
            NFKYOLONode([-1, 6], 1, .concat),
            NFKYOLONode([-1], 2, .a2c2f(out: 512, areaAttention: false, area: -1)),
            NFKYOLONode([-1], 1, .upsample),
            NFKYOLONode([-1, 4], 1, .concat),
            NFKYOLONode([-1], 2, .a2c2f(out: 256, areaAttention: false, area: -1)),
            NFKYOLONode([-1], 1, .conv(out: 256, kernel: 3, stride: 2)),
            NFKYOLONode([-1, 11], 1, .concat),
            NFKYOLONode([-1], 2, .a2c2f(out: 512, areaAttention: false, area: -1)),
            NFKYOLONode([-1], 1, .conv(out: 512, kernel: 3, stride: 2)),
            NFKYOLONode([-1, 8], 1, .concat),
            NFKYOLONode([-1], 2, .c3k2(out: 1024, c3k: true, expansion: 0.5, shortcut: true)),
            NFKYOLONode([14, 17, 20], 1, .detect),
        ]
    }

    /// YOLOv10: a `C2f` backbone with the separable downsamples and the partial self-attention stage,
    /// closing on the one-to-one head. Which stages run `C2fCIB` instead of `C2f`, and which of those
    /// take the large-kernel branch, is stated per size rather than derived, because the released
    /// configs differ stage by stage.
    static func yolo10(_ letter: NFKMLXYOLOScaleLetter) -> [NFKYOLONode] {
        // Layer index -> whether that stage's `C2fCIB` uses the large-kernel `RepVGGDW` branch.
        let cib: [Int: Bool]
        switch letter {
        case .nano: cib = [22: true]
        case .small: cib = [8: true, 22: true]
        case .medium: cib = [8: false, 19: false, 22: false]
        case .balanced, .large: cib = [8: false, 13: false, 19: false, 22: false]
        default: cib = [6: false, 8: false, 13: false, 19: false, 22: false]
        }
        func stage(_ index: Int, _ out: Int, _ shortcut: Bool) -> NFKYOLONode.Kind {
            guard let largeKernel = cib[index] else { return .c2f(out: out, shortcut: shortcut) }
            return .c2fCIB(out: out, shortcut: true, largeKernel: largeKernel)
        }
        return [
            NFKYOLONode([-1], 1, .conv(out: 64, kernel: 3, stride: 2)),
            NFKYOLONode([-1], 1, .conv(out: 128, kernel: 3, stride: 2)),
            NFKYOLONode([-1], 3, stage(2, 128, true)),
            NFKYOLONode([-1], 1, .conv(out: 256, kernel: 3, stride: 2)),
            NFKYOLONode([-1], 6, stage(4, 256, true)),
            NFKYOLONode([-1], 1, .scDown(out: 512, kernel: 3, stride: 2)),
            NFKYOLONode([-1], 6, stage(6, 512, true)),
            NFKYOLONode([-1], 1, .scDown(out: 1024, kernel: 3, stride: 2)),
            NFKYOLONode([-1], 3, stage(8, 1024, true)),
            NFKYOLONode([-1], 1, .sppf(out: 1024)),
            NFKYOLONode([-1], 1, .psa(out: 1024)),
            NFKYOLONode([-1], 1, .upsample),
            NFKYOLONode([-1, 6], 1, .concat),
            NFKYOLONode([-1], 3, stage(13, 512, false)),
            NFKYOLONode([-1], 1, .upsample),
            NFKYOLONode([-1, 4], 1, .concat),
            NFKYOLONode([-1], 3, stage(16, 256, false)),
            NFKYOLONode([-1], 1, .conv(out: 256, kernel: 3, stride: 2)),
            NFKYOLONode([-1, 13], 1, .concat),
            NFKYOLONode([-1], 3, stage(19, 512, false)),
            NFKYOLONode([-1], 1, .scDown(out: 512, kernel: 3, stride: 2)),
            NFKYOLONode([-1, 10], 1, .concat),
            NFKYOLONode([-1], 3, stage(22, 1024, true)),
            NFKYOLONode([16, 19, 22], 1, .detect),
        ]
    }

    /// YOLO26: YOLO11's skeleton with the shortcut SPPF, attention `C3k2` at the deepest neck stage,
    /// and the end-to-end head that regresses the box directly.
    static func yolo26(_ letter: NFKMLXYOLOScaleLetter) -> [NFKYOLONode] {
        let c3k = usesC3k(letter)
        return [
            NFKYOLONode([-1], 1, .conv(out: 64, kernel: 3, stride: 2)),
            NFKYOLONode([-1], 1, .conv(out: 128, kernel: 3, stride: 2)),
            NFKYOLONode([-1], 2, .c3k2(out: 256, c3k: c3k, expansion: 0.25, shortcut: true)),
            NFKYOLONode([-1], 1, .conv(out: 256, kernel: 3, stride: 2)),
            NFKYOLONode([-1], 2, .c3k2(out: 512, c3k: c3k, expansion: 0.25, shortcut: true)),
            NFKYOLONode([-1], 1, .conv(out: 512, kernel: 3, stride: 2)),
            NFKYOLONode([-1], 2, .c3k2(out: 512, c3k: true, expansion: 0.5, shortcut: true)),
            NFKYOLONode([-1], 1, .conv(out: 1024, kernel: 3, stride: 2)),
            NFKYOLONode([-1], 2, .c3k2(out: 1024, c3k: true, expansion: 0.5, shortcut: true)),
            NFKYOLONode([-1], 1, .sppfShortcut(out: 1024)),
            NFKYOLONode([-1], 2, .c2psa(out: 1024)),
            NFKYOLONode([-1], 1, .upsample),
            NFKYOLONode([-1, 6], 1, .concat),
            NFKYOLONode([-1], 2, .c3k2(out: 512, c3k: true, expansion: 0.5, shortcut: true)),
            NFKYOLONode([-1], 1, .upsample),
            NFKYOLONode([-1, 4], 1, .concat),
            NFKYOLONode([-1], 2, .c3k2(out: 256, c3k: true, expansion: 0.5, shortcut: true)),
            NFKYOLONode([-1], 1, .conv(out: 256, kernel: 3, stride: 2)),
            NFKYOLONode([-1, 13], 1, .concat),
            NFKYOLONode([-1], 2, .c3k2(out: 512, c3k: true, expansion: 0.5, shortcut: true)),
            NFKYOLONode([-1], 1, .conv(out: 512, kernel: 3, stride: 2)),
            NFKYOLONode([-1, 10], 1, .concat),
            NFKYOLONode([-1], 1, .c3k2Attention(out: 1024, expansion: 0.5)),
            NFKYOLONode([16, 19, 22], 1, .detect),
        ]
    }
}

extension NFKYOLOGraphs {

    /// YOLOv9's GELAN graph. The released sizes state their channel counts directly rather than
    /// scaling one graph by a letter, so each size is its own row list. The four here share a shape:
    /// a stem, four downsample-plus-`RepNCSPELAN4` stages, `SPPELAN`, and a `RepNCSPELAN4` neck.
    static func yolo9(_ letter: NFKMLXYOLOScaleLetter) -> [NFKYOLONode] {
        // The smaller sizes chain their stages through `AConv` and open on an `ELAN1`; the compact one
        // uses `ADown` and a `RepNCSPELAN4` throughout. The inner repeat count is the stage's own,
        // stated per size rather than scaled by a letter.
        switch letter {
        case .tiny: return yolo9Rows(repeats: 3, downsampleIsADown: false, stage2IsELAN1: true, letter: letter)
        case .small: return yolo9Rows(repeats: 3, downsampleIsADown: false, stage2IsELAN1: true, letter: letter)
        case .medium: return yolo9Rows(repeats: 1, downsampleIsADown: false, stage2IsELAN1: false, letter: letter)
        case .extended: return yolo9Extended()
        default: return yolo9Rows(repeats: 1, downsampleIsADown: true, stage2IsELAN1: false, letter: letter)
        }
    }

    /// YOLOv9e: the extended release, which keeps its programmable-gradient branch at inference. A
    /// `CBLinear` taps each backbone stage into per-level splits and a `CBFuse` sums the matching
    /// split into the main path, so the graph carries rows that produce several tensors.
    static func yolo9Extended() -> [NFKYOLONode] {
        func elan(_ out: Int, _ mid: Int, _ branch: Int) -> NFKYOLONode.Kind {
            .repNCSPELAN4(out: out, mid: mid, branch: branch, repeats: 2)
        }
        return [
            NFKYOLONode([-1], 1, .identity),
            NFKYOLONode([-1], 1, .conv(out: 64, kernel: 3, stride: 2)),
            NFKYOLONode([-1], 1, .conv(out: 128, kernel: 3, stride: 2)),
            NFKYOLONode([-1], 1, elan(256, 128, 64)),
            NFKYOLONode([-1], 1, .aDown(out: 256)),
            NFKYOLONode([-1], 1, elan(512, 256, 128)),
            NFKYOLONode([-1], 1, .aDown(out: 512)),
            NFKYOLONode([-1], 1, elan(1024, 512, 256)),
            NFKYOLONode([-1], 1, .aDown(out: 1024)),
            NFKYOLONode([-1], 1, elan(1024, 512, 256)),
            NFKYOLONode([1], 1, .cbLinear(outs: [64])),
            NFKYOLONode([3], 1, .cbLinear(outs: [64, 128])),
            NFKYOLONode([5], 1, .cbLinear(outs: [64, 128, 256])),
            NFKYOLONode([7], 1, .cbLinear(outs: [64, 128, 256, 512])),
            NFKYOLONode([9], 1, .cbLinear(outs: [64, 128, 256, 512, 1024])),
            NFKYOLONode([0], 1, .conv(out: 64, kernel: 3, stride: 2)),
            NFKYOLONode([10, 11, 12, 13, 14, -1], 1, .cbFuse(indices: [0, 0, 0, 0, 0])),
            NFKYOLONode([-1], 1, .conv(out: 128, kernel: 3, stride: 2)),
            NFKYOLONode([11, 12, 13, 14, -1], 1, .cbFuse(indices: [1, 1, 1, 1])),
            NFKYOLONode([-1], 1, elan(256, 128, 64)),
            NFKYOLONode([-1], 1, .aDown(out: 256)),
            NFKYOLONode([12, 13, 14, -1], 1, .cbFuse(indices: [2, 2, 2])),
            NFKYOLONode([-1], 1, elan(512, 256, 128)),
            NFKYOLONode([-1], 1, .aDown(out: 512)),
            NFKYOLONode([13, 14, -1], 1, .cbFuse(indices: [3, 3])),
            NFKYOLONode([-1], 1, elan(1024, 512, 256)),
            NFKYOLONode([-1], 1, .aDown(out: 1024)),
            NFKYOLONode([14, -1], 1, .cbFuse(indices: [4])),
            NFKYOLONode([-1], 1, elan(1024, 512, 256)),
            NFKYOLONode([-1], 1, .sppelan(out: 512, mid: 256)),
            NFKYOLONode([-1], 1, .upsample),
            NFKYOLONode([-1, 25], 1, .concat),
            NFKYOLONode([-1], 1, elan(512, 512, 256)),
            NFKYOLONode([-1], 1, .upsample),
            NFKYOLONode([-1, 22], 1, .concat),
            NFKYOLONode([-1], 1, elan(256, 256, 128)),
            NFKYOLONode([-1], 1, .aDown(out: 256)),
            NFKYOLONode([-1, 32], 1, .concat),
            NFKYOLONode([-1], 1, elan(512, 512, 256)),
            NFKYOLONode([-1], 1, .aDown(out: 512)),
            NFKYOLONode([-1, 29], 1, .concat),
            NFKYOLONode([-1], 1, elan(512, 1024, 512)),
            NFKYOLONode([35, 38, 41], 1, .detect),
        ]
    }

    private static func yolo9Rows(repeats: Int, downsampleIsADown: Bool,
                                  stage2IsELAN1: Bool, letter: NFKMLXYOLOScaleLetter) -> [NFKYOLONode] {
        // The compact release states (out, mid, branch) triples that do not follow the smaller sizes'
        // "out == mid" rule, so its stages are named rather than derived.
        let stages: [(out: Int, mid: Int, branch: Int)]
        let downsamples: [Int]
        let neck: [(out: Int, mid: Int, branch: Int)]
        let poolOut: Int, poolMid: Int
        switch letter {
        case .tiny:
            stages = [(32, 32, 16), (64, 64, 32), (96, 96, 48), (128, 128, 64)]
            downsamples = [64, 96, 128, 48, 64]
            neck = [(96, 96, 48), (64, 64, 32), (96, 96, 48), (128, 128, 64)]
            poolOut = 128; poolMid = 64
        case .small:
            stages = [(64, 64, 32), (128, 128, 64), (192, 192, 96), (256, 256, 128)]
            downsamples = [128, 192, 256, 96, 128]
            neck = [(192, 192, 96), (128, 128, 64), (192, 192, 96), (256, 256, 128)]
            poolOut = 256; poolMid = 128
        case .medium:
            stages = [(128, 128, 64), (240, 240, 120), (360, 360, 180), (480, 480, 240)]
            downsamples = [240, 360, 480, 180, 240]
            neck = [(360, 360, 180), (240, 240, 120), (360, 360, 180), (480, 480, 240)]
            poolOut = 480; poolMid = 240
        default:
            stages = [(256, 128, 64), (512, 256, 128), (512, 512, 256), (512, 512, 256)]
            downsamples = [256, 512, 512, 256, 512]
            neck = [(512, 512, 256), (256, 256, 128), (512, 512, 256), (512, 512, 256)]
            poolOut = 512; poolMid = 256
        }
        func down(_ out: Int) -> NFKYOLONode.Kind {
            downsampleIsADown ? .aDown(out: out) : .aConv(out: out)
        }
        let inner = repeats
        let stem = letter == .tiny ? (16, 32) : letter == .small ? (32, 64)
                 : letter == .medium ? (32, 64) : (64, 128)
        var rows: [NFKYOLONode] = [
            NFKYOLONode([-1], 1, .conv(out: stem.0, kernel: 3, stride: 2)),
            NFKYOLONode([-1], 1, .conv(out: stem.1, kernel: 3, stride: 2)),
        ]
        rows.append(stage2IsELAN1
            ? NFKYOLONode([-1], 1, .elan1(out: stages[0].out, mid: stages[0].mid, branch: stages[0].branch))
            : NFKYOLONode([-1], 1, .repNCSPELAN4(out: stages[0].out, mid: stages[0].mid,
                                                 branch: stages[0].branch, repeats: inner)))
        for index in 1 ..< 4 {
            rows.append(NFKYOLONode([-1], 1, down(downsamples[index - 1])))
            rows.append(NFKYOLONode([-1], 1, .repNCSPELAN4(out: stages[index].out, mid: stages[index].mid,
                                                           branch: stages[index].branch, repeats: inner)))
        }
        rows.append(NFKYOLONode([-1], 1, .sppelan(out: poolOut, mid: poolMid)))
        rows.append(NFKYOLONode([-1], 1, .upsample))
        rows.append(NFKYOLONode([-1, 6], 1, .concat))
        rows.append(NFKYOLONode([-1], 1, .repNCSPELAN4(out: neck[0].out, mid: neck[0].mid,
                                                       branch: neck[0].branch, repeats: inner)))
        rows.append(NFKYOLONode([-1], 1, .upsample))
        rows.append(NFKYOLONode([-1, 4], 1, .concat))
        rows.append(NFKYOLONode([-1], 1, .repNCSPELAN4(out: neck[1].out, mid: neck[1].mid,
                                                       branch: neck[1].branch, repeats: inner)))
        rows.append(NFKYOLONode([-1], 1, down(downsamples[3])))
        rows.append(NFKYOLONode([-1, 12], 1, .concat))
        rows.append(NFKYOLONode([-1], 1, .repNCSPELAN4(out: neck[2].out, mid: neck[2].mid,
                                                       branch: neck[2].branch, repeats: inner)))
        rows.append(NFKYOLONode([-1], 1, down(downsamples[4])))
        rows.append(NFKYOLONode([-1, 9], 1, .concat))
        rows.append(NFKYOLONode([-1], 1, .repNCSPELAN4(out: neck[3].out, mid: neck[3].mid,
                                                       branch: neck[3].branch, repeats: inner)))
        rows.append(NFKYOLONode([15, 18, 21], 1, .detect))
        return rows
    }
}
