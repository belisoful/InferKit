//
//  NFKMLXYOLOGenerations.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXNN

/// The released detector this port builds: a generation and the size letter its checkpoint carries.
@objc(NFKMLXYOLORelease)
public enum NFKMLXYOLORelease: Int {
    case v9Tiny, v9Small, v9Medium, v9Compact, v9Extended
    case v10Nano, v10Small, v10Medium, v10Balanced, v10Large, v10ExtraLarge
    case v11Nano, v11Small, v11Medium, v11Large, v11ExtraLarge
    case v12Nano, v12Small, v12Medium, v12Large, v12ExtraLarge
    case v26Nano, v26Small, v26Medium, v26Large, v26ExtraLarge
}

/// YOLOv9 through YOLO26 as InferKit detection backends.
///
/// The shipped `NFKMLXYOLO` is v8, whose graph is written out in Swift. These generations are built
/// from their reference layer rows instead, so one interpreter covers every release and a checkpoint's
/// `model.<N>.…` keys land on array index N.
@objc(NFKMLXYOLOGenerations)
public final class NFKMLXYOLOGenerations: NSObject {

    /// The registry name a release builds under, which is the reference's own checkpoint stem.
    @objc(modelNameForRelease:)
    public static func modelName(for release: NFKMLXYOLORelease) -> String {
        let spec = specs(for: release)
        let prefix: String
        switch spec.generation {
        case .v9: prefix = "yolov9"
        case .v10: prefix = "yolov10"
        case .v11: prefix = "yolo11"
        case .v12: prefix = "yolo12"
        case .v26: prefix = "yolo26"
        }
        return prefix + spec.letter.rawValue
    }

    static func specs(for release: NFKMLXYOLORelease)
        -> (generation: NFKMLXYOLOGeneration, letter: NFKMLXYOLOScaleLetter) {
        switch release {
        case .v9Tiny: return (.v9, .tiny)
        case .v9Small: return (.v9, .small)
        case .v9Medium: return (.v9, .medium)
        case .v9Compact: return (.v9, .compact)
        case .v9Extended: return (.v9, .extended)
        case .v10Nano: return (.v10, .nano)
        case .v10Small: return (.v10, .small)
        case .v10Medium: return (.v10, .medium)
        case .v10Balanced: return (.v10, .balanced)
        case .v10Large: return (.v10, .large)
        case .v10ExtraLarge: return (.v10, .extraLarge)
        case .v11Nano: return (.v11, .nano)
        case .v11Small: return (.v11, .small)
        case .v11Medium: return (.v11, .medium)
        case .v11Large: return (.v11, .large)
        case .v11ExtraLarge: return (.v11, .extraLarge)
        case .v12Nano: return (.v12, .nano)
        case .v12Small: return (.v12, .small)
        case .v12Medium: return (.v12, .medium)
        case .v12Large: return (.v12, .large)
        case .v12ExtraLarge: return (.v12, .extraLarge)
        case .v26Nano: return (.v26, .nano)
        case .v26Small: return (.v26, .small)
        case .v26Medium: return (.v26, .medium)
        case .v26Large: return (.v26, .large)
        case .v26ExtraLarge: return (.v26, .extraLarge)
        }
    }

    /// Every release this port covers, for a caller that wants to enumerate them.
    @objc public static var releases: [NSNumber] {
        (0 ... NFKMLXYOLORelease.v26ExtraLarge.rawValue).map { NSNumber(value: $0) }
    }

    static func makeNet(_ release: NFKMLXYOLORelease, classCount: Int = 80) -> NFKMLXYOLOGenerationNet {
        let spec = specs(for: release)
        return NFKMLXYOLOGenerationNet(generation: spec.generation, letter: spec.letter,
                                       classCount: classCount)
    }

    /// Loads a released ultralytics checkpoint. The graph rows put every module at the reference's own
    /// index, so only the detection head's positional Sequentials need translating.
    static func loadWeights(into net: NFKMLXYOLOGenerationNet, from url: URL, matchingShapesOnly: Bool = false) throws {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let head = "model.\(net.nodes.count - 1)."
        let mapped = checkpoint.arrays.compactMap { key, value -> (String, MLXArray)? in
            // A BatchNorm's batch counter is bookkeeping the port does not carry.
            guard !key.hasSuffix("num_batches_tracked") else { return nil }
            let renamed = key.hasPrefix(head)
                ? head + remapHeadKey(String(key.dropFirst(head.count)),
                                      legacy: NFKYOLOGraphs.usesLegacyHead(net.generation))
                : key
            return (renamed,
                    checkpoint.needsConvTranspose && value.ndim == 4 ? value.transposed(0, 2, 3, 1) : value)
        }
        if matchingShapesOnly {
            // ultralytics' `intersect_dicts`, as for YOLOv8: only the class branches may stay fresh.
            let shapes = Dictionary(uniqueKeysWithValues: net.parameters().flattened().map { ($0.0, $0.1.shape) })
            let kept = mapped.filter { shapes[$0.0] == $0.1.shape }
            try NFKMLXYOLO.requireCoverage(outsideClassBranches: kept, of: Array(shapes.keys), source: url)
            try NFKMLXWeights.apply(kept, to: net, strict: false)
        } else {
            try NFKMLXWeights.apply(mapped, to: net)
        }
        net.train(false)                                        // BatchNorm running statistics
    }

    /// The class count a checkpoint was trained for, read from its first class branch's output bias.
    static func classCount(in url: URL, release: NFKMLXYOLORelease) throws -> Int? {
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
        let legacy = NFKYOLOGraphs.usesLegacyHead(specs(for: release).generation)
        return checkpoint.arrays.first { key, _ in
            let parts = key.split(separator: ".")
            guard parts.count > 2, parts[0] == "model" else { return false }
            let head = parts.dropFirst(2).joined(separator: ".")
            return remapHeadKey(head, legacy: legacy).hasPrefix("cv3.0.out.bias")
        }?.value.dim(0)
    }

    /// Translates one head key. The box branches are three-slot Sequentials; the class branches are
    /// either the same shape (YOLOv9) or two depthwise-plus-pointwise pairs and an output convolution.
    static func remapHeadKey(_ key: String, legacy: Bool) -> String {
        var parts = key.split(separator: ".").map(String.init)
        guard parts.count >= 3, ["cv2", "cv3", "one2one_cv2", "one2one_cv3"].contains(parts[0]),
              Int(parts[1]) != nil else { return key }
        // YOLOv9's class branch is the same three-slot Sequential as the box branch.
        let boxLike = parts[0] == "cv2" || parts[0] == "one2one_cv2" || legacy
        if boxLike {
            let slot = ["0": "conv1", "1": "conv2", "2": "out"]
            guard let renamed = slot[parts[2]] else { return key }
            parts.replaceSubrange(2 ... 2, with: [renamed])
            return parts.joined(separator: ".")
        }
        // A class branch on the modern shape: `<scale>.<pair>.<inner>` for the first two slots and
        // `<scale>.2` for the output convolution.
        if parts[2] == "2" {
            parts.replaceSubrange(2 ... 2, with: ["out"])
            return parts.joined(separator: ".")
        }
        let names = ["0": ["0": "dw0", "1": "pw0"], "1": ["0": "dw1", "1": "pw1"]]
        guard parts.count >= 4, let renamed = names[parts[2]]?[parts[3]] else { return key }
        parts.replaceSubrange(2 ... 3, with: [renamed])
        return parts.joined(separator: ".")
    }
}

extension NFKMLXYOLOGenerations {

    /// Builds a detection backend for one release directly from optional local weights — no registry
    /// required. A nil `weightsURL` builds random weights (`isReady` is true). `labels` names classes
    /// when available. Run inference off the render thread.
    @objc(backendWithRelease:weightsURL:labels:error:)
    public static func backend(release: NFKMLXYOLORelease, weightsURL: URL?,
                               labels: [String]?) throws -> any NFKInferenceBackend {
        let classes = try weightsURL.flatMap { try classCount(in: $0, release: release) } ?? 80
        let net = makeNet(release, classCount: classes)
        if let weightsURL {
            try loadWeights(into: net, from: weightsURL)
        }
        return NFKMLXYOLOGenerationBackend(net: net, identifier: modelName(for: release), labels: labels)
    }

    /// Downloads the release's checkpoint from Hugging Face, then builds — no registry required.
    /// Blocking on the network; run off the render thread.
    @objc(backendWithRelease:repo:weightsPath:revision:cacheDirectoryURL:labels:error:)
    public static func backend(release: NFKMLXYOLORelease, repo: String, weightsPath: String,
                               revision: String?, cacheDirectoryURL: URL?,
                               labels: [String]?) throws -> any NFKInferenceBackend {
        let url = try NFKMLXDownload.weightsURL(repo: repo, weightsPath: weightsPath, revision: revision,
                                                cacheDirectoryURL: cacheDirectoryURL)
        return try backend(release: release, weightsURL: url, labels: labels)
    }

    /// The asynchronous form of the download factory.
    @objc(backendWithRelease:repo:weightsPath:revision:cacheDirectoryURL:labels:completionHandler:)
    public static func backend(release: NFKMLXYOLORelease, repo: String, weightsPath: String,
                               revision: String?, cacheDirectoryURL: URL?, labels: [String]?,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXDownload.backend(repo: repo, weightsPath: weightsPath, revision: revision,
                               cacheDirectoryURL: cacheDirectoryURL,
                               build: { try backend(release: release, weightsURL: $0, labels: labels) },
                               completionHandler: completionHandler)
    }

    /// Registers every release with `NFKMLXModelRegistry` under its reference checkpoint stem.
    @objc public static func register() {
        for raw in 0 ... NFKMLXYOLORelease.v26ExtraLarge.rawValue {
            guard let release = NFKMLXYOLORelease(rawValue: raw) else { continue }
            NFKMLXModelRegistry.register(name: modelName(for: release)) { weightsURL in
                try backend(release: release, weightsURL: weightsURL, labels: nil)
            }
        }
    }
}

private final class NFKMLXYOLOGenerationHolder: @unchecked Sendable {
    let net: NFKMLXYOLOGenerationNet
    init(_ net: NFKMLXYOLOGenerationNet) { self.net = net }
}
