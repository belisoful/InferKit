//
//  NFKMLXModelDescription.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXNN

/// A layer whose weights are held outside its parameters, as a paged load leaves them: in the
/// release, read as its router reaches them, or in memory in the form the release stores them. It
/// states how many parameters it stands for, the bytes it keeps in memory, and the quantization its
/// weights are stored at.
protocol NFKMLXPagedParameters {
    /// The parameters held outside the layer's own; 0 where its weights are resident.
    var pagedParameterCount: Int { get }
    /// The bytes of those weights kept in memory, 0 where they stay in the release.
    var pagedHeldBytes: Int { get }
    /// The bits and group size of a quantized paged layer, or nil.
    var pagedQuantization: (bits: Int, groupSize: Int)? { get }
}

extension NFKMLXPagedParameters {
    var pagedQuantization: (bits: Int, groupSize: Int)? { nil }
}

/// What a loaded network reports under a backend's `modelInfo`: its parameter count, the bytes its
/// arrays hold, the floating type its weights compute in, and its quantization.
///
/// A quantized layer keeps its weight packed into `uint32` words, with its scales and biases beside
/// it. Its parameters are the packed words times 32 over its bits; its scales and biases count toward
/// the bytes and not the parameters. A network quantized at more than one setting reports the one
/// that covers the most parameters. A paged layer (``NFKMLXPagedParameters``) adds the parameters it
/// stands for and, where it is quantized, its setting, and adds to the bytes only what it keeps in
/// memory: nothing where its weights stay in the release.
enum NFKMLXModelDescription {

    private struct Setting: Hashable {
        let bits: Int
        let groupSize: Int
    }

    static func info(of modules: [Module]) -> [String: Any] {
        var parameters = 0
        var bytes = 0
        var floatingBytes: [DType: Int] = [:]
        var quantizedParameters: [Setting: Int] = [:]
        for module in modules {
            var quantized: [String: Quantized] = [:]
            for (path, child) in module.namedModules() {
                if let layer = child as? Quantized {
                    quantized[path] = layer
                }
                if let paged = child as? NFKMLXPagedParameters {
                    let count = paged.pagedParameterCount
                    parameters += count
                    bytes += paged.pagedHeldBytes
                    if let setting = paged.pagedQuantization {
                        quantizedParameters[Setting(bits: setting.bits, groupSize: setting.groupSize), default: 0] += count
                    }
                }
            }
            for (key, array) in module.parameters().flattened() {
                bytes += array.nbytes
                let components = key.split(separator: ".")
                let name = components.last.map(String.init) ?? key
                let owner = components.dropLast().joined(separator: ".")
                if quantized[owner] != nil, name == "scales" || name == "biases" {
                    continue
                }
                if let layer = quantized[owner], name == "weight" {
                    let count = array.size * 32 / layer.bits
                    parameters += count
                    quantizedParameters[Setting(bits: layer.bits, groupSize: layer.groupSize), default: 0] += count
                    continue
                }
                parameters += array.size
                if array.dtype.isFloatingPoint {
                    floatingBytes[array.dtype, default: 0] += array.nbytes
                }
            }
        }
        var info: [String: Any] = [NFKModelInfoParameterCount: parameters, NFKModelInfoWeightBytes: bytes]
        if let dominant = floatingBytes.max(by: { $0.value < $1.value })?.key, let name = precisionName(dominant) {
            info[NFKModelInfoPrecision] = name
        }
        if let setting = quantizedParameters.max(by: { $0.value < $1.value })?.key {
            info[NFKModelInfoQuantizationBits] = setting.bits
            info[NFKModelInfoQuantizationGroupSize] = setting.groupSize
        }
        return info
    }

    /// The description of `modules`, with the bytes the release directory occupies on disk and the
    /// `model_type` and `max_position_embeddings` that its `config.json` states at its top level or,
    /// for a multimodal release, under `text_config`. With no modules, only the release's figures.
    static func info(of modules: [Module], releaseDirectoryURL: URL?) -> [String: Any] {
        var info = modules.isEmpty ? [:] : info(of: modules)
        guard let releaseDirectoryURL else {
            return info
        }
        info[NFKModelInfoStorageBytes] = storageBytes(at: releaseDirectoryURL)
        guard let data = try? Data(contentsOf: releaseDirectoryURL.appendingPathComponent("config.json")),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return info
        }
        let text = json["text_config"] as? [String: Any]
        if let type = json["model_type"] as? String {
            info[NFKModelInfoArchitecture] = type
        }
        if let positions = (json["max_position_embeddings"] ?? text?["max_position_embeddings"]) as? Int {
            info[NFKModelInfoContextLength] = positions
        }
        return info
    }

    /// The bytes a release occupies on disk: a single file's, or every file's under a directory, or
    /// nil where nothing is at `url`. A symbolic link counts its target, so a Hugging Face cache
    /// snapshot, whose files link to its blobs, counts the blobs.
    static func storageBytes(at url: URL) -> Int? {
        let resolved = url.resolvingSymlinksInPath()
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: resolved.path, isDirectory: &isDirectory) else {
            return nil
        }
        guard isDirectory.boolValue else {
            return (try? resolved.resourceValues(forKeys: [.totalFileAllocatedSizeKey]))?.totalFileAllocatedSize
        }
        let directoryURL = resolved
        guard let enumerator = FileManager.default.enumerator(at: directoryURL,
                                                              includingPropertiesForKeys: [.isSymbolicLinkKey]) else {
            return nil
        }
        var total = 0
        for case let url as URL in enumerator {
            let isLink = (try? url.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink) == true
            let file = isLink ? url.resolvingSymlinksInPath() : url
            guard let values = try? file.resourceValues(forKeys: [.isRegularFileKey, .totalFileAllocatedSizeKey]),
                  values.isRegularFile == true else {
                continue
            }
            total += values.totalFileAllocatedSize ?? 0
        }
        return total
    }

    private static func precisionName(_ type: DType) -> String? {
        switch type {
        case .float32: return "float32"
        case .float16: return "float16"
        case .bfloat16: return "bfloat16"
        default: return nil
        }
    }
}

/// A backend's model description, computed on first read: walking a large network's parameters
/// costs a status poll too much to repeat, and the loaded weights do not change.
final class NFKMLXModelInfoCache: @unchecked Sendable {
    private let lock = NSLock()
    private var cached: [String: Any]?

    func value(_ make: () -> [String: Any]) -> [String: Any] {
        lock.withLock {
            if let cached {
                return cached
            }
            let made = make()
            cached = made
            return made
        }
    }
}
