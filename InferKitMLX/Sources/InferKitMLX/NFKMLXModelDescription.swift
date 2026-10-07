//
//  NFKMLXModelDescription.swift
//  InferKitMLX
//

import Foundation
import InferKit
import MLX
import MLXNN

/// What a loaded network reports under a backend's `modelInfo`: its parameter count, the bytes its
/// arrays hold, the floating type its weights compute in, and its quantization.
///
/// A quantized layer keeps its weight packed into `uint32` words, with its scales and biases beside
/// it. Its parameters are the packed words times 32 over its bits; its scales and biases count toward
/// the bytes and not the parameters. A network quantized at more than one setting reports the one
/// that covers the most parameters.
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

    /// The description of `modules`, with the `model_type` and `max_position_embeddings` that the
    /// release's `config.json` states at its top level or, for a multimodal release, under
    /// `text_config`. With no modules, only the release's statements.
    static func info(of modules: [Module], releaseDirectoryURL: URL?) -> [String: Any] {
        var info = modules.isEmpty ? [:] : info(of: modules)
        guard let releaseDirectoryURL,
              let data = try? Data(contentsOf: releaseDirectoryURL.appendingPathComponent("config.json")),
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
