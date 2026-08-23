//
//  NFKMLXReleaseWeights.swift
//  InferKitMLX
//
//  Reading a downloaded release's weights, whether they sit in one file or across a shard index.
//  Every decoder family needs this, and each had written its own — which is how one of them came to
//  have no sharded path at all.
//
//  Introduced in InferKit 0.1.0.
//

import Foundation
import MLX

/// Reads a release's weights, whether they sit in one `model.safetensors` or across a shard index.
///
/// @discussion Every release above the smallest size splits its weights across files with a
/// `model.safetensors.index.json` naming which shard holds each tensor, so a loader reading only
/// `model.safetensors` covers one size and nothing else. A shard is read ONCE, not once per tensor:
/// the index names the same file for many.
enum NFKMLXReleaseWeights {

    /// The arrays a release holds, ready for `NFKMLXWeights.apply`.
    ///
    /// - Parameters:
    ///   - directory: the downloaded release directory.
    ///   - precision: `.checkpoint` keeps the stored element type, which halves the memory a
    ///     half-precision release needs and costs accuracy; anything else converts to float32.
    ///   - remap: the module key a checkpoint key maps to, or nil to skip that tensor. A multimodal
    ///     release carries towers this decoder does not implement, and skipping them is what lets the
    ///     apply stay strict.
    static func arrays(inDirectory directory: URL,
                       precision: NFKMLXWeightPrecision = .float32,
                       remap: (String) -> String? = { $0 }) throws -> [(String, MLXArray)] {
        var merged = [(String, MLXArray)]()
        for url in try files(inDirectory: directory) {
            let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
            for (key, value) in checkpoint.arrays {
                guard let name = remap(key) else { continue }
                merged.append((name, precision == .checkpoint ? value : value.asType(.float32)))
            }
        }
        return merged
    }

    /// The weight files a release directory holds, single or sharded.
    static func files(inDirectory directory: URL) throws -> [URL] {
        let single = directory.appendingPathComponent("model.safetensors")
        if FileManager.default.fileExists(atPath: single.path) { return [single] }

        let indexURL = directory.appendingPathComponent("model.safetensors.index.json")
        guard let data = try? Data(contentsOf: indexURL),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let weightMap = json["weight_map"] as? [String: String]
        else {
            throw NFKMLXError.unsupportedConfiguration(
                "\(directory.lastPathComponent) holds neither model.safetensors nor a shard index")
        }
        return Set(weightMap.values).sorted().map { directory.appendingPathComponent($0) }
    }
}
