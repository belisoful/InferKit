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

    /// The bytes a release's weight files occupy on disk, which is what they occupy resident once
    /// materialized at the stored element type.
    static func weightBytes(inDirectory directory: URL) throws -> Int {
        try files(inDirectory: directory).reduce(0) { total, url in
            let size = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int) ?? nil
            return total + (size ?? 0)
        }
    }

    /// Throws before any weight is materialized when the release cannot fit the memory budget, so a
    /// load that would kill the process becomes an error naming the shortfall.
    ///
    /// @discussion The stored bytes are the floor; a `.float32` load of a 16-bit release doubles them.
    /// `budget` defaults to Metal's recommended working set — what a resident model should be sized
    /// against — and `reserve` holds back room for activations and, for a generation model, the cache.
    /// The check is deliberately conservative: it is better to refuse a load that might have just fit
    /// than to let one that will not proceed to a process kill.
    static func verifyFits(inDirectory directory: URL,
                           precision: NFKMLXWeightPrecision = .float32,
                           budget: Int = NFKMLXGPU.recommendedWorkingSetSize,
                           reserve: Int = 0) throws {
        guard budget > 0 else { return }                        // an unknown machine does not gate a load
        let stored = try weightBytes(inDirectory: directory)
        let resident = precision == .checkpoint ? stored : stored * 2   // fp16/bf16 → fp32 doubles it
        guard resident + reserve <= budget else {
            let gib = { (bytes: Int) in String(format: "%.1f", Double(bytes) / 1_073_741_824) }
            throw NFKMLXError.unsupportedConfiguration(
                "\(directory.lastPathComponent) needs about \(gib(resident)) GiB resident"
                + (reserve > 0 ? " plus \(gib(reserve)) GiB reserved" : "")
                + ", but the machine's working set is \(gib(budget)) GiB; "
                + "load at .checkpoint precision, quantize, or use a smaller size")
        }
    }

    /// The weight files a release directory holds, single or sharded. Transformers releases name
    /// them `model.safetensors`; diffusers components name them `diffusion_pytorch_model.safetensors`.
    static func files(inDirectory directory: URL) throws -> [URL] {
        for base in ["model", "diffusion_pytorch_model"] {
            let single = directory.appendingPathComponent("\(base).safetensors")
            if FileManager.default.fileExists(atPath: single.path) { return [single] }

            let indexURL = directory.appendingPathComponent("\(base).safetensors.index.json")
            if let data = try? Data(contentsOf: indexURL),
               let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let weightMap = json["weight_map"] as? [String: String] {
                return Set(weightMap.values).sorted().map { directory.appendingPathComponent($0) }
            }
        }
        throw NFKMLXError.unsupportedConfiguration(
            "\(directory.lastPathComponent) holds neither model.safetensors nor a shard index")
    }
}
