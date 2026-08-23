//
//  NFKMLXWeights.swift
//  InferKitMLX
//
//  Applying a checkpoint to a module, with key-coverage verification.
//
//  MLX's `update(parameters:)` silently ignores keys it does not recognize: a checkpoint whose names do
//  not match the module leaves those parameters at their random initialization. The model still loads,
//  still runs, and produces confident-looking garbage — the failure mode that made a real U²-Net
//  checkpoint output noise (the module named its convolutions `conv`/`bn` where the reference uses
//  `conv_s1`/`bn_s1`). Verifying coverage turns that silent corruption into an immediate, specific error.
//

import Foundation
import MLX
import MLXNN

/// The precision a released checkpoint loads at.
///
/// MLX's `update(parameters:)` adopts a checkpoint's element type along with its values, so a
/// half-precision release silently turns a float32 module into a half-precision one.
@objc(NFKMLXWeightPrecision)
public enum NFKMLXWeightPrecision: Int, Sendable {
    /// Convert a half-precision checkpoint to the module's own float32. The model computes as it was
    /// built, which is what a measured parity number then describes.
    case float32
    /// Adopt the file's element type. A half-precision release runs at half precision: half the
    /// memory and faster, at three or four decimal digits rather than seven.
    case checkpoint
}

enum NFKMLXWeights {

    /// Applies `precision` to already-remapped pairs, leaving integer tensors alone — an index cast to
    /// float is not a rounding error, it is a different value.
    static func converted(_ mapped: [(String, MLXArray)],
                          to precision: NFKMLXWeightPrecision) -> [(String, MLXArray)] {
        guard precision == .float32 else { return mapped }
        return mapped.map { name, value in
            (name, value.dtype == .float16 || value.dtype == .bfloat16 ? value.asType(.float32) : value)
        }
    }

    /// Metadata written by ``save(_:to:)`` to mark a checkpoint as already being in the module's layout.
    private static let layoutKey = "inferkit.layout"
    private static let mlxLayout = "mlx"

    /// A checkpoint's arrays together with the layout its convolution weights are stored in.
    struct Checkpoint {
        let arrays: [String: MLXArray]

        /// True for a converted PyTorch checkpoint, whose 4-D weights are `[out, in, kH, kW]` and need
        /// the model's transpose to MLX's `[out, kH, kW, in]`. False for a checkpoint written by
        /// ``NFKMLXWeights/save(_:to:)``, whose weights are already in the module's own layout.
        ///
        /// A model's `loadWeights` skips its transpose when this is false. Skipping rather than
        /// inverting is what keeps the round trip exact for the models whose transpose is not the
        /// common one: SAM's `up1`/`up2` use `transposed(1, 2, 3, 0)` and Whisper handles 3-D Conv1d,
        /// and a generic inverse would corrupt both.
        let needsConvTranspose: Bool
    }

    /// Reads a checkpoint and reports which layout its convolution weights are in.
    ///
    /// A model's `loadWeights` calls this in place of `loadArrays(url:)` so that both a converted
    /// PyTorch checkpoint and one written by ``save(_:to:)`` load correctly through the same path.
    static func loadCheckpoint(url: URL) throws -> Checkpoint {
        let (arrays, metadata) = try loadArraysAndMetadata(url: url)
        return Checkpoint(arrays: arrays, needsConvTranspose: metadata[layoutKey] != mlxLayout)
    }

    /// Writes every parameter of `module` to a safetensors file in the module's own layout.
    ///
    /// This is the output of fine-tuning and the input of the model's existing
    /// `backendWith…weightsURL:` factory, so a customized model needs no separate loading path. The
    /// file records its layout in metadata, which ``loadCheckpoint(url:)`` reads back.
    ///
    /// Non-trainable parameters are included, so a model carrying batch-normalization running
    /// statistics reloads complete.
    static func save(_ module: Module, to url: URL) throws {
        guard url.pathExtension == "safetensors" else {
            throw NFKMLXError.checkpointNotWritable(
                "a checkpoint must be written as .safetensors to carry its layout metadata, "
                + "and \(url.lastPathComponent) is not")
        }
        eval(module)
        let arrays = Dictionary(uniqueKeysWithValues: module.parameters().flattened())

        // A periodic checkpoint overwrites the only copy of the run's progress. Writing in place means
        // a process killed part way through the write destroys both the new state and the previous
        // one, which is exactly the suspension this checkpoint exists to survive.
        let scratch = url.deletingLastPathComponent()
            .appendingPathComponent(".\(UUID().uuidString).safetensors")
        try MLX.save(arrays: arrays, metadata: [layoutKey: mlxLayout], url: scratch)
        do {
            if FileManager.default.fileExists(atPath: url.path) {
                _ = try FileManager.default.replaceItemAt(url, withItemAt: scratch)
            } else {
                try FileManager.default.moveItem(at: scratch, to: url)
            }
        } catch {
            try? FileManager.default.removeItem(at: scratch)
            throw error
        }
    }

    /// Receives the module's expected parameter names and the checkpoint's, whenever a load is verified.
    /// Working out a reference model's key remap means seeing both sides, and the thrown error can only
    /// carry a summary. Tests set this; nothing in the shipping path does.
    nonisolated(unsafe) static var diagnosticsHandler: ((_ expected: [String], _ provided: [String]) -> Void)?

    /// Applies `mapped` (checkpoint keys already remapped and transposed to MLX layout) to `module`,
    /// verifying first that the checkpoint covers every parameter the module expects.
    ///
    /// - Parameter strict: when true (the default), a parameter the checkpoint does not supply throws
    ///   ``NFKMLXError/weightsMismatch(_:)`` rather than leaving it randomly initialized. Pass false only
    ///   for a deliberate partial load.
    ///
    /// Keys the checkpoint carries that the module does not use are harmless (optimizer state, a teacher
    /// branch, `num_batches_tracked`), so they do not fail the load.
    static func apply(_ mapped: [(String, MLXArray)], to module: Module, strict: Bool = true) throws {
        if strict {
            try verifyCoverage(of: mapped, for: module)
        }
        module.update(parameters: ModuleParameters.unflattened(mapped))
        eval(module)
    }

    private static func verifyCoverage(of mapped: [(String, MLXArray)], for module: Module) throws {
        let expected = module.parameters().flattened().map(\.0)
        let provided = Set(mapped.map(\.0))
        diagnosticsHandler?(expected, mapped.map(\.0))
        let missing = expected.filter { !provided.contains($0) }
        guard missing.isEmpty else {
            throw NFKMLXError.weightsMismatch(describe(missing: missing, expected: expected.count,
                                                       provided: provided.count))
        }
    }

    private static func describe(missing: [String], expected: Int, provided: Int) -> String {
        let sample = missing.prefix(5).joined(separator: ", ")
        let more = missing.count > 5 ? " (and \(missing.count - 5) more)" : ""
        return """
            the checkpoint does not cover \(missing.count) of the model's \(expected) parameters, \
            so they would stay randomly initialized and the output would be meaningless: \(sample)\(more). \
            The checkpoint supplied \(provided) keys — the names likely need a remap for this model.
            """
    }
}
