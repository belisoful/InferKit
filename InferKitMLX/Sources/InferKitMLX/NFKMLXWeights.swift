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

/// Runtime MLX quantization: packs a module's `Linear` layers, a mixture's stacked experts, and
/// optionally its `Embedding` layers into affine 4- or 8-bit groups.
///
/// A layer quantizes only when its input width divides the group size; everything else (convolutions,
/// norms, the Snake alphas) computes as built. The replacement works because `QuantizedLinear`
/// subclasses `Linear`, which every module here stores behind `@ModuleInfo`, so it is assignable to
/// the property it replaces. The same subclassing is why the filter must exclude layers that are
/// already quantized, and why ``NFKMLXLoRA`` refuses a quantized layer rather than adapting one.
///
/// This is the step that follows a merged low-rank fine-tune: merge at float precision, then quantize.
/// The other order rounds the training away, because requantizing a weight plus a small delta snaps
/// back to the value it started from.
public enum NFKMLXQuantization {

    /// Replaces the module's eligible `Linear` layers with `QuantizedLinear` at the given geometry,
    /// and its `Embedding` layers with `QuantizedEmbedding` when `includeEmbeddings` is set. On a
    /// loaded module this quantizes the VALUES; on a freshly built one it shapes the structure a
    /// quantized checkpoint then loads into.
    ///
    /// `includeEmbeddings` is off by default because a TIED model reuses its input embedding as the
    /// output projection, so quantizing it quantizes the logit head too — a cost that has to be
    /// measured per model rather than assumed free. It is safe to enable for an untied model, whose
    /// embedding is a lookup table separate from `lm_head`.
    public static func quantize(module: Module, bits: Int = 4, groupSize: Int = 64,
                                includeEmbeddings: Bool = false) {
        MLXNN.quantize(model: module, groupSize: groupSize, bits: bits, mode: .affine, filter: { _, layer in
            if let linear = layer as? Linear, !(linear is QuantizedLinear) {
                return linear.weight.shape[1] % groupSize == 0
            }
            if let experts = layer as? NFKLMSwitchLinear, !(experts is NFKLMQuantizedSwitchLinear) {
                return experts.inputSize % groupSize == 0
            }
            if includeEmbeddings, let embedding = layer as? Embedding,
               !(embedding is QuantizedEmbedding) {
                return embedding.weight.shape[1] % groupSize == 0
            }
            return false
        }, apply: { layer, groupSize, bits, mode in
            // A mixture's stacked expert weights are not a `Linear`, so MLX's own quantizer does not
            // know them; everything else takes the standard path.
            if let experts = layer as? NFKLMSwitchLinear {
                return experts.quantized(groupSize: groupSize, bits: bits)
            }
            return quantizeSingle(layer: layer, groupSize: groupSize, bits: bits, mode: mode)
        })
    }

    /// Applies a checkpoint's recorded quantization to a freshly built module, so the packed arrays
    /// land on matching structure. A nil quantization leaves the module as built.
    ///
    /// The metadata records one bits/groupSize, not which layer KINDS were quantized, so whether the
    /// embedding was packed is read from the checkpoint itself: a quantized embedding weight is stored
    /// as `uint32` where an unquantized one is a float. This keeps a checkpoint self-describing, so a
    /// file saved before embeddings were quantizable still loads.
    static func matchStructure(of checkpoint: NFKMLXWeights.Checkpoint, on module: Module) {
        // An MXFP4 record describes packed experts alone, which the language loader installs from the
        // arrays' own dtypes; there is no affine structure to rebuild for it.
        guard let quantization = checkpoint.quantization, quantization.mode == .affine else { return }
        let embeddingsPacked = module.leafModules().flattened().contains { path, layer in
            guard layer is Embedding, !(layer is QuantizedEmbedding) else { return false }
            return checkpoint.arrays[path + ".weight"]?.dtype == .uint32
        }
        quantize(module: module, bits: quantization.bits, groupSize: quantization.groupSize,
                 includeEmbeddings: embeddingsPacked)
    }
}

public enum NFKMLXWeights {

    /// Applies `precision` to already-remapped pairs, leaving integer tensors alone — an index cast to
    /// float is not a rounding error, it is a different value.
    public static func converted(_ mapped: [(String, MLXArray)],
                                 to precision: NFKMLXWeightPrecision) -> [(String, MLXArray)] {
        guard precision == .float32 else { return mapped }
        return mapped.map { name, value in
            (name, value.dtype == .float16 || value.dtype == .bfloat16 ? value.asType(.float32) : value)
        }
    }

    /// Metadata written by ``save(_:to:)`` to mark a checkpoint as already being in the module's layout.
    private static let layoutKey = "inferkit.layout"
    private static let mlxLayout = "mlx"
    /// Metadata naming the MLX affine quantization a checkpoint's packed weights were stored under,
    /// as "bits:groupSize". Written automatically when the saved module holds quantized layers.
    private static let quantizationKey = "inferkit.quantization"

    /// The MLX quantization a checkpoint was stored under.
    public struct Quantization: Sendable, Equatable {
        public let bits: Int
        public let groupSize: Int
        public let mode: QuantizationMode
        init(bits: Int, groupSize: Int, mode: QuantizationMode = .affine) {
            self.bits = bits
            self.groupSize = groupSize
            self.mode = mode
        }
    }

    /// A checkpoint's arrays together with the layout its convolution weights are stored in.
    public struct Checkpoint {
        public let arrays: [String: MLXArray]

        /// True for a converted PyTorch checkpoint, whose 4-D weights are `[out, in, kH, kW]` and need
        /// the model's transpose to MLX's `[out, kH, kW, in]`. False for a checkpoint written by
        /// ``NFKMLXWeights/save(_:to:)``, whose weights are already in the module's own layout.
        ///
        /// A model's `loadWeights` skips its transpose when this is false. Skipping rather than
        /// inverting is what keeps the round trip exact for the models whose transpose is not the
        /// common one: SAM's `up1`/`up2` use `transposed(1, 2, 3, 0)` and Whisper handles 3-D Conv1d,
        /// and a generic inverse would corrupt both.
        public let needsConvTranspose: Bool

        /// Non-nil for a checkpoint whose weights are MLX-quantized. A loader quantizes the module's
        /// structure to match BEFORE applying: a packed uint32 weight loaded into a plain `Linear`
        /// adopts the wrong shape and dtype silently, which is exactly the hazard the metadata
        /// exists to close.
        public let quantization: Quantization?

        /// True when the file was a raw PyTorch checkpoint read by ``NFKMLXTorchFormat``. A model
        /// whose offline converter pre-permutes a tensor (a transposed-convolution axis swap) reads
        /// this to apply that permutation itself: the raw and converted files can carry identical
        /// key names, so the distinction cannot be recovered from the arrays.
        public let isNativeTorch: Bool

        init(arrays: [String: MLXArray], needsConvTranspose: Bool, quantization: Quantization?,
             isNativeTorch: Bool = false) {
            self.arrays = arrays
            self.needsConvTranspose = needsConvTranspose
            self.quantization = quantization
            self.isNativeTorch = isNativeTorch
        }
    }

    /// Reads a checkpoint and reports which layout its convolution weights are in.
    ///
    /// A model's `loadWeights` calls this in place of `loadArrays(url:)` so that both a converted
    /// PyTorch checkpoint and one written by ``save(_:to:)`` load correctly through the same path.
    ///
    /// The format is sniffed from the file's leading bytes rather than its extension: a raw PyTorch
    /// checkpoint (`.pth`, `.pt`, `.ckpt`, `.th`, or an HF `.bin`, which shares its extension with
    /// nothing that identifies it) routes through ``NFKMLXTorchFormat`` and reports PyTorch layout,
    /// so every model accepts one wherever it accepts a converted safetensors.
    public static func loadCheckpoint(url: URL) throws -> Checkpoint {
        if NFKMLXTorchFormat.isTorchCheckpoint(leadingBytes(of: url)) {
            let contents = try NFKMLXTorchFormat.read(url: url)
            return Checkpoint(arrays: try NFKMLXTorchFormat.arrays(from: contents),
                              needsConvTranspose: true, quantization: nil, isNativeTorch: true)
        }
        let (arrays, metadata) = try loadArraysAndMetadata(url: url)
        var quantization: Quantization?
        if let recorded = metadata[quantizationKey] {
            let fields = recorded.split(separator: ":")
            let parts = fields.prefix(2).compactMap { Int($0) }
            let mode = fields.count > 2 ? QuantizationMode(rawValue: String(fields[2])) : .affine
            guard parts.count == 2, (2 ... 3).contains(fields.count), let mode else {
                throw NFKMLXError.unsupportedConfiguration(
                    "\(url.lastPathComponent) records quantization \"\(recorded)\", which is not "
                    + "the bits:groupSize[:mode] form this loader reads")
            }
            quantization = Quantization(bits: parts[0], groupSize: parts[1], mode: mode)
        }
        return Checkpoint(arrays: arrays, needsConvTranspose: metadata[layoutKey] != mlxLayout,
                          quantization: quantization)
    }

    /// The file's first block, for format sniffing. One tar header (512 bytes) is read because a
    /// `.nemo`'s ustar magic sits at offset 257, past a 4-byte peek. An unreadable file returns
    /// empty, so the safetensors path raises its own error for a missing file as it always has.
    private static func leadingBytes(of url: URL) -> Data {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return Data() }
        defer { try? handle.close() }
        return (try? handle.read(upToCount: 512)) ?? Data()
    }

    /// Writes every parameter of `module` to a safetensors file in the module's own layout.
    ///
    /// This is the output of fine-tuning and the input of the model's existing
    /// `backendWith…weightsURL:` factory, so a customized model needs no separate loading path. The
    /// file records its layout in metadata, which ``loadCheckpoint(url:)`` reads back.
    ///
    /// Non-trainable parameters are included, so a model carrying batch-normalization running
    /// statistics reloads complete.
    public static func save(_ module: Module, to url: URL) throws {
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
        // A module holding quantized layers records their geometry, so the file cannot be read back
        // into an unquantized structure by mistake. Quantization here is uniform (one bits/groupSize
        // per module), which is what the single metadata entry can say.
        var metadata = [layoutKey: mlxLayout]
        // A module mixing affine layers with MXFP4 experts records the affine geometry, which is what
        // `matchStructure` rebuilds; the packed experts are recognized by their own dtypes on load.
        let quantizedLeaves = module.leafModules().flattened().compactMap { $0.1 as? Quantized }
        if let quantized = quantizedLeaves.first(where: { $0.mode == .affine }) ?? quantizedLeaves.first {
            metadata[quantizationKey] = "\(quantized.bits):\(quantized.groupSize)"
                + (quantized.mode == .affine ? "" : ":\(quantized.mode.rawValue)")
        }
        try MLX.save(arrays: arrays, metadata: metadata, url: scratch)
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
    /// - Parameter verifyShapes: when true, a parameter the checkpoint supplies at a shape the module
    ///   does not expect throws rather than being adopted silently. This is correct ONLY where every
    ///   built shape already equals the checkpoint's — the dense decoder, whose widths come straight
    ///   from `config.json`. It is WRONG wherever a module builder uses a shape the checkpoint then
    ///   corrects through MLX's adoption: Conv-TasNet's placeholder `.base` widths, and Gemma E4B's
    ///   feed-forward-doubling heuristic, both load right only because adoption reshapes them. Those
    ///   leave it off (the default), and shape adoption being load-bearing in several builders is why
    ///   this cannot be turned on globally. Default off.
    public static func apply(_ mapped: [(String, MLXArray)], to module: Module,
                             strict: Bool = true, verifyShapes: Bool = false) throws {
        if strict {
            try verifyCoverage(of: mapped, for: module, verifyShapes: verifyShapes)
        }
        module.update(parameters: ModuleParameters.unflattened(mapped))
        eval(module)
    }

    private static func verifyCoverage(of mapped: [(String, MLXArray)], for module: Module,
                                       verifyShapes: Bool) throws {
        let expectedParameters = module.parameters().flattened()
        let expected = expectedParameters.map(\.0)
        let provided = Set(mapped.map(\.0))
        diagnosticsHandler?(expected, mapped.map(\.0))
        let missing = expected.filter { !provided.contains($0) }
        guard missing.isEmpty else {
            throw NFKMLXError.weightsMismatch(describe(missing: missing, expected: expected.count,
                                                       provided: provided.count))
        }
        guard verifyShapes else { return }
        // A parameter the module expects at one shape, supplied at another, is adopted wholesale by
        // `update(parameters:)` — the checkpoint loads cleanly and the model computes wrong numbers.
        // Where the module's shapes come from the release's declared config, that is a config that
        // disagrees with the weights, which this turns into a load-time error.
        let expectedShapes = Dictionary(expectedParameters.map { ($0.0, $0.1.shape) },
                                        uniquingKeysWith: { first, _ in first })
        let mismatched = mapped.compactMap { name, value -> (name: String, expected: [Int], provided: [Int])? in
            guard let shape = expectedShapes[name], shape != value.shape else { return nil }
            return (name, shape, value.shape)
        }
        guard mismatched.isEmpty else {
            throw NFKMLXError.weightsMismatch(describe(mismatched: mismatched))
        }
    }

    private static func describe(mismatched: [(name: String, expected: [Int], provided: [Int])]) -> String {
        let sample = mismatched.prefix(3)
            .map { "\($0.name) expects \($0.expected) but the checkpoint has \($0.provided)" }
            .joined(separator: "; ")
        let more = mismatched.count > 3 ? " (and \(mismatched.count - 3) more)" : ""
        return """
            the checkpoint's shape does not match the model's for \(mismatched.count) parameter(s), \
            which would load cleanly and compute wrong numbers: \(sample)\(more). \
            The configuration this model was built with likely does not match this checkpoint.
            """
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
