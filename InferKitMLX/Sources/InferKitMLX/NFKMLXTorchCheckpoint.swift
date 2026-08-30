//
//  NFKMLXTorchCheckpoint.swift
//  InferKitMLX
//
//  The public face of the native PyTorch checkpoint reader, and its MLX materialization. A consumer
//  brings a `.pth` and inspects or converts it without a Python toolchain; the model loaders reach
//  the same reader through `NFKMLXWeights.loadCheckpoint`, which sniffs the file's leading bytes.
//
//  Introduced in InferKit 0.3.0.
//

import Foundation
import MLX

/// A checkpoint tensor's element type, as stored.
@objc(NFKMLXTorchScalarType)
public enum NFKMLXTorchScalarType: Int, Sendable {
    case float32
    case float16
    case bfloat16
    case float64
    case int64
    case int32
    case int16
    case int8
    case uint8
    case bool
}

/// One tensor's shape and element type, for inspecting a checkpoint before loading it.
@objc(NFKMLXTorchTensorInfo)
public final class NFKMLXTorchTensorInfo: NSObject {
    @objc public let shape: [Int]
    @objc public let scalarType: NFKMLXTorchScalarType

    init(shape: [Int], scalarType: NFKMLXTorchScalarType) {
        self.shape = shape
        self.scalarType = scalarType
    }
}

/// A PyTorch checkpoint (`.pth`, `.pt`, `.ckpt`, `.th`, HF `.bin`) read natively, with no Python
/// toolchain: the modern ZIP container and the pre-1.6 stream both parse, the file stays
/// memory-mapped, and no pickle code ever executes. The tensors are the flattened state dict, with
/// the training wrappers the offline converters unwrap (`state_dict`, `params_ema`, …) already
/// removed.
///
/// `writeSafetensors(to:)` converts on device: the output is what the model's `Tools` converter
/// produces for a passthrough model, and every `weightsURL:` factory reads it. A TorchScript
/// archive, a `.nemo` tar, and a checkpoint whose pickle wraps its weights in a framework class
/// (YOLO's `ultralytics` module tree) are refused with an error naming the offline converter to use
/// instead.
///
/// Introduced in InferKit 0.3.0.
@objc(NFKMLXTorchCheckpoint)
public final class NFKMLXTorchCheckpoint: NSObject {

    let contents: NFKMLXTorchFormat.Contents

    init(contents: NFKMLXTorchFormat.Contents) {
        self.contents = contents
    }

    /// Reads the checkpoint at `url`. The file is memory-mapped; tensors reference it until read.
    @objc(checkpointWithContentsOfURL:error:)
    public static func checkpoint(contentsOf url: URL) throws -> NFKMLXTorchCheckpoint {
        NFKMLXTorchCheckpoint(contents: try NFKMLXTorchFormat.read(url: url))
    }

    /// The state dict's tensor names, sorted.
    @objc public var tensorNames: [String] {
        contents.tensors.keys.sorted()
    }

    /// The named tensor's shape and stored element type, or nil when the checkpoint has no such
    /// tensor.
    @objc(infoForTensor:)
    public func info(forTensor name: String) -> NFKMLXTorchTensorInfo? {
        guard let tensor = contents.tensors[name] else { return nil }
        return NFKMLXTorchTensorInfo(shape: tensor.shape, scalarType: scalarType(of: tensor.scalarType))
    }

    /// The named tensor's row-major little-endian bytes at the stored element type. A tensor the
    /// checkpoint stored as a strided view (Whisper's transposed Linear weights) is gathered into
    /// row-major order first.
    @objc(dataForTensor:error:)
    public func data(forTensor name: String) throws -> Data {
        guard let tensor = contents.tensors[name] else {
            throw NFKMLXError.unsupportedConfiguration("the checkpoint holds no tensor named \(name)")
        }
        return try contents.bytes(for: tensor)
    }

    /// Writes the state dict as a safetensors file, which every `weightsURL:` factory loads. The
    /// file marks itself as PyTorch layout, so a model's convolution transposes apply exactly as
    /// they do to an offline-converted file; float64 narrows to float32, as the converters do.
    @objc(writeSafetensorsToURL:error:)
    public func writeSafetensors(to url: URL) throws {
        try NFKMLXTorchFormat.writeSafetensors(contents, to: url)
    }

    /// The state dict as MLX arrays, for a Swift consumer feeding a model directly. Float64 narrows
    /// to float32.
    public func arrays() throws -> [String: MLXArray] {
        try NFKMLXTorchFormat.arrays(from: contents)
    }

    private func scalarType(of stored: NFKMLXTorchFormat.ScalarType) -> NFKMLXTorchScalarType {
        switch stored {
        case .float32: return .float32
        case .float16: return .float16
        case .bfloat16: return .bfloat16
        case .float64: return .float64
        case .int64: return .int64
        case .int32: return .int32
        case .int16: return .int16
        case .int8: return .int8
        case .uint8: return .uint8
        case .bool: return .bool
        }
    }
}

extension NFKMLXTorchFormat {

    /// Materializes every tensor as an MLX array at its stored precision (float64 narrowed to
    /// float32). Each array copies out of the mapped file, so the peak is the arrays plus evictable
    /// mapped pages rather than two resident copies.
    static func arrays(from contents: Contents) throws -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]
        result.reserveCapacity(contents.tensors.count)
        for (name, tensor) in contents.tensors {
            let (bytes, scalarType) = try materialized(tensor, in: contents)
            result[name] = MLXArray(bytes, tensor.shape, dtype: dtype(of: scalarType))
        }
        return result
    }

    private static func dtype(of scalarType: ScalarType) -> DType {
        switch scalarType {
        case .float64, .float32: return .float32
        case .float16: return .float16
        case .bfloat16: return .bfloat16
        case .int64: return .int64
        case .int32: return .int32
        case .int16: return .int16
        case .int8: return .int8
        case .uint8: return .uint8
        case .bool: return .bool
        }
    }
}
