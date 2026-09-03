//
//  NFKMLXGGUF.swift
//  InferKitMLX
//
//  The public face of the native GGUF reader, and its MLX materialization. A consumer inspects a GGUF
//  model's metadata and tensors, or reads a tensor as a dequantized `MLXArray`, with no Python and no
//  llama.cpp. The parsing and dequantization are pure Foundation in NFKMLXGGUFFormat; this adds the MLX
//  array on top.
//

import Foundation
import MLX

/// One tensor's shape and quantization, for inspection before reading it.
@objc(NFKMLXGGUFTensorInfo)
public final class NFKMLXGGUFTensorInfo: NSObject {
    @objc public let name: String
    @objc public let shape: [Int]
    /// The GGML type name, e.g. `Q4_K`, `Q6_K`, `Q8_0`, `F16`, `F32`.
    @objc public let typeName: String

    init(_ descriptor: NFKMLXGGUFTensorDescriptor) {
        name = descriptor.name
        shape = descriptor.shape
        typeName = descriptor.type.map(NFKMLXGGUFTensorInfo.name(of:)) ?? "TYPE_\(descriptor.rawType)"
        super.init()
    }

    static func name(of type: NFKMLXGGMLType) -> String {
        switch type {
        case .f32: return "F32"
        case .f16: return "F16"
        case .q4_0: return "Q4_0"
        case .q5_0: return "Q5_0"
        case .q8_0: return "Q8_0"
        case .q4_K: return "Q4_K"
        case .q6_K: return "Q6_K"
        }
    }
}

/// A GGUF model file: its metadata, its tensor descriptors, and dequantized tensors.
///
/// @discussion GGUF is the format most quantized language models are distributed in. This reads the
/// container natively and dequantizes the block-quant formats a real model uses — `Q4_K` and `Q6_K` (the
/// k-quants a `Q4_K_M` model is built from), `Q8_0`, `Q4_0`, and `F16`/`F32` — into `MLXArray`s ready to
/// load into a model. A file naming a type this reader does not implement is refused rather than misread,
/// the same contract the native PyTorch reader keeps.
@objc(NFKMLXGGUF)
public final class NFKMLXGGUF: NSObject {

    private let format: NFKMLXGGUFFormat

    init(format: NFKMLXGGUFFormat) {
        self.format = format
        super.init()
    }

    /// Reads a GGUF file, memory-mapping it.
    @objc(GGUFWithContentsOfURL:error:)
    public static func gguf(contentsOf url: URL) throws -> NFKMLXGGUF {
        NFKMLXGGUF(format: try NFKMLXGGUFFormat(contentsOf: url))
    }

    /// Every tensor's name.
    @objc public var tensorNames: [String] { format.tensors.map(\.name) }

    /// One tensor's shape and quantization.
    @objc(infoForTensor:)
    public func info(forTensor name: String) -> NFKMLXGGUFTensorInfo? {
        format.descriptor(forTensor: name).map { NFKMLXGGUFTensorInfo($0) }
    }

    /// A string metadata value, e.g. `general.architecture`.
    @objc(metadataStringForKey:)
    public func metadataString(forKey key: String) -> String? {
        if case .string(let value)? = format.metadata[key] { return value }
        return nil
    }

    /// An integer metadata value, e.g. `llama.block_count`, or `defaultValue` when absent.
    @objc(metadataIntegerForKey:defaultValue:)
    public func metadataInteger(forKey key: String, defaultValue: Int) -> Int {
        switch format.metadata[key] {
        case .integer(let value)?: return Int(value)
        case .unsigned(let value)?: return Int(value)
        default: return defaultValue
        }
    }

    /// A floating metadata value, e.g. `llama.rope.freq_base`, or `defaultValue` when absent.
    @objc(metadataFloatForKey:defaultValue:)
    public func metadataFloat(forKey key: String, defaultValue: Float) -> Float {
        switch format.metadata[key] {
        case .double(let value)?: return Float(value)
        case .integer(let value)?: return Float(value)
        case .unsigned(let value)?: return Float(value)
        default: return defaultValue
        }
    }

    /// Whether a scalar boolean metadata value is present and true.
    @objc(metadataBoolForKey:defaultValue:)
    public func metadataBool(forKey key: String, defaultValue: Bool) -> Bool {
        if case .boolean(let value)? = format.metadata[key] { return value }
        return defaultValue
    }

    /// A string-array metadata value, e.g. `tokenizer.ggml.tokens`, or nil when absent or not an array
    /// of strings. This is how the embedded tokenizer's vocabulary and merges are read.
    public func metadataStringArray(forKey key: String) -> [String]? {
        guard case .array(let values)? = format.metadata[key] else { return nil }
        return values.compactMap { if case .string(let s) = $0 { return s }; return nil }
    }

    /// An integer-array metadata value, e.g. `tokenizer.ggml.token_type`.
    public func metadataIntegerArray(forKey key: String) -> [Int]? {
        guard case .array(let values)? = format.metadata[key] else { return nil }
        return values.compactMap {
            switch $0 {
            case .integer(let v): return Int(v)
            case .unsigned(let v): return Int(v)
            default: return nil
            }
        }
    }

    /// One tensor dequantized into an `MLXArray` of its row-major shape. Throws for a tensor whose type
    /// this reader does not implement.
    public func array(forTensor name: String) throws -> MLXArray? {
        guard let descriptor = format.descriptor(forTensor: name) else { return nil }
        return MLXArray(try format.dequantized(descriptor)).reshaped(descriptor.shape)
    }

    /// Every tensor dequantized, keyed by name.
    public func arrays() throws -> [String: MLXArray] {
        var result = [String: MLXArray]()
        for descriptor in format.tensors {
            result[descriptor.name] = MLXArray(try format.dequantized(descriptor)).reshaped(descriptor.shape)
        }
        return result
    }
}
