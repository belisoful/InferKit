//
//  NFKMLXGGUFFormat.swift
//  InferKitMLX
//
//  A native GGUF reader, the sequel to the native PyTorch checkpoint reader. GGUF is the format most
//  quantized language models are distributed in, and the package could not read one. This parses the
//  container and dequantizes the block-quant formats a real model uses into plain floats, with no
//  Python toolchain and no llama.cpp.
//
//  Everything here is pure Foundation below the MLX materialization (see NFKMLXGGUF), so the parsing and
//  dequantization run under `swift test`. The container is a header of typed key/value metadata and a
//  table of tensor descriptors, then the tensor data aligned to a boundary the metadata names.
//

import Foundation

/// The GGML tensor types this reader dequantizes. A file naming any other type is refused rather than
/// misread.
public enum NFKMLXGGMLType: Int, Sendable {
    case f32 = 0
    case f16 = 1
    case q4_0 = 2
    case q5_0 = 6
    case q8_0 = 8
    case q4_K = 12
    case q6_K = 14

    /// How many values one block holds.
    var blockSize: Int {
        switch self {
        case .f32, .f16: return 1
        case .q4_0, .q5_0, .q8_0: return 32
        case .q4_K, .q6_K: return 256
        }
    }

    /// How many bytes one block occupies.
    var typeSize: Int {
        switch self {
        case .f32: return 4
        case .f16: return 2
        case .q4_0: return 18          // fp16 d + 16 packed nibbles
        case .q5_0: return 22          // fp16 d + 4 high-bit bytes + 16 packed nibbles
        case .q8_0: return 34          // fp16 d + 32 int8
        case .q4_K: return 144         // fp16 d + fp16 dmin + 12 packed scales + 128 nibbles
        case .q6_K: return 210         // 128 low nibbles + 64 high bits + 16 int8 scales + fp16 d
        }
    }
}

/// One tensor's descriptor: its name, its shape in row-major (MLX) order, its type, and where its data
/// begins relative to the file's tensor-data region.
public struct NFKMLXGGUFTensorDescriptor: Sendable {
    public let name: String
    public let shape: [Int]
    /// The type, or nil when the file names a GGML type this reader does not dequantize. The tensor is
    /// still listed so a consumer sees the whole model; reading it throws.
    public let type: NFKMLXGGMLType?
    /// The raw GGML type number, for reporting an unsupported type.
    public let rawType: Int
    let offset: Int
    let elementCount: Int
}

/// A value read from GGUF metadata. Only the shapes a loader reads are surfaced.
public enum NFKMLXGGUFValue: Sendable {
    case integer(Int64)
    case unsigned(UInt64)
    case double(Double)
    case boolean(Bool)
    case string(String)
    case array([NFKMLXGGUFValue])
}

/// The parsed GGUF container: its metadata, its tensor descriptors, and a dequantizer over the mapped
/// data region.
public struct NFKMLXGGUFFormat {
    public let metadata: [String: NFKMLXGGUFValue]
    public let tensors: [NFKMLXGGUFTensorDescriptor]
    private let data: Data
    private let dataStart: Int

    private static let magic: UInt32 = 0x4655_4747       // "GGUF" little-endian

    public init(contentsOf url: URL) throws {
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        var cursor = 0
        guard data.count >= 8, NFKMLXGGUFFormat.readU32(data, &cursor) == NFKMLXGGUFFormat.magic else {
            throw NFKMLXError.malformedCheckpoint("not a GGUF file")
        }
        let version = NFKMLXGGUFFormat.readU32(data, &cursor)
        guard version == 2 || version == 3 else {
            throw NFKMLXError.malformedCheckpoint("GGUF version \(version) is not supported")
        }
        let tensorCount = Int(NFKMLXGGUFFormat.readU64(data, &cursor))
        let metadataCount = Int(NFKMLXGGUFFormat.readU64(data, &cursor))

        var metadata = [String: NFKMLXGGUFValue]()
        for _ in 0 ..< metadataCount {
            let key = try NFKMLXGGUFFormat.readString(data, &cursor)
            metadata[key] = try NFKMLXGGUFFormat.readValue(data, &cursor)
        }

        var tensors = [NFKMLXGGUFTensorDescriptor]()
        for _ in 0 ..< tensorCount {
            let name = try NFKMLXGGUFFormat.readString(data, &cursor)
            let dimensionCount = Int(NFKMLXGGUFFormat.readU32(data, &cursor))
            // GGUF stores the fastest-varying dimension first; a row-major shape is the reverse.
            var dimensions = (0 ..< dimensionCount).map { _ in Int(NFKMLXGGUFFormat.readU64(data, &cursor)) }
            let elementCount = dimensions.reduce(1, *)
            dimensions.reverse()
            // An unknown type does not fail the whole file: the tensor is listed and only reading it throws.
            let rawType = Int(NFKMLXGGUFFormat.readU32(data, &cursor))
            let offset = Int(NFKMLXGGUFFormat.readU64(data, &cursor))
            tensors.append(NFKMLXGGUFTensorDescriptor(name: name, shape: dimensions,
                                                      type: NFKMLXGGMLType(rawValue: rawType),
                                                      rawType: rawType, offset: offset,
                                                      elementCount: elementCount))
        }

        let alignment: Int
        if case .unsigned(let value)? = metadata["general.alignment"] {
            alignment = Int(value)
        } else if case .integer(let value)? = metadata["general.alignment"] {
            alignment = Int(value)
        } else {
            alignment = 32
        }
        self.metadata = metadata
        self.tensors = tensors
        self.data = data
        self.dataStart = (cursor + alignment - 1) / alignment * alignment
    }

    public func descriptor(forTensor name: String) -> NFKMLXGGUFTensorDescriptor? {
        tensors.first { $0.name == name }
    }

    /// The dequantized values of one tensor, in row-major order.
    public func dequantized(_ descriptor: NFKMLXGGUFTensorDescriptor) throws -> [Float] {
        guard let type = descriptor.type else {
            throw NFKMLXError.malformedCheckpoint(
                "\(descriptor.name) uses GGML type \(descriptor.rawType), which this reader does not dequantize")
        }
        let blocks = descriptor.elementCount / type.blockSize
        let base = dataStart + descriptor.offset
        var output = [Float](repeating: 0, count: descriptor.elementCount)
        data.withUnsafeBytes { raw in
            let bytes = raw.baseAddress!.advanced(by: base).assumingMemoryBound(to: UInt8.self)
            output.withUnsafeMutableBufferPointer { out in
                for block in 0 ..< blocks {
                    let source = bytes.advanced(by: block * type.typeSize)
                    let destination = out.baseAddress!.advanced(by: block * type.blockSize)
                    dequantizeBlock(type, source, into: destination)
                }
            }
        }
        return output
    }

    // MARK: Dequantizers

    private func dequantizeBlock(_ type: NFKMLXGGMLType, _ source: UnsafePointer<UInt8>,
                                 into destination: UnsafeMutablePointer<Float>) {
        switch type {
        case .f32:
            destination[0] = source.withMemoryRebound(to: Float.self, capacity: 1) { $0[0] }
        case .f16:
            destination[0] = Float(halfBits: readU16(source))
        case .q8_0:
            let d = Float(halfBits: readU16(source))
            let qs = source.advanced(by: 2)
            for i in 0 ..< 32 { destination[i] = d * Float(Int8(bitPattern: qs[i])) }
        case .q4_0:
            let d = Float(halfBits: readU16(source))
            let qs = source.advanced(by: 2)
            for i in 0 ..< 16 {
                destination[i] = d * Float(Int(qs[i] & 0x0F) - 8)
                destination[i + 16] = d * Float(Int(qs[i] >> 4) - 8)
            }
        case .q5_0:
            let d = Float(halfBits: readU16(source))
            let qh = UInt32(source[2]) | (UInt32(source[3]) << 8) | (UInt32(source[4]) << 16) | (UInt32(source[5]) << 24)
            let qs = source.advanced(by: 6)
            for i in 0 ..< 16 {
                let high0 = Int((qh >> UInt32(i)) & 1) << 4
                let high1 = Int((qh >> UInt32(i + 16)) & 1) << 4
                destination[i] = d * Float((Int(qs[i] & 0x0F) | high0) - 16)
                destination[i + 16] = d * Float((Int(qs[i] >> 4) | high1) - 16)
            }
        case .q4_K:
            dequantizeQ4K(source, into: destination)
        case .q6_K:
            dequantizeQ6K(source, into: destination)
        }
    }

    /// Q4_K: a 256-value super-block of eight 32-value sub-blocks. A block-wide `d` and `dmin` scale
    /// per-sub-block 6-bit scales and mins, and each value is a 4-bit quant: `d·sc·q − dmin·min`.
    private func dequantizeQ4K(_ source: UnsafePointer<UInt8>, into destination: UnsafeMutablePointer<Float>) {
        let d = Float(halfBits: readU16(source))
        let dmin = Float(halfBits: readU16(source.advanced(by: 2)))
        let scales = source.advanced(by: 4)                    // 12 packed bytes
        let qs = source.advanced(by: 16)                       // 128 nibble bytes

        for sub in 0 ..< 8 {
            let (scale, minimum) = scaleAndMinimumQ4K(scales, sub)
            let d1 = d * Float(scale)
            let m1 = dmin * Float(minimum)
            let group = sub / 2, nibble = (sub % 2) * 4
            for i in 0 ..< 32 {
                let q = Int((qs[group * 32 + i] >> UInt8(nibble)) & 0x0F)
                destination[sub * 32 + i] = d1 * Float(q) - m1
            }
        }
    }

    /// The 6-bit scale and min of sub-block `index`, unpacked from Q4_K's 12 scale bytes.
    private func scaleAndMinimumQ4K(_ scales: UnsafePointer<UInt8>, _ index: Int) -> (Int, Int) {
        if index < 4 {
            return (Int(scales[index] & 0x3F), Int(scales[index + 4] & 0x3F))
        }
        let low = index - 4
        let scale = Int(scales[8 + low] & 0x0F) | (Int(scales[low] >> 6) << 4)
        let minimum = Int(scales[8 + low] >> 4) | (Int(scales[4 + low] >> 6) << 4)
        return (scale, minimum)
    }

    /// Q6_K: a 256-value super-block of sixteen 16-value sub-blocks. Each value is a 6-bit quant (a 4-bit
    /// low part and a 2-bit high part) centered at 32, scaled by `d` and a per-sub-block int8 scale.
    private func dequantizeQ6K(_ source: UnsafePointer<UInt8>, into destination: UnsafeMutablePointer<Float>) {
        let ql = source                                        // 128 bytes
        let qh = source.advanced(by: 128)                      // 64 bytes
        let scales = source.advanced(by: 192)                  // 16 int8
        let d = Float(halfBits: readU16(source.advanced(by: 208)))

        for group in 0 ..< 2 {                                 // two halves of 128 values
            let qlBase = ql.advanced(by: group * 64)
            let qhBase = qh.advanced(by: group * 32)
            let scaleBase = group * 8
            let outBase = destination.advanced(by: group * 128)
            for l in 0 ..< 32 {
                let is0 = l / 16
                let low = qlBase[l], low2 = qlBase[l + 32], high = qhBase[l]
                let q1 = (Int(low & 0x0F) | ((Int(high >> 0) & 3) << 4)) - 32
                let q2 = (Int(low2 & 0x0F) | ((Int(high >> 2) & 3) << 4)) - 32
                let q3 = (Int(low >> 4) | ((Int(high >> 4) & 3) << 4)) - 32
                let q4 = (Int(low2 >> 4) | ((Int(high >> 6) & 3) << 4)) - 32
                outBase[l] = d * Float(Int8(bitPattern: scales[scaleBase + is0])) * Float(q1)
                outBase[l + 32] = d * Float(Int8(bitPattern: scales[scaleBase + is0 + 2])) * Float(q2)
                outBase[l + 64] = d * Float(Int8(bitPattern: scales[scaleBase + is0 + 4])) * Float(q3)
                outBase[l + 96] = d * Float(Int8(bitPattern: scales[scaleBase + is0 + 6])) * Float(q4)
            }
        }
    }

    // MARK: Reading primitives

    private func readU16(_ pointer: UnsafePointer<UInt8>) -> UInt16 {
        UInt16(pointer[0]) | (UInt16(pointer[1]) << 8)
    }

    private static func readU32(_ data: Data, _ cursor: inout Int) -> UInt32 {
        defer { cursor += 4 }
        return data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: cursor, as: UInt32.self) }
    }

    private static func readU64(_ data: Data, _ cursor: inout Int) -> UInt64 {
        defer { cursor += 8 }
        return data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: cursor, as: UInt64.self) }
    }

    private static func readString(_ data: Data, _ cursor: inout Int) throws -> String {
        let length = Int(readU64(data, &cursor))
        defer { cursor += length }
        let bytes = data.subdata(in: cursor ..< cursor + length)
        return String(decoding: bytes, as: UTF8.self)
    }

    private static func readValue(_ data: Data, _ cursor: inout Int) throws -> NFKMLXGGUFValue {
        let type = readU32(data, &cursor)
        return try readValue(ofType: type, data, &cursor)
    }

    private static func readValue(ofType type: UInt32, _ data: Data, _ cursor: inout Int) throws -> NFKMLXGGUFValue {
        switch type {
        case 0: defer { cursor += 1 }; return .unsigned(UInt64(data[data.startIndex + cursor]))
        case 1: defer { cursor += 1 }; return .integer(Int64(Int8(bitPattern: data[data.startIndex + cursor])))
        case 2: return .unsigned(UInt64(readU16Scalar(data, &cursor)))
        case 3: return .integer(Int64(Int16(bitPattern: readU16Scalar(data, &cursor))))
        case 4: return .unsigned(UInt64(readU32(data, &cursor)))
        case 5: return .integer(Int64(Int32(bitPattern: readU32(data, &cursor))))
        case 6: return .double(Double(Float(bitPattern: readU32(data, &cursor))))
        case 7: defer { cursor += 1 }; return .boolean(data[data.startIndex + cursor] != 0)
        case 8: return .string(try readString(data, &cursor))
        case 9:
            let elementType = readU32(data, &cursor)
            let count = Int(readU64(data, &cursor))
            var elements = [NFKMLXGGUFValue]()
            elements.reserveCapacity(count)
            for _ in 0 ..< count { elements.append(try readValue(ofType: elementType, data, &cursor)) }
            return .array(elements)
        case 10: return .unsigned(readU64(data, &cursor))
        case 11: return .integer(Int64(bitPattern: readU64(data, &cursor)))
        case 12: return .double(readU64(data, &cursor).withDouble)
        default: throw NFKMLXError.malformedCheckpoint("GGUF metadata value type \(type) is not supported")
        }
    }

    private static func readU16Scalar(_ data: Data, _ cursor: inout Int) -> UInt16 {
        defer { cursor += 2 }
        return data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: cursor, as: UInt16.self) }
    }
}

private extension UInt64 {
    var withDouble: Double { Double(bitPattern: self) }
}

private extension Float {
    /// A 32-bit float from IEEE half-precision bits.
    init(halfBits bits: UInt16) {
        self = Float(Float16(bitPattern: bits))
    }
}
