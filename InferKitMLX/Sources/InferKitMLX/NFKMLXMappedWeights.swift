//
//  NFKMLXMappedWeights.swift
//  InferKitMLX
//
//  Reading a safetensors file's layout, and mapping one so a model can copy out the bytes it reads
//  instead of holding the whole tensor.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import MLX

/// Where a tensor sits inside a safetensors file.
///
/// @discussion The format is an 8-byte little-endian header length, that many bytes of JSON, and
/// then the data. Each tensor's `data_offsets` are relative to the end of the header, so a reader
/// that wants bytes rather than arrays needs only the header and the file.
///
/// Introduced in InferKit 0.4.0.
public struct NFKMLXSafetensorsEntry: Sendable, Equatable {
    public let dtype: String
    public let shape: [Int]
    /// Absolute byte range in the file, header included.
    public let start: Int
    public let end: Int

    public var byteCount: Int { end - start }
}

public enum NFKMLXSafetensors {

    /// Every tensor's dtype, shape and byte range, read from the header alone.
    ///
    /// @discussion This reads the header and stops. A release shard is gigabytes and its header is
    /// kilobytes, which is the whole point: a caller that means to map the file needs the layout
    /// without paying for the contents.
    public static func entries(inFile url: URL) throws -> [String: NFKMLXSafetensorsEntry] {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        guard let prefix = try handle.read(upToCount: 8), prefix.count == 8 else {
            throw NFKMLXError.unsupportedConfiguration(
                "\(url.lastPathComponent) is too short to be safetensors")
        }
        let headerLength = prefix.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self).littleEndian }
        guard headerLength > 0, headerLength < 1 << 30 else {
            throw NFKMLXError.unsupportedConfiguration(
                "\(url.lastPathComponent) declares a \(headerLength)-byte safetensors header")
        }
        guard let headerData = try handle.read(upToCount: Int(headerLength)),
              headerData.count == Int(headerLength),
              let json = try JSONSerialization.jsonObject(with: headerData) as? [String: Any] else {
            throw NFKMLXError.unsupportedConfiguration(
                "\(url.lastPathComponent) has no readable safetensors header")
        }
        let base = 8 + Int(headerLength)
        var entries = [String: NFKMLXSafetensorsEntry]()
        for (name, value) in json where name != "__metadata__" {
            guard let fields = value as? [String: Any],
                  let dtype = fields["dtype"] as? String,
                  let shape = fields["shape"] as? [Int],
                  let offsets = fields["data_offsets"] as? [Int], offsets.count == 2 else {
                continue
            }
            entries[name] = NFKMLXSafetensorsEntry(dtype: dtype, shape: shape,
                                                   start: base + offsets[0], end: base + offsets[1])
        }
        return entries
    }
}

/// A read-only mapping of a weight file, from which a model copies the bytes it actually reads.
///
/// @discussion Mapping is what turns a held tensor into a page cache. The alternative this replaces
/// is holding the bytes: a released V4.1 Flash n-gram table is 384 million rows of 256 channels, 98
/// GB stored, and a step reads a handful of those rows. Mapped, only the pages a lookup touches are
/// resident, and the operating system decides which of them to keep.
///
/// The mapping deliberately never becomes an `MLXArray`. A gather run as an MLX operation would take
/// the whole tensor as its source, and a source that large would be made resident to run it; copying
/// the rows out first means the only thing MLX ever sees is the few kilobytes that were read.
///
/// Introduced in InferKit 0.4.0.
public final class NFKMLXMappedFile: @unchecked Sendable {
    private let base: UnsafeRawPointer
    public let byteCount: Int
    public let url: URL

    public init(url: URL) throws {
        self.url = url
        let descriptor = open(url.path, O_RDONLY)
        guard descriptor >= 0 else {
            throw NFKMLXError.unsupportedConfiguration(
                "\(url.lastPathComponent) cannot be opened for mapping: \(String(cString: strerror(errno)))")
        }
        defer { close(descriptor) }              // the mapping outlives the descriptor
        var status = stat()
        guard fstat(descriptor, &status) == 0, status.st_size > 0 else {
            throw NFKMLXError.unsupportedConfiguration(
                "\(url.lastPathComponent) reports no size to map")
        }
        byteCount = Int(status.st_size)
        guard let mapped = mmap(nil, byteCount, PROT_READ, MAP_PRIVATE, descriptor, 0),
              mapped != MAP_FAILED else {
            throw NFKMLXError.unsupportedConfiguration(
                "\(url.lastPathComponent) could not be mapped: \(String(cString: strerror(errno)))")
        }
        base = UnsafeRawPointer(mapped)
    }

    deinit { munmap(UnsafeMutableRawPointer(mutating: base), byteCount) }

    /// Copies `count` bytes from `offset` into `destination`. The caller has checked the range.
    func copy(to destination: UnsafeMutableRawPointer, offset: Int, count: Int) {
        destination.copyMemory(from: base + offset, byteCount: count)
    }

    /// A whole one-byte tensor, copied out of the mapping.
    ///
    /// @discussion Copying rather than wrapping is deliberate: an `MLXArray` over the mapping is a
    /// source MLX may make resident to run an operation, and the point of mapping is that it never
    /// is. What is copied here is one expert's matrix, a few megabytes, decoded and then let go.
    func bytes(at offset: Int, shape: [Int]) -> MLXArray {
        let count = shape.reduce(1, *)
        var buffer = [UInt8](repeating: 0, count: count)
        buffer.withUnsafeMutableBytes { copy(to: $0.baseAddress!, offset: offset, count: count) }
        return MLXArray(buffer).reshaped(shape)
    }

    /// Whether `offset ..< offset + count` lies inside the mapping.
    func contains(offset: Int, count: Int) -> Bool {
        offset >= 0 && count >= 0 && offset + count <= byteCount
    }
}
