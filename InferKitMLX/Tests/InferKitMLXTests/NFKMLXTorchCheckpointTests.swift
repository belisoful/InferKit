//
//  NFKMLXTorchCheckpointTests.swift
//  InferKitMLXTests
//
//  The torch container layer is exercised two ways: against checkpoints built byte by byte in the
//  test, which pins each container's layout without a download, and against the real checkpoints in
//  ~/.inferkit-validation/raw (skipped when absent), whose converted safetensors are a byte-level
//  oracle the offline converters already proved. Everything here is pure parsing, so it runs under
//  `swift test`.
//

import XCTest
@testable import InferKitMLX

final class NFKMLXTorchCheckpointTests: XCTestCase {

    // MARK: - Fixture checkpoints

    private func floatBytes(_ values: [Float]) -> Data {
        values.withUnsafeBytes { Data($0) }
    }

    private func zipCheckpoint(root: [UInt8], storages: [(key: String, bytes: Data)],
                               prefix: String = "archive/", extraEntries: [String] = []) -> Data {
        var builder = ZipBuilder()
        builder.add(name: prefix + "data.pkl", contents: Data([0x80, 0x02] + root + [0x2e]))
        for storage in storages {
            builder.add(name: prefix + "data/" + storage.key, contents: storage.bytes,
                        localExtra: Data(repeating: 0, count: 12))
        }
        for name in extraEntries {
            builder.add(name: prefix + name, contents: Data([0x80, 0x02, 0x4e, 0x2e]))
        }
        return builder.finished()
    }

    func testAZipCheckpointParsesTensorsAndSharedStorage() throws {
        let root = TorchOps.dict([
            ("w", TorchOps.tensor(key: "0", numel: 4, offset: 0, shape: [2, 2], stride: [2, 1])),
            ("w2", TorchOps.tensor(key: "0", numel: 4, offset: 2, shape: [2], stride: [1])),
            ("h", TorchOps.tensor(storageType: "HalfStorage", key: "1", numel: 3, offset: 0,
                                  shape: [3], stride: [1])),
        ])
        let halfBytes = Data([0x00, 0x3c, 0x00, 0x40, 0x00, 0x42])  // 1.0, 2.0, 3.0 as fp16
        let archive = zipCheckpoint(root: root, storages: [("0", floatBytes([1, 2, 3, 4])), ("1", halfBytes)])

        let contents = try NFKMLXTorchFormat.read(data: archive)
        XCTAssertEqual(Set(contents.tensors.keys), ["w", "w2", "h"])
        let w = contents.tensors["w"]!
        XCTAssertEqual(w.shape, [2, 2])
        XCTAssertEqual(w.scalarType, .float32)
        XCTAssertEqual(try contents.bytes(for: w), floatBytes([1, 2, 3, 4]))

        // The second tensor views the same storage two elements in, which is what tied weights
        // look like; its bytes are the tail without any copy of the head.
        let w2 = contents.tensors["w2"]!
        XCTAssertTrue(w2.storage === w.storage)
        XCTAssertEqual(try contents.bytes(for: w2), floatBytes([3, 4]))

        let h = contents.tensors["h"]!
        XCTAssertEqual(h.scalarType, .float16)
        XCTAssertEqual(try contents.bytes(for: h), halfBytes)
    }

    func testAWrapperKeyWinsOverRootSiblings() throws {
        let inner = TorchOps.dict([("conv.weight", TorchOps.tensor(key: "0", numel: 2, offset: 0,
                                                                   shape: [2], stride: [1]))])
        let root = TorchOps.dict([("params_ema", .raw(inner)), ("epoch", .raw(TorchOps.int(7)))])
        let archive = zipCheckpoint(root: root, storages: [("0", floatBytes([5, 6]))])
        let contents = try NFKMLXTorchFormat.read(data: archive)
        XCTAssertEqual(Array(contents.tensors.keys), ["conv.weight"])
    }

    func testNonTensorLeavesDropAndNestedDictsJoinDotted() throws {
        let nested = TorchOps.dict([("inner", TorchOps.tensor(key: "0", numel: 1, offset: 0,
                                                              shape: [1], stride: [1]))])
        let sidecar: [UInt8] = TorchOps.global("mmengine.logging", "HistoryBuffer")
            + [0x29, 0x52]  // REDUCE with no arguments: an inert node
        let root = TorchOps.dict([
            ("block", .raw(nested)),
            ("meta", .raw(sidecar)),
            ("count", .raw(TorchOps.int(3))),
        ])
        let archive = zipCheckpoint(root: root, storages: [("0", floatBytes([9]))])
        let contents = try NFKMLXTorchFormat.read(data: archive)
        XCTAssertEqual(Array(contents.tensors.keys), ["block.inner"])
    }

    func testAnOpaqueModelWrapperIsRefusedNamingItsClass() throws {
        let model: [UInt8] = TorchOps.global("ultralytics.nn.tasks", "DetectionModel") + [0x29, 0x52]
        let root = TorchOps.dict([("model", .raw(model))])
        let archive = zipCheckpoint(root: root, storages: [])
        XCTAssertThrowsError(try NFKMLXTorchFormat.read(data: archive)) { error in
            let description = (error as? NFKMLXError)?.errorDescription ?? ""
            XCTAssertTrue(description.contains("ultralytics.nn.tasks.DetectionModel"),
                          "expected the refusal to name the class: \(description)")
        }
    }

    func testATorchScriptArchiveIsRefused() throws {
        let root = TorchOps.dict([])
        let archive = zipCheckpoint(root: root, storages: [], extraEntries: ["constants.pkl"])
        XCTAssertThrowsError(try NFKMLXTorchFormat.read(data: archive)) { error in
            let description = (error as? NFKMLXError)?.errorDescription ?? ""
            XCTAssertTrue(description.contains("TorchScript"), "expected the TorchScript refusal: \(description)")
        }
    }

    func testABigEndianCheckpointIsRefused() throws {
        var builder = ZipBuilder()
        builder.add(name: "archive/data.pkl", contents: Data([0x80, 0x02, 0x7d, 0x2e]))
        builder.add(name: "archive/byteorder", contents: Data("big".utf8))
        XCTAssertThrowsError(try NFKMLXTorchFormat.read(data: builder.finished())) { error in
            let description = (error as? NFKMLXError)?.errorDescription ?? ""
            XCTAssertTrue(description.contains("big-endian"), "expected the endianness refusal: \(description)")
        }
    }

    func testATransposedViewGathersToRowMajor() throws {
        // Whisper's releases store Linear weights as transposed views: shape [2, 3] over stride
        // (1, 2) reads column-wise from the storage. The reader must hand back row-major bytes.
        let root = TorchOps.dict([("t", TorchOps.tensor(key: "0", numel: 6, offset: 0,
                                                        shape: [2, 3], stride: [1, 2]))])
        let archive = zipCheckpoint(root: root, storages: [("0", floatBytes([1, 2, 3, 4, 5, 6]))])
        let contents = try NFKMLXTorchFormat.read(data: archive)
        let tensor = try XCTUnwrap(contents.tensors["t"])
        XCTAssertFalse(tensor.isContiguous)
        XCTAssertEqual(try contents.bytes(for: tensor), floatBytes([1, 3, 5, 2, 4, 6]))
    }

    func testAStridedViewRespectsItsStorageOffset() throws {
        // A [2, 2] view into the middle of a wider matrix: offset one element in, row stride 3.
        let root = TorchOps.dict([("t", TorchOps.tensor(key: "0", numel: 6, offset: 1,
                                                        shape: [2, 2], stride: [3, 1]))])
        let archive = zipCheckpoint(root: root, storages: [("0", floatBytes([1, 2, 3, 4, 5, 6]))])
        let contents = try NFKMLXTorchFormat.read(data: archive)
        XCTAssertEqual(try contents.bytes(for: contents.tensors["t"]!), floatBytes([2, 3, 5, 6]))
    }

    func testASizeOneDimensionAcceptsAnyStride() throws {
        // PyTorch leaves the stride of an extent-1 dimension arbitrary; [1, 4] for shape [4, 1] is
        // contiguous in every byte that matters.
        let root = TorchOps.dict([("t", TorchOps.tensor(key: "0", numel: 4, offset: 0,
                                                        shape: [4, 1], stride: [1, 4]))])
        let archive = zipCheckpoint(root: root, storages: [("0", floatBytes([1, 2, 3, 4]))])
        let contents = try NFKMLXTorchFormat.read(data: archive)
        XCTAssertEqual(contents.tensors["t"]?.shape, [4, 1])
    }

    func testTheLegacyStreamParsesIncludingAStorageView() throws {
        var stream = Data([0x80, 0x02, 0x8a, 0x0a] + TorchOps.legacyMagic + [0x2e])
        stream.append(Data([0x80, 0x02] + TorchOps.int(1001) + [0x2e]))
        stream.append(Data([0x80, 0x02] + TorchOps.dict([
            ("little_endian", .raw([0x88])),
            ("protocol_version", .raw(TorchOps.int(1001))),
        ]) + [0x2e]))
        stream.append(Data([0x80, 0x02] + TorchOps.dict([
            ("a", TorchOps.legacyTensor(rootKey: "s0", numel: 4, offset: 0, shape: [4], stride: [1])),
            ("v", TorchOps.legacyTensor(rootKey: "s0", numel: 4, offset: 0, shape: [2], stride: [1],
                                        view: (key: "s0view", offset: 2, count: 2))),
        ]) + [0x2e]))
        stream.append(Data([0x80, 0x02, 0x5d, 0x28] + TorchOps.unicode("s0") + [0x65, 0x2e]))
        stream.append(contentsOf: [4, 0, 0, 0, 0, 0, 0, 0])  // the storage's element count
        stream.append(floatBytes([1, 2, 3, 4]))

        let contents = try NFKMLXTorchFormat.read(data: stream)
        XCTAssertEqual(Set(contents.tensors.keys), ["a", "v"])
        XCTAssertEqual(try contents.bytes(for: contents.tensors["a"]!), floatBytes([1, 2, 3, 4]))
        // The view starts two elements into the root storage.
        XCTAssertEqual(try contents.bytes(for: contents.tensors["v"]!), floatBytes([3, 4]))
    }

    func testALegacyBigEndianStreamIsRefused() throws {
        var stream = Data([0x80, 0x02, 0x8a, 0x0a] + TorchOps.legacyMagic + [0x2e])
        stream.append(Data([0x80, 0x02] + TorchOps.int(1001) + [0x2e]))
        stream.append(Data([0x80, 0x02] + TorchOps.dict([("little_endian", .raw([0x89]))]) + [0x2e]))
        XCTAssertThrowsError(try NFKMLXTorchFormat.read(data: stream))
    }

    func testAPickleStreamWithoutTheLegacyMagicIsMalformed() throws {
        XCTAssertThrowsError(try NFKMLXTorchFormat.read(data: Data([0x80, 0x02, 0x4e, 0x2e]))) { error in
            guard case NFKMLXError.malformedCheckpoint = error else {
                return XCTFail("expected malformedCheckpoint, got \(error)")
            }
        }
    }

    func testSafetensorsBytesAreNotMistakenForACheckpoint() {
        // A safetensors file opens with its little-endian header length.
        XCTAssertFalse(NFKMLXTorchFormat.isTorchCheckpoint(Data([0x40, 0, 0, 0, 0, 0, 0, 0, 0x7b])))
        XCTAssertTrue(NFKMLXTorchFormat.isTorchCheckpoint(Data([0x50, 0x4b, 0x03, 0x04])))
        XCTAssertTrue(NFKMLXTorchFormat.isTorchCheckpoint(Data([0x80, 0x02])))
    }

    // MARK: - The safetensors writer

    func testTheWriterEmitsAFileTheHeaderDescribes() throws {
        let root = TorchOps.dict([
            ("b", TorchOps.tensor(key: "0", numel: 2, offset: 0, shape: [2], stride: [1])),
            ("a", TorchOps.tensor(storageType: "DoubleStorage", key: "1", numel: 2, offset: 0,
                                  shape: [2], stride: [1])),
        ])
        let doubles: [Float64] = [1.5, -2.25]
        let archive = zipCheckpoint(root: root, storages: [
            ("0", floatBytes([7, 8])),
            ("1", doubles.withUnsafeBytes { Data($0) }),
        ])
        let contents = try NFKMLXTorchFormat.read(data: archive)

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("NFKMLXTorchCheckpointTests-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try NFKMLXTorchFormat.writeSafetensors(contents, to: url)

        let file = try Data(contentsOf: url)
        let parsed = try SafetensorsFile(file)
        XCTAssertEqual(Set(parsed.tensors.keys), ["a", "b"])
        XCTAssertEqual(parsed.metadata["format"], "pt")
        XCTAssertEqual(parsed.tensors["b"]?.dtype, "F32")
        XCTAssertEqual(try parsed.bytes("b", in: file), floatBytes([7, 8]))
        // Double-precision narrows to float32, as the offline converters do.
        XCTAssertEqual(parsed.tensors["a"]?.dtype, "F32")
        XCTAssertEqual(try parsed.bytes("a", in: file), floatBytes([1.5, -2.25]))
    }

    // MARK: - The real checkpoints, against the converted oracle

    private lazy var config: [String: String] = {
        var merged = ProcessInfo.processInfo.environment
        let url = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".inferkit-validation.json")
        if let data = try? Data(contentsOf: url),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: String] {
            json.forEach { merged[$0.key] = $0.value }
        }
        return merged
    }()

    private func rawPath(_ key: String, _ fileName: String) throws -> URL {
        let fallback = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".inferkit-validation/raw/\(fileName)").path
        let path = config[key] ?? fallback
        try XCTSkipUnless(FileManager.default.fileExists(atPath: path),
                          "fetch \(fileName) with Tools/validation-assets/fetch.py (looked at \(path))")
        return URL(fileURLWithPath: path)
    }

    func testTheU2NetLegacyCheckpointParses() throws {
        let contents = try NFKMLXTorchFormat.read(url: try rawPath("IK_RAW_U2NET", "u2net.pth"))
        XCTAssertEqual(contents.tensors.count, 686)
        let first = try XCTUnwrap(contents.tensors["stage1.rebnconvin.conv_s1.weight"])
        XCTAssertEqual(first.shape, [64, 3, 3, 3])
        XCTAssertEqual(first.scalarType, .float32)
        XCTAssertEqual(try contents.bytes(for: first).count, 64 * 27 * 4)
    }

    func testTheWhisperCheckpointReportsHalfPrecision() throws {
        let contents = try NFKMLXTorchFormat.read(url: try rawPath("IK_RAW_WHISPER", "whisper_tiny.pt"))
        let conv = try XCTUnwrap(contents.tensors["encoder.conv1.weight"])
        XCTAssertEqual(conv.scalarType, .float16)
    }

    func testTheRAFTCheckpointParsesWithItsSharedNormTensors() throws {
        // The reference reuses each block's norm3 inside its downsample Sequential, so the state
        // dict lists the same storage under two names; both must resolve.
        let contents = try NFKMLXTorchFormat.read(url: try rawPath("IK_RAW_RAFT", "raft_things.pth"))
        XCTAssertEqual(contents.tensors.count, 179)
        let direct = try XCTUnwrap(contents.tensors["module.cnet.layer2.0.norm3.weight"])
        let viaDownsample = try XCTUnwrap(contents.tensors["module.cnet.layer2.0.downsample.1.weight"])
        XCTAssertEqual(try contents.bytes(for: direct), try contents.bytes(for: viaDownsample))
    }

    func testThePoseCheckpointParsesDespiteItsTrainingSidecar() throws {
        // The checkpoint carries an mmengine object beside its state_dict; the offline converter
        // needs an import-hook stub to unpickle it, and this reader needs nothing.
        let contents = try NFKMLXTorchFormat.read(url: try rawPath("IK_RAW_POSE", "pose_resnet50.pth"))
        XCTAssertNotNil(contents.tensors["backbone.conv1.weight"])
        XCTAssertGreaterThan(contents.tensors.count, 100)
    }

    func testTheReaderMatchesTheConvertedFileByteForByte() throws {
        // The offline converter's own output is the oracle: for a passthrough model the raw
        // checkpoint and the converted safetensors must agree key for key and byte for byte.
        let raw = try rawPath("IK_RAW_REALESRGAN", "realesrgan_x4.pth")
        let convertedPath = try XCTUnwrap(config["IK_VAL_REALESRGAN"],
                                          "set IK_VAL_REALESRGAN in ~/.inferkit-validation.json")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: convertedPath),
                          "IK_VAL_REALESRGAN points at a missing file: \(convertedPath)")

        let contents = try NFKMLXTorchFormat.read(url: raw)
        let convertedData = try Data(contentsOf: URL(fileURLWithPath: convertedPath), options: .alwaysMapped)
        let converted = try SafetensorsFile(convertedData)
        XCTAssertEqual(Set(contents.tensors.keys), Set(converted.tensors.keys))
        for (name, tensor) in contents.tensors {
            XCTAssertEqual(try contents.bytes(for: tensor), try converted.bytes(name, in: convertedData),
                           "\(name) differs from the converted file")
        }
    }
}

/// A minimal safetensors reader for the oracle comparisons: the 8-byte header length, the JSON
/// header, and each tensor's byte range. Kept in the tests because production loading goes through
/// MLX's own reader.
private struct SafetensorsFile {
    struct Entry {
        let dtype: String
        let shape: [Int]
        let offsets: [Int]
    }
    let tensors: [String: Entry]
    let metadata: [String: String]
    let payloadStart: Int

    init(_ data: Data) throws {
        var headerLength = 0
        for shift in stride(from: 0, to: 64, by: 8) {
            headerLength |= Int(data[data.startIndex + shift / 8]) << shift
        }
        let headerData = data[data.startIndex + 8 ..< data.startIndex + 8 + headerLength]
        let json = try JSONSerialization.jsonObject(with: headerData) as? [String: Any] ?? [:]
        var tensors: [String: Entry] = [:]
        var metadata: [String: String] = [:]
        for (key, value) in json {
            if key == "__metadata__" {
                metadata = value as? [String: String] ?? [:]
                continue
            }
            guard let record = value as? [String: Any],
                  let dtype = record["dtype"] as? String,
                  let shape = record["shape"] as? [Int],
                  let offsets = record["data_offsets"] as? [Int] else { continue }
            tensors[key] = Entry(dtype: dtype, shape: shape, offsets: offsets)
        }
        self.tensors = tensors
        self.metadata = metadata
        self.payloadStart = 8 + headerLength
    }

    func bytes(_ name: String, in data: Data) throws -> Data {
        let entry = try XCTUnwrap(tensors[name], "\(name) is not in the safetensors header")
        let base = data.startIndex + payloadStart
        return data[base + entry.offsets[0] ..< base + entry.offsets[1]]
    }
}

/// Emits torch checkpoint pickle opcodes for the fixtures.
private enum TorchOps {
    static let legacyMagic: [UInt8] = [0x6c, 0xfc, 0x9c, 0x46, 0xf9, 0x20, 0x6a, 0xa8, 0x50, 0x19]

    enum Value {
        case raw([UInt8])
    }

    static func unicode(_ text: String) -> [UInt8] {
        let bytes = Array(text.utf8)
        return [0x8c, UInt8(bytes.count)] + bytes
    }

    static func global(_ module: String, _ name: String) -> [UInt8] {
        Array("c\(module)\n\(name)\n".utf8)
    }

    static func int(_ value: Int) -> [UInt8] {
        if (0 ..< 256).contains(value) {
            return [0x4b, UInt8(value)]
        }
        if (0 ..< 65536).contains(value) {
            return [0x4d, UInt8(value & 0xff), UInt8(value >> 8)]
        }
        let pattern = UInt32(bitPattern: Int32(value))
        return [0x4a] + (0 ..< 4).map { UInt8((pattern >> (8 * $0)) & 0xff) }
    }

    static func tuple(_ parts: [[UInt8]]) -> [UInt8] {
        [0x28] + parts.flatMap { $0 } + [0x74]
    }

    static func dict(_ items: [(String, Value)]) -> [UInt8] {
        var ops: [UInt8] = [0x7d]
        guard !items.isEmpty else { return ops }
        ops.append(0x28)
        for (key, value) in items {
            ops += unicode(key)
            switch value {
            case .raw(let bytes): ops += bytes
            }
        }
        ops.append(0x75)
        return ops
    }

    /// One `_rebuild_tensor_v2` over a ZIP-container persistent identifier.
    static func tensor(storageType: String = "FloatStorage", key: String, numel: Int,
                       offset: Int, shape: [Int], stride: [Int]) -> Value {
        let identifier = tuple([unicode("storage"), global("torch", storageType),
                                unicode(key), unicode("cpu"), int(numel)]) + [0x51]
        return .raw(rebuild(identifier: identifier, offset: offset, shape: shape, stride: stride))
    }

    /// One `_rebuild_tensor_v2` over a legacy persistent identifier, whose sixth element is the
    /// view entry (`NONE` when the tensor uses the root storage directly).
    static func legacyTensor(rootKey: String, numel: Int, offset: Int, shape: [Int], stride: [Int],
                             view: (key: String, offset: Int, count: Int)? = nil) -> Value {
        var viewOps: [UInt8] = [0x4e]
        if let view {
            viewOps = tuple([unicode(view.key), int(view.offset), int(view.count)])
        }
        let identifier = tuple([unicode("storage"), global("torch", "FloatStorage"),
                                unicode(rootKey), unicode("cpu"), int(numel), viewOps]) + [0x51]
        return .raw(rebuild(identifier: identifier, offset: offset, shape: shape, stride: stride))
    }

    private static func rebuild(identifier: [UInt8], offset: Int, shape: [Int], stride: [Int]) -> [UInt8] {
        global("torch._utils", "_rebuild_tensor_v2")
            + tuple([identifier, int(offset),
                     tuple(shape.map(int)), tuple(stride.map(int)),
                     [0x89], [0x7d]])
            + [0x52]
    }
}
