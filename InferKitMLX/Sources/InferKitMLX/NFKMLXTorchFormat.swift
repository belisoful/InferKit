//
//  NFKMLXTorchFormat.swift
//  InferKitMLX
//
//  Reads a PyTorch checkpoint (`.pth`, `.pt`, `.ckpt`, `.th`, HF `.bin`) without Python: the modern
//  ZIP container and the pre-1.6 multi-pickle stream. The pickle machine resolves nothing but data;
//  this layer interprets the dense-tensor reducers and turns storages into byte ranges over the
//  memory-mapped file. Foundation only, so parsing is tested directly under `swift test`; MLX
//  materialization is a separate extension.
//

import Foundation

enum NFKMLXTorchFormat {

    /// The element types a checkpoint storage can hold here. MLX-free so the parse layer needs no
    /// Metal; the MLX mapping lives with the materialization.
    enum ScalarType: Equatable {
        case float64, float32, float16, bfloat16
        case int64, int32, int16, int8, uint8, bool

        var elementSize: Int {
            switch self {
            case .float64, .int64: return 8
            case .float32, .int32: return 4
            case .float16, .bfloat16, .int16: return 2
            case .int8, .uint8, .bool: return 1
            }
        }
    }

    /// One storage blob, shared by every tensor that views it. The legacy container announces
    /// storages before their bytes arrive, so the source is filled in after the stream's data
    /// section is walked.
    final class Storage {
        enum Source {
            /// A byte range in the backing data (a stored ZIP entry, or the legacy data section).
            case range(Range<Int>)
            /// An inflated copy, for a deflated ZIP entry.
            case buffer(Data)
            /// A legacy view: a slice of another storage.
            case view(Storage, byteOffset: Int)
        }
        let scalarType: ScalarType
        let byteCount: Int
        var source: Source?

        init(scalarType: ScalarType, byteCount: Int, source: Source?) {
            self.scalarType = scalarType
            self.byteCount = byteCount
            self.source = source
        }
    }

    struct Tensor {
        let shape: [Int]
        /// Per-dimension element strides. Not always row-major: Whisper's releases store their
        /// Linear weights as transposed views, so a reader refusing non-contiguous tensors refuses
        /// every Whisper checkpoint.
        let stride: [Int]
        let scalarType: ScalarType
        let storage: Storage
        /// Into the storage, in bytes (the pickle records it in elements).
        let byteOffset: Int

        var byteCount: Int { shape.reduce(1, *) * scalarType.elementSize }

        /// Whether the elements lie row-major and dense. The stride of an extent-1 dimension is
        /// arbitrary and ignored, as PyTorch leaves it.
        var isContiguous: Bool {
            var expected = 1
            for axis in stride.indices.reversed() {
                if shape[axis] != 1, stride[axis] != expected {
                    return false
                }
                expected *= shape[axis]
            }
            return true
        }
    }

    /// A parsed checkpoint: the flattened state dict and the mapped file its ranges point into.
    struct Contents {
        let tensors: [String: Tensor]
        let backing: Data

        /// A tensor's row-major little-endian bytes: a zero-copy slice for a contiguous tensor in a
        /// stored entry, a gathered copy for a strided view (Whisper's transposed Linear weights).
        func bytes(for tensor: Tensor) throws -> Data {
            var storage = tensor.storage
            var offset = tensor.byteOffset
            while case .view(let parent, let viewOffset)? = storage.source {
                offset += viewOffset
                storage = parent
            }
            let source: Data
            let start: Int
            let limit: Int
            switch storage.source {
            case .range(let range):
                source = backing; start = range.lowerBound + offset; limit = range.upperBound
            case .buffer(let buffer):
                source = buffer; start = offset; limit = buffer.count
            case .view, .none:
                throw NFKMLXError.malformedCheckpoint("a tensor's storage was announced but its bytes never arrived")
            }
            if tensor.isContiguous {
                guard start + tensor.byteCount <= limit else {
                    throw NFKMLXError.malformedCheckpoint(
                        "a tensor reads \(tensor.byteCount) bytes at offset \(offset) of its storage's \(limit - start + offset) available bytes")
                }
                let base = source.startIndex
                return source[base + start ..< base + start + tensor.byteCount]
            }
            return try gathered(tensor, from: source, start: start, limit: limit)
        }

        /// Copies a strided tensor into row-major order, element by element with the storage offset
        /// walked incrementally, so a transposed view reads in one pass.
        private func gathered(_ tensor: Tensor, from source: Data, start: Int, limit: Int) throws -> Data {
            let shape = tensor.shape
            let stride = tensor.stride
            let elementSize = tensor.scalarType.elementSize
            guard stride.allSatisfy({ $0 >= 0 }) else {
                throw NFKMLXError.unsupportedConfiguration(
                    "a tensor with a negative stride \(stride) is not supported; save it with .contiguous()")
            }
            var lastElement = 0
            for (extent, step) in zip(shape, stride) where extent > 0 {
                lastElement += (extent - 1) * step
            }
            guard start + (lastElement + 1) * elementSize <= limit else {
                throw NFKMLXError.malformedCheckpoint(
                    "a strided tensor (shape \(shape), stride \(stride)) reads past its storage")
            }

            let count = shape.reduce(1, *)
            var result = Data(count: count * elementSize)
            result.withUnsafeMutableBytes { destinationBuffer in
                source.withUnsafeBytes { sourceBuffer in
                    let destination = destinationBuffer.baseAddress!
                    let sourceBase = sourceBuffer.baseAddress! + start
                    var indices = [Int](repeating: 0, count: shape.count)
                    var elementOffset = 0
                    var written = 0
                    for _ in 0 ..< count {
                        memcpy(destination + written, sourceBase + elementOffset * elementSize, elementSize)
                        written += elementSize
                        var axis = shape.count - 1
                        while axis >= 0 {
                            indices[axis] += 1
                            elementOffset += stride[axis]
                            if indices[axis] < shape[axis] {
                                break
                            }
                            elementOffset -= stride[axis] * shape[axis]
                            indices[axis] = 0
                            axis -= 1
                        }
                    }
                }
            }
            return result
        }
    }

    /// Whether the leading bytes look like a PyTorch checkpoint: a ZIP archive or a protocol-2+
    /// pickle stream. Safetensors starts with its little-endian header length, which collides with
    /// neither.
    static func isTorchCheckpoint(_ header: Data) -> Bool {
        let bytes = [UInt8](header.prefix(512))
        guard bytes.count >= 2 else { return false }
        if bytes.count >= 4, bytes[0] == 0x50, bytes[1] == 0x4b, bytes[2] == 0x03, bytes[3] == 0x04 {
            return true
        }
        if bytes[0] == 0x80, (0x02 ... 0x05).contains(bytes[1]) {
            return true
        }
        return isTar(bytes)
    }

    /// A POSIX/ustar tar carries its magic at offset 257, so the sniff needs the first block, not
    /// the first four bytes. A `.nemo` release is a tar wrapping an ordinary torch checkpoint.
    private static func isTar(_ bytes: [UInt8]) -> Bool {
        guard bytes.count >= 262 else { return false }
        return bytes[257] == 0x75 && bytes[258] == 0x73 && bytes[259] == 0x74
            && bytes[260] == 0x61 && bytes[261] == 0x72   // "ustar"
    }

    static func read(url: URL) throws -> Contents {
        let data = try Data(contentsOf: url, options: .alwaysMapped)
        do {
            return try read(data: data)
        } catch let error as NFKMLXError {
            throw annotated(error, with: url.lastPathComponent)
        }
    }

    static func read(data: Data) throws -> Contents {
        if isTar([UInt8](data.prefix(512))) {
            return try readTar(data)
        }
        guard isTorchCheckpoint(data) else {
            throw NFKMLXError.malformedCheckpoint(
                "the data starts with neither a ZIP archive nor a pickle stream, so it is not a PyTorch checkpoint")
        }
        if data.first == 0x50 {
            return try readZip(data)
        }
        return try readLegacy(data)
    }

    // MARK: - The tar wrapper (.nemo)

    /// A `.nemo` (and any tar wrapping a checkpoint) holds a config beside the weights. The first
    /// member whose bytes are themselves a torch checkpoint is read; the rest (a YAML config) are
    /// skipped. PAX/GNU metadata entries carry their payload in a data block that is stepped over.
    private static func readTar(_ data: Data) throws -> Contents {
        let reader = NFKMLXByteReader(data)
        var offset = 0
        while offset + 512 <= reader.count {
            // A zero block marks the end of the archive.
            let nameFirst = try reader.u8(offset)
            if nameFirst == 0 {
                break
            }
            let size = try tarOctal(reader, at: offset + 124, count: 12)
            let typeFlag = try reader.u8(offset + 156)
            let bodyStart = offset + 512
            let paddedSize = (size + 511) / 512 * 512
            // '0' and '\0' are regular files; a member of any other type (PAX 'x'/'g', GNU 'L'/'K',
            // directories) carries no checkpoint, so its body is stepped over.
            if typeFlag == 0x30 || typeFlag == 0x00 {
                let head = try reader.bytes(bodyStart, count: min(512, size))
                if isTorchCheckpoint(head) || isTar([UInt8](head.prefix(512))) {
                    let base = data.startIndex
                    return try read(data: data[base + bodyStart ..< base + bodyStart + size])
                }
            }
            offset = bodyStart + paddedSize
        }
        throw NFKMLXError.unsupportedConfiguration(
            "the tar archive holds no checkpoint member (a .nemo should carry model_weights.ckpt)")
    }

    /// Parses a tar header's octal size field. GNU marks a size too large for octal by setting the
    /// high bit of the first byte and storing base-256, which a real checkpoint member can reach.
    private static func tarOctal(_ reader: NFKMLXByteReader, at offset: Int, count: Int) throws -> Int {
        let first = try reader.u8(offset)
        if first & 0x80 != 0 {
            var value = first & 0x7f
            for index in 1 ..< count {
                value = value << 8 | (try reader.u8(offset + index))
            }
            return value
        }
        var value = 0
        for index in 0 ..< count {
            let byte = try reader.u8(offset + index)
            guard byte >= 0x30, byte <= 0x37 else { break }   // an octal digit; stop at NUL or space
            value = value * 8 + (byte - 0x30)
        }
        return value
    }

    // MARK: - The modern ZIP container

    private static func readZip(_ data: Data) throws -> Contents {
        let entries = try NFKMLXZipArchive.entries(in: data)
        var entriesByName: [String: NFKMLXZipArchive.Entry] = [:]
        for entry in entries {
            entriesByName[entry.name] = entry
        }

        guard let pickleEntry = entries.first(where: { $0.name == "data.pkl" || $0.name.hasSuffix("/data.pkl") }) else {
            throw NFKMLXError.malformedCheckpoint("the ZIP archive holds no data.pkl")
        }
        let prefix = String(pickleEntry.name.dropLast("data.pkl".count))
        // A TorchScript archive carries `constants.pkl` and `code/` beside `data.pkl`, and its
        // `data.pkl` stores the module as attribute-keyed state rather than the eager
        // `_parameters`/`_buffers`/`_modules` layout — walked differently, but from the same tensors.
        let isTorchScript = entriesByName["\(prefix)constants.pkl"] != nil
        if let byteorderEntry = entriesByName["\(prefix)byteorder"] {
            let order = String(decoding: try NFKMLXZipArchive.contents(of: byteorderEntry, in: data), as: UTF8.self)
            guard order.trimmingCharacters(in: .whitespacesAndNewlines) != "big" else {
                throw NFKMLXError.unsupportedConfiguration("the checkpoint was saved big-endian, which this reader does not support")
            }
        }

        var storages: [String: Storage] = [:]
        let pickleData = try NFKMLXZipArchive.contents(of: pickleEntry, in: data)
        let (root, _) = try NFKMLXPickle.load(pickleData) { identifier in
            let reference = try storageReference(from: identifier, allowView: false)
            if let existing = storages[reference.key] {
                return .external(existing)
            }
            guard let dataEntry = entriesByName["\(prefix)data/\(reference.key)"] else {
                throw NFKMLXError.malformedCheckpoint("the archive holds no storage entry data/\(reference.key)")
            }
            let source: Storage.Source
            switch dataEntry.method {
            case .stored:
                source = .range(dataEntry.dataRange)
            default:
                source = .buffer(try NFKMLXZipArchive.contents(of: dataEntry, in: data))
            }
            let storage = Storage(scalarType: reference.scalarType,
                                  byteCount: dataEntry.uncompressedSize, source: source)
            storages[reference.key] = storage
            return .external(storage)
        }
        return Contents(tensors: try stateDict(from: root, torchScript: isTorchScript), backing: data)
    }

    // MARK: - The legacy multi-pickle stream

    /// The stream is five pickles back to back — a magic number, a protocol version, system info,
    /// the checkpoint object, and the storage keys in write order — followed by each storage's
    /// element count and raw bytes. Storages are announced inside the fourth pickle and their byte
    /// ranges assigned while walking the data section.
    private static func readLegacy(_ data: Data) throws -> Contents {
        let legacyMagic: [UInt8] = [0x6c, 0xfc, 0x9c, 0x46, 0xf9, 0x20, 0x6a, 0xa8, 0x50, 0x19]
        var cursor = 0

        let (magic, afterMagic) = try NFKMLXPickle.load(data, from: cursor) { $0 }
        guard case .bigint(let payload) = magic, [UInt8](payload) == legacyMagic else {
            throw NFKMLXError.malformedCheckpoint("the stream does not open with the legacy checkpoint magic number")
        }
        cursor = afterMagic
        let (_, afterProtocol) = try NFKMLXPickle.load(data, from: cursor) { $0 }
        cursor = afterProtocol
        let (systemInfo, afterSystemInfo) = try NFKMLXPickle.load(data, from: cursor) { $0 }
        if case .dict(let info) = systemInfo, case .bool(false)? = info["little_endian"] {
            throw NFKMLXError.unsupportedConfiguration("the checkpoint was saved big-endian, which this reader does not support")
        }
        cursor = afterSystemInfo

        var storages: [String: Storage] = [:]
        let (root, afterRoot) = try NFKMLXPickle.load(data, from: cursor) { identifier in
            let reference = try storageReference(from: identifier, allowView: true)
            let rootStorage: Storage
            if let existing = storages[reference.key] {
                rootStorage = existing
            } else {
                rootStorage = Storage(scalarType: reference.scalarType,
                                      byteCount: reference.elementCount * reference.scalarType.elementSize,
                                      source: nil)
                storages[reference.key] = rootStorage
            }
            guard let view = reference.view else {
                return .external(rootStorage)
            }
            if let existing = storages[view.key] {
                return .external(existing)
            }
            let viewStorage = Storage(scalarType: reference.scalarType,
                                      byteCount: view.elementCount * reference.scalarType.elementSize,
                                      source: .view(rootStorage, byteOffset: view.elementOffset * reference.scalarType.elementSize))
            storages[view.key] = viewStorage
            return .external(viewStorage)
        }
        cursor = afterRoot
        let (keyList, afterKeys) = try NFKMLXPickle.load(data, from: cursor) { $0 }
        guard case .list(let keys) = keyList else {
            throw NFKMLXError.malformedCheckpoint("the legacy stream's fifth pickle is not the storage key list")
        }
        cursor = afterKeys

        let reader = NFKMLXByteReader(data)
        for keyValue in keys.items {
            guard case .string(let key) = keyValue else {
                throw NFKMLXError.malformedCheckpoint("the legacy storage key list holds a non-string key")
            }
            guard let storage = storages[key] else {
                throw NFKMLXError.malformedCheckpoint(
                    "the legacy data section carries storage \(key), which no tensor announced")
            }
            var elementCount = 0
            for shift in stride(from: 0, to: 64, by: 8) {
                elementCount |= try reader.u8(cursor) << shift
                cursor += 1
            }
            let byteCount = elementCount * storage.scalarType.elementSize
            guard cursor + byteCount <= data.count else {
                throw NFKMLXError.malformedCheckpoint("the legacy storage \(key) claims bytes past the end of the file")
            }
            storage.source = .range(cursor ..< cursor + byteCount)
            cursor += byteCount
        }
        return Contents(tensors: try stateDict(from: root), backing: data)
    }

    // MARK: - Storage identifiers

    private struct StorageReference {
        let scalarType: ScalarType
        let key: String
        let elementCount: Int
        let view: (key: String, elementOffset: Int, elementCount: Int)?
    }

    /// Interprets a `BINPERSID` tuple: `('storage', type, key, location, numel)` in the ZIP
    /// container, with a trailing view entry in the legacy stream. `location` (`cpu`, `cuda:0`) is
    /// where the tensor lived when saved and does not affect its bytes.
    private static func storageReference(from identifier: NFKMLXPickleValue,
                                         allowView: Bool) throws -> StorageReference {
        guard case .tuple(let elements) = identifier, elements.count >= 5,
              case .string("storage") = elements[0],
              case .string(let key) = elements[2],
              case .int(let elementCount) = elements[4] else {
            throw NFKMLXError.malformedCheckpoint("a persistent identifier is not a storage tuple")
        }
        let scalarType = try self.scalarType(of: elements[1])
        var view: (key: String, elementOffset: Int, elementCount: Int)?
        if allowView, elements.count >= 6, case .tuple(let viewElements) = elements[5] {
            guard viewElements.count >= 3,
                  case .string(let viewKey) = viewElements[0],
                  case .int(let offset) = viewElements[1],
                  case .int(let count) = viewElements[2] else {
                throw NFKMLXError.malformedCheckpoint("a legacy storage view entry is not a (key, offset, size) tuple")
            }
            view = (viewKey, Int(offset), Int(count))
        }
        return StorageReference(scalarType: scalarType, key: key,
                                elementCount: Int(elementCount), view: view)
    }

    private static func scalarType(of value: NFKMLXPickleValue) throws -> ScalarType {
        guard case .global(let module, let name) = value, module.hasPrefix("torch") else {
            throw NFKMLXError.malformedCheckpoint("a storage tuple's type is not a torch storage class")
        }
        switch name {
        case "FloatStorage": return .float32
        case "HalfStorage": return .float16
        case "BFloat16Storage": return .bfloat16
        case "DoubleStorage": return .float64
        case "LongStorage": return .int64
        case "IntStorage": return .int32
        case "ShortStorage": return .int16
        case "CharStorage": return .int8
        case "ByteStorage": return .uint8
        case "BoolStorage": return .bool
        default:
            throw NFKMLXError.unsupportedConfiguration(
                "torch.\(name) storages (quantized, complex, or sparse) are not supported")
        }
    }

    // MARK: - The state dict

    /// The wrapper keys the offline converters collectively unwrap. The first tensor-bearing one
    /// wins; a root that carries tensors directly is already the state dict.
    private static let wrapperKeys = [
        "state_dict", "model_state_dict", "params_ema", "params", "model", "generator", "state",
    ]

    private static func stateDict(from root: NFKMLXPickleValue,
                                  torchScript: Bool = false) throws -> [String: Tensor] {
        // A TorchScript archive stores its module as attribute-keyed state (`visual`, `conv1`,
        // `weight`, …) rather than `_parameters`/`_buffers`/`_modules`. The tensors are the same
        // `_rebuild_tensor_v2` records, and walking the attributes reproduces the names
        // `module.state_dict()` composes — no serialized `code/` is interpreted.
        if torchScript {
            if case .opaque(let node) = root {
                var result: [String: Tensor] = [:]
                walkScriptedModule(node, prefix: "", into: &result)
                if !result.isEmpty {
                    return result
                }
            }
            throw NFKMLXError.unsupportedConfiguration(
                "the TorchScript archive's root is not a walkable module; use the model's Tools converter")
        }
        // A checkpoint that pickled a live `nn.Module` (YOLO's DetectionModel) is walked into its
        // state dict, the same names `module.state_dict()` composes. No class is constructed; only
        // the standard `_parameters`/`_buffers`/`_modules` state the module carries is read.
        if case .opaque(let node) = root {
            if isModuleNode(node) {
                var result: [String: Tensor] = [:]
                walkModule(node, prefix: "", into: &result)
                if !result.isEmpty {
                    return result
                }
            }
            throw NFKMLXError.unsupportedConfiguration(
                "the checkpoint's root is a \(node.qualifiedName) object, which requires its Python class to interpret; use the model's Tools converter")
        }
        guard case .dict(let rootDict) = root else {
            throw NFKMLXError.malformedCheckpoint("the checkpoint's root is not a dictionary")
        }
        // A wrapper wins over the root: a training checkpoint keeps optimizer tensors beside its
        // state_dict, and flattening the root first would sweep those in.
        var flattened: [String: Tensor] = [:]
        for wrapper in wrapperKeys {
            guard case .dict(let candidate)? = rootDict[wrapper] else { continue }
            flatten(candidate, prefix: "", into: &flattened)
            if !flattened.isEmpty {
                return flattened
            }
        }
        // A pickled module under a wrapper key: ultralytics saves `checkpoint["model"]` as a live
        // DetectionModel, which its converter reads with `.state_dict()`. Walking it here reproduces
        // that with no ultralytics classes present.
        for wrapper in wrapperKeys {
            guard case .opaque(let node)? = rootDict[wrapper], isModuleNode(node) else { continue }
            var result: [String: Tensor] = [:]
            walkModule(node, prefix: "", into: &result)
            if !result.isEmpty {
                return result
            }
        }
        flatten(rootDict, prefix: "", into: &flattened)
        if !flattened.isEmpty {
            return flattened
        }
        for (_, value) in rootDict.entries {
            if case .opaque(let node) = value {
                throw NFKMLXError.unsupportedConfiguration(
                    "the checkpoint wraps its weights in a \(node.qualifiedName) object, which requires its Python class to interpret; use the model's Tools converter")
            }
        }
        throw NFKMLXError.malformedCheckpoint(
            "no tensors were found under the checkpoint's root; its keys are \(rootDict.entries.compactMap { key, _ in describe(key) }.joined(separator: ", "))")
    }

    /// Whether a constructed object carries the state an `nn.Module` pickles: the `_parameters` /
    /// `_buffers` / `_modules` OrderedDicts. Any class whose instance holds those is walkable,
    /// whatever its type name, so ultralytics's own module classes need no definitions here.
    private static func isModuleNode(_ node: NFKMLXPickleOpaque) -> Bool {
        guard case .dict(let state)? = node.state else { return false }
        return state["_modules"] != nil || state["_parameters"] != nil
    }

    /// Emits a module's state dict, exactly as `nn.Module.state_dict()` composes it: its parameters
    /// and its persistent buffers under `prefix`, then each submodule under `prefix<name>.`. A None
    /// parameter and a non-persistent buffer are skipped, and a plain tensor attribute (a Detect
    /// head's `stride`/`anchors`) is not emitted, because it lives outside `_parameters`/`_buffers`.
    private static func walkModule(_ node: NFKMLXPickleOpaque, prefix: String,
                                   into result: inout [String: Tensor]) {
        guard case .dict(let state)? = node.state else { return }
        if case .dict(let parameters)? = state["_parameters"] {
            for (keyValue, value) in parameters.entries {
                guard let name = describe(keyValue), let tensor = tensor(from: value) else { continue }
                result[prefix + name] = tensor
            }
        }
        if case .dict(let buffers)? = state["_buffers"] {
            let nonPersistent = stringSet(state["_non_persistent_buffers_set"])
            for (keyValue, value) in buffers.entries {
                guard let name = describe(keyValue), !nonPersistent.contains(name),
                      let tensor = tensor(from: value) else { continue }
                result[prefix + name] = tensor
            }
        }
        if case .dict(let modules)? = state["_modules"] {
            for (keyValue, value) in modules.entries {
                guard let name = describe(keyValue), case .opaque(let submodule) = value else { continue }
                walkModule(submodule, prefix: prefix + name + ".", into: &result)
            }
        }
    }

    /// Emits a scripted module's state dict. A TorchScript module's state is a flat dict keyed by
    /// attribute name: a tensor entry is a leaf (`weight`, `in_proj_weight`), an object entry is a
    /// submodule to recurse (`visual`, `resblocks`, whose keys are `0`/`1`/…), and `training` and
    /// other scalars are skipped. This composes the same names `nn.Module.state_dict()` does.
    private static func walkScriptedModule(_ node: NFKMLXPickleOpaque, prefix: String,
                                           into result: inout [String: Tensor]) {
        guard case .dict(let state)? = node.state else { return }
        for (keyValue, value) in state.entries {
            guard let name = describe(keyValue) else { continue }
            if let tensor = tensor(from: value) {
                result[prefix + name] = tensor
            } else if case .opaque(let submodule) = value, case .dict? = submodule.state {
                walkScriptedModule(submodule, prefix: prefix + name + ".", into: &result)
            }
        }
    }

    /// The string members of a pickled `set` (the value model holds a set as a list).
    private static func stringSet(_ value: NFKMLXPickleValue?) -> Set<String> {
        guard case .list(let list)? = value else { return [] }
        var names = Set<String>()
        for item in list.items {
            if case .string(let name) = item {
                names.insert(name)
            }
        }
        return names
    }

    private static func flatten(_ dict: NFKMLXPickleDict, prefix: String,
                                into result: inout [String: Tensor]) {
        for (keyValue, value) in dict.entries {
            guard let key = describe(keyValue) else { continue }
            let name = prefix.isEmpty ? key : "\(prefix).\(key)"
            if let tensor = tensor(from: value) {
                result[name] = tensor
            } else if case .dict(let nested) = value {
                flatten(nested, prefix: name, into: &result)
            }
        }
    }

    private static func describe(_ key: NFKMLXPickleValue) -> String? {
        switch key {
        case .string(let name): return name
        case .int(let value): return String(value)
        default: return nil
        }
    }

    private static func tensor(from value: NFKMLXPickleValue) -> Tensor? {
        guard case .opaque(let node) = value else { return nil }
        if node.module == "torch._utils", node.name == "_rebuild_parameter" {
            guard let wrapped = node.args.first else { return nil }
            return tensor(from: wrapped)
        }
        guard node.module == "torch._utils",
              node.name == "_rebuild_tensor_v2" || node.name == "_rebuild_tensor",
              node.args.count >= 4,
              case .external(let boxed) = node.args[0],
              let storage = boxed as? Storage,
              case .int(let elementOffset) = node.args[1],
              let shape = integers(node.args[2]),
              let stride = integers(node.args[3]),
              shape.count == stride.count else {
            return nil
        }
        return Tensor(shape: shape, stride: stride, scalarType: storage.scalarType, storage: storage,
                      byteOffset: Int(elementOffset) * storage.scalarType.elementSize)
    }

    private static func integers(_ value: NFKMLXPickleValue) -> [Int]? {
        var elements: [NFKMLXPickleValue]
        switch value {
        case .tuple(let items): elements = items
        case .list(let list): elements = list.items
        // torch.Size reduces to a construction over the underlying tuple.
        case .opaque(let node) where node.name == "Size":
            guard let first = node.args.first else { return nil }
            return integers(first)
        default: return nil
        }
        var result: [Int] = []
        result.reserveCapacity(elements.count)
        for element in elements {
            guard case .int(let item) = element else { return nil }
            result.append(Int(item))
        }
        return result
    }

    // MARK: - Materialization and the safetensors writer

    /// A tensor's bytes as a consumer holds them: float64 narrows to float32, matching the offline
    /// converters, so a double-precision release does not force a precision MLX has no fast path for.
    static func materialized(_ tensor: Tensor, in contents: Contents) throws -> (bytes: Data, scalarType: ScalarType) {
        let raw = try contents.bytes(for: tensor)
        guard tensor.scalarType == .float64 else {
            return (raw, tensor.scalarType)
        }
        var doubles = [Float64](repeating: 0, count: raw.count / 8)
        _ = doubles.withUnsafeMutableBytes { raw.copyBytes(to: $0) }
        let floats = doubles.map { Float32($0) }
        return (floats.withUnsafeBytes { Data($0) }, .float32)
    }

    /// Writes the state dict as a safetensors file, the format every loader here reads. The file
    /// carries no `inferkit.layout` metadata, which is the marker for PyTorch tensor layout: a
    /// loader applies its conv transposes exactly as it does to an offline-converted file.
    static func writeSafetensors(_ contents: Contents, to url: URL) throws {
        var header: [String: Any] = ["__metadata__": ["format": "pt"]]
        var payload = Data()
        for name in contents.tensors.keys.sorted() {
            let tensor = contents.tensors[name]!
            let (bytes, scalarType) = try materialized(tensor, in: contents)
            header[name] = [
                "dtype": safetensorsName(scalarType),
                "shape": tensor.shape,
                "data_offsets": [payload.count, payload.count + bytes.count],
            ]
            payload.append(bytes)
        }
        var headerData = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        while !headerData.count.isMultiple(of: 8) {
            headerData.append(0x20)
        }
        var file = Data()
        for shift in stride(from: 0, to: 64, by: 8) {
            file.append(UInt8((UInt64(headerData.count) >> UInt64(shift)) & 0xff))
        }
        file.append(headerData)
        file.append(payload)
        try file.write(to: url)
    }

    private static func safetensorsName(_ scalarType: ScalarType) -> String {
        switch scalarType {
        case .float64: return "F64"
        case .float32: return "F32"
        case .float16: return "F16"
        case .bfloat16: return "BF16"
        case .int64: return "I64"
        case .int32: return "I32"
        case .int16: return "I16"
        case .int8: return "I8"
        case .uint8: return "U8"
        case .bool: return "BOOL"
        }
    }

    private static func annotated(_ error: NFKMLXError, with name: String) -> NFKMLXError {
        switch error {
        case .malformedCheckpoint(let detail): return .malformedCheckpoint("\(name): \(detail)")
        case .unsupportedConfiguration(let detail): return .unsupportedConfiguration("\(name): \(detail)")
        default: return error
        }
    }
}
