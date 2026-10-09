//
//  NFKMLXLayerStreaming.swift
//  InferKitMLX
//
//  Running a dense decoder larger than the working set one layer at a time: the layers that do not
//  fit stay in the release and are read in their turn on every pass, the next one read while the
//  current one computes.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import MLX
import MLXNN

/// One tensor of a streamed layer: its name inside the layer's module and where the release stores it.
struct NFKMLXStreamedTensor {
    let name: String
    let file: Int
    let entry: NFKMLXSafetensorsEntry
    let dtype: DType
}

/// A tensor read out of the release into memory of its own, before MLX wraps it.
private struct NFKMLXStreamedBuffer {
    let name: String
    let pointer: UnsafeMutableRawPointer
    let shape: [Int]
    let dtype: DType
}

/// One layer's read, finished or still running on the reader.
private final class NFKMLXPendingLayer: @unchecked Sendable {
    private let done = DispatchSemaphore(value: 0)
    private var buffers = [NFKMLXStreamedBuffer]()
    private var failure: Error?

    func finish(_ buffers: [NFKMLXStreamedBuffer]) {
        self.buffers = buffers
        done.signal()
    }

    func fail(_ error: Error) {
        failure = error
        done.signal()
    }

    /// The read's buffers once it finishes; the caller owns them from here.
    func wait() throws -> [NFKMLXStreamedBuffer] {
        done.wait()
        if let failure { throw failure }
        return buffers
    }
}

/// The decoder layers of a release left on disk and read in their turn.
///
/// @discussion A dense decoder computes every layer on every token, so a release larger than the
/// working set has nothing to leave behind the way a mixture leaves its routed experts. What it can do
/// is hold one layer at a time. The stream holds the layers that fit and reads each of the rest from
/// the release when the pass reaches it, then lets it go once the layer's output is evaluated.
///
/// - The read bypasses the file cache (`F_NOCACHE`). Every streamed layer is read again on the next
///   pass, and on a machine that cannot hold the release the cache cannot either, so caching the
///   reads only evicts what other programs hold.
/// - Each tensor is read in one `pread` into page-aligned memory of its own, which MLX wraps without
///   copying where Metal accepts the allocation and copies once where it does not.
/// - The next streamed layer is read on a background queue while the current one computes, so a pass
///   holds at most two streamed layers. Reading is cyclic: after the last streamed layer the first is
///   read, ready for the next pass.
///
/// A layer streams with the values a resident load gives it: the same stored tensors, widened to
/// float32 by the same exact conversion at `.float32` precision. A streamed pass therefore computes
/// what a resident one does, read for read.
///
/// Introduced in InferKit 0.4.0.
public final class NFKMLXLayerStream: @unchecked Sendable {
    private let files: [URL]
    private let descriptors: [Int32]
    private let tensors: [Int: [NFKMLXStreamedTensor]]
    /// The streamed layer indices, ascending, the order a pass reads them in.
    public let layers: [Int]
    private let precision: NFKMLXWeightPrecision
    private let reader = DispatchQueue(label: "inferkit.mlx.layer-stream", qos: .userInitiated)
    private let lock = NSLock()
    private var pending = [Int: NFKMLXPendingLayer]()
    private var failure: Error?
    private var placeholders = [Int: [(String, MLXArray)]]()
    private var readBytes = 0
    private var readSeconds = 0.0

    /// Opens the release's weight files for the tensors of `layers`.
    ///
    /// - Parameters:
    ///   - files: the release's weight files.
    ///   - tensors: each streamed layer's tensors, named as the layer's module names its parameters.
    ///   - precision: `.float32` widens a 16-bit tensor as it is installed; `.checkpoint` keeps it.
    init(files: [URL], tensors: [Int: [NFKMLXStreamedTensor]], precision: NFKMLXWeightPrecision) throws {
        var descriptors = [Int32]()
        for url in files {
            let descriptor = Darwin.open(url.path, O_RDONLY)
            guard descriptor >= 0 else {
                descriptors.forEach { close($0) }
                throw NFKMLXError.unsupportedConfiguration(
                    "\(url.lastPathComponent) cannot be opened for streaming: \(String(cString: strerror(errno)))")
            }
            _ = fcntl(descriptor, F_NOCACHE, 1)
            descriptors.append(descriptor)
        }
        self.files = files
        self.descriptors = descriptors
        self.tensors = tensors
        self.layers = tensors.keys.sorted()
        self.precision = precision
    }

    deinit {
        for layer in pending.values {
            (try? layer.wait())?.forEach { free($0.pointer) }
        }
        descriptors.forEach { close($0) }
    }

    /// Whether layer `index` is streamed rather than held.
    public func streams(_ index: Int) -> Bool { tensors[index] != nil }

    /// The bytes one pass reads: every streamed layer's tensors as stored.
    public var bytesPerPass: Int {
        tensors.values.reduce(0) { total, layer in total + layer.reduce(0) { $0 + $1.entry.byteCount } }
    }

    /// The bytes read from the release so far, and the seconds the reader spent reading them.
    public var readStatistics: (bytes: Int, seconds: Double) {
        lock.lock(); defer { lock.unlock() }
        return (readBytes, readSeconds)
    }

    /// The first read that failed, or nil while every read has succeeded. A layer whose read failed is
    /// skipped, so a pass that meets a failure computes nothing meaningful; a caller that can throw
    /// checks this after each pass.
    public var readFailure: Error? {
        lock.lock(); defer { lock.unlock() }
        return failure
    }

    /// Starts reading the first streamed layer, so it is ready when a pass first reaches it.
    func prime() {
        guard let first = layers.first else { return }
        schedule(first)
    }

    /// Fills `module` with layer `index`'s weights for its turn, and starts reading the next streamed
    /// layer. Returns false, recording ``readFailure``, when the read failed and the layer must be
    /// skipped.
    func install(_ index: Int, into module: Module) -> Bool {
        let layer = schedule(index)
        lock.lock()
        pending[index] = nil
        lock.unlock()
        if let position = layers.firstIndex(of: index) {
            schedule(layers[(position + 1) % layers.count])
        }
        do {
            let arrays = try layer.wait().map { buffer -> (String, MLXArray) in
                let array = MLXArray(rawPointer: buffer.pointer, buffer.shape, dtype: buffer.dtype) {
                    free(buffer.pointer)
                }
                let widens = precision == .float32 && (buffer.dtype == .float16 || buffer.dtype == .bfloat16)
                return (buffer.name, widens ? array.asType(.float32) : array)
            }
            try NFKMLXWeights.apply(arrays, to: module)
            return true
        } catch {
            lock.lock()
            if failure == nil { failure = error }
            lock.unlock()
            return false
        }
    }

    /// Lets layer `index`'s weights go once `outputs` (the layer's result, and whatever state it wrote)
    /// are evaluated, so nothing the pass keeps still refers to them.
    func release(_ index: Int, from module: Module, after outputs: [MLXArray]) {
        eval(outputs)
        let empty = placeholders[index] ?? module.parameters().flattened().map { name, _ in
            (name, MLXArray.zeros([0]))
        }
        placeholders[index] = empty
        module.update(parameters: ModuleParameters.unflattened(empty))
    }

    /// Replaces a freshly built layer's parameters with empty placeholders, so the random weights its
    /// initializer described are dropped before anything evaluates them.
    func vacate(_ index: Int, module: Module) {
        let empty = module.parameters().flattened().map { name, _ in (name, MLXArray.zeros([0])) }
        placeholders[index] = empty
        module.update(parameters: ModuleParameters.unflattened(empty))
    }

    /// The read of layer `index`, started now unless it is already running.
    @discardableResult
    private func schedule(_ index: Int) -> NFKMLXPendingLayer {
        lock.lock()
        if let running = pending[index] {
            lock.unlock()
            return running
        }
        let layer = NFKMLXPendingLayer()
        pending[index] = layer
        lock.unlock()
        let parts = tensors[index] ?? []
        reader.async { [descriptors, files] in
            let start = Date()
            var buffers = [NFKMLXStreamedBuffer]()
            do {
                for part in parts {
                    buffers.append(try Self.read(part, descriptor: descriptors[part.file], url: files[part.file]))
                }
            } catch {
                buffers.forEach { free($0.pointer) }
                layer.fail(error)
                return
            }
            let seconds = Date().timeIntervalSince(start)
            self.lock.lock()
            self.readBytes += parts.reduce(0) { $0 + $1.entry.byteCount }
            self.readSeconds += seconds
            self.lock.unlock()
            layer.finish(buffers)
        }
        return layer
    }

    /// One tensor read into page-aligned memory the caller frees.
    private static func read(_ tensor: NFKMLXStreamedTensor, descriptor: Int32, url: URL) throws -> NFKMLXStreamedBuffer {
        let count = tensor.entry.byteCount
        let page = Int(getpagesize())
        var pointer: UnsafeMutableRawPointer?
        guard posix_memalign(&pointer, page, max(page, (count + page - 1) / page * page)) == 0, let pointer else {
            throw NFKMLXError.unsupportedConfiguration("no memory for \(tensor.name), \(count) bytes")
        }
        var done = 0
        while done < count {
            let got = pread(descriptor, pointer + done, count - done, off_t(tensor.entry.start + done))
            if got < 0, errno == EINTR { continue }
            guard got > 0 else {
                free(pointer)
                let reason = got < 0 ? String(cString: strerror(errno)) : "the file ends early"
                throw NFKMLXError.unsupportedConfiguration(
                    "\(url.lastPathComponent): reading \(tensor.name) failed: \(reason)")
            }
            done += got
        }
        return NFKMLXStreamedBuffer(name: tensor.name, pointer: pointer, shape: tensor.entry.shape, dtype: tensor.dtype)
    }
}

extension NFKMLXLayerStream {

    /// A stream over the layers of a release directory that `layerOf` names, keeping only `streamed`.
    ///
    /// - Parameters:
    ///   - directory: the release, single-file or sharded.
    ///   - streamed: the layer indices to stream.
    ///   - layerOf: a checkpoint key's layer index and its name inside the layer's module, or nil for a
    ///     key outside the repeated layers.
    static func streaming(directory: URL, layers streamed: Set<Int>, precision: NFKMLXWeightPrecision,
                          layerOf: (String) -> (layer: Int, name: String)?) throws -> NFKMLXLayerStream {
        let files = try NFKMLXReleaseWeights.files(inDirectory: directory)
        var tensors = [Int: [NFKMLXStreamedTensor]]()
        for (fileIndex, url) in files.enumerated() {
            for (key, entry) in try NFKMLXSafetensors.entries(inFile: url) {
                guard let (layer, name) = layerOf(key), streamed.contains(layer) else { continue }
                guard let dtype = NFKMLXSafetensors.dtype(entry.dtype),
                      entry.byteCount == entry.shape.reduce(dtype.size, *) else {
                    throw NFKMLXError.unsupportedConfiguration(
                        "\(url.lastPathComponent) stores \(key) as \(entry.dtype), which a stream cannot read")
                }
                tensors[layer, default: []].append(
                    NFKMLXStreamedTensor(name: name, file: fileIndex, entry: entry, dtype: dtype))
            }
        }
        for layer in streamed where tensors[layer] == nil {
            throw NFKMLXError.unsupportedConfiguration("\(directory.lastPathComponent) holds no tensors for layer \(layer)")
        }
        // A layer's tensors are read in file order, so a pass sweeps each shard forward.
        for layer in tensors.keys {
            tensors[layer]?.sort { ($0.file, $0.entry.start) < ($1.file, $1.entry.start) }
        }
        return try NFKMLXLayerStream(files: files, tensors: tensors, precision: precision)
    }

    /// The layer a key of the form `<prefix>layers.<n>.<rest>` belongs to, after `decoderName` strips
    /// the release's prefix.
    static func layer(ofDecoderName name: String, layersKey: String = "layers") -> (layer: Int, name: String)? {
        let prefix = layersKey + "."
        guard name.hasPrefix(prefix) else { return nil }
        let rest = name.dropFirst(prefix.count)
        guard let dot = rest.firstIndex(of: "."), let index = Int(rest[..<dot]) else { return nil }
        return (index, String(rest[rest.index(after: dot)...]))
    }
}
