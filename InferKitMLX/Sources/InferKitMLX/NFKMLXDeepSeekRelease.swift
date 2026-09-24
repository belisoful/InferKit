//
//  NFKMLXDeepSeekRelease.swift
//  InferKitMLX
//
//  Building a DeepSeek V4 / V4.1 decoder from a released directory: the fit check, the weight load,
//  and the backend factories.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import InferKit
import MLX
import MLXNN

public extension NFKMLXDeepSeek {

    /// The registry name a DeepSeek backend reports.
    @objc static let deepSeekModelName = "deepseek-v4.1-flash"

    /// The bytes this configuration's parameters occupy once loaded.
    ///
    /// @discussion The release stores its weights fp8 and fp4, and the modules hold floats, so what
    /// a load needs is not what the directory measures. A released V4.1 Flash is 510 GB stored and
    /// four times that decoded, which is the number that decides whether a machine can run it.
    ///
    /// - Parameter c: the configuration whose parameters are counted.
    /// - Parameter paging: the groups held in the form the release stores them instead. Those count
    ///   at their stored size, and everything else at the width the configuration computes in.
    static func residentBytes(for c: NFKMLXDeepSeekConfiguration,
                              paging: NFKMLXDeepSeekPaging = .none) -> Int {
        expectedParameters(for: c).reduce(0) { $0 + loadedBytes($1.key, $1.value, c, paging) }
    }

    /// What one released parameter occupies once loaded.
    private static func loadedBytes(_ key: String, _ shape: [Int], _ c: NFKMLXDeepSeekConfiguration,
                                    _ paging: NFKMLXDeepSeekPaging) -> Int {
        if paging.routedExperts, routedExpertAddress(key) != nil {
            // Mapped, a group's bytes are page cache rather than an allocation, so they are not
            // counted: what the machine has to find room for is everything else.
            return paging.mapsRoutedExperts ? 0 : storedFourBitBytes(shape)
        }
        if paging.ngramTables, ngramTableLayer(key) != nil {
            return paging.mapsNgramTables ? 0 : storedEightBitBytes(shape)
        }
        let width = c.computesInBFloat16
            && !heldInFloat32(moduleKey(forRelease: key), configuration: c) ? 2 : 4
        return shape.reduce(width, *)
    }

    /// What an fp8 weight with per-row scales occupies as the release stores it: a byte a value,
    /// plus one `e8m0` scale per row per 32 columns.
    private static func storedEightBitBytes(_ shape: [Int]) -> Int {
        guard shape.count == 2 else { return shape.reduce(1, *) }
        let block = NFKMLXDeepSeekQuantization.fp4BlockSize
        return shape[0] * (shape[1] + (shape[1] + block - 1) / block)
    }

    /// What a 4-bit routed-expert matrix occupies as the release stores it: two values a byte, plus
    /// one `e8m0` scale per row per 32 columns.
    private static func storedFourBitBytes(_ shape: [Int]) -> Int {
        guard shape.count == 2 else { return shape.reduce(1, *) }
        let block = NFKMLXDeepSeekQuantization.fp4BlockSize
        return shape[0] * (shape[1] / 2 + (shape[1] + block - 1) / block)
    }

    /// The bytes the DECODER allocates, which is what a load actually needs.
    ///
    /// @discussion `residentBytes(for:paging:)` measures the release, and the release includes the
    /// DSpark draft stack: the enumeration covers `mtp.` because the structural check measures
    /// against the reference's whole module tree. `NFKMLXDeepSeekNet` builds none of it and the
    /// loader drops the prefix, so counting it in a fit check refuses a load that would have
    /// fitted. On V4.1 Flash in float32 it is 53.0 GiB of a fully paged 543.5, and the decoder is
    /// 490.5, which is the difference between a 512 GiB machine being over budget and under it.
    static func decoderBytes(for c: NFKMLXDeepSeekConfiguration,
                             paging: NFKMLXDeepSeekPaging = .none) -> Int {
        residentBytes(for: c, paging: paging) - draftStackBytes(for: c, paging: paging)
    }

    /// What the draft stack would occupy, which the decoder never holds.
    static func draftStackBytes(for c: NFKMLXDeepSeekConfiguration,
                                paging: NFKMLXDeepSeekPaging = .none) -> Int {
        expectedParameters(for: c).reduce(0) { total, entry in
            entry.key.hasPrefix("mtp.") ? total + loadedBytes(entry.key, entry.value, c, paging) : total
        }
    }

    /// Refuses a release the machine cannot hold, before any of it is materialized.
    ///
    /// @discussion The general release check reads the directory's bytes and doubles them for a
    /// half-precision release. That under-counts a block-quantized one by half: an fp8 byte becomes
    /// a float32, and a packed fp4 nibble becomes one too, so the decoded size is four to eight
    /// times the stored size rather than twice. This counts the parameters the configuration
    /// declares instead, which is the same enumeration the structural check is built on.
    ///
    /// A machine that cannot hold the decoded weights is not one step away from running the model,
    /// so the refusal names what changes the figure: holding a group in the form the release stores
    /// it and decoding what a step reads, which ``NFKMLXDeepSeekPaging`` selects.
    ///
    /// - Parameter c: the configuration to load.
    /// - Parameter budget: the bytes the machine can hold; 0 (an unknown machine) passes every load.
    /// - Parameter reserve: bytes the caller needs beside the weights, counted against the budget.
    /// - Parameter paging: what the caller has already chosen to hold stored. The check measures
    ///   that, because a refusal quoting the resident figure would send a caller to a remedy it is
    ///   already using.
    /// - Parameter includesDraftStack: counts the draft stack too, because a caller that speculates
    ///   loads it. On V4.1 Flash it is 26.6 GiB in bf16, and a check that left it out
    ///   would pass a load that then runs out of memory, which is the wrong direction to be wrong.
    static func verifyFits(_ c: NFKMLXDeepSeekConfiguration,
                           budget: Int = NFKMLXGPU.recommendedWorkingSetSize,
                           reserve: Int = 0, paging: NFKMLXDeepSeekPaging = .none,
                           includesDraftStack: Bool = false) throws {
        guard budget > 0 else { return }                // an unknown machine does not gate a load
        let resident = decoderBytes(for: c, paging: paging)
            + (includesDraftStack ? draftStackBytes(for: c, paging: paging) : 0)
        guard resident + reserve > budget else { return }
        let gib = { (bytes: Int) in String(format: "%.1f", Double(bytes) / 1_073_741_824) }
        let everything = decoderBytes(for: c, paging: .all)
        // A refusal that names only the shortfall leaves the reader to find what is big. Naming the
        // largest parameter still held decoded is that answer, and it is how the n-gram tables
        // were found to be the rest of the release once the experts were paged.
        let largest = expectedParameters(for: c)
            .filter { !(paging.routedExperts && routedExpertAddress($0.key) != nil)
                   && !(paging.ngramTables && ngramTableLayer($0.key) != nil) }
            .map { ($0.key, loadedBytes($0.key, $0.value, c, .none)) }
            .max { $0.1 < $1.1 }
        let dominant = largest.map {
            " The largest still held decoded is \($0.0), at \(gib($0.1)) GiB."
        } ?? ""
        let remedy = resident <= everything
            ? "Everything this loader can hold stored is already stored, and what remains is the "
                + "rest of the decoder." + dominant
            : "Holding every group the release stores block-quantized and decoding what a step "
                + "reads needs about \(gib(everything)) GiB, which is what NFKMLXDeepSeekPaging.all "
                + "does." + dominant
        throw NFKMLXError.unsupportedConfiguration(
            "this release needs about \(gib(resident)) GiB of parameters"
            + (reserve > 0 ? " plus \(gib(reserve)) GiB reserved" : "")
            + ", and the machine's working set is \(gib(budget)) GiB. " + remedy)
    }

    /// Loads a released directory's weights into a decoder.
    ///
    /// @discussion A key belongs to the decoder when `expectedParameters(for:)` declares it, which
    /// drops the vision tower, the aligner and the draft stack without a list of prefixes that
    /// would rot as the port grew. The names are the release's own and differ from the module's in
    /// the two places MLX's key rules force a nesting, so each one goes through
    /// `moduleKey(forRelease:)`; the decode reads the release's names, because the shape that
    /// separates a 4-bit weight from an fp8 one is keyed by them.
    ///
    /// Decoding runs a shard at a time. A weight and its `.scale` companion are usually in the same
    /// shard; where the index puts them in different ones, the earlier is held until the other
    /// arrives rather than reading the whole release into memory to pair them.
    /// A decoder built with a `paging` policy takes the same path: the groups it holds stored are
    /// diverted from the dequantizer into the modules that decode on demand. Those groups are not
    /// parameters of the module, so the coverage check cannot see one missing; each group is
    /// verified on its own terms at the end.
    static func loadWeights(into net: NFKMLXDeepSeekNet, fromDirectory directory: URL) throws {
        let paging = net.paging
        let store = net.expertStore
        let shapes = expectedParameters(for: net.configuration)
        // The enumeration covers the draft stack too, because the structural check measures it
        // against the reference's whole module tree. The decoder is not the draft stack and holds
        // none of it, so decoding those keys here would cost a release's worth of memory to
        // produce parameters nothing accepts.
        let declared = Set(shapes.keys.filter { !$0.hasPrefix("mtp.") })
        // Matched on the MODULE key, so a checkpoint written by `NFKMLXWeights/save(_:to:)` — which
        // records the module's own nesting — reloads through the same path a release takes. The
        // mapping is idempotent on a name already in module form, which is what lets one test cover
        // both.
        let declaredModuleKeys = Set(declared.map(moduleKey(forRelease:)))
        let scales = scaleOwners(inDirectory: directory, declared: declared)

        var pending = [String: MLXArray]()
        var mapped = [(String, MLXArray)]()
        for url in try NFKMLXReleaseWeights.files(inDirectory: directory) {
            // A shard holding a mapped group is never read. Reading it would materialize the very
            // tensor the mapping exists to avoid: on V4.1 Flash an n-gram table is 98 GB, and
            // `loadCheckpoint` has no way to leave one tensor behind.
            if paging.mapsNgramTables, paging.ngramTables,
               try mapTables(inFile: url, into: net, declared: declaredModuleKeys) {
                continue
            }
            // Mapping the experts still reads the shard, because the parameters beside them are
            // what the decoder is built from and the array loader cannot leave one tensor behind.
            // An expert shard is bounded — a release caps them so a loader can stream one at a time
            // — so the peak is one shard rather than the release. What changes is what is KEPT: the
            // experts become byte ranges and their arrays go out of scope with the shard.
            let mappedShard = paging.mapsRoutedExperts && store != nil
                ? (file: try NFKMLXMappedFile(url: url),
                   entries: try NFKMLXSafetensors.entries(inFile: url))
                : nil
            var shard = pending
            pending = [:]
            for (key, value) in try NFKMLXWeights.loadCheckpoint(url: url).arrays
            where declaredModuleKeys.contains(
                moduleKey(forRelease: parameterName(of: key, declared: declaredModuleKeys))) {
                shard[key] = value
            }
            // A pair split across shards waits for its other half rather than decoding against a
            // scale that is not there, which would load the raw bytes as though they were floats.
            let split = shard.keys.filter {
                isHalfOfASplitPair($0, in: shard, scales: scales, declared: declaredModuleKeys)
            }
            for key in split {
                pending[key] = shard.removeValue(forKey: key)
            }
            // A hyper-connection's own `scale` is a PARAMETER whose name ends the way a block
            // scale's does. The two are told apart by what the module declares, not by the suffix:
            // a release spells the parameter `hc_attn_scale` and never collides, but a checkpoint
            // written in the module's own layout spells it `hc_attn.scale`, and the decode — which
            // skips every `.scale` as a companion — would drop it silently.
            for key in shard.keys where isDeclaredScaleParameter(key, declared: declaredModuleKeys) {
                mapped.append((moduleKey(forRelease: key), shard.removeValue(forKey: key)!))
            }
            // A paged group never reaches the dequantizer: the point of holding it is that its
            // bytes stay bytes until a step asks for them.
            if let store {
                for key in Array(shard.keys) {
                    guard let address = routedExpertAddress(key),
                          let bytes = shard.removeValue(forKey: key) else { continue }
                    let scaleKey = key.replacingOccurrences(of: ".weight", with: ".scale")
                    let scale = shard.removeValue(forKey: scaleKey)
                    let matrix: NFKDeepSeekStoredMatrix
                    if let mappedShard, let values = mappedShard.entries[key] {
                        guard oneByteWide(values.dtype) else {
                            throw NFKMLXError.unsupportedConfiguration(
                                "\(key) is stored \(values.dtype); a mapped expert is copied out "
                                + "as bytes and this loader maps the one-byte forms only")
                        }
                        matrix = NFKDeepSeekStoredMatrix(
                            mapping: .init(file: mappedShard.file, values: values.start,
                                           valueShape: values.shape,
                                           scales: mappedShard.entries[scaleKey]?.start,
                                           scaleShape: mappedShard.entries[scaleKey]?.shape ?? []),
                            shape: shapes[key] ?? bytes.shape)
                    } else {
                        matrix = NFKDeepSeekStoredMatrix(stored: bytes, scale: scale,
                                                         shape: shapes[key] ?? bytes.shape)
                    }
                    store.store(matrix, layer: address.layer, expert: address.expert,
                                matrix: address.matrix)
                }
            }
            if paging.ngramTables {
                for key in Array(shard.keys) {
                    guard let layer = ngramTableLayer(key),
                          let bytes = shard.removeValue(forKey: key) else { continue }
                    let scale = shard.removeValue(
                        forKey: key.replacingOccurrences(of: ".weight", with: ".scale"))
                    net.layers[layer].engram?.storedTable = NFKDeepSeekStoredTable(
                        stored: bytes, scale: scale,
                        dimensions: net.configuration.engramHeadDimensions,
                        blockSize: net.configuration.fp8BlockSize)
                }
            }
            for (name, value) in dequantized(shard, shapes: shapes,
                                             fp8BlockSize: net.configuration.fp8BlockSize) {
                mapped.append((moduleKey(forRelease: name), value))
            }
        }
        guard pending.isEmpty else {
            throw NFKMLXError.unsupportedConfiguration(
                "\(pending.count) quantized tensors in \(directory.lastPathComponent) never met "
                + "the scale that decodes them, starting with \(pending.keys.sorted()[0])")
        }
        try NFKMLXWeights.apply(mapped, to: net, verifyShapes: true)
        adoptComputeType(net, configuration: net.configuration)
        try store?.verifyComplete(net.configuration)
        if paging.ngramTables {
            let absent = net.configuration.engramLayerIDs
                .filter { net.layers[$0].engram?.storedTable == nil }
            guard absent.isEmpty else {
                throw NFKMLXError.weightsMismatch(
                    "the n-gram tables of layers \(absent) are held stored and never arrived; "
                    + "a paged table is not a parameter, so nothing else reports it missing")
            }
        }
    }

    /// Loads a release's image stack: the tower, the aligner and the span's learned delimiters.
    ///
    /// @discussion None of these is a parameter of the decoder, so `expectedParameters(for:)` does
    /// not declare them and `loadWeights(into:fromDirectory:)` skips them. They load on their own
    /// terms here. The tower is the one part of this checkpoint that ships unquantized, so there is
    /// nothing to decode: a `.scale` beside any of these keys would mean the release changed, and
    /// the load says so rather than reading the bytes as floats.
    static func loadImageStack(into stack: NFKMLXDeepSeekImageStack,
                               fromDirectory directory: URL,
                               configuration: NFKMLXDeepSeekConfiguration) throws {
        var tower = [(String, MLXArray)]()
        var aligner = [(String, MLXArray)]()
        var spans = [(String, MLXArray)]()
        let delimiters = ["image_start", "image_end", "image_newline"]
        for url in try NFKMLXReleaseWeights.files(inDirectory: directory) {
            for (key, value) in try NFKMLXWeights.loadCheckpoint(url: url).arrays {
                if key.hasPrefix("vision.") {
                    tower.append((String(key.dropFirst("vision.".count)), value))
                } else if key.hasPrefix("aligner.") {
                    aligner.append((String(key.dropFirst("aligner.".count)), value))
                } else if delimiters.contains(key) {
                    spans.append((key, value))
                } else {
                    continue
                }
                guard !key.hasSuffix(".scale") else {
                    throw NFKMLXError.unsupportedConfiguration(
                        "\(key) carries a block scale, and this release's image stack is expected "
                        + "unquantized; decoding it here would read its bytes as floats")
                }
            }
        }
        try NFKMLXWeights.apply(tower, to: stack.tower, verifyShapes: true)
        try NFKMLXWeights.apply(aligner, to: stack.aligner, verifyShapes: true)
        try NFKMLXWeights.apply(spans, to: stack.delimiters, verifyShapes: true)
        adoptComputeType(stack, configuration: configuration)
    }

    /// Holds an image stack's parameters in the configuration's compute type, except the ones the
    /// release holds float32. The release stores the stack bf16, so a float32 decoder widens it.
    static func adoptComputeType(_ stack: NFKMLXDeepSeekImageStack,
                                 configuration c: NFKMLXDeepSeekConfiguration) {
        for (prefix, module) in [("vision.", stack.tower as Module), ("aligner.", stack.aligner),
                                 ("", stack.delimiters)] {
            let cast = module.parameters().flattened().map { key, value -> (String, MLXArray) in
                let held = heldInFloat32(prefix + key, configuration: c) ? DType.float32 : c.computeType
                return (key, value.asType(held))
            }
            module.update(parameters: ModuleParameters.unflattened(cast))
            eval(module)
        }
    }

    /// Loads a release's DSpark draft stack.
    ///
    /// @discussion The decoder drops `mtp.` because it builds none of it, so the draft stack loads
    /// on its own terms here — without which `generate(prompt:draft:…)` could only be handed a
    /// stack of zeros. Its experts are stored 4-bit like the decoder's and its attention fp8, so it
    /// goes through the same decode; it is small enough beside the decoder that nothing here is
    /// paged.
    static func loadDraftStack(into stack: NFKMLXDeepSeekDraftStack,
                               fromDirectory directory: URL,
                               configuration: NFKMLXDeepSeekConfiguration) throws {
        let shapes = expectedParameters(for: configuration).filter { $0.key.hasPrefix("mtp.") }
        let declared = Set(shapes.keys)
        let declaredModuleKeys = Set(declared.map(moduleKey(forRelease:)))
        let scales = scaleOwners(inDirectory: directory, declared: declared)

        var pending = [String: MLXArray]()
        var mapped = [(String, MLXArray)]()
        for url in try NFKMLXReleaseWeights.files(inDirectory: directory) {
            var shard = pending
            pending = [:]
            for (key, value) in try NFKMLXWeights.loadCheckpoint(url: url).arrays
            where declaredModuleKeys.contains(
                moduleKey(forRelease: parameterName(of: key, declared: declaredModuleKeys))) {
                shard[key] = value
            }
            for key in shard.keys.filter({
                isHalfOfASplitPair($0, in: shard, scales: scales, declared: declaredModuleKeys)
            }) {
                pending[key] = shard.removeValue(forKey: key)
            }
            for key in shard.keys where isDeclaredScaleParameter(key, declared: declaredModuleKeys) {
                mapped.append((moduleKey(forRelease: key), shard.removeValue(forKey: key)!))
            }
            for (name, value) in dequantized(shard, shapes: shapes,
                                             fp8BlockSize: configuration.fp8BlockSize) {
                // The stack holds its stages as `mtp`, so its own keys are the release's
                // `mtp.<stage>.…` and only the hyper-connection flattening has to be undone.
                mapped.append((moduleKey(forRelease: name), value))
            }
        }
        guard pending.isEmpty else {
            throw NFKMLXError.unsupportedConfiguration(
                "\(pending.count) of the draft stack's quantized tensors never met the scale that "
                + "decodes them, starting with \(pending.keys.sorted()[0])")
        }
        try NFKMLXWeights.apply(mapped, to: stack, verifyShapes: true)
        adoptComputeType(stack, configuration: configuration)
    }

    /// Registers any n-gram table this shard holds as a mapping, and reports whether the shard was
    /// taken over entirely.
    ///
    /// @discussion Returning true means the caller must not read the file: its declared contents are
    /// the mapped tables and their scales, and nothing else. A shard that mixes a table with other
    /// declared parameters cannot be handled this way, because leaving one tensor unread is not
    /// something the array loader can do; the refusal says so rather than quietly loading 98 GB.
    private static func mapTables(inFile url: URL, into net: NFKMLXDeepSeekNet,
                                  declared: Set<String>) throws -> Bool {
        let entries = try NFKMLXSafetensors.entries(inFile: url)
        let tables = entries.keys.filter { ngramTableLayer($0) != nil }
        guard !tables.isEmpty else { return false }

        let scales = Set(tables.map { $0.replacingOccurrences(of: ".weight", with: ".scale") })
        let others = entries.keys.filter {
            !tables.contains($0) && !scales.contains($0)
                && declared.contains(moduleKey(forRelease: parameterName(of: $0, declared: declared)))
        }
        guard others.isEmpty else {
            throw NFKMLXError.unsupportedConfiguration(
                "\(url.lastPathComponent) holds an n-gram table beside \(others.count) other "
                + "parameters, starting with \(others.sorted()[0]); mapping a table means never "
                + "reading its shard, and the array loader cannot leave one tensor behind")
        }

        let file = try NFKMLXMappedFile(url: url)
        for key in tables {
            guard let layer = ngramTableLayer(key), let values = entries[key] else { continue }
            let scale = entries[key.replacingOccurrences(of: ".weight", with: ".scale")]
            guard values.shape.count == 2, oneByteWide(values.dtype) else {
                throw NFKMLXError.unsupportedConfiguration(
                    "\(key) is stored \(values.dtype) at shape \(values.shape); a mapped table is "
                    + "read a row of bytes at a time and this loader maps the one-byte forms only")
            }
            net.layers[layer].engram?.storedTable = NFKDeepSeekStoredTable(
                mapping: .init(file: file, values: values.start, valueStride: values.shape[1],
                               scales: scale?.start, scaleStride: scale?.shape.last ?? 0),
                rowCount: values.shape[0],
                dimensions: net.configuration.engramHeadDimensions,
                blockSize: net.configuration.fp8BlockSize)
        }
        return true
    }

    /// Whether a safetensors dtype is one byte a value, which a row-at-a-time copy needs.
    private static func oneByteWide(_ dtype: String) -> Bool {
        ["U8", "I8", "F8_E4M3", "F8_E5M2", "F8_E8M0"].contains(dtype)
    }

    /// The layer whose n-gram table a key names, or nil where the key is anything else.
    internal static func ngramTableLayer(_ key: String) -> Int? {
        let parts = key.split(separator: ".").map(String.init)
        guard parts.count == 5, parts[0] == "layers", parts[2] == "engram", parts[3] == "embed",
              parts[4] == "weight", let layer = Int(parts[1]) else { return nil }
        return layer
    }

    /// Whether a module parameter is one the release holds float32, which a bf16 decoder leaves
    /// float32.
    ///
    /// @discussion The rule is the reference's own constructor under `set_dtype(bfloat16)`: the
    /// hyper-connection coefficients, the attention sink, the router bias, the head, the draft
    /// stack's confidence projection and Markov head, the image tower's norms, and the `wkv` and
    /// `wgate` of a compressor that pools more
    /// than one position. The release's headers agree entry for entry, each stored F32 or stored
    /// bf16 and held float32. A pooling compressor's NORM is not on the list: it runs after the
    /// pool is cast back, on a bf16 input.
    static func heldInFloat32(_ key: String, configuration c: NFKMLXDeepSeekConfiguration) -> Bool {
        let parts = key.split(separator: ".").map(String.init)
        if parts.contains(where: { ["hc_attn", "hc_ffn", "hc_head"].contains($0) }) { return true }
        if key.hasSuffix("attn_sink") || key.hasSuffix("gate.bias") || key.hasSuffix("gate.bias_vl") {
            return true
        }
        if parts.first == "vision", parts.contains(where: { ["norm", "norm1", "norm2"].contains($0) }) {
            return true
        }
        if c.holdsNormWeightsInFloat32, parts.last == "weight", parts.count >= 2,
           parts[parts.count - 2].hasSuffix("norm") {
            return true
        }
        if key == "head.weight" || key.hasSuffix("confidence_head.proj.weight")
            || key.hasSuffix("markov_head.head.weight") {
            return true
        }
        if let at = parts.firstIndex(of: "compressor"), parts.count > at + 1,
           ["wkv", "wgate", "ape"].contains(parts[at + 1]),
           parts.first == "layers", parts.count > 1, let layer = Int(parts[1]) {
            return c.compressRatio(of: layer) > 1
        }
        return false
    }

    /// Casts a module's float32 parameters to the configuration's compute type, leaving the ones
    /// the release holds float32. An identity for a float32 decoder, which is what keeps the
    /// float32 decoder exactly as it was.
    static func adoptComputeType(_ module: Module, configuration c: NFKMLXDeepSeekConfiguration) {
        guard c.computesInBFloat16 else { return }
        let cast = module.parameters().flattened().map { key, value -> (String, MLXArray) in
            guard value.dtype == .float32, !heldInFloat32(key, configuration: c) else {
                return (key, value)
            }
            return (key, value.asType(.bfloat16))
        }
        module.update(parameters: ModuleParameters.unflattened(cast))
        eval(module)
    }

    /// The layer, expert and matrix a routed-expert key names, or nil where the key is anything
    /// else. A draft stage's experts are keyed under `mtp.<stage>.ffn.` and are not these.
    internal static func routedExpertAddress(_ key: String)
        -> (layer: Int, expert: Int, matrix: String)? {
        let parts = key.split(separator: ".").map(String.init)
        guard parts.count == 7, parts[0] == "layers", parts[2] == "ffn", parts[3] == "experts",
              parts[6] == "weight", ["w1", "w2", "w3"].contains(parts[5]),
              let layer = Int(parts[1]), let expert = Int(parts[4]) else { return nil }
        return (layer, expert, parts[5])
    }

    /// The parameter a checkpoint key belongs to: itself, or the weight a block `.scale` decodes.
    private static func parameterName(of key: String, declared: Set<String>) -> String {
        guard isBlockScale(key, declared: declared) else { return key }
        return key.replacingOccurrences(of: ".scale", with: ".weight")
    }

    /// Whether this key is a block scale rather than a parameter in its own right.
    private static func isBlockScale(_ key: String, declared: Set<String>) -> Bool {
        key.hasSuffix(".scale") && !declared.contains(moduleKey(forRelease: key))
    }

    /// The converse: a declared parameter whose name happens to end in `.scale`.
    private static func isDeclaredScaleParameter(_ key: String, declared: Set<String>) -> Bool {
        key.hasSuffix(".scale") && declared.contains(moduleKey(forRelease: key))
    }

    /// Every declared weight the release stores a separate scale for, from the shard index. A
    /// single-file release has no index and needs none: nothing can be split across one shard.
    private static func scaleOwners(inDirectory directory: URL, declared: Set<String>) -> Set<String> {
        let indexURL = directory.appendingPathComponent("model.safetensors.index.json")
        guard let data = try? Data(contentsOf: indexURL),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let map = json["weight_map"] as? [String: String] else { return [] }
        return Set(map.keys.filter { $0.hasSuffix(".scale") }
            .map { $0.replacingOccurrences(of: ".scale", with: ".weight") }
            .filter(declared.contains))
    }

    /// Whether this key is one half of a weight/scale pair whose other half is in another shard.
    private static func isHalfOfASplitPair(_ key: String, in shard: [String: MLXArray],
                                           scales: Set<String>, declared: Set<String>) -> Bool {
        guard !isDeclaredScaleParameter(key, declared: declared) else { return false }
        let weight = parameterName(of: key, declared: declared)
        guard scales.contains(weight) else { return false }     // stored plain: nothing to pair
        let partner = isBlockScale(key, declared: declared) ? weight
            : weight.replacingOccurrences(of: ".weight", with: ".scale")
        return shard[partner] == nil
    }

    /// Builds a text-generation backend from a released directory.
    ///
    /// @discussion The directory supplies all three things the decoder cannot be built without: its
    /// `config.json` geometry, its `tokenizer.json`, and — for a release carrying the n-gram memory
    /// — the collapsed token map that memory addresses through, which is derived from that same
    /// tokenizer rather than shipped.
    /// - Parameter paging: what to hold in the form the release stores it rather than as float
    ///   parameters. It is what makes a release whose decoded weights exceed the machine loadable,
    ///   and it costs a decode of what each step reads.
    /// - Parameter speculates: loads the release's DSpark draft stack, which a request then turns on
    ///   with `NFKMLXGenerationParameterKey.draftTokens`. It is a choice made here rather than per
    ///   request because the stack is parameters: 26.6 GiB of them on V4.1 Flash in bf16, which the fit
    ///   check counts when this is set and the load would otherwise not have paid for.
    /// - Parameter quantizesActivations: rounds activations where the release's own
    ///   `inference/model.py` rounds them, which is closer to what DeepSeek serves. Off, the
    ///   decoder is the unquantized model those round trips approximate.
    /// - Parameter computesInFloat32: computes in float32 in place of the bf16 the release declares
    ///   and runs in. Off, the decoder is the release's own arithmetic, bit for bit; on, it is the
    ///   model that arithmetic approximates, at twice the bytes a step reads.
    static func backend(directoryURL: URL,
                        paging: NFKMLXDeepSeekPaging,
                        speculates: Bool = false,
                        quantizesActivations: Bool = false,
                        computesInFloat32: Bool = false,
                        options: NFKMLXGenerationOptions = NFKMLXGenerationOptions())
        throws -> any NFKInferenceBackend {
        var configuration = try self.configuration(
            fromHuggingFace: directoryURL.appendingPathComponent("config.json"))
        configuration.quantizesActivations = quantizesActivations
        if computesInFloat32 { configuration.computesInBFloat16 = false }
        let drafts = speculates && makeDraftStack(configuration) != nil
        try verifyFits(configuration, reserve: paging.routedExperts ? paging.expertCacheBytes : 0,
                       paging: paging, includesDraftStack: drafts)
        let tokenizerURL = directoryURL.appendingPathComponent("tokenizer.json")
        guard let tokenizer = self.tokenizer(fromTokenizerJSON: tokenizerURL) else {
            throw NFKMLXError.unsupportedConfiguration(
                "\(directoryURL.lastPathComponent) has no readable tokenizer.json")
        }
        let collapsed = configuration.engramLayerIDs.isEmpty ? nil
            : try compressedTokens(fromTokenizerJSON: tokenizerURL, in: configuration)
        let net = try makeNet(configuration, compressedTokens: collapsed, paging: paging)
        try loadWeights(into: net, fromDirectory: directoryURL)
        // A release that carries a tower carries an aligner and the span's delimiters with it, and
        // a decoder built without them would route an image's tokens with the second router bias
        // while having nothing to put in those positions.
        let images = try configuration.vision.map { vision -> NFKMLXDeepSeekImageStack in
            let stack = NFKMLXDeepSeekImageStack(vision)
            try loadImageStack(into: stack, fromDirectory: directoryURL, configuration: configuration)
            return stack
        }
        // The draft stack is loaded where the release names one, so speculation is reachable from
        // a directory rather than only from a caller that built a stack itself.
        let draft = try (drafts ? makeDraftStack(configuration) : nil)
            .map { stack -> NFKMLXDeepSeekDraftStack in
            try loadDraftStack(into: stack, fromDirectory: directoryURL,
                               configuration: configuration)
            return stack
        }
        var resolved = options
        if case .none = resolved.chatTemplate,
           let template = NFKMLXReleaseChatTemplate(inDirectory: directoryURL) {
            resolved.chatTemplate = .jinja(template: template)
        }
        return NFKMLXDeepSeekBackend(net: net, tokenizer: tokenizer, identifier: deepSeekModelName,
                                     images: images, draft: draft, options: resolved)
    }

    /// Builds a text-generation backend from a released directory, holding it as `residency` says.
    ///
    /// @discussion ``paging(for:residency:budget:includesDraftStack:)`` turns the residency into a
    /// paging policy; the other arguments are ``backend(directoryURL:paging:speculates:quantizesActivations:computesInFloat32:options:)``'s.
    /// Under the default ``NFKMLXResidency/automatic`` a release that fits loads resident and one that
    /// does not is paged, where it was refused before.
    ///
    /// Introduced in InferKit 0.4.0.
    static func backend(directoryURL: URL,
                        residency: NFKMLXResidency = .automatic,
                        speculates: Bool = false,
                        quantizesActivations: Bool = false,
                        computesInFloat32: Bool = false,
                        options: NFKMLXGenerationOptions = NFKMLXGenerationOptions())
        throws -> any NFKInferenceBackend {
        var configuration = try self.configuration(
            fromHuggingFace: directoryURL.appendingPathComponent("config.json"))
        if computesInFloat32 { configuration.computesInBFloat16 = false }
        let paging = try self.paging(for: configuration, residency: residency, budget: NFKMLXResidencyBudget.current(),
                                     includesDraftStack: speculates && makeDraftStack(configuration) != nil)
        return try backend(directoryURL: directoryURL, paging: paging, speculates: speculates,
                           quantizesActivations: quantizesActivations, computesInFloat32: computesInFloat32,
                           options: options)
    }

    /// The paging policy a residency comes to for a configuration, planned by
    /// ``NFKMLXResidencyBudget`` like every other paged model.
    ///
    /// @discussion The release is one stage whose pageable bytes are what ``NFKMLXDeepSeekPaging/fullyMapped``
    /// leaves in the release: the routed experts and the n-gram tables.
    /// - `.resident` and `.staged` → ``NFKMLXDeepSeekPaging/none``. One stage has nothing to take turns
    ///   with, so staging it is loading it; `.resident` throws where the decoded weights are known not
    ///   to fit.
    /// - `.paged` → ``NFKMLXDeepSeekPaging/fullyMapped``, every paged group left in the release.
    /// - `.automatic` → `none` where the decoded weights fit; otherwise ``NFKMLXDeepSeekPaging/mapped``
    ///   where the experts held stored in memory fit beside the cache, which decodes without reading
    ///   the release; otherwise `fullyMapped`.
    ///
    /// A paged policy's `expertCacheBytes` is the plan's cache.
    static func paging(for configuration: NFKMLXDeepSeekConfiguration, residency: NFKMLXResidency, budget: Int,
                       includesDraftStack: Bool) throws -> NFKMLXDeepSeekPaging {
        func bytes(_ paging: NFKMLXDeepSeekPaging) -> Int {
            decoderBytes(for: configuration, paging: paging)
                + (includesDraftStack ? draftStackBytes(for: configuration, paging: paging) : 0)
        }
        let whole = bytes(.none)
        let plan = try NFKMLXResidencyBudget.plan(
            [NFKMLXStageFootprint(bytes: whole, pageableBytes: whole - bytes(.fullyMapped))],
            residency: residency, budget: budget)
        guard plan.pagesExperts else { return .none }
        let holdsStored = residency == .automatic
            && NFKMLXResidencyBudget.holds(bytes(.mapped) + plan.expertCacheBytes, budget: budget)
        var paging: NFKMLXDeepSeekPaging = holdsStored ? .mapped : .fullyMapped
        paging.expertCacheBytes = plan.expertCacheBytes
        return paging
    }

    /// ``NFKMLXDeepSeekPaging/defaultExpertCacheBytes``, where Objective-C can read it: the policy
    /// itself is a Swift struct and does not bridge.
    @objc static let defaultExpertCacheBytes = NFKMLXDeepSeekPaging.defaultExpertCacheBytes

    /// The Objective-C entry: builds a DeepSeek text-generation backend from a release directory, held
    /// as ``NFKMLXResidency/automatic`` decides.
    @objc(deepSeekBackendWithDirectoryURL:error:)
    static func deepSeekBackend(directoryURL: URL) throws -> any NFKInferenceBackend {
        try backend(directoryURL: directoryURL, residency: .automatic)
    }

    /// The Objective-C entry that holds the release as `residency` says. Introduced in InferKit 0.4.0.
    @objc(deepSeekBackendWithDirectoryURL:residency:error:)
    static func deepSeekBackend(directoryURL: URL, residency: NFKMLXResidency) throws -> any NFKInferenceBackend {
        try backend(directoryURL: directoryURL, residency: residency)
    }

    /// The Objective-C entry that reaches every load choice.
    @objc(deepSeekBackendWithDirectoryURL:options:error:)
    static func deepSeekBackend(directoryURL: URL,
                                options: NFKMLXDeepSeekLoadOptions) throws -> any NFKInferenceBackend {
        guard options.paging != .none else {
            return try backend(directoryURL: directoryURL, residency: options.residency,
                               speculates: options.speculates,
                               quantizesActivations: options.quantizesActivations,
                               computesInFloat32: options.computesInFloat32)
        }
        var policy = options.paging.policy
        policy.expertCacheBytes = options.expertCacheBytes
        return try backend(directoryURL: directoryURL, paging: policy,
                           speculates: options.speculates,
                           quantizesActivations: options.quantizesActivations,
                           computesInFloat32: options.computesInFloat32)
    }

    /// The Objective-C entry for a release whose decoded weights exceed the machine.
    ///
    /// @discussion This holds every group the loader can hold stored, which is what a caller that
    /// needs paging at all is asking for; the Swift `paging:` argument selects them individually.
    /// Pass `0` for `expertCacheBytes` to decode every routed expert on every chunk that reaches it,
    /// which is the least memory and the most work.
    @objc(deepSeekPagedBackendWithDirectoryURL:expertCacheBytes:error:)
    static func deepSeekPagedBackend(directoryURL: URL,
                                     expertCacheBytes: Int) throws -> any NFKInferenceBackend {
        var paging = NFKMLXDeepSeekPaging.all
        paging.expertCacheBytes = expertCacheBytes
        return try backend(directoryURL: directoryURL, paging: paging)
    }
}
