//
//  NFKMLXRepositoryMatchingTests.swift
//  InferKitMLXTests
//
//  Whether the language loader reads a repository, judged from files placed in a hub cache so no request
//  leaves the process: a small Qwen3 decoder measured into a descriptor, a small Llama decoder resolved
//  through NFKModelResolver against a stubbed hub, each refusal and its code, the
//  catalog's model types, the mlx-community releases in the validation store judged from their own files,
//  and, gated behind INFERKIT_LIVE_HUB, the runtime measurement of a shipped release against the figures
//  Tools/catalog/measure.py recorded for it. Nothing here reaches MLX.
//

import CryptoKit
import XCTest
import InferKit
@testable import InferKitMLX

/// Answers a hub's requests from files served in memory, as the hub at `http://hub.test` would.
private final class NFKMLXStubHub: URLProtocol {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var responses = [String: (data: Data, headers: [String: String])]()

    static let endpointURL = URL(string: "http://hub.test")!

    static var session: URLSession {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [NFKMLXStubHub.self]
        return URLSession(configuration: configuration)
    }

    static func reset() {
        lock.withLock { responses = [:] }
    }

    /// Serves `files` at `commit`, its tree listing, and its revision information naming `commit`.
    static func serve(_ repository: String, commit: String, files: [String: Data], card: [String: Any]) {
        var listing = [[String: Any]]()
        lock.withLock {
            for (path, data) in files.sorted(by: { $0.key < $1.key }) {
                let sha = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
                listing.append(["type": "file", "path": path, "size": data.count, "oid": String(sha.prefix(40))])
                responses["/\(repository)/resolve/\(commit)/\(path)"] = (data, ["ETag": "\"\(sha)\""])
            }
            let tree = try! JSONSerialization.data(withJSONObject: listing)
            responses["/api/models/\(repository)/tree/\(commit)"] = (tree, [:])
            let information = try! JSONSerialization.data(withJSONObject: [
                "id": repository, "sha": commit, "gated": false, "cardData": card] as [String: Any])
            responses["/api/models/\(repository)/revision/main"] = (information, [:])
        }
    }

    override class func canInit(with request: URLRequest) -> Bool { request.url?.host == endpointURL.host }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        let url = request.url!
        let served = Self.lock.withLock { Self.responses[url.path] }
        let response = HTTPURLResponse(url: url, statusCode: served == nil ? 404 : 200, httpVersion: "HTTP/1.1",
                                       headerFields: served?.headers ?? [:])!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: served?.data ?? Data())
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}

final class NFKMLXRepositoryMatchingTests: XCTestCase {

    private static let repository = "org/qwen3-tiny"
    private static let commit = "0123456789abcdef0123456789abcdef01234567"
    private static let qwen2Pattern =
        #"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#
    private static let llama3Pattern =
        #"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#

    private var cache: URL!

    override func setUp() {
        super.setUp()
        cache = FileManager.default.temporaryDirectory
            .appendingPathComponent("NFKMLXRepositoryMatchingTests-\(UUID().uuidString)")
    }

    override func tearDown() {
        NFKMLXStubHub.reset()
        try? FileManager.default.removeItem(at: cache)
        super.tearDown()
    }

    // MARK: Fixtures

    /// A two-layer Qwen3 decoder's configuration: 64 wide, two query heads and one key-value head of 32.
    private static var configuration: [String: Any] {
        ["model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"], "hidden_size": 64, "num_hidden_layers": 2,
         "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 32, "intermediate_size": 128,
         "vocab_size": 256, "max_position_embeddings": 512, "tie_word_embeddings": true, "rope_theta": 1_000_000,
         "rms_norm_eps": 1e-6, "torch_dtype": "bfloat16"]
    }

    /// A two-layer Llama 3 decoder of ``configuration``'s geometry: a llama3-scaled rotary, tied embeddings,
    /// and no query and key norms.
    private static var llamaConfiguration: [String: Any] {
        ["model_type": "llama", "architectures": ["LlamaForCausalLM"], "hidden_size": 64, "num_hidden_layers": 2,
         "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 32, "intermediate_size": 128,
         "vocab_size": 256, "max_position_embeddings": 131_072, "tie_word_embeddings": true, "rope_theta": 500_000,
         "rope_scaling": ["rope_type": "llama3", "factor": 32, "low_freq_factor": 1, "high_freq_factor": 4,
                          "original_max_position_embeddings": 8192],
         "rms_norm_eps": 1e-5, "hidden_act": "silu", "mlp_bias": false, "attention_bias": false,
         "torch_dtype": "bfloat16"]
    }

    /// The tensors of ``llamaConfiguration`` in bfloat16.
    private static var llamaTensors: [(String, [Int])] {
        tensors.filter { !$0.0.hasSuffix("q_norm.weight") && !$0.0.hasSuffix("k_norm.weight") }
    }

    private static let llamaRepository = "org/llama-tiny"
    private static let mistralTemplate = "mistral-small-24b-instruct-2501"

    /// The tensors of ``configuration`` in bfloat16, by name and shape.
    private static var tensors: [(String, [Int])] {
        var tensors: [(String, [Int])] = [("model.embed_tokens.weight", [256, 64]), ("model.norm.weight", [64])]
        for layer in 0..<2 {
            let prefix = "model.layers.\(layer)."
            tensors += [
                (prefix + "self_attn.q_proj.weight", [64, 64]), (prefix + "self_attn.k_proj.weight", [32, 64]),
                (prefix + "self_attn.v_proj.weight", [32, 64]), (prefix + "self_attn.o_proj.weight", [64, 64]),
                (prefix + "self_attn.q_norm.weight", [32]), (prefix + "self_attn.k_norm.weight", [32]),
                (prefix + "mlp.gate_proj.weight", [128, 64]), (prefix + "mlp.up_proj.weight", [128, 64]),
                (prefix + "mlp.down_proj.weight", [64, 128]), (prefix + "input_layernorm.weight", [64]),
                (prefix + "post_attention_layernorm.weight", [64]),
            ]
        }
        return tensors
    }

    /// The tensors of ``configuration`` as mlx_lm stores them at 4 bits in groups of 64: every projection
    /// and the embedding packed into `U32` words with bfloat16 scales and biases, the norms in bfloat16.
    private static var packedTensors: [(String, String, [Int])] {
        tensors.flatMap { name, shape -> [(String, String, [Int])] in
            guard shape.count == 2 else { return [(name, "BF16", shape)] }
            let base = String(name.dropLast(".weight".count))
            return [(name, "U32", [shape[0], shape[1] / 8]), (base + ".scales", "BF16", [shape[0], shape[1] / 64]),
                    (base + ".biases", "BF16", [shape[0], shape[1] / 64])]
        }
    }

    /// The bytes a tensor list holds, by its dtypes.
    private static func storedBytes(_ tensors: [(String, String, [Int])]) -> Int64 {
        tensors.reduce(0) { $0 + Int64($1.2.reduce(1, *) * (elementBytes[$1.1] ?? 0)) }
    }

    private static let elementBytes = ["BF16": 2, "F16": 2, "F32": 4, "U32": 4, "U8": 1, "I8": 1, "I32": 4, "I64": 8,
                                       "F8_E4M3": 1, "F8_E5M2": 1]

    /// A safetensors file of `tensors`, each in its own dtype, zero-filled.
    private static func safetensors(typed tensors: [(String, String, [Int])], metadata: [String: String]? = nil) -> Data {
        var header = [String: Any]()
        var offset = 0
        for (name, dtype, shape) in tensors {
            let bytes = shape.reduce(1, *) * (elementBytes[dtype] ?? 0)
            header[name] = ["dtype": dtype, "shape": shape, "data_offsets": [offset, offset + bytes]]
            offset += bytes
        }
        header["__metadata__"] = metadata
        let json = try! JSONSerialization.data(withJSONObject: header)
        var length = UInt64(json.count).littleEndian
        var data = Data(bytes: &length, count: 8)
        data.append(json)
        data.append(Data(count: offset))
        return data
    }

    /// A safetensors file of `tensors` in `dtype`, zero-filled.
    private static func safetensors(_ tensors: [(String, [Int])], dtype: String = "BF16", elementBytes: Int = 2,
                                    metadata: [String: String]? = nil) -> Data {
        var header = [String: Any]()
        var offset = 0
        for (name, shape) in tensors {
            let bytes = shape.reduce(1, *) * elementBytes
            header[name] = ["dtype": dtype, "shape": shape, "data_offsets": [offset, offset + bytes]]
            offset += bytes
        }
        header["__metadata__"] = metadata
        let json = try! JSONSerialization.data(withJSONObject: header)
        var length = UInt64(json.count).littleEndian
        var data = Data(bytes: &length, count: 8)
        data.append(json)
        data.append(Data(count: offset))
        return data
    }

    private static func tokenizer(pattern: String = qwen2Pattern, model: [String: Any] = [:],
                                  preTokenizer: [String: Any]? = nil,
                                  decoder: [String: Any] = ["type": "ByteLevel"]) -> [String: Any] {
        let split: [String: Any] = ["type": "Split", "pattern": ["Regex": pattern], "behavior": "Isolated",
                                    "invert": false]
        let byteLevel: [String: Any] = ["type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false,
                                        "use_regex": false]
        return ["model": ["type": "BPE", "vocab": [String: Int](), "merges": [String]()].merging(model) { $1 },
                "pre_tokenizer": preTokenizer ?? ["type": "Sequence", "pretokenizers": [split, byteLevel]],
                "decoder": decoder]
    }

    /// Mistral's SentencePiece-style tokenizer: a Metaspace that prepends `▁` to the first segment only,
    /// with byte fallback.
    private static var metaspaceTokenizer: [String: Any] {
        let decoders: [[String: Any]] = [
            ["type": "Replace", "pattern": ["String": "▁"], "content": " "], ["type": "ByteFallback"],
            ["type": "Fuse"], ["type": "Strip", "content": " ", "start": 1, "stop": 0]]
        return tokenizer(model: ["byte_fallback": true, "unk_token": "<unk>"],
                         preTokenizer: ["type": "Metaspace", "replacement": "▁", "prepend_scheme": "first",
                                        "split": false],
                         decoder: ["type": "Sequence", "decoders": decoders])
    }

    private static func json(_ object: Any) -> Data {
        try! JSONSerialization.data(withJSONObject: object)
    }

    /// The repository's files placed in the hub cache at ``commit``, and the repository a resolver reads.
    /// A repository's files: its configuration, tokenizer, and weights.
    private static func files(configuration: [String: Any], tokenizer: [String: Any]?, weights: [String: Data]?,
                              extra: [String: Data] = [:]) -> [String: Data] {
        var files: [String: Data] = ["config.json": json(configuration),
                                     "tokenizer_config.json": json(["tokenizer_class": "Qwen2Tokenizer"])]
        files["tokenizer.json"] = tokenizer.map(json)
        files.merge(weights ?? ["model.safetensors": safetensors(tensors)]) { $1 }
        files.merge(extra) { $1 }
        return files
    }

    private func repository(configuration: [String: Any] = configuration, tokenizer: [String: Any]? = tokenizer(),
                            weights: [String: Data]? = nil, extra: [String: Data] = [:],
                            name: String = repository) throws -> NFKModelRepository {
        let files = Self.files(configuration: configuration, tokenizer: tokenizer, weights: weights, extra: extra)
        let hub = NFKHFHub(cacheDirectoryURL: cache)
        var listed = [NFKDownloadPlanFile]()
        for (path, data) in files.sorted(by: { $0.key < $1.key }) {
            let local = try XCTUnwrap(hub.localURL(forRepo: name, revision: Self.commit, path: path))
            try FileManager.default.createDirectory(at: local.deletingLastPathComponent(),
                                                    withIntermediateDirectories: true)
            try data.write(to: local)
            let remote = try XCTUnwrap(hub.remoteURL(forRepo: name, revision: Self.commit, path: path))
            listed.append(NFKDownloadPlanFile(path: path, remoteURL: remote, localURL: local,
                                              byteCount: Int64(data.count), sha256: nil, cached: true, optional: false))
        }
        let license = NFKModelLicense(identifier: "apache-2.0", name: "apache-2.0", url: nil)
        return NFKModelRepository(repository: name, revision: Self.commit, requestedRevision: nil,
                                  configuration: configuration, files: listed, license: license, hub: hub)
    }

    /// The Llama fixture's repository, judged directly.
    private func llamaRepository(configuration: [String: Any] = llamaConfiguration) throws -> NFKModelRepository {
        try repository(configuration: configuration, tokenizer: Self.tokenizer(pattern: Self.llama3Pattern,
                                                                              model: ["ignore_merges": true]),
                       weights: ["model.safetensors": Self.safetensors(Self.llamaTensors)], name: Self.llamaRepository)
    }

    private func template(_ identifier: String = "qwen3-0.6b") throws -> NFKModelDescriptor {
        try XCTUnwrap(NFKModelCatalog.shared.descriptor(withIdentifier: identifier))
    }

    private func assertRefusal(_ code: NFKModelResolutionError.Code, file: StaticString = #filePath, line: UInt = #line,
                               _ repository: @autoclosure () throws -> NFKModelRepository,
                               template identifier: String = "qwen3-0.6b") throws {
        let read = try repository()
        let matching = try template(identifier)
        XCTAssertThrowsError(try NFKMLXModelDescriptors.descriptor(for: read, matching: matching),
                             file: file, line: line) { error in
            let error = error as NSError
            XCTAssertEqual(error.domain, NFKModelResolutionErrorDomain, file: file, line: line)
            XCTAssertEqual(error.code, code.rawValue, "\(error.localizedDescription)", file: file, line: line)
        }
    }

    // MARK: The descriptor

    func testAQwen3RepositoryBuildsADescriptorOnTheShippedLoader() throws {
        let read = try repository()
        let descriptor = try NFKMLXModelDescriptors.descriptor(for: read, matching: try template())
        XCTAssertEqual(descriptor.identifier, Self.repository)
        XCTAssertEqual(descriptor.registryName, "qwen3-0.6b")
        XCTAssertEqual(descriptor.family, "qwen3")
        XCTAssertEqual(descriptor.engine, NFKModelEngineMLX)
        XCTAssertEqual(descriptor.modelTypes, [])
        XCTAssertEqual(descriptor.license?.identifier, "apache-2.0")
        let variant = try XCTUnwrap(descriptor.defaultVariant)
        XCTAssertEqual(variant.identifier, "qwen3-tiny")
        XCTAssertEqual(variant.registryName, "qwen3-0.6b")
        XCTAssertEqual(variant.source?.repository, Self.repository)
        XCTAssertEqual(variant.source?.revision, Self.commit)
        XCTAssertEqual(variant.parameterCount, 90_560)
        XCTAssertEqual(variant.precision, NFKModelPrecisionBFloat16)
        let listed = ["config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"]
            .map { read.file(atPath: $0)!.byteCount }.reduce(0, +)
        XCTAssertEqual(variant.downloadByteCount?.int64Value, listed)
        XCTAssertEqual(Set(variant.source?.files.map(\.path) ?? []),
                       ["config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors", "vocab.json",
                        "merges.txt", "added_tokens.json", "chat_template.jinja"])
        XCTAssertEqual(variant.source?.files.first { $0.path == "model.safetensors" }?.byteCount?.int64Value,
                       read.file(atPath: "model.safetensors")?.byteCount)
        let facts = try XCTUnwrap(variant.memoryFacts)
        XCTAssertEqual(facts.basis, NFKModelMemoryBasisMeasured)
        let stage = try XCTUnwrap(facts.stages.first)
        XCTAssertEqual(stage.byteCountByPrecision[NFKModelPrecisionCheckpoint], 181_120)
        XCTAssertEqual(stage.byteCountByPrecision[NFKModelPrecisionFloat32], 362_240)
        XCTAssertEqual(stage.quantizableElementCount, 73_728)
        XCTAssertEqual(facts.vocabularySize, 256)
        XCTAssertNotNil(facts.keyValueByteCountPerToken)
        XCTAssertEqual(descriptor.loadOptions.map(\.key), try template().loadOptions.map(\.key))
        XCTAssertNoThrow(try descriptor.estimatedMemory(forOptions: nil))
    }

    func testTheDescriptorRoundTripsThroughItsDictionary() throws {
        let descriptor = try NFKMLXModelDescriptors.descriptor(for: try repository(), matching: try template())
        let data = try JSONSerialization.data(withJSONObject: descriptor.dictionaryRepresentation())
        let object = try XCTUnwrap(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(try NFKModelDescriptor(dictionary: object), descriptor)
    }

    func testAShardedRepositoryReadsEveryShardItsIndexNames() throws {
        let tensors = Self.tensors
        let first = Array(tensors.prefix(10)), second = Array(tensors.dropFirst(10))
        var weightMap = [String: String]()
        first.forEach { weightMap[$0.0] = "model-00001-of-00002.safetensors" }
        second.forEach { weightMap[$0.0] = "model-00002-of-00002.safetensors" }
        let read = try repository(weights: [
            "model.safetensors.index.json": Self.json(["metadata": ["total_size": 181_120], "weight_map": weightMap]),
            "model-00001-of-00002.safetensors": Self.safetensors(first),
            "model-00002-of-00002.safetensors": Self.safetensors(second),
        ])
        let variant = try XCTUnwrap(try NFKMLXModelDescriptors.descriptor(for: read, matching: try template()).defaultVariant)
        XCTAssertEqual(variant.parameterCount, 90_560)
        XCTAssertEqual(variant.memoryFacts?.stages.first?.byteCountByPrecision[NFKModelPrecisionCheckpoint], 181_120)
        let paths = Set(variant.source?.files.map(\.path) ?? [])
        XCTAssertTrue(paths.contains("model.safetensors.index.json"))
        XCTAssertFalse(paths.contains("model.safetensors"))
    }

    func testAnMLXQuantizedRepositoryIsMeasuredAsStored() throws {
        var configuration = Self.configuration
        configuration["quantization"] = ["group_size": 64, "bits": 4]
        configuration["quantization_config"] = ["group_size": 64, "bits": 4]
        let packed = Self.packedTensors
        let read = try repository(configuration: configuration,
                                  weights: ["model.safetensors": Self.safetensors(typed: packed)])
        let descriptor = try NFKMLXModelDescriptors.descriptor(for: read, matching: try template())
        let variant = try XCTUnwrap(descriptor.defaultVariant)
        let stage = try XCTUnwrap(variant.memoryFacts?.stages.first)
        XCTAssertEqual(stage.byteCountByPrecision[NFKModelPrecisionCheckpoint]?.int64Value, Self.storedBytes(packed))
        XCTAssertEqual(variant.parameterCount, 90_560, "the packed words count as the weights they pack")
        XCTAssertEqual(stage.quantizableElementCount, 0)
        let keys = Set(descriptor.loadOptions.map(\.key))
        XCTAssertFalse(keys.contains(NFKModelLoadOptionPrecision), "the loader reads an mlx_lm release as stored")
        XCTAssertFalse(keys.contains(NFKModelLoadOptionQuantizationBits), "the release's layers are packed already")
        XCTAssertTrue(keys.contains(NFKModelLoadOptionContextLength))
        XCTAssertNil(descriptor.memoryModel?.precisionOptionKey)
        XCTAssertNil(descriptor.memoryModel?.quantizationBitsOptionKey)
        XCTAssertNoThrow(try descriptor.estimatedMemory(forOptions: nil))
    }

    func testAShardedMLXQuantizedRepositoryReadsEveryShard() throws {
        var configuration = Self.configuration
        configuration["quantization"] = ["group_size": 64, "bits": 4]
        let packed = Self.packedTensors
        let first = Array(packed.prefix(12)), second = Array(packed.dropFirst(12))
        var weightMap = [String: String]()
        first.forEach { weightMap[$0.0] = "model-00001-of-00002.safetensors" }
        second.forEach { weightMap[$0.0] = "model-00002-of-00002.safetensors" }
        let read = try repository(configuration: configuration, weights: [
            "model.safetensors.index.json": Self.json(["weight_map": weightMap]),
            "model-00001-of-00002.safetensors": Self.safetensors(typed: first),
            "model-00002-of-00002.safetensors": Self.safetensors(typed: second),
        ])
        let variant = try XCTUnwrap(try NFKMLXModelDescriptors.descriptor(for: read, matching: try template()).defaultVariant)
        XCTAssertEqual(variant.memoryFacts?.stages.first?.byteCountByPrecision[NFKModelPrecisionCheckpoint]?.int64Value,
                       Self.storedBytes(packed))
    }

    func testAPerLayerWidthCountsThatLayersParameters() throws {
        var configuration = Self.configuration
        configuration["quantization"] = ["group_size": 64, "bits": 4,
                                         "model.embed_tokens": ["group_size": 64, "bits": 8]]
        let packed = Self.packedTensors.map { name, dtype, shape -> (String, String, [Int]) in
            name == "model.embed_tokens.weight" ? (name, dtype, [shape[0], shape[1] * 2]) : (name, dtype, shape)
        }
        let read = try repository(configuration: configuration,
                                  weights: ["model.safetensors": Self.safetensors(typed: packed)])
        let variant = try XCTUnwrap(try NFKMLXModelDescriptors.descriptor(for: read, matching: try template()).defaultVariant)
        XCTAssertEqual(variant.parameterCount, 90_560)
    }

    func testAnInferKitQuantizedSaveIsMeasuredAsStored() throws {
        let packed = Self.packedTensors
        let read = try repository(weights: [
            "model.safetensors": Self.safetensors(typed: packed, metadata: ["inferkit.quantization": "4:64"])])
        let descriptor = try NFKMLXModelDescriptors.descriptor(for: read, matching: try template())
        XCTAssertEqual(descriptor.defaultVariant?.memoryFacts?.stages.first?
                        .byteCountByPrecision[NFKModelPrecisionCheckpoint]?.int64Value, Self.storedBytes(packed))
        XCTAssertFalse(descriptor.loadOptions.contains { $0.key == NFKModelLoadOptionPrecision })
    }

    func testTheTokenizersTheLoaderReproducesAreRead() throws {
        let tokenizers: [String: [String: Any]] = [
            "llama3": Self.tokenizer(pattern: Self.llama3Pattern),
            "ignore_merges": Self.tokenizer(pattern: Self.llama3Pattern, model: ["ignore_merges": true]),
            "nfc": Self.tokenizer().merging(["normalizer": ["type": "NFC"]]) { $1 },
            "metaspace": Self.metaspaceTokenizer,
        ]
        for (name, tokenizer) in tokenizers.sorted(by: { $0.key < $1.key }) {
            XCTAssertNoThrow(try NFKMLXModelDescriptors.descriptor(for: try repository(tokenizer: tokenizer),
                                                                   matching: try template()), name)
        }
    }

    func testALlamaRepositoryBuildsOnMistralSmall3sLoader() throws {
        let mistral = try template(Self.mistralTemplate)
        let descriptor = try NFKMLXModelDescriptors.descriptor(for: try llamaRepository(), matching: mistral)
        XCTAssertEqual(descriptor.identifier, Self.llamaRepository)
        XCTAssertEqual(descriptor.registryName, mistral.registryName)
        XCTAssertEqual(descriptor.defaultVariant?.registryName, mistral.registryName)
        XCTAssertEqual(descriptor.defaultVariant?.parameterCount, 90_432, "the Qwen3 fixture without its query and key norms")
    }

    func testALlamaRepositoryResolvesThroughTheFamilyRoute() throws {
        let files = Self.files(configuration: Self.llamaConfiguration,
                               tokenizer: Self.tokenizer(pattern: Self.llama3Pattern, model: ["ignore_merges": true]),
                               weights: ["model.safetensors": Self.safetensors(Self.llamaTensors)])
        NFKMLXStubHub.serve(Self.llamaRepository, commit: Self.commit, files: files, card: ["license": "llama3.2"])
        let hub = NFKHFHub(cacheDirectoryURL: cache)
        hub.endpointURL = NFKMLXStubHub.endpointURL
        hub.session = NFKMLXStubHub.session
        hub.accessToken = ""
        hub.excludesCacheFromBackup = false
        let resolver = NFKModelResolver(catalog: NFKModelCatalog.shared, hub: hub)
        resolver.memoryBudget = NFKModelMemoryBudget(workingSetByteCount: 16 << 30)

        let unverified = try resolver.resolution(forRepository: Self.llamaRepository, revision: nil)
        XCTAssertFalse(unverified.isResolved)
        XCTAssertEqual(unverified.reasons.last.map { $0 as NSError }?.code,
                       NFKModelResolutionError.Code.unverifiedRouteNotAllowed.rawValue)

        resolver.allowsUnverifiedRoutes = true
        let resolution = try resolver.resolution(forRepository: Self.llamaRepository, revision: nil)
        XCTAssertTrue(resolution.isResolved, "\(resolution.reasons)")
        XCTAssertEqual(resolution.route, .family)
        XCTAssertEqual(resolution.verification, .configCompatible)
        XCTAssertEqual(resolution.revision, Self.commit)
        XCTAssertEqual(resolution.descriptor?.registryName, try template(Self.mistralTemplate).registryName)
        XCTAssertEqual(resolution.descriptor?.license?.identifier, "llama3.2")
    }

    func testFieldsTheReaderDoesNotReadAreRefused() throws {
        var biased = Self.llamaConfiguration
        biased["mlp_bias"] = true
        var gelu = Self.llamaConfiguration
        gelu["hidden_act"] = "gelu"
        var unstatedBase = Self.llamaConfiguration
        unstatedBase["rope_theta"] = nil
        var windowed = Self.llamaConfiguration
        windowed["sliding_window"] = 4096
        var qwenWindowed = Self.configuration
        qwenWindowed["sliding_window"] = 128
        qwenWindowed["use_sliding_window"] = true
        for configuration in [biased, gelu, unstatedBase, windowed] {
            try assertRefusal(.unsupportedConfiguration, try llamaRepository(configuration: configuration),
                              template: Self.mistralTemplate)
        }
        try assertRefusal(.unsupportedConfiguration, try repository(configuration: qwenWindowed))
    }

    func testFieldsStatedAtTheReadersValuesAreRead() throws {
        var nestedBase = Self.llamaConfiguration
        nestedBase["rope_theta"] = nil
        nestedBase["rope_parameters"] = ["rope_theta": 500_000, "rope_type": "default"]
        var unusedWindow = Self.configuration
        unusedWindow["sliding_window"] = 4096
        unusedWindow["use_sliding_window"] = false
        var contextWideWindow = Self.llamaConfiguration
        contextWideWindow["sliding_window"] = 131_072
        XCTAssertNoThrow(try NFKMLXModelDescriptors.descriptor(for: try llamaRepository(configuration: nestedBase),
                                                               matching: try template(Self.mistralTemplate)))
        XCTAssertNoThrow(try NFKMLXModelDescriptors.descriptor(for: try llamaRepository(configuration: contextWideWindow),
                                                               matching: try template(Self.mistralTemplate)))
        XCTAssertNoThrow(try NFKMLXModelDescriptors.descriptor(for: try repository(configuration: unusedWindow),
                                                               matching: try template()))
    }

    func testANullQuantizationKeyStatesNothing() {
        let nested: [String: Any] = ["quantization": NSNull(),
                                     "text_config": ["quantization": ["bits": 4, "group_size": 64]]]
        XCTAssertTrue(NFKMLXLanguage.isMLXQuantization(nested))
        XCTAssertFalse(NFKMLXLanguage.isMLXQuantization(["quantization": NSNull()]))
        XCTAssertTrue(NFKMLXLanguage.isMLXQuantization(["quantization_config": ["bits": 4, "quant_method": NSNull()]]))
    }

    func testTheMeasurementCountsWhatMeasurePyCounts() throws {
        var structure = NFKMLXTensorStructure()
        for (name, shape) in Self.tensors {
            structure.add(name: name, dtype: "BF16", shape: shape.map(Int64.init), bytes: Int64(shape.reduce(1, *) * 2))
        }
        structure.add(name: "model.layers.0.mlp.experts.3.up_proj.weight", dtype: "BF16", shape: [8, 64], bytes: 1024)
        XCTAssertEqual(structure.quantizableElements, 73_728 + 512)
        XCTAssertEqual(structure.expertBytes, ["BF16": 1024])
        let layers = try XCTUnwrap(structure.layers())
        XCTAssertEqual(layers.count, 2)
        XCTAssertEqual(layers.largestBytes, ["BF16": 37_056 * 2 + 1024])
    }

    func testTheGeometryReadsANestedDecoderConfiguration() throws {
        let geometry = try XCTUnwrap(NFKMLXMeasuredGeometry(configuration: [
            "hidden_size": 8, "text_config": ["hidden_size": 64, "num_key_value_heads": 4, "use_cache": true]]))
        XCTAssertEqual(geometry.hiddenSize, 64)
        XCTAssertEqual(geometry.keyValueHeads, 4)
        XCTAssertNil(NFKMLXMeasuredGeometry(configuration: ["use_cache": true]))
    }

    // MARK: Refusals

    func testAnotherQuantizationMethodIsRefused() throws {
        var configuration = Self.configuration
        configuration["quantization_config"] = ["quant_method": "gptq", "bits": 4]
        try assertRefusal(.unsupportedQuantization, try repository(configuration: configuration))
    }

    func testAQuantizationBlockWithoutItsGeometryIsRefused() throws {
        var configuration = Self.configuration
        configuration["quantization"] = ["mode": "affine"]
        try assertRefusal(.unsupportedQuantization, try repository(configuration: configuration))
    }

    func testPackedWeightsTheConfigurationDoesNotDescribeAreRefused() throws {
        let packed = Self.safetensors([("model.layers.0.mlp.up_proj.weight", [128, 8])], dtype: "U32", elementBytes: 4)
        try assertRefusal(.unsupportedQuantization, try repository(weights: ["model.safetensors": packed]))
    }

    func testPackedOrEightBitFloatWeightsWithoutAQuantizationAreRefused() throws {
        for dtype in ["I8", "U8", "I32", "F8_E4M3", "F8_E5M2"] {
            let weights = Self.safetensors(typed: [("model.layers.0.mlp.up_proj.weight", dtype, [128, 64])])
            try assertRefusal(.unsupportedQuantization, try repository(weights: ["model.safetensors": weights]))
        }
    }

    func testAnIntegerBufferBesideFloatWeightsIsRead() throws {
        let tensors = Self.tensors.map { ($0.0, "BF16", $0.1) } + [("model.position_ids", "I64", [1, 512])]
        XCTAssertNoThrow(try NFKMLXModelDescriptors.descriptor(
            for: try repository(weights: ["model.safetensors": Self.safetensors(typed: tensors)]), matching: try template()))
    }

    func testMXFP4OnAFamilyWithoutFusedExpertsIsRefused() throws {
        var configuration = Self.configuration
        configuration["quantization_config"] = ["quant_method": "mxfp4"]
        try assertRefusal(.unsupportedQuantization, try repository(configuration: configuration))
    }

    func testAnInferKitQuantizedSaveSplitAcrossFilesIsRefused() throws {
        let packed = Self.packedTensors
        let first = Array(packed.prefix(12)), second = Array(packed.dropFirst(12))
        var weightMap = [String: String]()
        first.forEach { weightMap[$0.0] = "model-00001-of-00002.safetensors" }
        second.forEach { weightMap[$0.0] = "model-00002-of-00002.safetensors" }
        let metadata = ["inferkit.quantization": "4:64"]
        try assertRefusal(.unsupportedQuantization, try repository(weights: [
            "model.safetensors.index.json": Self.json(["weight_map": weightMap]),
            "model-00001-of-00002.safetensors": Self.safetensors(typed: first, metadata: metadata),
            "model-00002-of-00002.safetensors": Self.safetensors(typed: second, metadata: metadata),
        ]))
    }

    func testMLXLMsGptOssLayoutIsRefused() throws {
        let entry: [String: Any] = ["dtype": "U32", "shape": [4, 8, 8], "data_offsets": [0, 1024]]
        for name in ["model.layers.0.mlp.experts.gate_proj.weight", "model.layers.0.mlp.switch_mlp.gate_proj.weight"] {
            XCTAssertThrowsError(try NFKMLXRepositoryJudgment.verifyPacking(
                headers: [[name: entry]], quantization: .mlx, fusesExperts: true, repository: "org/gpt-oss-4bit"), name) {
                XCTAssertEqual(($0 as NSError).code, NFKModelResolutionError.Code.unsupportedQuantization.rawValue)
            }
            XCTAssertFalse(try NFKMLXRepositoryJudgment.verifyPacking(
                headers: [[name: entry]], quantization: .mlx, fusesExperts: false, repository: "org/mixture-4bit"))
        }
    }

    func testAMetaspaceTokenizerOutsideMistralsFormIsRefused() throws {
        var tokenizer = Self.metaspaceTokenizer
        tokenizer["pre_tokenizer"] = ["type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": false]
        try assertRefusal(.unsupportedTokenizer, try repository(tokenizer: tokenizer))
    }

    func testAPreTokenizationPatternTheTokenizerDoesNotReproduceIsRefused() throws {
        try assertRefusal(.unsupportedTokenizer, try repository(tokenizer: Self.tokenizer(pattern: #"\s+|\S+"#)))
    }

    func testANormalizerTheTokenizerDoesNotReproduceIsRefused() throws {
        var tokenizer = Self.tokenizer()
        tokenizer["normalizer"] = ["type": "NFKC"]
        try assertRefusal(.unsupportedTokenizer, try repository(tokenizer: tokenizer))
    }

    func testAMissingTokenizerIsRefused() throws {
        try assertRefusal(.unsupportedTokenizer, try repository(tokenizer: nil))
    }

    func testAHybridConfigurationIsRefused() throws {
        var configuration = Self.configuration
        configuration["layer_types"] = ["linear_attention", "full_attention"]
        try assertRefusal(.unsupportedConfiguration, try repository(configuration: configuration))
    }

    func testAHeaderShapeWhoseProductOverflowsIsRefused() throws {
        let header: [String: Any] = ["model.layers.0.mlp.up_proj.weight": [
            "dtype": "BF16", "shape": [4_294_967_296, 4_294_967_296], "data_offsets": [0, 2]]]
        let json = try JSONSerialization.data(withJSONObject: header)
        var length = UInt64(json.count).littleEndian
        var file = Data(bytes: &length, count: 8)
        file.append(json)
        file.append(Data(count: 2))
        try assertRefusal(.unsupportedWeightFormat, try repository(weights: ["model.safetensors": file]))
    }

    func testNegativeHeaderOffsetsAreRefused() throws {
        let header: [String: Any] = ["model.norm.weight": ["dtype": "BF16", "shape": [64], "data_offsets": [128, 0]]]
        let json = try JSONSerialization.data(withJSONObject: header)
        var length = UInt64(json.count).littleEndian
        var file = Data(bytes: &length, count: 8)
        file.append(json)
        try assertRefusal(.unsupportedWeightFormat, try repository(weights: ["model.safetensors": file]))
    }

    func testWeightsOutsideTheFilesTheLoaderReadsAreRefused() throws {
        try assertRefusal(.unsupportedWeightFormat,
                          try repository(weights: ["weights/model.safetensors": Self.safetensors(Self.tensors)]))
    }

    func testAModelWhoseLoaderReadsNoOtherRepositoryRefuses() throws {
        try assertRefusal(.unsupportedArchitecture, try repository(), template: "whisper-base")
    }

    // MARK: Catalog

    func testTheLanguageReleasesListTheModelTypesTheirLoaderReads() throws {
        let catalog = NFKModelCatalog.shared
        let expected = ["qwen3-0.6b": ["qwen3"], "qwen3-30b-a3b": ["qwen3_moe"], "qwen1.5-moe-a2.7b": ["qwen2_moe"],
                        "mixtral-8x7b-instruct": ["mixtral"], "gpt-oss-20b": ["gpt_oss"],
                        "mistral-small-24b-instruct-2501": ["mistral", "llama"]]
        for (identifier, modelTypes) in expected {
            XCTAssertEqual(catalog.descriptor(withIdentifier: identifier)?.modelTypes, modelTypes, identifier)
        }
        for descriptor in NFKMLXModelDescriptors.modelDescriptors() where !descriptor.modelTypes.isEmpty {
            let entry = try XCTUnwrap(NFKMLXModelDescriptors.entries.first { $0.identifier == descriptor.identifier })
            XCTAssertTrue(NFKMLXCatalogLanguage.readsReleases(of: entry),
                          "\(descriptor.identifier) lists model types its loader does not judge")
        }
    }

    // MARK: The validation store

    /// The release directory a validation key names, placed in the hub cache at ``commit`` by symbolic links,
    /// and the repository a resolver reads.
    private func storedRelease(_ key: String, repository name: String) throws -> NFKModelRepository {
        var isDirectory: ObjCBool = false
        guard let path = NFKMLXValidationConfig.environment[key],
              FileManager.default.fileExists(atPath: path, isDirectory: &isDirectory), isDirectory.boolValue else {
            throw XCTSkip("set \(key) to the mlx-community release directory")
        }
        let directory = URL(fileURLWithPath: path)
        let hub = NFKHFHub(cacheDirectoryURL: cache)
        var listed = [NFKDownloadPlanFile]()
        // Only the release's own files: `hf download --local-dir` leaves a `.cache` directory beside them.
        for file in try FileManager.default.contentsOfDirectory(atPath: path).sorted() where file != "REVISION" {
            let source = directory.appendingPathComponent(file)
            guard try source.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true else { continue }
            let local = try XCTUnwrap(hub.localURL(forRepo: name, revision: Self.commit, path: file))
            try FileManager.default.createDirectory(at: local.deletingLastPathComponent(),
                                                    withIntermediateDirectories: true)
            try FileManager.default.createSymbolicLink(at: local, withDestinationURL: source)
            let size = try XCTUnwrap(try source.resourceValues(forKeys: [.fileSizeKey]).fileSize)
            let remote = try XCTUnwrap(hub.remoteURL(forRepo: name, revision: Self.commit, path: file))
            listed.append(NFKDownloadPlanFile(path: file, remoteURL: remote, localURL: local, byteCount: Int64(size),
                                              sha256: nil, cached: true, optional: false))
        }
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let configuration = try XCTUnwrap(try JSONSerialization.jsonObject(with: data) as? [String: Any])
        return NFKModelRepository(repository: name, revision: Self.commit, requestedRevision: nil,
                                  configuration: configuration, files: listed, license: nil, hub: hub)
    }

    /// Resolves a stored mlx-community release on `template` and holds its stored bytes to the
    /// `total_size` mlx_lm wrote in its weight index.
    private func assertStoredReleaseResolves(_ key: String, repository name: String, template identifier: String) throws {
        let read = try storedRelease(key, repository: name)
        guard read.file(atPath: "model.safetensors.index.json") != nil else {
            throw XCTSkip("\(name) in the validation store has no model.safetensors.index.json to hold its bytes to")
        }
        let descriptor = try NFKMLXModelDescriptors.descriptor(for: read, matching: try template(identifier))
        let index = try XCTUnwrap(try read.jsonObjectOfFile(atPath: "model.safetensors.index.json") as? [String: Any])
        let total = try XCTUnwrap((index["metadata"] as? [String: Any])?["total_size"] as? NSNumber)
        let stage = try XCTUnwrap(descriptor.defaultVariant?.memoryFacts?.stages.first)
        XCTAssertEqual(stage.byteCountByPrecision[NFKModelPrecisionCheckpoint]?.int64Value, total.int64Value)
        XCTAssertFalse(descriptor.loadOptions.contains { $0.key == NFKModelLoadOptionPrecision })
        XCTAssertFalse(descriptor.loadOptions.contains { $0.key == NFKModelLoadOptionQuantizationBits })
        XCTAssertEqual(descriptor.registryName, try template(identifier).registryName)
    }

    func testTheStoredMLXCommunityQwen3FourBitResolves() throws {
        try assertStoredReleaseResolves("IK_VAL_MLX_QWEN3_06B_4BIT", repository: "mlx-community/Qwen3-0.6B-4bit",
                                        template: "qwen3-0.6b")
    }

    func testTheStoredMLXCommunityLlama32FourBitResolves() throws {
        try assertStoredReleaseResolves("IK_VAL_MLX_LLAMA32_1B_4BIT", repository: "mlx-community/Llama-3.2-1B-Instruct-4bit",
                                        template: "mistral-small-24b-instruct-2501")
    }

    // MARK: Live (network-gated)

    // Set INFERKIT_LIVE_HUB=1 to measure Qwen/Qwen3-0.6B at its measured commit over the network, reading
    // only its listing, its JSON files, and its safetensors header, and compare the figures with the row
    // Tools/catalog/measure.py recorded for it.
    func testLiveMeasurementMatchesTheRecordedMeasurement() throws {
        guard ProcessInfo.processInfo.environment["INFERKIT_LIVE_HUB"] == "1" else {
            throw XCTSkip("set INFERKIT_LIVE_HUB=1 to measure a repository over the network")
        }
        let recorded = try XCTUnwrap(NFKMLXCatalogMeasurements.variant(model: "qwen3-0.6b", variant: "qwen3-0.6b"))
        let commit = try XCTUnwrap(recorded.commit)
        let repositoryName = "Qwen/Qwen3-0.6B"
        let hub = NFKHFHub(cacheDirectoryURL: cache)
        let plan = try hub.planDownload(ofRepo: repositoryName, revision: commit, paths: nil)
        let listing = NFKModelRepository(repository: repositoryName, revision: commit, requestedRevision: nil,
                                         configuration: [:], files: plan.files, license: nil, hub: hub)
        let configuration = try XCTUnwrap(try listing.jsonObjectOfFile(atPath: "config.json") as? [String: Any])
        let read = NFKModelRepository(repository: repositoryName, revision: commit, requestedRevision: nil,
                                      configuration: configuration, files: plan.files, license: nil, hub: hub)
        let template = try XCTUnwrap(NFKMLXModelDescriptors.entries.first { $0.identifier == "qwen3-0.6b" })
        let measured = try NFKMLXRepositoryJudgment(repository: read, template: template).measurement
        XCTAssertEqual(measured.downloadBytes, recorded.downloadBytes)
        XCTAssertEqual(measured.parameterCount, recorded.parameterCount)
        XCTAssertEqual(measured.tensorBytes, recorded.tensorBytes)
        XCTAssertEqual(measured.quantizableElements, recorded.quantizableElements)
        XCTAssertEqual(measured.layers[""]?.count, recorded.layers[""]?.count)
        XCTAssertEqual(measured.layers[""]?.largestBytes, recorded.layers[""]?.largestBytes)
        XCTAssertEqual(measured.geometry?.layerCount, recorded.geometry?.layerCount)
        XCTAssertEqual(measured.geometry?.keyValueHeads, recorded.geometry?.keyValueHeads)
        XCTAssertEqual(measured.geometry?.headDimension, recorded.geometry?.headDimension)
        XCTAssertEqual(measured.geometry?.maximumPositions, recorded.geometry?.maximumPositions)
        XCTAssertEqual(measured.geometry?.vocabularySize, recorded.geometry?.vocabularySize)
    }
}
