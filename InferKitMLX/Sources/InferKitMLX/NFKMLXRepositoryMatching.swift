//
//  NFKMLXRepositoryMatching.swift
//  InferKitMLX
//
//  The InferKitMLX side of the core's model resolver: whether a shipped model's loader reads a Hugging
//  Face repository, and the descriptor that builds it when it does.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import InferKit

extension NFKMLXModelDescriptors {

    /// A descriptor for a repository whose configuration names a model type `descriptor` lists, built on
    /// `descriptor`'s loader.
    ///
    /// @discussion The core's `NFKModelResolver` calls this for a shipped descriptor whose `modelTypes`
    /// holds the repository's `model_type`. The dense and mixture decoders ``NFKMLXLanguage`` loads judge
    /// a repository; another descriptor refuses with `NFKModelResolutionErrorUnsupportedArchitecture`.
    /// The judgment refuses what the loader would read wrongly with no error:
    /// - a configuration ``NFKMLXLanguage/configuration(fromJSON:)`` rejects
    ///   (`NFKModelResolutionErrorUnsupportedConfiguration`);
    /// - quantized weights other than gpt-oss's MXFP4 experts and a single-file InferKit save
    ///   (`NFKModelResolutionErrorUnsupportedQuantization`);
    /// - a tokenizer that is not byte-level BPE, or whose pre-tokenization pattern is not one the core
    ///   tokenizer implements exactly (`NFKModelResolutionErrorUnsupportedTokenizer`);
    /// - weights in neither `model.safetensors` nor a `model.safetensors.index.json`, and a header whose
    ///   shape or offsets are outside what a release holds (`NFKModelResolutionErrorUnsupportedWeightFormat`).
    ///
    /// The Mixtral, gpt-oss, and Mistral tokenizers fail the tokenizer test today, so their families'
    /// repositories are refused there.
    ///
    /// The repository is measured as `Tools/catalog/measure.py` measures a shipped release, from its file
    /// listing, its safetensors headers, and its `config.json`, and the descriptor takes the load options,
    /// memory model, and memory facts a shipped release of the family does. Its identifier is the
    /// repository name, its one variant reads the repository at `repository.revision` and builds under
    /// `descriptor`'s registry name, and its license is the model card's.
    ///
    /// - Throws: an error in `NFKModelResolutionErrorDomain` naming the refusal, or the hub's error when a
    ///   header or a file cannot be read.
    ///
    /// Introduced in InferKit 0.4.0.
    @objc(descriptorForRepository:matchingDescriptor:error:)
    public static func descriptor(for repository: NFKModelRepository,
                                  matching descriptor: NFKModelDescriptor) throws -> NFKModelDescriptor {
        guard let template = entries.first(where: { $0.identifier == descriptor.identifier }),
              !template.modelTypes.isEmpty,
              NFKMLXCatalogLanguage.readsReleases(of: template) else {
            throw NFKMLXRepositoryJudgment.refusal(.unsupportedArchitecture,
                "\(descriptor.identifier) has no loader that reads another repository")
        }
        let judgment = try NFKMLXRepositoryJudgment(repository: repository, template: template)
        return try judgment.descriptor()
    }
}

/// What the language loader would read of one repository, judged and measured.
struct NFKMLXRepositoryJudgment {
    let repository: NFKModelRepository
    let template: NFKMLXCatalogEntry
    /// The weight files the loader reads, in the order it reads them.
    let weightFiles: [String]
    let measurement: NFKMLXMeasuredVariant

    init(repository: NFKModelRepository, template: NFKMLXCatalogEntry) throws {
        self.repository = repository
        self.template = template
        let configuration = repository.configuration
        do {
            _ = try NFKMLXLanguage.configuration(fromJSON: configuration)
        } catch {
            throw Self.refusal(.unsupportedConfiguration,
                "the \(template.identifier) loader rejects the configuration of \(repository.repository): "
                + error.localizedDescription)
        }
        let mxfp4 = try Self.verifyQuantization(of: configuration, repository: repository.repository)
        try Self.verifyTokenizer(of: repository)
        weightFiles = try Self.weightFiles(of: repository)
        let headers = try weightFiles.map { try repository.safetensorsHeaderOfFile(atPath: $0) }
        try Self.verifyPacking(headers: headers, mxfp4: mxfp4, repository: repository.repository)
        let tensors = try Self.tensors(of: headers, repository: repository.repository)
        measurement = try Self.measurement(of: repository, template: template, weightFiles: weightFiles,
                                           tensors: tensors)
    }

    /// The descriptor of the repository, read through the template's loader.
    func descriptor() throws -> NFKModelDescriptor {
        var entry = template
        let name = repository.repository
        entry.identifier = name
        entry.displayName = name
        entry.summary = "\(name), read by the loader of \(template.displayName ?? template.identifier)."
        entry.registryName = template.registryName ?? template.identifier
        entry.license = repository.license ?? .license(nil, name: "Not stated on the model card",
                                                       url: repository.hub.endpointURL.appendingPathComponent(name)
                                                           .absoluteString)
        entry.modelTypes = []
        entry.minimumMemoryByteCount = nil
        entry.variants = [.variant(Self.variantIdentifier(name), displayName: name, isDefault: true,
                                   registryName: entry.registryName, source: source())]
        let measurement = self.measurement
        let measured = try entry.measured { _ in measurement }.descriptor()
        let catalog = NFKModelCatalog()
        catalog.addDescriptors([measured])
        return NFKMLXModelDescriptors.withLeastVariantMemory(measured.derivingMemoryFigures(in: catalog))
    }

    /// The template's files at the repository's commit, each with its listed size and digest, and the
    /// weight entry the loader reads.
    private func source() -> NFKModelSource {
        let templateFiles = template.variants.first?.source?.files ?? []
        let files = templateFiles.compactMap { file -> NFKModelFile? in
            let listed = repository.file(atPath: file.path)
            if file.role == NFKModelFileRoleWeights {
                return listed == nil ? nil : listedFile(file.path, role: file.role, optional: false, listed: listed)
            }
            return listedFile(file.path, role: file.role, optional: file.isOptional, listed: listed)
        }
        return NFKModelSource(kind: NFKModelSourceKindHuggingFace, repository: repository.repository,
                              revision: repository.revision, url: nil, layout: NFKModelSourceLayoutDirectory,
                              files: files, requiresConversion: false, extensions: nil)
    }

    private func listedFile(_ path: String, role: String, optional: Bool, listed: NFKDownloadPlanFile?) -> NFKModelFile {
        NFKModelFile(path: path, byteCount: listed.map { NSNumber(value: $0.byteCount) }, sha256: listed?.sha256,
                     role: role, optional: optional, extensions: nil)
    }

    /// A variant's identifier: the repository's own name, lowercased, as the catalog's releases name theirs.
    static func variantIdentifier(_ repository: String) -> String {
        (repository.split(separator: "/").last.map(String.init) ?? repository).lowercased()
    }

    static func refusal(_ code: NFKModelResolutionError.Code, _ reason: String) -> NSError {
        NSError(domain: NFKModelResolutionErrorDomain, code: code.rawValue,
                userInfo: [NSLocalizedDescriptionKey: reason])
    }
}

// MARK: - Refusals

extension NFKMLXRepositoryJudgment {

    /// Refuses a quantization the loader does not read, and returns whether the release stores MXFP4
    /// experts, which it does.
    static func verifyQuantization(of configuration: [String: Any], repository: String) throws -> Bool {
        if configuration["quantization"] != nil {
            throw refusal(.unsupportedQuantization,
                "\(repository) is quantized as its config.json's \"quantization\" states, which the loader does not read")
        }
        guard let quantization = configuration["quantization_config"] as? [String: Any] else {
            return false
        }
        let method = quantization["quant_method"] as? String ?? "unnamed"
        guard method == "mxfp4" else {
            throw refusal(.unsupportedQuantization, "\(repository) is quantized by \(method), which the loader does not read")
        }
        return true
    }

    /// Refuses packed weights the loader would take as floats: packed words a configuration does not
    /// describe, and an InferKit save split across files.
    static func verifyPacking(headers: [[String: Any]], mxfp4: Bool, repository: String) throws {
        let saved = headers.contains { (($0["__metadata__"] as? [String: Any])?["inferkit.quantization"]) != nil }
        if saved && headers.count > 1 {
            throw refusal(.unsupportedQuantization,
                "\(repository) is an InferKit quantized save split across files, which the loader reads only whole")
        }
        let packed = headers.contains { header in
            header.contains { name, entry in
                name != "__metadata__" && ((entry as? [String: Any])?["dtype"] as? String) == "U32"
            }
        }
        if packed && !saved && !mxfp4 {
            throw refusal(.unsupportedQuantization,
                "\(repository) holds packed weights its configuration does not describe")
        }
    }

    /// The pre-tokenization patterns the core byte-level tokenizer implements, as a release's
    /// `tokenizer.json` writes them (`NFKByteLevelBPETokenizer.m`): Qwen2's, Qwen3.5's, and o200k's.
    static let readablePreTokenizationPatterns: Set<String> = [
        #"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#,
        #"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#,
        #"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?"#
            + #"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?"#
            + #"|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"#,
    ]

    /// Refuses a tokenizer the loader's byte-level reader does not read as the release defines it.
    static func verifyTokenizer(of repository: NFKModelRepository) throws {
        let name = repository.repository
        for path in NFKMLXLanguage.requiredFiles where path.hasPrefix("tokenizer") && repository.file(atPath: path) == nil {
            throw refusal(.unsupportedTokenizer, "\(name) has no \(path), which the loader reads")
        }
        guard let tokenizer = try repository.jsonObjectOfFile(atPath: "tokenizer.json") as? [String: Any],
              let model = tokenizer["model"] as? [String: Any] else {
            throw refusal(.unsupportedTokenizer, "the tokenizer.json of \(name) is not a tokenizer definition")
        }
        guard model["type"] as? String == "BPE" else {
            throw refusal(.unsupportedTokenizer,
                "\(name) tokenizes with \(model["type"] as? String ?? "an unnamed model"), not byte-level BPE")
        }
        if model["byte_fallback"] as? Bool == true || contains(tokenizer["pre_tokenizer"], type: "Metaspace")
            || contains(tokenizer["decoder"], type: "Metaspace") {
            throw refusal(.unsupportedTokenizer,
                "\(name) tokenizes with a SentencePiece-style BPE (Metaspace, byte fallback), not byte-level BPE")
        }
        if model["ignore_merges"] as? Bool == true {
            throw refusal(.unsupportedTokenizer,
                "\(name) looks whole words up before merging (ignore_merges), which the tokenizer does not")
        }
        let patterns = splitPatterns(tokenizer["pre_tokenizer"])
        guard patterns.count == 1, let pattern = patterns.first, readablePreTokenizationPatterns.contains(pattern) else {
            throw refusal(.unsupportedTokenizer,
                "\(name) splits words with a pre-tokenization pattern the tokenizer does not implement")
        }
    }

    /// Whether a tokenizer component, or a step of a sequence of them, has the type.
    private static func contains(_ component: Any?, type: String) -> Bool {
        guard let component = component as? [String: Any] else { return false }
        if component["type"] as? String == type {
            return true
        }
        let steps = (component["pretokenizers"] as? [Any]) ?? (component["decoders"] as? [Any]) ?? []
        return steps.contains { contains($0, type: type) }
    }

    /// The regular expressions of a pre-tokenizer's `Split` steps.
    private static func splitPatterns(_ preTokenizer: Any?) -> [String] {
        guard let preTokenizer = preTokenizer as? [String: Any] else { return [] }
        let steps = (preTokenizer["pretokenizers"] as? [[String: Any]]) ?? [preTokenizer]
        return steps.compactMap { step in
            guard step["type"] as? String == "Split" else { return nil }
            return (step["pattern"] as? [String: Any])?["Regex"] as? String
        }
    }

    /// The safetensors files the loader reads: the shards a `model.safetensors.index.json` names, else
    /// `model.safetensors`.
    static func weightFiles(of repository: NFKModelRepository) throws -> [String] {
        let name = repository.repository
        let index = "model.safetensors.index.json"
        if repository.file(atPath: index) != nil {
            guard let weightMap = (try repository.jsonObjectOfFile(atPath: index) as? [String: Any])?["weight_map"]
                    as? [String: String], !weightMap.isEmpty else {
                throw refusal(.unsupportedWeightFormat, "the \(index) of \(name) names no shards")
            }
            let shards = Set(weightMap.values).sorted()
            if let missing = shards.first(where: { repository.file(atPath: $0) == nil }) {
                throw refusal(.unsupportedWeightFormat, "\(name) does not list \(missing), which its \(index) names")
            }
            return shards
        }
        guard repository.file(atPath: "model.safetensors") != nil else {
            throw refusal(.unsupportedWeightFormat,
                "\(name) holds its weights in neither model.safetensors nor a \(index), the files the loader reads")
        }
        return ["model.safetensors"]
    }
}

// MARK: - Measurement

extension NFKMLXRepositoryJudgment {

    /// The largest element count or byte count a measurement takes from a remote header or listing, and
    /// the largest total of them: 2^53, past any release and inside every sum the measurement forms.
    static let measurableLimit: Int64 = 1 << 53

    /// Every tensor the headers describe, with its element count and stored bytes. A header whose shape or
    /// offsets are negative, malformed, or past ``measurableLimit``, alone or in total, is refused, so no
    /// sum the measurement forms overflows.
    static func tensors(of headers: [[String: Any]], repository: String) throws -> [NFKMLXHeaderTensor] {
        func refused(_ name: String) -> NSError {
            refusal(.unsupportedWeightFormat,
                    "a safetensors header of \(repository) describes \(name) with a shape or offsets outside "
                    + "what a release holds")
        }
        var tensors = [NFKMLXHeaderTensor]()
        var totalElements: Int64 = 0
        var totalBytes: Int64 = 0
        for header in headers {
            for (name, value) in header where name != "__metadata__" {
                guard let entry = value as? [String: Any], let dtype = entry["dtype"] as? String,
                      let shape = entry["shape"] as? [NSNumber], let offsets = entry["data_offsets"] as? [NSNumber],
                      offsets.count == 2 else {
                    throw refused(name)
                }
                let dimensions = shape.map(\.int64Value)
                let start = offsets[0].int64Value, end = offsets[1].int64Value
                guard dimensions.allSatisfy({ $0 >= 0 && $0 <= measurableLimit }), start >= 0, end >= start,
                      end <= measurableLimit,
                      let elements = product(dimensions), elements <= measurableLimit else {
                    throw refused(name)
                }
                totalElements += elements
                totalBytes += end - start
                guard totalElements <= measurableLimit, totalBytes <= measurableLimit else {
                    throw refused(name)
                }
                tensors.append(NFKMLXHeaderTensor(name: name, dtype: dtype, shape: dimensions, elements: elements,
                                                  bytes: end - start))
            }
        }
        return tensors
    }

    /// The product of the dimensions, or nil where it passes ``measurableLimit`` or overflows.
    private static func product(_ dimensions: [Int64]) -> Int64? {
        var result: Int64 = 1
        for dimension in dimensions {
            let (next, overflow) = result.multipliedReportingOverflow(by: dimension)
            guard !overflow, next <= measurableLimit else { return nil }
            result = next
        }
        return result
    }

    /// The repository measured as `Tools/catalog/measure.py` measures a release: the bytes the download
    /// transfers, the tensor bytes by dtype, the parameters, the repeated layers, the routed experts, the
    /// elements a weight quantization packs, and the geometry the key-value cache follows.
    static func measurement(of repository: NFKModelRepository, template: NFKMLXCatalogEntry, weightFiles: [String],
                            tensors: [NFKMLXHeaderTensor]) throws -> NFKMLXMeasuredVariant {
        var tensorBytes = [String: Int64]()
        var elements: Int64 = 0
        var structure = NFKMLXTensorStructure()
        for tensor in tensors {
            tensorBytes[tensor.dtype, default: 0] += tensor.bytes
            elements += tensor.elements
            structure.add(name: tensor.name, dtype: tensor.dtype, shape: tensor.shape, bytes: tensor.bytes)
        }
        let variant = variantIdentifier(repository.repository)
        var measurement = NFKMLXMeasuredVariant(
            model: repository.repository, variant: variant, status: .measured, note: nil, commit: repository.revision,
            downloadBytes: try downloadBytes(of: repository, template: template, weightFiles: weightFiles),
            parameterCount: elements, tensorBytes: ["": tensorBytes], opaqueBytes: [:])
        if let layers = structure.layers() {
            measurement.layers = ["": layers]
        }
        if !structure.expertBytes.isEmpty {
            measurement.expertBytes = ["": structure.expertBytes]
        }
        if structure.quantizableElements > 0 {
            measurement.quantizableElements = ["": structure.quantizableElements]
        }
        measurement.geometry = NFKMLXMeasuredGeometry(configuration: repository.configuration)
        return measurement
    }

    /// The bytes the variant's download transfers: each template file the repository lists, the weight
    /// index, and the shards. A listed size past ``measurableLimit`` is refused.
    private static func downloadBytes(of repository: NFKModelRepository, template: NFKMLXCatalogEntry,
                                      weightFiles: [String]) throws -> Int64 {
        let templateFiles = template.variants.first?.source?.files ?? []
        let paths = Set(templateFiles.map(\.path)).union(weightFiles)
        var total: Int64 = 0
        for path in paths {
            let size = repository.file(atPath: path)?.byteCount ?? 0
            guard size >= 0, size <= measurableLimit, total + size <= measurableLimit else {
                throw refusal(.unsupportedWeightFormat,
                              "\(repository.repository) lists \(path) at a size outside what a release holds")
            }
            total += size
        }
        return total
    }
}

/// One tensor a safetensors header describes, with its counts bounded by
/// ``NFKMLXRepositoryJudgment/measurableLimit``.
struct NFKMLXHeaderTensor {
    let name: String
    let dtype: String
    let shape: [Int64]
    let elements: Int64
    let bytes: Int64
}

/// What one set of tensors holds beyond its bytes, as `Tools/catalog/measure.py`'s `Structure` reads it.
struct NFKMLXTensorStructure {
    /// The bytes of each layer of each repeated-layer container, by dtype.
    private var containers = [String: [Int: [String: Int64]]]()
    private(set) var expertBytes = [String: Int64]()
    private(set) var quantizableElements: Int64 = 0

    /// The first repeated-layer container in a tensor name: `model.layers.12.`, `transformer_blocks.3.`.
    private static let layerPattern = try! NSRegularExpression(
        pattern: #"^((?:.*?\.)?(?:layers|h|blocks|transformer_blocks))\.(\d+)\."#)
    /// The group a runtime weight quantization packs a row in.
    private static let quantizationGroup: Int64 = 64

    /// Adds one tensor. The caller bounds the shape's product and every total, as
    /// ``NFKMLXRepositoryJudgment/tensors(of:repository:)`` does, so the sums here cannot overflow.
    mutating func add(name: String, dtype: String, shape: [Int64], bytes: Int64) {
        let range = NSRange(name.startIndex..., in: name)
        if let match = Self.layerPattern.firstMatch(in: name, range: range),
           let container = Range(match.range(at: 1), in: name), let index = Range(match.range(at: 2), in: name),
           let layer = Int(name[index]) {
            containers[String(name[container]), default: [:]][layer, default: [:]][dtype, default: 0] += bytes
        }
        if name.contains(".experts.") {
            expertBytes[dtype, default: 0] += bytes
        }
        let lowered = name.lowercased()
        if shape.count == 2, name.hasSuffix(".weight"), shape[1] % Self.quantizationGroup == 0,
           !lowered.contains("embed"), !lowered.contains("norm") {
            quantizableElements += shape[0] * shape[1]
        }
    }

    /// The count and the largest layer's bytes by dtype of the container holding the most bytes.
    func layers() -> NFKMLXMeasuredLayers? {
        func total(_ layers: [Int: [String: Int64]]) -> Int64 {
            layers.values.reduce(0) { $0 + $1.values.reduce(0, +) }
        }
        guard let (_, layers) = containers.max(by: { (total($0.value), $0.key) < (total($1.value), $1.key) }),
              let largest = layers.values.max(by: { lhs, rhs in
                  (lhs.values.reduce(0, +), lhs.sorted { $0.key < $1.key }.description)
                      < (rhs.values.reduce(0, +), rhs.sorted { $0.key < $1.key }.description)
              }) else {
            return nil
        }
        return NFKMLXMeasuredLayers(count: layers.count, largestBytes: largest)
    }
}

extension NFKMLXMeasuredGeometry {

    /// The geometry fields a configuration states, read from its root and from a nested decoder
    /// configuration, which wins, as `Tools/catalog/measure.py`'s `geometry_of` reads them. nil when it
    /// states none.
    init?(configuration: [String: Any]) {
        let scopes = [configuration] + ["text_config", "llm_config", "language_config"].compactMap {
            configuration[$0] as? [String: Any]
        }
        var geometry = NFKMLXMeasuredGeometry()
        var stated = false
        func integer(_ scope: [String: Any], _ key: String) -> Int? {
            guard let number = scope[key] as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID() else { return nil }
            stated = true
            return number.intValue
        }
        for scope in scopes {
            geometry.layerCount = integer(scope, "num_hidden_layers") ?? geometry.layerCount
            geometry.attentionHeads = integer(scope, "num_attention_heads") ?? geometry.attentionHeads
            geometry.keyValueHeads = integer(scope, "num_key_value_heads") ?? geometry.keyValueHeads
            geometry.headDimension = integer(scope, "head_dim") ?? geometry.headDimension
            geometry.hiddenSize = integer(scope, "hidden_size") ?? geometry.hiddenSize
            geometry.maximumPositions = integer(scope, "max_position_embeddings") ?? geometry.maximumPositions
            geometry.slidingWindow = integer(scope, "sliding_window") ?? geometry.slidingWindow
            geometry.slidingWindowPattern = integer(scope, "sliding_window_pattern") ?? geometry.slidingWindowPattern
            geometry.sharedKeyValueLayers = integer(scope, "num_kv_shared_layers") ?? geometry.sharedKeyValueLayers
            geometry.globalHeadDimension = integer(scope, "global_head_dim") ?? geometry.globalHeadDimension
            geometry.globalKeyValueHeads = integer(scope, "num_global_key_value_heads") ?? geometry.globalKeyValueHeads
            geometry.vocabularySize = integer(scope, "vocab_size") ?? geometry.vocabularySize
            if let types = (scope["layer_types"] ?? scope["layers_block_type"]) as? [String] {
                geometry.layerTypes = types
                stated = true
            }
        }
        guard stated else { return nil }
        self = geometry
    }
}
