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
    /// holds the repository's `model_type`. Each family whose loader reads another release has a judge
    /// (`NFKMLXFamilyJudge`): the dense and mixture decoders ``NFKMLXLanguage`` loads, the speech encoders,
    /// TimesFM 2.5, Gemma 3, 3n, and 4, Qwen3-VL, SmolVLM2, TrOCR, the translators, and the hybrid
    /// decoders. A descriptor whose family has none, or a repository whose `model_type` the descriptor does
    /// not list, refuses with `NFKModelResolutionErrorUnsupportedArchitecture`.
    ///
    /// A judge refuses what its loader would read wrongly with no error, through the family's own readers:
    /// - a configuration the family's reader rejects, or one stating a field it does not read at a value the
    ///   output depends on (`NFKModelResolutionErrorUnsupportedConfiguration`);
    /// - a head or a layer kind the network does not build (`NFKModelResolutionErrorUnsupportedArchitecture`);
    /// - a quantization the loader does not read, or packed weights no configuration describes
    ///   (`NFKModelResolutionErrorUnsupportedQuantization`);
    /// - a tokenizer the family's reader does not reproduce (`NFKModelResolutionErrorUnsupportedTokenizer`);
    /// - weights outside the files the loader reads, and a header whose shape or offsets are outside what a
    ///   release holds (`NFKModelResolutionErrorUnsupportedWeightFormat`).
    ///
    /// The repository is measured as `Tools/catalog/measure.py` measures a shipped release, from its file
    /// listing, its safetensors headers, and its `config.json`, and the descriptor takes the load options,
    /// memory model, and memory facts a shipped release of the family does. A repository the loader reads
    /// as stored (an mlx_lm quantized release, an InferKit quantized save) is measured at its stored
    /// bytes and offers no precision or weight quantization option. Its identifier is the repository
    /// name, its one variant downloads the weight files the judgment measured at `repository.revision`
    /// and builds under the registry name of the family's entry that describes the repository, and its
    /// license is the model card's.
    ///
    /// - Throws: an error in `NFKModelResolutionErrorDomain` naming the refusal, or the hub's error when a
    ///   header or a file cannot be read.
    ///
    /// Introduced in InferKit 0.4.0.
    @objc(descriptorForRepository:matchingDescriptor:error:)
    public static func descriptor(for repository: NFKModelRepository,
                                  matching descriptor: NFKModelDescriptor) throws -> NFKModelDescriptor {
        guard let template = entries.first(where: { $0.identifier == descriptor.identifier }),
              !template.modelTypes.isEmpty, let judge = NFKMLXFamilyJudge.judge(for: template) else {
            throw NFKMLXRepositoryJudgment.refusal(.unsupportedArchitecture,
                "\(descriptor.identifier) has no loader that reads another repository")
        }
        guard let modelType = repository.modelType, template.modelTypes.contains(modelType) else {
            throw NFKMLXRepositoryJudgment.refusal(.unsupportedArchitecture,
                "\(descriptor.identifier) reads " + template.modelTypes.joined(separator: ", ")
                + ", not the \(repository.modelType ?? "unstated") model type of \(repository.repository)")
        }
        let judgment = try NFKMLXRepositoryJudgment(repository: repository, template: judge.template(repository, template),
                                                    judge: judge)
        return try judgment.descriptor()
    }
}

/// What one family's loader requires of a repository beyond what every judgment checks: its configuration
/// and tokenizer read as the loader reads them, the weight files it reads, and the tensors it places.
struct NFKMLXFamilyJudge {
    /// What a family's configuration states about the weights, which the shared packing check reads.
    struct Reading {
        var quantization = NFKMLXRepositoryJudgment.StatedQuantization.none
        /// True where the module fuses its experts' gate and up projections (gpt-oss).
        var fusesExperts = false
    }

    /// Whether the judge answers for the entry.
    var reads: (NFKMLXCatalogEntry) -> Bool
    /// The entry of the family whose tasks and files describe the repository: the asked entry, or a sibling
    /// with the head the repository carries.
    var template: (NFKModelRepository, NFKMLXCatalogEntry) -> NFKMLXCatalogEntry = { $1 }
    /// Refuses a configuration or tokenizer the family's readers would read wrongly, and returns what the
    /// configuration states about the weights.
    var verify: (NFKModelRepository, NFKMLXCatalogEntry) throws -> Reading
    /// The weight files the loader reads, in the order it reads them.
    var weightFiles: (NFKModelRepository) throws -> [String] = NFKMLXRepositoryJudgment.weightFiles(of:)
    /// Files beyond the template's that the loader reads where a repository has them, by path, with their
    /// roles: a fine-tuned head's vocabulary on an encoder template.
    var optionalFiles: [String: String] = [:]
    /// Refuses tensors the loader would place wrongly or not at all.
    var verifyTensors: ([NFKMLXHeaderTensor], NFKModelRepository) throws -> Void = { _, _ in }
    /// The family a repository of each listed `model_type` names, where it is another family the loader
    /// also reads. Any other `model_type` names the template's family.
    var families: [String: String] = [:]

    /// The judges, in the order they are asked.
    static let judges: [NFKMLXFamilyJudge] = [.language, .wav2Vec2, .w2vBert, .timesFM, .gemma3, .gemma3n, .gemma4,
                                               .qwen3VL, .smolVLM, .marian, .m2m100, .nllb, .trocr, .graniteH,
                                               .nemotronH]

    /// The judge for an entry, or nil where no loader of the entry reads another repository.
    static func judge(for entry: NFKMLXCatalogEntry) -> NFKMLXFamilyJudge? {
        judges.first { $0.reads(entry) }
    }

    /// The dense and mixture decoders ``NFKMLXLanguage`` loads.
    static let language = NFKMLXFamilyJudge(
        reads: NFKMLXCatalogLanguage.readsReleases(of:),
        verify: { repository, template in
            let configuration = repository.configuration
            let name = repository.repository
            let read = try NFKMLXRepositoryJudgment.reading(repository, template: template) {
                try NFKMLXLanguage.configuration(fromJSON: configuration)
            }
            try NFKMLXRepositoryJudgment.verifyUnreadFields(of: configuration, repository: name)
            let quantization = try NFKMLXRepositoryJudgment.verifyQuantization(of: configuration, repository: name)
            try NFKMLXRepositoryJudgment.verifyTokenizer(of: repository)
            return Reading(quantization: quantization, fusesExperts: read.clampedSwiGLU != nil)
        },
        families: ["llama": "llama"])
}

/// What a family's loader would read of one repository, judged and measured.
struct NFKMLXRepositoryJudgment {
    let repository: NFKModelRepository
    let template: NFKMLXCatalogEntry
    /// The family the resolved descriptor names.
    let family: String
    /// The weight files the loader reads, in the order it reads them.
    let weightFiles: [String]
    /// The weight files the variant downloads: the shard index where the repository lists one, then
    /// ``weightFiles``.
    var downloadedWeightFiles: [String] { Self.downloadedWeightFiles(of: repository, weightFiles: weightFiles) }
    /// The files beyond the template's the repository lists and the loader reads, by path, with their roles.
    let optionalFiles: [String: String]
    let measurement: NFKMLXMeasuredVariant
    /// True where the loader reads the weights as stored whatever the request: an mlx_lm quantized
    /// release, or an InferKit quantized save.
    let loadsAsStored: Bool

    init(repository: NFKModelRepository, template: NFKMLXCatalogEntry,
         judge: NFKMLXFamilyJudge = .language) throws {
        self.repository = repository
        self.template = template
        family = repository.modelType.flatMap { judge.families[$0] } ?? template.family
        let configuration = repository.configuration
        let reading = try judge.verify(repository, template)
        weightFiles = try judge.weightFiles(repository)
        let templatePaths = Set((template.variants.first?.source?.files ?? []).map(\.path))
        optionalFiles = judge.optionalFiles.filter { !templatePaths.contains($0.key) && repository.file(atPath: $0.key) != nil }
        let headers = try weightFiles.map { try repository.safetensorsHeaderOfFile(atPath: $0) }
        let saved = try Self.verifyPacking(headers: headers, quantization: reading.quantization,
                                           fusesExperts: reading.fusesExperts, repository: repository.repository)
        loadsAsStored = reading.quantization == .mlx || saved
        let tensors = try Self.tensors(of: headers, repository: repository.repository)
        try judge.verifyTensors(tensors, repository)
        let packedBits = loadsAsStored ? Self.packedBits(of: configuration, headers: headers) : nil
        measurement = try Self.measurement(of: repository, template: template,
                                           weightFiles: Self.downloadedWeightFiles(of: repository, weightFiles: weightFiles)
                                               + optionalFiles.keys.sorted(),
                                           tensors: tensors, packedBits: packedBits)
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
        // The template's profile and plan are fixed before the family changes, since both are keyed by it.
        var load = NFKMLXLoadProfile.profile(for: template) ?? .plain
        if loadsAsStored {
            load.precisions = []
            load.defaultPrecision = NFKModelPrecisionCheckpoint
            load.quantizes = false
        }
        entry.load = load
        entry.family = family
        let measurement = self.measurement
        let plan = loadsAsStored ? .stored : NFKMLXLoadPlan.plan(for: template)
        let measured = try entry.measured(by: { _ in measurement }, plan: plan).descriptor()
        let catalog = NFKModelCatalog()
        catalog.addDescriptors([measured])
        return NFKMLXModelDescriptors.withLeastVariantMemory(measured.derivingMemoryFigures(in: catalog))
    }

    /// The template's other files at the repository's commit, each with its listed size and digest, the
    /// weight files the judgment measured, and the files beyond the template's the loader reads.
    private func source() -> NFKModelSource {
        let templateFiles = (template.variants.first?.source?.files ?? []).filter { $0.role != NFKModelFileRoleWeights }
        let files = templateFiles.map { file in
            listedFile(file.path, role: file.role, optional: file.isOptional, listed: repository.file(atPath: file.path))
        } + downloadedWeightFiles.map { path in
            listedFile(path, role: NFKModelFileRoleWeights, optional: false, listed: repository.file(atPath: path))
        } + optionalFiles.sorted { $0.key < $1.key }.map { path, role in
            listedFile(path, role: role, optional: true, listed: repository.file(atPath: path))
        }
        return NFKModelSource(kind: NFKModelSourceKindHuggingFace, repository: repository.repository,
                              revision: repository.revision, url: nil, layout: NFKModelSourceLayoutDirectory,
                              files: files, requiresConversion: false, extensions: nil)
    }

    private func listedFile(_ path: String, role: String, optional: Bool, listed: NFKDownloadPlanFile?) -> NFKModelFile {
        NFKModelFile(path: path, byteCount: listed.map { NSNumber(value: $0.byteCount) }, sha256: listed?.sha256,
                     role: role, optional: optional, extensions: nil)
    }

    /// The shard index where the repository lists one, then `weightFiles`.
    static func downloadedWeightFiles(of repository: NFKModelRepository, weightFiles: [String]) -> [String] {
        let index = "model.safetensors.index.json"
        return repository.file(atPath: index) != nil && !weightFiles.contains(index) ? [index] + weightFiles : weightFiles
    }

    /// A variant's identifier: the repository's own name, lowercased, as the catalog's releases name theirs.
    static func variantIdentifier(_ repository: String) -> String {
        (repository.split(separator: "/").last.map(String.init) ?? repository).lowercased()
    }

    /// The family reader's result for a repository's configuration, its error refused as
    /// `NFKModelResolutionErrorUnsupportedConfiguration`.
    static func reading<Result>(_ repository: NFKModelRepository, template: NFKMLXCatalogEntry,
                                _ read: () throws -> Result) throws -> Result {
        do {
            return try read()
        } catch {
            throw refusal(.unsupportedConfiguration,
                "the \(template.identifier) loader rejects the configuration of \(repository.repository): "
                + error.localizedDescription)
        }
    }

    /// The directory of the repository's snapshot in the hub cache, holding each of `paths` the repository
    /// lists, fetched where the cache lacks it, so a family's own readers read the files as its load does.
    static func snapshot(of repository: NFKModelRepository, holding paths: [String]) throws -> URL {
        var directory: URL?
        for path in paths {
            guard let file = repository.file(atPath: path) else { continue }
            let local = try repository.hub.downloadRepo(repository.repository, revision: repository.revision,
                                                        path: path, sha256: file.sha256)
            directory = directory ?? local.deletingLastPathComponent()
        }
        guard let directory else {
            throw refusal(.noConfiguration, "\(repository.repository) lists none of \(paths.joined(separator: ", "))")
        }
        return directory
    }

    /// The one weight file a loader that reads no shard index takes, refusing a repository without it.
    static func singleWeightFile(_ path: String) -> (NFKModelRepository) throws -> [String] {
        { repository in
            guard repository.file(atPath: path) != nil else {
                throw refusal(.unsupportedWeightFormat,
                    "\(repository.repository) holds its weights outside \(path), the one file the loader reads whole")
            }
            return [path]
        }
    }

    /// Refuses a configuration whose `architectures` names a head the family's loader does not build.
    static func verifyArchitecture(of configuration: [String: Any], repository: String, suffixes: [String]) throws {
        let architectures = configuration["architectures"] as? [String] ?? []
        if let unread = architectures.first(where: { name in !suffixes.contains { name.hasSuffix($0) } }) {
            throw refusal(.unsupportedArchitecture,
                "\(repository) is a \(unread), a head the loader does not build (it reads "
                + suffixes.joined(separator: ", ") + ")")
        }
    }

    /// Refuses a configuration stating a field at a value other than the one the family's network builds,
    /// where the reader does not read the field. A field the configuration does not state passes.
    static func verifyFixedFields(of configuration: [String: Any], repository: String,
                                  _ fields: [String: Any]) throws {
        for (key, built) in fields.sorted(by: { $0.key < $1.key }) {
            guard let stated = configuration[key], !(stated is NSNull) else { continue }
            if !(stated as AnyObject).isEqual(built) {
                throw refusal(.unsupportedConfiguration,
                    "\(repository) states \(key) \(stated), where the loader builds \(built)")
            }
        }
    }

    /// Refuses a configuration that states a quantization, for a family whose loader reads floats only.
    static func verifyFloatWeights(of configuration: [String: Any], repository: String) throws {
        for key in ["quantization", "quantization_config"] {
            if let stated = configuration[key], !(stated is NSNull) {
                throw refusal(.unsupportedQuantization,
                    "\(repository) is quantized (\(key)), and this family's loader reads float weights only")
            }
        }
    }

    static func refusal(_ code: NFKModelResolutionError.Code, _ reason: String) -> NSError {
        NSError(domain: NFKModelResolutionErrorDomain, code: code.rawValue,
                userInfo: [NSLocalizedDescriptionKey: reason])
    }
}

// MARK: - Refusals

extension NFKMLXRepositoryJudgment {

    /// How a repository's `config.json` states its weights are quantized.
    enum StatedQuantization: Equatable {
        /// No quantization is stated.
        case none
        /// mlx_lm's `quantization` block, or a `quantization_config` with no `quant_method`.
        case mlx
        /// A `quant_method` the loader reads: gpt-oss's MXFP4 experts.
        case method
    }

    /// The dtypes a release stores unpacked tensors in: the floats, and the integer buffers (position ids,
    /// masks) a release carries beside its weights.
    static let unpackedDtypes: Set<String> = ["F64", "F32", "F16", "BF16", "I64", "BOOL"]

    /// The quantization a configuration states, refusing one the loader does not read.
    static func verifyQuantization(of configuration: [String: Any], repository: String) throws -> StatedQuantization {
        do {
            guard try NFKMLXQuantization.releaseQuantization(configJSON: configuration) != nil else { return .none }
        } catch {
            throw refusal(.unsupportedQuantization, "\(repository): " + error.localizedDescription)
        }
        return NFKMLXLanguage.isMLXQuantization(configuration) ? .mlx : .method
    }

    /// Refuses a configuration that states, with a value the output depends on, a field
    /// ``NFKMLXLanguage/configuration(fromJSON:)`` does not read: an MLP bias (`mlp_bias`), an activation
    /// other than SiLU (`hidden_act`), an unstated rotary base, which the reader takes as 10^6 where
    /// transformers takes 10^4, and a sliding window narrower than the context outside gpt-oss, whose
    /// alternating windows the reader builds.
    static func verifyUnreadFields(of configuration: [String: Any], repository: String) throws {
        if (configuration["mlp_bias"] as? NSNumber)?.boolValue == true {
            throw refusal(.unsupportedConfiguration,
                "\(repository) adds biases to its MLP projections (mlp_bias), which the loader does not build")
        }
        if let activation = configuration["hidden_act"], !(activation is NSNull), activation as? String != "silu" {
            throw refusal(.unsupportedConfiguration,
                "\(repository) activates its MLP with \(activation), where the loader applies SiLU")
        }
        let parameters = configuration["rope_parameters"] as? [String: Any]
        if !(configuration["rope_theta"] is NSNumber) && !(parameters?["rope_theta"] is NSNumber) {
            throw refusal(.unsupportedConfiguration,
                "\(repository) states no rope_theta, whose default the loader and transformers set differently")
        }
        guard configuration["model_type"] as? String != "gpt_oss",
              let window = (configuration["sliding_window"] as? NSNumber)?.intValue else { return }
        let windowed = (configuration["use_sliding_window"] as? NSNumber)?.boolValue ?? true
        let positions = (configuration["max_position_embeddings"] as? NSNumber)?.intValue ?? Int.max
        if windowed && window < positions {
            throw refusal(.unsupportedConfiguration,
                "\(repository) attends within a sliding window of \(window) tokens, which the loader does not apply")
        }
    }

    /// Refuses packed weights the loader would take as floats or would not place, and returns whether
    /// the weights are an InferKit quantized save. Refused: MXFP4 on a family whose experts the loader
    /// does not fuse, packed or 8-bit float tensors no configuration or save describes, an InferKit save
    /// split across files, and mlx_lm's gpt-oss layout.
    static func verifyPacking(headers: [[String: Any]], quantization: StatedQuantization, fusesExperts: Bool,
                              repository: String) throws -> Bool {
        if quantization == .method && !fusesExperts {
            throw refusal(.unsupportedQuantization,
                "\(repository) is quantized with mxfp4, which the loader reads only for gpt-oss's experts")
        }
        let saved = headers.contains { (($0["__metadata__"] as? [String: Any])?["inferkit.quantization"]) != nil }
        if saved && headers.count > 1 {
            throw refusal(.unsupportedQuantization,
                "\(repository) is an InferKit quantized save split across files, which the loader reads only whole")
        }
        let names = headers.flatMap { $0.keys.filter { $0 != "__metadata__" } }
        func dtype(_ entry: Any) -> String { ((entry as? [String: Any])?["dtype"] as? String) ?? "an unnamed dtype" }
        let packed = headers.lazy.flatMap { $0 }.first { $0.key != "__metadata__" && !unpackedDtypes.contains(dtype($0.value)) }
        if let packed, !saved, quantization == .none {
            throw refusal(.unsupportedQuantization,
                "\(repository) stores \(packed.key) as \(dtype(packed.value)), a packing its configuration does not "
                + "describe")
        }
        if quantization == .mlx && fusesExperts && names.contains(where: separatesFusedExperts) {
            throw refusal(.unsupportedQuantization,
                "\(repository) stores gpt-oss's expert gate and up projections apart (mlx_lm's layout), where the "
                + "loader reads them fused")
        }
        return saved
    }

    /// Whether a release tensor holds an expert gate projection apart from its up projection, under the
    /// module name the loader files it as.
    private static func separatesFusedExperts(_ name: String) -> Bool {
        let key = NFKMLXLanguage.moduleKey(forRelease: name)
        return key.contains(".mlp.experts.") && key.contains(".gate_proj.")
    }

    /// Refuses a tokenizer the loader's reader does not reproduce
    /// (`NFKMLXLanguage.tokenizerScheme(tokenizerJSON:tokenizerConfiguration:)`).
    static func verifyTokenizer(of repository: NFKModelRepository) throws {
        let name = repository.repository
        for path in NFKMLXLanguage.requiredFiles where path.hasPrefix("tokenizer") && repository.file(atPath: path) == nil {
            throw refusal(.unsupportedTokenizer, "\(name) has no \(path), which the loader reads")
        }
        guard let tokenizer = try repository.jsonObjectOfFile(atPath: "tokenizer.json") as? [String: Any] else {
            throw refusal(.unsupportedTokenizer, "the tokenizer.json of \(name) is not a tokenizer definition")
        }
        let tokenizerConfiguration = try repository.jsonObjectOfFile(atPath: "tokenizer_config.json") as? [String: Any]
        do {
            _ = try NFKMLXLanguage.tokenizerScheme(tokenizerJSON: tokenizer, tokenizerConfiguration: tokenizerConfiguration)
        } catch {
            throw refusal(.unsupportedTokenizer, "the tokenizer of \(name) is not one the loader reproduces: "
                          + error.localizedDescription)
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

    /// The width of each packed layer of a release the loader reads as stored, by the layer's path:
    /// mlx_lm's per-layer override in `config.json`, else its default, else an InferKit save's recorded
    /// width.
    static func packedBits(of configuration: [String: Any], headers: [[String: Any]]) -> (String) -> Int? {
        let nested = configuration["text_config"] as? [String: Any]
        let stated = (configuration["quantization"] as? [String: Any])
            ?? (configuration["quantization_config"] as? [String: Any])
            ?? (nested?["quantization"] as? [String: Any]) ?? (nested?["quantization_config"] as? [String: Any])
        let saved = headers.lazy.compactMap { ($0["__metadata__"] as? [String: Any])?["inferkit.quantization"] as? String }
            .first.flatMap { $0.split(separator: ":").first.flatMap { Int($0) } }
        let defaultBits = (stated?["bits"] as? NSNumber)?.intValue ?? saved
        return { layer in ((stated?[layer] as? [String: Any])?["bits"] as? NSNumber)?.intValue ?? defaultBits }
    }

    /// The parameters the tensors hold. A packed layer's `U32` words count as the elements they pack at
    /// its width, and its `scales` and `biases` describe the packing, so they count as none.
    private static func parameterCount(of tensors: [NFKMLXHeaderTensor], packedBits: ((String) -> Int?)?) -> Int64 {
        guard let packedBits else {
            return tensors.reduce(0) { $0 + $1.elements }
        }
        let packedLayers = Set(tensors.compactMap { tensor -> String? in
            guard tensor.dtype == "U32", tensor.name.hasSuffix(".weight") else { return nil }
            return String(tensor.name.dropLast(".weight".count))
        })
        return tensors.reduce(0) { total, tensor in
            let layer = tensor.name.split(separator: ".").dropLast().joined(separator: ".")
            guard packedLayers.contains(layer) else { return total + tensor.elements }
            guard tensor.name.hasSuffix(".weight") else {
                let describesPacking = tensor.name.hasSuffix(".scales") || tensor.name.hasSuffix(".biases")
                return describesPacking ? total : total + tensor.elements
            }
            guard let bits = packedBits(layer), bits > 0 else { return total + tensor.elements }
            return total + tensor.elements * 32 / Int64(bits)
        }
    }

    /// The repository measured as `Tools/catalog/measure.py` measures a release: the bytes the download
    /// transfers, the tensor bytes by dtype, the parameters, the repeated layers, the routed experts, the
    /// elements a weight quantization packs, and the geometry the key-value cache follows. A release the
    /// loader reads as stored (`packedBits` non-nil) counts its packed layers' parameters unpacked and
    /// records no quantizable elements, since its layers are packed already.
    static func measurement(of repository: NFKModelRepository, template: NFKMLXCatalogEntry, weightFiles: [String],
                            tensors: [NFKMLXHeaderTensor],
                            packedBits: ((String) -> Int?)? = nil) throws -> NFKMLXMeasuredVariant {
        var tensorBytes = [String: Int64]()
        var structure = NFKMLXTensorStructure()
        for tensor in tensors {
            tensorBytes[tensor.dtype, default: 0] += tensor.bytes
            structure.add(name: tensor.name, dtype: tensor.dtype, shape: tensor.shape, bytes: tensor.bytes)
        }
        let elements = parameterCount(of: tensors, packedBits: packedBits)
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
        if structure.quantizableElements > 0 && packedBits == nil {
            measurement.quantizableElements = ["": structure.quantizableElements]
        }
        measurement.geometry = NFKMLXMeasuredGeometry(configuration: repository.configuration)
        return measurement
    }

    /// The bytes the variant's download transfers: each of the template's files other than its weights
    /// the repository lists, and `weightFiles`, the weight index, the shards, and the other files the
    /// loader reads. A listed size past ``measurableLimit`` is refused.
    private static func downloadBytes(of repository: NFKModelRepository, template: NFKMLXCatalogEntry,
                                      weightFiles: [String]) throws -> Int64 {
        let templateFiles = (template.variants.first?.source?.files ?? []).filter { $0.role != NFKModelFileRoleWeights }
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
