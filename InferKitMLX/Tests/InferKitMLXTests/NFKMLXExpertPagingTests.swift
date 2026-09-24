//
//  NFKMLXExpertPagingTests.swift
//  InferKitMLXTests
//
//  Paging a mixture's routed experts: the plan that chooses it beside staging, the store's cache,
//  and, for every release layout the language loader reads, a paged load reproducing the resident
//  one's logits exactly.
//

import XCTest
import InferKit
import MLX
import MLXNN
@testable import InferKitMLX

final class NFKMLXExpertPagingTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    private let gib = 1 << 30
    private let tokens = MLXArray([3, 17, 42, 8, 91, 7, 64, 5].map { Int32($0) }).reshaped([1, 8])

    private func scratchDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("expert-paging-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }

    // MARK: The plan

    func testAutomaticPagesOnlyAgainstAKnownShortfall() throws {
        let budget = Int(21.25 * Double(gib))
        let fits = NFKMLXStageFootprint(bytes: 12 * gib, pageableBytes: 10 * gib)
        XCTAssertEqual(try NFKMLXResidencyBudget.plan([fits], residency: .automatic, budget: budget), .resident)

        let large = NFKMLXStageFootprint(bytes: 60 * gib, pageableBytes: 55 * gib)
        let paged = try NFKMLXResidencyBudget.plan([large], residency: .automatic, budget: budget)
        XCTAssertTrue(paged.pagesExperts)
        XCTAssertFalse(paged.holdsStagesResident)
        XCTAssertEqual(paged.expertCacheBytes, (budget - NFKMLXResidencyBudget.reserve - 5 * gib) / 2,
                       "half of what the unpaged weights and the reserve leave")

        let dense = NFKMLXStageFootprint(bytes: 60 * gib)
        XCTAssertEqual(try NFKMLXResidencyBudget.plan([dense], residency: .automatic, budget: budget), .staged,
                       "nothing to page is staged, as it always was")
        XCTAssertEqual(try NFKMLXResidencyBudget.plan([large], residency: .automatic, budget: 0), .staged,
                       "a machine that reports no budget is never paged")
    }

    // Two stages that fit one at a time stage; a stage that fits only with its experts paged pages.
    func testStagingIsTriedBeforePaging() throws {
        let budget = Int(21.25 * Double(gib))
        let encoder = NFKMLXStageFootprint(bytes: 10 * gib)
        let mixture = NFKMLXStageFootprint(bytes: 12 * gib, pageableBytes: 9 * gib)
        XCTAssertEqual(try NFKMLXResidencyBudget.plan([encoder, mixture], residency: .automatic, budget: budget),
                       .staged)
        let large = NFKMLXStageFootprint(bytes: 40 * gib, pageableBytes: 36 * gib)
        let plan = try NFKMLXResidencyBudget.plan([encoder, large], residency: .automatic, budget: budget)
        XCTAssertTrue(plan.pagesExperts)
        XCTAssertFalse(plan.holdsStagesResident)
        XCTAssertEqual(plan.expertCacheBytes, (budget - NFKMLXResidencyBudget.reserve - 10 * gib) / 2,
                       "the largest unpaged stage, the encoder, bounds the cache")
    }

    func testEachExplicitResidencyDecidesAsItSays() throws {
        let budget = Int(21.25 * Double(gib))
        let small = NFKMLXStageFootprint(bytes: 2 * gib, pageableBytes: 1 * gib)
        let paged = try NFKMLXResidencyBudget.plan([small], residency: .paged, budget: budget)
        XCTAssertTrue(paged.pagesExperts, "paged pages even where everything would fit")
        XCTAssertEqual(paged.expertCacheBytes, 1 * gib, "and caches no more than the experts weigh")
        XCTAssertEqual(try NFKMLXResidencyBudget.plan([NFKMLXStageFootprint(bytes: 2 * gib)], residency: .paged,
                                                      budget: budget), .staged,
                       "a model without experts is held as staged")
        XCTAssertEqual(try NFKMLXResidencyBudget.plan([small], residency: .staged, budget: budget), .staged)

        let large = NFKMLXStageFootprint(bytes: 60 * gib, pageableBytes: 55 * gib)
        XCTAssertThrowsError(try NFKMLXResidencyBudget.plan([large], residency: .resident, budget: budget))
        let unpageable = NFKMLXStageFootprint(bytes: 60 * gib, pageableBytes: 10 * gib)
        XCTAssertThrowsError(try NFKMLXResidencyBudget.plan([unpageable], residency: .paged, budget: budget),
                             "what is left after paging is known not to fit")
        XCTAssertThrowsError(try NFKMLXResidencyBudget.plan([unpageable], residency: .automatic, budget: budget))
        XCTAssertEqual(try NFKMLXResidencyBudget.plan([large], residency: .paged, budget: 0).expertCacheBytes,
                       NFKMLXResidencyBudget.defaultExpertCacheBytes)
    }

    // MARK: The store

    private struct Counted: NFKMLXExpertSource {
        let value: Float
        let reads: () -> Void
        var heldBytes: Int { 4 * 4 }
        var mappedBytes: Int { 0 }
        func materialize() -> MLXArray {
            reads()
            return MLXArray([value, value, value, value]).reshaped([2, 2])
        }
    }

    func testTheCacheHoldsWhatItsBudgetAllowsAndEvictsTheLeastRecent() throws {
        try requireMLXRuntime()
        var reads = 0
        let store = NFKMLXExpertStore(cacheByteBudget: 0)
        for expert in 0 ..< 3 {
            store.store(Counted(value: Float(expert), reads: { reads += 1 }), group: "g", expert: expert,
                        part: "weight")
        }
        XCTAssertEqual(store.expertCount, 3)
        XCTAssertEqual(store.heldBytes, 48)
        _ = store.expert(group: "g", index: 0)
        _ = store.expert(group: "g", index: 0)
        XCTAssertEqual(reads, 2, "no budget materializes on every request")
        XCTAssertEqual(store.cachedBytes, 0)

        store.cacheByteBudget = 32
        _ = store.expert(group: "g", index: 0)
        _ = store.expert(group: "g", index: 1)
        _ = store.expert(group: "g", index: 0)
        XCTAssertEqual(reads, 4)
        XCTAssertEqual(store.cacheHitCount, 1)
        _ = store.expert(group: "g", index: 2)
        XCTAssertEqual(store.cachedBytes, 32, "the budget holds two experts")
        _ = store.expert(group: "g", index: 0)
        XCTAssertEqual(store.cacheHitCount, 2, "expert 0 was used more recently than expert 1")
        _ = store.expert(group: "g", index: 1)
        XCTAssertEqual(reads, 6, "expert 1 was evicted")

        let bank = store.bank(group: "g", experts: [2, 0])
        XCTAssertEqual(bank["weight"]?.shape, [2, 2, 2])
        XCTAssertEqual(bank["weight"]?[0].asArray(Float.self), [2, 2, 2, 2])
        XCTAssertTrue(store.missing(group: "g", experts: 4, parts: ["weight"]).map(\.expert) == [3])
        store.cacheByteBudget = 0
        XCTAssertEqual(store.cachedBytes, 0, "lowering the budget evicts at once")
    }

    // A paged switch linear reads only the routed experts and computes what the resident stack computes.
    func testAPagedSwitchLinearIsTheResidentOne() throws {
        try requireMLXRuntime()
        MLXRandom.seed(3)
        let resident = NFKLMSwitchLinear(experts: 16, inputSize: 64, outputSize: 32)
        let store = NFKMLXExpertStore(cacheByteBudget: 0)
        for expert in 0 ..< 16 {
            store.store(NFKMLXExpertTensor(storage: .held(resident.weight[expert]), dtype: .float32, shape: [32, 64]),
                        group: "proj", expert: expert, part: "weight")
        }
        let paged = NFKLMPagedSwitchLinear(pager: NFKMLXExpertPager(store: store, group: "proj"), experts: 16,
                                           outputSize: 32, inputSize: 64)
        let x = MLXRandom.normal([1, 5, 1, 1, 64])
        let experts = MLXArray([3, 11, 3, 0, 15, 7, 11, 2, 9, 3].map { UInt32($0) }).reshaped([1, 5, 2])
        let expected = resident(x, experts: experts)
        let actual = paged(x, experts: experts)
        XCTAssertEqual(actual.shape, expected.shape)
        XCTAssertEqual(actual.asArray(Float.self), expected.asArray(Float.self))
        XCTAssertEqual(store.materializeCount, 7, "the seven routed experts, and none of the other nine")
    }

    // MARK: Releases

    /// Writes `arrays` as a two-shard release with its index, the layout every size above the smallest
    /// ships in.
    private func writeRelease(_ arrays: [(String, MLXArray)], to directory: URL) throws {
        let sorted = arrays.sorted { $0.0 < $1.0 }
        let halves = [Array(sorted.prefix(sorted.count / 2)), Array(sorted.dropFirst(sorted.count / 2))]
        var weightMap = [String: String]()
        for (index, half) in halves.enumerated() {
            let name = "model-0000\(index + 1)-of-00002.safetensors"
            try MLX.save(arrays: Dictionary(half, uniquingKeysWith: { first, _ in first }),
                         url: directory.appendingPathComponent(name))
            for (key, _) in half {
                weightMap[key] = name
            }
        }
        let index = try JSONSerialization.data(withJSONObject: ["weight_map": weightMap])
        try index.write(to: directory.appendingPathComponent("model.safetensors.index.json"))
    }

    /// Loads `directory` into a fresh net both ways and holds the paged logits to the resident ones.
    @discardableResult
    private func assertPagedMatchesResident(_ directory: URL, geometry: NFKMLXLanguageConfiguration,
                                            precision: NFKMLXWeightPrecision,
                                            file: StaticString = #filePath, line: UInt = #line) throws
        -> NFKMLXLanguageNet {
        let resident = NFKMLXLanguage.makeNet(geometry)
        try NFKMLXLanguage.loadWeights(into: resident, fromDirectory: directory, precision: precision,
                                       residency: .resident)
        XCTAssertNil(resident.expertStore, file: file, line: line)
        let paged = NFKMLXLanguage.makeNet(geometry)
        try NFKMLXLanguage.loadWeights(into: paged, fromDirectory: directory, precision: precision,
                                       residency: .paged)
        let store = try XCTUnwrap(paged.expertStore, file: file, line: line)
        XCTAssertFalse(paged.parameters().flattened().contains { $0.0.contains(".mlp.experts.") && !$0.0.hasSuffix("_bias") },
                       "a paged net holds no expert matrix as a parameter", file: file, line: line)
        XCTAssertEqual(store.heldBytes, 0, "every expert stays in the release", file: file, line: line)
        XCTAssertGreaterThan(store.mappedBytes, 0, file: file, line: line)

        let expected = resident(tokens)
        let actual = paged(tokens)
        XCTAssertEqual(actual.asArray(Float.self), expected.asArray(Float.self),
                       "a paged load computes the resident one's logits", file: file, line: line)
        store.cacheByteBudget = 0
        XCTAssertEqual(paged(tokens).asArray(Float.self), expected.asArray(Float.self),
                       "with or without a cache", file: file, line: line)
        var greedy = NFKMLXGenerationOptions()
        greedy.temperature = 0
        greedy.maxTokens = 6
        XCTAssertEqual(paged.generate(prompt: [3, 17, 42], options: greedy),
                       resident.generate(prompt: [3, 17, 42], options: greedy),
                       "and generates what it generates", file: file, line: line)
        return paged
    }

    // Qwen3-MoE's one tensor per expert and Mixtral's spelling of the same, stored bf16 and loaded at
    // float32, so the conversion happens per expert on the paged side.
    func testPerExpertReleasesPageExactly() throws {
        try requireMLXRuntime()
        MLXRandom.seed(11)
        let source = NFKMLXLanguage.makeNet(.tinyMixture)
        var released = [(String, MLXArray)]()
        for (name, value) in source.parameters().flattened() {
            guard name.contains(".mlp.experts.") else {
                released.append((name, value.asType(.bfloat16)))
                continue
            }
            let parts = name.components(separatedBy: ".")
            let layer = parts[2], projection = parts[5]
            for expert in 0 ..< value.dim(0) {
                let key = layer == "0"
                    ? "model.layers.0.mlp.experts.\(expert).\(projection).weight"
                    : "model.layers.\(layer).block_sparse_moe.experts.\(expert)."
                        + ["gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"][projection]! + ".weight"
                released.append((key, value[expert].asType(.bfloat16)))
            }
        }
        let directory = try scratchDirectory()
        try writeRelease(released, to: directory)
        let paged = try assertPagedMatchesResident(directory, geometry: .tinyMixture, precision: .float32)
        XCTAssertEqual(paged.expertStore?.expertCount, 2 * 3 * 8, "two layers, three projections, eight experts")
    }

    private func gptOSSGeometry() -> NFKMLXLanguageConfiguration {
        var geometry = NFKMLXLanguageConfiguration(
            hiddenSize: 64, layerCount: 2, headCount: 4, keyValueHeadCount: 2, headDimensions: 16,
            intermediateSize: 32, vocabularySize: 128, ropeTheta: 150_000, rmsEpsilon: 1e-5,
            tiesWordEmbeddings: false, normalizesQueryAndKey: false, attentionBias: true)
        geometry.expertCount = 4
        geometry.activeExpertCount = 2
        geometry.expertIntermediateSize = 32
        geometry.slidingWindows = [4, nil]
        geometry.attentionSinks = true
        geometry.outputProjectionBias = true
        geometry.routerBias = true
        geometry.clampedSwiGLU = NFKMLXClampedSwiGLU()
        return geometry
    }

    // gpt-oss at bf16 stores its fused projections stacked `[experts, in, out]`; a paged load transposes
    // each expert's slice where the resident load transposes the stack.
    func testAStackedTransposedReleasePagesExactly() throws {
        try requireMLXRuntime()
        MLXRandom.seed(12)
        let source = NFKMLXLanguage.makeNet(gptOSSGeometry())
        let released = source.parameters().flattened().map { name, value -> (String, MLXArray) in
            if name.hasSuffix(".mlp.experts.gate_up_proj.weight") || name.hasSuffix(".mlp.experts.down_proj.weight") {
                return (String(name.dropLast(".weight".count)), value.swappedAxes(-1, -2).asType(.bfloat16))
            }
            return (name.replacingOccurrences(of: ".mlp.gate.", with: ".mlp.router."), value.asType(.bfloat16))
        }
        let directory = try scratchDirectory()
        try writeRelease(released, to: directory)
        try assertPagedMatchesResident(directory, geometry: gptOSSGeometry(), precision: .float32)
    }

    // gpt-oss as released: MXFP4 `_blocks` bytes and `_scales`, paged as packed words and multiplied
    // packed, as the resident load multiplies them.
    func testAnMXFP4ReleasePagesExactly() throws {
        try requireMLXRuntime()
        MLXRandom.seed(13)
        let source = NFKMLXLanguage.makeNet(gptOSSGeometry())
        var released = [(String, MLXArray)]()
        for (name, value) in source.parameters().flattened() {
            if name.hasSuffix(".mlp.experts.gate_up_proj.weight") || name.hasSuffix(".mlp.experts.down_proj.weight") {
                let (words, scales, _) = MLX.quantized(value, groupSize: 32, bits: 4, mode: .mxfp4)
                let base = String(name.dropLast(".weight".count))
                let blocks = words.view(dtype: .uint8).reshaped([words.dim(0), words.dim(1), -1, 16])
                released.append((base + "_blocks", blocks))
                released.append((base + "_scales", scales))
                continue
            }
            released.append((name.replacingOccurrences(of: ".mlp.gate.", with: ".mlp.router."), value.asType(.bfloat16)))
        }
        let directory = try scratchDirectory()
        try writeRelease(released, to: directory)
        try assertPagedMatchesResident(directory, geometry: gptOSSGeometry(), precision: .checkpoint)
    }

    // A checkpoint this package saved from a quantized mixture: stacked, affine-packed, one file.
    func testAQuantizedCheckpointPagesExactly() throws {
        try requireMLXRuntime()
        MLXRandom.seed(14)
        let source = NFKMLXLanguage.makeNet(.tinyMixture)
        NFKMLXQuantization.quantize(module: source, bits: 8, groupSize: 32)
        let directory = try scratchDirectory()
        try NFKMLXWeights.save(source, to: directory.appendingPathComponent("model.safetensors"))
        let paged = try assertPagedMatchesResident(directory, geometry: .tinyMixture, precision: .float32)
        let mixture = try XCTUnwrap(paged.model.layers[0].feedForward as? NFKLMMixtureFeedForward)
        let gate = try XCTUnwrap((mixture.experts as? NFKLMSwitchGLU)?.gate as? NFKLMPagedSwitchLinear)
        XCTAssertEqual(gate.quantization, NFKMLXWeights.Quantization(bits: 8, groupSize: 32, mode: .affine))
    }

    // MARK: The other mixture families

    /// Writes `source`'s parameters as a release under `releaseKey`, loads it into two fresh nets
    /// resident and paged, and holds the paged logits to the resident ones.
    private func assertFamilyPagesExactly<Net: Module>(
        _ make: () -> Net, releaseKey: (String, MLXArray) -> (String, MLXArray),
        fuse: ([(String, MLXArray)]) -> [(String, MLXArray)] = { $0 },
        load: (Net, URL, NFKMLXResidency) throws -> Void, store: (Net) -> NFKMLXExpertStore?,
        forward: (Net) -> MLXArray, file: StaticString = #filePath, line: UInt = #line) throws {
        let source = make()
        let directory = try scratchDirectory()
        try writeRelease(fuse(source.parameters().flattened().map(releaseKey)), to: directory)
        let resident = make()
        try load(resident, directory, .resident)
        XCTAssertNil(store(resident), file: file, line: line)
        let paged = make()
        try load(paged, directory, .paged)
        let pagedStore = try XCTUnwrap(store(paged), file: file, line: line)
        XCTAssertGreaterThan(pagedStore.expertCount, 0, file: file, line: line)
        XCTAssertLessThan(paged.parameters().flattened().count, resident.parameters().flattened().count,
                          "the paged net holds no expert projection", file: file, line: line)
        let expected = forward(resident)
        XCTAssertEqual(forward(paged).asArray(Float.self), expected.asArray(Float.self),
                       "a paged load computes the resident one's logits", file: file, line: line)
        XCTAssertGreaterThan(pagedStore.materializeCount, 0, file: file, line: line)
    }

    func testGemmaMixturePagesExactly() throws {
        try requireMLXRuntime()
        MLXRandom.seed(21)
        try assertFamilyPagesExactly(
            { NFKMLXGemmaLanguage.makeNet(.tinyMixture) },
            releaseKey: { ("model.language_model." + $0, $1) },
            load: { try NFKMLXGemmaLanguage.loadWeights(into: $0, fromDirectory: $1, precision: .float32, residency: $2) },
            store: \.expertStore, forward: { $0(self.tokens) })
    }

    func testQwen4ExpPagesExactly() throws {
        try requireMLXRuntime()
        MLXRandom.seed(22)
        try assertFamilyPagesExactly(
            { NFKMLXQwen4Exp.makeNet(.tiny) },
            releaseKey: { key, value in
                (key, key.hasSuffix("conv1d.weight") && value.ndim == 3 ? value.transposed(0, 2, 1) : value)
            },
            load: { try NFKMLXQwen4Exp.loadWeights(into: $0, fromDirectory: $1, precision: .float32, residency: $2) },
            store: \.expertStore, forward: { $0(self.tokens) })
    }

    func testGraniteMixturePagesExactly() throws {
        try requireMLXRuntime()
        MLXRandom.seed(23)
        let geometry = NFKMLXGraniteHybridConfiguration(
            hiddenSize: 64, layerCount: 3, vocabularySize: 128, rmsEpsilon: 1e-5,
            tiesWordEmbeddings: false, headCount: 4, keyValueHeadCount: 2, headDimensions: 16,
            mambaHeadCount: 8, mambaHeadDimensions: 16, mambaGroupCount: 1, mambaStateSize: 16,
            mambaConvolutionKernel: 4, mambaExpand: 2, mambaConvolutionBias: true,
            mambaProjectionBias: false, sharedIntermediateSize: 96, expertCount: 8,
            expertsPerToken: 2, expertIntermediateSize: 32,
            embeddingMultiplier: 2.0, residualMultiplier: 0.5, attentionMultiplier: 0.25,
            logitsScaling: 3.0, layerTypes: [.mamba, .attention, .mamba])
        try assertFamilyPagesExactly(
            { NFKMLXGraniteHybrid.makeNet(geometry) }, releaseKey: { ($0, $1) },
            load: { try NFKMLXGraniteHybrid.loadWeights(into: $0, fromDirectory: $1, precision: .float32, residency: $2) },
            store: \.expertStore, forward: { $0(self.tokens) })
    }

    // Qwen3-VL 30B-A3B fuses each layer's gate and up experts `[experts, hidden, 2·width]` and stores
    // `down_proj` `[experts, width, hidden]`; a paged load splits each expert's slice as the resident
    // load splits the stack.
    func testQwen3VLFusedMixturePagesExactly() throws {
        try requireMLXRuntime()
        MLXRandom.seed(24)
        let geometry = NFKMLXLanguageConfiguration.tinyMixture
        try assertFamilyPagesExactly(
            { NFKMLXLanguageNet(geometry) },
            releaseKey: { key, value in
                let released = key.hasPrefix("model.") ? "model.language_model." + key.dropFirst("model.".count) : key
                return (released, value)
            },
            fuse: { released in
                var fused = [(String, MLXArray)]()
                var gates = [String: MLXArray]()
                for (key, value) in released {
                    if key.hasSuffix(".mlp.experts.gate_proj.weight") {
                        gates[String(key.dropLast("gate_proj.weight".count))] = value
                    } else if key.hasSuffix(".mlp.experts.up_proj.weight") {
                        let base = String(key.dropLast("up_proj.weight".count))
                        fused.append((base + "gate_up_proj",
                                      concatenated([gates[base]!, value], axis: 1).swappedAxes(1, 2)))
                    } else if key.hasSuffix(".mlp.experts.down_proj.weight") {
                        fused.append((String(key.dropLast(".weight".count)), value.swappedAxes(1, 2)))
                    } else {
                        fused.append((key, value))
                    }
                }
                return fused
            },
            load: { try NFKMLXQwen3VL.loadDecoderWeights(into: $0, fromDirectory: $1, precision: .float32, residency: $2) },
            store: \.expertStore, forward: { $0(self.tokens) })
    }

    // A dense release has nothing to page and loads resident whatever the residency asks.
    func testADenseReleaseIsNeverPaged() throws {
        try requireMLXRuntime()
        let source = NFKMLXLanguage.makeNet(.tiny)
        let directory = try scratchDirectory()
        try writeRelease(source.parameters().flattened(), to: directory)
        let net = NFKMLXLanguage.makeNet(.tiny)
        try NFKMLXLanguage.loadWeights(into: net, fromDirectory: directory, precision: .float32, residency: .paged)
        XCTAssertNil(net.expertStore)
        XCTAssertEqual(net(tokens).asArray(Float.self), source(tokens).asArray(Float.self))
    }

    // A paged load missing one expert's matrix is refused at load, since no parameter check can see it.
    func testAPagedReleaseMissingAnExpertIsRefused() throws {
        try requireMLXRuntime()
        let source = NFKMLXLanguage.makeNet(.tinyMixture)
        var released = [(String, MLXArray)]()
        for (name, value) in source.parameters().flattened() {
            guard name.contains(".mlp.experts.") else {
                released.append((name, value))
                continue
            }
            let parts = name.components(separatedBy: ".")
            for expert in 0 ..< value.dim(0) where !(parts[2] == "1" && parts[5] == "up_proj" && expert == 5) {
                released.append(("model.layers.\(parts[2]).mlp.experts.\(expert).\(parts[5]).weight", value[expert]))
            }
        }
        let directory = try scratchDirectory()
        try writeRelease(released, to: directory)
        let net = NFKMLXLanguage.makeNet(.tinyMixture)
        XCTAssertThrowsError(try NFKMLXLanguage.loadWeights(into: net, fromDirectory: directory, precision: .float32,
                                                            residency: .paged)) { error in
            XCTAssertTrue("\(error)".contains("model.layers.1.mlp.experts.up_proj.5.weight"), "\(error)")
        }
    }

    // The released gpt-oss-20b, its MXFP4 experts paged from the release, against its own resident
    // load: every tensor but the experts loads, and the logits and a greedy continuation match.
    func testTheReleasedGPTOSSPagesExactly() throws {
        try requireMLXRuntime()
        guard let path = NFKMLXValidationConfig.environment["IK_VAL_GPT_OSS"] else {
            throw XCTSkip("set IK_VAL_GPT_OSS (openai/gpt-oss-20b)")
        }
        let directory = URL(fileURLWithPath: path)
        let geometry = try NFKMLXLanguage.configuration(fromHuggingFace: directory.appendingPathComponent("config.json"))
        let prompt = MLXArray([976, 9029, 328, 10128, 382].map { Int32($0) }).reshaped([1, 5])
        var greedy = NFKMLXGenerationOptions()
        greedy.temperature = 0
        greedy.maxTokens = 8

        let paged = NFKMLXLanguage.makeNet(geometry)
        try NFKMLXLanguage.loadWeights(into: paged, fromDirectory: directory, precision: .checkpoint, residency: .paged)
        let store = try XCTUnwrap(paged.expertStore)
        let pagedLogits = paged(prompt)
        eval(pagedLogits)
        let pagedTokens = paged.generate(prompt: [976, 9029, 328, 10128, 382], options: greedy)
        print("VALIDATION runtime gpt-oss-20b-paged: \(store.expertCount) experts, \(store.mappedBytes / 1_048_576) MiB "
              + "left in the release, \(store.materializeCount) materialized, \(store.cacheHitCount) cache hits")
        NFKMLXGPU.clearCache()

        let resident = NFKMLXLanguage.makeNet(geometry)
        try NFKMLXLanguage.loadWeights(into: resident, fromDirectory: directory, precision: .checkpoint,
                                       residency: .resident)
        let residentLogits = resident(prompt)
        eval(residentLogits)
        let difference = abs(pagedLogits.asType(.float32) - residentLogits.asType(.float32)).max().item(Float.self)
        print("VALIDATION PARITY gpt-oss-20b-paged: logits max abs difference \(difference)")
        XCTAssertEqual(difference, 0, "a paged load computes the resident load's logits")
        XCTAssertEqual(pagedTokens, resident.generate(prompt: [976, 9029, 328, 10128, 382], options: greedy),
                       "and generates what it generates")
    }

    // Throughput of a paged gathered multiply over a synthetic stacked release, left mapped: the cost of
    // reading routed experts out of the mapping with no cache, and with every expert cached. Runs only
    // where IK_BENCH_EXPERT_PAGING is set, since it writes a 384 MB file and measures rather than checks.
    func testPagedGatherThroughput() throws {
        try requireMLXRuntime()
        try XCTSkipUnless(ProcessInfo.processInfo.environment["IK_BENCH_EXPERT_PAGING"] != nil,
                          "set IK_BENCH_EXPERT_PAGING to measure paged throughput")
        let (experts, output, input) = (64, 1536, 2048)
        // A fixed name, so a run that dies before its cleanup leaves one file that the next run reuses.
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("inferkit-expert-paging-bench.safetensors")
        MLXRandom.seed(51)
        let stack = MLXRandom.normal([experts, output, input]).asType(.bfloat16)
        try MLX.save(arrays: ["proj.weight": stack], url: url)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        let entry = try XCTUnwrap(try NFKMLXSafetensors.entries(inFile: url)["proj.weight"])
        let file = try NFKMLXMappedFile(url: url)
        let store = NFKMLXExpertStore(cacheByteBudget: 0)
        for expert in 0 ..< experts {
            store.store(try XCTUnwrap(NFKMLXExpertTensor.mapped(entry, expert: expert, in: file)),
                        group: "proj", expert: expert, part: "weight")
        }
        let paged = NFKLMPagedSwitchLinear(pager: NFKMLXExpertPager(store: store, group: "proj"), experts: experts,
                                           outputSize: output, inputSize: input)
        let x = MLXRandom.normal([1, 32, 1, 1, input]).asType(.bfloat16)
        let routes = (0 ..< 10).map { call in
            MLXArray((0 ..< 32 * 8).map { UInt32(($0 * 7 + call * 13) % experts) }).reshaped([1, 32, 8])
        }
        eval(paged(x, experts: routes[0]))

        func milliseconds(_ body: () -> Void) -> Double {
            let start = Date()
            body()
            return Date().timeIntervalSince(start) * 1000
        }
        let uncached = milliseconds { for route in routes { eval(paged(x, experts: route)) } } / Double(routes.count)
        let expertBytes = Double(output * input * 2)
        let readPerCall = Double(store.materializeCount - 1) / Double(routes.count + 1) * expertBytes
        store.cacheByteBudget = experts * output * input * 2
        eval(paged(MLXRandom.normal([1, experts, 1, 1, input]).asType(.bfloat16),
                   experts: MLXArray((0 ..< experts).map { UInt32($0) }).reshaped([1, experts, 1])))
        let cached = milliseconds { for route in routes { eval(paged(x, experts: route)) } } / Double(routes.count)
        print("VALIDATION bench expert-paging: uncached \(String(format: "%.1f", uncached)) ms/call "
              + "(\(String(format: "%.2f", readPerCall / uncached / 1e6)) GB/s of experts), cached "
              + "\(String(format: "%.1f", cached)) ms/call, \(store.cacheHitCount) cache hits")
    }
}
