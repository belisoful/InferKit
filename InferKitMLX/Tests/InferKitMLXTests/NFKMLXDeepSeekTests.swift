//
//  NFKMLXDeepSeekTests.swift
//  InferKitMLXTests
//
//  The DeepSeek V4 decoder. Nothing here loads weights: the released Flash model is hundreds of
//  gigabytes and cannot be instantiated at float precision on any single machine here, let alone run.
//
//  The verification is weaker than the hybrid decoder's, and deliberately says so. That checkpoint is
//  bf16, so a float module's shapes match it exactly. This one is QUANTIZED — attention in fp8, routed
//  experts 4-bit packed two to a byte — so the check has to derive what each float parameter looks like
//  stored. That derivation is an assumption, so it is asserted against the observed shapes rather than
//  trusted.
//

import XCTest
import MLX
import MLXNN
@testable import InferKitMLX

final class NFKMLXDeepSeekTests: XCTestCase {

    private lazy var config: [String: String] = {
        var merged = ProcessInfo.processInfo.environment
        let url = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".inferkit-validation.json")
        if let data = try? Data(contentsOf: url),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: String] {
            json.forEach { merged[$0.key] = $0.value }
        }
        return merged
    }()

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    private func released(_ suffix: String = "") throws
        -> (NFKMLXDeepSeekConfiguration, [String: [String: Any]]) {
        guard let shapesPath = config["IK_SHAPES_DEEPSEEK_V4" + suffix],
              let configPath = config["IK_CONFIG_DEEPSEEK_V4" + suffix],
              let data = FileManager.default.contents(atPath: shapesPath),
              let raw = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { throw XCTSkip("set IK_SHAPES_DEEPSEEK_V4\(suffix) and IK_CONFIG_DEEPSEEK_V4\(suffix)") }
        let geometry = try NFKMLXDeepSeek.configuration(
            fromHuggingFace: URL(fileURLWithPath: configPath))
        return (geometry, try headers(raw, shapesPath: shapesPath))
    }

    /// `name -> {shape, dtype}`, from either capture format: the combined one the earlier releases
    /// were recorded in, or `shapes.py`'s `shapes.json` beside its `dtypes.json`.
    private func headers(_ raw: [String: Any], shapesPath: String) throws -> [String: [String: Any]] {
        if let combined = raw as? [String: [String: Any]] { return combined }
        guard let shapes = raw as? [String: [Int]] else {
            throw XCTSkip("\(shapesPath) is neither capture format")
        }
        let dtypesURL = URL(fileURLWithPath: shapesPath)
            .deletingLastPathComponent().appendingPathComponent("dtypes.json")
        let dtypes = (FileManager.default.contents(atPath: dtypesURL.path))
            .flatMap { try? JSONSerialization.jsonObject(with: $0) } as? [String: String] ?? [:]
        return shapes.mapValues { ["shape": $0] }.merging(
            dtypes.mapValues { ["dtype": $0] }) { shape, dtype in shape.merging(dtype) { a, _ in a } }
    }

    /// Small enough to build and run while keeping every structural ratio.
    private var small: NFKMLXDeepSeekConfiguration {
        NFKMLXDeepSeekConfiguration(hiddenSize: 64, layerCount: 4, vocabularySize: 128,
                                    headCount: 4, headDimensions: 16, ropeHeadDimensions: 4,
                                    queryLoRARank: 16, outputLoRARank: 16, outputGroups: 2,
                                    slidingWindow: 8, routedExpertCount: 8, activatedExpertCount: 2,
                                    expertIntermediateSize: 32, hashLayerCount: 1)
    }

    // MARK: The configuration

    // A preset is what a caller with no config.json builds from, so it has to be the release's own
    // configuration in every field, not only in the shapes the structural check sees. Every stored
    // property is compared, so a field added later is covered without naming it here.
    func testEveryPresetIsItsReleasesConfiguration() throws {
        for (key, preset, name) in [("IK_CONFIG_DEEPSEEK_V4", NFKMLXDeepSeekConfiguration.v4Flash, "v4Flash"),
                                    ("IK_CONFIG_DEEPSEEK_V4_PRO", .v4Pro, "v4Pro"),
                                    ("IK_CONFIG_DEEPSEEK_V41", .v41Flash, "v41Flash")] {
            guard let path = config[key] else { throw XCTSkip("set \(key)") }
            let read = try NFKMLXDeepSeek.configuration(fromHuggingFace: URL(fileURLWithPath: path))
            let fields = Dictionary(uniqueKeysWithValues: Mirror(reflecting: read).children
                .compactMap { child in child.label.map { ($0, String(describing: child.value)) } })
            var differences = [String]()
            for child in Mirror(reflecting: preset).children {
                guard let label = child.label else { continue }
                let mine = String(describing: child.value)
                if fields[label] != mine {
                    differences.append("\(label): preset \(mine.prefix(60)), release \((fields[label] ?? "absent").prefix(60))")
                }
            }
            print("SEAM deepseek preset \(name) differs from its release in: \(differences.isEmpty ? "nothing" : differences.joined(separator: " | "))")
            XCTAssertEqual(differences, [], "\(name) is its release's configuration")
        }
    }

    func testTheReleasedConfigurationIsRead() throws {
        let (geometry, _) = try released()
        XCTAssertTrue(geometry.computesInBFloat16, "V4 declares bf16 and its own code runs in it")
        XCTAssertEqual(geometry.hiddenSize, 4096)
        XCTAssertEqual(geometry.layerCount, 43)
        XCTAssertEqual(geometry.headCount, 64)
        XCTAssertEqual(geometry.headDimensions, 512)
        XCTAssertEqual(geometry.routedExpertCount, 256)
        XCTAssertEqual(geometry.activatedExpertCount, 6)
        XCTAssertEqual(geometry.outputGroups, 8)
        XCTAssertEqual(geometry.hashLayerCount, 3)
        // The first two layers attend over the window alone; compression starts after them.
        XCTAssertEqual(geometry.compressRatios[0], 0)
        XCTAssertEqual(geometry.compressRatios[1], 0)
        XCTAssertGreaterThan(geometry.compressRatios[2], 0)
    }

    // The first layers route by a table indexed by token id, the rest by score with a learned bias.
    // Which a layer uses decides which parameters it carries, so a wrong boundary is a wrong model.
    func testRoutingChangesAfterTheHashLayers() throws {
        let (geometry, observed) = try released()
        for layer in 0 ..< geometry.hashLayerCount {
            XCTAssertNotNil(observed["layers.\(layer).ffn.gate.tid2eid"], "layer \(layer) routes by table")
            XCTAssertNil(observed["layers.\(layer).ffn.gate.bias"])
        }
        let scored = geometry.hashLayerCount
        XCTAssertNil(observed["layers.\(scored).ffn.gate.tid2eid"])
        XCTAssertNotNil(observed["layers.\(scored).ffn.gate.bias"], "layer \(scored) routes by score")
    }

    // MARK: Structure against the released checkpoint

    // The derivation of a quantized shape is an assumption; this is where it is tested. An attention
    // weight is fp8 and keeps its shape; a routed expert is 4-bit packed two to a byte, so its last
    // axis halves. If either is wrong the comparison below would be meaningless.
    func testTheQuantizedLayoutIsWhatTheReleaseUses() throws {
        let (_, observed) = try released()
        let attention = try XCTUnwrap(observed["layers.0.attn.wq_a.weight"])
        XCTAssertEqual(attention["dtype"] as? String, "F8_E4M3", "attention is fp8")
        XCTAssertEqual(attention["shape"] as? [Int], [1024, 4096], "and keeps its float shape")

        let expert = try XCTUnwrap(observed["layers.0.ffn.experts.0.w1.weight"])
        XCTAssertEqual(expert["dtype"] as? String, "I8", "a routed expert is packed into bytes")
        XCTAssertEqual(expert["shape"] as? [Int], [2048, 2048],
                       "two 4-bit values a byte, so 4096 columns are stored as 2048")

        let shared = try XCTUnwrap(observed["layers.0.ffn.shared_experts.w1.weight"])
        XCTAssertEqual(shared["dtype"] as? String, "F8_E4M3", "the shared expert is not 4-bit")
        XCTAssertEqual(shared["shape"] as? [Int], [2048, 4096])
    }

    // Every parameter the architecture declares, against the checkpoint's own headers. Enumerated
    // analytically: 43 layers of 257 experts cannot be instantiated at float precision.
    func testEveryDeclaredParameterMatchesTheReleasedCheckpoint() throws {
        try assertDeclaredParametersMatch(release: "", label: "deepseek-v4")
    }

    /// Flash's checks, run against another release of the same architecture.
    ///
    /// The Pro sizes everything up — 61 layers, hidden 7168, 384 experts, 128 heads — and adds a
    /// YaRN `rope_scaling`, which carries no parameters and so is invisible to a structural check.
    /// This is the Qwen3.8-27B treatment: the architecture is enumerated analytically and compared
    /// against captured shard headers, never instantiated.
    func testTheProReleaseMatchesStructurally() throws {
        try assertDeclaredParametersMatch(release: "_PRO", label: "deepseek-v4-pro")
    }

    private func assertDeclaredParametersMatch(release suffix: String, label: String) throws {
        let (geometry, observed) = try released(suffix)
        let expected = NFKMLXDeepSeek.expectedParameters(for: geometry)

        // Headers were captured for a subset of shards, so only those layers can be compared. The
        // meaningful coverage assertion is that EVERY expected parameter of a captured layer is
        // checked — an arbitrary count would pass while silently skipping a whole kind of parameter.
        let capturedLayers = Set(observed.keys.compactMap { key -> Int? in
            guard key.hasPrefix("layers.") else { return nil }
            return Int(key.split(separator: ".")[1])
        })
        XCTAssertFalse(capturedLayers.isEmpty, "no layer headers were captured")

        var checked = 0
        var mismatched = [String]()
        var uncomparable = [String]()
        for (name, floatShape) in expected {
            let layer = name.hasPrefix("layers.") ? Int(name.split(separator: ".")[1]) : nil
            let shouldCompare = layer.map(capturedLayers.contains) ?? (observed[name] != nil)
            guard shouldCompare else { continue }
            guard let shape = observed[name]?["shape"] as? [Int] else {
                uncomparable.append(name); continue
            }
            checked += 1
            let stored = NFKMLXDeepSeek.quantizedShape(of: name, float: floatShape)
            if stored != shape {
                mismatched.append("\(name): expected \(stored), released \(shape)")
            }
        }
        print("VALIDATION structure \(label): \(checked) parameters checked across "
              + "\(capturedLayers.count) captured layers, \(mismatched.count) mismatched")
        XCTAssertTrue(mismatched.isEmpty, "shape mismatches:\n" + mismatched.prefix(8).joined(separator: "\n"))
        XCTAssertTrue(uncomparable.isEmpty, "declared but absent from the release:\n"
                      + uncomparable.prefix(8).joined(separator: "\n"))
    }

    // THE CONVERSE, and the assertion that matters most: every tensor in the release is either a
    // parameter this port declares or falls in a named unimplemented group. Without this the check
    // only proves "what I declare exists", which is how an entire mechanism hid here — the
    // hyper-connection weights were in the checkpoint and absent from the port, and the one-directional
    // comparison reported zero problems.
    func testEveryReleasedTensorIsDeclaredOrNamed() throws {
        try assertEveryTensorAccounted(release: "", label: "deepseek-v4")
    }

    func testEveryProTensorIsDeclaredOrNamed() throws {
        try assertEveryTensorAccounted(release: "_PRO", label: "deepseek-v4-pro")
    }

    // V4.1 Flash: the same architecture at 763B, and the first release of it this port reads. Its
    // shapes are what say the V4.1 differences were understood — four layers own a compressor where
    // V4 gave one to every compressed layer, the hash-routed layers are gone, the copies collapse
    // without a learned head, and two layers carry an n-gram memory of 384 million rows.
    func testTheV41ReleaseMatchesStructurally() throws {
        try assertDeclaredParametersMatch(release: "1", label: "deepseek-v4.1-flash")
    }

    func testEveryV41TensorIsDeclaredOrNamed() throws {
        try assertEveryTensorAccounted(release: "1", label: "deepseek-v4.1-flash")
    }

    // The same assertion for V4.1, whose storage differs in a way the shape check cannot see: the
    // block scales are not blocked alike. An attention weight carries one scale per 32x32 block,
    // while a routed expert and the n-gram table carry one per ROW per 32 columns — finer, and a
    // dequantizer that assumed the square blocking would read the wrong scale for every row.
    func testTheV41QuantizedLayoutIsWhatTheReleaseUses() throws {
        let (_, observed) = try released("1")

        let attention = try XCTUnwrap(observed["layers.0.attn.wq_a.weight"])
        XCTAssertEqual(attention["dtype"] as? String, "F8_E4M3", "attention is fp8")
        XCTAssertEqual(attention["shape"] as? [Int], [1280, 5120], "and keeps its float shape")
        XCTAssertEqual(try XCTUnwrap(observed["layers.0.attn.wq_a.scale"])["shape"] as? [Int],
                       [40, 160], "one scale per 32x32 block")

        let expert = try XCTUnwrap(observed["layers.0.ffn.experts.0.w1.weight"])
        XCTAssertEqual(expert["dtype"] as? String, "I8", "a routed expert is packed into bytes")
        XCTAssertEqual(expert["shape"] as? [Int], [2304, 2560],
                       "two 4-bit values a byte, so 5120 columns are stored as 2560")
        XCTAssertEqual(try XCTUnwrap(observed["layers.0.ffn.experts.0.w1.scale"])["shape"] as? [Int],
                       [2304, 160], "one scale per row per 32 columns, not per 32x32 block")

        let shared = try XCTUnwrap(observed["layers.0.ffn.shared_experts.w1.weight"])
        XCTAssertEqual(shared["dtype"] as? String, "F8_E4M3", "the shared expert is not 4-bit")
        XCTAssertEqual(shared["shape"] as? [Int], [2304, 5120])

        // The n-gram table is the release's largest tensor and the only embedding stored quantized.
        let table = try XCTUnwrap(observed["layers.1.engram.embed.weight"])
        XCTAssertEqual(table["dtype"] as? String, "F8_E4M3", "the n-gram table stays fp8 as stored")
        XCTAssertEqual(table["shape"] as? [Int], [384_006_168, 256], "and keeps its float shape")
        let tableScale = try XCTUnwrap(observed["layers.1.engram.embed.scale"])
        XCTAssertEqual(tableScale["dtype"] as? String, "F8_E8M0", "the scale is an exponent alone")
        XCTAssertEqual(tableScale["shape"] as? [Int], [384_006_168, 8],
                       "one scale per row per 32 channels, so a row dequantizes on its own")

        XCTAssertEqual(try XCTUnwrap(observed["embed.weight"])["dtype"] as? String, "BF16",
                       "the token embedding is not quantized")
    }

    /// The configuration the `deepseek_v41` oracle builds the reference from.
    ///
    /// Its layer pattern differs from the release's on purpose: two key-value sources against four
    /// index sources, a ratio-2 and a ratio-1 compressor, and a non-source layer between them. A
    /// rule that happens to be right for the released pattern and wrong in general fails here.
    private var oracleShaped: NFKMLXDeepSeekConfiguration {
        NFKMLXDeepSeekConfiguration(
            hiddenSize: 64, layerCount: 6, vocabularySize: 256, rmsEpsilon: 1e-20,
            headCount: 4, headDimensions: 32, ropeHeadDimensions: 8,
            queryLoRARank: 32, outputLoRARank: 16, outputGroups: 2, slidingWindow: 8,
            ropeTheta: 10_000, compressRopeTheta: 40_000,
            hyperConnectionCopies: 3, sinkhornIterations: 4,
            indexHeadCount: 8, indexHeadDimensions: 16, indexTopK: 4,
            routedExpertCount: 8, activatedExpertCount: 2, expertIntermediateSize: 32,
            routeScale: 1, swigluLimit: 0,
            hashLayerCount: 0, routerHasVisionBias: false,
            compressRatios: [0, 0, 2, 2, 1, 1], compressorHasPositionBias: false,
            keyValueSourceLayers: [2, 4], indexSourceLayers: [2, 3, 4, 5],
            candidateSourceLayer: 4, candidateTopKBlocks: 3, candidateBlockSize: 2,
            collapsesThroughLearnedHead: false, indexerDerivesKeysFromCompressor: true,
            normalizesQueryHeads: false, compressedLayersRotateAtCompressedBase: true,
            pipelinesHyperConnectionRead: true,
            engramLayerIDs: [1, 3], engramEmbeddingCounts: [290, 370],
            engramMaxNgramSize: 3, engramHeadCount: 2, engramHeadDimensions: 32,
            engramVocabularySize: 64, engramCompressedVocabularySize: 256, engramPadToken: 2)
    }

    /// The collapsed id space the oracle hashes over. Its tokens all normalize apart, so the map is
    /// the identity — which is what makes the compressed size equal the vocabulary size there, and
    /// what lets the reference's own derivation run without a 129,280-entry tokenizer in the record.
    private var oracleTokens: [Int] { Array(0 ..< 256) }

    private func oracleNet() -> NFKMLXDeepSeekNet {
        NFKMLXDeepSeekNet(oracleShaped, compressedTokens: oracleTokens)
    }

    // The enumeration is otherwise held to one release, which fixes its rules at one size and one
    // layer pattern. The oracle builds the reference's OWN module tree at a different size, so
    // comparing names and shapes against it checks the rules rather than one instance of them.
    func testTheEnumerationMatchesTheReferencesModuleTree() throws {
        guard let path = config["IK_PARITY_DEEPSEEK_V41"] else {
            throw XCTSkip("set IK_PARITY_DEEPSEEK_V41 (Tools/reference-parity deepseek_v41)")
        }
        let record = try MLX.loadArrays(url: URL(fileURLWithPath: path))
        var reference = [String: [Int]]()
        for (key, value) in record where key.hasPrefix("w::") {
            let name = String(key.dropFirst(3))
            // A `.scale` is a quantized weight's companion, not a parameter of its own. The release
            // accounting counts those with the block scales, and a float module has none.
            if name.hasSuffix(".scale") { continue }
            reference[name] = value.shape
        }
        XCTAssertFalse(reference.isEmpty, "the record carries no w:: weights")

        let declared = NFKMLXDeepSeek.expectedParameters(for: oracleShaped)
        XCTAssertEqual(Set(declared.keys), Set(reference.keys),
                       "the enumeration names exactly what the reference builds")
        var mismatched = [String]()
        for (name, shape) in declared where reference[name] != nil && reference[name] != shape {
            mismatched.append("\(name): declared \(shape), reference \(reference[name]!)")
        }
        print("VALIDATION structure deepseek-v4.1-oracle: \(declared.count) parameters, "
              + "\(mismatched.count) mismatched against the reference's own module tree")
        XCTAssertTrue(mismatched.isEmpty, mismatched.prefix(8).joined(separator: "\n"))
    }

    /// The release writes a block's hyper-connection coefficients as three flat tensors per part
    /// (`hc_attn_fn`); the module holds a submodule with three parameters (`hc_attn.fn`). The
    /// enumeration speaks the release's naming, so the bridge belongs here.
    private static func hyperConnectionNaming(_ key: String) -> String {
        var name = key
        for part in ["attn", "ffn"] {
            for field in ["fn", "base", "scale"] {
                name = name.replacingOccurrences(of: "hc_\(part)_\(field)",
                                                 with: "hc_\(part).\(field)")
            }
        }
        return name
    }

    private func cosine(_ a: [Double], _ b: [Double]) -> Double {
        let dot = zip(a, b).reduce(0) { $0 + $1.0 * $1.1 }
        let na = sqrt(a.reduce(0) { $0 + $1 * $1 }), nb = sqrt(b.reduce(0) { $0 + $1 * $1 })
        return na == 0 || nb == 0 ? 0 : dot / (na * nb)
    }

    /// The record's weights under the names this module's parameters carry.
    ///
    /// The release writes a block's hyper-connection coefficients as three flat tensors per part
    /// (`hc_attn_fn`), where the module holds a submodule with three parameters (`hc_attn.fn`). The
    /// enumeration speaks the release's naming, so the bridge belongs here.
    private func oracleWeights(_ record: [String: MLXArray]) -> [(String, MLXArray)] {
        record.compactMap { key, value in
            guard key.hasPrefix("w::") else { return nil }
            var name = String(key.dropFirst(3))
            name = Self.hyperConnectionNaming(name)
            // The n-gram table stays quantized in the release and the reference dequantizes a row
            // as it looks it up. The module holds the table dequantized, so the scale is folded in
            // here rather than carried as a parameter — which is also why the release counts it
            // among the block scales rather than among the parameters.
            if name.hasSuffix("engram.embed.weight"),
               let scale = record["w::" + name.replacingOccurrences(of: ".weight", with: ".scale")] {
                let block = value.shape[1] / scale.shape[1]
                return (name, value * repeated(scale, count: block, axis: -1))
            }
            if name.hasSuffix("engram.embed.scale") { return nil }
            return (name, value)
        }
    }

    // The first numeric check of a V4.1 mechanism: the hyper-connection coefficients at the first
    // layer, which depend on nothing but the expanded embedding. It covers the normalized
    // projection, the three splits, both sigmoid forms, and the Sinkhorn projection at an iteration
    // count the released configuration does not use.
    func testTheHyperConnectionCoefficientsMatchTheReference() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_DEEPSEEK_V41"] else {
            throw XCTSkip("set IK_PARITY_DEEPSEEK_V41 (Tools/reference-parity deepseek_v41)")
        }
        let record = try MLX.loadArrays(url: URL(fileURLWithPath: path))
        let tokens = try XCTUnwrap(record["tokens"]).asArray(Int32.self)

        let net = oracleNet()
        try NFKMLXWeights.apply(oracleWeights(record), to: net)

        let ids = MLXArray(tokens).reshaped([1, tokens.count])
        let expanded = repeated(net.embed(ids).expandedDimensions(axis: 2),
                                count: oracleShaped.hyperConnectionCopies, axis: 2)
        let (read, write, combine) = net.layers[0].attentionConnection.weights(expanded)

        for (name, mine) in [("pre", read), ("post", write), ("comb", combine)] {
            let reference = try XCTUnwrap(record["seam.hc.attn.\(name).0"])
            eval(mine)
            let similarity = cosine(mine.reshaped([-1]).asArray(Float.self).map(Double.init),
                                    reference.reshaped([-1]).asArray(Float.self).map(Double.init))
            print("SEAM deepseek-v4.1 hyper-connection \(name): cosine \(similarity)")
            XCTAssertGreaterThan(similarity, 0.9999, "the \(name) coefficients match the reference")
        }
    }

    // The decoder's own copies stay within a unit in the last place of one another, so which index
    // the residual mix sums over is invisible in it. The record's probe hands the release's
    // `hc_post` and `hc_pre` copies that differ by several units; in bf16 each result is bit-exact.
    func testTheHyperConnectionMixesDistinctCopiesAsTheReleaseDoes() throws {
        try requireMLXRuntime()
        for (key, bfloat16) in [("IK_PARITY_DEEPSEEK_V41", false), ("IK_PARITY_DEEPSEEK_V41_BF16_PLAIN", true)] {
            guard let path = config[key] else { throw XCTSkip("set \(key) (Tools/reference-parity)") }
            let record = try MLX.loadArrays(url: URL(fileURLWithPath: path))
            func probe(_ name: String) throws -> MLXArray {
                let value = try XCTUnwrap(record["probe.hc.\(name)"], "regenerate \(key)")
                    .expandedDimensions(axis: 0)
                return bfloat16 && (name == "x" || name == "residual") ? value.asType(.bfloat16) : value
            }
            var configuration = oracleShaped
            configuration.computesInBFloat16 = bfloat16
            let net = try NFKMLXDeepSeek.makeNet(configuration, compressedTokens: oracleTokens)
            try NFKMLXWeights.apply(oracleWeights(record), to: net)
            NFKMLXDeepSeek.adoptComputeType(net, configuration: configuration)
            let connection = net.layers[1].attentionConnection

            let residual = try probe("residual")
            let (read, write, combine) = connection.weights(residual)
            let expanded = connection.expand(try probe("x"), residual: residual,
                                             write: try probe("post"), combine: try probe("comb"))
            let reduced = connection.reduce(expanded, read: try probe("pre"))
            for (name, mine) in [("pre", read), ("post", write), ("comb", combine),
                                 ("expanded", expanded), ("reduced", reduced)] {
                let reference = try probe(name).asType(.float32).reshaped([-1])
                let ours = mine.asType(.float32).reshaped([-1])
                let gap = abs(ours - reference).max().item(Float.self)
                let differing = (ours .!= reference).asType(.int32).sum().item(Int.self)
                print("SEAM deepseek-v4.1 hyper-connection probe \(bfloat16 ? "bf16" : "float32") \(name): "
                      + "largest gap \(gap), \(differing) of \(ours.size) differ")
                if bfloat16, name == "expanded" || name == "reduced" {
                    XCTAssertEqual(differing, 0, "the bf16 \(name) stream is the release's, bit for bit")
                } else {
                    XCTAssertLessThan(gap, 1e-5, "the \(name) \(bfloat16 ? "bf16" : "float32") step")
                }
            }
        }
    }

    // The compressor, at both ratios the release uses. Layer 2 pools two positions into one and
    // needs the gate that makes them compete; layer 4 pools one, which the reference makes a plain
    // projection with no gate at all. The record's own `hc.ffn.pre` from the layer before is the
    // read weight each layer collapses with, which is the pipeline this port now reproduces.
    func testTheCompressorMatchesTheReferenceAtBothRatios() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_DEEPSEEK_V41"] else {
            throw XCTSkip("set IK_PARITY_DEEPSEEK_V41 (Tools/reference-parity deepseek_v41)")
        }
        let record = try MLX.loadArrays(url: URL(fileURLWithPath: path))
        let net = oracleNet()
        try NFKMLXWeights.apply(oracleWeights(record), to: net)

        for layer in [2, 4] {
            let state = try XCTUnwrap(record["hidden.\(layer)"]).expandedDimensions(axis: 0)
            let read = try XCTUnwrap(record["seam.hc.ffn.pre.\(layer - 1)"]).expandedDimensions(axis: 0)
            let block = net.layers[layer]
            let reduced = block.attentionConnection.reduce(state, read: read)
            let compressor = try XCTUnwrap(block.attention.compressor)
            // Localize first: a pooled latent that disagrees is indistinguishable from a block
            // that handed the compressor the wrong input.
            let input = block.attentionNorm(reduced)
            let referenceInput = try XCTUnwrap(record["seam.compressor.in.\(layer)"])
            eval(input)
            let inputSimilarity = cosine(
                input.reshaped([-1]).asArray(Float.self).map(Double.init),
                referenceInput.reshaped([-1]).asArray(Float.self).map(Double.init))
            print("SEAM deepseek-v4.1 compressor input layer \(layer): cosine \(inputSimilarity)")
            XCTAssertGreaterThan(inputSimilarity, 0.9999, "the block hands it the same input")

            let mine = try XCTUnwrap(compressor(input))
            eval(mine)
            let reference = try XCTUnwrap(record["seam.compressor.\(layer)"])
            XCTAssertEqual(mine.shape.dropFirst().map { $0 }, reference.shape,
                           "layer \(layer) pools to the same number of positions")
            let similarity = cosine(mine.reshaped([-1]).asArray(Float.self).map(Double.init),
                                    reference.reshaped([-1]).asArray(Float.self).map(Double.init))
            print("SEAM deepseek-v4.1 compressor layer \(layer) "
                  + "(ratio \(oracleShaped.compressRatio(of: layer))): cosine \(similarity)")
            XCTAssertGreaterThan(similarity, 0.9999, "the compressed latent matches the reference")
        }
        XCTAssertNil(net.layers[3].attention.compressor, "a non-source layer owns no compressor")
        XCTAssertNil(try XCTUnwrap(net.layers[4].attention.compressor).scoreProjection,
                     "a compressor that pools one position per group carries no gate")
    }


    // The block size the dequantizer uses and the scale shapes the release actually stores have to
    // agree, and until this test they did not: the decoder declared 128 for every version while
    // V4.1 states 32, which reads a quarter of the scale grid and decodes wrong values with no
    // error. The structural test could not catch it — it checks the shapes the release HAS, not the
    // shape the block size implies — so the two are compared here directly.
    func testTheBlockSizeAgreesWithTheStoredScaleShapes() throws {
        for (suffix, label) in [("", "deepseek-v4"), ("_PRO", "deepseek-v4-pro"), ("1", "deepseek-v4.1")] {
            guard let (geometry, shapes) = try? released(suffix) else { continue }
            let block = geometry.fp8BlockSize
            var checked = 0
            for (name, entry) in shapes where name.hasSuffix(".scale") {
                guard let scale = entry["shape"] as? [Int], scale.count == 2,
                      let weight = shapes[name.replacingOccurrences(of: ".scale", with: ".weight")],
                      let float = weight["shape"] as? [Int], float.count == 2 else { continue }
                // A square-blocked weight has one scale per block on BOTH axes. A row-wise one has
                // as many scale rows as the weight has, which is the routed-expert and n-gram form.
                guard scale[0] != float[0] else { continue }
                XCTAssertEqual(scale[0], (float[0] + block - 1) / block,
                               "\(label) \(name): \(scale) scales for a \(float) weight is not "
                               + "block \(block) on the first axis")
                XCTAssertEqual(scale[1], (float[1] + block - 1) / block,
                               "\(label) \(name): \(scale) scales for a \(float) weight is not "
                               + "block \(block) on the second axis")
                checked += 1
            }
            print("VALIDATION structure \(label): \(checked) square-blocked weights decode at "
                  + "block \(block), which is what the release stores")
            XCTAssertGreaterThan(checked, 0, "\(label) has square-blocked fp8 weights to check")
        }
    }

    // The router's second correction bias, which only an image span reaches. A text-only parity run
    // cannot see it: every measurement above routes with `bias`, and a port that never built
    // `bias_vl` matches all of them while sending an image's tokens to the wrong experts.
    // The same router and mixture in bf16: both biases stay float32 and the routing is computed in
    // float32, so the choice is the reference's exactly and the mixture's output bit for bit.
    func testTheImageRouterInBFloat16MatchesTheReleasesOwnCode() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_DEEPSEEK_V41_VL_ROUTER_BF16"] else {
            throw XCTSkip("set IK_PARITY_DEEPSEEK_V41_VL_ROUTER_BF16 "
                          + "(Tools/reference-parity deepseek_v41_vl_router_bf16)")
        }
        let record = try MLX.loadArrays(url: URL(fileURLWithPath: path))
        var geometry = NFKMLXDeepSeekConfiguration(
            hiddenSize: 32, layerCount: 1, vocabularySize: 128, rmsEpsilon: 1e-20,
            headCount: 2, headDimensions: 16, ropeHeadDimensions: 4,
            queryLoRARank: 16, outputLoRARank: 8, outputGroups: 1, slidingWindow: 8,
            hyperConnectionCopies: 2, sinkhornIterations: 2,
            indexHeadCount: 2, indexHeadDimensions: 8, indexTopK: 4,
            routedExpertCount: 8, activatedExpertCount: 2, expertIntermediateSize: 16,
            routeScale: 1, swigluLimit: 0, hashLayerCount: 0, routerHasVisionBias: true,
            compressRatios: [0], collapsesThroughLearnedHead: false)
        geometry.computesInBFloat16 = true
        let experts = NFKDeepSeekMoE(geometry, layer: 0)
        try NFKMLXWeights.apply(weights(record, under: ""), to: experts)
        NFKMLXDeepSeek.adoptComputeType(experts, configuration: geometry)

        let hidden = try XCTUnwrap(record["hidden"]).asType(.bfloat16)
        let mask = try XCTUnwrap(record["image_mask"]) .> 0
        let (_, text) = experts.gate(hidden, tokens: nil)
        let (_, vision) = experts.gate(hidden, tokens: nil, images: mask)
        let lifted = experts(hidden.expandedDimensions(axis: 0), tokens: nil, images: mask)
        eval(text, vision, lifted)
        XCTAssertEqual(text.asArray(Int32.self), try XCTUnwrap(record["text_indices"]).asArray(Int32.self),
                       "outside an image span the text bias still selects")
        XCTAssertEqual(vision.asArray(Int32.self), try XCTUnwrap(record["vl_indices"]).asArray(Int32.self),
                       "inside one the vision bias selects")
        let differences = differing(lifted, try XCTUnwrap(record["ffn_vl"]).reshaped(lifted.shape))
        print("VALIDATION PARITY deepseek-v4.1-vl-router-bf16: every choice the reference's, the mixture "
              + "inside an image span \(differences) of \(lifted.size) elements differing (\(lifted.dtype))")
        XCTAssertEqual(lifted.dtype, .bfloat16)
        XCTAssertEqual(differences, 0, "the mixture an image span reaches is the release's, bit for bit")
    }

    func testTheRouterUsesTheVisionBiasInsideAnImageSpan() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_DEEPSEEK_V41_VL_ROUTER"] else {
            throw XCTSkip("set IK_PARITY_DEEPSEEK_V41_VL_ROUTER "
                          + "(Tools/reference-parity deepseek_v41_vl_router)")
        }
        let record = try MLX.loadArrays(url: URL(fileURLWithPath: path))
        let geometry = NFKMLXDeepSeekConfiguration(
            hiddenSize: 32, layerCount: 1, vocabularySize: 128, rmsEpsilon: 1e-20,
            headCount: 2, headDimensions: 16, ropeHeadDimensions: 4,
            queryLoRARank: 16, outputLoRARank: 8, outputGroups: 1, slidingWindow: 8,
            hyperConnectionCopies: 2, sinkhornIterations: 2,
            indexHeadCount: 2, indexHeadDimensions: 8, indexTopK: 4,
            routedExpertCount: 8, activatedExpertCount: 2, expertIntermediateSize: 16,
            routeScale: 1, swigluLimit: 0, hashLayerCount: 0, routerHasVisionBias: true,
            compressRatios: [0], collapsesThroughLearnedHead: false)

        let experts = NFKDeepSeekMoE(geometry, layer: 0)
        try NFKMLXWeights.apply(weights(record, under: ""), to: experts)
        XCTAssertNotNil(experts.gate.visionBias, "a release with a tower carries the second bias")

        let hidden = try XCTUnwrap(record["hidden"])
        let mask = try XCTUnwrap(record["image_mask"]) .> 0
        let (_, text) = experts.gate(hidden, tokens: nil)
        let (_, vision) = experts.gate(hidden, tokens: nil, images: mask)
        eval(text, vision)
        XCTAssertEqual(text.asArray(Int32.self),
                       try XCTUnwrap(record["text_indices"]).asArray(Int32.self),
                       "outside an image span the text bias still selects")
        XCTAssertEqual(vision.asArray(Int32.self),
                       try XCTUnwrap(record["vl_indices"]).asArray(Int32.self),
                       "inside one the vision bias selects")

        // The record is only worth anything if the two biases disagree somewhere.
        let moved = Int(try XCTUnwrap(record["tokens_rerouted"]).item(Int32.self))
        XCTAssertGreaterThan(moved, 0)
        print("VALIDATION structure deepseek-v4.1-vl-router: the image bias reroutes \(moved) of "
              + "\(hidden.shape[0]) tokens, and every one of them matches the reference")

        let lifted = experts(hidden.expandedDimensions(axis: 0), tokens: nil, images: mask)
        eval(lifted)
        let similarity = cosine(lifted.reshaped([-1]).asArray(Float.self).map(Double.init),
                                try XCTUnwrap(record["ffn_vl"]).reshaped([-1])
                                    .asArray(Float.self).map(Double.init))
        print("SEAM deepseek-v4.1 experts inside an image span: cosine \(similarity)")
        XCTAssertGreaterThan(similarity, 0.9999, "the experts an image span reaches match")
    }

    // Writing an image into the text stream must not modify the stream it was given. MLXArray is a
    // class whose subscript setter writes through, so a scatter into the parameter would mutate the
    // caller's embedding and return the same object — and Swift cannot warn about it, because from
    // its side nothing is mutated.
    func testMergingAnImageLeavesTheCallersEmbeddingAlone() throws {
        try requireMLXRuntime()
        let delimiters = NFKMLXDeepSeekImageDelimiters(outputSize: 4)
        let embedded = MLXArray.zeros([1, 5, 4]) + 7
        let before = embedded.asArray(Float.self)
        let aligned = MLXArray.ones([2, 4]) * 3
        let slots: [(position: Int, slot: NFKMLXDeepSeekImageDelimiters.Slot)] =
            [(1, .start), (2, .token(0)), (3, .token(1)), (4, .end)]

        let merged = delimiters.merged(into: embedded, aligned: aligned, slots: slots)
        eval(merged, embedded)
        XCTAssertEqual(embedded.asArray(Float.self), before,
                       "the caller's embedding is untouched")
        XCTAssertNotEqual(merged.asArray(Float.self), before,
                          "the merge wrote something")
    }

    // Real V4.1 bytes, in both of the blockings it ships. The square case is an attention weight;
    // the row-wise case is the n-gram table, whose scale has a row per weight row rather than one
    // per block of rows. Decoding the second as if it were square reads one row's scale for the
    // next thirty-one, which no shape check sees because both scales are correctly shaped.
    func testTheReleasedBytesDecodeInBothBlockings() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_DEEPSEEK_V41_QUANT"] else {
            throw XCTSkip("set IK_PARITY_DEEPSEEK_V41_QUANT (Tools/reference-parity "
                          + "deepseek_v41_quant --checkpoint <repo resolve base>)")
        }
        let record = try MLX.loadArrays(url: URL(fileURLWithPath: path))
        let block = NFKMLXDeepSeekConfiguration.v41Flash.fp8BlockSize
        XCTAssertEqual(block, 32, "the released V4.1 blocks at 32")

        for name in ["square", "rowwise_near", "rowwise_deep"] {
            let bytes = try XCTUnwrap(record["\(name)_bytes"])
            let scale = try XCTUnwrap(record["\(name)_scale_bytes"])
            let expected = try XCTUnwrap(record["\(name)_expected"])
            let decoded = NFKMLXDeepSeekQuantization.dequantizeFP8(bytes: bytes, scaleBytes: scale,
                                                                   blockSize: block)
            eval(decoded)
            XCTAssertEqual(decoded.shape, expected.shape)
            let worst = abs(decoded - expected).max().item(Float.self)
            print("VALIDATION structure deepseek-v4.1-quant \(name): \(bytes.shape) decoded from "
                  + "\(scale.shape) scales, worst absolute difference \(worst)")
            XCTAssertEqual(worst, 0, accuracy: 0,
                           "\(name) decodes exactly: both sides are the same table lookup")
        }
    }

    // MARK: The three decoder mechanisms

    // The n-gram memory's addressing, which no shape check can see. Every multiplier is drawn from
    // NumPy's own generator seeded by the layer id, and every bucket count is a prime picked in
    // order above a floor, so a port that derives either differently reads other rows of the table
    // and is wrong only in the forward pass.
    func testTheNgramAddressingMatchesTheReference() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_DEEPSEEK_V41"] else {
            throw XCTSkip("set IK_PARITY_DEEPSEEK_V41 (Tools/reference-parity deepseek_v41)")
        }
        let record = try MLX.loadArrays(url: URL(fileURLWithPath: path))
        let layout = NFKDeepSeekEngramLayout(oracleShaped)
        XCTAssertEqual(layout.multipliers.flatMap { $0 }.map(Int64.init),
                       try XCTUnwrap(record["engram.multipliers"]).asArray(Int64.self),
                       "the hash multipliers come from NumPy's own stream")
        XCTAssertEqual(layout.primes.flatMap { $0.flatMap { $0 } }.map(Int64.init),
                       try XCTUnwrap(record["engram.primes"]).asArray(Int64.self),
                       "each bucket count is the next unused prime above the floor")
        XCTAssertEqual(layout.offsets.flatMap { $0 }.map(Int64.init),
                       try XCTUnwrap(record["engram.offsets"]).asArray(Int64.self))
        XCTAssertEqual(layout.rowCounts, oracleShaped.engramEmbeddingCounts,
                       "a layer's table is exactly as many rows as its primes sum to")

        let tokens = try XCTUnwrap(record["tokens"]).asArray(Int32.self)
        let hash = NFKDeepSeekNgramHash(oracleShaped, compressedTokens: oracleTokens)
        let mine = hash(MLXArray(tokens).reshaped([1, tokens.count]))
        eval(mine)
        XCTAssertEqual(mine.asType(.int32).reshaped([-1]).asArray(Int32.self),
                       try XCTUnwrap(record["engram.hashes"]).reshaped([-1]).asArray(Int32.self),
                       "the look-back, the rolling hash and the offsets all agree")
        print("VALIDATION structure deepseek-v4.1-engram: "
              + "\(layout.rowCounts) rows derived, \(mine.shape[3]) hashed columns a position")
    }

    // The collapsed id space, which is the one input `makeNet` cannot derive for itself. The
    // reference decides it with a normalizer chain over the release's own tokenizer, and the SIZE it
    // arrives at is what every hash multiplier comes from — so the whole lookup is compared, not the
    // size: two derivations can agree on how many buckets exist and disagree about which ids share
    // one, and only the second kind of disagreement reads the wrong rows.
    func testTheCollapsedTokenMapMatchesTheReference() throws {
        guard let tokenizerPath = config["IK_TOKENIZER_DEEPSEEK_V41"],
              let recordPath = config["IK_PARITY_DEEPSEEK_V41_TOKENS"] else {
            throw XCTSkip("set IK_TOKENIZER_DEEPSEEK_V41 and IK_PARITY_DEEPSEEK_V41_TOKENS "
                          + "(Tools/reference-parity deepseek_v41_tokens --checkpoint <tokenizer.json>)")
        }
        let record = try MLX.loadArrays(url: URL(fileURLWithPath: recordPath))
        let reference = try XCTUnwrap(record["lookup"]).asArray(Int32.self)
        let collapsed = Int(try XCTUnwrap(record["collapsed_size"]).item(Int32.self))
        let vocabulary = Int(try XCTUnwrap(record["vocab_size"]).item(Int32.self))

        let url = URL(fileURLWithPath: tokenizerPath)
        let tokenizer = try XCTUnwrap(NFKMLXDeepSeek.tokenizer(fromTokenizerJSON: url),
                                      "the release's tokenizer.json reads as byte-level BPE")
        // A fragment token is keyed by its spelling, so a spelling table that came back short would
        // fall those ids back onto their replacement characters, which are identical across
        // fragments and would merge them into one bucket.
        let spellings = NFKMLXDeepSeek.tokenSpellings(fromTokenizerJSON: url)
        XCTAssertEqual(spellings.count, vocabulary, "every id has a spelling")
        let space = NFKMLXDeepSeek.collapsedTokenSpace(for: tokenizer, spellings: spellings,
                                                       vocabularySize: vocabulary)
        XCTAssertEqual(space.size, collapsed,
                       "the collapsed space is the size the release states")
        // Compare the PARTITION rather than the numbering. Buckets are numbered in first-occurrence
        // order, so one early disagreement renumbers every id after it and the count of differing
        // numbers says nothing about how many ids actually landed with the wrong neighbours.
        func leaders(_ map: [Int]) -> [Int] {
            var first = [Int: Int]()
            var out = [Int](repeating: 0, count: map.count)
            for (id, bucket) in map.enumerated() {
                if let leader = first[bucket] { out[id] = leader } else {
                    first[bucket] = id
                    out[id] = id
                }
            }
            return out
        }
        func text(_ id: Int) -> String {
            let bytes = tokenizer.bytes(forTokenId: id) ?? Data()
            return String(data: bytes, encoding: .utf8) ?? bytes.map {
                String(format: "\\x%02x", $0)
            }.joined()
        }
        let mine = leaders(space.map)
        let theirs = leaders(reference.map(Int.init))
        let differing = (0 ..< vocabulary).filter { mine[$0] != theirs[$0] }
        print("VALIDATION structure deepseek-v4.1-tokens: \(vocabulary) ids collapse to "
              + "\(space.size), \(differing.count) sharing a bucket the reference does not")
        XCTAssertTrue(differing.isEmpty, "first divergences: " + differing.prefix(4).map {
            "id \($0) \(text($0).debugDescription) groups with \(mine[$0]) "
            + "\(text(mine[$0]).debugDescription), reference groups it with \(theirs[$0]) "
            + "\(text(theirs[$0]).debugDescription)"
        }.joined(separator: "; "))

        // The released size is what the refusal is about, so the configuration that states it now
        // builds, and one handed a map that collapses to the wrong size still does not.
        let map = try NFKMLXDeepSeek.compressedTokens(fromTokenizerJSON: url, in: .v41Flash)
        XCTAssertEqual(map.count, vocabulary)
        XCTAssertNil(NFKMLXDeepSeek.unbuiltMechanism(in: .v41Flash, compressedTokens: map))
        XCTAssertNoThrow(try NFKMLXDeepSeek.makeNet(.v41Flash, compressedTokens: map))
    }

    // The normalizer chain the collapse runs on, at the cases that decide the bucket count: a token
    // that is one space must not be trimmed away into the empty bucket, accents and case must fold,
    // and a run of whitespace must become a single space.
    func testTheCollapseFoldsCaseAccentsAndWhitespace() throws {
        let collapse = NFKMLXDeepSeek.collapsedForm(of:)
        XCTAssertEqual(collapse(" The"), "the")
        XCTAssertEqual(collapse("THE"), "the")
        XCTAssertEqual(collapse("the"), "the")
        XCTAssertEqual(collapse("á"), "a")
        XCTAssertEqual(collapse("Ä"), "a")
        // A SPACING combining mark goes the same way a nonspacing one does, which is what collapses
        // an Indic vowel sign onto its consonant. Reading `StripAccents` as nonspacing-only splits
        // 418 groups the reference merges.
        XCTAssertEqual(collapse("\u{09BE}\u{09B0}"), "\u{09B0}", "a spacing mark is stripped")
        XCTAssertEqual(collapse("\u{09CD}\u{09B0}"), "\u{09B0}", "a nonspacing mark is stripped")
        XCTAssertEqual(collapse(" "), " ", "one space survives the trim")
        XCTAssertEqual(collapse("\t\n  \r"), " ", "a run of whitespace is one space")
        XCTAssertEqual(collapse("a  b"), "a b")
        XCTAssertEqual(collapse(""), "")
    }

    // The released tables are the check the oracle's size cannot be: two layers of 384 million rows
    // each, whose exact counts fall out of the same prime search at the released floor.
    func testTheReleasedNgramTablesAreDerived() throws {
        let layout = NFKDeepSeekEngramLayout(.v41Flash)
        XCTAssertEqual(layout.rowCounts, [384_006_168, 384_016_682])
    }

    // The engram write itself: the gate is a normalized dot product of stream against key, taken
    // per copy rather than jointly, with a signed square root before the sigmoid.
    func testTheNgramMemoryMatchesTheReference() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_DEEPSEEK_V41"] else {
            throw XCTSkip("set IK_PARITY_DEEPSEEK_V41 (Tools/reference-parity deepseek_v41)")
        }
        let record = try MLX.loadArrays(url: URL(fileURLWithPath: path))
        let net = oracleNet()
        try NFKMLXWeights.apply(oracleWeights(record), to: net)
        let tokens = try XCTUnwrap(record["tokens"]).asArray(Int32.self)
        let hashes = try XCTUnwrap(net.ngramHash)(MLXArray(tokens).reshaped([1, tokens.count]))

        for (column, layer) in oracleShaped.engramLayerIDs.enumerated() {
            let engram = try XCTUnwrap(net.layers[layer].engram)
            let input = try XCTUnwrap(record["seam.engram.in.\(layer)"]).expandedDimensions(axis: 0)
            let mine = engram(input, hashes: hashes[0..., 0..., column])
            eval(mine)
            let similarity = cosine(mine.reshaped([-1]).asArray(Float.self).map(Double.init),
                                    try XCTUnwrap(record["seam.engram.\(layer)"])
                                        .reshaped([-1]).asArray(Float.self).map(Double.init))
            print("SEAM deepseek-v4.1 n-gram memory layer \(layer): cosine \(similarity)")
            XCTAssertGreaterThan(similarity, 0.9999, "the gated n-gram write matches")
        }
    }

    // Candidate block selection and the indexer above it. The reference reports the chosen
    // positions as an index list per query and this port as a mask over the same positions, which
    // are the same statement: order does not matter to the attention that consumes it, and an
    // unreachable slot is a -1 there and a false here.
    func testTheCandidateBlocksAndChosenPositionsMatchTheReference() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_DEEPSEEK_V41"] else {
            throw XCTSkip("set IK_PARITY_DEEPSEEK_V41 (Tools/reference-parity deepseek_v41)")
        }
        let record = try MLX.loadArrays(url: URL(fileURLWithPath: path))
        let net = oracleNet()
        try NFKMLXWeights.apply(oracleWeights(record), to: net)
        let tokens = try XCTUnwrap(record["tokens"]).asArray(Int32.self)
        let states = net.hiddenStates(MLXArray(tokens).reshaped([1, tokens.count]))
        XCTAssertEqual(states.count, oracleShaped.layerCount + 1)

        // Rerunning the stack refills the shared runtime, so the candidate mask and the per-layer
        // selections below are the ones this pass produced rather than a previous one's leftovers.
        let shared = NFKDeepSeekSharedAttention()
        var read = try XCTUnwrap(record["seam.hc.ffn.pre.5"]).expandedDimensions(axis: 0) * 0
        read[0..., 0..., 0] = MLXArray(Float(1))
        var hidden = states[0]
        var selections = [Int: MLXArray]()
        for layer in 0 ..< oracleShaped.layerCount {
            if layer > 0, oracleShaped.engramLayerIDs.contains(layer) {
                hidden = states[layer]
            }
            (hidden, read) = net.layers[layer](hidden, mask: nil, tokens: nil,
                                               incoming: read, shared: shared)
            if let selection = shared.selection { selections[layer] = selection }
            if layer == oracleShaped.candidateSourceLayer, let candidates = shared.candidates {
                let mine = MLX.where(candidates, MLXArray(Int32(1)), MLXArray(Int32(0)))
                eval(mine)
                let theirs = try XCTUnwrap(record["seam.candidates"])
                    .reshaped([-1]).asArray(Int32.self)
                let differing = zip(mine.reshaped([-1]).asArray(Int32.self), theirs)
                    .filter { $0 != $1 }.count
                print("VALIDATION structure deepseek-v4.1-candidates: \(candidates.shape) block "
                      + "mask, \(differing) of \(theirs.count) entries differ from the reference")
                XCTAssertEqual(differing, 0,
                               "the same blocks of compressed positions survive level one")
            }
        }

        // The window is the offset the reference adds, so a compressed position t is reported as
        // t + length and an unreachable slot as -1.
        let length = tokens.count
        var differingSelections = 0
        for layer in [2, 3, 4, 5] {
            let selection = try XCTUnwrap(selections[layer])
            let chosen = MLX.where(selection, MLXArray(Int32(1)), MLXArray(Int32(0)))
            eval(chosen)
            let mine = chosen.asArray(Int32.self)
            let positions = selection.shape[2]
            let reference = try XCTUnwrap(record["seam.indexer.\(layer)"]).asArray(Int32.self)
            let perQuery = reference.count / length
            for query in 0 ..< length {
                var expected = [Int32](repeating: 0, count: positions)
                for slot in 0 ..< perQuery {
                    let value = reference[query * perQuery + slot]
                    if value >= 0 { expected[Int(value) - length] = 1 }
                }
                let row = Array(mine[(query * positions) ..< ((query + 1) * positions)])
                if row != expected { differingSelections += 1 }
                XCTAssertEqual(row, expected,
                               "layer \(layer) query \(query) keeps the same positions")
            }
        }
        print("VALIDATION structure deepseek-v4.1-indexer: \(differingSelections) of "
              + "\(oracleShaped.layerCount - 2) x \(length) queries choose different positions")
        XCTAssertEqual(differingSelections, 0,
                       "every layer chooses the compressed positions the reference does")
    }

    // The whole decoder, end to end: the sliding window, the shared compressed cache read by layers
    // that do not own one, both compression ratios, the two-level position choice, the n-gram
    // memory, the pipelined hyper-connection read, and the collapse by identity at the top.
    /// The tiny V4 geometry the release-code oracle builds, Flash's layout or Pro 0813's.
    private func v4ReleaseShaped(pro: Bool) -> NFKMLXDeepSeekConfiguration {
        var geometry = NFKMLXDeepSeekConfiguration(
            hiddenSize: 64, layerCount: 6, vocabularySize: 128, rmsEpsilon: 1e-6,
            headCount: 4, headDimensions: 32, ropeHeadDimensions: 8,
            queryLoRARank: 32, outputLoRARank: 16, outputGroups: 2, slidingWindow: 8,
            ropeTheta: 10_000, compressRopeTheta: 40_000,
            hyperConnectionCopies: 3, sinkhornIterations: 4,
            indexHeadCount: 4, indexHeadDimensions: 32, indexTopK: 3,
            routedExpertCount: 8, activatedExpertCount: 2, expertIntermediateSize: 32,
            routeScale: pro ? 2.5 : 1.5, swigluLimit: 10, hashLayerCount: 1,
            compressRatios: pro ? [8, 8, 4, 8, 4, 0] : [0, 0, 4, 8, 4, 8])
        // What `configuration(fromHuggingFace:)` makes of the releases' `rope_scaling`, scaled down
        // to an original length the 20-token sequence exceeds.
        geometry.ropeScaling = NFKMLXRoPEScaling(kind: .yarn, factor: 4,
                                                 originalMaxPositionEmbeddings: 16,
                                                 betaFast: 32, betaSlow: 1)
        geometry.holdsNormWeightsInFloat32 = true
        return geometry
    }

    // V4 held to its release's OWN `inference/model.py`, which reaches what the transformers
    // comparison cannot: both compression ratios, the overlapping pool, the indexer's own
    // compressor and its selection, the token-table routing, and YaRN on the compressed layers.
    // Each layer runs on the reference's own input, so a disagreement is the layer's.
    func testTheV4DecoderMatchesItsReleasesOwnCode() throws {
        try requireMLXRuntime()
        for (key, pro) in [("IK_PARITY_DEEPSEEK_V4_RELEASE", false), ("IK_PARITY_DEEPSEEK_V4_PRO_RELEASE", true)] {
            guard let path = config[key] else { throw XCTSkip("set \(key) (Tools/reference-parity)") }
            let record = try MLX.loadArrays(url: URL(fileURLWithPath: path))
            let geometry = v4ReleaseShaped(pro: pro)
            let net = try NFKMLXDeepSeek.makeNet(geometry)
            let stored = record.compactMap { key, value -> (String, MLXArray)? in
                key.hasPrefix("w::") ? (NFKMLXDeepSeek.moduleKey(forRelease: String(key.dropFirst(3))), value) : nil
            }
            try NFKMLXWeights.apply(stored, to: net)
            let tokens = try XCTUnwrap(record["tokens"]).asArray(Int32.self)
            let ids = MLXArray(tokens).reshaped([1, tokens.count])
            let label = pro ? "v4-pro" : "v4"

            var perLayer = [String]()
            for (index, block) in net.layers.enumerated() {
                let entering = try XCTUnwrap(record["hidden.\(index)"]).expandedDimensions(axis: 0)
                let output = block(entering, mask: nil, tokens: ids, incoming: nil,
                                   shared: NFKDeepSeekSharedAttention())
                eval(output.state)
                let theirs = try XCTUnwrap(record["hidden.\(index + 1)"])
                perLayer.append(String(format: "%d:%.12f", index,
                    cosine(output.state.reshaped([-1]).asArray(Float.self).map(Double.init),
                           theirs.reshaped([-1]).asArray(Float.self).map(Double.init))))
            }
            let logits = net(ids)
            eval(logits)
            let similarity = cosine(logits.reshaped([-1]).asArray(Float.self).map(Double.init),
                                    try XCTUnwrap(record["output"]).reshaped([-1]).asArray(Float.self)
                                        .map(Double.init))
            print("SEAM deepseek-\(label)-release each layer on the reference's input: "
                  + perLayer.joined(separator: " "))
            print("VALIDATION PARITY deepseek-\(label)-release: logit cosine \(similarity) against the "
                  + "release's own inference/model.py")
            XCTAssertGreaterThan(similarity, 0.99999999, "\(label): the logits")
        }
    }

    // V4 in bf16, against its release's own code built in bf16: what it holds float32 is the
    // reference's constructor's choice, and every layer's stream is the reference's element for
    // element.
    func testV4InBFloat16MatchesItsReleasesOwnCode() throws {
        try requireMLXRuntime()
        for (key, pro) in [("IK_PARITY_DEEPSEEK_V4_RELEASE_BF16", false),
                           ("IK_PARITY_DEEPSEEK_V4_PRO_RELEASE_BF16", true)] {
            guard let path = config[key] else { throw XCTSkip("set \(key) (Tools/reference-parity)") }
            let record = try MLX.loadArrays(url: URL(fileURLWithPath: path))
            var geometry = v4ReleaseShaped(pro: pro)
            geometry.computesInBFloat16 = true
            let label = pro ? "v4-pro" : "v4"

            var ruleMismatches = [String]()
            for (key, value) in record where key.hasPrefix("dtype::") {
                let name = String(key.dropFirst("dtype::".count))
                let reference = value.item(Int32.self) == 1
                if NFKMLXDeepSeek.heldInFloat32(NFKMLXDeepSeek.moduleKey(forRelease: name),
                                                configuration: geometry) != reference {
                    ruleMismatches.append("\(name): reference \(reference)")
                }
            }
            XCTAssertEqual(ruleMismatches.sorted(), [], "\(label): the float32 set is the reference's")

            let net = try NFKMLXDeepSeek.makeNet(geometry)
            let stored = record.compactMap { key, value -> (String, MLXArray)? in
                key.hasPrefix("w::") ? (NFKMLXDeepSeek.moduleKey(forRelease: String(key.dropFirst(3))), value) : nil
            }
            try NFKMLXWeights.apply(stored, to: net)
            NFKMLXDeepSeek.adoptComputeType(net, configuration: geometry)
            let tokens = try XCTUnwrap(record["tokens"]).asArray(Int32.self)
            let ids = MLXArray(tokens).reshaped([1, tokens.count])

            var perLayer = [Int]()
            for (index, block) in net.layers.enumerated() {
                let entering = try XCTUnwrap(record["hidden.\(index)"]).asType(.bfloat16)
                    .expandedDimensions(axis: 0)
                let output = block(entering, mask: nil, tokens: ids, incoming: nil,
                                   shared: NFKDeepSeekSharedAttention())
                eval(output.state)
                perLayer.append(differing(output.state,
                                          try XCTUnwrap(record["hidden.\(index + 1)"]).reshaped(output.state.shape)))
            }
            let logits = net(ids)
            eval(logits)
            let similarity = cosine(logits.asType(.float32).reshaped([-1]).asArray(Float.self).map(Double.init),
                                    try XCTUnwrap(record["output"]).reshaped([-1]).asArray(Float.self)
                                        .map(Double.init))
            print("VALIDATION PARITY deepseek-\(label)-release-bf16: elements differing per layer on the "
                  + "reference's input \(perLayer), logits \(similarity), \(ruleMismatches.count) parameters "
                  + "held at a dtype the reference does not hold")
            XCTAssertEqual(perLayer, [Int](repeating: 0, count: perLayer.count),
                           "\(label): every layer is the release's, bit for bit")
            XCTAssertGreaterThan(similarity, 0.99999999, "\(label): the logits")
        }
    }

    /// The tiny V4 Pro 0813 geometry with its draft stack, as the release-code oracle builds it.
    private var v4ProDraftShaped: NFKMLXDeepSeekConfiguration {
        var geometry = v4ReleaseShaped(pro: true)
        geometry.nextTokenPredictionLayers = 2
        geometry.dsparkBlockSize = 3
        geometry.dsparkTargetLayers = [3, 4, 5]
        geometry.dsparkMarkovRank = 8
        geometry.dsparkNoiseToken = 120
        geometry.draftReadsTargetLayerOutputs = true
        geometry.draftMarkovTableNames = ["markov_w1", "markov_w2"]
        return geometry
    }

    // V4 Pro 0813's draft stack against its release's own code: the target layers' outputs as its
    // context, V4 blocks, the last stage's learned collapse, and 0813's Markov table names. Driven
    // from this port's own decoder, so the connection between the two is measured too.
    func testV4ProDraftStackMatchesItsReleasesOwnCode() throws {
        try requireMLXRuntime()
        for (key, bfloat16) in [("IK_PARITY_DEEPSEEK_V4_PRO_DSPARK", false),
                                ("IK_PARITY_DEEPSEEK_V4_PRO_DSPARK_BF16", true)] {
            guard let path = config[key] else { throw XCTSkip("set \(key) (Tools/reference-parity)") }
            let record = try MLX.loadArrays(url: URL(fileURLWithPath: path))
            var geometry = v4ProDraftShaped
            geometry.computesInBFloat16 = bfloat16
            let label = "v4-pro-dspark" + (bfloat16 ? "-bf16" : "")

            var ruleMismatches = [String]()
            for (key, value) in record where key.hasPrefix("dtype::mtp.") {
                let name = String(key.dropFirst("dtype::".count))
                if NFKMLXDeepSeek.heldInFloat32(NFKMLXDeepSeek.moduleKey(forRelease: name),
                                                configuration: geometry) != (value.item(Int32.self) == 1) {
                    ruleMismatches.append(name)
                }
            }
            XCTAssertEqual(ruleMismatches.sorted(), [], "\(label): the float32 set is the reference's")

            let stack = try XCTUnwrap(NFKMLXDeepSeek.makeDraftStack(geometry), "0813 builds a draft stack")
            let stored = weights(record, under: "stack.").map { (NFKMLXDeepSeek.moduleKey(forRelease: $0.0), $0.1) }
            try NFKMLXWeights.apply(stored.filter { $0.0.hasPrefix("mtp.") }, to: stack)
            NFKMLXDeepSeek.adoptComputeType(stack, configuration: geometry)
            let decoder = try NFKMLXDeepSeek.makeNet(geometry)
            try NFKMLXWeights.apply(stored.filter { !$0.0.hasPrefix("mtp.") }, to: decoder)
            NFKMLXDeepSeek.adoptComputeType(decoder, configuration: geometry)

            var report = [String]()
            var failures = [String]()
            // A stage's attention output passes through two bf16 GEMMs whose float32 sums MLX takes
            // in a different order than torch; where a sum sits on a bf16 boundary it lands one
            // step away. Measured: stage 1's `wo_a` output differs in 1 of 96 elements, where
            // torch's value is the exactly rounded one, and `wo_b` spreads that one step across 9
            // of its outputs. The block's output, the drafted block and every other seam stay
            // exact, so only the attention seams are held to a cosine and their count reported.
            func agree(_ name: String, _ mine: MLXArray, _ key: String, exact: Bool,
                       accumulationOrder: Bool = false) throws {
                let theirs = try XCTUnwrap(record[key], "the record carries \(key)").reshaped(mine.shape)
                eval(mine)
                if exact {
                    let count = differing(mine, theirs)
                    guard accumulationOrder else {
                        report.append("\(name) \(count) differ")
                        if count > 0 { failures.append(name) }
                        return
                    }
                    let similarity = cosine(mine.asType(.float32).reshaped([-1]).asArray(Float.self).map(Double.init),
                                            theirs.asType(mine.dtype).asType(.float32).reshaped([-1])
                                                .asArray(Float.self).map(Double.init))
                    report.append(String(format: "%@ %d differ (cosine %.12f)", name, count, similarity))
                    // One bf16 step in one element moves a cosine by about 1e-7.
                    if similarity < 0.999999 { failures.append(name) }
                } else {
                    let similarity = cosine(mine.asType(.float32).reshaped([-1]).asArray(Float.self).map(Double.init),
                                            theirs.reshaped([-1]).asArray(Float.self).map(Double.init))
                    report.append(String(format: "%@ %.15f", name, similarity))
                    if similarity < 0.9999999999 { failures.append(name) }
                }
            }
            let width: DType = bfloat16 ? .bfloat16 : .float32
            let last = try XCTUnwrap(stack.stages.last)
            try agree("main projection", stack.projectedMainState(
                try XCTUnwrap(record["main_hidden"]).asType(width).expandedDimensions(axis: 0)),
                "main_state", exact: bfloat16)
            let (bias, embedded) = try XCTUnwrap(last.markov)(try XCTUnwrap(record["tokens"]))
            try agree("markov embedding", embedded, "markov_embed", exact: bfloat16)
            try agree("markov bias", bias, "markov_bias", exact: false)

            let committed = try XCTUnwrap(record["loop.committed"])
            let prompt = try XCTUnwrap(record["loop.prompt"])
            let sequence = concatenated([prompt.asType(.int32), committed.asType(.int32)])
                .reshaped([1, prompt.shape[0] + 1])
            let mainStates = try XCTUnwrap(decoder.draftStates(forTokens: sequence))
            try agree("main states from this decoder", mainStates, "loop.main_states", exact: bfloat16)

            let context = NFKDeepSeekDraftContext(mainState: stack.projectedMainState(mainStates),
                                                  offset: mainStates.shape[1])
            let ids = NFKMLXDeepSeekDraftStack.draftTokens(continuing: committed,
                                                           blockSize: geometry.dsparkBlockSize,
                                                           noiseToken: geometry.dsparkNoiseToken)
            var hidden = repeated(decoder.embed(ids).expandedDimensions(axis: 2),
                                  count: geometry.hyperConnectionCopies, axis: 2)
            var read = concatenated(
                [MLXArray.ones([1, geometry.dsparkBlockSize, 1]),
                 MLXArray.zeros([1, geometry.dsparkBlockSize, geometry.hyperConnectionCopies - 1])],
                axis: -1)
            for (index, stage) in stack.stages.enumerated() {
                let (stageRead, _, _) = stage.attentionConnection.weights(hidden)
                let attended = stage.attention.draft(
                    stage.attentionNorm(stage.attentionConnection.reduce(hidden, read: stageRead)),
                    context: context)
                try agree("attention stage \(index)", attended, "seam.draft.attn.\(index)", exact: bfloat16,
                          accumulationOrder: true)
                (hidden, read) = stage(hidden, mask: nil, tokens: nil, incoming: read, drafting: context)
                try agree("block stage \(index)", hidden, "seam.draft.block.\(index)", exact: bfloat16)
            }

            let (tokens, logits, confidence) = stack.propose(continuing: committed,
                                                             mainStates: mainStates, through: decoder)
            eval(tokens, logits, confidence)
            try agree("draft logits", logits, "loop.logits", exact: false)
            try agree("confidence", confidence, "loop.confidence", exact: false)
            let proposed = tokens.reshaped([-1]).asArray(Int32.self)
            let expected = try XCTUnwrap(record["loop.drafted"]).asArray(Int32.self)
            print("VALIDATION PARITY deepseek-\(label): block \(proposed == expected ? "the reference's" : "DIFFERENT \(proposed) vs \(expected)"); "
                  + report.joined(separator: ", ")
                  + "; \(ruleMismatches.count) parameters held at a dtype the reference does not hold")
            XCTAssertEqual(proposed, expected, "\(label): the same block, in the same order")
            XCTAssertEqual(failures, [], "\(label): every seam matches the release's code")
        }
    }

    // V4 and V4 Pro decoding one token at a time through the port's cache, against the release's
    // own buffers: the overlapping compressor's two-window state, the plain compressor's parked
    // group, and the indexer's own compressor and keys all close a group at decode as well as at
    // prefill. The steps follow the reference's tokens, so each step is measured on its own.
    func testV4DecodesAsItsReleaseDoes() throws {
        try requireMLXRuntime()
        for (key, pro, bfloat16) in [("IK_PARITY_DEEPSEEK_V4_RELEASE_DECODE", false, false),
                                     ("IK_PARITY_DEEPSEEK_V4_RELEASE_DECODE_BF16", false, true),
                                     ("IK_PARITY_DEEPSEEK_V4_PRO_RELEASE_DECODE", true, false),
                                     ("IK_PARITY_DEEPSEEK_V4_PRO_RELEASE_DECODE_BF16", true, true)] {
            guard let path = config[key] else { throw XCTSkip("set \(key) (Tools/reference-parity)") }
            let record = try MLX.loadArrays(url: URL(fileURLWithPath: path))
            var geometry = v4ReleaseShaped(pro: pro)
            geometry.computesInBFloat16 = bfloat16
            let net = try NFKMLXDeepSeek.makeNet(geometry)
            let stored = record.compactMap { key, value -> (String, MLXArray)? in
                key.hasPrefix("w::") ? (NFKMLXDeepSeek.moduleKey(forRelease: String(key.dropFirst(3))), value) : nil
            }
            try NFKMLXWeights.apply(stored, to: net)
            NFKMLXDeepSeek.adoptComputeType(net, configuration: geometry)
            let label = (pro ? "v4-pro" : "v4") + (bfloat16 ? "-bf16" : "")
            let prompt = try XCTUnwrap(record["prompt"])
            let expected = try XCTUnwrap(record["generated"]).asArray(Int32.self)
            let cache = NFKMLXDeepSeekCache(geometry)

            var worstLogits = 1.0
            var scoreGaps = [String]()
            var emitted = [Int32]()
            func measure(_ logits: MLXArray, _ tag: String) throws {
                let last = logits[0, -1]
                eval(last)
                worstLogits = min(worstLogits, cosine(
                    last.asType(.float32).asArray(Float.self).map(Double.init),
                    try XCTUnwrap(record["\(tag).logits"]).asArray(Float.self).map(Double.init)))
                emitted.append(argMax(last, axis: -1).item(Int32.self))
                for layer in 0 ..< geometry.layerCount {
                    guard let theirs = record["\(tag).score.\(layer)"],
                          let mine = cache.selectionScores[layer] else { continue }
                    let live = theirs.reshaped([-1]) .> MLXArray(Float(-1e29))
                    let ours = MLX.where(live, mine.reshaped([-1]).asType(.float32), MLXArray(Float(0)))
                    let refs = MLX.where(live, theirs.reshaped([-1]).asType(mine.dtype).asType(.float32),
                                         MLXArray(Float(0)))
                    eval(ours, refs)
                    let gap = bfloat16
                        ? Float((ours .!= refs).asType(.int32).sum().item(Int.self))
                        : abs(ours - refs).max().item(Float.self) / max(abs(refs).max().item(Float.self), 1e-30)
                    if gap > (bfloat16 ? 0 : 1e-4) { scoreGaps.append("\(tag) layer \(layer): \(gap)") }
                }
            }
            try measure(net(prompt.reshaped([1, prompt.shape[0]]), cache: cache), "prefill")
            for step in 0 ..< 5 {
                try measure(net(MLXArray([expected[step]]).reshaped([1, 1]), cache: cache), "step\(step)")
            }
            print("VALIDATION PARITY deepseek-\(label)-release-decode: prefill and 5 steps, logits "
                  + "\(worstLogits) at worst, tokens \(emitted == expected ? "the reference's" : "DIFFERENT \(emitted) vs \(expected)"), "
                  + "index scores \(scoreGaps.isEmpty ? (bfloat16 ? "identical" : "within 1e-4") : scoreGaps.joined(separator: "; "))")
            XCTAssertEqual(emitted, expected, "\(label): every token the reference emitted")
            XCTAssertGreaterThan(worstLogits, bfloat16 ? 0.99999999 : 0.9999999999, "\(label): the logits")
            XCTAssertEqual(scoreGaps, [], "\(label): the indexer scores what the release scores")
        }
    }

    func testTheV41DecoderMatchesTheReferenceEndToEnd() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_DEEPSEEK_V41"] else {
            throw XCTSkip("set IK_PARITY_DEEPSEEK_V41 (Tools/reference-parity deepseek_v41)")
        }
        let record = try MLX.loadArrays(url: URL(fileURLWithPath: path))
        let net = oracleNet()
        try NFKMLXWeights.apply(oracleWeights(record), to: net)
        let tokens = try XCTUnwrap(record["tokens"]).asArray(Int32.self)
        let ids = MLXArray(tokens).reshaped([1, tokens.count])

        // Localize first: a logit that disagrees says nothing about which layer lost it.
        var read: MLXArray?
        let states = net.hiddenStates(ids, finalRead: &read)
        for (index, state) in states.enumerated() {
            guard let reference = record["hidden.\(index)"] else { continue }
            eval(state)
            let similarity = cosine(state.reshaped([-1]).asArray(Float.self).map(Double.init),
                                    reference.reshaped([-1]).asArray(Float.self).map(Double.init))
            print("SEAM deepseek-v4.1 residual stream entering layer \(index): cosine \(similarity)")
            XCTAssertGreaterThan(similarity, 0.9999, "the stream matches at layer \(index)")
        }

        // The sublayers of each block, fed the state the reference itself entered them with, so a
        // stack that drifts says which sublayer drifts rather than only where the drift shows.
        let shared = NFKDeepSeekSharedAttention()
        var carried = concatenated(
            [MLXArray.ones([1, tokens.count, 1]),
             MLXArray.zeros([1, tokens.count, oracleShaped.hyperConnectionCopies - 1])], axis: -1)
        for index in 0 ..< oracleShaped.layerCount {
            let block = net.layers[index]
            let entering = try XCTUnwrap(record["hidden.\(index)"]).expandedDimensions(axis: 0)
            let attended = block.attention(block.attentionNorm(
                block.attentionConnection.reduce(entering, read: carried)), mask: nil, shared: shared)
            eval(attended)
            let similarity = cosine(attended.reshaped([-1]).asArray(Float.self).map(Double.init),
                                    try XCTUnwrap(record["seam.attn.\(index)"])
                                        .reshaped([-1]).asArray(Float.self).map(Double.init))
            print("SEAM deepseek-v4.1 attention layer \(index): cosine \(similarity)")
            XCTAssertGreaterThan(similarity, 0.9999, "attention matches at layer \(index)")
            if let reference = record["seam.ffn.\(index)"] {
                let lifted = block.feedForward(block.feedForwardNorm(
                    block.feedForwardConnection.reduce(
                        block.attentionConnection.expand(
                            attended, residual: entering,
                            write: block.attentionConnection.weights(entering).write,
                            combine: block.attentionConnection.weights(entering).combine),
                        read: block.attentionConnection.weights(entering).read)), tokens: ids)
                eval(lifted)
                let match = cosine(lifted.reshaped([-1]).asArray(Float.self).map(Double.init),
                                   reference.reshaped([-1]).asArray(Float.self).map(Double.init))
                print("SEAM deepseek-v4.1 mixture of experts layer \(index): cosine \(match)")
                XCTAssertGreaterThan(match, 0.9999, "the experts match at layer \(index)")
            }
            carried = block(entering, mask: nil, tokens: ids, incoming: carried, shared: shared).read
        }

        // `finalState` norms what it collapses, because that is what the head reads; the record
        // holds the collapse before the norm, so the reference's own norm goes on it here.
        let collapsed = net.finalState(states.last!, read: read)
        eval(collapsed)
        XCTAssertGreaterThan(
            cosine(collapsed.reshaped([-1]).asArray(Float.self).map(Double.init),
                   net.norm(try XCTUnwrap(record["collapsed"]).expandedDimensions(axis: 0))
                       .reshaped([-1]).asArray(Float.self).map(Double.init)),
            0.9999, "the copies collapse the way the last block's read weight says")

        let logits = net(ids)
        eval(logits)
        let mine = logits.reshaped([-1]).asArray(Float.self).map(Double.init)
        let reference = try XCTUnwrap(record["output"]).reshaped([-1]).asArray(Float.self)
            .map(Double.init)
        let similarity = cosine(mine, reference)
        print("VALIDATION PARITY deepseek-v4.1-oracle: logit cosine \(similarity)")
        XCTAssertGreaterThan(similarity, 0.9999, "the decoder matches the reference end to end")

        let width = oracleShaped.vocabularySize
        for position in 0 ..< tokens.count {
            let row = position * width
            let mineTop = (0 ..< width).max { mine[row + $0] < mine[row + $1] }
            let theirTop = (0 ..< width).max { reference[row + $0] < reference[row + $1] }
            XCTAssertEqual(mineTop, theirTop, "position \(position) predicts the same token")
        }
    }

    // MARK: Decoding one token at a time

    // The whole point of the cache. Prefill sees every position at once; a decode step sees one and
    // has to remember five separate things. The record carries the reference's own buffers after
    // every step, so a wrong logit names the mechanism rather than only the step.
    /// The oracle decoder in bf16, as the release builds it, loaded from a bf16 record.
    private func bfloat16OracleNet(_ record: [String: MLXArray]) throws
        -> (NFKMLXDeepSeekNet, NFKMLXDeepSeekConfiguration) {
        var geometry = oracleShaped
        geometry.indexHeadDimensions = 32
        geometry.computesInBFloat16 = true
        let net = try NFKMLXDeepSeek.makeNet(geometry, compressedTokens: oracleTokens)
        try NFKMLXWeights.apply(oracleWeights(record), to: net)
        NFKMLXDeepSeek.adoptComputeType(net, configuration: geometry)
        return (net, geometry)
    }

    /// How many elements of `mine` differ from `theirs` once `theirs` is rounded to `mine`'s dtype.
    private func differing(_ mine: MLXArray, _ theirs: MLXArray) -> Int {
        let ours = mine.asType(.float32).reshaped([-1])
        let reference = theirs.asType(mine.dtype).asType(.float32).reshaped([-1])
        return (ours .!= reference).asType(.int32).sum().item(Int.self)
    }

    // The same decode in bf16, which is how the release runs it: every buffer a step carries is then
    // held narrow, and each is compared element for element rather than by cosine.
    func testDecodingInBFloat16MatchesTheReleasesOwnCode() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_DEEPSEEK_V41_DECODE_BF16"] else {
            throw XCTSkip("set IK_PARITY_DEEPSEEK_V41_DECODE_BF16 (Tools/reference-parity deepseek_v41_decode_bf16)")
        }
        let record = try MLX.loadArrays(url: URL(fileURLWithPath: path))
        let (net, geometry) = try bfloat16OracleNet(record)
        let prompt = try XCTUnwrap(record["prompt"])
        let expectedTokens = try XCTUnwrap(record["generated"]).asArray(Int32.self)
        let cache = NFKMLXDeepSeekCache(geometry)
        let window = geometry.slidingWindow

        var mismatches = [String]()
        var worstLogits = 1.0
        var worstFloat32Gap: Float = 0
        // A buffer the release holds float32 (a pooling compressor's parked group, which its float32
        // projection writes) is summed in a different order by torch on the CPU and MLX on the GPU,
        // so it is held to a relative gap. Every bf16 buffer is held bit for bit.
        func held(_ mine: MLXArray?, _ key: String, _ label: String,
                  _ shape: (MLXArray) -> MLXArray = { $0 }) throws {
            guard let mine else { return }
            let theirs = shape(try XCTUnwrap(record[key], "the record carries \(key)"))
                .reshaped(mine.shape)
            eval(mine)
            if mine.dtype == .float32 {
                let gap = (abs(mine - theirs).max() / abs(theirs).max()).item(Float.self)
                worstFloat32Gap = max(worstFloat32Gap, gap)
                if gap > 1e-5 { mismatches.append("\(label): relative gap \(gap)") }
                return
            }
            let count = differing(mine, theirs)
            if count > 0 { mismatches.append("\(label): \(count) of \(mine.size) (\(mine.dtype))") }
        }
        func ring(_ seen: Int) -> (MLXArray) -> MLXArray {
            { take($0, MLXArray(((seen - min(window, seen)) ..< seen).map { Int32($0 % window) }),
                    axis: 0) }
        }
        func logitsAgree(_ mine: MLXArray, _ key: String) throws {
            let theirs = try XCTUnwrap(record[key])
            eval(mine)
            worstLogits = min(worstLogits,
                              cosine(mine.asType(.float32).reshaped([-1]).asArray(Float.self).map(Double.init),
                                     theirs.reshaped([-1]).asArray(Float.self).map(Double.init)))
        }
        func buffers(_ tag: String) throws {
            for layer in 0 ..< geometry.layerCount {
                try held(cache.window[layer]?[0, 0], "\(tag).window.\(layer)", "\(tag) window \(layer)",
                         ring(cache.offset))
            }
            for layer in geometry.keyValueSourceLayers ?? [] {
                if let compressed = cache.compressed[layer] {
                    try held(compressed, "\(tag).compress.\(layer)", "\(tag) compressed \(layer)") {
                        $0[0 ..< compressed.dim(2)]
                    }
                }
                if let keys = cache.indexKeys[layer] {
                    try held(keys, "\(tag).index_k.\(layer)", "\(tag) index keys \(layer)") {
                        $0[0 ..< keys.dim(1)]
                    }
                }
                // Only the slots the group has written: an unwritten slot carries a score of
                // negative infinity, so what it holds never reaches the pooled value.
                if let parked = cache.pendingValues[layer]?[0],
                   let scores = record["\(tag).score_state.\(layer)"] {
                    let live = scores.max(axis: -1).asArray(Float.self).enumerated()
                        .filter { $0.element > -1e29 }.map { Int32($0.offset) }
                    try held(take(parked, MLXArray(live), axis: 0), "\(tag).kv_state.\(layer)",
                             "\(tag) parked group \(layer)") { take($0, MLXArray(live), axis: 0) }
                }
            }
            // A prefill chooses per query; one decode step is one query, which is what a mask of
            // kept positions describes.
            guard tag != "prefill" else { return }
            for layer in [2, 3, 4, 5] {
                guard let mask = cache.selected[layer],
                      let theirs = record["\(tag).chosen.\(layer)"] else { continue }
                eval(mask)
                let mine = Set(MLX.where(mask, MLXArray(Int32(1)), MLXArray(Int32(0)))
                    .reshaped([-1]).asArray(Int32.self).enumerated()
                    .filter { $0.element == 1 }.map { $0.offset })
                let expected = Set(theirs.asArray(Int32.self).filter { $0 >= 0 }
                    .map { Int($0) - window })
                if mine != expected { mismatches.append("\(tag) layer \(layer) selection") }
            }
        }

        var logits = net(prompt.reshaped([1, prompt.shape[0]]), cache: cache)
        try logitsAgree(logits[0, -1], "prefill.logits")
        try buffers("prefill")
        var token = MLXArray([expectedTokens[0]])
        var emitted = [expectedTokens[0]]
        for step in 0 ..< 3 {
            logits = net(token.reshaped([1, 1]), cache: cache)
            try logitsAgree(logits[0, -1], "step\(step).logits")
            try buffers("step\(step)")
            token = argMax(logits[0, -1], axis: -1).reshaped([1])
            eval(token)
            emitted.append(token.item(Int32.self))
        }
        print("VALIDATION PARITY deepseek-v4.1-decode-bf16: \(expectedTokens.count - 1) steps, "
              + "logits \(worstLogits) at worst, tokens \(emitted == expectedTokens ? "the reference's" : "DIFFERENT"), "
              + "bf16 buffers differing: \(mismatches.isEmpty ? "none" : mismatches.joined(separator: "; ")), "
              + "float32 parked group within \(worstFloat32Gap) relative")
        XCTAssertEqual(mismatches, [], "every buffer a bf16 step carries is the release's, bit for bit")
        XCTAssertGreaterThan(worstLogits, 0.99999999, "the logits at every step")
        XCTAssertEqual(emitted, expectedTokens, "every token the reference emitted")
    }

    func testDecodingOneTokenAtATimeMatchesTheReference() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_DEEPSEEK_V41_DECODE"] else {
            throw XCTSkip("set IK_PARITY_DEEPSEEK_V41_DECODE (Tools/reference-parity deepseek_v41_decode)")
        }
        let record = try MLX.loadArrays(url: URL(fileURLWithPath: path))
        let net = oracleNet()
        try NFKMLXWeights.apply(oracleWeights(record), to: net)

        let prompt = try XCTUnwrap(record["prompt"])
        let expectedTokens = try XCTUnwrap(record["generated"]).asArray(Int32.self)
        let cache = NFKMLXDeepSeekCache(oracleShaped)

        func compare(_ mine: MLXArray, _ key: String, _ label: String) throws {
            let reference = try XCTUnwrap(record[key], "the record carries \(key)")
            eval(mine)
            let similarity = cosine(mine.reshaped([-1]).asArray(Float.self).map(Double.init),
                                    reference.reshaped([-1]).asArray(Float.self).map(Double.init))
            print("SEAM deepseek-v4.1 decode \(label): cosine \(similarity)")
            XCTAssertGreaterThan(similarity, 0.9999, "\(label) matches the reference")
        }

        // The reference keeps its window as a RING, slot = position % window. This port keeps the
        // same positions in order, so the ring is read back in position order to compare them.
        func ringInPositionOrder(_ ring: MLXArray, seen: Int) -> MLXArray {
            let window = oracleShaped.slidingWindow
            let kept = min(window, seen)
            let rows = ((seen - kept) ..< seen).map { Int32($0 % window) }
            return take(ring, MLXArray(rows), axis: 0)
        }

        var logits = net(prompt.reshaped([1, prompt.shape[0]]), cache: cache)
        try compare(logits[0, -1], "prefill.logits", "prefill logits")
        XCTAssertEqual(cache.offset, prompt.shape[0])
        for layer in 0 ..< oracleShaped.layerCount {
            let mine = try XCTUnwrap(cache.window[layer])[0, 0]
            let theirs = ringInPositionOrder(try XCTUnwrap(record["prefill.window.\(layer)"]),
                                             seen: cache.offset)
            eval(mine)
            XCTAssertEqual(cosine(mine.reshaped([-1]).asArray(Float.self).map(Double.init),
                                  theirs.reshaped([-1]).asArray(Float.self).map(Double.init)) > 0.9999,
                           true, "prefill window at layer \(layer)")
        }

        var token = MLXArray([expectedTokens[0]])
        for step in 0 ..< 3 {
            logits = net(token.reshaped([1, 1]), cache: cache)
            try compare(logits[0, -1], "step\(step).logits", "step \(step) logits")
            for layer in 0 ..< oracleShaped.layerCount {
                let mine = try XCTUnwrap(cache.window[layer])[0, 0]
                let theirs = ringInPositionOrder(
                    try XCTUnwrap(record["step\(step).window.\(layer)"]), seen: cache.offset)
                eval(mine)
                XCTAssertGreaterThan(
                    cosine(mine.reshaped([-1]).asArray(Float.self).map(Double.init),
                           theirs.reshaped([-1]).asArray(Float.self).map(Double.init)),
                    0.9999, "step \(step) window at layer \(layer)")
            }
            for layer in oracleShaped.keyValueSourceLayers ?? [] {
                if let held = cache.compressed[layer] {
                    let groups = held.dim(2)
                    let theirs = try XCTUnwrap(record["step\(step).compress.\(layer)"])[0 ..< groups]
                    eval(held)
                    XCTAssertGreaterThan(
                        cosine(held.reshaped([-1]).asArray(Float.self).map(Double.init),
                               theirs.reshaped([-1]).asArray(Float.self).map(Double.init)),
                        0.9999, "step \(step) compressed cache at layer \(layer)")
                }
                // The index keys the same source publishes, which is the half the reference does
                // not republish on a step that emits nothing.
                if let keys = cache.indexKeys[layer] {
                    let groups = keys.dim(1)
                    let theirs = try XCTUnwrap(record["step\(step).index_k.\(layer)"])[0 ..< groups]
                    eval(keys)
                    XCTAssertGreaterThan(
                        cosine(keys.reshaped([-1]).asArray(Float.self).map(Double.init),
                               theirs.reshaped([-1]).asArray(Float.self).map(Double.init)),
                        0.9999, "step \(step) index keys at layer \(layer)")
                }
                // The group a compressor has started and not finished. Every slot is written by the
                // end of the prompt here, so there is no negative infinity left to confuse a cosine.
                if let parked = cache.pendingValues[layer] {
                    let theirs = try XCTUnwrap(record["step\(step).kv_state.\(layer)"])
                    eval(parked)
                    XCTAssertGreaterThan(
                        cosine(parked[0].reshaped([-1]).asArray(Float.self).map(Double.init),
                               theirs.reshaped([-1]).asArray(Float.self).map(Double.init)),
                        0.9999, "step \(step) parked group at layer \(layer)")
                }
            }
            // What each layer CHOSE, which every buffer matching does not settle. The reference
            // reports positions offset by the window it also attends to; a mask is the same claim.
            for layer in [2, 3, 4, 5] {
                guard let mask = cache.selected[layer],
                      let theirs = record["step\(step).chosen.\(layer)"] else { continue }
                let window = oracleShaped.slidingWindow
                eval(mask)
                let mine = MLX.where(mask, MLXArray(Int32(1)), MLXArray(Int32(0)))
                    .reshaped([-1]).asArray(Int32.self)
                let theirPositions = Set(theirs.asArray(Int32.self)
                    .filter { $0 >= 0 }.map { Int($0) - window })
                let minePositions = Set(mine.enumerated().filter { $0.element == 1 }.map { $0.offset })
                // The SCORE the choice was ranked from, which is the stronger claim. These scores
                // live entirely at the float32 noise floor — every one in this configuration is
                // between 1e-6 and 1e-5 — so a wrong score can still rank right by luck on a step
                // or two. Holding the score itself catches a degraded ranking on the step it is
                // degraded, which comparing kept positions does not.
                if let scores = cache.selectionScores[layer],
                   let theirScores = record["step\(step).score.\(layer)"] {
                    eval(scores)
                    let live = zip(scores.reshaped([-1]).asArray(Float.self),
                                   theirScores.reshaped([-1]).asArray(Float.self))
                        .filter { $0.1 > -1e29 }
                    let drift = live.map { abs(Double($0.0) - Double($0.1)) }.max() ?? 0
                    let scale = live.map { abs(Double($0.1)) }.max() ?? 0
                    XCTAssertLessThan(drift, scale * 0.01,
                                      "step \(step) layer \(layer) scores the same positions")
                }
                XCTAssertEqual(minePositions, theirPositions,
                               "step \(step) layer \(layer) keeps the same compressed positions")
            }
            if let history = cache.engramHistory {
                let held = history.dim(1)
                let all = try XCTUnwrap(record["step\(step).engram_ids"]).asArray(Int32.self)
                let seen = cache.offset
                eval(history)
                XCTAssertEqual(history.asType(.int32).reshaped([-1]).asArray(Int32.self),
                               Array(all[(seen - held) ..< seen]),
                               "step \(step) n-gram history is the tail the reference wrote")
            }
            token = argMax(logits[0, -1], axis: -1).reshaped([1])
            eval(token)
            XCTAssertEqual(token.item(Int32.self), expectedTokens[step + 1],
                           "step \(step) emits the token the reference emitted")
        }
        print("VALIDATION PARITY deepseek-v4.1-decode: \(expectedTokens.count - 1) steps, "
              + "every token the reference's")
    }

    // MARK: The draft stack

    /// The configuration the `deepseek_v41_dspark` oracle builds its reference from: the decoder
    /// oracle's shape with two draft stages on top, routing over four experts of their own rather
    /// than the decoder's eight, and a Markov rank that differs from the hidden width on purpose.
    private var draftShapedOracle: NFKMLXDeepSeekConfiguration {
        NFKMLXDeepSeekConfiguration(
            hiddenSize: 64, layerCount: 6, vocabularySize: 256, rmsEpsilon: 1e-20,
            headCount: 4, headDimensions: 32, ropeHeadDimensions: 8,
            queryLoRARank: 32, outputLoRARank: 16, outputGroups: 2, slidingWindow: 8,
            ropeTheta: 10_000, compressRopeTheta: 40_000,
            hyperConnectionCopies: 3, sinkhornIterations: 4,
            indexHeadCount: 8, indexHeadDimensions: 16, indexTopK: 8,
            routedExpertCount: 8, activatedExpertCount: 2, expertIntermediateSize: 32,
            routeScale: 1, swigluLimit: 0,
            hashLayerCount: 0, routerHasVisionBias: false,
            compressRatios: [0, 0, 2, 2, 1, 1], compressorHasPositionBias: false,
            keyValueSourceLayers: [2, 4], indexSourceLayers: [2, 3, 4, 5],
            collapsesThroughLearnedHead: false, indexerDerivesKeysFromCompressor: true,
            normalizesQueryHeads: false, compressedLayersRotateAtCompressedBase: true,
            pipelinesHyperConnectionRead: true,
            nextTokenPredictionLayers: 2,
            dsparkBlockSize: 3, dsparkTargetLayers: [3, 4, 5], dsparkMarkovRank: 8,
            dsparkExpertCount: 4, dsparkActiveExpertCount: 2, dsparkNoiseToken: 250)
    }

    // MARK: The release's own activation round trips

    // MLX has no fp8 or fp4 type, so the round to each grid is spacing arithmetic that has to land
    // exactly where the formats do. These expected values come from torch's float8_e4m3fn and
    // ml_dtypes' float4_e2m1fn, which are independent of this port, and they include the ties that
    // decide the rounding mode: 1.25, 2.5 and 5.0 each sit exactly between two e2m1 values, and 432
    // exactly between two e4m3 ones.
    func testTheGridRoundingIsTheFormatsOwn() throws {
        try requireMLXRuntime()
        let fourBit = NFKMLXDeepSeekQuantization.roundedE2M1(
            MLXArray([0.25, 0.75, 2.5, 5.0, 5.5, -2.5, 1.25] as [Float]))
        XCTAssertEqual(fourBit.asArray(Float.self), [0, 1, 2, 4, 6, -2, 1],
                       "e2m1 rounds to nearest, ties to the even mantissa")
        let eightBit = NFKMLXDeepSeekQuantization.roundedE4M3(
            MLXArray([432, 424, 0.00292, 1.0625] as [Float]))
        XCTAssertEqual(eightBit.asArray(Float.self), [448, 416, 0.001953125, 1],
                       "e4m3 rounds the same way, through its subnormal step of 2^-9")
        // A power of two is its own ceiling; one ulp above it is the next.
        let ceiling = NFKMLXDeepSeekQuantization.nextPowerOfTwo(
            MLXArray([0.25, Float(0.25).nextUp, 3] as [Float]))
        XCTAssertEqual(ceiling.asArray(Float.self), [0.25, 0.5, 4])
    }

    /// The oracle's geometry as the release's code ran it with its quantizers on.
    ///
    /// Two things differ from `oracleShaped`, and both describe the reference rather than tune the
    /// port. The index heads are 32 wide because the kernel asserts that a row divides into its fp4
    /// blocks. And the fp8 block is 32 because V4.1's `inference/model.py` holds ONE global for the
    /// weight blocks and the activation blocks alike; `oracleShaped` keeps V4's 128 because its
    /// weights are floats and the block never mattered until the activations were rounded by it.
    private var quantizedShaped: NFKMLXDeepSeekConfiguration {
        var c = oracleShaped
        c.indexHeadDimensions = 32
        c.fp8BlockSize = 32
        c.quantizesActivations = true
        return c
    }

    // The mode's whole claim: with the round trips on, the port computes what the release's own
    // `inference/model.py` computes with its quantizers running. The record comes from that code with
    // the kernel shim's quantizers made real rather than neutral.
    //
    // The same weights run WITHOUT the round trips are measured against the same record too, which
    // is what shows the mode is doing the work: a port that ignored the flag would match the model
    // and not the release's code, and would score lower here than a port that honored it.
    func testQuantizedActivationsMatchTheReleasesOwnCode() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_DEEPSEEK_V41_QUANTIZED"] else {
            throw XCTSkip("set IK_PARITY_DEEPSEEK_V41_QUANTIZED "
                          + "(Tools/reference-parity deepseek_v41_quantized)")
        }
        let record = try MLX.loadArrays(url: URL(fileURLWithPath: path))
        let tokens = try XCTUnwrap(record["tokens"]).asArray(Int32.self)
        let ids = MLXArray(tokens).reshaped([1, tokens.count])
        let reference = try XCTUnwrap(record["output"]).reshaped([-1]).asArray(Float.self)
            .map(Double.init)

        XCTAssertNil(NFKMLXDeepSeek.unroundableWidth(in: quantizedShaped),
                     "the configuration divides into the quantizers' blocks")
        func scored(_ geometry: NFKMLXDeepSeekConfiguration) throws -> (Double, [MLXArray]) {
            // Through `makeNet`, so a configuration the quantizers cannot divide throws here rather
            // than trapping mid-forward, which would take every other test's result with it.
            let net = try NFKMLXDeepSeek.makeNet(geometry, compressedTokens: oracleTokens)
            try NFKMLXWeights.apply(oracleWeights(record), to: net)
            let logits = net(ids)
            eval(logits)
            return (cosine(logits.reshaped([-1]).asArray(Float.self).map(Double.init), reference),
                    net.hiddenStates(ids))
        }
        let (rounded, states) = try scored(quantizedShaped)
        var unrounded = quantizedShaped
        unrounded.quantizesActivations = false
        let (plain, _) = try scored(unrounded)

        // Per layer, so a divergence localizes: the stream entering each block.
        for (index, state) in states.enumerated() {
            guard let expected = record["hidden.\(index)"] else { continue }
            eval(state)
            let agreement = cosine(state[0].reshaped([-1]).asArray(Float.self).map(Double.init),
                                   expected.reshaped([-1]).asArray(Float.self).map(Double.init))
            print("SEAM deepseek-v4.1-quantized hidden \(index): cosine \(agreement)")
        }
        // Isolation: each layer fed the reference's OWN entering state, so drift carried in from
        // earlier layers cannot tip a rounding the other way. A layer that matches here and not
        // end to end is compounding; one that fails here has a site wrong.
        let isolated = try NFKMLXDeepSeek.makeNet(quantizedShaped, compressedTokens: oracleTokens)
        try NFKMLXWeights.apply(oracleWeights(record), to: isolated)
        let shared = NFKDeepSeekSharedAttention()
        var carried = concatenated(
            [MLXArray.ones([1, tokens.count, 1]),
             MLXArray.zeros([1, tokens.count, quantizedShaped.hyperConnectionCopies - 1])], axis: -1)
        for index in 0 ..< quantizedShaped.layerCount {
            let block = isolated.layers[index]
            let entering = try XCTUnwrap(record["hidden.\(index)"]).expandedDimensions(axis: 0)
            let attended = block.attention(block.attentionNorm(
                block.attentionConnection.reduce(entering, read: carried)), mask: nil, shared: shared)
            eval(attended)
            let attention = cosine(attended.reshaped([-1]).asArray(Float.self).map(Double.init),
                                   try XCTUnwrap(record["seam.attn.\(index)"])
                                       .reshaped([-1]).asArray(Float.self).map(Double.init))
            let output = block(entering, mask: nil, tokens: ids, incoming: carried, shared: shared)
            carried = output.read
            // The next layer's recorded state already carries its n-gram write, so the whole block
            // compares directly only where the next layer has none.
            var whole = "-"
            if !quantizedShaped.engramLayerIDs.contains(index + 1),
               let next = record["hidden.\(index + 1)"] {
                eval(output.state)
                whole = "\(cosine(output.state.reshaped([-1]).asArray(Float.self).map(Double.init), next.reshaped([-1]).asArray(Float.self).map(Double.init)))"
            }
            // Which compressed positions each query kept, against the reference's own choice.
            var differing = "-"
            if let selection = shared.selection, let chosen = record["seam.indexer.\(index)"] {
                let mine = MLX.where(selection, MLXArray(Int32(1)), MLXArray(Int32(0)))
                    .asArray(Int32.self)
                let theirs = chosen.asArray(Int32.self)
                let positions = selection.shape[2]
                let perQuery = theirs.count / tokens.count
                let scores = try XCTUnwrap(record["seam.score.\(index)"]).asArray(Float.self)
                var ties = [Int]()
                for query in 0 ..< tokens.count {
                    var expected = [Int32](repeating: 0, count: positions)
                    for slot in 0 ..< perQuery where theirs[query * perQuery + slot] >= 0 {
                        expected[Int(theirs[query * perQuery + slot]) - tokens.count] = 1
                    }
                    guard Array(mine[(query * positions) ..< ((query + 1) * positions)])
                        != expected else { continue }
                    // A kept set that differs is only acceptable where the reference's OWN scores
                    // tie at the boundary: then the choice was arbitrary, and the reference's
                    // `topk` leaves tie order unspecified. Anywhere else it is a defect.
                    let row = Array(scores[(query * positions) ..< ((query + 1) * positions)])
                        .filter { $0 > -1e29 }.sorted(by: >)
                    let kept = perQuery
                    let boundary = row.count > kept ? (row[kept - 1], row[kept]) : (0, 1)
                    XCTAssertEqual(boundary.0, boundary.1,
                                   "layer \(index) query \(query) keeps different positions where "
                                   + "the reference's scores do NOT tie: \(boundary)")
                    ties.append(query)
                }
                differing = ties.isEmpty ? "no queries"
                    : "queries \(ties), each an exact tie in the reference's own scores"
                if ties.isEmpty {
                    XCTAssertGreaterThan(attention, 0.99999999,
                                         "layer \(index)'s attention matches where nothing ties")
                }
            }
            print("SEAM deepseek-v4.1-quantized isolated layer \(index): attention \(attention), "
                  + "whole block \(whole), selection differs in \(differing)")
        }
        print("VALIDATION PARITY deepseek-v4.1-quantized: logit cosine \(rounded) with the round "
              + "trips, \(plain) without them, against the release's code with its quantizers on")
        XCTAssertGreaterThan(rounded, plain,
                             "honoring the flag matches the release's code better than ignoring it")
        XCTAssertGreaterThan(rounded, 0.999, "the port computes what the release's code computes")
    }

    // The draft stack rounds two key-values of its own that the decoder does not: the main states
    // through each stage's `wkv`, and the drafted block's. Measured the same way as the decoder:
    // against the release's code with its quantizers on, and with the flag both ways, so the
    // improvement is the stack's two sites doing their work rather than the decoder's four.
    func testTheQuantizedDraftLoopMatchesTheReleasesOwnCode() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_DEEPSEEK_V41_DSPARK_QUANTIZED"] else {
            throw XCTSkip("set IK_PARITY_DEEPSEEK_V41_DSPARK_QUANTIZED "
                          + "(Tools/reference-parity deepseek_v41_dspark_quantized)")
        }
        let record = try MLX.loadArrays(url: URL(fileURLWithPath: path))
        var geometry = draftShapedOracle
        geometry.indexHeadDimensions = 32
        geometry.fp8BlockSize = 32
        geometry.quantizesActivations = true
        XCTAssertNil(NFKMLXDeepSeek.unroundableWidth(in: geometry))

        func scored(_ g: NFKMLXDeepSeekConfiguration) throws
            -> (stages: [Double], logits: Double, drafted: [Int32]) {
            let stack = try XCTUnwrap(NFKMLXDeepSeek.makeDraftStack(g))
            let stored = weights(record, under: "stack.")
                .map { (Self.hyperConnectionNaming($0.0), $0.1) }
            try NFKMLXWeights.apply(stored.filter { $0.0.hasPrefix("mtp.") }, to: stack)
            let decoder = try NFKMLXDeepSeek.makeNet(g)
            try NFKMLXWeights.apply(stored.filter { !$0.0.hasPrefix("mtp.") }, to: decoder)

            let committed = try XCTUnwrap(record["loop.committed"])
            let prompt = try XCTUnwrap(record["loop.prompt"])
            let sequence = concatenated([prompt.asType(.int32), committed.asType(.int32)])
                .reshaped([1, prompt.shape[0] + 1])
            let mainStates = try XCTUnwrap(decoder.draftStates(forTokens: sequence))
            let projected = stack.projectedMainState(mainStates)
            let context = NFKDeepSeekDraftContext(mainState: projected,
                                                  offset: mainStates.shape[1])
            let ids = NFKMLXDeepSeekDraftStack.draftTokens(continuing: committed,
                                                           blockSize: g.dsparkBlockSize,
                                                           noiseToken: g.dsparkNoiseToken)
            var hidden = repeated(decoder.embed(ids).expandedDimensions(axis: 2),
                                  count: g.hyperConnectionCopies, axis: 2)
            var read = concatenated(
                [MLXArray.ones([1, g.dsparkBlockSize, 1]),
                 MLXArray.zeros([1, g.dsparkBlockSize, g.hyperConnectionCopies - 1])], axis: -1)
            var stages = [Double]()
            for (index, stage) in stack.stages.enumerated() {
                let attended = stage.attention.draft(
                    stage.attentionNorm(stage.attentionConnection.reduce(hidden, read: read)),
                    context: context)
                eval(attended)
                stages.append(cosine(
                    attended.reshaped([-1]).asArray(Float.self).map(Double.init),
                    try XCTUnwrap(record["seam.draft.attn.\(index)"])
                        .reshaped([-1]).asArray(Float.self).map(Double.init)))
                (hidden, read) = stage(hidden, mask: nil, tokens: nil, incoming: read,
                                       drafting: context)
            }
            let (tokens, logits, _) = stack.propose(continuing: committed, mainStates: mainStates,
                                                    through: decoder)
            eval(tokens, logits)
            let agreement = cosine(logits.reshaped([-1]).asArray(Float.self).map(Double.init),
                                   try XCTUnwrap(record["loop.logits"])
                                       .reshaped([-1]).asArray(Float.self).map(Double.init))
            return (stages, agreement, tokens.reshaped([-1]).asArray(Int32.self))
        }

        let rounded = try scored(geometry)
        var unrounded = geometry
        unrounded.quantizesActivations = false
        let plain = try scored(unrounded)

        for (index, (mine, without)) in zip(rounded.stages, plain.stages).enumerated() {
            print("SEAM deepseek-v4.1-dspark-quantized attention stage \(index): cosine \(mine) "
                  + "with the round trips, \(without) without")
            XCTAssertGreaterThan(mine, 0.9999, "stage \(index) reads the rounded window and block")
        }
        print("VALIDATION PARITY deepseek-v4.1-dspark-quantized: draft logit cosine "
              + "\(rounded.logits) with the round trips, \(plain.logits) without them")
        XCTAssertEqual(rounded.drafted, try XCTUnwrap(record["loop.drafted"]).asArray(Int32.self),
                       "the same block is proposed, in the same order")
        XCTAssertGreaterThan(rounded.logits, plain.logits,
                             "honoring the flag matches the release's code better than ignoring it")
        XCTAssertGreaterThan(rounded.logits, 0.999)
    }

    // MARK: Computing in bf16

    // The release computes in bf16. What a bf16 decoder holds float32 is the reference's own
    // constructor's choice, and the record carries that choice per parameter, so the rule here is
    // measured against it name by name rather than against a list written down on both sides.
    func testTheBFloat16WeightRuleIsTheReferencesOwn() throws {
        guard let path = config["IK_PARITY_DEEPSEEK_V41_BF16_PLAIN"] else {
            throw XCTSkip("set IK_PARITY_DEEPSEEK_V41_BF16_PLAIN "
                          + "(Tools/reference-parity deepseek_v41_bf16_plain)")
        }
        try requireMLXRuntime()
        let record = try MLX.loadArrays(url: URL(fileURLWithPath: path))
        var geometry = oracleShaped
        geometry.indexHeadDimensions = 32
        geometry.computesInBFloat16 = true
        var mismatched = [String]()
        var float32 = 0
        for (key, value) in record where key.hasPrefix("dtype::") {
            let name = String(key.dropFirst("dtype::".count))
            let reference = value.item(Int32.self) == 1
            let mine = NFKMLXDeepSeek.heldInFloat32(NFKMLXDeepSeek.moduleKey(forRelease: name),
                                                    configuration: geometry)
            if reference { float32 += 1 }
            if mine != reference { mismatched.append("\(name): reference \(reference), port \(mine)") }
        }
        XCTAssertGreaterThan(float32, 0, "the record says which parameters are float32")
        XCTAssertEqual(mismatched.sorted(), [],
                       "the port holds float32 exactly what the reference's constructor does")
        print("VALIDATION structure deepseek-v4.1-bf16-weights: \(float32) float32 parameters, "
              + "\(mismatched.count) disagreeing with the reference's constructor")
    }

    // The mode's claim: a decoder computing in bf16 matches the release's code run in bf16. It
    // cannot be held to the float32 standard, because torch on the CPU and MLX on the GPU round the
    // same bf16 arithmetic in different orders; what it can be held to is being closer to the bf16
    // reference than the float32 decoder is, on the same weights, and to a floor measured here.
    func testComputingInBFloat16MatchesTheReleasesOwnCode() throws {
        guard let plainPath = config["IK_PARITY_DEEPSEEK_V41_BF16_PLAIN"],
              let servedPath = config["IK_PARITY_DEEPSEEK_V41_BF16"] else {
            throw XCTSkip("set IK_PARITY_DEEPSEEK_V41_BF16 and IK_PARITY_DEEPSEEK_V41_BF16_PLAIN "
                          + "(Tools/reference-parity deepseek_v41_bf16, deepseek_v41_bf16_plain)")
        }
        try requireMLXRuntime()

        func scored(_ geometry: NFKMLXDeepSeekConfiguration, against path: String) throws
            -> (logits: Double, argmax: Int, hidden: [Int]) {
            let record = try MLX.loadArrays(url: URL(fileURLWithPath: path))
            let tokens = try XCTUnwrap(record["tokens"]).asArray(Int32.self)
            let ids = MLXArray(tokens).reshaped([1, tokens.count])
            let net = try NFKMLXDeepSeek.makeNet(geometry, compressedTokens: oracleTokens)
            try NFKMLXWeights.apply(oracleWeights(record), to: net)
            NFKMLXDeepSeek.adoptComputeType(net, configuration: geometry)
            let logits = net(ids)
            eval(logits)
            let mine = logits.reshaped([-1]).asArray(Float.self).map(Double.init)
            let reference = try XCTUnwrap(record["output"]).reshaped([-1]).asArray(Float.self)
                .map(Double.init)
            let width = geometry.vocabularySize
            var agree = 0
            for position in 0 ..< tokens.count {
                let row = position * width
                if (0 ..< width).max(by: { mine[row + $0] < mine[row + $1] })
                    == (0 ..< width).max(by: { reference[row + $0] < reference[row + $1] }) {
                    agree += 1
                }
            }
            var hidden = [Int]()
            for (index, state) in net.hiddenStates(ids).enumerated() {
                guard let expected = record["hidden.\(index)"] else { continue }
                let rounded = expected.asType(state.dtype).asType(.float32)
                hidden.append((state.asType(.float32) .!= rounded.reshaped(state.shape))
                    .asType(.int32).sum().item(Int.self))
            }
            return (cosine(mine, reference), agree, hidden)
        }

        var plain = oracleShaped
        plain.indexHeadDimensions = 32
        plain.computesInBFloat16 = true
        var plainFloat = plain
        plainFloat.computesInBFloat16 = false
        var served = quantizedShaped
        served.computesInBFloat16 = true
        var servedFloat = served
        servedFloat.computesInBFloat16 = false

        let results = [("bf16", try scored(plain, against: plainPath),
                        try scored(plainFloat, against: plainPath)),
                       ("bf16 + round trips + narrow GEMMs", try scored(served, against: servedPath),
                        try scored(servedFloat, against: servedPath))]
        for (label, bf16, float) in results {
            print("SEAM deepseek-v4.1-\(label) hidden elements differing, per layer: "
                  + bf16.hidden.map(String.init).joined(separator: " "))
            print("VALIDATION PARITY deepseek-v4.1-\(label): logit cosine \(bf16.logits) computing "
                  + "in bf16 (argmax \(bf16.argmax)/16), \(float.logits) in float32 "
                  + "(argmax \(float.argmax)/16), against the release's code run the same way")
            XCTAssertEqual(bf16.hidden, [Int](repeating: 0, count: bf16.hidden.count),
                           "\(label): every layer's bf16 stream is the release's, bit for bit")
            XCTAssertGreaterThan(bf16.logits, 0.99999999, "\(label): the logits")
            XCTAssertEqual(bf16.argmax, 16, "\(label): every position picks the release's token")
            XCTAssertGreaterThan(bf16.logits, float.logits,
                                 "\(label): computing in bf16 matches the bf16 reference better")
        }
    }

    // A configuration that asks for rounding its widths cannot support is refused at construction.
    // The alternative is a trap mid-forward, which is what the first run of the parity test hit
    // when it built the decoder by hand with V4's 128-wide block against 32-wide heads.
    func testAQuantizingConfigurationItsBlocksCannotDivideIsRefused() throws {
        var c = oracleShaped
        c.quantizesActivations = true            // 32-wide heads against V4's default 128 block
        XCTAssertNotNil(NFKMLXDeepSeek.unroundableWidth(in: c))
        XCTAssertThrowsError(try NFKMLXDeepSeek.makeNet(c, compressedTokens: oracleTokens)) {
            XCTAssertTrue("\($0)".contains("divides into its blocks"), "the refusal says why: \($0)")
        }
        XCTAssertNil(NFKMLXDeepSeek.unroundableWidth(in: .v41Flash.withActivationRounding),
                     "and the released geometry divides cleanly")
    }

    // MARK: Speculative decoding

    // The property that makes speculation worth having: the output is the SAME sequence plain
    // decoding produces. Every kept token is this decoder's own argmax given what precedes it, so a
    // greedy speculative run is token-for-token identical to a greedy plain one. Anything else
    // means the rollback lost something.
    // V4 Pro 0813's draft stack from a directory written in the release's OWN names — the Markov
    // tables as `markov_w1` and `markov_w2`, the last stage's collapse as `hc_head_fn` — so the
    // loader's mapping to module keys is what is measured, not a round trip of the module's names.
    func testAV4ProDraftStackLoadsFromItsReleasesNames() throws {
        try requireMLXRuntime()
        let c = v4ProDraftShaped
        let directory = try temporaryDirectory("draft-v4pro")
        let saved = try XCTUnwrap(NFKMLXDeepSeek.makeDraftStack(c))
        func releaseName(_ key: String) -> String {
            var name = key.replacingOccurrences(of: "markov_head.embed.", with: "markov_head.markov_w1.")
                .replacingOccurrences(of: "markov_head.head.", with: "markov_head.markov_w2.")
            for site in ["attn", "ffn", "head"] {
                for field in ["fn", "base", "scale"] {
                    name = name.replacingOccurrences(of: "hc_\(site).\(field)", with: "hc_\(site)_\(field)")
                }
            }
            return name
        }
        let arrays = Dictionary(uniqueKeysWithValues: saved.parameters().flattened()
            .map { (releaseName($0.0), $0.1) })
        XCTAssertNotNil(arrays["mtp.1.markov_head.markov_w1.weight"], "written in 0813's names")
        XCTAssertNotNil(arrays["mtp.1.hc_head_fn"], "the last stage carries its own collapse")
        try MLX.save(arrays: arrays, url: directory.appendingPathComponent("model.safetensors"))

        let loaded = try XCTUnwrap(NFKMLXDeepSeek.makeDraftStack(c))
        try NFKMLXDeepSeek.loadDraftStack(into: loaded, fromDirectory: directory, configuration: c)
        let mine = Dictionary(uniqueKeysWithValues: saved.parameters().flattened())
        var differing = [String]()
        for (name, value) in Dictionary(uniqueKeysWithValues: loaded.parameters().flattened()) {
            guard let other = mine[name] else { differing.append("\(name): absent"); continue }
            let delta = (value - other).abs().max()
            eval(delta)
            if delta.item(Float.self) != 0 { differing.append(name) }
        }
        XCTAssertEqual(differing.sorted(), [], "every draft parameter reloads under its release name")
        print("VALIDATION runtime deepseek-v4-pro-draft-load: \(arrays.count) draft parameters "
              + "reload exactly from 0813's own names")
    }

    func testSpeculativeDecodingProducesThePlainSequence() throws {
        try requireMLXRuntime()
        // Both draft forms: V4.1's, and V4 Pro 0813's, which reads the target layers' outputs and
        // decodes through V4's overlapping compressor state between rounds.
        for (label, c) in [("v4.1", draftShapedOracle), ("v4-pro", v4ProDraftShaped)] {
            let net = NFKMLXDeepSeekNet(c, compressedTokens: c.engramLayerIDs.isEmpty
                                            ? nil : Array(0 ..< c.vocabularySize))
            let draft = try XCTUnwrap(NFKMLXDeepSeek.makeDraftStack(c))
            var options = NFKMLXGenerationOptions()
            options.maxTokens = 8
            let prompt = [3, 9, 14, 2, 7, 11, 5, 1, 13]

            let plain = net.generate(prompt: prompt, options: options)
            var report = NFKMLXSpeculativeReport()
            let speculative = net.generate(prompt: prompt, draft: draft, options: options,
                                           report: &report)
            XCTAssertEqual(speculative, plain,
                           "\(label): a greedy speculative run is the greedy run, token for token")
            XCTAssertGreaterThan(report.rounds, 0, "\(label): the draft stack was actually consulted")
            XCTAssertEqual(report.proposed, report.rounds * c.dsparkBlockSize,
                           "\(label): every round proposed a whole block")
            print("VALIDATION runtime deepseek-\(label)-dspark-decode: \(speculative.count) tokens, "
                  + "\(report.accepted) of \(report.proposed) proposals accepted over "
                  + "\(report.rounds) rounds, identical to plain decoding")
        }
    }

    // The rollback on its own. A snapshot has to put back what an append replaced, and the sliding
    // window is the piece that proves it: it is a ring, so the speculative positions evicted the
    // oldest ones and trimming the tail would not bring them back.
    func testACacheSnapshotUndoesABlockExactly() throws {
        try requireMLXRuntime()
        let c = draftShapedOracle
        let net = NFKMLXDeepSeekNet(c, compressedTokens: Array(0 ..< c.vocabularySize))
        let prompt = [3, 9, 14, 2, 7, 11, 5, 1, 13, 6, 4]
        XCTAssertGreaterThan(prompt.count, c.slidingWindow, "the ring has already wrapped")

        let cache = NFKMLXDeepSeekCache(c)
        cache.collectsDraftStates = true
        eval(net.prefill(prompt, cache: cache, chunkSize: nil))
        let saved = cache.snapshot()

        // A block runs and is undone; what follows must be what would have followed without it.
        let rejected = MLXArray([21, 37, 44].map(Int32.init)).reshaped([1, 3])
        eval(net(rejected, cache: cache))
        cache.restore(saved)

        let after = net(MLXArray([Int32(19)]).reshaped([1, 1]), cache: cache)
        let fresh = NFKMLXDeepSeekCache(c)
        fresh.collectsDraftStates = true
        eval(net.prefill(prompt, cache: fresh, chunkSize: nil))
        let expected = net(MLXArray([Int32(19)]).reshaped([1, 1]), cache: fresh)
        eval(after, expected)
        XCTAssertEqual((after - expected).abs().max().item(Float.self), 0,
                       "a restored cache is the cache that was saved")
        XCTAssertEqual(cache.offset, fresh.offset, "including the position it is at")
    }

    // Speculation that a release cannot reach is speculation nobody runs. The draft stack has to
    // load from the directory beside the decoder, because the decoder's own load drops `mtp.`.
    func testADraftStackLoadsFromTheReleaseItShipsWith() throws {
        try requireMLXRuntime()
        let c = draftShapedOracle
        let directory = try temporaryDirectory("draft")
        let saved = try XCTUnwrap(NFKMLXDeepSeek.makeDraftStack(c))
        // The stack holds its stages as `mtp`, so its parameter names are already the release's.
        let arrays = Dictionary(uniqueKeysWithValues: saved.parameters().flattened())
        try MLX.save(arrays: arrays, url: directory.appendingPathComponent("model.safetensors"))

        let loaded = try XCTUnwrap(NFKMLXDeepSeek.makeDraftStack(c))
        try NFKMLXDeepSeek.loadDraftStack(into: loaded, fromDirectory: directory, configuration: c)

        let mine = Dictionary(uniqueKeysWithValues: saved.parameters().flattened())
        var differing = [String]()
        for (name, value) in Dictionary(uniqueKeysWithValues: loaded.parameters().flattened()) {
            guard let other = mine[name] else { differing.append("\(name): absent"); continue }
            let delta = (value - other).abs().max()
            eval(delta)
            if delta.item(Float.self) != 0 { differing.append(name) }
        }
        XCTAssertEqual(differing.sorted(), [], "every draft parameter reloads as it was saved")

        // And the loaded stack proposes, which a stack of zeros would also do — so the check that
        // matters is that it proposes something DIFFERENT from an unloaded one.
        let net = NFKMLXDeepSeekNet(c, compressedTokens: Array(0 ..< c.vocabularySize))
        let states = try XCTUnwrap(net.draftStates(forTokens:
            MLXArray([3, 9, 14, 2, 7].map(Int32.init)).reshaped([1, 5])))
        let fromLoaded = loaded.propose(continuing: MLXArray([Int32(11)]), mainStates: states,
                                        through: net).tokens
        let fromEmpty = NFKMLXDeepSeekDraftStack(c).propose(
            continuing: MLXArray([Int32(11)]), mainStates: states, through: net).tokens
        eval(fromLoaded, fromEmpty)
        XCTAssertNotEqual(fromLoaded.asArray(Int32.self), fromEmpty.asArray(Int32.self),
                          "the loaded weights are the ones proposing")
        print("VALIDATION runtime deepseek-v4.1-draft-load: \(arrays.count) draft parameters "
              + "reload exactly and propose from their own weights")
    }

    // The draft loop, which is what the rest of the stack exists for. One call proposes a block of
    // tokens after a committed one: each stage attends to the main stack's key-value over the
    // sliding window and to the whole drafted block WITHOUT a causal mask between the drafts, then
    // the head walks the block so each position is biased by the token chosen for the one before it.
    //
    // The prompt is longer than the window, so the reference's ring buffer has already wrapped and a
    // port that read the whole prefix instead would not line up.
    // The draft stack in bf16, as the release runs it, reading this port's own bf16 decoder. What
    // a stage holds float32 is held to the reference's constructor, and every bf16 stream to the
    // reference's element for element.
    func testTheDraftStackInBFloat16MatchesTheReleasesOwnCode() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_DEEPSEEK_V41_DSPARK_BF16"] else {
            throw XCTSkip("set IK_PARITY_DEEPSEEK_V41_DSPARK_BF16 (Tools/reference-parity deepseek_v41_dspark_bf16)")
        }
        let record = try MLX.loadArrays(url: URL(fileURLWithPath: path))
        var geometry = draftShapedOracle
        geometry.indexHeadDimensions = 32
        geometry.computesInBFloat16 = true

        // `mtp.<n>.embed` and `mtp.<n>.head` alias the decoder's own, which a stage does not hold.
        var ruleMismatches = [String]()
        for (key, value) in record where key.hasPrefix("dtype::mtp.") {
            let name = String(key.dropFirst("dtype::".count))
            if name.range(of: #"^mtp\.\d+\.(embed|head)\.weight$"#, options: .regularExpression) != nil {
                continue
            }
            let reference = value.item(Int32.self) == 1
            let mine = NFKMLXDeepSeek.heldInFloat32(NFKMLXDeepSeek.moduleKey(forRelease: name),
                                                    configuration: geometry)
            if mine != reference { ruleMismatches.append("\(name): reference \(reference), port \(mine)") }
        }
        XCTAssertEqual(ruleMismatches.sorted(), [],
                       "a stage holds float32 exactly what the reference's constructor does")

        let stack = try XCTUnwrap(NFKMLXDeepSeek.makeDraftStack(geometry))
        let stored = weights(record, under: "stack.").map { (Self.hyperConnectionNaming($0.0), $0.1) }
        try NFKMLXWeights.apply(stored.filter { $0.0.hasPrefix("mtp.") }, to: stack)
        NFKMLXDeepSeek.adoptComputeType(stack, configuration: geometry)
        let decoder = NFKMLXDeepSeekNet(geometry)
        try NFKMLXWeights.apply(stored.filter { !$0.0.hasPrefix("mtp.") }, to: decoder)
        NFKMLXDeepSeek.adoptComputeType(decoder, configuration: geometry)

        var differences = [String]()
        func exact(_ label: String, _ mine: MLXArray, _ key: String) throws {
            let theirs = try XCTUnwrap(record[key], "the record carries \(key)").reshaped(mine.shape)
            eval(mine)
            let count = differing(mine, theirs)
            print("SEAM deepseek-v4.1 dspark bf16 \(label): \(count) of \(mine.size) differ (\(mine.dtype))")
            if count > 0 { differences.append(label) }
        }

        let mainHidden = try XCTUnwrap(record["main_hidden"]).asType(.bfloat16).expandedDimensions(axis: 0)
        try exact("main projection on its own input", stack.projectedMainState(mainHidden), "main_state")

        let committed = try XCTUnwrap(record["loop.committed"])
        let prompt = try XCTUnwrap(record["loop.prompt"])
        let sequence = concatenated([prompt.asType(.int32), committed.asType(.int32)])
            .reshaped([1, prompt.shape[0] + 1])
        let mainStates = try XCTUnwrap(decoder.draftStates(forTokens: sequence))
        try exact("main states from this decoder", mainStates, "loop.main_states")

        let context = NFKDeepSeekDraftContext(mainState: stack.projectedMainState(mainStates),
                                              offset: mainStates.shape[1])
        let ids = NFKMLXDeepSeekDraftStack.draftTokens(continuing: committed,
                                                       blockSize: geometry.dsparkBlockSize,
                                                       noiseToken: geometry.dsparkNoiseToken)
        var hidden = repeated(decoder.embed(ids).expandedDimensions(axis: 2),
                              count: geometry.hyperConnectionCopies, axis: 2)
        var read = concatenated(
            [MLXArray.ones([1, geometry.dsparkBlockSize, 1]),
             MLXArray.zeros([1, geometry.dsparkBlockSize, geometry.hyperConnectionCopies - 1])],
            axis: -1)
        for (index, stage) in stack.stages.enumerated() {
            let attended = stage.attention.draft(
                stage.attentionNorm(stage.attentionConnection.reduce(hidden, read: read)),
                context: context)
            try exact("attention stage \(index)", attended, "seam.draft.attn.\(index)")
            (hidden, read) = stage(hidden, mask: nil, tokens: nil, incoming: read, drafting: context)
        }

        let (tokens, logits, confidence) = stack.propose(continuing: committed,
                                                         mainStates: mainStates, through: decoder)
        eval(tokens, logits, confidence)
        var agreement = [String: Double]()
        for (name, mine, key) in [("logits", logits, "loop.logits"),
                                  ("confidence", confidence, "loop.confidence")] {
            agreement[name] = cosine(mine.asType(.float32).reshaped([-1]).asArray(Float.self).map(Double.init),
                                     try XCTUnwrap(record[key]).reshaped([-1]).asArray(Float.self)
                                        .map(Double.init))
        }
        let proposed = tokens.reshaped([-1]).asArray(Int32.self)
        let expected = try XCTUnwrap(record["loop.drafted"]).asArray(Int32.self)
        print("VALIDATION PARITY deepseek-v4.1-dspark-bf16: block \(proposed == expected ? "the reference's" : "DIFFERENT"), "
              + "bf16 streams differing: \(differences.isEmpty ? "none" : differences.joined(separator: ", ")), "
              + "draft logits \(agreement["logits"]!), confidence \(agreement["confidence"]!), "
              + "\(ruleMismatches.count) parameters held at a dtype the reference does not hold")
        XCTAssertEqual(differences, [], "every bf16 stream in the draft pass is the release's, bit for bit")
        XCTAssertEqual(proposed, expected, "the same block is proposed, in the same order")
        XCTAssertGreaterThan(agreement["logits"]!, 0.99999999, "the draft logits")
        XCTAssertGreaterThan(agreement["confidence"]!, 0.99999999, "the confidence")
    }

    func testTheDraftLoopMatchesTheReference() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_DEEPSEEK_V41_DSPARK"] else {
            throw XCTSkip("set IK_PARITY_DEEPSEEK_V41_DSPARK (Tools/reference-parity deepseek_v41_dspark)")
        }
        let record = try MLX.loadArrays(url: URL(fileURLWithPath: path))
        let geometry = draftShapedOracle
        let stack = try XCTUnwrap(NFKMLXDeepSeek.makeDraftStack(geometry))
        // The stack's own parameters only: `embed` and `head` belong to the decoder, which the
        // reference shares by aliasing the modules rather than by copying them.
        // The record now carries the whole model, so the decoder the draft stack reads is this
        // port's own rather than the reference's recorded states.
        let stored = weights(record, under: "stack.")
            .map { (Self.hyperConnectionNaming($0.0), $0.1) }
        try NFKMLXWeights.apply(stored.filter { $0.0.hasPrefix("mtp.") }, to: stack)
        let decoder = NFKMLXDeepSeekNet(geometry)
        try NFKMLXWeights.apply(stored.filter { !$0.0.hasPrefix("mtp.") }, to: decoder)

        let committed = try XCTUnwrap(record["loop.committed"])
        let prompt = try XCTUnwrap(record["loop.prompt"])
        let reference = try XCTUnwrap(record["loop.main_states"]).expandedDimensions(axis: 0)
        XCTAssertEqual(reference.shape[1], prompt.shape[0] + 1,
                       "the draft stack reads every committed position, the newest included")

        // The states come from THIS decoder. The reference builds them across a prefill and one
        // decode step; a prefill-only port runs the whole committed sequence at once, which is the
        // same arithmetic because every path into them is causal.
        let sequence = concatenated([prompt.asType(.int32), committed.asType(.int32)])
            .reshaped([1, prompt.shape[0] + 1])
        let mainStates = try XCTUnwrap(decoder.draftStates(forTokens: sequence),
                                       "the decoder reports the states its target layers see")
        eval(mainStates)
        XCTAssertEqual(mainStates.shape, reference.shape)
        let agreement = cosine(mainStates.reshaped([-1]).asArray(Float.self).map(Double.init),
                               reference.reshaped([-1]).asArray(Float.self).map(Double.init))
        print("SEAM deepseek-v4.1 dspark main states from this decoder: cosine \(agreement)")
        XCTAssertGreaterThan(agreement, 0.9999,
                             "the decoder hands the draft stack what the reference handed it")

        // Localize first: a proposal that disagrees says nothing about which stage lost it.
        let projected = try XCTUnwrap(stack.stages[0].mainNorm)(
            try XCTUnwrap(stack.stages[0].mainProjection)(mainStates))
        let context = NFKDeepSeekDraftContext(mainState: projected, offset: mainStates.shape[1])
        let ids = NFKMLXDeepSeekDraftStack.draftTokens(continuing: committed,
                                                       blockSize: geometry.dsparkBlockSize,
                                                       noiseToken: geometry.dsparkNoiseToken)
        var hidden = repeated(decoder.embed(ids).expandedDimensions(axis: 2),
                              count: geometry.hyperConnectionCopies, axis: 2)
        var read = concatenated(
            [MLXArray.ones([1, geometry.dsparkBlockSize, 1]),
             MLXArray.zeros([1, geometry.dsparkBlockSize, geometry.hyperConnectionCopies - 1])],
            axis: -1)
        for (index, stage) in stack.stages.enumerated() {
            let attended = stage.attention.draft(
                stage.attentionNorm(stage.attentionConnection.reduce(hidden, read: read)),
                context: context)
            eval(attended)
            let similarity = cosine(attended.reshaped([-1]).asArray(Float.self).map(Double.init),
                                    try XCTUnwrap(record["seam.draft.attn.\(index)"])
                                        .reshaped([-1]).asArray(Float.self).map(Double.init))
            print("SEAM deepseek-v4.1 dspark attention stage \(index): cosine \(similarity)")
            XCTAssertGreaterThan(similarity, 0.9999,
                                 "stage \(index) reads the same window and the same block")
            (hidden, read) = stage(hidden, mask: nil, tokens: nil, incoming: read, drafting: context)
        }

        let (tokens, logits, confidence) = stack.propose(continuing: committed,
                                                         mainStates: mainStates, through: decoder)
        eval(tokens, logits, confidence)
        XCTAssertEqual(tokens.reshaped([-1]).asArray(Int32.self),
                       try XCTUnwrap(record["loop.drafted"]).asArray(Int32.self),
                       "the same block is proposed, in the same order")
        for (name, mine, key) in [("logits", logits, "loop.logits"),
                                  ("confidence", confidence, "loop.confidence")] {
            let similarity = cosine(mine.reshaped([-1]).asArray(Float.self).map(Double.init),
                                    try XCTUnwrap(record[key]).reshaped([-1]).asArray(Float.self)
                                        .map(Double.init))
            print("SEAM deepseek-v4.1 dspark draft \(name): cosine \(similarity)")
            XCTAssertGreaterThan(similarity, 0.9999, "the draft \(name) match the reference")
        }
        print("VALIDATION PARITY deepseek-v4.1-dspark: \(geometry.dsparkBlockSize) tokens proposed "
              + "from this decoder's own states, argmax exact")
    }

    // The three pieces of DSpark that stand alone, measured at inputs of their own so that a
    // disagreement in the loop above localizes to the walk rather than to a head. The Markov rank
    // differs from the hidden width in the oracle's configuration, so a head that confused the two
    // would not line up.
    func testTheDraftHeadsMatchTheReference() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_DEEPSEEK_V41_DSPARK"] else {
            throw XCTSkip("set IK_PARITY_DEEPSEEK_V41_DSPARK (Tools/reference-parity deepseek_v41_dspark)")
        }
        let record = try MLX.loadArrays(url: URL(fileURLWithPath: path))
        let blockSize = Int(try XCTUnwrap(record["block_size"]).item(Int32.self))
        let noiseToken = Int(try XCTUnwrap(record["noise_token"]).item(Int32.self))

        var geometry = oracleShaped
        geometry.dsparkBlockSize = blockSize
        geometry.dsparkTargetLayers = [3, 4, 5]
        geometry.dsparkMarkovRank = 8
        geometry.dsparkNoiseToken = noiseToken
        let heads = try XCTUnwrap(NFKMLXDeepSeek.makeDraftHeads(geometry))
        try NFKMLXWeights.apply(weights(record, under: "main_proj.").map { ("main_proj." + $0.0, $0.1) }
                                + weights(record, under: "main_norm.").map { ("main_norm." + $0.0, $0.1) },
                                to: heads.input)
        try NFKMLXWeights.apply(weights(record, under: "markov."), to: heads.markov)
        try NFKMLXWeights.apply(weights(record, under: "confidence."), to: heads.confidence)

        func compare(_ name: String, _ mine: MLXArray, _ key: String) throws {
            let reference = try XCTUnwrap(record[key])
            eval(mine)
            let similarity = cosine(mine.reshaped([-1]).asArray(Float.self).map(Double.init),
                                    reference.reshaped([-1]).asArray(Float.self).map(Double.init))
            print("SEAM deepseek-v4.1 dspark \(name): cosine \(similarity)")
            XCTAssertGreaterThan(similarity, 0.9999, "\(name) matches the reference")
        }

        let mainHidden = try XCTUnwrap(record["main_hidden"]).expandedDimensions(axis: 0)
        try compare("main projection", heads.input.mainState(mainHidden), "main_state")

        let tokens = try XCTUnwrap(record["tokens"])
        let (bias, embedded) = heads.markov(tokens)
        try compare("markov embedding", embedded, "markov_embed")
        try compare("markov logit bias", bias, "output")

        let hidden = try XCTUnwrap(record["hidden"]).expandedDimensions(axis: 0)
        try compare("confidence",
                    heads.confidence(hidden, markov: embedded.expandedDimensions(axis: 0)),
                    "confidence")

        // A draft position holds the noise id until it is proposed; only the committed token is real.
        let drafted = heads.input.draftTokens(continuing: MLXArray([Int32(41)]))
        eval(drafted)
        XCTAssertEqual(drafted.asArray(Int32.self),
                       [41] + Array(repeating: Int32(noiseToken), count: blockSize - 1))
    }

    // MARK: The image tower

    // The tower and the aligner, against the release's own `vision.py`. The grid is 5 by 7 and the
    // ratio is 3, so neither axis divides: an aligner that dropped the padding would pass on a grid
    // that happens to divide and fail here.
    // The image tower and aligner in bf16, as the release runs them: its norms float32, its rotary
    // in float32, everything else bf16, and every output held element for element.
    func testTheImageStackInBFloat16MatchesTheReleasesOwnCode() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_DEEPSEEK_V41_VISION_BF16"] else {
            throw XCTSkip("set IK_PARITY_DEEPSEEK_V41_VISION_BF16 (Tools/reference-parity deepseek_v41_vision_bf16)")
        }
        let record = try MLX.loadArrays(url: URL(fileURLWithPath: path))
        var decoder = oracleShaped
        decoder.computesInBFloat16 = true

        var ruleMismatches = [String]()
        for (key, value) in record where key.hasPrefix("dtype::") {
            let name = String(key.dropFirst("dtype::".count))
            let reference = value.item(Int32.self) == 1
            if NFKMLXDeepSeek.heldInFloat32(name, configuration: decoder) != reference {
                ruleMismatches.append("\(name): reference \(reference)")
            }
        }
        XCTAssertEqual(ruleMismatches.sorted(), [],
                       "the stack holds float32 exactly what the reference's constructor does")

        let rows = Int(try XCTUnwrap(record["rows"]).item(Int32.self))
        let columns = Int(try XCTUnwrap(record["columns"]).item(Int32.self))
        let geometry = NFKMLXDeepSeekVisionConfiguration(
            layerCount: 3, hiddenSize: 32, headCount: 4, intermediateSize: 48, patchSize: 4,
            downsampleRatio: 3, outputSize: 64)
        let stack = NFKMLXDeepSeekImageStack(geometry)
        try NFKMLXWeights.apply(weights(record, under: "vision."), to: stack.tower)
        try NFKMLXWeights.apply(weights(record, under: "aligner."), to: stack.aligner)
        NFKMLXDeepSeek.adoptComputeType(stack, configuration: decoder)

        let features = stack.tower(try XCTUnwrap(record["patches"]), rows: rows, columns: columns)
        let aligned = stack.aligner(features, rows: rows, columns: columns)
        eval(features, aligned)
        let featureDifferences = differing(features, try XCTUnwrap(record["features"]).reshaped(features.shape))
        let alignedDifferences = differing(aligned, try XCTUnwrap(record["output"]).reshaped(aligned.shape))
        // The record is taken on torch's MATH backend, its definition of the attention; its CPU
        // flash backend rounds differently in bf16, and this is how far apart the two are.
        let defaultBackend = differing(aligned, try XCTUnwrap(record["default_sdpa.aligned"]).reshaped(aligned.shape))
        print("VALIDATION PARITY deepseek-v4.1-vision-bf16: features \(featureDifferences) of \(features.size) "
              + "differ, aligned \(alignedDifferences) of \(aligned.size) differ (\(aligned.dtype)), "
              + "\(ruleMismatches.count) parameters held at a dtype the reference does not hold; "
              + "torch's default CPU attention backend differs from its definition in \(defaultBackend) aligned elements")
        XCTAssertEqual(features.dtype, .bfloat16, "the tower computes in bf16")
        XCTAssertEqual(featureDifferences, 0, "the patch features are the release's, bit for bit")
        XCTAssertEqual(alignedDifferences, 0, "and so are the aligned tokens")
    }

    func testTheVisionTowerAndAlignerMatchTheReference() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_DEEPSEEK_V41_VISION"] else {
            throw XCTSkip("set IK_PARITY_DEEPSEEK_V41_VISION (Tools/reference-parity deepseek_v41_vision)")
        }
        let record = try MLX.loadArrays(url: URL(fileURLWithPath: path))
        let rows = Int(try XCTUnwrap(record["rows"]).item(Int32.self))
        let columns = Int(try XCTUnwrap(record["columns"]).item(Int32.self))
        let patches = try XCTUnwrap(record["patches"])

        let geometry = NFKMLXDeepSeekVisionConfiguration(
            layerCount: 3, hiddenSize: 32, headCount: 4, intermediateSize: 48, patchSize: 4,
            downsampleRatio: 3, outputSize: 64)
        let tower = NFKMLXDeepSeekVisionNet(geometry)
        let aligner = NFKMLXDeepSeekAligner(geometry)
        try NFKMLXWeights.apply(weights(record, under: "vision."), to: tower)
        try NFKMLXWeights.apply(weights(record, under: "aligner."), to: aligner)

        let features = tower(patches, rows: rows, columns: columns)
        eval(features)
        let referenceFeatures = try XCTUnwrap(record["features"])
        let featureSimilarity = cosine(
            features.reshaped([-1]).asArray(Float.self).map(Double.init),
            referenceFeatures.reshaped([-1]).asArray(Float.self).map(Double.init))
        print("SEAM deepseek-v4.1 vision tower: cosine \(featureSimilarity)")
        XCTAssertGreaterThan(featureSimilarity, 0.9999, "the patch features match the reference")

        let aligned = aligner(features, rows: rows, columns: columns)
        eval(aligned)
        let reference = try XCTUnwrap(record["output"])
        XCTAssertEqual(aligned.shape, reference.shape,
                       "a 5 by 7 grid at ratio 3 pads to 6 by 9 and pools to six tokens")
        let similarity = cosine(aligned.reshaped([-1]).asArray(Float.self).map(Double.init),
                                reference.reshaped([-1]).asArray(Float.self).map(Double.init))
        print("VALIDATION PARITY deepseek-v4.1-vision: aligned cosine \(similarity)")
        XCTAssertGreaterThan(similarity, 0.9999, "the aligned tokens match the reference")
    }

    /// The record's `w::<prefix>…` weights, under the names the module's parameters carry.
    private func weights(_ record: [String: MLXArray], under prefix: String) -> [(String, MLXArray)] {
        record.compactMap { key, value in
            guard key.hasPrefix("w::" + prefix) else { return nil }
            return (String(key.dropFirst(3 + prefix.count)), value)
        }
    }

    // The parameters are enumerated; the modules are not built. A factory that quietly produced a
    // V4-shaped module under V4.1 geometry would load a checkpoint without complaint — MLX adopts a
    // checkpoint's shapes — and be wrong only in the forward pass.
    func testBuildingAnUnimplementedArrangementIsRefused() throws {
        XCTAssertEqual(NFKMLXDeepSeek.unbuiltMechanism(in: .v41Flash),
                       "the n-gram memory's collapsed token map")
        XCTAssertNil(NFKMLXDeepSeek.unbuiltMechanism(in: .v4Flash))
        XCTAssertNil(NFKMLXDeepSeek.unbuiltMechanism(in: .v4Pro))
        XCTAssertThrowsError(try NFKMLXDeepSeek.makeNet(.v41Flash))
        // Handed the map, the same configuration builds: every mechanism it names is implemented.
        let map = Array(0 ..< NFKMLXDeepSeekConfiguration.v41Flash.vocabularySize)
        XCTAssertNil(NFKMLXDeepSeek.unbuiltMechanism(in: .v41Flash, compressedTokens: map))
        // The same holds at the oracle's size, where the map happens to be the identity: the
        // refusal is about being given one, not about how much it collapses.
        XCTAssertNotNil(NFKMLXDeepSeek.unbuiltMechanism(in: oracleShaped))
        XCTAssertNoThrow(try NFKMLXDeepSeek.makeNet(oracleShaped, compressedTokens: oracleTokens))
        // A configuration with no n-gram memory at all needs no map.
        XCTAssertNil(NFKMLXDeepSeek.unbuiltMechanism(in: draftShapedOracle))
    }

    func testTheV41ConfigurationIsRead() throws {
        let (geometry, _) = try released("1")
        XCTAssertEqual(geometry.hiddenSize, 5120)
        XCTAssertEqual(geometry.layerCount, 40)
        XCTAssertEqual(geometry.routedExpertCount, 384)
        XCTAssertEqual(geometry.hashLayerCount, 0, "V4.1 drops the table-routed layers")
        XCTAssertEqual(geometry.engramLayerIDs, [1, 14])
        XCTAssertEqual(geometry.engramEmbeddingCounts, [384_006_168, 384_016_682])
        XCTAssertEqual(geometry.keyValueSourceLayers, [2, 8, 14, 20])
        XCTAssertEqual(geometry.indexSourceLayers, [2, 8, 14, 20, 24, 28, 32, 36])
        XCTAssertEqual(geometry.candidateSourceLayer, 20)
        XCTAssertEqual(geometry.nextTokenPredictionLayers, 3)
        XCTAssertFalse(geometry.collapsesThroughLearnedHead)
        XCTAssertFalse(geometry.compressorHasPositionBias)
        // The declared preset and the release's own configuration have to agree, or one of them is
        // a guess: the preset is what a caller with no config.json builds from.
        let preset = NFKMLXDeepSeekConfiguration.v41Flash
        XCTAssertEqual(NFKMLXDeepSeek.expectedParameters(for: preset).count,
                       NFKMLXDeepSeek.expectedParameters(for: geometry).count)
        XCTAssertEqual(preset.compressRatios, geometry.compressRatios)
        // The release declares bf16 and its own code runs in it, so the decoder does too.
        XCTAssertTrue(geometry.computesInBFloat16, "the release's declared dtype is read")
        XCTAssertEqual(preset.computesInBFloat16, geometry.computesInBFloat16)
    }

    // A layer that pools more than one position needs the gate that makes them compete; a ratio of
    // one is a plain projection and the release ships no gate for it. Reading that backwards would
    // declare a tensor the checkpoint does not have.
    func testOnlyAPoolingCompressorCarriesAGate() throws {
        let (geometry, _) = try released("1")
        let declared = NFKMLXDeepSeek.expectedParameters(for: geometry)
        for layer in try XCTUnwrap(geometry.keyValueSourceLayers) {
            let gate = "layers.\(layer).attn.compressor.wgate.weight"
            XCTAssertEqual(declared[gate] != nil, geometry.compressRatio(of: layer) > 1,
                           "layer \(layer) pools \(geometry.compressRatio(of: layer)) per group")
        }
        XCTAssertEqual(geometry.compressRatio(of: 20), 1, "the candidate source pools one per group")
    }

    private func assertEveryTensorAccounted(release suffix: String, label: String) throws {
        guard let indexPath = config["IK_INDEX_DEEPSEEK_V4" + suffix],
              let data = FileManager.default.contents(atPath: indexPath),
              let index = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let map = index["weight_map"] as? [String: String],
              let configPath = config["IK_CONFIG_DEEPSEEK_V4" + suffix]
        else { throw XCTSkip("set IK_INDEX_DEEPSEEK_V4\(suffix) and IK_CONFIG_DEEPSEEK_V4\(suffix)") }

        let geometry = try NFKMLXDeepSeek.configuration(fromHuggingFace: URL(fileURLWithPath: configPath))
        let declared = Set(NFKMLXDeepSeek.expectedParameters(for: geometry).keys)

        // Deliberately outside this port, each named rather than merely absent. Every release's
        // draft stack is enumerated now, so nothing is left to name.
        func unimplemented(_ key: String) -> Bool { key.contains(".dspark") }

        // A block scale is not a parameter, it is how the weight beside it is stored, and
        // `dequantized(_:shapes:)` consumes it. Accounted for by the weight it decodes.
        func decodesADeclaredWeight(_ key: String) -> Bool {
            key.hasSuffix(".scale")
                && declared.contains(key.replacingOccurrences(of: ".scale", with: ".weight"))
        }

        var unaccounted = [String]()
        for key in map.keys
        where !declared.contains(key) && !unimplemented(key) && !decodesADeclaredWeight(key) {
            unaccounted.append(key)
        }
        let scales = map.keys.filter(decodesADeclaredWeight).count
        let named = map.keys.filter(unimplemented).count
        print("VALIDATION structure \(label): \(scales) block scales decode a declared weight")
        print("VALIDATION structure \(label): \(declared.count) declared, \(named) named as "
              + "unimplemented, \(unaccounted.count) unaccounted")
        XCTAssertTrue(unaccounted.isEmpty, "released tensors this port neither declares nor names:\n"
                      + unaccounted.sorted().prefix(10).joined(separator: "\n"))
    }

    // MARK: It runs

    func testASmallConfigurationRunsEndToEnd() throws {
        try requireMLXRuntime()
        let net = try NFKMLXDeepSeek.makeNet(small)
        let logits = net(MLXArray([Int32(1), 2, 3]).reshaped([1, 3]))
        eval(logits)
        XCTAssertEqual(logits.shape, [1, 3, small.vocabularySize])
        XCTAssertTrue(logits.sum().item(Float.self).isFinite)
    }

    // Routing has to select exactly `activatedExpertCount` experts, and the weights it returns must
    // sum to the route scale — the reference renormalizes before scaling.
    func testTheGateSelectsAndNormalizes() throws {
        try requireMLXRuntime()
        let gate = NFKDeepSeekGate(small, layer: small.hashLayerCount)   // a scored layer
        let hidden = MLXArray.zeros([4, small.hiddenSize]) + 0.3
        let (weights, indices) = gate(hidden, tokens: nil)
        eval(weights, indices)
        XCTAssertEqual(indices.shape, [4, small.activatedExpertCount])
        let sums = weights.sum(axis: -1).asArray(Float.self)
        for total in sums {
            XCTAssertEqual(Double(total), Double(small.routeScale), accuracy: 1e-4,
                           "the selected weights renormalize, then scale")
        }
    }

    // MARK: The conventions the isolation harness taught us to pin

    // DeepSeek rotates ADJACENT channel pairs (the reference's `view_as_complex`), where the Qwen and
    // Gemma decoders here rotate halves. Nothing in a shape distinguishes them, and getting it wrong
    // is invisible until the numbers are compared — which cannot be done for this model, so the
    // convention is asserted directly instead.
    func testTheRotaryPairsAdjacentChannels() throws {
        try requireMLXRuntime()
        let rotary = NFKDeepSeekRotary(width: 4, theta: 10_000)
        // At position 1 the first pair turns by 1 radian-equivalent and the second by a smaller angle.
        let x = MLXArray([Float(1), 0, 1, 0]).reshaped([1, 1, 1, 4])
        let turned = rotary(x, offset: 1)
        eval(turned)
        let v = turned.reshaped([-1]).asArray(Float.self)

        // An interleaved rotation maps (1, 0) to (cos, sin) within EACH pair. A rotate-half one would
        // instead mix channel 0 with channel 2, leaving channel 1 untouched.
        XCTAssertEqual(Double(v[0]), Double(cos(Float(1))), accuracy: 1e-5)
        XCTAssertEqual(Double(v[1]), Double(sin(Float(1))), accuracy: 1e-5)
        XCTAssertNotEqual(Double(v[1]), 0, accuracy: 1e-3, "channel 1 is the partner of channel 0")
    }

    // The output's rotary component is undone on the way out, because the values share their latent
    // with the keys. An inverse that is not exactly the conjugate would leave a residual rotation.
    func testTheInverseRotationUndoesTheForwardOne() throws {
        try requireMLXRuntime()
        let rotary = NFKDeepSeekRotary(width: 8, theta: 10_000)
        let x = MLXArray((0 ..< 8).map { Float($0 + 1) }).reshaped([1, 1, 1, 8])
        let round = rotary(rotary(x, offset: 3), offset: 3, inverse: true)
        eval(round)
        let worst = (round - x).abs().max().item(Float.self)
        XCTAssertEqual(Double(worst), 0, accuracy: 1e-4, "rotating then de-rotating is the identity")
    }

    // The learned per-head sink drains probability mass without contributing a value, so raising it
    // must shrink the attended output rather than reorder it.
    //
    // The sink is mutated through `update(parameters:)`: assigning to a `@ParameterInfo` directly
    // aborts the process with "please call update() on the array rather than setting it", the same
    // family of MLX trap as giving a `@ModuleInfo` a numeric key.
    func testTheAttentionSinkDrainsMass() throws {
        try requireMLXRuntime()
        let configuration = small
        let net = try NFKMLXDeepSeek.makeNet(configuration)
        let attention = net.layers[0].attention
        let x = MLXArray.zeros([1, 4, configuration.hiddenSize]) + 0.2

        let quiet = attention(x, mask: nil)
        eval(quiet)
        let before = quiet.abs().sum().item(Float.self)

        attention.update(parameters: ModuleParameters.unflattened(
            [("attn_sink", MLXArray.zeros([configuration.headCount]) + 8)]))
        let drained = attention(x, mask: nil)
        eval(drained)
        let after = drained.abs().sum().item(Float.self)

        XCTAssertLessThan(after, before, "a stronger sink takes mass from the keys")
    }

    // MARK: The compressor and indexer

    // Compression pools `ratio` consecutive positions into one, so a sequence shortens by that factor
    // and a sequence too short to fill a window compresses to nothing.
    func testTheCompressorPoolsByItsRatio() throws {
        try requireMLXRuntime()
        let configuration = small
        for ratio in [4, 8] {
            let compressor = NFKDeepSeekCompressor(configuration, ratio: ratio,
                                                   headDimensions: configuration.headDimensions)
            let x = MLXArray.zeros([1, ratio * 3, configuration.hiddenSize]) + 0.1
            let pooled = try XCTUnwrap(compressor(x))
            eval(pooled)
            XCTAssertEqual(pooled.shape, [1, 3, configuration.headDimensions],
                           "ratio \(ratio) pools three windows")
            XCTAssertNil(compressor(MLXArray.zeros([1, ratio - 1, configuration.hiddenSize])),
                         "a partial window compresses to nothing")
        }
    }

    // The indexer selects compressed positions, and may only choose windows that are already complete
    // at the querying position — an early token has nothing to look at and chooses nothing at all.
    func testTheIndexerSelectsOnlyCompletedWindows() throws {
        try requireMLXRuntime()
        var configuration = small
        configuration.indexTopK = 2
        let indexer = NFKDeepSeekIndexer(configuration, ratio: 4)
        let length = 16
        let windows = length / 4
        let x = MLXArray.zeros([1, length, configuration.hiddenSize]) + 0.1
        let query = MLXArray.zeros([1, length, configuration.queryLoRARank]) + 0.1

        let chosen = indexer(x, lowRankQuery: query, latent: nil, positions: windows,
                             shared: NFKDeepSeekSharedAttention())
        eval(chosen)
        XCTAssertEqual(chosen.shape, [1, length, windows])
        let values = MLX.where(chosen, MLXArray(Int32(1)), MLXArray(Int32(0))).asArray(Int32.self)

        for token in 0 ..< length {
            let row = Array(values[(token * windows) ..< ((token + 1) * windows)])
            // Position 0 sits inside the first window, so no window is complete for it.
            XCTAssertLessThanOrEqual(row.reduce(0, +), Int32(min(2, (token + 1) / 4)),
                                     "token \(token) keeps at most its budget of complete windows")
            for window in 0 ..< windows where row[window] == 1 {
                XCTAssertLessThan(window, (token + 1) / 4, "token \(token) saw a future window")
            }
        }
    }

    private func worstDifference(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a - b).max().item(Float.self)
    }

    // MARK: Dequantization

    /// The record holds real bytes from the release beside the float weights they decode to.
    ///
    /// @discussion `output` is the fp8 expectation, which is the record's primary result; the 4-bit
    /// pair travels beside it.
    private func quantizationRecord() throws -> [String: MLXArray] {
        guard let path = config["IK_PARITY_DEEPSEEK_QUANT"] else {
            throw XCTSkip("set IK_PARITY_DEEPSEEK_QUANT (Tools/reference-parity deepseek_quant)")
        }
        return try loadArrays(url: URL(fileURLWithPath: path))
    }

    func testTheEightBitDequantizationMatchesTheReference() throws {
        try requireMLXRuntime()
        let record = try quantizationRecord()
        let decoded = NFKMLXDeepSeekQuantization.dequantizeFP8(
            bytes: record["fp8_bytes"]!, scaleBytes: record["fp8_scale_bytes"]!)
        let expected = record["output"]!
        XCTAssertEqual(decoded.shape, expected.shape)
        XCTAssertEqual(worstDifference(decoded, expected), 0, accuracy: 0,
                       "fp8 decode is exact — every value is representable in float32")
    }

    func testTheFourBitDequantizationMatchesTheReference() throws {
        try requireMLXRuntime()
        let record = try quantizationRecord()
        let decoded = NFKMLXDeepSeekQuantization.dequantizeFP4(
            packedBytes: record["fp4_bytes"]!, scaleBytes: record["fp4_scale_bytes"]!)
        let expected = record["fp4_expected"]!
        XCTAssertEqual(decoded.shape, expected.shape)
        XCTAssertEqual(worstDifference(decoded, expected), 0, accuracy: 0,
                       "fp4 decode is exact — eight magnitudes times a power of two")
    }

    /// The 4-bit blocks run along the LAST axis, which the checkpoint's own values corroborate.
    ///
    /// @discussion The reference quantizer clamps a block to ±6 and rounds its scale to the power of
    /// two that puts the block's largest magnitude in `(3, 6]`. So under the right grouping EVERY
    /// block lands in that range, and under a wrong one some do not. Grouping down the rows instead
    /// leaves about 1% of the blocks outside it, which is what makes this a check rather than a
    /// restatement.
    func testTheFourBitBlocksRunAlongTheLastAxis() throws {
        try requireMLXRuntime()
        let record = try quantizationRecord()
        let decoded = NFKMLXDeepSeekQuantization.dequantizeFP4(
            packedBytes: record["fp4_bytes"]!, scaleBytes: record["fp4_scale_bytes"]!)
        let scales = record["fp4_scale_bytes"]!.asType(.int32)
        let blockSize = NFKMLXDeepSeekQuantization.fp4BlockSize
        let blocks = abs(decoded).reshaped([decoded.shape[0], -1, blockSize]).max(axis: -1)
        let scale = MLXArray(NFKMLXDeepSeekQuantization.scaleValues).take(scales.flattened())
            .reshaped(scales.shape)
        let ratio = blocks / scale
        XCTAssertEqual(ratio.max().item(Float.self), 6, accuracy: 1e-6,
                       "no value exceeds the format's own maximum")
        XCTAssertGreaterThan(ratio.min().item(Float.self), 3,
                             "every block's scale is the tightest power of two that holds it")
    }

    /// A byte holds the earlier value in its LOW nibble.
    ///
    /// @discussion No statistic separates the two orders — a byte's pair decodes to the same values
    /// either way, and both stay inside one block — so the checkpoint cannot settle it. This pins the
    /// format's own convention against a hand-encoded byte instead.
    func testTheFourBitNibbleOrderIsTheFormatsOwn() throws {
        try requireMLXRuntime()
        // 0x1 is 0.5 and 0x7 is 6.0, so 0x71 is the pair (0.5, 6.0) in that order.
        let byte = MLXArray([UInt8(0x71)]).reshaped([1, 1])
        let unitScale = MLXArray([UInt8(127)]).reshaped([1, 1])
        let decoded = NFKMLXDeepSeekQuantization.dequantizeFP4(packedBytes: byte,
                                                              scaleBytes: unitScale)
        XCTAssertEqual(decoded.shape, [1, 2])
        XCTAssertEqual(decoded[0, 0].item(Float.self), 0.5, accuracy: 0)
        XCTAssertEqual(decoded[0, 1].item(Float.self), 6, accuracy: 0)
    }

    func testTheScaleFormatIsAnExponentAlone() throws {
        let values = NFKMLXDeepSeekQuantization.scaleValues
        XCTAssertEqual(values[127], 1, accuracy: 0)
        XCTAssertEqual(values[128], 2, accuracy: 0)
        XCTAssertEqual(values[126], 0.5, accuracy: 0)
        XCTAssertTrue(values[255].isNaN, "the all-ones exponent is the format's only NaN")
    }

    // A companion scale is named for the weight it decodes, so only a `.weight` has one. Deriving
    // that name by substring substitution made every other parameter find ITSELF as its scale and
    // decode against its own bytes — silently, in the forward pass alone.
    func testAParameterWithoutAScaleIsNotDecodedAgainstItself() throws {
        try requireMLXRuntime()
        let plain = MLXArray([1, 1, 1, 1].map(Float.init)).reshaped([2, 2])
        let arrays = ["layers.0.hc_attn_fn": plain,
                      "layers.0.engram.q_weight": plain,
                      "layers.0.ffn.gate.bias": plain]
        let decoded = NFKMLXDeepSeek.dequantized(arrays, shapes: [:])
        XCTAssertEqual(Set(decoded.keys), Set(arrays.keys))
        for (name, value) in decoded {
            eval(value)
            XCTAssertEqual(value.asArray(Float.self), [1, 1, 1, 1],
                           "\(name) carries no scale and passes through unchanged")
        }
    }

    // MARK: Generation

    // What the cache is FOR. A decode step reads one token and reconstructs from five buffers what a
    // prefill would have recomputed, so the claim that makes the runtime worth having is that the two
    // agree: generating through the cache produces the same tokens as running the whole growing
    // sequence through the decoder every step. The reference oracle holds three steps against the
    // reference; this holds the loop against the port's own prefill, which is the part a longer run
    // would drift in.
    func testCachedGenerationMatchesRunningTheWholeSequenceEachStep() throws {
        try requireMLXRuntime()
        let net = oracleNet()
        var options = NFKMLXGenerationOptions()
        options.maxTokens = 6
        let prompt = [3, 9, 14, 2, 7, 11, 5, 1, 13, 6, 4]

        let cached = net.generate(prompt: prompt, options: options)

        var tokens = prompt
        var expected = [Int]()
        for _ in 0 ..< options.maxTokens {
            let logits = net(MLXArray(tokens.map(Int32.init)).reshaped([1, tokens.count]))
            let next = logits[0, -1].argMax().item(Int.self)
            expected.append(next)
            tokens.append(next)
        }

        XCTAssertEqual(cached.count, options.maxTokens)
        XCTAssertEqual(cached, expected,
                       "the cached loop and the prefill-only loop produce the same tokens")
        print("VALIDATION runtime deepseek-v4.1-generate: \(cached.count) tokens, "
              + "cached decode equals prefill-only decode")
    }

    // A run stops on an end token rather than producing its full budget.
    func testGenerationStopsOnAStopToken() throws {
        try requireMLXRuntime()
        let net = oracleNet()
        var options = NFKMLXGenerationOptions()
        options.maxTokens = 6
        let prompt = [3, 9, 14, 2, 7, 11, 5, 1, 13, 6, 4]
        let produced = net.generate(prompt: prompt, options: options)
        let first = try XCTUnwrap(produced.first)

        options.stopTokens = [first]
        XCTAssertEqual(net.generate(prompt: prompt, options: options), [],
                       "a run whose first token is a stop token produces nothing")
    }

    // MARK: Loading a release

    func testALoadedDirectoryReproducesTheDecoderItWasSavedFrom() throws {
        try requireMLXRuntime()
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("deepseek-release-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let saved = oracleNet()
        try NFKMLXWeights.save(saved, to: directory.appendingPathComponent("model.safetensors"))

        let loaded = oracleNet()
        try NFKMLXDeepSeek.loadWeights(into: loaded, fromDirectory: directory)

        // Every parameter, not just the logits: a wrong one can hide behind a cosine.
        let mine = Dictionary(uniqueKeysWithValues: saved.parameters().flattened())
        let theirs = Dictionary(uniqueKeysWithValues: loaded.parameters().flattened())
        var differing = [String]()
        for (name, value) in mine {
            guard let other = theirs[name] else { differing.append("\(name): absent"); continue }
            if value.shape != other.shape {
                differing.append("\(name): \(value.shape) vs \(other.shape)"); continue
            }
            let delta = (value - other).abs().max()
            eval(delta)
            if delta.item(Float.self) != 0 { differing.append("\(name): delta \(delta.item(Float.self))") }
        }
        XCTAssertEqual(differing.sorted(), [],
                       "every parameter reloads as it was saved")

        let ids = MLXArray([3, 9, 14, 2, 7].map(Int32.init)).reshaped([1, 5])
        let (before, after) = (saved(ids), loaded(ids))
        eval(before, after)
        XCTAssertEqual(cosine(before.reshaped([-1]).asArray(Float.self).map(Double.init),
                              after.reshaped([-1]).asArray(Float.self).map(Double.init)),
                       1, accuracy: 1e-12,
                       "a loaded decoder is the one that was saved")
    }

    // The refusal that matters most on this architecture: the release stores its weights fp8 and fp4
    // and the modules hold them bf16, so what a load needs is two to four times what the directory
    // measures. A check that read the directory's bytes would under-count it and the process would
    // die instead of reporting.
    func testAReleaseLargerThanTheMachineIsRefusedBeforeAnythingIsRead() throws {
        let flash = NFKMLXDeepSeekConfiguration.v41Flash
        let resident = NFKMLXDeepSeek.residentBytes(for: flash)
        XCTAssertGreaterThan(resident, 1 << 40, "a 763-billion-parameter release exceeds a terabyte")
        print("VALIDATION runtime deepseek-v4.1-fit: decodes to "
              + String(format: "%.2f", Double(resident) / 1_099_511_627_776) + " TiB of bf16 "
              + "parameters, against a working set of "
              + String(format: "%.1f", Double(NFKMLXGPU.recommendedWorkingSetSize) / 1_073_741_824)
              + " GiB")

        XCTAssertThrowsError(try NFKMLXDeepSeek.verifyFits(flash, budget: 512 << 30)) { error in
            XCTAssertTrue("\(error)".contains("GiB"), "the refusal names the size: \(error)")
        }
        // An unknown machine reports no working set, and an unknown machine does not gate a load.
        XCTAssertNoThrow(try NFKMLXDeepSeek.verifyFits(flash, budget: 0))
    }

    // MARK: The image processor and a picture at decode

    /// The oracle's geometry with an image tower beside it.
    private var imageShaped: NFKMLXDeepSeekConfiguration {
        var c = oracleShaped
        c.routerHasVisionBias = true
        c.vision = NFKMLXDeepSeekVisionConfiguration(
            layerCount: 2, hiddenSize: 32, headCount: 4, intermediateSize: 48, patchSize: 4,
            downsampleRatio: 3, outputSize: c.hiddenSize, maximumTokenCount: 64,
            minimumPixels: 16 * 16, maximumWidthHeightRatio: 8, imageTokenID: 200)
        return c
    }

    /// The reference's own token-type codes, which the span layout is compared against.
    private func typeCode(_ slot: NFKMLXDeepSeekImageDelimiters.Slot) -> Int {
        switch slot {
        case .start: return 0
        case .token: return 1
        case .newline: return 2
        case .end: return 3
        }
    }

    // `deepseek_v41_vision` starts from patches, so the step that PRODUCES them was never measured,
    // and that step decides the grid every later shape follows from. The reference's preprocessing
    // is PIL's, which resamples an 8-bit picture in two passes and rounds to 8 bits between them;
    // a float resample of the same coefficients is a different picture by about one part in 255.
    func testTheImageProcessorMatchesTheReference() throws {
        try requireMLXRuntime()
        guard let path = config["IK_PARITY_DEEPSEEK_V41_IMAGE"] else {
            throw XCTSkip("set IK_PARITY_DEEPSEEK_V41_IMAGE "
                          + "(Tools/reference-parity deepseek_v41_image)")
        }
        let record = try MLX.loadArrays(url: URL(fileURLWithPath: path))
        func scalar(_ key: String) throws -> Int {
            Int(try XCTUnwrap(record[key]).item(Int32.self))
        }
        let processor = NFKMLXDeepSeekImageProcessor(
            patchSize: try scalar("patch_size"),
            downsampleRatio: try scalar("downsample_ratio"),
            maximumTokenCount: try scalar("max_n_token"),
            minimumPixels: try scalar("min_pixels"),
            maximumWidthHeightRatio: try scalar("max_wh_ratio"))

        for tag in ["a", "b"] {
            let size = try XCTUnwrap(record["\(tag).size"]).asArray(Int32.self).map(Int.init)
            let grid = try XCTUnwrap(record["\(tag).grid"]).asArray(Int32.self).map(Int.init)
            let plan = processor.plan(width: size[0], height: size[1])
            XCTAssertEqual([plan.patchRows, plan.patchColumns, plan.tokenRows, plan.tokenColumns],
                           grid, "the grid the reference planned for \(tag)")

            let pixels = try XCTUnwrap(record["\(tag).pixels"]).asArray(Float.self)
                .map { UInt8($0.rounded()) }
            let mine = processor.patches(from: pixels, width: size[0], height: size[1], plan: plan)
            let theirs = try XCTUnwrap(record["\(tag).patches"])
            eval(mine)
            XCTAssertEqual(mine.shape, theirs.shape, "\(tag): the patch grid")
            let delta = (mine - theirs).abs().max()
            eval(delta)
            XCTAssertEqual(delta.item(Float.self), 0,
                           "\(tag): the patches are the reference's, byte for byte")

            let types = try XCTUnwrap(record["\(tag).types"]).asArray(Int32.self).map(Int.init)
            XCTAssertEqual(processor.slots(for: plan, startingAt: 0).map { typeCode($0.slot) },
                           types, "\(tag): the span's layout")
            print("VALIDATION parity deepseek-v4.1-image-\(tag): \(size[0])x\(size[1]) plans to "
                  + "\(grid) and its patches match exactly")
        }
    }

    // A picture reaches the decoder as embeddings written over the placeholder's positions, with a
    // mask saying which positions those are. Both are needed: the mask is what the router selects an
    // image token's experts with, and the embeddings are what the positions hold.
    func testAPictureReachesTheDecoderThroughItsSpan() throws {
        try requireMLXRuntime()
        let c = imageShaped
        let net = NFKMLXDeepSeekNet(c, compressedTokens: oracleTokens)
        let stack = NFKMLXDeepSeekImageStack(try XCTUnwrap(c.vision))

        let (width, height) = (21, 13)
        let pixels = (0 ..< (width * height * 3)).map { UInt8(($0 * 37) % 256) }
        let text = [3, 9, 200, 14, 2, 7]
        let (tokens, spans) = try stack.expanded(text, pictures: [(pixels, width, height)])
        let span = try XCTUnwrap(spans.first)
        XCTAssertEqual(tokens.count, text.count - 1 + span.plan.tokenCount,
                       "the placeholder became its whole span")
        XCTAssertEqual(span.slots.count, span.plan.tokenCount)
        XCTAssertEqual(span.aligned.shape,
                       [span.plan.tokenRows * span.plan.tokenColumns, c.hiddenSize],
                       "the aligner pools into the decoder's width")

        let inputs = stack.inputs(for: tokens, spans: spans) { net.embed($0) }
        let ids = MLXArray(tokens.map(Int32.init)).reshaped([1, tokens.count])
        let withPicture = net(ids, images: inputs.images, embeddings: inputs.embeddings)
        let withoutPicture = net(ids)
        eval(withPicture, withoutPicture)
        XCTAssertGreaterThan((withPicture - withoutPicture).abs().max().item(Float.self), 1e-3,
                             "the picture changes what the decoder reads")

        // A span crosses chunk boundaries here, which is the state a chunked prefill could lose.
        var options = NFKMLXGenerationOptions()
        options.maxTokens = 3
        let whole = net.generate(prompt: tokens, embeddings: inputs.embeddings,
                                 images: inputs.images, options: options)
        options.prefillChunkSize = 4
        XCTAssertEqual(net.generate(prompt: tokens, embeddings: inputs.embeddings,
                                    images: inputs.images, options: options), whole,
                       "a chunked prefill carries the span across its boundaries")
        print("VALIDATION runtime deepseek-v4.1-image-decode: a \(width)x\(height) picture becomes "
              + "\(span.plan.tokenCount) positions and generates \(whole.count) tokens")
    }

    // The count has to agree, because a prompt that names two pictures and is handed one would
    // otherwise leave a span of placeholders holding nothing.
    func testAPictureCountThatDisagreesWithThePromptIsRefused() throws {
        try requireMLXRuntime()
        let stack = NFKMLXDeepSeekImageStack(try XCTUnwrap(imageShaped.vision))
        let pixels = [UInt8](repeating: 9, count: 8 * 8 * 3)
        XCTAssertThrowsError(try stack.expanded([3, 200, 9, 200],
                                                pictures: [(pixels, 8, 8)])) { error in
            XCTAssertTrue("\(error)".contains("2 image placeholders"),
                          "the refusal counts both sides: \(error)")
        }
    }

    // MARK: Chunked prefill

    // Chunking is exact or it is nothing: each chunk attends through the cache to the same prefix a
    // single pass would. The chunk sizes here divide neither the prompt (13) nor the compression
    // ratio (2), so a boundary lands inside a ratio-2 compressor's parked group, which is the state
    // a chunked prefill is most likely to lose. The tolerance is the engram's, not the chunking's:
    // its gate is discontinuous at a zero dot product, so a position whose dot sits within float
    // noise of zero can land on either side of the sign and move its gate by about 5e-4.
    //
    // A chunk of ONE is included because it is the case that does not work and is raised: below the
    // compression ratio a chunk finishes no group, emits no compressed position, and its queries
    // attend over a cache a single pass would have filled. Measured, that read 0.31 against a
    // tolerance of 2e-3, diverging at the first layer that owns a compressor.
    func testAChunkedPrefillMatchesASinglePass() throws {
        try requireMLXRuntime()
        let net = oracleNet()
        let prompt = [3, 9, 14, 2, 7, 11, 5, 1, 13, 6, 4, 8, 12]
        XCTAssertGreaterThan(prompt.count, oracleShaped.slidingWindow,
                             "the ring has wrapped before the last chunk")

        let single = net.prefill(prompt, cache: NFKMLXDeepSeekCache(oracleShaped), chunkSize: nil)
        eval(single)
        let expected = single[0, -1].asArray(Float.self)

        for chunkSize in [1, 3, 5, 12] {
            let stepped = net.prefill(prompt, cache: NFKMLXDeepSeekCache(oracleShaped),
                                      chunkSize: chunkSize)
            eval(stepped)
            let actual = stepped[0, -1].asArray(Float.self)
            let worst = zip(actual, expected).map { abs($0 - $1) }.max() ?? 0
            XCTAssertLessThan(worst, 2e-3,
                              "a prompt fed \(chunkSize) tokens at a time is the same prefill")
            XCTAssertEqual(stepped[0, -1].argMax().item(Int.self),
                           single[0, -1].argMax().item(Int.self),
                           "and picks the same token at chunk size \(chunkSize)")
        }

        // The point of chunking is the peak, so the option has to reach the loop a caller runs.
        var options = NFKMLXGenerationOptions()
        options.maxTokens = 4
        let produced = net.generate(prompt: prompt, options: options)
        options.prefillChunkSize = 5
        XCTAssertEqual(net.generate(prompt: prompt, options: options), produced,
                       "generation through a chunked prefill produces the same tokens")
        XCTAssertEqual(oracleShaped.minimumPrefillChunk, 2,
                       "the largest ratio among the layers that own a compressor")
        let raised = net.prefill(prompt, cache: NFKMLXDeepSeekCache(oracleShaped), chunkSize: 1)
        let honored = net.prefill(prompt, cache: NFKMLXDeepSeekCache(oracleShaped), chunkSize: 2)
        eval(raised, honored)
        XCTAssertEqual(raised[0, -1].asArray(Float.self), honored[0, -1].asArray(Float.self),
                       "a chunk below the ratio is raised to it rather than answered differently")

        print("VALIDATION runtime deepseek-v4.1-chunked-prefill: chunk sizes 1/3/5/12 over a "
              + "\(prompt.count)-token prompt all match the single pass, 1 raised to "
              + "\(oracleShaped.minimumPrefillChunk)")
    }

    // MARK: Paging what a step barely touches

    /// The oracle's geometry with an n-gram table wide enough to carry MORE THAN ONE scale a row.
    ///
    /// The release stores that table with one `e8m0` per row per 32 channels, which is what lets a
    /// row decode on its own. At the oracle's 32 channels a row is exactly one block, so the
    /// per-row blocking and the square blocking every other weight uses coincide and a rule that
    /// confused them would pass. At 64 they do not.
    private var pagedShaped: NFKMLXDeepSeekConfiguration {
        var c = oracleShaped
        c.engramHeadDimensions = 64
        c.fp8BlockSize = 32
        return c
    }

    /// Encodes a matrix the way the release stores a routed expert: `e2m1` values packed two to a
    /// byte, low nibble first, with one `e8m0` scale per row per 32 columns.
    private func fourBitStored(_ weight: MLXArray) -> (bytes: MLXArray, scale: MLXArray) {
        let magnitudes: [Float] = [0, 0.5, 1, 1.5, 2, 3, 4, 6]
        return blockStored(weight, peak: 6) { scaled in
            var best = 0
            for (code, magnitude) in magnitudes.enumerated()
            where abs(abs(scaled) - magnitude) < abs(abs(scaled) - magnitudes[best]) { best = code }
            return (scaled < 0 ? UInt8(8) : 0) | UInt8(best)
        }
    }

    /// The same for an `e4m3` weight, which the release uses for the n-gram tables: a byte a value
    /// rather than a nibble, with the same per-row scales.
    private func eightBitStored(_ weight: MLXArray) -> (bytes: MLXArray, scale: MLXArray) {
        let table = NFKMLXDeepSeekQuantization.fp8Values
        return blockStored(weight, peak: 448) { scaled in
            var best = 0
            for (code, value) in table.enumerated()
            where value.isFinite && abs(value - scaled) < abs(table[best] - scaled) { best = code }
            return UInt8(best)
        }
    }

    /// The shared encoder. `e8m0` is an exponent alone, so a block's scale is the power of two that
    /// brings its largest magnitude inside `peak`, the format's largest finite value.
    private func blockStored(_ weight: MLXArray, peak limit: Float,
                             code: (Float) -> UInt8) -> (bytes: MLXArray, scale: MLXArray) {
        let rows = weight.dim(0)
        let columns = weight.dim(1)
        let block = NFKMLXDeepSeekQuantization.fp4BlockSize
        let blocks = (columns + block - 1) / block
        let values = weight.asArray(Float.self)
        let packing = limit == 6 ? 2 : 1
        var packed = [UInt8](repeating: 0, count: rows * columns / packing)
        var scales = [UInt8](repeating: 0, count: rows * blocks)

        for row in 0 ..< rows {
            for group in 0 ..< blocks {
                let low = group * block
                let high = min(low + block, columns)
                let peak = (low ..< high).map { abs(values[row * columns + $0]) }.max() ?? 0
                let exponent = peak > 0 ? max(-127, min(127, Int(ceil(log2(peak / limit))))) : 0
                scales[row * blocks + group] = UInt8(exponent + 127)
                let scale = exp2(Float(exponent))
                for column in low ..< high {
                    let index = row * columns + column
                    let encoded = code(values[index] / scale)
                    if packing == 1 {
                        packed[index] = encoded
                    } else {
                        packed[index / 2] |= index % 2 == 0 ? encoded : encoded << 4
                    }
                }
            }
        }
        return (MLXArray(packed).reshaped([rows, columns / packing]),
                MLXArray(scales).reshaped([rows, blocks]))
    }

    /// Writes a release directory storing the routed experts 4-bit and the n-gram tables fp8, which
    /// is the shape of the released checkpoint, and everything else as floats.
    private func writeQuantizedRelease(of net: NFKMLXDeepSeekNet, to directory: URL) throws {
        var arrays = [String: MLXArray]()
        for (name, value) in net.parameters().flattened() {
            let stored: (bytes: MLXArray, scale: MLXArray)?
            if NFKMLXDeepSeek.routedExpertAddress(name) != nil {
                stored = fourBitStored(value)
            } else if NFKMLXDeepSeek.ngramTableLayer(name) != nil {
                stored = eightBitStored(value)
            } else {
                stored = nil
            }
            guard let stored else {
                arrays[name] = value
                continue
            }
            arrays[name] = stored.bytes
            arrays[name.replacingOccurrences(of: ".weight", with: ".scale")] = stored.scale
        }
        try MLX.save(arrays: arrays, url: directory.appendingPathComponent("model.safetensors"))
    }

    /// Writes the same release across two shards, the n-gram tables alone in the second.
    ///
    /// A released table is 98 GB, which no shard holds beside anything else, so this is the layout a
    /// mapped load meets: mapping a table means never reading its shard, and that is only possible
    /// when the shard holds nothing else the decoder needs.
    private func writeShardedRelease(of net: NFKMLXDeepSeekNet, to directory: URL) throws {
        var body = [String: MLXArray]()
        var tables = [String: MLXArray]()
        for (name, value) in net.parameters().flattened() {
            let isTable = NFKMLXDeepSeek.ngramTableLayer(name) != nil
            let stored: (bytes: MLXArray, scale: MLXArray)?
            if NFKMLXDeepSeek.routedExpertAddress(name) != nil {
                stored = fourBitStored(value)
            } else if isTable {
                stored = eightBitStored(value)
            } else {
                stored = nil
            }
            let scaleName = name.replacingOccurrences(of: ".weight", with: ".scale")
            if let stored {
                if isTable {
                    tables[name] = stored.bytes
                    tables[scaleName] = stored.scale
                } else {
                    body[name] = stored.bytes
                    body[scaleName] = stored.scale
                }
            } else {
                body[name] = value
            }
        }
        let first = "model-00001-of-00002.safetensors"
        let second = "model-00002-of-00002.safetensors"
        try MLX.save(arrays: body, url: directory.appendingPathComponent(first))
        try MLX.save(arrays: tables, url: directory.appendingPathComponent(second))
        var map = [String: String]()
        for key in body.keys { map[key] = first }
        for key in tables.keys { map[key] = second }
        let index = try JSONSerialization.data(withJSONObject: ["weight_map": map])
        try index.write(to: directory.appendingPathComponent("model.safetensors.index.json"))
    }

    // Mapping is what takes a held table to nothing held at all, and it has to be the SAME table:
    // the rows a lookup copies out of the mapping are the bytes the resident path would have
    // gathered, so the decode that follows is the same arithmetic on the same operands.
    func testAMappedNgramTableIsTheTableItMaps() throws {
        try requireMLXRuntime()
        let directory = try temporaryDirectory("mapped")
        try writeShardedRelease(
            of: NFKMLXDeepSeekNet(pagedShaped, compressedTokens: oracleTokens), to: directory)

        let resident = try loadedRelease(paging: .none, in: directory)
        var held = NFKMLXDeepSeekPaging.all
        held.expertCacheBytes = 0
        var maps = NFKMLXDeepSeekPaging.mapped
        maps.expertCacheBytes = 0
        let stored = try loadedRelease(paging: held, in: directory)
        let mapped = try loadedRelease(paging: maps, in: directory)

        for layer in pagedShaped.engramLayerIDs {
            let table = try XCTUnwrap(mapped.layers[layer].engram?.storedTable)
            XCTAssertEqual(table.storedBytes, 0, "a mapped table holds nothing in memory")
            XCTAssertGreaterThan(
                try XCTUnwrap(stored.layers[layer].engram?.storedTable).storedBytes, 0,
                "where a stored one holds its bytes")
        }

        let ids = MLXArray([3, 9, 14, 2, 7, 11, 5].map(Int32.init)).reshaped([1, 7])
        let (a, b, c) = (resident(ids), stored(ids), mapped(ids))
        eval(a, b, c)
        XCTAssertEqual((a - c).abs().max().item(Float.self), 0,
                       "a mapped decoder is the resident one")
        XCTAssertEqual((b - c).abs().max().item(Float.self), 0,
                       "and the same as holding the same bytes in memory")

        // The whole point is what the machine has to find room for.
        let flash = NFKMLXDeepSeekConfiguration.v41Flash
        let heldBytes = NFKMLXDeepSeek.decoderBytes(for: flash, paging: .all)
        let mappedBytes = NFKMLXDeepSeek.decoderBytes(for: flash, paging: .mapped)
        let gib = { (bytes: Int) in String(format: "%.1f", Double(bytes) / 1_073_741_824) }
        XCTAssertLessThan(mappedBytes, heldBytes, "mapping the tables takes them off the budget")
        print("VALIDATION runtime deepseek-v4.1-mapped: identical logits; the decoder goes from "
              + "\(gib(heldBytes)) GiB held to \(gib(mappedBytes)) GiB with the tables mapped")
    }

    // The experts map through the same machinery and a different trade: their shard is still read,
    // because the parameters beside them are what the decoder is built from, and what changes is
    // what is KEPT. The decoded expert has to be the same either way.
    func testMappedRoutedExpertsAreTheExpertsTheyMap() throws {
        try requireMLXRuntime()
        let directory = try temporaryDirectory("mapped-experts")
        try writeShardedRelease(
            of: NFKMLXDeepSeekNet(pagedShaped, compressedTokens: oracleTokens), to: directory)

        var held = NFKMLXDeepSeekPaging.all
        held.expertCacheBytes = 0
        var maps = NFKMLXDeepSeekPaging.fullyMapped
        maps.expertCacheBytes = 0
        let stored = try loadedRelease(paging: held, in: directory)
        let mapped = try loadedRelease(paging: maps, in: directory)

        XCTAssertEqual(try XCTUnwrap(mapped.expertStore).storedBytes, 0,
                       "a mapped expert store holds nothing in memory")
        XCTAssertGreaterThan(try XCTUnwrap(stored.expertStore).storedBytes, 0,
                             "where a stored one holds every expert's bytes")

        let ids = MLXArray([3, 9, 14, 2, 7, 11, 5].map(Int32.init)).reshaped([1, 7])
        let (a, b) = (stored(ids), mapped(ids))
        eval(a, b)
        XCTAssertEqual((a - b).abs().max().item(Float.self), 0,
                       "a mapped expert decodes to the expert it maps")

        let flash = NFKMLXDeepSeekConfiguration.v41Flash
        let gib = { (bytes: Int) in String(format: "%.1f", Double(bytes) / 1_073_741_824) }
        let everything = NFKMLXDeepSeek.decoderBytes(for: flash, paging: .fullyMapped)
        XCTAssertLessThan(everything, NFKMLXDeepSeek.decoderBytes(for: flash, paging: .mapped),
                          "mapping the experts too takes them off the budget")
        print("VALIDATION runtime deepseek-v4.1-fully-mapped: identical logits; the decoder holds "
              + "\(gib(everything)) GiB with every group mapped")
    }

    // Mapping a table means never reading its shard. A shard that mixes one with other parameters
    // cannot be handled that way, and loading it would materialize the tensor the mapping exists to
    // avoid, so the refusal names what it found rather than quietly reading 98 GB.
    func testAMappedTableSharingAShardIsRefused() throws {
        try requireMLXRuntime()
        let directory = try temporaryDirectory("mapped-mixed")
        try writeQuantizedRelease(
            of: NFKMLXDeepSeekNet(pagedShaped, compressedTokens: oracleTokens), to: directory)
        XCTAssertThrowsError(try loadedRelease(paging: .mapped, in: directory)) { error in
            XCTAssertTrue("\(error)".contains("mapping a table means never reading its shard"),
                          "the refusal says why: \(error)")
        }
    }

    /// A directory written from a resident decoder of `pagedShaped`, and a decoder loaded from it.
    private func loadedRelease(paging: NFKMLXDeepSeekPaging, in directory: URL) throws
        -> NFKMLXDeepSeekNet {
        let net = NFKMLXDeepSeekNet(pagedShaped, compressedTokens: oracleTokens, paging: paging)
        try NFKMLXDeepSeek.loadWeights(into: net, fromDirectory: directory)
        return net
    }

    private func temporaryDirectory(_ name: String) throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("deepseek-\(name)-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        return directory
    }

    // The claim paging rests on, for both groups at once. A paged decoder holds the routed experts
    // and the n-gram tables as the release stores them and decodes what a step reads, so it must be
    // the SAME decoder: same weights, same arithmetic, same numbers. Grouping by expert and
    // gathering rows change WHEN a decode happens, not what is multiplied, so this is an equality
    // rather than a cosine.
    func testAPagedDecoderProducesTheSameLogitsAsAResidentOne() throws {
        try requireMLXRuntime()
        let directory = try temporaryDirectory("paged")
        try writeQuantizedRelease(
            of: NFKMLXDeepSeekNet(pagedShaped, compressedTokens: oracleTokens), to: directory)

        let resident = try loadedRelease(paging: .none, in: directory)
        var policy = NFKMLXDeepSeekPaging.all
        policy.expertCacheBytes = 0
        let paged = try loadedRelease(paging: policy, in: directory)
        let store = try XCTUnwrap(paged.expertStore)

        // Neither group is a parameter of a paged decoder. If either were, it would be holding both
        // forms and saving nothing.
        let held = paged.parameters().flattened().map(\.0)
        XCTAssertEqual(held.filter { NFKMLXDeepSeek.routedExpertAddress($0) != nil }, [],
                       "a paged decoder holds no routed expert as a parameter")
        XCTAssertEqual(held.filter { NFKMLXDeepSeek.ngramTableLayer($0) != nil }, [],
                       "and no n-gram table either")
        XCTAssertEqual(store.expertCount, pagedShaped.layerCount * pagedShaped.routedExpertCount,
                       "the store holds every layer's experts")
        for layer in pagedShaped.engramLayerIDs {
            XCTAssertNotNil(paged.layers[layer].engram?.storedTable,
                            "layer \(layer) holds its table stored")
            XCTAssertNil(paged.layers[layer].engram?.table,
                         "and builds no float embedding for it")
        }

        let ids = MLXArray([3, 9, 14, 2, 7, 11, 5].map(Int32.init)).reshaped([1, 7])
        let (mine, theirs) = (resident(ids), paged(ids))
        eval(mine, theirs)
        let delta = (mine - theirs).abs().max()
        eval(delta)
        XCTAssertEqual(delta.item(Float.self), 0,
                       "a paged decoder is the resident one, not an approximation of it")
        XCTAssertGreaterThan(store.decodeCount, 0, "the run decoded experts from stored bytes")

        // A decode step is a chunk of one token, which the prefill above does not exercise: the
        // grouping has one member a group there, and the cache carries between steps.
        var options = NFKMLXGenerationOptions()
        options.maxTokens = 4
        let prompt = [3, 9, 14, 2, 7, 11, 5]
        XCTAssertEqual(paged.generate(prompt: prompt, options: options),
                       resident.generate(prompt: prompt, options: options),
                       "a paged decoder generates what the resident one generates")

        let tables = pagedShaped.engramLayerIDs
            .compactMap { paged.layers[$0].engram?.storedTable?.storedBytes }
            .reduce(0, +)
        print("VALIDATION runtime deepseek-v4.1-paged: identical logits, "
              + "\(store.decodeCount) expert decodes, experts stored \(store.storedBytes) bytes, "
              + "n-gram tables stored \(tables) bytes")
    }

    // What paging buys, at the size it can be measured at here: a 4-bit expert is half a byte a
    // value and an fp8 table is one, each plus a scale byte per 32, against four bytes decoded.
    func testAPagedStoreHoldsItsGroupsAtTheirStoredSize() throws {
        try requireMLXRuntime()
        let directory = try temporaryDirectory("paged-size")
        try writeQuantizedRelease(
            of: NFKMLXDeepSeekNet(pagedShaped, compressedTokens: oracleTokens), to: directory)
        let paged = try loadedRelease(paging: .all, in: directory)

        let c = pagedShaped
        let experts = c.layerCount * c.routedExpertCount
        let matrix = c.expertIntermediateSize * c.hiddenSize
        XCTAssertEqual(paged.expertStore?.storedBytes, experts * 3 * (matrix / 2 + matrix / 32),
                       "half a byte a value plus one e8m0 scale per 32")

        for (index, layer) in c.engramLayerIDs.enumerated() {
            let rows = c.engramEmbeddingCounts[index]
            let table = try XCTUnwrap(paged.layers[layer].engram?.storedTable)
            XCTAssertEqual(table.storedBytes, rows * (c.engramHeadDimensions
                                                      + c.engramHeadDimensions / 32),
                           "a byte a channel plus one e8m0 scale per 32 of them")
            XCTAssertLessThan(table.storedBytes * 3, rows * c.engramHeadDimensions * 4,
                              "the stored table is under a third of the decoded one")
        }
    }

    // A cache that never holds anything decodes on every request; one with room for an expert
    // decodes it once. The budget is the whole of the policy, so both ends of it are held.
    func testACachedExpertDecodesOnceAndAnUncachedOneEveryTime() throws {
        try requireMLXRuntime()
        let directory = try temporaryDirectory("paged-cache")
        try writeQuantizedRelease(
            of: NFKMLXDeepSeekNet(pagedShaped, compressedTokens: oracleTokens), to: directory)
        var policy = NFKMLXDeepSeekPaging(routedExperts: true)
        policy.expertCacheBytes = 0
        let store = try XCTUnwrap(try loadedRelease(paging: policy, in: directory).expertStore)

        XCTAssertNotNil(store.expert(layer: 0, index: 0))
        XCTAssertNotNil(store.expert(layer: 0, index: 0))
        XCTAssertEqual(store.decodeCount, 2, "no budget decodes every time")
        XCTAssertEqual(store.cacheHitCount, 0)
        XCTAssertEqual(store.cachedBytes, 0, "and holds nothing between the two")

        let expert = 3 * pagedShaped.expertIntermediateSize * pagedShaped.hiddenSize * 4
        store.cacheByteBudget = expert
        XCTAssertNotNil(store.expert(layer: 0, index: 0))
        XCTAssertNotNil(store.expert(layer: 0, index: 0))
        XCTAssertEqual(store.decodeCount, 3, "room for one expert decodes it once")
        XCTAssertEqual(store.cacheHitCount, 1)

        // One expert's room, a second expert asked for: the first is evicted, not both kept.
        XCTAssertNotNil(store.expert(layer: 0, index: 1))
        XCTAssertLessThanOrEqual(store.cachedBytes, store.cacheByteBudget,
                                 "the cache stays inside its budget")
        store.clearCache()
        XCTAssertEqual(store.cachedBytes, 0)
    }

    // A paged load drops its groups from the module, so the coverage check that catches a missing
    // parameter cannot see them. Without a check per group a release short of an expert or a table
    // would load clean and route into nothing.
    func testAPagedLoadMissingAGroupIsRefused() throws {
        try requireMLXRuntime()
        let store = NFKMLXDeepSeekExpertStore(configuration: pagedShaped, cacheByteBudget: 0)
        let plain = MLXArray.zeros([pagedShaped.expertIntermediateSize, pagedShaped.hiddenSize])
        store.store(NFKDeepSeekStoredMatrix(stored: plain, scale: nil, shape: plain.shape),
                    layer: 0, expert: 0, matrix: "w1")
        XCTAssertThrowsError(try store.verifyComplete(pagedShaped)) { error in
            XCTAssertTrue("\(error)".contains(".weight"),
                          "the refusal names a matrix that is absent: \(error)")
        }

        // The same for a table: a directory whose n-gram tables are absent is refused by name.
        let directory = try temporaryDirectory("paged-missing")
        let net = NFKMLXDeepSeekNet(pagedShaped, compressedTokens: oracleTokens)
        var arrays = Dictionary(uniqueKeysWithValues: net.parameters().flattened())
        for name in arrays.keys where NFKMLXDeepSeek.ngramTableLayer(name) != nil {
            arrays.removeValue(forKey: name)
        }
        try MLX.save(arrays: arrays, url: directory.appendingPathComponent("model.safetensors"))
        XCTAssertThrowsError(try loadedRelease(paging: .all, in: directory)) { error in
            XCTAssertTrue("\(error)".contains("n-gram tables of layers"),
                          "the refusal names the tables: \(error)")
        }
    }

    // The figure the whole exercise is for. The released decoder cannot be held as floats on any
    // machine that exists, and which parts move it is a measurement rather than an argument.
    func testPagingMovesTheReleasesFootprint() throws {
        let flash = NFKMLXDeepSeekConfiguration.v41Flash
        let resident = NFKMLXDeepSeek.residentBytes(for: flash)
        let experts = NFKMLXDeepSeek.residentBytes(for: flash, paging: .init(routedExperts: true))
        let tables = NFKMLXDeepSeek.residentBytes(for: flash, paging: .init(ngramTables: true))
        let both = NFKMLXDeepSeek.residentBytes(for: flash, paging: .all)
        let gib = { (bytes: Int) in String(format: "%.1f", Double(bytes) / 1_073_741_824) }
        print("VALIDATION runtime deepseek-v4.1-paging-fit: resident \(gib(resident)) GiB, "
              + "experts paged \(gib(experts)) GiB, n-gram tables paged \(gib(tables)) GiB, "
              + "both \(gib(both)) GiB")
        XCTAssertLessThan(experts, resident / 2, "the experts are more than half the decoder")
        XCTAssertLessThan(both, experts, "and the tables are most of what the experts leave")

        // The enumeration covers the draft stack because the structural check measures against the
        // reference's whole module tree, and the decoder never builds it, so the figure a load
        // actually allocates is lower by that much. The check keeps the conservative number;
        // reporting the difference is what stops a reader treating it as headroom that exists.
        let draft = NFKMLXDeepSeek.draftStackBytes(for: flash, paging: .all)
        let decoder = NFKMLXDeepSeek.decoderBytes(for: flash, paging: .all)
        XCTAssertEqual(decoder, both - draft, "the decoder is the release without its draft stack")
        XCTAssertLessThan(decoder, 512 << 30,
                          "a fully paged decoder is inside a 512 GiB machine's memory")
        print("VALIDATION runtime deepseek-v4.1-paging-draft: the draft stack is \(gib(draft)) GiB "
              + "of the fit check's figure and the decoder builds none of it, so a fully paged load "
              + "allocates about \(gib(decoder)) GiB")

        // The release computes in bf16, and the preset with it. Float32 doubles every parameter a
        // group does not already hold stored, except the few the release keeps float32 anyway.
        XCTAssertTrue(flash.computesInBFloat16, "the preset computes in the release's own dtype")
        var wide = flash
        wide.computesInBFloat16 = false
        let narrowResident = NFKMLXDeepSeek.decoderBytes(for: flash)
        let narrowMapped = NFKMLXDeepSeek.decoderBytes(for: flash, paging: .fullyMapped)
        let wideResident = NFKMLXDeepSeek.decoderBytes(for: wide)
        let wideHeld = NFKMLXDeepSeek.decoderBytes(for: wide, paging: .all)
        let wideMapped = NFKMLXDeepSeek.decoderBytes(for: wide, paging: .fullyMapped)
        print("VALIDATION runtime deepseek-v4.1-bf16-fit: the decoder is "
              + "\(gib(narrowResident)) GiB resident, \(gib(decoder)) GiB held stored, "
              + "\(gib(narrowMapped)) GiB fully mapped in bf16; in float32 \(gib(wideResident)), "
              + "\(gib(wideHeld)) and \(gib(wideMapped)) GiB")
        XCTAssertLessThan(narrowMapped, wideMapped * 6 / 10, "bf16 about halves the mapped decoder")
        XCTAssertGreaterThan(narrowMapped, wideMapped * 4 / 10, "and no more than halves it")

        // A caller that has already chosen paging is asking about the paged figure, and a refusal
        // quoting the resident one would send it to a remedy it is already using.
        XCTAssertThrowsError(try NFKMLXDeepSeek.verifyFits(flash, budget: 1 << 30,
                                                           paging: .all)) { error in
            XCTAssertTrue("\(error)".contains("already stored"),
                          "the fully paged refusal says so: \(error)")
        }
        XCTAssertThrowsError(try NFKMLXDeepSeek.verifyFits(flash, budget: 1 << 30)) { error in
            XCTAssertTrue("\(error)".contains("NFKMLXDeepSeekPaging.all"),
                          "and a resident refusal names the remedy: \(error)")
        }
        // An unknown machine reports no working set, and an unknown machine does not gate a load.
        XCTAssertNoThrow(try NFKMLXDeepSeek.verifyFits(flash, budget: 0, paging: .all))

        // Speculating loads the draft stack, which the decoder's own figure leaves out. At the SAME
        // budget the flag has to flip the verdict: a check that forgot the stack would pass a load
        // that then runs out of memory.
        let mappedDecoder = NFKMLXDeepSeek.decoderBytes(for: flash, paging: .fullyMapped)
        let mappedDraft = NFKMLXDeepSeek.draftStackBytes(for: flash, paging: .fullyMapped)
        let between = mappedDecoder + mappedDraft - (1 << 30)
        XCTAssertNoThrow(try NFKMLXDeepSeek.verifyFits(flash, budget: between,
                                                       paging: .fullyMapped),
                         "the decoder alone fits")
        XCTAssertThrowsError(try NFKMLXDeepSeek.verifyFits(flash, budget: between,
                                                           paging: .fullyMapped,
                                                           includesDraftStack: true),
                             "and the decoder with the draft stack it would load does not")
    }
}
