//
//  NFKMLXTimesFMTests.swift
//  InferKitMLXTests
//
//  TimesFM 2.5 (google/timesfm-2.5-200m-pytorch, Apache-2.0) against the official google-research/timesfm
//  package, `run_reference.py timesfm` (`IK_VAL_TIMESFM25_PYTORCH`, `IK_PARITY_TIMESFM25`): the module's
//  seams on the prefill of one forecast, and every recorded forecast under the release card's flags and
//  with every flag off. The transformers-format release (`IK_VAL_TIMESFM25`) must load the same weights.
//

import XCTest
import InferKit
import MLX
import MLXNN
import MLXRandom
@testable import InferKitMLX

final class NFKMLXTimesFMTests: XCTestCase {

    override func tearDown() {
        Memory.clearCache()
        super.tearDown()
    }

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    private func release() throws -> (forecaster: NFKMLXTimesFM, record: [String: MLXArray]) {
        let env = NFKMLXValidationConfig.environment
        guard let directory = env["IK_VAL_TIMESFM25_PYTORCH"], let path = env["IK_PARITY_TIMESFM25"],
              FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_VAL_TIMESFM25_PYTORCH (release directory) and IK_PARITY_TIMESFM25 (oracle record)")
        }
        return (try NFKMLXTimesFM.timesFM(directoryURL: URL(fileURLWithPath: directory)),
                try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: path)).arrays)
    }

    static var tiny: NFKMLXTimesFMConfiguration {
        var c = NFKMLXTimesFMConfiguration()
        c.hiddenSize = 64
        c.intermediateSize = 64
        c.numLayers = 2
        c.numHeads = 4
        c.outputPatchLength = 64
        c.outputQuantileLength = 128
        return c
    }

    /// The module keys mirror the official checkpoint.
    func testParameterNamesFollowTheOfficialCheckpoint() throws {
        try requireMLXRuntime()
        let names = Set(NFKMLXTimesFMNet(Self.tiny).parameters().flattened().map(\.0))
        for name in ["tokenizer.hidden_layer.weight", "tokenizer.hidden_layer.bias", "tokenizer.residual_layer.bias",
                     "stacked_xf.1.attn.query.weight", "stacked_xf.1.attn.key.weight",
                     "stacked_xf.1.attn.value.weight", "stacked_xf.1.attn.out.weight", "stacked_xf.1.attn.query_ln.scale",
                     "stacked_xf.1.attn.key_ln.scale", "stacked_xf.1.attn.per_dim_scale.per_dim_scale",
                     "stacked_xf.1.pre_attn_ln.scale", "stacked_xf.1.post_ff_ln.scale", "stacked_xf.1.ff0.weight",
                     "output_projection_point.output_layer.weight", "output_projection_quantiles.residual_layer.weight"] {
            XCTAssertTrue(names.contains(name), "missing \(name)")
        }
        XCTAssertFalse(names.contains("output_projection_point.hidden_layer.bias"), "the output projections carry no bias")
    }

    /// Filling follows the reference: leading NaNs dropped, interior NaNs interpolated, trailing NaNs held.
    func testNaNsAreFilledAsTheReferenceFillsThem() {
        let filled = NFKMLXTimesFM.filled([.nan, .nan, 1, .nan, 3, 4, .nan, .nan, 10, .nan])
        XCTAssertEqual(filled, [1, 2, 3, 4, 6, 8, 10, 10])
        XCTAssertTrue(NFKMLXTimesFM.filled([.nan, .nan]).isEmpty)
    }

    /// The transformers-format release holds the official weights under other names, with the query, key,
    /// and value projections stored apart where the official checkpoint fuses them; both load to the
    /// identical network.
    func testTheTransformersReleaseLoadsTheSameWeights() throws {
        try requireMLXRuntime()
        let env = NFKMLXValidationConfig.environment
        guard let official = env["IK_VAL_TIMESFM25_PYTORCH"], let transformers = env["IK_VAL_TIMESFM25"] else {
            throw XCTSkip("set IK_VAL_TIMESFM25_PYTORCH and IK_VAL_TIMESFM25")
        }
        let a = try NFKMLXTimesFM.timesFM(directoryURL: URL(fileURLWithPath: official)).net
        let b = try NFKMLXTimesFM.timesFM(directoryURL: URL(fileURLWithPath: transformers)).net
        XCTAssertEqual(a.configuration, b.configuration)
        let other = Dictionary(uniqueKeysWithValues: b.parameters().flattened())
        var largest: Float = 0
        for (name, value) in a.parameters().flattened() {
            let counterpart = try XCTUnwrap(other[name], name)
            XCTAssertEqual(value.shape, counterpart.shape, name)
            largest = max(largest, abs(value - counterpart).max().item(Float.self))
        }
        print("VALIDATION PARITY timesfm transformers-format weights: largest |difference| \(largest)")
        XCTAssertEqual(largest, 0)
    }

    /// PARITY at each seam of the prefill: the tokenizer's embeddings, the first and last layers, and both
    /// projections, over the patches the context fills (the reference's all-padding patches excluded).
    func testSeamParityOnTheReleasedWeights() throws {
        try requireMLXRuntime()
        let (forecaster, record) = try release()
        let net = forecaster.net
        let masks = try XCTUnwrap(record["seam.masks"])
        let padded = masks.asType(.int32).min(axis: -1).asArray(Int32.self).prefix { $0 == 1 }.count
        let inputs = try XCTUnwrap(record["seam.inputs"])[padded...]
        let patches = inputs.dim(0)
        let patchMasks = masks[padded...].asType(.bool)
        var caches = [NFKTimesFMLayerCache?](repeating: nil, count: net.configuration.numLayers)
        let positions = MLXArray(0 ..< patches).asType(.float32).reshaped([1, patches])
        let embeddings = net.tokenizer(concatenated([inputs, patchMasks.asType(.float32)], axis: -1).expandedDimensions(axis: 0))
        var hidden = embeddings
        var firstLayer = hidden
        for (index, layer) in net.layers.enumerated() {
            hidden = layer(hidden, positions: positions, cache: &caches[index])
            if index == 0 { firstLayer = hidden }
        }
        let seams: [(String, MLXArray, MLXArray)] = [
            ("embeddings", embeddings[0], record["seam.embeddings"]![padded...]),
            ("layer0", firstLayer[0], record["seam.layer0"]![padded...]),
            ("last layer", hidden[0], record["seam.layer_last"]![padded...]),
            ("point", net.pointProjection(hidden)[0], record["seam.point"]![padded...]),
            ("quantiles", net.quantileProjection(hidden)[0], record["seam.quantiles"]![padded...]),
        ]
        var readings = [String]()
        for (name, mine, reference) in seams {
            let similarity = NFKMLXWav2Vec2Tests.cosine(mine, reference)
            readings.append("\(name) \(similarity)")
            XCTAssertGreaterThan(similarity, 0.99999, "\(name) diverges")
        }
        print("VALIDATION PARITY timesfm seams (\(patches) patches after \(padded) padding): "
              + readings.joined(separator: ", "))
    }

    /// PARITY on every recorded forecast: five series (seasonal, positive counts past two autoregressive
    /// steps, a truncated long series, NaN gaps, a constant) under the release card's flags and with every
    /// flag off.
    func testForecastsMatchTheReference() throws {
        try requireMLXRuntime()
        let (forecaster, record) = try release()
        let names = ["seasonal", "counts", "long", "gappy", "constant"]
        let horizons = try XCTUnwrap(record["horizons"]).asArray(Int64.self).map(Int.init)
        let off = NFKMLXTimesFMForecastOptions()
        off.maximumContext = 512
        off.normalizesInputs = false
        off.usesContinuousQuantileHead = false
        off.forcesFlipInvariance = false
        off.infersPositivity = false
        off.fixesQuantileCrossing = false
        var lines = [String]()
        for (tag, options) in [("on", NFKMLXTimesFMForecastOptions()), ("off", off)] {
            for (name, horizon) in zip(names, horizons) {
                let series = try XCTUnwrap(record["series.\(name)"]).asArray(Float.self)
                let reference = try XCTUnwrap(record["\(tag).\(name)"])
                let forecast = try forecaster.forecast(context: series, horizon: horizon, options: options)
                let mine = MLXArray(forecast.values.flatMap { $0 }).reshaped([horizon, 10])
                let scale = max(abs(reference).max().item(Float.self), 1e-6)
                let relative = abs(mine - reference).max().item(Float.self) / scale
                let similarity = NFKMLXWav2Vec2Tests.cosine(mine, reference)
                lines.append("\(tag).\(name) h\(horizon) \(similarity) (max rel \(relative))")
                XCTAssertLessThan(relative, 1e-4, "\(tag).\(name)")
                XCTAssertEqual(forecast.pointForecast.count, horizon)
            }
        }
        print("VALIDATION PARITY timesfm forecasts: " + lines.joined(separator: ", "))
    }
}
