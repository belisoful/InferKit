//
//  NFKMLXWav2Vec2BertTests.swift
//  InferKitMLXTests
//
//  W2V-BERT 2.0 (facebook/w2v-bert-2.0, MIT) against transformers' `Wav2Vec2BertModel` and
//  `SeamlessM4TFeatureExtractor`, `run_reference.py w2v_bert` (`IK_VAL_W2V_BERT_2`, `IK_PARITY_W2V_BERT_2`):
//  the filterbank, the stacked features, the projection, three Conformer layers, and the output.
//

import XCTest
import InferKit
import MLX
import MLXNN
@testable import InferKitMLX

final class NFKMLXWav2Vec2BertTests: XCTestCase {

    override func tearDown() {
        Memory.clearCache()
        super.tearDown()
    }

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    /// The module keys mirror the transformers checkpoint.
    func testParameterNamesFollowTheCheckpoint() throws {
        try requireMLXRuntime()
        var c = NFKMLXWav2Vec2BertConfiguration()
        c.numHiddenLayers = 1
        let names = Set(NFKMLXWav2Vec2BertNet(c).parameters().flattened().map(\.0))
        for name in ["feature_projection.layer_norm.weight", "feature_projection.projection.bias", "masked_spec_embed",
                     "encoder.layers.0.ffn1_layer_norm.weight", "encoder.layers.0.ffn1.intermediate_dense.weight",
                     "encoder.layers.0.self_attn.distance_embedding.weight", "encoder.layers.0.self_attn.linear_q.bias",
                     "encoder.layers.0.conv_module.pointwise_conv1.weight", "encoder.layers.0.conv_module.depthwise_conv.weight",
                     "encoder.layers.0.conv_module.depthwise_layer_norm.bias", "encoder.layers.0.final_layer_norm.weight"] {
            XCTAssertTrue(names.contains(name), "missing \(name)")
        }
        XCTAssertFalse(names.contains("encoder.layers.0.conv_module.depthwise_conv.bias"))
    }

    /// PARITY on the release, float32, seam by seam.
    func testTheReleaseIsAtParity() throws {
        try requireMLXRuntime()
        let env = NFKMLXValidationConfig.environment
        guard let directory = env["IK_VAL_W2V_BERT_2"], let path = env["IK_PARITY_W2V_BERT_2"],
              FileManager.default.fileExists(atPath: path) else {
            throw XCTSkip("set IK_VAL_W2V_BERT_2 (release directory) and IK_PARITY_W2V_BERT_2 (oracle record)")
        }
        let url = URL(fileURLWithPath: directory)
        let net = try NFKMLXWav2Vec2BertNet(configurationURL: url.appendingPathComponent("config.json"))
        try net.loadWeights(fromDirectory: url)
        let record = try NFKMLXWeights.loadCheckpoint(url: URL(fileURLWithPath: path)).arrays
        let waveform = try XCTUnwrap(record["waveform"]).asArray(Float.self)

        let fbank = NFKMLXWav2Vec2BertProcessor.filterbank(waveform)
        let referenceFbank = try XCTUnwrap(record["fbank"])
        guard fbank.shape == referenceFbank.shape else {
            return XCTFail("filterbank \(fbank.shape) against the reference's \(referenceFbank.shape)")
        }
        let fbankDifference = abs(fbank - referenceFbank).max().item(Float.self)
        let (features, mask) = NFKMLXWav2Vec2BertProcessor.inputFeatures(waveform)
        let referenceFeatures = try XCTUnwrap(record["input_features"])
        let referenceMask = try XCTUnwrap(record["attention_mask"]).asType(.bool)
        guard features.shape == [1] + referenceFeatures.shape else {
            return XCTFail("features \(features.shape) against the reference's \(referenceFeatures.shape)")
        }
        XCTAssertEqual(mask[0].asArray(Bool.self), referenceMask.asArray(Bool.self), "frame mask")
        let featureDifference = abs(features[0] - referenceFeatures).max().item(Float.self)

        let seams = net.seams(referenceFeatures.expandedDimensions(axis: 0), mask: referenceMask.expandedDimensions(axis: 0))
        var measured: [(String, Double)] = [("features", NFKMLXWav2Vec2Tests.cosine(features[0], referenceFeatures)),
                                            ("projected", NFKMLXWav2Vec2Tests.cosine(seams.projected[0], record["projected"]!))]
        for index in try XCTUnwrap(record["layer_indices"]).asArray(Int64.self).map(Int.init) {
            measured.append(("layer\(index)", NFKMLXWav2Vec2Tests.cosine(seams.layers[index][0], try XCTUnwrap(record["layer\(index)"]))))
        }
        measured.append(("output", NFKMLXWav2Vec2Tests.cosine(seams.output[0], record["output"]!)))
        measured.append(("output from our features", NFKMLXWav2Vec2Tests.cosine(net(features, mask: mask)[0], record["output"]!)))
        print("VALIDATION PARITY w2v-bert-2.0: fbank max |d| \(fbankDifference), features max |d| \(featureDifference), "
              + measured.map { "\($0.0) \($0.1)" }.joined(separator: ", "))
        XCTAssertLessThan(fbankDifference, 1e-2, "filterbank")
        for (seam, similarity) in measured {
            XCTAssertGreaterThan(similarity, 0.99999, "\(seam) diverges")
        }

        let backend = try NFKMLXWav2Vec2Bert.backend(directoryURL: url)
        // The clip the oracle read, byte for byte: re-encoding the samples would move most of them by one
        // 16-bit step, since the writer truncates.
        let wav = try Data(contentsOf: URL(fileURLWithPath: NFKMLXValidationConfig.environment["IK_VAL_AUDIO"]
            ?? NFKMLXValidationConfig.root.appendingPathComponent("inputs/speech.wav").path))
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: [NFKInputAudio: wav]))
        let embedding = try XCTUnwrap(result.output(forKey: NFKOutputEmbedding) as? [NSNumber])
        XCTAssertEqual(embedding.count, 1024)
        let real = record["output"]![MLXArray(referenceMask.asArray(Bool.self).enumerated().filter(\.element).map { Int32($0.offset) })]
        XCTAssertGreaterThan(NFKMLXWav2Vec2Tests.cosine(MLXArray(embedding.map(\.floatValue)), real.mean(axis: 0)),
                             0.9999, "backend embedding")
    }
}
