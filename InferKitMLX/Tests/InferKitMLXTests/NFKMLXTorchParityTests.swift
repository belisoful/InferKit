//
//  NFKMLXTorchParityTests.swift
//  InferKitMLXTests
//
//  The native torch reader against the offline converters' own output: for a passthrough model the
//  raw checkpoint and the converted safetensors must agree key for key and value for value once both
//  are MLX arrays, and a model's loader must produce identical parameters from either file. These
//  evaluate arrays, so they run under xcodebuild (`swift test` has no metallib).
//

import XCTest
import MLX
import MLXNN
@testable import InferKitMLX

final class NFKMLXTorchParityTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test`; run via xcodebuild")
    }

    private lazy var config: [String: String] = {
        var merged = ProcessInfo.processInfo.environment
        let url = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".inferkit-validation.json")
        if let data = try? Data(contentsOf: url),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: String] {
            json.forEach { merged[$0.key] = $0.value }
        }
        return merged
    }()

    /// `fallbackFile` covers a checkpoint the manifest does not (yet) record: the file is looked up
    /// by its conventional name under `~/.inferkit-validation/<directory>/` when the config key is
    /// absent.
    private func filePath(_ key: String, fallbackFile: String? = nil,
                          directory: String = "raw") throws -> URL {
        if let path = config[key] {
            guard FileManager.default.fileExists(atPath: path) else {
                throw XCTSkip("\(key) points at a missing file: \(path)")
            }
            return URL(fileURLWithPath: path)
        }
        guard let fallbackFile else {
            throw XCTSkip("set \(key) in ~/.inferkit-validation.json (Tools/validation-assets/fetch.py)")
        }
        let fallback = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".inferkit-validation/\(directory)/\(fallbackFile)")
        guard FileManager.default.fileExists(atPath: fallback.path) else {
            throw XCTSkip("set \(key) or place \(fallbackFile) in ~/.inferkit-validation/\(directory)")
        }
        return fallback
    }

    private func assertArraysMatch(_ raw: [String: MLXArray], _ converted: [String: MLXArray],
                                   file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(Set(raw.keys), Set(converted.keys), file: file, line: line)
        for (name, array) in raw {
            guard let reference = converted[name] else { continue }
            XCTAssertEqual(array.shape, reference.shape, "\(name) shape", file: file, line: line)
            let difference = (array.asType(.float32) - reference.asType(.float32)).abs().max()
            eval(difference)
            XCTAssertEqual(difference.item(Float.self), 0, "\(name) differs", file: file, line: line)
        }
    }

    func testTheRawZipCheckpointMatchesTheConvertedArrays() throws {
        try requireMLXRuntime()
        let raw = try NFKMLXWeights.loadCheckpoint(url: try filePath("IK_RAW_REALESRGAN"))
        XCTAssertTrue(raw.needsConvTranspose)
        XCTAssertNil(raw.quantization)
        let converted = try loadArrays(url: try filePath("IK_VAL_REALESRGAN"))
        assertArraysMatch(raw.arrays, converted)
    }

    func testTheWhisperStridedTensorsMatchTheConvertedOracle() throws {
        // whisper_tiny.pt stores its Linear weights as transposed fp16 views, so this comparison is
        // the gather path held to torch's own materialization, tensor for tensor.
        try requireMLXRuntime()
        let raw = try NFKMLXWeights.loadCheckpoint(url: try filePath("IK_RAW_WHISPER"))
        let converted = try loadArrays(url: try filePath("IK_VAL_WHISPER"))
        assertArraysMatch(raw.arrays, converted)
    }

    /// A model's own loadWeights, handed the raw checkpoint, must land exactly the parameters the
    /// converted safetensors lands — remaps and transposes included.
    /// `tolerance` stays zero for a passthrough or rename-only path; a loader that computes (the
    /// weight-norm fusion) is held to float ulps instead, because its arithmetic and the offline
    /// converter's run the same formula in different operation orders.
    private func assertIdenticalParameters(_ rawNet: Module, _ convertedNet: Module,
                                           tolerance: Float = 0,
                                           file: StaticString = #filePath, line: UInt = #line) {
        let reference = Dictionary(uniqueKeysWithValues: convertedNet.parameters().flattened())
        for (name, value) in rawNet.parameters().flattened() {
            guard let expected = reference[name] else {
                XCTFail("\(name) is missing from the converted load", file: file, line: line)
                continue
            }
            let difference = (value.asType(.float32) - expected.asType(.float32)).abs().max()
            eval(difference)
            XCTAssertLessThanOrEqual(difference.item(Float.self), tolerance,
                                     "\(name) differs between the two loads", file: file, line: line)
        }
    }

    func testTheModelLoaderProducesIdenticalParametersFromEitherFile() throws {
        try requireMLXRuntime()
        let rawNet = NFKRealESRGANNet(blocks: 23, scale: 4)
        try NFKMLXRealESRGAN.loadWeights(into: rawNet, from: try filePath("IK_RAW_REALESRGAN"))
        let convertedNet = NFKRealESRGANNet(blocks: 23, scale: 4)
        try NFKMLXRealESRGAN.loadWeights(into: convertedNet, from: try filePath("IK_VAL_REALESRGAN"))
        assertIdenticalParameters(rawNet, convertedNet)
    }

    func testTheU2NetLoaderTranslatesTheRawNamesItself() throws {
        // The raw legacy .pth carries `rebnconvN` names the offline converter used to rename; the
        // loader's own translation must land the identical parameters.
        try requireMLXRuntime()
        let rawNet = NFKMLXU2Net.makeNet()
        try NFKMLXU2Net.loadWeights(into: rawNet, from: try filePath("IK_RAW_U2NET"))
        let convertedNet = NFKMLXU2Net.makeNet()
        try NFKMLXU2Net.loadWeights(into: convertedNet, from: try filePath("IK_VAL_U2NET"))
        assertIdenticalParameters(rawNet, convertedNet)
    }

    func testTheHiFiGANLoaderFusesTheRawWeightNormItself() throws {
        // The raw universal g_* release stores every convolution weight-normalized; the loader's
        // own fusion and renames must land the identical parameters the converted file lands.
        try requireMLXRuntime()
        let rawNet = NFKMLXHiFiGAN.makeNet()
        try NFKMLXHiFiGAN.loadWeights(into: rawNet, from: try filePath("IK_RAW_HIFIGAN_UNIVERSAL"))
        let convertedNet = NFKMLXHiFiGAN.makeNet()
        try NFKMLXHiFiGAN.loadWeights(into: convertedNet, from: try filePath("IK_VAL_HIFIGAN"))
        assertIdenticalParameters(rawNet, convertedNet, tolerance: 1e-6)
    }

    func testTheConvTasNetLoaderTakesTheRawDecoderLayout() throws {
        // The raw release's transposed-convolution decoder is 3-D [in, out, k]; the converted file
        // carries it pre-permuted 4-D. The loader's shape-keyed branches must land the same
        // parameters from either.
        try requireMLXRuntime()
        let rawNet = NFKMLXConvTasNetNet(.base)
        try NFKMLXConvTasNet.loadWeights(into: rawNet, from: try filePath("IK_RAW_CONVTASNET"))
        let convertedNet = NFKMLXConvTasNetNet(.base)
        try NFKMLXConvTasNet.loadWeights(into: convertedNet, from: try filePath("IK_VAL_CONVTASNET"))
        assertIdenticalParameters(rawNet, convertedNet)
    }

    func testTheDenoiserLoaderTakesTheRawLegacyCheckpoint() throws {
        try requireMLXRuntime()
        let rawNet = NFKMLXDenoiser.makeNet()
        try NFKMLXDemucs.loadWeights(into: rawNet, from: try filePath("IK_RAW_DENOISER"))
        let convertedNet = NFKMLXDenoiser.makeNet()
        try NFKMLXDemucs.loadWeights(into: convertedNet, from: try filePath("IK_VAL_DENOISER"))
        assertIdenticalParameters(rawNet, convertedNet)
    }

    func testThePoseLoaderTakesTheRawDeconvLayout() throws {
        // The raw and converted files carry IDENTICAL key names and differ only in the deconv
        // weights' axis order, which is what `Checkpoint.isNativeTorch` exists to distinguish.
        try requireMLXRuntime()
        let rawNet = NFKMLXPose.makeNet()
        try NFKMLXPose.loadWeights(into: rawNet, from: try filePath("IK_RAW_POSE"))
        let convertedNet = NFKMLXPose.makeNet()
        try NFKMLXPose.loadWeights(into: convertedNet, from: try filePath("IK_VAL_POSE"))
        assertIdenticalParameters(rawNet, convertedNet)
    }

    func testTheColorizerLoaderTranslatesTheRawNamesItself() throws {
        // The eccv16 release stores Sequential slots and one transposed convolution in the
        // ConvTranspose layout; the loader's translation and per-key transpose must land the
        // identical parameters the converted file lands.
        try requireMLXRuntime()
        let rawNet = NFKMLXColorizerNet(NFKMLXColorizerConfiguration())
        try NFKMLXColorizer.loadWeights(into: rawNet, from: try filePath("IK_RAW_COLORIZER"))
        let convertedNet = NFKMLXColorizerNet(NFKMLXColorizerConfiguration())
        try NFKMLXColorizer.loadWeights(into: convertedNet, from: try filePath("IK_VAL_COLORIZER"))
        assertIdenticalParameters(rawNet, convertedNet)
    }

    func testTheNAFNetLoaderTranslatesTheRawNamesItself() throws {
        try requireMLXRuntime()
        let rawNet = NFKMLXNAFNet.makeNet()
        try NFKMLXNAFNet.loadWeights(into: rawNet, from: try filePath("IK_RAW_NAFNET"))
        let convertedNet = NFKMLXNAFNet.makeNet()
        try NFKMLXNAFNet.loadWeights(into: convertedNet, from: try filePath("IK_VAL_NAFNET"))
        assertIdenticalParameters(rawNet, convertedNet)
    }

    func testTheRAFTLoaderTranslatesTheRawNamesItself() throws {
        try requireMLXRuntime()
        let rawNet = NFKMLXRAFT.makeNet()
        try NFKMLXRAFT.loadWeights(into: rawNet, from: try filePath("IK_RAW_RAFT"))
        let convertedNet = NFKMLXRAFT.makeNet()
        try NFKMLXRAFT.loadWeights(into: convertedNet, from: try filePath("IK_VAL_RAFT"))
        assertIdenticalParameters(rawNet, convertedNet)
    }

    func testTheRIFELoaderTakesTheRawCheckpoint() throws {
        // RIFE's remap already lives in Swift; the raw flownet.pkl adds only the teacher subtrees,
        // which the coverage rule ignores as extras.
        try requireMLXRuntime()
        let rawNet = NFKMLXRIFE.makeNet()
        try NFKMLXRIFE.loadWeights(into: rawNet, from: try filePath("IK_RAW_RIFE"))
        let convertedNet = NFKMLXRIFE.makeNet()
        try NFKMLXRIFE.loadWeights(into: convertedNet, from: try filePath("IK_VAL_RIFE"))
        assertIdenticalParameters(rawNet, convertedNet)
    }

    func testTheLaMaLoaderStripsTheGeneratorPrefixItself() throws {
        try requireMLXRuntime()
        let rawNet = NFKMLXLaMa.makeNet()
        try NFKMLXLaMa.loadWeights(into: rawNet, from: try filePath("IK_RAW_LAMA"))
        let convertedNet = NFKMLXLaMa.makeNet()
        try NFKMLXLaMa.loadWeights(into: convertedNet, from: try filePath("IK_VAL_LAMA"))
        assertIdenticalParameters(rawNet, convertedNet)
    }

    func testTheMODNetLoaderTakesTheRawCheckpoint() throws {
        // The Swift remap already strips the DataParallel prefix; the raw file adds only the
        // duplicated backbone reference, whose first copy wins.
        try requireMLXRuntime()
        let rawNet = NFKMLXMODNet.makeNet()
        try NFKMLXMODNet.loadWeights(into: rawNet, from: try filePath("IK_RAW_MODNET"))
        let convertedNet = NFKMLXMODNet.makeNet()
        try NFKMLXMODNet.loadWeights(into: convertedNet, from: try filePath("IK_VAL_MODNET"))
        assertIdenticalParameters(rawNet, convertedNet)
    }

    func testTheFastSpeech2LoaderStripsTheModelPrefixItself() throws {
        try requireMLXRuntime()
        let rawNet = NFKMLXFastSpeech2.makeNet()
        try NFKMLXFastSpeech2.loadWeights(into: rawNet, from: try filePath("IK_RAW_FASTSPEECH2"))
        let convertedNet = NFKMLXFastSpeech2.makeNet()
        try NFKMLXFastSpeech2.loadWeights(into: convertedNet, from: try filePath("IK_VAL_FASTSPEECH2"))
        assertIdenticalParameters(rawNet, convertedNet)
    }

    func testTheRetinaFaceLoaderTakesTheRawCheckpoint() throws {
        // The Swift remap already drops the classifier head and strips the training prefix; the
        // raw legacy .pth must land the identical parameters.
        try requireMLXRuntime()
        let rawNet = NFKMLXRetinaFace.makeNet()
        try NFKMLXRetinaFace.loadWeights(into: rawNet, from: try filePath("IK_RAW_RETINAFACE"))
        let convertedNet = NFKMLXRetinaFace.makeNet()
        try NFKMLXRetinaFace.loadWeights(into: convertedNet, from: try filePath("IK_VAL_RETINAFACE"))
        assertIdenticalParameters(rawNet, convertedNet)
    }

    func testTheStyleTransferLoaderTakesTheRawCheckpoint() throws {
        // The converter only drops the InstanceNorm running statistics, which the coverage rule
        // ignores as extras; the raw file must load unchanged. The only raw on hand is the seeded
        // checkpoint (the official release is unobtainable) — its values are meaningless for a
        // quality claim and byte-exact for an equivalence one, which is all this asserts.
        try requireMLXRuntime()
        let raw = try filePath("IK_RAW_STYLE", fallbackFile: "style_transfer_seeded.pth")
        let converted = try filePath("IK_VAL_STYLE_SEEDED", fallbackFile: "style-transfer-seeded.safetensors",
                                     directory: "converted")
        let rawNet = NFKStyleTransferNet()
        try NFKMLXStyleTransfer.loadWeights(into: rawNet, from: raw)
        let convertedNet = NFKStyleTransferNet()
        try NFKMLXStyleTransfer.loadWeights(into: convertedNet, from: converted)
        assertIdenticalParameters(rawNet, convertedNet)
    }

    func testTheObjCCheckpointConvertsOnDevice() throws {
        // The consumer path: inspect a raw checkpoint, convert it to safetensors with no Python, and
        // load the result through the ordinary checkpoint reader.
        try requireMLXRuntime()
        let checkpoint = try NFKMLXTorchCheckpoint.checkpoint(contentsOf: try filePath("IK_RAW_REALESRGAN"))
        XCTAssertEqual(checkpoint.tensorNames.count, checkpoint.contents.tensors.count)
        let info = try XCTUnwrap(checkpoint.info(forTensor: "conv_first.weight"))
        XCTAssertEqual(info.shape, [64, 3, 3, 3])
        XCTAssertEqual(info.scalarType, .float32)

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("NFKMLXTorchParityTests-\(UUID().uuidString).safetensors")
        defer { try? FileManager.default.removeItem(at: url) }
        try checkpoint.writeSafetensors(to: url)
        let reloaded = try NFKMLXWeights.loadCheckpoint(url: url)
        XCTAssertTrue(reloaded.needsConvTranspose)
        assertArraysMatch(try checkpoint.arrays(), reloaded.arrays)
    }
}
