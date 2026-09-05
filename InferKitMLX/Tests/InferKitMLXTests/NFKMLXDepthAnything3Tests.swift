//
//  NFKMLXDepthAnything3Tests.swift
//  InferKitMLXTests
//
//  Reference parity for Depth Anything 3 (monocular). The seams come from the authors'
//  `depth_anything_3` package via Tools/reference-parity (the da3_oracle capture): the input, the four
//  hooked backbone features, and the exp-depth map. Gated on the reference + released weights.
//
//    IK_VAL_DEPTH3_REF=~/.inferkit-validation/da3-reference.safetensors \
//    IK_VAL_DEPTH3_WEIGHTS=~/.inferkit-validation/da3-small/model.safetensors \
//    xcodebuild test -scheme InferKitMLXTests -destination 'platform=macOS' \
//      -skipPackagePluginValidation -only-testing:InferKitMLXTests/NFKMLXDepthAnything3Tests
//

import XCTest
import InferKit
import MLX
@testable import InferKitMLX

final class NFKMLXDepthAnything3Tests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test` (no bundled metallib); run via xcodebuild")
    }

    private func envPath(_ key: String) throws -> String {
        guard let path = ProcessInfo.processInfo.environment[key] else { throw XCTSkip("set \(key)") }
        let expanded = (path as NSString).expandingTildeInPath
        guard FileManager.default.fileExists(atPath: expanded) else { throw XCTSkip("\(key) missing: \(expanded)") }
        return expanded
    }

    private func cosine(_ a: MLXArray, _ b: MLXArray) -> Double {
        let x = a.reshaped([-1]).asType(.float32).asArray(Float.self).map(Double.init)
        let y = b.reshaped([-1]).asType(.float32).asArray(Float.self).map(Double.init)
        XCTAssertEqual(x.count, y.count, "shape mismatch")
        var dot = 0.0, na = 0.0, nb = 0.0
        for i in 0 ..< min(x.count, y.count) { dot += x[i] * y[i]; na += x[i] * x[i]; nb += y[i] * y[i] }
        return dot / (na.squareRoot() * nb.squareRoot() + 1e-12)
    }

    func testDepthAnything3MatchesTheReferenceSeamsAndDepth() throws {
        try requireMLXRuntime()
        let refURL = URL(fileURLWithPath: try envPath("IK_VAL_DEPTH3_REF"))
        let weightsURL = URL(fileURLWithPath: try envPath("IK_VAL_DEPTH3_WEIGHTS"))
        let reference = try loadArrays(url: refURL)

        let net = NFKMLXDepthAnything3.makeNet()
        try NFKMLXDepthAnything3.loadWeights(into: net, from: weightsURL)

        // The run_reference input_image is [H, W, 3]; the backbone takes NHWC with a batch axis.
        let input = reference["input_image"]!.reshaped([1, 518, 518, 3])

        let features = net.features(input)
        eval(features)
        for i in 0 ..< 4 {
            let c = cosine(features[i][0], reference["hook\(i)"]!)
            print("[DA3] hook\(i) cosine = \(c)")
            XCTAssertGreaterThan(c, 0.99999999, "hook\(i) must match the reference backbone feature")
        }

        // Localize any head divergence: compare each intermediate seam (NHWC → NCHW to match the
        // reference's [C,H,W]) by both cosine and mean-removed correlation.
        // The mean-removed correlation is what says the STRUCTURE matches, not just a near-constant
        // mean (the diffusion-preview lesson). The ConvTranspose resize layers need their own axis
        // order; a regression there collapses stage0/stage1 to ~0.01, which these seams catch.
        func meanRemoved(_ mineCHW: MLXArray, _ ref: MLXArray) -> Double {
            let x = mineCHW.reshaped([-1]).asType(.float32).asArray(Float.self).map(Double.init)
            let y = ref.reshaped([-1]).asType(.float32).asArray(Float.self).map(Double.init)
            let ma = x.reduce(0, +) / Double(x.count), mb = y.reduce(0, +) / Double(y.count)
            var dot = 0.0, na = 0.0, nb = 0.0
            for i in 0 ..< x.count { let da = x[i] - ma, db = y[i] - mb; dot += da * db; na += da * da; nb += db * db }
            return dot / (na.squareRoot() * nb.squareRoot() + 1e-12)
        }
        let headSeams = net.head.seams(features)
        eval(Array(headSeams.values))
        for name in ["stage0", "stage1", "stage2", "stage3", "fused", "logits"] {
            let mr = meanRemoved(headSeams[name]!.transposed(0, 3, 1, 2), reference[name]!)
            print("[DA3] \(name) mean-removed = \(mr)")
            XCTAssertGreaterThan(mr, 0.99999, "\(name) must match the reference head seam")
        }

        let (depth, _) = net.head(features)
        eval(depth)
        let c = cosine(depth[0], reference["output"]!)
        // The depth values cluster near 1.0, so raw cosine is dominated by the mean; the mean-removed
        // correlation is what says the depth STRUCTURE matches (the diffusion-preview lesson).
        let mine = depth[0].reshaped([-1]).asType(.float32).asArray(Float.self).map(Double.init)
        let ref = reference["output"]!.reshaped([-1]).asType(.float32).asArray(Float.self).map(Double.init)
        let meanA = mine.reduce(0, +) / Double(mine.count)
        let meanB = ref.reduce(0, +) / Double(ref.count)
        var dot = 0.0, na = 0.0, nb = 0.0, maxRel = 0.0
        for i in 0 ..< mine.count {
            let da = mine[i] - meanA, db = ref[i] - meanB
            dot += da * db; na += da * da; nb += db * db
            maxRel = max(maxRel, abs(mine[i] - ref[i]) / max(abs(ref[i]), 1e-6))
        }
        let meanRemoved = dot / (na.squareRoot() * nb.squareRoot() + 1e-12)
        print("[DA3] depth cosine = \(c)  mean-removed = \(meanRemoved)  maxRel = \(maxRel)")
        XCTAssertGreaterThan(c, 0.9999, "the depth map must match the reference")
        XCTAssertGreaterThan(meanRemoved, 0.9999, "the depth structure must match the reference")
    }

    // Every released tensor is either loaded by the monocular depth path or named as deliberately
    // unimplemented (the aux/ray fusion chain and heads, and the camera decoder/encoder), so a tensor
    // the port silently ignores is a failure rather than an oversight (the Gemma/DeepSeek discipline).
    func testEveryReleasedTensorIsLoadedOrNamedAsDropped() throws {
        try requireMLXRuntime()
        let weightsURL = URL(fileURLWithPath: try envPath("IK_VAL_DEPTH3_WEIGHTS"))
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: weightsURL)
        let net = NFKMLXDepthAnything3.makeNet()
        try NFKMLXDepthAnything3.loadWeights(into: net, from: weightsURL)
        let builtKeys = Set(net.parameters().flattened().map { $0.0 })

        var loaded = 0, dropped = 0, unaccounted = [String]()
        for (key, _) in checkpoint.arrays {
            if let remapped = NFKMLXDepthAnything3.remap(key) {
                if builtKeys.contains(remapped) { loaded += 1 } else { unaccounted.append(key) }
            } else if key.contains("_aux") || key.hasPrefix("model.cam_enc") || key.hasPrefix("model.cam_dec") {
                dropped += 1                                    // the aux/ray branch and the camera modules
            } else {
                unaccounted.append(key)
            }
        }
        print("[DA3] coverage: loaded=\(loaded) dropped=\(dropped) unaccounted=\(unaccounted.count)")
        XCTAssertEqual(unaccounted, [], "every released tensor must be loaded or named as deliberately dropped")
        XCTAssertEqual(loaded, builtKeys.count, "every built parameter must be loaded from the checkpoint")
    }
}
