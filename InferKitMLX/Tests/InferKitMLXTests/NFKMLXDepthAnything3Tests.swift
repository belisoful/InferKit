//
//  NFKMLXDepthAnything3Tests.swift
//  InferKitMLXTests
//
//  Reference parity for Depth Anything 3. The seams come from the authors' `depth_anything_3` package
//  via Tools/reference-parity (the da3_oracle capture): the input, the four hooked backbone features,
//  the head's stage and fusion seams, the exp-depth map, the aux (ray) pyramid and ray map, the camera
//  decoder's pose encoding, and the camera encoder's conditioning tokens. Gated on the reference +
//  released weights.
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
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    private func envPath(_ key: String) throws -> String {
        guard let path = NFKMLXValidationConfig.environment[key] else { throw XCTSkip("set \(key)") }
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
        for name in ["stage0", "stage1", "stage2", "stage3", "fused", "logits",
                     "aux0", "aux1", "aux2", "aux3", "aux_pos", "aux_conv0", "aux_norm", "ray_logits"] {
            let mr = meanRemoved(headSeams[name]!.transposed(0, 3, 1, 2), reference[name]!)
            print("[DA3] \(name) mean-removed = \(mr)")
            XCTAssertGreaterThan(mr, 0.99999, "\(name) must match the reference head seam")
        }

        let prediction = net.head(features)
        let depth = prediction.depth
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

        // The ray branch. `ray` is [h, w, 6] and `ray_conf` [h, w], both at the finest fusion
        // resolution rather than the image size, which is the reference's own convention.
        eval(prediction.ray, prediction.rayConfidence)
        let rayCosine = cosine(prediction.ray[0], reference["ray"]!)
        let rayConfidence = cosine(prediction.rayConfidence[0], reference["ray_conf"]!)
        print("[DA3] ray cosine = \(rayCosine)  ray_conf cosine = \(rayConfidence)")
        XCTAssertEqual(prediction.ray[0].shape, reference["ray"]!.shape)
        XCTAssertGreaterThan(rayCosine, 0.9999, "the ray map must match the reference")
        XCTAssertGreaterThan(rayConfidence, 0.9999, "the ray confidence must match the reference")

        // The camera decoder reads the last hook's camera token, which is the concatenated local and
        // global halves before the final LayerNorm.
        let cameras = net.backbone.hooked(input).cameraTokens
        eval(cameras)
        for i in 0 ..< 4 {
            let tokenCosine = cosine(cameras[i], reference["camera_token\(i)"]!)
            print("[DA3] camera_token\(i) cosine = \(tokenCosine)")
            XCTAssertGreaterThan(tokenCosine, 0.99999999, "camera token \(i) must match the reference")
        }
        let pose = net.cameraDecoder(cameras[cameras.count - 1])
        let encoding = pose.encoding
        eval(encoding)
        let poseCosine = cosine(encoding, reference["pose_enc"]!)
        print("[DA3] pose_enc cosine = \(poseCosine)")
        XCTAssertGreaterThan(poseCosine, 0.9999999, "the predicted pose encoding must match the reference")

        // The camera encoder: the reference's own extrinsic and intrinsic become the nine-number
        // encoding, and that becomes the conditioning token the backbone would read.
        let extrinsic = reference["cam_enc_extrinsic"]!
        let intrinsic = reference["cam_enc_intrinsic"]!.asType(.float32).asArray(Float.self)
        // `extri_intri_to_pose_encoding` reads the camera-to-world pose, so the world-to-camera
        // extrinsic the record carries is inverted first: R becomes Rᵀ and t becomes -Rᵀt.
        let rotation = extrinsic[0 ..< 3, 0 ..< 3].transposed(1, 0)
        let translation = matmul(rotation, extrinsic[0 ..< 3, 3].reshaped([3, 1])).reshaped([3]) * Float(-1)
        let ours = NFKDA3CameraEncoder.encoding(
            rotation: rotation, translation: translation,
            focalLengths: (intrinsic[0], intrinsic[4]), imageSize: (518, 518))
        eval(ours)
        let encodingCosine = cosine(ours, reference["cam_enc_pose_encoding"]!)
        print("[DA3] cam_enc_pose_encoding cosine = \(encodingCosine)")
        XCTAssertGreaterThan(encodingCosine, 0.9999999, "the pose encoding must match the reference")

        let tokens = net.cameraEncoder(reference["cam_enc_pose_encoding"]!.reshaped([1, 1, 9]))
        eval(tokens)
        let tokenCosine = cosine(tokens, reference["cam_enc_tokens"]!)
        print("[DA3] cam_enc_tokens cosine = \(tokenCosine)")
        XCTAssertGreaterThan(tokenCosine, 0.9999999, "the camera encoder tokens must match the reference")
    }

    // Every released tensor is loaded: the backbone, both DualDPT branches, the camera decoder, and
    // the camera encoder. The converse holds but for six parameters, which is a property of the
    // release rather than of this port — see the assertion below.
    func testEveryReleasedTensorIsLoadedOrNamedAsDropped() throws {
        try requireMLXRuntime()
        let weightsURL = URL(fileURLWithPath: try envPath("IK_VAL_DEPTH3_WEIGHTS"))
        let checkpoint = try NFKMLXWeights.loadCheckpoint(url: weightsURL)
        let net = NFKMLXDepthAnything3.makeNet()
        try NFKMLXDepthAnything3.loadWeights(into: net, from: weightsURL)
        let builtKeys = Set(net.parameters().flattened().map { $0.0 })

        var loadedKeys = Set<String>()
        var unaccounted = [String]()
        for (key, _) in checkpoint.arrays {
            guard let remapped = NFKMLXDepthAnything3.remap(key), builtKeys.contains(remapped) else {
                unaccounted.append(key)
                continue
            }
            loadedKeys.insert(remapped)
        }
        // The release carries the aux head's channel LayerNorm for level 0 only, and the reference's
        // own loader (`utils/model_loading.py`) loads `strict=False`, so levels 1-3 run at the
        // `nn.LayerNorm` init of weight 1 and bias 0. Inference reads level 3, so the ray head's
        // normalization is unweighted by construction, in the reference and here alike.
        let unlearned = Set((1 ... 3).flatMap { level in
            ["weight", "bias"].map { "head.scratch.output_conv2_aux.\(level).2.\($0)" }
        })
        let built = builtKeys.subtracting(loadedKeys)
        print("[DA3] coverage: loaded=\(loadedKeys.count) built=\(builtKeys.count) "
              + "unloaded=\(built.count) unaccounted=\(unaccounted.count)")
        XCTAssertEqual(unaccounted, [], "every released tensor must map onto a built parameter")
        XCTAssertEqual(built, unlearned,
                       "the only parameters the release does not carry are the three unlearned aux LayerNorms")
    }
}
