//
//  NFKMLXSAM3Backend.swift
//  InferKitMLX
//
//  SAM 3's image path behind the InferKit contract: a picture and a worded concept in, every instance
//  the concept names out, as boxes and a mask.
//
//  Introduced in InferKit 0.4.0.
//

import CoreGraphics
import Foundation
import InferKit
import MLX

/// The request parameters a SAM 3 backend reads beyond the core keys.
///
/// Introduced in InferKit 0.4.0.
@objc(NFKMLXSAM3ParameterKey)
public final class NFKMLXSAM3ParameterKey: NSObject {
    /// The score an instance must exceed to be returned, as an `NSNumber` in `0...1` (default 0.5). An
    /// instance's score is its query's probability times the probability that the concept is present.
    /// Introduced in InferKit 0.4.0.
    @objc public static let scoreThreshold = "NFKMLXParameterSAM3ScoreThreshold"
    /// The probability a pixel must exceed to belong to an instance's mask, as an `NSNumber` in `0...1`
    /// (default 0.5). Introduced in InferKit 0.4.0.
    @objc public static let maskThreshold = "NFKMLXParameterSAM3MaskThreshold"
}

extension NFKMLXSAM3 {

    /// The registry name the image path builds under. Introduced in InferKit 0.4.0.
    @objc public static let modelName = "sam3"

    /// The Hugging Face repository the download factories read when none is named. It is gated: its
    /// license is accepted once on the Hub, and the token is `NFKHFHub.defaultAccessToken`. Introduced
    /// in InferKit 0.4.0.
    @objc public static let releaseRepo = "facebook/sam3"

    /// The request inputs the backend reads.
    static let backendInputKeys: Set<String> = [NFKInputImage, NFKInputPrompt]

    /// The request parameters the backend reads.
    static let backendParameterKeys: Set<String> = [NFKMLXSAM3ParameterKey.scoreThreshold,
                                                    NFKMLXSAM3ParameterKey.maskThreshold]

    /// The image path and its CLIP tokenizer behind the InferKit contract.
    ///
    /// @discussion The backend reads a `CGImage` or texture under `NFKInputImage` and the concept to find
    /// under `NFKInputPrompt` (both required). The image is resized to the vision configuration's square
    /// size and normalized by mean 0.5 and deviation 0.5, as the release's processor does; the prompt
    /// is padded to the text tower's context with its padding masked. It returns every instance whose
    /// score exceeds ``NFKMLXSAM3ParameterKey/scoreThreshold`` as an `NFKDetection` under
    /// `NFKOutputDetections`, most confident first, labeled with the prompt and boxed in `0...1` of the
    /// input image, and the union of their masks at the input's size under `NFKOutputMask` as a
    /// grayscale `CGImage`. No mask is returned where no instance passes. The resize is bilinear
    /// without the antialiasing the reference processor applies when it shrinks an image.
    static func inferenceBackend(model: NFKMLXSAM3ImageModel, tokenizer: NFKMLXSDPromptTokenizer,
                                 identifier: String = modelName) -> any NFKInferenceBackend {
        NFKMLXGenerationBackend(identifier: identifier, inputKeys: backendInputKeys,
                                parameterKeys: backendParameterKeys) { request, job in
            let concept = try NFKMLXGenerationBackend.requiredText(request, NFKInputPrompt)
            guard let value = request.input(forKey: NFKInputImage) else { throw NFKMLXError.unsupportedInput }
            let plate = try NFKMLXImageBridge.tensor(from: value, channels: 3, colorSpace: CGColorSpaceCreateDeviceRGB())
            let (height, width) = (plate.dim(0), plate.dim(1))
            let side = model.vision.configuration.imageSize
            let pixels = (NFKMLXResample.resizeBilinear(plate.expandedDimensions(axis: 0), height: side, width: side)
                          - 0.5) / 0.5

            let context = model.text.configuration.contextLength
            let ids = tokenizer.tokens(for: concept, contextLength: context)
            let real = min(tokenizer.encode(concept).count + 2, context)
            let valid = MLXArray((0 ..< context).map { Float($0 < real ? 1 : 0) }).reshaped([1, context])
            job.reportProgress(0)
            let found = model.detect(image: pixels, tokens: MLXArray(ids.map(Int32.init)).reshaped([1, context]),
                                     valid: valid)
            if job.status == .cancelled { return nil }

            func threshold(_ key: String) -> Float { NFKMLXGenerationBackend.number(request, key)?.floatValue ?? 0.5 }
            let result = try detectionResult(found, label: concept, width: width, height: height,
                                             scoreThreshold: threshold(NFKMLXSAM3ParameterKey.scoreThreshold),
                                             maskThreshold: threshold(NFKMLXSAM3ParameterKey.maskThreshold))
            job.reportProgress(1)
            return result
        }
    }

    /// The instances that pass `scoreThreshold`, as `NFKDetection`s, and the union of their masks at
    /// `width` × `height`, thresholded at `maskThreshold`.
    static func detectionResult(_ found: NFKMLXSAM3Detection, label: String, width: Int, height: Int,
                                scoreThreshold: Float, maskThreshold: Float) throws -> NFKInferenceResult {
        let scores = sigmoid(found.logits[0]) * sigmoid(found.presence.reshaped([-1])[0])
        let boxes = found.boxes[0]
        eval(scores, boxes)
        let scoreValues = scores.asArray(Float.self)
        let boxValues = boxes.asArray(Float.self)
        let kept = scoreValues.indices.filter { scoreValues[$0] > scoreThreshold }
            .sorted { scoreValues[$0] > scoreValues[$1] }
        let detections = kept.map { query -> NFKDetection in
            func coordinate(_ offset: Int) -> CGFloat { CGFloat(min(max(boxValues[query * 4 + offset], 0), 1)) }
            let (minX, minY, maxX, maxY) = (coordinate(0), coordinate(1), coordinate(2), coordinate(3))
            return NFKDetection(label: label, classIndex: 0, confidence: Double(scoreValues[query]),
                                boundingBox: CGRect(x: minX, y: minY, width: max(maxX - minX, 0),
                                                    height: max(maxY - minY, 0)))
        }
        var outputs: [String: Any] = [NFKOutputDetections: detections]
        if !kept.isEmpty {
            // One instance at a time, so a large input never holds every lifted mask at once.
            var union = MLXArray.zeros([height, width, 1])
            for query in kept {
                let probability = sigmoid(found.masks[0, query]).reshaped([1, found.masks.dim(2), found.masks.dim(3), 1])
                let lifted = NFKMLXResample.resizeBilinear(probability, height: height, width: width)[0]
                union = maximum(union, (lifted .> maskThreshold).asType(.float32))
                eval(union)
            }
            outputs[NFKOutputMask] = try NFKMLXImageBridge.cgImage(from: union, options: NFKMLXImageOptions())
        }
        return NFKInferenceResult(outputs: outputs)
    }

    /// Builds the image path from a release directory, which holds `config.json`, the transformers
    /// `model.safetensors`, and the CLIP tokenizer's `vocab.json` and `merges.txt`, and returns it behind
    /// the InferKit contract.
    ///
    /// @discussion The release is one 3.44 GB float32 file, read once. Box prompts and video are not
    /// ported, so the backend takes a worded concept only. `facebook/sam3.1` serves its checkpoint as
    /// the original `sam3.1_multiplex.pt` only, whose names this loader does not read; a 3.1
    /// checkpoint saved in the transformers layout loads here. Introduced in InferKit 0.4.0.
    @objc(backendWithDirectoryURL:error:)
    public static func backend(directoryURL: URL) throws -> any NFKInferenceBackend {
        let model = try makeImageModel(fromHuggingFace: directoryURL.appendingPathComponent("config.json"))
        let files = try NFKMLXReleaseWeights.files(inDirectory: directoryURL)
        guard files.count == 1 else {
            throw NFKMLXError.unsupportedConfiguration("SAM 3 is read from one model.safetensors")
        }
        try loadWeights(into: model, from: files[0])
        return inferenceBackend(model: model, tokenizer: try NFKMLXSDPromptTokenizer(directoryURL: directoryURL))
    }

    /// The files a download fetches.
    static let releaseComponents = [
        NFKMLXReleaseComponent(required: ["config.json", "vocab.json", "merges.txt"],
                               optional: ["special_tokens_map.json", "tokenizer_config.json"],
                               weights: ["model.safetensors"]),
    ]

    /// Downloads the release (``releaseRepo`` when `repo` is nil) and returns the image path behind the
    /// InferKit contract.
    ///
    /// @discussion The download is about 3.5 GB. A file already in the cache is not fetched again. The
    /// call blocks on the network, so run it off the main and render threads. Introduced in InferKit
    /// 0.4.0.
    @objc(backendWithRepo:revision:cacheDirectoryURL:error:)
    public static func backend(repo: String?, revision: String?, cacheDirectoryURL: URL?) throws -> any NFKInferenceBackend {
        try backend(directoryURL: try NFKMLXReleaseDownload.directory(
            repo: repo ?? releaseRepo, revision: revision, cacheDirectoryURL: cacheDirectoryURL,
            components: releaseComponents))
    }

    /// The asynchronous form of ``backend(repo:revision:cacheDirectoryURL:)``. The handler runs on a
    /// background queue. Introduced in InferKit 0.4.0.
    @objc(backendWithRepo:revision:cacheDirectoryURL:completionHandler:)
    public static func backend(repo: String?, revision: String?, cacheDirectoryURL: URL?,
                               completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXReleaseDownload.async(completionHandler) {
            try backend(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL)
        }
    }

    /// Registers the backend under ``modelName``. The registry's URL is the release directory; nil is
    /// refused. Introduced in InferKit 0.4.0.
    @objc public static func register() {
        NFKMLXGenerationBackend.registerDirectoryModel(modelName) { try backend(directoryURL: $0) }
    }
}
