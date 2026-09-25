//
//  NFKMLXSa2VABackend.swift
//  InferKitMLX
//

// The consumer surface for Sa2VA-4B: the weight loader that partitions the released four-shard
// checkpoint across the InternViT tower, the projector, the Qwen2.5 decoder, the `[SEG]` bridge, and
// the SAM 2 grounding encoder; the processor (InternVL dynamic tiling for the understanding stream and
// the 1024-square grounding image for SAM 2); the tokenizer; the directory `@objc` factory; and an
// NFKInferenceBackend that runs one image plus a referring prompt to text (`NFKOutputText`) and, when
// the decoder emits `[SEG]`, a segmentation mask (`NFKOutputMask`).

import Foundation
import CoreGraphics
import InferKit
import MLX
import MLXNN

// MARK: - Configuration parsing + weight loading

extension NFKMLXSa2VANet {

    /// Reads `config.json` from a released Sa2VA directory. The decoder geometry comes from the nested
    /// `llm_config` (a plain qwen2 block); the vision and projector fields from `vision_config` and the
    /// top-level `downsample_ratio`.
    public static func configuration(fromDirectory directory: URL) throws -> NFKMLXSa2VAConfiguration {
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw NFKMLXError.unsupportedConfiguration("Sa2VA config.json is not an object")
        }
        let vision = (json["vision_config"] as? [String: Any]) ?? [:]
        guard let llm = json["llm_config"] as? [String: Any] else {
            throw NFKMLXError.unsupportedConfiguration("Sa2VA config.json has no llm_config")
        }
        func vInt(_ key: String, _ fallback: Int) -> Int { (vision[key] as? NSNumber)?.intValue ?? fallback }
        func vReal(_ key: String, _ fallback: Float) -> Float { (vision[key] as? NSNumber)?.floatValue ?? fallback }
        // A Sa2VA turn is at most 13 tiles of 256 image tokens plus its text and the reply, far below
        // the 32,768 positions past which the dynamic NTK rotary of InternLM2 and of InternVL3's Qwen2
        // would rescale.
        let decoder = try NFKMLXLanguage.configuration(
            fromJSON: NFKMLXInternLM2.describes(llm)
                ? NFKMLXInternLM2.denseConfigurationJSON(llm, sequencesStayBelowWindow: true)
                : NFKMLXRoPEScaling.droppingDynamic(llm))
        let imageSize = (json["force_image_size"] as? NSNumber)?.intValue ?? vInt("image_size", 448)
        var configuration = NFKMLXSa2VAConfiguration(
            visionHiddenSize: vInt("hidden_size", 1024),
            visionLayers: vInt("num_hidden_layers", 24),
            visionHeads: vInt("num_attention_heads", 16),
            visionIntermediateSize: vInt("intermediate_size", 4096),
            patchSize: vInt("patch_size", 14),
            imageSize: imageSize,
            visionLayerNormEps: vReal("layer_norm_eps", 1e-6),
            qkvBias: (vision["qkv_bias"] as? NSNumber)?.boolValue ?? true,
            downsampleRatio: (json["downsample_ratio"] as? NSNumber)?.doubleValue ?? 0.5,
            decoderHiddenSize: decoder.hiddenSize,
            decoder: decoder,
            imageContextTokenId: NFKMLXSa2VAProcessor.tokenId("<IMG_CONTEXT>", inDirectory: directory) ?? 151_667,
            segmentationTokenId: NFKMLXSa2VAProcessor.tokenId("[SEG]", inDirectory: directory) ?? 151_674)
        configuration.template = try NFKMLXSa2VATemplate.named(json["template"] as? String)
        configuration.visionRMSNorm = (vision["norm_type"] as? String) == "rms_norm"
        configuration.visionQueryKeyNormalization = (vision["qk_normalization"] as? NSNumber)?.boolValue ?? false
        if let endToken = NFKMLXSa2VAProcessor.endToken(inDirectory: directory),
           let id = NFKMLXSa2VAProcessor.tokenId(endToken, inDirectory: directory) {
            configuration.endTokenId = id
        }
        // A release on InternLM2 ships its SentencePiece model and no tokenizer.json; its special ids
        // come from that tokenizer.
        if !FileManager.default.fileExists(atPath: directory.appendingPathComponent("tokenizer.json").path),
           let tokenizer = try? NFKMLXInternLM2Tokenizer(directory: directory) {
            configuration.imageContextTokenId = tokenizer.id(ofToken: "<IMG_CONTEXT>") ?? configuration.imageContextTokenId
            configuration.segmentationTokenId = tokenizer.id(ofToken: "[SEG]") ?? configuration.segmentationTokenId
            configuration.endTokenId = tokenizer.eosTokenId
        }
        return configuration
    }

    /// Loads the four-shard release. The understanding stream, projector, and `[SEG]` bridge keep their
    /// checkpoint names; the InternViT patch convolution transposes to channels-last; the grounding
    /// subtree strips the `grounding_encoder.sam2_model.` prefix, runs the SAM 2 tracker's own key remap,
    /// applies the tracker's convolution transposes, and is put back under the prefix.
    ///
    /// `dtype` defaults to bfloat16, the dtype the release declares (it stores float32), which halves a
    /// 4B model's resident footprint; float32 computes as the reference's float32 run does.
    public func loadWeights(fromDirectory directory: URL, dtype: DType = .bfloat16) throws {
        // A directory `NFKMLXSa2VA.save` wrote (a fine-tune) holds this module's own names and layout.
        let saved = directory.appendingPathComponent("model.safetensors")
        if FileManager.default.fileExists(atPath: saved.path),
           let checkpoint = try? NFKMLXWeights.loadCheckpoint(url: saved), !checkpoint.needsConvTranspose {
            try NFKMLXWeights.apply(checkpoint.arrays.map { ($0.key, $0.value.asType(dtype)) }, to: self)
            return
        }
        var merged = try NFKMLXReleaseWeights.arrays(inDirectory: directory).map {
            ($0.0, $0.1.asType(dtype))
        }
        if merged.contains(where: { $0.0.hasSuffix(".attention.wqkv.weight") }) {
            merged = NFKMLXInternLM2.denseWeights(merged, prefix: "language_model.",
                                                  configuration: configuration.decoder)
        }
        let groundingPrefix = "grounding_encoder.sam2_model."
        var mapped = [(String, MLXArray)]()
        for (key, value) in merged {
            if key.hasPrefix(groundingPrefix) {
                let inner = String(key.dropFirst(groundingPrefix.count))
                guard let remapped = NFKMLXSAM2.remapTrackerKey(inner) else { continue }
                // Sa2VA's SAM 2 fork renames the memory-encoder ConvNeXt layer scale `gamma` to
                // `g_weight` in the checkpoint; the shared module keeps the original `gamma`.
                let named = remapped.replacingOccurrences(of: ".g_weight", with: ".gamma")
                mapped.append((groundingPrefix + named, NFKMLXSa2VANet.groundingTransposed(named, value)))
            } else if key == "vision_model.embeddings.patch_embedding.weight" {
                mapped.append((key, value.ndim == 4 ? value.transposed(0, 2, 3, 1) : value))
            } else if key.hasPrefix("mlp1.") || key.hasPrefix("text_hidden_fcs.") {
                // The projector and the [SEG] bridge are numeric `nn.Sequential`s; MLX reads a module
                // whose child keys are all integers as an array, so the slots are given names.
                mapped.append((NFKMLXSa2VANet.sequentialName(key), value))
            } else if key.hasPrefix("vision_model.") || key.hasPrefix("language_model.") {
                mapped.append((key, value))
            }
            // Anything else (there is nothing else) is dropped.
        }
        try NFKMLXWeights.apply(mapped, to: self)
    }

    /// The convolution transpose the SAM 2 tracker's loader applies, replayed for the grounding subtree:
    /// the mask decoder's transposed-convolution upscalers are stored `[in, out, kH, kW]`, every other
    /// 4-D tensor is a forward convolution, and the temporal encoding is a parameter, not a weight.
    /// The named key for a numeric `nn.Sequential` slot: `mlp1` is `LayerNorm(0) → Linear(1) → GELU(2)
    /// → Linear(3)`, and `text_hidden_fcs` is `Linear(0) → ReLU(1) → Linear(2)`.
    static func sequentialName(_ key: String) -> String {
        key.replacingOccurrences(of: "mlp1.0.", with: "mlp1.norm.")
            .replacingOccurrences(of: "mlp1.1.", with: "mlp1.fc1.")
            .replacingOccurrences(of: "mlp1.3.", with: "mlp1.fc2.")
            .replacingOccurrences(of: "text_hidden_fcs.0.", with: "text_hidden_fcs.fc1.")
            .replacingOccurrences(of: "text_hidden_fcs.2.", with: "text_hidden_fcs.fc2.")
    }

    static func groundingTransposed(_ remapped: String, _ value: MLXArray) -> MLXArray {
        guard value.ndim == 4 else { return value }
        if remapped == "maskmem_tpos_enc" { return value }
        if remapped.contains("sam_mask_decoder.upscale") { return value.transposed(1, 2, 3, 0) }
        return value.transposed(0, 2, 3, 1)
    }
}

// MARK: - Processor

public enum NFKMLXSa2VAProcessor {
    static let imageNetMean = MLXArray([Float(0.485), 0.456, 0.406]).reshaped([1, 1, 1, 3])
    static let imageNetStd = MLXArray([Float(0.229), 0.224, 0.225]).reshaped([1, 1, 1, 3])
    // The releases' `preprocess_image` builds the ImageNet mean and deviation as bfloat16 tensors, so
    // the grounding image is normalized by the rounded values (0.485 is 0.484375).
    static let groundingMean = MLXArray([Float(0.484375), 0.455078125, 0.40625]).reshaped([1, 1, 1, 3])
    static let groundingStd = MLXArray([Float(0.228515625), 0.2236328125, 0.224609375]).reshaped([1, 1, 1, 3])
    public static let groundingSize = 1024

    /// The id of an added token, read from `tokenizer.json`. Sa2VA adds `<IMG_CONTEXT>` and `[SEG]` past
    /// the Qwen vocabulary; their ids are needed before the tokenizer is built.
    static func tokenId(_ token: String, inDirectory directory: URL) -> Int? {
        guard let data = try? Data(contentsOf: directory.appendingPathComponent("tokenizer.json")),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let added = json["added_tokens"] as? [[String: Any]] else { return nil }
        for entry in added where (entry["content"] as? String) == token {
            return entry["id"] as? Int
        }
        return nil
    }

    /// The tokenizer's end token, `tokenizer_config.json`'s `eos_token` (a string or an added-token
    /// object).
    static func endToken(inDirectory directory: URL) -> String? {
        guard let data = try? Data(contentsOf: directory.appendingPathComponent("tokenizer_config.json")),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
        if let token = json["eos_token"] as? String { return token }
        return (json["eos_token"] as? [String: Any])?["content"] as? String
    }

    /// The channels-first `[tiles, 3, side, side]` pixel tensor for the understanding stream: each tile
    /// ImageNet-normalized, the reference's `T.Normalize`. The reference's `T.Resize` to `side` leaves
    /// a ``dynamicTiles(_:side:minTiles:maxTiles:useThumbnail:)`` tile unchanged; a tile of another size
    /// is resized bilinearly.
    public static func tilePixels(_ tiles: [MLXArray], side: Int) -> MLXArray {
        let normalized = tiles.map { tile -> MLXArray in
            var batched = tile.reshaped([1, tile.dim(0), tile.dim(1), 3])
            if tile.dim(0) != side || tile.dim(1) != side {
                batched = NFKMLXResample.resizeBilinear(batched, height: side, width: side)
            }
            return ((batched - imageNetMean) / imageNetStd).transposed(0, 3, 1, 2)   // [1, 3, side, side]
        }
        return concatenated(normalized, axis: 0)
    }

    /// The channels-first `[1, 3, side, side]` grounding image the grounding encoder reads, the
    /// releases' `DirectResize` then `preprocess_image`: the whole image resized to a `side` square
    /// (SAM 2: 1024, SAM 3: 1008) by PIL's default bicubic filter on 8-bit pixels, scaled to `0…1`,
    /// and normalized by the bfloat16-rounded ImageNet statistics.
    public static func groundingPixels(_ image: Any, side: Int = groundingSize) throws -> MLXArray {
        let (rgb, width, height) = try rgbBytes(image)
        let resized = NFKMLXPILResample.resampled(rgb, width: width, height: height, toWidth: side, toHeight: side,
                                                  filter: .bicubic)
        let pixels = MLXArray(resized.map { Float($0) / 255 }).reshaped([1, side, side, 3])
        return ((pixels - groundingMean) / groundingStd).transposed(0, 3, 1, 2)
    }

    /// The image's 8-bit RGB bytes, row-major, with its width and height.
    static func rgbBytes(_ image: Any) throws -> (bytes: [UInt8], width: Int, height: Int) {
        let (rgba, width, height) = try NFKMLXImageBridge.rgbaBytes(from: image, colorSpace: CGColorSpaceCreateDeviceRGB())
        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        for pixel in 0 ..< width * height {
            rgb[pixel * 3] = rgba[pixel * 4]
            rgb[pixel * 3 + 1] = rgba[pixel * 4 + 1]
            rgb[pixel * 3 + 2] = rgba[pixel * 4 + 2]
        }
        return (rgb, width, height)
    }

    /// InternVL dynamic tiling, the reference's `dynamic_preprocess`: the image is resized to the best
    /// whole-tile aspect ratio (up to `maxTiles` tiles of `side`×`side`) by PIL's bicubic filter on
    /// 8-bit pixels, split into tiles row-major, and a full-frame thumbnail resized the same way is
    /// appended when there is more than one tile. Returns each tile as an `[H, W, 3]` `0…1` tensor.
    public static func dynamicTiles(_ image: Any, side: Int = 448, minTiles: Int = 1,
                                    maxTiles: Int = 12, useThumbnail: Bool = true) throws -> [MLXArray] {
        let (rgb, width, height) = try rgbBytes(image)
        let aspect = Double(width) / Double(height)

        var ratios = Set<[Int]>()
        for n in minTiles ... maxTiles {
            for i in 1 ... n {
                for j in 1 ... n where i * j <= maxTiles && i * j >= minTiles {
                    ratios.insert([i, j])
                }
            }
        }
        let sorted = ratios.sorted { $0[0] * $0[1] < $1[0] * $1[1] }
        var best = [1, 1]
        var bestDiff = Double.greatestFiniteMagnitude
        let area = Double(width * height)
        for ratio in sorted {
            let target = Double(ratio[0]) / Double(ratio[1])
            let diff = abs(aspect - target)
            if diff < bestDiff {
                bestDiff = diff
                best = ratio
            } else if diff == bestDiff, area > 0.5 * Double(side * side) * Double(ratio[0] * ratio[1]) {
                best = ratio
            }
        }

        // The reference resizes with PIL's default filter, bicubic, on the 8-bit image.
        func resized(toWidth: Int, toHeight: Int) -> MLXArray {
            let bytes = NFKMLXPILResample.resampled(rgb, width: width, height: height, toWidth: toWidth,
                                                    toHeight: toHeight, filter: .bicubic)
            return MLXArray(bytes.map { Float($0) / 255 }).reshaped([toHeight, toWidth, 3])
        }
        let targetWidth = side * best[0], targetHeight = side * best[1]
        let grid = resized(toWidth: targetWidth, toHeight: targetHeight)
        let columns = targetWidth / side
        let blocks = best[0] * best[1]
        var tiles = [MLXArray]()
        for index in 0 ..< blocks {
            let x = (index % columns) * side
            let y = (index / columns) * side
            tiles.append(grid[y ..< (y + side), x ..< (x + side), 0...])
        }
        if useThumbnail && blocks != 1 {
            tiles.append(resized(toWidth: side, toHeight: side))
        }
        return tiles
    }

    /// The prompt text with the image placeholder expanded and wrapped in the release's instruction
    /// template. `<image>` in the user text is replaced by `<img>` + `<IMG_CONTEXT>`×n + `</img>`.
    public static func promptText(_ text: String, imageTokens: Int, template: NFKMLXSa2VATemplate = .phi3) -> String {
        let placeholder = "<img>" + String(repeating: "<IMG_CONTEXT>", count: imageTokens) + "</img>\n"
        let body = text.contains("<image>")
            ? text.replacingOccurrences(of: "<image>", with: placeholder)
            : placeholder + text
        return template.formatted(body)
    }
}

// MARK: - Backend

@objc(NFKMLXSa2VA)
public final class NFKMLXSa2VA: NSObject {

    @objc public static let modelName = "sa2va-4b"
    static let requiredFiles = ["config.json", "tokenizer_config.json"]
    static let optionalFiles = ["tokenizer.json", "tokenizer.model", "vocab.json", "merges.txt", "added_tokens.json",
                                "preprocessor_config.json"]
    static let weightFiles = ["model.safetensors", "model.safetensors.index.json"]

    /// The maximum tokens generated after the prompt.
    @objc public static var maximumTokens = 256

    /// Builds a backend from a released Sa2VA directory (weights + tokenizer + `config.json`). The
    /// configuration's `architectures` picks the family: `Sa2VAChatModel` is InternVL-based (the 1B, 4B,
    /// 8B, 26B, and InternVL3 releases), `Sa2VAChatModelQwen` wraps Qwen3-VL or Qwen2.5-VL, and
    /// `Sa2VAChatModelLlava` wraps LLaVA-1.5.
    @objc(backendWithDirectoryURL:error:)
    public static func backend(directoryURL: URL) throws -> NFKMLXSa2VABackend {
        if try architecture(inDirectory: directoryURL) == "Sa2VAChatModelLlava" {
            return NFKMLXSa2VABackend(llava: try NFKMLXSa2VALLaVANet.load(directoryURL: directoryURL),
                                      tokenizer: try NFKMLXInternLM2Tokenizer(directory: directoryURL, decoding: .llamaFast))
        }
        guard let tokenizer = tokenizer(inDirectory: directoryURL) else {
            throw NFKMLXError.unsupportedConfiguration("Sa2VA tokenizer files are missing or unreadable")
        }
        if try architecture(inDirectory: directoryURL) == "Sa2VAChatModelQwen" {
            return NFKMLXSa2VABackend(qwen: try NFKMLXSa2VAQwenNet.load(directoryURL: directoryURL, dtype: .bfloat16),
                                      tokenizer: tokenizer)
        }
        let configuration = try NFKMLXSa2VANet.configuration(fromDirectory: directoryURL)
        let net = NFKMLXSa2VANet(configuration)
        try net.loadWeights(fromDirectory: directoryURL)
        return NFKMLXSa2VABackend(net: net, tokenizer: tokenizer)
    }

    /// The release's tokenizer: its `tokenizer.json`, or an InternLM2 release's SentencePiece model.
    public static func tokenizer(inDirectory directory: URL) -> NFKTokenizer? {
        if let tokenizer = NFKMLXLanguage.releaseTokenizer(inDirectory: directory) { return tokenizer }
        return try? NFKMLXInternLM2Tokenizer(directory: directory)
    }

    /// The first of `config.json`'s `architectures`.
    static func architecture(inDirectory directory: URL) throws -> String? {
        let data = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        return (json?["architectures"] as? [String])?.first
    }

    /// Downloads a release into the hub cache and builds the backend.
    ///
    /// @discussion The download fetches `config.json`, `tokenizer_config.json`, the tokenizer files the
    /// repo serves (`tokenizer.json`, or an InternLM2 release's `tokenizer.model`; `vocab.json`,
    /// `merges.txt`, `added_tokens.json`, `preprocessor_config.json`), and the weights, single-file or
    /// sharded. The grounding encoder is part of the release's own checkpoint, so one repo holds the
    /// whole model. A file the cache already holds is not fetched again. The call blocks on the network
    /// for several gigabytes on a first download; call it off the render thread. The InternVL releases
    /// are `ByteDance/Sa2VA-1B`, `-4B`, `-8B`, `-26B`, and `-InternVL3-2B` / `-8B` / `-14B`; the Qwen-VL
    /// releases are `ByteDance/Sa2VA-Qwen3-VL-2B`, `-4B`, `-4B-SAM3`, and `-Qwen2_5-VL-3B` / `-7B`; the
    /// LLaVA release is `ByteDance/Sa2VA-LLaVA-1.5-7B`. None is gated.
    ///
    /// Introduced in InferKit 0.4.0.
    @objc(backendWithRepo:revision:cacheDirectoryURL:error:)
    public static func backend(repo: String, revision: String?, cacheDirectoryURL: URL?) throws -> NFKMLXSa2VABackend {
        try backend(directoryURL: try NFKMLXReleaseDownload.directory(
            repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL,
            required: requiredFiles, optional: optionalFiles, weights: weightFiles))
    }

    /// The asynchronous form of ``backend(repo:revision:cacheDirectoryURL:)``. The download and the
    /// build run at user-initiated quality of service off the calling thread.
    ///
    /// Introduced in InferKit 0.4.0.
    @objc(backendWithRepo:revision:cacheDirectoryURL:completionHandler:)
    public static func backend(repo: String, revision: String?, cacheDirectoryURL: URL?,
                               completionHandler: @escaping (NFKMLXSa2VABackend?, Error?) -> Void) {
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                completionHandler(try backend(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL), nil)
            } catch {
                completionHandler(nil, error)
            }
        }
    }
}

/// The Sa2VA inference backend: an image plus a referring prompt in, the decoder's text under
/// `NFKOutputText` and, when it emits `[SEG]`, the first object's mask under `NFKOutputMask`.
public final class NFKMLXSa2VABackend: NSObject, NFKInferenceBackend {
    private let holder: Holder
    private let tokenizer: NFKTokenizer

    final class Holder: @unchecked Sendable {
        let net: NFKMLXSa2VANet?
        let qwen: NFKMLXSa2VAQwenNet?
        let llava: NFKMLXSa2VALLaVANet?
        init(net: NFKMLXSa2VANet? = nil, qwen: NFKMLXSa2VAQwenNet? = nil, llava: NFKMLXSa2VALLaVANet? = nil) {
            self.net = net
            self.qwen = qwen
            self.llava = llava
        }
    }

    init(net: NFKMLXSa2VANet, tokenizer: NFKTokenizer) {
        holder = Holder(net: net)
        self.tokenizer = tokenizer
    }

    init(qwen: NFKMLXSa2VAQwenNet, tokenizer: NFKTokenizer) {
        holder = Holder(qwen: qwen)
        self.tokenizer = tokenizer
    }

    init(llava: NFKMLXSa2VALLaVANet, tokenizer: NFKTokenizer) {
        holder = Holder(llava: llava)
        self.tokenizer = tokenizer
    }

    public var supportedInputKeys: Set<String> { [NFKInputImage, NFKInputPrompt] }
    public var isReady: Bool { true }
    public var backendIdentifier: String { NFKMLXSa2VA.modelName }

    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        guard let imageValue = request.input(forKey: NFKInputImage) else {
            throw NFKMLXError.unsupportedConfiguration("Sa2VA needs an NFKInputImage")
        }
        let prompt = (request.input(forKey: NFKInputPrompt) as? String)
            ?? "<image>Please segment the main object."
        if let qwen = holder.qwen {
            return try runQwen(qwen, image: imageValue, prompt: prompt)
        }
        if let llava = holder.llava {
            return try runLLaVA(llava, image: imageValue, prompt: prompt)
        }
        guard let net = holder.net else { throw NFKMLXError.noOutput }
        let configuration = net.configuration

        let tiles = try NFKMLXSa2VAProcessor.dynamicTiles(imageValue, side: configuration.imageSize)
        let pixelValues = NFKMLXSa2VAProcessor.tilePixels(tiles, side: configuration.imageSize)
        let imageTokens = tiles.count * configuration.tokensPerTile

        let text = NFKMLXSa2VAProcessor.promptText(prompt, imageTokens: imageTokens, template: configuration.template)
        let inputIds = tokenizer.encode(text).map { Int(truncating: $0) }

        let features = net.imageFeatures(pixelValues: pixelValues)
        let tokenizer = self.tokenizer
        let template = configuration.template
        let generated = net.generate(inputIds: inputIds, imageFeatures: features,
                                     maximumTokens: NFKMLXSa2VA.maximumTokens, endToken: configuration.endTokenId,
                                     stopsAfter: { tokens in
                                         template.stops(after: tokens) { tokenizer.decode($0.map { NSNumber(value: $0) }) }
                                     })
        let answer = Self.answer(generated.tokens, ending: generated.ending, tokenizer: tokenizer)

        var outputs: [String: Any] = [NFKOutputText: answer]
        if let segHidden = generated.segmentationHiddenStates.first {
            let groundingImage = try NFKMLXSa2VAProcessor.groundingPixels(imageValue)
                .transposed(0, 2, 3, 1)                                        // [1, 1024, 1024, 3]
            let mask = net.segment(segHidden: segHidden, groundingImage: groundingImage)
            outputs[NFKOutputMask] = NFKMLXSa2VABackend.maskImage(mask.highResolution)
        }
        return NFKInferenceResult(outputs: outputs)
    }

    /// The Qwen-VL releases' turn: the image at the reference's pixel bounds, the chat template's user
    /// turn (any `<image>` marker dropped, as `predict_forward` drops it), greedy generation to
    /// `<|im_end|>` or `<|endoftext|>`, and the first `[SEG]`'s mask.
    private func runQwen(_ net: NFKMLXSa2VAQwenNet, image imageValue: Any, prompt: String) throws -> NFKInferenceResult {
        let cf = imageValue as CFTypeRef
        guard CFGetTypeID(cf) == CGImage.typeID else {
            throw NFKMLXError.unsupportedConfiguration("the Qwen-VL Sa2VA releases read a CGImage")
        }
        let (pixelValues, grid) = net.imageProcessor.process(cf as! CGImage)
        let features = net.imageFeatures(pixelValues: pixelValues, grid: grid)
        let text = net.promptText(prompt.replacingOccurrences(of: "<image>", with: ""),
                                  imageTokens: grid.h * grid.w / 4)
        let inputIds = tokenizer.encode(text).map { Int(truncating: $0) }
        let generated = net.generate(inputIds: inputIds, features: features, grid: grid,
                                     maximumTokens: NFKMLXSa2VA.maximumTokens, endTokens: [151_645, 151_643])
        let answer = Self.answer(generated.tokens, ending: generated.ending, tokenizer: tokenizer)
        var outputs: [String: Any] = [NFKOutputText: answer]
        if let segHidden = generated.segmentationHiddenStates.first {
            let groundingImage = try NFKMLXSa2VAProcessor.groundingPixels(imageValue, side: net.groundingSize)
                .transposed(0, 2, 3, 1)
            outputs[NFKOutputMask] = NFKMLXSa2VABackend.maskImage(net.segment(segHidden: segHidden,
                                                                              groundingImage: groundingImage).highResolution)
        }
        return NFKInferenceResult(outputs: outputs)
    }

    /// The LLaVA release's turn: the image resized to 336 (PIL bicubic) and CLIP-normalized, the Vicuna
    /// instruction with 576 `<image>` tokens, greedy generation to `</s>`, and the first `[SEG]`'s mask.
    private func runLLaVA(_ net: NFKMLXSa2VALLaVANet, image imageValue: Any, prompt: String) throws -> NFKInferenceResult {
        let pixels = try NFKMLXSa2VALLaVA.pixelValues(imageValue, side: net.visionConfiguration.imageSize)
        let features = net.imageFeatures(pixels)
        let grid = net.visionConfiguration.grid
        let text = NFKMLXSa2VALLaVA.promptText(prompt, imageTokens: grid * grid)
        let inputIds = tokenizer.encode(text).map { Int(truncating: $0) }
        let generated = net.generate(inputIds: inputIds, imageFeatures: features,
                                     maximumTokens: NFKMLXSa2VA.maximumTokens, endToken: tokenizer.eosTokenId)
        let answer = Self.answer(generated.tokens, ending: generated.ending, tokenizer: tokenizer)
        var outputs: [String: Any] = [NFKOutputText: answer]
        if let segHidden = generated.segmentationHiddenStates.first {
            let groundingImage = try NFKMLXSa2VAProcessor.groundingPixels(imageValue).transposed(0, 2, 3, 1)
            outputs[NFKOutputMask] = NFKMLXSa2VABackend.maskImage(net.segment(segHidden: segHidden,
                                                                              groundingImage: groundingImage).highResolution)
        }
        return NFKInferenceResult(outputs: outputs)
    }

    /// The reference's answer: the generated tokens, with the end token generation stopped on, decoded
    /// with their special tokens and trimmed (`predict_forward` decodes `generate`'s output, which keeps
    /// the end token).
    static func answer(_ tokens: [Int], ending: Int?, tokenizer: NFKTokenizer) -> String {
        let sequence = tokens + [ending].compactMap { $0 }
        return tokenizer.decode(sequence.map { NSNumber(value: $0) }).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// A binary mask CGImage from `[1, side, side]` logits, thresholded at zero (sigmoid > 0.5).
    static func maskImage(_ logits: MLXArray) -> CGImage? {
        let side = logits.dim(1)
        let binary = (logits[0] .> 0)
        let bytes = MLX.where(binary, MLXArray(UInt8(255)), MLXArray(UInt8(0))).asType(.uint8)
        eval(bytes)
        let raw = bytes.asArray(UInt8.self)
        guard let provider = CGDataProvider(data: Data(raw) as CFData) else { return nil }
        return CGImage(width: side, height: side, bitsPerComponent: 8, bitsPerPixel: 8,
                       bytesPerRow: side, space: CGColorSpaceCreateDeviceGray(),
                       bitmapInfo: CGBitmapInfo(rawValue: 0), provider: provider,
                       decode: nil, shouldInterpolate: false, intent: .defaultIntent)
    }
}
