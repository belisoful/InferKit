//
//  NFKMLXSa2VAQwen.swift
//  InferKitMLX
//
//  Sa2VA on Qwen-VL: transformers' Qwen3-VL (`ByteDance/Sa2VA-Qwen3-VL-2B`, `-4B`) or Qwen2.5-VL
//  (`-Qwen2_5-VL-3B`, `-7B`) whole — the vision tower and the M-RoPE decoder — under the checkpoint's
//  `model.`, the `[SEG]` bridge, and the SAM 2 Hiera-Large grounding encoder. The VLM half is the
//  toolkit's own Qwen3-VL or Qwen2.5-VL network, loaded under that outer prefix; the bridge and the
//  grounding are Sa2VA-4B's.
//

import Foundation
import CoreGraphics
import InferKit
import MLX
import MLXNN

/// Sa2VA on Qwen-VL: the Qwen3-VL or Qwen2.5-VL vision tower and decoder, the `[SEG]` bridge, and SAM 2
/// grounding.
///
/// @discussion The reference formats the prompt with the release's chat template, generates greedily
/// with the VLM's own `generate`, and drives SAM 2 from the decoder's final hidden state at each `[SEG]`
/// it emitted, exactly as the InternVL releases do. Qwen2.5-VL's tower has no deepstack, and its
/// decoder lays M-RoPE out in contiguous sections.
///
/// Introduced in InferKit 0.4.0.
public final class NFKMLXSa2VAQwenNet: Module {
    @ModuleInfo(key: "text_hidden_fcs") var textHiddenFCS: NFKSa2VATextHiddenFCS
    @ModuleInfo(key: "grounding_encoder") var grounding: NFKSa2VAGrounding

    enum Tower {
        case qwen3(NFKMLXQwen3VLVisionNet)
        case qwen25(NFKMLXQwen25VLVisionNet)
    }

    let tower: Tower
    /// The tower as a child module, so the parameter tree, and a saved fine-tune, carry it.
    let visual: Module
    /// The VLM decoder.
    public let decoder: NFKMLXLanguageNet
    /// The `[SEG]` id, read from the release's `tokenizer.json`.
    public let segmentationTokenId: Int
    /// The decoder's M-RoPE layout: Qwen3-VL's interleaved, or Qwen2.5-VL's contiguous sections.
    public let layout: NFKMLXMRoPELayout

    init(tower: Tower, decoder: NFKMLXLanguageNet, segmentationTokenId: Int, layout: NFKMLXMRoPELayout,
         groundsWithSAM3: Bool) {
        self.tower = tower
        switch tower {
        case .qwen3(let vision): visual = vision
        case .qwen25(let vision): visual = vision
        }
        self.decoder = decoder
        self.segmentationTokenId = segmentationTokenId
        self.layout = layout
        let grounding = NFKMLXSAM2TrackerConfiguration.geometry(.large, release: .sam2)
        _textHiddenFCS.wrappedValue = NFKSa2VATextHiddenFCS(inputSize: decoder.configuration.hiddenSize,
                                                           outputSize: grounding.hiddenDimensions)
        _grounding.wrappedValue = groundsWithSAM3 ? NFKSa2VASAM3GroundingEncoder() : NFKSa2VAGroundingEncoder(grounding)
        super.init()
    }

    /// The side the grounding image is resized to: 1024 for SAM 2, 1008 for SAM 3.
    public var groundingSize: Int { grounding.imageSize }

    /// Builds the network from a released directory at float32: the VLM half from its nested `model.`
    /// subtree (the tower chosen by `vision_config.model_type`), the bridge, and the grounding encoder.
    public static func load(directoryURL: URL) throws -> NFKMLXSa2VAQwenNet {
        try load(directoryURL: directoryURL, parts: .all, dtype: .float32)
    }

    /// Builds the network with its weights at `dtype`. The releases store float32 and their `text_config`
    /// declares bfloat16; ``NFKMLXSa2VA/backend(directoryURL:)`` loads bfloat16, which halves a 4B release's
    /// resident weights. Parity is measured at float32.
    public static func load(directoryURL: URL, dtype: DType) throws -> NFKMLXSa2VAQwenNet {
        try load(directoryURL: directoryURL, parts: .all, dtype: dtype)
    }

    /// The parts of a release a load reads.
    struct Parts: OptionSet {
        let rawValue: Int
        /// The vision tower and the decoder.
        static let language = Parts(rawValue: 1)
        /// The `[SEG]` bridge and the grounding encoder.
        static let grounding = Parts(rawValue: 2)
        static let all: Parts = [.language, .grounding]
    }

    /// Builds the network, reading only `parts` of a release. A part left out keeps its initialization
    /// unevaluated, which holds no memory, so a float32 4B release's decoder and its SAM 3 grounding can be
    /// measured one at a time. A saved fine-tune loads whole.
    static func load(directoryURL: URL, parts: Parts, dtype: DType = .float32) throws -> NFKMLXSa2VAQwenNet {
        let data = try Data(contentsOf: directoryURL.appendingPathComponent("config.json"))
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
        let visionJSON = json["vision_config"] as? [String: Any] ?? [:]
        // A directory `NFKMLXSa2VA.save` wrote (a fine-tune) holds this module's own names and layout.
        let savedURL = directoryURL.appendingPathComponent("model.safetensors")
        var saved = FileManager.default.fileExists(atPath: savedURL.path)
            ? (try? NFKMLXWeights.loadCheckpoint(url: savedURL)).flatMap { $0.needsConvTranspose ? nil : $0 } : nil
        let tower: Tower
        var layout = NFKMLXMRoPELayout.interleaved
        if (visionJSON["model_type"] as? String) == "qwen2_5_vl" {
            let configuration = NFKMLXQwen25VLVisionConfiguration(json: visionJSON)
            tower = .qwen25(saved != nil || !parts.contains(.language)
                ? NFKMLXQwen25VLVisionNet(configuration)
                : try NFKMLXQwen25VLVisionNet.load(directoryURL: directoryURL, configuration: configuration,
                                                   outerPrefix: "model.", dtype: dtype))
            let scaling = (json["text_config"] as? [String: Any])?["rope_scaling"] as? [String: Any]
            layout = .chunked((scaling?["mrope_section"] as? [NSNumber])?.map(\.intValue) ?? [16, 24, 24])
        } else {
            let vision = NFKMLXQwen3VLVisionNet(try NFKMLXQwen3VLVisionConfiguration.configuration(
                fromHuggingFace: directoryURL.appendingPathComponent("config.json")))
            if saved == nil && parts.contains(.language) {
                try NFKMLXQwen3VL.loadVisionWeights(into: vision, directoryURL: directoryURL, outerPrefix: "model.",
                                                    dtype: dtype)
            }
            tower = .qwen3(vision)
        }
        // A saved fine-tune keeps the head under the module's own name and has no shard index.
        let decoder = NFKMLXLanguageNet(try NFKMLXQwen3VL.decoderConfiguration(
            directoryURL: directoryURL, outerPrefix: "model.",
            shipsHead: saved.map { $0.arrays["decoder.lm_head.weight"] != nil }))
        if saved == nil && parts.contains(.language) {
            // The dense decoder loads whole under any plan; `.resident` would only refuse a float32 4B
            // release, which exceeds a 32 GB machine's recommended working set.
            try NFKMLXQwen3VL.loadDecoderWeights(into: decoder, fromDirectory: directoryURL, precision: .float32,
                                                 residency: .automatic, outerPrefix: "model.", dtype: dtype)
        }
        let groundsWithSAM3 = try saved.map { $0.arrays.keys.contains { $0.hasPrefix("grounding_encoder.neck.") } }
            ?? (NFKMLXReleaseWeights.arrays(inDirectory: directoryURL) { key in
                key.hasPrefix("grounding_encoder.sam2_model.backbone.vision_backbone.trunk.pos_embed") ? key : nil
            }.isEmpty == false)
        let net = NFKMLXSa2VAQwenNet(
            tower: tower, decoder: decoder,
            segmentationTokenId: NFKMLXSa2VAProcessor.tokenId("[SEG]", inDirectory: directoryURL) ?? 151_674,
            layout: layout, groundsWithSAM3: groundsWithSAM3)
        if saved != nil {
            // The checkpoint is dropped before the converted tensors evaluate, so it is not held beside them.
            let typed = saved!.arrays.map { ($0.key, $0.value.asType(dtype)) }
            saved = nil
            try NFKMLXWeights.apply(typed, to: net)
        } else if parts.contains(.grounding) {
            try net.loadBridgeAndGrounding(fromDirectory: directoryURL, dtype: dtype)
        }
        return net
    }

    /// Loads `text_hidden_fcs.*` and `grounding_encoder.sam2_model.*`: SAM 2 through Sa2VA-4B's remaps,
    /// SAM 3 through its own.
    func loadBridgeAndGrounding(fromDirectory directory: URL, dtype: DType = .float32) throws {
        let prefix = "grounding_encoder.sam2_model."
        let arrays = try NFKMLXReleaseWeights.arrays(inDirectory: directory, converting: dtype) { key in
            key.hasPrefix("text_hidden_fcs.") || key.hasPrefix(prefix) ? key : nil
        }
        // Each part loads into its own module with its own coverage check: the network also holds the
        // tower and the decoder, which their own loaders fill.
        let bridge = arrays.filter { $0.0.hasPrefix("text_hidden_fcs.") }.map {
            (String(NFKMLXSa2VANet.sequentialName($0.0).dropFirst("text_hidden_fcs.".count)), $0.1)
        }
        try NFKMLXWeights.apply(bridge, to: textHiddenFCS)
        if let sam3 = grounding as? NFKSa2VASAM3GroundingEncoder {
            try sam3.load(arrays, dtype: dtype)
            return
        }
        var mapped = [(String, MLXArray)]()
        for (key, value) in arrays where key.hasPrefix(prefix) {
            guard let remapped = NFKMLXSAM2.remapTrackerKey(String(key.dropFirst(prefix.count))) else { continue }
            let named = remapped.replacingOccurrences(of: ".g_weight", with: ".gamma")
            mapped.append(("sam2_model." + named, NFKMLXSa2VANet.groundingTransposed(named, value)))
        }
        try NFKMLXWeights.apply(mapped, to: grounding)
    }

    /// The merged vision tokens and the deepstack (empty for Qwen2.5-VL) for processed pixels over `grid`.
    public func imageFeatures(pixelValues: MLXArray, grid: (t: Int, h: Int, w: Int))
        -> (output: MLXArray, deepstack: [MLXArray]) {
        switch tower {
        case .qwen3(let vision): return vision(pixelValues, grid: grid)
        case .qwen25(let vision): return (vision(pixelValues, grid: grid), [])
        }
    }

    /// The image processor the reference's `predict_forward` runs: the release family's patch size and
    /// normalization at `512·28²` to `2048·28²` pixels.
    public var imageProcessor: NFKMLXQwen3VLImageProcessor {
        switch tower {
        case .qwen3: return NFKMLXSa2VAQwen.imageProcessor
        case .qwen25: return NFKMLXSa2VAQwen.qwen25ImageProcessor
        }
    }

    /// The decoder's final hidden states over a fused sequence, `[1, sequence, hidden]`.
    public func hiddenStates(inputIds: [Int], features: (output: MLXArray, deepstack: [MLXArray]),
                             grid: (t: Int, h: Int, w: Int)) -> MLXArray {
        NFKMLXQwen3VL.hiddenStates(decoder: decoder, inputIds: inputIds, visionFeatures: features.output,
                                   deepstack: features.deepstack, gridT: grid.t, gridH: grid.h, gridW: grid.w,
                                   layout: layout)
    }

    /// The release's prompt for one image of `imageTokens` merged tokens: Qwen2.5-VL's template opens
    /// with its default system turn, and Qwen3-VL's opens with the user turn.
    public func promptText(_ text: String, imageTokens: Int) -> String {
        NFKMLXSa2VAQwen.promptText(text, imageTokens: imageTokens,
                                   systemTurn: layout == .interleaved ? "" : NFKMLXSa2VAQwen.qwen25SystemTurn)
    }

    /// Whether the `[SEG]` bridge reads the last decoder layer's output before the final norm. Sa2VA
    /// reads `hidden_states[-1]`, in training and inference alike. transformers records Qwen3-VL's
    /// hidden states layer by layer and leaves the last one un-normalized, while Qwen2.5-VL's last entry
    /// is normalized, so this is true for Qwen3-VL and false for Qwen2.5-VL.
    public var segmentationReadsBeforeFinalNorm: Bool { layout == .interleaved }

    /// The states the `[SEG]` bridge reads over a fused sequence, `[1, sequence, hidden]`: the reference's
    /// `hidden_states[-1]` (see ``segmentationReadsBeforeFinalNorm``).
    public func segmentationStates(inputIds: [Int], features: (output: MLXArray, deepstack: [MLXArray]),
                                   grid: (t: Int, h: Int, w: Int)) -> MLXArray {
        NFKMLXQwen3VL.hiddenStates(decoder: decoder, inputIds: inputIds, visionFeatures: features.output,
                                   deepstack: features.deepstack, gridT: grid.t, gridH: grid.h, gridW: grid.w,
                                   applyFinalNorm: !segmentationReadsBeforeFinalNorm, layout: layout)
    }

    /// Greedy generation, then the decoder's hidden state at each `[SEG]` of the realized sequence,
    /// read from one teacher-forced pass, as the InternVL releases read theirs. `ending` is the end token
    /// generation stopped on, which the reference's decoded answer keeps; nil when it ran to the budget.
    public func generate(inputIds: [Int], features: (output: MLXArray, deepstack: [MLXArray]),
                         grid: (t: Int, h: Int, w: Int), maximumTokens: Int, endTokens: Set<Int>)
        -> (tokens: [Int], ending: Int?, segmentationHiddenStates: [MLXArray]) {
        let (tokens, ending) = NFKMLXQwen3VL.generateToEnd(
            decoder: decoder, inputIds: inputIds, visionFeatures: features.output, deepstack: features.deepstack,
            gridT: grid.t, gridH: grid.h, gridW: grid.w, maxTokens: maximumTokens, endTokens: endTokens,
            layout: layout)
        let full = inputIds + tokens
        guard full.contains(segmentationTokenId) else { return (tokens, ending, []) }
        let hidden = segmentationStates(inputIds: full, features: features, grid: grid)
        let segments = full.enumerated().filter { $0.element == segmentationTokenId }.map { hidden[0, $0.offset] }
        return (tokens, ending, segments)
    }

    /// A mask from a `[SEG]` hidden state and the normalized SAM input `[1, 1024, 1024, 3]`.
    public func segment(segHidden: MLXArray, groundingImage: MLXArray)
        -> (highResolution: MLXArray, lowResolution: MLXArray) {
        let embedding = textHiddenFCS(segHidden.reshaped([1, -1]))
        return grounding.segment(image: groundingImage, languageEmbedding: embedding.reshaped([1, 1, -1]))
    }
}

/// The Qwen3-VL releases' processing: the pixel bounds `predict_forward` passes and the release chat
/// template's single-image user turn.
public enum NFKMLXSa2VAQwen {
    /// The Qwen3-VL image processor at the bounds the reference passes (`512·28²` to `2048·28²` pixels).
    public static let imageProcessor = NFKMLXQwen3VLImageProcessor(minPixels: 512 * 28 * 28,
                                                                   maxPixels: 2048 * 28 * 28)

    /// The Qwen2.5-VL image processor at the same bounds: 14-pixel patches and CLIP's normalization.
    public static let qwen25ImageProcessor = NFKMLXQwen3VLImageProcessor(
        patchSize: 14, minPixels: 512 * 28 * 28, maxPixels: 2048 * 28 * 28,
        mean: [0.48145466, 0.4578275, 0.40821073], std: [0.26862954, 0.26130258, 0.27577711])

    /// The system turn Qwen2.5-VL's chat template opens a conversation with when it has none. Qwen3-VL's
    /// template adds no system turn.
    public static let qwen25SystemTurn = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"

    /// The chat template's user turn with one image of `imageTokens` merged tokens, and the assistant
    /// prompt that follows it, after `systemTurn`.
    public static func promptText(_ text: String, imageTokens: Int, systemTurn: String = "") -> String {
        systemTurn + "<|im_start|>user\n<|vision_start|>" + String(repeating: "<|image_pad|>", count: imageTokens)
            + "<|vision_end|>" + text + "<|im_end|>\n<|im_start|>assistant\n"
    }
}
