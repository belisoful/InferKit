//
//  NFKMLXLTX2Connectors.swift
//  InferKitMLX
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import InferKit
import MLX
import MLXNN

// The LTX-2 text connectors (`LTX2TextConnectors`, Lightricks), the stage between the Gemma text
// encoder and the transformer. The encoder's hidden state at EVERY layer, the embedding included,
// arrives stacked on a trailing axis and flattened hidden-major. Each token's state is RMS-normalized
// over the hidden axis per layer (eps 1e-6, no weight), scaled by `sqrt(modality width / hidden)`, and
// projected separately for the video and the audio streams (LTX-2.3's per-modality projections).
//
// Each modality then runs its own small 1-D transformer. The valid tokens move to the FRONT in their
// original order and the rest of the sequence fills with learnable registers indexed by absolute
// position, so after the fill nothing is masked and the transformer's text mask is all ones. The blocks
// are pre-normalized (affine-free RMS norms): a gated attention with the across-heads query and key norm
// and a split rotary at positions `i / 4096`, and a tanh-GELU feed-forward; an affine-free RMS norm
// closes the stack. The attention and feed-forward are the transformer's own (`NFKLTX2Attention`,
// `NFKLTX2FeedForward`).

/// The geometry of a release's `connectors/config.json`.
public struct NFKMLXLTX2ConnectorConfiguration: Sendable {
    /// The text encoder's hidden width.
    public var captionChannels: Int
    /// The hidden states stacked per token: the encoder's layers plus its embedding.
    public var layerCount: Int
    public var videoHeads: Int
    public var videoHeadDim: Int
    public var videoLayers: Int
    public var videoRegisters: Int
    public var audioHeads: Int
    public var audioHeadDim: Int
    public var audioLayers: Int
    public var audioRegisters: Int
    public var ropeBaseLength: Int
    public var ropeTheta: Float
    public var videoWidth: Int
    public var audioWidth: Int
    public var projectionBias: Bool

    public init(captionChannels: Int = 3840, layerCount: Int = 49, videoHeads: Int = 32, videoHeadDim: Int = 128,
                videoLayers: Int = 8, videoRegisters: Int = 128, audioHeads: Int = 32, audioHeadDim: Int = 64,
                audioLayers: Int = 8, audioRegisters: Int = 128, ropeBaseLength: Int = 4096,
                ropeTheta: Float = 10000, videoWidth: Int = 4096, audioWidth: Int = 2048,
                projectionBias: Bool = true) {
        self.captionChannels = captionChannels
        self.layerCount = layerCount
        self.videoHeads = videoHeads
        self.videoHeadDim = videoHeadDim
        self.videoLayers = videoLayers
        self.videoRegisters = videoRegisters
        self.audioHeads = audioHeads
        self.audioHeadDim = audioHeadDim
        self.audioLayers = audioLayers
        self.audioRegisters = audioRegisters
        self.ropeBaseLength = ropeBaseLength
        self.ropeTheta = ropeTheta
        self.videoWidth = videoWidth
        self.audioWidth = audioWidth
        self.projectionBias = projectionBias
    }

    /// The LTX-2.3 release's connectors (`diffusers/LTX-2.3-Diffusers`, `connectors/config.json`).
    public static let ltx23 = NFKMLXLTX2ConnectorConfiguration()

    /// The release's arrangement at a tiny width, for weight-free tests.
    public static let tiny = NFKMLXLTX2ConnectorConfiguration(
        captionChannels: 12, layerCount: 3, videoHeads: 2, videoHeadDim: 8, videoLayers: 2, videoRegisters: 4,
        audioHeads: 2, audioHeadDim: 4, audioLayers: 2, audioRegisters: 4, videoWidth: 16, audioWidth: 8)
}

/// One connector block: pre-normalized attention and feed-forward, both residual.
final class NFKLTX2ConnectorBlock: Module {
    @ModuleInfo(key: "attn1") var attn1: NFKLTX2Attention
    @ModuleInfo(key: "ff") var ff: NFKLTX2FeedForward

    init(dim: Int, heads: Int, headDim: Int) {
        _attn1.wrappedValue = NFKLTX2Attention(queryDim: dim, contextDim: dim, heads: heads, headDim: headDim,
                                               bias: true, outBias: true, eps: 1e-6, gated: true)
        _ff.wrappedValue = NFKLTX2FeedForward(dim: dim, bias: true)
    }

    func callAsFunction(_ x: MLXArray, rotary: (MLXArray, MLXArray)) -> MLXArray {
        let normed = ltx2AffineFreeRMSNorm(x, eps: 1e-6)
        let attended = x + attn1(normed, context: normed, queryRotary: rotary, keyRotary: nil)
        return attended + ff(ltx2AffineFreeRMSNorm(attended, eps: 1e-6))
    }
}

/// One modality's connector transformer: the register fill, the blocks, and the closing norm.
final class NFKLTX2ConnectorTransformer: Module {
    @ParameterInfo(key: "learnable_registers") var registers: MLXArray
    @ModuleInfo(key: "transformer_blocks") var blocks: [NFKLTX2ConnectorBlock]

    let heads: Int
    let ropeBaseLength: Int
    let ropeTheta: Float

    init(heads: Int, headDim: Int, layers: Int, registers: Int, ropeBaseLength: Int, ropeTheta: Float) {
        self.heads = heads
        self.ropeBaseLength = ropeBaseLength
        self.ropeTheta = ropeTheta
        let dim = heads * headDim
        _registers.wrappedValue = MLXArray.zeros([registers, dim])
        _blocks.wrappedValue = (0 ..< layers).map { _ in NFKLTX2ConnectorBlock(dim: dim, heads: heads, headDim: headDim) }
    }

    /// The projected valid tokens of one prompt `[1, valid, dim]`, front-aligned, filled with registers to
    /// `length`, and run through the blocks → `[1, length, dim]`.
    func callAsFunction(validTokens: MLXArray, length: Int) -> MLXArray {
        let count = registers.dim(0)
        precondition(length % count == 0, "the text length divides by the connector's registers")
        let valid = validTokens.dim(1)
        var sequence = validTokens
        if valid < length {
            let tiled = MLX.tiled(registers, repetitions: [length / count, 1])
            sequence = concatenated([validTokens, tiled[valid ..< length].expandedDimensions(axis: 0)
                .asType(validTokens.dtype)], axis: 1)
        }
        let rope = NFKLTX2Rope(dim: registers.dim(1), heads: heads, theta: ropeTheta)
        let positions = (0 ..< length).map { Float($0) / Float(ropeBaseLength) }
        let rotary = rope.table(positions: [positions])
        for block in blocks { sequence = block(sequence, rotary: rotary) }
        return ltx2AffineFreeRMSNorm(sequence, eps: 1e-6)
    }
}

/// The LTX-2 text connectors: a Gemma text encoder's stacked hidden states → the video and audio text
/// streams the transformer cross-attends to.
public final class NFKMLXLTX2ConnectorsNet: Module {
    @ModuleInfo(key: "video_text_proj_in") var videoProjection: Linear
    @ModuleInfo(key: "audio_text_proj_in") var audioProjection: Linear
    @ModuleInfo(key: "video_connector") var videoConnector: NFKLTX2ConnectorTransformer
    @ModuleInfo(key: "audio_connector") var audioConnector: NFKLTX2ConnectorTransformer

    public let configuration: NFKMLXLTX2ConnectorConfiguration

    public init(_ c: NFKMLXLTX2ConnectorConfiguration) {
        configuration = c
        let input = c.captionChannels * c.layerCount
        _videoProjection.wrappedValue = Linear(input, c.videoWidth, bias: c.projectionBias)
        _audioProjection.wrappedValue = Linear(input, c.audioWidth, bias: c.projectionBias)
        _videoConnector.wrappedValue = NFKLTX2ConnectorTransformer(
            heads: c.videoHeads, headDim: c.videoHeadDim, layers: c.videoLayers, registers: c.videoRegisters,
            ropeBaseLength: c.ropeBaseLength, ropeTheta: c.ropeTheta)
        _audioConnector.wrappedValue = NFKLTX2ConnectorTransformer(
            heads: c.audioHeads, headDim: c.audioHeadDim, layers: c.audioLayers, registers: c.audioRegisters,
            ropeBaseLength: c.ropeBaseLength, ropeTheta: c.ropeTheta)
    }

    /// The per-token, per-layer RMS norm of stacked hidden states `[B, T, hidden · layers]`
    /// (hidden-major), flattened back.
    func normalized(_ states: MLXArray) -> MLXArray {
        let c = configuration
        let (b, t) = (states.dim(0), states.dim(1))
        let stacked = states.reshaped([b, t, c.captionChannels, c.layerCount]).asType(.float32)
        let normed = stacked * rsqrt(stacked.square().mean(axis: 2, keepDims: true) + 1e-6)
        return normed.reshaped([b, t, c.captionChannels * c.layerCount]).asType(states.dtype)
    }

    /// The two text streams for ONE prompt's valid tokens `[1, valid, hidden · layers]`, padded with
    /// registers to `length` tokens: `video` `[1, length, videoWidth]`, `audio` `[1, length, audioWidth]`.
    public func callAsFunction(validStates: MLXArray, length: Int) -> (video: MLXArray, audio: MLXArray) {
        let c = configuration
        let normed = normalized(validStates)
        let video = videoProjection(normed * sqrt(Float(c.videoWidth) / Float(c.captionChannels)))
        let audio = audioProjection(normed * sqrt(Float(c.audioWidth) / Float(c.captionChannels)))
        return (videoConnector(validTokens: video, length: length), audioConnector(validTokens: audio, length: length))
    }

    /// The reference's batched form: stacked hidden states `[B, T, hidden · layers]` and the tokenizer's
    /// mask `[B, T]` (1 for a real token, either side padded) → the two streams, each row's valid tokens
    /// front-aligned and the rest registers.
    public func callAsFunction(_ states: MLXArray, mask: MLXArray) -> (video: MLXArray, audio: MLXArray) {
        let flags = mask.asType(.int32).asArray(Int32.self)
        let length = states.dim(1)
        var videos = [MLXArray](), audios = [MLXArray]()
        for row in 0 ..< states.dim(0) {
            let valid = (0 ..< length).filter { flags[row * length + $0] != 0 }
            let rowStates = states[row].take(MLXArray(valid.map(Int32.init)), axis: 0).expandedDimensions(axis: 0)
            let (video, audio) = self(validStates: rowStates, length: length)
            videos.append(video)
            audios.append(audio)
        }
        return (concatenated(videos, axis: 0), concatenated(audios, axis: 0))
    }
}

/// Reading, building, and loading the LTX-2 text connectors.
public enum NFKMLXLTX2Connectors {

    /// The geometry a `connectors/config.json` describes. Refuses by name what this port does not build:
    /// LTX-2.0's shared projection, connectors without registers, the interleaved rotary, causal
    /// positions, and ungated attention.
    public static func configuration(fromHuggingFace url: URL) throws -> NFKMLXLTX2ConnectorConfiguration {
        guard let json = try JSONSerialization.jsonObject(with: try Data(contentsOf: url)) as? [String: Any] else {
            throw NFKMLXError.unsupportedConfiguration("\(url.lastPathComponent) is not a JSON object")
        }
        if let kind = json["_class_name"] as? String, kind != "LTX2TextConnectors" {
            throw NFKMLXError.unsupportedConfiguration("this reads LTX-2 text connectors, not \(kind)")
        }
        func flag(_ key: String, _ fallback: Bool) -> Bool { (json[key] as? NSNumber)?.boolValue ?? fallback }
        func integer(_ key: String, _ fallback: Int) -> Int { (json[key] as? NSNumber)?.intValue ?? fallback }
        if !flag("per_modality_projections", false) {
            throw NFKMLXError.unsupportedConfiguration(
                "this builds LTX-2.3's per-modality text projections; the release shares one projection (LTX-2.0)")
        }
        if let rope = json["rope_type"] as? String, rope != "split" {
            throw NFKMLXError.unsupportedConfiguration("this builds the split rotary; the release asks for \(rope)")
        }
        if flag("causal_temporal_positioning", false) {
            throw NFKMLXError.unsupportedConfiguration(
                "this builds the connectors' plain token positions; the release asks for causal temporal ones")
        }
        if !flag("video_gated_attn", true) || !flag("audio_gated_attn", true) {
            throw NFKMLXError.unsupportedConfiguration(
                "this builds the gated connector attention; the release turns the gates off")
        }
        for key in ["video_connector_num_learnable_registers", "audio_connector_num_learnable_registers"]
            where json.keys.contains(key) && !(json[key] is NSNumber) {
            throw NFKMLXError.unsupportedConfiguration(
                "this builds connectors that fill padding with learnable registers; the release has none")
        }
        let base = NFKMLXLTX2ConnectorConfiguration.ltx23
        return NFKMLXLTX2ConnectorConfiguration(
            captionChannels: integer("caption_channels", base.captionChannels),
            layerCount: integer("text_proj_in_factor", base.layerCount),
            videoHeads: integer("video_connector_num_attention_heads", base.videoHeads),
            videoHeadDim: integer("video_connector_attention_head_dim", base.videoHeadDim),
            videoLayers: integer("video_connector_num_layers", base.videoLayers),
            videoRegisters: integer("video_connector_num_learnable_registers", base.videoRegisters),
            audioHeads: integer("audio_connector_num_attention_heads", base.audioHeads),
            audioHeadDim: integer("audio_connector_attention_head_dim", base.audioHeadDim),
            audioLayers: integer("audio_connector_num_layers", base.audioLayers),
            audioRegisters: integer("audio_connector_num_learnable_registers", base.audioRegisters),
            ropeBaseLength: integer("connector_rope_base_seq_len", base.ropeBaseLength),
            ropeTheta: (json["rope_theta"] as? NSNumber)?.floatValue ?? base.ropeTheta,
            videoWidth: integer("video_hidden_dim", base.videoWidth),
            audioWidth: integer("audio_hidden_dim", base.audioWidth),
            projectionBias: flag("proj_bias", base.projectionBias))
    }

    /// Loads a release's `connectors/` directory into `net`. Every tensor is at most 2-D.
    public static func loadWeights(into net: NFKMLXLTX2ConnectorsNet, from directory: URL,
                                   precision: NFKMLXWeightPrecision = .float32) throws {
        let arrays = try NFKMLXReleaseWeights.materializedArrays(inDirectory: directory, precision: precision)
        try NFKMLXWeights.apply(arrays, to: net)
    }
}
