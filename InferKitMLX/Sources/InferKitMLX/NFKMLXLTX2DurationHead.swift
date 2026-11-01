//
//  NFKMLXLTX2DurationHead.swift
//  InferKitMLX
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import InferKit
import MLX
import MLXNN

// LTX-2.5's duration head (`LTX2DurationHead`, Lightricks), which predicts the natural length of the shot a
// caption describes from the text connectors' outputs. Each stream is projected into a shared width and
// tagged with a learned modality embedding, the two are concatenated, a learned query attends over them
// without a mask (the connectors fill every padded position with a register), and a two-layer MLP with a
// tanh-approximated GELU turns the pooled vector into a log-duration in seconds. The pipeline converts the
// prediction into a frame count clamped to a window and snapped to the autoencoder's `8k + 1` grid.

/// The geometry of an LTX-2.5 duration head, read from a release's `duration_head/config.json`.
public struct NFKMLXLTX2DurationHeadConfiguration: Sendable {
    public var videoCrossAttentionDim: Int
    public var audioCrossAttentionDim: Int
    public var poolerHiddenDim: Int
    public var queries: Int
    public var poolerHeads: Int
    public var mlpHiddenDim: Int

    public init(videoCrossAttentionDim: Int = 4096, audioCrossAttentionDim: Int = 2048, poolerHiddenDim: Int = 256,
                queries: Int = 1, poolerHeads: Int = 4, mlpHiddenDim: Int = 256) {
        self.videoCrossAttentionDim = videoCrossAttentionDim
        self.audioCrossAttentionDim = audioCrossAttentionDim
        self.poolerHiddenDim = poolerHiddenDim
        self.queries = queries
        self.poolerHeads = poolerHeads
        self.mlpHiddenDim = mlpHiddenDim
    }

    /// LTX-2.5's head (`Lightricks/LTX-2.5-Diffusers`, `duration_head/config.json`).
    public static let ltx25 = NFKMLXLTX2DurationHeadConfiguration()

    /// The release's arrangement at a sixteenth of its width, for weight-free tests.
    public static let tiny = NFKMLXLTX2DurationHeadConfiguration(
        videoCrossAttentionDim: 32, audioCrossAttentionDim: 16, poolerHiddenDim: 16, queries: 2, poolerHeads: 2,
        mlpHiddenDim: 16)
}

/// The learned queries' attention over the projected caption tokens.
final class NFKLTX2DurationPooler: Module {
    @ParameterInfo(key: "query_tokens") var queryTokens: MLXArray
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "to_out") var toOut: Linear
    let heads: Int

    init(hidden: Int, queries: Int, heads: Int) {
        self.heads = heads
        _queryTokens.wrappedValue = MLXArray.zeros([queries, hidden])
        _toQ.wrappedValue = Linear(hidden, hidden)
        _toK.wrappedValue = Linear(hidden, hidden)
        _toV.wrappedValue = Linear(hidden, hidden)
        _toOut.wrappedValue = Linear(hidden, hidden)
    }

    func callAsFunction(_ tokens: MLXArray) -> MLXArray {
        let batch = tokens.dim(0), width = queryTokens.dim(1), headDim = width / heads
        func split(_ x: MLXArray) -> MLXArray {
            x.reshaped([batch, -1, heads, headDim]).transposed(0, 2, 1, 3)
        }
        let queries = broadcast(queryTokens.expandedDimensions(axis: 0), to: [batch, queryTokens.dim(0), width])
        let attended = NFKReferenceRounding.flashAttention(
            queries: split(toQ(queries)), keys: split(toK(tokens)), values: split(toV(tokens)),
            scale: 1 / sqrt(Float(headDim)), mask: nil)
        return toOut(attended.transposed(0, 2, 1, 3).reshaped([batch, -1, width]))
    }
}

/// LTX-2.5's duration head: the video and audio connector outputs `[B, tokens, width]` → the predicted shot
/// length in seconds `[B]`.
public final class NFKMLXLTX2DurationHeadNet: Module {
    @ModuleInfo(key: "video_input_proj") var videoInputProj: Linear
    @ParameterInfo(key: "video_modality_emb") var videoModalityEmbedding: MLXArray
    @ModuleInfo(key: "audio_input_proj") var audioInputProj: Linear
    @ParameterInfo(key: "audio_modality_emb") var audioModalityEmbedding: MLXArray
    @ModuleInfo(key: "attention_pooler") var pooler: NFKLTX2DurationPooler
    @ModuleInfo(key: "mlp_hidden") var mlpHidden: Linear
    @ModuleInfo(key: "mlp_out") var mlpOut: Linear

    public let configuration: NFKMLXLTX2DurationHeadConfiguration

    public init(_ c: NFKMLXLTX2DurationHeadConfiguration) {
        configuration = c
        _videoInputProj.wrappedValue = Linear(c.videoCrossAttentionDim, c.poolerHiddenDim)
        _videoModalityEmbedding.wrappedValue = MLXArray.zeros([c.poolerHiddenDim])
        _audioInputProj.wrappedValue = Linear(c.audioCrossAttentionDim, c.poolerHiddenDim)
        _audioModalityEmbedding.wrappedValue = MLXArray.zeros([c.poolerHiddenDim])
        _pooler.wrappedValue = NFKLTX2DurationPooler(hidden: c.poolerHiddenDim, queries: c.queries,
                                                     heads: c.poolerHeads)
        _mlpHidden.wrappedValue = Linear(c.poolerHiddenDim * c.queries, c.mlpHiddenDim)
        _mlpOut.wrappedValue = Linear(c.mlpHiddenDim, 1)
    }

    /// The predicted duration in seconds `[B]` from either or both connector outputs, cast to the head's type.
    public func callAsFunction(video: MLXArray?, audio: MLXArray?) -> MLXArray {
        let dtype = mlpOut.weight.dtype
        var groups = [MLXArray]()
        if let video { groups.append(videoInputProj(video.asType(dtype)) + videoModalityEmbedding) }
        if let audio { groups.append(audioInputProj(audio.asType(dtype)) + audioModalityEmbedding) }
        let pooled = pooler(concatenated(groups, axis: 1))
        let hidden = NFKReferenceRounding.geluTanh(mlpHidden(pooled.reshaped([pooled.dim(0), -1])))
        return exp(mlpOut(hidden).squeezed(axis: -1))
    }

    /// The frame count for a predicted duration of `seconds` at `frameRate`: rounded, clamped to
    /// `[minimumSeconds, maximumSeconds]`, and snapped down to the `k · compression + 1` grid, or up where
    /// snapping down leaves the window. A window that holds no grid point takes the nearer neighbour.
    public static func frames(forSeconds seconds: Double, frameRate: Double, compression: Int = 8,
                              minimumSeconds: Double = 1, maximumSeconds: Double = 20) -> Int {
        // Python's `round` rounds half to even.
        func round(_ x: Double) -> Int { Int(x.rounded(.toNearestOrEven)) }
        let minimum = max(1, round(minimumSeconds * frameRate)), maximum = round(maximumSeconds * frameRate)
        let clamped = max(minimum, min(round(seconds * frameRate), maximum))
        var frames = ((clamped - 1) / compression) * compression + 1
        if frames < minimum {
            let up = frames + compression
            if up <= maximum || abs(up - clamped) < abs(frames - clamped) {
                frames = up
            }
        }
        return frames
    }
}

/// Reading and loading LTX-2.5's duration head.
public enum NFKMLXLTX2DurationHead {

    /// The head geometry a `duration_head/config.json` describes.
    public static func configuration(fromHuggingFace url: URL) throws -> NFKMLXLTX2DurationHeadConfiguration {
        guard let json = try JSONSerialization.jsonObject(with: try Data(contentsOf: url)) as? [String: Any] else {
            throw NFKMLXError.unsupportedConfiguration("\(url.lastPathComponent) is not a JSON object")
        }
        if let kind = json["_class_name"] as? String, kind != "LTX2DurationHead" {
            throw NFKMLXError.unsupportedConfiguration("this reads an LTX-2 duration head, not \(kind)")
        }
        let base = NFKMLXLTX2DurationHeadConfiguration.ltx25
        func integer(_ key: String, _ fallback: Int) -> Int { (json[key] as? NSNumber)?.intValue ?? fallback }
        return NFKMLXLTX2DurationHeadConfiguration(
            videoCrossAttentionDim: integer("video_cross_attention_dim", base.videoCrossAttentionDim),
            audioCrossAttentionDim: integer("audio_cross_attention_dim", base.audioCrossAttentionDim),
            poolerHiddenDim: integer("pooler_hidden_dim", base.poolerHiddenDim),
            queries: integer("num_queries", base.queries), poolerHeads: integer("num_pooler_heads", base.poolerHeads),
            mlpHiddenDim: integer("mlp_hidden_dim", base.mlpHiddenDim))
    }

    /// Loads a release's `duration_head/` directory into `head`.
    public static func loadWeights(into head: NFKMLXLTX2DurationHeadNet, from directory: URL,
                                   precision: NFKMLXWeightPrecision = .float32) throws {
        let arrays = try NFKMLXReleaseWeights.materializedArrays(inDirectory: directory, precision: precision)
        try NFKMLXWeights.apply(arrays, to: head, verifyShapes: true)
    }
}
