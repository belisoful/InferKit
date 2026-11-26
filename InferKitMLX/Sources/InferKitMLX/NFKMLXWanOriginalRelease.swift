//
//  NFKMLXWanOriginalRelease.swift
//  InferKitMLX
//
//  Reading a Wan release in the original layout: Wan-Video's own repositories and the releases built on
//  them (the VideoX-Fun Fun releases, S2V, VACE-Fun, Wan-Dancer, SkyReels V3). The layout is a
//  transformer `config.json` and safetensors under the original Wan names, the autoencoder as
//  `Wan2.1_VAE.pth` or `Wan2.2_VAE.pth`, umT5-XXL as `models_t5_umt5-xxl-enc-bf16.pth`, and its tokenizer
//  under `google/umt5-xxl/`. The family shares these three files byte for byte.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import MLX

/// The original Wan layout's names, configuration, and files.
enum NFKMLXWanOriginalRelease {

    /// The umT5 encoder's file name.
    static let textEncoderFile = "models_t5_umt5-xxl-enc-bf16.pth"
    /// The tokenizer directory.
    static let tokenizerDirectory = "google/umt5-xxl"
    /// The Wan 2.1 autoencoder's file name (16 latent channels).
    static let wan21AutoencoderFile = "Wan2.1_VAE.pth"
    /// The Wan 2.2 autoencoder's file name (48 latent channels, compression 4 × 16 × 16).
    static let wan22AutoencoderFile = "Wan2.2_VAE.pth"

    /// The Wan 2.2 autoencoder's per-channel latent statistics, which the original code states and the
    /// `.pth` does not carry.
    static let wan22LatentsMean: [Float] = [
        -0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557, -0.1382, 0.0542, 0.2813, 0.0891,
        0.1570, -0.0098, 0.0375, -0.1825, -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108, -0.2158, 0.2502,
        -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.1230, -0.0313, -0.1649, 0.0117, 0.0723,
        -0.2839, -0.2083, -0.0520, 0.3748, 0.0152, 0.1957, 0.1433, -0.2944, 0.3573, -0.0548, -0.1681, -0.0667,
    ]
    static let wan22LatentsStd: [Float] = [
        0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.4990, 0.4818, 0.5013, 0.8158, 1.0344, 0.5894, 1.0901,
        0.6885, 0.6165, 0.8454, 0.4978, 0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659,
        0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093, 0.5709, 0.6065, 0.6415, 0.4944,
        0.5726, 1.2042, 0.5458, 1.6887, 0.3971, 1.0600, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744,
    ]

    // MARK: Transformer

    /// The module key an original transformer tensor name maps to, the renaming diffusers'
    /// `convert_wan_transformer_to_diffusers` applies: attention projections and norms move under
    /// `attn1` / `attn2`, the affine cross-attention norm `norm3` becomes `norm2`, `modulation` becomes
    /// `scale_shift_table`, and the embedders move under `condition_embedder`. `ref_conv` and
    /// `control_adapter` keep their names.
    static func transformerModuleKey(forRelease name: String) -> String {
        if name == "head.modulation" { return "scale_shift_table" }
        if let renamed = topLevelNames.first(where: { name.hasPrefix($0.key + ".") }) {
            return renamed.value + name.dropFirst(renamed.key.count)
        }
        let parts = name.split(separator: ".").map(String.init)
        guard parts.count >= 3, parts[0] == "blocks", Int(parts[1]) != nil else { return name }
        let block = "blocks.\(parts[1])"
        let rest = parts[2...].joined(separator: ".")
        for (site, module) in [("self_attn", "attn1"), ("cross_attn", "attn2")] where rest.hasPrefix(site + ".") {
            let tail = rest.dropFirst(site.count + 1)
            for (released, renamed) in attentionNames where tail.hasPrefix(released + ".") {
                return "\(block).\(module).\(renamed)" + tail.dropFirst(released.count)
            }
            return "\(block).\(module).\(tail)"
        }
        if rest.hasPrefix("norm3.") { return "\(block).norm2" + rest.dropFirst("norm3".count) }
        if rest == "modulation" { return "\(block).scale_shift_table" }
        if rest.hasPrefix("ffn.0.") { return "\(block).ffn.net.0.proj" + rest.dropFirst("ffn.0".count) }
        if rest.hasPrefix("ffn.2.") { return "\(block).ffn.net.2" + rest.dropFirst("ffn.2".count) }
        return name
    }

    private static let attentionNames: [(String, String)] = [
        ("q", "to_q"), ("k", "to_k"), ("v", "to_v"), ("o", "to_out.0"), ("k_img", "add_k_proj"),
        ("v_img", "add_v_proj"), ("norm_k_img", "norm_added_k"),
    ]

    private static let topLevelNames: [String: String] = [
        "time_embedding.0": "condition_embedder.time_embedder.linear_1",
        "time_embedding.2": "condition_embedder.time_embedder.linear_2",
        "text_embedding.0": "condition_embedder.text_embedder.linear_1",
        "text_embedding.2": "condition_embedder.text_embedder.linear_2",
        "time_projection.1": "condition_embedder.time_proj",
        "head.head": "proj_out",
    ]

    /// A renamed transformer tensor in the module's layout: the 5-D patch embedding moves to
    /// `[out, t, h, w, in]`, and a 4-D convolution (`ref_conv`, the control adapter's) to `[out, h, w, in]`.
    static func adaptedTransformer(key: String, value: MLXArray) -> (String, MLXArray) {
        switch value.ndim {
        case 5: return (key, value.transposed(0, 2, 3, 4, 1))
        case 4: return (key, value.transposed(0, 2, 3, 1))
        default: return (key, value)
        }
    }

    /// The transformer geometry an original `config.json` states. Both spellings occur: the original
    /// `WanModel` keys (`dim`, `in_dim`, `num_heads`) and VideoX-Fun's `Wan2_2Transformer3DModel` copy of
    /// them beside diffusers-style aliases.
    ///
    /// @discussion The configuration is refused where the original forward reads a setting this module
    /// does not implement: a sliding attention window, no query/key norm, or an un-normalized
    /// cross-attention.
    static func transformerConfiguration(fromConfigURL url: URL) throws -> NFKMLXWanConfiguration {
        let json = try NFKMLXWanRelease.json(url)
        func int(_ keys: String...) -> Int? { keys.lazy.compactMap { json[$0] as? Int }.first }
        guard let dim = int("dim", "hidden_size"), let heads = int("num_heads"), let layers = int("num_layers"),
              let ffn = int("ffn_dim"), let inChannels = int("in_dim", "in_channels") else {
            throw NFKMLXError.unsupportedConfiguration("\(url.lastPathComponent) is not an original Wan transformer configuration")
        }
        if let window = json["window_size"] as? [Int], window != [-1, -1] {
            throw NFKMLXError.unsupportedConfiguration("a Wan attention window \(window) is not implemented")
        }
        if (json["qk_norm"] as? NSNumber)?.boolValue == false || (json["cross_attn_norm"] as? NSNumber)?.boolValue == false {
            throw NFKMLXError.unsupportedConfiguration("a Wan block without its query/key or cross-attention norm is not implemented")
        }
        let addsReference = (json["add_ref_conv"] as? NSNumber)?.boolValue ?? false
        let addsAdapter = (json["add_control_adapter"] as? NSNumber)?.boolValue ?? false
        return NFKMLXWanConfiguration(
            inChannels: inChannels, heads: heads, headDim: dim / heads, layers: layers, ffnDim: ffn,
            textDim: int("text_dim") ?? 4096, freqDim: int("freq_dim") ?? 256,
            patchSize: json["patch_size"] as? [Int] ?? [1, 2, 2],
            eps: (json["eps"] as? NSNumber)?.floatValue ?? 1e-6, outChannels: int("out_dim", "out_channels") ?? 16,
            referenceChannels: addsReference ? int("in_dim_ref_conv") ?? 16 : nil,
            controlAdapterChannels: addsAdapter ? int("in_dim_control_adapter") ?? 24 : nil,
            controlAdapterDownscale: int("downscale_factor_control_adapter") ?? 8,
            formsSinusoidsInFloat64: true)
    }

    /// Loads an original transformer directory (one safetensors file or a shard index) into `net`.
    /// Tensors the module does not have (`img_emb` of an `i2v` configuration that carries no image
    /// branch) are refused by the strict apply, so a release that needs them fails at load.
    static func loadTransformer(into net: NFKMLXWanTransformerNet, fromDirectory directory: URL,
                                precision: NFKMLXWeightPrecision = .float32) throws {
        let arrays = try NFKMLXReleaseWeights.materializedArrays(inDirectory: directory, precision: precision,
                                                                 remap: transformerModuleKey(forRelease:),
                                                                 transform: adaptedTransformer)
        try NFKMLXWeights.apply(arrays, to: net, verifyShapes: true)
    }

    // MARK: Autoencoder

    /// The module key an original autoencoder tensor name maps to, the renaming diffusers'
    /// `convert_vae` (Wan 2.1) and `convert_vae_22` (Wan 2.2) apply.
    ///
    /// @discussion `residual.0`, `.2`, `.3`, and `.6` are a residual block's first norm, first convolution,
    /// second norm, and second convolution. Wan 2.2 nests each level's blocks under
    /// `downsamples.<level>.downsamples.<i>`, the resampler last. Wan 2.1 lists every block of the encoder
    /// flat, and its decoder groups the flat list by level: `numResBlocks + 1` residual blocks, then one
    /// upsampler, for every level but the last.
    static func vaeModuleKey(forRelease name: String, configuration: NFKMLXWanVAEConfiguration) -> String {
        if let renamed = vaeFixedNames[name] { return renamed }
        var parts = name.split(separator: ".").map(String.init)
        guard parts.count >= 3 else { return name }
        if ["middle", "head"].contains(parts[1]) {
            return vaeMiddleOrHead(parts)
        }
        guard parts[1] == "downsamples" || parts[1] == "upsamples", let level = Int(parts[2]) else { return name }
        let encoder = parts[1] == "downsamples"
        let blocks = encoder ? "down_blocks" : "up_blocks"
        if configuration.isResidual {
            guard parts.count >= 5, let item = Int(parts[4]) else { return name }
            let tail = Array(parts[5...])
            if tail.first == "residual" || tail.first == "shortcut" {
                return (["\(parts[0]).\(blocks).\(level).resnets.\(item)"] + residualTail(tail)).joined(separator: ".")
            }
            return (["\(parts[0]).\(blocks).\(level).\(encoder ? "downsampler" : "upsampler")"] + tail).joined(separator: ".")
        }
        let tail = Array(parts[3...])
        if encoder {
            parts = ["encoder.down_blocks.\(level)"] + residualTail(tail)
            return parts.joined(separator: ".")
        }
        let group = configuration.numResBlocks + 2
        let upLevel = level / group, position = level % group
        if position == configuration.numResBlocks + 1 {
            return (["decoder.up_blocks.\(upLevel).upsamplers.0"] + tail).joined(separator: ".")
        }
        return (["decoder.up_blocks.\(upLevel).resnets.\(position)"] + residualTail(tail)).joined(separator: ".")
    }

    private static let vaeFixedNames: [String: String] = [
        "conv1.weight": "quant_conv.weight", "conv1.bias": "quant_conv.bias",
        "conv2.weight": "post_quant_conv.weight", "conv2.bias": "post_quant_conv.bias",
        "encoder.conv1.weight": "encoder.conv_in.weight", "encoder.conv1.bias": "encoder.conv_in.bias",
        "decoder.conv1.weight": "decoder.conv_in.weight", "decoder.conv1.bias": "decoder.conv_in.bias",
    ]

    private static func residualTail(_ tail: [String]) -> [String] {
        guard let first = tail.first else { return tail }
        if first == "shortcut" { return ["conv_shortcut"] + tail.dropFirst() }
        guard first == "residual", tail.count >= 2,
              let renamed = ["0": "norm1", "2": "conv1", "3": "norm2", "6": "conv2"][tail[1]] else { return tail }
        return [renamed] + tail.dropFirst(2)
    }

    private static func vaeMiddleOrHead(_ parts: [String]) -> String {
        let side = parts[0]
        if parts[1] == "head", parts.count >= 4 {
            let module = parts[2] == "0" ? "norm_out" : "conv_out"
            return ([side, module] + parts[3...]).joined(separator: ".")
        }
        guard parts.count >= 4, let index = Int(parts[2]) else { return parts.joined(separator: ".") }
        if index == 1 {
            return (["\(side).mid_block.attentions.0"] + parts[3...]).joined(separator: ".")
        }
        return (["\(side).mid_block.resnets.\(index / 2)"] + residualTail(Array(parts[3...]))).joined(separator: ".")
    }

    /// The autoencoder an original file holds, its geometry read from the tensors (the file carries no
    /// configuration): the released `.wan22` or `.wan21` layout by the file's name, the encoder's width from
    /// `encoder.conv1` and the decoder's from `decoder.conv1`, and the latent statistics the original code
    /// states.
    static func autoencoder(from url: URL, precision: NFKMLXWeightPrecision = .float32) throws
        -> (net: NFKMLXWanVideoVAENet, mean: MLXArray, std: MLXArray) {
        let stored = try NFKMLXWeights.loadCheckpoint(url: url).arrays
        let wan22 = url.lastPathComponent == wan22AutoencoderFile
        var configuration: NFKMLXWanVAEConfiguration = wan22 ? .wan22 : .wan21
        guard let encoderIn = stored["encoder.conv1.weight"], let decoderIn = stored["decoder.conv1.weight"] else {
            throw NFKMLXError.unsupportedConfiguration("\(url.lastPathComponent) is not an original Wan autoencoder")
        }
        configuration.baseDim = encoderIn.dim(0)
        configuration.decoderBaseDim = decoderIn.dim(0) / (configuration.dimMult.last ?? 1)
        let net = NFKMLXWanVideoVAENet(configuration)
        let arrays = stored.map { key, value -> (String, MLXArray) in
            let typed = precision == .checkpoint || !value.dtype.isFloatingPoint ? value : value.asType(.float32)
            return NFKMLXWanRelease.adaptedVAE(key: vaeModuleKey(forRelease: key, configuration: configuration), value: typed)
        }
        try NFKMLXWeights.apply(arrays, to: net, verifyShapes: true)
        let statistics = wan22 ? (wan22LatentsMean, wan22LatentsStd)
            : (NFKMLXWanAnimatePipeline.wan21LatentsMean, NFKMLXWanAnimatePipeline.wan21LatentsStd)
        return (net, MLXArray(statistics.0), MLXArray(statistics.1))
    }

    // MARK: Text encoder

    /// The module key an original umT5 tensor name maps to: the original `T5Encoder` names
    /// (`token_embedding`, `blocks.<i>.attn`, `.pos_embedding.embedding`, `.ffn.gate.0` / `.fc1` / `.fc2`,
    /// `norm`) as transformers' `UMT5EncoderModel` names them. The original's gated feed-forward is
    /// `fc2(fc1(x) · gelu(gate(x)))`, so `gate` is `wi_0` and `fc1` is `wi_1`.
    static func textEncoderModuleKey(forRelease name: String) -> String {
        if name == "token_embedding.weight" { return "shared.weight" }
        if name == "norm.weight" { return "encoder.final_layer_norm.weight" }
        let parts = name.split(separator: ".").map(String.init)
        guard parts.count >= 4, parts[0] == "blocks", let index = Int(parts[1]) else { return name }
        let block = "encoder.block.\(index).layer"
        let rest = parts[2...].joined(separator: ".")
        let renamed: [String: String] = [
            "norm1.weight": "\(block).0.layer_norm.weight",
            "attn.q.weight": "\(block).0.SelfAttention.q.weight",
            "attn.k.weight": "\(block).0.SelfAttention.k.weight",
            "attn.v.weight": "\(block).0.SelfAttention.v.weight",
            "attn.o.weight": "\(block).0.SelfAttention.o.weight",
            "pos_embedding.embedding.weight": "\(block).0.SelfAttention.relative_attention_bias.weight",
            "norm2.weight": "\(block).1.layer_norm.weight",
            "ffn.gate.0.weight": "\(block).1.DenseReluDense.wi_0.weight",
            "ffn.fc1.weight": "\(block).1.DenseReluDense.wi_1.weight",
            "ffn.fc2.weight": "\(block).1.DenseReluDense.wo.weight",
        ]
        return renamed[rest] ?? name
    }

    /// The umT5 geometry the original file's tensors imply: the original ships no configuration, so the
    /// width, depth, heads, and feed-forward width are read from the shapes.
    static func textEncoderConfiguration(of arrays: [String: MLXArray]) throws -> NFKMLXT5Configuration {
        guard let embedding = arrays["token_embedding.weight"], let bias = arrays["blocks.0.pos_embedding.embedding.weight"],
              let query = arrays["blocks.0.attn.q.weight"], let fc1 = arrays["blocks.0.ffn.fc1.weight"] else {
            throw NFKMLXError.unsupportedConfiguration("the umT5 file lacks the original encoder's tensors")
        }
        let layers = Set(arrays.keys.compactMap { key -> Int? in
            let parts = key.split(separator: ".")
            return parts.count > 1 && parts[0] == "blocks" ? Int(parts[1]) : nil
        }).count
        let heads = bias.dim(1)
        return NFKMLXT5Configuration(dModel: embedding.dim(1), layers: layers, heads: heads, keyDim: query.dim(0) / heads,
                                     ffDim: fc1.dim(0), vocabularySize: embedding.dim(0),
                                     relativeBuckets: bias.dim(0), perLayerBias: true)
    }

    /// The umT5 encoder from an original file, its geometry read from the tensors.
    static func textEncoder(from url: URL, precision: NFKMLXWeightPrecision = .float32) throws -> NFKMLXT5EncoderNet {
        let stored = try NFKMLXWeights.loadCheckpoint(url: url).arrays
        let net = NFKMLXT5Encoder.makeNet(try textEncoderConfiguration(of: stored))
        let arrays = stored.map { key, value -> (String, MLXArray) in
            (textEncoderModuleKey(forRelease: key),
             precision == .checkpoint || !value.dtype.isFloatingPoint ? value : value.asType(.float32))
        }
        try NFKMLXWeights.apply(arrays, to: net, verifyShapes: true)
        return net
    }
}
