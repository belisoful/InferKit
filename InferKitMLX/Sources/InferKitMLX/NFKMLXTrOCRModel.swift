import Foundation
import MLX
import MLXNN

// MARK: - Network

/// The TrOCR vision-encoder-decoder: a ViT image encoder whose output is the memory a BART-style
/// text decoder reads. The decoder is the shared ``NFKMLXSeq2SeqNet`` in its decoder-only shape.
///
/// @discussion `microsoft/trocr-base-handwritten` reads a line of handwriting into text. The image
/// encoder produces one memory per generation; the decoder cross-attends to it while generating the
/// transcription. The decoder is 1024-wide over 768-wide image features, so its cross-attention
/// projects the memory from 768 (``NFKMLXSeq2SeqConfiguration/crossAttentionWidth``).
///
/// Introduced in InferKit 0.4.0.
public final class NFKMLXTrOCRNet: Module {
    @ModuleInfo(key: "encoder") var vision: NFKMLXTrOCRVisionNet
    @ModuleInfo(key: "decoder") var language: NFKMLXSeq2SeqNet

    public let visionConfiguration: NFKMLXTrOCRVisionConfiguration
    public let languageConfiguration: NFKMLXSeq2SeqConfiguration

    public init(vision: NFKMLXTrOCRVisionConfiguration, language: NFKMLXSeq2SeqConfiguration) {
        visionConfiguration = vision
        languageConfiguration = language
        _vision.wrappedValue = NFKMLXTrOCRVisionNet(vision)
        _language.wrappedValue = NFKMLXSeq2SeqNet(language)
        super.init()
    }

    /// Reads the ViT geometry and the decoder geometry from a vision-encoder-decoder `config.json`.
    public convenience init(configurationURL url: URL) throws {
        let data = try Data(contentsOf: url)
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw NFKMLXError.unsupportedConfiguration("\(url.lastPathComponent) is not a JSON object")
        }
        guard let decoder = json["decoder"] as? [String: Any] else {
            throw NFKMLXError.unsupportedConfiguration("config.json lacks a decoder block")
        }
        self.init(vision: try NFKMLXTrOCRVisionConfiguration(huggingFaceConfig: json),
                  language: try NFKMLXSeq2SeqConfiguration(huggingFaceConfig: decoder))
    }

    /// - Parameter image: `[B, H, W, 3]`, already resized and normalized.
    /// - Returns: the image memory `[B, patches + 1, hidden]` the decoder cross-attends.
    public func imageFeatures(_ image: MLXArray) -> MLXArray { vision(image) }

    /// Decoder ids `[B, T]` against `memory` → logits `[B, T, vocabulary]`, extending `cache`.
    public func decode(_ tokens: MLXArray, memory: MLXArray, cache: NFKMLXSeq2SeqCache) -> MLXArray {
        language.decode(tokens, memory: memory, cache: cache)
    }

    public func makeCache() -> NFKMLXSeq2SeqCache { language.makeCache() }

    // MARK: Weights

    /// Loads a released TrOCR directory. The ViT encoder tensors carry an `encoder.` prefix and load
    /// through this port's remap (with the 4-D patch convolution transposed to MLX layout); the
    /// decoder tensors carry a `decoder.` prefix and load through the shared seq2seq remap. A fine-tuned
    /// directory (``NFKMLXTrOCR/save(_:toDirectoryURL:release:)``) loads in the module's own names.
    public func loadWeights(fromDirectory directory: URL) throws {
        var visionArrays: [(String, MLXArray)] = []
        var decoderArrays: [(String, MLXArray)] = []
        for url in try NFKMLXTrOCRNet.weightFiles(in: directory) {
            let checkpoint = try NFKMLXWeights.loadCheckpoint(url: url)
            // A file `NFKMLXWeights.save` wrote (a fine-tune) holds this module's own names and layout.
            guard checkpoint.needsConvTranspose else {
                try NFKMLXWeights.apply(checkpoint.arrays.map { ($0.key, $0.value) }, to: self)
                return
            }
            let hasShared = checkpoint.arrays.keys.contains("shared.weight")
            for (key, value) in checkpoint.arrays {
                if key.hasPrefix("encoder.") {
                    guard let mapped = NFKMLXTrOCRVisionWeights.visionKey(key) else { continue }
                    let transposed = value.ndim == 4 ? value.transposed(0, 2, 3, 1) : value
                    visionArrays.append((mapped, transposed.asType(.float32)))
                } else if let mapped = NFKMLXSeq2SeqNet.moduleKey(for: key, configuration: languageConfiguration, hasShared: hasShared) {
                    decoderArrays.append((mapped, value.asType(.float32)))
                }
            }
        }
        try NFKMLXWeights.apply(visionArrays, to: vision)
        try NFKMLXWeights.apply(decoderArrays, to: language)
    }

    static func weightFiles(in directory: URL) throws -> [URL] {
        if let files = try? NFKMLXReleaseWeights.files(inDirectory: directory) { return files }
        let safetensors = directory.appendingPathComponent("model.safetensors")
        if FileManager.default.fileExists(atPath: safetensors.path) { return [safetensors] }
        let bin = directory.appendingPathComponent("pytorch_model.bin")
        guard FileManager.default.fileExists(atPath: bin.path) else {
            throw NFKMLXError.unsupportedConfiguration("\(directory.lastPathComponent) holds no model.safetensors or pytorch_model.bin")
        }
        return [bin]
    }
}

// MARK: - Decoding wrapper

/// Drives generation from a fixed image memory: the ``NFKMLXSeq2SeqDecoder`` asks for a source
/// encoding, and this returns the ViT features computed once for the page.
final class NFKTrOCRDecodeModel: NFKMLXSeq2SeqDecodable {
    let net: NFKMLXTrOCRNet
    let memory: MLXArray

    init(net: NFKMLXTrOCRNet, memory: MLXArray) {
        self.net = net
        self.memory = memory
    }

    func encodeSource(_ tokens: MLXArray) -> MLXArray { memory }
    func makeDecodingCache() -> NFKMLXSeq2SeqCache { net.makeCache() }
    func decodeStep(_ tokens: MLXArray, memory: MLXArray, cache: NFKMLXSeq2SeqCache) -> MLXArray {
        net.decode(tokens, memory: memory, cache: cache)
    }
    func reorderCache(_ cache: NFKMLXSeq2SeqCache, rows: MLXArray) { cache.reorder(rows) }
}
