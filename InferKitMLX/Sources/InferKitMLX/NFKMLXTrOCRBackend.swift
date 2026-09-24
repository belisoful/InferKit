import Foundation
import CoreGraphics
import InferKit
import MLX
import MLXNN

// The TrOCR usable layer: an image processor (PIL's resample to 384, scale to [-1, 1]), the release's
// tokenizer for decoding (RoBERTa byte-level BPE, or the small releases' XLM-R SentencePiece), an @objc
// directory factory, and an NFKInferenceBackend that reads a line of text into a transcription.
// Generation drives the shared seq2seq decoder from the image memory.

/// The TrOCR processor: the image preprocessing and the tokenizer the reference `TrOCRProcessor` pairs.
public enum NFKMLXTrOCRProcessor {

    /// The release's tokenizer: the RoBERTa byte-level BPE from `vocab.json` and `merges.txt`, or, for
    /// a small release, XLM-R's SentencePiece from `sentencepiece.bpe.model`.
    public static func tokenizer(inDirectory directory: URL) -> NFKTokenizer? {
        bytePairTokenizer(inDirectory: directory) ?? sentencePieceTokenizer(inDirectory: directory)
    }

    /// XLM-R's numbering of a SentencePiece model: fairseq's `<s>` 0, `<pad>` 1, `</s>` 2, `<unk>` 3, every
    /// other piece at its model index plus one, and `<mask>` after the last.
    static func sentencePieceTokenizer(inDirectory directory: URL) -> NFKTokenizer? {
        let modelURL = directory.appendingPathComponent("sentencepiece.bpe.model")
        guard let segmenter = try? NFKMLXSentencePieceSegmenter(contentsOf: modelURL) else { return nil }
        var vocabulary = ["<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3]
        for index in 3 ..< segmenter.pieceCount {
            if let piece = segmenter.piece(at: index) { vocabulary[piece] = index + 1 }
        }
        vocabulary["<mask>"] = segmenter.pieceCount + 1
        return NFKMLXSentencePieceTokenizer(segmenter: segmenter, vocabulary: vocabulary, unknownToken: "<unk>",
                                            eosTokenId: 2, bosTokenId: 0)
    }

    /// The RoBERTa byte-level BPE from `vocab.json` and `merges.txt`, declaring the five special tokens
    /// so they decode to their literals rather than byte-decoding.
    static func bytePairTokenizer(inDirectory directory: URL) -> NFKTokenizer? {
        let vocabURL = directory.appendingPathComponent("vocab.json")
        let mergesURL = directory.appendingPathComponent("merges.txt")
        guard FileManager.default.fileExists(atPath: vocabURL.path),
              FileManager.default.fileExists(atPath: mergesURL.path),
              let data = try? Data(contentsOf: vocabURL),
              let vocabulary = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return nil }
        var specials = [String: Int]()
        for token in ["<s>", "<pad>", "</s>", "<unk>", "<mask>"] {
            if let id = vocabulary[token] as? Int { specials[token] = id }
        }
        let manifest: [String: Any] = ["tokenizer": ["type": "bpe-bytelevel", "specialTokens": specials]]
        return try? NFKTokenizer(forManifest: manifest, directory: directory)
    }

    /// The `[1, side, side, 3]` NHWC pixel tensor: the image's 8-bit RGB resampled to `side` exactly as
    /// PIL resamples it (bilinear for the ViT releases, bicubic for the DeiT ones), then scaled to
    /// `[-1, 1]` (mean 0.5, std 0.5), matching the reference `ViTImageProcessor` / `DeiTImageProcessor`.
    public static func pixelValues(_ image: Any, side: Int = 384, bicubic: Bool = false) throws -> MLXArray {
        let (rgba, width, height) = try NFKMLXImageBridge.rgbaBytes(from: image, colorSpace: CGColorSpaceCreateDeviceRGB())
        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        for pixel in 0 ..< width * height {
            rgb[pixel * 3] = rgba[pixel * 4]
            rgb[pixel * 3 + 1] = rgba[pixel * 4 + 1]
            rgb[pixel * 3 + 2] = rgba[pixel * 4 + 2]
        }
        let resized = NFKMLXPILResample.resampled(rgb, width: width, height: height, toWidth: side, toHeight: side,
                                                  filter: bicubic ? .bicubic : .bilinear)
        let pixels = MLXArray(resized.map { Float($0) / 255 }).reshaped([1, side, side, 3])
        return (pixels - 0.5) / 0.5
    }

    /// Whether the release's `preprocessor_config.json` resamples bicubic (PIL code 3), as the DeiT
    /// releases do; the ViT releases resample bilinear (2).
    static func resamplesBicubic(inDirectory directory: URL) -> Bool {
        guard let data = try? Data(contentsOf: directory.appendingPathComponent("preprocessor_config.json")),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return false }
        return (json["resample"] as? NSNumber)?.intValue == 3
    }

    /// The transcription with the RoBERTa markers removed.
    public static func cleanText(_ text: String) -> String {
        var out = text
        for marker in ["<s>", "</s>", "<pad>"] { out = out.replacingOccurrences(of: marker, with: "") }
        return out.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

// MARK: - Factory

/// TrOCR: a text-line reader (`microsoft/trocr-*`, MIT): handwritten, printed, and scene-text lines.
///
/// @discussion The backend reads one image to a transcription (`NFKOutputText`). A released
/// directory supplies the image-encoder / trocr-decoder weights, the `config.json` geometry, the
/// `preprocessor_config.json` resample, and the tokenizer (`vocab.json` / `merges.txt`, or a small
/// release's `sentencepiece.bpe.model`).
///
/// Introduced in InferKit 0.4.0.
@objc(NFKMLXTrOCR)
public final class NFKMLXTrOCR: NSObject {

    @objc public static let modelName = "trocr-base-handwritten"
    static let requiredFiles = ["config.json"]
    static let optionalFiles = ["vocab.json", "merges.txt", "sentencepiece.bpe.model", "preprocessor_config.json"]
    static let weightFiles = ["model.safetensors", "model.safetensors.index.json", "pytorch_model.bin"]

    /// Builds a backend from a released TrOCR directory: the weights, `config.json`, the tokenizer
    /// files, and `preprocessor_config.json` when the release ships one.
    @objc(backendWithDirectoryURL:error:)
    public static func backend(directoryURL: URL) throws -> NFKMLXTrOCRBackend {
        let net = try NFKMLXTrOCRNet(configurationURL: directoryURL.appendingPathComponent("config.json"))
        try net.loadWeights(fromDirectory: directoryURL)
        guard let tokenizer = NFKMLXTrOCRProcessor.tokenizer(inDirectory: directoryURL) else {
            throw NFKMLXError.unsupportedConfiguration(
                "TrOCR needs vocab.json and merges.txt, or sentencepiece.bpe.model; neither is readable")
        }
        return NFKMLXTrOCRBackend(net: net, tokenizer: tokenizer,
                                  bicubic: NFKMLXTrOCRProcessor.resamplesBicubic(inDirectory: directoryURL))
    }

    /// Downloads a release into the hub cache and builds the backend.
    ///
    /// @discussion The download fetches `config.json`, the tokenizer files the release holds
    /// (`vocab.json` and `merges.txt`, or `sentencepiece.bpe.model`), `preprocessor_config.json`, and the
    /// weights: `model.safetensors`, a shard index, or `pytorch_model.bin`, in that preference. A file the
    /// cache already holds is not fetched again. The call blocks on the network; call it off the render
    /// thread. The public releases are `microsoft/trocr-{small,base,large}-{handwritten,printed,stage1}`
    /// and `microsoft/trocr-{base,large}-str`; none is gated.
    ///
    /// Introduced in InferKit 0.4.0.
    @objc(backendWithRepo:revision:cacheDirectoryURL:error:)
    public static func backend(repo: String, revision: String?, cacheDirectoryURL: URL?) throws -> NFKMLXTrOCRBackend {
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
                               completionHandler: @escaping (NFKMLXTrOCRBackend?, Error?) -> Void) {
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                completionHandler(try backend(repo: repo, revision: revision, cacheDirectoryURL: cacheDirectoryURL), nil)
            } catch {
                completionHandler(nil, error)
            }
        }
    }

    /// The maximum tokens generated after the start token.
    @objc public static var maximumTokens = 64
    /// The beam count for generation; greedy (1) matches the release's generation config.
    @objc public static var beams = 1
}

// MARK: - Backend

/// The TrOCR inference backend: one image in, its transcription (`NFKOutputText`) out.
public final class NFKMLXTrOCRBackend: NSObject, NFKInferenceBackend {
    private let holder: Holder
    private let tokenizer: NFKTokenizer
    private let bicubic: Bool

    final class Holder: @unchecked Sendable { let net: NFKMLXTrOCRNet; init(_ n: NFKMLXTrOCRNet) { net = n } }

    init(net: NFKMLXTrOCRNet, tokenizer: NFKTokenizer, bicubic: Bool) {
        self.holder = Holder(net)
        self.tokenizer = tokenizer
        self.bicubic = bicubic
    }

    public var supportedInputKeys: Set<String> { [NFKInputImage] }
    public var isReady: Bool { true }
    public var backendIdentifier: String { NFKMLXTrOCR.modelName }

    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        guard let imageValue = request.input(forKey: NFKInputImage) else {
            throw NFKMLXError.unsupportedConfiguration("TrOCR needs an NFKInputImage")
        }
        let net = holder.net
        let pixels = try NFKMLXTrOCRProcessor.pixelValues(imageValue, side: net.visionConfiguration.imageSize,
                                                          bicubic: bicubic)
        let memory = net.imageFeatures(pixels)
        let model = NFKTrOCRDecodeModel(net: net, memory: memory)
        let configuration = net.languageConfiguration
        let decoding = NFKMLXSeq2SeqDecoding(beams: NFKMLXTrOCR.beams, maxTokens: NFKMLXTrOCR.maximumTokens,
                                            startToken: configuration.decoderStartTokenId,
                                            endToken: configuration.eosTokenId)
        let ids = NFKMLXSeq2SeqDecoder.generate(model, source: [configuration.decoderStartTokenId], decoding: decoding)
        let text = tokenizer.decode(ids.map { NSNumber(value: $0) })
        return NFKInferenceResult(outputs: [NFKOutputText: NFKMLXTrOCRProcessor.cleanText(text)])
    }
}
