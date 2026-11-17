//
//  NFKMLXVisionChatTests.swift
//  InferKitMLXTests
//
//  The chat over images Qwen3-VL, Pixtral, and SmolVLM2 share. On a tiny Qwen3-VL, whose rotary reads the
//  images and whose deepstack adds to their positions: the engine's splice is the model's own, a follow-up
//  that adds a turn and an image resumes from the kept cache and gives the logits a full pass gives, its kept
//  images not encoded again, another image at a kept image's place shares nothing past it, a conversation
//  places each image in the turn that carried it, and the backend continues conversations keyed or not and
//  from disk. Pixtral and SmolVLM2 continue a conversation through their backends too.
//

import CoreGraphics
import ImageIO
import XCTest
import InferKit
import MLX
import MLXNN
@testable import InferKitMLX

final class NFKMLXVisionChatTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    override func tearDown() {
        if NFKMLXGPU.metalLibraryURL != nil {
            NFKMLXGPU.clearCache()
        }
        super.tearDown()
    }

    /// A Qwen-format template that writes each image part as one `<|image_pad|>` between its markers.
    static let template = "{% for message in messages %}<|im_start|>{{ message.role }}\n{% if message.content is string %}"
        + "{{ message.content }}{% else %}{% for part in message.content %}{% if part.type == 'image' %}"
        + "<|vision_start|><|image_pad|><|vision_end|>{% elif part.type == 'text' %}{{ part.text }}{% endif %}"
        + "{% endfor %}{% endif %}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"

    /// A tiny Qwen3-VL with a random decoder: an 8 × 8 image is four merged tokens, a 16 × 8 one eight.
    private func qwen(seed: UInt64 = 7) throws -> NFKMLXQwen3VL {
        NFKMLXRandom.seed(seed)
        let decoder = NFKMLXLanguage.makeNet(NFKMLXQwen3VLBackendTests.decoder)
        eval(decoder)
        let model = NFKMLXQwen3VL(visionNet: NFKMLXQwen3VLVisionNet(NFKMLXQwen3VLBackendTests.vision), decoder: decoder,
                                  tokenizer: try NFKMLXQwen3VLBackendTests.tokenizer(), endTokens: [501],
                                  processor: NFKMLXQwen3VLImageProcessor(patchSize: 2, temporalPatchSize: 2, mergeSize: 2,
                                                                         minPixels: 16, maxPixels: 128))
        model.releaseTemplate = NFKMLXVisionTemplate(source: Self.template)
        return model
    }

    /// A `width × height` image whose pixels follow `seed`, so two seeds give two images.
    private func image(width: Int = 8, height: Int = 8, seed: Int) -> CGImage {
        var bytes = (0 ..< width * height * 4).map { UInt8(truncatingIfNeeded: ($0 * 37 + seed * 101) % 251) }
        for pixel in 0 ..< width * height { bytes[pixel * 4 + 3] = 255 }
        let context = CGContext(data: &bytes, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4,
                                space: CGColorSpaceCreateDeviceRGB(),
                                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        return context.makeImage()!
    }

    private func worst(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }

    /// The logits after the last of `inputIds` from one uncached pass over the whole prompt.
    private func fullPass(_ model: NFKMLXVisionChatModel, _ inputIds: [Int], _ images: [NFKMLXVisionImage]) throws -> MLXArray {
        let positions = model.chatPositions(inputIds: inputIds, images: images)
        let input = try NFKMLXVisionChat.freshInput(model: model, inputIds: inputIds, images: images, from: 0,
                                                    positions: positions)
        let hidden = model.chatDecoder.hiddenStates(fromEmbeddings: input.embeddings, multimodal: input.multimodal)
        let logits = model.chatDecoder.logits(fromHidden: hidden)[0, inputIds.count - 1]
        eval(logits)
        return logits
    }

    private func ids(_ model: NFKMLXVisionChatModel, _ messages: [[String: Any]], _ images: [NFKMLXVisionImage]) throws -> [Int] {
        model.chatTokenizer.encode(try NFKMLXVisionPrompt.render(messages, images: images, model: model)).map(\.intValue)
    }

    private func user(_ text: String, images: Int = 0) -> [String: Any] {
        var message: [String: Any] = ["role": "user", "content": text]
        if images > 0 {
            message[NFKMessageMetadataKey] = ["imageCount": images]
        }
        return message
    }

    // MARK: The engine

    // The engine's splice, rotary, and deepstack over a lone question are the model's own.
    func testTheEnginesInputIsTheModelsOwnSplice() throws {
        try requireMLXRuntime()
        let model = try qwen()
        let prepared = model.prepareChatImage(image(seed: 1))
        let inputIds = model.chatTokenizer.encode(model.singleTurnPrompt(question: "What is it?", images: [prepared]))
            .map(\.intValue)
        let (features, deepstack) = model.encodeChatImage(prepared)
        let own = NFKMLXQwen3VL.hiddenStates(decoder: model.textDecoder, inputIds: inputIds, visionFeatures: features,
                                             deepstack: deepstack, gridT: prepared.grid.t, gridH: prepared.grid.h,
                                             gridW: prepared.grid.w)
        let logits = try fullPass(model, inputIds, [prepared])
        let expected = model.textDecoder.logits(fromHidden: own)[0, inputIds.count - 1]
        XCTAssertLessThan(worst(logits, expected), 1e-5)
    }

    // A follow-up that adds a reply, a turn, and a third image resumes from the kept cache in passes of five
    // and gives the logits a full pass gives; only the new image runs through the vision tower.
    func testAFollowUpResumesFromTheKeptCacheAsAFullPassReadsIt() throws {
        try requireMLXRuntime()
        let counting = NFKMLXCountingVisionModel(try qwen())
        let first = [image(seed: 1), image(width: 16, seed: 2)].map(counting.prepareChatImage)
        let opening = [user("Compare the two pictures.", images: 2)]
        let firstIds = try ids(counting, opening, first)
        let cache = NFKMLXPromptCache(layerCount: counting.chatDecoder.configuration.layerCount)

        func turn(_ inputIds: [Int], _ images: [NFKMLXVisionImage]) throws -> (start: Int, logits: MLXArray) {
            let key = try NFKMLXVisionChat.key(inputIds: inputIds, images: images, imageTokenId: counting.chatImageTokenId)
            let start = cache.align(to: key)
            let positions = counting.chatPositions(inputIds: inputIds, images: images)
            let input = try NFKMLXVisionChat.freshInput(model: counting, inputIds: inputIds, images: images, from: start,
                                                        positions: positions)
            let hidden = NFKMLXVisionChat.prefill(counting.chatDecoder, embeddings: input.embeddings,
                                                  multimodal: input.multimodal, cache: cache.cache, limit: 5)
            cache.record(Array(key[start...]))
            let logits = counting.chatDecoder.logits(fromHidden: hidden)[0, hidden.dim(1) - 1]
            eval(logits)
            return (start, logits)
        }

        let opened = try turn(firstIds, first)
        XCTAssertEqual(opened.start, 0)
        XCTAssertEqual(counting.encoded, 2)
        XCTAssertLessThan(worst(opened.logits, try fullPass(counting.inner, firstIds, first)), 1e-4)

        let all = first + [counting.prepareChatImage(image(seed: 3))]
        let followUp = opening + [["role": "assistant", "content": "They differ."], user("And this one?", images: 1)]
        let secondIds = try ids(counting, followUp, all)
        let resumed = try turn(secondIds, all)
        XCTAssertEqual(resumed.start, firstIds.count, "the whole kept prompt")
        XCTAssertEqual(counting.encoded, 3, "the kept images are not encoded again")
        XCTAssertLessThan(worst(resumed.logits, try fullPass(counting.inner, secondIds, all)), 1e-4)
    }

    // A prompt with another image where the kept prompt had one shares nothing from that image's first position.
    func testAnotherImageAtAKeptImagesPlaceSharesNothingPastIt() throws {
        try requireMLXRuntime()
        let model = try qwen()
        let kept = model.prepareChatImage(image(seed: 1)), other = model.prepareChatImage(image(seed: 4))
        let inputIds = try ids(model, [user("What is it?", images: 1)], [kept])
        let keptKey = try NFKMLXVisionChat.key(inputIds: inputIds, images: [kept], imageTokenId: model.chatImageTokenId)
        let otherKey = try NFKMLXVisionChat.key(inputIds: inputIds, images: [other], imageTokenId: model.chatImageTokenId)
        let firstImage = try XCTUnwrap(inputIds.firstIndex(of: model.chatImageTokenId))
        XCTAssertEqual(zip(keptKey, otherKey).prefix { $0.0 == $0.1 }.count, firstImage)
        XCTAssertEqual(keptKey, try NFKMLXVisionChat.key(inputIds: inputIds, images: [model.prepareChatImage(image(seed: 1))],
                                                         imageTokenId: model.chatImageTokenId), "the same pixels key alike")
    }

    // Each message takes the images its metadata counts, before its text; without counts the first user turn
    // takes them all; counts that disagree with the request's images are refused.
    func testAConversationPlacesEachImageInTheTurnThatCarriedIt() throws {
        try requireMLXRuntime()
        let model = try qwen()
        let images = [image(seed: 1), image(seed: 2)].map(model.prepareChatImage)
        let pad = images[0].expansion
        let placed = try NFKMLXVisionPrompt.render(
            [user("First.", images: 1), ["role": "assistant", "content": "Yes."], user("Second.", images: 1)],
            images: images, model: model)
        XCTAssertEqual(placed, "<|im_start|>user\n<|vision_start|>\(pad)<|vision_end|>First.<|im_end|>\n"
            + "<|im_start|>assistant\nYes.<|im_end|>\n<|im_start|>user\n<|vision_start|>\(images[1].expansion)<|vision_end|>"
            + "Second.<|im_end|>\n<|im_start|>assistant\n")
        let first = try NFKMLXVisionPrompt.render([user("Both."), ["role": "assistant", "content": "Yes."], user("Again.")],
                                                  images: images, model: model)
        XCTAssertTrue(first.hasPrefix("<|im_start|>user\n<|vision_start|>\(pad)<|vision_end|><|vision_start|>"))
        XCTAssertThrowsError(try NFKMLXVisionPrompt.render([user("One.", images: 1)], images: images, model: model))
    }

    // MARK: The released model

    private func cosine(_ a: MLXArray, _ b: MLXArray) -> Double {
        let x = a.asType(.float32).asArray(Float.self).map(Double.init)
        let y = b.asType(.float32).asArray(Float.self).map(Double.init)
        let dot = zip(x, y).reduce(0) { $0 + $1.0 * $1.1 }
        let norms = sqrt(x.reduce(0) { $0 + $1 * $1 }) * sqrt(y.reduce(0) { $0 + $1 * $1 })
        return dot / norms
    }

    // Qwen3-VL 2B at float32: a follow-up turn that adds a reply, a question, and a crop of the picture resumes
    // from the kept cache, its prompt-end logits those of a fresh run of the whole prompt and of one uncached
    // pass, and the backend's greedy reply the one a run that keeps nothing writes.
    func testTheReleasedQwen3VLContinuesAConversationAsAFullPassDoes() throws {
        try requireMLXRuntime()
        let environment = NFKMLXValidationConfig.environment
        guard let release = environment["IK_VAL_QWEN3_VL"], let photoPath = environment["IK_VAL_IMAGE"],
              let source = CGImageSourceCreateWithURL(URL(fileURLWithPath: photoPath) as CFURL, nil),
              let photo = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw XCTSkip("set IK_VAL_QWEN3_VL and IK_VAL_IMAGE")
        }
        let crop = try XCTUnwrap(photo.cropping(to: CGRect(x: 0, y: 0, width: photo.width / 2, height: photo.height / 2)))
        let model = try NFKMLXQwen3VL.model(directoryURL: URL(fileURLWithPath: release))
        let backend = model.inferenceBackend()
        let opening = [user("Describe the picture in one sentence.", images: 1)]
        var inputs: [String: Any] = [NFKInputImage: photo, NFKInputMessages: opening]
        let first = try reply(backend, inputs, [NFKParameterConversationKey: "released", NFKParameterMaxTokens: 24])
        let followUp = opening + [["role": "assistant", "content": first.text],
                                  user("Is this a part of the same picture?", images: 1)]
        inputs[NFKInputImages] = [crop]
        inputs[NFKInputMessages] = followUp
        let continued = try reply(backend, inputs, [NFKParameterConversationKey: "released", NFKParameterMaxTokens: 16])
        let fresh = try reply(backend, inputs, [NFKMLXGenerationParameterKey.reusesPromptCache: false, NFKParameterMaxTokens: 16])
        XCTAssertGreaterThan(continued.cached, 0)
        XCTAssertEqual(continued.text, fresh.text, "the greedy reply")

        let firstImages = [model.prepareChatImage(photo)], allImages = firstImages + [model.prepareChatImage(crop)]
        let firstIds = try ids(model, opening, firstImages), secondIds = try ids(model, followUp, allImages)
        func prefilled(_ cache: NFKMLXPromptCache, _ inputIds: [Int], _ images: [NFKMLXVisionImage]) throws -> (Int, MLXArray) {
            let key = try NFKMLXVisionChat.key(inputIds: inputIds, images: images, imageTokenId: model.chatImageTokenId)
            let start = cache.align(to: key)
            let input = try NFKMLXVisionChat.freshInput(model: model, inputIds: inputIds, images: images, from: start,
                                                        positions: model.chatPositions(inputIds: inputIds, images: images))
            let hidden = NFKMLXVisionChat.prefill(model.textDecoder, embeddings: input.embeddings,
                                                  multimodal: input.multimodal, cache: cache.cache)
            cache.record(Array(key[start...]))
            let logits = model.textDecoder.logits(fromHidden: hidden)[0, hidden.dim(1) - 1]
            eval(logits)
            return (start, logits)
        }
        let kept = NFKMLXPromptCache(layerCount: model.textDecoder.configuration.layerCount)
        _ = try prefilled(kept, firstIds, firstImages)
        let (start, resumed) = try prefilled(kept, secondIds, allImages)
        let (_, whole) = try prefilled(NFKMLXPromptCache(layerCount: model.textDecoder.configuration.layerCount),
                                       secondIds, allImages)
        let uncached = try fullPass(model, secondIds, allImages)
        let resumedAgainstUncached = cosine(resumed, uncached), wholeAgainstUncached = cosine(whole, uncached)
        let resumedAgainstWhole = cosine(resumed, whole)
        print("VALIDATION PARITY qwen3-vl-2b-conversation: turn 2 resumed at \(start) of \(secondIds.count), "
              + "\(continued.cached) positions reused by the backend; prompt-end logit cosine resumed against one "
              + "uncached pass \(resumedAgainstUncached), a fresh cached run against it \(wholeAgainstUncached), "
              + "resumed against fresh \(resumedAgainstWhole); greedy reply \"\(continued.text)\"")
        XCTAssertEqual(start, firstIds.count, "the whole first prompt")
        XCTAssertEqual(resumed.argMax().item(Int.self), uncached.argMax().item(Int.self))
        XCTAssertGreaterThan(resumedAgainstUncached, 0.9999999999)
        XCTAssertGreaterThan(wholeAgainstUncached, 0.9999999999)
        XCTAssertGreaterThan(resumedAgainstWhole, 0.9999999999, "resuming adds nothing to a fresh run's error")
    }

    // MARK: The backends

    private func reply(_ backend: NFKInferenceBackend, _ inputs: [String: Any], _ parameters: [String: Any] = [:])
        throws -> (text: String, cached: Int) {
        var all: [String: Any] = [NFKParameterMaxTokens: 4]
        all.merge(parameters) { $1 }
        let result = try backend.runInference(for: NFKInferenceRequest(inputs: inputs, parameters: all))
        let usage = try XCTUnwrap(result.output(forKey: NFKOutputUsage) as? [String: Int])
        return (try XCTUnwrap(result.output(forKey: NFKOutputText) as? String), usage[NFKUsageCachedTokens] ?? -1)
    }

    // The backend continues a conversation with or without a key, answers as a run that keeps nothing, and a
    // released conversation continues from the spill directory.
    func testTheBackendContinuesConversationsKeyedOrNotAndFromDisk() throws {
        try requireMLXRuntime()
        let backend = NFKMLXVisionChatBackend(model: try qwen(), identifier: "qwen3-vl")
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("vision-chat-spill-\(UUID().uuidString)", isDirectory: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        backend.conversationSpillDirectoryURL = directory
        let pictures: [String: Any] = [NFKInputImage: image(seed: 1), NFKInputImages: [image(seed: 2)]]
        let opening = [user("Compare them.", images: 2)]
        for key in ["a", nil] as [String?] {
            let parameters: [String: Any] = key.map { [NFKParameterConversationKey: $0] } ?? [:]
            let name = key ?? "keyless"
            var inputs = pictures
            inputs[NFKInputMessages] = opening
            let first = try reply(backend, inputs, parameters)
            XCTAssertEqual(first.cached, 0, name)
            inputs[NFKInputMessages] = opening + [["role": "assistant", "content": first.text], user("Which is brighter?")]
            let continued = try reply(backend, inputs, parameters)
            XCTAssertGreaterThan(continued.cached, 0, "\(name) continues its conversation")
            XCTAssertEqual(continued.text, try reply(backend, inputs, [NFKMLXGenerationParameterKey.reusesPromptCache: false]).text,
                           name)
            XCTAssertGreaterThan(backend.releaseIdleConversationCaches(), 0, name)
            XCTAssertEqual(backend.conversationCacheCount, 0, name)
            inputs[NFKInputMessages] = opening + [["role": "assistant", "content": first.text], user("And the larger?")]
            XCTAssertGreaterThan(try reply(backend, inputs, parameters).cached, 0, "\(name) continues from disk")
            backend.resetPromptCache()
        }
    }

    // A new question about the same image reuses the kept image prefix, and one about another image does not.
    func testANewQuestionAboutTheSameImageReusesItsPrefix() throws {
        try requireMLXRuntime()
        let backend = NFKMLXVisionChatBackend(model: try qwen(), identifier: "qwen3-vl")
        let picture = image(seed: 1)
        let opening = "Look at this picture carefully and tell me "
        XCTAssertEqual(try reply(backend, [NFKInputImage: picture, NFKInputPrompt: opening + "what it shows."]).cached, 0)
        let again = try reply(backend, [NFKInputImage: picture, NFKInputPrompt: opening + "its colour."])
        XCTAssertGreaterThan(again.cached, opening.count, "the image's positions and the shared words are kept")
        let other = try reply(backend, [NFKInputImage: image(seed: 5), NFKInputPrompt: opening + "its colour."])
        XCTAssertEqual(other.cached, 0, "another image at the kept image's place leaves too little to continue")
    }

    // Pixtral's and SmolVLM2's backends continue a conversation and answer as a run that keeps nothing.
    func testPixtralAndSmolVLMContinueAConversation() throws {
        try requireMLXRuntime()
        NFKMLXRandom.seed(9)
        let pixtralTokenizer = try NFKMLXBackendTestSupport.tokenizer(specials: [
            "[IMG]": NFKMLXPixtral.imageTokenId, "<s>": 500, "[INST]": 501, "[/INST]": 502,
            "[IMG_BREAK]": 503, "[IMG_END]": 504, "</s>": 505])
        let pixtral = NFKMLXPixtral(
            visionNet: NFKMLXPixtralVisionNet(NFKMLXPixtralVisionConfiguration(hiddenSize: 32, depth: 2, headCount: 2,
                                                                                intermediateSize: 64, patchSize: 2,
                                                                                imageSize: 16)),
            connector: NFKMLXPixtralConnector(visionSize: 32, textSize: 64), decoder: NFKMLXLanguage.makeNet(.tiny),
            tokenizer: pixtralTokenizer, endTokens: [505], processor: NFKMLXPixtralImageProcessor(patchSize: 2, longestEdge: 8))
        pixtral.releaseTemplate = NFKMLXVisionTemplate(
            source: "{{ bos_token }}{% for message in messages %}{% if message.role == 'user' %}[INST]"
                + "{% if message.content is string %}{{ message.content }}{% else %}{% for chunk in message.content %}"
                + "{% if chunk.type == 'image' %}[IMG]{% else %}{{ chunk.text }}{% endif %}{% endfor %}{% endif %}[/INST]"
                + "{% else %}{{ message.content }}</s>{% endif %}{% endfor %}", beginToken: "<s>")

        let imageToken = 500
        let smolTokenizer = try NFKMLXBackendTestSupport.tokenizer(specials: [
            "<image>": imageToken, "<end_of_utterance>": 501, "<fake_token_around_image>": 502, "<global-img>": 503,
            "<|im_start|>": 504])
        let smol = NFKMLXSmolVLM(
            net: NFKMLXSmolVLMNet(vision: NFKMLXSigLIPNet(.tiny),
                                  connector: NFKMLXSmolVLMConnector(visionHidden: 32, decoderHidden: 64, scaleFactor: 2),
                                  decoder: NFKMLXLanguage.makeNet(.tiny), imageTokenId: imageToken),
            tokenizer: smolTokenizer, endToken: 501,
            release: NFKMLXSmolVLM.Release(vision: .tiny, decoder: .tiny, scaleFactor: 2, imageTokenId: imageToken,
                                           tileSize: 64, longestEdge: 128))
        smol.releaseTemplate = NFKMLXVisionTemplate(
            source: "<|im_start|>{% for message in messages %}{{ message.role | capitalize }}"
                + "{% if message.content[0].type == 'image' %}:{% else %}: {% endif %}{% for line in message.content %}"
                + "{% if line.type == 'text' %}{{ line.text }}{% elif line.type == 'image' %}<image>{% endif %}{% endfor %}"
                + "<end_of_utterance>\n{% endfor %}{% if add_generation_prompt %}Assistant:{% endif %}")

        for (label, backend, size) in [("pixtral", pixtral.inferenceBackend(), 8), ("smolvlm", smol.inferenceBackend(), 64)] {
            let opening = [user("Describe it.", images: 1)]
            var inputs: [String: Any] = [NFKInputImage: image(width: size, height: size, seed: 1), NFKInputMessages: opening]
            let first = try reply(backend, inputs, [NFKParameterConversationKey: label])
            inputs[NFKInputMessages] = opening + [["role": "assistant", "content": first.text], user("More detail.")]
            let continued = try reply(backend, inputs, [NFKParameterConversationKey: label])
            XCTAssertGreaterThan(continued.cached, 0, label)
            XCTAssertEqual(continued.text,
                           try reply(backend, inputs, [NFKMLXGenerationParameterKey.reusesPromptCache: false]).text, label)
        }
    }
}

/// A vision chat model that forwards to another and counts the images it encodes.
final class NFKMLXCountingVisionModel: NFKMLXVisionChatModel {
    let inner: NFKMLXVisionChatModel
    private(set) var encoded = 0

    init(_ inner: NFKMLXVisionChatModel) { self.inner = inner }

    var chatDecoder: NFKMLXLanguageNet { inner.chatDecoder }
    var chatTokenizer: NFKTokenizer { inner.chatTokenizer }
    var chatEndTokens: Set<Int> { inner.chatEndTokens }
    var chatImageTokenId: Int { inner.chatImageTokenId }
    var chatTemplate: NFKMLXVisionTemplate? { inner.chatTemplate }
    var chatImageMarker: String { inner.chatImageMarker }
    var chatTextAsParts: Bool { inner.chatTextAsParts }
    var chatModules: [Module] { inner.chatModules }
    var chatReleaseDirectoryURL: URL? { inner.chatReleaseDirectoryURL }
    var chatAdapterStrengths: [Float] { inner.chatAdapterStrengths }

    func prepareChatImage(_ image: CGImage) -> NFKMLXVisionImage { inner.prepareChatImage(image) }
    func singleTurnPrompt(question: String, images: [NFKMLXVisionImage]) -> String {
        inner.singleTurnPrompt(question: question, images: images)
    }
    func encodeChatImage(_ image: NFKMLXVisionImage) -> (features: MLXArray, deepstack: [MLXArray]) {
        encoded += 1
        return inner.encodeChatImage(image)
    }
    func chatPositions(inputIds: [Int], images: [NFKMLXVisionImage]) -> [[Int]]? {
        inner.chatPositions(inputIds: inputIds, images: images)
    }
    func chatRope(_ positions: [[Int]]) -> (cos: MLXArray, sin: MLXArray) { inner.chatRope(positions) }
}
