//
//  NFKMLXLazyReadSitesTests.swift
//  InferKitMLXTests
//
//  Every lazy weight read in the package's sources, held to a list. A loader that applies a release or
//  a checkpoint reads it first (`NFKMLXReleaseWeights.materializedArrays`, `arrays(inDirectory:converting:)`,
//  `NFKMLXWeights.materializedCheckpoint`), so the model's first evaluation waits on no file read. A lazy
//  read is right only where the loader probes a few keys, takes one part of a shared file, or builds
//  new tensors from many of its tensors, and each such file is listed below with its reason. Reads the
//  source text, so it runs under `swift test` without MLX.
//

import XCTest

final class NFKMLXLazyReadSitesTests: XCTestCase {

    /// Files that call the lazy `NFKMLXReleaseWeights.arrays(inDirectory:precision:remap:)`, with the count.
    private static let lazyReleaseReads: [String: (count: Int, reason: String)] = [
        "NFKMLXFlux2Transformer.swift": (1, "the latent codec takes two bn statistics from the autoencoder"),
        "NFKMLXLanguageBackend.swift": (1, "per-expert tensors are read as their stack forms"),
        "NFKMLXPhi4MM.swift": (1, "the decoder's LoRA fold spans tensors"),
        "NFKMLXSa2VAQwen.swift": (1, "a probe for one key's presence"),
    ]

    /// Files that call the lazy `NFKMLXWeights.loadCheckpoint(url:)`, with the count.
    private static let lazyCheckpointReads: [String: (count: Int, reason: String)] = [
        "NFKMLXAdaIN.swift": (1, "the encoder keeps VGG layers through conv4_1"),
        "NFKMLXBasicPitch.swift": (1, "a probe for log_norm.weight"),
        "NFKMLXBiSeNet.swift": (1, "the aux heads, the file's largest tensors, are dropped"),
        "NFKMLXChatterbox.swift": (1, "the tokenizer's part of s3gen, which the vocoder also reads"),
        "NFKMLXChatterboxHiFT.swift": (1, "weight-norm fusion"),
        "NFKMLXChatterboxPipeline.swift": (1, "conditioning tensors read as host values"),
        "NFKMLXConvTasNet.swift": (1, "a probe for the configuration's shapes"),
        "NFKMLXCosmosTokenizerTraining.swift": (1, "VGG16's features without its classifier"),
        "NFKMLXDAC.swift": (1, "weight-norm fusion"),
        "NFKMLXDDColor.swift": (1, "spectral-norm fusion"),
        "NFKMLXDeepSeekRelease.swift": (3, "paged experts are mapped, never read; image and draft stacks share shards"),
        "NFKMLXDenoiserTraining.swift": (1, "a probe for the encoder's width"),
        "NFKMLXDepthAnythingTraining.swift": (1, "the encoder without its head"),
        "NFKMLXFRCRN.swift": (1, "the checkpoint's duplicate stage copies are dropped"),
        "NFKMLXFlorence2.swift": (1, "the vision half; the language half loads separately"),
        "NFKMLXGraniteSpeech.swift": (1, "the adapter fold spans tensors"),
        "NFKMLXHiFiGAN.swift": (1, "weight-norm fusion"),
        "NFKMLXKokoro.swift": (2, "weight-norm fusion and stacked alphas; a voicepack read as data"),
        "NFKMLXMODNet.swift": (1, "the backbone's duplicate copy is dropped"),
        "NFKMLXMossFormer2SR.swift": (1, "weight-norm fusion"),
        "NFKMLXMusic3.swift": (1, "the vocoder's weight-norm fusion"),
        "NFKMLXQwen3VL.swift": (1, "a probe for lm_head.weight"),
        "NFKMLXRIFEv4Training.swift": (1, "VGG19's first features"),
        "NFKMLXRTDetr.swift": (1, "a probe for the class count"),
        "NFKMLXReleaseWeights.swift": (3, "the readers' own grouping reads each group as it is taken"),
        "NFKMLXResembleEnhance.swift": (1, "weight-norm fusion"),
        "NFKMLXSAM2.swift": (3, "one component each of a file the tracker reads whole"),
        "NFKMLXSDTextEncoder.swift": (1, "q, k, and v joined into one projection"),
        "NFKMLXSNAC.swift": (1, "weight-norm fusion"),
        "NFKMLXSa2VABackend.swift": (1, "a probe for a saved fine-tune's layout"),
        "NFKMLXSa2VALLaVA.swift": (1, "a probe for a saved fine-tune's layout"),
        "NFKMLXSa2VAQwen.swift": (1, "a probe for a saved fine-tune's layout"),
        "NFKMLXSegFormer.swift": (1, "key and value joined into one projection"),
        "NFKMLXSileroVAD.swift": (1, "the 8 kHz branch is dropped"),
        "NFKMLXSmolVLM.swift": (1, "a probe for lm_head.weight"),
        "NFKMLXStableDiffusionModels.swift": (1, "a stored text context read as data"),
        "NFKMLXVoiceRestoreVocoder.swift": (1, "weight-norm fusion"),
        "NFKMLXWeights.swift": (1, "materializedCheckpoint opens the file before reading it"),
        "NFKMLXYOLO.swift": (1, "a probe for the class count"),
        "NFKMLXYOLOGenerations.swift": (1, "a probe for the class count"),
    ]

    private func sources() throws -> [(name: String, text: String)] {
        let directory = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/InferKitMLX")
        let names = try XCTUnwrap(try? FileManager.default.contentsOfDirectory(atPath: directory.path),
                                  "the package's sources are not beside its tests at \(directory.path)")
        return try names.filter { $0.hasSuffix(".swift") }.sorted().map { name in
            let text = try String(contentsOf: directory.appendingPathComponent(name), encoding: .utf8)
            // Comment lines name the readers in prose.
            let code = text.split(separator: "\n", omittingEmptySubsequences: false)
                .filter { !$0.drop(while: { $0 == " " }).hasPrefix("//") }
                .joined(separator: "\n")
            return (name, code)
        }
    }

    /// The text of each call's argument list, from its opening parenthesis to the matching close.
    private func arguments(of call: String, in text: String) -> [Substring] {
        var found = [Substring]()
        var searchStart = text.startIndex
        while let range = text.range(of: call, range: searchStart ..< text.endIndex) {
            var depth = 1
            var index = range.upperBound
            while depth > 0, index < text.endIndex {
                if text[index] == "(" {
                    depth += 1
                } else if text[index] == ")" {
                    depth -= 1
                }
                index = text.index(after: index)
            }
            found.append(text[range.upperBound ..< index])
            searchStart = index
        }
        return found
    }

    private func assertMatches(_ counted: [String: Int], _ listed: [String: (count: Int, reason: String)],
                               reader: String, readFirst: String, file: StaticString = #filePath, line: UInt = #line) {
        for (name, count) in counted.sorted(by: { $0.key < $1.key }) where count != listed[name]?.count {
            XCTFail("\(name) calls \(reader) \(count) time(s), \(listed[name]?.count ?? 0) listed. A loader that "
                    + "applies what it reads uses \(readFirst); a lazy read that must stay lazy is listed here "
                    + "with its reason.", file: file, line: line)
        }
        for name in listed.keys.sorted() where counted[name] == nil {
            XCTFail("\(name) is listed for \(reader) but no longer calls it; drop its entry.", file: file, line: line)
        }
    }

    func testEveryLazyReleaseReadIsListed() throws {
        var counted = [String: Int]()
        for (name, text) in try sources() {
            let lazy = arguments(of: "NFKMLXReleaseWeights.arrays(", in: text).filter { !$0.contains("converting:") }
            if !lazy.isEmpty {
                counted[name] = lazy.count
            }
        }
        assertMatches(counted, Self.lazyReleaseReads, reader: "NFKMLXReleaseWeights.arrays(inDirectory:precision:remap:)",
                      readFirst: "materializedArrays(inDirectory:precision:remap:transform:)")
    }

    func testEveryLazyCheckpointReadIsListed() throws {
        var counted = [String: Int]()
        for (name, text) in try sources() {
            let definitions = text.components(separatedBy: "func loadCheckpoint(").count - 1
            let calls = arguments(of: "loadCheckpoint(", in: text).count - definitions
            if calls > 0 {
                counted[name] = calls
            }
        }
        assertMatches(counted, Self.lazyCheckpointReads, reader: "NFKMLXWeights.loadCheckpoint(url:)",
                      readFirst: "materializedCheckpoint(url:reading:)")
    }
}
