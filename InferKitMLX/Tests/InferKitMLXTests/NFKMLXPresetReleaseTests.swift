//
//  NFKMLXPresetReleaseTests.swift
//  InferKitMLXTests
//
//  A preset is what a caller with no config.json builds from, so it has to equal what the family's
//  reader makes of its release's config.json in every stored field. A structural check sees only the
//  fields that decide a tensor's shape; a rotary base, a routing scale, a soft-cap, or a window does
//  not change a shape and is invisible to it. Every stored property is compared through reflection, so
//  a field added later is covered without naming it here.
//
//  The config paths come from `IK_SHAPES_ROOT/<release>/config.json` and the usual `IK_VAL_*` /
//  `IK_CONFIG_*` keys, provisioned through `~/.inferkit-validation.json`.
//

import XCTest
import Foundation
@testable import InferKitMLX

final class NFKMLXPresetReleaseTests: XCTestCase {

    private var config: [String: String] { NFKMLXValidationConfig.environment }

    /// `IK_SHAPES_ROOT/<release>/config.json`.
    private func shapesConfig(_ release: String) throws -> URL {
        guard let root = config["IK_SHAPES_ROOT"] else { throw XCTSkip("set IK_SHAPES_ROOT") }
        return try existing(URL(fileURLWithPath: root).appendingPathComponent(release)
            .appendingPathComponent("config.json"))
    }

    /// `<key>/config.json` when `key` names a directory, `key` itself when it names a file.
    private func keyedConfig(_ key: String) throws -> URL {
        guard let path = config[key] else { throw XCTSkip("set \(key)") }
        var isDirectory: ObjCBool = false
        FileManager.default.fileExists(atPath: path, isDirectory: &isDirectory)
        let url = URL(fileURLWithPath: path)
        return try existing(isDirectory.boolValue ? url.appendingPathComponent("config.json") : url)
    }

    private func existing(_ url: URL) throws -> URL {
        guard FileManager.default.fileExists(atPath: url.path) else { throw XCTSkip("no \(url.path)") }
        return url
    }

    private func json(_ url: URL) throws -> [String: Any] {
        try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
    }

    /// The stored fields where `preset` and `read` differ, `label: preset …, release …`.
    private func differences<T>(_ preset: T, _ read: T) -> [String] {
        let released = Dictionary(uniqueKeysWithValues: Mirror(reflecting: read).children
            .compactMap { child in child.label.map { ($0, String(describing: child.value)) } })
        return Mirror(reflecting: preset).children.compactMap { child in
            guard let label = child.label else { return nil }
            let mine = String(describing: child.value)
            let theirs = released[label] ?? "absent"
            return mine == theirs ? nil : "\(label): preset \(mine.prefix(80)), release \(theirs.prefix(80))"
        }
    }

    /// Compares every `(name, preset, read)` that resolves; a missing config skips only its own entry.
    private func assertPresets<T>(_ family: String, _ entries: [(String, T, () throws -> T)],
                                  file: StaticString = #filePath, line: UInt = #line) throws {
        var compared = 0
        for (name, preset, read) in entries {
            let released: T
            do { released = try read() } catch is XCTSkip { continue }
            compared += 1
            let found = differences(preset, released)
            print("SEAM preset \(family).\(name) differs from its release in: "
                  + (found.isEmpty ? "nothing" : found.joined(separator: " | ")))
            XCTAssertEqual(found, [], "\(family).\(name) is its release's configuration", file: file, line: line)
        }
        if compared == 0 { throw XCTSkip("no \(family) release config provisioned") }
    }

    func testQwen3AndMistralPresetsAreTheirReleases() throws {
        let read = { (url: URL) in try NFKMLXLanguage.configuration(fromHuggingFace: url) }
        try assertPresets("NFKMLXLanguageConfiguration", [
            ("qwen3_0_6B", .qwen3_0_6B, { try read(self.keyedConfig("IK_VAL_QWEN3")) }),
            ("qwen3_1_7B", .qwen3_1_7B, { try read(self.keyedConfig("IK_VAL_QWEN3_1_7B")) }),
            ("qwen3_4B", .qwen3_4B, { try read(self.keyedConfig("IK_VAL_QWEN3_4B")) }),
            ("qwen3_8B", .qwen3_8B, { try read(self.shapesConfig("qwen3-8b")) }),
            ("qwen3_14B", .qwen3_14B, { try read(self.shapesConfig("qwen3-14b")) }),
            ("qwen3_32B", .qwen3_32B, { try read(self.shapesConfig("qwen3-32b")) }),
            ("mistralSmall3", .mistralSmall3, { try read(self.shapesConfig("mistral-small-3.2-24b")) }),
        ])
    }

    func testGemmaPresetsAreTheirReleases() throws {
        let gemma3 = { (url: URL) in try NFKMLXGemma3Language.configuration(fromHuggingFace: url) }
        try assertPresets("NFKMLXGemma3Configuration", [
            ("gemma3_270M", .gemma3_270M, { try gemma3(self.keyedConfig("IK_VAL_GEMMA3_270M")) }),
            ("gemma3_1B", .gemma3_1B, { try gemma3(self.keyedConfig("IK_VAL_GEMMA3_1B")) }),
            ("gemma3_4B", .gemma3_4B, { try gemma3(self.keyedConfig("IK_VAL_GEMMA3_4B")) }),
        ])
        try assertPresets("NFKMLXGemmaConfiguration", [
            ("e2b", .e2b, { try NFKMLXGemmaLanguage.configuration(fromHuggingFace: self.keyedConfig("IK_CONFIG_GEMMA4")) }),
            ("twelveB", .twelveB, {
                try NFKMLXGemmaLanguage.unifiedConfiguration(fromHuggingFace: self.shapesConfig("gemma-4-12b"))
            }),
        ])
        let gemma2 = { (url: URL) in try NFKMLXGemma2Configuration.configuration(fromHuggingFace: url) }
        try assertPresets("NFKMLXGemma2Configuration", [
            ("gemma2_9B", .gemma2_9B, { try gemma2(self.shapesConfig("gemma-2-9b")) }),
            ("gemma2_27B", .gemma2_27B, { try gemma2(self.shapesConfig("gemma-2-27b")) }),
        ])
    }

    func testImageAndVideoTransformerPresetsAreTheirReleases() throws {
        try assertPresets("NFKMLXFluxConfiguration", [
            ("schnell", .schnell, {
                try NFKMLXFluxTransformerNet.configuration(fromHuggingFace: self.keyedConfig("IK_VAL_FLUX_SCHNELL"))
            }),
            ("dev", .dev, { try NFKMLXFluxTransformerNet.configuration(fromHuggingFace: self.shapesConfig("flux-dev")) }),
        ])
        let flux2 = { (url: URL) in try NFKMLXFlux2TransformerNet.configuration(fromHuggingFace: url) }
        try assertPresets("NFKMLXFlux2Configuration", [
            ("klein4B", .klein4B, { try flux2(self.shapesConfig("flux2-klein-4b")) }),
            ("klein9B", .klein9B, { try flux2(self.shapesConfig("flux2-klein-base-9b")) }),
        ])
        let sd3 = { (url: URL) in try NFKMLXSD3TransformerNet.configuration(fromHuggingFace: url) }
        try assertPresets("NFKMLXSD3Configuration", [
            ("sd35Large", .sd35Large, { try sd3(self.shapesConfig("sd35-large")) }),
            ("sd35Medium", .sd35Medium, { try sd3(self.shapesConfig("sd35-medium")) }),
        ])
        try assertPresets("NFKMLXLTX2Configuration", [
            ("ltx23", .ltx23, { try NFKMLXLTX2TransformerNet.configuration(fromHuggingFace: self.shapesConfig("ltx2-23-22b")) }),
        ])
        try assertPresets("NFKMLXQwenImageConfiguration", [
            ("base", .base, {
                try NFKMLXQwenImage.configuration(fromHuggingFace: self.shapesConfig("qwen-image-2.1-transformer"))
            }),
        ])
    }

    func testRecurrentAndHybridPresetsAreTheirReleases() throws {
        try assertPresets("NFKMLXHybridConfiguration", [
            ("qwen3_8_27B", .qwen3_8_27B, {
                try NFKMLXHybridLanguage.configuration(fromHuggingFace: self.keyedConfig("IK_CONFIG_QWEN3_8"))
            }),
        ])
        try assertPresets("NFKMLXQwen4ExpConfiguration", [
            ("qwen3_8FlashNext", .qwen3_8FlashNext, {
                try NFKMLXQwen4Exp.configuration(fromHuggingFace: self.shapesConfig("qwen3.8-flash-next"))
            }),
        ])
        try assertPresets("NFKMLXMamba2Configuration", [
            ("codestral7B", .codestral7B, {
                try NFKMLXMamba.configuration(fromDirectory: self.keyedConfig("IK_CONFIG_MAMBA2").deletingLastPathComponent())
            }),
        ])
        try assertPresets("NFKMLXNemotronHConfiguration", [
            ("nano9B", .nano9B, {
                try NFKMLXNemotronHConfiguration.configuration(
                    fromHuggingFace: self.json(self.keyedConfig("IK_CONFIG_NEMOTRON_H")))
            }),
        ])
        try assertPresets("NFKMLXGraniteHybridConfiguration", [
            ("h1B", .h1B, {
                try NFKMLXGraniteHybridConfiguration.configuration(
                    fromHuggingFace: self.json(self.keyedConfig("IK_CONFIG_GRANITE")))
            }),
        ])
        try assertPresets("NFKMLXPixtralVisionConfiguration", [
            ("pixtral12B", .pixtral12B, {
                try NFKMLXPixtralVisionConfiguration.configuration(fromHuggingFace: self.keyedConfig("IK_VAL_PIXTRAL"))
            }),
        ])
    }
}
