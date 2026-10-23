//
//  NFKMLXCatalogCompletenessTests.swift
//  InferKitMLXTests
//
//  The catalog describes every model the registry builds, and each description carries what an app
//  needs to list, download, and size the model. Building descriptors and registering factories touches
//  no MLX, so every test runs under `swift test` without a Metal library.
//

import XCTest
import InferKit
@testable import InferKitMLX

final class NFKMLXCatalogCompletenessTests: XCTestCase {

    /// The reference stand-ins and the two test geometries no released checkpoint fits.
    private static let standInNames: Set<String> = [
        "diffusion-controlnet", "diffusion-depth", "diffusion-inpaint", "diffusion-upscaler", "green-screen-keyer",
        "tone-speech", "sam", "swinir-x4",
    ]

    /// The variant sources the measurement does not size, by `model/variant`, with the reason.
    private static let unsizedSources: [String: String] = [
        "stable-diffusion-2-1-base/default":
            "the publisher no longer serves stabilityai/stable-diffusion-2-1-base publicly; a local copy still builds",
        "fast-style-transfer/default": "a GitHub project page, which serves no file size",
        "hat-x4/default": "a GitHub project page, which serves no file size",
        "modnet/default": "a GitHub project page, which serves no file size",
        "nuwave2/official": "a Google Drive file, which serves no size",
        "rife-v4/default": "a GitHub project page, which serves no file size",
        "storm/storm": "a Google Drive folder, which serves no size",
        "u2netp/default": "a GitHub project page, which serves no file size",
    ]

    /// The provider classes `swift run inferkit-catalog export` links: the core's Apple engines and this
    /// package's models.
    private static let snapshotProviders = ["NFKAppleModelDescriptors", "NFKMLXModelDescriptors"]

    private let descriptors = NFKMLXModelDescriptors.modelDescriptors()

    // A shipped model whose source downloads files is refused, never built, while those files are absent:
    // a factory given no weights builds its network on initial values, which run and answer nonsense.
    func testNoModelWithFilesBuildsWithoutThem() throws {
        NFKMLXReferenceModels.registerAll()
        let catalog = NFKModelCatalog()
        for descriptor in descriptors {
            catalog.addDescriptor(descriptor)
        }
        let empty = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: empty, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: empty) }
        catalog.locator = NFKModelCatalog.hubCacheLocator(forCacheDirectoryURL: empty)

        var refused = 0
        for descriptor in descriptors {
            for variant in descriptor.variants {
                guard let source = variant.source, source.layout != NFKModelSourceLayoutNone,
                      [NFKModelSourceKindHuggingFace, NFKModelSourceKindURL].contains(source.kind) else { continue }
                // A descriptor with one variant declares no variant option; its default is the variant built.
                let choosesVariant = descriptor.option(withKey: NFKModelLoadOptionVariant) != nil
                guard choosesVariant || variant.identifier == descriptor.defaultVariant?.identifier else { continue }
                var options: [String: Any] = [NFKModelLoadOptionIgnoreMemoryBudget: true]
                if choosesVariant {
                    options[NFKModelLoadOptionVariant] = variant.identifier
                }
                XCTAssertThrowsError(try catalog.backend(forModelIdentifier: descriptor.identifier, options: options),
                                     "\(descriptor.identifier)/\(variant.identifier)") { error in
                    let code = (error as NSError).code
                    XCTAssertTrue([NFKModelCatalogError.Code.notDownloaded.rawValue,
                                   NFKModelCatalogError.Code.unavailable.rawValue].contains(code),
                                  "\(descriptor.identifier)/\(variant.identifier): \(error)")
                    if code == NFKModelCatalogError.Code.notDownloaded.rawValue { refused += 1 }
                }
            }
        }
        XCTAssertGreaterThan(refused, 300, "the shipped variants with files")
    }

    func testTheRegistryAndTheCatalogNameTheSameModels() {
        NFKMLXReferenceModels.registerAll()
        // A test that registers a model of its own names it with the "test-" prefix.
        let registered = Set(NFKMLXModelRegistry.registeredModelNames.filter { !$0.hasPrefix("test-") })
        let cataloged = Set(descriptors.flatMap { descriptor in
            [descriptor.registryName].compactMap { $0 } + descriptor.variants.compactMap(\.registryName)
        })
        XCTAssertEqual(registered.subtracting(cataloged).sorted(), [], "registered without a descriptor")
        XCTAssertEqual(cataloged.subtracting(registered).sorted(), [], "described without a registration")
    }

    /// Every model that takes a token limit declares the most tokens one reply may run to, so a caller that
    /// sets none can send the model's greatest.
    func testEveryTokenLimitDeclaresItsMaximum() throws {
        for descriptor in descriptors {
            guard let option = descriptor.options.first(where: { $0.key == NFKParameterMaxTokens }) else { continue }
            XCTAssertGreaterThan(option.maximum?.intValue ?? 0, 0, descriptor.identifier)
        }
        let qwen = try XCTUnwrap(descriptors.first { $0.identifier == "qwen3-0.6b" })
        XCTAssertEqual(qwen.options.first { $0.key == NFKParameterMaxTokens }?.maximum, 40_960,
                       "a decoder's reply runs at most to its trained context")
    }

    func testEveryIdentifierIsUnique() {
        let identifiers = descriptors.map(\.identifier)
        XCTAssertEqual(Set(identifiers).count, identifiers.count,
                       "duplicates: \(Dictionary(grouping: identifiers, by: { $0 }).filter { $1.count > 1 }.keys.sorted())")
        for descriptor in descriptors {
            let variants = descriptor.variants.map(\.identifier)
            XCTAssertEqual(Set(variants).count, variants.count, descriptor.identifier)
        }
        let apple = NFKAppleModelDescriptors.modelDescriptors().map(\.identifier)
        XCTAssertEqual(Set(apple).intersection(identifiers).sorted(), [], "an Apple engine and an MLX model share a name")
    }

    func testTheStandInsAreTheEightTestModels() {
        XCTAssertEqual(Set(descriptors.filter(\.isStandIn).map(\.identifier)), Self.standInNames)
    }

    func testEveryShippedModelDescribesWhatAnAppNeeds() {
        for descriptor in descriptors where !descriptor.isStandIn {
            let name = descriptor.identifier
            XCTAssertFalse(descriptor.outputs.isEmpty, "\(name) declares no output")
            XCTAssertFalse(descriptor.tasks.isEmpty, "\(name) declares no task")
            XCTAssertNotNil(descriptor.license, "\(name) states no license")
            XCTAssertNotNil(descriptor.defaultVariant?.source, "\(name) has no default variant with a source")
        }
    }

    func testEveryMeasurableSourceHasADownloadSize() {
        var seen = Set<String>()
        for descriptor in descriptors where !descriptor.isStandIn {
            for variant in descriptor.variants {
                guard let kind = variant.source?.kind,
                      [NFKModelSourceKindHuggingFace, NFKModelSourceKindURL].contains(kind) else { continue }
                let key = "\(descriptor.identifier)/\(variant.identifier)"
                if Self.unsizedSources[key] != nil {
                    seen.insert(key)
                    XCTAssertNil(variant.downloadByteCount, "\(key) is listed as unsized but has a size")
                } else {
                    XCTAssertGreaterThan(variant.downloadByteCount?.int64Value ?? 0, 0, "\(key) has no download size")
                }
            }
        }
        XCTAssertEqual(Set(Self.unsizedSources.keys).subtracting(seen).sorted(), [], "listed but not in the catalog")
    }

    func testTheCommittedSnapshotIsTheCatalogsManifest() throws {
        let catalog = NFKModelCatalog()
        for provider in Self.snapshotProviders {
            XCTAssertTrue(catalog.addDescriptors(fromProviderClassName: provider), provider)
        }
        let generated = try catalog.manifestData() + Data("\n".utf8)
        let snapshot = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("Docs/model-catalog.json")
        let committed = try Data(contentsOf: snapshot)
        XCTAssertTrue(generated == committed,
                      "Docs/model-catalog.json differs from the catalog. Regenerate it from InferKitMLX/: "
                          + "swift run inferkit-catalog export --output ../Docs/model-catalog.json")
    }
}
