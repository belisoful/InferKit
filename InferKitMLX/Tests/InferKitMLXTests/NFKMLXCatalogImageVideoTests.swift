//
//  NFKMLXCatalogImageVideoTests.swift
//  InferKitMLXTests
//
//  The image and video models' catalog entries. Building descriptors and registering factories touches
//  no MLX, so every test runs under `swift test` without a Metal library.
//

import XCTest
import InferKit
@testable import InferKitMLX

final class NFKMLXCatalogImageVideoTests: XCTestCase {

    private var entries: [NFKMLXCatalogEntry] {
        NFKMLXCatalogImage.entries + NFKMLXCatalogVideo.entries
    }

    private func entry(_ name: String) -> NFKMLXCatalogEntry? {
        entries.first { $0.identifier == name }
    }

    func testEveryEntryBuildsADescriptor() {
        for entry in entries {
            XCTAssertNoThrow(try entry.descriptor(), entry.identifier)
        }
    }

    func testEachIdentifierIsUniqueAndIsItsRegistryName() {
        let identifiers = entries.map(\.identifier)
        XCTAssertEqual(Set(identifiers).count, identifiers.count,
                       "duplicate identifiers: \(Dictionary(grouping: identifiers, by: { $0 }).filter { $1.count > 1 }.keys)")
        for entry in entries {
            XCTAssertEqual(entry.registryName, entry.identifier, entry.identifier)
            XCTAssertEqual(entry.variants.compactMap(\.registryName), [entry.identifier], entry.identifier)
        }
    }

    func testEveryRegistryNameIsRegisteredAfterRegisterAll() {
        NFKMLXReferenceModels.registerAll()
        let registered = Set(NFKMLXModelRegistry.registeredModelNames)
        for entry in entries {
            XCTAssertTrue(registered.contains(entry.identifier), "\(entry.identifier) is not registered")
        }
    }

    func testEveryEntryHasALicenseAndADefaultVariantWithASource() throws {
        for entry in entries {
            let descriptor = try entry.descriptor()
            XCTAssertNotNil(descriptor.license, entry.identifier)
            XCTAssertNotNil(descriptor.defaultVariant?.source, entry.identifier)
            XCTAssertFalse(descriptor.tasks.isEmpty, entry.identifier)
        }
    }

    /// Asserts that the entry under `name` declares exactly the backend's key sets.
    private func assertPorts(_ name: String, inputs: Set<String>, parameters: Set<String>, outputs: Set<String>,
                             file: StaticString = #filePath, line: UInt = #line) {
        guard let entry = entry(name) else {
            XCTFail("no entry is named \(name)", file: file, line: line)
            return
        }
        XCTAssertEqual(Set(entry.inputs.map(\.key)), inputs, "\(name) inputs", file: file, line: line)
        XCTAssertEqual(Set(entry.parameters.map(\.key)), parameters, "\(name) parameters", file: file, line: line)
        XCTAssertEqual(Set(entry.outputs.map(\.key)), outputs, "\(name) outputs", file: file, line: line)
    }

    func testTheImageGeneratorsDeclareTheirBackendsKeys() {
        for name in [NFKMLXFlux.modelName, NFKMLXCatalogImage.fluxDevName] {
            assertPorts(name, inputs: NFKMLXFlux.backendInputKeys, parameters: NFKMLXFlux.backendParameterKeys,
                        outputs: [NFKOutputImage])
        }
        for name in [NFKMLXFlux2.modelName] + NFKMLXCatalogImage.flux2AdditionalReleases.map(\.name) {
            assertPorts(name, inputs: NFKMLXFlux2.backendInputKeys, parameters: NFKMLXFlux2.backendParameterKeys,
                        outputs: [NFKOutputImage])
        }
        for name in [NFKMLXSD3Generator.modelName] + NFKMLXCatalogImage.sd3AdditionalReleases.map(\.name) {
            assertPorts(name, inputs: NFKMLXSD3Generator.backendInputKeys,
                        parameters: NFKMLXSD3Generator.backendParameterKeys, outputs: [NFKOutputImage])
        }
        assertPorts(NFKMLXFluxControlNetGenerator.modelName, inputs: NFKMLXFluxControlNetGenerator.backendInputKeys,
                    parameters: NFKMLXFluxControlNetGenerator.backendParameterKeys, outputs: [NFKOutputImage])
        assertPorts(NFKMLXSD3ControlNetGenerator.modelName, inputs: NFKMLXSD3ControlNetGenerator.backendInputKeys,
                    parameters: NFKMLXSD3ControlNetGenerator.backendParameterKeys, outputs: [NFKOutputImage])
        for name in [NFKMLXSANAGenerator.modelName, NFKMLXCatalogImage.sana1600MName] {
            assertPorts(name, inputs: NFKMLXSANAGenerator.backendInputKeys,
                        parameters: NFKMLXSANAGenerator.backendParameterKeys, outputs: [NFKOutputImage])
        }
        for name in [NFKMLXZImageGenerator.modelName, NFKMLXCatalogImage.zImageBaseName] {
            assertPorts(name, inputs: NFKMLXZImageGenerator.backendInputKeys,
                        parameters: NFKMLXZImageGenerator.backendParameterKeys, outputs: [NFKOutputImage])
        }
        assertPorts(NFKMLXQwenImageGenerator.modelName, inputs: NFKMLXQwenImageGenerator.backendInputKeys,
                    parameters: NFKMLXQwenImageGenerator.backendParameterKeys, outputs: [NFKOutputImage])
    }

    func testTheStableDiffusionReleasesDeclareTheirBackendsKeys() {
        for model in NFKMLXCatalogImage.stableDiffusionReleases {
            let backend = NFKMLXBackend(model: model)
            assertPorts(NFKMLXCatalogImage.stableDiffusionName(model), inputs: backend.supportedInputKeys,
                        parameters: backend.supportedParameterKeys, outputs: backend.supportedOutputKeys)
        }
    }

    func testTheConceptSegmentersDeclareTheirBackendsKeys() {
        for name in [NFKMLXSAM3.modelName, NFKMLXCatalogImage.sam31Name] {
            assertPorts(name, inputs: NFKMLXSAM3.backendInputKeys, parameters: NFKMLXSAM3.backendParameterKeys,
                        outputs: [NFKOutputDetections, NFKOutputMask])
        }
    }

    func testTheVideoModelsDeclareTheirBackendsKeys() {
        assertPorts(NFKMLXLTXVideoGenerator.modelName, inputs: NFKMLXLTXVideoGenerator.backendInputKeys,
                    parameters: NFKMLXLTXVideoGenerator.backendParameterKeys, outputs: [NFKOutputVideo])
        for name in [NFKMLXWanVideoGenerator.modelName] + NFKMLXCatalogVideo.wanAdditionalReleases.map(\.name) {
            assertPorts(name, inputs: NFKMLXWanVideoGenerator.backendInputKeys,
                        parameters: NFKMLXWanVideoGenerator.backendParameterKeys, outputs: [NFKOutputVideo])
        }
        let sam2VideoNames = [NFKMLXSAM2.videoModelName]
            + NFKMLXCatalogImage.sam2AdditionalReleases.map(NFKMLXCatalogVideo.sam2VideoName)
        for name in sam2VideoNames {
            assertPorts(name, inputs: NFKMLXSAM2.videoInputKeys, parameters: NFKMLXSAM2.videoParameterKeys,
                        outputs: [NFKOutputVideo, NFKOutputStructured])
        }
    }

    func testEveryReleasedSizeHasItsOwnName() {
        let names = Set(entries.map(\.identifier))
        let expected = ["sam-vit-b", "sam-vit-l", "sam-vit-h", "swinir-classical-x4", "nafnet-reds",
                        "yolov8x", "sam2.1-hiera-large", "sam2-hiera-tiny-video", "flux.1-dev", "sd3.5-large",
                        "florence-2-base", "trocr-small-printed", "vjepa2-vitg-fpc32-384-diving48",
                        "stable-diffusion-v1-5", "sdxl-turbo"]
        for name in expected {
            XCTAssertTrue(names.contains(name), name)
        }
    }
}
