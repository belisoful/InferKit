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

    func testEachSa2VAReleaseCarriesTheLicenseOfItsMostRestrictiveComponent() throws {
        let expected: [String: (name: String, restrictsCommercialUse: Bool)] = [
            "sa2va-1b": ("Apache License 2.0", false),
            "sa2va-4b": ("Qwen Research License", true),
            "sa2va-internvl3-2b": ("Apache License 2.0", false),
            "sa2va-qwen2.5-vl-3b": ("Qwen Research License", true),
            "sa2va-qwen3-vl-2b": ("Apache License 2.0", false),
            "sa2va-qwen3-vl-4b": ("Apache License 2.0", false),
            "sa2va-qwen3-vl-4b-sam3": ("SAM License", false),
            "sa2va-llava-1.5-7b": ("Llama 2 Community License", true),
        ]
        XCTAssertEqual(Set(NFKMLXCatalogImage.sa2VAEntries.map(\.identifier)), Set(expected.keys))
        for (identifier, license) in expected {
            let descriptor = try XCTUnwrap(entry(identifier)).descriptor()
            let stated = try XCTUnwrap(descriptor.license, identifier)
            XCTAssertEqual(stated.name, license.name, identifier)
            XCTAssertEqual(stated.restrictsCommercialUse, license.restrictsCommercialUse, identifier)
            XCTAssertFalse(stated.isGated, identifier)
            if license.name != "Apache License 2.0" {
                XCTAssertTrue(stated.note?.hasPrefix("The repository states Apache-2.0.") ?? false, identifier)
            }
        }
    }

    func testEveryRFDetrReleaseInTheCatalogIsApache() throws {
        let identifiers = (NFKMLXCatalogImage.rfDetrEntries + NFKMLXCatalogImage.rfDetrSegmentationEntries).map(\.identifier)
        XCTAssertEqual(identifiers.count, 12)
        for identifier in identifiers {
            let license = try XCTUnwrap(XCTUnwrap(entry(identifier)).descriptor().license, identifier)
            XCTAssertEqual(license.identifier, "Apache-2.0", identifier)
            XCTAssertFalse(license.restrictsCommercialUse, identifier)
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

    func testTheImageGeneratorsDeclareTheirBackendsKeys() throws {
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
        let large = try XCTUnwrap(entry("sd3.5-large")).descriptor()
        XCTAssertEqual(large.option(withKey: NFKParameterSteps)?.defaultValue as? Int, 28)
        XCTAssertEqual(large.option(withKey: NFKParameterGuidanceScale)?.defaultValue as? Double, 7)
        let turbo = try XCTUnwrap(entry(NFKMLXCatalogImage.sd35LargeTurboName)).descriptor()
        XCTAssertEqual(turbo.option(withKey: NFKParameterSteps)?.defaultValue as? Int, 4)
        XCTAssertEqual(turbo.option(withKey: NFKParameterGuidanceScale)?.defaultValue as? Double, 0)
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
        for adapter in NFKMLXCatalogImage.stableDiffusionAdapters {
            let backend = NFKMLXBackend(model: adapter.base)
            assertPorts(adapter.name, inputs: backend.supportedInputKeys,
                        parameters: backend.supportedParameterKeys, outputs: backend.supportedOutputKeys)
        }
    }

    func testEachStableDiffusionAdapterListsItsBaseReleaseAndItsFileAtItsTrainedSchedule() throws {
        let adapters = NFKMLXCatalogImage.stableDiffusionAdapters
        XCTAssertEqual(adapters.map(\.name), ["lcm-lora-sdv1-5", "sdxl-lightning-4step-lora", "hyper-sd15-4steps-lora",
                                              "hyper-sdxl-4steps-lora", "hyper-sd15-8steps-cfg-lora",
                                              "hyper-sdxl-8steps-cfg-lora", "hyper-sd15-1step-lora",
                                              "hyper-sdxl-1step-lora"])
        for adapter in adapters {
            let descriptor = try XCTUnwrap(entry(adapter.name)).descriptor()
            let base = try XCTUnwrap(entry(NFKMLXCatalogImage.stableDiffusionName(adapter.base))).descriptor()
            let source = try XCTUnwrap(descriptor.defaultVariant?.source)
            let baseSource = try XCTUnwrap(base.defaultVariant?.source)
            XCTAssertEqual(source.repository, baseSource.repository, adapter.name)
            XCTAssertEqual(source.files.filter { !$0.overridesSource }.map(\.path), baseSource.files.map(\.path),
                           adapter.name)
            let adapterFiles = source.files.filter(\.overridesSource)
            XCTAssertEqual(adapterFiles.map(\.path), [adapter.weightsPath], adapter.name)
            XCTAssertEqual(adapterFiles.first?.repository, adapter.repository, adapter.name)
            XCTAssertEqual(adapterFiles.first?.role, NFKModelFileRoleWeights, adapter.name)
            XCTAssertEqual(adapterFiles.first?.isOptional, false, adapter.name)
            XCTAssertEqual(descriptor.option(withKey: NFKParameterSteps)?.defaultValue as? Int,
                           adapter.configuration.steps, adapter.name)
            XCTAssertEqual(adapter.configuration.sampler, adapter.sampler, adapter.name)
        }
        let lcm = try XCTUnwrap(entry("lcm-lora-sdv1-5")).descriptor()
        XCTAssertEqual(lcm.option(withKey: NFKParameterSteps)?.defaultValue as? Int, 4)
        XCTAssertEqual(lcm.option(withKey: NFKParameterGuidanceScale)?.defaultValue as? Double, 1)
        let lightning = try XCTUnwrap(entry("sdxl-lightning-4step-lora")).descriptor()
        XCTAssertEqual(lightning.option(withKey: NFKParameterSteps)?.defaultValue as? Int, 4)
        XCTAssertEqual(lightning.option(withKey: NFKParameterGuidanceScale)?.defaultValue as? Double, 0)
        XCTAssertEqual(lightning.option(withKey: NFKParameterWidth)?.defaultValue as? Int, 1024)
        let hyperFour = try XCTUnwrap(entry("hyper-sdxl-4steps-lora")).descriptor()
        XCTAssertEqual(hyperFour.option(withKey: NFKParameterSteps)?.defaultValue as? Int, 4)
        XCTAssertEqual(hyperFour.option(withKey: NFKParameterGuidanceScale)?.defaultValue as? Double, 0)
        let hyperCFG = try XCTUnwrap(entry("hyper-sd15-8steps-cfg-lora")).descriptor()
        XCTAssertEqual(hyperCFG.option(withKey: NFKParameterSteps)?.defaultValue as? Int, 8)
        XCTAssertEqual(hyperCFG.option(withKey: NFKParameterGuidanceScale)?.defaultValue as? Double, 5)
        let unified = try XCTUnwrap(entry("hyper-sd15-1step-lora")).descriptor()
        XCTAssertEqual(unified.option(withKey: NFKParameterSteps)?.defaultValue as? Int, 1)
        XCTAssertEqual(unified.option(withKey: NFKParameterGuidanceScale)?.defaultValue as? Double, 0)
        XCTAssertEqual(NFKMLXCatalogImage.hyperSDAdapters.count, 6)
        XCTAssertTrue(NFKMLXCatalogImage.hyperSDAdapters.allSatisfy { $0.repository == "ByteDance/Hyper-SD" })
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

    func testEachControlNetNamesTheBaseReleaseItSteers() throws {
        let controlNets: [(name: String, repository: String, base: String, components: [NFKMLXReleaseComponent])] = [
            (NFKMLXFluxControlNetGenerator.modelName, NFKMLXFluxControlNetGenerator.controlNetReleaseRepo,
             NFKMLXFluxControlNetGenerator.releaseRepo, NFKMLXFlux.releaseComponents),
            (NFKMLXSD3ControlNetGenerator.modelName, NFKMLXSD3ControlNetGenerator.controlNetReleaseRepo,
             NFKMLXSD3ControlNetGenerator.releaseRepo, NFKMLXSD3Generator.releaseComponents),
        ]
        for controlNet in controlNets {
            let source = try XCTUnwrap(try XCTUnwrap(entry(controlNet.name)).descriptor().defaultVariant?.source)
            XCTAssertEqual(source.repository, controlNet.repository, controlNet.name)
            let base = source.files.filter { $0.repository == controlNet.base }
            XCTAssertEqual(base.map(\.path), NFKMLXCatalogImage.componentFiles(controlNet.components).map(\.path),
                           controlNet.name)
            XCTAssertTrue(source.files.filter { $0.repository == nil }.allSatisfy { !$0.path.contains("/") }, controlNet.name)
        }
    }

    func testSourcesTheLoaderCannotReadAsServedSayWhy() throws {
        let sam31 = try XCTUnwrap(try XCTUnwrap(entry(NFKMLXCatalogImage.sam31Name)).descriptor().defaultVariant?.source)
        XCTAssertTrue(sam31.requiresConversion)
        XCTAssertTrue(sam31.files.contains { $0.path == NFKMLXCatalogImage.sam31Checkpoint && $0.role == NFKModelFileRoleWeights })
        XCTAssertNotNil(sam31.extensions["note"] as? String)

        let withdrawn = try XCTUnwrap(try XCTUnwrap(entry("stable-diffusion-2-1-base")).descriptor().defaultVariant?.source)
        XCTAssertEqual(withdrawn.repository, "stabilityai/stable-diffusion-2-1-base")
        XCTAssertEqual(withdrawn.extensions["publiclyServed"] as? Bool, false)
        XCTAssertNotNil(withdrawn.extensions["note"] as? String)
    }

    func testEveryReleasedSizeHasItsOwnName() {
        let names = Set(entries.map(\.identifier))
        let expected = ["sam-vit-b", "sam-vit-l", "sam-vit-h", "swinir-classical-x4", "nafnet-reds",
                        "yolov8x", "sam2.1-hiera-large", "sam2-hiera-tiny-video", "flux.1-dev", "sd3.5-large", "sd3.5-large-turbo",
                        "florence-2-base", "trocr-small-printed", "vjepa2-vitg-fpc32-384-diving48",
                        "stable-diffusion-v1-5", "sdxl-turbo", "stable-diffusion-xl-base-1.0", "lcm-dreamshaper-v7",
                        "lcm-lora-sdv1-5", "sdxl-lightning-4step-lora", "hyper-sd15-4steps-lora",
                        "hyper-sdxl-4steps-lora", "hyper-sd15-8steps-cfg-lora", "hyper-sdxl-8steps-cfg-lora",
                        "hyper-sd15-1step-lora", "hyper-sdxl-1step-lora"]
        for name in expected {
            XCTAssertTrue(names.contains(name), name)
        }
    }
}
