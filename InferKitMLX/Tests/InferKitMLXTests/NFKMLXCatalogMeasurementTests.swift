//
//  NFKMLXCatalogMeasurementTests.swift
//  InferKitMLXTests
//
//  How a catalog entry reads its measurements: the precision conversion, the stage split, the residency
//  budget's least working set, and the fields a measured variant carries. Nothing here reaches MLX.
//

import XCTest
import InferKit
@testable import InferKitMLX

final class NFKMLXCatalogMeasurementTests: XCTestCase {

    private func measurement(tensorBytes: [String: [String: Int64]],
                             opaqueBytes: [String: Int64] = [:]) -> NFKMLXMeasuredVariant {
        NFKMLXMeasuredVariant(model: "m", variant: "v", status: .measured, note: nil, commit: "abc", downloadBytes: 100,
                              parameterCount: 10, tensorBytes: tensorBytes, opaqueBytes: opaqueBytes)
    }

    func testAPrecisionConvertsFloatingTensorsAndKeepsPackedOnes() {
        let stored: [String: Int64] = ["BF16": 200, "F32": 400, "U32": 64]
        XCTAssertEqual(NFKMLXLoadPlan.bytes(stored, at: .stored), 664)
        XCTAssertEqual(NFKMLXLoadPlan.bytes(stored, at: .float32), 400 + 400 + 64)
        XCTAssertEqual(NFKMLXLoadPlan.bytes(stored, at: .bfloat16), 200 + 200 + 64)
    }

    func testTheLeastWorkingSetUsesTheResidencyBudgetsReserveAndShare() {
        let bytes: Int64 = 10 << 30
        let expected = (Double(bytes) + Double(NFKMLXResidencyBudget.reserve)) / NFKMLXResidencyBudget.workingSetFraction
        XCTAssertEqual(NFKMLXResidencyBudget.leastWorkingSet(holding: bytes), Int64(expected.rounded(.up)))
    }

    func testAStagedModelNeedsItsLargestStage() {
        let plan = NFKMLXLoadPlan(precision: .stored, directories: ["text_encoder": .float32],
                                  stages: [["text_encoder"], ["transformer", "vae"]])
        let figures = NFKMLXVariantFigures(measurement(tensorBytes: ["text_encoder": ["BF16": 1000],
                                                                     "transformer": ["BF16": 1500],
                                                                     "vae": ["F32": 300]]),
                                           plan: plan, converted: false)
        XCTAssertEqual(figures.residentBytes, 2000 + 1500 + 300)
        XCTAssertEqual(figures.minimumMemoryBytes, NFKMLXResidencyBudget.leastWorkingSet(holding: 2000))
        XCTAssertEqual(figures.precision, "bfloat16")
        XCTAssertFalse(figures.estimatesStoredBytes)
    }

    func testAnUnknownLoaderOrAConvertedSourceEstimatesTheStoredBytes() {
        let tensors: [String: [String: Int64]] = ["": ["F16": 500]]
        let unknown = NFKMLXVariantFigures(measurement(tensorBytes: tensors), plan: .unknown, converted: false)
        XCTAssertEqual(unknown.residentBytes, 500)
        XCTAssertTrue(unknown.estimatesStoredBytes)

        let converted = NFKMLXVariantFigures(measurement(tensorBytes: tensors), plan: .float32, converted: true)
        XCTAssertEqual(converted.residentBytes, 500)
        XCTAssertTrue(converted.estimatesStoredBytes)

        let pickled = NFKMLXVariantFigures(measurement(tensorBytes: [:], opaqueBytes: ["": 700]), plan: .float32,
                                           converted: false)
        XCTAssertEqual(pickled.residentBytes, 700)
        XCTAssertTrue(pickled.estimatesStoredBytes)

        let stored = NFKMLXVariantFigures(measurement(tensorBytes: [:], opaqueBytes: ["": 700]), plan: .stored,
                                          converted: false)
        XCTAssertFalse(stored.estimatesStoredBytes)
        XCTAssertEqual(stored.minimumMemoryBytes, NFKMLXResidencyBudget.leastWorkingSet(holding: 700))
    }

    func testAnUnmeasuredVariantCarriesNoFigures() {
        let unmeasured = NFKMLXMeasuredVariant(model: "m", variant: "v", status: .gated, note: "refused", commit: nil,
                                           downloadBytes: nil, parameterCount: nil, tensorBytes: [:], opaqueBytes: [:])
        let figures = NFKMLXVariantFigures(unmeasured, plan: .stored, converted: false)
        XCTAssertNil(figures.downloadBytes)
        XCTAssertNil(figures.residentBytes)
        XCTAssertNil(figures.minimumMemoryBytes)
    }

    func testEveryMeasuredFileIsCopiedIntoItsVariant() throws {
        for row in NFKMLXCatalogMeasurements.measuredVariants.prefix(50) where row.status == .measured {
            let descriptor = try XCTUnwrap(NFKMLXModelDescriptors.modelDescriptors().first { $0.identifier == row.model },
                                           row.model)
            let variant = try XCTUnwrap(descriptor.variant(withIdentifier: row.variant), "\(row.model)/\(row.variant)")
            XCTAssertEqual(variant.downloadByteCount?.int64Value, row.downloadBytes, "\(row.model)/\(row.variant)")
            XCTAssertEqual(variant.parameterCount?.int64Value, row.parameterCount, "\(row.model)/\(row.variant)")
            XCTAssertEqual(variant.source?.extensions["measuredRevision"] as? String ?? variant.source?.revision,
                           row.commit, "\(row.model)/\(row.variant)")
        }
    }

    func testTheStagedGeneratorsSplitTheirStagesAsTheirLoadersDo() throws {
        let staged = [NFKMLXFlux.modelName, "flux.1-dev", NFKMLXFluxControlNetGenerator.modelName, NFKMLXFlux2.modelName,
                      NFKMLXSD3Generator.modelName, NFKMLXSD3ControlNetGenerator.modelName, NFKMLXSANAGenerator.modelName,
                      NFKMLXZImageGenerator.modelName, NFKMLXQwenImageGenerator.modelName,
                      NFKMLXLTXVideoGenerator.modelName, NFKMLXWanVideoGenerator.modelName, NFKMLXMusic3.modelName]
        let entries = NFKMLXModelDescriptors.entries
        for name in staged {
            let entry = try XCTUnwrap(entries.first { $0.identifier == name }, name)
            XCTAssertFalse(NFKMLXLoadPlan.plan(for: entry).stages.isEmpty, name)
            XCTAssertTrue(entry.residencies.contains(NFKModelResidencyStaged), name)
        }
        let controlNet = try XCTUnwrap(entries.first { $0.identifier == NFKMLXSD3ControlNetGenerator.modelName })
        XCTAssertEqual(NFKMLXLoadPlan.plan(for: controlNet).stages.last, ["transformer", "vae", ""])
    }
}
