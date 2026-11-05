//
//  NFKMLXCatalogQualityTests.swift
//  InferKitMLXTests
//
//  The published benchmark scores in the package's catalog and the rank the core computes from them.
//  Building descriptors touches no MLX.
//

import XCTest
import InferKit
@testable import InferKitMLX

final class NFKMLXCatalogQualityTests: XCTestCase {

    private lazy var descriptors = NFKMLXModelDescriptors.modelDescriptors()

    private lazy var catalog: NFKModelCatalog = {
        let catalog = NFKModelCatalog()
        catalog.addDescriptors(descriptors)
        return catalog
    }()

    func testEveryRecordNamesAShippedEntryAndItsVariant() {
        let entries = Dictionary(uniqueKeysWithValues: NFKMLXModelDescriptors.entries.map { ($0.identifier, $0) })
        for record in NFKMLXCatalogQuality.records {
            guard let entry = entries[record.model] else {
                XCTFail("\(record.model) has a score but no catalog entry")
                continue
            }
            if let variant = record.variant {
                XCTAssertTrue(entry.variants.contains { $0.identifier == variant }, "\(record.model) has no variant \(variant)")
            }
            XCTAssertEqual(URL(string: record.source)?.scheme, "https", record.model)
            XCTAssertEqual(NFKModelCatalog.referenceBenchmark(forTask: record.task), record.benchmark, record.model)
        }
    }

    func testEveryScoredEntryCarriesItsScores() throws {
        for (model, records) in NFKMLXCatalogQuality.byModel {
            let descriptor = try XCTUnwrap(descriptors.first { $0.identifier == model }, model)
            XCTAssertEqual(descriptor.qualityScores.count, records.count, model)
            XCTAssertEqual(Set(descriptor.qualityScores.map(\.benchmarkIdentifier)),
                           Set(records.map { "\($0.benchmark)@\($0.snapshot)" }), model)
        }
    }

    func testAnEntryWithACodingScoreListsCodeGeneration() throws {
        let qwen = try XCTUnwrap(descriptors.first { $0.identifier == "qwen3-4b" })
        XCTAssertEqual(qwen.tasks.first, NFKModelTaskTextGeneration)
        XCTAssertTrue(qwen.tasks.contains(NFKModelTaskCodeGeneration))
        let mixtral = try XCTUnwrap(descriptors.first { $0.identifier == "mixtral-8x7b-instruct" })
        XCTAssertFalse(mixtral.tasks.contains(NFKModelTaskCodeGeneration))
    }

    func testEachRankedTaskRanksItsScoredModels() {
        let leaders: [String: String] = [
            NFKModelTaskTextGeneration: "gemma-4-31b-it",
            NFKModelTaskCodeGeneration: "qwen4-exp",
            NFKModelTaskTranscription: "parakeet-tdt",
            NFKModelTaskTranslation: "nllb-200-distilled-1.3b",
            NFKModelTaskTextToImage: "z-image",
            NFKModelTaskSpeechSynthesis: "kokoro-82m",
        ]
        for (task, leader) in leaders {
            let rankings = catalog.qualityRankings(forTask: task)
            XCTAssertEqual(rankings.first?.modelIdentifier, leader, task)
            XCTAssertEqual(rankings.first?.rank, 1, task)
        }
    }

    func testACodingRankComparesOneSnapshot() {
        let rankings = catalog.qualityRankings(forTask: NFKModelTaskCodeGeneration)
        XCTAssertEqual(Set(rankings.map(\.score.snapshot)), ["v6"])
        XCTAssertNil(catalog.qualityRanking(forModelIdentifier: "qwen3-4b", task: NFKModelTaskCodeGeneration),
                     "a v5 score does not rank against v6 scores")
    }

    func testATranscriptionRankPutsTheLowestErrorFirst() {
        let rankings = catalog.qualityRankings(forTask: NFKModelTaskTranscription)
        let values = rankings.map(\.score.value)
        XCTAssertEqual(values, values.sorted())
        XCTAssertEqual(rankings.count, 5)
    }
}
