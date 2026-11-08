//
//  NFKMLXCatalogTiers.swift
//  InferKitMLX
//
//  What the residency planner does with each catalog variant at its default options on Macs of 8, 16, 24,
//  and 32 GB. Docs/model-memory-tiers.json holds the result, so a change to a figure or to the planner
//  shows as a change in that file.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import InferKit

extension NFKModelCatalog {

    /// The Metal working set assumed per Mac size: two thirds of the memory at 8, 16, and 24 GB, and at 32 GB
    /// the 26.8 GB an M1 Max reports.
    private static let tierWorkingSets: [(gigabytes: Int, bytes: Int)] = [
        (8, (8 << 30) * 2 / 3), (16, (16 << 30) * 2 / 3), (24, (24 << 30) * 2 / 3), (32, 26_800_000_000),
    ]

    /// The bytes of Docs/model-memory-tiers.json: each variant with memory facts, at each Mac size, as
    /// `residency peakBytes fits|refused` at its default options.
    ///
    /// Introduced in InferKit 0.4.0.
    @objc public func memoryTierSnapshotData() throws -> Data {
        var variants = [String: [String: String]]()
        for descriptor in descriptors {
            for variant in descriptor.variants where variant.memoryFacts != nil {
                var cells = [String: String]()
                for tier in Self.tierWorkingSets {
                    let estimate = try descriptor.estimatedMemory(
                        forOptions: descriptor.variants.count > 1 ? [NFKModelLoadOptionVariant: variant.identifier] : nil,
                        memoryBudget: NFKModelMemoryBudget(workingSetByteCount: tier.bytes))
                    cells[String(tier.gigabytes)] = "\(estimate.residency) \(estimate.peakByteCount) "
                        + (estimate.fits ? "fits" : "refused")
                }
                variants["\(descriptor.identifier)/\(variant.identifier)"] = cells
            }
        }
        let workingSets = Dictionary(uniqueKeysWithValues: Self.tierWorkingSets.map { (String($0.gigabytes), $0.bytes) })
        let document: [String: Any] = ["kind": "inferkit.catalog-memory-tiers", "workingSetBytes": workingSets,
                                       "variants": variants]
        return try JSONSerialization.data(withJSONObject: document,
                                          options: [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes])
    }
}
