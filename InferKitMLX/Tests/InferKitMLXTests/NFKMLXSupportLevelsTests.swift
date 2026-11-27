//
//  NFKMLXSupportLevelsTests.swift
//  InferKitMLX tests
//

import XCTest

final class NFKMLXSupportLevelsTests: XCTestCase {
    private func document(_ path: String) throws -> String {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
        return try String(contentsOf: root.appendingPathComponent(path), encoding: .utf8)
    }

    func testEveryRetiredReleaseIsRecordedAtItsLevelInTheRemovalCandidates() throws {
        let text = try document("Docs/removal-candidates.md")
        for (release, level) in NFKMLXSupportLevels.retired {
            XCTAssertTrue(text.contains("`\(release)` | \(level)"),
                          "\(release) is retired at level \(level) but Docs/removal-candidates.md does not record it")
        }
    }

    func testEveryRetiredReleaseIsListedInTheModelIndexAndTheLedger() throws {
        for path in ["Docs/model-index.md", "Docs/agent-reference/mlx-customization-ledger.md"] {
            let text = try document(path)
            guard let start = text.range(of: "## Support levels") else {
                XCTFail("\(path) has no Support levels section")
                continue
            }
            let section = text[start.upperBound...].components(separatedBy: "\n## ").first ?? ""
            for release in NFKMLXSupportLevels.retired.keys {
                XCTAssertTrue(section.contains("`\(release)`"), "\(path) does not list the retired release \(release)")
            }
        }
    }

    func testOnlyLevelsTwoAndThreeRetireTests() {
        for (release, level) in NFKMLXSupportLevels.retired {
            XCTAssertTrue((2...3).contains(level), "\(release): level \(level) retires no released-weight test")
        }
    }
}
