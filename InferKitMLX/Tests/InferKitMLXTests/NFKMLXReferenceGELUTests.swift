//
//  NFKMLXReferenceGELUTests.swift
//  InferKitMLXTests
//
//  The package's exact GELU: a float32 input takes the trainable kernel, and the sources are held to
//  `NFKReferenceRounding.gelu`. MLX's GPU error function reads high for arguments from 0.93 to 2, so MLXNN's
//  `gelu` lands apart from torch's in value and gradient.
//

import Foundation
import MLX
import XCTest
@testable import InferKitMLX

final class NFKMLXReferenceGELUTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    // The value and the gradient under a cotangent of ones, on x = (i − 4096) / 1024 for i in 0 ..< 8192, equal
    // `torchTrainableGELU`'s bit for bit. NFKMLXBFloat16ParityTests holds that kernel to torch's distance from
    // float64.
    func testFloat32GELUTakesTheTrainableKernel() throws {
        try requireMLXRuntime()
        let x = MLXArray((0 ..< 8192).map { Float($0 - 4096) / 1024 })
        func valueAndGradient(_ gelu: @escaping (MLXArray) -> MLXArray) -> [[Float]] {
            let (y, dx) = vjp({ [gelu($0[0])] }, primals: [x], cotangents: [MLXArray.ones([8192])])
            return [y[0].asArray(Float.self), dx[0].asArray(Float.self)]
        }
        XCTAssertEqual(valueAndGradient { NFKReferenceRounding.gelu($0) },
                       valueAndGradient { NFKReferenceRounding.torchTrainableGELU($0) })
    }

    // A bare `gelu(` is MLXNN's, and a `GELU` module's call is too. Reads the source text, so it runs under
    // `swift test` without MLX.
    func testEveryErfGELUTakesTheReferenceFunction() throws {
        let directory = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/InferKitMLX")
        let names = try XCTUnwrap(try? FileManager.default.contentsOfDirectory(atPath: directory.path),
                                  "the package's sources are not beside its tests at \(directory.path)")
        let exempt: Set = ["NFKMLXReferenceRounding.swift", "NFKMLXReferenceKernels.swift"]
        let bare = try NSRegularExpression(pattern: #"(?<![\w.])(?<!func )gelu\(|MLXNN\.gelu\(|as! GELU\b"#)
        var sites = [String]()
        for name in names.filter({ $0.hasSuffix(".swift") && !exempt.contains($0) }).sorted() {
            let text = try String(contentsOf: directory.appendingPathComponent(name), encoding: .utf8)
            for (number, line) in text.split(separator: "\n", omittingEmptySubsequences: false).enumerated() {
                let code = String(line.split(separator: "//", maxSplits: 1, omittingEmptySubsequences: false)[0])
                if bare.firstMatch(in: code, range: NSRange(code.startIndex..., in: code)) != nil {
                    sites.append("\(name):\(number + 1)")
                }
            }
        }
        XCTAssertEqual(sites, [], "each takes NFKReferenceRounding.gelu")
    }
}
