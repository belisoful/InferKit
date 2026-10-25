//
//  NFKMLXReferenceGELUTests.swift
//  InferKitMLXTests
//
//  The package's exact GELU: a float32 input takes the trainable kernel, torch 2.14's bit for bit, and the
//  sources are held to `NFKReferenceRounding.gelu`. MLX's GPU error function reads high for arguments from 0.93
//  to 2, so MLXNN's `gelu` lands apart from torch's in value and gradient.
//

import CryptoKit
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

    // torch 2.14's `gelu` and its `backward` under a cotangent of ones, as SHA-256 of the float32 words, on
    // x = (i − 4096) / 1024 for i in 0 ..< 8192 and on the same grid with every sixth value replaced by
    // 15 + (i mod 7) / 4, which puts an argument beyond 87.34 into two of every three 4-lane groups. torch's
    // vectorized loop computes every element of a length that fills its vectors; the remainder of any other
    // length takes `std::erf`.
    func testFloat32GELUIsTorchsBitForBit() throws {
        try requireMLXRuntime()
        func grid(_ i: Int) -> Float { Float(i - 4096) / 1024 }
        func mixed(_ i: Int) -> Float { i % 6 == 0 ? 15 + Float(i % 7) / 4 : grid(i) }
        let cases: [(String, (Int) -> Float, String, String)] = [
            ("grid", grid, "80855992122507a860205ea305591815a5406e623dd06c267f3f77968729ec9a",
             "37177e2fd7cf823cb9e8a44761feb4545d76d658f98b61c657783626702664dd"),
            ("mixed", mixed, "2491a4d0f4f39655a82f698bb70e515214b40e95dc1f2cce488074461013fe1d",
             "e2ec44a89e59c63bf52fec07206efc2452025d40494dafd944cc5fe107f5515b"),
        ]
        func digest(_ y: MLXArray) -> String {
            SHA256.hash(data: y.asArray(Float.self).withUnsafeBufferPointer { Data(buffer: $0) })
                .map { String(format: "%02x", $0) }.joined()
        }
        for (name, value, forward, backward) in cases {
            let x = MLXArray((0 ..< 8192).map(value))
            let (y, dx) = vjp({ [NFKReferenceRounding.gelu($0[0])] }, primals: [x], cotangents: [MLXArray.ones([8192])])
            XCTAssertEqual(digest(y[0]), forward, "\(name): torch's gelu")
            XCTAssertEqual(digest(dx[0]), backward, "\(name): torch's GeluBackward")
        }
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
