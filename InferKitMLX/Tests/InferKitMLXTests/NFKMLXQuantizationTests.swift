//
//  NFKMLXQuantizationTests.swift
//  InferKitMLXTests
//
//  Runtime quantization of layers held in module arrays and dictionaries. A layer the filter skips
//  sits beside layers it packs, so the container has to come back whole with only the eligible
//  entries replaced.
//

import XCTest
import MLX
import MLXNN
@testable import InferKitMLX

final class NFKMLXQuantizationTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(NFKMLXGPU.metalLibraryURL == nil,
                      "no Metal library for MLX; run Tools/mlx-metallib.sh or xcodebuild")
    }

    override func setUp() {
        super.setUp()
        // Seeding initializes MLX's runtime, which needs a Metal library it can find; the methods
        // skip without one.
        guard NFKMLXGPU.metalLibraryURL != nil else { return }
        NFKMLXRandom.seed(20_260_930)
    }

    /// A module array whose first and last Linears have input widths of 48, which a group size of 32
    /// does not divide.
    private final class ArrayHeld: Module {
        @ModuleInfo(key: "net") var net: [Module]

        override init() {
            _net.wrappedValue = [Linear(48, 64), GELU(), Linear(64, 48), GELU(), Linear(48, 32), Dropout(p: 0)]
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            net.reduce(x) { ($1 as! UnaryLayer)($0) }
        }
    }

    private final class DictionaryHeld: Module {
        @ModuleInfo(key: "heads") var heads: [String: Linear]

        override init() {
            _heads.wrappedValue = ["a": Linear(64, 8), "b": Linear(48, 8)]
            super.init()
        }
    }

    private final class Block: Module {
        @ModuleInfo(key: "q") var q: Linear
        @ModuleInfo(key: "out") var out: Linear

        override init() {
            _q.wrappedValue = Linear(64, 64)
            _out.wrappedValue = Linear(48, 64)
        }
    }

    private final class Stack: Module {
        @ModuleInfo(key: "blocks") var blocks: [Block]

        override init() {
            _blocks.wrappedValue = [Block(), Block()]
            super.init()
        }
    }

    private final class Grid: Module {
        @ModuleInfo(key: "grid") var grid: [[Module]]

        override init() {
            _grid.wrappedValue = [[Linear(64, 32), GELU()]]
            super.init()
        }
    }

    func testQuantizingAnArrayReplacesOnlyItsEligibleEntries() throws {
        try requireMLXRuntime()
        let model = ArrayHeld()
        let input = MLXArray((0 ..< 96).map { Float($0 % 11) / 11.0 }).reshaped([2, 48])
        let before = model(input)
        try NFKMLXQuantization.quantize(module: model, bits: 8, groupSize: 32)
        XCTAssertEqual(model.net.count, 6, "the entries after the last quantized layer stay")
        XCTAssertTrue(type(of: model.net[0]) == Linear.self, "an ineligible first entry is kept as built")
        XCTAssertTrue(model.net[1] is GELU)
        XCTAssertTrue(model.net[2] is QuantizedLinear)
        XCTAssertTrue(model.net[3] is GELU)
        XCTAssertTrue(type(of: model.net[4]) == Linear.self)
        XCTAssertTrue(model.net[5] is Dropout)
        let after = model(input)
        XCTAssertLessThan(abs(after - before).max().item(Float.self), 0.05, "8-bit groups stay close")
    }

    func testQuantizingADictionaryKeepsItsIneligibleKeys() throws {
        try requireMLXRuntime()
        let model = DictionaryHeld()
        try NFKMLXQuantization.quantize(module: model, bits: 4, groupSize: 32)
        XCTAssertEqual(Set(model.heads.keys), ["a", "b"])
        XCTAssertTrue(model.heads["a"] is QuantizedLinear)
        XCTAssertTrue(type(of: model.heads["b"]!) == Linear.self)
    }

    func testQuantizingAStackReplacesEveryBlocksEligibleLayers() throws {
        try requireMLXRuntime()
        let model = Stack()
        try NFKMLXQuantization.quantize(module: model, bits: 4, groupSize: 32)
        XCTAssertEqual(model.blocks.count, 2)
        for block in model.blocks {
            XCTAssertTrue(block.q is QuantizedLinear)
            XCTAssertTrue(type(of: block.out) == Linear.self)
        }
    }

    func testALayerDirectlyInANestedArrayIsRefused() throws {
        try requireMLXRuntime()
        let model = Grid()
        XCTAssertThrowsError(try NFKMLXQuantization.quantize(module: model, bits: 4, groupSize: 32))
        XCTAssertEqual(model.grid.count, 1)
        XCTAssertEqual(model.grid[0].count, 2)
        XCTAssertTrue(type(of: model.grid[0][0]) == Linear.self)
    }

    func testPackedExpertsStayFrozenUnderTheirOwnUnfreeze() throws {
        try requireMLXRuntime()
        let packed = NFKLMSwitchLinear(experts: 2, inputSize: 64, outputSize: 32).quantized(groupSize: 32, bits: 4)
        XCTAssertTrue(packed.trainableParameters().flattened().isEmpty, "built frozen")
        packed.unfreeze()
        XCTAssertTrue(packed.trainableParameters().flattened().isEmpty, "packed experts have no gradient")
    }
}
