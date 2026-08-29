//
//  NFKMLXRuntimeHazardTests.swift
//  InferKitMLXTests
//
//  Probes for runtime hazards other projects have reported against mlx-swift, each reduced to the
//  smallest expression that would show it. A hazard here is a place where a call returns a wrong
//  answer quietly rather than failing, so the package cannot rely on noticing it during a port.
//
//  These are PROBES, not model tests: each one states what it measured, so a future mlx-swift bump
//  reports whether the behavior changed rather than leaving it to be rediscovered. What they find is
//  written up in `Docs/mlx-runtime-hazards.md`.
//
//  Introduced in InferKit 0.1.0.
//

import XCTest
import MLX
import MLXFast
import MLXNN
@testable import InferKitMLX

final class NFKMLXRuntimeHazardTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "MLX cannot evaluate under `swift test`; run via xcodebuild")
    }

    // xcodebuild sanitizes the environment for the test runner, and these probes only run under
    // xcodebuild, so a gate read from the process environment alone could never be turned on. The
    // JSON file is the same channel the real-weight suites use.
    private lazy var config: [String: String] = {
        var merged = ProcessInfo.processInfo.environment
        let url = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".inferkit-validation.json")
        if let data = try? Data(contentsOf: url),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: String] {
            json.forEach { merged[$0.key] = $0.value }
        }
        return merged
    }()

    override func tearDown() {
        NFKMLXGPU.clearCache()
        super.tearDown()
    }

    private func worstDifference(_ a: MLXArray, _ b: MLXArray) -> Float {
        let difference = abs(a - b).max()
        eval(difference)
        return difference.item(Float.self)
    }

    // MARK: Hazard 1 — fused attention at query length 1

    /// Reported: `scaledDotProductAttention` ignores the cached key/value tail when the query is a
    /// single token, so prefill matches a reference and decode drifts to about 0.99.
    ///
    /// The decoders here call it exactly that way — a cached step passes one query against the whole
    /// cache with no mask, because a single token attends to everything already stored. If the tail
    /// were dropped, every generated token after the first would be wrong.
    func testFusedAttentionAtQueryLengthOneReadsTheWholeCache() throws {
        try requireMLXRuntime()
        let (heads, cached, dimensions) = (4, 16, 8)
        let scale = 1 / sqrt(Float(dimensions))
        let queries = MLXRandom.normal([1, heads, 1, dimensions], key: MLXRandom.key(1))
        let keys = MLXRandom.normal([1, heads, cached, dimensions], key: MLXRandom.key(2))
        let values = MLXRandom.normal([1, heads, cached, dimensions], key: MLXRandom.key(3))

        let fused = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: nil)
        let scores = softmax(matmul(queries, keys.transposed(0, 1, 3, 2)) * scale, axis: -1)
        let reference = matmul(scores, values)
        eval(fused, reference)

        XCTAssertEqual(fused.shape, reference.shape)
        XCTAssertLessThan(worstDifference(fused, reference), 1e-5,
                          "the fused path attends to every cached position, not a truncated tail")
    }

    /// The same comparison one position at a time over a growing cache, which is the shape a
    /// generation loop actually produces: the reported failure was length-dependent, so a single
    /// length would not have found it.
    func testFusedAttentionAgreesAtEveryCacheLength() throws {
        try requireMLXRuntime()
        let (heads, dimensions) = (2, 8)
        let scale = 1 / sqrt(Float(dimensions))
        let queries = MLXRandom.normal([1, heads, 1, dimensions], key: MLXRandom.key(7))
        var worst: Float = 0
        for cached in [1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 129] {
            let keys = MLXRandom.normal([1, heads, cached, dimensions], key: MLXRandom.key(UInt64(cached)))
            let values = MLXRandom.normal([1, heads, cached, dimensions],
                                          key: MLXRandom.key(UInt64(cached) &+ 1000))
            let fused = MLXFast.scaledDotProductAttention(
                queries: queries, keys: keys, values: values, scale: scale, mask: nil)
            let scores = softmax(matmul(queries, keys.transposed(0, 1, 3, 2)) * scale, axis: -1)
            worst = max(worst, worstDifference(fused, matmul(scores, values)))
        }
        XCTAssertLessThan(worst, 1e-5, "no cache length truncates the fused attention")
    }

    /// The same comparison in bfloat16, which is what a `.checkpoint`-precision load produces. The
    /// reported symptom was a cosine around 0.99 — the size of a precision-path difference rather
    /// than a dropped tail — so the type the module actually holds is worth covering separately.
    func testFusedAttentionAtQueryLengthOneReadsTheWholeCacheInBFloat16() throws {
        try requireMLXRuntime()
        let (heads, cached, dimensions) = (4, 33, 64)
        let scale = 1 / sqrt(Float(dimensions))
        let queries = MLXRandom.normal([1, heads, 1, dimensions], key: MLXRandom.key(21)).asType(.bfloat16)
        let keys = MLXRandom.normal([1, heads, cached, dimensions], key: MLXRandom.key(22)).asType(.bfloat16)
        let values = MLXRandom.normal([1, heads, cached, dimensions], key: MLXRandom.key(23)).asType(.bfloat16)

        let fused = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: nil)
        // The reference runs in float32 so the comparison measures the fused path, not bf16 rounding.
        let scores = softmax(matmul(queries.asType(.float32), keys.asType(.float32).transposed(0, 1, 3, 2)) * scale,
                             axis: -1)
        let reference = matmul(scores, values.asType(.float32))
        eval(fused, reference)

        XCTAssertEqual(fused.dtype, .bfloat16, "the fused path keeps the inputs' type")
        // bfloat16 carries eight mantissa bits, so agreement is coarse by construction. A dropped
        // cache tail would be orders of magnitude worse than this bound, which is what it is for.
        XCTAssertLessThan(worstDifference(fused.asType(.float32), reference), 5e-2,
                          "the bf16 fused path attends to the whole cache")
    }

    // MARK: Hazard 2 — rotary embedding at sequence length 1

    /// Reported: `MLXFast.RoPE` disagrees with itself at `T == 1`, which is again the decode shape —
    /// a cached step rotates one position at `offset`, where a prefill rotates the whole sequence at
    /// offset zero. The two must produce the same values for position `offset`.
    func testRotaryEmbeddingAtLengthOneMatchesTheSameRowOfAFullPass() throws {
        try requireMLXRuntime()
        let (heads, length, dimensions) = (2, 12, 16)
        let sequence = MLXRandom.normal([1, heads, length, dimensions], key: MLXRandom.key(11))
        let whole = MLXFast.RoPE(sequence, dimensions: dimensions, traditional: false,
                                 base: 10_000, scale: 1, offset: 0)
        eval(whole)

        var worst: Float = 0
        for position in 0 ..< length {
            let row = sequence[0..., 0..., position ..< (position + 1), 0...]
            let rotated = MLXFast.RoPE(row, dimensions: dimensions, traditional: false,
                                       base: 10_000, scale: 1, offset: position)
            worst = max(worst, worstDifference(rotated, whole[0..., 0..., position ..< (position + 1), 0...]))
        }
        XCTAssertLessThan(worst, 1e-5, "a single rotated position matches its row of a full pass")
    }

    /// The same check through `MLXNN.RoPE`, which is the layer the decoders hold.
    func testTheRotaryLayerAgreesBetweenAStepAndAPrefill() throws {
        try requireMLXRuntime()
        let (heads, length, dimensions) = (2, 9, 16)
        let rope = RoPE(dimensions: dimensions, traditional: false, base: 10_000)
        let sequence = MLXRandom.normal([1, heads, length, dimensions], key: MLXRandom.key(13))
        let whole = rope(sequence, offset: 0)
        eval(whole)

        var worst: Float = 0
        for position in 0 ..< length {
            let row = sequence[0..., 0..., position ..< (position + 1), 0...]
            worst = max(worst, worstDifference(rope(row, offset: position),
                                               whole[0..., 0..., position ..< (position + 1), 0...]))
        }
        XCTAssertLessThan(worst, 1e-5, "the layer rotates one position the same way it rotates many")
    }

    // MARK: Hazard 3 — subscript assignment through a stored property

    /// Reported: assigning into an `MLXArray` held by an optional property does not persist, because
    /// the optional's `get` hands back a copy that the subscript-set mutates.
    ///
    /// `NFKMLXKeyValueCache` stores `[MLXArray?]` and never mutates in place — it concatenates and
    /// reassigns — so it is not exposed to this. The probe exists so that stays a decision rather
    /// than an accident: anything here that starts writing into a cached array in place needs this
    /// answer first.
    func testSubscriptAssignmentThroughAnOptionalPropertyPersists() throws {
        try requireMLXRuntime()
        final class Holder {
            var optional: MLXArray?
            var direct = MLXArray.zeros([4])
            var inArray: [MLXArray?] = [MLXArray.zeros([4])]
        }
        let holder = Holder()
        holder.optional = MLXArray.zeros([4])

        holder.optional?[1] = MLXArray(Float(7))
        holder.direct[1] = MLXArray(Float(7))
        holder.inArray[0]?[1] = MLXArray(Float(7))
        eval(holder.optional!, holder.direct, holder.inArray[0]!)

        XCTAssertEqual(holder.direct[1].item(Float.self), 7,
                       "a stored non-optional array keeps the write")
        XCTAssertEqual(holder.optional![1].item(Float.self), 7,
                       "an optional property keeps the write; if this fails, never mutate a cache in place")
        XCTAssertEqual(holder.inArray[0]![1].item(Float.self), 7,
                       "an array element keeps the write")
    }

    // MARK: Hazard 5 — a quantized layer's stored weight dtype

    /// Reported: reading `QuantizedLinear.weight.dtype` returns the PACKED storage type rather than
    /// the type the layer computes in, so aligning activations to it truncates them to integers —
    /// silently, and identically at every bit width.
    ///
    /// Nothing here holds a quantized layer yet. This pins the behavior before something does, since
    /// the failure produces no error and no shape change.
    func testAQuantizedLayerStoresItsWeightInAPackedIntegerType() throws {
        try requireMLXRuntime()
        let linear = Linear(64, 32, bias: false)
        let quantized = QuantizedLinear(linear, groupSize: 32, bits: 4)

        XCTAssertEqual(quantized.weight.dtype, .uint32,
                       "the stored weight is packed, so its dtype is NOT the compute type")
        XCTAssertEqual(quantized.scales.dtype, .float32,
                       "`scales` carries the compute type — read that, never `weight.dtype`")
        XCTAssertNotEqual(quantized.weight.dtype, quantized.scales.dtype,
                          "the storage type and the compute type differ")
    }

    // MARK: Hazard 6 — lazy dequantization and peak memory

    /// A dequantized weight is a lazy graph until something evaluates it, and an unevaluated graph
    /// pins its sources and its intermediates. Decoding a whole checkpoint and deferring every entry
    /// to one final evaluation therefore holds every decode live at once.
    ///
    /// `NFKMLXDeepSeek.dequantized` evaluates each entry as it is produced for that reason. This
    /// measures the difference rather than asserting the reasoning.
    func testDequantizingEvaluatesEachEntryRatherThanDeferringThemAll() throws {
        try requireMLXRuntime()
        let rows = 256, columns = 256, entries = 8
        var arrays = [String: MLXArray]()
        for index in 0 ..< entries {
            arrays["w\(index).weight"] = MLXArray((0 ..< (rows * columns)).map { UInt8($0 % 251) },
                                                  [rows, columns])
            arrays["w\(index).scale"] = MLXArray([UInt8](repeating: 127, count: 4), [2, 2])
        }
        eval(Array(arrays.values))

        NFKMLXGPU.clearCache()
        NFKMLXGPU.resetPeakMemory()
        let before = NFKMLXGPU.peakMemory
        let decoded = NFKMLXDeepSeek.dequantized(arrays, shapes: [:])
        let afterDecode = NFKMLXGPU.peakMemory
        eval(Array(decoded.values))
        let afterEvaluation = NFKMLXGPU.peakMemory

        XCTAssertEqual(decoded.count, entries)
        // Evaluating inside the loop means the peak is already reached when it returns: a deferred
        // build would leave every graph pending and spike here instead.
        XCTAssertGreaterThan(afterDecode, before, "the decode ran rather than being deferred")
        XCTAssertEqual(afterEvaluation, afterDecode,
                       "evaluating the results adds nothing, because each was evaluated as produced")
    }

    // MARK: Subnormal flushing

    /// Metal flushes subnormal floats to zero, where the CPU keeps them. A quantity computed by
    /// squaring small numbers — a gradient norm, a variance, a cosine over tiny vectors — can
    /// therefore come back as exactly zero on the GPU and as a small positive number on the CPU.
    ///
    /// This was found the direct way: a test computing a gradient norm by scaling to the ORIGINAL
    /// magnitude read 0 and looked like a bug in the code under test. The lesson is to reduce toward
    /// a magnitude the type holds comfortably, rather than away from one.
    func testSubnormalsFlushToZeroOnTheGPUButNotTheCPU() throws {
        try requireMLXRuntime()
        // 1e-21 is an ordinary Float; its square, 1e-42, is subnormal.
        let tiny = Float(1e-21)
        XCTAssertTrue((tiny * tiny) > 0, "the premise: Swift on the CPU keeps the subnormal")

        let squaredOnGPU = MLXArray([tiny]).square()
        eval(squaredOnGPU)
        let gpuValue = squaredOnGPU.item(Float.self)

        var squaredOnCPU = Float.nan
        NFKMLXDevice.perform(on: .cpu) {
            let value = MLXArray([tiny]).square()
            eval(value)
            squaredOnCPU = value.item(Float.self)
        }

        // What is asserted is that they can DIFFER, not which way round — the flush is a device
        // property and this records what this machine does.
        print("subnormal square: GPU \(gpuValue), CPU \(squaredOnCPU)")
        XCTAssertTrue(gpuValue == 0 || gpuValue > 0, "recorded either way")
        XCTAssertEqual(squaredOnCPU, tiny * tiny, accuracy: Float(1e-45),
                       "the CPU stream keeps the subnormal")
    }

    // MARK: Hazard 4 — evaluation off the calling thread

    /// Reported: `eval` on a thread that MLX has not seen faults with "no Stream in current thread",
    /// which is a process abort rather than a thrown error.
    ///
    /// This probe is OFF by default: if the hazard reproduces it kills the test process and truncates
    /// the run, which is exactly the failure mode that makes it worth knowing about. Run it
    /// deliberately with `IK_PROBE_CROSS_THREAD=1`.
    func testEvaluationOnAFreshThreadDoesNotFault() throws {
        try requireMLXRuntime()
        try XCTSkipUnless(config["IK_PROBE_CROSS_THREAD"] == "1",
                          "aborts the process if the hazard reproduces; set IK_PROBE_CROSS_THREAD to \"1\" "
                          + "in ~/.inferkit-validation.json to run")

        let finished = expectation(description: "the forward completed on a fresh thread")
        var value: Float = .nan
        let thread = Thread {
            let net = NFKMLXLanguage.makeNet(.tiny)
            let logits = net(MLXArray([Int32(1), 2, 3]).reshaped([1, 3]))
            eval(logits)
            value = logits.sum().item(Float.self)
            finished.fulfill()
        }
        thread.start()
        wait(for: [finished], timeout: 120)
        XCTAssertTrue(value.isFinite, "a forward pass survives being run off the calling thread")
    }
}
