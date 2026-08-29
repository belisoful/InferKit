//
//  NFKMLXModelFitTests.swift
//  InferKitMLXTests
//
//  Sizing a model before loading it. The load-bearing test is the first one: the parameter count is
//  computed from the geometry so a model too large to instantiate can still be sized, which means the
//  arithmetic has no natural check — except against a model small enough to build, which is what this
//  does. Everything else rests on that.
//

import XCTest
import InferKit
import MLX
import MLXNN
@testable import InferKitMLX

final class NFKMLXModelFitTests: XCTestCase {

    private func requireMLXRuntime() throws {
        try XCTSkipIf(Bundle(for: type(of: self)).bundlePath.contains("/.build/"),
                      "builds MLX modules; run via xcodebuild")
    }

    override func tearDown() {
        NFKMLXGPU.clearCache()
        super.tearDown()
    }

    // MARK: The arithmetic, against a real module

    /// The count is derived from the configuration; this checks it against what the module actually
    /// declares. A geometry the count gets wrong would otherwise show up as a fit verdict that is
    /// quietly optimistic, which is the one kind of wrong answer this whole file exists to prevent.
    func testTheParameterCountMatchesABuiltModule() throws {
        try requireMLXRuntime()
        var geometries: [NFKMLXLanguageConfiguration] = [.tiny]

        // Each flag changes which parameters exist, so each is counted separately.
        var untied = NFKMLXLanguageConfiguration.tiny
        untied.tiesWordEmbeddings = false
        geometries.append(untied)

        var withoutQueryKeyNorm = NFKMLXLanguageConfiguration.tiny
        withoutQueryKeyNorm.normalizesQueryAndKey = false
        geometries.append(withoutQueryKeyNorm)

        var biased = NFKMLXLanguageConfiguration.tiny
        biased.attentionBias = true
        geometries.append(biased)

        // A grouped-query geometry, where the key and value widths differ from the query width.
        var grouped = NFKMLXLanguageConfiguration.tiny
        grouped.headCount = 8
        grouped.keyValueHeadCount = 2
        geometries.append(grouped)

        for geometry in geometries {
            let net = NFKMLXLanguage.makeNet(geometry)
            let built = net.parameters().flattened().reduce(0) { $0 + $1.1.size }
            let counted = NFKMLXModelSizing.parameterCount(of: geometry)
            XCTAssertEqual(counted, built,
                           "tied=\(geometry.tiesWordEmbeddings) qkNorm=\(geometry.normalizesQueryAndKey) "
                           + "bias=\(geometry.attentionBias) heads=\(geometry.headCount)/\(geometry.keyValueHeadCount)")
        }
    }

    /// The released geometries, so a change to the count is caught at the sizes that matter rather
    /// than only at the tiny one. Qwen3-0.6B is about 750M parameters once its tied embedding —
    /// 151936 × 1024, a fifth of the model on its own — is counted.
    func testTheReleasedGeometriesCountToTheirPublishedSizes() {
        let sizes: [(NFKMLXLanguageConfiguration, ClosedRange<Int>)] = [
            (.qwen3_0_6B, 550_000_000 ... 800_000_000),
            (.qwen3_1_7B, 1_500_000_000 ... 2_200_000_000),
            (.qwen3_4B, 3_500_000_000 ... 4_500_000_000),
            (.qwen3_8B, 7_000_000_000 ... 9_000_000_000),
        ]
        for (geometry, expected) in sizes {
            let counted = NFKMLXModelSizing.parameterCount(of: geometry)
            XCTAssertTrue(expected.contains(counted),
                          "counted \(counted), expected within \(expected)")
        }
    }

    /// Grouped-query attention is what makes a cache affordable, and counting it at the query width
    /// would overstate it fourfold on a released geometry.
    func testTheCacheIsCountedAtTheKeyValueWidth() {
        let configuration = NFKMLXLanguageConfiguration.qwen3_0_6B
        let perToken = NFKMLXModelSizing.keyValueBytesPerToken(of: configuration, precision: .float32)
        let expected = 2 * configuration.layerCount * configuration.keyValueHeadCount
            * configuration.headDimensions * 4
        XCTAssertEqual(perToken, expected)

        // The query head count is twice the key-value head count here, so the wrong reading is
        // exactly double — large enough to matter and small enough to overlook.
        XCTAssertEqual(configuration.headCount, 2 * configuration.keyValueHeadCount)
    }

    // `.checkpoint` holds a release at the type it shipped in, which for every decoder here is
    // 16-bit, so both costs halve against `.float32`.
    func testCheckpointPrecisionHalvesBothCosts() {
        let configuration = NFKMLXLanguageConfiguration.qwen3_4B
        let wide = NFKMLXModelSizing.fit(of: configuration, tokens: 1024,
                                         precision: .float32, budget: 1 << 62)
        let narrow = NFKMLXModelSizing.fit(of: configuration, tokens: 1024,
                                           precision: .checkpoint, budget: 1 << 62)
        XCTAssertEqual(wide.weightBytes, 2 * narrow.weightBytes)
        XCTAssertEqual(wide.keyValueBytesPerToken, 2 * narrow.keyValueBytesPerToken)
    }

    // MARK: The verdict

    func testAModelThatFitsSaysSo() {
        let fit = NFKMLXModelSizing.fit(of: .qwen3_0_6B, tokens: 2048, precision: .checkpoint,
                                        budget: 8 << 30)
        XCTAssertEqual(fit.verdict, NFKMLXModelFit.Verdict.fits)
        XCTAssertTrue(fit.fits)
        XCTAssertNil(fit.recommendedContextWindow, "nothing needs bounding")
    }

    /// The case the whole thing is for: the weights fit, the requested context does not, and the
    /// answer is the window that would.
    func testAModelThatNeedsAWindowReportsTheWindow() {
        let configuration = NFKMLXLanguageConfiguration.qwen3_4B
        let weights = NFKMLXModelSizing.parameterCount(of: configuration) * 2
        let perToken = NFKMLXModelSizing.keyValueBytesPerToken(of: configuration, precision: .checkpoint)
        // Room for the weights and exactly 1000 positions, against a request for far more.
        let budget = weights + perToken * 1000

        let fit = NFKMLXModelSizing.fit(of: configuration, tokens: 131_072,
                                        precision: .checkpoint, budget: budget)
        XCTAssertTrue(fit.fits)
        XCTAssertEqual(fit.recommendedContextWindow, 1000)
        XCTAssertEqual(fit.verdict, NFKMLXModelFit.Verdict.fitsWithinWindow(1000))
    }

    func testAModelTooLargeForTheBudgetReportsTheShortfall() {
        let configuration = NFKMLXLanguageConfiguration.qwen3_8B
        let weights = NFKMLXModelSizing.parameterCount(of: configuration) * 4
        let budget = weights / 2

        let fit = NFKMLXModelSizing.fit(of: configuration, tokens: 1024,
                                        precision: .float32, budget: budget)
        XCTAssertFalse(fit.fits)
        XCTAssertNil(fit.recommendedContextWindow, "there is no window that helps")
        guard case .tooLarge(let shortfall) = fit.verdict else {
            return XCTFail("expected tooLarge, got \(fit.verdict)")
        }
        XCTAssertEqual(shortfall, weights - budget)
    }

    /// A model whose weights only just fit leaves room for no cache at all. Zero is a real answer
    /// here, not a failure to compute one.
    func testAModelWithNoRoomLeftReportsAWindowOfZero() {
        let configuration = NFKMLXLanguageConfiguration.qwen3_0_6B
        let weights = NFKMLXModelSizing.parameterCount(of: configuration) * 2
        let fit = NFKMLXModelSizing.fit(of: configuration, tokens: 4096, precision: .checkpoint,
                                        budget: weights + 1)
        XCTAssertEqual(fit.verdict, NFKMLXModelFit.Verdict.fitsWithinWindow(0))
    }

    func testTheSummaryNamesTheVerdict() {
        let fit = NFKMLXModelSizing.fit(of: .qwen3_0_6B, tokens: 1024, precision: .checkpoint,
                                        budget: 8 << 30)
        XCTAssertTrue(fit.describedFit.contains("fits"), fit.describedFit)
        XCTAssertTrue(fit.describedFit.contains("GB"), fit.describedFit)
    }

    // MARK: The budget

    /// The budget must reflect what is free rather than what is installed, and must leave a margin.
    func testTheBudgetStaysInsideWhatTheMachineHas() {
        let profile = NFKHardwareProfile.current
        let budget = NFKMLXModelSizing.availableBudget()

        XCTAssertGreaterThan(budget, 0)
        XCTAssertLessThan(budget, profile.physicalMemory,
                          "a budget equal to the machine leaves nothing for the machine")
        if profile.recommendedWorkingSetSize > 0 {
            XCTAssertLessThanOrEqual(budget, profile.recommendedWorkingSetSize)
        }
        // A larger margin holds back more.
        XCTAssertLessThan(NFKMLXModelSizing.availableBudget(margin: 0.5),
                          NFKMLXModelSizing.availableBudget(margin: 0.1))
    }

    // MARK: The hardware profile

    func testTheProfileReadsThisMachine() {
        let profile = NFKHardwareProfile.current
        XCTAssertGreaterThan(profile.physicalMemory, 0, "the machine reports its memory")
        XCTAssertFalse(profile.chipName.isEmpty, "and its chip")
        XCTAssertGreaterThan(NFKHardwareProfile.availableMemory(), 0)
        XCTAssertFalse(profile.describedMachine.isEmpty)

        // Metal's recommendation is a real budget, below the physical total.
        if profile.recommendedWorkingSetSize > 0 {
            XCTAssertLessThan(profile.recommendedWorkingSetSize, profile.physicalMemory)
        }
        // The largest single buffer is a SEPARATE ceiling: a tensor cannot exceed it however much of
        // the budget is unspent.
        XCTAssertGreaterThan(profile.maximumBufferLength, 0)
    }

    /// Reading it twice returns the same object, since the static facts are read once.
    func testTheProfileIsCached() {
        XCTAssertTrue(NFKHardwareProfile.current === NFKHardwareProfile.current)
    }

    // MARK: The decode ceiling

    func testTheMeasuredBandwidthIsPlausibleAndCached() throws {
        try requireMLXRuntime()
        NFKMLXModelSizing.resetMeasuredBandwidth()
        // A small probe on purpose: this checks the plumbing, not the machine's real rate, and the
        // size that measures the rate honestly costs a gigabyte.
        let measured = NFKMLXModelSizing.measuredMemoryBandwidth(megabytes: 64)
        XCTAssertGreaterThan(measured, 1_000_000_000, "above 1 GB/s on any machine this runs on")
        XCTAssertLessThan(measured, 100e12, "and below anything physically possible")

        // The second call is the cache, so it returns the same figure without measuring again.
        XCTAssertEqual(NFKMLXModelSizing.measuredMemoryBandwidth(megabytes: 4096), measured,
                       "the cached value is returned whatever size is asked for")
        NFKMLXModelSizing.resetMeasuredBandwidth()
    }

    /// The ceiling falls as the model grows and as the context grows, because both add bytes that a
    /// decode step has to read.
    func testTheCeilingFallsWithModelSizeAndContextLength() {
        let bandwidth = 400e9

        let small = NFKMLXModelSizing.decodeCeiling(for: .qwen3_0_6B, contextLength: 1024,
                                                    precision: .checkpoint, bandwidth: bandwidth)
        let large = NFKMLXModelSizing.decodeCeiling(for: .qwen3_8B, contextLength: 1024,
                                                    precision: .checkpoint, bandwidth: bandwidth)
        XCTAssertGreaterThan(small, large, "a bigger model reads more per token")

        let shortContext = NFKMLXModelSizing.decodeCeiling(for: .qwen3_0_6B, contextLength: 128,
                                                           precision: .checkpoint, bandwidth: bandwidth)
        let longContext = NFKMLXModelSizing.decodeCeiling(for: .qwen3_0_6B, contextLength: 100_000,
                                                          precision: .checkpoint, bandwidth: bandwidth)
        XCTAssertGreaterThan(shortContext, longContext, "a longer cache is more to read")
    }

    /// Inverting the ceiling is what makes it a diagnostic rather than a number.
    func testTheAchievedFractionInvertsTheCeiling() {
        let bandwidth = 400e9
        let ceiling = NFKMLXModelSizing.decodeCeiling(for: .qwen3_0_6B, contextLength: 1024,
                                                      precision: .checkpoint, bandwidth: bandwidth)
        let half = NFKMLXModelSizing.achievedFraction(tokensPerSecond: ceiling / 2,
                                                      for: .qwen3_0_6B, contextLength: 1024,
                                                      precision: .checkpoint, bandwidth: bandwidth)
        XCTAssertEqual(half, 0.5, accuracy: 1e-6)

        // Above 1 means the model is not reading every parameter, which is what a sparse model doing
        // its job looks like — and a dense model cannot.
        let impossible = NFKMLXModelSizing.achievedFraction(tokensPerSecond: ceiling * 3,
                                                            for: .qwen3_0_6B, contextLength: 1024,
                                                            precision: .checkpoint, bandwidth: bandwidth)
        XCTAssertGreaterThan(impossible, 1)
    }

    func testAnUnknownBandwidthReportsNoCeilingRatherThanGuessing() {
        XCTAssertEqual(NFKMLXModelSizing.decodeCeiling(for: .qwen3_0_6B, bandwidth: 0), 0)
        XCTAssertEqual(NFKMLXModelSizing.achievedFraction(tokensPerSecond: 10, for: .qwen3_0_6B,
                                                          bandwidth: 0), 0)
    }

    // MARK: What this machine says

    /// The report, on whatever machine the suite is running on. It asserts the ordering — a larger
    /// model can never fit more comfortably than a smaller one — and prints the table, which is the
    /// thing a developer actually wants before choosing a size to ship.
    func testTheMachineReportsWhatEachReleasedSizeWouldDo() throws {
        try requireMLXRuntime()
        let sizes: [(String, NFKMLXLanguageConfiguration)] = [
            ("Qwen3-0.6B", .qwen3_0_6B), ("Qwen3-1.7B", .qwen3_1_7B),
            ("Qwen3-4B", .qwen3_4B), ("Qwen3-8B", .qwen3_8B),
        ]
        let budget = NFKMLXModelSizing.availableBudget()
        // The cache is process-wide, so another test's small probe would otherwise be reported here.
        NFKMLXModelSizing.resetMeasuredBandwidth()
        let bandwidth = NFKMLXModelSizing.measuredMemoryBandwidth()

        print("\n\(NFKHardwareProfile.current.describedMachine)")
        print(String(format: "budget %.2f GB, measured bandwidth %.0f GB/s",
                     Double(budget) / 1_073_741_824, bandwidth / 1e9))
        var previousHeadroom = Int.max
        for (name, geometry) in sizes {
            for precision in [NFKMLXWeightPrecision.float32, .checkpoint] {
                let fit = NFKMLXModelSizing.fit(of: geometry, tokens: 32_768,
                                                precision: precision, budget: budget)
                let ceiling = NFKMLXModelSizing.decodeCeiling(for: geometry, contextLength: 4096,
                                                              precision: precision,
                                                              bandwidth: bandwidth)
                let label = precision == .float32 ? "float32" : "16-bit "
                print(String(format: "  %-11@ %@  %@  ceiling %.1f tok/s",
                             name as NSString, label, fit.describedFit, ceiling))
            }
            // Ordering: a larger model leaves no more room than a smaller one did.
            let headroom = budget - NFKMLXModelSizing.fit(of: geometry, tokens: 1,
                                                          precision: .float32,
                                                          budget: budget).weightBytes
            XCTAssertLessThanOrEqual(headroom, previousHeadroom, "\(name) leaves no more room")
            previousHeadroom = headroom
        }
    }

    // MARK: Sized options

    /// When the requested length fits, nothing is bounded — an unnecessary window would drop context
    /// the machine had room for.
    func testSizedOptionsLeaveTheWindowUnsetWhenNothingNeedsDropping() throws {
        let options = try NFKMLXModelSizing.options(for: .qwen3_0_6B, requesting: 2048,
                                                    precision: .checkpoint, budget: 8 << 30)
        XCTAssertNil(options.contextWindow)
    }

    /// When it does not fit, the window comes out of the arithmetic rather than out of a guess.
    func testSizedOptionsDeriveTheWindowFromWhatIsLeft() throws {
        let configuration = NFKMLXLanguageConfiguration.qwen3_4B
        let weights = NFKMLXModelSizing.parameterCount(of: configuration) * 2
        let perToken = NFKMLXModelSizing.keyValueBytesPerToken(of: configuration, precision: .checkpoint)
        let budget = weights + perToken * 512

        let options = try NFKMLXModelSizing.options(for: configuration, requesting: 100_000,
                                                    precision: .checkpoint, budget: budget)
        XCTAssertEqual(options.contextWindow, 512)
    }

    /// The caller's other settings survive being sized.
    func testSizedOptionsKeepTheCallersOtherSettings() throws {
        var base = NFKMLXGenerationOptions()
        base.maxTokens = 77
        base.temperature = 0.8
        base.stopTokens = [2]

        let options = try NFKMLXModelSizing.options(for: .qwen3_0_6B, requesting: 1024,
                                                    precision: .checkpoint, base: base,
                                                    budget: 8 << 30)
        XCTAssertEqual(options.maxTokens, 77)
        XCTAssertEqual(options.temperature, 0.8)
        XCTAssertEqual(options.stopTokens, [2])
    }

    /// A model whose weights do not fit throws here rather than at the load, where the failure is a
    /// process kill rather than an error.
    func testSizedOptionsThrowWhenNoWindowCanHelp() {
        XCTAssertThrowsError(try NFKMLXModelSizing.options(for: .qwen3_8B, requesting: 1024,
                                                           precision: .float32,
                                                           budget: 2 << 30)) { error in
            guard case NFKMLXError.unsupportedConfiguration(let message) = error else {
                return XCTFail("expected unsupportedConfiguration, got \(error)")
            }
            XCTAssertTrue(message.contains("does not fit"), message)
        }
    }
}
