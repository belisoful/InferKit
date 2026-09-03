//
//  NFKMLXFlowMatchSchedulerTests.swift
//  InferKitMLXTests
//
//  The rectified-flow schedule is pure Float math, so these run under `swift test`.
//

import XCTest
@testable import InferKitMLX

final class NFKMLXFlowMatchSchedulerTests: XCTestCase {

    // The sigma and timestep schedule against diffusers' FlowMatchEulerDiscreteScheduler (LTX config),
    // 20 steps at a latent sequence length of 2048.
    func testTheScheduleMatchesTheReference() {
        var scheduler = NFKMLXFlowMatchScheduler(.ltxVideo)
        scheduler.setTimesteps(20, sequenceLength: 2048)

        XCTAssertEqual(scheduler.shift(forSequenceLength: 2048), 1.316667, accuracy: 1e-4)

        let referenceSigmas: [Float] = [1.0, 0.98676, 0.97242, 0.95682, 0.93981, 0.92118, 0.90068, 0.87802,
                                        0.85285, 0.82471, 0.79304, 0.75715, 0.71613, 0.66879, 0.61354, 0.54824,
                                        0.46986, 0.37402, 0.25417, 0.1, 0.0]
        XCTAssertEqual(scheduler.sigmas.count, referenceSigmas.count, "steps + 1 sigmas ending in 0")
        for (mine, theirs) in zip(scheduler.sigmas, referenceSigmas) {
            XCTAssertEqual(mine, theirs, accuracy: 1e-4)
        }

        let referenceTimesteps: [Float] = [1000.0, 986.76, 972.415, 956.823, 939.811, 921.179, 900.681,
                                           878.024, 852.848, 824.706, 793.043, 757.153, 716.13, 668.788,
                                           613.544, 548.241, 469.856, 374.018, 254.171, 100.0]
        XCTAssertEqual(scheduler.timesteps.count, referenceTimesteps.count)
        for (mine, theirs) in zip(scheduler.timesteps, referenceTimesteps) {
            XCTAssertEqual(mine, theirs, accuracy: 5e-2)
        }
    }

    func testTheTerminalSigmaIsTheConfiguredValue() {
        var scheduler = NFKMLXFlowMatchScheduler(.ltxVideo)
        scheduler.setTimesteps(30, sequenceLength: 4096)
        // The last non-zero sigma lands on shift_terminal.
        XCTAssertEqual(scheduler.sigmas[scheduler.sigmas.count - 2], 0.1, accuracy: 1e-5)
        XCTAssertEqual(scheduler.sigmas.last, 0)
    }
}
