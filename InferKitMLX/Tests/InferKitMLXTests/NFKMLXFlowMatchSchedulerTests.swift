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

    // FLUX.2 against diffusers at the released config, fed the pipeline's `linspace(1, 1/steps,
    // steps)` ramp. FLUX.2 replaces `calculate_shift` with `compute_empirical_mu`, which depends on
    // the STEP COUNT as well as the sequence length, so the cases vary both. 4300 and 6000 sit either
    // side of the crossover above which the 200-step line is used alone.
    func testTheFlux2ScheduleMatchesTheReference() {
        let reference: [(steps: Int, sequence: Int, mu: Float, sigmas: [Float])] = [
            (4, 256, 1.965567112,
             [1.000000, 0.955391, 0.877134, 0.704112, 0.0]),
            (20, 1024, 1.916347623,
             [1.000000, 0.992315, 0.983914, 0.974691, 0.964519, 0.953245, 0.940679, 0.926586,
              0.910668, 0.892546, 0.871731, 0.847571, 0.819192, 0.785382, 0.744416, 0.693755,
              0.629496, 0.545312, 0.430239, 0.263454, 0.0]),
            (28, 4096, 2.151443243,
             [1.000000, 0.995710, 0.991132, 0.986234, 0.980983, 0.975338, 0.969253, 0.962675,
              0.955542, 0.947780, 0.939302, 0.930005, 0.919763, 0.908424, 0.895804, 0.881670,
              0.865735, 0.847629, 0.826877, 0.802854, 0.774719, 0.741318, 0.701020, 0.651443,
              0.588964, 0.507794, 0.398072, 0.241515, 0.0]),
            (28, 6000, 1.472286701,
             [1.000000, 0.991575, 0.982660, 0.973210, 0.963175, 0.952499, 0.941120, 0.928965,
              0.915952, 0.901987, 0.886962, 0.870750, 0.853206, 0.834158, 0.813405, 0.790706,
              0.765775, 0.738265, 0.707754, 0.673723, 0.635526, 0.592347, 0.543143, 0.486561,
              0.420804, 0.343446, 0.251117, 0.139008, 0.0]),
        ]
        for (steps, sequence, mu, referenceSigmas) in reference {
            var scheduler = NFKMLXFlowMatchScheduler(.flux2)
            XCTAssertEqual(scheduler.shift(forSequenceLength: sequence, steps: steps), mu, accuracy: 1e-6,
                           "the empirical shift at \(steps) steps over \(sequence) tokens")
            scheduler.setTimesteps(steps, sequenceLength: sequence)
            XCTAssertEqual(scheduler.sigmas.count, referenceSigmas.count,
                           "\(steps) steps produce \(steps) sigmas and a zero")
            for (index, expected) in referenceSigmas.enumerated() {
                XCTAssertEqual(scheduler.sigmas[index], expected, accuracy: 1e-5,
                               "sigma \(index) at \(steps) steps over \(sequence) tokens")
            }
        }
    }

    // The crossover is a real branch, not a rounding detail: above 4300 tokens the fit drops the
    // step-count interpolation, so the same step count either side of it gives a different shift.
    func testTheFlux2ShiftChangesBranchAboveTheCrossover() {
        // Every figure here is the reference's own `compute_empirical_mu`, read off it rather than
        // derived by hand.
        let scheduler = NFKMLXFlowMatchScheduler(.flux2)
        XCTAssertEqual(scheduler.shift(forSequenceLength: 4300, steps: 28), 2.170851490, accuracy: 1e-5)
        XCTAssertEqual(scheduler.shift(forSequenceLength: 6000, steps: 28), 1.472286660, accuracy: 1e-5)
        // Below the crossover the step count moves the shift, and by a lot.
        XCTAssertEqual(scheduler.shift(forSequenceLength: 4300, steps: 10), 2.274071425, accuracy: 1e-5)
        XCTAssertEqual(scheduler.shift(forSequenceLength: 4300, steps: 200), 1.184527660, accuracy: 1e-5)
        // Above it the 200-step line is used alone, so the step count drops out entirely.
        XCTAssertEqual(scheduler.shift(forSequenceLength: 6000, steps: 10),
                       scheduler.shift(forSequenceLength: 6000, steps: 200), accuracy: 1e-9)
    }

    // FLUX.1 [dev] against diffusers' FlowMatchEulerDiscreteScheduler at the release's config
    // (dynamic shift 0.5…1.15 over 256…4096 tokens) fed the `linspace(1, 1/steps, steps)` ramp
    // `pipeline_flux.py` passes, 20 steps at two packed sequence lengths.
    func testTheFluxScheduleMatchesTheReference() {
        let reference: [Int: [Float]] = [
            1024: [1.0, 0.972733, 0.944129, 0.914088, 0.882497, 0.849235, 0.814164, 0.777133, 0.737974,
                   0.696497, 0.652489, 0.605713, 0.555899, 0.502740, 0.445888, 0.384945, 0.319451,
                   0.248879, 0.172612, 0.089934, 0.0],
            4096: [1.0, 0.983608, 0.966014, 0.947080, 0.926647, 0.904531, 0.880513, 0.854338, 0.825702,
                   0.794239, 0.759511, 0.720980, 0.677987, 0.629707, 0.575103, 0.512844, 0.441200,
                   0.357875, 0.259758, 0.142529, 0.0],
        ]
        for (sequence, referenceSigmas) in reference {
            var scheduler = NFKMLXFlowMatchScheduler(.flux)
            scheduler.setTimesteps(20, sequenceLength: sequence)
            XCTAssertEqual(scheduler.sigmas.count, referenceSigmas.count)
            for (mine, theirs) in zip(scheduler.sigmas, referenceSigmas) {
                XCTAssertEqual(mine, theirs, accuracy: 1e-5, "sequence \(sequence)")
            }
            for (mine, theirs) in zip(scheduler.timesteps, referenceSigmas) {
                XCTAssertEqual(mine, theirs * 1000, accuracy: 1e-2, "sequence \(sequence)")
            }
        }
    }

    // FLUX.1 [schnell]'s static shift of 1.0 leaves the pipeline's ramp as the schedule: four steps
    // run at sigmas 1, 0.75, 0.5, 0.25 (diffusers, same config and ramp).
    func testTheFluxSchnellScheduleIsThePipelineRamp() {
        var scheduler = NFKMLXFlowMatchScheduler(.fluxSchnell)
        scheduler.setTimesteps(4, sequenceLength: 4096)
        let referenceSigmas: [Float] = [1.0, 0.75, 0.5, 0.25, 0.0]
        XCTAssertEqual(scheduler.sigmas.count, referenceSigmas.count)
        for (mine, theirs) in zip(scheduler.sigmas, referenceSigmas) {
            XCTAssertEqual(mine, theirs, accuracy: 1e-6)
        }
        XCTAssertEqual(scheduler.timesteps, [1000, 750, 500, 250])
    }

    // Stable Diffusion 3 against the same scheduler at its release config (static shift 3.0) with NO
    // pipeline ramp: `pipeline_stable_diffusion_3.py` passes `sigmas=None`, so the scheduler's own
    // ramp to `1 / num_train_timesteps` applies and the last step lands near 0.009.
    func testTheSD3ScheduleMatchesTheReference() {
        var scheduler = NFKMLXFlowMatchScheduler(.sd3)
        scheduler.setTimesteps(20, sequenceLength: 4096)
        let referenceSigmas: [Float] = [1.0, 0.981875, 0.962386, 0.941373, 0.918652, 0.894003, 0.867172,
                                        0.837855, 0.805689, 0.770239, 0.730974, 0.687244, 0.638240,
                                        0.582948, 0.520074, 0.447944, 0.364352, 0.266330, 0.149787,
                                        0.008929, 0.0]
        XCTAssertEqual(scheduler.sigmas.count, referenceSigmas.count)
        for (mine, theirs) in zip(scheduler.sigmas, referenceSigmas) {
            XCTAssertEqual(mine, theirs, accuracy: 1e-5)
        }
        XCTAssertEqual(scheduler.timesteps[19], 8.929, accuracy: 1e-2)
    }

    func testTheTerminalSigmaIsTheConfiguredValue() {
        var scheduler = NFKMLXFlowMatchScheduler(.ltxVideo)
        scheduler.setTimesteps(30, sequenceLength: 4096)
        // The last non-zero sigma lands on shift_terminal.
        XCTAssertEqual(scheduler.sigmas[scheduler.sigmas.count - 2], 0.1, accuracy: 1e-5)
        XCTAssertEqual(scheduler.sigmas.last, 0)
    }
}
