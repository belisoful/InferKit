//
//  NFKMLXBarTracker.swift
//  InferKitMLX
//

import Foundation
import InferKit

// Turning a beat probability per frame into beat times is a decoding problem, not a threshold. This
// is madmom's `DBNDownBeatTrackingProcessor`: a bar-pointer hidden Markov model whose state is a
// position in the bar together with the tempo it is being played at, decoded by Viterbi over the
// whole track. It reads the two probabilities All-In-One scores per frame (a beat, and that beat
// being a downbeat) and returns beats with their position in the bar.
//
// The state space is one BEAT state space repeated per beat of the bar: for each candidate tempo,
// `interval` states walk from the beat to the next one, so a state's position inside its beat is
// `state / interval`. Transitions are deterministic inside a beat (each state follows the previous
// one) and choose a tempo at the beat boundary, under an exponential penalty on the tempo change.
//
// Two bar lengths are decoded (three and four beats) and the one with the higher path probability
// wins, which is how the meter is inferred rather than assumed.

/// The tempo grid and the bar lengths the tracker decodes.
public struct NFKMLXBarTrackerConfiguration: Sendable {
    /// The bar lengths to try, in beats.
    public var beatsPerBar: [Int]
    public var minimumBPM: Double
    public var maximumBPM: Double
    /// The tempi the state space samples between the two bounds.
    public var tempoCount: Int
    /// How sharply a tempo change at a beat boundary is penalized.
    public var transitionLambda: Double
    /// The reciprocal of the fraction of a beat that counts as "at the beat": at 16, the first
    /// sixteenth of a beat observes the beat probability and the rest observes its complement.
    public var observationLambda: Int
    /// The frames per second the probabilities are scored at.
    public var framesPerSecond: Int

    public init(beatsPerBar: [Int] = [3, 4], minimumBPM: Double = 55, maximumBPM: Double = 215,
                tempoCount: Int = 60, transitionLambda: Double = 100, observationLambda: Int = 16,
                framesPerSecond: Int = 100) {
        self.beatsPerBar = beatsPerBar
        self.minimumBPM = minimumBPM
        self.maximumBPM = maximumBPM
        self.tempoCount = tempoCount
        self.transitionLambda = transitionLambda
        self.observationLambda = observationLambda
        self.framesPerSecond = framesPerSecond
    }

    public static let `default` = NFKMLXBarTrackerConfiguration()
}

/// The states of one bar: every position in the bar, at every tempo the grid carries.
struct NFKBarStateSpace {
    /// The beat intervals, in frames, the grid samples.
    let intervals: [Int]
    /// Each state's position in the bar, from 0 to `beats`.
    let positions: [Double]
    /// Each state's own beat interval.
    let stateIntervals: [Int]
    /// The first state of each (beat, interval) pair, beat-major.
    let firstStates: [[Int]]
    /// The last state of each (beat, interval) pair.
    let lastStates: [[Int]]
    let beats: Int
    var stateCount: Int { positions.count }

    init(beats: Int, minimumInterval: Double, maximumInterval: Double, tempoCount: Int) {
        self.beats = beats
        intervals = Self.intervals(minimum: minimumInterval, maximum: maximumInterval, count: tempoCount)

        var positions = [Double]()
        var stateIntervals = [Int]()
        var firstStates = [[Int]]()
        var lastStates = [[Int]]()
        var offset = 0
        for beat in 0 ..< beats {
            var first = [Int]()
            var last = [Int]()
            for interval in intervals {
                first.append(offset)
                for step in 0 ..< interval {
                    positions.append(Double(beat) + Double(step) / Double(interval))
                    stateIntervals.append(interval)
                }
                offset += interval
                last.append(offset - 1)
            }
            firstStates.append(first)
            lastStates.append(last)
        }
        self.positions = positions
        self.stateIntervals = stateIntervals
        self.firstStates = firstStates
        self.lastStates = lastStates
    }

    /// The beat intervals the grid samples: whole frame counts, spaced geometrically between the
    /// bounds. The reference widens the sampling until enough distinct whole intervals fall out of
    /// it, which is why this is a loop rather than one `logspace`.
    static func intervals(minimum: Double, maximum: Double, count: Int) -> [Int] {
        let low = minimum.rounded()
        let high = maximum.rounded()
        let whole = Array(Int(low) ... Int(high))
        if count >= whole.count { return whole }

        var samples = count
        var intervals = [Int]()
        while intervals.count < count {
            var seen = Set<Int>()
            intervals = []
            for index in 0 ..< samples {
                let fraction = samples > 1 ? Double(index) / Double(samples - 1) : 0
                let exponent = log2(minimum) + fraction * (log2(maximum) - log2(minimum))
                let value = Int(pow(2.0, exponent).rounded())
                if seen.insert(value).inserted { intervals.append(value) }
            }
            intervals.sort()
            samples += 1
        }
        return intervals
    }
}

/// The transitions into each state, as the Viterbi decode reads them.
struct NFKBarTransitions {
    /// For each state, the index into `sources` where its incoming transitions start.
    let offsets: [Int]
    /// The previous state of each incoming transition.
    let sources: [Int]
    /// The log probability of each incoming transition.
    let logProbabilities: [Double]
    /// Whether a state is the first of its beat, which is where the tempo is chosen. Every other
    /// state has exactly one predecessor, the state before it, so the decode skips the edge list for
    /// it — which is most of the state space.
    let choosesTempo: [Bool]

    init(space: NFKBarStateSpace, transitionLambda: Double) {
        var incoming = [[(Int, Double)]](repeating: [], count: space.stateCount)
        // Inside a beat the pointer advances one state per frame, with no choice to make.
        var firstStates = Set<Int>()
        for beat in space.firstStates { firstStates.formUnion(beat) }
        for state in 0 ..< space.stateCount where !firstStates.contains(state) {
            incoming[state].append((state - 1, 0))
        }
        // At a beat boundary the pointer enters the next beat at some tempo, and changing tempo costs
        // `exp(-lambda · |ratio − 1|)`, normalized over the tempi reachable from each source.
        for beat in 0 ..< space.beats {
            let to = space.firstStates[beat]
            let from = space.lastStates[(beat + space.beats - 1) % space.beats]
            for source in from {
                let fromInterval = Double(space.stateIntervals[source])
                var probabilities = [Double](repeating: 0, count: to.count)
                var total = 0.0
                for (index, target) in to.enumerated() {
                    let ratio = Double(space.stateIntervals[target]) / fromInterval
                    let probability = exp(-transitionLambda * abs(ratio - 1))
                    // The reference drops anything below the smallest representable step.
                    probabilities[index] = probability < .ulpOfOne ? 0 : probability
                    total += probabilities[index]
                }
                guard total > 0 else { continue }
                for (index, target) in to.enumerated() where probabilities[index] > 0 {
                    incoming[target].append((source, log(probabilities[index] / total)))
                }
            }
        }

        var offsets = [Int](repeating: 0, count: space.stateCount + 1)
        var sources = [Int]()
        var logProbabilities = [Double]()
        var choosesTempo = [Bool](repeating: false, count: space.stateCount)
        for state in 0 ..< space.stateCount {
            offsets[state] = sources.count
            choosesTempo[state] = firstStates.contains(state)
            for (source, logProbability) in incoming[state] {
                sources.append(source)
                logProbabilities.append(logProbability)
            }
        }
        offsets[space.stateCount] = sources.count
        self.offsets = offsets
        self.sources = sources
        self.logProbabilities = logProbabilities
        self.choosesTempo = choosesTempo
    }
}

/// The beat tracker: All-In-One's beat and downbeat probabilities in, beats out.
public final class NFKMLXBarTracker {

    public let configuration: NFKMLXBarTrackerConfiguration

    public init(_ configuration: NFKMLXBarTrackerConfiguration = .default) {
        self.configuration = configuration
    }

    /// Decodes beats from the two probabilities scored per frame.
    ///
    /// The three columns the observation model reads are the ones the reference builds: a beat that
    /// is not a downbeat, a downbeat, and neither, normalized to sum to one.
    ///
    /// `threshold` trims the track to the span that reaches it before decoding, which is what the
    /// reference does with the value its checkpoint tuned. Trimming moves where the beats stop: a
    /// quiet tail below the threshold is not decoded at all, so the last beat of a fade-out is the
    /// last one above it rather than the last the model scored.
    public func track(beat: [Float], downbeat: [Float], threshold: Double? = nil) -> [NFKMusicBeat] {
        let count = min(beat.count, downbeat.count)
        guard count > 1 else { return [] }

        var all = [Double](repeating: 0, count: count * 2)
        for frame in 0 ..< count {
            let beatValue = Double(beat[frame])
            let downbeatValue = Double(downbeat[frame])
            let offBeat = max(1e-8, beatValue - downbeatValue)
            let neither = ((1 - beatValue) + (1 - downbeatValue)) / 2
            let total = offBeat + downbeatValue + neither
            all[frame * 2] = offBeat / total
            all[frame * 2 + 1] = downbeatValue / total
        }

        var first = 0
        var last = count
        if let threshold, threshold > 0 {
            let reaching = (0 ..< count).filter { all[$0 * 2] >= threshold || all[$0 * 2 + 1] >= threshold }
            if let low = reaching.first, let high = reaching.last {
                first = low
                last = high + 1
            } else {
                return []
            }
        }
        let frames = last - first
        guard frames > 1 else { return [] }
        let observations = Array(all[(first * 2) ..< (last * 2)])

        let minimumInterval = 60.0 * Double(configuration.framesPerSecond) / configuration.maximumBPM
        let maximumInterval = 60.0 * Double(configuration.framesPerSecond) / configuration.minimumBPM

        var best: (probability: Double, path: [Int], space: NFKBarStateSpace)?
        for beats in configuration.beatsPerBar {
            let space = NFKBarStateSpace(beats: beats, minimumInterval: minimumInterval,
                                         maximumInterval: maximumInterval, tempoCount: configuration.tempoCount)
            let transitions = NFKBarTransitions(space: space, transitionLambda: configuration.transitionLambda)
            let (path, probability) = viterbi(space: space, transitions: transitions,
                                              observations: observations, frames: frames)
            if best == nil || probability > best!.probability {
                best = (probability, path, space)
            }
        }
        guard let best, !best.path.isEmpty else { return [] }
        return beats(from: best.path, space: best.space, observations: observations, frames: frames,
                     offset: first)
    }

    /// Which of the three observation columns a state reads: 2 for the downbeat, 1 for another beat,
    /// 0 for the rest of the bar.
    private func pointer(position: Double) -> Int {
        let border = 1.0 / Double(configuration.observationLambda)
        if position < border { return 2 }
        return position.truncatingRemainder(dividingBy: 1) < border ? 1 : 0
    }

    /// The most probable path through the state space, in log space.
    private func viterbi(space: NFKBarStateSpace, transitions: NFKBarTransitions,
                         observations: [Double], frames: Int) -> (path: [Int], probability: Double) {
        let stateCount = space.stateCount
        let pointers = space.positions.map { pointer(position: $0) }
        let lambda = Double(configuration.observationLambda)

        var current = [Double](repeating: log(1.0 / Double(stateCount)), count: stateCount)
        var previous = [Double](repeating: 0, count: stateCount)
        var backtrack = [Int32](repeating: 0, count: frames * stateCount)

        let boundary = transitions.choosesTempo
        var densities = [Double](repeating: 0, count: 3)
        for frame in 0 ..< frames {
            let offBeat = observations[frame * 2]
            let downbeat = observations[frame * 2 + 1]
            densities[0] = log(max(1e-30, (1 - offBeat - downbeat) / (lambda - 1)))
            densities[1] = log(max(1e-30, offBeat))
            densities[2] = log(max(1e-30, downbeat))

            swap(&previous, &current)
            transitions.offsets.withUnsafeBufferPointer { offsets in
                transitions.sources.withUnsafeBufferPointer { sources in
                    transitions.logProbabilities.withUnsafeBufferPointer { weights in
                        previous.withUnsafeBufferPointer { source in
                            current.withUnsafeMutableBufferPointer { destination in
                                backtrack.withUnsafeMutableBufferPointer { trail in
                                    for state in 0 ..< stateCount {
                                        var bestValue: Double
                                        var bestSource: Int32
                                        if boundary[state] {
                                            bestValue = -Double.infinity
                                            bestSource = 0
                                            for edge in offsets[state] ..< offsets[state + 1] {
                                                let value = source[sources[edge]] + weights[edge]
                                                if value > bestValue {
                                                    bestValue = value
                                                    bestSource = Int32(sources[edge])
                                                }
                                            }
                                        } else {
                                            // Inside a beat the pointer only advances, at no cost.
                                            bestSource = Int32(state - 1)
                                            bestValue = source[state - 1]
                                        }
                                        destination[state] = bestValue + densities[pointers[state]]
                                        trail[frame * stateCount + state] = bestSource
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        var state = 0
        var probability = -Double.infinity
        for candidate in 0 ..< stateCount where current[candidate] > probability {
            probability = current[candidate]
            state = candidate
        }
        var path = [Int](repeating: 0, count: frames)
        for frame in stride(from: frames - 1, through: 0, by: -1) {
            path[frame] = state
            state = Int(backtrack[frame * stateCount + state])
        }
        return (path, probability)
    }

    /// The beats the path passes through: one per run of frames inside a beat's own window, placed at
    /// the frame whose probability is highest.
    private func beats(from path: [Int], space: NFKBarStateSpace, observations: [Double],
                       frames: Int, offset: Int) -> [NFKMusicBeat] {
        let inBeat = path.map { pointer(position: space.positions[$0]) >= 1 }
        var runs = [(Int, Int)]()
        var start: Int?
        for frame in 0 ..< frames {
            if inBeat[frame] {
                if start == nil { start = frame }
            } else if let open = start {
                runs.append((open, frame))
                start = nil
            }
        }
        if let open = start { runs.append((open, frames)) }

        var beats = [NFKMusicBeat]()
        for (left, right) in runs {
            var peak = left
            var best = -Double.infinity
            for frame in left ..< right {
                // The reference reads the flattened two-column activations, so a beat and a downbeat
                // frame compete on the same scale.
                let value = max(observations[frame * 2], observations[frame * 2 + 1])
                if value > best {
                    best = value
                    peak = frame
                }
            }
            let position = Int(space.positions[path[peak]]) + 1
            beats.append(NFKMusicBeat(timeSeconds: Double(peak + offset) / Double(configuration.framesPerSecond),
                                      positionInBar: position))
        }
        return beats
    }
}
