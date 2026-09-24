//
//  NFKMLXMuScriptorEvents.swift
//  InferKitMLX
//

import Foundation
import InferKit

// MuScriptor transcribes by generating a token sequence, so its output is a language before it is a
// performance. The vocabulary is MT3's: a token is a time shift, a pitch, a velocity, a program, a
// drum hit, or the `tie` marker that closes a chunk's opening section. This file is the vocabulary
// and the state machine that turns a stream of those tokens back into notes.
//
// The model reads five seconds at a time, and a note can be held across that boundary. Each chunk
// therefore opens with a "tie section": the notes still sounding, declared as program and pitch pairs
// and terminated by `tie`. Anything the previous chunk left open that the tie section does not
// declare ends at the boundary. A chunk that reaches a time shift without ever emitting `tie` is
// malformed, and the reference closes every open note and drops the rest of that chunk.

/// One entry of the MT3 vocabulary.
public struct NFKMuScriptorEvent: Equatable, Sendable {
    public enum Kind: String, Sendable {
        case pad, eos, unknown, shift, pitch, velocity, tie, program, drum
    }
    public var kind: Kind
    public var value: Int
}

/// The token to event table, which is fixed by one number: how many time shifts the vocabulary spans.
///
/// The layout is the reference's `build_event_vocab`: the three special tokens, the shifts, then
/// pitch, velocity, tie, program, and drum. At the released 1001 shifts that is 1393 tokens, which is
/// also where the model masks its own logits — the released heads are wider (1395 for medium and
/// large), and the tokens past the vocabulary are never sampled.
public struct NFKMuScriptorVocabulary: Sendable {
    public let events: [NFKMuScriptorEvent]
    /// The shifts the vocabulary spans. One shift step is one frame, 10 ms at the released frame rate.
    public let maxShiftSteps: Int
    private let indices: [String: Int]

    public init(maxShiftSteps: Int = 1001) {
        self.maxShiftSteps = maxShiftSteps
        var events: [NFKMuScriptorEvent] = [
            NFKMuScriptorEvent(kind: .pad, value: 0),
            NFKMuScriptorEvent(kind: .eos, value: 0),
            NFKMuScriptorEvent(kind: .unknown, value: 0),
        ]
        for value in 0 ..< maxShiftSteps { events.append(NFKMuScriptorEvent(kind: .shift, value: value)) }
        for value in 0 ... 127 { events.append(NFKMuScriptorEvent(kind: .pitch, value: value)) }
        for value in 0 ... 1 { events.append(NFKMuScriptorEvent(kind: .velocity, value: value)) }
        events.append(NFKMuScriptorEvent(kind: .tie, value: 0))
        for value in 0 ... 129 { events.append(NFKMuScriptorEvent(kind: .program, value: value)) }
        for value in 0 ... 127 { events.append(NFKMuScriptorEvent(kind: .drum, value: value)) }
        self.events = events

        var indices = [String: Int]()
        for (index, event) in events.enumerated() {
            indices["\(event.kind.rawValue):\(event.value)"] = index
        }
        self.indices = indices
    }

    /// The token count, which is also the first token the model masks.
    public var count: Int { events.count }
    /// The end-of-sequence token.
    public var endOfSequence: Int { 1 }

    public func event(at token: Int) -> NFKMuScriptorEvent? {
        token >= 0 && token < events.count ? events[token] : nil
    }

    public func token(kind: NFKMuScriptorEvent.Kind, value: Int) -> Int? {
        indices["\(kind.rawValue):\(value)"]
    }

    /// The tie section that opens a chunk: the sustained notes as program and pitch pairs, sorted,
    /// each program written once for its run of pitches, terminated by `tie`.
    ///
    /// Teacher-forcing this pins the chunk's opening to the notes actually held rather than letting
    /// the model guess them.
    public func tieSectionTokens(openNotes: [(program: Int, pitch: Int)]) -> [Int] {
        var tokens = [Int]()
        var program: Int?
        for note in openNotes.sorted(by: { ($0.program, $0.pitch) < ($1.program, $1.pitch) }) {
            if note.program != program, let token = token(kind: .program, value: note.program) {
                tokens.append(token)
                program = note.program
            }
            if let token = token(kind: .pitch, value: note.pitch) {
                tokens.append(token)
            }
        }
        if let token = token(kind: .tie, value: 0) {
            tokens.append(token)
        }
        return tokens
    }
}

/// Where a chunk starts, and where the next one does.
public struct NFKMuScriptorChunkBoundary: Sendable {
    public var seekSeconds: Double
    /// The next chunk's start, or nil for the last chunk. A note the model places past it belongs to
    /// the next chunk, so it is dropped here.
    public var nextSeekSeconds: Double?

    public init(seekSeconds: Double, nextSeekSeconds: Double?) {
        self.seekSeconds = seekSeconds
        self.nextSeekSeconds = nextSeekSeconds
    }
}

/// What the tracker says happened.
enum NFKMuScriptorAction {
    case start(program: Int, pitch: Int, time: Double)
    case end(program: Int, pitch: Int, time: Double)
    case drum(pitch: Int, time: Double)
}

/// The decode state machine: tokens and chunk boundaries in, note actions out.
///
/// The reference runs one of these for both jobs it has — turning the stream into notes, and reading
/// back which notes are open so the next chunk's tie section can be forced. Keeping one machine for
/// both is what keeps the two consistent.
final class NFKMuScriptorNoteTracker {
    private let vocabulary: NFKMuScriptorVocabulary
    private let frameRate: Int
    /// Open notes in the order they started, which is the order the end-of-stream closes replay in.
    private var open: [(key: NoteKey, onset: Double)] = []
    private var seekSeconds: Double = 0
    private var nextSeekSeconds: Double?
    private var startTick = 0
    private var tick = 0
    private var program: Int?
    private var velocity: Int?
    private var inPrologue = true
    private var skipRest = false
    private var tieSet = Set<NoteKey>()
    private var chunkStarted = false

    struct NoteKey: Hashable { var program: Int; var pitch: Int }

    init(vocabulary: NFKMuScriptorVocabulary, frameRate: Int = 100) {
        self.vocabulary = vocabulary
        self.frameRate = frameRate
    }

    /// The `(program, pitch)` pairs currently held open, sorted.
    var openNotes: [(program: Int, pitch: Int)] {
        open.map { ($0.key.program, $0.key.pitch) }
            .sorted { ($0.program, $0.pitch) < ($1.program, $1.pitch) }
    }

    func feed(boundary: NFKMuScriptorChunkBoundary) -> [NFKMuScriptorAction] {
        var actions = [NFKMuScriptorAction]()
        // A chunk that never closed its tie section is malformed, so nothing it left open survives.
        if chunkStarted && inPrologue {
            actions = endAll(at: seekSeconds)
        }
        seekSeconds = boundary.seekSeconds
        nextSeekSeconds = boundary.nextSeekSeconds
        startTick = Int((boundary.seekSeconds * Double(frameRate)).rounded())
        tick = startTick
        program = nil
        velocity = nil
        inPrologue = true
        skipRest = false
        tieSet = []
        chunkStarted = true
        return actions
    }

    func feed(token: Int) -> [NFKMuScriptorAction] {
        guard let event = vocabulary.event(at: token) else { return [] }

        if inPrologue {
            switch event.kind {
            case .tie:
                inPrologue = false
                velocity = nil
                let ended = open.filter { !tieSet.contains($0.key) }
                open.removeAll { !tieSet.contains($0.key) }
                return ended.map { .end(program: $0.key.program, pitch: $0.key.pitch, time: seekSeconds) }
            case .shift:
                // No tie token: the chunk is malformed.
                inPrologue = false
                skipRest = true
                return endAll(at: seekSeconds)
            case .program:
                program = event.value
            case .pitch:
                if let program {
                    tieSet.insert(NoteKey(program: program, pitch: event.value))
                }
            default:
                break
            }
            return []
        }

        if skipRest { return [] }

        switch event.kind {
        case .shift:
            if event.value > 0 { tick = startTick + event.value }
        case .program:
            program = event.value
        case .velocity:
            velocity = event.value
        case .drum:
            let time = Double(tick) / Double(frameRate)
            if nextSeekSeconds == nil || time < nextSeekSeconds! {
                return [.drum(pitch: event.value, time: time)]
            }
        case .pitch:
            guard let program, let velocity else { return [] }
            let time = Double(tick) / Double(frameRate)
            if let next = nextSeekSeconds, time >= next { return [] }
            let key = NoteKey(program: program, pitch: event.value)
            var actions = [NFKMuScriptorAction]()
            if let index = open.firstIndex(where: { $0.key == key }) {
                open.remove(at: index)
                actions.append(.end(program: key.program, pitch: key.pitch, time: time))
            }
            if velocity > 0 {
                open.append((key, time))
                actions.append(.start(program: key.program, pitch: key.pitch, time: time))
            }
            return actions
        default:
            break
        }
        return []
    }

    /// Closes whatever is still sounding at the end of the stream.
    func finish() -> [NFKMuScriptorAction] {
        if chunkStarted && inPrologue {
            return endAll(at: seekSeconds)
        }
        let actions = open.map {
            NFKMuScriptorAction.end(program: $0.key.program, pitch: $0.key.pitch,
                                    time: $0.onset + NFKMuScriptorNoteTracker.minimumNoteSeconds)
        }
        open.removeAll()
        return actions
    }

    private func endAll(at time: Double) -> [NFKMuScriptorAction] {
        let actions = open.map {
            NFKMuScriptorAction.end(program: $0.key.program, pitch: $0.key.pitch, time: time)
        }
        open.removeAll()
        return actions
    }

    /// A drum hit has no offset, and a note still open at the end of the stream gets this length.
    static let minimumNoteSeconds = 0.01
}

/// Assembles the tracker's actions into notes.
enum NFKMuScriptorNotes {
    /// The velocity the reference writes. The model scores a note on or off rather than a dynamic, so
    /// every note carries the same one.
    static let midiVelocity = 100
    /// The program number the vocabulary reserves for drums.
    static let drumProgram = 128

    /// A token stream, chunk by chunk, as MIDI notes.
    static func notes(chunks: [(boundary: NFKMuScriptorChunkBoundary, tokens: [Int])],
                      vocabulary: NFKMuScriptorVocabulary, frameRate: Int = 100) -> [NFKMIDINote] {
        let tracker = NFKMuScriptorNoteTracker(vocabulary: vocabulary, frameRate: frameRate)
        var notes = [NFKMIDINote]()
        var pending = [NFKMuScriptorNoteTracker.NoteKey: Double]()

        func apply(_ actions: [NFKMuScriptorAction]) {
            for action in actions {
                switch action {
                case let .start(program, pitch, time):
                    pending[.init(program: program, pitch: pitch)] = time
                case let .end(program, pitch, time):
                    let key = NFKMuScriptorNoteTracker.NoteKey(program: program, pitch: pitch)
                    guard let onset = pending.removeValue(forKey: key) else { continue }
                    notes.append(NFKMIDINote(pitch: pitch, startSeconds: onset, endSeconds: max(time, onset),
                                             velocity: midiVelocity, program: program,
                                             percussion: false, pitchBend: nil))
                case let .drum(pitch, time):
                    notes.append(NFKMIDINote(pitch: pitch, startSeconds: time,
                                             endSeconds: time + NFKMuScriptorNoteTracker.minimumNoteSeconds,
                                             velocity: midiVelocity, program: 0,
                                             percussion: true, pitchBend: nil))
                }
            }
        }

        for chunk in chunks {
            apply(tracker.feed(boundary: chunk.boundary))
            for token in chunk.tokens {
                if token == vocabulary.endOfSequence { break }
                apply(tracker.feed(token: token))
            }
        }
        apply(tracker.finish())
        return notes.sorted { ($0.startSeconds, $0.pitch) < ($1.startSeconds, $1.pitch) }
    }
}
