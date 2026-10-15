//
//  NFKMLXLifecycle.swift
//  InferKitMLX
//
//  Unloading a backend's model, loading it again, and measuring what a load holds. The bring-your-own
//  backends keep their loaded state in an `NFKMLXLifecycle`; every other backend reaches the same path
//  through `NFKMLXReloadableBackend`. `NFKMLXReload.reloadable(_:)` is the one place the registry and the
//  download factories give a backend the closure that loads it again.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import InferKit
import MLX

/// What one load added to MLX's memory.
struct NFKMLXLoadMeasurement: Sendable {

    /// MLX's active memory after the load less its active memory before it.
    let residentBytes: Int

    /// The peak of MLX's active memory during the load less its active memory before it.
    let peakBytes: Int

    /// Runs `load` and measures it: the peak is reset, and active memory read, before the load and after
    /// it. Without MLX's Metal library a memory reading aborts the process, so `load` then runs unmeasured.
    static func measuring<T>(_ load: () throws -> T) rethrows -> (value: T, measurement: NFKMLXLoadMeasurement?) {
        guard NFKMLXGPU.metalLibraryURL != nil else {
            return (try load(), nil)
        }
        let before = NFKMLXGPU.activeMemory
        NFKMLXGPU.resetPeakMemory()
        let value = try load()
        let rise = NFKMLXGPU.activeMemory - before
        let peak = NFKMLXGPU.peakMemory - before
        return (value, NFKMLXLoadMeasurement(residentBytes: max(rise, 0), peakBytes: max(peak, rise, 0)))
    }

    /// The figures under the `NFKModelInfo*` keys.
    var modelInfo: [String: Any] {
        [NFKModelInfoResidentBytes: residentBytes, NFKModelInfoPeakBytes: peakBytes]
    }
}

/// The runs and loads in flight across every lifecycle in the process.
///
/// An unload returns MLX's buffer cache to the system only while this count is zero: a cache trim frees
/// buffers that in-flight work still reads (`Docs/agent-reference/mlx-runtime-gotchas.md`). The trim
/// runs under the count's lock, so no managed run or load starts during it.
enum NFKMLXActivity {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var inFlight = 0

    static func begin() {
        lock.withLock { inFlight += 1 }
    }

    static func end() {
        lock.withLock { inFlight -= 1 }
    }

    /// Synchronizes the GPU stream and clears MLX's cache when no managed run or load is in flight.
    static func releaseCachedMemoryWhenIdle() {
        lock.withLock {
            if inFlight == 0 {
                NFKMLXGPU.releaseCachedMemory()
            }
        }
    }
}

/// A backend's loaded state, with what it needs to free that state and load it again.
///
/// @discussion A lifecycle starts loaded with the state its backend was built with. It can unload only
/// after ``enableReload(_:measurement:)`` gives it a closure that loads the state again; until then
/// ``unload()`` does nothing and ``isLoaded`` stays true.
///
/// A run takes the state through ``withState(_:)``, or ``begin()`` and ``end()`` when it outlives a call,
/// and counts as in flight meanwhile. An unload with runs in flight is recorded, and the last run to end
/// frees the state. Loads are serialized, so concurrent runs that find the model unloaded share one load.
final class NFKMLXLifecycle<State>: @unchecked Sendable {

    typealias Load = () throws -> State

    private let lock = NSLock()
    private let loadLock = NSLock()
    private var state: State?
    private var load: Load?
    private var runsInFlight = 0
    private var unloadRequested = false
    private var measurement: NFKMLXLoadMeasurement?
    private var freeHandler: (() -> Void)?

    init(_ state: State) {
        self.state = state
    }

    /// Gives the lifecycle the closure that loads the state again, and the figures the first load measured.
    func enableReload(_ load: @escaping Load, measurement: NFKMLXLoadMeasurement?) {
        lock.withLock {
            self.load = load
            if let measurement {
                self.measurement = measurement
            }
        }
    }

    /// Sets the work that runs each time the state is freed, before the lifecycle drops it: a backend's
    /// own caches over the state.
    func setFreeHandler(_ handler: @escaping () -> Void) {
        lock.withLock { freeHandler = handler }
    }

    var canReload: Bool { lock.withLock { load != nil } }

    var isLoaded: Bool { lock.withLock { state != nil } }

    var residentByteCount: Int {
        lock.withLock { state != nil ? measurement?.residentBytes ?? 0 : 0 }
    }

    var peakByteCount: Int { lock.withLock { measurement?.peakBytes ?? 0 } }

    /// The last measured load's figures under the `NFKModelInfo*` keys, or none before a measured load.
    var measurementInfo: [String: Any] { lock.withLock { measurement?.modelInfo ?? [:] } }

    /// The loaded state; throws when the lifecycle is not loaded. A fresh backend's state is read this way
    /// when an older backend of its class loads again.
    func loadedState() throws -> State {
        guard let state = lock.withLock({ state }) else {
            throw NFKMLXError.notReady
        }
        return state
    }

    /// Runs `body` on the loaded state, loading it first when it was unloaded.
    func withState<T>(_ body: (State) throws -> T) throws -> T {
        try begin()
        defer { end() }
        return try body(try loadedState())
    }

    /// Runs `body` on the state when it is loaded, and returns nil without loading when it is not.
    func ifLoaded<T>(_ body: (State) throws -> T) rethrows -> T? {
        let entered = lock.withLock { () -> Bool in
            guard state != nil else {
                return false
            }
            runsInFlight += 1
            return true
        }
        guard entered else {
            return nil
        }
        defer { leave() }
        guard let current = try? loadedState() else {
            return nil
        }
        return try body(current)
    }

    /// Loads the state when it is not loaded, and withdraws an unload waiting for runs to end.
    func prepare() throws {
        lock.withLock { unloadRequested = false }
        try begin()
        end()
    }

    /// Counts a run in flight, loading the state first when it was unloaded. Pair with ``end()``.
    func begin() throws {
        NFKMLXActivity.begin()
        do {
            if case .entered = lock.withLock({ enter() }) {
                return
            }
            try loadLock.withLock {
                switch lock.withLock({ enter() }) {
                case .entered:
                    return
                case .unavailable:
                    throw NFKMLXError.notReady
                case .load(let load):
                    let (loaded, measured) = try NFKMLXLoadMeasurement.measuring(load)
                    lock.withLock {
                        state = loaded
                        if let measured {
                            measurement = measured
                        }
                        runsInFlight += 1
                    }
                }
            }
        } catch {
            NFKMLXActivity.end()
            throw error
        }
    }

    /// Ends a run ``begin()`` counted, and frees the state when an unload waits for the last run.
    func end() {
        NFKMLXActivity.end()
        leave()
    }

    /// Frees the state now, or when the last run in flight ends. It does nothing without a way to load the
    /// state again.
    func unload() {
        var taken: State? = lock.withLock {
            guard load != nil, state != nil else {
                return nil
            }
            unloadRequested = true
            return takeIfUnloadable()
        }
        if taken != nil {
            free(&taken)
        }
    }

    private enum Entry {
        case entered
        case load(Load)
        case unavailable
    }

    /// Counts a run when the state is loaded, or answers how to load it. Runs on the lock.
    private func enter() -> Entry {
        if state != nil {
            runsInFlight += 1
            return .entered
        }
        if let load {
            return .load(load)
        }
        return .unavailable
    }

    private func leave() {
        var taken: State? = lock.withLock {
            runsInFlight -= 1
            return takeIfUnloadable()
        }
        if taken != nil {
            free(&taken)
        }
    }

    /// Takes the state out when an unload waits and no run is in flight. Runs on the lock.
    private func takeIfUnloadable() -> State? {
        guard unloadRequested, runsInFlight == 0, let taken = state else {
            return nil
        }
        state = nil
        unloadRequested = false
        return taken
    }

    /// Drops the last reference the lifecycle held, so MLX frees the arrays, then returns MLX's cache.
    private func free(_ taken: inout State?) {
        let handler = lock.withLock { freeHandler }
        handler?()
        autoreleasepool {
            taken = nil
        }
        NFKMLXActivity.releaseCachedMemoryWhenIdle()
    }
}

extension NFKMLXLifecycle {

    /// Enables reload through `build`, which builds a fresh backend of class `Backend` from the same source,
    /// whose loaded state `take` reads.
    func enableReload<Backend: AnyObject>(building build: @escaping () throws -> any NFKInferenceBackend,
                                          as type: Backend.Type,
                                          measurement: NFKMLXLoadMeasurement?,
                                          state take: @escaping (Backend) throws -> State) {
        enableReload({
            guard let fresh = try build() as? Backend else {
                throw NFKMLXError.unsupportedConfiguration("loading the model again built a backend of another class")
            }
            return try take(fresh)
        }, measurement: measurement)
    }
}

/// A backend that keeps its loaded state in an ``NFKMLXLifecycle`` and loads it again from a fresh backend
/// of its own class.
protocol NFKMLXReloading: AnyObject {

    /// Whether the backend already holds a way to load its model again.
    var canReload: Bool { get }

    /// Gives the backend `build`, which builds a fresh backend of the same class from the same source, and
    /// the figures the first build measured.
    func enableReload(_ build: @escaping () throws -> any NFKInferenceBackend, measurement: NFKMLXLoadMeasurement?)
}

/// The paths that build a backend from a weights file or a release directory give it the closure that loads
/// it again.
enum NFKMLXReload {

    /// Builds a backend through `build`, measured, and returns it able to unload and load again: a
    /// lifecycle backend takes `build` itself, and any other backend is wrapped in an
    /// ``NFKMLXReloadableBackend``. A backend that can already reload is returned as built.
    static func reloadable(_ build: @escaping () throws -> any NFKInferenceBackend) throws -> any NFKInferenceBackend {
        let (backend, measurement) = try NFKMLXLoadMeasurement.measuring(build)
        if backend is NFKMLXReloadableBackend {
            return backend
        }
        if let reloading = backend as? NFKMLXReloading {
            if !reloading.canReload {
                reloading.enableReload(build, measurement: measurement)
            }
            return backend
        }
        return NFKMLXReloadableBackend(backend: backend, measurement: measurement, build: build)
    }
}
