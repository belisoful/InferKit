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

    /// The runs and loads in flight.
    static var inFlightCount: Int { lock.withLock { inFlight } }

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
/// An unload during a load is recorded the same way: the load completes, the run that asked for it
/// proceeds, and the state is freed when that run ends. A load reads local files only
/// (``NFKMLXReload/reloading(_:)``).
final class NFKMLXLifecycle<State>: @unchecked Sendable {

    typealias Load = () throws -> State

    private let lock = NSLock()
    private let loadLock = NSLock()
    private var state: State?
    private var load: Load?
    private var runsInFlight = 0
    private var unloadRequested = false
    private var isLoading = false
    private var measurement: NFKMLXLoadMeasurement?
    private var freeHandler: (() -> Void)?
    private var loadOptions: NFKMLXLoadOptions?

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

    /// Keeps the load options the state was built with, which ``loadOptionValues`` reports.
    func setLoadOptions(_ options: NFKMLXLoadOptions) {
        lock.withLock { loadOptions = options }
    }

    /// The load options the state was built with, as requested with defaults filled; empty for a state
    /// built without options.
    var loadOptionValues: [String: Any] { lock.withLock { loadOptions?.values ?? [:] } }

    var canReload: Bool { lock.withLock { load != nil } }

    var isLoaded: Bool { lock.withLock { state != nil } }

    var residentByteCount: Int {
        lock.withLock { state != nil ? measurement?.residentBytes ?? 0 : 0 }
    }

    var peakByteCount: Int { lock.withLock { measurement?.peakBytes ?? 0 } }

    /// The last measured load's figures, and the residency the last load resolved to, under the
    /// `NFKModelInfo*` keys; empty before a measured load of a backend built without options.
    var loadInfo: [String: Any] {
        lock.withLock {
            var info = measurement?.modelInfo ?? [:]
            if let residency = loadOptions?.resolvedResidency {
                info[NFKModelInfoResidency] = residency.optionValue
            }
            return info
        }
    }

    /// `info` with the last measured load's figures and residency over it, and, after a measured load
    /// that neither reported nor showed a residency, ``NFKMLXResidency/resident``: the factory read the
    /// weights in and the measurement holds them. A backend's own residency (a language backend's paged
    /// experts) stands where the load reported none.
    func modelInfo(over info: [String: Any]) -> [String: Any] {
        var merged = info.merging(loadInfo) { _, measured in measured }
        let measured = lock.withLock { measurement != nil }
        if measured && merged[NFKModelInfoResidency] == nil {
            merged[NFKModelInfoResidency] = NFKMLXResidency.resident.optionValue
        }
        return merged
    }

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
                    try loadState(load)
                }
            }
        } catch {
            NFKMLXActivity.end()
            throw error
        }
    }

    /// Runs `load` on the load lock and counts the run that asked for it. An unload during the load is
    /// recorded, so the state is freed when that run ends; a failed load withdraws it.
    private func loadState(_ load: Load) throws {
        lock.withLock { isLoading = true }
        do {
            let (loaded, measured) = try NFKMLXLoadMeasurement.measuring {
                try NFKMLXBuildHolds.keeping { try NFKMLXReload.reloading(load) }
            }
            lock.withLock {
                isLoading = false
                state = loaded
                if let measured {
                    measurement = measured
                }
                runsInFlight += 1
            }
        } catch {
            lock.withLock {
                isLoading = false
                unloadRequested = false
            }
            throw error
        }
    }

    /// Ends a run ``begin()`` counted, and frees the state when an unload waits for the last run.
    func end() {
        NFKMLXActivity.end()
        leave()
    }

    /// Frees the state now, or when the last run in flight ends. During a load it frees the state when the
    /// run that asked for the load ends. It does nothing without a way to load the state again, or when
    /// the state is neither loaded nor loading.
    func unload() {
        var taken: State? = lock.withLock {
            guard load != nil, state != nil || isLoading else {
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

    /// Gives the backend the load options it was built with, which its `loadOptionValues` reports.
    func setLoadOptions(_ options: NFKMLXLoadOptions)
}

/// The paths that build a backend from a weights file or a release directory give it the closure that loads
/// it again.
///
/// A load after an unload reads local files only. It runs inside ``reloading(_:)``, where the download
/// helpers take a file from the cache and make no request: a file missing from the cache throws
/// ``NFKMLXReloadError/modelFilesRemoved``. A build from a local file or folder records the files it
/// opened (``NFKMLXFileReads``) and the files its catalog variant lists, and a load fails with the same
/// error when one of them is gone.
enum NFKMLXReload {

    @TaskLocal private static var reloadInProgress = false

    /// Whether the calling code runs inside a load after an unload, where a download helper reads the
    /// cache and makes no request.
    static var isReloading: Bool { reloadInProgress }

    /// Runs `load`, a load after an unload, with every download helper reading local files only.
    static func reloading<T>(_ load: () throws -> T) rethrows -> T {
        try $reloadInProgress.withValue(true, operation: load)
    }

    /// The error a load after an unload reports when the model's files are no longer on disk: `path` is
    /// the first missing file, and `source` names the repository or folder the model came from.
    static func modelFilesRemovedError(path: String, source: String) -> NSError {
        let reason = "the model's files were removed: \(source) no longer holds \(path). Loading the model again "
            + "reads local files only; download them again through an NFKDownload, and the next run loads them"
        return NSError(domain: NFKMLXReloadError.errorDomain, code: NFKMLXReloadError.modelFilesRemoved.rawValue,
                       userInfo: [NSLocalizedDescriptionKey: reason, NSFilePathErrorKey: path])
    }

    /// Builds a backend through `build`, measured, and returns it able to unload and load again: a
    /// lifecycle backend takes `build` itself, and any other backend is wrapped in an
    /// ``NFKMLXReloadableBackend``. A backend that can already reload is returned as built.
    static func reloadable(_ build: @escaping () throws -> any NFKInferenceBackend) throws -> any NFKInferenceBackend {
        try reloadable(loadOptions: nil, build)
    }

    /// ``reloadable(_:)`` for a backend built with `loadOptions`, which its `loadOptionValues` reports.
    /// When `source` is a local file or folder, a load after an unload requires on disk the files the first
    /// build opened and the files of `listed` (paths under a folder `source`) it found there, and otherwise
    /// throws ``NFKMLXReloadError/modelFilesRemoved``. The same error replaces a build error that reports a
    /// missing file under `source`.
    static func reloadable(loadOptions: NFKMLXLoadOptions?, reading source: URL? = nil, listing listed: [String] = [],
                           _ build: @escaping () throws -> any NFKInferenceBackend) throws -> any NFKInferenceBackend {
        let opened = NFKMLXFileReads()
        let (backend, measurement) = try NFKMLXLoadMeasurement.measuring { try opened.recording(build) }
        if let reloadable = backend as? NFKMLXReloadableBackend {
            reportLoadOptions(of: reloadable, loadOptions)
            return backend
        }
        let reload = NFKMLXLocalFiles(source: source, listed: listed, opened: opened.urls).map { files in
            { () throws -> any NFKInferenceBackend in
                try files.requirePresent()
                do {
                    return try build()
                } catch {
                    throw files.attributing(error)
                }
            }
        } ?? build
        if let reloading = backend as? NFKMLXReloading {
            if !reloading.canReload {
                reloading.enableReload(reload, measurement: measurement)
            }
            reportLoadOptions(of: backend, loadOptions)
            return backend
        }
        return NFKMLXReloadableBackend(backend: backend, measurement: measurement, build: reload,
                                       loadOptions: loadOptions)
    }

    /// Gives a lifecycle backend, or a wrapper, the load options it was built with. Any other backend
    /// reports none.
    static func reportLoadOptions(of backend: any NFKInferenceBackend, _ loadOptions: NFKMLXLoadOptions?) {
        guard let loadOptions else {
            return
        }
        if let reloadable = backend as? NFKMLXReloadableBackend {
            reloadable.setLoadOptions(loadOptions)
        } else if let reloading = backend as? NFKMLXReloading {
            reloading.setLoadOptions(loadOptions)
        }
    }
}

/// The files one build opens, which the package's weight readers and download helpers note while a
/// build records them.
final class NFKMLXFileReads: @unchecked Sendable {

    @TaskLocal private static var current: NFKMLXFileReads?

    private let lock = NSLock()
    private var recorded = [URL]()

    /// Notes `url` in the record of the build running on the calling task, if one is recording.
    static func note(_ url: URL) {
        current?.append(url)
    }

    /// Runs `build` with every file it opens noted here.
    func recording<T>(_ build: () throws -> T) rethrows -> T {
        try Self.$current.withValue(self, operation: build)
    }

    /// The files noted, in the order first noted.
    var urls: [URL] { lock.withLock { recorded } }

    private func append(_ url: URL) {
        lock.withLock {
            if !recorded.contains(url) {
                recorded.append(url)
            }
        }
    }
}

/// The model's own files at a local source, which a load after an unload requires on disk.
///
/// @discussion The record holds the source, the files the first build opened, and the listed files under a
/// folder source, each only when it was on disk at the build. A folder's other files are not recorded:
/// a shared snapshot holds other variants' files, and a transfer's `*.download` partial goes away when it
/// completes.
struct NFKMLXLocalFiles {

    let source: URL
    let paths: [String]

    /// nil when `source` is nil, not a file URL, or not on disk. `listed` paths are relative to a folder
    /// `source` and ignored for a file `source`.
    init?(source: URL?, listed: [String], opened: [URL]) {
        guard let source, source.isFileURL else {
            return nil
        }
        var isFolder: ObjCBool = false
        guard FileManager.default.fileExists(atPath: source.path, isDirectory: &isFolder) else {
            return nil
        }
        let listedURLs = isFolder.boolValue ? listed.map { source.appendingPathComponent($0) } : []
        var paths = [source.path]
        for url in listedURLs + opened where url.isFileURL && !paths.contains(url.path) && Self.isModelFile(url) {
            paths.append(url.path)
        }
        self.source = source
        self.paths = paths
    }

    private static func isModelFile(_ url: URL) -> Bool {
        url.pathExtension != "download" && FileManager.default.fileExists(atPath: url.path)
    }

    /// Throws ``NFKMLXReloadError/modelFilesRemoved`` naming the first recorded file no longer on disk. A
    /// link whose target is gone counts as missing.
    func requirePresent() throws {
        if let missing = paths.first(where: { !FileManager.default.fileExists(atPath: $0) }) {
            throw NFKMLXReload.modelFilesRemovedError(path: missing, source: source.path)
        }
    }

    /// `error` as ``NFKMLXReloadError/modelFilesRemoved`` when it, or an error under it, reports a file
    /// under the source as missing and the file is still missing; otherwise `error` itself.
    func attributing(_ error: Error) -> Error {
        var next: NSError? = error as NSError
        while let current = next {
            if let path = Self.missingFilePath(in: current), isUnderSource(path),
               !FileManager.default.fileExists(atPath: path) {
                return NFKMLXReload.modelFilesRemovedError(path: path, source: source.path)
            }
            next = current.userInfo[NSUnderlyingErrorKey] as? NSError
        }
        return error
    }

    /// The path a missing-file error names, or nil for any other error.
    private static func missingFilePath(in error: NSError) -> String? {
        let isMissing = (error.domain == NSCocoaErrorDomain
                         && [NSFileNoSuchFileError, NSFileReadNoSuchFileError].contains(error.code))
            || (error.domain == NSPOSIXErrorDomain && error.code == Int(ENOENT))
        guard isMissing else {
            return nil
        }
        return error.userInfo[NSFilePathErrorKey] as? String ?? (error.userInfo[NSURLErrorKey] as? URL)?.path
    }

    /// Whether `path` is the source or under it. `/private/var` and `/var` name the same folder.
    private func isUnderSource(_ path: String) -> Bool {
        let folder = Self.canonical(source.path)
        let candidate = Self.canonical(path)
        return candidate == folder || candidate.hasPrefix(folder + "/")
    }

    private static func canonical(_ path: String) -> String {
        let standardized = (path as NSString).standardizingPath
        return standardized.hasPrefix("/private/") ? String(standardized.dropFirst("/private".count)) : standardized
    }
}
