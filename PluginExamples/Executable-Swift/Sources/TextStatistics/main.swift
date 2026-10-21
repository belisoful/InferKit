//
//  main.swift
//  InferKit plugin examples
//
//  An executable plugin written in Swift. The app launches this program with the plugin channel on
//  descriptor 3, and NFKPluginService speaks the plugin protocol on it for the program's own NFKPlugin
//  object. The program links InferKit, and any other library it needs, into its own process.
//

import Foundation
import InferKit

private let wordsPerMinuteKey = "wordsPerMinute"
private let defaultWordsPerMinute = 238

/// What the program gives its plugin object. NFKPluginService serves a plugin it is handed, and the
/// program makes that plugin before the app's hello arrives, so the context carries no option values.
final class StandaloneContext: NSObject, NFKPluginContext {
    let temporaryDirectoryURL: URL
    let hostAPIVersion = NFKInferKit.apiVersion
    let options: [String: Any] = [:]
    let runsInHostProcess = false

    init(temporaryDirectoryURL: URL) {
        self.temporaryDirectoryURL = temporaryDirectoryURL
    }

    func logMessage(_ message: String, level: NFKPluginLogLevel) {
        FileHandle.standardError.write(Data("text-statistics: \(message)\n".utf8))
    }
}

/// Counts what a text holds and estimates its reading time.
final class TextStatisticsBackend: NSObject, NFKInferenceBackend {
    private let defaultSpeed: Int

    init(wordsPerMinute: Int) {
        defaultSpeed = wordsPerMinute
    }

    var isReady: Bool { true }
    var backendIdentifier: String { "text-statistics" }
    var supportedInputKeys: Set<String> { [NFKInputPrompt] }
    var supportedParameterKeys: Set<String> { [wordsPerMinuteKey] }
    var supportedOutputKeys: Set<String> { [NFKOutputStructured] }
    var modelInfo: [String: Any] { [NFKModelInfoArchitecture: "text-statistics"] }

    func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        guard let text = request.inputs[NFKInputPrompt] as? String else {
            throw NSError(domain: NFKInferenceErrorDomain, code: NFKInferenceError.error_InferenceMissingInput.rawValue,
                          userInfo: [NSLocalizedDescriptionKey: "text statistics need a text under \"prompt\""])
        }
        let speed = (request.parameters[wordsPerMinuteKey] as? NSNumber)?.intValue ?? defaultSpeed
        guard speed > 0 else {
            throw NSError(domain: NFKInferenceErrorDomain, code: NFKInferenceError.error_InferenceUnsupported.rawValue,
                          userInfo: [NSLocalizedDescriptionKey: "\"wordsPerMinute\" is at least 1"])
        }
        return NFKInferenceResult(outputs: [NFKOutputStructured: statistics(of: text, wordsPerMinute: speed)])
    }

    private func count(_ text: String, by unit: String.EnumerationOptions) -> (count: Int, characters: Int) {
        var count = 0
        var characters = 0
        text.enumerateSubstrings(in: text.startIndex..., options: unit) { substring, _, _, _ in
            let trimmed = substring?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            if !trimmed.isEmpty {
                count += 1
                characters += trimmed.count
            }
        }
        return (count, characters)
    }

    private func statistics(of text: String, wordsPerMinute: Int) -> [String: Any] {
        let words = count(text, by: .byWords)
        let averageWordLength = words.count > 0 ? Double(words.characters) / Double(words.count) : 0
        return [
            "characters": text.count,
            "words": words.count,
            "sentences": count(text, by: .bySentences).count,
            "lines": count(text, by: .byLines).count,
            "paragraphs": count(text, by: .byParagraphs).count,
            "averageWordLength": (averageWordLength * 100).rounded() / 100,
            "readingSeconds": Int((Double(words.count) * 60 / Double(wordsPerMinute)).rounded(.up)),
        ]
    }
}

/// The plugin object. Its models come from the bundle's manifest.
final class TextStatisticsPlugin: NSObject, NFKPlugin {
    private let context: NFKPluginContext
    private let models: [NFKModelDescriptor]

    required init(context: NFKPluginContext) {
        self.context = context
        models = (try? NFKPluginManifest(bundleURL: pluginBundleURL()))?.models ?? []
    }

    func modelDescriptors() -> [NFKModelDescriptor] {
        models
    }

    func backend(forModelIdentifier modelIdentifier: String, options: [String: Any]) throws -> NFKInferenceBackend {
        guard modelIdentifier == "text-statistics" else {
            throw NSError(domain: NFKPluginErrorDomain, code: NFKPluginError.unknownModel.rawValue,
                          userInfo: [NSLocalizedDescriptionKey: "the plugin has no model \"\(modelIdentifier)\""])
        }
        let speed = (options[wordsPerMinuteKey] as? NSNumber)?.intValue ?? defaultWordsPerMinute
        return TextStatisticsBackend(wordsPerMinute: speed)
    }
}

/// The plugin bundle that holds this program, which lies at Contents/MacOS/TextStatistics inside it.
func pluginBundleURL() -> URL {
    let executable = (Bundle.main.executableURL ?? URL(fileURLWithPath: CommandLine.arguments[0])).resolvingSymlinksInPath()
    return executable.deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
}

private func serve() -> Int32 {
    let manifest: NFKPluginManifest
    do {
        manifest = try NFKPluginManifest(bundleURL: pluginBundleURL())
    } catch {
        FileHandle.standardError.write(Data("text-statistics: \(error.localizedDescription)\n".utf8))
        return EX_DATAERR
    }
    let temporary = FileManager.default.temporaryDirectory
        .appendingPathComponent("\(manifest.identifier).\(UUID().uuidString)", isDirectory: true)
    try? FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: temporary) }

    let context = StandaloneContext(temporaryDirectoryURL: temporary)
    let service = NFKPluginService(plugin: TextStatisticsPlugin(context: context), manifest: manifest)
    let channel = ProcessInfo.processInfo.environment["INFERKIT_PLUGIN_CHANNEL_FD"].flatMap { Int32($0) } ?? 3
    do {
        try service.run(withChannelDescriptor: channel)
    } catch {
        FileHandle.standardError.write(Data("text-statistics: \(error.localizedDescription)\n".utf8))
        return EX_PROTOCOL
    }
    return 0
}

exit(serve())
