//
//  NFKMLXPhonemizer.swift
//  InferKitMLX
//

import Foundation

/// Maps text to phoneme symbols — the front-end of a text-to-speech pipeline. Two implementations
/// ship: `NFKMLXEspeakPhonemizer` (uses a system espeak-ng if installed; macOS) and `NFKMLXNeuralG2P`
/// (an in-toolkit MLX grapheme-to-phoneme model, no external dependency).
public protocol NFKMLXPhonemizer: Sendable {
    /// The phoneme symbols for `text`.
    func phonemes(for text: String) -> [String]
}

#if os(macOS)
/// A phonemizer backed by a system-installed espeak-ng, invoked as a subprocess. InferKit does not
/// bundle espeak-ng (it is GPLv3); `Tools/espeak/install.sh` installs it, and this uses it only when
/// present. macOS only — iOS cannot spawn a subprocess, so use `NFKMLXNeuralG2P` there.
public struct NFKMLXEspeakPhonemizer: NFKMLXPhonemizer {

    public let executableURL: URL
    public let voice: String

    /// Fails if espeak-ng is not found (in `PATH`-typical locations, or `executableURL`).
    public init?(voice: String = "en", executableURL: URL? = nil) {
        self.voice = voice
        if let executableURL {
            self.executableURL = executableURL
            return
        }
        let candidates = ["/opt/homebrew/bin/espeak-ng", "/usr/local/bin/espeak-ng", "/usr/bin/espeak-ng"]
        guard let found = candidates.first(where: { FileManager.default.isExecutableFile(atPath: $0) }) else {
            return nil
        }
        self.executableURL = URL(fileURLWithPath: found)
    }

    /// Whether a usable espeak-ng is installed.
    public static var isInstalled: Bool { NFKMLXEspeakPhonemizer() != nil }

    public func phonemes(for text: String) -> [String] {
        let process = Process()
        process.executableURL = executableURL
        process.arguments = ["-q", "--ipa=3", "-v", voice, text]      // --ipa=3 separates phonemes with '_'
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = Pipe()
        do {
            try process.run()
            process.waitUntilExit()
        } catch {
            return []
        }
        let data = pipe.fileHandleForReading.readDataToEndOfFile()
        let output = String(data: data, encoding: .utf8) ?? ""
        return output.split(whereSeparator: { $0 == "_" || $0.isWhitespace }).map(String.init)
    }
}
#endif
