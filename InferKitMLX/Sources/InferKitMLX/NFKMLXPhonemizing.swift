//
//  NFKMLXPhonemizing.swift
//  InferKitMLX
//
//  The phonemizer as Objective-C reaches it: a protocol an Objective-C or Swift object adopts, which the
//  text-to-speech factories take, and the neural G2P's own adoption of it.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation

/// Turns text into the phoneme symbols an acoustic model reads.
///
/// @discussion The Objective-C face of ``NFKMLXPhonemizer``. The text-to-speech factories take an
/// object adopting it; ``NFKMLXNeuralG2P`` adopts it, and so can an application's own phonemizer. The
/// symbols are the ones the acoustic model's vocabulary names (the FastSpeech2 LJSpeech release reads
/// ARPAbet with stress, such as `HH AH0 L OW1`).
///
/// Introduced in InferKit 0.4.0.
@objc(NFKMLXPhonemizing)
public protocol NFKMLXPhonemizing: NSObjectProtocol {
    /// The phoneme symbols for `text`. Introduced in InferKit 0.4.0.
    @objc(phonemesForText:error:)
    func phonemes(forText text: String) throws -> [String]
}

extension NFKMLXNeuralG2P: NFKMLXPhonemizing {

    /// ``phonemes(for:)`` for Objective-C. Introduced in InferKit 0.4.0.
    @objc(phonemesForText:error:)
    public func phonemes(forText text: String) throws -> [String] {
        phonemes(for: text)
    }

    /// A G2P model at the default configuration, loaded from the safetensors checkpoint at `weightsURL`,
    /// that names a phoneme id by `phonemeSymbols[id]` (by its number where nil or out of range).
    /// Introduced in InferKit 0.4.0.
    @objc(g2pWithWeightsURL:phonemeSymbols:error:)
    public static func g2p(weightsURL: URL, phonemeSymbols: [String]?) throws -> NFKMLXNeuralG2P {
        let g2p = NFKMLXNeuralG2P(phonemeSymbols: phonemeSymbols)
        try g2p.loadWeights(from: weightsURL)
        return g2p
    }
}

/// An ``NFKMLXPhonemizing`` object where Swift takes an ``NFKMLXPhonemizer``. The Swift requirement
/// cannot throw, so a failure answers no symbols and is kept for the caller to raise after the run.
final class NFKObjCPhonemizer: NFKMLXPhonemizer, @unchecked Sendable {
    private let phonemizer: any NFKMLXPhonemizing
    private let lock = NSLock()
    private var failure: Error?

    init(_ phonemizer: any NFKMLXPhonemizing) { self.phonemizer = phonemizer }

    func phonemes(for text: String) -> [String] {
        do {
            return try phonemizer.phonemes(forText: text)
        } catch {
            lock.withLock { failure = error }
            return []
        }
    }

    /// Raises the failure of the last run, clearing it.
    func raiseFailure() throws {
        let error = lock.withLock { () -> Error? in
            defer { failure = nil }
            return failure
        }
        if let error { throw error }
    }
}
