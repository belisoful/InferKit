//
//  NFKMLXLiveAudio.swift
//  InferKitMLX
//
//  The audio of one live-stream unit, read from the forms a unit carries it in.
//
//  Introduced in InferKit 0.4.0.
//

import AVFoundation
import Foundation
import InferKit

/// Reads one unit's audio under `NFKInputAudio`: an `AVAudioPCMBuffer` (float32 or int16), WAV bytes, or
/// an `NFKAudioAsset` whose file is a WAV.
enum NFKMLXLiveAudio {

    /// The unit's samples, interleaved by frame, its sample rate, and its channel count, or nil for a value
    /// of another form.
    static func samples(from value: Any?) -> (samples: [Float], sampleRate: Int, channels: Int)? {
        if let buffer = value as? AVAudioPCMBuffer {
            return samples(from: buffer)
        }
        if let asset = value as? NFKAudioAsset, let url = asset.fileURL, let data = try? Data(contentsOf: url) {
            return NFKMLXWaveFile.readChannels(data)
        }
        if let data = value as? Data {
            return NFKMLXWaveFile.readChannels(data)
        }
        return nil
    }

    private static func samples(from buffer: AVAudioPCMBuffer) -> (samples: [Float], sampleRate: Int, channels: Int)? {
        let frames = Int(buffer.frameLength)
        let channels = Int(buffer.format.channelCount)
        let interleaved = buffer.format.isInterleaved
        var samples = [Float](repeating: 0, count: frames * channels)
        if let data = buffer.floatChannelData {
            for frame in 0 ..< frames {
                for channel in 0 ..< channels {
                    samples[frame * channels + channel] = interleaved
                        ? data[0][frame * channels + channel] : data[channel][frame]
                }
            }
        } else if let data = buffer.int16ChannelData {
            for frame in 0 ..< frames {
                for channel in 0 ..< channels {
                    let value = interleaved ? data[0][frame * channels + channel] : data[channel][frame]
                    samples[frame * channels + channel] = Float(value) / 32768
                }
            }
        } else {
            return nil
        }
        return (samples, Int(buffer.format.sampleRate.rounded()), channels)
    }

    /// Mono float32 samples as a non-interleaved buffer, the form a stream answers an audio unit in. A unit
    /// that completes no output answers an empty buffer.
    static func buffer(_ samples: [Float], sampleRate: Int) -> AVAudioPCMBuffer {
        let format = AVAudioFormat(standardFormatWithSampleRate: Double(sampleRate), channels: 1)!
        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(max(samples.count, 1)))!
        buffer.frameLength = AVAudioFrameCount(samples.count)
        samples.withUnsafeBufferPointer { source in
            if let base = source.baseAddress {
                buffer.floatChannelData![0].update(from: base, count: samples.count)
            }
        }
        return buffer
    }

    /// The samples of one mono unit at `sampleRate`, or the error the unit fails with.
    static func monoSamples(_ unit: NFKInferenceRequest, sampleRate: Int) -> Result<[Float], NSError> {
        guard let (samples, rate, channels) = samples(from: unit.input(forKey: NFKInputAudio)) else {
            return .failure(NFKMLXError.unsupportedInput as NSError)
        }
        guard rate == sampleRate, channels == 1 else {
            return .failure(formatError(sampleRate: rate, channels: channels, expectedRate: sampleRate, expectedChannels: 1))
        }
        return .success(samples)
    }

    /// The error a unit at another rate or channel count than the stream takes fails with.
    static func formatError(sampleRate: Int, channels: Int, expectedRate: Int, expectedChannels: Int) -> NSError {
        NSError(domain: NFKInferenceErrorDomain, code: NFKInferenceError.error_InferenceUnsupported.rawValue, userInfo: [
            NSLocalizedDescriptionKey: "The unit is \(channels)-channel audio at \(sampleRate) Hz; the stream takes "
                + "\(expectedChannels)-channel audio at \(expectedRate) Hz.",
        ])
    }
}
