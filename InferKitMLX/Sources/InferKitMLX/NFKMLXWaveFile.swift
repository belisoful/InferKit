//
//  NFKMLXWaveFile.swift
//  InferKitMLX
//

import Foundation

/// Encodes float samples to a 16-bit PCM WAV file. Foundation only, so it holds no MLX and is tested
/// directly. Samples outside `-1...1` clamp; interleave channels for multi-channel audio.
enum NFKMLXWaveFile {

    static func data(samples: [Float], sampleRate: Int, channels: Int = 1) -> Data {
        let bitsPerSample = 16
        let blockAlign = channels * bitsPerSample / 8
        let byteRate = sampleRate * blockAlign
        let dataSize = samples.count * bitsPerSample / 8

        var data = Data()
        data.append(ascii: "RIFF")
        data.append(littleEndian: UInt32(36 + dataSize))
        data.append(ascii: "WAVE")

        data.append(ascii: "fmt ")
        data.append(littleEndian: UInt32(16))                  // PCM fmt chunk size
        data.append(littleEndian: UInt16(1))                   // audioFormat = PCM
        data.append(littleEndian: UInt16(channels))
        data.append(littleEndian: UInt32(sampleRate))
        data.append(littleEndian: UInt32(byteRate))
        data.append(littleEndian: UInt16(blockAlign))
        data.append(littleEndian: UInt16(bitsPerSample))

        data.append(ascii: "data")
        data.append(littleEndian: UInt32(dataSize))
        for sample in samples {
            let clamped = max(-1, min(1, sample))
            let value = Int16(clamped * Float(Int16.max))
            data.append(littleEndian: UInt16(bitPattern: value))
        }
        return data
    }

    /// Writes the samples to `url` as a WAV file.
    static func write(samples: [Float], sampleRate: Int, channels: Int = 1, to url: URL) throws {
        try data(samples: samples, sampleRate: sampleRate, channels: channels).write(to: url)
    }

    /// Reads a 16-bit PCM WAV file's samples (`-1...1`), downmixing to mono. Returns nil for a
    /// non-PCM16 file. Foundation only; a caller needing other formats decodes with AVFoundation.
    static func read(_ data: Data) -> (samples: [Float], sampleRate: Int)? {
        let bytes = [UInt8](data)
        guard bytes.count > 44,
              String(bytes: bytes[0 ..< 4], encoding: .ascii) == "RIFF",
              String(bytes: bytes[8 ..< 12], encoding: .ascii) == "WAVE" else { return nil }

        func u16(_ i: Int) -> Int { Int(bytes[i]) | (Int(bytes[i + 1]) << 8) }
        func u32(_ i: Int) -> Int { u16(i) | (u16(i + 2) << 16) }

        var sampleRate = 0, channels = 1, bitsPerSample = 16
        var offset = 12
        var dataStart = -1, dataSize = 0
        while offset + 8 <= bytes.count {
            let id = String(bytes: bytes[offset ..< offset + 4], encoding: .ascii)
            let size = u32(offset + 4)
            if id == "fmt " {
                channels = max(u16(offset + 10), 1)
                sampleRate = u32(offset + 12)
                bitsPerSample = u16(offset + 22)
            } else if id == "data" {
                dataStart = offset + 8
                dataSize = min(size, bytes.count - dataStart)
            }
            offset += 8 + size + (size & 1)
        }
        guard dataStart >= 0, bitsPerSample == 16, sampleRate > 0 else { return nil }

        let frameCount = dataSize / (2 * channels)
        var samples = [Float](repeating: 0, count: frameCount)
        for frame in 0 ..< frameCount {
            var sum: Float = 0
            for channel in 0 ..< channels {
                let index = dataStart + (frame * channels + channel) * 2
                let value = Int16(bitPattern: UInt16(bytes[index]) | (UInt16(bytes[index + 1]) << 8))
                sum += Float(value) / Float(Int16.max)
            }
            samples[frame] = sum / Float(channels)
        }
        return (samples, sampleRate)
    }
}

private extension Data {
    mutating func append(ascii string: String) {
        append(contentsOf: string.utf8)
    }
    mutating func append(littleEndian value: UInt16) {
        append(UInt8(value & 0xff)); append(UInt8((value >> 8) & 0xff))
    }
    mutating func append(littleEndian value: UInt32) {
        for shift in stride(from: 0, to: 32, by: 8) {
            append(UInt8((value >> UInt32(shift)) & 0xff))
        }
    }
}
