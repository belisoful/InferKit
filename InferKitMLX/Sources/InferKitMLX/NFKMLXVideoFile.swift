//
//  NFKMLXVideoFile.swift
//  InferKitMLX
//
//  Decoding a clip into frames and encoding frames back into a clip, which is what a video backend
//  needs and the core does not carry: `NFKVideoAsset` names a file, and reading or writing one is
//  AVFoundation work the core stays free of.
//
//  This is the video counterpart of `NFKMLXWaveFile`.
//

import AVFoundation
import CoreGraphics
import CoreVideo
import Foundation

/// Carries an asynchronous load's result across the semaphore that waits for it.
private final class NFKMLXVideoLoadBox<T>: @unchecked Sendable {
    var value: Result<T, Error>?
}

/// Reads and writes the clips an `NFKVideoAsset` points at.
public enum NFKMLXVideoFile {

    /// What a decoded clip carries: its frames in presentation order, and the timing needed to write
    /// an equivalent one back out.
    public struct Clip {
        public let frames: [CGImage]
        public let framesPerSecond: Double
        public var size: CGSize { frames.first.map { CGSize(width: $0.width, height: $0.height) } ?? .zero }
    }

    /// Decodes every frame of the clip at `url`.
    ///
    /// @discussion Frames are decoded eagerly into memory, which suits the clip lengths on-device
    /// inference is run over and keeps the backend contract synchronous. `frameLimit` bounds the read
    /// so a caller cannot exhaust memory on a long source by accident.
    public static func read(_ url: URL, frameLimit: Int = 900) throws -> Clip {
        let asset = AVURLAsset(url: url)
        guard let track = try blocking({ try await asset.loadTracks(withMediaType: .video).first }) else {
            throw NFKMLXError.unsupportedInput
        }
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(
            track: track,
            outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
        output.alwaysCopiesSampleData = false
        reader.add(output)
        guard reader.startReading() else {
            throw NFKMLXError.unsupportedConfiguration("the clip at \(url.lastPathComponent) could not be read")
        }

        var frames = [CGImage]()
        while frames.count < frameLimit, let sample = output.copyNextSampleBuffer() {
            guard let buffer = CMSampleBufferGetImageBuffer(sample) else { continue }
            if let image = self.image(from: buffer) {
                frames.append(image)
            }
        }
        reader.cancelReading()

        guard !frames.isEmpty else {
            throw NFKMLXError.unsupportedConfiguration("the clip at \(url.lastPathComponent) decoded no frames")
        }
        let rate = Double(try blocking { try await track.load(.nominalFrameRate) })
        return Clip(frames: frames, framesPerSecond: rate > 0 ? rate : 30)
    }

    /// Encodes `frames` to an H.264 clip at `url`, replacing anything already there.
    ///
    /// @discussion Every frame must share the first one's dimensions; the encoder is configured from
    /// them. A mismatch throws rather than being scaled silently, because which frame was resized
    /// would otherwise be invisible in the result.
    public static func write(_ frames: [CGImage], framesPerSecond: Double, to url: URL) throws {
        guard let first = frames.first else {
            throw NFKMLXError.unsupportedConfiguration("a clip needs at least one frame")
        }
        let (width, height) = (first.width, first.height)
        guard frames.allSatisfy({ $0.width == width && $0.height == height }) else {
            throw NFKMLXError.unsupportedConfiguration("every frame of a clip must share its dimensions")
        }
        // A codec dimension has to be even; an odd side is rejected rather than cropped silently.
        guard width % 2 == 0, height % 2 == 0 else {
            throw NFKMLXError.unsupportedConfiguration(
                "H.264 needs even frame dimensions; got \(width)×\(height)")
        }
        try? FileManager.default.removeItem(at: url)

        let writer = try AVAssetWriter(outputURL: url, fileType: .mov)
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
        ])
        input.expectsMediaDataInRealTime = false
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: input,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: width,
                kCVPixelBufferHeightKey as String: height,
            ])
        writer.add(input)
        guard writer.startWriting() else {
            throw NFKMLXError.unsupportedConfiguration("the clip at \(url.lastPathComponent) could not be written")
        }
        writer.startSession(atSourceTime: .zero)

        let scale: Int32 = 600                                  // divides the common frame rates exactly
        for (index, frame) in frames.enumerated() {
            guard let buffer = pixelBuffer(from: frame, pool: adaptor.pixelBufferPool) else {
                writer.cancelWriting()
                throw NFKMLXError.unsupportedConfiguration("frame \(index) could not be encoded")
            }
            // The writer accepts frames only while it has room, and this path is deliberately
            // synchronous: the backend contract is a blocking call the caller already runs off the
            // render thread.
            while !input.isReadyForMoreMediaData {
                Thread.sleep(forTimeInterval: 0.002)
            }
            let time = CMTime(value: CMTimeValue(Double(index) * Double(scale) / framesPerSecond),
                              timescale: scale)
            adaptor.append(buffer, withPresentationTime: time)
        }
        input.markAsFinished()

        let finished = DispatchSemaphore(value: 0)
        writer.finishWriting { finished.signal() }
        finished.wait()
        if writer.status == .failed {
            throw NFKMLXError.unsupportedConfiguration(
                "writing \(url.lastPathComponent) failed: \(writer.error?.localizedDescription ?? "unknown")")
        }
    }

    /// Runs an asynchronous AVFoundation load and waits for it.
    ///
    /// AVFoundation deprecated its synchronous property accessors, and this file's contract is
    /// synchronous: a backend's `runInferenceForRequest:` blocks, and the caller already runs it off
    /// the main thread. The wait must therefore not happen on the cooperative pool.
    private static func blocking<T: Sendable>(_ work: @escaping @Sendable () async throws -> T) throws -> T {
        let box = NFKMLXVideoLoadBox<T>()
        let finished = DispatchSemaphore(value: 0)
        Task.detached(priority: .userInitiated) {
            do { box.value = .success(try await work()) } catch { box.value = .failure(error) }
            finished.signal()
        }
        finished.wait()
        switch box.value {
        case .success(let value): return value
        case .failure(let error): throw error
        case nil: throw NFKMLXError.unsupportedConfiguration("the asset load produced no result")
        }
    }

    private static func image(from buffer: CVPixelBuffer) -> CGImage? {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else { return nil }
        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        let context = CGContext(
            data: base, width: width, height: height, bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)
        return context?.makeImage()
    }

    private static func pixelBuffer(from image: CGImage, pool: CVPixelBufferPool?) -> CVPixelBuffer? {
        var buffer: CVPixelBuffer?
        if let pool {
            CVPixelBufferPoolCreatePixelBuffer(nil, pool, &buffer)
        } else {
            CVPixelBufferCreate(nil, image.width, image.height, kCVPixelFormatType_32BGRA,
                                [kCVPixelBufferCGImageCompatibilityKey: true] as CFDictionary, &buffer)
        }
        guard let buffer else { return nil }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let base = CVPixelBufferGetBaseAddress(buffer),
              let context = CGContext(
                data: base, width: CVPixelBufferGetWidth(buffer), height: CVPixelBufferGetHeight(buffer),
                bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)
        else { return nil }
        context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
        return buffer
    }
}
