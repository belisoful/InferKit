//
//  NFKVisionDocumentBackend.swift
//  InferKitAppleSwift
//

import CoreGraphics
import Foundation
import InferKit
import Vision

/// Reads a document's structure with Vision's `RecognizeDocumentsRequest`, which ships only in
/// Vision's Swift module and so cannot be reached from the core's Objective-C target.
///
/// The core's `NFKVisionTextBackend` reads the lines in an image. This one reads what the lines
/// belong to: paragraphs, lists, and tables with their rows and columns, which is what turns a
/// photographed receipt or form into data.
///
/// - Input: `NFKInputImage`.
/// - Output: the whole transcript under `NFKOutputText`; the structure under `NFKOutputStructured`
///   as `paragraphs` (an array of strings), `lists` (arrays of item strings), and `tables` (arrays
///   of rows, each row an array of cell strings).
///
/// Needs macOS 26 / iOS 26. Introduced in InferKit 0.4.0.
@objc(NFKVisionDocumentBackend)
public final class NFKVisionDocumentBackend: NSObject, NFKInferenceBackend {

    @objc public override init() {
        super.init()
    }

    @objc public var isReady: Bool { true }

    @objc public var backendIdentifier: String { "vision-document" }

    @objc public var supportedInputKeys: Set<String> { [NFKInputImage] }

    @objc(runInferenceForRequest:error:)
    public func runInference(for request: NFKInferenceRequest) throws -> NFKInferenceResult {
        guard let image = NFKAppleImage.cgImage(in: request, key: NFKInputImage) else {
            throw NFKAppleImage.error(.error_InferenceMissingInput,
                                      "the request carries no image under NFKInputImage")
        }

        let outcome = NFKAppleOutcome<NFKInferenceResult>()
        let semaphore = DispatchSemaphore(value: 0)
        Task.detached(priority: .userInitiated) {
            do {
                let observations = try await RecognizeDocumentsRequest().perform(on: image)
                outcome.succeed(Self.result(for: observations))
            } catch {
                outcome.fail(error)
            }
            semaphore.signal()
        }
        semaphore.wait()
        return try outcome.value()
    }

    // MARK: Reading the structure

    private static func result(for observations: [DocumentObservation]) -> NFKInferenceResult {
        var transcripts: [String] = []
        var paragraphs: [String] = []
        var lists: [[String]] = []
        var tables: [[[String]]] = []

        for observation in observations {
            let document = observation.document
            transcripts.append(document.text.transcript)
            paragraphs.append(contentsOf: document.paragraphs.map(\.transcript))
            lists.append(contentsOf: document.lists.map { list in
                list.items.map(\.content.text.transcript)
            })
            tables.append(contentsOf: document.tables.map { table in
                table.rows.map { row in row.map(\.content.text.transcript) }
            })
        }

        return NFKInferenceResult(outputs: [
            NFKOutputText: transcripts.joined(separator: "\n"),
            NFKOutputStructured: ["paragraphs": paragraphs, "lists": lists, "tables": tables],
        ])
    }
}
