//
//  NFKMLXCLIPProbeFactories.swift
//  InferKitMLX
//
//  A saved CLIP probe behind the InferKit contract, built from Objective-C: the CLIP weights, the probe
//  file `NFKMLXWeights.save` wrote, and the class names.
//
//  Introduced in InferKit 0.4.0.
//

import Foundation
import InferKit

extension NFKMLXCLIP {

    /// The registry name a CLIP probe backend builds under. Introduced in InferKit 0.4.0.
    @objc public static let probeModelName = "clip-probe"

    /// The CLIP network at `variant` with a saved probe over it, as an image classifier: an image under
    /// `NFKInputImage` becomes ranked `NFKClassification`s under `NFKOutputClassifications`.
    ///
    /// @discussion `weightsURL` holds the CLIP checkpoint at `variant` (nil builds random weights);
    /// `probeURL` is the file `NFKMLXWeights.save` wrote for an ``NFKMLXCLIPProbe`` trained through
    /// ``trainProbe(_:embeddings:labels:sampler:optimizer:steps:clipGradientNorm:checkpoint:observer:)``;
    /// `labels` names the classes in order, or nil to report class indices. A probe whose width is not
    /// the variant's embedding width is refused. Introduced in InferKit 0.4.0.
    @objc(probeBackendWithVariant:weightsURL:probeURL:labels:error:)
    public static func probeBackend(variant: NFKMLXCLIPVariant, weightsURL: URL?, probeURL: URL,
                                    labels: [String]?) throws -> any NFKInferenceBackend {
        let configuration = specs(for: variant).configuration
        let probe = try NFKMLXCLIPProbe(weightsURL: probeURL)
        guard probe.embedDimensions == configuration.embedDimensions else {
            throw NFKMLXError.trainingDataMismatch(
                "the probe reads \(probe.embedDimensions)-wide embeddings and this CLIP produces \(configuration.embedDimensions)")
        }
        return probeBackend(net: try network(weightsURL: weightsURL, configuration: configuration), probe: probe,
                            labels: labels)
    }

    /// ``probeBackend(variant:weightsURL:probeURL:labels:)`` with the CLIP checkpoint downloaded from
    /// Hugging Face. The call blocks on the network, so run it off the main and render threads.
    /// Introduced in InferKit 0.4.0.
    @objc(probeBackendWithVariant:repo:weightsPath:revision:cacheDirectoryURL:probeURL:labels:error:)
    public static func probeBackend(variant: NFKMLXCLIPVariant, repo: String, weightsPath: String, revision: String?,
                                    cacheDirectoryURL: URL?, probeURL: URL,
                                    labels: [String]?) throws -> any NFKInferenceBackend {
        try NFKMLXBuildHolds.keeping {
            let url = try NFKMLXDownload.weightsURL(repo: repo, weightsPath: weightsPath, revision: revision,
                                                    cacheDirectoryURL: cacheDirectoryURL)
            return try probeBackend(variant: variant, weightsURL: url, probeURL: probeURL, labels: labels)
        }
    }

    /// The asynchronous form of
    /// ``probeBackend(variant:repo:weightsPath:revision:cacheDirectoryURL:probeURL:labels:)``. The handler
    /// runs on a background queue. Introduced in InferKit 0.4.0.
    @objc(probeBackendWithVariant:repo:weightsPath:revision:cacheDirectoryURL:probeURL:labels:completionHandler:)
    public static func probeBackend(variant: NFKMLXCLIPVariant, repo: String, weightsPath: String, revision: String?,
                                    cacheDirectoryURL: URL?, probeURL: URL, labels: [String]?,
                                    completionHandler: @escaping ((any NFKInferenceBackend)?, Error?) -> Void) {
        NFKMLXReleaseDownload.async(completionHandler) {
            try probeBackend(variant: variant, repo: repo, weightsPath: weightsPath, revision: revision,
                             cacheDirectoryURL: cacheDirectoryURL, probeURL: probeURL, labels: labels)
        }
    }

    /// The probe directory the registry reads: `model.safetensors` (the CLIP ViT-B/32 checkpoint),
    /// `probe.safetensors`, and, where present, `labels.json` (an array of class names).
    static func probeBackend(directoryURL: URL) throws -> any NFKInferenceBackend {
        let labelsURL = directoryURL.appendingPathComponent("labels.json")
        var labels: [String]?
        if FileManager.default.fileExists(atPath: labelsURL.path) {
            guard let names = try JSONSerialization.jsonObject(with: Data(contentsOf: labelsURL)) as? [String] else {
                throw NFKMLXError.unsupportedConfiguration("labels.json is not an array of class names")
            }
            labels = names
        }
        return try probeBackend(variant: .vitB32, weightsURL: directoryURL.appendingPathComponent("model.safetensors"),
                                probeURL: directoryURL.appendingPathComponent("probe.safetensors"), labels: labels)
    }

    /// Registers the probe backend under ``probeModelName``. The registry's URL is a directory holding
    /// `model.safetensors` (the CLIP ViT-B/32 checkpoint), `probe.safetensors`, and optionally
    /// `labels.json`; nil is refused. Introduced in InferKit 0.4.0.
    @objc public static func registerProbe() {
        NFKMLXGenerationBackend.registerDirectoryModel(probeModelName) { try probeBackend(directoryURL: $0) }
    }
}
