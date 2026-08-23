//
//  NFKMLXModelRegistry.swift
//  InferKitMLX
//

import Foundation
import InferKit

/// A registry of MLX model backends by name, so an Objective-C consumer (an FCPX plugin, an app) can
/// build and run a local MLX model without writing Swift.
///
/// An MLX model's forward pass is Swift over `MLXArray`, which Objective-C cannot express — so the
/// bring-your-own-closure backends (`NFKMLXModuleBackend`, `NFKMLXMattingBackend`,
/// `NFKMLXTensorBackend`) are constructed from Swift. This registry bridges the gap: a model author
/// registers a factory once from Swift (capturing the forward), and the consumer builds the backend
/// by name and drives it through the Objective-C `NFKInferenceBackend` protocol.
///
/// ```objc
/// // Objective-C, after the Swift side has registered "corridor-key":
/// NSError *error = nil;
/// id<NFKInferenceBackend> keyer = [NFKMLXModelRegistry backendNamed:@"corridor-key" weightsURL:url error:&error];
/// NFKInferenceResult *result = [keyer runInferenceForRequest:request error:&error];
/// ```
@objc(NFKMLXModelRegistry)
public final class NFKMLXModelRegistry: NSObject {

    /// Builds a backend, loading from `weightsURL` when the model needs it. Runs on the caller's thread.
    public typealias Factory = @Sendable (_ weightsURL: URL?) throws -> any NFKInferenceBackend

    private static let lock = NSLock()
    private static var factories: [String: Factory] = [:]

    /// Registers a model factory under `name` (call once, e.g. at launch). Swift-only: the factory
    /// captures the model's `MLXArray` forward. Registering a name again replaces it.
    public static func register(name: String, factory: @escaping Factory) {
        lock.lock()
        factories[name] = factory
        lock.unlock()
    }

    /// Removes a registered model.
    @objc public static func unregisterModel(named name: String) {
        lock.lock()
        factories[name] = nil
        lock.unlock()
    }

    /// The registered model names, sorted.
    @objc public static var registeredModelNames: [String] {
        lock.lock()
        defer { lock.unlock() }
        return factories.keys.sorted()
    }

    /// Whether a model is registered under `name`.
    @objc public static func isModelRegistered(_ name: String) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return factories[name] != nil
    }

    /// Builds the backend registered under `name`, passing `weightsURL` to its factory. Returns nil
    /// with an error when no model is registered under the name or the factory throws.
    @objc(backendNamed:weightsURL:error:)
    public static func backend(named name: String, weightsURL: URL?) throws -> any NFKInferenceBackend {
        lock.lock()
        let factory = factories[name]
        lock.unlock()
        guard let factory else {
            throw NSError(domain: NFKInferenceErrorDomain,
                          code: NFKInferenceError.error_InferenceNotReady.rawValue,
                          userInfo: [NSLocalizedDescriptionKey: "no MLX model is registered under '\(name)'"])
        }
        return try factory(weightsURL)
    }
}
