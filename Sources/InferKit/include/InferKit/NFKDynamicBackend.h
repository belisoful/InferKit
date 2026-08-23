//
//  NFKDynamicBackend.h
//  InferKit
//

#ifndef NFKDynamicBackend_h
#define NFKDynamicBackend_h

#import <Foundation/Foundation.h>
#import <InferKit/NFKInferenceBackend.h>

NS_ASSUME_NONNULL_BEGIN

/*!
	@protocol   NFKDynamicBackendProvider
	@abstract   A class that supplies an InferKit backend, discovered at runtime by name.
	@discussion InferKit ships only zero-dependency backends. A heavier
				engine (a Stable Diffusion pipeline, a Core ML or MLX model, a C/Rust runtime) is
				brought by the consumer's build. Rather than link that engine, InferKit discovers it:
				the consumer adds a small class conforming to this protocol that builds a backend around
				their engine, and InferKit resolves it by class name only if it is present in the running
				process. The engine itself needs no InferKit awareness; the provider is the adapter
				between them.

				A provider is looked up with NSClassFromString, so InferKit never references the engine's
				symbols. When the engine is not linked, the class is absent and resolution returns nil —
				the feature is simply unavailable, with no link error and no crash.
*/
@protocol NFKDynamicBackendProvider <NSObject>

/*! Builds a ready backend, or nil when the engine cannot be constructed (missing weights, unsupported
	hardware). InferKit calls this on the provider class. */
+ (nullable id<NFKInferenceBackend>)makeInferenceBackend;

@end

/*!
	@class      NFKDynamicBackend
	@abstract   Resolves optional backends that are linked into the process at runtime, without a build
				dependency on them.
	@discussion Two ways to resolve:

				- By provider class name: backendForProviderClassName: looks up a class, checks that it
				  conforms to NFKDynamicBackendProvider, and asks it to build a backend.
				- By capability: a consumer registers one or more provider class names under a capability
				  string (for example "stable-diffusion"); backendForCapability: activates the first
				  provider that is present. This lets a consumer wire several possible engines and let
				  whichever is linked win.

				The built-in "stable-diffusion" capability additionally tries a default provider class
				name, NFKStableDiffusionProvider, so a consumer that adopts that name needs no
				registration call. Registration is process-wide and thread-safe.
*/
@interface NFKDynamicBackend : NSObject

/*! YES when a class named className is linked in the process and conforms to NFKDynamicBackendProvider. */
+ (BOOL)isProviderAvailable:(NSString *)className;

/*! Resolves a backend from the named provider class. Returns nil and sets error
	(NFKInferenceErrorDomain, kNFKError_InferenceUnsupported) when the class is absent, does not conform,
	or returns no backend. */
+ (nullable id<NFKInferenceBackend>)backendForProviderClassName:(NSString *)className
														 error:(NSError * _Nullable * _Nullable)error;

/*! Registers a provider class name under a capability. Later registrations take precedence (they are
	tried first), so a consumer can override a default. */
+ (void)registerProviderClassName:(NSString *)className forCapability:(NSString *)capability;

/*! YES when any provider registered (or built-in) for the capability is present in the process. */
+ (BOOL)isCapabilityAvailable:(NSString *)capability;

/*! Resolves a backend for a capability, trying its registered providers in most-recently-registered
	order. Returns nil and sets error when none is available. */
+ (nullable id<NFKInferenceBackend>)backendForCapability:(NSString *)capability
												  error:(NSError * _Nullable * _Nullable)error;

/*! The built-in "stable-diffusion" capability. Activates the consumer's registered Stable Diffusion
	provider, or a class named NFKStableDiffusionProvider if present. Returns nil and sets error when no
	Stable Diffusion implementation is linked. */
+ (nullable id<NFKInferenceBackend>)stableDiffusionBackendWithError:(NSError * _Nullable * _Nullable)error;

@end

/*! The built-in capability string for a Stable Diffusion implementation. Its default provider class is
	NFKStableDiffusionProvider (shipped by InferKitMLX). */
extern NSString * const NFKCapabilityStableDiffusion;
/*! The built-in capability string for an on-device text-generation (LLM) implementation. Its default
	provider class is NFKFoundationModelsProvider (shipped by InferKitFoundationModels). */
extern NSString * const NFKCapabilityTextGeneration;
/*! The built-in capability string for audio transcription. Its default provider class is
	NFKMLXWhisperProvider (shipped by InferKitMLX). A consumer may register a native engine (whisper.cpp)
	to override it. */
extern NSString * const NFKCapabilityTranscription;
/*! The built-in capability string for a ControlNet implementation. It has no shipped default; a
	consumer brings a ControlNet/Stable Diffusion engine and registers its provider (conventionally
	named NFKControlNetProvider) under this capability. */
extern NSString * const NFKCapabilityControlNet;

NS_ASSUME_NONNULL_END

#endif /* NFKDynamicBackend_h */
