//
//  NFKAppleModelDescriptors.h
//  InferKit
//

#ifndef NFKAppleModelDescriptors_h
#define NFKAppleModelDescriptors_h

#import <Foundation/Foundation.h>
#import <InferKit/NFKInferenceBackend.h>
#import <InferKit/NFKModelCatalog.h>

NS_ASSUME_NONNULL_BEGIN

/*!
	@class      NFKAppleModelDescriptors
	@abstract   The model descriptors of the core's Apple-framework engines, and the registry that
				builds their backends.
	@discussion One descriptor per engine: the Vision backends, VideoToolbox's three frame
				processors, Apple's speech recognizer, AVFoundation's voices, SoundAnalysis, and
				NaturalLanguage's sentence embedding. Each descriptor's engine is
				NFKModelEngineApple, and its inputs, options, and outputs are the key sets of the
				backend backendNamed: builds for its registry name. The registry name is the
				descriptor's identifier.

				- The identifier is the backend's backendIdentifier, such as "vision-text". Where one
				  backend class runs several tasks with different keys, each task has its own
				  descriptor: "videotoolbox-super-resolution", "videotoolbox-frame-interpolation",
				  and "videotoolbox-optical-flow", and "vision-tracking" beside
				  "vision-tracking-trajectory".
				- Every descriptor has one default variant whose source is NFKModelSourceKindSystem
				  with NFKModelSourceLayoutNone: the model is part of the operating system and has no
				  files of its own to download.
				- The license is "Apple system framework". The requirements give the least OS version
				  at which the backend runs, per platform.
				- A descriptor whose engine also answers an NFKDynamicBackend capability names that
				  provider class in providerClassName.

				NFKVisionCoreMLBackend has no descriptor, because it runs a Core ML model the app
				supplies. The shared NFKModelCatalog loads these descriptors first. Introduced in
				InferKit 0.4.0.
*/
@interface NFKAppleModelDescriptors : NSObject <NFKModelDescriptorProvider>

/*! The descriptors of the core's Apple-framework engines, sorted by identifier. */
+ (NSArray<NFKModelDescriptor *> *)modelDescriptors;

/*! Whether a descriptor lists the choices this machine has, such as the voices apple-speech-synthesis
	offers. YES by default, and each call to modelDescriptors reads them again. A manifest written for
	every machine, such as the catalog snapshot, sets NO, and those options list no choices.
	Introduced in InferKit 0.4.0. */
@property (class, nonatomic) BOOL includesMachineChoices;

/*! The registry names backendNamed: builds, sorted. */
@property (class, nonatomic, readonly, copy) NSArray<NSString *> *registryNames;

/*! A new backend for a descriptor's registry name, configured for its task, or nil for a name this
	registry does not know. The backend is built whether or not this machine runs it, so isReady
	answers that. */
+ (nullable id<NFKInferenceBackend>)backendNamed:(NSString *)registryName;

/*! The backend the catalog's build entry asks for: backendNamed: of the variant's registry name, else
	the descriptor's, with the request's property options set on it. Returns nil with
	NFKModelCatalogErrorUnknownModel for a name this registry does not know. Introduced in
	InferKit 0.4.0. */
+ (nullable id<NFKInferenceBackend>)backendForBuildRequest:(NFKModelBuildRequest *)request error:(NSError **)error;

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKAppleModelDescriptors_h */
