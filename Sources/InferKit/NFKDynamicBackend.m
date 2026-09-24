//
//  NFKDynamicBackend.m
//  InferKit
//

#import "NFKDynamicBackend.h"
#import "NFKErrors.h"

NSString * const NFKCapabilityStableDiffusion = @"stable-diffusion";
NSString * const NFKCapabilityTextGeneration = @"text-generation";
NSString * const NFKCapabilityTranscription = @"transcription";
NSString * const NFKCapabilityControlNet = @"controlnet";
NSString * const NFKCapabilityTextRecognition = @"text-recognition";
NSString * const NFKCapabilitySegmentation = @"segmentation";
NSString * const NFKCapabilityPose = @"pose";
NSString * const NFKCapabilityFaceDetection = @"face-detection";
NSString * const NFKCapabilityImageEmbedding = @"image-embedding";
NSString * const NFKCapabilityUpscaling = @"upscaling";
NSString * const NFKCapabilityOpticalFlow = @"optical-flow";
NSString * const NFKCapabilityTranslation = @"translation";

// The provider class names tried for each built-in capability when the consumer registers nothing,
// in order. A companion package (InferKitMLX, InferKitFoundationModels) ships the first names, so
// linking it activates the capability with a chosen model. The core's own Apple-framework engines
// come last: they need no download and no companion, so they answer when nothing else is linked.
static NSDictionary<NSString *, NSArray<NSString *> *> *NFKBuiltInDefaultProviders(void)
{
	static NSDictionary *providers;
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		providers = @{
			NFKCapabilityStableDiffusion: @[ @"NFKStableDiffusionProvider" ],
			NFKCapabilityTextGeneration:  @[ @"NFKFoundationModelsProvider" ],
			// Whisper where the consumer brought it, then Apple's newer analyzer where the Swift
			// companion is linked, then the core's own recognizer, which is always present.
			NFKCapabilityTranscription:   @[ @"NFKMLXWhisperProvider", @"NFKSpeechAnalyzerProvider",
											 @"NFKSpeechRecognitionProvider" ],
			NFKCapabilityControlNet:      @[ @"NFKControlNetProvider" ],
			NFKCapabilityTextRecognition: @[ @"NFKVisionTextProvider" ],
			NFKCapabilitySegmentation:    @[ @"NFKVisionSegmentationProvider" ],
			NFKCapabilityPose:            @[ @"NFKVisionPoseProvider" ],
			NFKCapabilityFaceDetection:   @[ @"NFKVisionFaceProvider" ],
			NFKCapabilityImageEmbedding:  @[ @"NFKVisionFeaturePrintProvider" ],
			NFKCapabilityUpscaling:       @[ @"NFKVideoToolboxUpscalingProvider" ],
			NFKCapabilityOpticalFlow:     @[ @"NFKVideoToolboxOpticalFlowProvider" ],
			// The core ships no translator; the Swift companion's wrapper answers when linked, and a
			// model-backed one registers ahead of it.
			NFKCapabilityTranslation:     @[ @"NFKMLXTranslationProvider", @"NFKTranslationProvider" ],
		};
	});
	return providers;
}

@implementation NFKDynamicBackend

// capability -> NSMutableArray<NSString *> of provider class names, newest first. Guarded by the lock.
+ (NSMutableDictionary<NSString *, NSMutableArray<NSString *> *> *)registry
{
	static NSMutableDictionary *registry;
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		registry = [NSMutableDictionary dictionary];
	});
	return registry;
}

+ (NSObject *)lock
{
	static NSObject *lock;
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		lock = [[NSObject alloc] init];
	});
	return lock;
}

+ (nullable Class)providerClassNamed:(NSString *)className
{
	Class candidate = NSClassFromString(className);
	if (candidate != Nil && [candidate conformsToProtocol:@protocol(NFKDynamicBackendProvider)]) {
		return candidate;
	}
	return Nil;
}

+ (BOOL)isProviderAvailable:(NSString *)className
{
	return [self providerClassNamed:className] != Nil;
}

+ (nullable id<NFKInferenceBackend>)backendForProviderClassName:(NSString *)className
														 error:(NSError * _Nullable * _Nullable)error
{
	Class provider = [self providerClassNamed:className];
	if (provider == Nil) {
		if (error != NULL) {
			*error = [NSError errorWithDomain:NFKInferenceErrorDomain
										 code:kNFKError_InferenceUnsupported
									 userInfo:@{ NSLocalizedDescriptionKey:
													 [NSString stringWithFormat:@"No backend provider named '%@' is linked.", className] }];
		}
		return nil;
	}
	id<NFKInferenceBackend> backend = [provider makeInferenceBackend];
	if (backend == nil && error != NULL) {
		*error = [NSError errorWithDomain:NFKInferenceErrorDomain
									 code:kNFKError_InferenceUnsupported
								 userInfo:@{ NSLocalizedDescriptionKey:
												 [NSString stringWithFormat:@"Provider '%@' returned no backend.", className] }];
	}
	return backend;
}

+ (void)registerProviderClassName:(NSString *)className forCapability:(NSString *)capability
{
	@synchronized (self.lock) {
		NSMutableArray<NSString *> *names = self.registry[capability];
		if (names == nil) {
			names = [NSMutableArray array];
			self.registry[capability] = names;
		}
		// Newest first, and de-duplicated so a re-registration moves the name to the front.
		[names removeObject:className];
		[names insertObject:className atIndex:0];
	}
}

+ (NSArray<NSString *> *)providerNamesForCapability:(NSString *)capability
{
	NSMutableArray<NSString *> *names;
	@synchronized (self.lock) {
		names = [self.registry[capability] mutableCopy] ?: [NSMutableArray array];
	}
	// A built-in capability has default providers, tried last and in their own order (a registered
	// override wins over all of them).
	for (NSString *builtIn in NFKBuiltInDefaultProviders()[capability]) {
		[names removeObject:builtIn];
		[names addObject:builtIn];
	}
	return names;
}

+ (BOOL)isCapabilityAvailable:(NSString *)capability
{
	for (NSString *className in [self providerNamesForCapability:capability]) {
		if ([self isProviderAvailable:className]) {
			return YES;
		}
	}
	return NO;
}

+ (nullable id<NFKInferenceBackend>)backendForCapability:(NSString *)capability
												  error:(NSError * _Nullable * _Nullable)error
{
	for (NSString *className in [self providerNamesForCapability:capability]) {
		if (![self isProviderAvailable:className]) {
			continue;
		}
		// A linked provider that cannot serve this machine returns nil, which is a pass rather than
		// a failure: the VideoToolbox engines answer only where the hardware carries the processor.
		id<NFKInferenceBackend> backend = [self backendForProviderClassName:className error:NULL];
		if (backend != nil) {
			return backend;
		}
	}
	if (error != NULL) {
		*error = [NSError errorWithDomain:NFKInferenceErrorDomain
									 code:kNFKError_InferenceUnsupported
								 userInfo:@{ NSLocalizedDescriptionKey:
												 [NSString stringWithFormat:@"No provider for capability '%@' answered on this machine.", capability] }];
	}
	return nil;
}

+ (nullable id<NFKInferenceBackend>)stableDiffusionBackendWithError:(NSError * _Nullable * _Nullable)error
{
	return [self backendForCapability:NFKCapabilityStableDiffusion error:error];
}

@end
