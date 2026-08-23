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

// The default provider class name tried for each built-in capability when the consumer registers
// nothing. A consumer that names their provider this needs no registration call; a companion package
// (InferKitMLX, InferKitFoundationModels) ships the class so linking it activates the capability.
static NSDictionary<NSString *, NSString *> *NFKBuiltInDefaultProviders(void)
{
	static NSDictionary *providers;
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		providers = @{
			NFKCapabilityStableDiffusion: @"NFKStableDiffusionProvider",
			NFKCapabilityTextGeneration:  @"NFKFoundationModelsProvider",
			NFKCapabilityTranscription:   @"NFKMLXWhisperProvider",
			NFKCapabilityControlNet:      @"NFKControlNetProvider",
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
	// A built-in capability has a default provider, tried last (a registered override wins).
	NSString *builtIn = NFKBuiltInDefaultProviders()[capability];
	if (builtIn != nil) {
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
		if ([self isProviderAvailable:className]) {
			return [self backendForProviderClassName:className error:error];
		}
	}
	if (error != NULL) {
		*error = [NSError errorWithDomain:NFKInferenceErrorDomain
									 code:kNFKError_InferenceUnsupported
								 userInfo:@{ NSLocalizedDescriptionKey:
												 [NSString stringWithFormat:@"No provider for capability '%@' is linked.", capability] }];
	}
	return nil;
}

+ (nullable id<NFKInferenceBackend>)stableDiffusionBackendWithError:(NSError * _Nullable * _Nullable)error
{
	return [self backendForCapability:NFKCapabilityStableDiffusion error:error];
}

@end
