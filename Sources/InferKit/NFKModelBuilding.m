//
//  NFKModelBuilding.m
//  InferKit
//

#import "NFKModelBuilding.h"
#import "NFKModelDescriptorPrivate.h"
#import "NFKModelOptions.h"
#import "NFKModelPreset.h"
#import "NFKDynamicBackend.h"
#import "NFKHardwareProfile.h"
#import "NFKHFHub.h"

static NSError *NFKModelBuildError(NFKModelCatalogError code, NSString *reason)
{
	return [NSError errorWithDomain:NFKModelCatalogErrorDomain code:code userInfo:@{ NSLocalizedDescriptionKey: reason }];
}

#pragma mark - NFKModelBuildRequest

@implementation NFKModelBuildRequest

- (instancetype)initWithDescriptor:(NFKModelDescriptor *)descriptor
						   variant:(nullable NFKModelVariant *)variant
					   loadOptions:(nullable NSDictionary<NSString *, id> *)loadOptions
				   requestDefaults:(nullable NSDictionary<NSString *, id> *)requestDefaults
						  location:(nullable NSURL *)location
					memoryEstimate:(nullable NFKModelMemoryEstimate *)memoryEstimate
						   catalog:(NFKModelCatalog *)catalog
{
	self = [super init];
	if (self != nil) {
		_descriptor = descriptor;
		_variant = variant;
		_loadOptions = [loadOptions copy] ?: @{};
		_requestDefaults = [requestDefaults copy] ?: @{};
		_location = [location copy];
		_memoryEstimate = memoryEstimate;
		_catalog = catalog;
		NSMutableDictionary<NSString *, id> *arguments = [NSMutableDictionary dictionary];
		NSMutableDictionary<NSString *, id> *properties = [NSMutableDictionary dictionary];
		[_loadOptions enumerateKeysAndObjectsUsingBlock:^(NSString *key, id value, BOOL *stop) {
			NSString *keyPath = [descriptor optionWithKey:key].propertyKeyPath;
			if (keyPath != nil) {
				properties[key] = value;
			} else {
				arguments[key] = value;
			}
		}];
		_argumentOptions = [arguments copy];
		_propertyOptions = [properties copy];
	}
	return self;
}

- (nullable NSURL *)locationForModelIdentifier:(NSString *)identifier
									   variant:(nullable NSString *)variantIdentifier
										 error:(NSError **)error
{
	NFKModelDescriptor *descriptor = [self.catalog descriptorWithIdentifier:identifier];
	if (descriptor == nil) {
		if (error != NULL) {
			*error = NFKModelBuildError(NFKModelCatalogErrorUnknownModel,
										[NSString stringWithFormat:@"The catalog has no model \"%@\".", identifier]);
		}
		return nil;
	}
	NFKModelVariant *variant = variantIdentifier != nil ? [descriptor variantWithIdentifier:variantIdentifier]
		: descriptor.defaultVariant;
	return self.catalog.locator(descriptor, variant, error);
}

// Each key path's object answers the getter of every component but the last, and the setter of the last.
- (nullable NSString *)missingSetterOfKeyPath:(NSString *)keyPath onBackend:(id)backend
{
	NSArray<NSString *> *components = [keyPath componentsSeparatedByString:@"."];
	id target = backend;
	for (NSUInteger index = 0; index + 1 < components.count; index += 1) {
		if (![target respondsToSelector:NSSelectorFromString(components[index])]) {
			return components[index];
		}
		target = [target valueForKey:components[index]];
	}
	NSString *last = components.lastObject;
	if (last.length == 0) {
		return keyPath;
	}
	NSString *setter = [NSString stringWithFormat:@"set%@%@:", [[last substringToIndex:1] uppercaseString],
						[last substringFromIndex:1]];
	return [target respondsToSelector:NSSelectorFromString(setter)] ? nil : last;
}

- (BOOL)applyPropertyOptionsToBackend:(id)backend error:(NSError **)error
{
	NSArray<NSString *> *keys = [self.propertyOptions.allKeys sortedArrayUsingSelector:@selector(compare:)];
	for (NSString *key in keys) {
		NSString *keyPath = [self.descriptor optionWithKey:key].propertyKeyPath;
		NSString *missing = [self missingSetterOfKeyPath:keyPath onBackend:backend];
		if (missing != nil) {
			if (error != NULL) {
				NSString *reason = [NSString stringWithFormat:@"The %@ backend has no property \"%@\" to set \"%@\" on.",
									NSStringFromClass([backend class]), keyPath, key];
				*error = NFKModelBuildError(NFKModelCatalogErrorInvalidOptions, reason);
			}
			return NO;
		}
	}
	for (NSString *key in keys) {
		[backend setValue:self.propertyOptions[key] forKeyPath:[self.descriptor optionWithKey:key].propertyKeyPath];
	}
	return YES;
}

@end

#pragma mark - NFKDefaultedBackend

// The optional members whose availability follows the wrapped backend's.
static NSSet<NSString *> *NFKDefaultedForwardedSelectors(void)
{
	static NSSet<NSString *> *selectors;
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		selectors = [NSSet setWithArray:@[
			@"prepareWithError:", @"submitInferenceJobForRequest:", @"supportedParameterKeys", @"supportedInputKeys",
			@"supportedOutputKeys", @"modelInfo", @"backendStatus", @"tokenCountForRequest:error:", @"unload",
			@"isLoaded", @"residentByteCount", @"peakByteCount",
		]];
	});
	return selectors;
}

// The members a backend gains from the unload work, declared here so the wrapper compiles before and
// after the protocol declares them.
@protocol NFKDefaultedBackendLoading <NSObject>
@optional
- (void)unload;
@property (nonatomic, readonly) BOOL isLoaded;
@property (nonatomic, readonly) NSInteger residentByteCount;
@property (nonatomic, readonly) NSInteger peakByteCount;
@end

@interface NFKDefaultedBackend ()
@property (nonatomic, readonly, copy) NSDictionary<NSString *, id> *givenLoadOptionValues;
@end

@implementation NFKDefaultedBackend

- (instancetype)initWithBackend:(id<NFKInferenceBackend>)backend
				requestDefaults:(NSDictionary<NSString *, id> *)requestDefaults
			   loadOptionValues:(nullable NSDictionary<NSString *, id> *)loadOptionValues
{
	self = [super init];
	if (self != nil) {
		_backend = backend;
		_requestDefaults = [requestDefaults copy];
		_givenLoadOptionValues = [loadOptionValues copy] ?: @{};
	}
	return self;
}

- (BOOL)respondsToSelector:(SEL)selector
{
	if ([NFKDefaultedForwardedSelectors() containsObject:NSStringFromSelector(selector)]) {
		return [self.backend respondsToSelector:selector];
	}
	return [super respondsToSelector:selector] || [self.backend respondsToSelector:selector];
}

- (id)forwardingTargetForSelector:(SEL)selector
{
	return self.backend;
}

- (NFKInferenceRequest *)requestByFillingDefaults:(NFKInferenceRequest *)request
{
	if (self.requestDefaults.count == 0) {
		return request;
	}
	NSMutableDictionary<NSString *, id> *parameters = [self.requestDefaults mutableCopy];
	[parameters addEntriesFromDictionary:request.parameters];
	return [[NFKInferenceRequest alloc] initWithInputs:request.inputs parameters:parameters
										outputModality:request.outputModality];
}

- (BOOL)isReady
{
	return self.backend.isReady;
}

- (NSString *)backendIdentifier
{
	return self.backend.backendIdentifier;
}

- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request error:(NSError **)error
{
	return [self.backend runInferenceForRequest:[self requestByFillingDefaults:request] error:error];
}

- (BOOL)prepareWithError:(NSError **)error
{
	return NFKInferencePrepare(self.backend, error);
}

- (NFKInferenceJob *)submitInferenceJobForRequest:(NFKInferenceRequest *)request
{
	return NFKInferenceSubmit(self.backend, [self requestByFillingDefaults:request], NULL);
}

- (NSSet<NSString *> *)supportedParameterKeys
{
	return [self.backend respondsToSelector:@selector(supportedParameterKeys)] ? self.backend.supportedParameterKeys : [NSSet set];
}

- (NSSet<NSString *> *)supportedInputKeys
{
	return [self.backend respondsToSelector:@selector(supportedInputKeys)] ? self.backend.supportedInputKeys : [NSSet set];
}

- (NSSet<NSString *> *)supportedOutputKeys
{
	return [self.backend respondsToSelector:@selector(supportedOutputKeys)] ? self.backend.supportedOutputKeys : [NSSet set];
}

- (NSDictionary<NSString *, id> *)modelInfo
{
	return [self.backend respondsToSelector:@selector(modelInfo)] ? self.backend.modelInfo : @{};
}

- (NSDictionary<NSString *, id> *)backendStatus
{
	return [self.backend respondsToSelector:@selector(backendStatus)] ? self.backend.backendStatus : @{};
}

- (nullable NSNumber *)tokenCountForRequest:(NFKInferenceRequest *)request error:(NSError **)error
{
	if (![self.backend respondsToSelector:@selector(tokenCountForRequest:error:)]) {
		return nil;
	}
	return [self.backend tokenCountForRequest:[self requestByFillingDefaults:request] error:error];
}

- (NSDictionary<NSString *, id> *)loadOptionValues
{
	NSDictionary<NSString *, id> *reported = NFKInferenceLoadOptionValues(self.backend);
	return reported.count > 0 ? reported : self.givenLoadOptionValues;
}

- (id<NFKDefaultedBackendLoading>)loadingBackend
{
	return (id<NFKDefaultedBackendLoading>)self.backend;
}

- (void)unload
{
	if ([self.loadingBackend respondsToSelector:@selector(unload)]) {
		[self.loadingBackend unload];
	}
}

- (BOOL)isLoaded
{
	if ([self.loadingBackend respondsToSelector:@selector(isLoaded)]) {
		return self.loadingBackend.isLoaded;
	}
	return self.backend.isReady;
}

- (NSInteger)residentByteCount
{
	return [self.loadingBackend respondsToSelector:@selector(residentByteCount)] ? self.loadingBackend.residentByteCount : 0;
}

- (NSInteger)peakByteCount
{
	return [self.loadingBackend respondsToSelector:@selector(peakByteCount)] ? self.loadingBackend.peakByteCount : 0;
}

- (NSString *)description
{
	return [NSString stringWithFormat:@"<%@ %@ defaults %@>", NSStringFromClass(self.class), self.backend,
			self.requestDefaults];
}

@end

#pragma mark - Building

@implementation NFKModelCatalog (NFKModelBuilding)

+ (NSMutableDictionary<NSString *, id<NFKModelBackendBuilding>> *)backendBuilders
{
	static NSMutableDictionary<NSString *, id<NFKModelBackendBuilding>> *builders;
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		builders = [NSMutableDictionary dictionary];
	});
	return builders;
}

+ (void)registerBackendBuilder:(nullable id<NFKModelBackendBuilding>)builder forEngine:(NSString *)engine
{
	NSMutableDictionary<NSString *, id<NFKModelBackendBuilding>> *builders = [self backendBuilders];
	@synchronized (builders) {
		builders[engine] = builder;
	}
}

+ (nullable id<NFKModelBackendBuilding>)backendBuilderForEngine:(NSString *)engine
{
	NSMutableDictionary<NSString *, id<NFKModelBackendBuilding>> *builders = [self backendBuilders];
	@synchronized (builders) {
		return builders[engine];
	}
}

// Whether a source's files come from a download the hub cache holds: a Hugging Face repository or a URL,
// loaded as files. A system or none source, and a layout of none, build without files.
static BOOL NFKModelSourceDownloadsFiles(NFKModelSource * _Nullable source)
{
	if (source == nil || [source.layout isEqualToString:NFKModelSourceLayoutNone]) {
		return NO;
	}
	BOOL fromHub = [source.kind isEqualToString:NFKModelSourceKindHuggingFace] && source.repository.length > 0;
	BOOL fromURL = [source.kind isEqualToString:NFKModelSourceKindURL] && source.URL != nil;
	return fromHub || fromURL;
}

+ (NFKModelLocator)hubCacheLocator
{
	return ^NSURL * _Nullable (NFKModelDescriptor *descriptor, NFKModelVariant * _Nullable variant, NSError **error) {
		return [NFKModelCatalog hubCacheLocatorForCacheDirectoryURL:[NFKHFHub defaultCacheDirectoryURL]](descriptor, variant,
																										error);
	};
}

+ (NFKModelLocator)hubCacheLocatorForCacheDirectoryURL:(NSURL *)cacheDirectoryURL
{
	return ^NSURL * _Nullable (NFKModelDescriptor *descriptor, NFKModelVariant * _Nullable variant, NSError **error) {
		NFKModelSource *source = variant.source;
		if (!NFKModelSourceDownloadsFiles(source)) {
			return nil;
		}
		NFKHFHub *hub = [NFKHFHub hubWithCacheDirectoryURL:cacheDirectoryURL];
		NFKDownloadStatus *status = [hub downloadStatusForModelSource:source];
		if (status.presence != NFKDownloadPresenceComplete) {
			return nil;
		}
		if ([source.kind isEqualToString:NFKModelSourceKindURL]) {
			return status.localURL;
		}
		NFKModelFile *file = nil;
		for (NFKModelFile *candidate in source.files) {
			if (candidate.overridesSource || candidate.isOptional) {
				continue;
			}
			if (file == nil || ([candidate.role isEqualToString:NFKModelFileRoleWeights]
								&& ![file.role isEqualToString:NFKModelFileRoleWeights])) {
				file = candidate;
			}
		}
		NSString *path = file.path ?: @"config.json";
		NSURL *fileURL = [hub localURLForRepo:source.repository revision:source.revision path:path];
		if (fileURL == nil) {
			return nil;
		}
		if ([source.layout isEqualToString:NFKModelSourceLayoutFile]) {
			return [NSFileManager.defaultManager fileExistsAtPath:fileURL.path] ? fileURL : nil;
		}
		NSURL *directory = fileURL;
		for (NSUInteger index = 0; index < path.pathComponents.count; index += 1) {
			directory = directory.URLByDeletingLastPathComponent;
		}
		BOOL isDirectory = NO;
		BOOL exists = [NSFileManager.defaultManager fileExistsAtPath:directory.path isDirectory:&isDirectory];
		return exists && isDirectory ? directory : nil;
	};
}

- (NFKModelLocator)locator
{
	return self.storedLocator ?: [NFKModelCatalog hubCacheLocator];
}

- (void)setLocator:(nullable NFKModelLocator)locator
{
	self.storedLocator = locator;
}

- (NFKModelMemoryBudget *)memoryBudget
{
	return self.storedMemoryBudget ?: [NFKModelMemoryBudget currentBudget];
}

- (void)setMemoryBudget:(nullable NFKModelMemoryBudget *)memoryBudget
{
	self.storedMemoryBudget = memoryBudget;
}

// The descriptor a name reaches, and the preset when the name is model@preset.
- (nullable NFKModelDescriptor *)descriptorForName:(NSString *)name preset:(NFKModelPreset * _Nullable * _Nonnull)preset
{
	*preset = nil;
	NFKModelDescriptor *descriptor = [self descriptorWithIdentifier:name];
	if (descriptor != nil) {
		return descriptor;
	}
	NSString *modelName = nil, *presetName = nil;
	BOOL split = NFKModelNameSplitAtPreset(name, ^BOOL(NSString *model, NSString *candidate) {
		return [[self descriptorWithIdentifier:model] presetWithIdentifier:candidate] != nil;
	}, &modelName, &presetName);
	if (!split) {
		return nil;
	}
	descriptor = [self descriptorWithIdentifier:modelName];
	*preset = [descriptor presetWithIdentifier:presetName];
	return descriptor;
}

- (nullable NFKModelBuildRequest *)buildRequestForModelIdentifier:(NSString *)identifier
														   options:(nullable NSDictionary<NSString *, id> *)options
														  location:(nullable NSURL *)location
															 error:(NSError **)error
{
	NFKModelPreset *preset = nil;
	NFKModelDescriptor *descriptor = [self descriptorForName:identifier preset:&preset];
	if (descriptor == nil) {
		if (error != NULL) {
			*error = NFKModelBuildError(NFKModelCatalogErrorUnknownModel,
										[NSString stringWithFormat:@"The catalog has no model \"%@\".", identifier]);
		}
		return nil;
	}
	NSMutableDictionary<NSString *, id> *given = [NSMutableDictionary dictionary];
	[given addEntriesFromDictionary:preset.loadOptions];
	[given addEntriesFromDictionary:preset.requestDefaults];
	[given addEntriesFromDictionary:options ?: @{}];
	NFKHardwareProfile *profile = NFKHardwareProfile.currentProfile;
	NSDictionary<NSString *, id> *resolved = [self resolveOptions:given descriptor:descriptor profile:profile error:error];
	if (resolved == nil) {
		return nil;
	}
	NSString *problem = [descriptor.requirements problemForHardwareProfile:profile];
	if (problem != nil) {
		if (error != NULL) {
			*error = NFKModelBuildError(NFKModelCatalogErrorUnavailable,
										[NSString stringWithFormat:@"%@ cannot run on this machine. %@",
										 descriptor.displayName, problem]);
		}
		return nil;
	}
	NSMutableDictionary<NSString *, id> *loadOptions = [NSMutableDictionary dictionary];
	NSMutableDictionary<NSString *, id> *requestDefaults = [NSMutableDictionary dictionary];
	NSSet<NSString *> *overridable = descriptor.requestOverridableOptionKeys;
	[resolved enumerateKeysAndObjectsUsingBlock:^(NSString *key, id value, BOOL *stop) {
		NFKModelOption *option = [descriptor optionWithKey:key];
		if ([option.stage isEqualToString:NFKModelOptionStageLoad]) {
			loadOptions[key] = value;
			if ([overridable containsObject:key]) {
				requestDefaults[key] = value;
			}
		} else if (given[key] != nil) {
			requestDefaults[key] = value;
		}
	}];
	NFKModelVariant *variant = [descriptor variantWithIdentifier:[loadOptions[NFKModelLoadOptionVariant] description] ?: @""]
		?: descriptor.defaultVariant;
	NFKModelMemoryEstimate *estimate = [descriptor estimateForResolvedOptions:loadOptions memoryBudget:self.memoryBudget
																	  catalog:self];
	BOOL ignoresBudget = [given[NFKModelLoadOptionIgnoreMemoryBudget] respondsToSelector:@selector(boolValue)]
		&& [given[NFKModelLoadOptionIgnoreMemoryBudget] boolValue];
	if (!estimate.fits && !ignoresBudget) {
		if (error != NULL) {
			*error = [self doesNotFitErrorFor:descriptor estimate:estimate loadOptions:loadOptions profile:profile];
		}
		return nil;
	}
	NSURL *located = location;
	if (located == nil) {
		NSError *locateError = nil;
		located = self.locator(descriptor, variant, &locateError);
		if (located == nil && locateError != nil) {
			if (error != NULL) {
				*error = locateError;
			}
			return nil;
		}
	}
	if (located == nil && NFKModelSourceDownloadsFiles(variant.source)) {
		if (error != NULL) {
			*error = NFKModelBuildError(NFKModelCatalogErrorNotDownloaded,
										[NSString stringWithFormat:@"%@'s files are not downloaded. Download them with "
										 @"-[NFKHFHub downloadModelSource:progressHandler:completionHandler:], or pass "
										 @"their location.", descriptor.displayName ?: descriptor.identifier]);
		}
		return nil;
	}
	return [[NFKModelBuildRequest alloc] initWithDescriptor:descriptor variant:variant loadOptions:loadOptions
											requestDefaults:requestDefaults location:located memoryEstimate:estimate
													catalog:self];
}

// The refusal of a combination that does not fit, with the fitting combination nearest the requested one
// as the way out, or the smallest combination's need when nothing fits.
- (NSError *)doesNotFitErrorFor:(NFKModelDescriptor *)descriptor
					   estimate:(NFKModelMemoryEstimate *)estimate
					loadOptions:(NSDictionary<NSString *, id> *)loadOptions
						profile:(NFKHardwareProfile *)profile
{
	NFKModelMemoryBudget *budget = estimate.budget ?: self.memoryBudget;
	NFKModelMemoryConstraints *constraints = [descriptor constraintsForOptions:nil memoryBudget:budget
																	   profile:profile catalog:self
																	 preferred:loadOptions];
	NSMutableDictionary<NSString *, id> *userInfo = [NSMutableDictionary dictionary];
	userInfo[NFKModelMemoryEstimateKey] = estimate;
	NSString *description = estimate.reason ?: @"The model does not fit.";
	if (constraints.fittingOptions != nil && constraints.fittingEstimate != nil) {
		userInfo[NFKModelMemoryFittingOptionsKey] = constraints.fittingOptions;
		userInfo[NFKModelMemoryFittingEstimateKey] = constraints.fittingEstimate;
		NSString *changes = [self changesFrom:loadOptions to:constraints.fittingOptions];
		if (changes.length > 0) {
			description = [NSString stringWithFormat:@"%@ Choosing %@ fits: it holds %@.", description, changes,
						   [NFKModelMemoryBudget formattedByteCount:constraints.fittingEstimate.peakByteCount]];
		}
	} else if (constraints.reason != nil) {
		description = [NSString stringWithFormat:@"%@ %@", description, constraints.reason];
	}
	userInfo[NSLocalizedDescriptionKey] = description;
	return [NSError errorWithDomain:NFKModelCatalogErrorDomain code:NFKModelCatalogErrorDoesNotFit userInfo:userInfo];
}

// The options that differ, as "key value" pairs in key order: "residency paged, contextLength 4096".
- (NSString *)changesFrom:(NSDictionary<NSString *, id> *)requested to:(NSDictionary<NSString *, id> *)fitting
{
	NSMutableSet<NSString *> *keys = [NSMutableSet setWithArray:requested.allKeys];
	[keys addObjectsFromArray:fitting.allKeys];
	NSMutableArray<NSString *> *changes = [NSMutableArray array];
	for (NSString *key in [keys.allObjects sortedArrayUsingSelector:@selector(compare:)]) {
		NSString *wanted = [requested[key] description] ?: @"none";
		NSString *fits = [fitting[key] description] ?: @"none";
		if (![wanted isEqualToString:fits]) {
			[changes addObject:[NSString stringWithFormat:@"%@ %@", key, fits]];
		}
	}
	return [changes componentsJoinedByString:@", "];
}

// The options resolved at every stage. A problem other than an unavailable choice is InvalidOptions; a set
// of problems that are all unavailable choices is Unavailable.
- (nullable NSDictionary<NSString *, id> *)resolveOptions:(NSDictionary<NSString *, id> *)given
											   descriptor:(NFKModelDescriptor *)descriptor
												  profile:(NFKHardwareProfile *)profile
													error:(NSError **)error
{
	NSError *resolveError = nil;
	NSDictionary<NSString *, id> *resolved = [descriptor resolvedOptions:given stage:nil catalog:self profile:profile
																   error:&resolveError];
	if (resolved != nil) {
		return resolved;
	}
	NSArray<NFKModelOptionProblem *> *problems = resolveError.userInfo[NFKModelOptionProblemsKey];
	BOOL allUnavailable = problems.count > 0;
	for (NFKModelOptionProblem *problem in problems) {
		allUnavailable = allUnavailable && [problem.code isEqualToString:NFKModelOptionProblemUnavailable];
	}
	if (error != NULL) {
		*error = allUnavailable
			? [NSError errorWithDomain:NFKModelCatalogErrorDomain code:NFKModelCatalogErrorUnavailable
							  userInfo:resolveError.userInfo]
			: resolveError;
	}
	return nil;
}

- (nullable id<NFKInferenceBackend>)backendForModelIdentifier:(NSString *)identifier
													  options:(nullable NSDictionary<NSString *, id> *)options
														error:(NSError **)error
{
	return [self backendForModelIdentifier:identifier options:options location:nil error:error];
}

- (nullable id<NFKInferenceBackend>)backendForModelIdentifier:(NSString *)identifier
													  options:(nullable NSDictionary<NSString *, id> *)options
													 location:(nullable NSURL *)location
														error:(NSError **)error
{
	NFKModelBuildRequest *request = [self buildRequestForModelIdentifier:identifier options:options location:location
																   error:error];
	if (request == nil) {
		return nil;
	}
	id<NFKInferenceBackend> backend = [self buildBackendForRequest:request error:error];
	if (backend == nil) {
		return nil;
	}
	if (request.requestDefaults.count == 0) {
		return backend;
	}
	return [[NFKDefaultedBackend alloc] initWithBackend:backend requestDefaults:request.requestDefaults
									   loadOptionValues:request.loadOptions];
}

- (nullable id<NFKInferenceBackend>)buildBackendForRequest:(NFKModelBuildRequest *)request error:(NSError **)error
{
	NFKModelDescriptor *descriptor = request.descriptor;
	NSString *providerName = [self providerClassNameForDescriptorIdentifier:descriptor.identifier];
	Class provider = providerName != nil ? NSClassFromString(providerName) : Nil;
	if ([provider respondsToSelector:@selector(backendForBuildRequest:error:)]) {
		return [(Class<NFKModelDescriptorProvider>)provider backendForBuildRequest:request error:error];
	}
	id<NFKModelBackendBuilding> builder = [NFKModelCatalog backendBuilderForEngine:descriptor.engine];
	if (builder != nil) {
		return [builder backendForBuildRequest:request error:error];
	}
	if (descriptor.providerClassName == nil) {
		if (error != NULL) {
			*error = NFKModelBuildError(NFKModelCatalogErrorNoBuilder,
										[NSString stringWithFormat:@"No builder builds models of the engine \"%@\".",
										 descriptor.engine]);
		}
		return nil;
	}
	id<NFKInferenceBackend> backend = [NFKDynamicBackend backendForProviderClassName:descriptor.providerClassName
																			   error:error];
	if (backend == nil || ![request applyPropertyOptionsToBackend:backend error:error]) {
		return nil;
	}
	return backend;
}

@end
