//
//  NFKModelCatalog.m
//  InferKit
//

#import "NFKModelCatalog.h"
#import "NFKModelDescriptorPrivate.h"

NSString * const NFKModelCatalogErrorDomain = @"NFKModelCatalogErrorDomain";
NSString * const NFKModelCatalogManifestKind = @"inferkit.model-catalog";

static const NSInteger NFKModelCatalogSchemaVersion = 1;

@interface NFKModelCatalog ()
// identifier -> descriptor. Guarded by @synchronized on itself, as is providerNames.
@property (nonatomic, readonly) NSMutableDictionary<NSString *, NFKModelDescriptor *> *storage;
// identifier -> the provider class that supplied the descriptor.
@property (nonatomic, readonly) NSMutableDictionary<NSString *, NSString *> *providerNames;
@end

@implementation NFKModelCatalog

#pragma mark Provider registry

// Guards the provider names and the shared catalog's publication.
+ (NSObject *)registryLock
{
	static NSObject *lock;
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		lock = [[NSObject alloc] init];
	});
	return lock;
}

+ (NSMutableArray<NSString *> *)registeredProviderClassNames
{
	static NSMutableArray<NSString *> *names;
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		names = [@[ @"NFKAppleModelDescriptors", @"NFKMLXModelDescriptors", @"NFKFoundationModelsDescriptors",
					@"NFKAppleSwiftModelDescriptors" ] mutableCopy];
	});
	return names;
}

// Set once, under the registry lock, before the shared catalog loads its providers, so a provider that
// registers another provider while it loads reaches the catalog without waiting on its creation.
static NFKModelCatalog *NFKSharedModelCatalog;

+ (NFKModelCatalog *)sharedCatalog
{
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		NFKModelCatalog *catalog = [[NFKModelCatalog alloc] init];
		NSArray<NSString *> *names = nil;
		@synchronized ([self registryLock]) {
			names = [[self registeredProviderClassNames] copy];
			NFKSharedModelCatalog = catalog;
		}
		for (NSString *name in names) {
			[catalog addDescriptorsFromProviderClassName:name];
		}
	});
	return NFKSharedModelCatalog;
}

+ (NSInteger)schemaVersion
{
	return NFKModelCatalogSchemaVersion;
}

+ (NSArray<NSString *> *)providerClassNames
{
	@synchronized ([self registryLock]) {
		return [[self registeredProviderClassNames] copy];
	}
}

+ (void)registerProviderClassName:(NSString *)className
{
	NFKModelCatalog *shared = nil;
	@synchronized ([self registryLock]) {
		NSMutableArray<NSString *> *names = [self registeredProviderClassNames];
		if (![names containsObject:className]) {
			[names addObject:className];
		}
		shared = NFKSharedModelCatalog;
	}
	if (shared != nil) {
		[shared addDescriptorsFromProviderClassName:className];
	}
}

#pragma mark Instance

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_storage = [NSMutableDictionary dictionary];
		_providerNames = [NSMutableDictionary dictionary];
	}
	return self;
}

- (NSArray<NFKModelDescriptor *> *)descriptors
{
	NSArray<NFKModelDescriptor *> *all = nil;
	@synchronized (self.storage) {
		all = self.storage.allValues;
	}
	return [all sortedArrayUsingComparator:^NSComparisonResult(NFKModelDescriptor *first, NFKModelDescriptor *second) {
		return [first.identifier compare:second.identifier];
	}];
}

- (NSArray<NFKModelDescriptor *> *)visibleDescriptors
{
	return [self descriptorsMatching:^BOOL(NFKModelDescriptor *descriptor) {
		return !descriptor.isStandIn;
	}];
}

- (nullable NFKModelDescriptor *)descriptorWithIdentifier:(NSString *)identifier
{
	@synchronized (self.storage) {
		return self.storage[identifier];
	}
}

- (NSArray<NFKModelDescriptor *> *)descriptorsForTask:(NSString *)task
{
	return [self descriptorsMatching:^BOOL(NFKModelDescriptor *descriptor) {
		return [descriptor.tasks containsObject:task];
	}];
}

- (NSArray<NFKModelDescriptor *> *)descriptorsInFamily:(NSString *)family
{
	return [self descriptorsMatching:^BOOL(NFKModelDescriptor *descriptor) {
		return [descriptor.family isEqualToString:family];
	}];
}

- (NSArray<NFKModelDescriptor *> *)descriptorsMatching:(BOOL (^)(NFKModelDescriptor *descriptor))predicate
{
	NSMutableArray<NFKModelDescriptor *> *matching = [NSMutableArray array];
	for (NFKModelDescriptor *descriptor in self.descriptors) {
		if (predicate(descriptor)) {
			[matching addObject:descriptor];
		}
	}
	return [matching copy];
}

- (void)addDescriptor:(NFKModelDescriptor *)descriptor
{
	[self addDescriptors:@[ descriptor ]];
}

- (void)addDescriptors:(NSArray<NFKModelDescriptor *> *)descriptors
{
	[self addDescriptors:descriptors providerClassName:nil];
}

- (void)addDescriptors:(NSArray<NFKModelDescriptor *> *)descriptors providerClassName:(nullable NSString *)className
{
	@synchronized (self.storage) {
		for (NFKModelDescriptor *descriptor in descriptors) {
			self.storage[descriptor.identifier] = descriptor;
			self.providerNames[descriptor.identifier] = className;
		}
	}
}

- (void)removeDescriptorWithIdentifier:(NSString *)identifier
{
	@synchronized (self.storage) {
		[self.storage removeObjectForKey:identifier];
		[self.providerNames removeObjectForKey:identifier];
	}
}

- (nullable NSString *)providerClassNameForDescriptorIdentifier:(NSString *)identifier
{
	@synchronized (self.storage) {
		return self.providerNames[identifier];
	}
}

- (BOOL)addDescriptorsFromProviderClassName:(NSString *)className
{
	Class provider = NSClassFromString(className);
	if (provider == Nil || ![provider conformsToProtocol:@protocol(NFKModelDescriptorProvider)]) {
		return NO;
	}
	NSArray<NFKModelDescriptor *> *descriptors = [(Class<NFKModelDescriptorProvider>)provider modelDescriptors];
	[self addDescriptors:descriptors ?: @[] providerClassName:className];
	return YES;
}

#pragma mark Manifests

// A manifest that does not parse, with the parser's own account as the underlying error.
static BOOL NFKModelCatalogFailParsing(NSError * _Nullable parseError, NSError **error)
{
	if (error == NULL) {
		return NO;
	}
	NSString *detail = parseError.userInfo[NSDebugDescriptionErrorKey] ?: parseError.localizedDescription;
	NSMutableDictionary<NSString *, id> *userInfo = [NSMutableDictionary dictionary];
	userInfo[NSLocalizedDescriptionKey] = detail != nil ? [NSString stringWithFormat:@"The manifest is not valid JSON: %@", detail]
		: @"The manifest is not valid JSON.";
	userInfo[NSUnderlyingErrorKey] = parseError;
	*error = [NSError errorWithDomain:NFKModelCatalogErrorDomain code:NFKModelCatalogErrorMalformedManifest
							 userInfo:userInfo];
	return NO;
}

- (BOOL)addDescriptorsFromManifestData:(NSData *)data error:(NSError **)error
{
	NSError *parseError = nil;
	id manifest = [NSJSONSerialization JSONObjectWithData:data options:0 error:&parseError];
	if (manifest == nil) {
		return NFKModelCatalogFailParsing(parseError, error);
	}
	if (![manifest isKindOfClass:NSDictionary.class]) {
		return NFKModelCatalogFail(NFKModelCatalogErrorMalformedManifest, @"The manifest is not a JSON object.", error);
	}
	if (![self checkManifestHeader:manifest error:error]) {
		return NO;
	}
	id models = manifest[@"models"];
	if (![models isKindOfClass:NSArray.class]) {
		return NFKModelCatalogFail(NFKModelCatalogErrorMalformedManifest, @"The manifest's models is not an array.",
								   error);
	}
	NSMutableArray<NFKModelDescriptor *> *descriptors = [NSMutableArray arrayWithCapacity:[models count]];
	NSMutableDictionary<NSString *, NSNumber *> *indexByIdentifier = [NSMutableDictionary dictionary];
	NSUInteger index = 0;
	for (id model in models) {
		NSString *path = [NSString stringWithFormat:@"models[%lu]", (unsigned long)index];
		NFKModelDescriptor *descriptor = [NFKModelDescriptor descriptorFromObject:model path:path error:error];
		if (descriptor == nil) {
			return NO;
		}
		NSNumber *earlier = indexByIdentifier[descriptor.identifier];
		if (earlier != nil) {
			NSString *reason = [NSString stringWithFormat:@"%@.id \"%@\" repeats models[%lu].id; each model's id is unique within a manifest.",
								path, descriptor.identifier, (unsigned long)earlier.unsignedIntegerValue];
			return NFKModelCatalogFail(NFKModelCatalogErrorMalformedManifest, reason, error);
		}
		indexByIdentifier[descriptor.identifier] = @(index);
		[descriptors addObject:descriptor];
		index += 1;
	}
	[self addDescriptors:descriptors];
	return YES;
}

- (BOOL)checkManifestHeader:(NSDictionary<NSString *, id> *)manifest error:(NSError **)error
{
	id version = manifest[@"schemaVersion"];
	if (![version isKindOfClass:NSNumber.class] || [version integerValue] < 1) {
		return NFKModelCatalogFail(NFKModelCatalogErrorMalformedManifest,
								   @"The manifest's schemaVersion is missing or not a positive integer.", error);
	}
	if ([version integerValue] > NFKModelCatalogSchemaVersion) {
		NSString *reason = [NSString stringWithFormat:@"The manifest's schemaVersion %ld is newer than %ld, the newest this release reads.",
							(long)[version integerValue], (long)NFKModelCatalogSchemaVersion];
		return NFKModelCatalogFail(NFKModelCatalogErrorUnsupportedSchemaVersion, reason, error);
	}
	if (![manifest[@"kind"] isEqual:NFKModelCatalogManifestKind]) {
		NSString *reason = [NSString stringWithFormat:@"The manifest's kind is not \"%@\".", NFKModelCatalogManifestKind];
		return NFKModelCatalogFail(NFKModelCatalogErrorMalformedManifest, reason, error);
	}
	return YES;
}

- (nullable NSData *)manifestDataWithError:(NSError **)error
{
	NSMutableArray<NSDictionary<NSString *, id> *> *models = [NSMutableArray array];
	for (NFKModelDescriptor *descriptor in self.descriptors) {
		NSDictionary<NSString *, id> *model = descriptor.dictionaryRepresentation;
		if (![NSJSONSerialization isValidJSONObject:model]) {
			NSString *reason = [NSString stringWithFormat:@"The descriptor %@ holds a value JSON cannot carry.",
								descriptor.identifier];
			NFKModelCatalogFail(NFKModelCatalogErrorMalformedDescriptor, reason, error);
			return nil;
		}
		[models addObject:model];
	}
	NSDictionary<NSString *, id> *manifest = @{ @"schemaVersion": @(NFKModelCatalogSchemaVersion),
												 @"kind": NFKModelCatalogManifestKind,
												 @"models": models };
	NSJSONWritingOptions options = NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys
		| NSJSONWritingWithoutEscapingSlashes;
	return [NSJSONSerialization dataWithJSONObject:manifest options:options error:error];
}

@end
