//
//  NFKPluginRegistry.m
//  InferKit
//

#import "NFKPluginRegistry.h"
#import "NFKPluginBackend.h"
#import "NFKPluginWire.h"

#if TARGET_OS_OSX

#import <dlfcn.h>
#import "NFKDynamicBackend.h"
#import "NFKModelBuilding.h"
#import "NFKModelDescriptorPrivate.h"
#import "NFKModelOptions.h"
#import "NFKModelVocabulary.h"
#import "NFKPluginLoaderPrivate.h"
#import "NFKPluginManifestPrivate.h"
#import "NFKPluginProcessPrivate.h"

NSString * const NFKPluginInProcessWarningNoUnload =
	@"A plugin that runs in the app's process stays loaded until the app quits.";
NSString * const NFKPluginInProcessWarningCrashTakesDownApp =
	@"A crash in a plugin that runs in the app's process ends the app.";
NSString * const NFKPluginInProcessWarningFullAccess =
	@"A plugin that runs in the app's process has every access the app has.";

/*! The plugin host's file name. */
static NSString * const NFKPluginHostExecutableName = @"inferkit-plugin-host";

/*! The descriptor extension that names the plugin serving a catalog model. */
static NSString * const NFKPluginDescriptorExtensionKey = @"plugin";

/*! The executable file at url, or nil. */
static NSURL * _Nullable NFKPluginExecutableFile(NSURL * _Nullable url)
{
	if (url == nil) {
		return nil;
	}
	BOOL isDirectory = NO;
	BOOL exists = [NSFileManager.defaultManager fileExistsAtPath:url.path isDirectory:&isDirectory];
	return exists && !isDirectory && [NSFileManager.defaultManager isExecutableFileAtPath:url.path] ? url : nil;
}

/*! The path with symbolic links resolved, so two spellings of one bundle compare equal. */
static NSString *NFKPluginRegistryResolvedPath(NSURL *url)
{
	char resolved[PATH_MAX];
	return realpath(url.fileSystemRepresentation, resolved) != NULL ? [NSString stringWithUTF8String:resolved]
		: url.URLByStandardizingPath.path;
}

/*! The catalog identifier of one of a plugin's models. */
static NSString *NFKPluginCatalogIdentifier(NSString *pluginIdentifier, NSString *modelIdentifier)
{
	return [NSString stringWithFormat:@"%@/%@", pluginIdentifier, modelIdentifier];
}

/*! Whether a plugin may run in the app's process: its manifest allows it and it is not a program of
	its own. */
static BOOL NFKPluginRunsInProcess(NFKPluginManifest *manifest)
{
	return manifest.allowsInProcess && ![manifest.entryKind isEqualToString:NFKPluginEntryKindExecutable];
}

#pragma mark - Entry

/*! One plugin the registry lists. Read and written under the registry's lock. */
@interface NFKPluginRegistryEntry : NSObject
@property (nonatomic, copy) NFKPluginManifest *manifest;
@property (nonatomic, copy) NSArray<NSString *> *catalogIdentifiers;
@property (nonatomic, strong, nullable) NFKPluginProcess *process;
@property (nonatomic, strong, nullable) NFKLoadedPlugin *loadedPlugin;
@end

@implementation NFKPluginRegistryEntry
@end

#pragma mark - Engine builder

/*! Builds the catalog's "plugin" models through the registry that lists them. A "plugin" descriptor no
	registry lists, such as one derived from an app's own backend, builds from its provider class as the
	catalog builds any descriptor without a builder. */
@interface NFKPluginEngineBuilder : NSObject <NFKModelBackendBuilding>
@end

@interface NFKPluginRegistry ()
@property (nonatomic, readonly) NSObject *lock;
@property (nonatomic, readonly) NSMutableDictionary<NSString *, NFKPluginRegistryEntry *> *entries;
@property (nonatomic, readonly) NSMutableDictionary<NSString *, NSString *> *pluginIdentifiersByCatalogIdentifier;
@property (nonatomic, readonly) NSMutableSet<NSString *> *trustedIdentifiers;
@property (nonatomic, readonly) NSMutableDictionary<NSString *, NSDictionary<NSString *, id> *> *optionValuesByPlugin;
@property (nonatomic, copy) NFKPluginSignaturePolicy *storedSignaturePolicy;
@property (nonatomic, copy, nullable) NSString *storedSharedMemoryPrefix;
@property (nonatomic, copy, nullable) NSURL *assignedHostExecutableURL;
+ (nullable NFKPluginRegistry *)registryListingModelIdentifier:(NSString *)catalogIdentifier
													   catalog:(NFKModelCatalog *)catalog;
@end

@implementation NFKPluginEngineBuilder

- (nullable id<NFKInferenceBackend>)backendForBuildRequest:(NFKModelBuildRequest *)request error:(NSError **)error
{
	NFKModelDescriptor *descriptor = request.descriptor;
	NFKPluginRegistry *registry = [NFKPluginRegistry registryListingModelIdentifier:descriptor.identifier
																			catalog:request.catalog];
	if (registry != nil) {
		return [registry backendForModelIdentifier:descriptor.identifier options:request.loadOptions error:error];
	}
	if (descriptor.providerClassName == nil) {
		NFKModelCatalogFail(NFKModelCatalogErrorNoBuilder,
							[NSString stringWithFormat:@"No builder builds models of the engine \"%@\".", descriptor.engine],
							error);
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

#pragma mark - Registry

@implementation NFKPluginRegistry

+ (NFKPluginRegistry *)sharedRegistry
{
	static NFKPluginRegistry *shared = nil;
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		shared = [[NFKPluginRegistry alloc] initWithCatalog:NFKModelCatalog.sharedCatalog];
	});
	return shared;
}

/*! Every registry alive, held weakly, which the engine builder searches. Guarded by @synchronized on
	itself. */
+ (NSHashTable<NFKPluginRegistry *> *)liveRegistries
{
	static NSHashTable<NFKPluginRegistry *> *registries = nil;
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		registries = [NSHashTable weakObjectsHashTable];
		[NFKModelCatalog registerBackendBuilder:[[NFKPluginEngineBuilder alloc] init] forEngine:NFKModelEnginePlugin];
	});
	return registries;
}

+ (nullable NFKPluginRegistry *)registryListingModelIdentifier:(NSString *)catalogIdentifier catalog:(NFKModelCatalog *)catalog
{
	NSArray<NFKPluginRegistry *> *registries = nil;
	NSHashTable<NFKPluginRegistry *> *live = [self liveRegistries];
	@synchronized (live) {
		registries = live.allObjects;
	}
	for (NFKPluginRegistry *registry in registries) {
		if (registry.catalog == catalog && [registry manifestForModelIdentifier:catalogIdentifier] != nil) {
			return registry;
		}
	}
	return nil;
}

- (instancetype)initWithCatalog:(NFKModelCatalog *)catalog
{
	self = [super init];
	if (self != nil) {
		_catalog = catalog;
		_lock = [[NSObject alloc] init];
		_entries = [NSMutableDictionary dictionary];
		_pluginIdentifiersByCatalogIdentifier = [NSMutableDictionary dictionary];
		_trustedIdentifiers = [NSMutableSet set];
		_optionValuesByPlugin = [NSMutableDictionary dictionary];
		_storedSignaturePolicy = [NFKPluginSignaturePolicy defaultPolicy];
		NSHashTable<NFKPluginRegistry *> *live = [NFKPluginRegistry liveRegistries];
		@synchronized (live) {
			[live addObject:self];
		}
	}
	return self;
}

#pragma mark Settings

- (NFKPluginSignaturePolicy *)signaturePolicy
{
	@synchronized (self.lock) {
		return self.storedSignaturePolicy;
	}
}

- (void)setSignaturePolicy:(NFKPluginSignaturePolicy *)signaturePolicy
{
	NSArray<NFKPluginProcess *> *processes = nil;
	@synchronized (self.lock) {
		self.storedSignaturePolicy = signaturePolicy;
		processes = [self processes];
	}
	for (NFKPluginProcess *process in processes) {
		process.signaturePolicy = signaturePolicy;
	}
}

- (nullable NSString *)sharedMemoryPrefix
{
	@synchronized (self.lock) {
		return self.storedSharedMemoryPrefix;
	}
}

- (void)setSharedMemoryPrefix:(nullable NSString *)sharedMemoryPrefix
{
	NSArray<NFKPluginProcess *> *processes = nil;
	@synchronized (self.lock) {
		self.storedSharedMemoryPrefix = sharedMemoryPrefix;
		processes = [self processes];
	}
	for (NFKPluginProcess *process in processes) {
		process.sharedMemoryPrefix = sharedMemoryPrefix;
	}
}

/*! The processes the registry has made. Runs under the lock. */
- (NSArray<NFKPluginProcess *> *)processes
{
	NSMutableArray<NFKPluginProcess *> *processes = [NSMutableArray array];
	for (NFKPluginRegistryEntry *entry in self.entries.allValues) {
		if (entry.process != nil) {
			[processes addObject:entry.process];
		}
	}
	return processes;
}

- (nullable NSURL *)hostExecutableURL
{
	NSURL *assigned = nil;
	@synchronized (self.lock) {
		assigned = self.assignedHostExecutableURL;
	}
	return assigned ?: [NFKPluginRegistry defaultHostExecutableURL];
}

- (void)setHostExecutableURL:(nullable NSURL *)hostExecutableURL
{
	@synchronized (self.lock) {
		self.assignedHostExecutableURL = hostExecutableURL;
	}
}

/*! The first plugin host found in the main bundle's Contents/Helpers, among its auxiliary executables,
	beside its executable, or beside the image that contains InferKit. */
+ (nullable NSURL *)defaultHostExecutableURL
{
	NSBundle *main = NSBundle.mainBundle;
	NSMutableArray<NSURL *> *candidates = [NSMutableArray array];
	[candidates addObject:[main.bundleURL URLByAppendingPathComponent:@"Contents/Helpers/inferkit-plugin-host"]];
	NSURL *auxiliary = [main URLForAuxiliaryExecutable:NFKPluginHostExecutableName];
	if (auxiliary != nil) {
		[candidates addObject:auxiliary];
	}
	NSURL *executable = main.executableURL;
	if (executable != nil) {
		[candidates addObject:[executable.URLByDeletingLastPathComponent URLByAppendingPathComponent:NFKPluginHostExecutableName]];
	}
	Dl_info info;
	if (dladdr((const void *)&NFKPluginCatalogIdentifier, &info) != 0 && info.dli_fname != NULL) {
		NSURL *image = [NSURL fileURLWithPath:[NSString stringWithUTF8String:info.dli_fname]];
		[candidates addObject:[image.URLByDeletingLastPathComponent URLByAppendingPathComponent:NFKPluginHostExecutableName]];
	}
	for (NSURL *candidate in candidates) {
		NSURL *found = NFKPluginExecutableFile(candidate);
		if (found != nil) {
			return found;
		}
	}
	return nil;
}

#pragma mark Adding and removing

/*! The manifest at url once the bundle passes the ownership rule and this process can run it. */
- (nullable NFKPluginManifest *)checkedManifestAtURL:(NSURL *)url error:(NSError **)error
{
	NFKPluginManifest *manifest = [NFKPluginManifest manifestWithBundleURL:url error:error];
	if (manifest == nil || !NFKPluginCheckBundleOwnership(url, error) || ![manifest checkHostCompatibilityWithError:error]) {
		return nil;
	}
	return manifest;
}

- (nullable NFKPluginManifest *)addPluginAtURL:(NSURL *)url error:(NSError **)error
{
	NFKPluginManifest *manifest = [self checkedManifestAtURL:url error:error];
	if (manifest == nil) {
		return nil;
	}
	NSDictionary<NSString *, id> *kept = [self optionValues:[self optionValuesForPluginIdentifier:manifest.identifier]
											 keptByManifest:manifest];
	NFKPluginRegistryEntry *existing = nil;
	@synchronized (self.lock) {
		existing = self.entries[manifest.identifier];
		if (existing == nil) {
			NSArray<NFKModelDescriptor *> *descriptors = [self catalogDescriptorsForManifest:manifest error:error];
			if (descriptors == nil || ![self checkToolNamesOfManifest:manifest error:error]) {
				return nil;
			}
			NFKPluginRegistryEntry *entry = [[NFKPluginRegistryEntry alloc] init];
			entry.manifest = manifest;
			entry.catalogIdentifiers = @[];
			self.entries[manifest.identifier] = entry;
			[self listDescriptors:descriptors forEntry:entry];
			if (self.optionValuesByPlugin[manifest.identifier] != nil) {
				self.optionValuesByPlugin[manifest.identifier] = kept;
			}
			return manifest;
		}
	}
	NSString *listedPath = NFKPluginRegistryResolvedPath(existing.manifest.bundleURL ?: url);
	if (![listedPath isEqualToString:NFKPluginRegistryResolvedPath(url)]) {
		NFKPluginSetError(error, NFKPluginErrorIdentifierInUse,
						  [NSString stringWithFormat:@"the plugin %@ is already listed from %@", manifest.identifier, listedPath]);
		return nil;
	}
	return [self replaceManifestOfPlugin:manifest.identifier withManifest:manifest restartCompletion:nil error:error] ? manifest : nil;
}

- (NSArray<NFKPluginManifest *> *)addPluginsInDirectoryURL:(NSURL *)directoryURL errors:(NSArray<NSError *> **)errors
{
	NSError *listError = nil;
	NSArray<NSURL *> *contents = [NSFileManager.defaultManager contentsOfDirectoryAtURL:directoryURL includingPropertiesForKeys:nil
																				options:NSDirectoryEnumerationSkipsHiddenFiles
																				  error:&listError];
	if (contents == nil) {
		if (errors != NULL) {
			*errors = @[ listError ];
		}
		return @[];
	}
	NSArray<NSURL *> *sorted = [contents sortedArrayUsingComparator:^NSComparisonResult(NSURL *first, NSURL *second) {
		return [first.lastPathComponent compare:second.lastPathComponent];
	}];
	NSMutableArray<NFKPluginManifest *> *added = [NSMutableArray array];
	NSMutableArray<NSError *> *failures = [NSMutableArray array];
	for (NSURL *bundle in sorted) {
		if (![bundle.pathExtension isEqualToString:@"inferkitplugin"]) {
			continue;
		}
		NSError *addError = nil;
		NFKPluginManifest *manifest = [self addPluginAtURL:bundle error:&addError];
		if (manifest != nil) {
			[added addObject:manifest];
		} else if (addError != nil) {
			[failures addObject:addError];
		}
	}
	if (errors != NULL) {
		*errors = [failures copy];
	}
	return [added copy];
}

- (void)removePluginWithIdentifier:(NSString *)identifier
{
	@synchronized (self.lock) {
		NFKPluginRegistryEntry *entry = self.entries[identifier];
		if (entry == nil) {
			return;
		}
		[self listDescriptors:@[] forEntry:entry];
		[self.entries removeObjectForKey:identifier];
	}
}

- (BOOL)reloadPluginWithIdentifier:(NSString *)identifier error:(NSError **)error
{
	return [self reloadPluginWithIdentifier:identifier restartCompletion:nil error:error];
}

- (void)reloadPluginWithIdentifier:(NSString *)identifier completionHandler:(void (^)(NSError * _Nullable))completionHandler
{
	void (^finish)(NSError * _Nullable) = ^(NSError * _Nullable failure) {
		dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
			completionHandler(failure);
		});
	};
	NSError *error = nil;
	if (![self reloadPluginWithIdentifier:identifier restartCompletion:finish error:&error]) {
		finish(error);
	}
}

/*! Reloads the plugin and calls restartCompletion, when there is one, once its process has taken the new
	manifest: at once for a stopped process, after the restart for a running one. Returns NO without
	calling it when the reload fails. */
- (BOOL)reloadPluginWithIdentifier:(NSString *)identifier
				 restartCompletion:(nullable void (^)(NSError * _Nullable error))restartCompletion
							 error:(NSError **)error
{
	NSURL *bundle = nil;
	@synchronized (self.lock) {
		bundle = self.entries[identifier].manifest.bundleURL;
	}
	if (bundle == nil) {
		return NFKPluginSetError(error, NFKPluginErrorUnknownPlugin, [NSString stringWithFormat:@"no plugin %@ is listed", identifier]);
	}
	NFKPluginManifest *manifest = [self checkedManifestAtURL:bundle error:error];
	if (manifest == nil) {
		return NO;
	}
	if (![manifest.identifier isEqualToString:identifier]) {
		return NFKPluginSetError(error, NFKPluginErrorManifestMismatch,
								 [NSString stringWithFormat:@"the bundle at %@ now holds the plugin %@, not %@", bundle.path,
								  manifest.identifier, identifier]);
	}
	return [self replaceManifestOfPlugin:identifier withManifest:manifest restartCompletion:restartCompletion error:error];
}

/*! Lists manifest in place of the plugin's current one, and hands it to the plugin's process: a stopped
	process takes it for its next start, and a running one restarts with it. restartCompletion receives
	the restart's outcome. */
- (BOOL)replaceManifestOfPlugin:(NSString *)identifier
				   withManifest:(NFKPluginManifest *)manifest
			  restartCompletion:(nullable void (^)(NSError * _Nullable error))restartCompletion
						  error:(NSError **)error
{
	NSDictionary<NSString *, id> *kept = [self optionValues:[self optionValuesForPluginIdentifier:identifier] keptByManifest:manifest];
	NFKPluginProcess *process = nil;
	@synchronized (self.lock) {
		NFKPluginRegistryEntry *entry = self.entries[identifier];
		if (entry == nil) {
			return NFKPluginSetError(error, NFKPluginErrorUnknownPlugin, [NSString stringWithFormat:@"no plugin %@ is listed", identifier]);
		}
		NFKLoadedPlugin *loaded = entry.loadedPlugin;
		if (loaded != nil && ![loaded.manifest.version isEqualToString:manifest.version]) {
			return NFKPluginSetError(error, NFKPluginErrorLoadFailed,
									 [NSString stringWithFormat:@"the plugin %@ runs version %@ in the app's process until the app quits, so version %@ cannot replace it",
									  identifier, loaded.manifest.version, manifest.version]);
		}
		NSArray<NFKModelDescriptor *> *descriptors = [self catalogDescriptorsForManifest:manifest error:error];
		if (descriptors == nil || ![self checkToolNamesOfManifest:manifest error:error]) {
			return NO;
		}
		entry.manifest = manifest;
		[self listDescriptors:descriptors forEntry:entry];
		if (self.optionValuesByPlugin[identifier] != nil) {
			self.optionValuesByPlugin[identifier] = kept;
		}
		process = entry.process;
	}
	if (process != nil) {
		process.pluginOptions = kept;
	}
	[self handManifest:manifest toProcess:process completion:restartCompletion];
	return YES;
}

/*! The values a new manifest still takes: each key it declares whose value resolves against the new
	declaration, converted. A key it no longer declares, and a value that no longer resolves, are left
	out. */
- (NSDictionary<NSString *, id> *)optionValues:(NSDictionary<NSString *, id> *)values keptByManifest:(NFKPluginManifest *)manifest
{
	NSMutableDictionary<NSString *, id> *kept = [NSMutableDictionary dictionary];
	[values enumerateKeysAndObjectsUsingBlock:^(NSString *key, id value, BOOL *stop) {
		NSDictionary<NSString *, id> *resolved = [self resolvedOptionValues:@{ key: value } ofManifest:manifest error:NULL];
		if (resolved[key] != nil) {
			kept[key] = resolved[key];
		}
	}];
	return kept;
}

/*! Gives a process the manifest: a stopped process takes it for its next start and completion runs at
	once; a running process restarts with it and completion runs when the restart ends. */
- (void)handManifest:(NFKPluginManifest *)manifest
		   toProcess:(nullable NFKPluginProcess *)process
		  completion:(nullable void (^)(NSError * _Nullable error))completion
{
	if (process == nil || [process replaceManifestWhileStopped:manifest]) {
		if (completion != nil) {
			completion(nil);
		}
		return;
	}
	[process restartWithManifest:manifest completionHandler:completion];
}

/*! Refuses a manifest that declares a tool another listed plugin declares. Runs under the lock. */
- (BOOL)checkToolNamesOfManifest:(NFKPluginManifest *)manifest error:(NSError **)error
{
	for (NSDictionary<NSString *, id> *declaration in manifest.tools) {
		NSString *name = [declaration[@"name"] isKindOfClass:NSString.class] ? declaration[@"name"] : nil;
		if (name.length == 0) {
			continue;
		}
		for (NFKPluginRegistryEntry *other in self.entries.allValues) {
			if ([other.manifest.identifier isEqualToString:manifest.identifier]) {
				continue;
			}
			if ([[other.manifest.tools valueForKey:@"name"] containsObject:name]) {
				return NFKPluginSetError(error, NFKPluginErrorIdentifierInUse,
										 [NSString stringWithFormat:@"the plugin %@ declares the tool %@, which the listed plugin %@ already declares",
										  manifest.identifier, name, other.manifest.identifier]);
			}
		}
	}
	return YES;
}

/*! The catalog descriptors of a manifest's models, or nil with NFKPluginErrorIdentifierInUse when the
	catalog lists a model under one of their identifiers that this plugin did not add. Runs under the
	lock. */
- (nullable NSArray<NFKModelDescriptor *> *)catalogDescriptorsForManifest:(NFKPluginManifest *)manifest error:(NSError **)error
{
	NSMutableArray<NFKModelDescriptor *> *descriptors = [NSMutableArray arrayWithCapacity:manifest.models.count];
	for (NFKModelDescriptor *model in manifest.models) {
		NSString *catalogIdentifier = NFKPluginCatalogIdentifier(manifest.identifier, model.identifier);
		BOOL ours = [self.pluginIdentifiersByCatalogIdentifier[catalogIdentifier] isEqualToString:manifest.identifier];
		if (!ours && [self.catalog descriptorWithIdentifier:catalogIdentifier] != nil) {
			NFKPluginSetError(error, NFKPluginErrorIdentifierInUse,
							  [NSString stringWithFormat:@"the catalog already lists a model %@, which the plugin %@ cannot replace",
							   catalogIdentifier, manifest.identifier]);
			return nil;
		}
		NSMutableDictionary<NSString *, id> *dictionary = [model.dictionaryRepresentation mutableCopy];
		// A plugin's model builds through the registry only, never through a class the manifest names.
		[dictionary removeObjectForKey:@"providerClass"];
		dictionary[@"id"] = catalogIdentifier;
		dictionary[@"registryName"] = model.identifier;
		dictionary[@"engine"] = NFKModelEnginePlugin;
		dictionary[NFKPluginDescriptorExtensionKey] = @{ @"id": manifest.identifier, @"version": manifest.version,
														 @"kind": manifest.entryKind };
		NFKModelDescriptor *descriptor = [NFKModelDescriptor descriptorWithDictionary:dictionary error:error];
		if (descriptor == nil) {
			return nil;
		}
		[descriptors addObject:descriptor];
	}
	return descriptors;
}

/*! Replaces the entry's descriptors in the catalog with descriptors. Runs under the lock. */
- (void)listDescriptors:(NSArray<NFKModelDescriptor *> *)descriptors forEntry:(NFKPluginRegistryEntry *)entry
{
	NSArray<NSString *> *identifiers = [descriptors valueForKey:@"identifier"];
	for (NSString *stale in entry.catalogIdentifiers) {
		if (![identifiers containsObject:stale]) {
			[self.catalog removeDescriptorWithIdentifier:stale];
			[self.pluginIdentifiersByCatalogIdentifier removeObjectForKey:stale];
		}
	}
	for (NSString *identifier in identifiers) {
		self.pluginIdentifiersByCatalogIdentifier[identifier] = entry.manifest.identifier;
	}
	[self.catalog addDescriptors:descriptors];
	entry.catalogIdentifiers = identifiers;
}

#pragma mark Reading

- (NSArray<NFKPluginManifest *> *)manifests
{
	NSMutableArray<NFKPluginManifest *> *manifests = [NSMutableArray array];
	@synchronized (self.lock) {
		for (NSString *identifier in [self.entries.allKeys sortedArrayUsingSelector:@selector(compare:)]) {
			[manifests addObject:self.entries[identifier].manifest];
		}
	}
	return [manifests copy];
}

- (nullable NFKPluginManifest *)manifestForModelIdentifier:(NSString *)catalogIdentifier
{
	@synchronized (self.lock) {
		NSString *plugin = self.pluginIdentifiersByCatalogIdentifier[catalogIdentifier];
		return plugin != nil ? self.entries[plugin].manifest : nil;
	}
}

#pragma mark Trust

- (void)setTrusted:(BOOL)trusted forPluginIdentifier:(NSString *)identifier
{
	@synchronized (self.lock) {
		if (trusted) {
			[self.trustedIdentifiers addObject:identifier];
		} else {
			[self.trustedIdentifiers removeObject:identifier];
		}
	}
}

- (BOOL)isTrustedPluginIdentifier:(NSString *)identifier
{
	@synchronized (self.lock) {
		return [self.trustedIdentifiers containsObject:identifier];
	}
}

#pragma mark Option values

- (BOOL)setOptionValues:(nullable NSDictionary<NSString *, id> *)values
	forPluginIdentifier:(NSString *)identifier
				  error:(NSError **)error
{
	NFKPluginManifest *manifest = nil;
	@synchronized (self.lock) {
		manifest = self.entries[identifier].manifest;
	}
	if (manifest == nil) {
		return NFKPluginSetError(error, NFKPluginErrorUnknownPlugin, [NSString stringWithFormat:@"no plugin %@ is listed", identifier]);
	}
	NSDictionary<NSString *, id> *resolved = [self resolvedOptionValues:values ?: @{} ofManifest:manifest error:error];
	if (resolved == nil) {
		return NO;
	}
	NFKPluginProcess *process = nil;
	@synchronized (self.lock) {
		self.optionValuesByPlugin[identifier] = resolved;
		process = self.entries[identifier].process;
	}
	if (process != nil) {
		process.pluginOptions = resolved;
		[self handManifest:process.manifest toProcess:process completion:nil];
	}
	return YES;
}

- (NSDictionary<NSString *, id> *)optionValuesForPluginIdentifier:(NSString *)identifier
{
	@synchronized (self.lock) {
		return self.optionValuesByPlugin[identifier] ?: @{};
	}
}

/*! The values each converted to its option's kind, keeping only the keys given, or nil with
	NFKModelCatalogErrorInvalidOptions. The options resolve through a descriptor that carries them, so
	their problems are the model options' own. */
- (nullable NSDictionary<NSString *, id> *)resolvedOptionValues:(NSDictionary<NSString *, id> *)values
													 ofManifest:(NFKPluginManifest *)manifest
														  error:(NSError **)error
{
	if (values.count == 0) {
		return @{};
	}
	NSMutableArray<NSDictionary<NSString *, id> *> *options = [NSMutableArray array];
	for (NFKModelOption *option in manifest.options) {
		[options addObject:option.dictionaryRepresentation];
	}
	NFKModelDescriptor *carrier = [NFKModelDescriptor descriptorWithDictionary:@{ @"id": manifest.identifier, @"family": manifest.identifier,
																				   @"engine": NFKModelEnginePlugin, @"options": options }
																		 error:error];
	NSDictionary<NSString *, id> *resolved = [carrier resolvedOptions:values stage:nil error:error];
	if (resolved == nil) {
		return nil;
	}
	NSMutableDictionary<NSString *, id> *given = [NSMutableDictionary dictionary];
	for (NSString *key in values) {
		given[key] = resolved[key];
	}
	return given;
}

#pragma mark Backends and tools

- (nullable NFKPluginBackend *)backendForModelIdentifier:(NSString *)catalogIdentifier
												 options:(nullable NSDictionary<NSString *, id> *)options
												   error:(NSError **)error
{
	NFKPluginManifest *manifest = nil;
	BOOL trusted = NO;
	@synchronized (self.lock) {
		NSString *plugin = self.pluginIdentifiersByCatalogIdentifier[catalogIdentifier];
		manifest = plugin != nil ? self.entries[plugin].manifest : nil;
		trusted = plugin != nil && [self.trustedIdentifiers containsObject:plugin];
	}
	NSString *modelIdentifier = [catalogIdentifier substringFromIndex:MIN(manifest.identifier.length + 1, catalogIdentifier.length)];
	NFKModelDescriptor *model = manifest != nil ? [manifest modelWithIdentifier:modelIdentifier] : nil;
	if (model == nil) {
		NFKPluginSetError(error, NFKPluginErrorUnknownModel, [NSString stringWithFormat:@"no listed plugin serves the model %@", catalogIdentifier]);
		return nil;
	}
	NSDictionary<NSString *, id> *values = options ?: @{};
	if (trusted) {
		return [self inProcessBackendForModel:model manifest:manifest options:values error:error];
	}
	NFKPluginProcess *process = [self processForPluginIdentifier:manifest.identifier];
	if (process == nil) {
		NFKPluginSetError(error, NFKPluginErrorUnknownModel, [NSString stringWithFormat:@"no listed plugin serves the model %@", catalogIdentifier]);
		return nil;
	}
	return [process backendForModelIdentifier:modelIdentifier options:values error:error];
}

/*! A backend over the plugin's own backend in this process, loading the plugin on first use. */
- (nullable NFKPluginBackend *)inProcessBackendForModel:(NFKModelDescriptor *)model
											   manifest:(NFKPluginManifest *)manifest
												options:(NSDictionary<NSString *, id> *)options
												  error:(NSError **)error
{
	NFKLoadedPlugin *loaded = [self loadedPluginForManifest:manifest error:error];
	if (loaded == nil) {
		return nil;
	}
	id<NFKInferenceBackend> pluginBackend = [loaded backendForModelIdentifier:model.identifier options:options error:error];
	if (pluginBackend == nil) {
		return nil;
	}
	return [[NFKPluginBackend alloc] initWithPluginIdentifier:manifest.identifier descriptor:model options:options
												pluginBackend:pluginBackend];
}

/*! The plugin loaded into this process under signaturePolicy, or nil with NFKPluginErrorNotTrusted for a
	plugin that cannot run here, or with the loader's error. */
- (nullable NFKLoadedPlugin *)loadedPluginForManifest:(NFKPluginManifest *)manifest error:(NSError **)error
{
	if (!NFKPluginRunsInProcess(manifest)) {
		NFKPluginSetError(error, NFKPluginErrorNotTrusted,
						  [NSString stringWithFormat:@"the plugin %@ runs only in a process of its own", manifest.identifier]);
		return nil;
	}
	NFKLoadedPlugin *loaded = [NFKPluginLoader loadPluginAtURL:manifest.bundleURL signaturePolicy:self.signaturePolicy
													   options:[self optionValuesForPluginIdentifier:manifest.identifier]
											 runsInHostProcess:YES logHandler:nil error:error];
	if (loaded == nil) {
		return nil;
	}
	@synchronized (self.lock) {
		self.entries[manifest.identifier].loadedPlugin = loaded;
	}
	return loaded;
}

/*! The plugin's supervisor, made on first use with the registry's policy and host, or nil for a plugin
	the registry does not list. Makes no process. */
- (nullable NFKPluginProcess *)processForPluginIdentifier:(NSString *)identifier
{
	NSURL *host = self.hostExecutableURL;
	@synchronized (self.lock) {
		NFKPluginRegistryEntry *entry = self.entries[identifier];
		if (entry == nil) {
			return nil;
		}
		if (entry.process == nil) {
			NFKPluginProcess *process = [[NFKPluginProcess alloc] initWithManifest:entry.manifest hostExecutableURL:host];
			process.signaturePolicy = self.storedSignaturePolicy;
			process.sharedMemoryPrefix = self.storedSharedMemoryPrefix;
			process.pluginOptions = self.optionValuesByPlugin[identifier] ?: @{};
			entry.process = process;
		}
		return entry.process;
	}
}

- (NSArray<NFKInferenceTool *> *)tools
{
	NSMutableArray<NFKInferenceTool *> *tools = [NSMutableArray array];
	for (NFKPluginManifest *manifest in self.manifests) {
		// A trusted plugin's tools run where its backends do: in this process, and a manifest that does not
		// allow it fails each call with NFKPluginErrorNotTrusted, as its backends do.
		if ([self isTrustedPluginIdentifier:manifest.identifier]) {
			[tools addObjectsFromArray:[self inProcessToolsOfManifest:manifest]];
			continue;
		}
		NFKPluginProcess *process = [self processForPluginIdentifier:manifest.identifier];
		if (process != nil) {
			[tools addObjectsFromArray:process.tools];
		}
	}
	return [tools copy];
}

/*! Tools from the manifest's declarations that load the plugin into this process on their first call
	and run its own tool of the same name. */
- (NSArray<NFKInferenceTool *> *)inProcessToolsOfManifest:(NFKPluginManifest *)manifest
{
	NSMutableArray<NFKInferenceTool *> *tools = [NSMutableArray array];
	__weak NFKPluginRegistry *weakSelf = self;
	for (NSDictionary<NSString *, id> *declaration in manifest.tools) {
		NSString *name = [declaration[@"name"] isKindOfClass:NSString.class] ? declaration[@"name"] : nil;
		if (name.length == 0) {
			continue;
		}
		NSString *summary = [declaration[@"description"] isKindOfClass:NSString.class] ? declaration[@"description"] : @"";
		NSDictionary *parameters = [declaration[@"parameters"] isKindOfClass:NSDictionary.class] ? declaration[@"parameters"]
			: @{ @"type": @"object" };
		NFKInferenceTool *tool = [NFKInferenceTool toolWithName:name description:summary parameters:parameters
														handler:^NFKToolResult * _Nullable(NSDictionary<NSString *, id> *arguments, NSError **error) {
			NFKPluginRegistry *registry = weakSelf;
			NFKLoadedPlugin *loaded = registry != nil ? [registry loadedPluginForManifest:manifest error:error] : nil;
			for (NFKInferenceTool *own in loaded.tools) {
				if ([own.name isEqualToString:name]) {
					return [own resultForArguments:arguments error:error];
				}
			}
			if (loaded != nil || registry == nil) {
				NFKPluginSetError(error, NFKPluginErrorUnknownTool,
								  [NSString stringWithFormat:@"the plugin %@ has no tool %@", manifest.identifier, name]);
			}
			return nil;
		}];
		[tools addObject:tool];
	}
	return tools;
}

@end

#endif
