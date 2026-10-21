//
//  NFKPluginLoader.m
//  InferKit
//

#import "NFKPluginLoader.h"
#import "NFKPluginLoaderPrivate.h"
#import "NFKPluginManifestPrivate.h"
#import "NFKPluginCAdapter.h"
#import "NFKPluginWire.h"
#import "NFKInferKit.h"

#if TARGET_OS_OSX

#import <dlfcn.h>
#import <mach-o/dyld.h>
#import <objc/runtime.h>
#import <os/log.h>
#import <stdlib.h>
#import <sys/stat.h>
#import <unistd.h>

#pragma mark - Context

@interface NFKPluginHostContext ()
@property (nonatomic, readwrite, copy) NSString *pluginIdentifier;
@property (nonatomic, readwrite, copy) NSURL *temporaryDirectoryURL;
@property (nonatomic, readwrite, copy) NSDictionary<NSString *, id> *options;
@property (nonatomic, readwrite) BOOL runsInHostProcess;
@property (nonatomic, copy, nullable) NFKPluginLogHandler logHandler;
@end

@implementation NFKPluginHostContext

- (instancetype)initWithPluginIdentifier:(NSString *)pluginIdentifier
								 options:(NSDictionary<NSString *, id> *)options
					   runsInHostProcess:(BOOL)runsInHostProcess
							  logHandler:(nullable NFKPluginLogHandler)logHandler
{
	self = [super init];
	if (self != nil) {
		_pluginIdentifier = [pluginIdentifier copy];
		_options = [options copy];
		_runsInHostProcess = runsInHostProcess;
		_logHandler = [logHandler copy];
		NSURL *plugins = [[NSURL fileURLWithPath:NSTemporaryDirectory()] URLByAppendingPathComponent:@"InferKit/Plugins" isDirectory:YES];
		NSString *name = [NSString stringWithFormat:@"%@.%@", pluginIdentifier, NSUUID.UUID.UUIDString];
		_temporaryDirectoryURL = [plugins URLByAppendingPathComponent:name isDirectory:YES];
		[NSFileManager.defaultManager createDirectoryAtURL:_temporaryDirectoryURL withIntermediateDirectories:YES
												attributes:@{ NSFilePosixPermissions: @0700 } error:NULL];
	}
	return self;
}

- (NSString *)hostAPIVersion
{
	return NFKInferKit.APIVersion;
}

- (void)logMessage:(NSString *)message level:(NFKPluginLogLevel)level
{
	NFKPluginLogHandler handler = self.logHandler;
	if (handler != nil) {
		handler(level, message);
		return;
	}
	static os_log_t log;
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		log = os_log_create("com.inferkit", "plugin");
	});
	os_log_type_t type = level >= NFKPluginLogLevelError ? OS_LOG_TYPE_ERROR
		: level == NFKPluginLogLevelWarning ? OS_LOG_TYPE_DEFAULT
		: level == NFKPluginLogLevelInfo ? OS_LOG_TYPE_INFO : OS_LOG_TYPE_DEBUG;
	os_log_with_type(log, type, "%{public}@: %{public}@", self.pluginIdentifier, message);
}

- (void)removeTemporaryDirectory
{
	[NSFileManager.defaultManager removeItemAtURL:self.temporaryDirectoryURL error:NULL];
}

@end

#pragma mark - Loaded plugin

@interface NFKLoadedPlugin ()
@property (nonatomic, readwrite, copy) NFKPluginManifest *manifest;
/*! The bundle's resolved path, which identifies it across URLs that spell it differently. */
@property (nonatomic, copy) NSString *resolvedBundlePath;
@end

@implementation NFKLoadedPlugin

- (instancetype)initWithManifest:(NFKPluginManifest *)manifest
						  plugin:(id<NFKPlugin>)plugin
						 context:(NFKPluginHostContext *)context
			  resolvedBundlePath:(NSString *)resolvedBundlePath
{
	self = [super init];
	if (self != nil) {
		_manifest = [manifest copy];
		_plugin = plugin;
		_context = context;
		_resolvedBundlePath = [resolvedBundlePath copy];
	}
	return self;
}

- (NSArray<NFKModelDescriptor *> *)describedModels
{
	return [self.plugin modelDescriptors] ?: @[];
}

- (NSArray<NFKInferenceTool *> *)tools
{
	return [self.plugin respondsToSelector:@selector(tools)] ? ([self.plugin tools] ?: @[]) : @[];
}

- (nullable id<NFKInferenceBackend>)backendForModelIdentifier:(NSString *)modelIdentifier
													  options:(NSDictionary<NSString *, id> *)options
														error:(NSError **)error
{
	if ([self.manifest modelWithIdentifier:modelIdentifier] == nil) {
		NFKPluginSetError(error, NFKPluginErrorUnknownModel,
						  [NSString stringWithFormat:@"the plugin %@ lists no model \"%@\"", self.manifest.identifier, modelIdentifier]);
		return nil;
	}
	NSError *failure = nil;
	id<NFKInferenceBackend> backend = [self.plugin backendForModelIdentifier:modelIdentifier options:options error:&failure];
	if (backend == nil) {
		if (failure == nil) {
			NFKPluginSetError(&failure, NFKPluginErrorLoadFailed,
							  [NSString stringWithFormat:@"the plugin %@ made no backend for \"%@\"", self.manifest.identifier, modelIdentifier]);
		}
		if (error != NULL) {
			*error = failure;
		}
	}
	return backend;
}

- (NSString *)description
{
	return [NSString stringWithFormat:@"<%@ %@ %@ at %@>", self.class, self.manifest.identifier, self.manifest.version, self.resolvedBundlePath];
}

@end

#pragma mark - Loader

/*! The resolved path of a file, so two spellings of one file compare equal. */
static NSString *NFKPluginResolvedPath(NSURL *url)
{
	char resolved[PATH_MAX];
	return realpath(url.fileSystemRepresentation, resolved) != NULL ? [NSString stringWithUTF8String:resolved] : url.URLByStandardizingPath.path;
}

/*! The name dyld recorded for the loaded image at url, which objc_copyClassNamesForImage matches. */
static const char * _Nullable NFKPluginLoadedImageName(NSURL *url)
{
	NSString *target = NFKPluginResolvedPath(url);
	uint32_t count = _dyld_image_count();
	for (uint32_t index = 0; index < count; index++) {
		const char *name = _dyld_get_image_name(index);
		char resolved[PATH_MAX];
		if (name != NULL && realpath(name, resolved) != NULL && [target isEqualToString:[NSString stringWithUTF8String:resolved]]) {
			return name;
		}
	}
	return NULL;
}

/*! Why an item fails the ownership rule, or nil when it passes. */
static NSString * _Nullable NFKPluginOwnershipProblem(const struct stat *status)
{
	uid_t user = getuid();
	if (status->st_uid != user && status->st_uid != 0) {
		return [NSString stringWithFormat:@"is owned by user %u, neither this user (%u) nor root", status->st_uid, user];
	}
	if (!S_ISLNK(status->st_mode) && (status->st_mode & S_IWOTH) != 0) {
		return @"is writable by every user";
	}
	if (!S_ISLNK(status->st_mode) && (status->st_mode & S_IWGRP) != 0) {
		return @"is writable by its group";
	}
	return nil;
}

BOOL NFKPluginCheckBundleOwnership(NSURL *bundleURL, NSError **error)
{
	NSString *bundle = NFKPluginResolvedPath(bundleURL);
	NSMutableArray<NSString *> *paths = [NSMutableArray arrayWithObjects:bundle.stringByDeletingLastPathComponent, bundle, nil];
	NSDirectoryEnumerator<NSString *> *enumerator = [NSFileManager.defaultManager enumeratorAtPath:bundle];
	for (NSString *relative in enumerator) {
		[paths addObject:[bundle stringByAppendingPathComponent:relative]];
	}
	for (NSString *path in paths) {
		struct stat status;
		if (lstat(path.fileSystemRepresentation, &status) != 0) {
			int code = errno;
			return NFKPluginSetError(error, NFKPluginErrorLoadFailed, [NSString stringWithFormat:@"%@ could not be examined: %s", path, strerror(code)]);
		}
		NSString *problem = NFKPluginOwnershipProblem(&status);
		if (problem != nil) {
			return NFKPluginSetError(error, NFKPluginErrorLoadFailed,
									 [NSString stringWithFormat:@"%@ %@, so another account could change the plugin after its signature is checked",
									  path, problem]);
		}
	}
	return YES;
}

@implementation NFKPluginLoader

/*! The plugins this process loaded, by identifier. */
+ (NSMutableDictionary<NSString *, NFKLoadedPlugin *> *)loadedPlugins
{
	static NSMutableDictionary<NSString *, NFKLoadedPlugin *> *plugins = nil;
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		plugins = [NSMutableDictionary dictionary];
	});
	return plugins;
}

/*! The loads that failed after code entered the process, by resolved bundle path. Such code stays, so a
	second attempt answers the same error. A load that fails before its code enters the process, such as a
	library dlopen refuses, is not kept, so a retry after the bundle is fixed loads it. */
+ (NSMutableDictionary<NSString *, NSError *> *)failedLoads
{
	static NSMutableDictionary<NSString *, NSError *> *failures = nil;
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		failures = [NSMutableDictionary dictionary];
	});
	return failures;
}

+ (nullable NFKLoadedPlugin *)loadPluginAtURL:(NSURL *)url signaturePolicy:(NFKPluginSignaturePolicy *)policy error:(NSError **)error
{
	return [self loadPluginAtURL:url signaturePolicy:policy options:@{} runsInHostProcess:YES logHandler:nil error:error];
}

+ (nullable NFKLoadedPlugin *)loadPluginAtURL:(NSURL *)url
							  signaturePolicy:(NFKPluginSignaturePolicy *)policy
									  options:(NSDictionary<NSString *, id> *)options
							runsInHostProcess:(BOOL)runsInHostProcess
								   logHandler:(nullable NFKPluginLogHandler)logHandler
										error:(NSError **)error
{
	NFKPluginManifest *manifest = [NFKPluginManifest manifestWithBundleURL:url error:error];
	if (manifest == nil) {
		return nil;
	}
	if ([manifest.entryKind isEqualToString:NFKPluginEntryKindExecutable]) {
		NFKPluginSetError(error, NFKPluginErrorNotTrusted,
						  [NSString stringWithFormat:@"the plugin %@ is an executable, which runs only as its own process", manifest.identifier]);
		return nil;
	}
	if (![manifest checkHostCompatibilityWithError:error]) {
		return nil;
	}
	if (!NFKPluginCheckBundleOwnership(url, error)) {
		return nil;
	}
	NFKPluginSignaturePolicy *checks = policy ?: NFKPluginSignaturePolicy.defaultPolicy;
	NSString *bundlePath = NFKPluginResolvedPath(url);
	NSMutableDictionary<NSString *, NFKLoadedPlugin *> *loaded = self.loadedPlugins;
	@synchronized (loaded) {
		NFKLoadedPlugin *existing = loaded[manifest.identifier];
		if (existing != nil) {
			// The policy applies to every call, so a caller with a stricter policy never receives a
			// loaded plugin it would refuse. A cached success makes the check cheap.
			if (![checks validateCodeAtURL:url error:error]) {
				return nil;
			}
			return [self existingPlugin:existing matchingManifest:manifest bundlePath:bundlePath error:error];
		}
		NSError *earlier = self.failedLoads[bundlePath];
		if (earlier != nil) {
			if (error != NULL) {
				*error = earlier;
			}
			return nil;
		}
		NFKPluginHostContext *context = [[NFKPluginHostContext alloc] initWithPluginIdentifier:manifest.identifier
																					   options:[manifest optionValuesWithDefaults:options ?: @{}]
																			  runsInHostProcess:runsInHostProcess logHandler:logHandler];
		NSError *loadError = nil;
		BOOL codeLoaded = NO;
		id<NFKPlugin> plugin = [manifest.entryKind isEqualToString:NFKPluginEntryKindObjC]
			? [self objectiveCPluginWithManifest:manifest policy:checks context:context codeLoaded:&codeLoaded error:&loadError]
			: [self CPluginWithManifest:manifest policy:checks context:context codeLoaded:&codeLoaded error:&loadError];
		if (plugin == nil) {
			[context removeTemporaryDirectory];
			if (codeLoaded) {
				self.failedLoads[bundlePath] = loadError;
			}
			if (error != NULL) {
				*error = loadError;
			}
			return nil;
		}
		NFKLoadedPlugin *result = [[NFKLoadedPlugin alloc] initWithManifest:manifest plugin:plugin context:context resolvedBundlePath:bundlePath];
		loaded[manifest.identifier] = result;
		return result;
	}
}

+ (nullable NFKLoadedPlugin *)existingPlugin:(NFKLoadedPlugin *)existing
							matchingManifest:(NFKPluginManifest *)manifest
								  bundlePath:(NSString *)bundlePath
									   error:(NSError **)error
{
	if ([existing.resolvedBundlePath isEqualToString:bundlePath] && [existing.manifest.version isEqualToString:manifest.version]) {
		return existing;
	}
	NFKPluginSetError(error, NFKPluginErrorLoadFailed,
					  [NSString stringWithFormat:@"the plugin %@ %@ is loaded from %@; a second version or copy, %@ from %@, cannot load into the same process",
					   manifest.identifier, existing.manifest.version, existing.resolvedBundlePath, manifest.version, bundlePath]);
	return nil;
}

#pragma mark Objective-C

/*! Sets *codeLoaded once the bundle's code has entered the process, so a failure after that point is
	remembered. */
+ (nullable id<NFKPlugin>)objectiveCPluginWithManifest:(NFKPluginManifest *)manifest
												 policy:(NFKPluginSignaturePolicy *)policy
												context:(NFKPluginHostContext *)context
											 codeLoaded:(BOOL *)codeLoaded
												  error:(NSError **)error
{
	*codeLoaded = NO;
	if (manifest.executablePath != nil && ![manifest.executablePath isEqualToString:manifest.bundleExecutableName ?: @""]) {
		NFKPluginSetError(error, NFKPluginErrorManifestMismatch,
						  [NSString stringWithFormat:@"an \"objc\" plugin's code is its bundle's executable, and %@ names %@",
						   manifest.identifier, manifest.executablePath]);
		return nil;
	}
	NSBundle *bundle = [NSBundle bundleWithURL:manifest.bundleURL];
	NSError *bundleError = nil;
	// Checked and verified immediately before the load, so the window an edit could use is as short as it
	// can be. NSBundle loads the file CFBundleExecutable names, which the check resolved.
	NSURL *executable = [manifest containedExecutableURLWithError:error];
	if (executable == nil || ![policy validateCodeAtURL:manifest.bundleURL error:error]) {
		return nil;
	}
	if (bundle == nil || ![bundle loadAndReturnError:&bundleError]) {
		NFKPluginSetError(error, NFKPluginErrorLoadFailed,
						  [NSString stringWithFormat:@"the plugin %@ did not load: %@", manifest.identifier,
						   bundleError.localizedDescription ?: @"its bundle could not be opened"]);
		return nil;
	}
	*codeLoaded = YES;
	const char *image = NFKPluginLoadedImageName(executable);
	if (image == NULL) {
		NFKPluginSetError(error, NFKPluginErrorLoadFailed, [NSString stringWithFormat:@"the plugin %@'s code is not among the loaded images", manifest.identifier]);
		return nil;
	}
	if (![self checkSingleRuntimeInImage:image manifest:manifest error:error]) {
		return nil;
	}
	Class principal = manifest.principalClassName != nil ? NSClassFromString(manifest.principalClassName) : bundle.principalClass;
	const char *principalImage = principal != Nil ? class_getImageName(principal) : NULL;
	if (principalImage == NULL || strcmp(principalImage, image) != 0) {
		NFKPluginSetError(error, NFKPluginErrorLoadFailed,
						  [NSString stringWithFormat:@"the plugin %@ defines no principal class %@", manifest.identifier,
						   manifest.principalClassName ?: @"(NSPrincipalClass)"]);
		return nil;
	}
	BOOL adopts = [principal conformsToProtocol:@protocol(NFKPlugin)]
		|| ([principal instancesRespondToSelector:@selector(initWithContext:)] && [principal instancesRespondToSelector:@selector(modelDescriptors)]
			&& [principal instancesRespondToSelector:@selector(backendForModelIdentifier:options:error:)]);
	if (!adopts) {
		NFKPluginSetError(error, NFKPluginErrorLoadFailed,
						  [NSString stringWithFormat:@"the plugin %@'s principal class %s does not adopt NFKPlugin", manifest.identifier, class_getName(principal)]);
		return nil;
	}
	id<NFKPlugin> plugin = nil;
	@try {
		plugin = [(id<NFKPlugin>)[principal alloc] initWithContext:context];
	} @catch (NSException *exception) {
		NFKPluginSetError(error, NFKPluginErrorLoadFailed,
						  [NSString stringWithFormat:@"the plugin %@ raised %@ when created: %@", manifest.identifier, exception.name, exception.reason]);
		return nil;
	}
	if (plugin == nil) {
		NFKPluginSetError(error, NFKPluginErrorLoadFailed, [NSString stringWithFormat:@"the plugin %@'s initWithContext: returned nil", manifest.identifier]);
	}
	return plugin;
}

/*! Refuses an image that defines an NFK class: it carries its own copy of InferKit, and the runtime keeps
	one of the two definitions of each class without saying which. */
+ (BOOL)checkSingleRuntimeInImage:(const char *)image manifest:(NFKPluginManifest *)manifest error:(NSError **)error
{
	unsigned int count = 0;
	const char **names = objc_copyClassNamesForImage(image, &count);
	NSMutableArray<NSString *> *duplicates = [NSMutableArray array];
	for (unsigned int index = 0; index < count; index++) {
		if (strncmp(names[index], "NFK", 3) == 0) {
			[duplicates addObject:[NSString stringWithUTF8String:names[index]]];
		}
	}
	free(names);
	if (duplicates.count == 0) {
		return YES;
	}
	[duplicates sortUsingSelector:@selector(compare:)];
	return NFKPluginSetError(error, NFKPluginErrorDuplicateRuntime,
							 [NSString stringWithFormat:@"the plugin %@ carries its own copy of InferKit (it defines %@); a plugin links InferKit's interface with -undefined dynamic_lookup or the app's InferKit framework",
							  manifest.identifier, [duplicates componentsJoinedByString:@", "]]);
}

#pragma mark C

/*! Sets *codeLoaded once the library has entered the process, so a failure after that point is
	remembered. */
+ (nullable id<NFKPlugin>)CPluginWithManifest:(NFKPluginManifest *)manifest
										policy:(NFKPluginSignaturePolicy *)policy
									   context:(NFKPluginHostContext *)context
									codeLoaded:(BOOL *)codeLoaded
										 error:(NSError **)error
{
	*codeLoaded = NO;
	NSURL *library = [manifest containedExecutableURLWithError:error];
	if (library == nil || ![policy validateCodeAtURL:manifest.bundleURL error:error]) {
		return nil;
	}
	// An in-process plugin is never closed: code the process has called cannot be unloaded safely.
	void *handle = dlopen(library.fileSystemRepresentation, RTLD_NOW | RTLD_LOCAL);
	if (handle == NULL) {
		const char *reason = dlerror();
		NFKPluginSetError(error, NFKPluginErrorLoadFailed,
						  [NSString stringWithFormat:@"the plugin %@'s library did not load: %s", manifest.identifier, reason != NULL ? reason : "unknown reason"]);
		return nil;
	}
	*codeLoaded = YES;
	const char *image = NFKPluginLoadedImageName(library);
	if (image != NULL && ![self checkSingleRuntimeInImage:image manifest:manifest error:error]) {
		return nil;
	}
	NSString *symbol = manifest.entrySymbol ?: @NFK_PLUGIN_ENTRY_SYMBOL;
	nfk_plugin_entry_function entry = (nfk_plugin_entry_function)dlsym(handle, symbol.UTF8String);
	if (entry == NULL) {
		NFKPluginSetError(error, NFKPluginErrorLoadFailed,
						  [NSString stringWithFormat:@"the plugin %@'s library exports no %@", manifest.identifier, symbol]);
		return nil;
	}
	return [[NFKPluginCAdapter alloc] initWithEntry:entry manifest:manifest context:context error:error];
}

@end

#endif
