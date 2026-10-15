//
//  NFKPluginFixtures.m
//  InferKitTests
//

#import "NFKPluginFixtures.h"

#if TARGET_OS_OSX

#import <CommonCrypto/CommonDigest.h>
#import <errno.h>
#import <stdio.h>

NSString * const NFKPluginFixturesErrorDomain = @"NFKPluginFixturesErrorDomain";
NSString * const NFKPluginFixtureInvertModel = @"invert";
NSString * const NFKPluginFixtureSlowModel = @"slow";
NSString * const NFKPluginFixtureEchoTool = @"echo";
NSString * const NFKPluginFixtureEchoPrefixOption = @"echoPrefix";
NSString * const NFKPluginFixtureValuesKey = @"values";
NSString * const NFKPluginFixtureVersionKey = @"pluginVersion";
NSString * const NFKPluginFixtureStepsKey = @"steps";
NSString * const NFKPluginFixtureStepMillisecondsKey = @"stepMilliseconds";
NSString * const NFKPluginFixtureExitAtStepKey = @"exitAtStep";
NSString * const NFKPluginFixtureAllocationCounterSymbol = @"sample_plugin_live_allocations";

/*! Changes whenever the build steps change in a way the hashed inputs do not show. */
static NSString * const NFKPluginFixturesRecipeRevision = @"1";

static NSString * const NFKPluginFixturesDuplicateRuntime = @"duplicate-runtime";

static NSString * const NFKPluginFixturesManifestPath = @"Contents/Resources/inferkit-plugin.json";

static const NSTimeInterval NFKPluginFixturesToolTimeout = 300;

static BOOL NFKPluginFixturesSetError(NSError **error, NSString *reason)
{
	if (error != NULL) {
		*error = [NSError errorWithDomain:NFKPluginFixturesErrorDomain code:1 userInfo:@{ NSLocalizedDescriptionKey: reason }];
	}
	return NO;
}

static NSString *NFKPluginFixturesArchitecture(void)
{
#if defined(__arm64__)
	return @"arm64";
#else
	return @"x86_64";
#endif
}

@implementation NFKPluginFixtures

+ (NSURL *)sourceRootURL
{
	NSURL *file = [NSURL fileURLWithPath:@__FILE__];
	return file.URLByDeletingLastPathComponent.URLByDeletingLastPathComponent.URLByDeletingLastPathComponent;
}

+ (NSString *)pluginIdentifierForKind:(NFKPluginEntryKind)kind
{
	return [self recipeNamed:kind][@"identifier"] ?: @"";
}

#pragma mark Recipes

/*! The compiler arguments shared by both Objective-C bundles. $ROOT, $OUT, $ARCH, and $VERSION are
	substituted at build time; the hash reads them unsubstituted. */
+ (NSArray<NSString *> *)bundleArgumentsForSource:(NSString *)source
{
	return @[ @"clang", @"-bundle", @"-fobjc-arc", @"-undefined", @"dynamic_lookup",
			  @"-I", @"$ROOT/Sources/InferKit/include", @"-I", @"$ROOT/Sources/InferKitC/include",
			  @"-arch", @"$ARCH", @"-mmacosx-version-min=11.0", @"-Wall", @"-Wextra", @"-Wno-unused-parameter", @"-Werror",
			  @"-DSAMPLE_PLUGIN_VERSION=\"$VERSION\"", @"-framework", @"Foundation", @"-framework", @"CoreVideo",
			  @"-o", @"$OUT", [@"$ROOT/Tests/PluginFixtures/" stringByAppendingString:source] ];
}

/*! How to build each fixture: its sources, its templates, its names, and its compiler arguments. */
+ (nullable NSDictionary<NSString *, id> *)recipeNamed:(NSString *)name
{
	NSDictionary *recipes = @{
		NFKPluginEntryKindObjC: @{
			@"identifier": @"com.example.inferkit.sample-objc",
			@"bundleName": @"SampleObjCPlugin.inferkitplugin",
			@"executable": @"SampleObjCPlugin",
			@"principalClass": @"SamplePlugin",
			@"infoTemplate": @"objc/Info.plist.in",
			@"manifestTemplate": @"manifests/objc.json.in",
			@"source": @"objc/SamplePlugin.m",
			@"readsCoreHeaders": @YES,
			@"arguments": [self bundleArgumentsForSource:@"objc/SamplePlugin.m"],
		},
		NFKPluginFixturesDuplicateRuntime: @{
			@"identifier": @"com.example.inferkit.sample-duplicate-runtime",
			@"bundleName": @"SampleDuplicateRuntimePlugin.inferkitplugin",
			@"executable": @"DuplicateRuntimePlugin",
			@"principalClass": @"DuplicateRuntimePlugin",
			@"infoTemplate": @"objc/Info.plist.in",
			@"manifestTemplate": @"manifests/objc.json.in",
			@"source": @"objc/DuplicateRuntimePlugin.m",
			@"readsCoreHeaders": @YES,
			@"arguments": [self bundleArgumentsForSource:@"objc/DuplicateRuntimePlugin.m"],
		},
		NFKPluginEntryKindCABI: @{
			@"identifier": @"com.example.inferkit.sample-c",
			@"bundleName": @"SampleCPlugin.inferkitplugin",
			@"executable": @"SampleCPlugin",
			@"infoTemplate": @"c/Info.plist.in",
			@"manifestTemplate": @"manifests/c-abi.json.in",
			@"source": @"c/sample_plugin.c",
			@"readsCoreHeaders": @NO,
			@"arguments": @[ @"clang", @"-dynamiclib", @"-std=c99", @"-pedantic", @"-fvisibility=hidden",
							 @"-Wall", @"-Wextra", @"-Wno-unused-parameter", @"-Werror", @"-I", @"$ROOT/Sources/InferKitC/include",
							 @"-arch", @"$ARCH", @"-mmacosx-version-min=11.0", @"-install_name", @"@rpath/SampleCPlugin",
							 @"-DSAMPLE_PLUGIN_VERSION=\"$VERSION\"", @"-o", @"$OUT", @"$ROOT/Tests/PluginFixtures/c/sample_plugin.c" ],
		},
		NFKPluginEntryKindExecutable: @{
			@"identifier": @"com.example.inferkit.sample-executable",
			@"bundleName": @"SampleExecutablePlugin.inferkitplugin",
			@"executable": @"SampleExecutablePlugin",
			@"infoTemplate": @"executable/Info.plist.in",
			@"manifestTemplate": @"manifests/executable.json.in",
			@"source": @"executable/sample_executable.m",
			@"readsCoreHeaders": @NO,
			@"arguments": @[ @"clang", @"-fobjc-arc", @"-arch", @"$ARCH", @"-mmacosx-version-min=11.0",
							 @"-Wall", @"-Wextra", @"-Wno-unused-parameter", @"-Werror", @"-DSAMPLE_PLUGIN_VERSION=\"$VERSION\"",
							 @"-framework", @"Foundation", @"-o", @"$OUT", @"$ROOT/Tests/PluginFixtures/executable/sample_executable.m" ],
		},
	};
	return recipes[name];
}

#pragma mark Building

+ (nullable NSURL *)bundleURLForKind:(NFKPluginEntryKind)kind version:(NSString *)version error:(NSError **)error
{
	if ([kind isEqualToString:NFKPluginFixturesDuplicateRuntime] || [self recipeNamed:kind] == nil) {
		NFKPluginFixturesSetError(error, [NSString stringWithFormat:@"no fixture of kind \"%@\"", kind]);
		return nil;
	}
	return [self buildFixtureNamed:kind version:version error:error];
}

+ (nullable NSURL *)duplicateRuntimeBundleURLWithError:(NSError **)error
{
	return [self buildFixtureNamed:NFKPluginFixturesDuplicateRuntime version:@"1.0.0" error:error];
}

+ (BOOL)isPlainVersion:(NSString *)version
{
	NSCharacterSet *allowed = [NSCharacterSet characterSetWithCharactersInString:@"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.-+"];
	return version.length > 0 && [version rangeOfCharacterFromSet:allowed.invertedSet].location == NSNotFound;
}

+ (nullable NSURL *)buildFixtureNamed:(NSString *)name version:(NSString *)version error:(NSError **)error
{
	if (![self isPlainVersion:version]) {
		NFKPluginFixturesSetError(error, [NSString stringWithFormat:@"the version \"%@\" is not a plain version string", version]);
		return nil;
	}
	@synchronized (self) {
		NSDictionary *recipe = [self recipeNamed:name];
		NSString *digest = [self digestOfRecipe:recipe named:name version:version error:error];
		if (digest == nil) {
			return nil;
		}
		NSURL *temporary = [NSURL fileURLWithPath:NSTemporaryDirectory() isDirectory:YES];
		NSString *cacheName = [@"inferkit-plugin-fixtures-" stringByAppendingString:digest];
		NSURL *cache = [temporary URLByAppendingPathComponent:cacheName isDirectory:YES];
		NSURL *bundle = [cache URLByAppendingPathComponent:recipe[@"bundleName"] isDirectory:YES];
		if ([NSFileManager.defaultManager fileExistsAtPath:[bundle URLByAppendingPathComponent:NFKPluginFixturesManifestPath].path]) {
			return bundle;
		}
		NSString *stagingName = [NSString stringWithFormat:@"%@.staging-%@", cacheName, NSUUID.UUID.UUIDString];
		NSURL *staging = [temporary URLByAppendingPathComponent:stagingName isDirectory:YES];
		BOOL built = [self buildRecipe:recipe version:version intoDirectory:staging error:error];
		if (built && [self moveStagingDirectory:staging toCacheDirectory:cache error:error]) {
			return bundle;
		}
		[NSFileManager.defaultManager removeItemAtURL:staging error:NULL];
		return nil;
	}
}

/*! Renames the finished staging directory into place in one step, so a cache directory that exists is
	complete. Another process that finished the same build first wins, and its copy is used. */
+ (BOOL)moveStagingDirectory:(NSURL *)staging toCacheDirectory:(NSURL *)cache error:(NSError **)error
{
	if (rename(staging.fileSystemRepresentation, cache.fileSystemRepresentation) == 0) {
		return YES;
	}
	if ([NSFileManager.defaultManager fileExistsAtPath:cache.path]) {
		[NSFileManager.defaultManager removeItemAtURL:staging error:NULL];
		return YES;
	}
	return NFKPluginFixturesSetError(error, [NSString stringWithFormat:@"the fixture could not be moved into %@: %s", cache.path, strerror(errno)]);
}

+ (NSString *)substitute:(NSString *)text recipe:(NSDictionary *)recipe version:(NSString *)version
{
	NSDictionary<NSString *, NSString *> *values = @{
		@"@IDENTIFIER@": recipe[@"identifier"],
		@"@VERSION@": version,
		@"@EXECUTABLE@": recipe[@"executable"],
		@"@PRINCIPAL_CLASS@": recipe[@"principalClass"] ?: @"",
		@"@ARCH@": NFKPluginFixturesArchitecture(),
	};
	NSMutableString *result = [text mutableCopy];
	[values enumerateKeysAndObjectsUsingBlock:^(NSString *placeholder, NSString *value, BOOL *stop) {
		[result replaceOccurrencesOfString:placeholder withString:value options:0 range:NSMakeRange(0, result.length)];
	}];
	return result;
}

+ (NSArray<NSString *> *)argumentsOfRecipe:(NSDictionary *)recipe version:(NSString *)version output:(NSString *)output
{
	NSMutableArray<NSString *> *arguments = [NSMutableArray array];
	for (NSString *argument in recipe[@"arguments"]) {
		NSString *expanded = [argument stringByReplacingOccurrencesOfString:@"$ROOT" withString:self.sourceRootURL.path];
		expanded = [expanded stringByReplacingOccurrencesOfString:@"$OUT" withString:output];
		expanded = [expanded stringByReplacingOccurrencesOfString:@"$ARCH" withString:NFKPluginFixturesArchitecture()];
		expanded = [expanded stringByReplacingOccurrencesOfString:@"$VERSION" withString:version];
		[arguments addObject:expanded];
	}
	return arguments;
}

+ (BOOL)writeTemplate:(NSString *)templatePath recipe:(NSDictionary *)recipe version:(NSString *)version
				toURL:(NSURL *)destination error:(NSError **)error
{
	NSURL *source = [self.sourceRootURL URLByAppendingPathComponent:[@"Tests/PluginFixtures/" stringByAppendingString:templatePath]];
	NSString *text = [NSString stringWithContentsOfURL:source encoding:NSUTF8StringEncoding error:error];
	if (text == nil) {
		return NO;
	}
	NSString *filled = [self substitute:text recipe:recipe version:version];
	return [filled writeToURL:destination atomically:NO encoding:NSUTF8StringEncoding error:error];
}

/*! Compiles the fixture into a .inferkitplugin bundle inside directory, writes its Info.plist and
	manifest, signs it ad hoc, and records the commands beside it. */
+ (BOOL)buildRecipe:(NSDictionary *)recipe version:(NSString *)version intoDirectory:(NSURL *)directory error:(NSError **)error
{
	NSFileManager *files = NSFileManager.defaultManager;
	NSURL *bundle = [directory URLByAppendingPathComponent:recipe[@"bundleName"] isDirectory:YES];
	NSURL *contents = [bundle URLByAppendingPathComponent:@"Contents" isDirectory:YES];
	NSURL *executables = [contents URLByAppendingPathComponent:@"MacOS" isDirectory:YES];
	NSURL *resources = [contents URLByAppendingPathComponent:@"Resources" isDirectory:YES];
	if (![files createDirectoryAtURL:executables withIntermediateDirectories:YES attributes:nil error:error]
		|| ![files createDirectoryAtURL:resources withIntermediateDirectories:YES attributes:nil error:error]) {
		return NO;
	}
	NSString *output = [executables URLByAppendingPathComponent:recipe[@"executable"]].path;
	NSArray<NSString *> *compile = [self argumentsOfRecipe:recipe version:version output:output];
	NSArray<NSString *> *sign = @[ @"--force", @"--sign", @"-", bundle.path ];
	BOOL written = [self runProgram:@"/usr/bin/xcrun" arguments:compile error:error] != nil
		&& [self writeTemplate:recipe[@"infoTemplate"] recipe:recipe version:version
						 toURL:[contents URLByAppendingPathComponent:@"Info.plist"] error:error]
		&& [self writeTemplate:recipe[@"manifestTemplate"] recipe:recipe version:version
						 toURL:[bundle URLByAppendingPathComponent:NFKPluginFixturesManifestPath] error:error]
		&& [self runProgram:@"/usr/bin/codesign" arguments:sign error:error] != nil;
	if (!written) {
		return NO;
	}
	NSString *log = [NSString stringWithFormat:@"xcrun %@\ncodesign %@\n", [compile componentsJoinedByString:@" "], [sign componentsJoinedByString:@" "]];
	return [log writeToURL:[directory URLByAppendingPathComponent:@"build-commands.txt"] atomically:NO encoding:NSUTF8StringEncoding error:error];
}

#pragma mark Hashing

/*! Every file whose contents a build reads, relative to the source root, sorted. */
+ (nullable NSArray<NSString *> *)inputPathsOfRecipe:(NSDictionary *)recipe error:(NSError **)error
{
	NSMutableArray<NSString *> *paths = [NSMutableArray arrayWithObjects:
		[@"Tests/PluginFixtures/" stringByAppendingString:recipe[@"source"]],
		[@"Tests/PluginFixtures/" stringByAppendingString:recipe[@"infoTemplate"]],
		[@"Tests/PluginFixtures/" stringByAppendingString:recipe[@"manifestTemplate"]], nil];
	NSMutableArray<NSString *> *headerDirectories = [NSMutableArray arrayWithObject:@"Sources/InferKitC/include/InferKitC"];
	if ([recipe[@"readsCoreHeaders"] boolValue]) {
		[headerDirectories addObject:@"Sources/InferKit/include/InferKit"];
	}
	for (NSString *directory in headerDirectories) {
		NSURL *url = [self.sourceRootURL URLByAppendingPathComponent:directory isDirectory:YES];
		NSArray<NSString *> *names = [NSFileManager.defaultManager contentsOfDirectoryAtPath:url.path error:error];
		if (names == nil) {
			return nil;
		}
		for (NSString *name in names) {
			if ([name.pathExtension isEqualToString:@"h"]) {
				[paths addObject:[directory stringByAppendingPathComponent:name]];
			}
		}
	}
	return [paths sortedArrayUsingSelector:@selector(compare:)];
}

/*! A SHA-256 prefix over the recipe, the version, the architecture, and every input file's path and
	contents. The source root's own path is left out, so two checkouts of the same sources share a build. */
+ (nullable NSString *)digestOfRecipe:(NSDictionary *)recipe named:(NSString *)name version:(NSString *)version error:(NSError **)error
{
	NSArray<NSString *> *paths = [self inputPathsOfRecipe:recipe error:error];
	if (paths == nil) {
		return nil;
	}
	NSMutableData *material = [NSMutableData data];
	NSArray<NSString *> *header = @[ NFKPluginFixturesRecipeRevision, name, version, NFKPluginFixturesArchitecture(),
									 [recipe[@"arguments"] componentsJoinedByString:@"\x1f"] ];
	[material appendData:[[header componentsJoinedByString:@"\x1e"] dataUsingEncoding:NSUTF8StringEncoding]];
	for (NSString *path in paths) {
		NSData *contents = [NSData dataWithContentsOfURL:[self.sourceRootURL URLByAppendingPathComponent:path] options:0 error:error];
		if (contents == nil) {
			return nil;
		}
		[material appendData:[[NSString stringWithFormat:@"\x1e%@\x1e%lu\x1e", path, (unsigned long)contents.length] dataUsingEncoding:NSUTF8StringEncoding]];
		[material appendData:contents];
	}
	uint8_t digest[CC_SHA256_DIGEST_LENGTH];
	CC_SHA256(material.bytes, (CC_LONG)material.length, digest);
	NSMutableString *hex = [NSMutableString string];
	for (NSUInteger index = 0; index < 8; index++) {
		[hex appendFormat:@"%02x", digest[index]];
	}
	return hex;
}

#pragma mark Other fixtures

+ (nullable NSURL *)hostExecutableURL
{
	NSString *override = NSProcessInfo.processInfo.environment[@"NFK_PLUGIN_HOST_PATH"];
	NSURL *candidate = override.length > 0
		? [NSURL fileURLWithPath:override]
		: [[NSBundle bundleForClass:self].bundleURL.URLByDeletingLastPathComponent URLByAppendingPathComponent:@"inferkit-plugin-host"];
	return [NSFileManager.defaultManager isExecutableFileAtPath:candidate.path] ? candidate : nil;
}

+ (nullable NSURL *)tamperedCopyOfBundleAtURL:(NSURL *)bundleURL error:(NSError **)error
{
	NSString *directoryName = [@"inferkit-plugin-tampered-" stringByAppendingString:NSUUID.UUID.UUIDString];
	NSURL *directory = [[NSURL fileURLWithPath:NSTemporaryDirectory() isDirectory:YES] URLByAppendingPathComponent:directoryName isDirectory:YES];
	NSURL *copy = [directory URLByAppendingPathComponent:bundleURL.lastPathComponent isDirectory:YES];
	NSFileManager *files = NSFileManager.defaultManager;
	if (![files createDirectoryAtURL:directory withIntermediateDirectories:YES attributes:nil error:error]
		|| ![files copyItemAtURL:bundleURL toURL:copy error:error]) {
		return nil;
	}
	NSURL *manifestURL = [copy URLByAppendingPathComponent:NFKPluginFixturesManifestPath];
	NSData *data = [NSData dataWithContentsOfURL:manifestURL options:0 error:error];
	NSMutableDictionary *manifest = data != nil ? [NSJSONSerialization JSONObjectWithData:data options:NSJSONReadingMutableContainers error:error] : nil;
	if (![manifest isKindOfClass:NSMutableDictionary.class]) {
		NFKPluginFixturesSetError(error, @"the manifest to tamper with is not a JSON object");
		return nil;
	}
	NSString *summary = [manifest[@"summary"] isKindOfClass:NSString.class] ? manifest[@"summary"] : @"";
	manifest[@"summary"] = [summary stringByAppendingString:@" Edited after signing."];
	NSData *edited = [NSJSONSerialization dataWithJSONObject:manifest options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys error:error];
	return edited != nil && [edited writeToURL:manifestURL options:0 error:error] ? copy : nil;
}

+ (nullable NSURL *)executableURLOfBundleAtURL:(NSURL *)bundleURL
{
	NSURL *contents = [bundleURL URLByAppendingPathComponent:@"Contents" isDirectory:YES];
	NSDictionary *information = [NSDictionary dictionaryWithContentsOfURL:[contents URLByAppendingPathComponent:@"Info.plist"]];
	NSString *name = [information[@"CFBundleExecutable"] isKindOfClass:NSString.class] ? information[@"CFBundleExecutable"] : nil;
	return name != nil ? [[contents URLByAppendingPathComponent:@"MacOS" isDirectory:YES] URLByAppendingPathComponent:name] : nil;
}

#pragma mark Running tools

/*! The toolchain's codesign_allocate, found once with xcrun, or nil. codesign runs codesign_allocate to
	make room for a signature, through the /usr/bin shim unless CODESIGN_ALLOCATE names it, and the shim
	resolves the toolchain through xcrun on every call. */
+ (nullable NSString *)codesignAllocatePath
{
	static NSString *path = nil;
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		NSString *found = [self outputOfProgram:@"/usr/bin/xcrun" arguments:@[ @"--find", @"codesign_allocate" ] environment:@{} error:NULL];
		found = [[found componentsSeparatedByString:@"\n"].firstObject stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceCharacterSet];
		path = found.length > 0 && [NSFileManager.defaultManager isExecutableFileAtPath:found] ? found : nil;
	});
	return path;
}

+ (nullable NSString *)runProgram:(NSString *)path arguments:(NSArray<NSString *> *)arguments error:(NSError **)error
{
	NSString *allocate = [self codesignAllocatePath];
	return [self outputOfProgram:path arguments:arguments environment:allocate != nil ? @{ @"CODESIGN_ALLOCATE": allocate } : @{}
						   error:error];
}

/*! The processes below pid, one "pid ppid state elapsed command" line each, or an empty string. */
+ (NSString *)descendantsOfProcess:(pid_t)pid
{
	NSString *table = [self outputOfProgram:@"/bin/ps" arguments:@[ @"-A", @"-o", @"pid=,ppid=,stat=,etime=,command=" ]
								environment:@{} error:NULL] ?: @"";
	NSMutableDictionary<NSNumber *, NSNumber *> *parents = [NSMutableDictionary dictionary];
	NSMutableDictionary<NSNumber *, NSString *> *rows = [NSMutableDictionary dictionary];
	for (NSString *row in [table componentsSeparatedByString:@"\n"]) {
		NSArray<NSString *> *fields = [[row stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceCharacterSet]
									   componentsSeparatedByCharactersInSet:NSCharacterSet.whitespaceCharacterSet];
		if (fields.count >= 2) {
			parents[@(fields[0].intValue)] = @([fields[1] intValue]);
			rows[@(fields[0].intValue)] = row;
		}
	}
	NSMutableArray<NSString *> *found = [NSMutableArray array];
	NSMutableArray<NSNumber *> *frontier = [NSMutableArray arrayWithObject:@(pid)];
	while (frontier.count > 0) {
		NSNumber *parent = frontier.lastObject;
		[frontier removeLastObject];
		[parents enumerateKeysAndObjectsUsingBlock:^(NSNumber *child, NSNumber *childParent, BOOL *stop) {
			if ([childParent isEqualToNumber:parent] && child.intValue != parent.intValue) {
				[found addObject:rows[child]];
				[frontier addObject:child];
			}
		}];
	}
	return [found componentsJoinedByString:@"\n"];
}

+ (nullable NSString *)outputOfProgram:(NSString *)path
							 arguments:(NSArray<NSString *> *)arguments
						   environment:(NSDictionary<NSString *, NSString *> *)environment
								 error:(NSError **)error
{
	NFKSubprocess *process = [[NFKSubprocess alloc] initWithCommand:path arguments:arguments];
	process.environment = environment;
	NSMutableArray<NSString *> *lines = [NSMutableArray array];
	dispatch_semaphore_t ended = dispatch_semaphore_create(0);
	process.lineHandler = ^(NSData *line) {
		NSString *text = [[NSString alloc] initWithData:line encoding:NSUTF8StringEncoding] ?: @"";
		@synchronized (lines) {
			[lines addObject:text];
		}
	};
	process.stateHandler = ^(NFKSubprocess *subprocess, NFKSubprocessState state) {
		if (state == NFKSubprocessStateExited || state == NFKSubprocessStateFailedToLaunch) {
			dispatch_semaphore_signal(ended);
		}
	};
	if (![process startWithError:error]) {
		return nil;
	}
	// No tool here reads its input; an open pipe would leave one that does waiting for ever.
	[process closeStandardInput];
	BOOL finished = dispatch_semaphore_wait(ended, dispatch_time(DISPATCH_TIME_NOW, (int64_t)(NFKPluginFixturesToolTimeout * NSEC_PER_SEC))) == 0;
	NSString *stalled = nil;
	if (!finished) {
		stalled = [self descendantsOfProcess:process.processIdentifier];
		[process stop];
	}
	NSString *output;
	@synchronized (lines) {
		NSArray<NSString *> *streams = @[ [lines componentsJoinedByString:@"\n"], process.recentStandardError ];
		output = [streams componentsJoinedByString:@"\n"];
	}
	BOOL succeeded = finished && process.terminationReason == NFKSubprocessTerminationReasonExit && process.terminationStatus == 0;
	if (!succeeded) {
		NSString *command = [[@[ path ] arrayByAddingObjectsFromArray:arguments] componentsJoinedByString:@" "];
		NSString *outcome = finished ? [NSString stringWithFormat:@"ended with status %d", process.terminationStatus]
			: [NSString stringWithFormat:@"timed out; its processes then: %@", stalled.length > 0 ? stalled : @"(none below it)"];
		NFKPluginFixturesSetError(error, [NSString stringWithFormat:@"%@ %@:\n%@", command, outcome, output]);
		return nil;
	}
	return output;
}

@end

#endif
