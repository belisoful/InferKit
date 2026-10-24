//
//  NFKPluginFixturesTests.m
//  InferKitTests
//
//  The sample plugins themselves: each builds as a signed bundle, a broken cached build is built again,
//  a tampered copy fails verification,
//  the C library honors its interface's version and frees everything it allocates, and the executable
//  speaks the protocol over a channel this test drives.
//

#import <XCTest/XCTest.h>
#import <InferKit/InferKit.h>

#if TARGET_OS_OSX

#import "NFKPluginFixtures.h"
#import "../../Sources/InferKit/NFKInferenceWireCoding.h"
#import "../../Sources/InferKit/NFKPluginChannel.h"
#import "../../Sources/InferKit/NFKPluginWire.h"
#import "../../Sources/InferKit/NFKSubprocessPrivate.h"
#import <InferKitC/NFKPluginABI.h>
#import <dlfcn.h>
#import <unistd.h>

#pragma mark - C host

/*! What the test's host table records while the C plugin runs. */
typedef struct NFKFixtureHostState {
	int progressReports;
	int partialReports;
	int cancelAfterReports;
	int logMessages;
} NFKFixtureHostState;

static void NFKFixtureHostLog(void *context, nfk_plugin_log_level level, const char *message)
{
	((NFKFixtureHostState *)context)->logMessages += 1;
}

static void NFKFixtureHostReportProgress(void *context, nfk_plugin_run *run, double fraction, const nfk_plugin_payload *partial)
{
	NFKFixtureHostState *state = context;
	state->progressReports += 1;
	if (partial != NULL && partial->json != NULL && strstr(partial->json, "\"step\"") != NULL) {
		state->partialReports += 1;
	}
}

static int32_t NFKFixtureHostIsCancelled(void *context, nfk_plugin_run *run)
{
	NFKFixtureHostState *state = context;
	return state->cancelAfterReports > 0 && state->progressReports >= state->cancelAfterReports;
}

static const char *NFKFixtureHostOutputDirectory(void *context, nfk_plugin_run *run)
{
	return "/tmp";
}

/*! Copies a string the plugin returned and frees it through the plugin: empty for NULL. */
static NSString *NFKFixtureTakeString(const nfk_plugin *plugin, char * _Nullable string)
{
	if (string == NULL) {
		return @"";
	}
	NSString *copy = [NSString stringWithUTF8String:string] ?: @"";
	plugin->free_string(plugin->context, string);
	return copy;
}

/*! The directory a fixture build renames a broken cache into, beside the cache. */
static NSURL *NFKFixtureDiscardedDirectoryOfCache(NSURL *cache)
{
	return [cache.URLByDeletingLastPathComponent URLByAppendingPathComponent:[cache.lastPathComponent stringByAppendingString:@".discarded"]
																 isDirectory:YES];
}

#pragma mark - Channel replies

/*! Collects what the executable sends: replies by id, and every other message in order. */
@interface NFKFixtureReplies : NSObject
@property (nonatomic, strong) NSMutableDictionary<NSNumber *, NSDictionary *> *replies;
@property (nonatomic, strong) NSMutableDictionary<NSNumber *, NSArray<NFKPluginDescriptor *> *> *replyDescriptors;
@property (nonatomic, strong) NSMutableArray<NSDictionary *> *others;
@end

@implementation NFKFixtureReplies

- (instancetype)initWithChannel:(NFKPluginChannel *)channel
{
	self = [super init];
	if (self != nil) {
		_replies = [NSMutableDictionary dictionary];
		_replyDescriptors = [NSMutableDictionary dictionary];
		_others = [NSMutableArray array];
		channel.messageHandler = ^(NSDictionary *message, NSArray<NFKPluginDescriptor *> *descriptors) {
			@synchronized (self) {
				if ([message[@"type"] isEqual:@"reply"] && [message[@"id"] isKindOfClass:NSNumber.class]) {
					self.replies[message[@"id"]] = message;
					self.replyDescriptors[message[@"id"]] = descriptors;
				} else {
					[self.others addObject:message];
				}
			}
		};
	}
	return self;
}

- (nullable NSDictionary *)replyTo:(NSInteger)identifier timeout:(NSTimeInterval)timeout
{
	NSDate *deadline = [NSDate dateWithTimeIntervalSinceNow:timeout];
	while (deadline.timeIntervalSinceNow > 0) {
		@synchronized (self) {
			NSDictionary *reply = self.replies[@(identifier)];
			if (reply != nil) {
				return reply;
			}
		}
		[NSThread sleepForTimeInterval:0.01];
	}
	return nil;
}

- (NSArray<NFKPluginDescriptor *> *)descriptorsOfReplyTo:(NSInteger)identifier
{
	@synchronized (self) {
		return self.replyDescriptors[@(identifier)] ?: @[];
	}
}

@end

#pragma mark - Tests

@interface NFKPluginFixturesTests : XCTestCase
@end

@implementation NFKPluginFixturesTests

- (BOOL)waitUntil:(BOOL (^)(void))condition timeout:(NSTimeInterval)timeout
{
	NSDate *deadline = [NSDate dateWithTimeIntervalSinceNow:timeout];
	while (!condition()) {
		if (deadline.timeIntervalSinceNow < 0) {
			return NO;
		}
		[NSThread sleepForTimeInterval:0.01];
	}
	return YES;
}

- (nullable NSDictionary *)manifestOfBundleAtURL:(NSURL *)bundleURL
{
	NSData *data = [NSData dataWithContentsOfURL:[bundleURL URLByAppendingPathComponent:@"Contents/Resources/inferkit-plugin.json"]];
	return data != nil ? [NSJSONSerialization JSONObjectWithData:data options:0 error:NULL] : nil;
}

/*! Builds the fixture of kind at version and checks its layout, its names, and its signature. */
- (nullable NSURL *)assertFixtureOfKind:(NFKPluginEntryKind)kind version:(NSString *)version
{
	NSError *error = nil;
	NSURL *bundle = [NFKPluginFixtures bundleURLForKind:kind version:version error:&error];
	XCTAssertNotNil(bundle, @"%@", error.localizedDescription);
	if (bundle == nil) {
		return nil;
	}
	XCTAssertEqualObjects(bundle.pathExtension, @"inferkitplugin");
	NSDictionary *manifest = [self manifestOfBundleAtURL:bundle];
	NSString *identifier = [NFKPluginFixtures pluginIdentifierForKind:kind];
	XCTAssertEqualObjects(manifest[@"id"], identifier);
	XCTAssertEqualObjects(manifest[@"version"], version);
	XCTAssertEqualObjects(manifest[@"entry"][@"kind"], kind);
	NSArray *modelIdentifiers = [manifest[@"models"] valueForKey:@"id"];
	XCTAssertEqualObjects(modelIdentifiers, (@[ NFKPluginFixtureInvertModel, NFKPluginFixtureSlowModel ]));
	XCTAssertEqualObjects([manifest[@"tools"] valueForKey:@"name"], (@[ NFKPluginFixtureEchoTool ]));
	NSDictionary *information = [NSDictionary dictionaryWithContentsOfURL:[bundle URLByAppendingPathComponent:@"Contents/Info.plist"]];
	XCTAssertEqualObjects(information[@"CFBundleIdentifier"], identifier);
	XCTAssertEqualObjects(information[@"CFBundleShortVersionString"], version);
	NSURL *executable = [NFKPluginFixtures executableURLOfBundleAtURL:bundle];
	XCTAssertTrue([NSFileManager.defaultManager isExecutableFileAtPath:executable.path], @"%@", executable);
	NSString *verified = [NFKPluginFixtures runProgram:@"/usr/bin/codesign" arguments:@[ @"--verify", @"--strict", bundle.path ] error:&error];
	XCTAssertNotNil(verified, @"the ad hoc signature verifies: %@", error.localizedDescription);
	return bundle;
}

- (void)testTheObjectiveCFixtureBuildsAsASignedBundle
{
	NSURL *bundle = [self assertFixtureOfKind:NFKPluginEntryKindObjC version:@"1.0.0"];
	NSDictionary *information = [NSDictionary dictionaryWithContentsOfURL:[bundle URLByAppendingPathComponent:@"Contents/Info.plist"]];
	XCTAssertEqualObjects(information[@"NSPrincipalClass"], @"SamplePlugin");
	XCTAssertEqualObjects([self manifestOfBundleAtURL:bundle][@"entry"][@"principalClass"], @"SamplePlugin");
}

- (void)testTheCFixtureBuildsAsASignedBundle
{
	NSURL *bundle = [self assertFixtureOfKind:NFKPluginEntryKindCABI version:@"1.0.0"];
	NSString *executable = [NFKPluginFixtures executableURLOfBundleAtURL:bundle].path ?: @"";
	NSError *error = nil;
	NSString *symbols = [NFKPluginFixtures runProgram:@"/usr/bin/xcrun" arguments:@[ @"nm", @"-gUj", executable ] error:&error];
	XCTAssertNotNil(symbols, @"%@", error.localizedDescription);
	NSArray<NSString *> *exported = [[symbols stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceAndNewlineCharacterSet]
									 componentsSeparatedByString:@"\n"];
	XCTAssertEqualObjects([exported sortedArrayUsingSelector:@selector(compare:)],
						  (@[ @"_nfk_plugin_entry", @"_sample_plugin_live_allocations" ]),
						  @"the library exports its entry and its counter and nothing else");
}

- (void)testTheExecutableFixtureBuildsAsASignedBundle
{
	[self assertFixtureOfKind:NFKPluginEntryKindExecutable version:@"1.0.0"];
}

- (void)testASecondVersionBuildsApartFromTheFirst
{
	NSURL *first = [self assertFixtureOfKind:NFKPluginEntryKindCABI version:@"1.0.0"];
	NSURL *second = [self assertFixtureOfKind:NFKPluginEntryKindCABI version:@"2.0.0"];
	XCTAssertNotEqualObjects(first.URLByDeletingLastPathComponent, second.URLByDeletingLastPathComponent);
	NSError *error = nil;
	NSURL *again = [NFKPluginFixtures bundleURLForKind:NFKPluginEntryKindCABI version:@"1.0.0" error:&error];
	XCTAssertEqualObjects(again, first, @"a second request reuses the cached build");
	XCTAssertNil([NFKPluginFixtures bundleURLForKind:NFKPluginEntryKindCABI version:@"1.0 \"quoted\"" error:&error]);
	XCTAssertEqualObjects(error.domain, NFKPluginFixturesErrorDomain, @"a version that is not plain is refused before any build");
}

- (void)testTheDuplicateRuntimeFixtureDefinesItsOwnInferenceRequest
{
	NSError *error = nil;
	NSURL *bundle = [NFKPluginFixtures duplicateRuntimeBundleURLWithError:&error];
	XCTAssertNotNil(bundle, @"%@", error.localizedDescription);
	NSString *executable = [NFKPluginFixtures executableURLOfBundleAtURL:bundle].path ?: @"";
	NSString *symbols = [NFKPluginFixtures runProgram:@"/usr/bin/xcrun" arguments:@[ @"nm", @"-gUj", executable ] error:&error];
	XCTAssertTrue([symbols containsString:@"_OBJC_CLASS_$_NFKInferenceRequest"], @"%@", symbols ?: error.localizedDescription);
	XCTAssertEqualObjects([self manifestOfBundleAtURL:bundle][@"entry"][@"kind"], NFKPluginEntryKindObjC);
}

- (void)testATamperedCopyNoLongerVerifies
{
	NSError *error = nil;
	NSURL *bundle = [NFKPluginFixtures bundleURLForKind:NFKPluginEntryKindObjC version:@"1.0.0" error:&error];
	XCTAssertNotNil(bundle, @"%@", error.localizedDescription);
	NSURL *tampered = [NFKPluginFixtures tamperedCopyOfBundleAtURL:bundle error:&error];
	XCTAssertNotNil(tampered, @"%@", error);
	XCTAssertNotEqualObjects(tampered, bundle);
	XCTAssertTrue([[self manifestOfBundleAtURL:tampered][@"summary"] hasSuffix:@"Edited after signing."]);
	NSString *verified = [NFKPluginFixtures runProgram:@"/usr/bin/codesign" arguments:@[ @"--verify", @"--strict", tampered.path ] error:&error];
	XCTAssertNil(verified, @"an edited manifest breaks the sealed resources");
	NSArray<NSString *> *verifyOriginal = @[ @"--verify", @"--strict", bundle.path ];
	XCTAssertNotNil([NFKPluginFixtures runProgram:@"/usr/bin/codesign" arguments:verifyOriginal error:NULL], @"the original still verifies");
}

#pragma mark The build cache

/*! A version no other test builds, so the cache directory of a fixture at it belongs to this test. */
- (NSString *)privateVersion
{
	return [@"1.0.0-cache-" stringByAppendingString:NSUUID.UUID.UUIDString];
}

/*! The C fixture at a private version. Its cache and every directory renamed aside from it are removed
	at teardown. */
- (nullable NSURL *)privateCFixtureAtVersion:(NSString *)version
{
	NSURL *bundle = [self assertFixtureOfKind:NFKPluginEntryKindCABI version:version];
	if (bundle == nil) {
		return nil;
	}
	NSURL *cache = bundle.URLByDeletingLastPathComponent;
	[self addTeardownBlock:^{
		NSFileManager *files = NSFileManager.defaultManager;
		[files removeItemAtURL:cache error:NULL];
		[files removeItemAtURL:NFKFixtureDiscardedDirectoryOfCache(cache) error:NULL];
	}];
	return bundle;
}

- (NSUInteger)countOfDirectoriesDiscardedFromCacheOf:(NSURL *)bundle
{
	NSURL *discarded = NFKFixtureDiscardedDirectoryOfCache(bundle.URLByDeletingLastPathComponent);
	return [NSFileManager.defaultManager contentsOfDirectoryAtPath:discarded.path error:NULL].count;
}

- (void)testACachedBuildWhoseFilesWereRemovedIsBuiltAgain
{
	NSString *version = [self privateVersion];
	NSURL *bundle = [self privateCFixtureAtVersion:version];
	if (bundle == nil) {
		return;
	}
	NSFileManager *files = NSFileManager.defaultManager;
	NSArray<NSURL *> *purged = @[ [bundle URLByAppendingPathComponent:@"Contents/Resources/inferkit-plugin.json"],
								   [bundle URLByAppendingPathComponent:@"Contents/Info.plist"],
								   [NFKPluginFixtures executableURLOfBundleAtURL:bundle] ];
	NSError *error = nil;
	for (NSURL *file in purged) {
		XCTAssertTrue([files removeItemAtURL:file error:&error], @"%@", error);
	}
	BOOL isDirectory = NO;
	XCTAssertTrue([files fileExistsAtPath:[bundle URLByAppendingPathComponent:@"Contents/Resources"].path isDirectory:&isDirectory] && isDirectory,
				  @"the directories stay, as the system's temporary-file cleanup leaves them");

	NSURL *rebuilt = [self assertFixtureOfKind:NFKPluginEntryKindCABI version:version];
	XCTAssertEqualObjects(rebuilt, bundle, @"the build lands at the same path");
	XCTAssertEqual([self countOfDirectoriesDiscardedFromCacheOf:bundle], 1u,
				   @"the emptied cache is renamed aside");
}

- (void)testACachedBuildWhoseSignatureNoLongerVerifiesIsBuiltAgain
{
	NSString *version = [self privateVersion];
	NSURL *bundle = [self privateCFixtureAtVersion:version];
	if (bundle == nil) {
		return;
	}
	NSURL *infoURL = [bundle URLByAppendingPathComponent:@"Contents/Info.plist"];
	NSMutableDictionary *information = [NSMutableDictionary dictionaryWithContentsOfURL:infoURL];
	information[@"NFKEditedAfterSigning"] = @YES;
	NSError *error = nil;
	XCTAssertTrue([information writeToURL:infoURL error:&error], @"%@", error);
	NSArray<NSString *> *verify = @[ @"--verify", @"--strict", bundle.path ];
	XCTAssertNil([NFKPluginFixtures runProgram:@"/usr/bin/codesign" arguments:verify error:NULL], @"an edited Info.plist breaks the signature");

	NSURL *rebuilt = [self assertFixtureOfKind:NFKPluginEntryKindCABI version:version];
	XCTAssertEqualObjects(rebuilt, bundle, @"the build lands at the same path");
	XCTAssertNil([NSDictionary dictionaryWithContentsOfURL:infoURL][@"NFKEditedAfterSigning"], @"the Info.plist is the one the build writes");
	XCTAssertEqual([self countOfDirectoriesDiscardedFromCacheOf:bundle], 1u,
				   @"the cache that no longer verifies is renamed aside");
}

#pragma mark The C library

/*! The C fixture's library copied to a path of its own and opened, so its counter starts at zero no
	matter what else in this process opened the fixture, with its entry and its allocation counter. */
- (BOOL)openPrivateCopyOfCLibraryWithEntry:(nfk_plugin_entry_function _Nullable * _Nonnull)outEntry
						   liveAllocations:(int64_t (* _Nullable * _Nonnull)(void))outLiveAllocations
{
	NSError *error = nil;
	NSURL *bundle = [NFKPluginFixtures bundleURLForKind:NFKPluginEntryKindCABI version:@"1.0.0" error:&error];
	XCTAssertNotNil(bundle, @"%@", error.localizedDescription);
	NSURL *library = bundle != nil ? [NFKPluginFixtures executableURLOfBundleAtURL:bundle] : nil;
	NSString *name = [NSString stringWithFormat:@"inferkit-sample-c-%@.dylib", NSUUID.UUID.UUIDString];
	NSURL *copy = [[NSURL fileURLWithPath:NSTemporaryDirectory()] URLByAppendingPathComponent:name];
	if (library == nil || ![NSFileManager.defaultManager copyItemAtURL:library toURL:copy error:&error]) {
		XCTFail(@"the library could not be copied: %@", error);
		return NO;
	}
	void *handle = dlopen(copy.fileSystemRepresentation, RTLD_NOW | RTLD_LOCAL);
	[NSFileManager.defaultManager removeItemAtURL:copy error:NULL];
	*outEntry = handle != NULL ? (nfk_plugin_entry_function)dlsym(handle, NFK_PLUGIN_ENTRY_SYMBOL) : NULL;
	*outLiveAllocations = handle != NULL ? (int64_t (*)(void))dlsym(handle, NFKPluginFixtureAllocationCounterSymbol.UTF8String) : NULL;
	if (*outEntry == NULL || *outLiveAllocations == NULL) {
		XCTFail(@"the library does not export its entry and its counter: %s", dlerror());
		return NO;
	}
	return YES;
}

- (nfk_plugin_host)hostWithState:(NFKFixtureHostState *)state
{
	nfk_plugin_host host;
	memset(&host, 0, sizeof host);
	host.struct_size = (uint32_t)sizeof host;
	host.abi_version = NFK_PLUGIN_ABI_VERSION;
	host.context = state;
	host.inferkit_api_version = NFK_API_VERSION;
	host.options_json = "{\"echoPrefix\":\"> \"}";
	host.temporary_directory = "/tmp";
	host.log = NFKFixtureHostLog;
	host.report_progress = NFKFixtureHostReportProgress;
	host.is_cancelled = NFKFixtureHostIsCancelled;
	host.output_directory = NFKFixtureHostOutputDirectory;
	return host;
}

- (void)testTheCEntryRefusesAnotherHostMajorVersion
{
	nfk_plugin_entry_function entry = NULL;
	int64_t (*liveAllocations)(void) = NULL;
	if (![self openPrivateCopyOfCLibraryWithEntry:&entry liveAllocations:&liveAllocations]) {
		return;
	}
	NFKFixtureHostState state = { 0 };
	nfk_plugin_host host = [self hostWithState:&state];
	nfk_plugin sentinel;
	const nfk_plugin *plugin = &sentinel;
	uint32_t nextMajor = (uint32_t)(NFK_PLUGIN_ABI_VERSION_MAJOR + 1) << 16;
	XCTAssertEqual(entry(nextMajor, &host, &plugin), NFK_PLUGIN_STATUS_INCOMPATIBLE);
	XCTAssertTrue(plugin == NULL, @"a refused entry writes no table");
	XCTAssertEqual(entry(NFK_PLUGIN_ABI_VERSION_MAJOR << 16 | 7, &host, &plugin), NFK_PLUGIN_STATUS_OK, @"a later minor version is accepted");
	if (plugin == NULL) {
		XCTFail(@"an accepted entry writes its table");
		return;
	}
	plugin->shutdown(plugin->context);
	XCTAssertEqual(liveAllocations(), 0);
}

- (void)testTheCAllocationCounterReturnsToZero
{
	nfk_plugin_entry_function entry = NULL;
	int64_t (*liveAllocations)(void) = NULL;
	if (![self openPrivateCopyOfCLibraryWithEntry:&entry liveAllocations:&liveAllocations]) {
		return;
	}
	XCTAssertEqual(liveAllocations(), 0);
	NFKFixtureHostState state = { 0 };
	nfk_plugin_host host = [self hostWithState:&state];
	const nfk_plugin *plugin = NULL;
	XCTAssertEqual(entry(NFK_PLUGIN_ABI_VERSION, &host, &plugin), NFK_PLUGIN_STATUS_OK);
	if (plugin == NULL) {
		XCTFail(@"the entry writes its table");
		return;
	}
	void *context = plugin->context;

	char *description = NULL;
	XCTAssertEqual(plugin->describe(context, &description), NFK_PLUGIN_STATUS_OK);
	NSData *describedJSON = [NFKFixtureTakeString(plugin, description) dataUsingEncoding:NSUTF8StringEncoding];
	NSDictionary *described = [NSJSONSerialization JSONObjectWithData:describedJSON options:0 error:NULL];
	XCTAssertEqualObjects([described[@"models"] valueForKey:@"id"], (@[ NFKPluginFixtureInvertModel, NFKPluginFixtureSlowModel ]));

	nfk_plugin_backend *backend = NULL;
	char *errorJSON = NULL;
	XCTAssertEqual(plugin->create_backend(context, "missing", "{}", &backend, &errorJSON), NFK_PLUGIN_STATUS_UNKNOWN_MODEL);
	XCTAssertTrue([NFKFixtureTakeString(plugin, errorJSON) containsString:@"\"code\":14"]);
	XCTAssertEqual(plugin->create_backend(context, "invert", "{}", &backend, &errorJSON), NFK_PLUGIN_STATUS_OK);

	uint8_t pixels[64 * 64 * 4];
	for (size_t index = 0; index < sizeof pixels; index++) {
		pixels[index] = (uint8_t)index;
	}
	nfk_plugin_buffer image;
	memset(&image, 0, sizeof image);
	image.struct_size = (uint32_t)sizeof image;
	image.kind = NFK_PLUGIN_BUFFER_BYTES;
	image.bytes = pixels;
	image.length = sizeof pixels;
	image.descriptor = -1;
	nfk_plugin_payload request;
	memset(&request, 0, sizeof request);
	request.struct_size = (uint32_t)sizeof request;
	request.json = "{\"inputs\":{\"image\":{\"$nfk\":\"buffer\",\"index\":0,\"as\":\"pixelBuffer\",\"format\":1111970369,"
		"\"width\":64,\"height\":64,\"planes\":[{\"offset\":0,\"bytesPerRow\":256,\"rows\":64}],\"length\":16384},"
		"\"values\":{\"$nfk\":\"float32Array\",\"base64\":\"AACAPgAAAEA=\"}},\"parameters\":{}}";
	request.buffer_count = 1;
	request.buffers = &image;
	nfk_plugin_payload *result = NULL;
	XCTAssertEqual(plugin->run(context, backend, (nfk_plugin_run *)&state, &request, &result, &errorJSON), NFK_PLUGIN_STATUS_OK);
	if (result == NULL || result->buffer_count != 1) {
		XCTFail(@"the run returns one attachment: %@", NFKFixtureTakeString(plugin, errorJSON));
		return;
	}
	const uint8_t *inverted = result->buffers[0].bytes;
	XCTAssertEqual(result->buffers[0].length, sizeof pixels);
	XCTAssertEqual(inverted[0], 255 - pixels[0]);
	XCTAssertEqual(inverted[3], pixels[3], @"alpha is kept");
	XCTAssertEqual(inverted[sizeof pixels - 2], 255 - pixels[sizeof pixels - 2]);
	NSString *resultJSON = [NSString stringWithUTF8String:result->json] ?: @"";
	XCTAssertTrue([resultJSON containsString:@"\"base64\":\"AABAPwAAgL8=\""], @"0.25 and 2 become 0.75 and -1: %@", resultJSON);
	XCTAssertTrue([resultJSON containsString:@"\"pluginVersion\":\"1.0.0\""]);
	plugin->free_payload(context, result);

	char *backendDescription = NULL;
	XCTAssertEqual(plugin->describe_backend(context, backend, &backendDescription), NFK_PLUGIN_STATUS_OK);
	XCTAssertTrue([NFKFixtureTakeString(plugin, backendDescription) containsString:@"\"loaded\":true"]);
	plugin->unload(context, backend);
	plugin->destroy_backend(context, backend);

	XCTAssertEqual(plugin->create_backend(context, "slow", "{}", &backend, &errorJSON), NFK_PLUGIN_STATUS_OK);
	request.json = "{\"inputs\":{},\"parameters\":{\"steps\":5,\"stepMilliseconds\":1}}";
	request.buffer_count = 0;
	request.buffers = NULL;
	XCTAssertEqual(plugin->run(context, backend, (nfk_plugin_run *)&state, &request, &result, &errorJSON), NFK_PLUGIN_STATUS_OK);
	if (result == NULL) {
		XCTFail(@"the slow run returns a result: %@", NFKFixtureTakeString(plugin, errorJSON));
		return;
	}
	XCTAssertEqual(state.progressReports, 5);
	XCTAssertEqual(state.partialReports, 5, @"each step reports a partial result");
	plugin->free_payload(context, result);
	state.progressReports = 0;
	state.cancelAfterReports = 2;
	request.json = "{\"inputs\":{},\"parameters\":{\"steps\":1000,\"stepMilliseconds\":1}}";
	XCTAssertEqual(plugin->run(context, backend, (nfk_plugin_run *)&state, &request, &result, &errorJSON), NFK_PLUGIN_STATUS_CANCELLED);
	XCTAssertEqual(state.progressReports, 2, @"the run stops at the first step after it is cancelled");
	XCTAssertTrue([NFKFixtureTakeString(plugin, errorJSON) containsString:@"cancelled"]);
	plugin->destroy_backend(context, backend);

	char *toolResult = NULL;
	XCTAssertEqual(plugin->call_tool(context, "echo", "{\"text\":\"hi\"}", &toolResult, &errorJSON), NFK_PLUGIN_STATUS_OK);
	NSString *echoed = NFKFixtureTakeString(plugin, toolResult);
	XCTAssertTrue([echoed containsString:@"\"text\":\"> hi\""], @"%@", echoed);
	XCTAssertEqual(plugin->call_tool(context, "missing", "{}", &toolResult, &errorJSON), NFK_PLUGIN_STATUS_UNKNOWN_TOOL);
	XCTAssertTrue([NFKFixtureTakeString(plugin, errorJSON) containsString:@"\"code\":15"]);

	plugin->shutdown(context);
	XCTAssertEqual(liveAllocations(), 0, @"describe, create, run, free, and destroy leave nothing allocated");
	XCTAssertEqual(state.logMessages, 2, @"the plugin logs its start and its shutdown");
}

#pragma mark The executable

/*! Launches the executable fixture with one end of a socket pair as descriptor 3, and accepts its
	hand-off. */
- (nullable NFKPluginChannel *)launchExecutableFixture:(NFKSubprocess * _Nullable * _Nonnull)outProcess
{
	NSError *error = nil;
	NSURL *bundle = [NFKPluginFixtures bundleURLForKind:NFKPluginEntryKindExecutable version:@"1.0.0" error:&error];
	XCTAssertNotNil(bundle, @"%@", error.localizedDescription);
	NSURL *executable = [NFKPluginFixtures executableURLOfBundleAtURL:bundle];
	int ours = -1;
	int theirs = -1;
	XCTAssertTrue([NFKPluginChannel createSocketPair:&ours second:&theirs error:&error], @"%@", error);
	NFKSubprocess *process = [[NFKSubprocess alloc] initWithCommand:executable.path ?: @"" arguments:@[]];
	process.environment = @{ NFKPluginChannelEnvironmentVariable: [NSString stringWithFormat:@"%d", NFKPluginChannelDescriptorNumber] };
	process.ownerInheritedDescriptors = @{ @(NFKPluginChannelDescriptorNumber): @(theirs) };
	XCTAssertTrue([process startWithError:&error], @"%@", error);
	process.ownerInheritedDescriptors = nil;
	close(theirs);
	*outProcess = process;
	NFKPluginChannel *channel = [NFKPluginChannel channelByAcceptingHandOffOnDescriptor:ours timeout:10 error:&error];
	XCTAssertNotNil(channel, @"%@", error);
	return channel;
}

- (NSDictionary *)helloMessage
{
	return @{ @"type": @"hello", @"id": @1, @"protocol": @{ @"major": @1, @"minor": @0 }, @"inferkitAPI": @"1.0",
			  @"capabilities": @[ @"attachments" ], @"options": @{ NFKPluginFixtureEchoPrefixOption: @"> " },
			  @"limits": @{ @"inlineBytes": @(NFKPluginDefaultInlineByteLimit) } };
}

- (void)shutDownExecutable:(NFKSubprocess *)process channel:(NFKPluginChannel *)channel replies:(NFKFixtureReplies *)replies identifier:(NSInteger)identifier
{
	NSError *error = nil;
	NSDictionary *shutdown = @{ @"type": @"shutdown", @"id": @(identifier) };
	XCTAssertTrue([channel sendMessage:shutdown error:&error], @"%@", error);
	XCTAssertNotNil([replies replyTo:identifier timeout:10][@"result"]);
	XCTAssertTrue([self waitUntil:^BOOL { return process.state == NFKSubprocessStateExited; } timeout:10]);
	XCTAssertEqual(process.terminationReason, NFKSubprocessTerminationReasonExit);
	XCTAssertEqual(process.terminationStatus, 0);
	[channel close];
}

- (void)testTheExecutableAnswersHello
{
	NFKSubprocess *process = nil;
	NFKPluginChannel *channel = [self launchExecutableFixture:&process];
	if (channel == nil) {
		[process stop];
		return;
	}
	XCTAssertEqual(channel.peerProcessIdentifier, process.processIdentifier, @"the handed socket names the plugin's process");
	NFKFixtureReplies *replies = [[NFKFixtureReplies alloc] initWithChannel:channel];
	[channel resume];
	NSError *error = nil;
	XCTAssertTrue([channel sendMessage:[self helloMessage] error:&error], @"%@", error);
	NSDictionary *hello = [replies replyTo:1 timeout:10][@"result"];
	XCTAssertEqualObjects(hello[@"plugin"][@"id"], [NFKPluginFixtures pluginIdentifierForKind:NFKPluginEntryKindExecutable]);
	XCTAssertEqualObjects(hello[@"plugin"][@"version"], @"1.0.0");
	XCTAssertEqualObjects(hello[@"protocol"][@"major"], @1);
	XCTAssertEqualObjects(hello[@"pid"], @(process.processIdentifier));

	NSDictionary *unknown = @{ @"type": @"unknownMessage", @"id": @2 };
	XCTAssertTrue([channel sendMessage:unknown error:&error], @"%@", error);
	NSError *refusal = [NFKInferenceWireCoding errorFromJSONObject:@{ @"error": [replies replyTo:2 timeout:10][@"error"] ?: @{} }];
	XCTAssertEqualObjects(refusal.domain, NFKPluginErrorDomain);
	XCTAssertEqual(refusal.code, NFKPluginErrorProtocolUnsupported, @"an unknown request is refused, not ignored");
	[self shutDownExecutable:process channel:channel replies:replies identifier:3];
}

- (void)testTheExecutableInvertsAttachmentsTheCoreWrites
{
	NFKSubprocess *process = nil;
	NFKPluginChannel *channel = [self launchExecutableFixture:&process];
	if (channel == nil) {
		[process stop];
		return;
	}
	NFKFixtureReplies *replies = [[NFKFixtureReplies alloc] initWithChannel:channel];
	[channel resume];
	NSError *error = nil;
	XCTAssertTrue([channel sendMessage:[self helloMessage] error:&error], @"%@", error);
	XCTAssertNotNil([replies replyTo:1 timeout:10][@"result"]);
	NSDictionary *build = @{ @"type": @"build", @"id": @2, @"model": NFKPluginFixtureInvertModel, @"options": @{} };
	XCTAssertTrue([channel sendMessage:build error:&error], @"%@", error);
	NSNumber *backend = [replies replyTo:2 timeout:10][@"result"][@"backend"];
	XCTAssertNotNil(backend);

	NSMutableArray<NSNumber *> *values = [NSMutableArray array];
	for (NSUInteger index = 0; index < 40000; index++) {
		[values addObject:@((float)index * 0.25f)];
	}
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKPluginFixtureValuesKey: values } parameters:@{}
														   outputModality:NFKModalityImage];
	NFKPluginAttachmentWriter *writer = [[NFKPluginAttachmentWriter alloc] init];
	NFKInferenceWireCoding *encoder = [[NFKInferenceWireCoding alloc] initWithTemporaryDirectory:nil];
	encoder.attachmentSink = writer;
	id requestJSON = [encoder JSONObjectForRequest:request error:&error];
	XCTAssertEqualObjects(requestJSON[@"inputs"][NFKPluginFixtureValuesKey][@"$nfk"], @"buffer", @"160000 bytes leave the JSON");
	NSDictionary *run = @{ @"type": @"run", @"id": @3, @"backend": backend ?: @0, @"request": requestJSON ?: @{}, @"stream": @NO,
						   @"outputDirectory": NSTemporaryDirectory() };
	XCTAssertTrue([channel sendMessage:run descriptors:writer.descriptors error:&error], @"%@", error);
	NSDictionary *reply = [replies replyTo:3 timeout:30];
	XCTAssertNotNil(reply[@"result"], @"%@", reply);

	NFKInferenceWireCoding *decoder = [[NFKInferenceWireCoding alloc] initWithTemporaryDirectory:nil];
	decoder.attachmentSource = [[NFKPluginAttachmentReader alloc] initWithDescriptors:[replies descriptorsOfReplyTo:3]];
	NFKInferenceResult *result = [decoder resultFromJSONObject:reply[@"result"] error:&error];
	XCTAssertNotNil(result, @"%@", error);
	NSArray<NSNumber *> *inverted = result.outputs[NFKPluginFixtureValuesKey];
	XCTAssertEqual(inverted.count, values.count);
	XCTAssertEqual(inverted.firstObject.floatValue, 1.0f);
	XCTAssertEqual(inverted.lastObject.floatValue, 1.0f - values.lastObject.floatValue);
	XCTAssertEqualObjects(result.outputs[NFKPluginFixtureVersionKey], @"1.0.0");

	NSDictionary *echo = @{ @"type": @"callTool", @"id": @4, @"tool": NFKPluginFixtureEchoTool, @"arguments": @{ @"text": @"hi" } };
	XCTAssertTrue([channel sendMessage:echo error:&error], @"%@", error);
	XCTAssertEqualObjects([replies replyTo:4 timeout:10][@"result"][@"toolResult"][@"text"], @"> hi");
	[self shutDownExecutable:process channel:channel replies:replies identifier:5];
}

@end

#endif
