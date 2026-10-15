//
//  NFKPluginLoaderTests.m
//  InferKitTests
//
//  Loading plugin code into this process: the Objective-C bundle and the C library load and run, a
//  plugin ABI of another major version is refused, the C fixture frees what it allocates on every
//  path, a second version and an executable are refused, and the same bundle loads once. The bundle
//  that carries its own InferKit loads in a child process (NFKPluginServiceTests), since loading it here
//  would leave two NFKInferenceRequest classes in the test process. A loaded plugin stays for the life of the process, so these tests
//  load the fixtures at 1.0.0 only and keep 2.0.0 for the second-version refusal; a test that needs
//  another identity loads a re-signed copy under an identifier of its own.
//

#import <XCTest/XCTest.h>
#import <InferKit/InferKit.h>

#if TARGET_OS_OSX

#import "NFKPluginFixtures.h"
#import "../../Sources/InferKit/NFKPluginLoaderPrivate.h"
#import <CoreVideo/CoreVideo.h>
#import <dlfcn.h>
#import <mach/mach.h>
#import <sys/stat.h>

/*! The number of threads this process runs now. */
static NSInteger NFKLoaderTestThreadCount(void)
{
	thread_act_array_t threads = NULL;
	mach_msg_type_number_t count = 0;
	if (task_threads(mach_task_self(), &threads, &count) != KERN_SUCCESS) {
		return -1;
	}
	for (mach_msg_type_number_t index = 0; index < count; index++) {
		mach_port_deallocate(mach_task_self(), threads[index]);
	}
	vm_deallocate(mach_task_self(), (vm_address_t)threads, sizeof(thread_act_t) * count);
	return (NSInteger)count;
}

/*! A C plugin whose function table claims plugin ABI 2.0. */
static NSString * const NFKLoaderTestsSecondMajorSource =
	@"#include <InferKitC/NFKPluginABI.h>\n"
	@"static nfk_plugin_status describe(void *c, char **j) { return NFK_PLUGIN_STATUS_FAILED; }\n"
	@"static void shutdown_plugin(void *c) { }\n"
	@"static const nfk_plugin table = { sizeof(nfk_plugin), (2u << 16), 0, describe, 0, 0, 0, 0, 0, 0, 0, 0, 0, shutdown_plugin };\n"
	@"NFK_PLUGIN_EXPORT nfk_plugin_status nfk_plugin_entry(uint32_t v, const nfk_plugin_host *h, const nfk_plugin **out)\n"
	@"{ *out = &table; return NFK_PLUGIN_STATUS_OK; }\n";

@interface NFKPluginLoaderTests : XCTestCase
@property (nonatomic, strong) NSMutableArray<NSURL *> *temporaryURLs;
@end

@implementation NFKPluginLoaderTests

- (void)setUp
{
	[super setUp];
	self.temporaryURLs = [NSMutableArray array];
}

- (void)tearDown
{
	for (NSURL *url in self.temporaryURLs) {
		[NSFileManager.defaultManager removeItemAtURL:url error:NULL];
	}
	[super tearDown];
}

#pragma mark Helpers

- (NSURL *)fixtureOfKind:(NFKPluginEntryKind)kind version:(NSString *)version
{
	NSError *error = nil;
	NSURL *bundle = [NFKPluginFixtures bundleURLForKind:kind version:version error:&error];
	XCTAssertNotNil(bundle, @"%@", error);
	return bundle;
}

- (nullable NFKLoadedPlugin *)loadFixtureOfKind:(NFKPluginEntryKind)kind
{
	NSError *error = nil;
	NFKLoadedPlugin *plugin = [NFKPluginLoader loadPluginAtURL:[self fixtureOfKind:kind version:@"1.0.0"]
											   signaturePolicy:NFKPluginSignaturePolicy.developmentPolicy error:&error];
	XCTAssertNotNil(plugin, @"%@", error);
	return plugin;
}

- (void)assertError:(NSError *)error domain:(NSString *)domain code:(NSInteger)code
{
	XCTAssertEqualObjects(error.domain, domain, @"%@", error);
	XCTAssertEqual(error.code, code, @"%@", error);
}

/*! A copy of bundle under identifier, its manifest edited by edit, its executable replaced by
	executable when one is given, and signed ad hoc again. */
- (nullable NSURL *)copyOfBundle:(NSURL *)bundle
					  identifier:(NSString *)identifier
					  executable:(nullable NSURL *)executable
					editManifest:(nullable void (^)(NSMutableDictionary *manifest))edit
{
	NSURL *directory = [[NSURL fileURLWithPath:NSTemporaryDirectory()] URLByAppendingPathComponent:
						[NSString stringWithFormat:@"loader-test-%@", NSUUID.UUID.UUIDString] isDirectory:YES];
	[self.temporaryURLs addObject:directory];
	NSURL *copy = [directory URLByAppendingPathComponent:bundle.lastPathComponent isDirectory:YES];
	NSError *error = nil;
	if (![NSFileManager.defaultManager createDirectoryAtURL:directory withIntermediateDirectories:YES attributes:nil error:&error]
		|| ![NSFileManager.defaultManager copyItemAtURL:bundle toURL:copy error:&error]) {
		XCTFail(@"%@", error);
		return nil;
	}
	NSURL *infoURL = [copy URLByAppendingPathComponent:@"Contents/Info.plist"];
	NSMutableDictionary *info = [NSMutableDictionary dictionaryWithContentsOfURL:infoURL];
	info[@"CFBundleIdentifier"] = identifier;
	XCTAssertTrue([info writeToURL:infoURL error:NULL]);
	NSURL *manifestURL = [copy URLByAppendingPathComponent:@"Contents/Resources/inferkit-plugin.json"];
	NSMutableDictionary *manifest = [NSJSONSerialization JSONObjectWithData:[NSData dataWithContentsOfURL:manifestURL]
																	options:NSJSONReadingMutableContainers error:NULL];
	manifest[@"id"] = identifier;
	if (edit != nil) {
		edit(manifest);
	}
	XCTAssertTrue([[NSJSONSerialization dataWithJSONObject:manifest options:0 error:NULL] writeToURL:manifestURL atomically:YES]);
	if (executable != nil) {
		NSURL *target = [NFKPluginFixtures executableURLOfBundleAtURL:copy];
		XCTAssertTrue([NSFileManager.defaultManager removeItemAtURL:target error:NULL]);
		XCTAssertTrue([NSFileManager.defaultManager copyItemAtURL:executable toURL:target error:&error], @"%@", error);
	}
	if ([NFKPluginFixtures runProgram:@"/usr/bin/codesign" arguments:@[ @"--force", @"--sign", @"-", copy.path ] error:&error] == nil) {
		XCTFail(@"%@", error);
		return nil;
	}
	return copy;
}

/*! The C fixture's count of live allocations. */
- (int64_t)liveAllocationsOfCPlugin:(NFKLoadedPlugin *)plugin
{
	NSURL *library = [NFKPluginFixtures executableURLOfBundleAtURL:plugin.manifest.bundleURL];
	void *handle = dlopen(library.fileSystemRepresentation, RTLD_NOLOAD | RTLD_LOCAL);
	XCTAssertTrue(handle != NULL, @"the C fixture is loaded");
	int64_t (*counter)(void) = handle != NULL ? (int64_t (*)(void))dlsym(handle, NFKPluginFixtureAllocationCounterSymbol.UTF8String) : NULL;
	XCTAssertTrue(counter != NULL, @"the C fixture exports its allocation counter");
	return counter != NULL ? counter() : -1;
}

- (BOOL)waitUntil:(BOOL (^)(void))condition timeout:(NSTimeInterval)timeout
{
	NSDate *deadline = [NSDate dateWithTimeIntervalSinceNow:timeout];
	while (!condition()) {
		if (deadline.timeIntervalSinceNow <= 0) {
			return NO;
		}
		[NSThread sleepForTimeInterval:0.01];
	}
	return YES;
}

- (CVPixelBufferRef)newImageOfSide:(size_t)side CF_RETURNS_RETAINED
{
	CVPixelBufferRef buffer = NULL;
	XCTAssertEqual(CVPixelBufferCreate(kCFAllocatorDefault, side, side, kCVPixelFormatType_32BGRA, NULL, &buffer), kCVReturnSuccess);
	CVPixelBufferLockBaseAddress(buffer, 0);
	uint8_t *bytes = CVPixelBufferGetBaseAddress(buffer);
	size_t stride = CVPixelBufferGetBytesPerRow(buffer);
	for (size_t row = 0; row < side; row++) {
		for (size_t column = 0; column < side; column++) {
			uint8_t *pixel = bytes + row * stride + column * 4;
			pixel[0] = 10;
			pixel[1] = 20;
			pixel[2] = (uint8_t)row;
			pixel[3] = 255;
		}
	}
	CVPixelBufferUnlockBaseAddress(buffer, 0);
	return buffer;
}

#pragma mark Objective-C

- (void)testAnObjectiveCBundleLoadsAndRuns
{
	NFKLoadedPlugin *plugin = [self loadFixtureOfKind:NFKPluginEntryKindObjC];
	XCTAssertEqualObjects(plugin.manifest.identifier, [NFKPluginFixtures pluginIdentifierForKind:NFKPluginEntryKindObjC]);
	XCTAssertTrue(plugin.context.runsInHostProcess);
	XCTAssertEqualObjects(plugin.context.hostAPIVersion, NFKInferKit.APIVersion);
	NSArray<NSString *> *models = [plugin.describedModels valueForKey:@"identifier"];
	XCTAssertEqualObjects(models, (@[ NFKPluginFixtureInvertModel, NFKPluginFixtureSlowModel ]));
	XCTAssertEqualObjects([plugin.tools valueForKey:@"name"], @[ NFKPluginFixtureEchoTool ]);

	NSError *error = nil;
	id<NFKInferenceBackend> backend = [plugin backendForModelIdentifier:NFKPluginFixtureInvertModel options:@{} error:&error];
	XCTAssertNotNil(backend, @"%@", error);
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKPluginFixtureValuesKey: @[ @0.25, @1.0 ] }];
	NFKInferenceResult *result = [backend runInferenceForRequest:request error:&error];
	XCTAssertNotNil(result, @"%@", error);
	XCTAssertEqualObjects(result.outputs[NFKPluginFixtureValuesKey], (@[ @0.75, @0.0 ]));
	XCTAssertEqualObjects(result.outputs[NFKPluginFixtureVersionKey], @"1.0.0");
}

- (void)testLoadingTheSameBundleAgainReturnsTheSameObject
{
	NFKLoadedPlugin *first = [self loadFixtureOfKind:NFKPluginEntryKindObjC];
	NFKLoadedPlugin *second = [self loadFixtureOfKind:NFKPluginEntryKindObjC];
	XCTAssertTrue(first == second);
	XCTAssertTrue([self loadFixtureOfKind:NFKPluginEntryKindCABI] == [self loadFixtureOfKind:NFKPluginEntryKindCABI]);
}

- (void)testASecondVersionOfALoadedPluginIsRefused
{
	XCTAssertNotNil([self loadFixtureOfKind:NFKPluginEntryKindObjC]);
	NSError *error = nil;
	NSURL *second = [self fixtureOfKind:NFKPluginEntryKindObjC version:@"2.0.0"];
	XCTAssertNil([NFKPluginLoader loadPluginAtURL:second signaturePolicy:NFKPluginSignaturePolicy.developmentPolicy error:&error]);
	[self assertError:error domain:NFKPluginErrorDomain code:NFKPluginErrorLoadFailed];
	XCTAssertTrue([error.localizedDescription containsString:@"2.0.0"], @"%@", error);
}

- (void)testAnUnknownModelIsRefused
{
	NFKLoadedPlugin *plugin = [self loadFixtureOfKind:NFKPluginEntryKindObjC];
	NSError *error = nil;
	XCTAssertNil([plugin backendForModelIdentifier:@"absent" options:@{} error:&error]);
	[self assertError:error domain:NFKPluginErrorDomain code:NFKPluginErrorUnknownModel];
}

#pragma mark Refusals before code loads

- (void)testAnExecutablePluginIsRefusedInProcess
{
	NSError *error = nil;
	XCTAssertNil([NFKPluginLoader loadPluginAtURL:[self fixtureOfKind:NFKPluginEntryKindExecutable version:@"1.0.0"]
								  signaturePolicy:NFKPluginSignaturePolicy.developmentPolicy error:&error]);
	[self assertError:error domain:NFKPluginErrorDomain code:NFKPluginErrorNotTrusted];
}

- (void)testASignatureThePolicyRefusesIsRefused
{
	NSError *error = nil;
	XCTAssertNil([NFKPluginLoader loadPluginAtURL:[self fixtureOfKind:NFKPluginEntryKindObjC version:@"1.0.0"]
								  signaturePolicy:NFKPluginSignaturePolicy.defaultPolicy error:&error],
				 @"an ad hoc signature, refused even when the plugin is already loaded");
	[self assertError:error domain:NFKPluginErrorDomain code:NFKPluginErrorSignatureInvalid];

	NSURL *tampered = [NFKPluginFixtures tamperedCopyOfBundleAtURL:[self fixtureOfKind:NFKPluginEntryKindCABI version:@"1.0.0"] error:&error];
	XCTAssertNotNil(tampered, @"%@", error);
	[self.temporaryURLs addObject:tampered.URLByDeletingLastPathComponent];
	XCTAssertNil([NFKPluginLoader loadPluginAtURL:tampered signaturePolicy:NFKPluginSignaturePolicy.developmentPolicy error:&error]);
	[self assertError:error domain:NFKPluginErrorDomain code:NFKPluginErrorSignatureInvalid];
}

- (void)testAManifestNamingAnotherPluginABIMajorIsRefused
{
	NSURL *copy = [self copyOfBundle:[self fixtureOfKind:NFKPluginEntryKindCABI version:@"1.0.0"]
						  identifier:@"com.example.inferkit.loader-test-abi-manifest" executable:nil
						editManifest:^(NSMutableDictionary *manifest) {
		manifest[@"compatibility"][@"pluginABI"] = @"2.0";
	}];
	NSError *error = nil;
	XCTAssertNil([NFKPluginLoader loadPluginAtURL:copy signaturePolicy:NFKPluginSignaturePolicy.developmentPolicy error:&error]);
	[self assertError:error domain:NFKPluginErrorDomain code:NFKPluginErrorIncompatible];
}

- (void)testABundleAnotherAccountCouldChangeIsRefused
{
	NSURL *fixture = [self fixtureOfKind:NFKPluginEntryKindObjC version:@"1.0.0"];
	NSURL *copy = [self copyOfBundle:fixture identifier:@"com.example.inferkit.loader-test-ownership" executable:nil editManifest:nil];
	NSURL *info = [copy URLByAppendingPathComponent:@"Contents/Info.plist"];
	XCTAssertEqual(chmod(info.fileSystemRepresentation, 0664), 0);
	NSError *error = nil;
	XCTAssertNil([NFKPluginLoader loadPluginAtURL:copy signaturePolicy:NFKPluginSignaturePolicy.developmentPolicy error:&error]);
	[self assertError:error domain:NFKPluginErrorDomain code:NFKPluginErrorLoadFailed];
	XCTAssertTrue([error.localizedDescription containsString:@"Info.plist is writable by its group"], @"%@", error);

	XCTAssertEqual(chmod(info.fileSystemRepresentation, 0644), 0);
	NSURL *folder = copy.URLByDeletingLastPathComponent;
	XCTAssertEqual(chmod(folder.fileSystemRepresentation, 0777), 0);
	XCTAssertNil([NFKPluginLoader loadPluginAtURL:copy signaturePolicy:NFKPluginSignaturePolicy.developmentPolicy error:&error]);
	[self assertError:error domain:NFKPluginErrorDomain code:NFKPluginErrorLoadFailed];
	XCTAssertTrue([error.localizedDescription containsString:@"is writable by every user"], @"the folder that holds the bundle counts too: %@", error);
	XCTAssertEqual(chmod(folder.fileSystemRepresentation, 0755), 0);
	XCTAssertTrue(NFKPluginCheckBundleOwnership(copy, &error), @"%@", error);
}

#pragma mark C

- (void)testACLibraryLoadsAndRuns
{
	NFKLoadedPlugin *plugin = [self loadFixtureOfKind:NFKPluginEntryKindCABI];
	XCTAssertEqualObjects([plugin.describedModels valueForKey:@"identifier"], (@[ NFKPluginFixtureInvertModel, NFKPluginFixtureSlowModel ]));
	XCTAssertEqualObjects(plugin.describedModels.firstObject.engine, NFKModelEnginePlugin);
	NSError *error = nil;
	id<NFKInferenceBackend> backend = [plugin backendForModelIdentifier:NFKPluginFixtureInvertModel options:@{} error:&error];
	XCTAssertNotNil(backend, @"%@", error);
	XCTAssertEqualObjects(backend.backendIdentifier, ([NSString stringWithFormat:@"plugin:%@/invert", plugin.manifest.identifier]));
	XCTAssertTrue([backend.supportedInputKeys containsObject:NFKInputImage]);

	NFKInferenceResult *values = [backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKPluginFixtureValuesKey: @[ @0.25, @1.0 ] }] error:&error];
	XCTAssertNotNil(values, @"%@", error);
	XCTAssertEqualObjects(values.outputs[NFKPluginFixtureValuesKey], (@[ @0.75, @0.0 ]));
	XCTAssertEqualObjects(values.outputs[NFKPluginFixtureVersionKey], @"1.0.0");

	CVPixelBufferRef image = [self newImageOfSide:256];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputImage: (__bridge id)image }];
	CVPixelBufferRelease(image);
	NFKInferenceResult *inverted = [backend runInferenceForRequest:request error:&error];
	XCTAssertNotNil(inverted, @"a 256 KiB image crosses as an attachment: %@", error);
	CVPixelBufferRef output = (__bridge CVPixelBufferRef)inverted.outputs[NFKOutputImage];
	XCTAssertTrue(output != NULL && CFGetTypeID(output) == CVPixelBufferGetTypeID());
	if (output != NULL && CFGetTypeID(output) == CVPixelBufferGetTypeID()) {
		CVPixelBufferLockBaseAddress(output, kCVPixelBufferLock_ReadOnly);
		const uint8_t *pixel = (const uint8_t *)CVPixelBufferGetBaseAddress(output) + 7 * CVPixelBufferGetBytesPerRow(output) + 3 * 4;
		XCTAssertEqual(pixel[0], 245);
		XCTAssertEqual(pixel[1], 235);
		XCTAssertEqual(pixel[2], 248);
		XCTAssertEqual(pixel[3], 255, @"alpha is kept");
		CVPixelBufferUnlockBaseAddress(output, kCVPixelBufferLock_ReadOnly);
	}

	NFKToolResult *echo = [plugin.tools.firstObject invokeWithArguments:@{ @"text": @"hi" }];
	XCTAssertFalse(echo.isError, @"%@", echo.text);
	XCTAssertEqualObjects(echo.text, @"hi");
}

- (void)testACLibrarysRunReportsProgressAndIsCancelled
{
	NFKLoadedPlugin *plugin = [self loadFixtureOfKind:NFKPluginEntryKindCABI];
	NSError *error = nil;
	id<NFKInferenceBackend> backend = [plugin backendForModelIdentifier:NFKPluginFixtureSlowModel options:@{} error:&error];
	XCTAssertNotNil(backend, @"%@", error);

	NFKInferenceRequest *finishing = [NFKInferenceRequest requestWithInputs:@{} parameters:@{ NFKPluginFixtureStepsKey: @3, NFKPluginFixtureStepMillisecondsKey: @5 }
															outputModality:NFKModalityText];
	NFKInferenceJob *job = [backend submitInferenceJobForRequest:finishing];
	NSMutableArray<NSNumber *> *steps = [NSMutableArray array];
	job.progressHandler = ^(NFKInferenceJob *progressed) {
		id step = progressed.partialResult.outputs[@"step"];
		@synchronized (steps) {
			if (step != nil) {
				[steps addObject:step];
			}
		}
	};
	XCTAssertTrue([self waitUntil:^BOOL { return job.status == NFKInferenceJobStatusSucceeded || job.status == NFKInferenceJobStatusFailed; } timeout:10]);
	XCTAssertEqual(job.status, NFKInferenceJobStatusSucceeded, @"%@", job.error);
	XCTAssertEqualObjects(job.result.outputs[NFKPluginFixtureStepsKey], @3);
	@synchronized (steps) {
		XCTAssertTrue(steps.count > 0, @"progress arrives with a partial result");
	}

	NFKInferenceRequest *lengthy = [NFKInferenceRequest requestWithInputs:@{} parameters:@{ NFKPluginFixtureStepsKey: @1000, NFKPluginFixtureStepMillisecondsKey: @10 }
														 outputModality:NFKModalityText];
	NFKInferenceJob *cancelled = [backend submitInferenceJobForRequest:lengthy];
	XCTAssertTrue([self waitUntil:^BOOL { return cancelled.progress > 0; } timeout:10]);
	[cancelled cancel];
	XCTAssertEqual(cancelled.status, NFKInferenceJobStatusCancelled);
}

- (void)testTheCPluginFreesWhatItAllocatesOnEveryPath
{
	NFKLoadedPlugin *plugin = [self loadFixtureOfKind:NFKPluginEntryKindCABI];
	int64_t baseline = [self liveAllocationsOfCPlugin:plugin];
	@autoreleasepool {
		NSError *error = nil;
		XCTAssertNil([plugin backendForModelIdentifier:@"absent" options:@{} error:&error], @"the error string is freed");
		id<NFKInferenceBackend> invert = [plugin backendForModelIdentifier:NFKPluginFixtureInvertModel options:@{} error:&error];
		XCTAssertNotNil(invert, @"%@", error);
		XCTAssertEqualObjects(invert.modelInfo[NFKModelInfoArchitecture], @"sample-invert", @"describe_backend's string is freed");
		@autoreleasepool {
			NSMutableArray *values = [NSMutableArray array];
			for (NSUInteger index = 0; index < 40000; index++) {
				[values addObject:@(0.5f)];
			}
			NFKInferenceResult *result = [invert runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKPluginFixtureValuesKey: values }] error:&error];
			XCTAssertEqual([result.outputs[NFKPluginFixtureValuesKey] count], 40000u, @"%@", error);
			XCTAssertNil([invert runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{}] error:&error], @"the error path");
			[self assertError:error domain:NFKInferenceErrorDomain code:kNFKError_InferenceMissingInput];
		}
		XCTAssertFalse([plugin.tools.firstObject invokeWithArguments:@{ @"text": @"x" }].isError);

		id<NFKInferenceBackend> slow = [plugin backendForModelIdentifier:NFKPluginFixtureSlowModel options:@{} error:&error];
		NFKInferenceJob *job = [slow submitInferenceJobForRequest:[NFKInferenceRequest requestWithInputs:@{} parameters:@{ NFKPluginFixtureStepsKey: @1000 }
																							  outputModality:NFKModalityText]];
		XCTAssertTrue([self waitUntil:^BOOL { return job.progress > 0; } timeout:10]);
		[job cancel];
	}
	XCTAssertTrue([self waitUntil:^BOOL { return [self liveAllocationsOfCPlugin:plugin] == baseline; } timeout:10],
				  @"every payload, string, and backend is freed: %lld live, %lld before", [self liveAllocationsOfCPlugin:plugin], baseline);
}

- (void)testCRunsWaitingForASlotHoldNoThreads
{
	NFKLoadedPlugin *plugin = [self loadFixtureOfKind:NFKPluginEntryKindCABI];
	XCTAssertEqual(plugin.manifest.maximumConcurrentRuns, 1);
	NSError *error = nil;
	id<NFKInferenceBackend> backend = [plugin backendForModelIdentifier:NFKPluginFixtureSlowModel options:@{} error:&error];
	XCTAssertNotNil(backend, @"%@", error);
	NSInteger baseline = NFKLoaderTestThreadCount();
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{} parameters:@{ NFKPluginFixtureStepsKey: @1, NFKPluginFixtureStepMillisecondsKey: @10 }
														  outputModality:NFKModalityText];
	NSMutableArray<NFKInferenceJob *> *jobs = [NSMutableArray array];
	for (NSInteger index = 0; index < 64; index++) {
		[jobs addObject:[backend submitInferenceJobForRequest:request]];
	}
	__block NSInteger most = baseline;
	XCTAssertTrue([self waitUntil:^BOOL {
		most = MAX(most, NFKLoaderTestThreadCount());
		for (NFKInferenceJob *job in jobs) {
			if (job.status != NFKInferenceJobStatusSucceeded && job.status != NFKInferenceJobStatusFailed) {
				return NO;
			}
		}
		return YES;
	} timeout:30]);
	XCTAssertLessThanOrEqual(most - baseline, 16, @"64 runs queued on one slot ran on %ld threads more than the %ld before",
							 (long)(most - baseline), (long)baseline);
	for (NFKInferenceJob *job in jobs) {
		XCTAssertEqual(job.status, NFKInferenceJobStatusSucceeded, @"%@", job.error);
	}
}

- (void)testAFunctionTableOfAnotherABIMajorIsRefused
{
	NSURL *directory = [[NSURL fileURLWithPath:NSTemporaryDirectory()] URLByAppendingPathComponent:
						[NSString stringWithFormat:@"loader-test-abi-%@", NSUUID.UUID.UUIDString] isDirectory:YES];
	[self.temporaryURLs addObject:directory];
	XCTAssertTrue([NSFileManager.defaultManager createDirectoryAtURL:directory withIntermediateDirectories:YES attributes:nil error:NULL]);
	NSURL *source = [directory URLByAppendingPathComponent:@"second_major.c"];
	NSURL *library = [directory URLByAppendingPathComponent:@"second_major.dylib"];
	XCTAssertTrue([NFKLoaderTestsSecondMajorSource writeToURL:source atomically:YES encoding:NSUTF8StringEncoding error:NULL]);
	NSString *include = [NFKPluginFixtures.sourceRootURL URLByAppendingPathComponent:@"Sources/InferKitC/include"].path;
	NSError *error = nil;
	NSArray<NSString *> *compile = @[ @"clang", @"-dynamiclib", @"-I", include, @"-o", library.path, source.path ];
	XCTAssertNotNil([NFKPluginFixtures runProgram:@"/usr/bin/xcrun" arguments:compile error:&error], @"%@", error);

	NSURL *copy = [self copyOfBundle:[self fixtureOfKind:NFKPluginEntryKindCABI version:@"1.0.0"]
						  identifier:@"com.example.inferkit.loader-test-abi-table" executable:library editManifest:nil];
	XCTAssertNil([NFKPluginLoader loadPluginAtURL:copy signaturePolicy:NFKPluginSignaturePolicy.developmentPolicy error:&error]);
	[self assertError:error domain:NFKPluginErrorDomain code:NFKPluginErrorIncompatible];
	XCTAssertTrue([error.localizedDescription containsString:@"2.0"], @"%@", error);
}

@end

#endif
