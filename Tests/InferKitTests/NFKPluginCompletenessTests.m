//
//  NFKPluginCompletenessTests.m
//  InferKitTests
//
//  Every plugin form end to end: the Objective-C bundle, the C library, and the executable program, each
//  built by NFKPluginFixtures and run in a process of its own through the registry. The catalog lists
//  all three without loading code; each kind inverts an image and a float buffer, inline and as
//  attachments, and mirrors an NFKDetection that crosses as JSON both ways; it reports progress with
//  partial results, cancels, answers a tool, unloads, exits when idle, restarts to a second version
//  mid-run, and refuses a tampered bundle and a pinned team; the trusted kinds give their out-of-process
//  outputs in this process; the C API runs a plugin model through its process; and a server's idle
//  unload ends each kind's process. NFKPluginRegistryTests covers the registry's own rules on the C
//  fixture alone. In-process runs load the cached 1.0.0 fixtures only, as NFKPluginLoaderTests does.
//

#import <XCTest/XCTest.h>
#import <InferKit/InferKit.h>

#if TARGET_OS_OSX

#if __has_include(<InferKitC/NFKInferKitC.h>)
#import <InferKitC/NFKInferKitC.h>
#else
#import <InferKit/NFKInferKitC.h>
#endif

#import "NFKPluginFixtures.h"
#import "NFKPluginTestPeer.h"
#import "../../Sources/InferKit/NFKPluginProcessPrivate.h"
#import <mach-o/dyld.h>
#import <signal.h>

/*! The detection every kind receives under "region". */
static NFKDetection *NFKCompletenessRegion(void)
{
	return [[NFKDetection alloc] initWithLabel:@"cat" classIndex:3 confidence:0.25 boundingBox:CGRectMake(0.125, 0.25, 0.5, 0.5)
								 quadrilateral:nil];
}

/*! The path with symbolic links resolved, or the path itself. */
static NSString *NFKCompletenessResolvedPath(const char *path)
{
	char resolved[PATH_MAX];
	return realpath(path, resolved) != NULL ? [NSString stringWithUTF8String:resolved] : [NSString stringWithUTF8String:path];
}

@interface NFKPluginCompletenessTests : XCTestCase
@property (nonatomic, strong) NSMutableArray<NSURL *> *temporaryURLs;
@end

@implementation NFKPluginCompletenessTests

- (void)setUp
{
	[super setUp];
	self.temporaryURLs = [NSMutableArray array];
	NFKModelCatalog.sharedCatalog.memoryBudget = [self generousBudget];
}

- (void)tearDown
{
	NFKModelCatalog.sharedCatalog.memoryBudget = nil;
	for (NSURL *url in self.temporaryURLs) {
		[NSFileManager.defaultManager removeItemAtURL:url error:NULL];
	}
	[super tearDown];
}

#pragma mark Helpers

/*! A 64 GiB working set's budget, so no build depends on the memory of the machine that runs the tests. */
- (NFKModelMemoryBudget *)generousBudget
{
	return [NFKModelMemoryBudget budgetWithWorkingSetByteCount:(NSInteger)64 << 30];
}

- (NSArray<NFKPluginEntryKind> *)kinds
{
	return @[ NFKPluginEntryKindObjC, NFKPluginEntryKindCABI, NFKPluginEntryKindExecutable ];
}

- (void)requireHost
{
	if ([NFKPluginFixtures hostExecutableURL] == nil) {
		XCTSkip(@"the inferkit-plugin-host product is not built beside the tests");
	}
}

- (NSURL *)fixtureOfKind:(NFKPluginEntryKind)kind version:(NSString *)version
{
	NSError *error = nil;
	NSURL *bundle = [NFKPluginFixtures bundleURLForKind:kind version:version error:&error];
	XCTAssertNotNil(bundle, @"%@", error);
	return bundle;
}

- (NSURL *)folderWithCopiesOf:(NSArray<NSURL *> *)bundles
{
	NSError *error = nil;
	NSURL *folder = [NFKPluginFixtures folderWithCopiesOfBundles:bundles error:&error];
	XCTAssertNotNil(folder, @"%@", error);
	if (folder != nil) {
		[self.temporaryURLs addObject:folder];
	}
	return folder;
}

/*! A registry over an empty catalog that accepts ad hoc signatures and launches the host built beside
	the tests, listing the bundle when one is given. */
- (NFKPluginRegistry *)registryListing:(nullable NSURL *)bundle
{
	NFKModelCatalog *catalog = [[NFKModelCatalog alloc] init];
	catalog.memoryBudget = [self generousBudget];
	NFKPluginRegistry *registry = [[NFKPluginRegistry alloc] initWithCatalog:catalog];
	registry.signaturePolicy = NFKPluginSignaturePolicy.developmentPolicy;
	NSURL *host = [NFKPluginFixtures hostExecutableURL];
	if (host != nil) {
		registry.hostExecutableURL = host;
	}
	if (bundle != nil) {
		NSError *error = nil;
		XCTAssertNotNil([registry addPluginAtURL:bundle error:&error], @"%@", error);
	}
	return registry;
}

- (NSString *)model:(NSString *)model ofKind:(NFKPluginEntryKind)kind
{
	return [NSString stringWithFormat:@"%@/%@", [NFKPluginFixtures pluginIdentifierForKind:kind], model];
}

- (nullable NFKPluginBackend *)backendFor:(NSString *)model ofKind:(NFKPluginEntryKind)kind registry:(NFKPluginRegistry *)registry
{
	NSError *error = nil;
	NFKPluginBackend *backend = [registry backendForModelIdentifier:[self model:model ofKind:kind] options:nil error:&error];
	XCTAssertNotNil(backend, @"%@: %@", kind, error);
	return backend;
}

- (void)assertError:(nullable NSError *)error domain:(NSString *)domain code:(NSInteger)code kind:(NSString *)kind
{
	XCTAssertEqualObjects(error.domain, domain, @"%@: %@", kind, error);
	XCTAssertEqual(error.code, code, @"%@: %@", kind, error);
}

- (NSArray<NSNumber *> *)floatsOfCount:(NSUInteger)count
{
	NSMutableArray<NSNumber *> *values = [NSMutableArray arrayWithCapacity:count];
	for (NSUInteger index = 0; index < count; index++) {
		[values addObject:@((float)index / (float)count)];
	}
	return values;
}

/*! Runs invert on a side × side image and count floats, and checks every inverted pixel and value. */
- (void)assertInvertOf:(id<NFKInferenceBackend>)backend side:(size_t)side floats:(NSUInteger)count kind:(NSString *)kind
{
	CVPixelBufferRef image = [NFKPluginFixtures newPatternImageOfSide:side];
	NSArray<NSNumber *> *values = [self floatsOfCount:count];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputImage: (__bridge id)image, NFKPluginFixtureValuesKey: values }];
	NSData *original = [NFKPluginFixtures pixelsOfImage:(__bridge id)image];
	CVPixelBufferRelease(image);
	NSError *error = nil;
	NFKInferenceResult *result = [backend runInferenceForRequest:request error:&error];
	XCTAssertNotNil(result, @"%@ at %zu: %@", kind, side, error);
	XCTAssertEqualObjects(result.outputs[NFKPluginFixtureVersionKey], @"1.0.0", @"%@", kind);

	NSData *pixels = [NFKPluginFixtures pixelsOfImage:result.outputs[NFKOutputImage]];
	XCTAssertEqual(pixels.length, original.length, @"%@: a %zu × %zu image", kind, side, side);
	const uint8_t *in = original.bytes;
	const uint8_t *out = pixels.bytes;
	for (NSUInteger index = 0; index < MIN(pixels.length, original.length); index++) {
		uint8_t expected = index % 4 == 3 ? in[index] : (uint8_t)(255 - in[index]);
		if (out[index] != expected) {
			XCTFail(@"%@: byte %lu of the %zu × %zu image is %u, not %u", kind, (unsigned long)index, side, side, out[index], expected);
			break;
		}
	}

	NSArray<NSNumber *> *inverted = result.outputs[NFKPluginFixtureValuesKey];
	XCTAssertEqual(inverted.count, count, @"%@", kind);
	for (NSUInteger index = 0; index < MIN(inverted.count, count); index++) {
		if (inverted[index].floatValue != 1.0f - values[index].floatValue) {
			XCTFail(@"%@: value %lu of %lu is %g", kind, (unsigned long)index, (unsigned long)count, inverted[index].floatValue);
			break;
		}
	}
}

/*! Sends NFKCompletenessRegion and returns the detection that comes back. */
- (nullable NFKDetection *)mirroredRegionFrom:(id<NFKInferenceBackend>)backend kind:(NSString *)kind
{
	NSError *error = nil;
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKPluginFixtureRegionKey: NFKCompletenessRegion() }];
	NFKInferenceResult *result = [backend runInferenceForRequest:request error:&error];
	XCTAssertNotNil(result, @"%@: %@", kind, error);
	NSArray *detections = result.outputs[NFKOutputDetections];
	XCTAssertEqual(detections.count, 1u, @"%@: %@", kind, result.outputs);
	NFKDetection *detection = detections.firstObject;
	XCTAssertTrue([detection isKindOfClass:NFKDetection.class], @"%@: %@ crosses back as a detection", kind, detection);
	return [detection isKindOfClass:NFKDetection.class] ? detection : nil;
}

- (void)assertMirroredDetection:(nullable NFKDetection *)detection kind:(NSString *)kind
{
	XCTAssertEqualObjects(detection.label, @"cat", @"%@", kind);
	XCTAssertEqual(detection.classIndex, 3, @"%@", kind);
	XCTAssertEqualWithAccuracy(detection.confidence, 0.75, 1e-12, @"%@", kind);
	XCTAssertTrue(CGRectEqualToRect(detection.boundingBox, CGRectMake(0.375, 0.25, 0.5, 0.5)), @"%@: %@", kind,
				  NSStringFromRect(NSRectFromCGRect(detection.boundingBox)));
}

/*! Waits for the process to leave Ready and for its program to exit. */
- (void)assertProcessExits:(NFKPluginProcess *)process pid:(pid_t)pid kind:(NSString *)kind
{
	XCTAssertTrue(NFKPluginTestWaitUntil(30, ^BOOL {
		return process.state == NFKPluginProcessStateNotRunning;
	}), @"%@: %@", kind, process.diagnosticDescription);
	XCTAssertTrue(NFKPluginTestWaitUntil(30, ^BOOL {
		return kill(pid, 0) != 0;
	}), @"%@: process %d has exited", kind, pid);
}

#pragma mark Listing

- (void)testTheCatalogListsEveryKindWithNoCodeLoadedAndNoProcessRunning
{
	NSMutableArray<NSURL *> *fixtures = [NSMutableArray array];
	for (NFKPluginEntryKind kind in self.kinds) {
		[fixtures addObject:[self fixtureOfKind:kind version:@"1.0.0"]];
	}
	NSURL *folder = [self folderWithCopiesOf:fixtures];
	// Every fixture declares the echo tool, and a tool name is unique across a registry's plugins.
	for (NSURL *fixture in [fixtures subarrayWithRange:NSMakeRange(1, fixtures.count - 1)]) {
		NSError *error = nil;
		XCTAssertTrue([NFKPluginFixtures editManifestOfBundleAtURL:[folder URLByAppendingPathComponent:fixture.lastPathComponent]
															  edit:^(NSMutableDictionary *manifest) {
			[manifest removeObjectForKey:@"tools"];
		} error:&error], @"%@", error);
	}
	NFKPluginRegistry *registry = [self registryListing:nil];
	NSArray<NSError *> *errors = nil;
	NSArray<NFKPluginManifest *> *added = [registry addPluginsInDirectoryURL:folder errors:&errors];
	XCTAssertEqual(errors.count, 0u, @"%@", errors);
	XCTAssertEqual(added.count, 3u);
	XCTAssertEqualObjects([NSSet setWithArray:[registry.manifests valueForKey:@"entryKind"]], [NSSet setWithArray:self.kinds]);
	XCTAssertEqual(registry.catalog.descriptors.count, 6u, @"invert and slow from each kind");
	XCTAssertEqualObjects([registry.tools valueForKey:@"name"], @[ NFKPluginFixtureEchoTool ]);

	for (NFKPluginEntryKind kind in self.kinds) {
		for (NSString *model in @[ NFKPluginFixtureInvertModel, NFKPluginFixtureSlowModel ]) {
			NSError *error = nil;
			id<NFKInferenceBackend> built = [registry.catalog backendForModelIdentifier:[self model:model ofKind:kind] options:nil error:&error];
			NFKPluginBackend *backend = [(NSObject *)built isKindOfClass:NFKDefaultedBackend.class]
				? (NFKPluginBackend *)((NFKDefaultedBackend *)built).backend : (NFKPluginBackend *)built;
			XCTAssertTrue([backend isKindOfClass:NFKPluginBackend.class], @"%@ %@: %@", kind, model, error);
			XCTAssertEqual(backend.process.state, NFKPluginProcessStateNotRunning, @"%@ %@", kind, model);
			XCTAssertEqual(backend.process.processIdentifier, 0, @"%@ %@", kind, model);
		}
	}
	for (NSURL *fixture in fixtures) {
		NSURL *executable = [NFKPluginFixtures executableURLOfBundleAtURL:[folder URLByAppendingPathComponent:fixture.lastPathComponent]];
		NSString *target = NFKCompletenessResolvedPath(executable.fileSystemRepresentation);
		for (uint32_t index = 0; index < _dyld_image_count(); index++) {
			const char *name = _dyld_get_image_name(index);
			XCTAssertFalse(name != NULL && [NFKCompletenessResolvedPath(name) isEqualToString:target], @"%@ is loaded", target);
		}
	}
}

#pragma mark Each kind in its own process

- (void)testEveryKindCarriesAttachmentsAndADetectionThroughItsProcess
{
	[self requireHost];
	for (NFKPluginEntryKind kind in self.kinds) {
		NFKPluginRegistry *registry = [self registryListing:[self fixtureOfKind:kind version:@"1.0.0"]];
		NFKPluginBackend *invert = [self backendFor:NFKPluginFixtureInvertModel ofKind:kind registry:registry];
		XCTAssertEqual(invert.isolation, NFKPluginIsolationProcess);
		// 64 × 64 × 4 and 4096 × 4 bytes stay under the 64 KiB inline limit; 256 × 256 × 4 and 65536 × 4
		// bytes leave the JSON as shared-memory attachments.
		[self assertInvertOf:invert side:64 floats:4096 kind:kind];
		[self assertInvertOf:invert side:256 floats:65536 kind:kind];
		[self assertMirroredDetection:[self mirroredRegionFrom:invert kind:kind] kind:kind];
		XCTAssertGreaterThan(invert.process.processIdentifier, 0, @"%@ ran in a process of its own", kind);
	}
}

- (void)testEveryKindReportsProgressCancelsCallsAToolUnloadsAndExitsWhenIdle
{
	[self requireHost];
	for (NFKPluginEntryKind kind in self.kinds) {
		NFKPluginRegistry *registry = [self registryListing:[self fixtureOfKind:kind version:@"1.0.0"]];
		NFKPluginProcess *process = nil;
		pid_t pid = 0;
		@autoreleasepool {
			NFKPluginBackend *slow = [self backendFor:NFKPluginFixtureSlowModel ofKind:kind registry:registry];
			process = slow.process;
			process.idleExitInterval = 0.5;

			NFKInferenceRequest *counting = [NFKInferenceRequest requestWithInputs:@{}
																	   parameters:@{ NFKPluginFixtureStepsKey: @3, NFKPluginFixtureStepMillisecondsKey: @10 }];
			NFKInferenceJob *job = [slow submitInferenceJobForRequest:counting];
			NSMutableArray<NSNumber *> *steps = [NSMutableArray array];
			job.progressHandler = ^(NFKInferenceJob *updated) {
				id step = updated.partialResult.outputs[@"step"];
				@synchronized (steps) {
					if ([step isKindOfClass:NSNumber.class]) {
						[steps addObject:step];
					}
				}
			};
			XCTAssertTrue(NFKPluginTestWaitUntil(60, ^BOOL {
				return job.status >= NFKInferenceJobStatusSucceeded;
			}), @"%@: %@", kind, process.diagnosticDescription);
			XCTAssertEqual(job.status, NFKInferenceJobStatusSucceeded, @"%@: %@", kind, job.error);
			XCTAssertEqualObjects(job.result.outputs[NFKPluginFixtureStepsKey], @3, @"%@", kind);
			@synchronized (steps) {
				XCTAssertGreaterThan(steps.count, 0u, @"%@: progress carries the partial results", kind);
				XCTAssertLessThanOrEqual(steps.lastObject.integerValue, 3, @"%@", kind);
			}

			NFKInferenceRequest *lengthy = [NFKInferenceRequest requestWithInputs:@{}
																	parameters:@{ NFKPluginFixtureStepsKey: @500, NFKPluginFixtureStepMillisecondsKey: @20 }];
			NFKInferenceJob *cancelled = [slow submitInferenceJobForRequest:lengthy];
			XCTAssertTrue(NFKPluginTestWaitUntil(30, ^BOOL {
				return cancelled.progress > 0;
			}), @"%@: the long run reports progress", kind);
			[cancelled cancel];
			XCTAssertTrue(NFKPluginTestWaitUntil(30, ^BOOL {
				return cancelled.status >= NFKInferenceJobStatusSucceeded;
			}), @"%@: %@", kind, process.diagnosticDescription);
			XCTAssertEqual(cancelled.status, NFKInferenceJobStatusCancelled, @"%@", kind);
			XCTAssertEqual(process.state, NFKPluginProcessStateReady, @"%@: a cancelled run leaves the process serving", kind);

			NFKInferenceTool *echo = registry.tools.firstObject;
			XCTAssertEqualObjects(echo.name, NFKPluginFixtureEchoTool);
			NFKToolResult *answered = [echo invokeWithArguments:@{ @"text": @"hi" }];
			XCTAssertFalse(answered.isError, @"%@: %@", kind, answered.text);
			XCTAssertEqualObjects(answered.text, @"hi", @"%@", kind);

			XCTAssertTrue(slow.isLoaded, @"%@: the runs loaded the model's weights", kind);
			XCTAssertEqual(slow.residentByteCount, 4096, @"%@", kind);
			[slow unload];
			XCTAssertTrue(NFKPluginTestWaitUntil(10, ^BOOL {
				return !slow.isLoaded;
			}), @"%@: the unload reaches the plugin", kind);
			XCTAssertEqual(slow.residentByteCount, 0, @"%@", kind);
			XCTAssertEqual(slow.peakByteCount, 4096, @"%@: the peak stays", kind);
			pid = process.processIdentifier;
			XCTAssertGreaterThan(pid, 0, @"%@", kind);
		}
		[self assertProcessExits:process pid:pid kind:kind];
	}
}

- (void)testEveryKindRestartsToASecondVersionMidRun
{
	[self requireHost];
	for (NFKPluginEntryKind kind in self.kinds) {
		NSURL *first = [self fixtureOfKind:kind version:@"1.0.0"];
		NSURL *installed = [[self folderWithCopiesOf:@[ first ]] URLByAppendingPathComponent:first.lastPathComponent];
		NFKPluginRegistry *registry = [self registryListing:installed];
		NFKPluginBackend *slow = [self backendFor:NFKPluginFixtureSlowModel ofKind:kind registry:registry];
		NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{}
																   parameters:@{ NFKPluginFixtureStepsKey: @20, NFKPluginFixtureStepMillisecondsKey: @50 }];
		NFKInferenceJob *job = [slow submitInferenceJobForRequest:request];
		XCTAssertTrue(NFKPluginTestWaitUntil(60, ^BOOL {
			return job.progress > 0;
		}), @"%@: %@", kind, slow.process.diagnosticDescription);

		NSError *error = nil;
		XCTAssertTrue([NSFileManager.defaultManager removeItemAtURL:installed error:&error], @"%@", error);
		XCTAssertTrue([NSFileManager.defaultManager copyItemAtURL:[self fixtureOfKind:kind version:@"2.0.0"] toURL:installed error:&error], @"%@", error);
		XCTestExpectation *restarted = [self expectationWithDescription:[@"restarted " stringByAppendingString:kind]];
		[registry reloadPluginWithIdentifier:[NFKPluginFixtures pluginIdentifierForKind:kind] completionHandler:^(NSError *reloadError) {
			XCTAssertNil(reloadError, @"%@", kind);
			[restarted fulfill];
		}];
		[self waitForExpectations:@[ restarted ] timeout:120];

		XCTAssertEqual(job.status, NFKInferenceJobStatusSucceeded, @"%@: %@", kind, job.error);
		XCTAssertEqualObjects(job.result.outputs[NFKPluginFixtureVersionKey], @"1.0.0", @"%@: the run in flight finishes on the first version", kind);
		NFKInferenceRequest *brief = [NFKInferenceRequest requestWithInputs:@{} parameters:@{ NFKPluginFixtureStepsKey: @1 }];
		NFKInferenceResult *after = [slow runInferenceForRequest:brief error:&error];
		XCTAssertEqualObjects(after.outputs[NFKPluginFixtureVersionKey], @"2.0.0", @"%@: %@", kind, error);
	}
}

#pragma mark Trust

- (void)testTrustedKindsGiveTheirOutOfProcessOutputsAndAnExecutableIsRefused
{
	[self requireHost];
	for (NFKPluginEntryKind kind in self.kinds) {
		NSURL *bundle = [self fixtureOfKind:kind version:@"1.0.0"];
		NFKPluginRegistry *inside = [self registryListing:bundle];
		[inside setTrusted:YES forPluginIdentifier:[NFKPluginFixtures pluginIdentifierForKind:kind]];
		NSError *error = nil;
		NFKPluginBackend *loaded = [inside backendForModelIdentifier:[self model:NFKPluginFixtureInvertModel ofKind:kind] options:nil error:&error];
		if ([kind isEqualToString:NFKPluginEntryKindExecutable]) {
			XCTAssertNil(loaded);
			[self assertError:error domain:NFKPluginErrorDomain code:NFKPluginErrorNotTrusted kind:kind];
			continue;
		}
		XCTAssertEqual(loaded.isolation, NFKPluginIsolationInProcess, @"%@: %@", kind, error);
		NFKPluginRegistry *outside = [self registryListing:bundle];
		NFKPluginBackend *separate = [self backendFor:NFKPluginFixtureInvertModel ofKind:kind registry:outside];

		NFKDetection *fromProcess = [self mirroredRegionFrom:separate kind:kind];
		NFKDetection *fromApp = [self mirroredRegionFrom:loaded kind:kind];
		[self assertMirroredDetection:fromApp kind:kind];
		XCTAssertEqualObjects(fromApp.label, fromProcess.label, @"%@", kind);
		XCTAssertTrue(CGRectEqualToRect(fromApp.boundingBox, fromProcess.boundingBox), @"%@", kind);
		XCTAssertEqual(fromApp.confidence, fromProcess.confidence, @"%@", kind);

		CVPixelBufferRef image = [NFKPluginFixtures newPatternImageOfSide:64];
		NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputImage: (__bridge id)image,
																				 NFKPluginFixtureValuesKey: [self floatsOfCount:4096] }];
		CVPixelBufferRelease(image);
		NFKInferenceResult *processResult = [separate runInferenceForRequest:request error:&error];
		NFKInferenceResult *appResult = [loaded runInferenceForRequest:request error:&error];
		XCTAssertEqualObjects(appResult.outputs[NFKPluginFixtureValuesKey], processResult.outputs[NFKPluginFixtureValuesKey], @"%@", kind);
		XCTAssertEqualObjects([NFKPluginFixtures pixelsOfImage:appResult.outputs[NFKOutputImage]],
							  [NFKPluginFixtures pixelsOfImage:processResult.outputs[NFKOutputImage]], @"%@", kind);
		XCTAssertEqualObjects(appResult.outputs[NFKPluginFixtureVersionKey], processResult.outputs[NFKPluginFixtureVersionKey], @"%@", kind);
	}
}

- (void)testEveryKindRefusesATamperedBundleAndATeamPin
{
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKPluginFixtureValuesKey: @[ @0.5 ] }];
	for (NFKPluginEntryKind kind in self.kinds) {
		NSURL *fixture = [self fixtureOfKind:kind version:@"1.0.0"];
		NSError *error = nil;
		NSURL *tampered = [NFKPluginFixtures tamperedCopyOfBundleAtURL:fixture error:&error];
		XCTAssertNotNil(tampered, @"%@", error);
		[self.temporaryURLs addObject:tampered.URLByDeletingLastPathComponent];
		NFKPluginRegistry *registry = [self registryListing:tampered];
		NFKPluginBackend *backend = [self backendFor:NFKPluginFixtureInvertModel ofKind:kind registry:registry];
		XCTAssertNil([backend runInferenceForRequest:request error:&error]);
		[self assertError:error domain:NFKPluginErrorDomain code:NFKPluginErrorSignatureInvalid kind:kind];
		XCTAssertEqual(backend.process.processIdentifier, 0, @"%@: nothing launched", kind);

		NFKPluginRegistry *pinned = [self registryListing:fixture];
		NFKPluginSignaturePolicy *policy = NFKPluginSignaturePolicy.developmentPolicy;
		policy.teamIdentifier = @"ABCDE12345";
		pinned.signaturePolicy = policy;
		NFKPluginBackend *refused = [self backendFor:NFKPluginFixtureInvertModel ofKind:kind registry:pinned];
		XCTAssertNil([refused runInferenceForRequest:request error:&error]);
		[self assertError:error domain:NFKPluginErrorDomain code:NFKPluginErrorTeamMismatch kind:kind];
		if ([kind isEqualToString:NFKPluginEntryKindExecutable]) {
			continue;
		}
		[registry setTrusted:YES forPluginIdentifier:[NFKPluginFixtures pluginIdentifierForKind:kind]];
		XCTAssertNil([registry backendForModelIdentifier:[self model:NFKPluginFixtureInvertModel ofKind:kind] options:nil error:&error]);
		[self assertError:error domain:NFKPluginErrorDomain code:NFKPluginErrorSignatureInvalid kind:kind];
		[pinned setTrusted:YES forPluginIdentifier:[NFKPluginFixtures pluginIdentifierForKind:kind]];
		XCTAssertNil([pinned backendForModelIdentifier:[self model:NFKPluginFixtureInvertModel ofKind:kind] options:nil error:&error]);
		[self assertError:error domain:NFKPluginErrorDomain code:NFKPluginErrorTeamMismatch kind:kind];
	}
}

#pragma mark The C API

- (void)testTheCAPIRunsAPluginModelThroughItsProcess
{
	[self requireHost];
	NFKPluginRegistry *shared = NFKPluginRegistry.sharedRegistry;
	NFKPluginSignaturePolicy *previousPolicy = shared.signaturePolicy;
	NSString *identifier = [NFKPluginFixtures pluginIdentifierForKind:NFKPluginEntryKindCABI];
	shared.signaturePolicy = NFKPluginSignaturePolicy.developmentPolicy;
	shared.hostExecutableURL = [NFKPluginFixtures hostExecutableURL];
	[self addTeardownBlock:^{
		[shared removePluginWithIdentifier:identifier];
		shared.signaturePolicy = previousPolicy;
		shared.hostExecutableURL = nil;
	}];
	NSURL *folder = [self folderWithCopiesOf:@[ [self fixtureOfKind:NFKPluginEntryKindCABI version:@"1.0.0"] ]];
	nfk_error *error = NULL;
	XCTAssertEqual(nfk_plugins_add_directory(folder.fileSystemRepresentation, &error), 1, @"%s", nfk_error_message(error));
	nfk_error_free(error);

	NSString *model = [self model:NFKPluginFixtureInvertModel ofKind:NFKPluginEntryKindCABI];
	error = NULL;
	nfk_backend *backend = nfk_backend_create(model.UTF8String, NULL, &error);
	XCTAssert(backend != NULL, @"%s", nfk_error_message(error));
	nfk_error_free(error);

	float values[4] = { 0.0f, 0.25f, 0.5f, 1.0f };
	nfk_buffer buffer;
	memset(&buffer, 0, sizeof buffer);
	buffer.struct_size = (uint32_t)sizeof buffer;
	buffer.kind = NFK_BUFFER_BYTES;
	buffer.bytes = values;
	buffer.length = sizeof values;
	buffer.descriptor = -1;
	const char *json = "{\"inputs\": {\"values\": {\"$nfk\": \"buffer\", \"index\": 0, \"as\": \"float32Array\", \"length\": 16, \"count\": 4}, "
		"\"region\": {\"$nfk\": \"detection\", \"label\": \"cat\", \"classIndex\": 3, \"confidence\": 0.25, "
		"\"boundingBox\": {\"x\": 0.125, \"y\": 0.25, \"width\": 0.5, \"height\": 0.5}}}, \"parameters\": {}}";
	nfk_payload request = { (uint32_t)sizeof(nfk_payload), 1, json, &buffer };
	error = NULL;
	nfk_result *result = backend != NULL ? nfk_backend_run(backend, &request, &error) : NULL;
	XCTAssert(result != NULL, @"%s", nfk_error_message(error));
	nfk_error_free(error);

	const char *resultJSON = nfk_result_json(result);
	NSData *resultData = resultJSON != NULL ? [NSData dataWithBytes:resultJSON length:strlen(resultJSON)] : nil;
	NSDictionary *outputs = resultData != nil ? [NSJSONSerialization JSONObjectWithData:resultData options:0 error:NULL][@"outputs"] : nil;
	XCTAssertEqualObjects(outputs[NFKPluginFixtureVersionKey], @"1.0.0", @"%s", resultJSON);
	NSDictionary *detection = [outputs[NFKOutputDetections] firstObject];
	XCTAssertEqualObjects(detection[@"$nfk"], @"detection", @"%s", resultJSON);
	XCTAssertEqualObjects(detection[@"label"], @"cat");
	XCTAssertEqualObjects(detection[@"boundingBox"], (@{ @"x": @0.375, @"y": @0.25, @"width": @0.5, @"height": @0.5 }));
	NSDictionary *inverted = outputs[NFKPluginFixtureValuesKey];
	XCTAssertEqualObjects(inverted[@"$nfk"], @"buffer", @"%s", resultJSON);
	const nfk_buffer *bytes = nfk_result_buffer(result, [inverted[@"index"] unsignedIntValue]);
	XCTAssert(bytes != NULL && bytes->length == sizeof values);
	if (bytes != NULL && bytes->length == sizeof values) {
		const float *out = bytes->bytes;
		XCTAssertEqual(out[0], 1.0f);
		XCTAssertEqual(out[1], 0.75f);
		XCTAssertEqual(out[3], 0.0f);
	}
	nfk_result_free(result);

	NSError *probeError = nil;
	NFKPluginBackend *probe = [shared backendForModelIdentifier:model options:nil error:&probeError];
	XCTAssertEqual(probe.isolation, NFKPluginIsolationProcess, @"%@", probeError);
	XCTAssertEqual(probe.process.state, NFKPluginProcessStateReady, @"the C API's run started the plugin's process");
	XCTAssertGreaterThan(probe.process.processIdentifier, 0);
	nfk_backend_free(backend);
}

#pragma mark Lazy hosting

- (void)testTheServersIdleUnloadEndsEveryKindsProcess
{
	[self requireHost];
	for (NFKPluginEntryKind kind in self.kinds) {
		NFKPluginRegistry *registry = [self registryListing:[self fixtureOfKind:kind version:@"1.0.0"]];
		NSString *model = [self model:NFKPluginFixtureInvertModel ofKind:kind];
		NFKInferenceServer *server = [[NFKInferenceServer alloc] init];
		server.residencyCheckInterval = 0;
		server.unloadGracePeriod = 0;
		server.idleUnloadInterval = 0.2;
		NSMutableArray<NFKPluginProcess *> *processes = [NSMutableArray array];
		[server addBackendProvider:^id<NFKInferenceBackend> _Nullable(NSError **providerError) {
			NFKPluginBackend *backend = [registry backendForModelIdentifier:model options:nil error:providerError];
			backend.process.idleExitInterval = 0.5;
			@synchronized (processes) {
				if (backend != nil) {
					[processes addObject:backend.process];
				}
			}
			return backend;
		} forModelName:kind];
		XCTestExpectation *loaded = [self expectationWithDescription:[@"loaded " stringByAppendingString:kind]];
		[server loadModelNamed:kind completionHandler:^(NSError *loadError) {
			XCTAssertNil(loadError, @"%@", kind);
			[loaded fulfill];
		}];
		[self waitForExpectations:@[ loaded ] timeout:30];
		@autoreleasepool {
			NSError *error = nil;
			NFKInferenceResult *result = [[server backendForModelName:kind] runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKPluginFixtureValuesKey: @[ @0.25 ] }]
																							 error:&error];
			XCTAssertEqualObjects(result.outputs[NFKPluginFixtureValuesKey], (@[ @0.75 ]), @"%@: %@", kind, error);
		}
		NFKPluginProcess *process = nil;
		@synchronized (processes) {
			process = processes.firstObject;
		}
		pid_t pid = process.processIdentifier;
		XCTAssertGreaterThan(pid, 0, @"%@", kind);
		XCTAssertTrue(NFKPluginTestWaitUntil(10, ^BOOL {
			return [[server unloadIdleModels] containsObject:kind];
		}), @"%@: the server unloads the idle model", kind);
		[self assertProcessExits:process pid:pid kind:kind];
		[server removeBackendForModelName:kind];
	}
}

@end

#endif
