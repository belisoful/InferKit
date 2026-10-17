//
//  NFKPluginRegistryTests.m
//  InferKitTests
//
//  The plugin registry over the sample plugins: listing reads manifests and starts nothing, each model
//  joins the catalog under "<plugin>/<model>" and builds through the catalog's build entry, a built-in
//  identifier and a folder another account could change are refused, a plugin is removed and reloaded to
//  a second version while a run is in flight, trusted plugins run in this process with the outputs they
//  give in their own, and a server's idle unload ends a plugin's process. In-process runs load the
//  fixtures at 1.0.0 from their cache only, as NFKPluginLoaderTests does, since a loaded plugin stays
//  for the life of the process.
//

#import <XCTest/XCTest.h>
#import <InferKit/InferKit.h>

#if TARGET_OS_OSX

#import "NFKPluginFixtures.h"
#import "NFKPluginTestPeer.h"
#import "../../Sources/InferKit/NFKPluginProcessPrivate.h"
#import <CoreVideo/CoreVideo.h>
#import <mach-o/dyld.h>
#import <signal.h>
#import <sys/stat.h>

/*! The path with symbolic links resolved, or the path itself. */
static NSString *NFKRegistryTestResolvedPath(const char *path)
{
	char resolved[PATH_MAX];
	return realpath(path, resolved) != NULL ? [NSString stringWithUTF8String:resolved] : [NSString stringWithUTF8String:path];
}

/*! Whether dyld has loaded the image at url into this process. */
static BOOL NFKRegistryTestImageIsLoaded(NSURL *url)
{
	NSString *target = NFKRegistryTestResolvedPath(url.fileSystemRepresentation);
	for (uint32_t index = 0; index < _dyld_image_count(); index++) {
		const char *name = _dyld_get_image_name(index);
		if (name != NULL && [NFKRegistryTestResolvedPath(name) isEqualToString:target]) {
			return YES;
		}
	}
	return NO;
}

@interface NFKPluginRegistryTests : XCTestCase
@property (nonatomic, strong) NSMutableArray<NSURL *> *temporaryURLs;
@end

@implementation NFKPluginRegistryTests

- (void)setUp
{
	[super setUp];
	self.temporaryURLs = [NSMutableArray array];
}

- (void)tearDown
{
	for (NSURL *url in self.temporaryURLs) {
		chmod(url.fileSystemRepresentation, 0755);
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

/*! A registry over an empty catalog that accepts the fixtures' ad hoc signatures and launches the host
	built beside the tests. The catalog plans against a 64 GiB working set, so no build depends on the
	memory of the machine that runs the tests. */
- (NFKPluginRegistry *)registry
{
	NFKModelCatalog *catalog = [[NFKModelCatalog alloc] init];
	catalog.memoryBudget = [NFKModelMemoryBudget budgetWithWorkingSetByteCount:(NSInteger)64 << 30];
	NFKPluginRegistry *registry = [[NFKPluginRegistry alloc] initWithCatalog:catalog];
	registry.signaturePolicy = NFKPluginSignaturePolicy.developmentPolicy;
	NSURL *host = [NFKPluginFixtures hostExecutableURL];
	if (host != nil) {
		registry.hostExecutableURL = host;
	}
	return registry;
}

- (void)requireHost
{
	if ([NFKPluginFixtures hostExecutableURL] == nil) {
		XCTSkip(@"the inferkit-plugin-host product is not built beside the tests");
	}
}

/*! A new folder, mode 0755, holding a copy of each bundle. */
- (NSURL *)pluginsFolderWithCopiesOf:(NSArray<NSURL *> *)bundles
{
	NSError *error = nil;
	NSURL *folder = [NFKPluginFixtures folderWithCopiesOfBundles:bundles error:&error];
	XCTAssertNotNil(folder, @"%@", error);
	if (folder != nil) {
		[self.temporaryURLs addObject:folder];
	}
	return folder;
}

/*! Rewrites the manifest of the bundle at url through edit. */
- (void)editManifestOfBundle:(NSURL *)url edit:(void (^)(NSMutableDictionary *manifest))edit
{
	NSError *error = nil;
	XCTAssertTrue([NFKPluginFixtures editManifestOfBundleAtURL:url edit:edit error:&error], @"%@", error);
}

/*! A copy of a fixture under identifier, in a folder of its own, signed ad hoc again. */
- (nullable NSURL *)signedCopyOfFixture:(NSURL *)fixture identifier:(NSString *)identifier
{
	NSURL *copy = [[self pluginsFolderWithCopiesOf:@[ fixture ]] URLByAppendingPathComponent:fixture.lastPathComponent];
	NSURL *infoURL = [copy URLByAppendingPathComponent:@"Contents/Info.plist"];
	NSMutableDictionary *info = [NSMutableDictionary dictionaryWithContentsOfURL:infoURL];
	info[@"CFBundleIdentifier"] = identifier;
	XCTAssertTrue([info writeToURL:infoURL error:NULL]);
	[self editManifestOfBundle:copy edit:^(NSMutableDictionary *manifest) {
		manifest[@"id"] = identifier;
	}];
	NSError *error = nil;
	if ([NFKPluginFixtures runProgram:@"/usr/bin/codesign" arguments:@[ @"--force", @"--sign", @"-", copy.path ] error:&error] == nil) {
		XCTFail(@"%@", error);
		return nil;
	}
	return copy;
}

/*! The text the registry's tool of that name returns for "hi". */
- (nullable NSString *)echoOf:(NFKPluginRegistry *)registry
{
	for (NFKInferenceTool *tool in registry.tools) {
		if ([tool.name isEqualToString:NFKPluginFixtureEchoTool]) {
			NFKToolResult *result = [tool invokeWithArguments:@{ @"text": @"hi" }];
			XCTAssertFalse(result.isError, @"%@", result.text);
			return result.text;
		}
	}
	XCTFail(@"no echo tool is listed");
	return nil;
}

- (NSString *)catalogIdentifierOfKind:(NFKPluginEntryKind)kind model:(NSString *)model
{
	return [NSString stringWithFormat:@"%@/%@", [NFKPluginFixtures pluginIdentifierForKind:kind], model];
}

- (void)assertError:(nullable NSError *)error domain:(NSString *)domain code:(NSInteger)code
{
	XCTAssertEqualObjects(error.domain, domain, @"%@", error);
	XCTAssertEqual(error.code, code, @"%@", error);
}

- (BOOL)waitUntil:(BOOL (^)(void))condition timeout:(NSTimeInterval)timeout
{
	NSDate *deadline = [NSDate dateWithTimeIntervalSinceNow:timeout];
	while (!condition()) {
		if (deadline.timeIntervalSinceNow <= 0) {
			return NO;
		}
		[NSThread sleepForTimeInterval:0.02];
	}
	return YES;
}

- (CVPixelBufferRef)newImageOfSide:(size_t)side CF_RETURNS_RETAINED
{
	CVPixelBufferRef image = [NFKPluginFixtures newPatternImageOfSide:side];
	XCTAssertTrue(image != NULL);
	return image;
}

- (nullable NSData *)pixelsOf:(nullable id)value
{
	return [NFKPluginFixtures pixelsOfImage:value];
}

/*! The invert model's outputs for a float array and a 64 × 64 image. */
- (NSDictionary<NSString *, id> *)invertOutputsOf:(id<NFKInferenceBackend>)backend
{
	NSError *error = nil;
	NFKInferenceResult *values = [backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKPluginFixtureValuesKey: @[ @0.25, @1.0 ] }]
														   error:&error];
	XCTAssertNotNil(values, @"%@", error);
	CVPixelBufferRef image = [self newImageOfSide:64];
	NFKInferenceResult *inverted = [backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputImage: (__bridge id)image }]
															 error:&error];
	CVPixelBufferRelease(image);
	XCTAssertNotNil(inverted, @"%@", error);
	NSMutableDictionary<NSString *, id> *outputs = [NSMutableDictionary dictionary];
	outputs[NFKPluginFixtureValuesKey] = values.outputs[NFKPluginFixtureValuesKey];
	outputs[NFKPluginFixtureVersionKey] = values.outputs[NFKPluginFixtureVersionKey];
	outputs[NFKOutputImage] = [self pixelsOf:inverted.outputs[NFKOutputImage]];
	return outputs;
}

#pragma mark Listing

- (void)testListingAFolderLoadsNoCodeAndStartsNoProcess
{
	NSURL *objc = [self fixtureOfKind:NFKPluginEntryKindObjC version:@"1.0.0"];
	NSURL *library = [self fixtureOfKind:NFKPluginEntryKindCABI version:@"1.0.0"];
	NSURL *folder = [self pluginsFolderWithCopiesOf:@[ objc, library ]];
	[self editManifestOfBundle:[folder URLByAppendingPathComponent:library.lastPathComponent] edit:^(NSMutableDictionary *manifest) {
		[manifest removeObjectForKey:@"tools"];
	}];
	NFKPluginRegistry *registry = [self registry];

	NSArray<NSError *> *errors = nil;
	NSArray<NFKPluginManifest *> *added = [registry addPluginsInDirectoryURL:folder errors:&errors];
	XCTAssertEqual(errors.count, 0u, @"%@", errors);
	NSArray<NSString *> *expected = [@[ [NFKPluginFixtures pluginIdentifierForKind:NFKPluginEntryKindObjC],
										[NFKPluginFixtures pluginIdentifierForKind:NFKPluginEntryKindCABI] ]
									 sortedArrayUsingSelector:@selector(compare:)];
	XCTAssertEqualObjects([[added valueForKey:@"identifier"] sortedArrayUsingSelector:@selector(compare:)], expected);
	XCTAssertEqualObjects([registry.manifests valueForKey:@"identifier"], expected);
	XCTAssertEqual(registry.catalog.descriptors.count, 4u);
	XCTAssertEqualObjects([registry.tools valueForKey:@"name"], @[ NFKPluginFixtureEchoTool ]);

	NSError *error = nil;
	NFKPluginBackend *direct = [registry backendForModelIdentifier:[self catalogIdentifierOfKind:NFKPluginEntryKindObjC model:NFKPluginFixtureInvertModel]
														   options:nil error:&error];
	XCTAssertNotNil(direct, @"%@", error);
	id<NFKInferenceBackend> built = [registry.catalog backendForModelIdentifier:[self catalogIdentifierOfKind:NFKPluginEntryKindCABI model:NFKPluginFixtureInvertModel]
																		 options:nil error:&error];
	XCTAssertTrue([(NSObject *)built isKindOfClass:NFKPluginBackend.class], @"%@ %@", built, error);
	XCTAssertTrue([direct.supportedInputKeys containsObject:NFKInputImage], @"the keys come from the manifest");

	for (NFKPluginBackend *backend in @[ direct, (NFKPluginBackend *)built ]) {
		XCTAssertEqual(backend.isolation, NFKPluginIsolationProcess);
		XCTAssertEqual(backend.process.state, NFKPluginProcessStateNotRunning);
		XCTAssertEqual(backend.process.processIdentifier, 0);
	}
	for (NSURL *bundle in @[ objc, library ]) {
		NSURL *copy = [folder URLByAppendingPathComponent:bundle.lastPathComponent];
		XCTAssertFalse(NFKRegistryTestImageIsLoaded([NFKPluginFixtures executableURLOfBundleAtURL:copy]), @"%@", copy);
	}
}

- (void)testCatalogIdentifiersAndDescriptorFields
{
	NFKPluginRegistry *registry = [self registry];
	NSError *error = nil;
	NFKPluginManifest *manifest = [registry addPluginAtURL:[self fixtureOfKind:NFKPluginEntryKindObjC version:@"1.0.0"] error:&error];
	XCTAssertNotNil(manifest, @"%@", error);
	NSString *invert = [self catalogIdentifierOfKind:NFKPluginEntryKindObjC model:NFKPluginFixtureInvertModel];
	NSString *slow = [self catalogIdentifierOfKind:NFKPluginEntryKindObjC model:NFKPluginFixtureSlowModel];
	XCTAssertEqualObjects([registry.catalog.descriptors valueForKey:@"identifier"], (@[ invert, slow ]));

	NFKModelDescriptor *descriptor = [registry.catalog descriptorWithIdentifier:invert];
	XCTAssertEqualObjects(descriptor.registryName, NFKPluginFixtureInvertModel);
	XCTAssertEqualObjects(descriptor.engine, NFKModelEnginePlugin);
	XCTAssertNil(descriptor.providerClassName);
	XCTAssertEqualObjects(descriptor.displayName, @"Invert");
	XCTAssertEqualObjects(descriptor.family, @"sample-invert");
	NSDictionary *plugin = @{ @"id": manifest.identifier, @"version": @"1.0.0", @"kind": NFKPluginEntryKindObjC };
	XCTAssertEqualObjects(descriptor.extensions[@"plugin"], plugin);
	XCTAssertEqualObjects([descriptor.inputs valueForKey:@"key"], (@[ NFKInputImage, NFKPluginFixtureValuesKey, NFKPluginFixtureRegionKey ]));
	XCTAssertNotNil([registry.catalog manifestDataWithError:&error], @"the descriptors are JSON: %@", error);

	XCTAssertEqualObjects([registry manifestForModelIdentifier:slow], manifest);
	XCTAssertNil([registry manifestForModelIdentifier:NFKPluginFixtureSlowModel], @"a model is reached by its catalog identifier");
	XCTAssertNil([registry backendForModelIdentifier:@"com.example.absent/invert" options:nil error:&error]);
	[self assertError:error domain:NFKPluginErrorDomain code:NFKPluginErrorUnknownModel];

	NSURL *elsewhere = [NSURL fileURLWithPath:@"/usr/bin/true"];
	registry.hostExecutableURL = elsewhere;
	XCTAssertEqualObjects(registry.hostExecutableURL, elsewhere);
	registry.hostExecutableURL = nil;
	NFKPluginRegistry *fresh = [[NFKPluginRegistry alloc] initWithCatalog:[[NFKModelCatalog alloc] init]];
	XCTAssertEqualObjects(registry.hostExecutableURL, fresh.hostExecutableURL, @"nil restores the search");
	XCTAssertEqualObjects(fresh.signaturePolicy, NFKPluginSignaturePolicy.defaultPolicy);
}

#pragma mark The catalog's build entry

- (void)testTheCatalogBuildEntryReachesAPluginBackendAndRunsIt
{
	[self requireHost];
	NFKPluginRegistry *registry = [self registry];
	NSError *error = nil;
	XCTAssertNotNil([registry addPluginAtURL:[self fixtureOfKind:NFKPluginEntryKindCABI version:@"1.0.0"] error:&error], @"%@", error);
	NSString *invert = [self catalogIdentifierOfKind:NFKPluginEntryKindCABI model:NFKPluginFixtureInvertModel];
	id<NFKInferenceBackend> backend = [registry.catalog backendForModelIdentifier:invert options:nil error:&error];
	XCTAssertTrue([(NSObject *)backend isKindOfClass:NFKPluginBackend.class], @"%@ %@", backend, error);
	XCTAssertEqualObjects(backend.backendIdentifier, ([NSString stringWithFormat:@"plugin:%@", invert]));
	NFKInferenceResult *result = [backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKPluginFixtureValuesKey: @[ @0.25, @1.0 ] }]
														   error:&error];
	XCTAssertNotNil(result, @"%@", error);
	XCTAssertEqualObjects(result.outputs[NFKPluginFixtureValuesKey], (@[ @0.75, @0.0 ]));
	XCTAssertEqualObjects(result.outputs[NFKPluginFixtureVersionKey], @"1.0.0");

	NSString *slow = [self catalogIdentifierOfKind:NFKPluginEntryKindCABI model:NFKPluginFixtureSlowModel];
	id<NFKInferenceBackend> counting = [registry.catalog backendForModelIdentifier:slow
																			options:@{ NFKPluginFixtureStepsKey: @3, NFKPluginFixtureStepMillisecondsKey: @1 }
																			  error:&error];
	XCTAssertTrue([(NSObject *)counting isKindOfClass:NFKDefaultedBackend.class], @"request options become defaults: %@ %@", counting, error);
	XCTAssertTrue([(NSObject *)((NFKDefaultedBackend *)counting).backend isKindOfClass:NFKPluginBackend.class]);
	NFKInferenceResult *counted = [counting runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{}] error:&error];
	XCTAssertNotNil(counted, @"%@", error);
	XCTAssertEqualObjects(counted.outputs[NFKPluginFixtureStepsKey], @3);

	XCTAssertNil([registry.catalog backendForModelIdentifier:slow options:@{ NFKPluginFixtureStepsKey: @0 } error:&error],
				 @"steps has a minimum of 1");
	[self assertError:error domain:NFKModelCatalogErrorDomain code:NFKModelCatalogErrorInvalidOptions];
}

- (void)testTheCatalogChecksAPluginModelsMemory
{
	NSURL *folder = [self pluginsFolderWithCopiesOf:@[ [self fixtureOfKind:NFKPluginEntryKindObjC version:@"1.0.0"] ]];
	NSURL *bundle = [NSFileManager.defaultManager contentsOfDirectoryAtURL:folder includingPropertiesForKeys:nil options:0 error:NULL].firstObject;
	NSURL *manifestURL = [bundle URLByAppendingPathComponent:@"Contents/Resources/inferkit-plugin.json"];
	NSMutableDictionary *manifest = [NSJSONSerialization JSONObjectWithData:[NSData dataWithContentsOfURL:manifestURL]
																	options:NSJSONReadingMutableContainers error:NULL];
	NSNumber *huge = @(1LL << 60);
	manifest[@"models"][0][@"variants"] = @[ @{ @"id": @"huge", @"isDefault": @YES, @"residentBytes": huge, @"peakBytes": huge } ];
	XCTAssertTrue([[NSJSONSerialization dataWithJSONObject:manifest options:0 error:NULL] writeToURL:manifestURL atomically:YES]);

	NFKPluginRegistry *registry = [self registry];
	NSError *error = nil;
	XCTAssertNotNil([registry addPluginAtURL:bundle error:&error], @"listing reads the manifest and checks no signature: %@", error);
	NSString *invert = [self catalogIdentifierOfKind:NFKPluginEntryKindObjC model:NFKPluginFixtureInvertModel];
	XCTAssertNil([registry.catalog backendForModelIdentifier:invert options:nil error:&error]);
	[self assertError:error domain:NFKModelCatalogErrorDomain code:NFKModelCatalogErrorDoesNotFit];

	id<NFKInferenceBackend> backend = [registry.catalog backendForModelIdentifier:invert
																		  options:@{ NFKModelLoadOptionIgnoreMemoryBudget: @YES }
																			error:&error];
	XCTAssertTrue([(NSObject *)backend isKindOfClass:NFKPluginBackend.class], @"%@ %@", backend, error);
	XCTAssertNil(((NFKPluginBackend *)backend).options[NFKModelLoadOptionIgnoreMemoryBudget], @"the plugin's build never sees it");
	XCTAssertEqual(((NFKPluginBackend *)backend).process.state, NFKPluginProcessStateNotRunning);
}

#pragma mark Refusals

- (void)testABuiltInIdentifierIsRefused
{
	NFKModelCatalog *catalog = [[NFKModelCatalog alloc] init];
	NSString *invert = [self catalogIdentifierOfKind:NFKPluginEntryKindObjC model:NFKPluginFixtureInvertModel];
	NSError *error = nil;
	NFKModelDescriptor *builtIn = [NFKModelDescriptor descriptorWithDictionary:@{ @"id": invert, @"family": @"built-in",
																				   @"engine": NFKModelEngineApple }
																		 error:&error];
	XCTAssertNotNil(builtIn, @"%@", error);
	[catalog addDescriptor:builtIn];
	NFKPluginRegistry *registry = [[NFKPluginRegistry alloc] initWithCatalog:catalog];
	NSURL *fixture = [self fixtureOfKind:NFKPluginEntryKindObjC version:@"1.0.0"];

	XCTAssertNil([registry addPluginAtURL:fixture error:&error]);
	[self assertError:error domain:NFKPluginErrorDomain code:NFKPluginErrorIdentifierInUse];
	XCTAssertEqualObjects([catalog descriptorWithIdentifier:invert].engine, NFKModelEngineApple);
	XCTAssertNil([catalog descriptorWithIdentifier:[self catalogIdentifierOfKind:NFKPluginEntryKindObjC model:NFKPluginFixtureSlowModel]],
				 @"a refused plugin lists none of its models");
	XCTAssertEqual(registry.manifests.count, 0u);

	NFKPluginRegistry *other = [self registry];
	XCTAssertNotNil([other addPluginAtURL:fixture error:&error], @"%@", error);
	XCTAssertNotNil([other addPluginAtURL:fixture error:&error], @"adding the same bundle again reloads it: %@", error);
	NSURL *copy = [[self pluginsFolderWithCopiesOf:@[ fixture ]] URLByAppendingPathComponent:fixture.lastPathComponent];
	XCTAssertNil([other addPluginAtURL:copy error:&error]);
	[self assertError:error domain:NFKPluginErrorDomain code:NFKPluginErrorIdentifierInUse];
	XCTAssertFalse([other reloadPluginWithIdentifier:@"com.example.absent" error:&error]);
	[self assertError:error domain:NFKPluginErrorDomain code:NFKPluginErrorUnknownPlugin];
}

- (void)testAGroupWritableFolderIsRefused
{
	NSURL *fixture = [self fixtureOfKind:NFKPluginEntryKindCABI version:@"1.0.0"];
	NSURL *folder = [self pluginsFolderWithCopiesOf:@[ fixture ]];
	XCTAssertEqual(chmod(folder.fileSystemRepresentation, 0775), 0);
	NFKPluginRegistry *registry = [self registry];

	NSArray<NSError *> *errors = nil;
	XCTAssertEqual([registry addPluginsInDirectoryURL:folder errors:&errors].count, 0u);
	XCTAssertEqual(errors.count, 1u, @"%@", errors);
	[self assertError:errors.firstObject domain:NFKPluginErrorDomain code:NFKPluginErrorLoadFailed];
	XCTAssertTrue([errors.firstObject.localizedDescription containsString:@"is writable by its group"], @"%@", errors.firstObject);
	XCTAssertEqual(registry.catalog.descriptors.count, 0u);

	XCTAssertEqual(chmod(folder.fileSystemRepresentation, 0755), 0);
	NSURL *library = [NFKPluginFixtures executableURLOfBundleAtURL:[folder URLByAppendingPathComponent:fixture.lastPathComponent]];
	XCTAssertEqual(chmod(library.fileSystemRepresentation, 0757), 0);
	XCTAssertEqual([registry addPluginsInDirectoryURL:folder errors:&errors].count, 0u, @"a file inside the bundle counts too");
	XCTAssertTrue([errors.firstObject.localizedDescription containsString:@"is writable by every user"], @"%@", errors.firstObject);

	XCTAssertEqual(chmod(library.fileSystemRepresentation, 0755), 0);
	XCTAssertEqual([registry addPluginsInDirectoryURL:folder errors:&errors].count, 1u, @"%@", errors);
	XCTAssertEqual(errors.count, 0u, @"%@", errors);
}

- (void)testAToolNameAnotherPluginDeclaresIsRefused
{
	NFKPluginRegistry *registry = [self registry];
	NSError *error = nil;
	NSString *holder = [NFKPluginFixtures pluginIdentifierForKind:NFKPluginEntryKindObjC];
	XCTAssertNotNil([registry addPluginAtURL:[self fixtureOfKind:NFKPluginEntryKindObjC version:@"1.0.0"] error:&error], @"%@", error);
	NSURL *library = [self fixtureOfKind:NFKPluginEntryKindCABI version:@"1.0.0"];

	XCTAssertNil([registry addPluginAtURL:library error:&error], @"both fixtures declare the echo tool");
	[self assertError:error domain:NFKPluginErrorDomain code:NFKPluginErrorIdentifierInUse];
	XCTAssertTrue([error.localizedDescription containsString:NFKPluginFixtureEchoTool], @"%@", error);
	XCTAssertTrue([error.localizedDescription containsString:holder], @"%@", error);
	XCTAssertNil([registry.catalog descriptorWithIdentifier:[self catalogIdentifierOfKind:NFKPluginEntryKindCABI model:NFKPluginFixtureInvertModel]]);
	XCTAssertEqualObjects([registry.manifests valueForKey:@"identifier"], @[ holder ]);

	XCTAssertTrue([registry reloadPluginWithIdentifier:holder error:&error], @"a plugin's tools do not conflict with its own: %@", error);
	[registry removePluginWithIdentifier:holder];
	XCTAssertNotNil([registry addPluginAtURL:library error:&error], @"a removed plugin's tool names are free: %@", error);
}

#pragma mark Plugin options and sandboxed hosts

- (void)testPluginOptionValuesAreCheckedAndReachThePluginsProcess
{
	[self requireHost];
	NFKPluginRegistry *registry = [self registry];
	NSError *error = nil;
	NFKPluginManifest *manifest = [registry addPluginAtURL:[self fixtureOfKind:NFKPluginEntryKindCABI version:@"1.0.0"] error:&error];
	XCTAssertNotNil(manifest, @"%@", error);
	NSString *identifier = manifest.identifier;
	XCTAssertEqualObjects([registry optionValuesForPluginIdentifier:identifier], @{});

	XCTAssertFalse([registry setOptionValues:@{ @"bogus": @1 } forPluginIdentifier:identifier error:&error]);
	[self assertError:error domain:NFKModelCatalogErrorDomain code:NFKModelCatalogErrorInvalidOptions];
	NSArray<NFKModelOptionProblem *> *problems = error.userInfo[NFKModelOptionProblemsKey];
	XCTAssertEqualObjects(problems.firstObject.code, NFKModelOptionProblemUnknownKey, @"%@", problems);
	XCTAssertEqualObjects(problems.firstObject.key, @"bogus");
	XCTAssertFalse([registry setOptionValues:@{ NFKPluginFixtureEchoPrefixOption: @[ @1 ] } forPluginIdentifier:identifier error:&error]);
	[self assertError:error domain:NFKModelCatalogErrorDomain code:NFKModelCatalogErrorInvalidOptions];
	XCTAssertEqualObjects([registry optionValuesForPluginIdentifier:identifier], @{}, @"a refused value changes nothing");
	XCTAssertFalse([registry setOptionValues:@{} forPluginIdentifier:@"com.example.absent" error:&error]);
	[self assertError:error domain:NFKPluginErrorDomain code:NFKPluginErrorUnknownPlugin];

	XCTAssertTrue([registry setOptionValues:@{ NFKPluginFixtureEchoPrefixOption: @"> " } forPluginIdentifier:identifier error:&error], @"%@", error);
	XCTAssertEqualObjects([registry optionValuesForPluginIdentifier:identifier], (@{ NFKPluginFixtureEchoPrefixOption: @"> " }));
	XCTAssertEqualObjects([self echoOf:registry], @"> hi", @"hello carries the values to the process's first start");

	NFKPluginBackend *backend = [registry backendForModelIdentifier:[self catalogIdentifierOfKind:NFKPluginEntryKindCABI model:NFKPluginFixtureInvertModel]
															options:nil error:&error];
	pid_t first = backend.process.processIdentifier;
	XCTAssertGreaterThan(first, 0);
	XCTAssertTrue([registry setOptionValues:@{ NFKPluginFixtureEchoPrefixOption: @"# " } forPluginIdentifier:identifier error:&error], @"%@", error);
	XCTAssertEqualObjects([self echoOf:registry], @"# hi", @"a running process restarts with the new values");
	XCTAssertNotEqual(backend.process.processIdentifier, first);
	XCTAssertTrue([registry setOptionValues:nil forPluginIdentifier:identifier error:&error], @"%@", error);
	XCTAssertEqualObjects([registry optionValuesForPluginIdentifier:identifier], @{});
}

- (void)testPluginOptionValuesReachATrustedPluginsContext
{
	NSString *identifier = @"com.example.inferkit.registry-options";
	NSURL *copy = [self signedCopyOfFixture:[self fixtureOfKind:NFKPluginEntryKindCABI version:@"1.0.0"] identifier:identifier];
	NFKPluginRegistry *registry = [self registry];
	NSError *error = nil;
	XCTAssertNotNil([registry addPluginAtURL:copy error:&error], @"%@", error);
	XCTAssertTrue([registry setOptionValues:@{ NFKPluginFixtureEchoPrefixOption: @"in: " } forPluginIdentifier:identifier error:&error], @"%@", error);
	[registry setTrusted:YES forPluginIdentifier:identifier];
	NFKPluginBackend *backend = [registry backendForModelIdentifier:[identifier stringByAppendingString:@"/invert"] options:nil error:&error];
	XCTAssertEqual(backend.isolation, NFKPluginIsolationInProcess, @"%@", error);
	XCTAssertEqualObjects([self echoOf:registry], @"in: hi", @"the C plugin reads the values from its host table's options");
}

- (void)testAReloadDropsOptionValuesTheNewManifestNoLongerTakes
{
	NSURL *fixture = [self fixtureOfKind:NFKPluginEntryKindCABI version:@"1.0.0"];
	NSURL *bundle = [[self pluginsFolderWithCopiesOf:@[ fixture ]] URLByAppendingPathComponent:fixture.lastPathComponent];
	[self editManifestOfBundle:bundle edit:^(NSMutableDictionary *manifest) {
		manifest[@"options"] = @[
			@{ @"key": NFKPluginFixtureEchoPrefixOption, @"kind": @"text", @"default": @"" },
			@{ @"key": @"tileSize", @"kind": @"integer", @"default": @64, @"minimum": @1, @"maximum": @512 },
			@{ @"key": @"mode", @"kind": @"choice", @"default": @"fast", @"choices": @[ @{ @"value": @"fast" }, @{ @"value": @"exact" } ] },
		];
	}];
	NFKPluginRegistry *registry = [self registry];
	NSError *error = nil;
	NFKPluginManifest *manifest = [registry addPluginAtURL:bundle error:&error];
	XCTAssertNotNil(manifest, @"listing reads the manifest and checks no signature: %@", error);
	NSString *identifier = manifest.identifier;
	NSDictionary *values = @{ NFKPluginFixtureEchoPrefixOption: @"> ", @"tileSize": @"256", @"mode": @"exact" };
	XCTAssertTrue([registry setOptionValues:values forPluginIdentifier:identifier error:&error], @"%@", error);
	XCTAssertEqualObjects([registry optionValuesForPluginIdentifier:identifier][@"tileSize"], @256);
	NFKPluginBackend *backend = [registry backendForModelIdentifier:[self catalogIdentifierOfKind:NFKPluginEntryKindCABI model:NFKPluginFixtureInvertModel]
															options:nil error:&error];
	XCTAssertNotNil(backend, @"%@", error);

	[self editManifestOfBundle:bundle edit:^(NSMutableDictionary *edited) {
		edited[@"options"] = @[
			@{ @"key": NFKPluginFixtureEchoPrefixOption, @"kind": @"text", @"default": @"" },
			@{ @"key": @"tileSize", @"kind": @"integer", @"default": @64, @"minimum": @1, @"maximum": @128 },
		];
	}];
	XCTAssertTrue([registry reloadPluginWithIdentifier:identifier error:&error], @"%@", error);
	NSDictionary *kept = @{ NFKPluginFixtureEchoPrefixOption: @"> " };
	XCTAssertEqualObjects([registry optionValuesForPluginIdentifier:identifier], kept,
						  @"mode is no longer declared, and 256 is above tileSize's new maximum");
	XCTAssertEqualObjects(backend.process.pluginOptions, kept, @"the process's next hello carries what remains");

	[registry removePluginWithIdentifier:identifier];
	[self editManifestOfBundle:bundle edit:^(NSMutableDictionary *edited) {
		[edited removeObjectForKey:@"options"];
	}];
	XCTAssertNotNil([registry addPluginAtURL:bundle error:&error], @"%@", error);
	XCTAssertEqualObjects([registry optionValuesForPluginIdentifier:identifier], @{}, @"adding the plugin again checks the values too");
}

- (void)testASandboxedPrefixOffersFileDescriptorsInHello
{
	NFKPluginTestHarness *harness = [[NFKPluginTestHarness alloc] init];
	for (NSString *prefix in @[ @"TEAMID.group.example/", @"" ]) {
		NFKPluginProcess *process = [[NFKPluginProcess alloc] initWithManifest:NFKPluginTestManifest(@"1.0.0", 300) hostExecutableURL:nil];
		process.launcherFactory = harness.factory;
		process.sharedMemoryPrefix = prefix.length > 0 ? prefix : nil;
		NSError *error = nil;
		XCTAssertTrue([process startWithError:&error], @"%@", error);
		NFKPluginTestPeer *peer = harness.peers.lastObject;
		NSDictionary *hello = [peer waitForMessageOfType:NFKPluginMessageHello timeout:10];
		XCTAssertNotNil(hello, @"%@", peer.diagnosticDescription);
		if (prefix.length > 0) {
			XCTAssertEqualObjects(hello[@"capabilities"], @[ @"file-descriptors" ]);
			XCTAssertEqualObjects(hello[@"sharedMemoryPrefix"], prefix);
		} else {
			XCTAssertEqualObjects(hello[@"capabilities"], @[], @"an unsandboxed host offers nothing more");
			XCTAssertNil(hello[@"sharedMemoryPrefix"]);
		}
		XCTestExpectation *stopped = [self expectationWithDescription:@"stopped"];
		[process stopWithCompletionHandler:^{
			[stopped fulfill];
		}];
		[self waitForExpectations:@[ stopped ] timeout:20];
	}

	NFKPluginRegistry *registry = [self registry];
	NSError *error = nil;
	XCTAssertNotNil([registry addPluginAtURL:[self fixtureOfKind:NFKPluginEntryKindCABI version:@"1.0.0"] error:&error], @"%@", error);
	NFKPluginBackend *backend = [registry backendForModelIdentifier:[self catalogIdentifierOfKind:NFKPluginEntryKindCABI model:NFKPluginFixtureInvertModel]
															options:nil error:&error];
	XCTAssertNil(backend.process.sharedMemoryPrefix, @"unsandboxed by default");
	registry.sharedMemoryPrefix = @"TEAMID.group.example/";
	XCTAssertEqualObjects(backend.process.sharedMemoryPrefix, @"TEAMID.group.example/", @"a process the registry made follows it");
}

- (void)testLargeValuesCrossUnderASharedMemoryPrefix
{
	[self requireHost];
	NFKPluginRegistry *registry = [self registry];
	registry.sharedMemoryPrefix = @"NFKT/";
	NSError *error = nil;
	XCTAssertNotNil([registry addPluginAtURL:[self fixtureOfKind:NFKPluginEntryKindCABI version:@"1.0.0"] error:&error], @"%@", error);
	NFKPluginBackend *backend = [registry backendForModelIdentifier:[self catalogIdentifierOfKind:NFKPluginEntryKindCABI model:NFKPluginFixtureInvertModel]
															options:nil error:&error];
	CVPixelBufferRef image = [self newImageOfSide:256];
	NFKInferenceResult *inverted = [backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputImage: (__bridge id)image }]
															 error:&error];
	NSData *expected = [self pixelsOf:(__bridge id)image];
	CVPixelBufferRelease(image);
	NSData *pixels = [self pixelsOf:inverted.outputs[NFKOutputImage]];
	XCTAssertEqual(pixels.length, 256u * 256u * 4u, @"a 256 KiB image crosses as an attachment: %@", error);
	const uint8_t *in = expected.bytes;
	const uint8_t *out = pixels.bytes;
	for (NSUInteger index = 0; index + 4 <= MIN(pixels.length, expected.length); index += 4) {
		if (out[index] != 255 - in[index] || out[index + 3] != in[index + 3]) {
			XCTFail(@"pixel %lu is %u %u, from %u %u", (unsigned long)(index / 4), out[index], out[index + 3], in[index], in[index + 3]);
			break;
		}
	}
}

#pragma mark Removing and reloading

- (void)testRemovingAPluginRemovesItsModels
{
	NFKPluginRegistry *registry = [self registry];
	NSError *error = nil;
	NFKPluginManifest *manifest = [registry addPluginAtURL:[self fixtureOfKind:NFKPluginEntryKindCABI version:@"1.0.0"] error:&error];
	XCTAssertNotNil(manifest, @"%@", error);
	NSString *invert = [self catalogIdentifierOfKind:NFKPluginEntryKindCABI model:NFKPluginFixtureInvertModel];
	XCTAssertNotNil([registry.catalog descriptorWithIdentifier:invert]);

	[registry removePluginWithIdentifier:manifest.identifier];
	XCTAssertEqual(registry.catalog.descriptors.count, 0u);
	XCTAssertEqual(registry.manifests.count, 0u);
	XCTAssertEqual(registry.tools.count, 0u);
	XCTAssertNil([registry manifestForModelIdentifier:invert]);
	XCTAssertNil([registry backendForModelIdentifier:invert options:nil error:&error]);
	[self assertError:error domain:NFKPluginErrorDomain code:NFKPluginErrorUnknownModel];
	XCTAssertNil([registry.catalog backendForModelIdentifier:invert options:nil error:&error]);
	[self assertError:error domain:NFKModelCatalogErrorDomain code:NFKModelCatalogErrorUnknownModel];
	[registry removePluginWithIdentifier:manifest.identifier];
}

- (void)testReloadingToASecondVersionMidRun
{
	[self requireHost];
	NSURL *first = [self fixtureOfKind:NFKPluginEntryKindCABI version:@"1.0.0"];
	NSURL *second = [self fixtureOfKind:NFKPluginEntryKindCABI version:@"2.0.0"];
	NSURL *folder = [self pluginsFolderWithCopiesOf:@[ first ]];
	NSURL *installed = [folder URLByAppendingPathComponent:first.lastPathComponent];
	NFKPluginRegistry *registry = [self registry];
	NSError *error = nil;
	NFKPluginManifest *manifest = [registry addPluginAtURL:installed error:&error];
	XCTAssertNotNil(manifest, @"%@", error);
	NSString *slow = [self catalogIdentifierOfKind:NFKPluginEntryKindCABI model:NFKPluginFixtureSlowModel];
	NFKPluginBackend *backend = [registry backendForModelIdentifier:slow options:nil error:&error];
	XCTAssertNotNil(backend, @"%@", error);

	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{}
															   parameters:@{ NFKPluginFixtureStepsKey: @20, NFKPluginFixtureStepMillisecondsKey: @50 }];
	NFKInferenceJob *job = [backend submitInferenceJobForRequest:request];
	__block NSInteger reports = 0;
	job.progressHandler = ^(NFKInferenceJob *updated) {
		@synchronized (self) {
			reports += 1;
		}
	};
	XCTAssertTrue([self waitUntil:^BOOL {
		@synchronized (self) {
			return reports > 0;
		}
	} timeout:60], @"the run reports progress: %@", backend.process.diagnosticDescription);

	XCTAssertTrue([NSFileManager.defaultManager removeItemAtURL:installed error:&error], @"%@", error);
	XCTAssertTrue([NSFileManager.defaultManager copyItemAtURL:second toURL:installed error:&error], @"%@", error);
	XCTAssertTrue([registry reloadPluginWithIdentifier:manifest.identifier error:&error], @"%@", error);
	XCTAssertEqualObjects([registry manifestForModelIdentifier:slow].version, @"2.0.0");
	XCTAssertEqualObjects([registry.catalog descriptorWithIdentifier:slow].extensions[@"plugin"][@"version"], @"2.0.0");

	XCTAssertTrue([self waitUntil:^BOOL {
		return job.status >= NFKInferenceJobStatusSucceeded;
	} timeout:120], @"%@", backend.process.diagnosticDescription);
	XCTAssertEqual(job.status, NFKInferenceJobStatusSucceeded, @"%@", job.error);
	XCTAssertEqualObjects(job.result.outputs[NFKPluginFixtureVersionKey], @"1.0.0", @"the run in flight finishes on the first version");
	XCTAssertEqualObjects(job.result.outputs[NFKPluginFixtureStepsKey], @20);

	NFKInferenceRequest *brief = [NFKInferenceRequest requestWithInputs:@{} parameters:@{ NFKPluginFixtureStepsKey: @1 }];
	NFKInferenceResult *after = [backend runInferenceForRequest:brief error:&error];
	XCTAssertNotNil(after, @"%@", error);
	XCTAssertEqualObjects(after.outputs[NFKPluginFixtureVersionKey], @"2.0.0", @"the next run reaches the second version");
	XCTAssertEqualObjects(backend.process.manifest.version, @"2.0.0");
	XCTAssertEqual(backend.process.state, NFKPluginProcessStateReady);
}

- (void)testReloadCompletionReportsTheRestart
{
	NFKPluginRegistry *registry = [self registry];
	XCTestExpectation *unknown = [self expectationWithDescription:@"unknown plugin"];
	[registry reloadPluginWithIdentifier:@"com.example.absent" completionHandler:^(NSError *error) {
		[self assertError:error domain:NFKPluginErrorDomain code:NFKPluginErrorUnknownPlugin];
		[unknown fulfill];
	}];
	[self waitForExpectations:@[ unknown ] timeout:10];

	NSURL *first = [self fixtureOfKind:NFKPluginEntryKindCABI version:@"1.0.0"];
	NSURL *folder = [self pluginsFolderWithCopiesOf:@[ first ]];
	NSURL *installed = [folder URLByAppendingPathComponent:first.lastPathComponent];
	NSError *error = nil;
	NFKPluginManifest *manifest = [registry addPluginAtURL:installed error:&error];
	XCTAssertNotNil(manifest, @"%@", error);
	XCTestExpectation *stopped = [self expectationWithDescription:@"no process"];
	[registry reloadPluginWithIdentifier:manifest.identifier completionHandler:^(NSError *reloadError) {
		XCTAssertNil(reloadError);
		[stopped fulfill];
	}];
	[self waitForExpectations:@[ stopped ] timeout:10];

	[self requireHost];
	NFKPluginBackend *backend = [registry backendForModelIdentifier:[self catalogIdentifierOfKind:NFKPluginEntryKindCABI model:NFKPluginFixtureInvertModel]
															options:nil error:&error];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKPluginFixtureValuesKey: @[ @0.5 ] }];
	XCTAssertEqualObjects([backend runInferenceForRequest:request error:&error].outputs[NFKPluginFixtureVersionKey], @"1.0.0", @"%@", error);
	XCTAssertTrue([NSFileManager.defaultManager removeItemAtURL:installed error:&error], @"%@", error);
	XCTAssertTrue([NSFileManager.defaultManager copyItemAtURL:[self fixtureOfKind:NFKPluginEntryKindCABI version:@"2.0.0"] toURL:installed error:&error], @"%@", error);
	XCTestExpectation *restarted = [self expectationWithDescription:@"restarted"];
	[registry reloadPluginWithIdentifier:manifest.identifier completionHandler:^(NSError *reloadError) {
		XCTAssertNil(reloadError);
		XCTAssertEqual(backend.process.state, NFKPluginProcessStateReady, @"the restart has finished");
		XCTAssertEqualObjects(backend.process.manifest.version, @"2.0.0");
		[restarted fulfill];
	}];
	[self waitForExpectations:@[ restarted ] timeout:120];
	XCTAssertEqualObjects([backend runInferenceForRequest:request error:&error].outputs[NFKPluginFixtureVersionKey], @"2.0.0", @"%@", error);

	[self editManifestOfBundle:installed edit:^(NSMutableDictionary *edited) {
		edited[@"summary"] = @"Edited after signing.";
	}];
	XCTestExpectation *refused = [self expectationWithDescription:@"refused"];
	[registry reloadPluginWithIdentifier:manifest.identifier completionHandler:^(NSError *reloadError) {
		[self assertError:reloadError domain:NFKPluginErrorDomain code:NFKPluginErrorSignatureInvalid];
		[refused fulfill];
	}];
	[self waitForExpectations:@[ refused ] timeout:60];
}

#pragma mark Trust

- (void)testTrustedPluginsRunInProcessWithTheirOutOfProcessOutputs
{
	[self requireHost];
	for (NFKPluginEntryKind kind in @[ NFKPluginEntryKindObjC, NFKPluginEntryKindCABI ]) {
		NSURL *bundle = [self fixtureOfKind:kind version:@"1.0.0"];
		NSString *invert = [self catalogIdentifierOfKind:kind model:NFKPluginFixtureInvertModel];
		NSString *identifier = [NFKPluginFixtures pluginIdentifierForKind:kind];
		NSError *error = nil;
		NFKPluginRegistry *outside = [self registry];
		NFKPluginRegistry *inside = [self registry];
		XCTAssertNotNil([outside addPluginAtURL:bundle error:&error], @"%@", error);
		XCTAssertNotNil([inside addPluginAtURL:bundle error:&error], @"%@", error);
		XCTAssertFalse([inside isTrustedPluginIdentifier:identifier]);
		[inside setTrusted:YES forPluginIdentifier:identifier];
		XCTAssertTrue([inside isTrustedPluginIdentifier:identifier]);

		NFKPluginBackend *separate = [outside backendForModelIdentifier:invert options:nil error:&error];
		XCTAssertEqual(separate.isolation, NFKPluginIsolationProcess, @"%@", error);
		NFKPluginBackend *loaded = [inside backendForModelIdentifier:invert options:nil error:&error];
		XCTAssertNotNil(loaded, @"%@", error);
		XCTAssertEqual(loaded.isolation, NFKPluginIsolationInProcess);
		XCTAssertNil(loaded.process);
		XCTAssertEqualObjects(loaded.backendIdentifier, separate.backendIdentifier);
		XCTAssertEqualObjects(loaded.supportedOutputKeys, separate.supportedOutputKeys);

		NSDictionary<NSString *, id> *expected = [self invertOutputsOf:separate];
		NSDictionary<NSString *, id> *actual = [self invertOutputsOf:loaded];
		XCTAssertEqualObjects(actual[NFKPluginFixtureValuesKey], (@[ @0.75, @0.0 ]), @"%@", kind);
		XCTAssertEqualObjects(actual, expected, @"%@", kind);
		XCTAssertEqual([actual[NFKOutputImage] length], 64u * 64u * 4u, @"%@", kind);

		NFKToolResult *echoed = [inside.tools.firstObject invokeWithArguments:@{ @"text": @"hi" }];
		XCTAssertFalse(echoed.isError, @"%@", echoed.text);
		XCTAssertEqualObjects(echoed.text, [outside.tools.firstObject invokeWithArguments:@{ @"text": @"hi" }].text);

		[inside setTrusted:NO forPluginIdentifier:identifier];
		XCTAssertEqual([inside backendForModelIdentifier:invert options:nil error:&error].isolation, NFKPluginIsolationProcess,
					   @"withdrawn trust applies from the next load");
	}
}

- (void)testAnExecutablePluginIsRefusedInProcess
{
	NFKPluginRegistry *registry = [self registry];
	NSError *error = nil;
	NFKPluginManifest *manifest = [registry addPluginAtURL:[self fixtureOfKind:NFKPluginEntryKindExecutable version:@"1.0.0"] error:&error];
	XCTAssertNotNil(manifest, @"%@", error);
	NSString *invert = [self catalogIdentifierOfKind:NFKPluginEntryKindExecutable model:NFKPluginFixtureInvertModel];
	[registry setTrusted:YES forPluginIdentifier:manifest.identifier];
	XCTAssertNil([registry backendForModelIdentifier:invert options:nil error:&error]);
	[self assertError:error domain:NFKPluginErrorDomain code:NFKPluginErrorNotTrusted];
	XCTAssertNil([registry.catalog backendForModelIdentifier:invert options:nil error:&error]);
	[self assertError:error domain:NFKPluginErrorDomain code:NFKPluginErrorNotTrusted];

	[registry setTrusted:NO forPluginIdentifier:manifest.identifier];
	NFKPluginBackend *backend = [registry backendForModelIdentifier:invert options:nil error:&error];
	XCTAssertEqual(backend.isolation, NFKPluginIsolationProcess, @"%@", error);
	NFKInferenceResult *result = [backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKPluginFixtureValuesKey: @[ @0.25, @1.0 ] }]
														   error:&error];
	XCTAssertEqualObjects(result.outputs[NFKPluginFixtureValuesKey], (@[ @0.75, @0.0 ]), @"%@", error);
}

- (void)testATrustedPluginThatCannotRunInProcessRefusesItsToolsAsItsBackends
{
	NFKPluginRegistry *registry = [self registry];
	NSError *error = nil;
	NFKPluginManifest *manifest = [registry addPluginAtURL:[self fixtureOfKind:NFKPluginEntryKindExecutable version:@"1.0.0"] error:&error];
	XCTAssertNotNil(manifest, @"%@", error);
	[registry setTrusted:YES forPluginIdentifier:manifest.identifier];
	NFKInferenceTool *echo = nil;
	for (NFKInferenceTool *tool in registry.tools) {
		if ([tool.name isEqualToString:NFKPluginFixtureEchoTool]) {
			echo = tool;
		}
	}
	XCTAssertNotNil(echo);
	XCTAssertNil([echo resultForArguments:@{ @"text": @"hi" } error:&error], @"the tool runs nowhere, as the backends do");
	[self assertError:error domain:NFKPluginErrorDomain code:NFKPluginErrorNotTrusted];

	[registry setTrusted:NO forPluginIdentifier:manifest.identifier];
	for (NFKInferenceTool *tool in registry.tools) {
		if ([tool.name isEqualToString:NFKPluginFixtureEchoTool]) {
			echo = tool;
		}
	}
	NFKToolResult *result = [echo resultForArguments:@{ @"text": @"hi" } error:&error];
	XCTAssertNotNil(result, @"an untrusted plugin's tool runs in its process: %@", error);
	XCTAssertFalse(result.isError, @"%@", result.text);
}

- (void)testATamperedBundleAndATeamPinAreRefused
{
	NSURL *fixture = [self fixtureOfKind:NFKPluginEntryKindCABI version:@"1.0.0"];
	NSError *error = nil;
	NSURL *tampered = [NFKPluginFixtures tamperedCopyOfBundleAtURL:fixture error:&error];
	XCTAssertNotNil(tampered, @"%@", error);
	[self.temporaryURLs addObject:tampered.URLByDeletingLastPathComponent];
	NSString *invert = [self catalogIdentifierOfKind:NFKPluginEntryKindCABI model:NFKPluginFixtureInvertModel];
	NSString *identifier = [NFKPluginFixtures pluginIdentifierForKind:NFKPluginEntryKindCABI];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKPluginFixtureValuesKey: @[ @0.5 ] }];

	NFKPluginRegistry *registry = [self registry];
	XCTAssertNotNil([registry addPluginAtURL:tampered error:&error], @"listing hashes nothing: %@", error);
	NFKPluginBackend *backend = [registry backendForModelIdentifier:invert options:nil error:&error];
	XCTAssertNotNil(backend, @"%@", error);
	XCTAssertNil([backend runInferenceForRequest:request error:&error]);
	[self assertError:error domain:NFKPluginErrorDomain code:NFKPluginErrorSignatureInvalid];
	XCTAssertEqual(backend.process.state, NFKPluginProcessStateFailed);
	XCTAssertEqual(backend.process.processIdentifier, 0, @"nothing launched");
	[registry setTrusted:YES forPluginIdentifier:identifier];
	XCTAssertNil([registry backendForModelIdentifier:invert options:nil error:&error]);
	[self assertError:error domain:NFKPluginErrorDomain code:NFKPluginErrorSignatureInvalid];

	NFKPluginRegistry *pinned = [self registry];
	NFKPluginSignaturePolicy *policy = NFKPluginSignaturePolicy.developmentPolicy;
	policy.teamIdentifier = @"ABCDE12345";
	pinned.signaturePolicy = policy;
	XCTAssertNotNil([pinned addPluginAtURL:fixture error:&error], @"%@", error);
	XCTAssertNil([[pinned backendForModelIdentifier:invert options:nil error:&error] runInferenceForRequest:request error:&error]);
	[self assertError:error domain:NFKPluginErrorDomain code:NFKPluginErrorTeamMismatch];
	[pinned setTrusted:YES forPluginIdentifier:identifier];
	XCTAssertNil([pinned backendForModelIdentifier:invert options:nil error:&error]);
	[self assertError:error domain:NFKPluginErrorDomain code:NFKPluginErrorTeamMismatch];
}

#pragma mark Lazy hosting

- (void)testTheServersIdleUnloadEndsThePluginsProcess
{
	[self requireHost];
	NFKPluginRegistry *registry = [self registry];
	NSError *error = nil;
	XCTAssertNotNil([registry addPluginAtURL:[self fixtureOfKind:NFKPluginEntryKindCABI version:@"1.0.0"] error:&error], @"%@", error);
	NSString *invert = [self catalogIdentifierOfKind:NFKPluginEntryKindCABI model:NFKPluginFixtureInvertModel];

	NFKInferenceServer *server = [[NFKInferenceServer alloc] init];
	server.residencyCheckInterval = 0;
	server.unloadGracePeriod = 0;
	server.idleUnloadInterval = 0.2;
	NSMutableArray<NFKPluginProcess *> *processes = [NSMutableArray array];
	NSHashTable<NFKPluginBackend *> *built = [NSHashTable weakObjectsHashTable];
	[server addBackendProvider:^id<NFKInferenceBackend> _Nullable(NSError **providerError) {
		NFKPluginBackend *backend = [registry backendForModelIdentifier:invert options:nil error:providerError];
		backend.process.idleExitInterval = 0.5;
		@synchronized (processes) {
			if (backend != nil) {
				[processes addObject:backend.process];
				[built addObject:backend];
			}
		}
		return backend;
	} forModelName:@"invert"];
	XCTestExpectation *loaded = [self expectationWithDescription:@"loaded"];
	[server loadModelNamed:@"invert" completionHandler:^(NSError *loadError) {
		XCTAssertNil(loadError);
		[loaded fulfill];
	}];
	[self waitForExpectations:@[ loaded ] timeout:30];

	@autoreleasepool {
		NFKInferenceResult *result = [[server backendForModelName:@"invert"] runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKPluginFixtureValuesKey: @[ @0.25 ] }]
																							  error:&error];
		XCTAssertEqualObjects(result.outputs[NFKPluginFixtureValuesKey], (@[ @0.75 ]), @"%@", error);
	}
	NFKPluginProcess *process = nil;
	@synchronized (processes) {
		process = processes.firstObject;
	}
	XCTAssertEqual(process.state, NFKPluginProcessStateReady);
	pid_t pid = process.processIdentifier;
	XCTAssertGreaterThan(pid, 0);

	XCTAssertTrue([self waitUntil:^BOOL {
		return [[server unloadIdleModels] containsObject:@"invert"];
	} timeout:10], @"the server unloads the idle model");
	XCTAssertEqual([server residencyForModelName:@"invert"], NFKServerModelResidencyNotLoaded);
	XCTAssertTrue([self waitUntil:^BOOL {
		// The array allObjects returns holds the backends it lists, so it is drained before the next poll.
		NSUInteger alive = 0;
		@autoreleasepool {
			@synchronized (processes) {
				alive = built.allObjects.count;
			}
		}
		return alive == 0;
	} timeout:10], @"the unload releases the last reference to the backend, so its handle is released");
	XCTAssertTrue([self waitUntil:^BOOL {
		return process.state == NFKPluginProcessStateNotRunning;
	} timeout:30], @"the process exits once no backend holds it: %@", process.diagnosticDescription);
	XCTAssertTrue([self waitUntil:^BOOL {
		return kill(pid, 0) != 0;
	} timeout:30], @"process %d has exited", pid);
	[server removeBackendForModelName:@"invert"];
}

@end

#endif
