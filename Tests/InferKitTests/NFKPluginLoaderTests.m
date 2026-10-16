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

/*! A C plugin whose function table, result payload, and result buffers are each NFK_TEST_LONGER bytes
	longer than plugin ABI 1.0's, as a plugin built against a later minor version writes them. Model
	"invert" answers every run with two attachments, "abcd" under "first" and "wxyz" under "second". */
static NSString * const NFKLoaderTestsLongerStructuresSource =
	@"#include <InferKitC/NFKPluginABI.h>\n"
	@"#include <stdlib.h>\n"
	@"#include <string.h>\n"
	@"#define STRIDE (sizeof(nfk_plugin_buffer) + NFK_TEST_LONGER)\n"
	@"typedef struct { nfk_plugin_payload payload; unsigned char later[NFK_TEST_LONGER + 1]; "
	@"unsigned char buffers[2 * STRIDE] __attribute__((aligned(8))); char first[4]; char second[4]; } result_t;\n"
	@"static int backend_token;\n"
	@"static char *copy_string(const char *text) { char *copy = malloc(strlen(text) + 1); strcpy(copy, text); return copy; }\n"
	@"static nfk_plugin_status describe(void *c, char **json) { *json = copy_string(\"{\\\"models\\\":[{\\\"id\\\":\\\"invert\\\"}]}\"); return NFK_PLUGIN_STATUS_OK; }\n"
	@"static nfk_plugin_status create_backend(void *c, const char *m, const char *o, nfk_plugin_backend **b, char **e)\n"
	@"{ *b = (nfk_plugin_backend *)&backend_token; *e = 0; return NFK_PLUGIN_STATUS_OK; }\n"
	@"static void put_buffer(result_t *r, int index, const char *bytes)\n"
	@"{ nfk_plugin_buffer buffer; memset(&buffer, 0, sizeof buffer); buffer.struct_size = (uint32_t)STRIDE; buffer.kind = NFK_PLUGIN_BUFFER_BYTES;\n"
	@"  buffer.bytes = bytes; buffer.length = 4; buffer.descriptor = -1; memcpy(r->buffers + index * STRIDE, &buffer, sizeof buffer); }\n"
	@"static nfk_plugin_status run(void *c, nfk_plugin_backend *b, nfk_plugin_run *run, const nfk_plugin_payload *request,\n"
	@"  nfk_plugin_payload **out, char **e)\n"
	@"{ result_t *r = calloc(1, sizeof *r); memcpy(r->first, \"abcd\", 4); memcpy(r->second, \"wxyz\", 4);\n"
	@"  put_buffer(r, 0, r->first); put_buffer(r, 1, r->second);\n"
	@"  r->payload.struct_size = (uint32_t)(sizeof(nfk_plugin_payload) + NFK_TEST_LONGER); r->payload.buffer_count = 2;\n"
	@"  r->payload.json = \"{\\\"outputs\\\":{\\\"first\\\":{\\\"$nfk\\\":\\\"buffer\\\",\\\"index\\\":0,\\\"as\\\":\\\"data\\\",\\\"length\\\":4},\"\n"
	@"    \"\\\"second\\\":{\\\"$nfk\\\":\\\"buffer\\\",\\\"index\\\":1,\\\"as\\\":\\\"data\\\",\\\"length\\\":4}}}\";\n"
	@"  r->payload.buffers = (const nfk_plugin_buffer *)r->buffers; *out = &r->payload; *e = 0; return NFK_PLUGIN_STATUS_OK; }\n"
	@"static void destroy_backend(void *c, nfk_plugin_backend *b) { }\n"
	@"static void free_payload(void *c, nfk_plugin_payload *p) { free(p); }\n"
	@"static void free_string(void *c, char *s) { free(s); }\n"
	@"static void shutdown_plugin(void *c) { }\n"
	@"static struct { nfk_plugin table; unsigned char later[NFK_TEST_LONGER + 1]; } longer_table = {\n"
	@"  { (uint32_t)(sizeof(nfk_plugin) + NFK_TEST_LONGER), NFK_PLUGIN_ABI_VERSION, 0, describe, create_backend, 0, run, 0, 0,\n"
	@"    destroy_backend, 0, free_payload, free_string, shutdown_plugin } };\n"
	@"NFK_PLUGIN_EXPORT nfk_plugin_status nfk_plugin_entry(uint32_t v, const nfk_plugin_host *h, const nfk_plugin **out)\n"
	@"{ *out = &longer_table.table; return NFK_PLUGIN_STATUS_OK; }\n";

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

/*! A copy of bundle under identifier in a directory of its own, edited by edit and not signed again. */
- (nullable NSURL *)unsignedCopyOfBundle:(NSURL *)bundle identifier:(NSString *)identifier edit:(nullable void (^)(NSURL *copy))edit
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
	XCTAssertTrue([[NSJSONSerialization dataWithJSONObject:manifest options:0 error:NULL] writeToURL:manifestURL atomically:YES]);
	if (edit != nil) {
		edit(copy);
	}
	return copy;
}

/*! A library compiled from source with the plugin headers and the given -D definitions, or nil. */
- (nullable NSURL *)libraryFromSource:(NSString *)source named:(NSString *)name definitions:(NSArray<NSString *> *)definitions
{
	NSURL *directory = [[NSURL fileURLWithPath:NSTemporaryDirectory()] URLByAppendingPathComponent:
						[NSString stringWithFormat:@"loader-test-library-%@", NSUUID.UUID.UUIDString] isDirectory:YES];
	[self.temporaryURLs addObject:directory];
	XCTAssertTrue([NSFileManager.defaultManager createDirectoryAtURL:directory withIntermediateDirectories:YES attributes:nil error:NULL]);
	NSURL *sourceURL = [directory URLByAppendingPathComponent:[name stringByAppendingPathExtension:@"c"]];
	NSURL *library = [directory URLByAppendingPathComponent:[name stringByAppendingPathExtension:@"dylib"]];
	XCTAssertTrue([source writeToURL:sourceURL atomically:YES encoding:NSUTF8StringEncoding error:NULL]);
	NSString *include = [NFKPluginFixtures.sourceRootURL URLByAppendingPathComponent:@"Sources/InferKitC/include"].path;
	NSMutableArray<NSString *> *compile = [NSMutableArray arrayWithObjects:@"clang", @"-dynamiclib", @"-I", include, nil];
	[compile addObjectsFromArray:definitions];
	[compile addObjectsFromArray:@[ @"-o", library.path, sourceURL.path ]];
	NSError *error = nil;
	if ([NFKPluginFixtures runProgram:@"/usr/bin/xcrun" arguments:compile error:&error] == nil) {
		XCTFail(@"%@", error);
		return nil;
	}
	return library;
}

/*! Whether the image at url is loaded in this process. */
- (BOOL)isLoaded:(NSURL *)url
{
	void *handle = dlopen(url.fileSystemRepresentation, RTLD_NOLOAD | RTLD_LOCAL);
	if (handle != NULL) {
		dlclose(handle);
	}
	return handle != NULL;
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

#pragma mark Containment

- (void)testALibraryOutsideTheBundleIsNeverLoaded
{
	NSURL *fixture = [self fixtureOfKind:NFKPluginEntryKindCABI version:@"1.0.0"];
	NSURL *outside = [self libraryFromSource:NFKLoaderTestsSecondMajorSource named:@"outside" definitions:@[]];
	XCTAssertNotNil(outside);
	NFKPluginSignaturePolicy *unsignedPolicy = NFKPluginSignaturePolicy.developmentPolicy;
	unsignedPolicy.requiresValidSignature = NO;

	NSURL *escaping = [self unsignedCopyOfBundle:fixture identifier:@"com.example.inferkit.loader-test-escape" edit:^(NSURL *copy) {
		NSURL *infoURL = [copy URLByAppendingPathComponent:@"Contents/Info.plist"];
		NSMutableDictionary *info = [NSMutableDictionary dictionaryWithContentsOfURL:infoURL];
		info[@"CFBundleExecutable"] = @"../../x.dylib";
		XCTAssertTrue([info writeToURL:infoURL error:NULL]);
		NSURL *manifestURL = [copy URLByAppendingPathComponent:@"Contents/Resources/inferkit-plugin.json"];
		NSMutableDictionary *manifest = [NSJSONSerialization JSONObjectWithData:[NSData dataWithContentsOfURL:manifestURL]
																		options:NSJSONReadingMutableContainers error:NULL];
		[manifest[@"entry"] removeObjectForKey:@"executable"];
		XCTAssertTrue([[NSJSONSerialization dataWithJSONObject:manifest options:0 error:NULL] writeToURL:manifestURL atomically:YES]);
		XCTAssertTrue([NSFileManager.defaultManager copyItemAtURL:outside toURL:[copy URLByAppendingPathComponent:@"x.dylib"] error:NULL]);
	}];
	NSError *error = nil;
	XCTAssertNil([NFKPluginLoader loadPluginAtURL:escaping signaturePolicy:unsignedPolicy error:&error]);
	[self assertError:error domain:NFKPluginErrorDomain code:NFKPluginErrorManifestMismatch];
	XCTAssertFalse([self isLoaded:[escaping URLByAppendingPathComponent:@"x.dylib"]], @"the refusal comes before dlopen");

	NSURL *linked = [self unsignedCopyOfBundle:fixture identifier:@"com.example.inferkit.loader-test-link" edit:^(NSURL *copy) {
		NSURL *executable = [NFKPluginFixtures executableURLOfBundleAtURL:copy];
		XCTAssertTrue([NSFileManager.defaultManager removeItemAtURL:executable error:NULL]);
		XCTAssertTrue([NSFileManager.defaultManager createSymbolicLinkAtURL:executable withDestinationURL:outside error:NULL]);
	}];
	XCTAssertNil([NFKPluginLoader loadPluginAtURL:linked signaturePolicy:unsignedPolicy error:&error]);
	[self assertError:error domain:NFKPluginErrorDomain code:NFKPluginErrorManifestMismatch];
	XCTAssertTrue([error.localizedDescription containsString:@"outside"], @"%@", error);
	XCTAssertFalse([self isLoaded:outside], @"a link out of the bundle is never followed into dlopen");
}

- (void)testALoadThatFailsBeforeItsCodeLoadsIsTriedAgain
{
	NSURL *fixture = [self fixtureOfKind:NFKPluginEntryKindCABI version:@"1.0.0"];
	NSURL *library = [NFKPluginFixtures executableURLOfBundleAtURL:fixture];
	NFKPluginSignaturePolicy *unsignedPolicy = NFKPluginSignaturePolicy.developmentPolicy;
	unsignedPolicy.requiresValidSignature = NO;
	__block NSURL *executable = nil;
	NSURL *copy = [self unsignedCopyOfBundle:fixture identifier:@"com.example.inferkit.loader-test-retry" edit:^(NSURL *bundle) {
		executable = [NFKPluginFixtures executableURLOfBundleAtURL:bundle];
		XCTAssertTrue([NSFileManager.defaultManager removeItemAtURL:executable error:NULL]);
		XCTAssertTrue([[@"not a library" dataUsingEncoding:NSUTF8StringEncoding] writeToURL:executable atomically:YES]);
	}];
	NSError *error = nil;
	XCTAssertNil([NFKPluginLoader loadPluginAtURL:copy signaturePolicy:unsignedPolicy error:&error]);
	[self assertError:error domain:NFKPluginErrorDomain code:NFKPluginErrorLoadFailed];
	XCTAssertTrue([error.localizedDescription containsString:@"did not load"], @"%@", error);

	XCTAssertTrue([NSFileManager.defaultManager removeItemAtURL:executable error:NULL]);
	XCTAssertTrue([NSFileManager.defaultManager copyItemAtURL:library toURL:executable error:&error], @"%@", error);
	NFKLoadedPlugin *plugin = [NFKPluginLoader loadPluginAtURL:copy signaturePolicy:unsignedPolicy error:&error];
	XCTAssertNotNil(plugin, @"a library dlopen refused left nothing behind, so the fixed bundle loads: %@", error);
}

#pragma mark Structure sizes

- (void)testStructuresAtTheOneZeroSizeAndLongerOnesAreRead
{
	NSURL *fixture = [self fixtureOfKind:NFKPluginEntryKindCABI version:@"1.0.0"];
	NSDictionary<NSString *, NSNumber *> *longerBy = @{ @"com.example.inferkit.loader-test-size-1-0": @0,
													  @"com.example.inferkit.loader-test-size-longer": @16 };
	for (NSString *identifier in longerBy) {
		NSString *definition = [NSString stringWithFormat:@"-DNFK_TEST_LONGER=%@", longerBy[identifier]];
		NSURL *library = [self libraryFromSource:NFKLoaderTestsLongerStructuresSource named:@"longer" definitions:@[ definition ]];
		NSURL *copy = library != nil ? [self copyOfBundle:fixture identifier:identifier executable:library editManifest:nil] : nil;
		XCTAssertNotNil(copy);
		if (copy == nil) {
			continue;
		}
		NSError *error = nil;
		NFKLoadedPlugin *plugin = [NFKPluginLoader loadPluginAtURL:copy signaturePolicy:NFKPluginSignaturePolicy.developmentPolicy error:&error];
		XCTAssertNotNil(plugin, @"a function table %@ bytes longer than 1.0's loads: %@", longerBy[identifier], error);
		id<NFKInferenceBackend> backend = [plugin backendForModelIdentifier:@"invert" options:@{} error:&error];
		XCTAssertNotNil(backend, @"%@", error);
		NFKInferenceResult *result = [backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{}] error:&error];
		XCTAssertNotNil(result, @"a payload and buffers %@ bytes longer than 1.0's are read: %@", longerBy[identifier], error);
		XCTAssertEqualObjects(result.outputs[@"first"], [@"abcd" dataUsingEncoding:NSUTF8StringEncoding]);
		XCTAssertEqualObjects(result.outputs[@"second"], [@"wxyz" dataUsingEncoding:NSUTF8StringEncoding],
							  @"the second buffer is found at its stride of %@ bytes past 1.0's", longerBy[identifier]);
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
