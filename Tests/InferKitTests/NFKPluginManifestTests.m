//
//  NFKPluginManifestTests.m
//  InferKitTests
//
//  The plugin manifest: reading every field, the defaults, the JSON round trip, the refusals of a
//  malformed or newer manifest, the checks against a bundle's Info.plist, and the compatibility checks.
//

#import <XCTest/XCTest.h>
#import <InferKit/InferKit.h>
#import "../../Sources/InferKit/NFKPluginManifestPrivate.h"

@interface NFKPluginManifestTests : XCTestCase
@property (nonatomic, strong) NSMutableArray<NSURL *> *temporaryURLs;
@end

@implementation NFKPluginManifestTests

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

/*! A complete manifest object, every field set. */
- (NSMutableDictionary<NSString *, id> *)fullObject
{
	return [@{
		@"schemaVersion": @1,
		@"kind": @"inferkit.plugin",
		@"id": @"com.example.denoise",
		@"version": @"1.2.0",
		@"displayName": @"Denoise",
		@"summary": @"Removes noise from audio.",
		@"compatibility": @{ @"inferkitAPI": @"1.0", @"pluginABI": @"1.0", @"protocol": @"1.0",
							 @"minimumOS": @{ @"macOS": @"11.0" }, @"architectures": @[ @"arm64", @"x86_64" ] },
		@"entry": @{ @"kind": @"c-abi", @"executable": @"libdenoise.dylib", @"symbol": @"denoise_entry" },
		@"process": @{ @"idleExitSeconds": @120, @"allowsInProcess": @YES, @"maximumConcurrentRuns": @2 },
		@"options": @[ @{ @"key": @"threads", @"kind": @"integer", @"default": @4 } ],
		@"models": @[ @{ @"id": @"denoise", @"family": @"denoise", @"engine": @"plugin",
						 @"inputs": @[ @{ @"key": @"audio", @"valueType": @"audio", @"required": @YES } ] } ],
		@"tools": @[ @{ @"name": @"measure_noise", @"description": @"Measures the noise floor.",
						@"parameters": @{ @"type": @"object", @"properties": @{} } } ],
		@"extensions": @{ @"com.example.color": @"teal" },
	} mutableCopy];
}

/*! The smallest manifest the reader accepts. */
- (NSMutableDictionary<NSString *, id> *)minimalObject
{
	return [@{ @"schemaVersion": @1, @"kind": @"inferkit.plugin", @"id": @"com.example.minimal", @"version": @"0.1.0",
			   @"compatibility": @{ @"inferkitAPI": @"1.0" }, @"entry": @{ @"kind": @"objc" } } mutableCopy];
}

- (nullable NFKPluginManifest *)manifestWithObject:(NSDictionary *)object error:(NSError **)error
{
	NSData *data = [NSJSONSerialization dataWithJSONObject:object options:0 error:NULL];
	return [NFKPluginManifest manifestWithJSONData:data error:error];
}

- (void)assertObject:(NSDictionary *)object failsWithCode:(NFKPluginError)code because:(NSString *)reason
{
	NSError *error = nil;
	XCTAssertNil([self manifestWithObject:object error:&error], @"%@", reason);
	XCTAssertEqualObjects(error.domain, NFKPluginErrorDomain, @"%@", reason);
	XCTAssertEqual(error.code, code, @"%@: %@", reason, error);
}

- (NSMutableDictionary *)object:(NSDictionary *)object setting:(id)value atPath:(NSString *)path
{
	NSMutableDictionary *copy = [NSJSONSerialization JSONObjectWithData:[NSJSONSerialization dataWithJSONObject:object options:0 error:NULL]
																options:NSJSONReadingMutableContainers error:NULL];
	NSArray<NSString *> *components = [path componentsSeparatedByString:@"."];
	NSMutableDictionary *container = copy;
	for (NSString *component in [components subarrayWithRange:NSMakeRange(0, components.count - 1)]) {
		container = container[component];
	}
	container[components.lastObject] = value;
	return copy;
}

/*! A bundle directory holding info, the manifest object, and an executable file named executable. */
- (NSURL *)bundleWithInfo:(NSDictionary *)info manifest:(NSDictionary *)manifest executable:(nullable NSString *)executable
{
	NSURL *bundle = [[NSURL fileURLWithPath:NSTemporaryDirectory()] URLByAppendingPathComponent:
					 [NSString stringWithFormat:@"manifest-test-%@.inferkitplugin", NSUUID.UUID.UUIDString] isDirectory:YES];
	[self.temporaryURLs addObject:bundle];
	NSURL *contents = [bundle URLByAppendingPathComponent:@"Contents" isDirectory:YES];
	NSURL *resources = [contents URLByAppendingPathComponent:@"Resources" isDirectory:YES];
	NSURL *macOS = [contents URLByAppendingPathComponent:@"MacOS" isDirectory:YES];
	XCTAssertTrue([NSFileManager.defaultManager createDirectoryAtURL:resources withIntermediateDirectories:YES attributes:nil error:NULL]);
	XCTAssertTrue([NSFileManager.defaultManager createDirectoryAtURL:macOS withIntermediateDirectories:YES attributes:nil error:NULL]);
	XCTAssertTrue([info writeToURL:[contents URLByAppendingPathComponent:@"Info.plist"] error:NULL]);
	NSData *JSON = [NSJSONSerialization dataWithJSONObject:manifest options:0 error:NULL];
	XCTAssertTrue([JSON writeToURL:[resources URLByAppendingPathComponent:@"inferkit-plugin.json"] atomically:YES]);
	if (executable != nil) {
		XCTAssertTrue([[NSData data] writeToURL:[macOS URLByAppendingPathComponent:executable] atomically:YES]);
	}
	return bundle;
}

- (NSDictionary *)infoForObject:(NSDictionary *)object
{
	return @{ @"CFBundleIdentifier": object[@"id"], @"CFBundleShortVersionString": object[@"version"],
			  @"CFBundleExecutable": @"Minimal", @"CFBundlePackageType": @"BNDL" };
}

#pragma mark Reading

- (void)testReadsEveryFieldOfAManifest
{
	NSError *error = nil;
	NFKPluginManifest *manifest = [self manifestWithObject:[self fullObject] error:&error];
	XCTAssertNotNil(manifest, @"%@", error);
	XCTAssertEqualObjects(manifest.identifier, @"com.example.denoise");
	XCTAssertEqualObjects(manifest.version, @"1.2.0");
	XCTAssertEqualObjects(manifest.displayName, @"Denoise");
	XCTAssertEqualObjects(manifest.summary, @"Removes noise from audio.");
	XCTAssertEqualObjects(manifest.entryKind, NFKPluginEntryKindCABI);
	XCTAssertEqualObjects(manifest.executablePath, @"libdenoise.dylib");
	XCTAssertEqualObjects(manifest.entrySymbol, @"denoise_entry");
	XCTAssertNil(manifest.principalClassName);
	XCTAssertEqualObjects(manifest.minimumAPIVersion, @"1.0");
	XCTAssertEqualObjects(manifest.pluginABIVersion, @"1.0");
	XCTAssertEqualObjects(manifest.protocolVersion, @"1.0");
	XCTAssertEqualObjects(manifest.minimumOSVersions, @{ @"macOS": @"11.0" });
	XCTAssertEqualObjects(manifest.architectures, (@[ @"arm64", @"x86_64" ]));
	XCTAssertEqual(manifest.idleExitInterval, 120);
	XCTAssertTrue(manifest.allowsInProcess);
	XCTAssertEqual(manifest.maximumConcurrentRuns, 2);
	XCTAssertEqual(manifest.options.count, 1u);
	XCTAssertEqualObjects(manifest.options.firstObject.key, @"threads");
	XCTAssertEqual(manifest.models.count, 1u);
	XCTAssertEqualObjects(manifest.models.firstObject.identifier, @"denoise");
	XCTAssertEqualObjects(manifest.models.firstObject.engine, NFKModelEnginePlugin);
	XCTAssertEqualObjects(manifest.models.firstObject.inputs.firstObject.key, @"audio");
	XCTAssertEqualObjects(manifest.tools.firstObject[@"name"], @"measure_noise");
	XCTAssertEqualObjects(manifest.extensions, @{ @"com.example.color": @"teal" });
	XCTAssertNil(manifest.bundleURL);
	XCTAssertEqualObjects([manifest modelWithIdentifier:@"denoise"].identifier, @"denoise");
	XCTAssertNil([manifest modelWithIdentifier:@"absent"]);
}

- (void)testFillsDefaultsForOptionalFields
{
	NSMutableDictionary *object = [self minimalObject];
	object[@"models"] = @[ @{ @"id": @"echo", @"family": @"echo" } ];
	object[@"tools"] = @[ @{ @"name": @"ping" } ];
	NSError *error = nil;
	NFKPluginManifest *manifest = [self manifestWithObject:object error:&error];
	XCTAssertNotNil(manifest, @"%@", error);
	XCTAssertEqualObjects(manifest.displayName, @"com.example.minimal", @"the display name falls back to the identifier");
	XCTAssertNil(manifest.summary);
	XCTAssertEqualObjects(manifest.protocolVersion, @"1.0");
	XCTAssertNil(manifest.pluginABIVersion);
	XCTAssertEqual(manifest.idleExitInterval, 300);
	XCTAssertFalse(manifest.allowsInProcess);
	XCTAssertEqual(manifest.maximumConcurrentRuns, 1);
	XCTAssertEqualObjects(manifest.arguments, @[]);
	XCTAssertEqualObjects(manifest.architectures, @[]);
	XCTAssertEqualObjects(manifest.minimumOSVersions, @{});
	XCTAssertEqualObjects(manifest.extensions, @{});
	XCTAssertEqualObjects(manifest.models.firstObject.engine, NFKModelEnginePlugin, @"a model without an engine runs on the plugin engine");
	NSDictionary *expectedTool = @{ @"name": @"ping", @"description": @"", @"parameters": @{ @"type": @"object", @"properties": @{} } };
	XCTAssertEqualObjects(manifest.tools.firstObject, expectedTool);
}

- (void)testIgnoresUnknownFields
{
	NSMutableDictionary *object = [self minimalObject];
	object[@"futureField"] = @{ @"anything": @[ @1, @2 ] };
	NSError *error = nil;
	XCTAssertNotNil([self manifestWithObject:object error:&error], @"%@", error);
}

#pragma mark Writing

- (void)testJSONRoundTripKeepsTheManifest
{
	NSError *error = nil;
	NFKPluginManifest *manifest = [self manifestWithObject:[self fullObject] error:&error];
	NSData *JSON = [manifest JSONDataWithError:&error];
	XCTAssertNotNil(JSON, @"%@", error);
	NFKPluginManifest *again = [NFKPluginManifest manifestWithJSONData:JSON error:&error];
	XCTAssertNotNil(again, @"%@", error);
	XCTAssertEqualObjects(again, manifest);
	XCTAssertEqual(again.hash, manifest.hash);
	NSString *text = [[NSString alloc] initWithData:JSON encoding:NSUTF8StringEncoding];
	NSRange compatibility = [text rangeOfString:@"\"compatibility\""];
	NSRange entry = [text rangeOfString:@"\"entry\""];
	NSRange version = [text rangeOfString:@"\"version\""];
	XCTAssertTrue(compatibility.location < entry.location && entry.location < version.location, @"keys are sorted: %@", text);
	XCTAssertFalse([text containsString:@"\\/"], @"slashes are written as they are");
}

- (void)testMinimalManifestRoundTripsWithItsDefaults
{
	NSError *error = nil;
	NFKPluginManifest *manifest = [self manifestWithObject:[self minimalObject] error:&error];
	NSData *JSON = [manifest JSONDataWithError:&error];
	NSDictionary *written = [NSJSONSerialization JSONObjectWithData:JSON options:0 error:NULL];
	XCTAssertEqualObjects(written[@"process"], (@{ @"idleExitSeconds": @300, @"allowsInProcess": @NO, @"maximumConcurrentRuns": @1 }));
	XCTAssertNil(written[@"models"], @"an empty list is left out");
	XCTAssertEqualObjects([NFKPluginManifest manifestWithJSONData:JSON error:NULL], manifest);
}

- (void)testACopyIsTheSameManifest
{
	NFKPluginManifest *manifest = [self manifestWithObject:[self fullObject] error:NULL];
	XCTAssertEqualObjects([manifest copy], manifest);
	XCTAssertNotEqualObjects([self manifestWithObject:[self minimalObject] error:NULL], manifest);
}

#pragma mark Refusals

- (void)testRefusesANewerSchemaVersion
{
	NSMutableDictionary *object = [self minimalObject];
	object[@"schemaVersion"] = @2;
	[self assertObject:object failsWithCode:NFKPluginErrorUnsupportedManifestVersion because:@"schema version 2"];
}

- (void)testRefusesMalformedManifests
{
	NSError *error = nil;
	XCTAssertNil([NFKPluginManifest manifestWithJSONData:[@"not json" dataUsingEncoding:NSUTF8StringEncoding] error:&error]);
	XCTAssertEqual(error.code, NFKPluginErrorMalformedManifest);
	XCTAssertNil([NFKPluginManifest manifestWithJSONData:[@"[1, 2]" dataUsingEncoding:NSUTF8StringEncoding] error:&error]);
	XCTAssertEqual(error.code, NFKPluginErrorMalformedManifest);

	NSDictionary *minimal = [self minimalObject];
	NSDictionary *full = [self fullObject];
	NSMutableDictionary *withoutIdentifier = [minimal mutableCopy];
	[withoutIdentifier removeObjectForKey:@"id"];
	NSMutableDictionary *withoutCompatibility = [minimal mutableCopy];
	[withoutCompatibility removeObjectForKey:@"compatibility"];
	NSDictionary *cases = @{
		@"no schema version": [self object:minimal setting:@"1" atPath:@"schemaVersion"],
		@"another kind": [self object:minimal setting:@"inferkit.model" atPath:@"kind"],
		@"no identifier": withoutIdentifier,
		@"an identifier with a slash": [self object:minimal setting:@"com.example/denoise" atPath:@"id"],
		@"a version with a space": [self object:minimal setting:@"1.0 beta" atPath:@"version"],
		@"no compatibility": withoutCompatibility,
		@"an API version that is not major.minor": [self object:minimal setting:@"one" atPath:@"compatibility.inferkitAPI"],
		@"an executable outside Contents/MacOS": [self object:minimal setting:@"../escape" atPath:@"entry.executable"],
		@"an absolute executable": [self object:minimal setting:@"/bin/sh" atPath:@"entry.executable"],
		@"a c-abi plugin without a plugin ABI": [self object:minimal setting:@"c-abi" atPath:@"entry.kind"],
		@"a negative idle exit": [self object:full setting:@-1 atPath:@"process.idleExitSeconds"],
		@"zero concurrent runs": [self object:full setting:@0 atPath:@"process.maximumConcurrentRuns"],
		@"a string for allowsInProcess": [self object:full setting:@"yes" atPath:@"process.allowsInProcess"],
		@"a model on another engine": [self object:full setting:@[ @{ @"id": @"x", @"family": @"x", @"engine": @"mlx" } ] atPath:@"models"],
		@"a model identifier with a slash": [self object:full setting:@[ @{ @"id": @"a/b", @"family": @"x" } ] atPath:@"models"],
		@"a model listed twice": [self object:full setting:@[ @{ @"id": @"x", @"family": @"x" }, @{ @"id": @"x", @"family": @"y" } ] atPath:@"models"],
		@"a model without a family": [self object:full setting:@[ @{ @"id": @"x" } ] atPath:@"models"],
		@"a tool name with a space": [self object:full setting:@[ @{ @"name": @"measure noise" } ] atPath:@"tools"],
		@"a tool listed twice": [self object:full setting:@[ @{ @"name": @"t" }, @{ @"name": @"t" } ] atPath:@"tools"],
		@"an option declared twice": [self object:full setting:@[ @{ @"key": @"k" }, @{ @"key": @"k" } ] atPath:@"options"],
		@"models that are not an array": [self object:full setting:@{} atPath:@"models"],
	};
	for (NSString *reason in cases) {
		[self assertObject:cases[reason] failsWithCode:NFKPluginErrorMalformedManifest because:reason];
	}
}

- (void)testRefusesAnUnknownEntryKind
{
	[self assertObject:[self object:[self minimalObject] setting:@"wasm" atPath:@"entry.kind"]
		 failsWithCode:NFKPluginErrorIncompatible because:@"an entry kind this library does not run"];
}

#pragma mark Bundles

- (void)testReadsAManifestFromABundleWithoutLoadingItsCode
{
	NSMutableDictionary *object = [self minimalObject];
	NSURL *bundle = [self bundleWithInfo:[self infoForObject:object] manifest:object executable:@"Minimal"];
	NSError *error = nil;
	NFKPluginManifest *manifest = [NFKPluginManifest manifestWithBundleURL:bundle error:&error];
	XCTAssertNotNil(manifest, @"%@", error);
	XCTAssertEqualObjects(manifest.identifier, @"com.example.minimal");
	XCTAssertEqualObjects(manifest.bundleURL, bundle);
	XCTAssertEqualObjects(manifest.bundleExecutableName, @"Minimal");
	XCTAssertEqualObjects(manifest.executableURL.lastPathComponent, @"Minimal");
	XCTAssertEqualObjects(manifest.executableURL.URLByDeletingLastPathComponent.lastPathComponent, @"MacOS");
	XCTAssertFalse([NSBundle bundleWithURL:bundle].isLoaded, @"reading a manifest loads no code");

	NFKPluginManifest *again = [NFKPluginManifest manifestWithBundleURL:bundle error:NULL];
	XCTAssertEqualObjects(again, manifest);
	XCTAssertNotEqualObjects([NFKPluginManifest manifestWithJSONData:[manifest JSONDataWithError:NULL] error:NULL], manifest,
							 @"a manifest read from JSON alone names no bundle");
}

- (void)testAManifestNamesItsOwnExecutable
{
	NSMutableDictionary *object = [self object:[self minimalObject] setting:@"Helpers/serve" atPath:@"entry.executable"];
	NSURL *bundle = [self bundleWithInfo:[self infoForObject:object] manifest:object executable:nil];
	NSURL *helpers = [bundle URLByAppendingPathComponent:@"Contents/MacOS/Helpers" isDirectory:YES];
	XCTAssertTrue([NSFileManager.defaultManager createDirectoryAtURL:helpers withIntermediateDirectories:YES attributes:nil error:NULL]);
	XCTAssertTrue([[NSData data] writeToURL:[helpers URLByAppendingPathComponent:@"serve"] atomically:YES]);
	NSError *error = nil;
	NFKPluginManifest *manifest = [NFKPluginManifest manifestWithBundleURL:bundle error:&error];
	XCTAssertNotNil(manifest, @"%@", error);
	XCTAssertTrue([manifest.executableURL.path hasSuffix:@"Contents/MacOS/Helpers/serve"], @"%@", manifest.executableURL);
}

- (void)testRefusesAnExecutableThatResolvesOutsideContentsMacOS
{
	NSMutableDictionary *object = [self minimalObject];
	NSMutableDictionary *escaping = [[self infoForObject:object] mutableCopy];
	escaping[@"CFBundleExecutable"] = @"../../x.dylib";
	NSURL *bundle = [self bundleWithInfo:escaping manifest:object executable:nil];
	XCTAssertTrue([[NSData data] writeToURL:[bundle URLByAppendingPathComponent:@"x.dylib"] atomically:YES], @"the file it names exists");
	NSError *error = nil;
	XCTAssertNil([NFKPluginManifest manifestWithBundleURL:bundle error:&error], @"CFBundleExecutable is checked as entry.executable is");
	XCTAssertEqual(error.code, NFKPluginErrorManifestMismatch, @"%@", error);
	XCTAssertTrue([error.localizedDescription containsString:@"../../x.dylib"], @"%@", error);

	NSURL *outside = [[NSURL fileURLWithPath:NSTemporaryDirectory()] URLByAppendingPathComponent:
					  [NSString stringWithFormat:@"manifest-test-outside-%@.dylib", NSUUID.UUID.UUIDString]];
	[self.temporaryURLs addObject:outside];
	XCTAssertTrue([[NSData data] writeToURL:outside atomically:YES]);
	NSURL *linked = [self bundleWithInfo:[self infoForObject:object] manifest:object executable:nil];
	NSURL *link = [linked URLByAppendingPathComponent:@"Contents/MacOS/Minimal"];
	XCTAssertTrue([NSFileManager.defaultManager createSymbolicLinkAtURL:link withDestinationURL:outside error:&error], @"%@", error);
	XCTAssertNil([NFKPluginManifest manifestWithBundleURL:linked error:&error], @"a link out of the bundle is refused");
	XCTAssertEqual(error.code, NFKPluginErrorManifestMismatch, @"%@", error);
	XCTAssertTrue([error.localizedDescription containsString:@"outside"], @"%@", error);

	NSMutableDictionary *named = [self object:object setting:@"Minimal" atPath:@"entry.executable"];
	NSURL *namedLink = [self bundleWithInfo:[self infoForObject:named] manifest:named executable:nil];
	XCTAssertTrue([NSFileManager.defaultManager createSymbolicLinkAtURL:[namedLink URLByAppendingPathComponent:@"Contents/MacOS/Minimal"]
													 withDestinationURL:outside error:&error], @"%@", error);
	XCTAssertNil([NFKPluginManifest manifestWithBundleURL:namedLink error:&error], @"entry.executable resolves the same way");
	XCTAssertEqual(error.code, NFKPluginErrorManifestMismatch, @"%@", error);

	NSURL *inside = [self bundleWithInfo:[self infoForObject:object] manifest:object executable:@"Real"];
	XCTAssertTrue([NSFileManager.defaultManager createSymbolicLinkAtPath:[inside URLByAppendingPathComponent:@"Contents/MacOS/Minimal"].path
												 withDestinationPath:@"Real" error:&error], @"%@", error);
	NFKPluginManifest *manifest = [NFKPluginManifest manifestWithBundleURL:inside error:&error];
	XCTAssertNotNil(manifest, @"a link that stays in Contents/MacOS is accepted: %@", error);
	XCTAssertEqualObjects(manifest.executableURL.lastPathComponent, @"Real", @"the executable is the file the link resolves to");
}

- (void)testRefusesABundleWhoseInfoDisagrees
{
	NSMutableDictionary *object = [self minimalObject];
	NSMutableDictionary *otherIdentifier = [[self infoForObject:object] mutableCopy];
	otherIdentifier[@"CFBundleIdentifier"] = @"com.example.other";
	NSMutableDictionary *otherVersion = [[self infoForObject:object] mutableCopy];
	otherVersion[@"CFBundleShortVersionString"] = @"9.9.9";
	for (NSDictionary *info in @[ otherIdentifier, otherVersion ]) {
		NSURL *bundle = [self bundleWithInfo:info manifest:object executable:@"Minimal"];
		NSError *error = nil;
		XCTAssertNil([NFKPluginManifest manifestWithBundleURL:bundle error:&error]);
		XCTAssertEqualObjects(error.domain, NFKPluginErrorDomain);
		XCTAssertEqual(error.code, NFKPluginErrorManifestMismatch, @"%@", error);
	}
}

- (void)testRefusesABundleWithoutItsExecutable
{
	NSMutableDictionary *object = [self minimalObject];
	NSURL *bundle = [self bundleWithInfo:[self infoForObject:object] manifest:object executable:nil];
	NSError *error = nil;
	XCTAssertNil([NFKPluginManifest manifestWithBundleURL:bundle error:&error]);
	XCTAssertEqual(error.code, NFKPluginErrorManifestMismatch, @"%@", error);
}

- (void)testRefusesADirectoryWithoutAManifest
{
	NSURL *directory = [[NSURL fileURLWithPath:NSTemporaryDirectory()] URLByAppendingPathComponent:NSUUID.UUID.UUIDString isDirectory:YES];
	[self.temporaryURLs addObject:directory];
	XCTAssertTrue([NSFileManager.defaultManager createDirectoryAtURL:directory withIntermediateDirectories:YES attributes:nil error:NULL]);
	NSError *error = nil;
	XCTAssertNil([NFKPluginManifest manifestWithBundleURL:directory error:&error]);
	XCTAssertEqual(error.code, NFKPluginErrorMalformedManifest);
}

#pragma mark Compatibility

- (void)testCompatibilityNeedsTheSameMajorAndAtLeastTheMinor
{
	NFKPluginManifest *manifest = [self manifestWithObject:[self object:[self minimalObject] setting:@"1.2" atPath:@"compatibility.inferkitAPI"] error:NULL];
	NSError *error = nil;
	XCTAssertTrue([manifest checkCompatibilityWithAPIVersion:@"1.2" error:&error], @"%@", error);
	XCTAssertTrue([manifest checkCompatibilityWithAPIVersion:@"1.7" error:&error], @"%@", error);
	for (NSString *version in @[ @"1.1", @"2.2", @"0.9", @"latest" ]) {
		error = nil;
		XCTAssertFalse([manifest checkCompatibilityWithAPIVersion:version error:&error], @"%@", version);
		XCTAssertEqualObjects(error.domain, NFKPluginErrorDomain);
		XCTAssertEqual(error.code, NFKPluginErrorIncompatible, @"%@", version);
	}
}

#if TARGET_OS_OSX

- (void)testHostCompatibilityChecksTheABIProtocolSystemAndArchitecture
{
	NSMutableDictionary *object = [self fullObject];
	object[@"compatibility"] = @{ @"inferkitAPI": @"1.0", @"pluginABI": @"1.0", @"protocol": @"1.0" };
	NSError *error = nil;
	XCTAssertTrue([[self manifestWithObject:object error:NULL] checkHostCompatibilityWithError:&error], @"%@", error);
	NSDictionary *cases = @{
		@"another plugin ABI major": [self object:object setting:@"2.0" atPath:@"compatibility.pluginABI"],
		@"another protocol major": [self object:object setting:@"2.1" atPath:@"compatibility.protocol"],
		@"a newer API minor": [self object:object setting:@"1.999" atPath:@"compatibility.inferkitAPI"],
		@"a future macOS": [self object:object setting:@{ @"macOS": @"99.0" } atPath:@"compatibility.minimumOS"],
		@"another architecture": [self object:object setting:@[ @"riscv64" ] atPath:@"compatibility.architectures"],
	};
	for (NSString *reason in cases) {
		NFKPluginManifest *manifest = [self manifestWithObject:cases[reason] error:&error];
		XCTAssertNotNil(manifest, @"%@: %@", reason, error);
		error = nil;
		XCTAssertFalse([manifest checkHostCompatibilityWithError:&error], @"%@", reason);
		XCTAssertEqual(error.code, NFKPluginErrorIncompatible, @"%@: %@", reason, error);
	}
}

#endif

@end
