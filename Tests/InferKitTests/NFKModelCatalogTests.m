//
//  NFKModelCatalogTests.m
//  InferKitTests
//

#import <XCTest/XCTest.h>
#import <InferKit/InferKit.h>

static NSString * const NFKCatalogTestProviderIdentifier = @"test.catalog-provider";

// Registered into the process-wide shared catalog, so its one descriptor is a stand-in that the
// visible list hides from every other test.
@interface NFKModelCatalogTestProvider : NSObject <NFKModelDescriptorProvider>
@end

@implementation NFKModelCatalogTestProvider

+ (NSArray<NFKModelDescriptor *> *)modelDescriptors
{
	return @[ [[NFKModelDescriptor alloc] initWithIdentifier:NFKCatalogTestProviderIdentifier displayName:nil
													 family:@"test" summary:nil engine:NFKModelEnginePlugin
											   registryName:nil providerClassName:nil
													  tasks:@[ NFKModelTaskClassification ] inputModalities:nil
										   outputModalities:nil inputs:nil outputs:nil parameters:nil variants:nil
													license:nil requirements:nil residencies:nil standIn:YES
										  introducedVersion:nil extensions:nil] ];
}

@end

@interface NFKModelCatalogTestNotAProvider : NSObject
@end

@implementation NFKModelCatalogTestNotAProvider
@end

@interface NFKModelCatalogTestBackend : NSObject <NFKInferenceBackend>
@property (nonatomic, copy) NSDictionary<NSString *, id> *modelInfo;
@end

@implementation NFKModelCatalogTestBackend

- (BOOL)isReady
{
	return YES;
}

- (NSString *)backendIdentifier
{
	return @"test-backend";
}

- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request error:(NSError **)error
{
	return [NFKInferenceResult resultWithOutputs:@{ NFKOutputText: request.prompt ?: @"" }];
}

- (NSSet<NSString *> *)supportedInputKeys
{
	return [NSSet setWithObjects:NFKInputPrompt, NFKInputImage, nil];
}

- (NSSet<NSString *> *)supportedParameterKeys
{
	return [NSSet setWithObjects:NFKParameterTemperature, NFKParameterMaxTokens, @"test.mode", nil];
}

- (NSSet<NSString *> *)supportedOutputKeys
{
	return [NSSet setWithObjects:NFKOutputText, NFKOutputCitations, nil];
}

@end

@interface NFKModelCatalogTests : XCTestCase
@end

@implementation NFKModelCatalogTests

- (NFKModelDescriptor *)descriptorWithIdentifier:(NSString *)identifier
										  family:(NSString *)family
										   tasks:(NSArray<NSString *> *)tasks
										 standIn:(BOOL)standIn
{
	return [[NFKModelDescriptor alloc] initWithIdentifier:identifier displayName:nil family:family summary:nil
												   engine:NFKModelEngineMLX registryName:nil providerClassName:nil
													tasks:tasks inputModalities:nil outputModalities:nil
												   inputs:@[ [NFKModelPort portWithKey:NFKInputImage required:YES] ]
												  outputs:@[ [NFKModelPort outputPortWithKey:NFKOutputImage required:YES] ]
											   parameters:@[ [NFKModelOption optionWithKey:NFKParameterSeed] ]
												 variants:@[ [NFKModelVariant variantWithIdentifier:@"base" displayName:@"Base"] ]
												  license:[NFKModelLicense licenseWithIdentifier:@"MIT" name:@"MIT" URL:nil]
											 requirements:nil residencies:@[ NFKModelResidencyResident ]
												  standIn:standIn introducedVersion:@"0.4.0"
											   extensions:@{ @"x-order": @1 }];
}

- (NSArray<NSString *> *)identifiersOf:(NSArray<NFKModelDescriptor *> *)descriptors
{
	return [descriptors valueForKey:@"identifier"];
}

- (NSData *)manifestWithHeader:(NSDictionary *)header models:(NSArray *)models
{
	NSMutableDictionary *manifest = [header mutableCopy];
	manifest[@"models"] = models;
	return [NSJSONSerialization dataWithJSONObject:manifest options:0 error:NULL];
}

#pragma mark Contents

- (void)testANewCatalogIsEmpty
{
	NFKModelCatalog *catalog = [[NFKModelCatalog alloc] init];
	XCTAssertEqual(catalog.descriptors.count, (NSUInteger)0);
	XCTAssertNil([catalog descriptorWithIdentifier:@"anything"]);
}

- (void)testAddingADescriptorReplacesOneWithTheSameIdentifier
{
	NFKModelCatalog *catalog = [[NFKModelCatalog alloc] init];
	[catalog addDescriptor:[self descriptorWithIdentifier:@"a" family:@"one" tasks:@[] standIn:NO]];
	[catalog addDescriptor:[self descriptorWithIdentifier:@"a" family:@"two" tasks:@[] standIn:NO]];
	XCTAssertEqual(catalog.descriptors.count, (NSUInteger)1);
	XCTAssertEqualObjects([catalog descriptorWithIdentifier:@"a"].family, @"two");
}

- (void)testRemovingADescriptorLeavesTheOthers
{
	NFKModelCatalog *catalog = [[NFKModelCatalog alloc] init];
	[catalog addDescriptors:@[ [self descriptorWithIdentifier:@"a" family:@"f" tasks:@[] standIn:NO],
							   [self descriptorWithIdentifier:@"b" family:@"f" tasks:@[] standIn:NO] ]];
	[catalog removeDescriptorWithIdentifier:@"a"];
	[catalog removeDescriptorWithIdentifier:@"absent"];
	XCTAssertEqualObjects([self identifiersOf:catalog.descriptors], (@[ @"b" ]));
}

- (void)testDescriptorsAreSortedByIdentifier
{
	NFKModelCatalog *catalog = [[NFKModelCatalog alloc] init];
	for (NSString *identifier in @[ @"mlx.whisper", @"apple.vision-text", @"mlx.qwen3", @"Z.upper" ]) {
		[catalog addDescriptor:[self descriptorWithIdentifier:identifier family:@"f" tasks:@[] standIn:NO]];
	}
	XCTAssertEqualObjects([self identifiersOf:catalog.descriptors],
						  (@[ @"Z.upper", @"apple.vision-text", @"mlx.qwen3", @"mlx.whisper" ]));
}

- (void)testVisibleDescriptorsHideStandIns
{
	NFKModelCatalog *catalog = [[NFKModelCatalog alloc] init];
	[catalog addDescriptors:@[ [self descriptorWithIdentifier:@"real" family:@"f" tasks:@[] standIn:NO],
							   [self descriptorWithIdentifier:@"reference" family:@"f" tasks:@[] standIn:YES] ]];
	XCTAssertEqualObjects([self identifiersOf:catalog.visibleDescriptors], (@[ @"real" ]));
	XCTAssertEqual(catalog.descriptors.count, (NSUInteger)2);
}

- (void)testDescriptorsAreFoundByTaskAndByFamily
{
	NFKModelCatalog *catalog = [[NFKModelCatalog alloc] init];
	[catalog addDescriptors:@[
		[self descriptorWithIdentifier:@"depth" family:@"depth-anything" tasks:@[ NFKModelTaskDepthEstimation ] standIn:NO],
		[self descriptorWithIdentifier:@"matte" family:@"birefnet" tasks:@[ NFKModelTaskMatting, NFKModelTaskSegmentation ] standIn:NO],
		[self descriptorWithIdentifier:@"mask" family:@"u2net" tasks:@[ NFKModelTaskSegmentation ] standIn:YES],
	]];
	XCTAssertEqualObjects([self identifiersOf:[catalog descriptorsForTask:NFKModelTaskSegmentation]], (@[ @"mask", @"matte" ]));
	XCTAssertEqualObjects([self identifiersOf:[catalog descriptorsInFamily:@"depth-anything"]], (@[ @"depth" ]));
	XCTAssertEqual([catalog descriptorsForTask:NFKModelTaskForecasting].count, (NSUInteger)0);
}

#pragma mark Manifests

- (void)testAManifestRoundTripsByteForByte
{
	NFKModelCatalog *catalog = [[NFKModelCatalog alloc] init];
	[catalog addDescriptors:@[ [self descriptorWithIdentifier:@"b" family:@"f" tasks:@[ NFKModelTaskUpscaling ] standIn:NO],
							   [self descriptorWithIdentifier:@"a" family:@"f" tasks:@[] standIn:YES] ]];
	NSError *error = nil;
	NSData *first = [catalog manifestDataWithError:&error];
	XCTAssertNotNil(first, @"%@", error);

	NFKModelCatalog *reread = [[NFKModelCatalog alloc] init];
	XCTAssertTrue([reread addDescriptorsFromManifestData:first error:&error], @"%@", error);
	XCTAssertEqualObjects(reread.descriptors, catalog.descriptors);
	XCTAssertEqualObjects([reread manifestDataWithError:NULL], first);

	NSDictionary *manifest = [NSJSONSerialization JSONObjectWithData:first options:0 error:NULL];
	XCTAssertEqualObjects(manifest[@"schemaVersion"], @1);
	XCTAssertEqualObjects(manifest[@"kind"], NFKModelCatalogManifestKind);
	XCTAssertEqualObjects([manifest[@"models"] valueForKey:@"id"], (@[ @"a", @"b" ]));
	XCTAssertEqualObjects(manifest[@"models"][0][@"x-order"], @1);
}

- (void)testAnEmptyCatalogWritesAnEmptyManifest
{
	NSData *data = [[[NFKModelCatalog alloc] init] manifestDataWithError:NULL];
	NSDictionary *manifest = [NSJSONSerialization JSONObjectWithData:data options:0 error:NULL];
	XCTAssertEqualObjects(manifest[@"models"], @[]);
}

- (void)testANewerSchemaVersionIsRefused
{
	NFKModelCatalog *catalog = [[NFKModelCatalog alloc] init];
	NSData *data = [self manifestWithHeader:@{ @"schemaVersion": @(NFKModelCatalog.schemaVersion + 1),
											   @"kind": NFKModelCatalogManifestKind }
									 models:@[]];
	NSError *error = nil;
	XCTAssertFalse([catalog addDescriptorsFromManifestData:data error:&error]);
	XCTAssertEqualObjects(error.domain, NFKModelCatalogErrorDomain);
	XCTAssertEqual(error.code, NFKModelCatalogErrorUnsupportedSchemaVersion);
}

- (void)testAMissingOrDifferentKindIsRefused
{
	NFKModelCatalog *catalog = [[NFKModelCatalog alloc] init];
	for (NSDictionary *header in @[ @{ @"schemaVersion": @1 },
									@{ @"schemaVersion": @1, @"kind": @"inferkit.plugin" } ]) {
		NSError *error = nil;
		XCTAssertFalse([catalog addDescriptorsFromManifestData:[self manifestWithHeader:header models:@[]] error:&error]);
		XCTAssertEqual(error.code, NFKModelCatalogErrorMalformedManifest, @"%@", header);
	}
}

- (void)testAManifestThatIsNotAJSONObjectIsRefused
{
	NFKModelCatalog *catalog = [[NFKModelCatalog alloc] init];
	NSArray<NSData *> *payloads = @[ [@"not json" dataUsingEncoding:NSUTF8StringEncoding],
									 [@"[]" dataUsingEncoding:NSUTF8StringEncoding],
									 [self manifestWithHeader:@{ @"kind": NFKModelCatalogManifestKind } models:@[]],
									 [NSJSONSerialization dataWithJSONObject:@{ @"schemaVersion": @1,
																				@"kind": NFKModelCatalogManifestKind,
																				@"models": @{} }
																	 options:0 error:NULL] ];
	for (NSData *payload in payloads) {
		NSError *error = nil;
		XCTAssertFalse([catalog addDescriptorsFromManifestData:payload error:&error]);
		XCTAssertEqual(error.code, NFKModelCatalogErrorMalformedManifest);
	}
}

- (void)testAMalformedDescriptorLeavesTheCatalogUnchanged
{
	NFKModelCatalog *catalog = [[NFKModelCatalog alloc] init];
	NSDictionary *header = @{ @"schemaVersion": @1, @"kind": NFKModelCatalogManifestKind };
	NSArray *models = @[ @{ @"id": @"good", @"family": @"f", @"engine": NFKModelEngineMLX },
						 @{ @"id": @"bad", @"family": @"f" } ];
	NSError *error = nil;
	XCTAssertFalse([catalog addDescriptorsFromManifestData:[self manifestWithHeader:header models:models] error:&error]);
	XCTAssertEqual(error.code, NFKModelCatalogErrorMalformedDescriptor);
	XCTAssertTrue([error.localizedDescription containsString:@"models[1].engine"], @"%@", error.localizedDescription);
	XCTAssertEqual(catalog.descriptors.count, (NSUInteger)0);
}

- (void)testAnExtensionJSONCannotCarryFailsTheManifest
{
	NFKModelCatalog *catalog = [[NFKModelCatalog alloc] init];
	[catalog addDescriptor:[[NFKModelDescriptor alloc] initWithIdentifier:@"m" displayName:nil family:@"f" summary:nil
																   engine:NFKModelEngineMLX registryName:nil
														providerClassName:nil tasks:nil inputModalities:nil
														 outputModalities:nil inputs:nil outputs:nil parameters:nil
																 variants:nil license:nil requirements:nil
															  residencies:nil standIn:NO introducedVersion:nil
															   extensions:@{ @"x-date": [NSDate date] }]];
	NSError *error = nil;
	XCTAssertNil([catalog manifestDataWithError:&error]);
	XCTAssertEqual(error.code, NFKModelCatalogErrorMalformedDescriptor);
}

#pragma mark Providers

- (void)testTheSharedCatalogNamesTheCompanionProviders
{
	NSArray<NSString *> *names = NFKModelCatalog.providerClassNames;
	NSArray<NSString *> *known = @[ @"NFKAppleModelDescriptors", @"NFKMLXModelDescriptors",
									@"NFKFoundationModelsDescriptors", @"NFKAppleSwiftModelDescriptors" ];
	XCTAssertEqualObjects([names subarrayWithRange:NSMakeRange(0, known.count)], known);
}

- (void)testARegisteredProviderReachesTheSharedCatalog
{
	[NFKModelCatalog registerProviderClassName:NSStringFromClass(NFKModelCatalogTestProvider.class)];
	XCTAssertTrue([NFKModelCatalog.providerClassNames containsObject:@"NFKModelCatalogTestProvider"]);

	NFKModelDescriptor *descriptor = [NFKModelCatalog.sharedCatalog descriptorWithIdentifier:NFKCatalogTestProviderIdentifier];
	XCTAssertNotNil(descriptor);
	XCTAssertTrue(descriptor.isStandIn);
	XCTAssertFalse([[self identifiersOf:NFKModelCatalog.sharedCatalog.visibleDescriptors]
					containsObject:NFKCatalogTestProviderIdentifier]);
}

- (void)testAProviderClassAddsItsDescriptorsOnlyWhenItAdoptsTheProtocol
{
	NFKModelCatalog *catalog = [[NFKModelCatalog alloc] init];
	XCTAssertFalse([catalog addDescriptorsFromProviderClassName:@"NFKNoSuchDescriptorProvider"]);
	XCTAssertFalse([catalog addDescriptorsFromProviderClassName:NSStringFromClass(NFKModelCatalogTestNotAProvider.class)]);
	XCTAssertEqual(catalog.descriptors.count, (NSUInteger)0);
	XCTAssertTrue([catalog addDescriptorsFromProviderClassName:NSStringFromClass(NFKModelCatalogTestProvider.class)]);
	XCTAssertEqualObjects([self identifiersOf:catalog.descriptors], (@[ NFKCatalogTestProviderIdentifier ]));
}

#pragma mark Descriptors from backends

- (void)testADescriptorDerivedFromABackendTypesItsKeys
{
	NFKModelCatalogTestBackend *backend = [[NFKModelCatalogTestBackend alloc] init];
	NFKModelDescriptor *descriptor = [NFKModelDescriptor descriptorWithIdentifier:@"app.captioner"
															   derivedFromBackend:backend];
	XCTAssertEqualObjects(descriptor.identifier, @"app.captioner");
	XCTAssertEqualObjects(descriptor.displayName, @"app.captioner");
	XCTAssertEqualObjects(descriptor.family, @"test-backend");
	XCTAssertEqualObjects(descriptor.engine, NFKModelEnginePlugin);

	XCTAssertEqualObjects([descriptor.inputs valueForKey:@"key"], (@[ NFKInputImage, NFKInputPrompt ]));
	XCTAssertEqualObjects(descriptor.inputModalities, (@[ NFKModelValueTypeImage, NFKModelValueTypeText ]));
	XCTAssertFalse(descriptor.inputs.firstObject.isRequired);

	XCTAssertEqualObjects([descriptor.outputs valueForKey:@"key"], (@[ NFKOutputCitations, NFKOutputText ]));
	XCTAssertEqualObjects([descriptor.outputs valueForKey:@"valueType"], (@[ NFKModelValueTypeJSON, NFKModelValueTypeText ]));

	XCTAssertEqualObjects([descriptor.parameters valueForKey:@"key"],
						  (@[ NFKParameterMaxTokens, NFKParameterTemperature, @"test.mode" ]));
	XCTAssertEqualObjects([descriptor.parameters valueForKey:@"kind"],
						  (@[ NFKModelOptionKindInteger, NFKModelOptionKindNumber, NFKModelOptionKindJSON ]));
}

- (void)testADerivedDescriptorTakesItsFamilyFromModelInfo
{
	NFKModelCatalogTestBackend *backend = [[NFKModelCatalogTestBackend alloc] init];
	backend.modelInfo = @{ NFKModelInfoArchitecture: @"qwen3" };
	XCTAssertEqualObjects([NFKModelDescriptor descriptorWithIdentifier:@"m" derivedFromBackend:backend].family, @"qwen3");
}

- (void)testABackendThatDeclaresNothingDerivesAnEmptyDescriptor
{
	NFKPassthroughBackend *backend = [NFKPassthroughBackend backend];
	NFKModelDescriptor *descriptor = [NFKModelDescriptor descriptorWithIdentifier:@"m" derivedFromBackend:backend];
	XCTAssertEqualObjects(descriptor.family, backend.backendIdentifier);
	XCTAssertEqual(descriptor.outputs.count, (NSUInteger)0);
}

@end
