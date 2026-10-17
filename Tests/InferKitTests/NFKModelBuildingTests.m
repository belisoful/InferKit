//
//  NFKModelBuildingTests.m
//  InferKitTests
//

#import <XCTest/XCTest.h>
#import <InferKit/InferKit.h>

static NSString * const NFKBuildTestEngine = @"build-test-engine";
static const NSInteger NFKBuildTestGiB = (NSInteger)1 << 30;

/*! A backend with two settable properties and one read-only one, which records the requests it runs. */
@interface NFKBuildTestBackend : NSObject <NFKInferenceBackend>
@property (nonatomic) NSInteger computeUnits;
@property (nonatomic, copy, nullable) NSString *mode;
@property (nonatomic, readonly) NSInteger quality;
@property (nonatomic, readonly) NSMutableArray<NFKInferenceRequest *> *requests;
@end

@implementation NFKBuildTestBackend

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_requests = [NSMutableArray array];
	}
	return self;
}

- (BOOL)isReady
{
	return YES;
}

- (NSString *)backendIdentifier
{
	return @"build-test";
}

- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request error:(NSError **)error
{
	[self.requests addObject:request];
	return [NFKInferenceResult resultWithOutputs:@{ NFKOutputText: @"ok" }];
}

@end

/*! An engine builder that records the request it was given. */
@interface NFKBuildTestBuilder : NSObject <NFKModelBackendBuilding>
@property (nonatomic, nullable) NFKModelBuildRequest *lastRequest;
@end

@implementation NFKBuildTestBuilder

- (nullable id<NFKInferenceBackend>)backendForBuildRequest:(NFKModelBuildRequest *)request error:(NSError **)error
{
	self.lastRequest = request;
	NFKBuildTestBackend *backend = [[NFKBuildTestBackend alloc] init];
	if (![request applyPropertyOptionsToBackend:backend error:error]) {
		return nil;
	}
	return backend;
}

@end

/*! A provider whose class builds its own descriptor's backends. */
@interface NFKBuildTestProvider : NSObject <NFKModelDescriptorProvider>
@end

@implementation NFKBuildTestProvider

+ (NSArray<NFKModelDescriptor *> *)modelDescriptors
{
	return @[ [[NFKModelDescriptor alloc] initWithIdentifier:@"provided.model" displayName:nil family:@"test" summary:nil
													  engine:NFKBuildTestEngine registryName:nil providerClassName:nil
													   tasks:nil inputModalities:nil outputModalities:nil inputs:nil
													 outputs:nil options:nil variants:nil license:nil requirements:nil
												 residencies:nil standIn:NO introducedVersion:nil extensions:nil] ];
}

+ (nullable id<NFKInferenceBackend>)backendForBuildRequest:(NFKModelBuildRequest *)request error:(NSError **)error
{
	NFKBuildTestBackend *backend = [[NFKBuildTestBackend alloc] init];
	backend.mode = @"from-provider";
	return backend;
}

@end

/*! A dynamic provider, the fallback a descriptor names in providerClass. */
@interface NFKBuildTestDynamicProvider : NSObject <NFKDynamicBackendProvider>
@end

@implementation NFKBuildTestDynamicProvider

+ (nullable id<NFKInferenceBackend>)makeInferenceBackend
{
	return [[NFKBuildTestBackend alloc] init];
}

@end

@interface NFKModelBuildingTests : XCTestCase
@property (nonatomic) NFKBuildTestBuilder *builder;
@property (nonatomic) NFKModelCatalog *catalog;
@end

@implementation NFKModelBuildingTests

// Each catalog builds against a 64 GiB working set, a 54.4 GiB budget, so no build depends on the memory of
// the machine that runs the tests.
- (void)setUp
{
	self.builder = [[NFKBuildTestBuilder alloc] init];
	[NFKModelCatalog registerBackendBuilder:self.builder forEngine:NFKBuildTestEngine];
	self.catalog = [[NFKModelCatalog alloc] init];
	self.catalog.memoryBudget = [self generousBudget];
	NFKModelCatalog.sharedCatalog.memoryBudget = [self generousBudget];
	[self.catalog addDescriptor:[self modelWithIdentifier:@"build.model" engine:NFKBuildTestEngine provider:nil
											 requirements:nil variant:nil]];
}

- (void)tearDown
{
	NFKModelCatalog.sharedCatalog.memoryBudget = nil;
	[NFKModelCatalog registerBackendBuilder:nil forEngine:NFKBuildTestEngine];
}

- (NFKModelMemoryBudget *)generousBudget
{
	return [NFKModelMemoryBudget budgetWithWorkingSetByteCount:64 * NFKBuildTestGiB];
}

- (NFKModelOption *)loadOption:(NSString *)key kind:(NSString *)kind property:(nullable NSString *)property
				  defaultValue:(nullable id)defaultValue choices:(nullable NSArray *)choices
{
	return [[NFKModelOption alloc] initWithKey:key kind:kind title:nil help:nil defaultValue:defaultValue minimum:nil
									   maximum:nil step:nil choices:choices stage:NFKModelOptionStageLoad group:nil
								  requirements:nil propertyKeyPath:property memoryBytesPerUnit:nil extensions:nil];
}

// A model with a property option, an argument option, a context length, a choice that needs a feature no
// machine has, a request option, and a preset.
- (NFKModelDescriptor *)modelWithIdentifier:(NSString *)identifier
									 engine:(NSString *)engine
								   provider:(nullable NSString *)provider
							   requirements:(nullable NFKModelRequirements *)requirements
									variant:(nullable NFKModelVariant *)variant
{
	NFKModelRequirements *unmeetable = [[NFKModelRequirements alloc] initWithMinimumOSVersions:nil requiresAppleSilicon:NO
																		minimumMemoryByteCount:nil
																			  requiredFeatures:@[ @"made-up" ]
																				   unmetReason:nil extensions:nil];
	NFKModelOptionChoice *exotic = [[NFKModelOptionChoice alloc] initWithValue:@"exotic" title:@"Exotic"
																  requirements:unmeetable memory:nil memoryByVariant:nil
																	extensions:nil];
	NSArray *options = @[
		[self loadOption:@"mode" kind:NFKModelOptionKindText property:@"mode" defaultValue:@"plain" choices:nil],
		[self loadOption:@"layout" kind:NFKModelOptionKindChoice property:nil defaultValue:@"standard"
				 choices:@[ [NFKModelOptionChoice choiceWithValue:@"standard" title:@"Standard"], exotic ]],
		[self loadOption:NFKModelLoadOptionContextLength kind:NFKModelOptionKindInteger property:nil defaultValue:@4096
				 choices:nil],
		[[NFKModelOption alloc] initWithKey:@"steps" kind:NFKModelOptionKindInteger title:nil help:nil defaultValue:@20
									minimum:@1 maximum:nil step:nil choices:nil stage:NFKModelOptionStageRequest group:nil
								 extensions:nil],
	];
	NFKModelMemoryModel *model = [[NFKModelMemoryModel alloc] initWithPrecisionOptionKey:nil quantizationBitsOptionKey:nil
														  quantizationGroupSizeOptionKey:nil
																  contextLengthOptionKey:NFKModelLoadOptionContextLength
																	  cacheBitsOptionKey:nil cacheGroupSizeOptionKey:nil
																		  draftOptionKey:nil draftBitsOptionKey:nil
																	  residencyOptionKey:nil cacheStage:nil draftStage:nil
																		 stagePrecisions:nil extensions:nil];
	NFKModelPreset *fast = [[NFKModelPreset alloc] initWithIdentifier:@"fast" title:@"Fast" help:nil baseModelIdentifier:nil
														  loadOptions:nil requestDefaults:@{ @"steps": @4 } suggested:YES
														   extensions:nil];
	return [[NFKModelDescriptor alloc] initWithIdentifier:identifier displayName:@"Build Model" family:@"test"
												  summary:nil engine:engine registryName:nil providerClassName:provider
													tasks:nil inputModalities:nil outputModalities:nil inputs:nil
												  outputs:nil options:options variants:variant != nil ? @[ variant ] : nil
												  license:nil requirements:requirements residencies:nil memoryModel:model
												  presets:@[ fast ] standIn:NO introducedVersion:nil extensions:nil];
}

- (NFKModelBuildRequest *)requestWithLoadOptions:(NSDictionary *)loadOptions
{
	NFKModelDescriptor *descriptor = [self.catalog descriptorWithIdentifier:@"build.model"];
	NSMutableArray *options = [descriptor.options mutableCopy];
	[options addObject:[self loadOption:@"quality" kind:NFKModelOptionKindInteger property:@"quality" defaultValue:nil
								choices:nil]];
	[options addObject:[self loadOption:@"units" kind:NFKModelOptionKindInteger property:@"computeUnits" defaultValue:nil
								choices:nil]];
	NFKModelDescriptor *withMore = [[NFKModelDescriptor alloc] initWithIdentifier:descriptor.identifier displayName:nil
																		   family:descriptor.family summary:nil
																		   engine:descriptor.engine registryName:nil
																providerClassName:nil tasks:nil inputModalities:nil
																 outputModalities:nil inputs:nil outputs:nil
																		  options:options variants:nil license:nil
																	 requirements:nil residencies:nil standIn:NO
																introducedVersion:nil extensions:nil];
	return [[NFKModelBuildRequest alloc] initWithDescriptor:withMore variant:nil loadOptions:loadOptions requestDefaults:nil
												   location:nil memoryEstimate:nil catalog:self.catalog];
}

#pragma mark Property options

- (void)testPropertyOptionsAreSetByKeyValueCoding
{
	NFKModelBuildRequest *request = [self requestWithLoadOptions:@{ @"mode": @"fast", @"units": @2, @"layout": @"standard" }];
	XCTAssertEqualObjects(request.propertyOptions, (@{ @"mode": @"fast", @"units": @2 }));
	XCTAssertEqualObjects(request.argumentOptions, (@{ @"layout": @"standard" }));
	NFKBuildTestBackend *backend = [[NFKBuildTestBackend alloc] init];
	NSError *error = nil;
	XCTAssertTrue([request applyPropertyOptionsToBackend:backend error:&error], @"%@", error);
	XCTAssertEqualObjects(backend.mode, @"fast");
	XCTAssertEqual(backend.computeUnits, 2);
}

- (void)testAMissingSetterFailsAndSetsNothing
{
	NFKModelBuildRequest *request = [self requestWithLoadOptions:@{ @"mode": @"fast", @"quality": @3 }];
	NFKBuildTestBackend *backend = [[NFKBuildTestBackend alloc] init];
	NSError *error = nil;
	XCTAssertFalse([request applyPropertyOptionsToBackend:backend error:&error]);
	XCTAssertEqual(error.code, NFKModelCatalogErrorInvalidOptions);
	XCTAssertEqualObjects(error.localizedDescription,
						  @"The NFKBuildTestBackend backend has no property \"quality\" to set \"quality\" on.");
	XCTAssertNil(backend.mode);
}

#pragma mark Building

- (void)testAnEngineBuilderBuildsWithTheResolvedOptions
{
	NSURL *location = [NSURL fileURLWithPath:@"/tmp/build-model"];
	NSError *error = nil;
	id<NFKInferenceBackend> backend = [self.catalog backendForModelIdentifier:@"build.model"
																	  options:@{ @"mode": @"fast", @"steps": @"5" }
																	 location:location error:&error];
	XCTAssertNotNil(backend, @"%@", error);
	NFKModelBuildRequest *request = self.builder.lastRequest;
	XCTAssertEqualObjects(request.location, location);
	XCTAssertEqualObjects(request.loadOptions, (@{ @"mode": @"fast", @"layout": @"standard",
												   NFKModelLoadOptionContextLength: @4096 }));
	XCTAssertEqualObjects(request.propertyOptions, (@{ @"mode": @"fast" }));
	XCTAssertEqualObjects(request.requestDefaults, (@{ @"steps": @5, NFKModelLoadOptionContextLength: @4096 }));
	XCTAssertNotNil(request.memoryEstimate);

	XCTAssertTrue([backend isKindOfClass:NFKDefaultedBackend.class]);
	NFKBuildTestBackend *built = (NFKBuildTestBackend *)((NFKDefaultedBackend *)backend).backend;
	XCTAssertEqualObjects(built.mode, @"fast");
	XCTAssertEqualObjects(NFKInferenceLoadOptionValues(backend), request.loadOptions);
}

- (void)testAProviderClassBuildsBeforeTheEngineBuilder
{
	XCTAssertTrue([self.catalog addDescriptorsFromProviderClassName:@"NFKBuildTestProvider"]);
	NSError *error = nil;
	id<NFKInferenceBackend> backend = [self.catalog backendForModelIdentifier:@"provided.model" options:nil error:&error];
	XCTAssertNotNil(backend, @"%@", error);
	XCTAssertEqualObjects(((NFKBuildTestBackend *)backend).mode, @"from-provider");
	XCTAssertNil(self.builder.lastRequest);
}

- (void)testTheDynamicProviderIsTheLastResortWithItsPropertiesSet
{
	[self.catalog addDescriptor:[self modelWithIdentifier:@"dynamic.model" engine:@"build-test-unbuilt"
												 provider:@"NFKBuildTestDynamicProvider" requirements:nil variant:nil]];
	NSError *error = nil;
	id<NFKInferenceBackend> backend = [self.catalog backendForModelIdentifier:@"dynamic.model"
																	  options:@{ @"mode": @"dynamic" } error:&error];
	XCTAssertNotNil(backend, @"%@", error);
	NFKBuildTestBackend *built = (NFKBuildTestBackend *)((NFKDefaultedBackend *)backend).backend;
	XCTAssertEqualObjects(built.mode, @"dynamic");
}

- (void)testAnEngineWithoutABuilderFails
{
	[self.catalog addDescriptor:[self modelWithIdentifier:@"orphan.model" engine:@"build-test-nowhere" provider:nil
											 requirements:nil variant:nil]];
	NSError *error = nil;
	XCTAssertNil([self.catalog backendForModelIdentifier:@"orphan.model" options:nil error:&error]);
	XCTAssertEqual(error.code, NFKModelCatalogErrorNoBuilder);
	XCTAssertEqualObjects(error.localizedDescription, @"No builder builds models of the engine \"build-test-nowhere\".");
}

- (void)testAnUnknownModelFails
{
	NSError *error = nil;
	XCTAssertNil([self.catalog backendForModelIdentifier:@"ghost" options:nil error:&error]);
	XCTAssertEqual(error.code, NFKModelCatalogErrorUnknownModel);
	XCTAssertEqualObjects(error.localizedDescription, @"The catalog has no model \"ghost\".");
	XCTAssertNil([self.catalog backendForModelIdentifier:@"build.model@slow" options:nil error:&error]);
	XCTAssertEqualObjects(error.localizedDescription, @"The catalog has no model \"build.model@slow\".");
}

- (void)testInvalidOptionsFail
{
	NSError *error = nil;
	XCTAssertNil([self.catalog backendForModelIdentifier:@"build.model" options:@{ @"steps": @0 } error:&error]);
	XCTAssertEqual(error.code, NFKModelCatalogErrorInvalidOptions);
	XCTAssertEqual([error.userInfo[NFKModelOptionProblemsKey] count], 1u);
}

- (void)testAModelOrAChoiceThisMachineCannotRunIsUnavailable
{
	NFKModelRequirements *unmeetable = [[NFKModelRequirements alloc] initWithMinimumOSVersions:nil requiresAppleSilicon:NO
																		minimumMemoryByteCount:nil
																			  requiredFeatures:@[ @"made-up" ]
																				   unmetReason:nil extensions:nil];
	[self.catalog addDescriptor:[self modelWithIdentifier:@"unavailable.model" engine:NFKBuildTestEngine provider:nil
											 requirements:unmeetable variant:nil]];
	NSError *error = nil;
	XCTAssertNil([self.catalog backendForModelIdentifier:@"unavailable.model" options:nil error:&error]);
	XCTAssertEqual(error.code, NFKModelCatalogErrorUnavailable);
	XCTAssertEqualObjects(error.localizedDescription, @"Build Model cannot run on this machine. This needs the made-up feature.");

	XCTAssertNil([self.catalog backendForModelIdentifier:@"build.model" options:@{ @"layout": @"exotic" } error:&error]);
	XCTAssertEqual(error.code, NFKModelCatalogErrorUnavailable);
	XCTAssertEqualObjects(error.localizedDescription,
						  @"\"layout\" cannot be \"exotic\" on this machine. This needs the made-up feature.");
}

- (void)testAModelThatDoesNotFitIsRefusedUnlessTheUserOverrides
{
	NFKModelVariant *huge = [[NFKModelVariant alloc] initWithIdentifier:@"huge" displayName:nil isDefault:YES
														   registryName:nil source:nil downloadByteCount:nil
													  residentByteCount:@((NSInteger)1 << 50) peakByteCount:nil
														 parameterCount:nil precision:nil minimumMemoryByteCount:nil
															 extensions:nil];
	[self.catalog addDescriptor:[self modelWithIdentifier:@"huge.model" engine:NFKBuildTestEngine provider:nil
											 requirements:nil variant:huge]];
	NSError *error = nil;
	XCTAssertNil([self.catalog backendForModelIdentifier:@"huge.model" options:nil error:&error]);
	XCTAssertEqual(error.code, NFKModelCatalogErrorDoesNotFit);
	NFKModelMemoryEstimate *estimate = error.userInfo[NFKModelMemoryEstimateKey];
	XCTAssertEqual(estimate.peakByteCount, (NSInteger)1 << 50);
	XCTAssertEqualObjects(estimate.budget, [self generousBudget]);
	NSString *expected = @"The model needs 1048576.0 GiB plus a 4.0 GiB reserve, against a 54.4 GiB budget. "
						  "No combination of options fits: the smallest needs 1048576.0 GiB plus a 4.0 GiB reserve, "
						  "against a 54.4 GiB budget.";
	XCTAssertEqualObjects(error.localizedDescription, expected, @"nothing fits, so the smallest need is named");
	XCTAssertNil(error.userInfo[NFKModelMemoryFittingOptionsKey]);
	XCTAssertNil(error.userInfo[NFKModelMemoryFittingEstimateKey]);

	id<NFKInferenceBackend> backend = [self.catalog backendForModelIdentifier:@"huge.model"
																	  options:@{ NFKModelLoadOptionIgnoreMemoryBudget: @YES }
																		error:&error];
	XCTAssertNotNil(backend, @"%@", error);
	XCTAssertNil(self.builder.lastRequest.loadOptions[NFKModelLoadOptionIgnoreMemoryBudget]);
	XCTAssertFalse(self.builder.lastRequest.memoryEstimate.fits);
}

// One stage of 2^50 bytes whose routed experts are all but 1 GiB, held resident by default.
- (NFKModelDescriptor *)pageableModel
{
	NSInteger stored = (NSInteger)1 << 50;
	NFKModelStageFacts *stage = [[NFKModelStageFacts alloc] initWithIdentifier:@"decoder" title:nil
														  byteCountByPrecision:@{ NFKModelPrecisionCheckpoint: @(stored) }
													   quantizableElementCount:0
												  pageableByteCountByPrecision:@{ NFKModelPrecisionCheckpoint: @(stored - ((NSInteger)1 << 30)) }
																	layerCount:0 largestLayerByteCountByPrecision:nil
															  widenedPrecision:nil extensions:nil];
	NFKModelMemoryFacts *facts = [[NFKModelMemoryFacts alloc] initWithStages:@[ stage ] keyValueByteCountPerToken:nil
													boundedKeyValueByteCount:nil maximumContextLength:nil vocabularySize:nil
													   loadOverheadByteCount:0 basis:nil extensions:nil];
	NFKModelVariant *variant = [[NFKModelVariant alloc] initWithIdentifier:@"base" displayName:nil isDefault:YES
															  registryName:nil source:nil downloadByteCount:nil
														 residentByteCount:nil peakByteCount:nil parameterCount:nil
																 precision:NFKModelPrecisionBFloat16 minimumMemoryByteCount:nil
																   license:nil memoryFacts:facts extensions:nil];
	NFKModelOption *residency = [self loadOption:NFKModelLoadOptionResidency kind:NFKModelOptionKindChoice property:nil
									defaultValue:NFKModelResidencyResident
										 choices:@[ [NFKModelOptionChoice choiceWithValue:NFKModelResidencyResident title:@"Resident"],
													[NFKModelOptionChoice choiceWithValue:NFKModelResidencyPaged title:@"Paged"] ]];
	NFKModelMemoryModel *model = [[NFKModelMemoryModel alloc] initWithPrecisionOptionKey:nil quantizationBitsOptionKey:nil
														  quantizationGroupSizeOptionKey:nil contextLengthOptionKey:nil
																	  cacheBitsOptionKey:nil cacheGroupSizeOptionKey:nil
																		  draftOptionKey:nil draftBitsOptionKey:nil
																	  residencyOptionKey:NFKModelLoadOptionResidency
																			  cacheStage:nil draftStage:nil
																		 stagePrecisions:nil extensions:nil];
	return [[NFKModelDescriptor alloc] initWithIdentifier:@"pageable.model" displayName:nil family:@"test" summary:nil
												   engine:NFKBuildTestEngine registryName:nil providerClassName:nil
													tasks:nil inputModalities:nil outputModalities:nil inputs:nil outputs:nil
												  options:@[ residency ] variants:@[ variant ] license:nil requirements:nil
											  residencies:nil memoryModel:model presets:nil standIn:NO
										introducedVersion:nil extensions:nil];
}

- (void)testARefusalNamesTheNearestCombinationThatFits
{
	[self.catalog addDescriptor:[self pageableModel]];
	NSError *error = nil;
	XCTAssertNil([self.catalog backendForModelIdentifier:@"pageable.model" options:nil error:&error]);
	XCTAssertEqual(error.code, NFKModelCatalogErrorDoesNotFit);
	NSDictionary *fitting = error.userInfo[NFKModelMemoryFittingOptionsKey];
	NFKModelMemoryEstimate *fittingEstimate = error.userInfo[NFKModelMemoryFittingEstimateKey];
	XCTAssertEqualObjects(fitting[NFKModelLoadOptionResidency], NFKModelResidencyPaged);
	XCTAssertTrue(fittingEstimate.fits);
	XCTAssertEqualObjects(fittingEstimate.residency, NFKModelResidencyPaged);
	NFKModelMemoryEstimate *estimate = error.userInfo[NFKModelMemoryEstimateKey];
	NSString *expected = [NSString stringWithFormat:@"%@ Choosing residency paged fits: it holds %@.", estimate.reason,
						  [NFKModelMemoryBudget formattedByteCount:fittingEstimate.peakByteCount]];
	XCTAssertEqualObjects(error.localizedDescription, expected);

	XCTAssertNotNil([self.catalog backendForModelIdentifier:@"pageable.model" options:fitting error:&error], @"%@", error);
}

- (void)testTheCatalogBuildsAgainstTheBudgetItIsGiven
{
	NFKModelVariant *twoGiB = [[NFKModelVariant alloc] initWithIdentifier:@"2g" displayName:nil isDefault:YES
															 registryName:nil source:nil downloadByteCount:nil
														residentByteCount:@(2 * NFKBuildTestGiB) peakByteCount:nil
														   parameterCount:nil precision:nil minimumMemoryByteCount:nil
															   extensions:nil];
	[self.catalog addDescriptor:[self modelWithIdentifier:@"small.model" engine:NFKBuildTestEngine provider:nil
											 requirements:nil variant:twoGiB]];

	self.catalog.memoryBudget = [NFKModelMemoryBudget budgetWithWorkingSetByteCount:4 * NFKBuildTestGiB];
	NSError *error = nil;
	XCTAssertNil([self.catalog backendForModelIdentifier:@"small.model" options:nil error:&error]);
	XCTAssertEqual(error.code, NFKModelCatalogErrorDoesNotFit);
	XCTAssertEqualObjects(error.localizedDescription,
						  @"The model needs 2.0 GiB plus a 1.7 GiB reserve, against a 3.4 GiB budget. No combination of "
						  "options fits: the smallest needs 2.0 GiB plus a 1.7 GiB reserve, against a 3.4 GiB budget.",
						  @"a budget below 8 GiB keeps half of itself");

	self.catalog.memoryBudget = [NFKModelMemoryBudget budgetWithWorkingSetByteCount:8 * NFKBuildTestGiB];
	XCTAssertNotNil([self.catalog backendForModelIdentifier:@"small.model" options:nil error:&error], @"%@", error);
	XCTAssertEqualObjects(self.builder.lastRequest.memoryEstimate.budget, self.catalog.memoryBudget);
	XCTAssertTrue(self.builder.lastRequest.memoryEstimate.fits, @"2 GiB beside a 3.4 GiB reserve fits a 6.8 GiB budget");

	self.catalog.memoryBudget = nil;
	XCTAssertEqualObjects(self.catalog.memoryBudget, [NFKModelMemoryBudget currentBudget], @"nil restores this machine's");
}

- (void)testAModelThatNeedsNoMemoryBuildsUnderEveryBudget
{
	for (NSNumber *workingSet in @[ @2, @(NFKBuildTestGiB), @(4 * NFKBuildTestGiB), @(64 * NFKBuildTestGiB) ]) {
		self.catalog.memoryBudget = [NFKModelMemoryBudget budgetWithWorkingSetByteCount:workingSet.integerValue];
		NSError *error = nil;
		XCTAssertNotNil([self.catalog backendForModelIdentifier:@"build.model" options:nil error:&error],
						@"%@: %@", workingSet, error);
	}
}

// A hosted runner's budget reads 4.0 GiB. A shipped model that declares no memory builds through the shared
// catalog's own builder there, and on the smaller budget of a 4 GiB device; the fixed 4 GiB reserve
// refused it on both.
- (void)testAShippedModelBuildsOnASmallDevicesBudget
{
	NFKModelCatalog *catalog = NFKModelCatalog.sharedCatalog;
	XCTAssertNotNil([catalog descriptorWithIdentifier:@"vision-feature-print"]);
	for (NSNumber *workingSet in @[ @(5053000000), @(4 * NFKBuildTestGiB) ]) {
		catalog.memoryBudget = [NFKModelMemoryBudget budgetWithWorkingSetByteCount:workingSet.integerValue];
		XCTAssertLessThan(catalog.memoryBudget.byteCount, 8 * NFKBuildTestGiB);
		NSError *error = nil;
		id<NFKInferenceBackend> backend = [catalog backendForModelIdentifier:@"vision-feature-print" options:nil error:&error];
		XCTAssertNotNil(backend, @"a %@-byte working set: %@", workingSet, error);
	}
}

- (void)testAPresetResolvesAsModelAtPreset
{
	NSError *error = nil;
	id<NFKInferenceBackend> backend = [self.catalog backendForModelIdentifier:@"build.model@fast" options:nil error:&error];
	XCTAssertNotNil(backend, @"%@", error);
	XCTAssertEqualObjects(self.builder.lastRequest.requestDefaults[@"steps"], @4);
	XCTAssertEqualObjects(self.builder.lastRequest.descriptor.identifier, @"build.model");

	[self.catalog backendForModelIdentifier:@"build.model@fast" options:@{ @"steps": @9 } error:&error];
	XCTAssertEqualObjects(self.builder.lastRequest.requestDefaults[@"steps"], @9, @"the given options win over the preset");
}

#pragma mark Locating

- (void)testTheCatalogsLocatorFindsTheFilesWhenNoLocationIsGiven
{
	NSURL *located = [NSURL fileURLWithPath:@"/tmp/located"];
	self.catalog.locator = ^NSURL * _Nullable (NFKModelDescriptor *descriptor, NFKModelVariant * _Nullable variant, NSError **error) {
		return located;
	};
	[self.catalog backendForModelIdentifier:@"build.model" options:nil error:NULL];
	XCTAssertEqualObjects(self.builder.lastRequest.location, located);
	XCTAssertEqualObjects([self.builder.lastRequest locationForModelIdentifier:@"build.model" variant:nil error:NULL], located);
	NSError *error = nil;
	XCTAssertNil([self.builder.lastRequest locationForModelIdentifier:@"ghost" variant:nil error:&error]);
	XCTAssertEqual(error.code, NFKModelCatalogErrorUnknownModel);

	self.catalog.locator = nil;
	XCTAssertNotNil(self.catalog.locator);
}

- (void)testTheHubCacheLocatorSkipsASystemModel
{
	NFKModelDescriptor *system = [NFKModelCatalog.sharedCatalog descriptorWithIdentifier:@"vision-text"];
	XCTAssertNil(NFKModelCatalog.hubCacheLocator(system, system.defaultVariant, NULL));
}

- (void)testAnAppleEngineBuildsThroughTheSharedCatalog
{
	NSError *error = nil;
	id<NFKInferenceBackend> backend = [NFKModelCatalog.sharedCatalog backendForModelIdentifier:@"vision-text" options:nil
																						 error:&error];
	XCTAssertNotNil(backend, @"%@", error);
	XCTAssertEqualObjects(backend.backendIdentifier, @"vision-text");
}

#pragma mark Defaulted backend

- (void)testTheDefaultedBackendFillsMissingParametersAndForwards
{
	NFKBuildTestBackend *inner = [[NFKBuildTestBackend alloc] init];
	NFKDefaultedBackend *backend = [[NFKDefaultedBackend alloc] initWithBackend:inner
																requestDefaults:@{ NFKParameterTemperature: @0.2,
																				   NFKParameterMaxTokens: @64 }
															   loadOptionValues:@{ @"mode": @"fast" }];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Hi" }
															   parameters:@{ NFKParameterMaxTokens: @10 }];
	XCTAssertNotNil([backend runInferenceForRequest:request error:NULL]);
	NFKInferenceRequest *seen = inner.requests.firstObject;
	XCTAssertEqualObjects(seen.parameters, (@{ NFKParameterTemperature: @0.2, NFKParameterMaxTokens: @10 }));
	XCTAssertEqualObjects(seen.inputs, request.inputs);

	XCTAssertEqualObjects(backend.backendIdentifier, @"build-test");
	XCTAssertTrue(backend.isReady);
	XCTAssertFalse([backend respondsToSelector:@selector(modelInfo)], @"the wrapped backend reports no model info");
	XCTAssertFalse([backend respondsToSelector:@selector(prepareWithError:)]);
	XCTAssertTrue([backend respondsToSelector:@selector(loadOptionValues)]);
	XCTAssertEqualObjects(backend.loadOptionValues, (@{ @"mode": @"fast" }));
	XCTAssertEqualObjects([backend valueForKey:@"isLoaded"], @YES, @"isLoaded falls back to isReady");
	XCTAssertEqualObjects([backend valueForKey:@"residentByteCount"], @0);
	XCTAssertEqualObjects([backend valueForKey:@"peakByteCount"], @0);
	XCTAssertTrue(NFKInferencePrepare(backend, NULL));
}

@end
