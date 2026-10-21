//
//  NFKModelResolverTests.m
//  InferKitTests
//
//  Resolving repositories against the in-process hub at http://hub.test: a shipped variant with no
//  request and at its measured commit, a family loader behind the opt-in, each refusal and its code, a
//  missing repository, the asynchronous form, the dictionary round trips, and the safetensors header read.
//

#import <XCTest/XCTest.h>
#import <InferKit/InferKit.h>
#import "NFKStubHubProtocol.h"

static NSString * const NFKResolverCommit = @"0123456789abcdef0123456789abcdef01234567";
static NSString * const NFKResolverMeasuredCommit = @"fedcba9876543210fedcba9876543210fedcba98";

/*! The refusal the toy provider gives, or nil to build a descriptor. */
static NSError *NFKResolverToyRefusal;
/*! The repository the toy provider was last asked about. */
static NFKModelRepository *NFKResolverToyAsked;
/*! How many times the toy provider was asked. */
static NSUInteger NFKResolverToyAskCount;

/*! A provider of one shipped toy model whose loader reads configurations of model type "toy". */
@interface NFKResolverToyProvider : NSObject <NFKModelDescriptorProvider>
@end

@implementation NFKResolverToyProvider

+ (NSArray<NFKModelDescriptor *> *)modelDescriptors
{
	NFKModelSource *source = [NFKModelSource huggingFaceSourceWithRepository:@"org/shipped"
																	revision:@"main"
																	  layout:NFKModelSourceLayoutDirectory];
	source = [[NFKModelSource alloc] initWithKind:source.kind repository:source.repository revision:source.revision
											  URL:nil layout:source.layout files:nil requiresConversion:NO
									   extensions:@{ @"measuredRevision": NFKResolverMeasuredCommit }];
	NFKModelVariant *variant = [[NFKModelVariant alloc] initWithIdentifier:@"shipped" displayName:@"Shipped" isDefault:YES
															  registryName:@"toy" source:source downloadByteCount:@1024
														 residentByteCount:@4096 peakByteCount:@4096 parameterCount:nil
																 precision:nil minimumMemoryByteCount:nil extensions:nil];
	NFKModelDescriptor *toy = [[NFKModelDescriptor alloc] initWithIdentifier:@"toy" displayName:@"Toy" family:@"toy"
																	 summary:nil engine:NFKModelEngineMLX
																registryName:@"toy" providerClassName:nil
																	   tasks:@[ NFKModelTaskTextGeneration ]
															 inputModalities:nil outputModalities:nil inputs:nil
																	 outputs:nil options:nil variants:@[ variant ]
																	 license:nil requirements:nil residencies:nil
																 memoryModel:nil presets:nil modelTypes:@[ @"toy" ]
																	 standIn:NO introducedVersion:@"0.4.0"
																  extensions:nil];
	NFKModelDescriptor *larger = [[NFKModelDescriptor alloc] initWithIdentifier:@"toy-large" displayName:nil
																		 family:@"toy" summary:nil
																		 engine:NFKModelEngineMLX
																   registryName:@"toy-large" providerClassName:nil
																		  tasks:@[ NFKModelTaskTextGeneration ]
																inputModalities:nil outputModalities:nil inputs:nil
																		outputs:nil options:nil variants:nil
																		license:nil requirements:nil residencies:nil
																	memoryModel:nil presets:nil
																	 modelTypes:@[ @"toy" ] standIn:NO
															  introducedVersion:@"0.4.0" extensions:nil];
	return @[ toy, larger ];
}

+ (nullable NFKModelDescriptor *)descriptorForRepository:(NFKModelRepository *)repository
									  matchingDescriptor:(NFKModelDescriptor *)descriptor
												   error:(NSError **)error
{
	NFKResolverToyAsked = repository;
	NFKResolverToyAskCount += 1;
	if (NFKResolverToyRefusal != nil) {
		if (error != NULL) {
			*error = NFKResolverToyRefusal;
		}
		return nil;
	}
	NFKModelSource *source = [NFKModelSource huggingFaceSourceWithRepository:repository.repository
																	revision:repository.revision
																	  layout:NFKModelSourceLayoutDirectory];
	long long bytes = [repository fileAtPath:@"model.safetensors"].byteCount;
	NFKModelVariant *variant = [[NFKModelVariant alloc] initWithIdentifier:@"resolved" displayName:nil isDefault:YES
															  registryName:descriptor.registryName source:source
														 downloadByteCount:@(bytes) residentByteCount:@(bytes)
															 peakByteCount:@(bytes) parameterCount:nil precision:nil
													minimumMemoryByteCount:nil license:repository.license
																extensions:nil];
	return [[NFKModelDescriptor alloc] initWithIdentifier:repository.repository displayName:nil family:descriptor.family
												  summary:nil engine:descriptor.engine registryName:descriptor.registryName
										providerClassName:nil tasks:descriptor.tasks inputModalities:nil
										 outputModalities:nil inputs:nil outputs:nil options:nil variants:@[ variant ]
												  license:repository.license requirements:nil residencies:nil
												  standIn:NO introducedVersion:nil extensions:nil];
}

@end

@interface NFKModelResolverTests : XCTestCase
@property (nonatomic, strong) NSURL *cacheDir;
@property (nonatomic, strong) NFKHFHub *hub;
@property (nonatomic, strong) NFKModelCatalog *catalog;
@property (nonatomic, strong) NFKModelResolver *resolver;
@end

@implementation NFKModelResolverTests

- (void)setUp
{
	[super setUp];
	[NFKStubHubProtocol reset];
	NFKResolverToyRefusal = nil;
	NFKResolverToyAsked = nil;
	NFKResolverToyAskCount = 0;
	NSString *name = [@"NFKModelResolverTests-" stringByAppendingString:NSUUID.UUID.UUIDString];
	self.cacheDir = [NSURL fileURLWithPath:[NSTemporaryDirectory() stringByAppendingPathComponent:name]];
	[NSFileManager.defaultManager createDirectoryAtURL:self.cacheDir withIntermediateDirectories:YES attributes:nil error:NULL];
	self.hub = [NFKHFHub hubWithCacheDirectoryURL:self.cacheDir];
	self.hub.endpointURL = NFKStubHubProtocol.endpointURL;
	self.hub.session = [NFKStubHubProtocol session];
	self.hub.accessToken = @"";
	self.hub.excludesCacheFromBackup = NO;
	self.catalog = [[NFKModelCatalog alloc] init];
	XCTAssertTrue([self.catalog addDescriptorsFromProviderClassName:NSStringFromClass(NFKResolverToyProvider.class)]);
	self.resolver = [[NFKModelResolver alloc] initWithCatalog:self.catalog hub:self.hub];
	self.resolver.memoryBudget = [NFKModelMemoryBudget budgetWithWorkingSetByteCount:(NSInteger)16 << 30];
}

- (void)tearDown
{
	[NSFileManager.defaultManager removeItemAtURL:self.cacheDir error:NULL];
	[NFKStubHubProtocol reset];
	NFKResolverToyRefusal = nil;
	NFKResolverToyAsked = nil;
	[super tearDown];
}

#pragma mark Fixtures

/*! A safetensors file of one float32 tensor of shape rows × 64. */
+ (NSData *)safetensorsWithRows:(NSUInteger)rows metadata:(nullable NSDictionary *)metadata
{
	NSUInteger byteCount = rows * 64 * 4;
	NSMutableDictionary *header = [@{ @"model.layers.0.mlp.weight": @{ @"dtype": @"F32", @"shape": @[ @(rows), @64 ],
																	  @"data_offsets": @[ @0, @(byteCount) ] } } mutableCopy];
	header[@"__metadata__"] = metadata;
	NSData *json = [NSJSONSerialization dataWithJSONObject:header options:0 error:NULL];
	uint64_t length = CFSwapInt64HostToLittle(json.length);
	NSMutableData *data = [NSMutableData dataWithBytes:&length length:8];
	[data appendData:json];
	[data increaseLengthBy:byteCount];
	return data;
}

+ (NSData *)JSON:(id)object
{
	return [NSJSONSerialization dataWithJSONObject:object options:0 error:NULL];
}

/*! Serves a repository at NFKResolverCommit, answering its revision information with that commit. */
- (void)serveRepo:(NSString *)repo files:(NSDictionary<NSString *, NSData *> *)files card:(nullable NSDictionary *)card
{
	[NFKStubHubProtocol serveRepo:repo revision:NFKResolverCommit files:files lfsPaths:[NSSet set]];
	NSMutableDictionary *information = [@{ @"id": repo, @"sha": NFKResolverCommit, @"gated": @NO } mutableCopy];
	information[@"cardData"] = card;
	[NFKStubHubProtocol setResponse:[NFKStubHubResponse responseWithStatus:200 body:[self.class JSON:information]]
							forPath:[NSString stringWithFormat:@"/api/models/%@/revision/main", repo]];
}

- (void)serveToyRepo:(NSString *)repo configuration:(NSDictionary *)configuration
{
	[self serveRepo:repo files:@{ @"config.json": [self.class JSON:configuration],
								  @"model.safetensors": [self.class safetensorsWithRows:4 metadata:nil] }
			   card:@{ @"license": @"apache-2.0" }];
}

- (NFKModelResolution *)resolve:(NSString *)repo revision:(nullable NSString *)revision
{
	NSError *error = nil;
	NFKModelResolution *resolution = [self.resolver resolutionForRepository:repo revision:revision error:&error];
	XCTAssertNotNil(resolution, @"%@", error);
	return resolution;
}

- (void)assertRefusal:(NFKModelResolution *)resolution code:(NFKModelResolutionError)code
{
	XCTAssertEqual(resolution.route, NFKModelRouteNone);
	XCTAssertFalse(resolution.isResolved);
	XCTAssertNil(resolution.descriptor);
	XCTAssertEqualObjects(resolution.reasons.lastObject.domain, NFKModelResolutionErrorDomain);
	XCTAssertEqual(resolution.reasons.lastObject.code, code, @"%@", resolution.reasons);
}

#pragma mark Catalog route

- (void)testAShippedRepositoryResolvesToItsVariantWithNoRequest
{
	NFKModelResolution *resolution = [self resolve:@"org/shipped" revision:nil];
	XCTAssertEqual(resolution.route, NFKModelRouteCatalog);
	XCTAssertEqualObjects(resolution.verification, NFKModelVerificationMeasured);
	XCTAssertEqualObjects(resolution.descriptor.identifier, @"toy");
	XCTAssertEqualObjects(resolution.variant.identifier, @"shipped");
	XCTAssertEqualObjects(resolution.downloadByteCount, @1024);
	XCTAssertEqual(resolution.memoryEstimate.residentByteCount, 4096);
	XCTAssertTrue(resolution.memoryEstimate.fits);
	XCTAssertEqual(NFKStubHubProtocol.requests.count, (NSUInteger)0);
}

- (void)testARepositoryNameMatchesItsShippedVariantWhateverItsCase
{
	XCTAssertEqual([self resolve:@"Org/Shipped" revision:nil].route, NFKModelRouteCatalog);
}

- (void)testAShippedRepositoryAtItsMeasuredCommitResolvesToItsVariant
{
	XCTAssertEqual([self resolve:@"org/shipped" revision:NFKResolverMeasuredCommit].route, NFKModelRouteCatalog);
	XCTAssertEqual(NFKStubHubProtocol.requests.count, (NSUInteger)0);
}

- (void)testABranchThatPointsAtTheMeasuredCommitResolvesToTheShippedVariant
{
	NSDictionary *information = @{ @"sha": NFKResolverMeasuredCommit };
	[NFKStubHubProtocol setResponse:[NFKStubHubResponse responseWithStatus:200 body:[self.class JSON:information]]
							forPath:@"/api/models/org/shipped/revision/release"];
	NFKModelResolution *resolution = [self resolve:@"org/shipped" revision:@"release"];
	XCTAssertEqual(resolution.route, NFKModelRouteCatalog);
	XCTAssertEqualObjects(resolution.revision, NFKResolverMeasuredCommit);
}

#pragma mark Family route

- (void)testAFamilyRouteIsRefusedWithoutTheOptIn
{
	[self serveToyRepo:@"org/fine-tune" configuration:@{ @"model_type": @"toy" }];
	NFKModelResolution *resolution = [self resolve:@"org/fine-tune" revision:nil];
	[self assertRefusal:resolution code:NFKModelResolutionErrorUnverifiedRouteNotAllowed];
	NSError *reason = resolution.reasons.lastObject;
	XCTAssertEqualObjects(reason.userInfo[NFKModelResolutionRouteKey], @(NFKModelRouteFamily));
	XCTAssertEqualObjects(reason.userInfo[NFKModelResolutionDescriptorKey], @"toy");
	XCTAssertEqualObjects(resolution.revision, NFKResolverCommit);
}

- (void)testAFamilyRouteResolvesWithTheOptIn
{
	self.resolver.allowsUnverifiedRoutes = YES;
	[self serveToyRepo:@"org/fine-tune" configuration:@{ @"model_type": @"toy" }];
	NFKModelResolution *resolution = [self resolve:@"org/fine-tune" revision:nil];
	XCTAssertEqual(resolution.route, NFKModelRouteFamily);
	XCTAssertEqualObjects(resolution.verification, NFKModelVerificationConfigCompatible);
	XCTAssertEqualObjects(resolution.revision, NFKResolverCommit);
	XCTAssertEqualObjects(resolution.descriptor.identifier, @"org/fine-tune");
	XCTAssertEqualObjects(resolution.variant.source.revision, NFKResolverCommit);
	XCTAssertEqualObjects(resolution.descriptor.license.identifier, @"apache-2.0");
	XCTAssertNotNil(resolution.memoryEstimate);
	XCTAssertEqualObjects(resolution.downloadByteCount, @([self.class safetensorsWithRows:4 metadata:nil].length));
	NSDictionary *expected = @{ @"verification": NFKModelVerificationConfigCompatible, @"repository": @"org/fine-tune",
								@"revision": NFKResolverCommit, @"basedOn": @"toy" };
	XCTAssertEqualObjects(resolution.descriptor.extensions[NFKModelResolutionExtensionKey], expected);
	XCTAssertEqualObjects(NFKResolverToyAsked.modelType, @"toy");
	XCTAssertEqualObjects(NFKResolverToyAsked.configuration[@"model_type"], @"toy");
	XCTAssertNotNil([NFKResolverToyAsked fileAtPath:@"config.json"]);
}

- (void)testAResolvedDescriptorRoundTripsThroughItsDictionaryIntoACatalog
{
	self.resolver.allowsUnverifiedRoutes = YES;
	[self serveToyRepo:@"org/fine-tune" configuration:@{ @"model_type": @"toy" }];
	NFKModelDescriptor *descriptor = [self resolve:@"org/fine-tune" revision:nil].descriptor;
	NSData *saved = [NSJSONSerialization dataWithJSONObject:descriptor.dictionaryRepresentation options:0 error:NULL];
	XCTAssertNotNil(saved);
	NSError *error = nil;
	NFKModelDescriptor *restored = [NFKModelDescriptor descriptorWithDictionary:
									[NSJSONSerialization JSONObjectWithData:saved options:0 error:NULL] error:&error];
	XCTAssertNotNil(restored, @"%@", error);
	XCTAssertEqualObjects(restored, descriptor);
	NFKModelCatalog *catalog = [[NFKModelCatalog alloc] init];
	[catalog addDescriptor:restored];
	XCTAssertEqualObjects([catalog descriptorWithIdentifier:@"org/fine-tune"].extensions[NFKModelResolutionExtensionKey][@"basedOn"],
						  @"toy");
}

- (void)testALoaderRefusalCarriesItsCodeAndTheDescriptor
{
	self.resolver.allowsUnverifiedRoutes = YES;
	NFKResolverToyRefusal = [NSError errorWithDomain:NFKModelResolutionErrorDomain
												code:NFKModelResolutionErrorUnsupportedTokenizer
											userInfo:@{ NSLocalizedDescriptionKey: @"the tokenizer is not read" }];
	[self serveToyRepo:@"org/fine-tune" configuration:@{ @"model_type": @"toy" }];
	NFKModelResolution *resolution = [self resolve:@"org/fine-tune" revision:nil];
	[self assertRefusal:resolution code:NFKModelResolutionErrorUnsupportedTokenizer];
	XCTAssertEqualObjects(resolution.reasons.lastObject.userInfo[NFKModelResolutionDescriptorKey], @"toy");
	XCTAssertEqualObjects(resolution.reasons.lastObject.localizedDescription, @"the tokenizer is not read");
}

- (void)testALoaderErrorOutsideTheDomainIsAnUnsupportedConfiguration
{
	self.resolver.allowsUnverifiedRoutes = YES;
	NFKResolverToyRefusal = [NSError errorWithDomain:NSCocoaErrorDomain code:1 userInfo:nil];
	[self serveToyRepo:@"org/fine-tune" configuration:@{ @"model_type": @"toy" }];
	NFKModelResolution *resolution = [self resolve:@"org/fine-tune" revision:nil];
	[self assertRefusal:resolution code:NFKModelResolutionErrorUnsupportedConfiguration];
	NSError *underlying = resolution.reasons.lastObject.userInfo[NSUnderlyingErrorKey];
	XCTAssertEqualObjects(underlying.domain, NSCocoaErrorDomain);
}

- (void)testOneFamilyOfOneProviderIsJudgedOnce
{
	self.resolver.allowsUnverifiedRoutes = YES;
	NFKResolverToyRefusal = [NSError errorWithDomain:NFKModelResolutionErrorDomain
												code:NFKModelResolutionErrorUnsupportedQuantization userInfo:nil];
	[self serveToyRepo:@"org/fine-tune" configuration:@{ @"model_type": @"toy" }];
	NFKModelResolution *resolution = [self resolve:@"org/fine-tune" revision:nil];
	[self assertRefusal:resolution code:NFKModelResolutionErrorUnsupportedQuantization];
	XCTAssertEqual(NFKResolverToyAskCount, (NSUInteger)1);
	XCTAssertEqual(resolution.reasons.count, (NSUInteger)1);
}

- (void)testAHubErrorFromTheLoaderFailsTheCall
{
	self.resolver.allowsUnverifiedRoutes = YES;
	NFKResolverToyRefusal = [NSError errorWithDomain:NFKDownloadErrorDomain code:NFKDownloadErrorNotFound userInfo:nil];
	[self serveToyRepo:@"org/fine-tune" configuration:@{ @"model_type": @"toy" }];
	NSError *error = nil;
	XCTAssertNil([self.resolver resolutionForRepository:@"org/fine-tune" revision:nil error:&error]);
	XCTAssertEqualObjects(error.domain, NFKDownloadErrorDomain);
	XCTAssertEqual(error.code, NFKDownloadErrorNotFound);
}

- (void)testATransportErrorFromTheLoaderFailsTheCall
{
	self.resolver.allowsUnverifiedRoutes = YES;
	NFKResolverToyRefusal = [NSError errorWithDomain:NSURLErrorDomain code:NSURLErrorNotConnectedToInternet userInfo:nil];
	[self serveToyRepo:@"org/fine-tune" configuration:@{ @"model_type": @"toy" }];
	NSError *error = nil;
	XCTAssertNil([self.resolver resolutionForRepository:@"org/fine-tune" revision:nil error:&error]);
	XCTAssertEqualObjects(error.domain, NSURLErrorDomain);
}

- (void)testARefusedReadFromTheLoaderIsGated
{
	self.resolver.allowsUnverifiedRoutes = YES;
	NFKResolverToyRefusal = [NSError errorWithDomain:NFKDownloadErrorDomain code:NFKDownloadErrorAuthenticationRequired
											userInfo:nil];
	[self serveToyRepo:@"org/fine-tune" configuration:@{ @"model_type": @"toy" }];
	NFKModelResolution *resolution = [self resolve:@"org/fine-tune" revision:nil];
	[self assertRefusal:resolution code:NFKModelResolutionErrorGated];
	XCTAssertEqualObjects(resolution.reasons.lastObject.userInfo[NFKModelResolutionRouteKey], @(NFKModelRouteFamily));
	XCTAssertEqualObjects(resolution.reasons.lastObject.userInfo[NFKModelResolutionDescriptorKey], @"toy");
}

#pragma mark Refusals

- (void)testWithheldMetadataIsAGatedRefusalThatNamesAMissingRepository
{
	[NFKStubHubProtocol setResponse:[NFKStubHubResponse responseWithStatus:401 body:nil]
							forPath:@"/api/models/org/private/revision/main"];
	NFKModelResolution *resolution = [self resolve:@"org/private" revision:nil];
	[self assertRefusal:resolution code:NFKModelResolutionErrorGated];
	XCTAssertTrue([resolution.reasons.lastObject.localizedDescription containsString:@"does not exist"]);
	NSError *underlying = resolution.reasons.lastObject.userInfo[NSUnderlyingErrorKey];
	XCTAssertEqual(underlying.code, NFKDownloadErrorAuthenticationRequired);
}

- (void)testAnUnknownModelTypeIsAnUnsupportedArchitecture
{
	[self serveToyRepo:@"org/other" configuration:@{ @"model_type": @"other" }];
	[self assertRefusal:[self resolve:@"org/other" revision:nil] code:NFKModelResolutionErrorUnsupportedArchitecture];
}

- (void)testAConfigurationWithoutAModelTypeIsAnUnsupportedArchitecture
{
	[self serveToyRepo:@"org/other" configuration:@{ @"hidden_size": @8 }];
	[self assertRefusal:[self resolve:@"org/other" revision:nil] code:NFKModelResolutionErrorUnsupportedArchitecture];
}

- (void)testShippedCodeIsRefusedBeforeAnyLoaderIsAsked
{
	self.resolver.allowsUnverifiedRoutes = YES;
	[self serveToyRepo:@"org/custom" configuration:@{ @"model_type": @"toy",
													  @"auto_map": @{ @"AutoModel": @"modeling.Model" } }];
	[self assertRefusal:[self resolve:@"org/custom" revision:nil] code:NFKModelResolutionErrorCustomCode];
	XCTAssertNil(NFKResolverToyAsked);
}

- (void)testPickledWeightsAreAnUnsupportedWeightFormat
{
	[self serveRepo:@"org/pickled" files:@{ @"config.json": [self.class JSON:@{ @"model_type": @"toy" }],
											@"pytorch_model.bin": [NSData dataWithBytes:"pk" length:2] } card:nil];
	NFKModelResolution *resolution = [self resolve:@"org/pickled" revision:nil];
	[self assertRefusal:resolution code:NFKModelResolutionErrorUnsupportedWeightFormat];
	XCTAssertTrue([resolution.reasons.lastObject.localizedDescription containsString:@"pytorch_model.bin"]);
}

- (void)testARepositoryWithoutAConfigurationIsRefused
{
	[self serveRepo:@"org/bare" files:@{ @"model.safetensors": [self.class safetensorsWithRows:1 metadata:nil] } card:nil];
	[self assertRefusal:[self resolve:@"org/bare" revision:nil] code:NFKModelResolutionErrorNoConfiguration];
}

- (void)testAGatedListingIsAGatedRefusalWithTheHubsError
{
	[NFKStubHubProtocol setResponse:[NFKStubHubResponse responseWithStatus:200 body:[self.class JSON:@{ @"sha": NFKResolverCommit }]]
							forPath:@"/api/models/org/gated/revision/main"];
	[NFKStubHubProtocol setResponse:[NFKStubHubResponse responseWithStatus:401 body:nil]
							forPath:[NSString stringWithFormat:@"/api/models/org/gated/tree/%@", NFKResolverCommit]];
	NFKModelResolution *resolution = [self resolve:@"org/gated" revision:nil];
	[self assertRefusal:resolution code:NFKModelResolutionErrorGated];
	NSError *underlying = resolution.reasons.lastObject.userInfo[NSUnderlyingErrorKey];
	XCTAssertEqualObjects(underlying.domain, NFKDownloadErrorDomain);
	XCTAssertEqual(underlying.code, NFKDownloadErrorAuthenticationRequired);
}

- (void)testAMissingRepositoryFailsWithTheHubsError
{
	NSError *error = nil;
	XCTAssertNil([self.resolver resolutionForRepository:@"org/missing" revision:nil error:&error]);
	XCTAssertEqualObjects(error.domain, NFKDownloadErrorDomain);
	XCTAssertEqual(error.code, NFKDownloadErrorNotFound);
}

- (void)testANameThatIsNotOwnerSlashNameFails
{
	NSError *error = nil;
	XCTAssertNil([self.resolver resolutionForRepository:@"no-owner" revision:nil error:&error]);
	XCTAssertEqualObjects(error.domain, NFKDownloadErrorDomain);
}

#pragma mark Forms

- (void)testTheAsynchronousFormAnswersOffTheCallingQueue
{
	XCTestExpectation *answered = [self expectationWithDescription:@"answered"];
	[self.resolver resolveRepository:@"org/shipped" revision:nil
				   completionHandler:^(NFKModelResolution *resolution, NSError *error) {
		XCTAssertFalse(NSThread.isMainThread);
		XCTAssertEqual(resolution.route, NFKModelRouteCatalog);
		XCTAssertNil(error);
		[answered fulfill];
	}];
	[self waitForExpectations:@[ answered ] timeout:10];
}

- (void)testAResolutionRoundTripsThroughItsDictionary
{
	self.resolver.allowsUnverifiedRoutes = YES;
	[self serveToyRepo:@"org/fine-tune" configuration:@{ @"model_type": @"toy" }];
	[self serveToyRepo:@"org/other" configuration:@{ @"model_type": @"other" }];
	for (NFKModelResolution *resolution in @[ [self resolve:@"org/fine-tune" revision:nil],
											  [self resolve:@"org/other" revision:nil] ]) {
		NSDictionary *dictionary = resolution.dictionaryRepresentation;
		XCTAssertTrue([NSJSONSerialization isValidJSONObject:dictionary]);
		NSError *error = nil;
		NFKModelResolution *restored = [NFKModelResolution resolutionWithDictionary:dictionary error:&error];
		XCTAssertNotNil(restored, @"%@", error);
		XCTAssertEqualObjects(restored.dictionaryRepresentation, dictionary);
		XCTAssertEqual(restored.route, resolution.route);
		XCTAssertEqual(restored.reasons.count, resolution.reasons.count);
	}
}

- (void)testAResolutionDictionaryWithoutARepositoryIsRefused
{
	NSError *error = nil;
	XCTAssertNil([NFKModelResolution resolutionWithDictionary:@{ @"route": @"none" } error:&error]);
	XCTAssertNotNil(error);
}

- (void)testRouteNames
{
	XCTAssertEqualObjects(NFKModelRouteName(NFKModelRouteNone), @"none");
	XCTAssertEqualObjects(NFKModelRouteName(NFKModelRouteCatalog), @"catalog");
	XCTAssertEqualObjects(NFKModelRouteName(NFKModelRouteFamily), @"family");
}

#pragma mark Safetensors headers

- (void)testASafetensorsHeaderIsReadByTwoRangeRequests
{
	NSData *file = [self.class safetensorsWithRows:1024 metadata:@{ @"format": @"pt" }];
	[self serveRepo:@"org/weights" files:@{ @"model.safetensors": file } card:nil];
	NSError *error = nil;
	NSDictionary *header = [self.hub safetensorsHeaderOfRepo:@"org/weights" revision:NFKResolverCommit
														path:@"model.safetensors" error:&error];
	XCTAssertNotNil(header, @"%@", error);
	XCTAssertEqualObjects(header[@"__metadata__"], @{ @"format": @"pt" });
	XCTAssertEqualObjects(header[@"model.layers.0.mlp.weight"][@"shape"], (@[ @1024, @64 ]));
	NSString *path = [NSString stringWithFormat:@"/org/weights/resolve/%@/model.safetensors", NFKResolverCommit];
	NSArray<NSURLRequest *> *requests = [NFKStubHubProtocol requestsForPath:path];
	XCTAssertEqual(requests.count, (NSUInteger)2);
	XCTAssertEqualObjects([requests.firstObject valueForHTTPHeaderField:@"Range"], @"bytes=0-7");
	XCTAssertFalse([self.hub isCachedRepo:@"org/weights" revision:NFKResolverCommit path:@"model.safetensors"]);
}

- (void)testACachedSafetensorsHeaderIsReadFromDisk
{
	NSData *file = [self.class safetensorsWithRows:2 metadata:nil];
	[self serveRepo:@"org/weights" files:@{ @"model.safetensors": file } card:nil];
	NSError *error = nil;
	XCTAssertNotNil([self.hub downloadRepo:@"org/weights" revision:NFKResolverCommit path:@"model.safetensors"
									sha256:nil error:&error], @"%@", error);
	[NFKStubHubProtocol reset];
	NSDictionary *header = [self.hub safetensorsHeaderOfRepo:@"org/weights" revision:NFKResolverCommit
														path:@"model.safetensors" error:&error];
	XCTAssertEqualObjects(header[@"model.layers.0.mlp.weight"][@"dtype"], @"F32", @"%@", error);
	XCTAssertEqual(NFKStubHubProtocol.requests.count, (NSUInteger)0);
}

- (void)testBytesThatAreNotASafetensorsHeaderAreRefused
{
	NSMutableData *file = [NSMutableData dataWithLength:16];
	((uint8_t *)file.mutableBytes)[7] = 0xff;
	[self serveRepo:@"org/weights" files:@{ @"model.safetensors": file } card:nil];
	NSError *error = nil;
	XCTAssertNil([self.hub safetensorsHeaderOfRepo:@"org/weights" revision:NFKResolverCommit
											  path:@"model.safetensors" error:&error]);
	XCTAssertEqualObjects(error.domain, NFKDownloadErrorDomain);
	XCTAssertEqual(error.code, NFKDownloadErrorUnsupportedSource);
}

- (void)testAMissingSafetensorsFileIsNotFound
{
	[self serveRepo:@"org/weights" files:@{ @"config.json": [self.class JSON:@{}] } card:nil];
	NSError *error = nil;
	XCTAssertNil([self.hub safetensorsHeaderOfRepo:@"org/weights" revision:NFKResolverCommit
											  path:@"model.safetensors" error:&error]);
	XCTAssertEqual(error.code, NFKDownloadErrorNotFound);
}

@end
