//
//  NFKHFHubModelFolderTests.m
//  InferKitTests
//
//  Model folders on temporary roots: REVISION read in either line order, a folder naming no
//  repository skipped, the repository compared without regard to case, pinned revisions, required and
//  optional files, the location of each layout, the order among roots and folders, the status and the
//  locators that read the folders, and removals that leave a folder's files in place.
//

#import <XCTest/XCTest.h>
#import <InferKit/InferKit.h>

static NSString * const NFKFolderTestRevision = @"1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0";
static NSString * const NFKFolderTestOtherRevision = @"f21265c1e2710b3bd2386596943f0007f55f802e";

@interface NFKHFHubModelFolderTests : XCTestCase
@property (nonatomic, copy) NSArray<NSURL *> *savedDefaultModelFolderURLs;
@property (nonatomic, strong) NSURL *root;
@property (nonatomic, strong) NSURL *cacheDir;
@property (nonatomic, strong) NFKHFHub *hub;
@end

@implementation NFKHFHubModelFolderTests

- (void)setUp
{
	[super setUp];
	self.savedDefaultModelFolderURLs = NFKHFHub.defaultModelFolderURLs;
	self.root = [self temporaryDirectory];
	self.cacheDir = [self temporaryDirectory];
	self.hub = [NFKHFHub hubWithCacheDirectoryURL:self.cacheDir];
	self.hub.excludesCacheFromBackup = NO;
	self.hub.modelCatalog = [[NFKModelCatalog alloc] init];
	self.hub.modelFolderURLs = @[ self.root ];
}

- (void)tearDown
{
	NFKHFHub.defaultModelFolderURLs = self.savedDefaultModelFolderURLs;
	[super tearDown];
}

#pragma mark Fixtures

- (NSURL *)temporaryDirectory
{
	NSString *name = [@"NFKHFHubModelFolderTests-" stringByAppendingString:NSUUID.UUID.UUIDString];
	NSURL *url = [NSURL fileURLWithPath:[NSTemporaryDirectory() stringByAppendingPathComponent:name] isDirectory:YES];
	XCTAssertTrue([NSFileManager.defaultManager createDirectoryAtURL:url withIntermediateDirectories:YES attributes:nil
															   error:NULL]);
	[self addTeardownBlock:^{
		[NSFileManager.defaultManager removeItemAtURL:url error:NULL];
	}];
	return url;
}

/*! Writes a release folder under root: its REVISION text, when given, and each path with small contents. */
- (NSURL *)folderNamed:(NSString *)name inRoot:(NSURL *)root revisionText:(nullable NSString *)revisionText
				 paths:(NSArray<NSString *> *)paths
{
	NSURL *folder = [root URLByAppendingPathComponent:name isDirectory:YES];
	XCTAssertTrue([NSFileManager.defaultManager createDirectoryAtURL:folder withIntermediateDirectories:YES attributes:nil
															   error:NULL]);
	if (revisionText != nil) {
		XCTAssertTrue([revisionText writeToURL:[folder URLByAppendingPathComponent:@"REVISION"] atomically:YES
									  encoding:NSUTF8StringEncoding error:NULL]);
	}
	for (NSString *path in paths) {
		NSURL *url = [folder URLByAppendingPathComponent:path isDirectory:NO];
		XCTAssertTrue([NSFileManager.defaultManager createDirectoryAtURL:url.URLByDeletingLastPathComponent
											 withIntermediateDirectories:YES attributes:nil error:NULL]);
		NSString *contents = [path hasSuffix:@".json"] ? @"{}" : [@"weights of " stringByAppendingString:path];
		XCTAssertTrue([contents writeToURL:url atomically:YES encoding:NSUTF8StringEncoding error:NULL]);
	}
	return folder;
}

- (NFKModelFile *)file:(NSString *)path optional:(BOOL)optional
{
	NSString *role = [path hasSuffix:@".safetensors"] ? NFKModelFileRoleWeights : NFKModelFileRoleConfig;
	return [[NFKModelFile alloc] initWithPath:path byteCount:nil sha256:nil role:role optional:optional extensions:nil];
}

- (NFKModelSource *)sourceOfRepository:(NSString *)repository
							  revision:(nullable NSString *)revision
								layout:(NSString *)layout
								 files:(NSArray<NFKModelFile *> *)files
{
	return [[NFKModelSource alloc] initWithKind:NFKModelSourceKindHuggingFace repository:repository revision:revision URL:nil
										 layout:layout files:files requiresConversion:NO extensions:nil];
}

/*! A directory source of org/model requiring config.json and model.safetensors. */
- (NFKModelSource *)modelSourceAtRevision:(nullable NSString *)revision
{
	return [self sourceOfRepository:@"org/model" revision:revision layout:NFKModelSourceLayoutDirectory
							  files:@[ [self file:@"config.json" optional:NO], [self file:@"model.safetensors" optional:NO] ]];
}

- (NFKModelDescriptor *)descriptorWithSource:(NFKModelSource *)source
{
	NFKModelVariant *variant = [[NFKModelVariant alloc] initWithIdentifier:@"default" displayName:nil isDefault:YES
															  registryName:nil source:source downloadByteCount:nil
														 residentByteCount:nil peakByteCount:nil parameterCount:nil
																 precision:nil minimumMemoryByteCount:nil license:nil
																extensions:nil];
	return [[NFKModelDescriptor alloc] initWithIdentifier:@"folder.model" displayName:nil family:@"test" summary:nil
												   engine:@"folder-test-engine" registryName:nil providerClassName:nil
													tasks:nil inputModalities:nil outputModalities:nil inputs:nil
												  outputs:nil options:nil variants:@[ variant ] license:nil requirements:nil
											  residencies:nil standIn:NO introducedVersion:nil extensions:nil];
}

- (void)assertURL:(nullable NSURL *)url isAtPath:(NSURL *)expected
{
	XCTAssertNotNil(url);
	XCTAssertEqualObjects(url.URLByStandardizingPath.path, expected.URLByStandardizingPath.path);
}

#pragma mark REVISION

- (void)testTheRevisionFileIsReadInEitherLineOrder
{
	NSArray<NSString *> *paths = @[ @"config.json", @"model.safetensors" ];
	NSString *hashFirst = [NSString stringWithFormat:@"%@\norg/model\n", NFKFolderTestRevision];
	NSURL *folder = [self folderNamed:@"model-hash-first" inRoot:self.root revisionText:hashFirst paths:paths];
	[self assertURL:[self.hub modelFolderLocationForModelSource:[self modelSourceAtRevision:NFKFolderTestRevision]]
		   isAtPath:folder];

	NSString *repositoryFirst = [NSString stringWithFormat:@"  org/other \r\n%@\n", NFKFolderTestOtherRevision];
	NSURL *other = [self folderNamed:@"other-repository-first" inRoot:self.root revisionText:repositoryFirst paths:paths];
	NFKModelSource *source = [self sourceOfRepository:@"org/other" revision:NFKFolderTestOtherRevision
											   layout:NFKModelSourceLayoutDirectory
												files:@[ [self file:@"config.json" optional:NO] ]];
	[self assertURL:[self.hub modelFolderLocationForModelSource:source] isAtPath:other];
}

- (void)testAFolderWhoseRevisionNamesNoRepositoryIsNotMatched
{
	NSString *revisionOnly = [NFKFolderTestRevision stringByAppendingString:@"\n"];
	[self folderNamed:@"model-transformer" inRoot:self.root revisionText:revisionOnly
				paths:@[ @"config.json", @"model.safetensors" ]];
	XCTAssertNil([self.hub modelFolderLocationForModelSource:[self modelSourceAtRevision:nil]]);
	XCTAssertNil([self.hub modelFolderLocationForModelSource:[self modelSourceAtRevision:NFKFolderTestRevision]]);
}

- (void)testHiddenFoldersAndFoldersWithoutARevisionFileAreSkipped
{
	NSArray<NSString *> *paths = @[ @"config.json", @"model.safetensors" ];
	[self folderNamed:@".hidden" inRoot:self.root revisionText:@"org/model\n" paths:paths];
	[self folderNamed:@"unkeyed" inRoot:self.root revisionText:nil paths:paths];
	XCTAssertNil([self.hub modelFolderLocationForModelSource:[self modelSourceAtRevision:nil]]);

	self.hub.modelFolderURLs = @[ [self.root URLByAppendingPathComponent:@"absent-volume" isDirectory:YES], self.root ];
	NSURL *keyed = [self folderNamed:@"keyed" inRoot:self.root revisionText:@"org/model\n" paths:paths];
	[self assertURL:[self.hub modelFolderLocationForModelSource:[self modelSourceAtRevision:nil]] isAtPath:keyed];
}

- (void)testTheRepositoryIsComparedWithoutRegardToCase
{
	NSURL *folder = [self folderNamed:@"Model" inRoot:self.root revisionText:@"Org/Model\n"
								paths:@[ @"config.json", @"model.safetensors" ]];
	[self assertURL:[self.hub modelFolderLocationForModelSource:[self modelSourceAtRevision:nil]] isAtPath:folder];
}

- (void)testAPinnedRevisionMatchesOnlyAFolderAtThatRevision
{
	NSArray<NSString *> *paths = @[ @"config.json", @"model.safetensors" ];
	NSString *text = [NSString stringWithFormat:@"org/model\n%@\n", NFKFolderTestRevision.uppercaseString];
	NSURL *folder = [self folderNamed:@"model" inRoot:self.root revisionText:text paths:paths];
	[self assertURL:[self.hub modelFolderLocationForModelSource:[self modelSourceAtRevision:NFKFolderTestRevision]]
		   isAtPath:folder];
	XCTAssertNil([self.hub modelFolderLocationForModelSource:[self modelSourceAtRevision:NFKFolderTestOtherRevision]]);
	[self assertURL:[self.hub modelFolderLocationForModelSource:[self modelSourceAtRevision:nil]] isAtPath:folder];
	[self assertURL:[self.hub modelFolderLocationForModelSource:[self modelSourceAtRevision:@"main"]] isAtPath:folder];

	NSURL *unpinnedRoot = [self temporaryDirectory];
	NSURL *unpinned = [self folderNamed:@"model" inRoot:unpinnedRoot revisionText:@"org/model\n" paths:paths];
	self.hub.modelFolderURLs = @[ unpinnedRoot ];
	XCTAssertNil([self.hub modelFolderLocationForModelSource:[self modelSourceAtRevision:NFKFolderTestRevision]],
				 @"a folder naming no revision does not match a pinned source");
	[self assertURL:[self.hub modelFolderLocationForModelSource:[self modelSourceAtRevision:nil]] isAtPath:unpinned];
}

#pragma mark Files and locations

- (void)testAMissingRequiredFileIsNotAMatchWhileAMissingOptionalFileIs
{
	NFKModelSource *source = [self sourceOfRepository:@"org/model" revision:nil layout:NFKModelSourceLayoutDirectory
												files:@[ [self file:@"config.json" optional:NO],
														 [self file:@"model.safetensors" optional:NO],
														 [self file:@"generation_config.json" optional:YES] ]];
	NSURL *partial = [self folderNamed:@"model-config-only" inRoot:self.root revisionText:@"org/model\n"
								 paths:@[ @"config.json", @"generation_config.json" ]];
	XCTAssertNil([self.hub modelFolderLocationForModelSource:source]);

	XCTAssertTrue([@"weights" writeToURL:[partial URLByAppendingPathComponent:@"model.safetensors"] atomically:YES
								encoding:NSUTF8StringEncoding error:NULL]);
	XCTAssertTrue([NSFileManager.defaultManager removeItemAtURL:[partial URLByAppendingPathComponent:@"generation_config.json"]
														  error:NULL]);
	[self assertURL:[self.hub modelFolderLocationForModelSource:source] isAtPath:partial];
	XCTAssertEqual([self.hub downloadStatusForModelSource:source].presence, NFKDownloadPresenceComplete);
}

- (void)testAShardIndexInTheFolderRequiresItsShards
{
	NFKModelSource *source = [self sourceOfRepository:@"org/model" revision:nil layout:NFKModelSourceLayoutDirectory
												files:@[ [self file:@"config.json" optional:NO],
														 [self file:@"model.safetensors.index.json" optional:YES] ]];
	NSURL *folder = [self folderNamed:@"model" inRoot:self.root revisionText:@"org/model\n"
								paths:@[ @"config.json", @"model-00001-of-00002.safetensors" ]];
	NSString *index = @"{\"weight_map\": {\"a\": \"model-00001-of-00002.safetensors\", \"b\": \"model-00002-of-00002.safetensors\"}}";
	XCTAssertTrue([index writeToURL:[folder URLByAppendingPathComponent:@"model.safetensors.index.json"] atomically:YES
						   encoding:NSUTF8StringEncoding error:NULL]);
	XCTAssertNil([self.hub modelFolderLocationForModelSource:source]);

	XCTAssertTrue([@"weights" writeToURL:[folder URLByAppendingPathComponent:@"model-00002-of-00002.safetensors"]
							  atomically:YES encoding:NSUTF8StringEncoding error:NULL]);
	[self assertURL:[self.hub modelFolderLocationForModelSource:source] isAtPath:folder];
}

- (void)testTheLocationIsTheFolderForADirectoryAndTheWeightsFileForAFile
{
	NSURL *folder = [self folderNamed:@"model" inRoot:self.root revisionText:@"org/model\n"
								paths:@[ @"config.json", @"weights/model.safetensors" ]];
	NSArray<NFKModelFile *> *files = @[ [self file:@"config.json" optional:NO],
										[self file:@"weights/model.safetensors" optional:NO] ];
	NFKModelSource *directory = [self sourceOfRepository:@"org/model" revision:nil layout:NFKModelSourceLayoutDirectory
												   files:files];
	[self assertURL:[self.hub modelFolderLocationForModelSource:directory] isAtPath:folder];

	NFKModelSource *file = [self sourceOfRepository:@"org/model" revision:nil layout:NFKModelSourceLayoutFile files:files];
	[self assertURL:[self.hub modelFolderLocationForModelSource:file]
		   isAtPath:[folder URLByAppendingPathComponent:@"weights/model.safetensors"]];
}

- (void)testTheFirstRootAndThenTheFirstFolderNameWins
{
	NSArray<NSString *> *paths = @[ @"config.json", @"model.safetensors" ];
	NSURL *secondRoot = [self temporaryDirectory];
	[self folderNamed:@"a-model" inRoot:secondRoot revisionText:@"org/model\n" paths:paths];
	NSURL *firstRootB = [self folderNamed:@"b-model" inRoot:self.root revisionText:@"org/model\n" paths:paths];
	self.hub.modelFolderURLs = @[ self.root, secondRoot ];
	[self assertURL:[self.hub modelFolderLocationForModelSource:[self modelSourceAtRevision:nil]] isAtPath:firstRootB];

	NSURL *firstRootA = [self folderNamed:@"a-model" inRoot:self.root revisionText:@"org/model\n" paths:paths];
	[self assertURL:[self.hub modelFolderLocationForModelSource:[self modelSourceAtRevision:nil]] isAtPath:firstRootA];
}

- (void)testOnlyAHuggingFaceSourceMatchesAFolder
{
	[self folderNamed:@"base" inRoot:self.root revisionText:@"org/model\n" paths:@[ @"base.pt" ]];
	NFKModelFile *file = [[NFKModelFile alloc] initWithPath:@"base.pt" byteCount:nil sha256:nil role:NFKModelFileRoleWeights
												   optional:NO extensions:nil];
	NFKModelSource *source = [[NFKModelSource alloc] initWithKind:NFKModelSourceKindURL repository:@"org/model" revision:nil
															  URL:[NSURL URLWithString:@"https://weights.test/org/model/base.pt"]
														   layout:NFKModelSourceLayoutFile files:@[ file ]
											   requiresConversion:NO extensions:nil];
	XCTAssertNil([self.hub modelFolderLocationForModelSource:source]);
	XCTAssertNotEqual([self.hub downloadStatusForModelSource:source].presence, NFKDownloadPresenceComplete);
}

#pragma mark Status

- (void)testTheStatusOfAMatchingFolderIsCompleteInsideTheFolder
{
	NSURL *folder = [self folderNamed:@"model" inRoot:self.root revisionText:@"org/model\n"
								paths:@[ @"config.json", @"model.safetensors" ]];
	NFKDownloadStatus *status = [self.hub downloadStatusForModelSource:[self modelSourceAtRevision:nil]];
	XCTAssertEqual(status.presence, NFKDownloadPresenceComplete);
	[self assertURL:status.localURL isAtPath:folder];
	XCTAssertEqual(status.missingPaths.count, 0u);
	NSArray<NSString *> *expectedPaths = @[ @"config.json", @"model.safetensors" ];
	XCTAssertEqualObjects(status.presentPaths, expectedPaths);
	long long expectedBytes = 0;
	for (NSString *path in expectedPaths) {
		NSNumber *size = nil;
		[[folder URLByAppendingPathComponent:path] getResourceValue:&size forKey:NSURLTotalFileAllocatedSizeKey error:NULL];
		expectedBytes += size.longLongValue;
	}
	XCTAssertGreaterThan(expectedBytes, 0);
	XCTAssertEqual(status.byteCount, expectedBytes);
}

- (void)testTheStatusOfASourceNoFolderMatchesIsTheCaches
{
	[self folderNamed:@"model" inRoot:self.root revisionText:@"org/model\n" paths:@[ @"config.json" ]];
	NSURL *cachedConfig = [self.hub localURLForRepo:@"org/model" revision:nil path:@"config.json"];
	XCTAssertTrue([NSFileManager.defaultManager createDirectoryAtURL:cachedConfig.URLByDeletingLastPathComponent
										 withIntermediateDirectories:YES attributes:nil error:NULL]);
	XCTAssertTrue([@"{}" writeToURL:cachedConfig atomically:YES encoding:NSUTF8StringEncoding error:NULL]);

	NFKModelSource *source = [self modelSourceAtRevision:nil];
	NFKDownloadStatus *status = [self.hub downloadStatusForModelSource:source];
	NFKHFHub *cacheOnly = [NFKHFHub hubWithCacheDirectoryURL:self.cacheDir];
	cacheOnly.modelFolderURLs = @[];
	NFKDownloadStatus *cacheStatus = [cacheOnly downloadStatusForModelSource:source];
	XCTAssertEqual(status.presence, NFKDownloadPresencePartial);
	XCTAssertEqual(status.presence, cacheStatus.presence);
	XCTAssertEqual(status.byteCount, cacheStatus.byteCount);
	XCTAssertEqualObjects(status.presentPaths, cacheStatus.presentPaths);
	XCTAssertEqualObjects(status.missingPaths, cacheStatus.missingPaths);
	XCTAssertEqualObjects(status.localURL, cacheStatus.localURL);
}

- (void)testAFileFromAnotherRepositoryIsReadInTheCache
{
	NFKModelFile *encoder = [[NFKModelFile alloc] initWithPath:@"text_encoder/model.safetensors" byteCount:nil sha256:nil
														  role:NFKModelFileRoleWeights optional:NO repository:@"org/encoder"
													  revision:nil URL:nil extensions:nil];
	NFKModelSource *source = [self sourceOfRepository:@"org/model" revision:nil layout:NFKModelSourceLayoutDirectory
												files:@[ [self file:@"config.json" optional:NO], encoder ]];
	NSURL *folder = [self folderNamed:@"model" inRoot:self.root revisionText:@"org/model\n" paths:@[ @"config.json" ]];
	[self folderNamed:@"encoder" inRoot:self.root revisionText:@"org/encoder\n" paths:@[ @"text_encoder/model.safetensors" ]];
	[self assertURL:[self.hub modelFolderLocationForModelSource:source] isAtPath:folder];
	XCTAssertNotEqual([self.hub downloadStatusForModelSource:source].presence, NFKDownloadPresenceComplete,
					  @"the encoder is read in the cache, which lacks it");
	NFKModelDescriptor *descriptor = [self descriptorWithSource:source];
	XCTAssertNil([NFKModelCatalog locatorForHub:self.hub](descriptor, descriptor.defaultVariant, NULL));

	NSURL *cachedEncoder = [self.hub localURLForRepo:@"org/encoder" revision:nil path:@"text_encoder/model.safetensors"];
	XCTAssertTrue([NSFileManager.defaultManager createDirectoryAtURL:cachedEncoder.URLByDeletingLastPathComponent
										 withIntermediateDirectories:YES attributes:nil error:NULL]);
	XCTAssertTrue([@"weights" writeToURL:cachedEncoder atomically:YES encoding:NSUTF8StringEncoding error:NULL]);
	NFKDownloadStatus *status = [self.hub downloadStatusForModelSource:source];
	XCTAssertEqual(status.presence, NFKDownloadPresenceComplete);
	[self assertURL:status.localURL isAtPath:folder];
	[self assertURL:[NFKModelCatalog locatorForHub:self.hub](descriptor, descriptor.defaultVariant, NULL) isAtPath:folder];
}

#pragma mark Locators

- (void)testTheHubCacheLocatorFindsAFolderThroughTheDefaultModelFolders
{
	NSURL *folder = [self folderNamed:@"model" inRoot:self.root revisionText:@"org/model\n"
								paths:@[ @"config.json", @"model.safetensors" ]];
	NFKModelDescriptor *descriptor = [self descriptorWithSource:[self modelSourceAtRevision:nil]];
	NSURL *emptyCache = [self temporaryDirectory];
	NFKModelLocator locator = [NFKModelCatalog hubCacheLocatorForCacheDirectoryURL:emptyCache];
	NFKHFHub.defaultModelFolderURLs = @[];
	XCTAssertNil(locator(descriptor, descriptor.defaultVariant, NULL));

	NFKHFHub.defaultModelFolderURLs = @[ self.root ];
	[self assertURL:locator(descriptor, descriptor.defaultVariant, NULL) isAtPath:folder];
	NFKHFHub *defaultHub = [NFKHFHub hubWithCacheDirectoryURL:emptyCache];
	XCTAssertEqualObjects(defaultHub.modelFolderURLs, @[ self.root ]);
	XCTAssertEqual([defaultHub downloadStatusForModelSource:descriptor.defaultVariant.source].presence,
				   NFKDownloadPresenceComplete, @"the status agrees with the build");
}

- (void)testADefaultSetAfterAHubIsMadeLeavesThatHubUnchanged
{
	NFKHFHub.defaultModelFolderURLs = @[];
	NFKHFHub *hub = [NFKHFHub hubWithCacheDirectoryURL:self.cacheDir];
	NFKHFHub.defaultModelFolderURLs = @[ self.root ];
	XCTAssertEqualObjects(hub.modelFolderURLs, @[]);
	XCTAssertEqualObjects([NFKHFHub hubWithCacheDirectoryURL:self.cacheDir].modelFolderURLs, @[ self.root ]);
}

- (void)testTheLocatorForAHubReadsThatHubsFolders
{
	NSURL *folder = [self folderNamed:@"model" inRoot:self.root revisionText:@"org/model\n"
								paths:@[ @"config.json", @"model.safetensors" ]];
	NFKModelDescriptor *descriptor = [self descriptorWithSource:[self modelSourceAtRevision:nil]];
	[self assertURL:[NFKModelCatalog locatorForHub:self.hub](descriptor, descriptor.defaultVariant, NULL) isAtPath:folder];

	NFKHFHub *withoutFolders = [NFKHFHub hubWithCacheDirectoryURL:self.cacheDir];
	withoutFolders.modelFolderURLs = @[];
	XCTAssertNil([NFKModelCatalog locatorForHub:withoutFolders](descriptor, descriptor.defaultVariant, NULL));
}

- (void)testTheLocatorForAHubFindsItsCacheWithoutAFolder
{
	NFKModelDescriptor *descriptor = [self descriptorWithSource:[self modelSourceAtRevision:nil]];
	NSURL *config = [self.hub localURLForRepo:@"org/model" revision:nil path:@"config.json"];
	XCTAssertTrue([NSFileManager.defaultManager createDirectoryAtURL:config.URLByDeletingLastPathComponent
										 withIntermediateDirectories:YES attributes:nil error:NULL]);
	XCTAssertTrue([@"{}" writeToURL:config atomically:YES encoding:NSUTF8StringEncoding error:NULL]);
	XCTAssertTrue([@"weights" writeToURL:[self.hub localURLForRepo:@"org/model" revision:nil path:@"model.safetensors"]
							  atomically:YES encoding:NSUTF8StringEncoding error:NULL]);
	[self assertURL:[NFKModelCatalog locatorForHub:self.hub](descriptor, descriptor.defaultVariant, NULL)
		   isAtPath:config.URLByDeletingLastPathComponent];
}

#pragma mark Removal

- (void)testARemovalOfASourceFoundOnlyInAFolderRemovesNothing
{
	NSArray<NSString *> *paths = @[ @"config.json", @"model.safetensors" ];
	NSURL *folder = [self folderNamed:@"model" inRoot:self.root revisionText:@"org/model\n" paths:paths];
	NFKModelSource *source = [self modelSourceAtRevision:nil];
	NFKRemovalPlan *plan = [self.hub removalPlanForModelSource:source error:NULL];
	XCTAssertNotNil(plan);
	XCTAssertEqual(plan.removedFiles.count, 0u);
	[self.hub removeFilesInPlan:plan error:NULL];
	for (NSString *path in paths) {
		XCTAssertTrue([NSFileManager.defaultManager fileExistsAtPath:[folder URLByAppendingPathComponent:path].path], @"%@", path);
	}
	XCTAssertEqual([self.hub downloadStatusForModelSource:source].presence, NFKDownloadPresenceComplete);
}

- (void)testARemovalOfACachedSourceLeavesTheFolderInPlace
{
	NSArray<NSString *> *paths = @[ @"config.json", @"model.safetensors" ];
	NSURL *folder = [self folderNamed:@"model" inRoot:self.root revisionText:@"org/model\n" paths:paths];
	for (NSString *path in paths) {
		NSURL *cached = [self.hub localURLForRepo:@"org/model" revision:nil path:path];
		XCTAssertTrue([NSFileManager.defaultManager createDirectoryAtURL:cached.URLByDeletingLastPathComponent
											 withIntermediateDirectories:YES attributes:nil error:NULL]);
		XCTAssertTrue([@"cached" writeToURL:cached atomically:YES encoding:NSUTF8StringEncoding error:NULL]);
	}
	XCTAssertTrue([self.hub adoptCachedRepo:@"org/model" revision:nil error:NULL]);
	NFKModelSource *source = [self modelSourceAtRevision:nil];
	NSError *error = nil;
	NFKRemovalPlan *plan = [self.hub removalPlanForModelSource:source error:&error];
	XCTAssertEqual(plan.removedFiles.count, paths.count, @"%@", error);
	for (NFKCachedFile *file in plan.removedFiles) {
		XCTAssertTrue([file.URL.URLByStandardizingPath.path hasPrefix:self.cacheDir.URLByStandardizingPath.path], @"%@", file.URL);
	}
	XCTAssertNotNil([self.hub removeFilesInPlan:plan error:&error], @"%@", error);
	for (NSString *path in paths) {
		XCTAssertFalse([NSFileManager.defaultManager fileExistsAtPath:[self.hub localURLForRepo:@"org/model" revision:nil
																						   path:path].path]);
		XCTAssertTrue([NSFileManager.defaultManager fileExistsAtPath:[folder URLByAppendingPathComponent:path].path], @"%@", path);
	}
	XCTAssertEqual([self.hub downloadStatusForModelSource:source].presence, NFKDownloadPresenceComplete,
				   @"the folder still holds the model");
}

@end
