//
//  NFKHFHubHuggingFaceCacheTests.m
//  InferKitTests
//
//  Hugging Face caches on temporary roots, laid out as huggingface_hub lays them out, with snapshot
//  files linked into blobs: refs and pinned commits, links whose blob is missing, required and optional
//  files, the location of each layout, the repository folder compared without regard to case, the order
//  among roots and against the model folders and the cache, the status and the locators that read the
//  roots, the default root, and removals that leave the roots in place.
//

#import <XCTest/XCTest.h>
#import <InferKit/InferKit.h>
#import "../../Sources/InferKit/NFKHFHubInternal.h"

static NSString * const NFKHFCacheTestCommit = @"1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0";
static NSString * const NFKHFCacheTestOtherCommit = @"f21265c1e2710b3bd2386596943f0007f55f802e";

@interface NFKHFHubHuggingFaceCacheTests : XCTestCase
@property (nonatomic, copy) NSArray<NSURL *> *savedDefaultHuggingFaceCacheURLs;
@property (nonatomic, copy) NSArray<NSURL *> *savedDefaultModelFolderURLs;
@property (nonatomic, strong) NSURL *root;
@property (nonatomic, strong) NSURL *cacheDir;
@property (nonatomic, strong) NFKHFHub *hub;
@end

@implementation NFKHFHubHuggingFaceCacheTests

- (void)setUp
{
	[super setUp];
	self.savedDefaultHuggingFaceCacheURLs = NFKHFHub.defaultHuggingFaceCacheURLs;
	self.savedDefaultModelFolderURLs = NFKHFHub.defaultModelFolderURLs;
	NFKHFHub.defaultHuggingFaceCacheURLs = @[];
	NFKHFHub.defaultModelFolderURLs = @[];
	self.root = [self temporaryDirectory];
	self.cacheDir = [self temporaryDirectory];
	self.hub = [NFKHFHub hubWithCacheDirectoryURL:self.cacheDir];
	self.hub.excludesCacheFromBackup = NO;
	self.hub.modelCatalog = [[NFKModelCatalog alloc] init];
	self.hub.huggingFaceCacheURLs = @[ self.root ];
}

- (void)tearDown
{
	NFKHFHub.defaultHuggingFaceCacheURLs = self.savedDefaultHuggingFaceCacheURLs;
	NFKHFHub.defaultModelFolderURLs = self.savedDefaultModelFolderURLs;
	[super tearDown];
}

#pragma mark Fixtures

- (NSURL *)temporaryDirectory
{
	NSString *name = [@"NFKHFHubHuggingFaceCacheTests-" stringByAppendingString:NSUUID.UUID.UUIDString];
	NSURL *url = [NSURL fileURLWithPath:[NSTemporaryDirectory() stringByAppendingPathComponent:name] isDirectory:YES];
	XCTAssertTrue([NSFileManager.defaultManager createDirectoryAtURL:url withIntermediateDirectories:YES attributes:nil
															   error:NULL]);
	[self addTeardownBlock:^{
		[NSFileManager.defaultManager removeItemAtURL:url error:NULL];
	}];
	return url;
}

- (void)writeContents:(NSString *)contents toURL:(NSURL *)url
{
	XCTAssertTrue([NSFileManager.defaultManager createDirectoryAtURL:url.URLByDeletingLastPathComponent
										 withIntermediateDirectories:YES attributes:nil error:NULL]);
	XCTAssertTrue([contents writeToURL:url atomically:YES encoding:NSUTF8StringEncoding error:NULL]);
}

/*! Writes a snapshot as huggingface_hub does: each path a relative link into the repository folder's
	blobs, and refs/ref holding the commit when ref is given. Returns the snapshot folder. */
- (NSURL *)snapshotInFolderNamed:(NSString *)folderName
						  inRoot:(NSURL *)root
						  commit:(NSString *)commit
							 ref:(nullable NSString *)ref
						   paths:(NSArray<NSString *> *)paths
{
	NSURL *repositoryURL = [root URLByAppendingPathComponent:folderName isDirectory:YES];
	NSURL *snapshotURL = [[repositoryURL URLByAppendingPathComponent:@"snapshots" isDirectory:YES]
						  URLByAppendingPathComponent:commit isDirectory:YES];
	XCTAssertTrue([NSFileManager.defaultManager createDirectoryAtURL:snapshotURL withIntermediateDirectories:YES
														  attributes:nil error:NULL]);
	for (NSString *path in paths) {
		NSString *blob = [NSUUID.UUID.UUIDString.lowercaseString stringByReplacingOccurrencesOfString:@"-" withString:@""];
		NSString *contents = [path hasSuffix:@".json"] ? @"{}" : [@"weights of " stringByAppendingString:path];
		[self writeContents:contents toURL:[[repositoryURL URLByAppendingPathComponent:@"blobs" isDirectory:YES]
											URLByAppendingPathComponent:blob isDirectory:NO]];
		NSURL *linkURL = [snapshotURL URLByAppendingPathComponent:path isDirectory:NO];
		XCTAssertTrue([NSFileManager.defaultManager createDirectoryAtURL:linkURL.URLByDeletingLastPathComponent
											 withIntermediateDirectories:YES attributes:nil error:NULL]);
		NSMutableString *destination = [NSMutableString string];
		for (NSUInteger index = 1; index < path.pathComponents.count; index += 1) {
			[destination appendString:@"../"];
		}
		[destination appendFormat:@"../../blobs/%@", blob];
		XCTAssertTrue([NSFileManager.defaultManager createSymbolicLinkAtPath:linkURL.path withDestinationPath:destination
																	   error:NULL]);
	}
	if (ref != nil) {
		[self writeContents:[commit stringByAppendingString:@"\n"]
					  toURL:[[repositoryURL URLByAppendingPathComponent:@"refs" isDirectory:YES]
							 URLByAppendingPathComponent:ref isDirectory:NO]];
	}
	return snapshotURL;
}

/*! A snapshot of org/model at NFKHFCacheTestCommit, which refs/main names. */
- (NSURL *)modelSnapshotWithPaths:(NSArray<NSString *> *)paths
{
	return [self snapshotInFolderNamed:@"models--org--model" inRoot:self.root commit:NFKHFCacheTestCommit ref:@"main"
								 paths:paths];
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
	return [[NFKModelDescriptor alloc] initWithIdentifier:@"hf-cache.model" displayName:nil family:@"test" summary:nil
												   engine:@"hf-cache-test-engine" registryName:nil providerClassName:nil
													tasks:nil inputModalities:nil outputModalities:nil inputs:nil
												  outputs:nil options:nil variants:@[ variant ] license:nil requirements:nil
											  residencies:nil standIn:NO introducedVersion:nil extensions:nil];
}

- (void)assertURL:(nullable NSURL *)url isAtPath:(NSURL *)expected
{
	XCTAssertNotNil(url);
	XCTAssertEqualObjects(url.URLByStandardizingPath.path, expected.URLByStandardizingPath.path);
}

/*! Writes org/model's config.json and model.safetensors into the hub's cache. */
- (void)cacheModelFiles
{
	for (NSString *path in @[ @"config.json", @"model.safetensors" ]) {
		[self writeContents:@"cached" toURL:[self.hub localURLForRepo:@"org/model" revision:nil path:path]];
	}
	XCTAssertTrue([self.hub adoptCachedRepo:@"org/model" revision:nil error:NULL]);
}

#pragma mark Revisions

- (void)testASourceWithoutARevisionReadsTheCommitRefsMainNames
{
	NSURL *snapshot = [self modelSnapshotWithPaths:@[ @"config.json", @"model.safetensors" ]];
	[self snapshotInFolderNamed:@"models--org--model" inRoot:self.root commit:NFKHFCacheTestOtherCommit ref:nil
						  paths:@[ @"config.json", @"model.safetensors" ]];
	[self assertURL:[self.hub huggingFaceCacheLocationForModelSource:[self modelSourceAtRevision:nil]] isAtPath:snapshot];
	[self assertURL:[self.hub huggingFaceCacheLocationForModelSource:[self modelSourceAtRevision:@""]] isAtPath:snapshot];
	[self assertURL:[self.hub huggingFaceCacheLocationForModelSource:[self modelSourceAtRevision:@"main"]] isAtPath:snapshot];
}

- (void)testAPinnedCommitNamesItsSnapshot
{
	NSArray<NSString *> *paths = @[ @"config.json", @"model.safetensors" ];
	[self modelSnapshotWithPaths:paths];
	NSURL *other = [self snapshotInFolderNamed:@"models--org--model" inRoot:self.root commit:NFKHFCacheTestOtherCommit
										   ref:nil paths:paths];
	NFKModelSource *pinned = [self modelSourceAtRevision:NFKHFCacheTestOtherCommit.uppercaseString];
	[self assertURL:[self.hub huggingFaceCacheLocationForModelSource:pinned] isAtPath:other];
}

- (void)testAPinnedCommitWithoutASnapshotIsNotAMatch
{
	[self modelSnapshotWithPaths:@[ @"config.json", @"model.safetensors" ]];
	XCTAssertNil([self.hub huggingFaceCacheLocationForModelSource:[self modelSourceAtRevision:NFKHFCacheTestOtherCommit]]);
}

- (void)testAMissingRefIsNotAMatchAndAnotherRefIsRead
{
	NSArray<NSString *> *paths = @[ @"config.json", @"model.safetensors" ];
	NSURL *snapshot = [self snapshotInFolderNamed:@"models--org--model" inRoot:self.root commit:NFKHFCacheTestCommit
											  ref:@"dev" paths:paths];
	XCTAssertNil([self.hub huggingFaceCacheLocationForModelSource:[self modelSourceAtRevision:nil]], @"no refs/main");
	[self assertURL:[self.hub huggingFaceCacheLocationForModelSource:[self modelSourceAtRevision:@"dev"]] isAtPath:snapshot];
	XCTAssertNil([self.hub huggingFaceCacheLocationForModelSource:[self modelSourceAtRevision:@"release"]]);
}

#pragma mark Files and locations

- (void)testALinkWhoseBlobIsMissingCountsAsAbsent
{
	NSURL *snapshot = [self modelSnapshotWithPaths:@[ @"config.json", @"model.safetensors" ]];
	NSURL *link = [snapshot URLByAppendingPathComponent:@"model.safetensors"];
	NSURL *blob = link.URLByResolvingSymlinksInPath;
	XCTAssertTrue([blob.path containsString:@"/blobs/"], @"%@", blob);
	XCTAssertTrue([NSFileManager.defaultManager removeItemAtURL:blob error:NULL]);
	XCTAssertNil([self.hub huggingFaceCacheLocationForModelSource:[self modelSourceAtRevision:nil]]);
	XCTAssertNotEqual([self.hub downloadStatusForModelSource:[self modelSourceAtRevision:nil]].presence,
					  NFKDownloadPresenceComplete);

	NFKModelSource *glob = [self sourceOfRepository:@"org/model" revision:nil layout:NFKModelSourceLayoutDirectory
											  files:@[ [self file:@"config.json" optional:NO],
													   [self file:@"*.safetensors" optional:NO] ]];
	XCTAssertNil([self.hub huggingFaceCacheLocationForModelSource:glob], @"a glob matches no dangling link");
}

- (void)testAMissingOptionalFileStillMatches
{
	NSURL *snapshot = [self modelSnapshotWithPaths:@[ @"config.json", @"model.safetensors" ]];
	NFKModelSource *source = [self sourceOfRepository:@"org/model" revision:nil layout:NFKModelSourceLayoutDirectory
												files:@[ [self file:@"config.json" optional:NO],
														 [self file:@"model.safetensors" optional:NO],
														 [self file:@"generation_config.json" optional:YES] ]];
	[self assertURL:[self.hub huggingFaceCacheLocationForModelSource:source] isAtPath:snapshot];
	XCTAssertEqual([self.hub downloadStatusForModelSource:source].presence, NFKDownloadPresenceComplete);
}

- (void)testAShardIndexInTheSnapshotRequiresItsShards
{
	NFKModelSource *source = [self sourceOfRepository:@"org/model" revision:nil layout:NFKModelSourceLayoutDirectory
												files:@[ [self file:@"config.json" optional:NO],
														 [self file:@"model.safetensors.index.json" optional:YES] ]];
	NSURL *snapshot = [self modelSnapshotWithPaths:@[ @"config.json", @"model-00001-of-00002.safetensors" ]];
	NSString *index = @"{\"weight_map\": {\"a\": \"model-00001-of-00002.safetensors\", \"b\": \"model-00002-of-00002.safetensors\"}}";
	[self writeContents:index toURL:[snapshot URLByAppendingPathComponent:@"model.safetensors.index.json"]];
	XCTAssertNil([self.hub huggingFaceCacheLocationForModelSource:source]);

	[self writeContents:@"weights" toURL:[snapshot URLByAppendingPathComponent:@"model-00002-of-00002.safetensors"]];
	[self assertURL:[self.hub huggingFaceCacheLocationForModelSource:source] isAtPath:snapshot];
}

- (void)testTheLocationIsTheSnapshotForADirectoryAndTheLinkForAFile
{
	NSURL *snapshot = [self modelSnapshotWithPaths:@[ @"config.json", @"weights/model.safetensors" ]];
	NSArray<NFKModelFile *> *files = @[ [self file:@"config.json" optional:NO],
										[self file:@"weights/model.safetensors" optional:NO] ];
	NFKModelSource *directory = [self sourceOfRepository:@"org/model" revision:nil layout:NFKModelSourceLayoutDirectory
												   files:files];
	[self assertURL:[self.hub huggingFaceCacheLocationForModelSource:directory] isAtPath:snapshot];

	NFKModelSource *file = [self sourceOfRepository:@"org/model" revision:nil layout:NFKModelSourceLayoutFile files:files];
	NSURL *location = [self.hub huggingFaceCacheLocationForModelSource:file];
	[self assertURL:location isAtPath:[snapshot URLByAppendingPathComponent:@"weights/model.safetensors"]];
	XCTAssertTrue([location.path containsString:@"/snapshots/"], @"%@", location);
	XCTAssertFalse([location.path containsString:@"/blobs/"], @"%@", location);
}

- (void)testTheRepositoryFolderIsComparedWithoutRegardToCase
{
	NSURL *snapshot = [self snapshotInFolderNamed:@"models--Org--Model" inRoot:self.root commit:NFKHFCacheTestCommit
											  ref:@"main" paths:@[ @"config.json", @"model.safetensors" ]];
	[self assertURL:[self.hub huggingFaceCacheLocationForModelSource:[self modelSourceAtRevision:nil]] isAtPath:snapshot];
	XCTAssertNil([self.hub huggingFaceCacheLocationForModelSource:[self sourceOfRepository:@"org/other" revision:nil
																				   layout:NFKModelSourceLayoutDirectory
																					files:@[ [self file:@"config.json" optional:NO] ]]]);
}

- (void)testOnlyAHuggingFaceSourceMatchesASnapshot
{
	[self modelSnapshotWithPaths:@[ @"base.pt" ]];
	NFKModelFile *file = [[NFKModelFile alloc] initWithPath:@"base.pt" byteCount:nil sha256:nil role:NFKModelFileRoleWeights
												   optional:NO extensions:nil];
	NFKModelSource *source = [[NFKModelSource alloc] initWithKind:NFKModelSourceKindURL repository:@"org/model" revision:nil
															  URL:[NSURL URLWithString:@"https://weights.test/org/model/base.pt"]
														   layout:NFKModelSourceLayoutFile files:@[ file ]
											   requiresConversion:NO extensions:nil];
	XCTAssertNil([self.hub huggingFaceCacheLocationForModelSource:source]);
	XCTAssertNotEqual([self.hub downloadStatusForModelSource:source].presence, NFKDownloadPresenceComplete);
}

#pragma mark Order

- (void)testTheFirstRootWins
{
	NSArray<NSString *> *paths = @[ @"config.json", @"model.safetensors" ];
	NSURL *secondRoot = [self temporaryDirectory];
	NSURL *second = [self snapshotInFolderNamed:@"models--org--model" inRoot:secondRoot commit:NFKHFCacheTestCommit
											ref:@"main" paths:paths];
	self.hub.huggingFaceCacheURLs = @[ [self.root URLByAppendingPathComponent:@"absent-volume" isDirectory:YES], self.root,
									   secondRoot ];
	[self assertURL:[self.hub huggingFaceCacheLocationForModelSource:[self modelSourceAtRevision:nil]] isAtPath:second];

	NSURL *first = [self modelSnapshotWithPaths:paths];
	[self assertURL:[self.hub huggingFaceCacheLocationForModelSource:[self modelSourceAtRevision:nil]] isAtPath:first];
}

- (void)testAModelFolderWinsOverASnapshotWhichWinsOverTheCache
{
	NFKModelSource *source = [self modelSourceAtRevision:nil];
	NFKModelDescriptor *descriptor = [self descriptorWithSource:source];
	[self cacheModelFiles];
	NSURL *cached = [self.hub localURLForRepo:@"org/model" revision:nil path:@"config.json"].URLByDeletingLastPathComponent;
	self.hub.huggingFaceCacheURLs = @[];
	[self assertURL:[self.hub downloadStatusForModelSource:source].localURL isAtPath:cached];

	NSURL *snapshot = [self modelSnapshotWithPaths:@[ @"config.json", @"model.safetensors" ]];
	self.hub.huggingFaceCacheURLs = @[ self.root ];
	[self assertURL:[self.hub downloadStatusForModelSource:source].localURL isAtPath:snapshot];
	[self assertURL:[NFKModelCatalog locatorForHub:self.hub](descriptor, descriptor.defaultVariant, NULL) isAtPath:snapshot];

	NSURL *folderRoot = [self temporaryDirectory];
	NSURL *folder = [folderRoot URLByAppendingPathComponent:@"model" isDirectory:YES];
	[self writeContents:@"org/model\n" toURL:[folder URLByAppendingPathComponent:@"REVISION"]];
	[self writeContents:@"{}" toURL:[folder URLByAppendingPathComponent:@"config.json"]];
	[self writeContents:@"weights" toURL:[folder URLByAppendingPathComponent:@"model.safetensors"]];
	self.hub.modelFolderURLs = @[ folderRoot ];
	[self assertURL:[self.hub downloadStatusForModelSource:source].localURL isAtPath:folder];
	[self assertURL:[NFKModelCatalog locatorForHub:self.hub](descriptor, descriptor.defaultVariant, NULL) isAtPath:folder];
	XCTAssertNil([self.hub modelFolderLocationForModelSource:[self modelSourceAtRevision:NFKHFCacheTestCommit]],
				 @"the folder names no revision");
	[self assertURL:[self.hub downloadStatusForModelSource:[self modelSourceAtRevision:NFKHFCacheTestCommit]].localURL
		   isAtPath:snapshot];
}

#pragma mark Status

- (void)testTheStatusIsCompleteInsideTheSnapshotWithTheResolvedSizes
{
	NSArray<NSString *> *expectedPaths = @[ @"config.json", @"model.safetensors" ];
	NSURL *snapshot = [self modelSnapshotWithPaths:expectedPaths];
	NFKDownloadStatus *status = [self.hub downloadStatusForModelSource:[self modelSourceAtRevision:nil]];
	XCTAssertEqual(status.presence, NFKDownloadPresenceComplete);
	[self assertURL:status.localURL isAtPath:snapshot];
	XCTAssertEqual(status.missingPaths.count, 0u);
	XCTAssertEqualObjects(status.presentPaths, expectedPaths);
	long long expectedBytes = 0;
	for (NSString *path in expectedPaths) {
		NSNumber *size = nil;
		NSURL *blob = [snapshot URLByAppendingPathComponent:path].URLByResolvingSymlinksInPath;
		XCTAssertTrue([blob getResourceValue:&size forKey:NSURLTotalFileAllocatedSizeKey error:NULL]);
		expectedBytes += size.longLongValue;
	}
	XCTAssertGreaterThan(expectedBytes, 0);
	XCTAssertEqual(status.byteCount, expectedBytes);

	NFKModelSource *file = [self sourceOfRepository:@"org/model" revision:nil layout:NFKModelSourceLayoutFile
											  files:@[ [self file:@"model.safetensors" optional:NO] ]];
	[self assertURL:[self.hub downloadStatusForModelSource:file].localURL
		   isAtPath:[snapshot URLByAppendingPathComponent:@"model.safetensors"]];
}

- (void)testAFileFromAnotherRepositoryIsReadInTheCache
{
	NFKModelFile *encoder = [[NFKModelFile alloc] initWithPath:@"text_encoder/model.safetensors" byteCount:nil sha256:nil
														  role:NFKModelFileRoleWeights optional:NO repository:@"org/encoder"
													  revision:nil URL:nil extensions:nil];
	NFKModelSource *source = [self sourceOfRepository:@"org/model" revision:nil layout:NFKModelSourceLayoutDirectory
												files:@[ [self file:@"config.json" optional:NO], encoder ]];
	NSURL *snapshot = [self modelSnapshotWithPaths:@[ @"config.json" ]];
	[self snapshotInFolderNamed:@"models--org--encoder" inRoot:self.root commit:NFKHFCacheTestOtherCommit ref:@"main"
						  paths:@[ @"text_encoder/model.safetensors" ]];
	[self assertURL:[self.hub huggingFaceCacheLocationForModelSource:source] isAtPath:snapshot];
	XCTAssertNotEqual([self.hub downloadStatusForModelSource:source].presence, NFKDownloadPresenceComplete,
					  @"the encoder is read in the cache, which lacks it");

	[self writeContents:@"weights" toURL:[self.hub localURLForRepo:@"org/encoder" revision:nil
															  path:@"text_encoder/model.safetensors"]];
	NFKDownloadStatus *status = [self.hub downloadStatusForModelSource:source];
	XCTAssertEqual(status.presence, NFKDownloadPresenceComplete);
	[self assertURL:status.localURL isAtPath:snapshot];
}

#pragma mark Locators and the default

- (void)testTheHubCacheLocatorFindsASnapshotThroughTheDefaultRoots
{
	NSURL *snapshot = [self modelSnapshotWithPaths:@[ @"config.json", @"model.safetensors" ]];
	NFKModelDescriptor *descriptor = [self descriptorWithSource:[self modelSourceAtRevision:nil]];
	NSURL *emptyCache = [self temporaryDirectory];
	NFKModelLocator locator = [NFKModelCatalog hubCacheLocatorForCacheDirectoryURL:emptyCache];
	XCTAssertNil(locator(descriptor, descriptor.defaultVariant, NULL));

	NFKHFHub.defaultHuggingFaceCacheURLs = @[ self.root ];
	[self assertURL:locator(descriptor, descriptor.defaultVariant, NULL) isAtPath:snapshot];
	NFKHFHub *defaultHub = [NFKHFHub hubWithCacheDirectoryURL:emptyCache];
	XCTAssertEqualObjects(defaultHub.huggingFaceCacheURLs, @[ self.root ]);
	XCTAssertEqual([defaultHub downloadStatusForModelSource:descriptor.defaultVariant.source].presence,
				   NFKDownloadPresenceComplete, @"the status agrees with the build");
}

- (void)testAnEmptyDefaultTurnsTheLookupOff
{
	[self modelSnapshotWithPaths:@[ @"config.json", @"model.safetensors" ]];
	NFKHFHub.defaultHuggingFaceCacheURLs = @[];
	NFKHFHub *hub = [NFKHFHub hubWithCacheDirectoryURL:self.cacheDir];
	XCTAssertEqualObjects(hub.huggingFaceCacheURLs, @[]);
	XCTAssertNil([hub huggingFaceCacheLocationForModelSource:[self modelSourceAtRevision:nil]]);
	XCTAssertEqual([hub downloadStatusForModelSource:[self modelSourceAtRevision:nil]].presence, NFKDownloadPresenceAbsent);
}

- (void)testTheDefaultRootIsTheOneHuggingFaceHubReads
{
	NSString *home = @"/home/someone";
	NSDictionary<NSString *, NSString *> *all = @{ @"HF_HUB_CACHE": @"/hub-cache", @"HUGGINGFACE_HUB_CACHE": @"/legacy",
												   @"HF_HOME": @"/hf-home", @"XDG_CACHE_HOME": @"/xdg" };
	XCTAssertEqualObjects(NFKHFHubHuggingFaceCacheURL(all, home).path, @"/hub-cache");
	NSDictionary<NSString *, NSString *> *legacy = @{ @"HUGGINGFACE_HUB_CACHE": @"/legacy", @"HF_HOME": @"/hf-home" };
	XCTAssertEqualObjects(NFKHFHubHuggingFaceCacheURL(legacy, home).path, @"/legacy");
	NSDictionary<NSString *, NSString *> *hfHome = @{ @"HF_HUB_CACHE": @"", @"HF_HOME": @"/hf-home", @"XDG_CACHE_HOME": @"/xdg" };
	XCTAssertEqualObjects(NFKHFHubHuggingFaceCacheURL(hfHome, home).path, @"/hf-home/hub");
	NSDictionary<NSString *, NSString *> *xdg = @{ @"XDG_CACHE_HOME": @"/xdg" };
	XCTAssertEqualObjects(NFKHFHubHuggingFaceCacheURL(xdg, home).path, @"/xdg/huggingface/hub");
	XCTAssertEqualObjects(NFKHFHubHuggingFaceCacheURL(@{}, home).path, @"/home/someone/.cache/huggingface/hub");
	NSDictionary<NSString *, NSString *> *tilde = @{ @"HF_HOME": @"~/models" };
	XCTAssertEqualObjects(NFKHFHubHuggingFaceCacheURL(tilde, home).path, @"/home/someone/models/hub");
	XCTAssertTrue(NFKHFHubHuggingFaceCacheURL(@{}, home).hasDirectoryPath);
}

#pragma mark Removal

- (void)testARemovalLeavesTheSnapshotAndBlobsInPlace
{
	NSArray<NSString *> *paths = @[ @"config.json", @"model.safetensors" ];
	NSURL *snapshot = [self modelSnapshotWithPaths:paths];
	NSURL *blobs = [[self.root URLByAppendingPathComponent:@"models--org--model"] URLByAppendingPathComponent:@"blobs"];
	NSArray<NSString *> *blobNames = [NSFileManager.defaultManager contentsOfDirectoryAtPath:blobs.path error:NULL];
	XCTAssertEqual(blobNames.count, paths.count);
	NFKModelSource *source = [self modelSourceAtRevision:nil];

	NFKRemovalPlan *plan = [self.hub removalPlanForModelSource:source error:NULL];
	XCTAssertNotNil(plan);
	XCTAssertEqual(plan.removedFiles.count, 0u, @"found only in a Hugging Face cache");
	XCTAssertNotNil([self.hub removeFilesInPlan:plan error:NULL]);

	[self cacheModelFiles];
	NSError *error = nil;
	plan = [self.hub removalPlanForModelSource:source error:&error];
	XCTAssertEqual(plan.removedFiles.count, paths.count, @"%@", error);
	NSString *rootPath = self.root.URLByResolvingSymlinksInPath.path;
	for (NFKCachedFile *file in [plan.removedFiles arrayByAddingObjectsFromArray:plan.keptFiles]) {
		XCTAssertFalse([file.URL.URLByResolvingSymlinksInPath.path hasPrefix:rootPath], @"%@", file.URL);
	}
	XCTAssertNotNil([self.hub removeFilesInPlan:plan error:&error], @"%@", error);
	for (NSString *path in paths) {
		XCTAssertFalse([NSFileManager.defaultManager fileExistsAtPath:[self.hub localURLForRepo:@"org/model" revision:nil
																						   path:path].path]);
		XCTAssertTrue([NSFileManager.defaultManager fileExistsAtPath:[snapshot URLByAppendingPathComponent:path].path], @"%@", path);
	}
	NSSet<NSString *> *remainingBlobs = [NSSet setWithArray:[NSFileManager.defaultManager contentsOfDirectoryAtPath:blobs.path
																											 error:NULL]];
	XCTAssertEqualObjects(remainingBlobs, [NSSet setWithArray:blobNames]);
	XCTAssertEqual([self.hub downloadStatusForModelSource:source].presence, NFKDownloadPresenceComplete,
				   @"the Hugging Face cache still holds the model");
}

@end
