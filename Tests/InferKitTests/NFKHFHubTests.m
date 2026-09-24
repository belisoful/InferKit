//
//  NFKHFHubTests.m
//  NFKTests
//
//  Covers URL construction, caching, and checksum through a stub transport. A live download
//  is integration-verified.
//

#import <XCTest/XCTest.h>
#import <InferKit/NFKHFHub.h>
#import <InferKit/NFKErrors.h>

/*! A hub whose transport writes staged bytes and counts fetches. */
@interface FxHFHubStub : NFKHFHub
@property (nonatomic, strong) NSData *stagedData;
@property (nonatomic, assign) NSUInteger fetchCount;
@end

@implementation FxHFHubStub

- (BOOL)fetchURL:(NSURL *)remoteURL toFileURL:(NSURL *)destinationURL error:(NSError **)outError
{
	self.fetchCount += 1;
	return [self.stagedData writeToURL:destinationURL atomically:YES];
}

@end

@interface NFKHFHubTests : XCTestCase
@property (nonatomic, strong) FxHFHubStub *hub;
@property (nonatomic, strong) NSURL *cacheDir;
@property (nonatomic, assign) long long savedDefaultLimit;
@property (nonatomic, assign) BOOL savedDefaultExclusion;
@property (nonatomic, copy) NSString *savedDefaultToken;
@end

@implementation NFKHFHubTests

- (void)setUp
{
	[super setUp];
	self.savedDefaultLimit = NFKHFHub.defaultCacheSizeLimit;
	self.savedDefaultExclusion = NFKHFHub.defaultExcludesCacheFromBackup;
	self.savedDefaultToken = NFKHFHub.defaultAccessToken;
	self.cacheDir = [NSURL fileURLWithPath:[NSTemporaryDirectory() stringByAppendingPathComponent:@"NFKHFHubTests"]];
	[NSFileManager.defaultManager removeItemAtURL:self.cacheDir error:NULL];
	[NSFileManager.defaultManager createDirectoryAtURL:self.cacheDir withIntermediateDirectories:YES attributes:nil error:NULL];

	self.hub = [[FxHFHubStub alloc] init];
	self.hub.cacheDirectoryURL = self.cacheDir;
	self.hub.stagedData = [@"hello" dataUsingEncoding:NSUTF8StringEncoding];
}

- (void)tearDown
{
	[NSFileManager.defaultManager removeItemAtURL:self.cacheDir error:NULL];
	NFKHFHub.defaultCacheSizeLimit = self.savedDefaultLimit;
	NFKHFHub.defaultExcludesCacheFromBackup = self.savedDefaultExclusion;
	NFKHFHub.defaultAccessToken = self.savedDefaultToken;
	[super tearDown];
}

- (void)testTheProcessDefaultTokenAppliesUntilAHubSetsItsOwn
{
	NFKHFHub *hub = [NFKHFHub hubWithCacheDirectoryURL:self.cacheDir];
	NFKHFHub.defaultAccessToken = @"hf_default";
	XCTAssertEqualObjects(hub.accessToken, @"hf_default", @"a default set after the hub was made still applies");
	hub.accessToken = @"hf_own";
	XCTAssertEqualObjects(hub.accessToken, @"hf_own", @"a hub's own token wins");
	hub.accessToken = nil;
	XCTAssertEqualObjects(hub.accessToken, @"hf_default");
	NFKHFHub.defaultAccessToken = nil;
	XCTAssertEqualObjects(hub.accessToken, NSProcessInfo.processInfo.environment[@"HF_TOKEN"],
		@"with neither, the environment's token or nil");
}

- (void)testTheRemoteURLFollowsTheResolveLayout
{
	NSURL *url = [self.hub remoteURLForRepo:@"apple/coreml-sd" revision:nil path:@"unet/model.mlpackage"];
	XCTAssertEqualObjects(url.absoluteString,
						@"https://huggingface.co/apple/coreml-sd/resolve/main/unet/model.mlpackage");
}

- (void)testTheLocalURLMirrorsTheRepoLayout
{
	NSURL *url = [self.hub localURLForRepo:@"apple/coreml-sd" revision:@"v2" path:@"unet/model.bin"];
	NSURL *expected = [[[[self.cacheDir URLByAppendingPathComponent:@"apple"]
						 URLByAppendingPathComponent:@"coreml-sd"]
						URLByAppendingPathComponent:@"v2"]
					   URLByAppendingPathComponent:@"unet/model.bin"];
	XCTAssertEqualObjects(url.path, expected.path);
}

- (void)testDownloadingWithoutACacheDirectoryFails
{
	self.hub.cacheDirectoryURL = nil;
	NSError *error = nil;
	NSURL *result = [self.hub downloadRepo:@"a/b" revision:nil path:@"f.bin" sha256:nil error:&error];
	XCTAssertNil(result);
	XCTAssertEqual(error.code, (NSInteger)kNFKError_InferenceNotReady);
}

- (void)testDownloadingCachesTheFileAndReturnsItsURL
{
	NSError *error = nil;
	NSURL *result = [self.hub downloadRepo:@"a/b" revision:nil path:@"f.bin" sha256:nil error:&error];
	XCTAssertNil(error);
	XCTAssertNotNil(result);
	XCTAssertTrue([NSFileManager.defaultManager fileExistsAtPath:result.path]);
	XCTAssertTrue([self.hub isCachedRepo:@"a/b" revision:nil path:@"f.bin"]);
	XCTAssertEqual(self.hub.fetchCount, (NSUInteger)1);
}

- (void)testASecondDownloadSkipsTheFetch
{
	[self.hub downloadRepo:@"a/b" revision:nil path:@"f.bin" sha256:nil error:NULL];
	[self.hub downloadRepo:@"a/b" revision:nil path:@"f.bin" sha256:nil error:NULL];
	XCTAssertEqual(self.hub.fetchCount, (NSUInteger)1, @"the cached file is reused");
}

- (void)testAChecksumMatchSucceeds
{
	// SHA-256 of "hello".
	NSString *sha = @"2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824";
	NSError *error = nil;
	NSURL *result = [self.hub downloadRepo:@"a/b" revision:nil path:@"f.bin" sha256:sha error:&error];
	XCTAssertNotNil(result);
	XCTAssertNil(error);
}

- (void)testAChecksumMismatchFailsAndDoesNotCache
{
	NSError *error = nil;
	NSURL *result = [self.hub downloadRepo:@"a/b" revision:nil path:@"f.bin" sha256:@"deadbeef" error:&error];
	XCTAssertNil(result);
	XCTAssertEqual(error.code, (NSInteger)kNFKError_InferenceBackendFailure);
	XCTAssertFalse([self.hub isCachedRepo:@"a/b" revision:nil path:@"f.bin"]);
}

- (void)testTheDefaultCacheDirectoryIsUnderApplicationSupport
{
	NSURL *directory = [NFKHFHub defaultCacheDirectoryURL];
	XCTAssertTrue([directory.path containsString:@"InferKit"]);
	XCTAssertTrue([directory.path hasSuffix:@"models"]);
	BOOL isDirectory = NO;
	XCTAssertTrue([NSFileManager.defaultManager fileExistsAtPath:directory.path isDirectory:&isDirectory]);
	XCTAssertTrue(isDirectory, @"the default cache directory is created on demand");
}

- (void)testTheAsynchronousDownloadDeliversTheLocalURL
{
	XCTestExpectation *done = [self expectationWithDescription:@"async download"];
	[self.hub downloadRepo:@"a/b" revision:nil path:@"f.bin" sha256:nil
		 completionHandler:^(NSURL *localURL, NSError *error) {
		XCTAssertNotNil(localURL);
		XCTAssertNil(error);
		[done fulfill];
	}];
	[self waitForExpectations:@[ done ] timeout:5];
	XCTAssertTrue([self.hub isCachedRepo:@"a/b" revision:nil path:@"f.bin"]);
}

#pragma mark Cache policy

- (NSURL *)snapshotURLForRepo:(NSString *)repo
{
	return [[self.cacheDir URLByAppendingPathComponent:repo] URLByAppendingPathComponent:@"main"];
}

- (void)markRepo:(NSString *)repo lastUsedSecondsAgo:(NSTimeInterval)seconds
{
	NSURL *marker = [[self snapshotURLForRepo:repo] URLByAppendingPathComponent:@".inferkit-owned"];
	NSDictionary *attributes = @{ NSFileModificationDate: [NSDate dateWithTimeIntervalSinceNow:-seconds] };
	XCTAssertTrue([NSFileManager.defaultManager setAttributes:attributes ofItemAtPath:marker.path error:NULL]);
}

- (void)testTheProcessWideDefaultsAreUnlimitedAndExcludedFromBackup
{
	XCTAssertEqual(NFKHFHub.defaultCacheSizeLimit, NFKHFHubUnlimitedCacheSize);
	XCTAssertTrue(NFKHFHub.defaultExcludesCacheFromBackup);
	XCTAssertEqual(self.hub.cacheSizeLimit, NFKHFHubUnlimitedCacheSize);
	XCTAssertTrue(self.hub.excludesCacheFromBackup);
}

- (void)testANewHubStartsFromTheProcessWideDefaults
{
	NFKHFHub.defaultCacheSizeLimit = 1234;
	NFKHFHub.defaultExcludesCacheFromBackup = NO;
	NFKHFHub *hub = [NFKHFHub hubWithCacheDirectoryURL:self.cacheDir];
	XCTAssertEqual(hub.cacheSizeLimit, 1234LL);
	XCTAssertFalse(hub.excludesCacheFromBackup);
	XCTAssertEqual(self.hub.cacheSizeLimit, NFKHFHubUnlimitedCacheSize, @"an existing hub keeps its setting");
	XCTAssertTrue(self.hub.excludesCacheFromBackup);
}

- (void)testADownloadExcludesTheCacheFromBackup
{
	XCTAssertFalse([NFKHFHub isExcludedFromBackup:self.cacheDir]);
	XCTAssertNotNil([self.hub downloadRepo:@"a/b" revision:nil path:@"f.bin" sha256:nil error:NULL]);
	XCTAssertTrue([NFKHFHub isExcludedFromBackup:self.cacheDir]);
}

- (void)testAHubThatDoesNotExcludeLeavesTheCacheIncluded
{
	self.hub.excludesCacheFromBackup = NO;
	XCTAssertNotNil([self.hub downloadRepo:@"a/b" revision:nil path:@"f.bin" sha256:nil error:NULL]);
	XCTAssertFalse([NFKHFHub isExcludedFromBackup:self.cacheDir]);
}

- (void)testExclusionFromBackupRoundTrips
{
	NSError *error = nil;
	XCTAssertTrue([NFKHFHub setExcludedFromBackup:YES forURL:self.cacheDir error:&error]);
	XCTAssertNil(error);
	XCTAssertTrue([NFKHFHub isExcludedFromBackup:self.cacheDir]);
	XCTAssertTrue([NFKHFHub setExcludedFromBackup:NO forURL:self.cacheDir error:&error]);
	XCTAssertFalse([NFKHFHub isExcludedFromBackup:self.cacheDir]);
}

- (void)testTheCacheSizeCountsDownloadedFiles
{
	XCTAssertEqual([self.hub cacheSize], 0LL);
	[self.hub downloadRepo:@"a/b" revision:nil path:@"f.bin" sha256:nil error:NULL];
	XCTAssertGreaterThan([self.hub cacheSize], 0LL);
	XCTAssertEqual([[NFKHFHub hubWithCacheDirectoryURL:nil] cacheSize], 0LL);
}

- (void)testTrimmingEvictsTheLeastRecentlyUsedSnapshotWhole
{
	self.hub.stagedData = [NSMutableData dataWithLength:64 * 1024];
	[self.hub downloadRepo:@"org/one" revision:nil path:@"f.bin" sha256:nil error:NULL];
	long long fileSize = [self.hub cacheSize];
	[self.hub downloadRepo:@"org/one" revision:nil path:@"g.bin" sha256:nil error:NULL];
	[self.hub downloadRepo:@"org/two" revision:nil path:@"f.bin" sha256:nil error:NULL];
	[self markRepo:@"org/one" lastUsedSecondsAgo:100];
	[self markRepo:@"org/two" lastUsedSecondsAgo:50];

	self.hub.cacheSizeLimit = 3 * fileSize;
	XCTAssertNotNil([self.hub downloadRepo:@"org/three" revision:nil path:@"f.bin" sha256:nil error:NULL]);

	XCTAssertFalse([NSFileManager.defaultManager fileExistsAtPath:[self snapshotURLForRepo:@"org/one"].path]);
	XCTAssertTrue([self.hub isCachedRepo:@"org/two" revision:nil path:@"f.bin"]);
	XCTAssertTrue([self.hub isCachedRepo:@"org/three" revision:nil path:@"f.bin"]);
}

- (void)testACacheHitRefreshesTheSnapshotsRecency
{
	self.hub.stagedData = [NSMutableData dataWithLength:64 * 1024];
	[self.hub downloadRepo:@"org/one" revision:nil path:@"f.bin" sha256:nil error:NULL];
	long long snapshotSize = [self.hub cacheSize];
	[self.hub downloadRepo:@"org/two" revision:nil path:@"f.bin" sha256:nil error:NULL];
	[self markRepo:@"org/one" lastUsedSecondsAgo:100];
	[self markRepo:@"org/two" lastUsedSecondsAgo:50];
	[self.hub downloadRepo:@"org/one" revision:nil path:@"f.bin" sha256:nil error:NULL];

	self.hub.cacheSizeLimit = 2 * snapshotSize;
	[self.hub downloadRepo:@"org/three" revision:nil path:@"f.bin" sha256:nil error:NULL];

	XCTAssertTrue([self.hub isCachedRepo:@"org/one" revision:nil path:@"f.bin"]);
	XCTAssertFalse([self.hub isCachedRepo:@"org/two" revision:nil path:@"f.bin"]);
}

- (void)testTheRequestedSnapshotStaysWhenItAloneExceedsTheLimit
{
	self.hub.cacheSizeLimit = 1;
	NSURL *result = [self.hub downloadRepo:@"a/b" revision:nil path:@"f.bin" sha256:nil error:NULL];
	XCTAssertNotNil(result);
	XCTAssertTrue([NSFileManager.defaultManager fileExistsAtPath:result.path]);
}

- (void)testAnUnlimitedCacheEvictsNothing
{
	[self.hub downloadRepo:@"org/one" revision:nil path:@"f.bin" sha256:nil error:NULL];
	[self.hub downloadRepo:@"org/two" revision:nil path:@"f.bin" sha256:nil error:NULL];
	XCTAssertTrue([self.hub trimCacheToSizeLimitWithError:NULL]);
	XCTAssertTrue([self.hub isCachedRepo:@"org/one" revision:nil path:@"f.bin"]);
	XCTAssertTrue([self.hub isCachedRepo:@"org/two" revision:nil path:@"f.bin"]);
}

- (void)testFilesTheHubDidNotDownloadAreNeverEvicted
{
	NSURL *foreign = [self.cacheDir URLByAppendingPathComponent:@"notes.txt"];
	XCTAssertTrue([[NSMutableData dataWithLength:8192] writeToURL:foreign atomically:YES]);
	[self.hub downloadRepo:@"a/b" revision:nil path:@"f.bin" sha256:nil error:NULL];

	self.hub.cacheSizeLimit = 0;
	NSError *error = nil;
	XCTAssertTrue([self.hub trimCacheToSizeLimitWithError:&error]);
	XCTAssertNil(error);
	XCTAssertFalse([self.hub isCachedRepo:@"a/b" revision:nil path:@"f.bin"]);
	XCTAssertTrue([NSFileManager.defaultManager fileExistsAtPath:foreign.path]);
}

- (void)testASnapshotWithAnInFlightDownloadIsKept
{
	[self.hub downloadRepo:@"a/b" revision:nil path:@"f.bin" sha256:nil error:NULL];
	NSURL *partial = [[self snapshotURLForRepo:@"a/b"] URLByAppendingPathComponent:@"g.bin.download"];
	XCTAssertTrue([self.hub.stagedData writeToURL:partial atomically:YES]);

	self.hub.cacheSizeLimit = 0;
	XCTAssertTrue([self.hub trimCacheToSizeLimitWithError:NULL]);
	XCTAssertTrue([self.hub isCachedRepo:@"a/b" revision:nil path:@"f.bin"]);
}

- (void)testRemovingACachedRepoRemovesTheFoldersItEmpties
{
	[self.hub downloadRepo:@"org/one" revision:nil path:@"f.bin" sha256:nil error:NULL];
	NSError *error = nil;
	XCTAssertTrue([self.hub removeCachedRepo:@"org/one" revision:nil error:&error]);
	XCTAssertNil(error);
	XCTAssertFalse([NSFileManager.defaultManager fileExistsAtPath:[self.cacheDir URLByAppendingPathComponent:@"org"].path]);
	XCTAssertTrue([NSFileManager.defaultManager fileExistsAtPath:self.cacheDir.path], @"the cache folder itself stays");
	XCTAssertTrue([self.hub removeCachedRepo:@"org/one" revision:nil error:&error], @"removing an absent snapshot succeeds");
}

- (NSURL *)writeLegacySnapshotForRepo:(NSString *)repo
{
	NSURL *snapshot = [self snapshotURLForRepo:repo];
	[NSFileManager.defaultManager createDirectoryAtURL:snapshot withIntermediateDirectories:YES attributes:nil error:NULL];
	XCTAssertTrue([self.hub.stagedData writeToURL:[snapshot URLByAppendingPathComponent:@"f.bin"] atomically:YES]);
	return snapshot;
}

- (void)testASnapshotTheHubDoesNotOwnIsKeptUntilAdopted
{
	NSURL *legacy = [self writeLegacySnapshotForRepo:@"org/legacy"];
	self.hub.cacheSizeLimit = 0;
	XCTAssertTrue([self.hub trimCacheToSizeLimitWithError:NULL]);
	XCTAssertTrue([NSFileManager.defaultManager fileExistsAtPath:legacy.path]);

	NSError *error = nil;
	XCTAssertTrue([self.hub adoptCachedRepo:@"org/legacy" revision:nil error:&error]);
	XCTAssertNil(error);
	XCTAssertTrue([self.hub trimCacheToSizeLimitWithError:NULL]);
	XCTAssertFalse([NSFileManager.defaultManager fileExistsAtPath:legacy.path]);
}

- (void)testACacheHitAdoptsASnapshotTheHubDoesNotOwn
{
	NSURL *legacy = [self writeLegacySnapshotForRepo:@"org/legacy"];
	XCTAssertNotNil([self.hub downloadRepo:@"org/legacy" revision:nil path:@"f.bin" sha256:nil error:NULL]);
	XCTAssertEqual(self.hub.fetchCount, (NSUInteger)0);
	NSString *marker = [legacy URLByAppendingPathComponent:@".inferkit-owned"].path;
	XCTAssertTrue([NSFileManager.defaultManager fileExistsAtPath:marker]);
}

- (void)testAdoptingAnUncachedRepoFails
{
	NSError *error = nil;
	XCTAssertFalse([self.hub adoptCachedRepo:@"org/absent" revision:nil error:&error]);
	XCTAssertEqual(error.code, (NSInteger)kNFKError_InferenceBackendFailure);
}

- (void)testAPinnedSnapshotIsNeverEvicted
{
	[self.hub downloadRepo:@"org/one" revision:nil path:@"f.bin" sha256:nil error:NULL];
	NSError *error = nil;
	XCTAssertTrue([self.hub pinCachedRepo:@"org/one" revision:nil error:&error]);
	XCTAssertTrue([self.hub isCachedRepoPinned:@"org/one" revision:nil]);

	self.hub.cacheSizeLimit = 0;
	XCTAssertTrue([self.hub trimCacheToSizeLimitWithError:NULL]);
	XCTAssertTrue([self.hub isCachedRepo:@"org/one" revision:nil path:@"f.bin"]);

	XCTAssertTrue([self.hub unpinCachedRepo:@"org/one" revision:nil error:&error]);
	XCTAssertFalse([self.hub isCachedRepoPinned:@"org/one" revision:nil]);
	XCTAssertTrue([self.hub trimCacheToSizeLimitWithError:NULL]);
	XCTAssertFalse([self.hub isCachedRepo:@"org/one" revision:nil path:@"f.bin"]);
}

- (void)testARepoCanBePinnedBeforeItsFirstDownload
{
	XCTAssertTrue([self.hub pinCachedRepo:@"org/later" revision:nil error:NULL]);
	XCTAssertFalse([self.hub isCachedRepo:@"org/later" revision:nil path:@"f.bin"]);
	[self.hub downloadRepo:@"org/later" revision:nil path:@"f.bin" sha256:nil error:NULL];
	self.hub.cacheSizeLimit = 0;
	XCTAssertTrue([self.hub trimCacheToSizeLimitWithError:NULL]);
	XCTAssertTrue([self.hub isCachedRepo:@"org/later" revision:nil path:@"f.bin"]);
}

- (void)testUnpinningAnEmptyPinRemovesItsFolders
{
	XCTAssertTrue([self.hub pinCachedRepo:@"org/later" revision:nil error:NULL]);
	XCTAssertTrue([self.hub unpinCachedRepo:@"org/later" revision:nil error:NULL]);
	XCTAssertFalse([NSFileManager.defaultManager fileExistsAtPath:[self.cacheDir URLByAppendingPathComponent:@"org"].path]);
	XCTAssertTrue([self.hub unpinCachedRepo:@"org/later" revision:nil error:NULL], @"unpinning twice succeeds");
}

- (void)testRemovingAPinnedRepoRemovesIt
{
	[self.hub downloadRepo:@"org/one" revision:nil path:@"f.bin" sha256:nil error:NULL];
	XCTAssertTrue([self.hub pinCachedRepo:@"org/one" revision:nil error:NULL]);
	XCTAssertTrue([self.hub removeCachedRepo:@"org/one" revision:nil error:NULL]);
	XCTAssertFalse([self.hub isCachedRepo:@"org/one" revision:nil path:@"f.bin"]);
	XCTAssertFalse([self.hub isCachedRepoPinned:@"org/one" revision:nil]);
}

@end
