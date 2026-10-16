//
//  NFKDownloadRemovalTests.m
//  InferKitTests
//
//  Download status and removal on a temporary cache: complete, partial, and absent; the files a
//  removal plan names; files another complete model uses, shared link targets, and files of other
//  snapshots kept; the bytes freed; a pin kept; a removal refused while a download writes into the
//  snapshot; sources spanning repositories; and unsafe names refused.
//

#import <XCTest/XCTest.h>
#import <InferKit/InferKit.h>
#import "NFKStubHubProtocol.h"

@interface NFKDownloadRemovalTests : XCTestCase
@property (nonatomic, strong) NSURL *cacheDir;
@property (nonatomic, strong) NSURL *outsideDir;
@property (nonatomic, strong) NFKHFHub *hub;
@property (nonatomic, strong) NFKModelCatalog *catalog;
@end

@implementation NFKDownloadRemovalTests

- (void)setUp
{
	[super setUp];
	[NFKStubHubProtocol reset];
	NSString *name = NSUUID.UUID.UUIDString;
	self.cacheDir = [NSURL fileURLWithPath:[NSTemporaryDirectory() stringByAppendingPathComponent:[@"NFKDownloadRemovalTests-" stringByAppendingString:name]]];
	self.outsideDir = [NSURL fileURLWithPath:[NSTemporaryDirectory() stringByAppendingPathComponent:[@"NFKDownloadRemovalOutside-" stringByAppendingString:name]]];
	[NSFileManager.defaultManager createDirectoryAtURL:self.cacheDir withIntermediateDirectories:YES attributes:nil error:NULL];
	[NSFileManager.defaultManager createDirectoryAtURL:self.outsideDir withIntermediateDirectories:YES attributes:nil error:NULL];
	self.hub = [NFKHFHub hubWithCacheDirectoryURL:self.cacheDir];
	self.hub.endpointURL = NFKStubHubProtocol.endpointURL;
	self.hub.session = [NFKStubHubProtocol session];
	self.hub.accessToken = @"";
	self.hub.excludesCacheFromBackup = NO;
	self.catalog = [[NFKModelCatalog alloc] init];
	self.hub.modelCatalog = self.catalog;
}

- (void)tearDown
{
	[NSFileManager.defaultManager removeItemAtURL:self.cacheDir error:NULL];
	[NSFileManager.defaultManager removeItemAtURL:self.outsideDir error:NULL];
	[NFKStubHubProtocol reset];
	[super tearDown];
}

#pragma mark Fixtures

- (NSData *)bytes:(NSUInteger)count
{
	NSMutableData *data = [NSMutableData dataWithLength:count];
	uint8_t *bytes = data.mutableBytes;
	for (NSUInteger i = 0; i < count; i++) {
		bytes[i] = (uint8_t)(i * 13 + 1);
	}
	return data;
}

/*! Writes a file into a repository's snapshot and marks the snapshot owned, as a download leaves it. */
- (NSURL *)place:(NSUInteger)byteCount inRepo:(NSString *)repo path:(NSString *)path
{
	NSURL *url = [self.hub localURLForRepo:repo revision:nil path:path];
	[NSFileManager.defaultManager createDirectoryAtURL:url.URLByDeletingLastPathComponent withIntermediateDirectories:YES attributes:nil error:NULL];
	XCTAssertTrue([[self bytes:byteCount] writeToURL:url atomically:YES]);
	XCTAssertTrue([self.hub adoptCachedRepo:repo revision:nil error:NULL]);
	return url;
}

- (NSURL *)linkInRepo:(NSString *)repo path:(NSString *)path to:(NSURL *)target
{
	NSURL *url = [self.hub localURLForRepo:repo revision:nil path:path];
	[NSFileManager.defaultManager createDirectoryAtURL:url.URLByDeletingLastPathComponent withIntermediateDirectories:YES attributes:nil error:NULL];
	XCTAssertTrue([NSFileManager.defaultManager createSymbolicLinkAtURL:url withDestinationURL:target error:NULL]);
	XCTAssertTrue([self.hub adoptCachedRepo:repo revision:nil error:NULL]);
	return url;
}

- (NFKModelSource *)sourceForRepo:(NSString *)repo paths:(NSArray<NSString *> *)paths optional:(NSArray<NSString *> *)optional
{
	NSMutableArray<NFKModelFile *> *files = [NSMutableArray array];
	for (NSString *path in paths) {
		[files addObject:[NFKModelFile fileWithPath:path role:NFKModelFileRoleOther]];
	}
	for (NSString *path in optional) {
		[files addObject:[[NFKModelFile alloc] initWithPath:path byteCount:nil sha256:nil role:nil optional:YES extensions:nil]];
	}
	return [[NFKModelSource alloc] initWithKind:NFKModelSourceKindHuggingFace repository:repo revision:nil URL:nil
										 layout:NFKModelSourceLayoutDirectory files:files requiresConversion:NO extensions:nil];
}

/*! Adds a model with one variant over a source to the hub's catalog. */
- (void)addModel:(NSString *)identifier source:(NFKModelSource *)source
{
	NSDictionary *dictionary = @{ @"id": identifier, @"family": identifier, @"engine": @"plugin",
								  @"variants": @[ @{ @"id": @"default", @"isDefault": @YES, @"source": source.dictionaryRepresentation } ] };
	NSError *error = nil;
	NFKModelDescriptor *descriptor = [NFKModelDescriptor descriptorWithDictionary:dictionary error:&error];
	XCTAssertNotNil(descriptor, @"%@", error);
	[self.catalog addDescriptor:descriptor];
}

- (NSArray<NSString *> *)pathsOf:(NSArray<NFKCachedFile *> *)files
{
	return [[files valueForKey:@"path"] sortedArrayUsingSelector:@selector(compare:)];
}

#pragma mark Status

- (void)testTheStatusReadsAbsentPartialAndComplete
{
	NFKModelSource *source = [self sourceForRepo:@"org/model" paths:@[ @"config.json", @"model.bin" ] optional:@[ @"extra.json" ]];
	NFKDownloadStatus *status = [self.hub downloadStatusForModelSource:source];
	XCTAssertEqual(status.presence, NFKDownloadPresenceAbsent);
	XCTAssertEqual(status.byteCount, 0LL);

	[self place:4096 inRepo:@"org/model" path:@"config.json"];
	status = [self.hub downloadStatusForModelSource:source];
	XCTAssertEqual(status.presence, NFKDownloadPresencePartial);
	NSArray<NSString *> *missing = @[ @"model.bin" ];
	XCTAssertEqualObjects(status.missingPaths, missing);
	XCTAssertGreaterThanOrEqual(status.byteCount, 4096LL);

	[self place:8192 inRepo:@"org/model" path:@"model.bin"];
	status = [self.hub downloadStatusForModelSource:source];
	XCTAssertEqual(status.presence, NFKDownloadPresenceComplete, @"an absent optional file does not make it partial");
	XCTAssertGreaterThanOrEqual(status.byteCount, 4096LL + 8192LL);
	XCTAssertEqualObjects(status.localURL.path, [self.cacheDir URLByAppendingPathComponent:@"org/model/main"].path);
}

- (void)testAWholeRepositoryReadsCompleteFromItsRecord
{
	[self place:100 inRepo:@"org/release" path:@"config.json"];
	XCTAssertEqual([self.hub downloadStatusForRepo:@"org/release" revision:nil paths:nil].presence, NFKDownloadPresencePartial,
		@"files without a record are a partial download");

	[self place:100 inRepo:@"org/release" path:@"weights/model.bin"];
	NSArray<NSString *> *paths = @[ @"config.json", @"weights/model.bin" ];
	XCTAssertTrue([self.hub recordCompleteDownloadOfRepo:@"org/release" revision:nil paths:paths error:NULL]);
	NFKDownloadStatus *status = [self.hub downloadStatusForRepo:@"org/release" revision:nil paths:nil];
	XCTAssertEqual(status.presence, NFKDownloadPresenceComplete);
	XCTAssertEqualObjects(status.presentPaths, paths, @"the hub's bookkeeping files are not the model's");

	[NSFileManager.defaultManager removeItemAtURL:[self.hub localURLForRepo:@"org/release" revision:nil path:@"weights/model.bin"] error:NULL];
	XCTAssertEqual([self.hub downloadStatusForRepo:@"org/release" revision:nil paths:nil].presence, NFKDownloadPresencePartial);
}

#pragma mark Removal plans

- (void)testARemovalPlanNamesExactlyTheSourcesFiles
{
	[self place:4096 inRepo:@"org/model" path:@"config.json"];
	[self place:8192 inRepo:@"org/model" path:@"weights/model.bin"];
	[self place:4096 inRepo:@"org/model" path:@"notes.txt"];
	NFKModelSource *source = [self sourceForRepo:@"org/model" paths:@[ @"config.json", @"weights/model.bin" ] optional:@[ @"absent.json" ]];

	NSError *error = nil;
	NFKRemovalPlan *plan = [self.hub removalPlanForModelSource:source error:&error];
	XCTAssertNotNil(plan, @"%@", error);
	NSArray<NSString *> *expected = @[ @"org/model/main/config.json", @"org/model/main/weights/model.bin" ];
	XCTAssertEqualObjects([self pathsOf:plan.removedFiles], expected);
	XCTAssertEqual(plan.keptFiles.count, (NSUInteger)0);
	long long sizes = 0;
	for (NFKCachedFile *file in plan.removedFiles) {
		XCTAssertNil(file.reason);
		sizes += file.byteCount;
	}
	XCTAssertEqual(plan.freedByteCount, sizes, @"the snapshot keeps a file, so its bookkeeping stays");
}

- (void)testARemovalKeepsAFileAnotherCompleteModelUses
{
	[self place:4096 inRepo:@"org/shared" path:@"tokenizer.json"];
	[self place:8192 inRepo:@"org/shared" path:@"a.bin"];
	[self place:8192 inRepo:@"org/shared" path:@"b.bin"];
	NFKModelSource *first = [self sourceForRepo:@"org/shared" paths:@[ @"a.bin", @"tokenizer.json" ] optional:@[]];
	NFKModelSource *second = [self sourceForRepo:@"org/shared" paths:@[ @"b.bin", @"tokenizer.json" ] optional:@[]];
	[self addModel:@"model-a" source:first];
	[self addModel:@"model-b" source:second];

	NFKRemovalPlan *plan = [self.hub removalPlanForModelSource:first error:NULL];
	NSArray<NSString *> *removed = @[ @"org/shared/main/a.bin" ];
	XCTAssertEqualObjects([self pathsOf:plan.removedFiles], removed);
	XCTAssertEqual(plan.keptFiles.count, (NSUInteger)1);
	XCTAssertEqualObjects(plan.keptFiles.firstObject.path, @"org/shared/main/tokenizer.json");
	XCTAssertTrue([plan.keptFiles.firstObject.reason containsString:@"model-b"], @"%@", plan.keptFiles.firstObject.reason);

	long long before = [self.hub cacheSize];
	NSError *error = nil;
	NFKRemovalPlan *removal = [self.hub removeFilesInPlan:plan error:&error];
	XCTAssertNotNil(removal, @"%@", error);
	XCTAssertEqual(before - [self.hub cacheSize], removal.freedByteCount);
	XCTAssertEqual([self.hub downloadStatusForModelSource:second].presence, NFKDownloadPresenceComplete,
		@"the other model still loads");
	XCTAssertEqual([self.hub downloadStatusForModelSource:first].presence, NFKDownloadPresencePartial,
		@"the kept tokenizer is all that is left of it");
}

- (void)testAFileOfAModelWhoseDownloadIsIncompleteIsNotKept
{
	[self place:4096 inRepo:@"org/shared" path:@"tokenizer.json"];
	[self place:8192 inRepo:@"org/shared" path:@"a.bin"];
	NFKModelSource *first = [self sourceForRepo:@"org/shared" paths:@[ @"a.bin", @"tokenizer.json" ] optional:@[]];
	[self addModel:@"model-b" source:[self sourceForRepo:@"org/shared" paths:@[ @"b.bin", @"tokenizer.json" ] optional:@[]]];
	NFKRemovalPlan *plan = [self.hub removalPlanForModelSource:first error:NULL];
	XCTAssertEqual(plan.removedFiles.count, (NSUInteger)2);
	XCTAssertEqual(plan.keptFiles.count, (NSUInteger)0);
}

- (void)testASharedLinkTargetStaysUntilItsLastLinkGoes
{
	NSURL *blob = [self.cacheDir URLByAppendingPathComponent:@"blobs/shared.bin"];
	[NSFileManager.defaultManager createDirectoryAtURL:blob.URLByDeletingLastPathComponent withIntermediateDirectories:YES attributes:nil error:NULL];
	XCTAssertTrue([[self bytes:16384] writeToURL:blob atomically:YES]);
	[self linkInRepo:@"org/one" path:@"model.bin" to:blob];
	[self linkInRepo:@"org/two" path:@"model.bin" to:blob];

	long long before = [self.hub cacheSize];
	NFKRemovalPlan *plan = [self.hub removalPlanForRepo:@"org/one" revision:nil paths:nil error:NULL];
	XCTAssertEqual(plan.keptFiles.count, (NSUInteger)1);
	XCTAssertEqualObjects(plan.keptFiles.firstObject.path, @"blobs/shared.bin");
	XCTAssertTrue([plan.keptFiles.firstObject.reason containsString:@"linked from 1 other"], @"%@", plan.keptFiles.firstObject.reason);
	NFKRemovalPlan *removal = [self.hub removeFilesInPlan:plan error:NULL];
	XCTAssertEqual(before - [self.hub cacheSize], removal.freedByteCount);
	XCTAssertTrue([NSFileManager.defaultManager fileExistsAtPath:blob.path]);
	XCTAssertEqual([self.hub downloadStatusForRepo:@"org/two" revision:nil paths:@[ @"model.bin" ]].presence, NFKDownloadPresenceComplete);

	before = [self.hub cacheSize];
	plan = [self.hub removalPlanForRepo:@"org/two" revision:nil paths:nil error:NULL];
	XCTAssertEqual(plan.keptFiles.count, (NSUInteger)0);
	XCTAssertTrue([[self pathsOf:plan.removedFiles] containsObject:@"blobs/shared.bin"]);
	removal = [self.hub removeFilesInPlan:plan error:NULL];
	XCTAssertEqual(before - [self.hub cacheSize], removal.freedByteCount);
	XCTAssertGreaterThanOrEqual(removal.freedByteCount, 16384LL);
	XCTAssertFalse([NSFileManager.defaultManager fileExistsAtPath:blob.path]);
}

- (void)testALinkTargetOutsideTheCacheIsKept
{
	NSURL *outside = [self.outsideDir URLByAppendingPathComponent:@"store.bin"];
	XCTAssertTrue([[self bytes:8192] writeToURL:outside atomically:YES]);
	[self linkInRepo:@"org/linked" path:@"model.bin" to:outside];

	NFKRemovalPlan *plan = [self.hub removalPlanForRepo:@"org/linked" revision:nil paths:nil error:NULL];
	XCTAssertEqual(plan.keptFiles.count, (NSUInteger)1);
	XCTAssertEqualObjects(plan.keptFiles.firstObject.reason, @"outside the cache folder");
	XCTAssertNotNil([self.hub removeFilesInPlan:plan error:NULL]);
	XCTAssertTrue([NSFileManager.defaultManager fileExistsAtPath:outside.path]);
	XCTAssertFalse([NSFileManager.defaultManager fileExistsAtPath:[self.cacheDir URLByAppendingPathComponent:@"org"].path]);
}

#pragma mark Removal

- (void)testRemovalFreesTheReportedBytesAndLeavesTheOtherModelLoadable
{
	[self place:4096 inRepo:@"org/first" path:@"config.json"];
	[self place:65536 inRepo:@"org/first" path:@"model.bin"];
	[self place:4096 inRepo:@"org/second" path:@"config.json"];
	[self place:32768 inRepo:@"org/second" path:@"model.bin"];
	NFKModelSource *second = [self sourceForRepo:@"org/second" paths:@[ @"config.json", @"model.bin" ] optional:@[]];

	long long before = [self.hub cacheSize];
	NFKRemovalPlan *plan = [self.hub removalPlanForRepo:@"org/first" revision:nil paths:nil error:NULL];
	XCTAssertGreaterThan(plan.freedByteCount, 65536LL);
	NSError *error = nil;
	NFKRemovalPlan *removal = [self.hub removeFilesInPlan:plan error:&error];
	XCTAssertNotNil(removal, @"%@", error);
	XCTAssertEqual(removal.freedByteCount, plan.freedByteCount);
	XCTAssertEqual(before - [self.hub cacheSize], removal.freedByteCount);

	XCTAssertFalse([NSFileManager.defaultManager fileExistsAtPath:[self.cacheDir URLByAppendingPathComponent:@"org/first"].path],
		@"the emptied snapshot and its folders go");
	XCTAssertEqual([self.hub downloadStatusForModelSource:second].presence, NFKDownloadPresenceComplete);
	NSURL *weights = [self.hub localURLForRepo:@"org/second" revision:nil path:@"model.bin"];
	XCTAssertEqual([NSData dataWithContentsOfURL:weights].length, (NSUInteger)32768);
	NSArray<NSString *> *left = [NSFileManager.defaultManager contentsOfDirectoryAtPath:self.cacheDir.path error:NULL];
	NSArray<NSString *> *expected = @[ @"org" ];
	XCTAssertEqualObjects(left, expected, @"no trash folder remains");
}

- (void)testRemovalKeepsAPin
{
	[self place:4096 inRepo:@"org/pinned" path:@"model.bin"];
	XCTAssertTrue([self.hub pinCachedRepo:@"org/pinned" revision:nil error:NULL]);
	NFKRemovalPlan *removal = [self.hub removeFilesInPlan:[self.hub removalPlanForRepo:@"org/pinned" revision:nil paths:nil error:NULL] error:NULL];
	XCTAssertNotNil(removal);
	XCTAssertFalse([self.hub isCachedRepo:@"org/pinned" revision:nil path:@"model.bin"]);
	XCTAssertTrue([self.hub isCachedRepoPinned:@"org/pinned" revision:nil], @"the app's pin is not a downloaded file");
	XCTAssertEqual([self.hub downloadStatusForRepo:@"org/pinned" revision:nil paths:nil].presence, NFKDownloadPresenceAbsent);
}

- (void)testAURLDownloadIsRemovedWithItsFolder
{
	NSURL *URL = [NSURL URLWithString:@"https://example.com/models/model.bin"];
	NSURL *local = [self.hub localURLForURL:URL fileName:nil];
	[NSFileManager.defaultManager createDirectoryAtURL:local.URLByDeletingLastPathComponent withIntermediateDirectories:YES attributes:nil error:NULL];
	XCTAssertTrue([[self bytes:4096] writeToURL:local atomically:YES]);
	XCTAssertEqual([self.hub downloadStatusForURL:URL].presence, NFKDownloadPresenceComplete);

	NFKRemovalPlan *plan = [self.hub removalPlanForURL:URL error:NULL];
	XCTAssertEqual(plan.removedFiles.count, (NSUInteger)1);
	XCTAssertNotNil([self.hub removeFilesInPlan:plan error:NULL]);
	XCTAssertEqual([self.hub downloadStatusForURL:URL].presence, NFKDownloadPresenceAbsent);
	XCTAssertFalse([NSFileManager.defaultManager fileExistsAtPath:[self.cacheDir URLByAppendingPathComponent:@"@url"].path]);
}

- (void)testRemovalIsRefusedWhileADownloadWritesIntoTheSnapshot
{
	[self place:4096 inRepo:@"org/model" path:@"config.json"];
	NSMutableData *weights = [NSMutableData dataWithLength:64 * 1024];
	NSDictionary<NSString *, NFKStubHubResponse *> *responses =
		[NFKStubHubProtocol serveRepo:@"org/model" revision:@"main" files:@{ @"model.bin": weights } lfsPaths:[NSSet set]];
	responses[@"model.bin"].chunkSize = 4096;
	responses[@"model.bin"].stallsAfterFirstChunk = YES;
	NFKDownload *download = [self.hub downloadRepo:@"org/model" revision:nil paths:@[ @"model.bin" ] progressHandler:nil completionHandler:nil];
	NSDate *deadline = [NSDate dateWithTimeIntervalSinceNow:10];
	while (download.currentProgress.receivedByteCount == 0 && deadline.timeIntervalSinceNow > 0) {
		[NSThread sleepForTimeInterval:0.01];
	}
	XCTAssertGreaterThan(download.currentProgress.receivedByteCount, 0LL);

	NFKRemovalPlan *plan = [self.hub removalPlanForRepo:@"org/model" revision:nil paths:nil error:NULL];
	NSError *error = nil;
	XCTAssertNil([self.hub removeFilesInPlan:plan error:&error]);
	XCTAssertEqualObjects(error.domain, NFKDownloadErrorDomain);
	XCTAssertEqual(error.code, NFKDownloadErrorInUse);
	XCTAssertTrue([self.hub isCachedRepo:@"org/model" revision:nil path:@"config.json"], @"nothing was removed");

	[download cancel];
	XCTAssertTrue([download waitUntilFinishedWithTimeout:10]);
	error = nil;
	XCTAssertNotNil([self.hub removeFilesInPlan:plan error:&error], @"%@", error);
	XCTAssertFalse([self.hub isCachedRepo:@"org/model" revision:nil path:@"config.json"]);
}

- (void)testTheAsynchronousRemovalDeliversTheBytesFreed
{
	[self place:8192 inRepo:@"org/model" path:@"model.bin"];
	NFKRemovalPlan *plan = [self.hub removalPlanForRepo:@"org/model" revision:nil paths:@[ @"*.bin" ] error:NULL];
	XCTAssertEqual(plan.removedFiles.count, (NSUInteger)1, @"a glob matches the cached files");
	XCTestExpectation *done = [self expectationWithDescription:@"removal"];
	[self.hub removeFilesInPlan:plan completionHandler:^(NFKRemovalPlan *removal, NSError *error) {
		XCTAssertNil(error);
		XCTAssertEqual(removal.freedByteCount, plan.freedByteCount);
		[done fulfill];
	}];
	[self waitForExpectations:@[ done ] timeout:10];
	XCTAssertFalse([self.hub isCachedRepo:@"org/model" revision:nil path:@"model.bin"]);
}

- (void)testASourceWithNoFilesPlansAnEmptyRemoval
{
	NFKModelSource *source = [[NFKModelSource alloc] initWithKind:NFKModelSourceKindNone repository:nil revision:nil URL:nil
														   layout:nil files:nil requiresConversion:NO extensions:nil];
	NFKRemovalPlan *plan = [self.hub removalPlanForModelSource:source error:NULL];
	XCTAssertEqual(plan.removedFiles.count, (NSUInteger)0);
	XCTAssertEqual(plan.freedByteCount, 0LL);
	XCTAssertNotNil([self.hub removeFilesInPlan:plan error:NULL]);
}

#pragma mark Files other snapshots use

- (void)testRemovingASnapshotKeepsAFileOfAnotherSnapshotItsLinkResolvesTo
{
	NSURL *shared = [self place:16384 inRepo:@"org/two" path:@"model.bin"];
	XCTAssertTrue([self.hub recordCompleteDownloadOfRepo:@"org/two" revision:nil paths:@[ @"model.bin" ] error:NULL]);
	[self linkInRepo:@"org/one" path:@"model.bin" to:shared];

	NFKRemovalPlan *plan = [self.hub removalPlanForRepo:@"org/one" revision:nil paths:nil error:NULL];
	XCTAssertEqual(plan.keptFiles.count, (NSUInteger)1);
	XCTAssertEqualObjects(plan.keptFiles.firstObject.path, @"org/two/main/model.bin");
	XCTAssertEqualObjects(plan.keptFiles.firstObject.reason, @"a file of another cached snapshot");
	XCTAssertFalse([[self pathsOf:plan.removedFiles] containsObject:@"org/two/main/model.bin"]);
	NSError *error = nil;
	NFKRemovalPlan *removal = [self.hub removeFilesInPlan:plan error:&error];
	XCTAssertNotNil(removal, @"%@", error);

	XCTAssertEqual([NSData dataWithContentsOfURL:shared].length, (NSUInteger)16384);
	NFKDownloadStatus *status = [self.hub downloadStatusForRepo:@"org/two" revision:nil paths:nil];
	XCTAssertEqual(status.presence, NFKDownloadPresenceComplete, @"the other snapshot's record still matches its files");
	XCTAssertEqual([self.hub downloadStatusForRepo:@"org/one" revision:nil paths:nil].presence, NFKDownloadPresenceAbsent);
}

- (void)testRemovingASnapshotKeepsItsFileWhileAnotherSnapshotLinksToIt
{
	NSURL *original = [self place:16384 inRepo:@"org/one" path:@"model.bin"];
	NSURL *link = [self linkInRepo:@"org/two" path:@"model.bin" to:original];
	XCTAssertTrue([self.hub recordCompleteDownloadOfRepo:@"org/two" revision:nil paths:@[ @"model.bin" ] error:NULL]);

	NFKRemovalPlan *plan = [self.hub removalPlanForRepo:@"org/one" revision:nil paths:nil error:NULL];
	XCTAssertEqual(plan.removedFiles.count, (NSUInteger)0);
	XCTAssertEqual(plan.keptFiles.count, (NSUInteger)1);
	XCTAssertTrue([plan.keptFiles.firstObject.reason containsString:@"linked from 1 other"], @"%@", plan.keptFiles.firstObject.reason);
	XCTAssertEqual(plan.freedByteCount, 0LL);
	XCTAssertNotNil([self.hub removeFilesInPlan:plan error:NULL]);

	XCTAssertEqual([NSData dataWithContentsOfURL:link].length, (NSUInteger)16384, @"the link still resolves");
	XCTAssertEqual([self.hub downloadStatusForRepo:@"org/two" revision:nil paths:nil].presence, NFKDownloadPresenceComplete);
}

#pragma mark Unsafe names

/*! A repository whose snapshot, <cache>/<repo>/main, is the outside folder's main/, holding victim.bin. */
- (NSString *)hostileRepoWithVictim
{
	NSURL *victim = [[self.outsideDir URLByAppendingPathComponent:@"main"] URLByAppendingPathComponent:@"victim.bin"];
	[NSFileManager.defaultManager createDirectoryAtURL:victim.URLByDeletingLastPathComponent withIntermediateDirectories:YES attributes:nil error:NULL];
	XCTAssertTrue([[self bytes:4096] writeToURL:victim atomically:YES]);
	return [@"../" stringByAppendingString:self.outsideDir.lastPathComponent];
}

- (void)assertTheVictimIsUntouched
{
	NSString *folder = [[self.outsideDir URLByAppendingPathComponent:@"main"] path];
	NSArray<NSString *> *contents = [NSFileManager.defaultManager contentsOfDirectoryAtPath:folder error:NULL];
	XCTAssertEqualObjects(contents, @[ @"victim.bin" ], @"nothing was written into or removed from the outside folder");
	XCTAssertEqualObjects([NSData dataWithContentsOfURL:[NSURL fileURLWithPath:[folder stringByAppendingPathComponent:@"victim.bin"]]],
						  [self bytes:4096]);
}

- (void)testAnUnsafeRepositoryReadsAbsentAndItsRemovalIsRefused
{
	NSString *repo = [self hostileRepoWithVictim];
	NFKDownloadStatus *status = [self.hub downloadStatusForRepo:repo revision:nil paths:nil];
	XCTAssertEqual(status.presence, NFKDownloadPresenceAbsent, @"the status does not read outside the cache");
	XCTAssertEqual(status.byteCount, 0LL);
	NFKModelSource *source = [self sourceForRepo:repo paths:@[ @"victim.bin" ] optional:@[]];
	XCTAssertEqual([self.hub downloadStatusForModelSource:source].presence, NFKDownloadPresenceAbsent);

	NSError *error = nil;
	XCTAssertNil([self.hub removalPlanForRepo:repo revision:nil paths:nil error:&error]);
	XCTAssertEqualObjects(error.domain, NFKDownloadErrorDomain);
	XCTAssertEqual(error.code, NFKDownloadErrorUnsafePath, @"%@", error);
	error = nil;
	XCTAssertNil([self.hub removalPlanForModelSource:source error:&error]);
	XCTAssertEqual(error.code, NFKDownloadErrorUnsafePath, @"%@", error);
	[self assertTheVictimIsUntouched];
}

- (void)testAnUnsafeRepositoryOrPathIsNotRecorded
{
	NSString *repo = [self hostileRepoWithVictim];
	NSError *error = nil;
	XCTAssertFalse([self.hub recordCompleteDownloadOfRepo:repo revision:nil paths:@[ @"victim.bin" ] error:&error]);
	XCTAssertEqual(error.code, NFKDownloadErrorUnsafePath, @"%@", error);
	error = nil;
	XCTAssertFalse([self.hub recordCompleteDownloadOfRepo:@"org/model" revision:nil paths:@[ @"../escape.bin" ] error:&error]);
	XCTAssertEqual(error.code, NFKDownloadErrorUnsafePath, @"%@", error);
	[self assertTheVictimIsUntouched];
	XCTAssertFalse([NSFileManager.defaultManager fileExistsAtPath:[self.cacheDir URLByAppendingPathComponent:@"org"].path]);
}

- (void)testAURLWhoseFileNameLeavesItsFolderIsRefused
{
	NSURL *URL = [NSURL URLWithString:@"https://example.com/models/.."];
	XCTAssertNil([self.hub localURLForURL:URL fileName:nil]);
	XCTAssertNil([self.hub localURLForURL:[NSURL URLWithString:@"https://example.com/models/model.bin"] fileName:@".."]);
	XCTAssertEqual([self.hub downloadStatusForURL:URL].presence, NFKDownloadPresenceAbsent);
}

#pragma mark Sources spanning repositories

/*! org/controlnet's controlnet.bin with org/base's vae.bin read from org/base. */
- (NFKModelSource *)controlNetSource
{
	NFKModelFile *controlNet = [NFKModelFile fileWithPath:@"controlnet.bin" role:NFKModelFileRoleWeights];
	NFKModelFile *vae = [[NFKModelFile alloc] initWithPath:@"vae.bin" byteCount:nil sha256:nil role:NFKModelFileRoleWeights
												  optional:NO repository:@"org/base" revision:nil URL:nil extensions:nil];
	return [[NFKModelSource alloc] initWithKind:NFKModelSourceKindHuggingFace repository:@"org/controlnet" revision:nil URL:nil
										 layout:NFKModelSourceLayoutDirectory files:@[ controlNet, vae ] requiresConversion:NO extensions:nil];
}

- (void)testASourceSpanningRepositoriesIsCompleteOnlyWhenEveryRepositoryIs
{
	NFKModelSource *source = [self controlNetSource];
	XCTAssertEqual([self.hub downloadStatusForModelSource:source].presence, NFKDownloadPresenceAbsent);
	[self place:4096 inRepo:@"org/controlnet" path:@"controlnet.bin"];
	NFKDownloadStatus *status = [self.hub downloadStatusForModelSource:source];
	XCTAssertEqual(status.presence, NFKDownloadPresencePartial, @"the base repository's file is missing");
	XCTAssertEqualObjects(status.missingPaths, @[ @"vae.bin" ]);
	XCTAssertTrue([self.hub isCachedRepo:@"org/controlnet" revision:nil path:@"controlnet.bin"]);
	XCTAssertFalse([self.hub isCachedRepo:@"org/controlnet" revision:nil path:@"vae.bin"]);

	[self place:8192 inRepo:@"org/base" path:@"vae.bin"];
	status = [self.hub downloadStatusForModelSource:source];
	XCTAssertEqual(status.presence, NFKDownloadPresenceComplete);
	XCTAssertGreaterThanOrEqual(status.byteCount, 4096LL + 8192LL, @"bytes count every repository");
}

- (void)testRemovingASourceSpanningRepositoriesKeepsTheBaseFilesAnotherModelUses
{
	[self place:4096 inRepo:@"org/controlnet" path:@"controlnet.bin"];
	NSURL *vae = [self place:8192 inRepo:@"org/base" path:@"vae.bin"];
	NSURL *unet = [self place:16384 inRepo:@"org/base" path:@"unet.bin"];
	NFKModelSource *base = [self sourceForRepo:@"org/base" paths:@[ @"vae.bin", @"unet.bin" ] optional:@[]];
	[self addModel:@"base-model" source:base];
	NFKModelSource *source = [self controlNetSource];
	[self addModel:@"controlnet-model" source:source];

	NFKRemovalPlan *plan = [self.hub removalPlanForModelSource:source error:NULL];
	XCTAssertEqualObjects([self pathsOf:plan.removedFiles], @[ @"org/controlnet/main/controlnet.bin" ]);
	XCTAssertEqualObjects([self pathsOf:plan.keptFiles], @[ @"org/base/main/vae.bin" ]);
	XCTAssertTrue([plan.keptFiles.firstObject.reason containsString:@"base-model"], @"%@", plan.keptFiles.firstObject.reason);
	long long before = [self.hub cacheSize];
	NSError *error = nil;
	NFKRemovalPlan *removal = [self.hub removeFilesInPlan:plan error:&error];
	XCTAssertNotNil(removal, @"%@", error);
	XCTAssertEqual(before - [self.hub cacheSize], removal.freedByteCount);
	XCTAssertEqual([NSData dataWithContentsOfURL:vae].length, (NSUInteger)8192);
	XCTAssertEqual([NSData dataWithContentsOfURL:unet].length, (NSUInteger)16384);
	XCTAssertEqual([self.hub downloadStatusForModelSource:base].presence, NFKDownloadPresenceComplete);
	XCTAssertFalse([NSFileManager.defaultManager fileExistsAtPath:[self.cacheDir URLByAppendingPathComponent:@"org/controlnet"].path]);

	[self place:4096 inRepo:@"org/controlnet" path:@"controlnet.bin"];
	plan = [self.hub removalPlanForModelSource:base error:NULL];
	XCTAssertEqualObjects([self pathsOf:plan.removedFiles], @[ @"org/base/main/unet.bin" ]);
	XCTAssertEqualObjects([self pathsOf:plan.keptFiles], @[ @"org/base/main/vae.bin" ],
		@"the base model's removal keeps the file the complete control net reads from its repository");
}

- (void)testAnUnsafeOverrideRepositoryIsRefusedAndReadsNothingOutsideTheCache
{
	NSString *repo = [self hostileRepoWithVictim];
	[self place:4096 inRepo:@"org/model" path:@"model.bin"];
	NFKModelFile *own = [NFKModelFile fileWithPath:@"model.bin" role:NFKModelFileRoleWeights];
	NFKModelFile *hostile = [[NFKModelFile alloc] initWithPath:@"victim.bin" byteCount:nil sha256:nil role:nil optional:NO
													repository:repo revision:nil URL:nil extensions:nil];
	NFKModelSource *source = [[NFKModelSource alloc] initWithKind:NFKModelSourceKindHuggingFace repository:@"org/model" revision:nil
															  URL:nil layout:NFKModelSourceLayoutDirectory files:@[ own, hostile ]
											   requiresConversion:NO extensions:nil];
	NFKDownloadStatus *status = [self.hub downloadStatusForModelSource:source];
	XCTAssertEqual(status.presence, NFKDownloadPresencePartial, @"the outside file does not count as present");
	XCTAssertEqualObjects(status.missingPaths, @[ @"victim.bin" ]);

	NSError *error = nil;
	XCTAssertNil([self.hub removalPlanForModelSource:source error:&error]);
	XCTAssertEqual(error.code, NFKDownloadErrorUnsafePath, @"%@", error);
	[self assertTheVictimIsUntouched];
	XCTAssertTrue([self.hub isCachedRepo:@"org/model" revision:nil path:@"model.bin"]);
}

@end
