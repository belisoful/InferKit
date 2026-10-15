//
//  NFKDownloadRemovalTests.m
//  InferKitTests
//
//  Download status and removal on a temporary cache: complete, partial, and absent; the files a
//  removal plan names; files another complete model uses and shared link targets kept; the bytes
//  freed; a pin kept; and a removal refused while a download writes into the snapshot.
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

@end
