//
//  NFKDownloadTests.m
//  InferKitTests
//
//  Download plans and observable downloads against the in-process hub at http://hub.test: sizes and
//  digests from the listing, globs, cache detection, progress, cancellation, the cache size limit,
//  resumption, gated repositories, catalog model sources and the files they read from other
//  repositories, listed sizes, unsafe names, and eviction while a download runs.
//

#import <XCTest/XCTest.h>
#import <InferKit/InferKit.h>
#import "NFKStubHubProtocol.h"
#import <sys/xattr.h>

static NSString * const NFKTestRepo = @"org/model";

/*! A hub that holds the fetch of one file until the test opens the gate, so a download stays
	between its files. */
@interface NFKGatedHub : NFKHFHub
@property (nonatomic, copy) NSString *gatedFileName;
@property (nonatomic, strong) dispatch_semaphore_t arrived;
@property (nonatomic, strong) dispatch_semaphore_t gate;
@end

@implementation NFKGatedHub

- (BOOL)fetchURL:(NSURL *)remoteURL toFileURL:(NSURL *)destinationURL error:(NSError **)outError
{
	if ([remoteURL.lastPathComponent isEqualToString:self.gatedFileName]) {
		dispatch_semaphore_signal(self.arrived);
		dispatch_semaphore_wait(self.gate, dispatch_time(DISPATCH_TIME_NOW, 20 * (int64_t)NSEC_PER_SEC));
	}
	return [super fetchURL:remoteURL toFileURL:destinationURL error:outError];
}

@end

@interface NFKDownloadTests : XCTestCase
@property (nonatomic, strong) NSURL *cacheDir;
@property (nonatomic, strong) NFKHFHub *hub;
@property (nonatomic, strong) NSData *weights;
@property (nonatomic, strong) NSData *config;
@property (nonatomic, copy) NSString *savedDefaultToken;
@end

@implementation NFKDownloadTests

- (void)setUp
{
	[super setUp];
	[NFKStubHubProtocol reset];
	self.savedDefaultToken = NFKHFHub.defaultAccessToken;
	NFKHFHub.defaultAccessToken = nil;
	NSString *name = [@"NFKDownloadTests-" stringByAppendingString:NSUUID.UUID.UUIDString];
	self.cacheDir = [NSURL fileURLWithPath:[NSTemporaryDirectory() stringByAppendingPathComponent:name]];
	[NSFileManager.defaultManager createDirectoryAtURL:self.cacheDir withIntermediateDirectories:YES attributes:nil error:NULL];

	self.hub = [NFKHFHub hubWithCacheDirectoryURL:self.cacheDir];
	self.hub.endpointURL = NFKStubHubProtocol.endpointURL;
	self.hub.session = [NFKStubHubProtocol session];
	self.hub.accessToken = @"";
	self.hub.excludesCacheFromBackup = NO;

	NSMutableData *weights = [NSMutableData dataWithLength:64 * 1024];
	uint8_t *bytes = weights.mutableBytes;
	for (NSUInteger i = 0; i < weights.length; i++) {
		bytes[i] = (uint8_t)(i * 31 + 7);
	}
	self.weights = weights;
	self.config = [@"{\"hidden_size\": 8}" dataUsingEncoding:NSUTF8StringEncoding];
}

- (void)tearDown
{
	[NSFileManager.defaultManager removeItemAtURL:self.cacheDir error:NULL];
	[NSFileManager.defaultManager removeItemAtURL:[NSURL fileURLWithPath:[self.cacheDir.path stringByAppendingString:@"-outside"]] error:NULL];
	NFKHFHub.defaultAccessToken = self.savedDefaultToken;
	[NFKStubHubProtocol reset];
	[super tearDown];
}

- (NSDictionary<NSString *, NFKStubHubResponse *> *)serveTheModel
{
	NSDictionary *files = @{ @"config.json": self.config, @"model.safetensors": self.weights,
							 @"tokenizer/vocab.json": [@"{}" dataUsingEncoding:NSUTF8StringEncoding] };
	NSDictionary<NSString *, NFKStubHubResponse *> *responses =
		[NFKStubHubProtocol serveRepo:NFKTestRepo revision:@"main" files:files lfsPaths:[NSSet setWithObject:@"model.safetensors"]];
	responses[@"model.safetensors"].chunkSize = 4096;
	return responses;
}

- (NFKDownloadPlan *)planPaths:(nullable NSArray<NSString *> *)paths
{
	NSError *error = nil;
	NFKDownloadPlan *plan = [self.hub planDownloadOfRepo:NFKTestRepo revision:nil paths:paths error:&error];
	XCTAssertNotNil(plan, @"%@", error);
	return plan;
}

/*! Runs a download to its end and returns its progress reports, in order. */
- (NSArray<NFKDownloadProgress *> *)runDownloadOfPlan:(NFKDownloadPlan *)plan
											 download:(NFKDownload * __strong *)outDownload
												error:(NSError * __strong *)outError
{
	NSMutableArray<NFKDownloadProgress *> *reports = [NSMutableArray array];
	XCTestExpectation *done = [self expectationWithDescription:@"download"];
	__block NSError *finalError = nil;
	NFKDownload *download = [self.hub downloadWithPlan:plan progressHandler:^(NFKDownloadProgress *progress) {
		@synchronized (reports) {
			[reports addObject:progress];
		}
	} completionHandler:^(NSURL *localURL, NSError *error) {
		finalError = error;
		[done fulfill];
	}];
	[self waitForExpectations:@[ done ] timeout:20];
	if (outDownload != NULL) {
		*outDownload = download;
	}
	if (outError != NULL) {
		*outError = finalError;
	}
	@synchronized (reports) {
		return [reports copy];
	}
}

#pragma mark Plans

- (void)testAPlanListsEachFilesSizeAndTheLargeFileDigest
{
	[self serveTheModel];
	NFKDownloadPlan *plan = [self planPaths:nil];

	XCTAssertEqual(plan.files.count, (NSUInteger)3, @"directories in the listing are not files");
	NFKDownloadPlanFile *weights = [plan fileAtPath:@"model.safetensors"];
	XCTAssertEqual(weights.byteCount, (long long)self.weights.length);
	XCTAssertEqualObjects(weights.sha256, [NFKStubHubProtocol sha256OfData:self.weights]);
	XCTAssertNil([plan fileAtPath:@"config.json"].sha256, @"a Git object ID is not a SHA-256 of the contents");
	XCTAssertEqualObjects(weights.remoteURL.absoluteString, @"http://hub.test/org/model/resolve/main/model.safetensors");
	XCTAssertEqual(plan.totalByteCount, (long long)(self.weights.length + self.config.length + 2));
	XCTAssertEqual(plan.remainingByteCount, plan.totalByteCount);
	XCTAssertTrue(plan.sizesKnown);
	XCTAssertFalse(plan.cached);
	XCTAssertEqualObjects(plan.revision, @"main");
	XCTAssertEqual([NFKStubHubProtocol requestsForPath:@"/api/models/org/model/tree/main"].count, (NSUInteger)1);
}

- (void)testAPlanSelectsFilesByGlob
{
	[self serveTheModel];
	NFKDownloadPlan *plan = [self planPaths:@[ @"*.json" ]];
	NSArray<NSString *> *paths = [plan.files valueForKey:@"path"];
	NSArray<NSString *> *expected = @[ @"config.json", @"tokenizer/vocab.json" ];
	XCTAssertEqualObjects(paths, expected, @"a star also matches a slash");
}

- (void)testAPlanFailsOnAFileTheRepositoryLacks
{
	[self serveTheModel];
	NSError *error = nil;
	XCTAssertNil([self.hub planDownloadOfRepo:NFKTestRepo revision:nil paths:@[ @"absent.bin" ] error:&error]);
	XCTAssertEqualObjects(error.domain, NFKDownloadErrorDomain);
	XCTAssertEqual(error.code, NFKDownloadErrorNotFound);
}

- (void)testAPlanCountsOnlyTheFilesTheCacheLacks
{
	[self serveTheModel];
	NSURL *cached = [self.hub localURLForRepo:NFKTestRepo revision:nil path:@"config.json"];
	[NSFileManager.defaultManager createDirectoryAtURL:cached.URLByDeletingLastPathComponent withIntermediateDirectories:YES attributes:nil error:NULL];
	XCTAssertTrue([self.config writeToURL:cached atomically:YES]);
	NSURL *truncated = [self.hub localURLForRepo:NFKTestRepo revision:nil path:@"model.safetensors"];
	XCTAssertTrue([[self.weights subdataWithRange:NSMakeRange(0, 10)] writeToURL:truncated atomically:YES]);

	NFKDownloadPlan *plan = [self planPaths:nil];
	XCTAssertTrue([plan fileAtPath:@"config.json"].cached);
	XCTAssertFalse([plan fileAtPath:@"model.safetensors"].cached, @"a file of the wrong size is not whole");
	XCTAssertEqual(plan.remainingByteCount, (long long)(self.weights.length + 2));
}

- (void)testAPlanFollowsTheListingsNextPage
{
	[self serveTheModel];
	NSArray *first = @[ @{ @"type": @"file", @"path": @"config.json", @"size": @(self.config.length) } ];
	NFKStubHubResponse *page = [NFKStubHubResponse responseWithStatus:200 body:[NSJSONSerialization dataWithJSONObject:first options:0 error:NULL]];
	page.headers = @{ @"Link": @"<http://hub.test/api/models/org/model/tree-page-2>; rel=\"next\"" };
	[NFKStubHubProtocol setResponse:page forPath:@"/api/models/org/model/tree/main"];
	NSArray *second = @[ @{ @"type": @"file", @"path": @"model.safetensors", @"size": @(self.weights.length) } ];
	[NFKStubHubProtocol setResponse:[NFKStubHubResponse responseWithStatus:200 body:[NSJSONSerialization dataWithJSONObject:second options:0 error:NULL]]
							forPath:@"/api/models/org/model/tree-page-2"];

	NFKDownloadPlan *plan = [self planPaths:nil];
	NSArray<NSString *> *expected = @[ @"config.json", @"model.safetensors" ];
	XCTAssertEqualObjects([plan.files valueForKey:@"path"], expected);
}

- (void)testAGatedRepositoryReportsThatItNeedsAnAccessToken
{
	[NFKStubHubProtocol setResponse:[NFKStubHubResponse responseWithStatus:401 body:nil] forPath:@"/api/models/org/gated/tree/main"];
	NSError *error = nil;
	XCTAssertNil([self.hub planDownloadOfRepo:@"org/gated" revision:nil paths:nil error:&error]);
	XCTAssertEqualObjects(error.domain, NFKDownloadErrorDomain);
	XCTAssertEqual(error.code, NFKDownloadErrorAuthenticationRequired);
	XCTAssertTrue([error.localizedDescription containsString:@"access token"], @"%@", error.localizedDescription);
	XCTAssertEqualObjects(error.userInfo[NFKDownloadHTTPStatusCodeKey], @401);
}

- (void)testAGatedRepositoryWhoseTermsAreNotAcceptedReportsAnAuthenticationError
{
	self.hub.accessToken = @"hf_test";
	[NFKStubHubProtocol setResponse:[NFKStubHubResponse responseWithStatus:403 body:nil] forPath:@"/api/models/org/gated/tree/main"];
	NSError *error = nil;
	XCTAssertNil([self.hub planDownloadOfRepo:@"org/gated" revision:nil paths:nil error:&error]);
	XCTAssertEqual(error.code, NFKDownloadErrorAuthenticationRequired);
	NSURLRequest *request = [NFKStubHubProtocol requestsForPath:@"/api/models/org/gated/tree/main"].firstObject;
	XCTAssertEqualObjects([request valueForHTTPHeaderField:@"Authorization"], @"Bearer hf_test");
}

- (void)testTheAccessTokenGoesOnlyToTheHub
{
	self.hub.accessToken = @"hf_test";
	NFKStubHubResponse *file = [NFKStubHubResponse responseWithStatus:200 body:self.config];
	[NFKStubHubProtocol setResponse:file forPath:@"/files/config.json"];
	self.hub.endpointURL = [NSURL URLWithString:@"https://huggingface.co"];
	NSError *error = nil;
	XCTAssertNotNil([self.hub planDownloadOfURL:[NSURL URLWithString:@"http://hub.test/files/config.json"] sha256:nil error:&error], @"%@", error);
	NSURLRequest *request = [NFKStubHubProtocol requestsForPath:@"/files/config.json"].firstObject;
	XCTAssertNil([request valueForHTTPHeaderField:@"Authorization"]);
}

- (void)testAURLPlanTakesItsSizeFromAHeadRequest
{
	[NFKStubHubProtocol setResponse:[NFKStubHubResponse responseWithStatus:200 body:self.weights] forPath:@"/files/model.bin"];
	NSURL *URL = [NSURL URLWithString:@"http://hub.test/files/model.bin"];
	NSError *error = nil;
	NFKDownloadPlan *plan = [self.hub planDownloadOfURL:URL sha256:nil error:&error];
	XCTAssertNotNil(plan, @"%@", error);
	XCTAssertEqual(plan.files.firstObject.byteCount, (long long)self.weights.length);
	XCTAssertEqualObjects(plan.sourceURL, URL);
	XCTAssertEqualObjects(plan.localURL, [self.hub localURLForURL:URL fileName:nil]);
	XCTAssertTrue([plan.localURL.path containsString:@"/@url/hub.test/"]);
	XCTAssertEqualObjects([NFKStubHubProtocol requestsForPath:@"/files/model.bin"].firstObject.HTTPMethod, @"HEAD");
}

#pragma mark Downloads

- (void)testADownloadReportsProgressThatOnlyGrowsAndReachesTheTotal
{
	[self serveTheModel];
	NFKDownloadPlan *plan = [self planPaths:nil];
	NFKDownload *download = nil;
	NSError *error = nil;
	NSArray<NFKDownloadProgress *> *reports = [self runDownloadOfPlan:plan download:&download error:&error];

	XCTAssertNil(error);
	XCTAssertEqual(download.state, NFKDownloadStateCompleted);
	XCTAssertEqualObjects(download.localURL, plan.localURL);
	XCTAssertGreaterThan(reports.count, (NSUInteger)2);
	long long previous = 0;
	for (NFKDownloadProgress *report in reports) {
		XCTAssertGreaterThanOrEqual(report.receivedByteCount, previous, @"received bytes never fall");
		XCTAssertLessThanOrEqual(report.receivedByteCount, report.expectedByteCount);
		previous = report.receivedByteCount;
	}
	NFKDownloadProgress *last = reports.lastObject;
	XCTAssertEqual(last.state, NFKDownloadStateCompleted);
	XCTAssertEqual(last.receivedByteCount, plan.totalByteCount);
	XCTAssertEqual(last.expectedByteCount, plan.totalByteCount);
	XCTAssertEqual(last.fractionCompleted, 1.0);
	XCTAssertEqual(last.fileCount, (NSUInteger)3);
	XCTAssertEqual(last.completedFileCount, (NSUInteger)3);
	XCTAssertNil(last.currentFile);
	BOOL sawWeightsAsCurrentFile = NO;
	for (NFKDownloadProgress *report in reports) {
		if ([report.currentFile.path isEqualToString:@"model.safetensors"]) {
			sawWeightsAsCurrentFile = YES;
			XCTAssertEqual(report.currentFile.expectedByteCount, (long long)self.weights.length);
		}
	}
	XCTAssertTrue(sawWeightsAsCurrentFile, @"the file in transfer is reported");
	XCTAssertEqual(download.progress.completedUnitCount, download.progress.totalUnitCount);

	NSURL *weightsURL = [self.hub localURLForRepo:NFKTestRepo revision:nil path:@"model.safetensors"];
	XCTAssertEqualObjects([NSData dataWithContentsOfURL:weightsURL], self.weights);
	XCTAssertEqual([self.hub downloadStatusForRepo:NFKTestRepo revision:nil paths:nil].presence, NFKDownloadPresenceComplete,
		@"a whole-repository download records its completion");
}

- (void)testADownloadSkipsTheFilesTheCacheHolds
{
	[self serveTheModel];
	NFKDownloadPlan *plan = [self planPaths:nil];
	[self runDownloadOfPlan:plan download:NULL error:NULL];
	[NFKStubHubProtocol reset];

	NFKDownload *download = nil;
	NSError *error = nil;
	NSArray<NFKDownloadProgress *> *reports = [self runDownloadOfPlan:plan download:&download error:&error];
	XCTAssertNil(error);
	XCTAssertEqual(NFKStubHubProtocol.requests.count, (NSUInteger)0, @"nothing is fetched again");
	XCTAssertEqual(reports.lastObject.completedFileCount, (NSUInteger)3);
	XCTAssertGreaterThan(reports.lastObject.receivedByteCount, 0LL, @"the cached files count their bytes");
	XCTAssertEqual(reports.lastObject.receivedByteCount, reports.lastObject.expectedByteCount);
	XCTAssertEqual(reports.lastObject.bytesPerSecond, 0.0, @"and none of them was transferred");
}

// A file the plan found in the cache and that is gone when its turn comes counts from zero while it
// transfers again.
- (void)testAFileThePlanFoundCachedAndThatIsGoneCountsFromZero
{
	[self serveTheModel];
	[self runDownloadOfPlan:[self planPaths:nil] download:NULL error:NULL];
	NFKDownloadPlan *plan = [self planPaths:nil];
	XCTAssertEqual(plan.remainingByteCount, 0LL);
	NSURL *weightsURL = [self.hub localURLForRepo:NFKTestRepo revision:nil path:@"model.safetensors"];
	XCTAssertTrue([NSFileManager.defaultManager removeItemAtURL:weightsURL error:NULL]);

	NSError *error = nil;
	NSArray<NFKDownloadProgress *> *reports = [self runDownloadOfPlan:plan download:NULL error:&error];
	XCTAssertNil(error);
	NFKDownloadProgress *start = nil;
	for (NFKDownloadProgress *report in reports) {
		if ([report.currentFile.path isEqualToString:@"model.safetensors"]) {
			start = report;
			break;
		}
	}
	XCTAssertNotNil(start);
	XCTAssertEqual(start.currentFile.receivedByteCount, 0LL);
	XCTAssertFalse(start.currentFile.isCached);
	XCTAssertLessThan(start.receivedByteCount, start.expectedByteCount);
	XCTAssertEqualObjects([NSData dataWithContentsOfURL:weightsURL], self.weights);
	XCTAssertEqual(reports.lastObject.receivedByteCount, plan.totalByteCount);
}

- (void)testADownloadVerifiesTheLargeFileDigest
{
	NSDictionary<NSString *, NFKStubHubResponse *> *responses = [self serveTheModel];
	NSMutableData *corrupt = [self.weights mutableCopy];
	((uint8_t *)corrupt.mutableBytes)[100] ^= 0xff;
	responses[@"model.safetensors"].body = corrupt;

	NFKDownload *download = nil;
	NSError *error = nil;
	[self runDownloadOfPlan:[self planPaths:@[ @"model.safetensors" ]] download:&download error:&error];
	XCTAssertEqual(download.state, NFKDownloadStateFailed);
	XCTAssertEqualObjects(error.domain, NFKDownloadErrorDomain);
	XCTAssertEqual(error.code, NFKDownloadErrorChecksumMismatch);
	XCTAssertFalse([self.hub isCachedRepo:NFKTestRepo revision:nil path:@"model.safetensors"]);
}

- (void)testCancellingADownloadLeavesNoFileInTheCache
{
	NSDictionary<NSString *, NFKStubHubResponse *> *responses = [self serveTheModel];
	responses[@"model.safetensors"].stallsAfterFirstChunk = YES;
	NFKDownloadPlan *plan = [self planPaths:@[ @"model.safetensors" ]];

	XCTestExpectation *done = [self expectationWithDescription:@"download ends"];
	__block NSError *finalError = nil;
	NFKDownload *download = [self.hub downloadWithPlan:plan progressHandler:nil completionHandler:^(NSURL *localURL, NSError *error) {
		finalError = error;
		[done fulfill];
	}];
	NSDate *deadline = [NSDate dateWithTimeIntervalSinceNow:10];
	while (download.currentProgress.receivedByteCount == 0 && deadline.timeIntervalSinceNow > 0) {
		[NSThread sleepForTimeInterval:0.01];
	}
	XCTAssertGreaterThan(download.currentProgress.receivedByteCount, 0LL, @"the first bytes arrive before the stall");
	XCTAssertEqual(download.state, NFKDownloadStateDownloading);
	[download cancel];
	[self waitForExpectations:@[ done ] timeout:10];

	XCTAssertEqual(download.state, NFKDownloadStateCancelled);
	XCTAssertTrue(download.isCancelled);
	XCTAssertEqual(finalError.code, NFKDownloadErrorCancelled);
	XCTAssertNil(download.localURL);
	NSURL *localURL = [self.hub localURLForRepo:NFKTestRepo revision:nil path:@"model.safetensors"];
	XCTAssertFalse([NSFileManager.defaultManager fileExistsAtPath:localURL.path]);
	XCTAssertFalse([NSFileManager.defaultManager fileExistsAtPath:[localURL URLByAppendingPathExtension:@"download"].path],
		@"the partial file is removed");
}

- (void)testCancellingThroughTheProgressObjectCancelsTheDownload
{
	NSDictionary<NSString *, NFKStubHubResponse *> *responses = [self serveTheModel];
	responses[@"model.safetensors"].stallsAfterFirstChunk = YES;
	NFKDownload *download = [self.hub downloadWithPlan:[self planPaths:@[ @"model.safetensors" ]] progressHandler:nil completionHandler:nil];
	[download.progress cancel];
	XCTAssertTrue([download waitUntilFinishedWithTimeout:10]);
	XCTAssertEqual(download.state, NFKDownloadStateCancelled);
}

- (void)testADownloadLargerThanTheCacheLimitIsRefusedBeforeAnyTransfer
{
	[self serveTheModel];
	NFKDownloadPlan *plan = [self planPaths:nil];
	self.hub.cacheSizeLimit = 1024;
	[NFKStubHubProtocol reset];

	NFKDownload *download = nil;
	NSError *error = nil;
	[self runDownloadOfPlan:plan download:&download error:&error];
	XCTAssertEqual(download.state, NFKDownloadStateFailed);
	XCTAssertEqual(error.code, NFKDownloadErrorExceedsCacheSizeLimit);
	XCTAssertEqual(NFKStubHubProtocol.requests.count, (NSUInteger)0);
}

- (void)testADroppedTransferResumesWithARangeRequest
{
	NSDictionary<NSString *, NFKStubHubResponse *> *responses = [self serveTheModel];
	responses[@"model.safetensors"].dropsAfterByteCount = 10000;

	NFKDownload *download = nil;
	NSError *error = nil;
	NSArray<NFKDownloadProgress *> *reports = [self runDownloadOfPlan:[self planPaths:@[ @"model.safetensors" ]] download:&download error:&error];
	XCTAssertNil(error);
	NSArray<NSURLRequest *> *requests = [NFKStubHubProtocol requestsForPath:@"/org/model/resolve/main/model.safetensors"];
	XCTAssertEqual(requests.count, (NSUInteger)2);
	XCTAssertEqualObjects([requests[1] valueForHTTPHeaderField:@"Range"], @"bytes=10000-");
	XCTAssertNotNil([requests[1] valueForHTTPHeaderField:@"If-Range"]);
	NSURL *weightsURL = [self.hub localURLForRepo:NFKTestRepo revision:nil path:@"model.safetensors"];
	XCTAssertEqualObjects([NSData dataWithContentsOfURL:weightsURL], self.weights);
	XCTAssertEqual(reports.lastObject.receivedByteCount, (long long)self.weights.length, @"resumed bytes are not counted twice");
}

#pragma mark Pausing

/*! Starts a download of model.safetensors whose transfer stalls after its first chunk, and pauses it once
	that chunk has arrived. Returns the bytes the pause kept, and the paused download in outDownload. */
- (long long)pauseAfterTheFirstChunkOf:(NSDictionary<NSString *, NFKStubHubResponse *> *)responses
							  download:(NFKDownload * __strong *)outDownload
{
	responses[@"model.safetensors"].chunkSize = 4096;
	responses[@"model.safetensors"].stallsAfterFirstChunk = YES;
	XCTestExpectation *done = [self expectationWithDescription:@"download pauses"];
	__block NSError *finalError = nil;
	NFKDownload *download = [self.hub downloadWithPlan:[self planPaths:@[ @"model.safetensors" ]] progressHandler:nil
									 completionHandler:^(NSURL *localURL, NSError *error) {
		finalError = error;
		[done fulfill];
	}];
	NSDate *deadline = [NSDate dateWithTimeIntervalSinceNow:10];
	while (download.currentProgress.receivedByteCount == 0 && deadline.timeIntervalSinceNow > 0) {
		[NSThread sleepForTimeInterval:0.01];
	}
	[download pause];
	[self waitForExpectations:@[ done ] timeout:10];
	XCTAssertEqual(download.state, NFKDownloadStatePaused);
	XCTAssertTrue(download.isPaused);
	XCTAssertFalse(download.isCancelled);
	XCTAssertEqual(finalError.code, NFKDownloadErrorPaused);
	NSURL *localURL = [self.hub localURLForRepo:NFKTestRepo revision:nil path:@"model.safetensors"];
	NSURL *partialURL = [localURL URLByAppendingPathExtension:@"download"];
	long long kept = [[NSFileManager.defaultManager attributesOfItemAtPath:partialURL.path error:NULL][NSFileSize] longLongValue];
	XCTAssertEqual(kept, 4096LL, @"the partial file keeps the bytes that arrived");
	XCTAssertFalse([NSFileManager.defaultManager fileExistsAtPath:localURL.path]);
	responses[@"model.safetensors"].stallsAfterFirstChunk = NO;
	if (outDownload != NULL) {
		*outDownload = download;
	}
	return kept;
}

// A paused download keeps its partial file, the status counts it, and a later download of the same files
// continues it with a Range request guarded by the file's ETag, counting the kept bytes as received.
- (void)testAPausedDownloadIsContinuedByTheNextDownload
{
	NSDictionary<NSString *, NFKStubHubResponse *> *responses = [self serveTheModel];
	long long kept = [self pauseAfterTheFirstChunkOf:responses download:NULL];

	NFKModelFile *weights = [[NFKModelFile alloc] initWithPath:@"model.safetensors" byteCount:@(self.weights.length) sha256:nil
														   role:NFKModelFileRoleWeights optional:NO extensions:nil];
	NFKModelSource *source = [self sourceWithFiles:@[ weights ] requiresConversion:NO];
	NFKDownloadStatus *status = [self.hub downloadStatusForModelSource:source];
	XCTAssertEqual(status.presence, NFKDownloadPresencePartial, @"a kept partial is not a download never started");
	XCTAssertEqual(status.partialByteCount, kept);
	XCTAssertEqualObjects(status.partialPaths, @[ @"model.safetensors" ]);

	[NFKStubHubProtocol reset];
	[self serveTheModel];
	NFKDownload *download = nil;
	NSError *error = nil;
	NSArray<NFKDownloadProgress *> *reports = [self runDownloadOfPlan:[self planPaths:@[ @"model.safetensors" ]]
															 download:&download error:&error];
	XCTAssertNil(error);
	NSArray<NSURLRequest *> *requests = [NFKStubHubProtocol requestsForPath:@"/org/model/resolve/main/model.safetensors"];
	XCTAssertEqual(requests.count, (NSUInteger)1);
	XCTAssertEqualObjects([requests[0] valueForHTTPHeaderField:@"Range"], ([NSString stringWithFormat:@"bytes=%lld-", kept]));
	XCTAssertNotNil([requests[0] valueForHTTPHeaderField:@"If-Range"]);
	NSURL *weightsURL = [self.hub localURLForRepo:NFKTestRepo revision:nil path:@"model.safetensors"];
	XCTAssertEqualObjects([NSData dataWithContentsOfURL:weightsURL], self.weights);
	XCTAssertEqual(reports.lastObject.receivedByteCount, (long long)self.weights.length,
				   @"the kept bytes count once, from the start");
	XCTAssertEqual([self.hub downloadStatusForModelSource:source].presence, NFKDownloadPresenceComplete);
}

// A file that changed while the download was paused is fetched again whole: the If-Range no longer names
// it, so the server answers the whole file, which replaces the kept bytes. The kept bytes are scrambled
// here, so splicing them in would fail the file's checksum.
- (void)testAPausedFileThatChangedStartsOver
{
	NSDictionary<NSString *, NFKStubHubResponse *> *responses = [self serveTheModel];
	long long kept = [self pauseAfterTheFirstChunkOf:responses download:NULL];
	NSURL *localURL = [self.hub localURLForRepo:NFKTestRepo revision:nil path:@"model.safetensors"];
	NSFileHandle *partial = [NSFileHandle fileHandleForUpdatingURL:[localURL URLByAppendingPathExtension:@"download"] error:NULL];
	XCTAssertTrue([partial writeData:[NSMutableData dataWithLength:(NSUInteger)kept] error:NULL]);
	XCTAssertTrue([partial closeAndReturnError:NULL]);

	[NFKStubHubProtocol reset];
	responses = [self serveTheModel];
	NSMutableDictionary *headers = [responses[@"model.safetensors"].headers mutableCopy];
	headers[@"ETag"] = @"\"changed\"";
	responses[@"model.safetensors"].headers = headers;

	NFKDownload *download = nil;
	NSError *error = nil;
	[self runDownloadOfPlan:[self planPaths:@[ @"model.safetensors" ]] download:&download error:&error];
	XCTAssertNil(error);
	NSArray<NSURLRequest *> *requests = [NFKStubHubProtocol requestsForPath:@"/org/model/resolve/main/model.safetensors"];
	XCTAssertEqualObjects([requests.firstObject valueForHTTPHeaderField:@"Range"], ([NSString stringWithFormat:@"bytes=%lld-", kept]),
						  @"the resume was asked for");
	XCTAssertEqualObjects([NSData dataWithContentsOfURL:localURL], self.weights, @"and the whole file came instead");
}

- (void)testCancellingAPausedDownloadRemovesItsKeptPartial
{
	NSDictionary<NSString *, NFKStubHubResponse *> *responses = [self serveTheModel];
	NFKDownload *download = nil;
	[self pauseAfterTheFirstChunkOf:responses download:&download];
	[download cancel];
	XCTAssertEqual(download.state, NFKDownloadStateCancelled);
	XCTAssertTrue(download.isCancelled);
	XCTAssertEqual(download.error.code, NFKDownloadErrorCancelled);
	NSURL *localURL = [self.hub localURLForRepo:NFKTestRepo revision:nil path:@"model.safetensors"];
	XCTAssertFalse([NSFileManager.defaultManager fileExistsAtPath:[localURL URLByAppendingPathExtension:@"download"].path]);
	XCTAssertFalse([NSFileManager.defaultManager fileExistsAtPath:localURL.path]);
}

- (void)testASynchronousFetchThatAPauseStopsReportsThePause
{
	NSDictionary<NSString *, NFKStubHubResponse *> *responses = [self serveTheModel];
	responses[@"model.safetensors"].stallsAfterFirstChunk = YES;
	NFKDownload *download = [NFKDownload downloadWithProgressHandler:nil];
	[download beginActivity];
	XCTestExpectation *returned = [self expectationWithDescription:@"the fetch returns"];
	__block NSURL *fetchedURL = nil;
	__block NSError *fetchError = nil;
	dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
		[download performAsCurrent:^{
			NSError *error = nil;
			fetchedURL = [self.hub downloadRepo:NFKTestRepo revision:nil path:@"model.safetensors" sha256:nil error:&error];
			fetchError = error;
		}];
		[returned fulfill];
	});
	NSDate *deadline = [NSDate dateWithTimeIntervalSinceNow:10];
	while (download.currentProgress.receivedByteCount == 0 && deadline.timeIntervalSinceNow > 0) {
		[NSThread sleepForTimeInterval:0.01];
	}
	[download pause];
	[self waitForExpectations:@[ returned ] timeout:10];

	XCTAssertNil(fetchedURL);
	XCTAssertEqualObjects(fetchError.domain, NFKDownloadErrorDomain);
	XCTAssertEqual(fetchError.code, NFKDownloadErrorPaused);
	NSURL *localURL = [self.hub localURLForRepo:NFKTestRepo revision:nil path:@"model.safetensors"];
	XCTAssertTrue([NSFileManager.defaultManager fileExistsAtPath:[localURL URLByAppendingPathExtension:@"download"].path],
		@"the pause keeps the partial file");
	[download endActivityWithError:fetchError];
	XCTAssertEqual(download.state, NFKDownloadStatePaused);
}

- (void)testAPauseAfterTheLastFileLeavesTheDownloadToComplete
{
	[self serveTheModel];
	NFKDownload *download = [NFKDownload downloadWithProgressHandler:nil];
	[download beginActivity];
	__block NSError *error = nil;
	[download performAsCurrent:^{
		NSError *fetchError = nil;
		[self.hub downloadRepo:NFKTestRepo revision:nil path:@"config.json" sha256:nil error:&fetchError];
		error = fetchError;
	}];
	XCTAssertNil(error);
	[download pause];
	XCTAssertEqual(download.state, NFKDownloadStateDownloading, @"the download waits for its activity to end");
	[download endActivityWithError:nil];
	XCTAssertTrue([download waitUntilFinishedWithTimeout:5]);
	XCTAssertEqual(download.state, NFKDownloadStateCompleted);
	XCTAssertNil(download.error);
}

// A partial that holds every listed byte skips its request and goes to verification. One whose bytes
// fail the file's checksum is fetched again from its first byte.
- (void)testAWholeKeptPartialThatFailsItsChecksumIsFetchedAgain
{
	[self serveTheModel];
	NFKDownloadPlan *plan = [self planPaths:@[ @"model.safetensors" ]];
	NSURL *localURL = [self.hub localURLForRepo:NFKTestRepo revision:nil path:@"model.safetensors"];
	NSURL *partialURL = [localURL URLByAppendingPathExtension:@"download"];
	[NSFileManager.defaultManager createDirectoryAtURL:localURL.URLByDeletingLastPathComponent
						   withIntermediateDirectories:YES attributes:nil error:NULL];
	XCTAssertTrue([[NSMutableData dataWithLength:self.weights.length] writeToURL:partialURL atomically:NO]);
	NSData *tag = [@"\"kept\"" dataUsingEncoding:NSUTF8StringEncoding];
	XCTAssertEqual(setxattr(partialURL.fileSystemRepresentation, "com.inferkit.download.etag", tag.bytes, tag.length, 0, 0), 0);

	NSError *error = nil;
	[self runDownloadOfPlan:plan download:NULL error:&error];
	XCTAssertNil(error);
	NSArray<NSURLRequest *> *requests = [NFKStubHubProtocol requestsForPath:@"/org/model/resolve/main/model.safetensors"];
	XCTAssertEqual(requests.count, (NSUInteger)1);
	XCTAssertNil([requests.firstObject valueForHTTPHeaderField:@"Range"]);
	XCTAssertEqualObjects([NSData dataWithContentsOfURL:localURL], self.weights);
}

// Removing a source takes a paused download's kept partial with it.
- (void)testRemovingASourceRemovesItsKeptPartial
{
	NSDictionary<NSString *, NFKStubHubResponse *> *responses = [self serveTheModel];
	[self pauseAfterTheFirstChunkOf:responses download:NULL];
	NFKModelFile *weights = [[NFKModelFile alloc] initWithPath:@"model.safetensors" byteCount:nil sha256:nil
														   role:NFKModelFileRoleWeights optional:NO extensions:nil];
	NFKModelSource *source = [self sourceWithFiles:@[ weights ] requiresConversion:NO];
	NSError *error = nil;
	NFKRemovalPlan *plan = [self.hub removalPlanForModelSource:source error:&error];
	XCTAssertNotNil(plan, @"%@", error);
	XCTAssertNotNil([self.hub removeFilesInPlan:plan error:&error], @"%@", error);
	NSURL *localURL = [self.hub localURLForRepo:NFKTestRepo revision:nil path:@"model.safetensors"];
	XCTAssertFalse([NSFileManager.defaultManager fileExistsAtPath:[localURL URLByAppendingPathExtension:@"download"].path]);
	XCTAssertEqual([self.hub downloadStatusForModelSource:source].presence, NFKDownloadPresenceAbsent);
}

#pragma mark Model sources

- (NFKModelSource *)sourceWithFiles:(NSArray<NFKModelFile *> *)files requiresConversion:(BOOL)requiresConversion
{
	return [[NFKModelSource alloc] initWithKind:NFKModelSourceKindHuggingFace
									 repository:NFKTestRepo
									   revision:nil
											URL:nil
										 layout:NFKModelSourceLayoutDirectory
										  files:files
							 requiresConversion:requiresConversion
									 extensions:nil];
}

- (void)testAModelSourceDownloadFetchesItsFilesAndLeavesOutAnAbsentOptionalOne
{
	[self serveTheModel];
	NFKModelFile *optional = [[NFKModelFile alloc] initWithPath:@"generation_config.json" byteCount:nil sha256:nil
														   role:NFKModelFileRoleConfig optional:YES extensions:nil];
	NFKModelSource *source = [self sourceWithFiles:@[ [NFKModelFile fileWithPath:@"config.json" role:NFKModelFileRoleConfig],
													  [NFKModelFile fileWithPath:@"model.safetensors" role:NFKModelFileRoleWeights], optional ]
								requiresConversion:NO];
	NSError *planError = nil;
	NFKDownloadPlan *plan = [self.hub planDownloadOfModelSource:source error:&planError];
	XCTAssertEqual(plan.files.count, (NSUInteger)2, @"%@", planError);

	XCTestExpectation *done = [self expectationWithDescription:@"download"];
	__block NSURL *location = nil;
	NFKDownload *download = [self.hub downloadModelSource:source progressHandler:nil completionHandler:^(NSURL *localURL, NSError *error) {
		XCTAssertNil(error);
		location = localURL;
		[done fulfill];
	}];
	[self waitForExpectations:@[ done ] timeout:20];
	XCTAssertEqual(download.state, NFKDownloadStateCompleted);
	XCTAssertEqualObjects(location.path, [self.cacheDir URLByAppendingPathComponent:@"org/model/main"].path);
	XCTAssertEqual([self.hub downloadStatusForModelSource:source].presence, NFKDownloadPresenceComplete);
	NSArray<NSString *> *absent = @[ @"generation_config.json" ];
	XCTAssertEqualObjects([self.hub recordedAbsentPathsOfRepo:NFKTestRepo revision:nil], absent);
}

- (void)testADownloadMadeBeforeTheSourceListedAnOptionalFileFetchesItNext
{
	[self serveTheModel];
	NSArray<NFKModelFile *> *required = @[ [NFKModelFile fileWithPath:@"config.json" role:NFKModelFileRoleConfig],
										   [NFKModelFile fileWithPath:@"model.safetensors" role:NFKModelFileRoleWeights] ];
	NFKModelSource *earlier = [self sourceWithFiles:required requiresConversion:NO];
	XCTestExpectation *first = [self expectationWithDescription:@"first download"];
	[self.hub downloadModelSource:earlier progressHandler:nil completionHandler:^(NSURL *localURL, NSError *error) {
		XCTAssertNil(error);
		[first fulfill];
	}];
	[self waitForExpectations:@[ first ] timeout:20];

	NFKModelFile *vocabulary = [[NFKModelFile alloc] initWithPath:@"tokenizer/vocab.json" byteCount:nil sha256:nil
															 role:NFKModelFileRoleTokenizer optional:YES extensions:nil];
	NFKModelSource *later = [self sourceWithFiles:[required arrayByAddingObject:vocabulary] requiresConversion:NO];
	NFKDownloadStatus *status = [self.hub downloadStatusForModelSource:later];
	XCTAssertEqual(status.presence, NFKDownloadPresencePartial);
	NSArray<NSString *> *unchecked = @[ @"tokenizer/vocab.json" ];
	XCTAssertEqualObjects(status.uncheckedOptionalPaths, unchecked);

	NFKDownloadPlan *plan = [self.hub planDownloadOfModelSource:later error:NULL];
	XCTAssertEqual(plan.remainingByteCount, 2LL, @"only the newly listed file is fetched");
	XCTestExpectation *second = [self expectationWithDescription:@"second download"];
	[self.hub downloadModelSource:later progressHandler:nil completionHandler:^(NSURL *localURL, NSError *error) {
		XCTAssertNil(error);
		[second fulfill];
	}];
	[self waitForExpectations:@[ second ] timeout:20];
	XCTAssertEqual([self.hub downloadStatusForModelSource:later].presence, NFKDownloadPresenceComplete);
}

- (void)testASourceThatRequiresConversionIsRefused
{
	NFKModelSource *source = [self sourceWithFiles:@[] requiresConversion:YES];
	NSError *error = nil;
	XCTAssertNil([self.hub planDownloadOfModelSource:source error:&error]);
	XCTAssertEqual(error.code, NFKDownloadErrorRequiresConversion);

	XCTestExpectation *done = [self expectationWithDescription:@"download"];
	NFKDownload *download = [self.hub downloadModelSource:source progressHandler:nil completionHandler:^(NSURL *localURL, NSError *downloadError) {
		XCTAssertEqual(downloadError.code, NFKDownloadErrorRequiresConversion);
		[done fulfill];
	}];
	[self waitForExpectations:@[ done ] timeout:10];
	XCTAssertEqual(download.state, NFKDownloadStateFailed);
	XCTAssertEqual(NFKStubHubProtocol.requests.count, (NSUInteger)0);
}

- (void)testASourceWithNoFilesCompletesAtOnce
{
	NFKModelSource *source = [[NFKModelSource alloc] initWithKind:NFKModelSourceKindSystem repository:nil revision:nil URL:nil
														   layout:nil files:nil requiresConversion:NO extensions:nil];
	XCTestExpectation *done = [self expectationWithDescription:@"download"];
	NFKDownload *download = [self.hub downloadModelSource:source progressHandler:nil completionHandler:^(NSURL *localURL, NSError *error) {
		XCTAssertNil(localURL);
		XCTAssertNil(error);
		[done fulfill];
	}];
	[self waitForExpectations:@[ done ] timeout:10];
	XCTAssertEqual(download.state, NFKDownloadStateCompleted);
	XCTAssertEqual([self.hub downloadStatusForModelSource:source].presence, NFKDownloadPresenceComplete);
}

- (void)testAURLSourceDownloadsIntoTheURLFolder
{
	[NFKStubHubProtocol setResponse:[NFKStubHubResponse responseWithStatus:200 body:self.weights] forPath:@"/files/model.bin"];
	NSURL *URL = [NSURL URLWithString:@"http://hub.test/files/model.bin"];
	NFKModelFile *file = [[NFKModelFile alloc] initWithPath:@"weights.bin" byteCount:nil
													 sha256:[NFKStubHubProtocol sha256OfData:self.weights]
													   role:NFKModelFileRoleWeights optional:NO extensions:nil];
	NFKModelSource *source = [[NFKModelSource alloc] initWithKind:NFKModelSourceKindURL repository:nil revision:nil URL:URL
														   layout:NFKModelSourceLayoutFile files:@[ file ] requiresConversion:NO extensions:nil];
	XCTestExpectation *done = [self expectationWithDescription:@"download"];
	__block NSURL *location = nil;
	[self.hub downloadModelSource:source progressHandler:nil completionHandler:^(NSURL *localURL, NSError *error) {
		XCTAssertNil(error);
		location = localURL;
		[done fulfill];
	}];
	[self waitForExpectations:@[ done ] timeout:10];
	XCTAssertEqualObjects(location.lastPathComponent, @"weights.bin", @"the listed file name names the file");
	XCTAssertEqualObjects([NSData dataWithContentsOfURL:location], self.weights);
	XCTAssertEqual([self.hub downloadStatusForModelSource:source].presence, NFKDownloadPresenceComplete);
}

#pragma mark Collecting work

- (void)testACurrentDownloadCollectsTheHubsFetches
{
	[self serveTheModel];
	NSMutableArray<NFKDownloadProgress *> *reports = [NSMutableArray array];
	NFKDownload *download = [NFKDownload downloadWithProgressHandler:^(NFKDownloadProgress *progress) {
		@synchronized (reports) {
			[reports addObject:progress];
		}
	}];
	XCTAssertEqual(download.state, NFKDownloadStatePending);
	XCTAssertNil(NFKDownload.currentDownload);

	[download beginActivity];
	__block NSError *error = nil;
	[download performAsCurrent:^{
		XCTAssertEqual(NFKDownload.currentDownload, download);
		NSError *fetchError = nil;
		[self.hub downloadRepo:NFKTestRepo revision:nil path:@"config.json" sha256:nil error:&fetchError];
		[self.hub downloadRepo:NFKTestRepo revision:nil path:@"model.safetensors" sha256:nil error:&fetchError];
		error = fetchError;
	}];
	XCTAssertNil(NFKDownload.currentDownload);
	XCTAssertEqual(download.state, NFKDownloadStateDownloading, @"the download waits for its activity to end");
	[download endActivityWithError:error];
	XCTAssertTrue([download waitUntilFinishedWithTimeout:5]);

	XCTAssertEqual(download.state, NFKDownloadStateCompleted);
	NFKDownloadProgress *progress = download.currentProgress;
	XCTAssertEqual(progress.fileCount, (NSUInteger)2);
	XCTAssertEqual(progress.receivedByteCount, (long long)(self.config.length + self.weights.length));
}

- (void)testACancelledCurrentDownloadStopsTheHubsFetches
{
	[self serveTheModel];
	NFKDownload *download = [NFKDownload downloadWithProgressHandler:nil];
	[download beginActivity];
	[download cancel];
	__block NSError *error = nil;
	[download performAsCurrent:^{
		NSError *fetchError = nil;
		XCTAssertNil([self.hub downloadRepo:NFKTestRepo revision:nil path:@"config.json" sha256:nil error:&fetchError]);
		error = fetchError;
	}];
	XCTAssertEqual(error.code, NFKDownloadErrorCancelled);
	XCTAssertEqual(NFKStubHubProtocol.requests.count, (NSUInteger)0);
	[download endActivityWithError:error];
	XCTAssertEqual(download.state, NFKDownloadStateCancelled);
}

- (void)testTheAsynchronousPlanDeliversThePlan
{
	[self serveTheModel];
	XCTestExpectation *done = [self expectationWithDescription:@"plan"];
	[self.hub planDownloadOfRepo:NFKTestRepo revision:nil paths:@[ @"config.json" ] completionHandler:^(NFKDownloadPlan *plan, NSError *error) {
		XCTAssertNil(error);
		XCTAssertEqual(plan.files.count, (NSUInteger)1);
		[done fulfill];
	}];
	[self waitForExpectations:@[ done ] timeout:10];
}

#pragma mark Listed sizes

- (void)testAFileWhoseSizeDiffersFromTheListingFailsAndIsNotKept
{
	NSDictionary<NSString *, NFKStubHubResponse *> *responses = [self serveTheModel];
	responses[@"config.json"].body = [@"<html><body>Sign in to this network</body></html>" dataUsingEncoding:NSUTF8StringEncoding];

	NFKDownload *download = nil;
	NSError *error = nil;
	[self runDownloadOfPlan:[self planPaths:@[ @"config.json" ]] download:&download error:&error];
	XCTAssertEqual(download.state, NFKDownloadStateFailed);
	XCTAssertEqualObjects(error.domain, NFKDownloadErrorDomain);
	XCTAssertEqual(error.code, NFKDownloadErrorSizeMismatch, @"%@", error);
	NSURL *localURL = [self.hub localURLForRepo:NFKTestRepo revision:nil path:@"config.json"];
	XCTAssertFalse([NSFileManager.defaultManager fileExistsAtPath:localURL.path]);
	XCTAssertFalse([NSFileManager.defaultManager fileExistsAtPath:[localURL URLByAppendingPathExtension:@"download"].path]);
	XCTAssertEqual([self.hub downloadStatusForRepo:NFKTestRepo revision:nil paths:@[ @"config.json" ]].presence, NFKDownloadPresenceAbsent);
}

- (void)testAURLFileWhoseSizeDiffersFromItsListedSizeFails
{
	[NFKStubHubProtocol setResponse:[NFKStubHubResponse responseWithStatus:200 body:self.weights] forPath:@"/files/model.bin"];
	NSURL *URL = [NSURL URLWithString:@"http://hub.test/files/model.bin"];
	NFKModelFile *file = [[NFKModelFile alloc] initWithPath:@"model.bin" byteCount:@(self.weights.length + 1) sha256:nil
													   role:NFKModelFileRoleWeights optional:NO extensions:nil];
	NFKModelSource *source = [[NFKModelSource alloc] initWithKind:NFKModelSourceKindURL repository:nil revision:nil URL:URL
														   layout:NFKModelSourceLayoutFile files:@[ file ] requiresConversion:NO extensions:nil];
	XCTestExpectation *done = [self expectationWithDescription:@"download"];
	__block NSError *finalError = nil;
	[self.hub downloadModelSource:source progressHandler:nil completionHandler:^(NSURL *localURL, NSError *error) {
		finalError = error;
		[done fulfill];
	}];
	[self waitForExpectations:@[ done ] timeout:10];
	XCTAssertEqual(finalError.code, NFKDownloadErrorSizeMismatch, @"%@", finalError);
	XCTAssertFalse([NSFileManager.defaultManager fileExistsAtPath:[self.hub localURLForURL:URL fileName:@"model.bin"].path]);
}

#pragma mark Unsafe names

- (void)testAListingThatNamesAFileOutsideTheSnapshotIsRefused
{
	NSURL *outside = [NSURL fileURLWithPath:[self.cacheDir.path stringByAppendingString:@"-outside"]];
	NSString *hostile = [NSString stringWithFormat:@"../../../../%@/planted.bin", outside.lastPathComponent];
	[NFKStubHubProtocol serveRepo:NFKTestRepo revision:@"main" files:@{ @"config.json": self.config, hostile: self.weights }
						 lfsPaths:[NSSet set]];

	NSError *error = nil;
	XCTAssertNil([self.hub planDownloadOfRepo:NFKTestRepo revision:nil paths:nil error:&error]);
	XCTAssertEqualObjects(error.domain, NFKDownloadErrorDomain);
	XCTAssertEqual(error.code, NFKDownloadErrorUnsafePath, @"%@", error);

	XCTestExpectation *done = [self expectationWithDescription:@"download"];
	__block NSError *finalError = nil;
	[self.hub downloadRepo:NFKTestRepo revision:nil paths:nil progressHandler:nil completionHandler:^(NSURL *localURL, NSError *downloadError) {
		finalError = downloadError;
		[done fulfill];
	}];
	[self waitForExpectations:@[ done ] timeout:10];
	XCTAssertEqual(finalError.code, NFKDownloadErrorUnsafePath, @"%@", finalError);
	XCTAssertFalse([NSFileManager.defaultManager fileExistsAtPath:outside.path], @"nothing is written outside the cache");
	XCTAssertEqual([NFKStubHubProtocol requestsForPath:@"/org/model/resolve/main/config.json"].count, (NSUInteger)0);
}

- (void)testAURLSourceWhoseFileNameLeavesItsFolderIsRefused
{
	[NFKStubHubProtocol setResponse:[NFKStubHubResponse responseWithStatus:200 body:self.weights] forPath:@"/files/model.bin"];
	NSURL *URL = [NSURL URLWithString:@"http://hub.test/files/model.bin"];
	NSURL *hostFolder = [[self.cacheDir URLByAppendingPathComponent:@"@url"] URLByAppendingPathComponent:@"hub.test"];
	NSURL *neighbor = [[hostFolder URLByAppendingPathComponent:@"0000000000000000"] URLByAppendingPathComponent:@"keep.bin"];
	[NSFileManager.defaultManager createDirectoryAtURL:neighbor.URLByDeletingLastPathComponent withIntermediateDirectories:YES attributes:nil error:NULL];
	XCTAssertTrue([self.config writeToURL:neighbor atomically:YES]);

	NFKModelFile *file = [[NFKModelFile alloc] initWithPath:@".." byteCount:nil sha256:nil
													   role:NFKModelFileRoleWeights optional:NO extensions:nil];
	NFKModelSource *source = [[NFKModelSource alloc] initWithKind:NFKModelSourceKindURL repository:nil revision:nil URL:URL
														   layout:NFKModelSourceLayoutFile files:@[ file ] requiresConversion:NO extensions:nil];
	NSError *error = nil;
	XCTAssertNil([self.hub planDownloadOfModelSource:source error:&error]);
	XCTAssertEqual(error.code, NFKDownloadErrorUnsafePath, @"%@", error);

	XCTestExpectation *done = [self expectationWithDescription:@"download"];
	__block NSError *finalError = nil;
	[self.hub downloadModelSource:source progressHandler:nil completionHandler:^(NSURL *localURL, NSError *downloadError) {
		finalError = downloadError;
		[done fulfill];
	}];
	[self waitForExpectations:@[ done ] timeout:10];
	XCTAssertEqual(finalError.code, NFKDownloadErrorUnsafePath, @"%@", finalError);
	XCTAssertEqual(NFKStubHubProtocol.requests.count, (NSUInteger)0, @"the refusal comes before any request");
	XCTAssertEqualObjects([NSData dataWithContentsOfURL:neighbor], self.config, @"the host folder's other downloads stay");
}

#pragma mark Eviction

- (void)testATrimLeavesTheSnapshotOfADownloadBetweenItsFiles
{
	[self serveTheModel];
	NFKGatedHub *hub = [NFKGatedHub hubWithCacheDirectoryURL:self.cacheDir];
	hub.endpointURL = NFKStubHubProtocol.endpointURL;
	hub.session = [NFKStubHubProtocol session];
	hub.accessToken = @"";
	hub.excludesCacheFromBackup = NO;
	hub.gatedFileName = @"model.safetensors";
	hub.arrived = dispatch_semaphore_create(0);
	hub.gate = dispatch_semaphore_create(0);
	NSError *error = nil;
	NFKDownloadPlan *plan = [hub planDownloadOfRepo:NFKTestRepo revision:nil paths:@[ @"config.json", @"model.safetensors" ] error:&error];
	XCTAssertNotNil(plan, @"%@", error);
	XCTAssertEqualObjects(plan.files.firstObject.path, @"config.json");

	NFKDownload *download = [hub downloadWithPlan:plan progressHandler:nil completionHandler:nil];
	long arrived = dispatch_semaphore_wait(hub.arrived, dispatch_time(DISPATCH_TIME_NOW, 10 * (int64_t)NSEC_PER_SEC));
	XCTAssertEqual(arrived, 0L, @"the download reaches its second file");
	NSURL *configURL = [hub localURLForRepo:NFKTestRepo revision:nil path:@"config.json"];
	XCTAssertTrue([NSFileManager.defaultManager fileExistsAtPath:configURL.path]);

	NFKHFHub *other = [NFKHFHub hubWithCacheDirectoryURL:self.cacheDir];
	other.cacheSizeLimit = 0;
	error = nil;
	XCTAssertTrue([other trimCacheToSizeLimitWithError:&error], @"%@", error);
	XCTAssertTrue([NSFileManager.defaultManager fileExistsAtPath:configURL.path], @"the running download's snapshot is not evicted");
	error = nil;
	XCTAssertFalse([other removeCachedRepo:NFKTestRepo revision:nil error:&error]);
	XCTAssertEqual(error.code, NFKDownloadErrorInUse, @"%@", error);

	dispatch_semaphore_signal(hub.gate);
	XCTAssertTrue([download waitUntilFinishedWithTimeout:20]);
	XCTAssertEqual(download.state, NFKDownloadStateCompleted, @"%@", download.error);
	XCTAssertTrue([NSFileManager.defaultManager fileExistsAtPath:configURL.path]);
}

#pragma mark Files from other repositories

/*! org/model's config.json, org/adapter's adapter.safetensors at revision v2, and a file at a URL. */
- (NFKModelSource *)sourceSpanningRepositoriesWithAdapter:(NSData *)adapter vocabulary:(NSData *)vocabulary
{
	NFKModelFile *config = [NFKModelFile fileWithPath:@"config.json" role:NFKModelFileRoleConfig];
	NFKModelFile *adapterFile = [[NFKModelFile alloc] initWithPath:@"adapter.safetensors" byteCount:nil sha256:nil
															  role:NFKModelFileRoleWeights optional:NO repository:@"org/adapter"
														  revision:@"v2" URL:nil extensions:nil];
	NFKModelFile *vocabularyFile = [[NFKModelFile alloc] initWithPath:@"vocab.txt" byteCount:@(vocabulary.length) sha256:nil
																 role:NFKModelFileRoleTokenizer optional:NO repository:nil revision:nil
																  URL:[NSURL URLWithString:@"http://hub.test/files/vocab.txt"] extensions:nil];
	return [[NFKModelSource alloc] initWithKind:NFKModelSourceKindHuggingFace repository:NFKTestRepo revision:nil URL:nil
										 layout:NFKModelSourceLayoutDirectory files:@[ config, adapterFile, vocabularyFile ]
							 requiresConversion:NO extensions:nil];
}

- (void)testASourceSpanningRepositoriesFetchesEachFileFromItsOwnRepository
{
	[self serveTheModel];
	NSData *adapter = [NSMutableData dataWithLength:8192];
	NSData *vocabulary = [@"a\nb\nc\n" dataUsingEncoding:NSUTF8StringEncoding];
	[NFKStubHubProtocol serveRepo:@"org/adapter" revision:@"v2" files:@{ @"adapter.safetensors": adapter }
						 lfsPaths:[NSSet setWithObject:@"adapter.safetensors"]];
	[NFKStubHubProtocol setResponse:[NFKStubHubResponse responseWithStatus:200 body:vocabulary] forPath:@"/files/vocab.txt"];
	NFKModelSource *source = [self sourceSpanningRepositoriesWithAdapter:adapter vocabulary:vocabulary];

	NSError *error = nil;
	NFKDownloadPlan *plan = [self.hub planDownloadOfModelSource:source error:&error];
	XCTAssertNotNil(plan, @"%@", error);
	XCTAssertEqual(plan.files.count, (NSUInteger)3);
	XCTAssertEqual(plan.totalByteCount, (long long)(self.config.length + adapter.length + vocabulary.length),
		@"the plan counts every repository's files");
	NFKDownloadPlanFile *adapterPlan = [plan fileAtPath:@"adapter.safetensors"];
	XCTAssertEqualObjects(adapterPlan.remoteURL.path, @"/org/adapter/resolve/v2/adapter.safetensors");
	XCTAssertEqualObjects(adapterPlan.localURL.path, [self.hub localURLForRepo:@"org/adapter" revision:@"v2" path:@"adapter.safetensors"].path);
	XCTAssertEqualObjects(adapterPlan.sha256, [NFKStubHubProtocol sha256OfData:adapter], @"the adapter repository's listing gives its digest");
	XCTAssertEqual([self.hub downloadStatusForModelSource:source].presence, NFKDownloadPresenceAbsent);

	NFKDownload *download = nil;
	[self runDownloadOfPlan:plan download:&download error:&error];
	XCTAssertEqual(download.state, NFKDownloadStateCompleted, @"%@", error);
	XCTAssertEqual([NFKStubHubProtocol requestsForPath:@"/org/model/resolve/main/config.json"].count, (NSUInteger)1);
	XCTAssertEqual([NFKStubHubProtocol requestsForPath:@"/org/adapter/resolve/v2/adapter.safetensors"].count, (NSUInteger)1);
	NSArray<NSURLRequest *> *vocabularyRequests = [NFKStubHubProtocol requestsForPath:@"/files/vocab.txt"];
	XCTAssertEqual([vocabularyRequests filteredArrayUsingPredicate:[NSPredicate predicateWithFormat:@"HTTPMethod == 'GET'"]].count,
				   (NSUInteger)1);
	XCTAssertEqual([NFKStubHubProtocol requestsForPath:@"/org/model/resolve/main/adapter.safetensors"].count, (NSUInteger)0,
		@"the adapter is not asked of the source's repository");
	XCTAssertEqualObjects([NSData dataWithContentsOfURL:adapterPlan.localURL], adapter);
	XCTAssertEqualObjects([NSData dataWithContentsOfURL:[self.hub localURLForURL:[NSURL URLWithString:@"http://hub.test/files/vocab.txt"]
																		 fileName:@"vocab.txt"]], vocabulary);
	XCTAssertEqual([self.hub downloadStatusForModelSource:source].presence, NFKDownloadPresenceComplete);

	XCTAssertTrue([self.hub removeCachedRepo:@"org/adapter" revision:@"v2" error:&error], @"%@", error);
	NFKDownloadStatus *status = [self.hub downloadStatusForModelSource:source];
	XCTAssertEqual(status.presence, NFKDownloadPresencePartial, @"complete only while every repository's files are");
	XCTAssertEqualObjects(status.missingPaths, @[ @"adapter.safetensors" ]);

	NFKRemovalPlan *removal = [self.hub removalPlanForModelSource:source error:&error];
	XCTAssertNotNil(removal, @"%@", error);
	XCTAssertNotNil([self.hub removeFilesInPlan:removal error:&error], @"%@", error);
	XCTAssertEqual([self.hub downloadStatusForModelSource:source].presence, NFKDownloadPresenceAbsent);
	XCTAssertFalse([NSFileManager.defaultManager fileExistsAtPath:[self.cacheDir URLByAppendingPathComponent:@"org"].path]);
	XCTAssertFalse([NSFileManager.defaultManager fileExistsAtPath:[self.cacheDir URLByAppendingPathComponent:@"@url"].path]);
}

- (void)testAnUnsafeOverrideRepositoryOrRevisionIsRefusedBeforeAnyRequest
{
	[self serveTheModel];
	NSString *outside = [@"../" stringByAppendingString:[[self.cacheDir.path stringByAppendingString:@"-outside"] lastPathComponent]];
	NFKModelFile *config = [NFKModelFile fileWithPath:@"config.json" role:NFKModelFileRoleConfig];
	NSArray<NFKModelFile *> *hostileFiles = @[
		[[NFKModelFile alloc] initWithPath:@"planted.bin" byteCount:nil sha256:nil role:nil optional:NO
								repository:outside revision:nil URL:nil extensions:nil],
		[[NFKModelFile alloc] initWithPath:@"planted.bin" byteCount:nil sha256:nil role:nil optional:NO
								repository:nil revision:@"../../../escape" URL:nil extensions:nil],
		[[NFKModelFile alloc] initWithPath:@".." byteCount:nil sha256:nil role:nil optional:NO
								repository:nil revision:nil URL:[NSURL URLWithString:@"http://hub.test/files/x.bin"] extensions:nil] ];
	for (NFKModelFile *hostile in hostileFiles) {
		[NFKStubHubProtocol reset];
		[self serveTheModel];
		NFKModelSource *source = [[NFKModelSource alloc] initWithKind:NFKModelSourceKindHuggingFace repository:NFKTestRepo revision:nil
																  URL:nil layout:NFKModelSourceLayoutDirectory
																files:@[ config, hostile ] requiresConversion:NO extensions:nil];
		NSError *error = nil;
		XCTAssertNil([self.hub planDownloadOfModelSource:source error:&error], @"%@", hostile.dictionaryRepresentation);
		XCTAssertEqualObjects(error.domain, NFKDownloadErrorDomain);
		XCTAssertEqual(error.code, NFKDownloadErrorUnsafePath, @"%@", error);
		XCTAssertEqual(NFKStubHubProtocol.requests.count, (NSUInteger)0, @"the refusal comes before any request");
		XCTAssertNotEqual([self.hub downloadStatusForModelSource:source].presence, NFKDownloadPresenceComplete);
		error = nil;
		XCTAssertNil([self.hub removalPlanForModelSource:source error:&error]);
		XCTAssertEqual(error.code, NFKDownloadErrorUnsafePath, @"%@", error);
	}
	XCTAssertFalse([NSFileManager.defaultManager fileExistsAtPath:[self.cacheDir.path stringByAppendingString:@"-outside"]]);
}

#pragma mark Streaming limits

- (void)testATransferStopsOnceItStreamsPastTheListedSize
{
	NSDictionary<NSString *, NFKStubHubResponse *> *responses = [self serveTheModel];
	responses[@"config.json"].body = self.weights;
	responses[@"config.json"].chunkSize = 4096;
	responses[@"config.json"].stallsAfterFirstChunk = YES;

	NFKDownload *download = nil;
	NSError *error = nil;
	[self runDownloadOfPlan:[self planPaths:@[ @"config.json" ]] download:&download error:&error];
	[download cancel];
	XCTAssertEqual(download.state, NFKDownloadStateFailed, @"a stalled body past the listed size ends without waiting for it");
	XCTAssertEqual(error.code, NFKDownloadErrorSizeMismatch, @"%@", error);
	NSURL *localURL = [self.hub localURLForRepo:NFKTestRepo revision:nil path:@"config.json"];
	XCTAssertFalse([NSFileManager.defaultManager fileExistsAtPath:localURL.path]);
	XCTAssertFalse([NSFileManager.defaultManager fileExistsAtPath:[localURL URLByAppendingPathExtension:@"download"].path]);
}

- (void)testAnUnlistedTransferStopsAtTheHubsMaximum
{
	NSDictionary<NSString *, NFKStubHubResponse *> *responses = [self serveTheModel];
	responses[@"model.safetensors"].stallsAfterFirstChunk = YES;
	self.hub.maximumFileByteCount = 1000;

	XCTestExpectation *done = [self expectationWithDescription:@"download"];
	__block NSError *finalError = nil;
	__block NSURL *location = nil;
	[self.hub downloadRepo:NFKTestRepo revision:nil path:@"model.safetensors" sha256:nil completionHandler:^(NSURL *localURL, NSError *error) {
		location = localURL;
		finalError = error;
		[done fulfill];
	}];
	[self waitForExpectations:@[ done ] timeout:10];
	XCTAssertNil(location);
	XCTAssertEqual(finalError.code, NFKDownloadErrorSizeMismatch, @"%@", finalError);
	XCTAssertFalse([self.hub isCachedRepo:NFKTestRepo revision:nil path:@"model.safetensors"]);

	self.hub.maximumFileByteCount = -1;
	responses[@"model.safetensors"].stallsAfterFirstChunk = NO;
	NSError *error = nil;
	XCTAssertNotNil([self.hub downloadRepo:NFKTestRepo revision:nil path:@"config.json" sha256:nil error:&error], @"%@", error);
}

@end
