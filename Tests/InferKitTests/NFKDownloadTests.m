//
//  NFKDownloadTests.m
//  InferKitTests
//
//  Download plans and observable downloads against the in-process hub at http://hub.test: sizes and
//  digests from the listing, globs, cache detection, progress, cancellation, the cache size limit,
//  resumption, gated repositories, and catalog model sources.
//

#import <XCTest/XCTest.h>
#import <InferKit/InferKit.h>
#import "NFKStubHubProtocol.h"

static NSString * const NFKTestRepo = @"org/model";

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
	XCTAssertEqual(reports.lastObject.receivedByteCount, 0LL);
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

@end
