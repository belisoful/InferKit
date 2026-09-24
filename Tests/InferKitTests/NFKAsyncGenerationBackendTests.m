//
//  NFKAsyncGenerationBackendTests.m
//  InferKitTests
//

#import <XCTest/XCTest.h>
#import <InferKit/NFKAsyncGenerationBackend.h>
#import <InferKit/NFKInferenceRequest.h>
#import <InferKit/NFKInferenceResult.h>
#import <InferKit/NFKInferenceKeys.h>
#import <InferKit/NFKVideoAsset.h>
#import <InferKit/NFKRemoteTransport.h>
#import <InferKit/NFKErrors.h>

/*! A generation backend whose transport replays scripted JSON responses in order. */
@interface NFKAsyncTestBackend : NFKAsyncGenerationBackend
@property (nonatomic, strong) NSMutableArray<NSDictionary *> *scriptedResponses;
@property (nonatomic, assign) NSUInteger requestCount;
@end

@implementation NFKAsyncTestBackend

- (instancetype)init
{
	self = [super init];
	if (self) {
		_scriptedResponses = NSMutableArray.new;
		self.submitURL = [NSURL URLWithString:@"https://example.test/jobs"];
		self.pollInterval = 0.0;
	}
	return self;
}

- (NSDictionary *)sendJSONRequest:(NSURLRequest *)request error:(NSError **)outError
{
	self.requestCount += 1;
	if (self.scriptedResponses.count == 0) {
		if (outError != NULL) {
			*outError = [NSError errorWithDomain:@"test" code:1 userInfo:nil];
		}
		return nil;
	}
	NSDictionary *next = self.scriptedResponses.firstObject;
	[self.scriptedResponses removeObjectAtIndex:0];
	return next;
}

@end

/*! Answers every request to async.test with one status and body, so the base's own transport is
	exercised without a network. */
@interface NFKAsyncStubProtocol : NSURLProtocol
@end

static NSInteger NFKAsyncStubStatus = 200;
static NSString *NFKAsyncStubBody = @"{}";

@implementation NFKAsyncStubProtocol
+ (BOOL)canInitWithRequest:(NSURLRequest *)request { return [request.URL.host isEqualToString:@"async.test"]; }
+ (NSURLRequest *)canonicalRequestForRequest:(NSURLRequest *)request { return request; }
- (void)startLoading
{
	NSHTTPURLResponse *response = [[NSHTTPURLResponse alloc] initWithURL:self.request.URL statusCode:NFKAsyncStubStatus
															 HTTPVersion:@"HTTP/1.1" headerFields:@{ @"Retry-After": @"7" }];
	[self.client URLProtocol:self didReceiveResponse:response cacheStoragePolicy:NSURLCacheStorageNotAllowed];
	[self.client URLProtocol:self didLoadData:[NFKAsyncStubBody dataUsingEncoding:NSUTF8StringEncoding]];
	[self.client URLProtocolDidFinishLoading:self];
}
- (void)stopLoading {}
@end

@interface NFKAsyncGenerationBackendTests : XCTestCase
@end

@implementation NFKAsyncGenerationBackendTests

- (void)testTheSubmitBodyCarriesPromptsAndParametersButNotMedia
{
	NFKAsyncTestBackend *backend = NFKAsyncTestBackend.new;
	backend.modelName = @"gen-1";
	NFKVideoAsset *clip = [NFKVideoAsset videoAssetWithFileURL:[NSURL fileURLWithPath:@"/in.mp4"]];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"a wave", NFKInputVideo: clip }
															 parameters:@{ NFKParameterSeed: @7 }
														 outputModality:NFKModalityVideo];

	NSDictionary *body = [backend submitBodyForRequest:request];
	XCTAssertEqualObjects(body[@"model"], @"gen-1");
	XCTAssertEqualObjects(body[NFKInputPrompt], @"a wave");
	XCTAssertEqualObjects(body[NFKParameterSeed], @7);
	XCTAssertNil(body[NFKInputVideo], @"a media asset is not JSON-encoded into the body");
}

- (void)testSubmitPollSucceedProducesAVideoResult
{
	NFKAsyncTestBackend *backend = NFKAsyncTestBackend.new;
	backend.scriptedResponses = [@[
		@{ @"id": @"job1" },
		@{ @"status": @"running", @"progress": @0.5 },
		@{ @"status": @"succeeded", @"output": @"https://example.test/out.mp4" },
	] mutableCopy];

	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"a wave" }
															 parameters:nil
														 outputModality:NFKModalityVideo];

	XCTestExpectation *done = [self expectationWithDescription:@"job completes"];
	NFKInferenceJob *job = [backend submitInferenceJobForRequest:request];
	job.completionHandler = ^(NFKInferenceJob *finished) { [done fulfill]; };
	[self waitForExpectations:@[done] timeout:2.0];

	XCTAssertEqual(job.status, NFKInferenceJobStatusSucceeded);
	NFKVideoAsset *asset = [job.result outputForKey:NFKOutputVideo];
	XCTAssertTrue([asset isKindOfClass:NFKVideoAsset.class]);
	XCTAssertEqualObjects(asset.fileURL.absoluteString, @"https://example.test/out.mp4");
	XCTAssertEqual(backend.requestCount, (NSUInteger)3, @"submit + two polls");
}

- (void)testAFailedStatusFailsTheJob
{
	NFKAsyncTestBackend *backend = NFKAsyncTestBackend.new;
	backend.scriptedResponses = [@[
		@{ @"id": @"job1" },
		@{ @"status": @"failed" },
	] mutableCopy];

	XCTestExpectation *done = [self expectationWithDescription:@"job completes"];
	NFKInferenceJob *job = [backend submitInferenceJobForRequest:[NFKInferenceRequest requestWithInputs:@{}]];
	job.completionHandler = ^(NFKInferenceJob *finished) { [done fulfill]; };
	[self waitForExpectations:@[done] timeout:2.0];

	XCTAssertEqual(job.status, NFKInferenceJobStatusFailed);
	XCTAssertNotNil(job.error);
}

// The base's own transport treats a failing status as the provider's answer rather than a body to
// read a job id from: a rate limit fails the job under its code with the reset date beside it.
- (void)testAFailingStatusOnSubmitFailsTheJobUnderItsCode
{
	NFKAsyncGenerationBackend *backend = [[NFKAsyncGenerationBackend alloc] init];
	backend.submitURL = [NSURL URLWithString:@"https://async.test/jobs"];
	NSURLSessionConfiguration *configuration = [NSURLSessionConfiguration ephemeralSessionConfiguration];
	configuration.protocolClasses = @[ NFKAsyncStubProtocol.class ];
	backend.session = [NSURLSession sessionWithConfiguration:configuration];
	NFKAsyncStubStatus = 429;
	NFKAsyncStubBody = @"{\"id\":\"job_1\",\"error\":\"slow down\"}";

	NFKInferenceJob *job = [backend submitInferenceJobForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"x" }]];
	XCTestExpectation *ended = [self expectationWithDescription:@"job ended"];
	job.completionHandler = ^(NFKInferenceJob *j) { [ended fulfill]; };
	[self waitForExpectations:@[ ended ] timeout:10];
	XCTAssertEqual(job.status, NFKInferenceJobStatusFailed);
	XCTAssertEqual(job.error.code, kNFKError_InferenceRateLimited);
	XCTAssertNotNil(job.error.userInfo[NFKRemoteErrorRetryAfterKey]);
	NFKAsyncStubStatus = 200;
}

- (void)testAnUnconfiguredBackendFailsTheJobImmediately
{
	NFKAsyncGenerationBackend *backend = NFKAsyncGenerationBackend.new;	// no submitURL
	NFKInferenceJob *job = [backend submitInferenceJobForRequest:[NFKInferenceRequest requestWithInputs:@{}]];
	XCTAssertEqual(job.status, NFKInferenceJobStatusFailed);
	XCTAssertFalse(backend.isReady);
}

@end
