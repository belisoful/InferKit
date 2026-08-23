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

- (void)testAnUnconfiguredBackendFailsTheJobImmediately
{
	NFKAsyncGenerationBackend *backend = NFKAsyncGenerationBackend.new;	// no submitURL
	NFKInferenceJob *job = [backend submitInferenceJobForRequest:[NFKInferenceRequest requestWithInputs:@{}]];
	XCTAssertEqual(job.status, NFKInferenceJobStatusFailed);
	XCTAssertFalse(backend.isReady);
}

@end
