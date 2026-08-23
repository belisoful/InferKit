//
//  NFKMultimodalTests.m
//  InferKitTests
//

#import <XCTest/XCTest.h>
#import <InferKit/NFKInferenceRequest.h>
#import <InferKit/NFKInferenceResult.h>
#import <InferKit/NFKInferenceBackend.h>
#import <InferKit/NFKInferenceJob.h>
#import <InferKit/NFKPassthroughBackend.h>
#import <InferKit/NFKVideoAsset.h>

@interface NFKMultimodalTests : XCTestCase
@end

@implementation NFKMultimodalTests

- (void)testARequestDefaultsToImageOutput
{
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ @"prompt": @"a cat" }];
	XCTAssertEqual(request.outputModality, NFKModalityImage);
}

- (void)testARequestCanDeclareVideoOutput
{
	// text + image → video: mix input modalities under one request, ask for a clip back.
	NSDictionary *inputs = @{ @"prompt": @"pan left", @"image": @"still" };
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:inputs
															 parameters:@{ @"seconds": @4 }
														 outputModality:NFKModalityVideo];
	XCTAssertEqual(request.outputModality, NFKModalityVideo);
	XCTAssertEqualObjects([request inputForKey:@"prompt"], @"pan left");
	XCTAssertEqualObjects([request inputForKey:@"image"], @"still");
}

- (void)testAVideoAssetFlowsAsAnInputValue
{
	NFKVideoAsset *clip = [NFKVideoAsset videoAssetWithFileURL:[NSURL fileURLWithPath:@"/in.mp4"]];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ @"video": clip }
															 parameters:nil
														 outputModality:NFKModalityVideo];
	XCTAssertEqualObjects([request inputForKey:@"video"], clip);
}

- (void)testSubmitWrapsASynchronousBackendIntoAJob
{
	NFKPassthroughBackend *backend = [NFKPassthroughBackend backend];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ @"image": @"plate" }];

	XCTestExpectation *done = [self expectationWithDescription:@"job completes"];
	NFKInferenceJob *job = NFKInferenceSubmit(backend, request, NULL);
	job.completionHandler = ^(NFKInferenceJob *j) {
		XCTAssertEqual(j.status, NFKInferenceJobStatusSucceeded);
		XCTAssertEqualObjects([j.result outputForKey:@"image"], @"plate");
		[done fulfill];
	};

	[self waitForExpectations:@[done] timeout:2.0];
	XCTAssertEqual(job.status, NFKInferenceJobStatusSucceeded);
}

@end
