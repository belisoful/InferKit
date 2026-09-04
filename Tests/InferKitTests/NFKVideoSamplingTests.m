//
//  NFKVideoSamplingTests.m
//  InferKitTests
//
//  Frames sampled from a clip written on the fly: the count, the order, and that each frame is the
//  one at its time. A live pass shows a sampled clip to a local vision model.
//

#import <XCTest/XCTest.h>
#import <InferKit/NFKVideoSampling.h>
#import <InferKit/NFKRemoteProvider.h>
#import <InferKit/NFKRemoteBackend.h>
#import <InferKit/NFKInferenceRequest.h>
#import <InferKit/NFKInferenceResult.h>
#import <InferKit/NFKInferenceKeys.h>
#import <InferKit/NFKVideoAsset.h>
#import <InferKit/NFKErrors.h>
#import "NFKTestClip.h"

@interface NFKVideoSamplingTests : XCTestCase
@property (nonatomic, strong) NSURL *clip;
@end

@implementation NFKVideoSamplingTests

// Four frames: red, green, blue, white, half a second each.
- (void)setUp
{
	[super setUp];
	NSError *error = nil;
	self.clip = [NFKTestClip writeClipWithColors:@[ @[ @1, @0, @0 ], @[ @0, @1, @0 ], @[ @0, @0, @1 ], @[ @1, @1, @1 ] ] error:&error];
	XCTAssertNotNil(self.clip, @"%@", error);
}

- (void)tearDown
{
	[NSFileManager.defaultManager removeItemAtURL:self.clip error:NULL];
	[super tearDown];
}

// H.264's chroma subsampling bleeds up to about 0.27 into a pure primary's other channels (a solid
// green frame decodes with r ≈ 0.27), so the tolerance admits that while still telling the colours apart.
- (void)assertImage:(CGImageRef)image isNear:(NSArray<NSNumber *> *)rgb
{
	NSArray<NSNumber *> *mean = [NFKTestClip meanColorOfImage:image];
	for (NSUInteger channel = 0; channel < 3; channel++) {
		XCTAssertEqualWithAccuracy(mean[channel].doubleValue, rgb[channel].doubleValue, 0.35,
								   @"channel %lu of %@", (unsigned long)channel, mean);
	}
}

- (void)testFourFramesFromAFourFrameClipAreTheFramesInOrder
{
	NSError *error = nil;
	NSArray *frames = [NFKVideoSampling framesOfVideoAtURL:self.clip count:4 error:&error];
	XCTAssertEqual(frames.count, 4, @"%@", error);
	[self assertImage:(__bridge CGImageRef)frames[0] isNear:@[ @1, @0, @0 ]];
	[self assertImage:(__bridge CGImageRef)frames[1] isNear:@[ @0, @1, @0 ]];
	[self assertImage:(__bridge CGImageRef)frames[2] isNear:@[ @0, @0, @1 ]];
	[self assertImage:(__bridge CGImageRef)frames[3] isNear:@[ @1, @1, @1 ]];
	XCTAssertEqual(CGImageGetWidth((__bridge CGImageRef)frames[0]), 64);
}

// Two samples of the two-second clip land at 0.5 s and 1.5 s, which are the starts of the second
// and fourth frames, so those are the frames that come back.
- (void)testFewerSamplesAreSpacedEvenlyThroughTheClip
{
	NSError *error = nil;
	NSArray *frames = [NFKVideoSampling framesOfVideoAtURL:self.clip count:2 error:&error];
	XCTAssertEqual(frames.count, 2, @"%@ — %@", error, error.userInfo);
	[self assertImage:(__bridge CGImageRef)frames[0] isNear:@[ @0, @1, @0 ]];
	[self assertImage:(__bridge CGImageRef)frames[1] isNear:@[ @1, @1, @1 ]];
}

- (void)testAMissingClipIsAnErrorNotAnEmptyList
{
	NSError *error = nil;
	NSArray *frames = [NFKVideoSampling framesOfVideoAtURL:[NSURL fileURLWithPath:@"/nonesuch/clip.mp4"] count:2 error:&error];
	XCTAssertNil(frames);
	XCTAssertNotNil(error);
}

#pragma mark A live local vision model

// Set INFERKIT_LIVE_VISION_MODEL to a vision-capable model the local server has. The clip is four
// solid frames, so the answer has to name the colours it saw.
- (void)testALocalVisionModelDescribesASampledClip
{
	NSDictionary *environment = NSProcessInfo.processInfo.environment;
	NSString *model = environment[@"INFERKIT_LIVE_VISION_MODEL"];
	if (model.length == 0) {
		XCTSkip("set INFERKIT_LIVE_VISION_MODEL (and run a local server) to exercise the live path");
	}
	NSString *identifier = environment[@"INFERKIT_LIVE_LOCAL_PROVIDER"] ?: @"ollama";
	id<NFKInferenceBackend> backend = [NFKRemoteProvider backendForProvider:[NFKRemoteProvider providerWithIdentifier:identifier]
																	 apiKey:nil modelName:model];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{
		NFKInputPrompt: @"These are frames from a short video, in order. Which single colour fills each frame? Answer with the colour names only.",
		NFKInputVideo: [NFKVideoAsset videoAssetWithFileURL:self.clip] }
		parameters:@{ NFKParameterVideoFrameCount: @4, NFKParameterMaxTokens: @512 } outputModality:NFKModalityText];
	NSError *error = nil;
	NFKInferenceResult *result = [backend runInferenceForRequest:request error:&error];
	XCTAssertNotNil(result, @"%@", error.localizedDescription);
	NSString *answer = result.text.lowercaseString;
	XCTAssertTrue([answer containsString:@"red"] && [answer containsString:@"blue"], @"the model answered: %@", result.text);
}

@end
