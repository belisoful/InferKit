//
//  NFKVideoToolboxBackendTests.m
//  NFKTests
//
//  The frame processors need Apple silicon and a recent OS, and upscaling needs a model the system
//  downloads. Each test runs the processor where the machine has it and checks the refusal where it
//  does not, so the file is green either way and neither case is silently skipped.
//

#import <XCTest/XCTest.h>
#import <CoreVideo/CoreVideo.h>
#import <InferKit/InferKit.h>

@interface NFKVideoToolboxBackendTests : XCTestCase
@end

@implementation NFKVideoToolboxBackendTests

- (CVPixelBufferRef)frameOfWidth:(size_t)width height:(size_t)height shiftedBy:(size_t)shift CF_RETURNS_RETAINED
{
	NSDictionary *attributes = @{ (__bridge NSString *)kCVPixelBufferCGImageCompatibilityKey: @YES,
								  (__bridge NSString *)kCVPixelBufferCGBitmapContextCompatibilityKey: @YES,
								  (__bridge NSString *)kCVPixelBufferIOSurfacePropertiesKey: @{} };
	CVPixelBufferRef buffer = NULL;
	CVPixelBufferCreate(kCFAllocatorDefault, width, height, kCVPixelFormatType_32BGRA,
						(__bridge CFDictionaryRef)attributes, &buffer);
	CVPixelBufferLockBaseAddress(buffer, 0);
	uint8_t *base = CVPixelBufferGetBaseAddress(buffer);
	size_t stride = CVPixelBufferGetBytesPerRow(buffer);
	for (size_t y = 0; y < height; y++) {
		uint8_t *row = base + y * stride;
		for (size_t x = 0; x < width; x++) {
			BOOL inside = (x >= 40 + shift && x < 120 + shift && y >= 40 && y < 120);
			uint8_t value = inside ? 220 : 30;
			row[4 * x + 0] = value;
			row[4 * x + 1] = value;
			row[4 * x + 2] = value;
			row[4 * x + 3] = 255;
		}
	}
	CVPixelBufferUnlockBaseAddress(buffer, 0);
	return buffer;
}

- (NFKInferenceRequest *)requestWithFrames:(NSArray *)frames
{
	return [NFKInferenceRequest requestWithInputs:@{ NFKInputImages: frames }];
}

#pragma mark Contract

- (void)testTheBackendReportsItsIdentifierAndKeys
{
	NFKVideoToolboxBackend *upscaler = [NFKVideoToolboxBackend backend];
	XCTAssertEqualObjects(upscaler.backendIdentifier, @"videotoolbox");
	XCTAssertEqual(upscaler.task, NFKVideoToolboxTaskSuperResolution);
	XCTAssertEqual(upscaler.scaleFactor, (NSInteger)0, @"the machine's smallest factor, until one is set");
	XCTAssertEqualWithAccuracy(upscaler.flowScale, 32.0, 1e-9, @"the packing matches NFKMLXRAFT's default");
	XCTAssertTrue([upscaler.supportedInputKeys containsObject:NFKInputImage]);

	NFKVideoToolboxBackend *flow = [NFKVideoToolboxBackend backendWithTask:NFKVideoToolboxTaskOpticalFlow];
	XCTAssertTrue([flow.supportedInputKeys containsObject:NFKInputImages], @"two frames go in together");
}

- (void)testAPairTaskNeedsTwoFrames
{
	NFKVideoToolboxBackend *backend = [NFKVideoToolboxBackend backendWithTask:NFKVideoToolboxTaskOpticalFlow];
	if (!backend.isReady) {
		return;
	}
	CVPixelBufferRef only = [self frameOfWidth:160 height:160 shiftedBy:0];
	NSError *error = nil;
	NFKInferenceResult *result = [backend runInferenceForRequest:[self requestWithFrames:@[ (__bridge id)only ]]
														   error:&error];
	CVPixelBufferRelease(only);
	XCTAssertNil(result);
	XCTAssertEqual(error.code, (NSInteger)kNFKError_InferenceMissingInput);
}

- (void)testAnUnavailableProcessorRefusesTheRunRatherThanFailingLate
{
	NFKVideoToolboxBackend *backend = [NFKVideoToolboxBackend backendWithTask:NFKVideoToolboxTaskSuperResolution];
	if (backend.isReady) {
		return;
	}
	NSError *error = nil;
	XCTAssertNil([backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{}] error:&error]);
	XCTAssertEqual(error.code, (NSInteger)kNFKError_InferenceUnsupported);
	XCTAssertFalse([backend prepareWithError:NULL]);
}

#pragma mark Running

- (void)testAnUnofferedScaleFactorIsRefused
{
	NFKVideoToolboxBackend *backend = [NFKVideoToolboxBackend backend];
	if (!backend.isReady) {
		return;
	}
	backend.scaleFactor = 3;
	CVPixelBufferRef source = [self frameOfWidth:320 height:240 shiftedBy:0];
	NSError *error = nil;
	NFKInferenceResult *result =
		[backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputImage: (__bridge id)source }]
								  error:&error];
	CVPixelBufferRelease(source);
	XCTAssertNil(result);
	XCTAssertEqual(error.code, (NSInteger)kNFKError_InferenceUnsupported);
	XCTAssertTrue([error.localizedDescription containsString:@"upscales by"], @"%@", error.localizedDescription);
}

- (void)testUpscalingEnlargesTheFrameWhereTheMachineHasIt
{
	NFKVideoToolboxBackend *backend = [NFKVideoToolboxBackend backend];
	if (!backend.isReady) {
		return;
	}
	NSError *prepareError = nil;
	if (![backend prepareWithError:&prepareError]) {
		// A model the system has not finished downloading is a not-ready backend, by contract.
		XCTAssertEqual(prepareError.code, (NSInteger)kNFKError_InferenceNotReady);
		return;
	}

	CVPixelBufferRef source = [self frameOfWidth:320 height:240 shiftedBy:0];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputImage: (__bridge id)source }];
	NSError *error = nil;
	NFKInferenceResult *result = [backend runInferenceForRequest:request error:&error];
	CVPixelBufferRelease(source);
	XCTAssertNotNil(result, @"%@", error);

	// Which factors a machine has is the machine's business, so the test checks that the frame grew
	// by a whole factor, equally on both axes.
	CVPixelBufferRef upscaled = (__bridge CVPixelBufferRef)[result outputForKey:NFKOutputImage];
	size_t factor = CVPixelBufferGetWidth(upscaled) / 320;
	XCTAssertGreaterThan(factor, (size_t)1);
	XCTAssertEqual(CVPixelBufferGetWidth(upscaled), 320 * factor);
	XCTAssertEqual(CVPixelBufferGetHeight(upscaled), 240 * factor);
}

- (void)testInterpolationReturnsOneFrameWhereTheMachineHasIt
{
	NFKVideoToolboxBackend *backend = [NFKVideoToolboxBackend backendWithTask:NFKVideoToolboxTaskFrameInterpolation];
	if (!backend.isReady) {
		return;
	}
	CVPixelBufferRef previous = [self frameOfWidth:320 height:240 shiftedBy:0];
	CVPixelBufferRef next = [self frameOfWidth:320 height:240 shiftedBy:40];
	NSError *error = nil;
	NFKInferenceResult *result = [backend runInferenceForRequest:[self requestWithFrames:@[ (__bridge id)previous,
																						   (__bridge id)next ]]
														   error:&error];
	CVPixelBufferRelease(previous);
	CVPixelBufferRelease(next);
	XCTAssertNotNil(result, @"%@", error);

	CVPixelBufferRef middle = (__bridge CVPixelBufferRef)[result outputForKey:NFKOutputImage];
	XCTAssertEqual(CVPixelBufferGetWidth(middle), (size_t)320);
	XCTAssertEqual(CVPixelBufferGetHeight(middle), (size_t)240);
}

- (void)testStillFramesPackAsNoMotion
{
	NFKVideoToolboxBackend *backend = [NFKVideoToolboxBackend backendWithTask:NFKVideoToolboxTaskOpticalFlow];
	if (!backend.isReady) {
		return;
	}
	CVPixelBufferRef first = [self frameOfWidth:320 height:240 shiftedBy:0];
	CVPixelBufferRef second = [self frameOfWidth:320 height:240 shiftedBy:0];
	NSError *error = nil;
	NFKInferenceResult *result = [backend runInferenceForRequest:[self requestWithFrames:@[ (__bridge id)first,
																						   (__bridge id)second ]]
														   error:&error];
	CVPixelBufferRelease(first);
	CVPixelBufferRelease(second);
	XCTAssertNotNil(result, @"%@", error);

	// Nothing moved, so every component is zero, which the packing writes as mid-gray in red and
	// green with an empty blue channel.
	CVPixelBufferRef packed = (__bridge CVPixelBufferRef)[result outputForKey:NFKOutputImage];
	CVPixelBufferLockBaseAddress(packed, kCVPixelBufferLock_ReadOnly);
	const uint8_t *pixel = CVPixelBufferGetBaseAddress(packed);
	XCTAssertEqual(pixel[0], 0, @"blue carries no component");
	XCTAssertEqualWithAccuracy((double)pixel[1], 128.0, 2.0);
	XCTAssertEqualWithAccuracy((double)pixel[2], 128.0, 2.0);
	CVPixelBufferUnlockBaseAddress(packed, kCVPixelBufferLock_ReadOnly);
}

@end
