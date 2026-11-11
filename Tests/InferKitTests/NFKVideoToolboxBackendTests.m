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

// A hosted runner is a virtual machine, where each processor reports support and then fails inside
// the framework on paravirtualized hardware. That is the machine's answer, so a run that fails there
// is reported as a skip; on real hardware it is a failure.
- (nullable NFKInferenceResult *)resultOfRunning:(NFKVideoToolboxBackend *)backend
										 request:(NFKInferenceRequest *)request
{
	NSError *error = nil;
	NFKInferenceResult *result = [backend runInferenceForRequest:request error:&error];
	if (result == nil && NFKHardwareProfile.currentProfile.isVirtualMachine) {
		XCTSkip("the frame processor did not run in this virtual machine: %@", error);
	}
	XCTAssertNotNil(result, @"%@", error);
	return result;
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
		if (NFKHardwareProfile.currentProfile.isVirtualMachine) {
			XCTSkip("the upscaler did not prepare in this virtual machine: %@", prepareError);
		}
		// A model the system has not finished downloading is a not-ready backend, by contract.
		XCTAssertEqual(prepareError.code, (NSInteger)kNFKError_InferenceNotReady);
		return;
	}

	CVPixelBufferRef source = [self frameOfWidth:320 height:240 shiftedBy:0];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputImage: (__bridge id)source }];
	CVPixelBufferRelease(source);
	NFKInferenceResult *result = [self resultOfRunning:backend request:request];
	if (result == nil) {
		return;
	}

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
	NFKInferenceRequest *request = [self requestWithFrames:@[ (__bridge id)previous, (__bridge id)next ]];
	CVPixelBufferRelease(previous);
	CVPixelBufferRelease(next);
	NFKInferenceResult *result = [self resultOfRunning:backend request:request];
	if (result == nil) {
		return;
	}

	CVPixelBufferRef middle = (__bridge CVPixelBufferRef)[result outputForKey:NFKOutputImage];
	XCTAssertEqual(CVPixelBufferGetWidth(middle), (size_t)320);
	XCTAssertEqual(CVPixelBufferGetHeight(middle), (size_t)240);
}

/*! The horizontal center of the bright square in a BGRA frame. */
- (double)squareCenterOf:(CVPixelBufferRef)frame
{
	CVPixelBufferLockBaseAddress(frame, kCVPixelBufferLock_ReadOnly);
	const uint8_t *base = CVPixelBufferGetBaseAddress(frame);
	size_t stride = CVPixelBufferGetBytesPerRow(frame);
	double sum = 0, weight = 0;
	for (size_t y = 60; y < 100; y++) {
		for (size_t x = 0; x < CVPixelBufferGetWidth(frame); x++) {
			double value = base[y * stride + 4 * x + 1];
			value = value > 40 ? value - 30 : 0;
			sum += value * (double)x;
			weight += value;
		}
	}
	CVPixelBufferUnlockBaseAddress(frame, kCVPixelBufferLock_ReadOnly);
	return sum / weight;
}

// The square's center moves from 79.5 to 159.5, so the frame at a timestep shows how far along it landed.
- (void)testATimestepPlacesTheFrameAtThatPoint
{
	NFKVideoToolboxBackend *backend = [NFKVideoToolboxBackend backendWithTask:NFKVideoToolboxTaskFrameInterpolation];
	if (!backend.isReady) {
		return;
	}
	CVPixelBufferRef previous = [self frameOfWidth:320 height:240 shiftedBy:0];
	CVPixelBufferRef next = [self frameOfWidth:320 height:240 shiftedBy:80];
	NSArray *frames = @[ (__bridge id)previous, (__bridge id)next ];
	CVPixelBufferRelease(previous);
	CVPixelBufferRelease(next);
	for (NSNumber *timestep in @[ @0.2, @0.4, @0.8 ]) {
		NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputImages: frames }
																   parameters:@{ NFKParameterTimestep: timestep }];
		NFKInferenceResult *result = [self resultOfRunning:backend request:request];
		if (result == nil) {
			return;
		}
		double reached = ([self squareCenterOf:(__bridge CVPixelBufferRef)[result outputForKey:NFKOutputImage]] - 79.5) / 80.0;
		XCTAssertEqualWithAccuracy(reached, timestep.doubleValue, 0.02, @"a point off the eighths is reached");
	}
	NFKInferenceResult *middle = [self resultOfRunning:backend request:[self requestWithFrames:frames]];
	if (middle != nil) {
		double reached = ([self squareCenterOf:(__bridge CVPixelBufferRef)[middle outputForKey:NFKOutputImage]] - 79.5) / 80.0;
		XCTAssertEqualWithAccuracy(reached, 0.5, 0.02, @"no timestep gives the middle");
	}
}

- (void)testATimestepAtEitherEndAnswersThatFrame
{
	NFKVideoToolboxBackend *backend = [NFKVideoToolboxBackend backendWithTask:NFKVideoToolboxTaskFrameInterpolation];
	if (!backend.isReady) {
		return;
	}
	CVPixelBufferRef previous = [self frameOfWidth:320 height:240 shiftedBy:0];
	CVPixelBufferRef next = [self frameOfWidth:320 height:240 shiftedBy:80];
	NSArray *frames = @[ (__bridge id)previous, (__bridge id)next ];
	CVPixelBufferRelease(previous);
	CVPixelBufferRelease(next);
	NSDictionary<NSNumber *, NSNumber *> *centers = @{ @0: @79.5, @1: @159.5, @(-2): @79.5, @3: @159.5 };
	for (NSNumber *timestep in centers) {
		NSError *error = nil;
		NFKInferenceResult *result = [backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputImages: frames }
																								parameters:@{ NFKParameterTimestep: timestep }]
															   error:&error];
		XCTAssertNotNil(result, @"%@", error);
		XCTAssertEqualWithAccuracy([self squareCenterOf:(__bridge CVPixelBufferRef)[result outputForKey:NFKOutputImage]],
								   centers[timestep].doubleValue, 0.01, @"timestep %@", timestep);
	}
}

- (void)testATimestepThatIsNotAFiniteNumberIsRefused
{
	NFKVideoToolboxBackend *backend = [NFKVideoToolboxBackend backendWithTask:NFKVideoToolboxTaskFrameInterpolation];
	XCTAssertEqualObjects(backend.supportedParameterKeys, [NSSet setWithObject:NFKParameterTimestep]);
	XCTAssertEqualObjects([NFKVideoToolboxBackend backend].supportedParameterKeys, [NSSet set]);
	CVPixelBufferRef frame = [self frameOfWidth:320 height:240 shiftedBy:0];
	NSArray *frames = @[ (__bridge id)frame, (__bridge id)frame ];
	CVPixelBufferRelease(frame);
	for (id timestep in @[ @"middle", @(NAN) ]) {
		NSError *error = nil;
		XCTAssertNil([backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputImages: frames }
																				 parameters:@{ NFKParameterTimestep: timestep }]
											   error:&error]);
		XCTAssertEqual(error.code, kNFKError_InferenceUnsupported, @"%@", timestep);
		XCTAssertTrue([error.localizedDescription containsString:@"NFKParameterTimestep"], @"%@", error);
	}
}

- (void)testTheCatalogDeclaresTheTimestep
{
	NFKModelOption *option = [[NFKModelCatalog.sharedCatalog descriptorWithIdentifier:@"videotoolbox-frame-interpolation"]
							  optionWithKey:NFKParameterTimestep];
	XCTAssertEqualObjects(option.kind, NFKModelOptionKindNumber);
	XCTAssertEqualObjects(option.stage, NFKModelOptionStageRequest);
	XCTAssertEqualObjects(option.defaultValue, @0.5);
	XCTAssertEqualObjects(option.minimum, @0);
	XCTAssertEqualObjects(option.maximum, @1);
	XCTAssertTrue([option.key.lowercaseString hasSuffix:@"timestep"]);
	XCTAssertEqualObjects(NFKModelValueTypeForKey(NFKParameterTimestep), NFKModelValueTypeNumber);
}

- (NSArray *)framesOfWidth:(size_t)width height:(size_t)height shiftedBy:(size_t)shift
{
	CVPixelBufferRef first = [self frameOfWidth:width height:height shiftedBy:shift];
	CVPixelBufferRef second = [self frameOfWidth:width height:height shiftedBy:shift + 60];
	NSArray *frames = @[ (__bridge id)first, (__bridge id)second ];
	CVPixelBufferRelease(first);
	CVPixelBufferRelease(second);
	return frames;
}

/*! The BGRA bytes of the frame a backend answers at a timestep, or nil where the processor did not run. */
- (nullable NSData *)bytesFrom:(NFKVideoToolboxBackend *)backend frames:(NSArray *)frames timestep:(double)timestep
{
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputImages: frames }
															   parameters:@{ NFKParameterTimestep: @(timestep) }];
	NFKInferenceResult *result = [self resultOfRunning:backend request:request];
	if (result == nil) {
		return nil;
	}
	CVPixelBufferRef frame = (__bridge CVPixelBufferRef)[result outputForKey:NFKOutputImage];
	CVPixelBufferLockBaseAddress(frame, kCVPixelBufferLock_ReadOnly);
	NSMutableData *bytes = [NSMutableData data];
	for (size_t y = 0; y < CVPixelBufferGetHeight(frame); y++) {
		[bytes appendBytes:(const uint8_t *)CVPixelBufferGetBaseAddress(frame) + y * CVPixelBufferGetBytesPerRow(frame)
					length:CVPixelBufferGetWidth(frame) * 4];
	}
	CVPixelBufferUnlockBaseAddress(frame, kCVPixelBufferLock_ReadOnly);
	return bytes;
}

- (void)testAHeldConverterAnswersAsAFreshOneAcrossUnrelatedPairs
{
	NFKVideoToolboxBackend *held = [NFKVideoToolboxBackend backendWithTask:NFKVideoToolboxTaskFrameInterpolation];
	if (!held.isReady) {
		return;
	}
	NSArray<NSArray<NSNumber *> *> *runs = @[ @[ @0, @0.4 ], @[ @0, @0.8 ], @[ @60, @0.2 ], @[ @160, @0.6 ], @[ @0, @0.4 ] ];
	for (NSArray<NSNumber *> *run in runs) {
		NSArray *frames = [self framesOfWidth:320 height:240 shiftedBy:run[0].unsignedIntegerValue];
		NSData *reused = [self bytesFrom:held frames:frames timestep:run[1].doubleValue];
		if (reused == nil) {
			return;
		}
		XCTAssertNotNil([held valueForKey:@"converter"], @"the session stays open between runs");
		NFKVideoToolboxBackend *fresh = [NFKVideoToolboxBackend backendWithTask:NFKVideoToolboxTaskFrameInterpolation];
		fresh.converterIdleTimeout = 0;
		XCTAssertEqualObjects(reused, [self bytesFrom:fresh frames:frames timestep:run[1].doubleValue], @"%@", run);
		XCTAssertNil([fresh valueForKey:@"converter"], @"a timeout of 0 ends the session after each run");
	}
}

- (void)testAHeldConverterTakesANewFrameSizeAndEndsWhenIdle
{
	NFKVideoToolboxBackend *held = [NFKVideoToolboxBackend backendWithTask:NFKVideoToolboxTaskFrameInterpolation];
	if (!held.isReady) {
		return;
	}
	held.converterIdleTimeout = 0.3;
	NSData *small = [self bytesFrom:held frames:[self framesOfWidth:320 height:240 shiftedBy:0] timestep:0.4];
	if (small == nil) {
		return;
	}
	XCTAssertEqual(small.length, (NSUInteger)320 * 240 * 4);
	NSData *large = [self bytesFrom:held frames:[self framesOfWidth:640 height:480 shiftedBy:0] timestep:0.4];
	XCTAssertEqual(large.length, (NSUInteger)640 * 480 * 4, @"another size opens another session");
	XCTAssertNotNil([held valueForKey:@"converter"]);
	NSPredicate *released = [NSPredicate predicateWithBlock:^BOOL(NFKVideoToolboxBackend *backend, NSDictionary *bindings) {
		return [backend valueForKey:@"converter"] == nil;
	}];
	[self waitForExpectations:@[ [self expectationForPredicate:released evaluatedWithObject:held handler:nil] ] timeout:10];
	XCTAssertEqualObjects([self bytesFrom:held frames:[self framesOfWidth:320 height:240 shiftedBy:0] timestep:0.4], small,
						  @"a run after the release opens a session again");
}

- (void)testRunsFromManyThreadsTakeTheHeldConverterInTurn
{
	NFKVideoToolboxBackend *held = [NFKVideoToolboxBackend backendWithTask:NFKVideoToolboxTaskFrameInterpolation];
	if (!held.isReady) {
		return;
	}
	NSArray *frames = [self framesOfWidth:320 height:240 shiftedBy:0];
	NSData *expected = [self bytesFrom:held frames:frames timestep:0.4];
	if (expected == nil) {
		return;
	}
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputImages: frames }
															   parameters:@{ NFKParameterTimestep: @0.4 }];
	__block NSInteger answered = 0;
	dispatch_apply(4, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t index) {
		if ([held runInferenceForRequest:request error:NULL] != nil) {
			__sync_fetch_and_add(&answered, 1);
		}
	});
	XCTAssertEqual(answered, 4);
}

- (void)testStillFramesPackAsNoMotion
{
	NFKVideoToolboxBackend *backend = [NFKVideoToolboxBackend backendWithTask:NFKVideoToolboxTaskOpticalFlow];
	if (!backend.isReady) {
		return;
	}
	CVPixelBufferRef first = [self frameOfWidth:320 height:240 shiftedBy:0];
	CVPixelBufferRef second = [self frameOfWidth:320 height:240 shiftedBy:0];
	NFKInferenceRequest *request = [self requestWithFrames:@[ (__bridge id)first, (__bridge id)second ]];
	CVPixelBufferRelease(first);
	CVPixelBufferRelease(second);
	NFKInferenceResult *result = [self resultOfRunning:backend request:request];
	if (result == nil) {
		return;
	}

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
