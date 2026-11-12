//
//  NFKLiveTimingTests.m
//  InferKitTests
//
//  The measured time per frame of the core's live engines through the live stream a client opens, at
//  1280 × 720 and 1920 × 1080. Set IK_TIME_LIVE_APPLE to the path of the JSON report to write. Runs only
//  in a release build, under the Large Model Coordination lock, on a quiet machine.
//

#import <XCTest/XCTest.h>
#import <CoreVideo/CoreVideo.h>
#import <InferKit/InferKit.h>

@interface NFKLiveTimingTests : XCTestCase
@end

@implementation NFKLiveTimingTests

static const NSInteger NFKLiveTimingWarmUpFrames = 3;
static const NSInteger NFKLiveTimingTimedFrames = 30;

/*! A BGRA frame: a gradient with a bright square in the middle, as a capture device hands one. */
static CVPixelBufferRef NFKLiveTimingFrame(size_t width, size_t height)
{
	CVPixelBufferRef buffer = NULL;
	NSDictionary *attributes = @{ (id)kCVPixelBufferIOSurfacePropertiesKey: @{} };
	if (CVPixelBufferCreate(NULL, width, height, kCVPixelFormatType_32BGRA, (__bridge CFDictionaryRef)attributes,
							&buffer) != kCVReturnSuccess) {
		return NULL;
	}
	CVPixelBufferLockBaseAddress(buffer, 0);
	uint8_t *pixels = CVPixelBufferGetBaseAddress(buffer);
	size_t rowBytes = CVPixelBufferGetBytesPerRow(buffer);
	for (size_t y = 0; y < height; y++) {
		for (size_t x = 0; x < width; x++) {
			BOOL inSquare = labs((long)x - (long)width / 2) < (long)width / 6 && labs((long)y - (long)height / 2) < (long)height / 4;
			uint8_t *pixel = pixels + y * rowBytes + x * 4;
			pixel[0] = (uint8_t)(x * 255 / width);
			pixel[1] = inSquare ? 230 : (uint8_t)(y * 255 / height);
			pixel[2] = inSquare ? 210 : 90;
			pixel[3] = 255;
		}
	}
	CVPixelBufferUnlockBaseAddress(buffer, 0);
	return buffer;
}

- (void)testEveryLiveEngineIsTimedThroughItsStream
{
	NSString *reportPath = NSProcessInfo.processInfo.environment[@"IK_TIME_LIVE_APPLE"];
	if (reportPath.length == 0) {
		XCTSkip(@"set IK_TIME_LIVE_APPLE to the report's path to time the live engines");
	}
#if DEBUG
	BOOL debugBuild = YES;
#else
	BOOL debugBuild = NO;
#endif
	if (debugBuild) {
		XCTSkip(@"time the live engines in a release build (swift test -c release)");
	}
	NSString *chip = NFKHardwareProfile.currentProfile.chipName;
	NSDateFormatter *formatter = [[NSDateFormatter alloc] init];
	formatter.dateFormat = @"yyyy-MM-dd";
	NSString *day = [formatter stringFromDate:[NSDate date]];
	NSMutableArray<NSDictionary *> *rows = [NSMutableArray array];
	for (NFKModelDescriptor *descriptor in [NFKAppleModelDescriptors modelDescriptors]) {
		if (descriptor.liveProfile == nil) {
			continue;
		}
		id<NFKInferenceBackend> backend = [NFKAppleModelDescriptors backendNamed:descriptor.identifier];
		XCTAssertNotNil(backend, @"%@", descriptor.identifier);
		for (NSArray<NSNumber *> *size in @[ @[ @1280, @720 ], @[ @1920, @1080 ] ]) {
			CVPixelBufferRef frame = NFKLiveTimingFrame(size[0].unsignedIntegerValue, size[1].unsignedIntegerValue);
			NFKInferenceRequest *unit = [NFKInferenceRequest requestWithInputs:@{ NFKInputImage: (__bridge id)frame }];
			NSError *error = nil;
			id<NFKLiveStream> stream = NFKInferenceOpenLiveStream(backend, nil, &error);
			XCTAssertNotNil(stream, @"%@", error);
			for (NSInteger index = 0; index < NFKLiveTimingWarmUpFrames; index++) {
				XCTAssertNotNil([stream resultForUnit:unit error:&error], @"%@ %@", descriptor.identifier, error);
			}
			NSMutableArray<NSNumber *> *times = [NSMutableArray array];
			for (NSInteger index = 0; index < NFKLiveTimingTimedFrames; index++) {
				uint64_t before = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
				XCTAssertNotNil([stream resultForUnit:unit error:&error], @"%@ %@", descriptor.identifier, error);
				[times addObject:@((double)(clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - before) / 1e6)];
			}
			[stream close];
			CVPixelBufferRelease(frame);
			NSArray<NSNumber *> *sorted = [times sortedArrayUsingSelector:@selector(compare:)];
			double median = (sorted[sorted.count / 2 - 1].doubleValue + sorted[sorted.count / 2].doubleValue) / 2;
			[rows addObject:@{ @"model": descriptor.identifier, @"chip": chip, @"width": size[0], @"height": size[1],
							   @"milliseconds": @(median), @"measured": day }];
			NSLog(@"LIVE-TIMING %@ %@x%@: %.3f ms per frame", descriptor.identifier, size[0], size[1], median);
		}
	}
	NSData *report = [NSJSONSerialization dataWithJSONObject:rows
													 options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys error:NULL];
	XCTAssertTrue([report writeToFile:reportPath atomically:YES]);
}

@end
