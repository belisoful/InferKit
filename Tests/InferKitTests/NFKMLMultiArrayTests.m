//
//  NFKMLMultiArrayTests.m
//  InferKitTests
//

#import <XCTest/XCTest.h>
#import <InferKit/NFKMLMultiArray.h>
#import <InferKit/NFKTensorConversion.h>
#import <CoreML/CoreML.h>

@interface NFKMLMultiArrayTests : XCTestCase
@end

@implementation NFKMLMultiArrayTests

static void FillSampleImage(float *rgba)
{
	float values[16] = {
		0.1f, 0.2f, 0.3f, 1.0f,   0.4f, 0.5f, 0.6f, 1.0f,
		0.7f, 0.8f, 0.9f, 1.0f,   0.15f, 0.25f, 0.35f, 1.0f,
	};
	memcpy(rgba, values, sizeof(values));
}

- (void)testCHWMultiArrayHasTheRightShapeAndPlanarValues
{
	float rgba[16];
	FillSampleImage(rgba);
	NFKTensorSpec spec = NFKTensorSpecMake(2, 2, 3);

	NSError *error = nil;
	MLMultiArray *array = NFKMultiArrayFromInterleaved(rgba, spec, &error);
	XCTAssertNil(error);
	XCTAssertNotNil(array);
	XCTAssertEqualObjects(array.shape, (@[@1, @3, @2, @2]));
	XCTAssertEqual(array.dataType, MLMultiArrayDataTypeFloat32);

	const float *data = (const float *)array.dataPointer;
	// R plane first: 0.1, 0.4, 0.7, 0.15
	XCTAssertEqualWithAccuracy(data[0], 0.1f, 1e-6);
	XCTAssertEqualWithAccuracy(data[1], 0.4f, 1e-6);
	// G plane starts at index 4
	XCTAssertEqualWithAccuracy(data[4], 0.2f, 1e-6);
	// B plane starts at index 8
	XCTAssertEqualWithAccuracy(data[8], 0.3f, 1e-6);
}

- (void)testHWCMultiArrayShape
{
	float rgba[16];
	FillSampleImage(rgba);
	NFKTensorSpec spec = NFKTensorSpecMake(2, 2, 3);
	spec.layout = NFKTensorLayoutHWC;

	MLMultiArray *array = NFKMultiArrayFromInterleaved(rgba, spec, NULL);
	XCTAssertEqualObjects(array.shape, (@[@1, @2, @2, @3]));
	const float *data = (const float *)array.dataPointer;
	// First pixel R,G,B contiguous
	XCTAssertEqualWithAccuracy(data[0], 0.1f, 1e-6);
	XCTAssertEqualWithAccuracy(data[1], 0.2f, 1e-6);
	XCTAssertEqualWithAccuracy(data[2], 0.3f, 1e-6);
}

- (void)testRoundTripThroughAMultiArrayRestoresTheImage
{
	float rgba[16];
	FillSampleImage(rgba);
	NFKTensorSpec spec = NFKTensorSpecMake(2, 2, 3);
	spec.mean[0] = spec.mean[1] = spec.mean[2] = 0.5f;
	spec.scale[0] = spec.scale[1] = spec.scale[2] = 2.0f;

	MLMultiArray *array = NFKMultiArrayFromInterleaved(rgba, spec, NULL);
	float restored[16];
	XCTAssertTrue(NFKInterleavedFromMultiArray(array, restored, spec));

	for (int i = 0; i < 4; i++) {
		XCTAssertEqualWithAccuracy(restored[i * 4 + 0], rgba[i * 4 + 0], 1e-5);
		XCTAssertEqualWithAccuracy(restored[i * 4 + 1], rgba[i * 4 + 1], 1e-5);
		XCTAssertEqualWithAccuracy(restored[i * 4 + 2], rgba[i * 4 + 2], 1e-5);
		XCTAssertEqualWithAccuracy(restored[i * 4 + 3], 1.0f, 1e-6);
	}
}

@end
