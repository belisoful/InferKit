//
//  NFKTensorConversionTests.m
//  NFKTests
//

#import <XCTest/XCTest.h>
#import <InferKit/NFKTensorConversion.h>

@interface NFKTensorConversionTests : XCTestCase
@end

@implementation NFKTensorConversionTests

// A 2x2 RGBA image (0..1), row-major, tightly packed.
static void FillSampleImage(float *rgba)
{
	// pixel (0,0) = (0.1,0.2,0.3,1), (1,0) = (0.4,0.5,0.6,1),
	//       (0,1) = (0.7,0.8,0.9,1), (1,1) = (0.15,0.25,0.35,1)
	float values[16] = {
		0.1f, 0.2f, 0.3f, 1.0f,   0.4f, 0.5f, 0.6f, 1.0f,
		0.7f, 0.8f, 0.9f, 1.0f,   0.15f, 0.25f, 0.35f, 1.0f,
	};
	memcpy(rgba, values, sizeof(values));
}

- (void)testSpecMakeIsIdentityRGBCHW
{
	NFKTensorSpec spec = NFKTensorSpecMake(4, 2, 3);
	XCTAssertEqual(spec.width, (NSUInteger)4);
	XCTAssertEqual(spec.height, (NSUInteger)2);
	XCTAssertEqual(spec.channelCount, (NSUInteger)3);
	XCTAssertEqual(spec.layout, NFKTensorLayoutCHW);
	XCTAssertEqual(spec.channelOrder, NFKTensorChannelOrderRGBA);
	XCTAssertEqual(spec.mean[0], 0.0f);
	XCTAssertEqual(spec.scale[0], 1.0f);
	XCTAssertEqual(NFKTensorElementCount(spec), (NSUInteger)(3 * 4 * 2));
}

- (void)testInterleavedToCHWIsPlanar
{
	float rgba[16];
	FillSampleImage(rgba);
	NFKTensorSpec spec = NFKTensorSpecMake(2, 2, 3);

	float tensor[12];
	NFKInterleavedToTensor(rgba, tensor, spec);

	// Plane R = all reds: (0,0)=0.1 (1,0)=0.4 (0,1)=0.7 (1,1)=0.15
	XCTAssertEqualWithAccuracy(tensor[0], 0.1f, 1e-6);
	XCTAssertEqualWithAccuracy(tensor[1], 0.4f, 1e-6);
	XCTAssertEqualWithAccuracy(tensor[2], 0.7f, 1e-6);
	XCTAssertEqualWithAccuracy(tensor[3], 0.15f, 1e-6);
	// Plane G starts at index 4: first green is 0.2
	XCTAssertEqualWithAccuracy(tensor[4], 0.2f, 1e-6);
	// Plane B starts at index 8: first blue is 0.3
	XCTAssertEqualWithAccuracy(tensor[8], 0.3f, 1e-6);
}

- (void)testInterleavedToHWCIsPixelInterleaved
{
	float rgba[16];
	FillSampleImage(rgba);
	NFKTensorSpec spec = NFKTensorSpecMake(2, 2, 3);
	spec.layout = NFKTensorLayoutHWC;

	float tensor[12];
	NFKInterleavedToTensor(rgba, tensor, spec);

	// Pixel 0 = R,G,B contiguous = 0.1,0.2,0.3
	XCTAssertEqualWithAccuracy(tensor[0], 0.1f, 1e-6);
	XCTAssertEqualWithAccuracy(tensor[1], 0.2f, 1e-6);
	XCTAssertEqualWithAccuracy(tensor[2], 0.3f, 1e-6);
	// Pixel 1 starts at index 3
	XCTAssertEqualWithAccuracy(tensor[3], 0.4f, 1e-6);
}

- (void)testBGRAOrderSwapsRedAndBlue
{
	float rgba[16];
	FillSampleImage(rgba);
	NFKTensorSpec spec = NFKTensorSpecMake(2, 2, 3);
	spec.channelOrder = NFKTensorChannelOrderBGRA;

	float tensor[12];
	NFKInterleavedToTensor(rgba, tensor, spec);

	// Output channel 0 now reads source blue: pixel (0,0) blue = 0.3
	XCTAssertEqualWithAccuracy(tensor[0], 0.3f, 1e-6);
	// Output channel 2 reads source red: 0.1
	XCTAssertEqualWithAccuracy(tensor[8], 0.1f, 1e-6);
}

- (void)testNormalizationSubtractsMeanAndScales
{
	float rgba[16];
	FillSampleImage(rgba);
	NFKTensorSpec spec = NFKTensorSpecMake(2, 2, 3);
	spec.mean[0] = spec.mean[1] = spec.mean[2] = 0.5f;
	spec.scale[0] = spec.scale[1] = spec.scale[2] = 2.0f;

	float tensor[12];
	NFKInterleavedToTensor(rgba, tensor, spec);

	// R channel pixel 0: (0.1 - 0.5) * 2 = -0.8
	XCTAssertEqualWithAccuracy(tensor[0], -0.8f, 1e-6);
}

- (void)testRoundTripRestoresTheImage
{
	float rgba[16];
	FillSampleImage(rgba);
	NFKTensorSpec spec = NFKTensorSpecMake(2, 2, 3);
	spec.mean[0] = spec.mean[1] = spec.mean[2] = 0.5f;
	spec.scale[0] = spec.scale[1] = spec.scale[2] = 2.0f;

	float tensor[12];
	NFKInterleavedToTensor(rgba, tensor, spec);

	float restored[16];
	NFKTensorToInterleaved(tensor, restored, spec);

	for (int i = 0; i < 4; i++) {
		XCTAssertEqualWithAccuracy(restored[i * 4 + 0], rgba[i * 4 + 0], 1e-5);
		XCTAssertEqualWithAccuracy(restored[i * 4 + 1], rgba[i * 4 + 1], 1e-5);
		XCTAssertEqualWithAccuracy(restored[i * 4 + 2], rgba[i * 4 + 2], 1e-5);
		XCTAssertEqualWithAccuracy(restored[i * 4 + 3], 1.0f, 1e-6, @"a 3-channel tensor restores opaque alpha");
	}
}

@end
