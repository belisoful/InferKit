//
//  NFKGaussianSplatsTests.m
//  InferKitTests
//

#import <XCTest/XCTest.h>
#import <InferKit/NFKGaussianSplats.h>
#import <InferKit/NFKInferenceResult.h>
#import <InferKit/NFKInferenceKeys.h>
#import <InferKit/NFKModelVocabulary.h>
#import <InferKit/NFKValueCoding.h>
#import <InferKit/NFKErrors.h>

static NSData *NFKFloats(NSArray<NSNumber *> *values)
{
	NSMutableData *data = [NSMutableData dataWithLength:values.count * sizeof(float)];
	float *out = data.mutableBytes;
	for (NSUInteger index = 0; index < values.count; index++) {
		out[index] = values[index].floatValue;
	}
	return data;
}

/*! Reads the float the PLY body holds at row and column (17 floats a row after the header). */
static float NFKPLYFloat(NSData *ply, NSUInteger row, NSUInteger column)
{
	NSRange end = [ply rangeOfData:[@"end_header\n" dataUsingEncoding:NSASCIIStringEncoding] options:0 range:NSMakeRange(0, ply.length)];
	float value = 0;
	[ply getBytes:&value range:NSMakeRange(NSMaxRange(end) + (row * 17 + column) * sizeof(float), sizeof(float))];
	return value;
}

@interface NFKGaussianSplatsTests : XCTestCase
@end

@implementation NFKGaussianSplatsTests

/*! Two Gaussians: one unrotated, one a quarter turn about y. */
- (NFKGaussianSplats *)pair
{
	float half = sqrtf(0.5f);
	NSError *error = nil;
	NFKGaussianSplats *splats = [[NFKGaussianSplats alloc] initWithPositions:NFKFloats(@[ @1, @2, @3, @-0.25, @0.5, @0 ])
																	 scales:NFKFloats(@[ @0.1, @0.2, @0.3, @0.01, @0.01, @0.02 ])
																  rotations:NFKFloats(@[ @1, @0, @0, @0, @(half), @0, @(half), @0 ])
																  opacities:NFKFloats(@[ @0.5, @0.9 ])
																	 colors:NFKFloats(@[ @0, @1, @-1, @0.5, @0.5, @0.5 ])
																	  error:&error];
	XCTAssertNotNil(splats, @"%@", error);
	return splats;
}

- (void)assertSplats:(NFKGaussianSplats *)a closeTo:(NFKGaussianSplats *)b
{
	XCTAssertEqual(a.count, b.count);
	for (NSUInteger index = 0; index < MIN(a.count, b.count); index++) {
		NFKMeshVector pa = [a positionAtIndex:index], pb = [b positionAtIndex:index];
		NFKMeshVector sa = [a scaleAtIndex:index], sb = [b scaleAtIndex:index];
		NFKGaussianRotation ra = [a rotationAtIndex:index], rb = [b rotationAtIndex:index];
		float sign = (ra.w * rb.w + ra.x * rb.x + ra.y * rb.y + ra.z * rb.z) < 0 ? -1 : 1;
		XCTAssertEqualWithAccuracy(pa.x, pb.x, 1e-6);
		XCTAssertEqualWithAccuracy(pa.y, pb.y, 1e-6);
		XCTAssertEqualWithAccuracy(pa.z, pb.z, 1e-6);
		XCTAssertEqualWithAccuracy(sa.x, sb.x, 1e-6);
		XCTAssertEqualWithAccuracy(sa.y, sb.y, 1e-6);
		XCTAssertEqualWithAccuracy(sa.z, sb.z, 1e-6);
		XCTAssertEqualWithAccuracy(ra.w, sign * rb.w, 1e-6);
		XCTAssertEqualWithAccuracy(ra.x, sign * rb.x, 1e-6);
		XCTAssertEqualWithAccuracy(ra.y, sign * rb.y, 1e-6);
		XCTAssertEqualWithAccuracy(ra.z, sign * rb.z, 1e-6);
		XCTAssertEqualWithAccuracy([a opacityAtIndex:index], [b opacityAtIndex:index], 1e-6);
		NFKMeshVector ca = [a colorAtIndex:index], cb = [b colorAtIndex:index];
		XCTAssertEqualWithAccuracy(ca.x, cb.x, 1e-6);
		XCTAssertEqualWithAccuracy(ca.y, cb.y, 1e-6);
		XCTAssertEqualWithAccuracy(ca.z, cb.z, 1e-6);
	}
}

- (void)testThePLYRoundTripsEveryAttribute
{
	NFKGaussianSplats *splats = self.pair;
	NSError *error = nil;
	NFKGaussianSplats *read = [NFKGaussianSplats splatsWithPLYData:splats.PLYData error:&error];
	XCTAssertNotNil(read, @"%@", error);
	[self assertSplats:splats closeTo:read];
}

- (void)testThePLYIsInTheSplattingFrameAndStoresLogitsAndLogarithms
{
	NSData *ply = self.pair.PLYData;
	NSString *header = [[NSString alloc] initWithData:[ply subdataWithRange:NSMakeRange(0, 120)] encoding:NSASCIIStringEncoding];
	XCTAssertTrue([header hasPrefix:@"ply\nformat binary_little_endian 1.0\nelement vertex 2\nproperty float x\n"], @"%@", header);
	XCTAssertEqual(NFKPLYFloat(ply, 0, 0), 1.0f);
	XCTAssertEqual(NFKPLYFloat(ply, 0, 1), -2.0f);
	XCTAssertEqual(NFKPLYFloat(ply, 0, 2), -3.0f);
	XCTAssertEqualWithAccuracy(NFKPLYFloat(ply, 0, 9), 0.0f, 1e-7);                       // logit(0.5)
	XCTAssertEqualWithAccuracy(NFKPLYFloat(ply, 0, 10), logf(0.1f), 1e-6);
	// Half a turn about x applied to the identity: rot_1 (x) is one.
	XCTAssertEqualWithAccuracy(fabsf(NFKPLYFloat(ply, 0, 14)), 1.0f, 1e-7);
	XCTAssertEqualWithAccuracy(NFKPLYFloat(ply, 0, 13), 0.0f, 1e-7);
	XCTAssertEqual(NFKPLYFloat(ply, 0, 6), 0.0f);
	XCTAssertEqual(NFKPLYFloat(ply, 0, 7), 1.0f);
}

- (void)testTheDisplayedColorComesFromTheDCCoefficient
{
	NFKMeshVector color = [self.pair colorAtIndex:0];
	XCTAssertEqualWithAccuracy(color.x, 0.5f, 1e-7);
	XCTAssertEqualWithAccuracy(color.y, 0.5f + 0.28209479f, 1e-6);
	XCTAssertEqualWithAccuracy(color.z, 0.5f - 0.28209479f, 1e-6);
}

- (void)testMismatchedLengthsAreRefused
{
	NSError *error = nil;
	NFKGaussianSplats *splats = [[NFKGaussianSplats alloc] initWithPositions:NFKFloats(@[ @0, @0, @0 ]) scales:NFKFloats(@[ @1, @1 ])
																  rotations:NFKFloats(@[ @1, @0, @0, @0 ]) opacities:NFKFloats(@[ @1 ])
																	 colors:NFKFloats(@[ @0, @0, @0 ]) error:&error];
	XCTAssertNil(splats);
	XCTAssertEqualObjects(error.domain, NFKInferenceErrorDomain);
	XCTAssertEqual(error.code, kNFKError_InferenceUnsupported);
	XCTAssertNil([NFKGaussianSplats splatsWithPLYData:[@"ply\nformat ascii 1.0\nend_header\n" dataUsingEncoding:NSASCIIStringEncoding]
												error:&error]);
	XCTAssertEqualObjects(error.domain, NSCocoaErrorDomain);
	XCTAssertEqual(error.code, NSFileReadCorruptFileError);
}

- (void)testTheSplatsArchiveAndCompareByValue
{
	NFKGaussianSplats *splats = self.pair;
	NSError *error = nil;
	NSData *archive = [NSKeyedArchiver archivedDataWithRootObject:splats requiringSecureCoding:YES error:&error];
	XCTAssertNotNil(archive, @"%@", error);
	NFKGaussianSplats *restored = [NSKeyedUnarchiver unarchivedObjectOfClass:NFKGaussianSplats.class fromData:archive error:&error];
	XCTAssertEqualObjects(restored, splats, @"%@", error);
	XCTAssertEqual(restored.hash, splats.hash);
}

- (void)testTheOutputKeyHasItsValueTypeAndConvenience
{
	XCTAssertEqualObjects(NFKModelValueTypeForOutputKey(NFKOutputGaussianSplats), NFKModelValueTypeGaussianSplats);
	NFKGaussianSplats *splats = self.pair;
	NFKInferenceResult *result = [NFKInferenceResult resultWithOutputs:@{ NFKOutputGaussianSplats: splats }];
	XCTAssertEqualObjects(result.gaussianSplats, splats);
	NFKInferenceResult *other = [NFKInferenceResult resultWithOutputs:@{ NFKOutputGaussianSplats: @"not splats" }];
	XCTAssertNil(other.gaussianSplats);
}

- (void)testValueCodingReadsThePLYAndCarriesTheSplatsAsJSON
{
	NFKGaussianSplats *splats = self.pair;
	NSError *error = nil;
	NSURL *directory = [NSFileManager.defaultManager.temporaryDirectory URLByAppendingPathComponent:NSUUID.UUID.UUIDString isDirectory:YES];
	XCTAssertTrue([NSFileManager.defaultManager createDirectoryAtURL:directory withIntermediateDirectories:YES attributes:nil error:&error], @"%@", error);
	NSURL *file = [directory URLByAppendingPathComponent:@"splats.ply"];
	XCTAssertTrue([splats writeToURL:file error:&error], @"%@", error);
	NFKValueCoding *coding = [[NFKValueCoding alloc] init];
	NFKGaussianSplats *read = [coding valueOfType:NFKModelValueTypeGaussianSplats fromFileURL:file error:&error];
	XCTAssertNotNil(read, @"%@", error);
	[self assertSplats:splats closeTo:read];
	[NSFileManager.defaultManager removeItemAtURL:directory error:NULL];

	id entry = [coding JSONValueForValue:splats key:NFKOutputGaussianSplats error:&error];
	XCTAssertNotNil(entry, @"%@", error);
	XCTAssertTrue([NSJSONSerialization isValidJSONObject:entry]);
	XCTAssertEqualObjects([coding valueFromJSONValue:entry key:NFKOutputGaussianSplats error:&error], splats, @"%@", error);
}

@end
