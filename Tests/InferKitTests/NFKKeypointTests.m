//
//  NFKKeypointTests.m
//  InferKitTests
//

#import <XCTest/XCTest.h>
#import <InferKit/InferKit.h>

@interface NFKKeypointTests : XCTestCase
@end

@implementation NFKKeypointTests

- (void)testFactoryCarriesEveryField
{
	NFKKeypoint *keypoint = [NFKKeypoint keypointWithName:@"nose" index:0
												 position:CGPointMake(0.5, 0.3) confidence:0.92];
	XCTAssertEqualObjects(keypoint.name, @"nose");
	XCTAssertEqual(keypoint.index, 0);
	XCTAssertTrue(CGPointEqualToPoint(keypoint.position, CGPointMake(0.5, 0.3)));
	XCTAssertEqual(keypoint.confidence, 0.92);
}

- (void)testANameIsOptional
{
	NFKKeypoint *keypoint = [NFKKeypoint keypointWithName:nil index:5 position:CGPointZero confidence:0.1];
	XCTAssertNil(keypoint.name, @"a backend may return a joint index only");
	XCTAssertEqual(keypoint.index, 5);
}

- (void)testEqualityAndCopy
{
	CGPoint p = CGPointMake(0.2, 0.4);
	NFKKeypoint *a = [NFKKeypoint keypointWithName:@"left_eye" index:1 position:p confidence:0.7];
	NFKKeypoint *b = [NFKKeypoint keypointWithName:@"left_eye" index:1 position:p confidence:0.7];
	NFKKeypoint *c = [NFKKeypoint keypointWithName:@"left_eye" index:1 position:CGPointMake(0.3, 0.4) confidence:0.7];
	XCTAssertEqualObjects(a, b);
	XCTAssertNotEqualObjects(a, c, @"a different position is a different keypoint");
	XCTAssertEqual(a.hash, b.hash);
	XCTAssertEqual([a copy], a, @"immutable value copies to itself");
}

- (void)testResultPoseAccessor
{
	NFKKeypoint *keypoint = [NFKKeypoint keypointWithName:@"nose" index:0 position:CGPointMake(0.5, 0.5) confidence:0.9];
	NFKInferenceResult *result = [NFKInferenceResult resultWithOutputs:@{ NFKOutputPose: @[ keypoint ] }];
	XCTAssertEqualObjects(result.pose, @[ keypoint ]);

	NFKInferenceResult *empty = [NFKInferenceResult resultWithOutputs:@{ NFKOutputImage: @"opaque" }];
	XCTAssertNil(empty.pose);
}

@end
