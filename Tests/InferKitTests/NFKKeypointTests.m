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

#pragma mark Archiving

- (void)testAKeypointSurvivesASecureRoundTrip
{
	NFKKeypoint *keypoint = [NFKKeypoint keypointWithName:@"left_wrist"
													index:9
												 position:CGPointMake(0.25, 0.75)
											   confidence:0.875];
	NSError *error = nil;
	NSData *data = [NSKeyedArchiver archivedDataWithRootObject:keypoint requiringSecureCoding:YES error:&error];
	XCTAssertNotNil(data, @"%@", error);

	NFKKeypoint *read = [NSKeyedUnarchiver unarchivedObjectOfClass:NFKKeypoint.class fromData:data error:&error];
	XCTAssertNotNil(read, @"%@", error);
	XCTAssertEqualObjects(read.name, @"left_wrist");
	XCTAssertEqual(read.index, 9);
	XCTAssertTrue(CGPointEqualToPoint(read.position, CGPointMake(0.25, 0.75)));
	XCTAssertEqual(read.confidence, 0.875);
	XCTAssertEqualObjects(read, keypoint);
}

- (void)testAPoseArchivesAsOneArray
{
	NSArray<NFKKeypoint *> *pose = @[
		[NFKKeypoint keypointWithName:nil index:0 position:CGPointMake(0.5, 0.1) confidence:0.6],
		[NFKKeypoint keypointWithName:@"nose" index:1 position:CGPointMake(0.5, 0.2) confidence:0.7],
	];
	NSData *data = [NSKeyedArchiver archivedDataWithRootObject:pose requiringSecureCoding:YES error:NULL];
	NSSet *classes = [NSSet setWithObjects:NSArray.class, NFKKeypoint.class, nil];
	XCTAssertEqualObjects([NSKeyedUnarchiver unarchivedObjectOfClasses:classes fromData:data error:NULL], pose);
}
@end
