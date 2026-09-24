//
//  NFKQuadrilateralTests.m
//  NFKTests
//

#import <XCTest/XCTest.h>
#import <InferKit/InferKit.h>

@interface NFKQuadrilateralTests : XCTestCase
@end

@implementation NFKQuadrilateralTests

- (NFKQuadrilateral *)slanted
{
	return [NFKQuadrilateral quadrilateralWithTopLeft:CGPointMake(0.1, 0.2)
											 topRight:CGPointMake(0.8, 0.1)
										   bottomLeft:CGPointMake(0.2, 0.9)
										  bottomRight:CGPointMake(0.9, 0.8)];
}

- (void)testACornerIsNamedByItsPlaceInTheShape
{
	NFKQuadrilateral *quad = [self slanted];
	XCTAssertTrue(CGPointEqualToPoint(quad.topLeft, CGPointMake(0.1, 0.2)));
	XCTAssertTrue(CGPointEqualToPoint(quad.bottomRight, CGPointMake(0.9, 0.8)));
}

- (void)testTheBoundingBoxHoldsEveryCorner
{
	CGRect box = [self slanted].boundingBox;
	XCTAssertEqualWithAccuracy(CGRectGetMinX(box), 0.1, 1e-9);
	XCTAssertEqualWithAccuracy(CGRectGetMinY(box), 0.1, 1e-9);
	XCTAssertEqualWithAccuracy(CGRectGetMaxX(box), 0.9, 1e-9);
	XCTAssertEqualWithAccuracy(CGRectGetMaxY(box), 0.9, 1e-9);
}

- (void)testEqualityAndCopy
{
	NFKQuadrilateral *a = [self slanted];
	NFKQuadrilateral *b = [self slanted];
	NFKQuadrilateral *square = [NFKQuadrilateral quadrilateralWithTopLeft:CGPointZero
																 topRight:CGPointMake(1, 0)
															   bottomLeft:CGPointMake(0, 1)
															  bottomRight:CGPointMake(1, 1)];
	XCTAssertEqualObjects(a, b);
	XCTAssertEqual(a.hash, b.hash);
	XCTAssertNotEqualObjects(a, square);
	XCTAssertEqual([a copy], a, @"immutable value copies to itself");
}

- (void)testAQuadrilateralSurvivesASecureRoundTrip
{
	NFKQuadrilateral *quad = [self slanted];
	NSError *error = nil;
	NSData *data = [NSKeyedArchiver archivedDataWithRootObject:quad requiringSecureCoding:YES error:&error];
	XCTAssertNotNil(data, @"%@", error);
	XCTAssertEqualObjects([NSKeyedUnarchiver unarchivedObjectOfClass:NFKQuadrilateral.class
														   fromData:data
															  error:&error], quad, @"%@", error);
}

#pragma mark On a detection

- (void)testADetectionWithCornersTakesItsBoxFromThem
{
	NFKDetection *detection = [NFKDetection detectionWithLabel:@"page"
													classIndex:0
													confidence:0.9
												 quadrilateral:[self slanted]];
	XCTAssertEqualObjects(detection.quadrilateral, [self slanted]);
	XCTAssertTrue(CGRectEqualToRect(detection.boundingBox, [self slanted].boundingBox));
}

- (void)testADetectionWithoutCornersReportsNone
{
	NFKDetection *detection = [NFKDetection detectionWithLabel:@"dog"
													classIndex:16
													confidence:0.8
												   boundingBox:CGRectMake(0, 0, 0.5, 0.5)];
	XCTAssertNil(detection.quadrilateral);
}

- (void)testTheCornersSurviveADetectionsRoundTrip
{
	NFKDetection *detection = [NFKDetection detectionWithLabel:@"page"
													classIndex:0
													confidence:0.9
												 quadrilateral:[self slanted]];
	NSData *data = [NSKeyedArchiver archivedDataWithRootObject:detection requiringSecureCoding:YES error:NULL];
	NFKDetection *read = [NSKeyedUnarchiver unarchivedObjectOfClass:NFKDetection.class fromData:data error:NULL];
	XCTAssertEqualObjects(read.quadrilateral, [self slanted]);
	XCTAssertEqualObjects(read, detection);
}

@end
