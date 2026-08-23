//
//  NFKDetectionTests.m
//  InferKitTests
//

#import <XCTest/XCTest.h>
#import <InferKit/InferKit.h>

@interface NFKDetectionTests : XCTestCase
@end

@implementation NFKDetectionTests

- (void)testFactoryCarriesEveryField
{
	NFKDetection *detection = [NFKDetection detectionWithLabel:@"person" classIndex:0 confidence:0.87
												  boundingBox:CGRectMake(0.25, 0.5, 0.1, 0.2)];
	XCTAssertEqualObjects(detection.label, @"person");
	XCTAssertEqual(detection.classIndex, 0);
	XCTAssertEqual(detection.confidence, 0.87);
	XCTAssertTrue(CGRectEqualToRect(detection.boundingBox, CGRectMake(0.25, 0.5, 0.1, 0.2)));
}

- (void)testALabelIsOptional
{
	NFKDetection *detection = [NFKDetection detectionWithLabel:nil classIndex:7 confidence:0.5
												  boundingBox:CGRectZero];
	XCTAssertNil(detection.label, @"a backend may return a class index only");
	XCTAssertEqual(detection.classIndex, 7);
}

- (void)testEqualityAndCopy
{
	CGRect box = CGRectMake(0.1, 0.1, 0.2, 0.2);
	NFKDetection *a = [NFKDetection detectionWithLabel:@"dog" classIndex:16 confidence:0.6 boundingBox:box];
	NFKDetection *b = [NFKDetection detectionWithLabel:@"dog" classIndex:16 confidence:0.6 boundingBox:box];
	NFKDetection *c = [NFKDetection detectionWithLabel:@"dog" classIndex:16 confidence:0.6
										  boundingBox:CGRectMake(0, 0, 0.2, 0.2)];
	XCTAssertEqualObjects(a, b);
	XCTAssertNotEqualObjects(a, c, @"a different box is a different detection");
	XCTAssertEqual(a.hash, b.hash);
	XCTAssertEqual([a copy], a, @"immutable value copies to itself");
}

@end
