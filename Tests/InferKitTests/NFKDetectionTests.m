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

#pragma mark Archiving

- (void)testADetectionSurvivesASecureRoundTrip
{
	NFKDetection *detection = [NFKDetection detectionWithLabel:@"gannet"
													classIndex:16
													confidence:0.625
												   boundingBox:CGRectMake(0.1, 0.2, 0.3, 0.4)];
	NSError *error = nil;
	NSData *data = [NSKeyedArchiver archivedDataWithRootObject:detection requiringSecureCoding:YES error:&error];
	XCTAssertNotNil(data, @"%@", error);

	NFKDetection *read = [NSKeyedUnarchiver unarchivedObjectOfClass:NFKDetection.class fromData:data error:&error];
	XCTAssertNotNil(read, @"%@", error);
	XCTAssertEqualObjects(read.label, @"gannet");
	XCTAssertEqual(read.classIndex, 16);
	XCTAssertEqual(read.confidence, 0.625);
	XCTAssertTrue(CGRectEqualToRect(read.boundingBox, CGRectMake(0.1, 0.2, 0.3, 0.4)));
	XCTAssertEqualObjects(read, detection);
}

- (void)testADetectionWithNoLabelSurvivesASecureRoundTrip
{
	NFKDetection *detection = [NFKDetection detectionWithLabel:nil classIndex:7 confidence:0.5
												   boundingBox:CGRectZero];
	NSData *data = [NSKeyedArchiver archivedDataWithRootObject:detection requiringSecureCoding:YES error:NULL];
	NFKDetection *read = [NSKeyedUnarchiver unarchivedObjectOfClass:NFKDetection.class fromData:data error:NULL];
	XCTAssertNil(read.label);
	XCTAssertEqualObjects(read, detection);
}

- (void)testAFramesDetectionsArchiveAsOneArray
{
	// A caller recording a result per frame archives the array the backend returned, which is what
	// the conformance is for.
	NSArray<NFKDetection *> *detections = @[
		[NFKDetection detectionWithLabel:@"dog" classIndex:16 confidence:0.9 boundingBox:CGRectMake(0, 0, 0.5, 0.5)],
		[NFKDetection detectionWithLabel:@"cat" classIndex:17 confidence:0.4 boundingBox:CGRectMake(0.5, 0.5, 0.5, 0.5)],
	];
	NSError *error = nil;
	NSData *data = [NSKeyedArchiver archivedDataWithRootObject:detections requiringSecureCoding:YES error:&error];
	XCTAssertNotNil(data, @"%@", error);

	NSSet *classes = [NSSet setWithObjects:NSArray.class, NFKDetection.class, nil];
	NSArray<NFKDetection *> *read = [NSKeyedUnarchiver unarchivedObjectOfClasses:classes fromData:data error:&error];
	XCTAssertNotNil(read, @"%@", error);
	XCTAssertEqualObjects(read, detections);
}
@end
