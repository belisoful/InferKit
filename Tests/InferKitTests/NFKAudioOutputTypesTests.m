//
//  NFKAudioOutputTypesTests.m
//  InferKitTests
//

#import <XCTest/XCTest.h>
#import <InferKit/InferKit.h>

@interface NFKAudioOutputTypesTests : XCTestCase
@end

@implementation NFKAudioOutputTypesTests

#pragma mark NFKClassification

- (void)testClassificationCarriesEveryField
{
	NFKClassification *c = [NFKClassification classificationWithLabel:@"Speech" classIndex:0 confidence:0.88];
	XCTAssertEqualObjects(c.label, @"Speech");
	XCTAssertEqual(c.classIndex, 0);
	XCTAssertEqual(c.confidence, 0.88);
	XCTAssertNil([NFKClassification classificationWithLabel:nil classIndex:3 confidence:0.1].label);
}

- (void)testClassificationEqualityAndCopy
{
	NFKClassification *a = [NFKClassification classificationWithLabel:@"Music" classIndex:137 confidence:0.6];
	NFKClassification *b = [NFKClassification classificationWithLabel:@"Music" classIndex:137 confidence:0.6];
	NFKClassification *c = [NFKClassification classificationWithLabel:@"Music" classIndex:137 confidence:0.5];
	XCTAssertEqualObjects(a, b);
	XCTAssertNotEqualObjects(a, c);
	XCTAssertEqual([a copy], a);
}

- (void)testResultClassificationsAccessor
{
	NFKClassification *c = [NFKClassification classificationWithLabel:@"Dog" classIndex:74 confidence:0.9];
	NFKInferenceResult *result = [NFKInferenceResult resultWithOutputs:@{ NFKOutputClassifications: @[ c ] }];
	XCTAssertEqualObjects(result.classifications, @[ c ]);
	XCTAssertNil([NFKInferenceResult resultWithOutputs:@{ NFKOutputText: @"x" }].classifications);
}

#pragma mark NFKAudioSegment

- (void)testSegmentCarriesEveryField
{
	NFKAudioSegment *s = [NFKAudioSegment segmentWithStartSeconds:1.2 endSeconds:3.4 label:nil confidence:0.95];
	XCTAssertEqual(s.startSeconds, 1.2);
	XCTAssertEqual(s.endSeconds, 3.4);
	XCTAssertNil(s.label, @"a plain speech span has no label");
	XCTAssertEqual(s.confidence, 0.95);
}

- (void)testSegmentEqualityAndCopy
{
	NFKAudioSegment *a = [NFKAudioSegment segmentWithStartSeconds:0 endSeconds:2 label:@"speaker-1" confidence:0.8];
	NFKAudioSegment *b = [NFKAudioSegment segmentWithStartSeconds:0 endSeconds:2 label:@"speaker-1" confidence:0.8];
	NFKAudioSegment *c = [NFKAudioSegment segmentWithStartSeconds:0 endSeconds:2.5 label:@"speaker-1" confidence:0.8];
	XCTAssertEqualObjects(a, b);
	XCTAssertNotEqualObjects(a, c);
	XCTAssertEqual([a copy], a);
}

- (void)testResultSegmentsAccessor
{
	NFKAudioSegment *s = [NFKAudioSegment segmentWithStartSeconds:0 endSeconds:1 label:nil confidence:0.7];
	NFKInferenceResult *result = [NFKInferenceResult resultWithOutputs:@{ NFKOutputSegments: @[ s ] }];
	XCTAssertEqualObjects(result.segments, @[ s ]);
	XCTAssertNil([NFKInferenceResult resultWithOutputs:@{ NFKOutputText: @"x" }].segments);
}

#pragma mark Archiving

- (void)testAClassificationSurvivesASecureRoundTrip
{
	NFKClassification *classification = [NFKClassification classificationWithLabel:@"speech"
																		classIndex:3
																		confidence:0.75];
	NSError *error = nil;
	NSData *data = [NSKeyedArchiver archivedDataWithRootObject:classification requiringSecureCoding:YES error:&error];
	XCTAssertNotNil(data, @"%@", error);

	NFKClassification *read = [NSKeyedUnarchiver unarchivedObjectOfClass:NFKClassification.class
																fromData:data
																   error:&error];
	XCTAssertNotNil(read, @"%@", error);
	XCTAssertEqualObjects(read.label, @"speech");
	XCTAssertEqual(read.classIndex, 3);
	XCTAssertEqual(read.confidence, 0.75);
	XCTAssertEqualObjects(read, classification);
}

- (void)testAnAudioSegmentSurvivesASecureRoundTrip
{
	NFKAudioSegment *segment = [NFKAudioSegment segmentWithStartSeconds:15.25
															 endSeconds:45.5
																  label:@"verse"
															 confidence:0.9];
	NSError *error = nil;
	NSData *data = [NSKeyedArchiver archivedDataWithRootObject:segment requiringSecureCoding:YES error:&error];
	XCTAssertNotNil(data, @"%@", error);

	NFKAudioSegment *read = [NSKeyedUnarchiver unarchivedObjectOfClass:NFKAudioSegment.class fromData:data error:&error];
	XCTAssertNotNil(read, @"%@", error);
	XCTAssertEqual(read.startSeconds, 15.25);
	XCTAssertEqual(read.endSeconds, 45.5);
	XCTAssertEqualObjects(read.label, @"verse");
	XCTAssertEqual(read.confidence, 0.9);
	XCTAssertEqualObjects(read, segment);
}

- (void)testATranscriptsSegmentsArchiveAsOneArray
{
	NSArray<NFKAudioSegment *> *segments = @[
		[NFKAudioSegment segmentWithStartSeconds:0 endSeconds:1.5 label:nil confidence:0.7],
		[NFKAudioSegment segmentWithStartSeconds:1.5 endSeconds:3 label:@"speech" confidence:0.8],
	];
	NSData *data = [NSKeyedArchiver archivedDataWithRootObject:segments requiringSecureCoding:YES error:NULL];
	NSSet *classes = [NSSet setWithObjects:NSArray.class, NFKAudioSegment.class, nil];
	XCTAssertEqualObjects([NSKeyedUnarchiver unarchivedObjectOfClasses:classes fromData:data error:NULL], segments);
}
@end
