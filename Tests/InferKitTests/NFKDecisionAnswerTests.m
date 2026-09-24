//
//  NFKDecisionAnswerTests.m
//  InferKitTests
//

#import <XCTest/XCTest.h>
#import <InferKit/NFKDecisionAnswer.h>

@interface NFKDecisionAnswerTests : XCTestCase
@end

@implementation NFKDecisionAnswerTests

- (void)testEachTypeReadsItsOwnFieldsAndLeavesTheOthersEmpty
{
	NFKDecisionAnswer *choice = [NFKDecisionAnswer answerWithDictionary:@{ @"type": @"choice", @"choice": @"sales",
		@"probabilities": @{ @"sales": @0.9, @"billing": @0.1 }, @"confidence": @0.8, @"score": @4 }];
	XCTAssertEqual(choice.type, NFKDecisionTypeChoice);
	XCTAssertEqualObjects(choice.choice, @"sales");
	XCTAssertEqualWithAccuracy(choice.probabilities[@"sales"].doubleValue, 0.9, 1e-9);
	XCTAssertEqualWithAccuracy(choice.confidence, 0.8, 1e-9);
	XCTAssertEqual(choice.score, 0, @"a score field on a choice is not the answer");
	XCTAssertEqual(choice.probability, 0);

	NFKDecisionAnswer *score = [NFKDecisionAnswer answerWithDictionary:@{ @"type": @"score", @"score": @2.25,
		@"legend": @{ @"0": @"low", @"1": @"high" }, @"probabilities": @{ @"0": @0.25, @"1": @0.75 }, @"confidence": @0.5 }];
	XCTAssertEqual(score.type, NFKDecisionTypeScore);
	XCTAssertEqualWithAccuracy(score.score, 2.25, 1e-9);
	XCTAssertEqualObjects(score.legend[@"1"], @"high");
	XCTAssertNil(score.choice);

	NFKDecisionAnswer *noul = [NFKDecisionAnswer answerWithDictionary:@{ @"type": @"noul", @"noul": @0.33 }];
	XCTAssertEqual(noul.type, NFKDecisionTypeNoul);
	XCTAssertEqualWithAccuracy(noul.probability, 0.33, 1e-9);
	XCTAssertEqual(noul.confidence, 0, @"the service reports none for a noul");
	XCTAssertNil(noul.probabilities);
	XCTAssertNil(noul.legend);
	XCTAssertEqualObjects(noul.raw[@"noul"], @0.33);
}

- (void)testAMissingOrUnknownTypeIsNoAnswer
{
	XCTAssertNil([NFKDecisionAnswer answerWithDictionary:@{ @"choice": @"a" }]);
	XCTAssertNil([NFKDecisionAnswer answerWithDictionary:@{ @"type": @"ranking" }]);
	XCTAssertNil([NFKDecisionAnswer answerWithDictionary:(NSDictionary *)(id)@[ @"noul" ]]);
}

- (void)testAnAnswerArchivesAndComparesByValue
{
	NFKDecisionAnswer *answer = [NFKDecisionAnswer answerWithDictionary:@{ @"type": @"score", @"score": @1.5,
		@"legend": @{ @"0": @"low", @"1": @"mid", @"2": @"high" }, @"probabilities": @{ @"1": @0.5, @"2": @0.5 }, @"confidence": @0.4 }];
	NSError *error = nil;
	NSData *archive = [NSKeyedArchiver archivedDataWithRootObject:@[ answer ] requiringSecureCoding:YES error:&error];
	XCTAssertNotNil(archive, @"%@", error);
	NSSet *classes = [NSSet setWithArray:@[ NSArray.class, NFKDecisionAnswer.class ]];
	NSArray<NFKDecisionAnswer *> *back = [NSKeyedUnarchiver unarchivedObjectOfClasses:classes fromData:archive error:&error];
	XCTAssertNotNil(back, @"%@", error);
	XCTAssertEqualObjects(back.firstObject, answer);
	XCTAssertEqual(back.firstObject.hash, answer.hash);
	XCTAssertEqualObjects(back.firstObject.raw, answer.raw);
	XCTAssertTrue([answer copy] == answer);

	NFKDecisionAnswer *other = [NFKDecisionAnswer answerWithDictionary:@{ @"type": @"score", @"score": @1.5, @"confidence": @0.41 }];
	XCTAssertNotEqualObjects(answer, other);
}

@end
