//
//  NFKDecisionQuestionTests.m
//  InferKitTests
//

#import <XCTest/XCTest.h>
#import <InferKit/NFKDecisionQuestion.h>

@interface NFKDecisionQuestionTests : XCTestCase
@end

@implementation NFKDecisionQuestionTests

- (void)testAChoiceSendsItsOptionsAsAMapOfNameToDescriptionOrNull
{
	NFKDecisionQuestion *question = [NFKDecisionQuestion choiceQuestionWithInstructions:@"Which team?"
																				options:@[ @"billing", @"technical" ]
																		   descriptions:@{ @"billing": @"money" }];
	XCTAssertEqual(question.type, NFKDecisionTypeChoice);
	XCTAssertEqualObjects(question.options, (@[ @"billing", @"technical" ]));
	NSDictionary *wire = question.dictionaryRepresentation;
	XCTAssertEqualObjects(wire[@"type"], @"choice");
	XCTAssertEqualObjects(wire[@"instructions"], @"Which team?");
	XCTAssertEqualObjects(wire[@"criteria"], (@{ @"billing": @"money", @"technical": NSNull.null }));
	XCTAssertTrue([NSJSONSerialization isValidJSONObject:wire]);
}

- (void)testAScoreSendsItsLevelsInOrder
{
	NFKDecisionQuestion *question = [NFKDecisionQuestion scoreQuestionWithInstructions:@"How severe?"
																				levels:@[ @"low", @"medium", @"high" ]];
	XCTAssertEqual(question.type, NFKDecisionTypeScore);
	NSDictionary *wire = question.dictionaryRepresentation;
	XCTAssertEqualObjects(wire[@"type"], @"score");
	XCTAssertEqualObjects(wire[@"criteria"], (@[ @"low", @"medium", @"high" ]));
	XCTAssertEqual(question.descriptions.count, 0);
}

- (void)testANoulSendsItsMeaningsOnlyWhenGiven
{
	NFKDecisionQuestion *bare = [NFKDecisionQuestion noulQuestionWithInstructions:@"Is it urgent?"];
	XCTAssertEqual(bare.type, NFKDecisionTypeNoul);
	XCTAssertNil(bare.dictionaryRepresentation[@"criteria"]);
	XCTAssertEqual(bare.options.count, 0);

	NFKDecisionQuestion *explained = [NFKDecisionQuestion noulQuestionWithInstructions:@"Is it urgent?"
																		   trueMeaning:@"needs an answer today"
																		  falseMeaning:nil];
	XCTAssertEqualObjects(explained.dictionaryRepresentation[@"criteria"], (@{ @"true": @"needs an answer today" }));
}

- (void)testTheWireShapeReadsBackIntoAQuestion
{
	NFKDecisionQuestion *choice = [NFKDecisionQuestion questionWithDictionary:@{ @"type": @"choice", @"instructions": @"Which?",
		@"criteria": @{ @"sales": NSNull.null, @"billing": @"money" } }];
	XCTAssertEqual(choice.type, NFKDecisionTypeChoice);
	XCTAssertEqualObjects(choice.options, (@[ @"billing", @"sales" ]), @"a dictionary carries no order, so the names are sorted");
	XCTAssertEqualObjects(choice.descriptions, (@{ @"billing": @"money" }));

	NFKDecisionQuestion *score = [NFKDecisionQuestion questionWithDictionary:@{ @"type": @"score", @"instructions": @"How?",
		@"criteria": @[ @"low", @"high" ] }];
	XCTAssertEqualObjects(score.options, (@[ @"low", @"high" ]));

	NFKDecisionQuestion *noul = [NFKDecisionQuestion questionWithDictionary:@{ @"type": @"noul", @"instructions": @"Is it?",
		@"criteria": @{ @"true": @"yes it is" } }];
	XCTAssertEqualObjects(noul.descriptions, (@{ @"true": @"yes it is" }));
	XCTAssertEqualObjects(noul.dictionaryRepresentation, (@{ @"type": @"noul", @"instructions": @"Is it?", @"criteria": @{ @"true": @"yes it is" } }));

	NSDictionary *ranking = @{ @"type": @"ranking", @"instructions": @"?" };
	NSDictionary *wordless = @{ @"type": @"noul" };
	XCTAssertNil([NFKDecisionQuestion questionWithDictionary:ranking]);
	XCTAssertNil([NFKDecisionQuestion questionWithDictionary:wordless], @"instructions are required");
}

- (void)testQuestionsCompareByValueAndCopyToThemselves
{
	NFKDecisionQuestion *a = [NFKDecisionQuestion scoreQuestionWithInstructions:@"?" levels:@[ @"1", @"2" ]];
	NFKDecisionQuestion *b = [NFKDecisionQuestion scoreQuestionWithInstructions:@"?" levels:@[ @"1", @"2" ]];
	NFKDecisionQuestion *c = [NFKDecisionQuestion choiceQuestionWithInstructions:@"?" options:@[ @"1", @"2" ]];
	XCTAssertEqualObjects(a, b);
	XCTAssertEqual(a.hash, b.hash);
	XCTAssertNotEqualObjects(a, c);
	XCTAssertTrue([a copy] == a);
	XCTAssertEqualObjects([NFKDecisionQuestion nameForType:NFKDecisionTypeNoul], @"noul");
}

@end
