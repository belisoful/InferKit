//
//  NFKToolMessagesTests.m
//  InferKitTests
//
//  The assistant turn and the tool turn of a call, in the OpenAI chat shape every chat backend reads.
//

#import <XCTest/XCTest.h>
#import <InferKit/NFKToolMessages.h>
#import <InferKit/NFKInferenceTool.h>
#import <InferKit/NFKInferenceResult.h>
#import <InferKit/NFKInferenceKeys.h>

@interface NFKToolMessagesTests : XCTestCase
@end

@implementation NFKToolMessagesTests

- (void)testAReplyWithCallsBecomesTheChatShapedAssistantTurn
{
	NFKInferenceResult *result = [NFKInferenceResult resultWithOutputs:@{
		NFKOutputToolCalls: @[ @{ @"id": @"call_1", @"name": @"get_weather", @"arguments": @{ @"city": @"Paris" },
								 @"argumentsJSON": @"{\"city\":\"Paris\"}" } ] }];
	NSDictionary *message = [NFKToolMessages assistantMessageForResult:result];
	XCTAssertEqualObjects(message[@"role"], @"assistant");
	XCTAssertEqualObjects(message[@"content"], NSNull.null, @"a turn of calls alone has null content, as the chat API writes it");
	NSDictionary *expected = @{ @"id": @"call_1", @"type": @"function",
								@"function": @{ @"name": @"get_weather", @"arguments": @"{\"city\":\"Paris\"}" } };
	XCTAssertEqualObjects(message[@"tool_calls"], @[ expected ]);
	XCTAssertNil(message[NFKMessageMetadataKey]);
}

- (void)testACallWithoutAnIdentifierIsGivenOneItsAnswerCanName
{
	NFKInferenceResult *result = [NFKInferenceResult resultWithOutputs:@{
		NFKOutputText: @"Checking.",
		NFKOutputToolCalls: @[ @{ @"id": @"", @"name": @"get_weather", @"arguments": @{ @"city": @"Oslo" } } ] }];
	NSDictionary *message = [NFKToolMessages assistantMessageForResult:result];
	XCTAssertEqualObjects(message[@"content"], @"Checking.");
	NSArray<NSDictionary *> *calls = [NFKToolMessages toolCallsOfAssistantMessage:message];
	XCTAssertEqual(calls.count, 1u);
	XCTAssertTrue([calls[0][@"id"] hasPrefix:@"call_"]);
	XCTAssertGreaterThan([calls[0][@"id"] length], 5u);
	XCTAssertEqualObjects(calls[0][@"arguments"], @{ @"city": @"Oslo" });
	XCTAssertEqualObjects(calls[0][@"argumentsJSON"], @"{\"city\":\"Oslo\"}");
}

- (void)testABackendsOwnAssistantTurnIsUsedAsItWrote
{
	NSDictionary *written = @{ @"role": @"assistant", @"content": @"hi", NFKMessageMetadataKey: @{ @"format": @"x", @"turn": @[] } };
	NFKInferenceResult *result = [NFKInferenceResult resultWithOutputs:@{ NFKOutputText: @"hi", NFKOutputAssistantMessage: written }];
	XCTAssertEqualObjects([NFKToolMessages assistantMessageForResult:result], written);
}

- (void)testATextReplyIsAPlainAssistantTurn
{
	NSDictionary *message = [NFKToolMessages assistantMessageForResult:[NFKInferenceResult resultWithOutputs:@{ NFKOutputText: @"Paris." }]];
	NSDictionary *expected = @{ @"role": @"assistant", @"content": @"Paris." };
	XCTAssertEqualObjects(message, expected);
	XCTAssertEqualObjects([NFKToolMessages toolCallsOfAssistantMessage:message], @[]);
}

- (void)testArgumentsThatAreNotAnObjectKeepTheModelsText
{
	NSDictionary *message = @{ @"role": @"assistant",
							   @"tool_calls": @[ @{ @"id": @"c", @"type": @"function", @"function": @{ @"name": @"f", @"arguments": @"{not json" } } ] };
	NSArray<NSDictionary *> *calls = [NFKToolMessages toolCallsOfAssistantMessage:message];
	XCTAssertEqualObjects(calls[0][@"arguments"], @{});
	XCTAssertEqualObjects(calls[0][@"argumentsJSON"], @"{not json");
}

- (void)testAToolTurnCarriesItsNameErrorFlagAndImagesBesideTheChatFields
{
	NSData *png = [NSData dataWithBytes:"\x89PNG" length:4];
	NFKToolResult *failed = [[NFKToolResult alloc] initWithText:@"no such city" outputs:nil imagePNGs:@[ png ] isError:YES];
	NSDictionary *message = [NFKToolMessages messageForToolCallIdentifier:@"call_1" name:@"get_weather" result:failed];
	XCTAssertEqualObjects(message[@"role"], @"tool");
	XCTAssertEqualObjects(message[@"tool_call_id"], @"call_1");
	XCTAssertEqualObjects(message[@"content"], @"no such city");
	NSDictionary *metadata = message[NFKMessageMetadataKey];
	XCTAssertEqualObjects(metadata[@"toolName"], @"get_weather");
	XCTAssertEqualObjects(metadata[@"isError"], @YES);
	XCTAssertEqualObjects(metadata[@"images"], @[ [png base64EncodedStringWithOptions:0] ]);
	XCTAssertTrue([NSJSONSerialization isValidJSONObject:message], @"a conversation stays JSON, so it crosses the native route");

	NSDictionary *stripped = [NFKToolMessages messagesWithoutMetadata:@[ message ]].firstObject;
	NSDictionary *expected = @{ @"role": @"tool", @"tool_call_id": @"call_1", @"content": @"no such city" };
	XCTAssertEqualObjects(stripped, expected);
}

- (void)testABackendOutsideTheCoreRecordsAndReadsBackItsOwnTurn
{
	NSArray *calls = @[ @{ @"id": @"c1", @"name": @"get_weather", @"argumentsJSON": @"{\"city\":\"Oslo\"}" } ];
	NSDictionary *turn = [NFKToolMessages assistantMessageWithText:@"Checking." toolCalls:calls
															record:@{ @"reasoning": @"Need the weather." }
															format:@"my-format" source:@"my-model"];
	XCTAssertEqualObjects(turn[@"content"], @"Checking.");
	XCTAssertEqualObjects([NFKToolMessages toolCallsOfAssistantMessage:turn].firstObject[@"id"], @"c1");
	NSDictionary *record = @{ @"reasoning": @"Need the weather." };
	XCTAssertEqualObjects([NFKToolMessages recordOfMessage:turn format:@"my-format" source:@"my-model"], record);
	XCTAssertEqualObjects([NFKToolMessages recordOfMessage:turn format:@"my-format" source:nil], record, @"any source reads it");
	XCTAssertNil([NFKToolMessages recordOfMessage:turn format:@"my-format" source:@"another-model"]);
	XCTAssertNil([NFKToolMessages recordOfMessage:turn format:@"other-format" source:@"my-model"]);
	XCTAssertNil([NFKToolMessages assistantMessageWithText:@"x" toolCalls:nil record:nil format:nil source:nil][NFKMessageMetadataKey]);
}

- (void)testAToolTurnsPartsAreReadThroughTheClass
{
	NSData *png = [NSData dataWithBytes:"\x89PNG" length:4];
	NFKToolResult *failed = [[NFKToolResult alloc] initWithText:@"no data" outputs:nil imagePNGs:@[ png ] isError:YES];
	NSDictionary *answer = [NFKToolMessages messageForToolCallIdentifier:@"c1" name:@"get_weather" result:failed];
	XCTAssertTrue([NFKToolMessages isErrorToolMessage:answer]);
	XCTAssertEqualObjects([NFKToolMessages textOfToolMessage:answer], @"no data");
	XCTAssertEqualObjects([NFKToolMessages imagePNGsOfToolMessage:answer], @[ png ]);
	XCTAssertEqualObjects([NFKToolMessages toolNameOfMessage:answer inConversation:@[]], @"get_weather");

	NSDictionary *bare = @{ @"role": @"tool", @"tool_call_id": @"c9",
							@"content": @[ @{ @"type": @"text", @"text": @"a" }, @{ @"type": @"text", @"text": @"b" } ] };
	NSDictionary *calling = @{ @"role": @"assistant",
							   @"tool_calls": @[ @{ @"id": @"c9", @"type": @"function", @"function": @{ @"name": @"lookup", @"arguments": @"{}" } } ] };
	NSArray *conversation = @[ calling, bare ];
	XCTAssertEqualObjects([NFKToolMessages toolNameOfMessage:bare inConversation:conversation], @"lookup");
	XCTAssertNil([NFKToolMessages toolNameOfMessage:bare inConversation:@[]]);
	XCTAssertEqualObjects([NFKToolMessages textOfToolMessage:bare], @"a\nb");
	XCTAssertFalse([NFKToolMessages isErrorToolMessage:bare]);
}

@end
