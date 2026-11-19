//
//  NFKToolConversationTests.m
//  InferKitTests
//
//  A tool conversation in the chat shape through each chat backend's stubbed transport: the wire
//  form of the calls and their answers, the turn each backend records, and that record replayed to
//  the endpoint that wrote it. A tool loop then runs end to end over the Anthropic wire.
//

#import <XCTest/XCTest.h>
#import <InferKit/InferKit.h>

#define NFK_STAGED_TRANSPORT \
- (NSData *)sendRequest:(NSURLRequest *)request response:(NSHTTPURLResponse **)outResponse error:(NSError **)outError \
{ \
	if (self.requests == nil) { \
		self.requests = [NSMutableArray array]; \
	} \
	[self.requests addObject:request]; \
	if (outResponse != NULL) { \
		*outResponse = [[NSHTTPURLResponse alloc] initWithURL:request.URL statusCode:200 HTTPVersion:@"HTTP/1.1" headerFields:nil]; \
	} \
	NSString *body = self.staged.firstObject ?: @"{}"; \
	if (self.staged.count > 1) { \
		[self.staged removeObjectAtIndex:0]; \
	} \
	return [body dataUsingEncoding:NSUTF8StringEncoding]; \
} \
- (NSDictionary *)bodyAt:(NSUInteger)index \
{ \
	return [NSJSONSerialization JSONObjectWithData:self.requests[index].HTTPBody options:0 error:NULL]; \
}

@interface NFKConversationChatStub : NFKRemoteBackend
@property (nonatomic, strong) NSMutableArray<NSURLRequest *> *requests;
@property (nonatomic, strong) NSMutableArray<NSString *> *staged;
@end
@implementation NFKConversationChatStub
NFK_STAGED_TRANSPORT
@end

@interface NFKConversationAnthropicStub : NFKAnthropicBackend
@property (nonatomic, strong) NSMutableArray<NSURLRequest *> *requests;
@property (nonatomic, strong) NSMutableArray<NSString *> *staged;
@end
@implementation NFKConversationAnthropicStub
NFK_STAGED_TRANSPORT
@end

@interface NFKConversationResponsesStub : NFKRemoteResponsesBackend
@property (nonatomic, strong) NSMutableArray<NSURLRequest *> *requests;
@property (nonatomic, strong) NSMutableArray<NSString *> *staged;
@end
@implementation NFKConversationResponsesStub
NFK_STAGED_TRANSPORT
@end

@interface NFKConversationInteractionsStub : NFKGeminiInteractionsBackend
@property (nonatomic, strong) NSMutableArray<NSURLRequest *> *requests;
@property (nonatomic, strong) NSMutableArray<NSString *> *staged;
@end
@implementation NFKConversationInteractionsStub
NFK_STAGED_TRANSPORT
@end

/*! An Anthropic backend that answers each call from a staged list of {status, body}: a blocking call
	from responses, and a stream from streams, where a 200's body is its SSE lines joined by newlines. */
@interface NFKPrefixCheckStub : NFKAnthropicBackend
@property (nonatomic, strong) NSMutableArray<NSURLRequest *> *requests;
@property (nonatomic, strong) NSMutableArray<NSArray *> *responses;
@property (nonatomic, strong) NSMutableArray<NSArray *> *streams;
@end

@implementation NFKPrefixCheckStub

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_requests = [NSMutableArray array];
		_responses = [NSMutableArray array];
		_streams = [NSMutableArray array];
	}
	return self;
}

- (NSArray *)nextOf:(NSMutableArray<NSArray *> *)staged
{
	NSArray *next = staged.firstObject ?: @[ @200, @"{}" ];
	if (staged.count > 0) {
		[staged removeObjectAtIndex:0];
	}
	return next;
}

- (NSData *)sendRequest:(NSURLRequest *)request response:(NSHTTPURLResponse **)outResponse error:(NSError **)outError
{
	[self.requests addObject:request];
	NSArray *next = [self nextOf:self.responses];
	if (outResponse != NULL) {
		*outResponse = [[NSHTTPURLResponse alloc] initWithURL:request.URL statusCode:[next[0] integerValue] HTTPVersion:@"HTTP/1.1" headerFields:nil];
	}
	return [next[1] dataUsingEncoding:NSUTF8StringEncoding];
}

- (void (^)(void))streamRequest:(NSURLRequest *)request
					lineHandler:(void (^)(NSString *))lineHandler
			  completionHandler:(void (^)(NSHTTPURLResponse *, NSData *, NSError *))completionHandler
{
	[self.requests addObject:request];
	NSArray *next = [self nextOf:self.streams];
	NSInteger status = [next[0] integerValue];
	NSHTTPURLResponse *response = [[NSHTTPURLResponse alloc] initWithURL:request.URL statusCode:status HTTPVersion:@"HTTP/1.1" headerFields:nil];
	if (status != 200) {
		completionHandler(response, [next[1] dataUsingEncoding:NSUTF8StringEncoding], nil);
		return ^{};
	}
	for (NSString *line in [next[1] componentsSeparatedByString:@"\n"]) {
		lineHandler(line);
	}
	completionHandler(response, nil, nil);
	return ^{};
}

- (NSDictionary *)bodyAt:(NSUInteger)index
{
	return [NSJSONSerialization JSONObjectWithData:self.requests[index].HTTPBody options:0 error:NULL];
}

@end

@interface NFKToolConversationTests : XCTestCase
@property (nonatomic, copy) NSData *png;
@end

@implementation NFKToolConversationTests

- (void)setUp
{
	[super setUp];
	self.png = [NSData dataWithBytes:"\x89PNG\r\n" length:6];
}

// A user question, an assistant turn calling two tools, a failed answer that carries an image, a good
// answer, and a user follow-up.
- (NSArray<NSDictionary *> *)conversation
{
	NFKInferenceResult *calling = [NFKInferenceResult resultWithOutputs:@{
		NFKOutputText: @"Let me check.",
		NFKOutputToolCalls: @[ @{ @"id": @"c1", @"name": @"get_weather", @"arguments": @{ @"city": @"Paris" } },
							   @{ @"id": @"c2", @"name": @"get_map", @"arguments": @{ @"city": @"Paris" } } ] }];
	NFKToolResult *failed = [[NFKToolResult alloc] initWithText:@"the station is down" outputs:nil imagePNGs:nil isError:YES];
	NFKToolResult *mapped = [NFKToolResult resultWithText:@"a map of Paris" outputs:nil imagePNGs:@[ self.png ]];
	return @[ @{ @"role": @"user", @"content": @"Weather and a map of Paris?" },
			  [NFKToolMessages assistantMessageForResult:calling],
			  [NFKToolMessages messageForToolCallIdentifier:@"c1" name:@"get_weather" result:failed],
			  [NFKToolMessages messageForToolCallIdentifier:@"c2" name:@"get_map" result:mapped],
			  @{ @"role": @"user", @"content": @"Thanks." } ];
}

- (NFKInferenceRequest *)requestWith:(NSArray<NSDictionary *> *)messages
{
	return [NFKInferenceRequest requestWithInputs:@{ NFKInputMessages: messages } parameters:nil outputModality:NFKModalityText];
}

- (NSString *)base64PNG
{
	return [self.png base64EncodedStringWithOptions:0];
}

#pragma mark OpenAI chat

- (NFKConversationChatStub *)chatStubAt:(NSString *)endpoint
{
	NFKConversationChatStub *stub = [[NFKConversationChatStub alloc] init];
	stub.endpointURL = [NSURL URLWithString:endpoint];
	stub.modelName = @"m";
	stub.staged = [NSMutableArray arrayWithObject:@"{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"ok\"}}]}"];
	return stub;
}

- (void)testTheChatShapeGoesOutWithoutMetadataAndToolImagesFollowInAUserTurn
{
	NFKConversationChatStub *chat = [self chatStubAt:@"https://api.deepseek.com/chat/completions"];
	NSError *error = nil;
	XCTAssertNotNil([chat runInferenceForRequest:[self requestWith:self.conversation] error:&error], @"%@", error);
	NSArray<NSDictionary *> *wire = [chat bodyAt:0][@"messages"];
	XCTAssertEqual(wire.count, 6u);
	XCTAssertNil(wire[1][NFKMessageMetadataKey]);
	XCTAssertEqualObjects(wire[1][@"tool_calls"][1][@"function"][@"name"], @"get_map");
	NSDictionary *failed = @{ @"role": @"tool", @"tool_call_id": @"c1", @"content": @"Error: the station is down" };
	XCTAssertEqualObjects(wire[2], failed, @"the chat API has no error flag, so the text says it");
	NSDictionary *mapped = @{ @"role": @"tool", @"tool_call_id": @"c2", @"content": @"a map of Paris" };
	XCTAssertEqualObjects(wire[3], mapped);
	XCTAssertEqualObjects(wire[4][@"role"], @"user");
	NSString *dataURL = [@"data:image/png;base64," stringByAppendingString:self.base64PNG];
	XCTAssertEqualObjects(wire[4][@"content"][1][@"image_url"][@"url"], dataURL, @"a tool answer carries text only, so its image follows");
	XCTAssertEqualObjects(wire[5][@"content"], @"Thanks.");
}

- (void)testTheChatReplyRecordsItsTurnAndOnlyItsOwnEndpointReplaysIt
{
	NFKConversationChatStub *deepSeek = [self chatStubAt:@"https://api.deepseek.com/chat/completions"];
	deepSeek.staged = [NSMutableArray arrayWithObjects:
		@"{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":null,\"reasoning_content\":\"Need the weather.\","
		"\"tool_calls\":[{\"id\":\"call_9\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"{\\\"city\\\":\\\"Oslo\\\"}\"}}]}}]}",
		@"{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"Cold.\"}}]}", nil];
	NFKInferenceResult *reply = [deepSeek runInferenceForRequest:[self requestWith:@[ @{ @"role": @"user", @"content": @"Oslo?" } ]] error:NULL];
	NSDictionary *assistant = [NFKToolMessages assistantMessageForResult:reply];
	XCTAssertEqualObjects(assistant[@"tool_calls"][0][@"id"], @"call_9");
	XCTAssertNil(assistant[@"reasoning_content"], @"the chat fields stay the shape every backend reads");

	NFKToolResult *answer = [NFKToolResult resultWithText:@"-3 C"];
	NSArray *conversation = @[ @{ @"role": @"user", @"content": @"Oslo?" }, assistant,
							   [NFKToolMessages messageForToolCallIdentifier:@"call_9" name:@"get_weather" result:answer] ];
	[deepSeek runInferenceForRequest:[self requestWith:conversation] error:NULL];
	NSDictionary *replayed = [deepSeek bodyAt:1][@"messages"][1];
	XCTAssertEqualObjects(replayed[@"reasoning_content"], @"Need the weather.", @"DeepSeek reads its reasoning back during tool use");
	XCTAssertEqualObjects(replayed[@"tool_calls"][0][@"type"], @"function");
	XCTAssertNil(replayed[NFKMessageMetadataKey]);

	NFKConversationChatStub *other = [self chatStubAt:@"https://api.openai.com/v1/chat/completions"];
	[other runInferenceForRequest:[self requestWith:conversation] error:NULL];
	NSDictionary *plain = [other bodyAt:0][@"messages"][1];
	XCTAssertNil(plain[@"reasoning_content"], @"another endpoint receives only the chat fields");
	XCTAssertNil(plain[NFKMessageMetadataKey]);
	XCTAssertEqualObjects(plain[@"tool_calls"][0][@"id"], @"call_9");
}

#pragma mark Anthropic

- (NFKConversationAnthropicStub *)anthropicStub
{
	NFKConversationAnthropicStub *stub = [[NFKConversationAnthropicStub alloc] init];
	stub.modelName = @"claude-sonnet-4-5";
	stub.staged = [NSMutableArray arrayWithObject:@"{\"content\":[{\"type\":\"text\",\"text\":\"ok\"}]}"];
	return stub;
}

- (void)testAnthropicTakesCallsAsToolUseAndAnswersAsToolResultsOpeningTheNextUserTurn
{
	NFKConversationAnthropicStub *anthropic = self.anthropicStub;
	NSError *error = nil;
	XCTAssertNotNil([anthropic runInferenceForRequest:[self requestWith:self.conversation] error:&error], @"%@", error);
	NSArray<NSDictionary *> *wire = [anthropic bodyAt:0][@"messages"];
	XCTAssertEqual(wire.count, 3u, @"the answers and the follow-up are one user turn");
	NSArray *assistant = wire[1][@"content"];
	XCTAssertEqualObjects(assistant[0], (@{ @"type": @"text", @"text": @"Let me check." }));
	NSDictionary *use = @{ @"type": @"tool_use", @"id": @"c1", @"name": @"get_weather", @"input": @{ @"city": @"Paris" } };
	XCTAssertEqualObjects(assistant[1], use);
	XCTAssertEqualObjects(assistant[2][@"id"], @"c2");

	NSArray *user = wire[2][@"content"];
	XCTAssertEqual(user.count, 3u);
	NSDictionary *failed = @{ @"type": @"tool_result", @"tool_use_id": @"c1", @"content": @"the station is down", @"is_error": @YES };
	XCTAssertEqualObjects(user[0], failed);
	XCTAssertEqualObjects(user[1][@"tool_use_id"], @"c2");
	XCTAssertEqualObjects(user[1][@"content"][0][@"text"], @"a map of Paris");
	XCTAssertEqualObjects(user[1][@"content"][1][@"source"][@"data"], self.base64PNG, @"Claude sees the image a tool returned");
	XCTAssertNil(user[1][@"is_error"]);
	XCTAssertEqualObjects(user[2], (@{ @"type": @"text", @"text": @"Thanks." }));
}

- (void)testAnthropicReplaysItsOwnBlocksWithTheirThinkingSignature
{
	NFKConversationAnthropicStub *anthropic = self.anthropicStub;
	anthropic.staged = [NSMutableArray arrayWithObject:@"{\"content\":["
		"{\"type\":\"thinking\",\"thinking\":\"Look it up.\",\"signature\":\"sig-1\"},"
		"{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"get_weather\",\"input\":{\"city\":\"Rome\"}}]}"];
	NFKInferenceResult *reply = [anthropic runInferenceForRequest:[self requestWith:@[ @{ @"role": @"user", @"content": @"Rome?" } ]] error:NULL];
	NSDictionary *assistant = [NFKToolMessages assistantMessageForResult:reply];
	XCTAssertEqualObjects(assistant[@"tool_calls"][0][@"id"], @"toolu_1");
	XCTAssertEqualObjects(assistant[@"tool_calls"][0][@"function"][@"arguments"], @"{\"city\":\"Rome\"}");

	NSArray *conversation = @[ @{ @"role": @"user", @"content": @"Rome?" }, assistant,
							   [NFKToolMessages messageForToolCallIdentifier:@"toolu_1" name:@"get_weather" result:[NFKToolResult resultWithText:@"21 C"]] ];
	[anthropic runInferenceForRequest:[self requestWith:conversation] error:NULL];
	NSArray *replayed = [anthropic bodyAt:1][@"messages"][1][@"content"];
	NSDictionary *thinking = @{ @"type": @"thinking", @"thinking": @"Look it up.", @"signature": @"sig-1" };
	XCTAssertEqualObjects(replayed[0], thinking, @"the API requires the signed thinking block back while a tool is in use");
	XCTAssertEqualObjects(replayed[1][@"id"], @"toolu_1");
}

// Meta's Messages route opens a reply with the encrypted reasoning and asks for it back as returned.
- (void)testAMuseTurnReplaysItsRedactedThinkingBesideTheToolUse
{
	NFKConversationAnthropicStub *muse = self.anthropicStub;
	muse.endpointURL = [NSURL URLWithString:@"https://api.meta.ai/v1/messages"];
	muse.modelName = @"muse-spark-1.3";
	muse.staged = [NSMutableArray arrayWithObject:@"{\"content\":["
		"{\"type\":\"redacted_thinking\",\"data\":\"enc-1\"},"
		"{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"get_weather\",\"input\":{\"city\":\"Rome\"}}],"
		"\"stop_reason\":\"tool_use\"}"];
	NFKInferenceResult *reply = [muse runInferenceForRequest:[self requestWith:@[ @{ @"role": @"user", @"content": @"Rome?" } ]] error:NULL];
	XCTAssertEqualObjects([reply outputForKey:NFKOutputFinishReason], NFKFinishReasonToolCalls);
	NSArray *conversation = @[ @{ @"role": @"user", @"content": @"Rome?" }, [NFKToolMessages assistantMessageForResult:reply],
							   [NFKToolMessages messageForToolCallIdentifier:@"toolu_1" name:@"get_weather" result:[NFKToolResult resultWithText:@"21 C"]] ];
	[muse runInferenceForRequest:[self requestWith:conversation] error:NULL];
	NSArray *replayed = [muse bodyAt:1][@"messages"][1][@"content"];
	XCTAssertEqualObjects(replayed[0], (@{ @"type": @"redacted_thinking", @"data": @"enc-1" }));
	XCTAssertEqualObjects(replayed[1][@"id"], @"toolu_1");
	XCTAssertEqualObjects([muse bodyAt:1][@"messages"][2][@"content"][0][@"tool_use_id"], @"toolu_1");
}

- (void)testAnthropicLeavesTheForcedSchemaToolOutOfTheRecordedTurn
{
	NFKConversationAnthropicStub *anthropic = self.anthropicStub;
	anthropic.staged = [NSMutableArray arrayWithObject:
		@"{\"content\":[{\"type\":\"tool_use\",\"id\":\"t\",\"name\":\"structured_output\",\"input\":{\"answer\":4}}]}"];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"2+2?" }
															   parameters:@{ NFKParameterJSONSchema: @{ @"type": @"object" } }
														   outputModality:NFKModalityText];
	NFKInferenceResult *reply = [anthropic runInferenceForRequest:request error:NULL];
	XCTAssertEqualObjects(reply.structured, @{ @"answer": @4 });
	NSDictionary *assistant = [NFKToolMessages assistantMessageForResult:reply];
	XCTAssertNil(assistant[@"tool_calls"]);
	XCTAssertEqualObjects(assistant[NFKMessageMetadataKey][@"turn"], @[], @"a call nobody answers is not replayed");
}

- (void)testAToolLoopRunsEndToEndOverTheAnthropicWire
{
	NFKConversationAnthropicStub *anthropic = self.anthropicStub;
	anthropic.staged = [NSMutableArray arrayWithObjects:
		@"{\"content\":[{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"ask_local\",\"input\":{\"prompt\":\"Summarize: the cat sat.\"}}],"
		"\"usage\":{\"input_tokens\":50,\"output_tokens\":10}}",
		@"{\"content\":[{\"type\":\"text\",\"text\":\"The local model says: a cat sat.\"}],\"usage\":{\"input_tokens\":80,\"output_tokens\":12}}", nil];
	__block NSString *asked = nil;
	NFKInferenceTool *local = [NFKInferenceTool toolWithName:@"ask_local" description:@"A local model."
												  parameters:[NFKBackendTool defaultParametersForOutputModality:NFKModalityText]
													 handler:^NFKToolResult *(NSDictionary *arguments, NSError **error) {
		asked = arguments[NFKInputPrompt];
		return [NFKToolResult resultWithText:@"A cat sat."];
	}];
	NFKToolLoop *loop = [[NFKToolLoop alloc] initWithBackend:anthropic tools:@[ local ]];
	NSError *error = nil;
	NFKInferenceResult *result = [loop runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Use the local model." }
																						  parameters:nil outputModality:NFKModalityText]
														 error:&error];
	XCTAssertEqualObjects(result.text, @"The local model says: a cat sat.", @"%@", error);
	XCTAssertEqualObjects(asked, @"Summarize: the cat sat.");
	XCTAssertEqualObjects([anthropic bodyAt:0][@"tools"][0][@"name"], @"ask_local");
	NSArray *second = [anthropic bodyAt:1][@"messages"];
	XCTAssertEqualObjects(second[1][@"content"][0][@"type"], @"tool_use");
	XCTAssertEqualObjects(second[2][@"content"][0][@"tool_use_id"], @"toolu_1");
	XCTAssertEqualObjects(second[2][@"content"][0][@"content"], @"A cat sat.");
	XCTAssertEqualObjects([result outputForKey:NFKOutputUsage][NFKUsageInputTokens], @130);
}

#pragma mark The prefix check

- (NFKPrefixCheckStub *)prefixCheckStub
{
	NFKPrefixCheckStub *stub = [[NFKPrefixCheckStub alloc] init];
	stub.modelName = @"claude-fable-5-2";
	stub.maxTokens = 4096;
	return stub;
}

// The 400 a replayed thinking block gets when the content before it changed, naming the block.
- (NSArray *)prefixMismatchAt:(NSString *)path
{
	NSString *message = [NSString stringWithFormat:@"%@: Invalid `signature` in `thinking` block. The block is bound to a different "
						 "conversation. Remove the block, or set `thinking.block_binding.prefix_mismatch_behavior` to \\\"drop_block\\\". "
						 "That setting requires the `thinking-binding-controls-2026-08-01` value in the `anthropic-beta` header.", path];
	return @[ @400, [NSString stringWithFormat:@"{\"type\":\"error\",\"error\":{\"type\":\"invalid_request_error\",\"message\":\"%@\"}}", message] ];
}

// Two tool turns, each opening with a signed thinking block: user, assistant, tool result, assistant,
// tool result. The assistant turns sit at messages.1 and messages.3 on the wire.
- (NSArray<NSDictionary *> *)twoThinkingTurnsThrough:(NFKPrefixCheckStub *)stub
{
	NSMutableArray<NSDictionary *> *conversation = [NSMutableArray arrayWithObject:@{ @"role": @"user", @"content": @"Rome, then Oslo?" }];
	NSArray<NSString *> *cities = @[ @"Rome", @"Oslo" ];
	for (NSUInteger turn = 0; turn < cities.count; turn++) {
		NSString *call = [NSString stringWithFormat:@"toolu_%lu", (unsigned long)turn + 1];
		NSString *reply = [NSString stringWithFormat:@"{\"content\":["
						   "{\"type\":\"thinking\",\"thinking\":\"\",\"signature\":\"sig-%lu\"},"
						   "{\"type\":\"tool_use\",\"id\":\"%@\",\"name\":\"get_weather\",\"input\":{\"city\":\"%@\"}}]}",
						   (unsigned long)turn + 1, call, cities[turn]];
		[stub.responses addObject:@[ @200, reply ]];
		NFKInferenceResult *result = [stub runInferenceForRequest:[self requestWith:conversation] error:NULL];
		[conversation addObject:[NFKToolMessages assistantMessageForResult:result]];
		[conversation addObject:[NFKToolMessages messageForToolCallIdentifier:call name:@"get_weather" result:[NFKToolResult resultWithText:@"21 C"]]];
	}
	[stub.requests removeAllObjects];
	return conversation;
}

- (NSArray<NSString *> *)blockTypesOf:(NSDictionary *)body atMessage:(NSUInteger)index
{
	return [body[@"messages"][index][@"content"] valueForKey:@"type"];
}

- (void)testAPrefixMismatchIsSentOnceMoreWithoutTheNamedThinkingAndTheThinkingAfterIt
{
	NFKPrefixCheckStub *stub = self.prefixCheckStub;
	NSArray<NSDictionary *> *conversation = [self twoThinkingTurnsThrough:stub];
	NSArray *edited = [@[ @{ @"role": @"system", @"content": @"Answer in Celsius." } ] arrayByAddingObjectsFromArray:conversation];
	[stub.responses addObject:[self prefixMismatchAt:@"messages.3.content.0"]];
	[stub.responses addObject:@[ @200, @"{\"content\":[{\"type\":\"text\",\"text\":\"Both are 21 C.\"}],\"stop_reason\":\"end_turn\"}" ]];

	NSError *error = nil;
	NFKInferenceResult *result = [stub runInferenceForRequest:[self requestWith:edited] error:&error];
	XCTAssertEqualObjects(result.text, @"Both are 21 C.", @"%@", error);
	XCTAssertEqual(stub.requests.count, 2, @"the refused request is sent once more");

	NSDictionary *refused = [stub bodyAt:0];
	XCTAssertEqualObjects([self blockTypesOf:refused atMessage:1], (@[ @"thinking", @"tool_use" ]));
	XCTAssertEqualObjects([self blockTypesOf:refused atMessage:3], (@[ @"thinking", @"tool_use" ]));

	NSDictionary *resent = [stub bodyAt:1];
	XCTAssertEqualObjects(resent[@"messages"][1], refused[@"messages"][1], @"the thinking before the named block stays");
	XCTAssertEqualObjects([self blockTypesOf:resent atMessage:3], (@[ @"tool_use" ]));
	XCTAssertEqualObjects(resent[@"messages"][3][@"content"][0], refused[@"messages"][3][@"content"][1]);
	NSMutableDictionary *rest = [resent mutableCopy];
	NSMutableDictionary *before = [refused mutableCopy];
	[rest removeObjectForKey:@"messages"];
	[before removeObjectForKey:@"messages"];
	XCTAssertEqualObjects(rest, before, @"nothing but the thinking changes");
	for (NSUInteger index = 0; index < [refused[@"messages"] count]; index++) {
		if (index != 3) {
			XCTAssertEqualObjects(resent[@"messages"][index], refused[@"messages"][index], @"messages.%lu", (unsigned long)index);
		}
	}
	XCTAssertEqualObjects(stub.requests[1].allHTTPHeaderFields, stub.requests[0].allHTTPHeaderFields);
	XCTAssertNil(stub.requests[1].allHTTPHeaderFields[@"anthropic-beta"], @"the recovery needs no beta");
	XCTAssertEqualObjects([result outputForKey:NFKAnthropicBackendInputTransformationsKey],
						  (@[ @{ @"type": @"thinking_removed", @"path": @"messages.3.content.0" } ]));
}

- (void)testAMismatchAtTheFirstThinkingBlockRemovesEveryOneAfterItToo
{
	NFKPrefixCheckStub *stub = self.prefixCheckStub;
	NSArray<NSDictionary *> *conversation = [self twoThinkingTurnsThrough:stub];
	[stub.responses addObject:[self prefixMismatchAt:@"messages.1.content.0"]];
	[stub.responses addObject:@[ @200, @"{\"content\":[{\"type\":\"redacted_thinking\",\"data\":\"enc\"},{\"type\":\"text\",\"text\":\"ok\"}]}" ]];
	NFKInferenceResult *result = [stub runInferenceForRequest:[self requestWith:conversation] error:NULL];
	XCTAssertEqualObjects(result.text, @"ok");
	XCTAssertEqualObjects([self blockTypesOf:[stub bodyAt:1] atMessage:1], (@[ @"tool_use" ]));
	XCTAssertEqualObjects([self blockTypesOf:[stub bodyAt:1] atMessage:3], (@[ @"tool_use" ]));
	NSArray *removed = @[ @{ @"type": @"thinking_removed", @"path": @"messages.1.content.0" },
						  @{ @"type": @"thinking_removed", @"path": @"messages.3.content.0" } ];
	XCTAssertEqualObjects([result outputForKey:NFKAnthropicBackendInputTransformationsKey], removed);
	NSArray *turn = [NFKToolMessages assistantMessageForResult:result][NFKMessageMetadataKey][@"turn"];
	XCTAssertEqualObjects(turn[0][@"type"], @"redacted_thinking", @"the new turn keeps the thinking the model produced since");
}

// The error counts two neighbouring messages of one role as one, so the position it names is not an
// index into the array, and every thinking block goes.
- (void)testAMismatchWhoseNamedPositionCannotBeReadRemovesEveryThinkingBlock
{
	NFKPrefixCheckStub *stub = self.prefixCheckStub;
	NSArray<NSDictionary *> *conversation = [self twoThinkingTurnsThrough:stub];
	NSArray *doubled = [@[ @{ @"role": @"user", @"content": @"A question first." } ] arrayByAddingObjectsFromArray:conversation];
	[stub.responses addObject:[self prefixMismatchAt:@"messages.3.content.0"]];
	[stub.responses addObject:@[ @200, @"{\"content\":[{\"type\":\"text\",\"text\":\"ok\"}]}" ]];
	NFKInferenceResult *result = [stub runInferenceForRequest:[self requestWith:doubled] error:NULL];
	XCTAssertEqualObjects(result.text, @"ok");
	NSDictionary *resent = [stub bodyAt:1];
	XCTAssertEqualObjects([self blockTypesOf:resent atMessage:2], (@[ @"tool_use" ]));
	XCTAssertEqualObjects([self blockTypesOf:resent atMessage:4], (@[ @"tool_use" ]));
	NSArray *removed = @[ @{ @"type": @"thinking_removed", @"path": @"messages.2.content.0" },
						  @{ @"type": @"thinking_removed", @"path": @"messages.4.content.0" } ];
	XCTAssertEqualObjects([result outputForKey:NFKAnthropicBackendInputTransformationsKey], removed);
}

- (void)testAPrefixMismatchIsReturnedWhenTheRecoveryIsOffOrTheSecondAttemptIsRefused
{
	NFKPrefixCheckStub *stub = self.prefixCheckStub;
	NSArray<NSDictionary *> *conversation = [self twoThinkingTurnsThrough:stub];
	stub.recoversFromPrefixMismatch = NO;
	[stub.responses addObject:[self prefixMismatchAt:@"messages.3.content.0"]];
	NSError *error = nil;
	XCTAssertNil([stub runInferenceForRequest:[self requestWith:conversation] error:&error]);
	XCTAssertEqual(stub.requests.count, 1);
	XCTAssertEqualObjects(error.userInfo[NFKRemoteErrorStatusCodeKey], @400);
	XCTAssertTrue([error.userInfo[NFKRemoteErrorBodyKey] containsString:@"bound to a different conversation"]);

	stub.recoversFromPrefixMismatch = YES;
	[stub.requests removeAllObjects];
	[stub.responses addObject:[self prefixMismatchAt:@"messages.1.content.0"]];
	[stub.responses addObject:[self prefixMismatchAt:@"messages.1.content.0"]];
	[stub.responses addObject:@[ @200, @"{\"content\":[{\"type\":\"text\",\"text\":\"never read\"}]}" ]];
	error = nil;
	XCTAssertNil([stub runInferenceForRequest:[self requestWith:conversation] error:&error]);
	XCTAssertEqual(stub.requests.count, 2, @"the request is sent again once");
	XCTAssertEqualObjects(error.userInfo[NFKRemoteErrorStatusCodeKey], @400);
}

// A signature that does not verify answers 400 without the sentence about another conversation, and
// removing thinking is no recovery for it.
- (void)testAnotherBadRequestAndAMismatchWithNoThinkingToRemoveAreNotSentAgain
{
	NFKPrefixCheckStub *stub = self.prefixCheckStub;
	NSArray<NSDictionary *> *conversation = [self twoThinkingTurnsThrough:stub];
	[stub.responses addObject:@[ @400, (@"{\"type\":\"error\",\"error\":{\"type\":\"invalid_request_error\","
										"\"message\":\"messages.1.content.0: Invalid `signature` in `thinking` block.\"}}") ]];
	NSError *error = nil;
	XCTAssertNil([stub runInferenceForRequest:[self requestWith:conversation] error:&error]);
	XCTAssertEqual(stub.requests.count, 1);
	XCTAssertEqualObjects(error.userInfo[NFKRemoteErrorStatusCodeKey], @400);

	[stub.requests removeAllObjects];
	[stub.responses addObject:[self prefixMismatchAt:@"messages.1.content.0"]];
	NSArray<NSDictionary *> *plain = @[ @{ @"role": @"user", @"content": @"Rome?" } ];
	XCTAssertNil([stub runInferenceForRequest:[self requestWith:plain] error:&error]);
	XCTAssertEqual(stub.requests.count, 1, @"a body without thinking has nothing to remove");
}

- (void)testARequestThatPassesThePrefixCheckGoesOutOnceAsBuilt
{
	NFKPrefixCheckStub *stub = self.prefixCheckStub;
	NSArray<NSDictionary *> *conversation = [self twoThinkingTurnsThrough:stub];
	[stub.responses addObject:@[ @200, @"{\"content\":[{\"type\":\"text\",\"text\":\"Both are 21 C.\"}]}" ]];
	NFKInferenceResult *result = [stub runInferenceForRequest:[self requestWith:conversation] error:NULL];
	XCTAssertEqualObjects(result.text, @"Both are 21 C.");
	XCTAssertEqual(stub.requests.count, 1);
	NSDictionary *body = [stub bodyAt:0];
	XCTAssertNil(body[@"thinking"], @"no setting rides on a request by default");
	XCTAssertNil(stub.requests[0].allHTTPHeaderFields[@"anthropic-beta"]);
	XCTAssertEqualObjects(body[@"messages"][1][@"content"][0], (@{ @"type": @"thinking", @"thinking": @"", @"signature": @"sig-1" }),
						  @"a thinking block goes back as returned, its text empty or not");
	XCTAssertEqualObjects(body[@"messages"][3][@"content"][0][@"signature"], @"sig-2");
	XCTAssertNil([result outputForKey:NFKAnthropicBackendInputTransformationsKey]);
}

- (void)testThePrefixMismatchBehaviorRidesOnAdaptiveThinkingUnderItsBeta
{
	NFKPrefixCheckStub *stub = self.prefixCheckStub;
	stub.prefixMismatchBehavior = NFKAnthropicPrefixMismatchBehaviorDropBlock;
	NSString *dropped = @"{\"content\":[{\"type\":\"text\",\"text\":\"ok\"}],\"input_transformations\":["
		"{\"type\":\"thinking_dropped\",\"path\":\"messages.1.content.0\",\"reason\":\"prefix_binding_mismatch\"}]}";
	[stub.responses addObject:@[ @200, dropped ]];
	NFKInferenceRequest *plain = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"hi" }];
	NFKInferenceResult *result = [stub runInferenceForRequest:plain error:NULL];
	NSDictionary *binding = @{ @"prefix_mismatch_behavior": @"drop_block" };
	XCTAssertEqualObjects([stub bodyAt:0][@"thinking"], (@{ @"type": @"adaptive", @"block_binding": binding }));
	XCTAssertEqualObjects(stub.requests[0].allHTTPHeaderFields[@"anthropic-beta"], @"thinking-binding-controls-2026-08-01");
	NSArray *reported = @[ @{ @"type": @"thinking_dropped", @"path": @"messages.1.content.0", @"reason": @"prefix_binding_mismatch" } ];
	XCTAssertEqualObjects([result outputForKey:NFKAnthropicBackendInputTransformationsKey], reported);

	stub.prefixMismatchBehavior = NFKAnthropicPrefixMismatchBehaviorError;
	NFKInferenceRequest *thinking = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"hi" }
															   parameters:@{ NFKParameterReasoningEffort: NFKReasoningEffortLight }];
	[stub runInferenceForRequest:thinking error:NULL];
	NSDictionary *sent = [stub bodyAt:1][@"thinking"];
	XCTAssertEqualObjects(sent, (@{ @"type": @"adaptive", @"display": @"summarized", @"block_binding": @{ @"prefix_mismatch_behavior": @"error" } }));

	[stub.responses addObject:@[ @200, @"{\"input_tokens\":9}" ]];
	XCTAssertEqualObjects([stub tokenCountForRequest:plain error:NULL], @9);
	XCTAssertEqualObjects([stub bodyAt:2][@"thinking"][@"block_binding"], (@{ @"prefix_mismatch_behavior": @"error" }));
	XCTAssertEqualObjects(stub.requests[2].allHTTPHeaderFields[@"anthropic-beta"], @"thinking-binding-controls-2026-08-01");

	// Claude Opus 5 and Claude Haiku 4.5 run no prefix check, and Meta's route refuses fields it does not name.
	for (NSString *model in @[ @"claude-opus-5", @"claude-haiku-4-5", @"muse-spark-1.3" ]) {
		stub.modelName = model;
		[stub.requests removeAllObjects];
		[stub runInferenceForRequest:plain error:NULL];
		XCTAssertNil([stub bodyAt:0][@"thinking"], @"%@", model);
		XCTAssertNil(stub.requests[0].allHTTPHeaderFields[@"anthropic-beta"], @"%@", model);
	}
}

- (void)testAStreamThePrefixCheckRefusesIsSentAgainWithoutTheThinking
{
	NFKPrefixCheckStub *stub = self.prefixCheckStub;
	NSArray<NSDictionary *> *conversation = [self twoThinkingTurnsThrough:stub];
	NSString *lines = @"data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n"
		"data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Both are 21 C.\"}}\n"
		"data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n"
		"data: {\"type\":\"message_stop\"}";
	[stub.streams addObject:[self prefixMismatchAt:@"messages.3.content.0"]];
	[stub.streams addObject:@[ @200, lines ]];
	NFKInferenceJob *job = [stub submitInferenceJobForRequest:[self requestWith:conversation]];
	XCTAssertEqual(job.status, NFKInferenceJobStatusSucceeded, @"%@", job.error);
	XCTAssertEqualObjects(job.result.text, @"Both are 21 C.");
	XCTAssertEqual(stub.requests.count, 2);
	XCTAssertEqualObjects([stub bodyAt:1][@"stream"], @YES);
	XCTAssertEqualObjects([self blockTypesOf:[stub bodyAt:1] atMessage:1], (@[ @"thinking", @"tool_use" ]));
	XCTAssertEqualObjects([self blockTypesOf:[stub bodyAt:1] atMessage:3], (@[ @"tool_use" ]));
	XCTAssertEqualObjects([job.result outputForKey:NFKAnthropicBackendInputTransformationsKey],
						  (@[ @{ @"type": @"thinking_removed", @"path": @"messages.3.content.0" } ]));

	[stub.requests removeAllObjects];
	[stub.streams addObject:[self prefixMismatchAt:@"messages.3.content.0"]];
	[stub.streams addObject:[self prefixMismatchAt:@"messages.3.content.0"]];
	NFKInferenceJob *refused = [stub submitInferenceJobForRequest:[self requestWith:conversation]];
	XCTAssertEqual(refused.status, NFKInferenceJobStatusFailed);
	XCTAssertEqual(stub.requests.count, 2, @"the stream is sent again once");
	XCTAssertEqualObjects(refused.error.userInfo[NFKRemoteErrorStatusCodeKey], @400);
}

// Under the beta the stream's opening message carries the blocks the API dropped.
- (void)testAStreamReportsTheBlocksTheAPIDropped
{
	NFKPrefixCheckStub *stub = self.prefixCheckStub;
	stub.prefixMismatchBehavior = NFKAnthropicPrefixMismatchBehaviorDropBlock;
	NSString *lines = @"data: {\"type\":\"message_start\",\"message\":{\"role\":\"assistant\",\"content\":[],\"input_transformations\":["
		"{\"type\":\"thinking_dropped\",\"path\":\"messages.1.content.0\",\"reason\":\"prefix_binding_mismatch\"}]}}\n"
		"data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"ok\"}}\n"
		"data: {\"type\":\"message_stop\"}";
	[stub.streams addObject:@[ @200, lines ]];
	NFKInferenceJob *job = [stub submitInferenceJobForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"hi" }]];
	XCTAssertEqualObjects(job.result.text, @"ok");
	XCTAssertEqualObjects([job.result outputForKey:NFKAnthropicBackendInputTransformationsKey][0][@"reason"], @"prefix_binding_mismatch");
	XCTAssertEqualObjects(stub.requests[0].allHTTPHeaderFields[@"anthropic-beta"], @"thinking-binding-controls-2026-08-01");
}

// count_tokens runs the same check, so a count of an edited conversation recovers the same way.
- (void)testACountThePrefixCheckRefusesIsSentAgainWithoutTheThinking
{
	NFKPrefixCheckStub *stub = self.prefixCheckStub;
	NSArray<NSDictionary *> *conversation = [self twoThinkingTurnsThrough:stub];
	[stub.responses addObject:[self prefixMismatchAt:@"messages.3.content.0"]];
	[stub.responses addObject:@[ @200, @"{\"input_tokens\":42}" ]];
	NSError *error = nil;
	XCTAssertEqualObjects([stub tokenCountForRequest:[self requestWith:conversation] error:&error], @42, @"%@", error);
	XCTAssertEqual(stub.requests.count, 2);
	XCTAssertEqualObjects(stub.requests[1].URL.absoluteString, @"https://api.anthropic.com/v1/messages/count_tokens");
	XCTAssertEqualObjects([self blockTypesOf:[stub bodyAt:1] atMessage:1], (@[ @"thinking", @"tool_use" ]));
	XCTAssertEqualObjects([self blockTypesOf:[stub bodyAt:1] atMessage:3], (@[ @"tool_use" ]));
}

#pragma mark Responses

- (void)testResponsesTakesCallsAsFunctionCallItemsAndReplaysItsOwnOutput
{
	NFKConversationResponsesStub *responses = [NFKConversationResponsesStub backendWithEndpointURL:[NSURL URLWithString:@"https://api.openai.com/v1/responses"]];
	responses.modelName = @"gpt-5.6-sol";
	responses.staged = [NSMutableArray arrayWithObject:@"{\"status\":\"completed\",\"output\":[{\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"ok\"}]}]}"];
	NSError *error = nil;
	XCTAssertNotNil([responses runInferenceForRequest:[self requestWith:self.conversation] error:&error], @"%@", error);
	NSArray<NSDictionary *> *items = [responses bodyAt:0][@"input"];
	XCTAssertEqual(items.count, 7u, @"user, assistant text, two calls, two outputs, user");
	XCTAssertEqualObjects(items[1][@"role"], @"assistant");
	NSDictionary *call = @{ @"type": @"function_call", @"call_id": @"c1", @"name": @"get_weather", @"arguments": @"{\"city\":\"Paris\"}" };
	XCTAssertEqualObjects(items[2], call);
	NSDictionary *failed = @{ @"type": @"function_call_output", @"call_id": @"c1", @"output": @"Error: the station is down" };
	XCTAssertEqualObjects(items[4], failed);
	XCTAssertEqualObjects(items[5][@"output"][1][@"type"], @"input_image");
	XCTAssertEqualObjects(items[6][@"role"], @"user");

	responses.staged = [NSMutableArray arrayWithObject:@"{\"status\":\"completed\",\"output\":["
		"{\"type\":\"reasoning\",\"id\":\"rs_1\",\"summary\":[]},"
		"{\"type\":\"function_call\",\"id\":\"fc_1\",\"call_id\":\"call_r\",\"name\":\"get_weather\",\"arguments\":\"{}\"}]}"];
	NFKInferenceResult *reply = [responses runInferenceForRequest:[self requestWith:@[ @{ @"role": @"user", @"content": @"x" } ]] error:NULL];
	NSArray *conversation = @[ @{ @"role": @"user", @"content": @"x" }, [NFKToolMessages assistantMessageForResult:reply],
							   [NFKToolMessages messageForToolCallIdentifier:@"call_r" name:@"get_weather" result:[NFKToolResult resultWithText:@"fine"]] ];
	[responses runInferenceForRequest:[self requestWith:conversation] error:NULL];
	NSArray<NSDictionary *> *replayed = [responses bodyAt:2][@"input"];
	XCTAssertEqualObjects(replayed[1][@"id"], @"rs_1", @"the reasoning item goes back with the call it led to");
	XCTAssertEqualObjects(replayed[2][@"id"], @"fc_1");
	XCTAssertEqualObjects(replayed[3][@"type"], @"function_call_output");
}

#pragma mark MCP servers the provider calls

- (NSArray<NSDictionary *> *)mcpServers
{
	return @[ @{ @"url": [NSURL URLWithString:@"https://tools.example.com/mcp"], @"name": @"inferkit",
				 @"authorizationToken": @"secret", @"allowedTools": @[ @"speak" ], @"description": @"Local models." } ];
}

- (void)testClaudesConnectorReachesAnMCPServerAndReportsWhatItRan
{
	NFKConversationAnthropicStub *anthropic = self.anthropicStub;
	anthropic.staged = [NSMutableArray arrayWithObject:@"{\"content\":["
		"{\"type\":\"mcp_tool_use\",\"id\":\"mcptoolu_1\",\"name\":\"speak\",\"server_name\":\"inferkit\",\"input\":{\"prompt\":\"Hi\"}},"
		"{\"type\":\"mcp_tool_result\",\"tool_use_id\":\"mcptoolu_1\",\"is_error\":false,\"content\":[{\"type\":\"text\",\"text\":\"Generated 1.0 seconds of audio.\"}]},"
		"{\"type\":\"text\",\"text\":\"Done.\"}]}"];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Say hi." }
															   parameters:@{ NFKParameterMCPServers: self.mcpServers }
														   outputModality:NFKModalityText];
	NSError *error = nil;
	NFKInferenceResult *result = [anthropic runInferenceForRequest:request error:&error];
	XCTAssertEqualObjects(result.text, @"Done.", @"%@", error);
	NSDictionary *body = [anthropic bodyAt:0];
	NSDictionary *server = @{ @"type": @"url", @"url": @"https://tools.example.com/mcp", @"name": @"inferkit", @"authorization_token": @"secret" };
	XCTAssertEqualObjects(body[@"mcp_servers"], @[ server ]);
	NSDictionary *toolset = @{ @"type": @"mcp_toolset", @"mcp_server_name": @"inferkit", @"default_config": @{ @"enabled": @NO },
							   @"configs": @{ @"speak": @{ @"enabled": @YES } } };
	XCTAssertEqualObjects(body[@"tools"], @[ toolset ], @"an allowlist turns every other tool off");
	XCTAssertEqualObjects([anthropic.requests[0] valueForHTTPHeaderField:@"anthropic-beta"], @"mcp-client-2025-11-20");
	NSDictionary *ran = [[result outputForKey:NFKOutputServerToolResults] firstObject];
	XCTAssertEqualObjects(ran[@"name"], @"speak");
	XCTAssertEqualObjects(ran[@"input"], @{ @"prompt": @"Hi" });
	XCTAssertEqualObjects(ran[@"output"][0][@"text"], @"Generated 1.0 seconds of audio.");
	XCTAssertNil(result.toolCalls, @"a call the service ran is not the caller's to run");

	NFKConversationAnthropicStub *plain = self.anthropicStub;
	[plain runInferenceForRequest:[self requestWith:@[ @{ @"role": @"user", @"content": @"x" } ]] error:NULL];
	XCTAssertNil([plain.requests[0] valueForHTTPHeaderField:@"anthropic-beta"], @"the beta goes out only with a server");
}

- (void)testResponsesCallsAnMCPServerWithoutAnApprovalRoundTrip
{
	NFKConversationResponsesStub *responses = [NFKConversationResponsesStub backendWithEndpointURL:[NSURL URLWithString:@"https://api.openai.com/v1/responses"]];
	responses.modelName = @"gpt-5.6-sol";
	responses.staged = [NSMutableArray arrayWithObject:@"{\"status\":\"completed\",\"output\":["
		"{\"type\":\"mcp_list_tools\",\"id\":\"mcpl_1\",\"server_label\":\"inferkit\",\"tools\":[]},"
		"{\"type\":\"mcp_call\",\"id\":\"mcp_1\",\"name\":\"speak\",\"server_label\":\"inferkit\",\"arguments\":\"{}\",\"output\":\"ok\"},"
		"{\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"Done.\"}]}]}"];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Say hi." }
															   parameters:@{ NFKParameterMCPServers: self.mcpServers }
														   outputModality:NFKModalityText];
	NSError *error = nil;
	NFKInferenceResult *result = [responses runInferenceForRequest:request error:&error];
	XCTAssertEqualObjects(result.text, @"Done.", @"%@", error);
	NSDictionary *tool = @{ @"type": @"mcp", @"server_label": @"inferkit", @"server_url": @"https://tools.example.com/mcp",
							@"require_approval": @"never", @"authorization": @"secret", @"allowed_tools": @[ @"speak" ],
							@"server_description": @"Local models." };
	XCTAssertEqualObjects([responses bodyAt:0][@"tools"], @[ tool ]);
	NSDictionary *ran = [[result outputForKey:NFKOutputServerToolResults] firstObject];
	XCTAssertEqualObjects(ran[@"name"], @"mcp_call");
	XCTAssertEqualObjects(ran[@"output"], @"ok");
}

- (void)testAChatBackendWithoutAConnectorSendsNothingForTheServers
{
	NFKConversationChatStub *chat = [self chatStubAt:@"https://api.openai.com/v1/chat/completions"];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"x" }
															   parameters:@{ NFKParameterMCPServers: self.mcpServers }
														   outputModality:NFKModalityText];
	NSError *error = nil;
	XCTAssertNotNil([chat runInferenceForRequest:request error:&error], @"%@", error);
	NSDictionary *body = [chat bodyAt:0];
	XCTAssertNil(body[NFKParameterMCPServers], @"the key is not folded in by name");
	XCTAssertNil(body[@"tools"]);
}

#pragma mark Skills

- (NSArray<NSDictionary *> *)skillReferences
{
	return @[ @{ @"id": @"skill_01Abc", @"source": @"custom" },
			  @{ @"id": @"pdf", @"source": @"anthropic", @"version": @"20251013" },
			  @{ @"type": @"builtin", @"name": @"office/pdf" } ];
}

// Claude runs a skill in a container, which the code-execution tool provides; a request that names
// its own code-execution tool keeps it.
- (void)testClaudeLoadsSkillsInAContainerWithTheCodeExecutionTool
{
	NFKConversationAnthropicStub *anthropic = self.anthropicStub;
	anthropic.staged = [NSMutableArray arrayWithObject:@"{\"content\":[{\"type\":\"text\",\"text\":\"Done.\"}]}"];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Make a report." }
															   parameters:@{ NFKParameterSkills: self.skillReferences }
														   outputModality:NFKModalityText];
	NSError *error = nil;
	XCTAssertNotNil([anthropic runInferenceForRequest:request error:&error], @"%@", error);
	NSDictionary *body = [anthropic bodyAt:0];
	NSArray *skills = @[ @{ @"type": @"custom", @"skill_id": @"skill_01Abc", @"version": @"latest" },
						 @{ @"type": @"anthropic", @"skill_id": @"pdf", @"version": @"20251013" },
						 @{ @"type": @"builtin", @"name": @"office/pdf" } ];
	XCTAssertEqualObjects(body[@"container"], @{ @"skills": skills });
	XCTAssertEqualObjects(body[@"tools"], (@[ @{ @"type": @"code_execution_20260521", @"name": @"code_execution" } ]));
	XCTAssertNil(body[NFKParameterSkills]);

	NFKConversationAnthropicStub *ownTool = self.anthropicStub;
	ownTool.staged = [NSMutableArray arrayWithObject:@"{\"content\":[{\"type\":\"text\",\"text\":\"Done.\"}]}"];
	NSDictionary *older = @{ @"type": @"code_execution_20250825", @"name": @"code_execution" };
	NFKInferenceRequest *withTool = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Make a report." }
																parameters:@{ NFKParameterSkills: self.skillReferences, NFKParameterTools: @[ older ] }
															outputModality:NFKModalityText];
	[ownTool runInferenceForRequest:withTool error:NULL];
	XCTAssertEqualObjects([ownTool bodyAt:0][@"tools"], @[ older ]);
}

- (void)testPerplexitysAgentNamesACustomSkillByIdAndPassesABuiltinThrough
{
	NFKConversationResponsesStub *agent = [NFKConversationResponsesStub backendWithEndpointURL:[NSURL URLWithString:@"https://api.perplexity.ai/v1/agent"]];
	agent.dialect = NFKRemoteResponsesDialectPerplexityAgent;
	agent.modelName = @"openai/gpt-5.5";
	agent.staged = [NSMutableArray arrayWithObject:@"{\"status\":\"completed\",\"output\":[]}"];
	NSArray *references = @[ @{ @"id": @"skill_9", @"version": @"rev_2" }, @{ @"type": @"builtin", @"name": @"office/pdf" } ];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Make a report." }
															   parameters:@{ NFKParameterSkills: references }
														   outputModality:NFKModalityText];
	XCTAssertNotNil([agent runInferenceForRequest:request error:NULL]);
	XCTAssertEqualObjects([agent bodyAt:0][@"skills"], (@[ @{ @"type": @"custom", @"id": @"skill_9", @"version": @"rev_2" },
														   @{ @"type": @"builtin", @"name": @"office/pdf" } ]));

	NFKConversationResponsesStub *openAI = [NFKConversationResponsesStub backendWithEndpointURL:[NSURL URLWithString:@"https://api.openai.com/v1/responses"]];
	openAI.modelName = @"gpt-5.6-sol";
	openAI.staged = [NSMutableArray arrayWithObject:@"{\"status\":\"completed\",\"output\":[]}"];
	[openAI runInferenceForRequest:request error:NULL];
	XCTAssertNil([openAI bodyAt:0][@"skills"], @"OpenAI's Responses API takes no skills");
}

- (void)testAChatBackendSendsNothingForTheSkills
{
	NFKConversationChatStub *chat = [self chatStubAt:@"https://api.openai.com/v1/chat/completions"];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"x" }
															   parameters:@{ NFKParameterSkills: self.skillReferences }
														   outputModality:NFKModalityText];
	XCTAssertNotNil([chat runInferenceForRequest:request error:NULL]);
	XCTAssertNil([chat bodyAt:0][NFKParameterSkills], @"the key is not folded in by name");
}

#pragma mark Gemini Interactions

- (void)testInteractionsTakesCallsAsStepsNamesTheirResultsAndReplaysItsOwnSteps
{
	NFKConversationInteractionsStub *gemini = [[NFKConversationInteractionsStub alloc] init];
	gemini.modelName = @"gemini-3-flash";
	gemini.staged = [NSMutableArray arrayWithObject:@"{\"status\":\"completed\",\"steps\":[{\"type\":\"model_output\",\"content\":[{\"type\":\"text\",\"text\":\"ok\"}]}]}"];
	NSError *error = nil;
	XCTAssertNotNil([gemini runInferenceForRequest:[self requestWith:self.conversation] error:&error], @"%@", error);
	NSArray<NSDictionary *> *steps = [gemini bodyAt:0][@"input"];
	XCTAssertEqual(steps.count, 7u);
	NSDictionary *call = @{ @"type": @"function_call", @"id": @"c1", @"name": @"get_weather", @"arguments": @{ @"city": @"Paris" } };
	XCTAssertEqualObjects(steps[2], call);
	NSDictionary *failed = @{ @"type": @"function_result", @"call_id": @"c1", @"name": @"get_weather",
							  @"result": @"the station is down", @"is_error": @YES };
	XCTAssertEqualObjects(steps[4], failed);
	XCTAssertEqualObjects(steps[5][@"result"][1][@"data"], self.base64PNG);
	XCTAssertEqualObjects(steps[6][@"type"], @"user_input");

	NSDictionary *unnamed = @{ @"role": @"tool", @"tool_call_id": @"c2", @"content": @"done" };
	NSArray *bare = @[ self.conversation[0], self.conversation[1], unnamed ];
	[gemini runInferenceForRequest:[self requestWith:bare] error:NULL];
	XCTAssertEqualObjects([[gemini bodyAt:1][@"input"] lastObject][@"name"], @"get_map", @"an answer's name is found from the call it answers");

	gemini.staged = [NSMutableArray arrayWithObject:@"{\"status\":\"completed\",\"steps\":["
		"{\"type\":\"thought\",\"signature\":\"sig-g\"},"
		"{\"type\":\"function_call\",\"id\":\"fn_1\",\"name\":\"get_weather\",\"arguments\":{\"city\":\"Kyoto\"}}]}"];
	NFKInferenceResult *reply = [gemini runInferenceForRequest:[self requestWith:@[ @{ @"role": @"user", @"content": @"Kyoto?" } ]] error:NULL];
	NSArray *conversation = @[ @{ @"role": @"user", @"content": @"Kyoto?" }, [NFKToolMessages assistantMessageForResult:reply],
							   [NFKToolMessages messageForToolCallIdentifier:@"fn_1" name:@"get_weather" result:[NFKToolResult resultWithText:@"mild"]] ];
	[gemini runInferenceForRequest:[self requestWith:conversation] error:NULL];
	NSArray<NSDictionary *> *replayed = [gemini bodyAt:3][@"input"];
	XCTAssertEqualObjects(replayed[1][@"signature"], @"sig-g", @"the thought's signature goes back with the call");
	XCTAssertEqualObjects(replayed[2][@"id"], @"fn_1");
	XCTAssertEqualObjects(replayed[3][@"type"], @"function_result");
}

@end
