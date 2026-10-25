//
//  NFKMistralConversationsBackendTests.m
//  InferKitTests
//
//  Mistral's Conversations API through stub transports: the inputs and completion_args on the wire,
//  built-in tools by wire shape, every output entry kind read back, a conversation continued and
//  restarted, a tool file downloaded as an image, the stream, and the factory. A live test is gated
//  on a key.
//

#import <XCTest/XCTest.h>
#import <InferKit/NFKMistralConversationsBackend.h>
#import <InferKit/NFKRemoteProvider.h>
#import <InferKit/NFKImageCoding.h>
#import <InferKit/NFKInferenceRequest.h>
#import <InferKit/NFKInferenceResult.h>
#import <InferKit/NFKInferenceJob.h>
#import <InferKit/NFKInferenceKeys.h>
#import <InferKit/NFKToolMessages.h>
#import <InferKit/NFKErrors.h>
#import <CoreGraphics/CoreGraphics.h>

@interface NFKStubConversationsBackend : NFKMistralConversationsBackend
@property (nonatomic, strong) NSMutableArray<NSURLRequest *> *requests;
@property (nonatomic, strong) NSMutableArray<NSData *> *staged;
@property (nonatomic, copy) NSArray<NSString *> *stagedLines;
@end

@implementation NFKStubConversationsBackend

- (void)stage:(NSString *)body
{
	if (self.staged == nil) {
		self.staged = [NSMutableArray array];
	}
	[self.staged addObject:[body dataUsingEncoding:NSUTF8StringEncoding]];
}

- (NSData *)sendRequest:(NSURLRequest *)request response:(NSHTTPURLResponse **)outResponse error:(NSError **)outError
{
	if (self.requests == nil) {
		self.requests = [NSMutableArray array];
	}
	[self.requests addObject:request];
	if (outResponse != NULL) {
		*outResponse = [[NSHTTPURLResponse alloc] initWithURL:request.URL statusCode:200 HTTPVersion:@"HTTP/1.1" headerFields:nil];
	}
	NSData *body = self.staged.firstObject ?: [@"{}" dataUsingEncoding:NSUTF8StringEncoding];
	if (self.staged.count > 0) {
		[self.staged removeObjectAtIndex:0];
	}
	return body;
}

- (void (^)(void))streamRequest:(NSURLRequest *)request
					lineHandler:(void (^)(NSString *))lineHandler
			  completionHandler:(void (^)(NSHTTPURLResponse *, NSData *, NSError *))completionHandler
{
	if (self.requests == nil) {
		self.requests = [NSMutableArray array];
	}
	[self.requests addObject:request];
	for (NSString *line in self.stagedLines) {
		lineHandler(line);
	}
	completionHandler([[NSHTTPURLResponse alloc] initWithURL:request.URL statusCode:200 HTTPVersion:@"HTTP/1.1" headerFields:nil], nil, nil);
	return ^{};
}

- (NSDictionary *)bodyAt:(NSUInteger)index
{
	return [NSJSONSerialization JSONObjectWithData:self.requests[index].HTTPBody options:0 error:NULL];
}

@end

@interface NFKMistralConversationsBackendTests : XCTestCase
@property (nonatomic, strong) NFKStubConversationsBackend *backend;
@end

@implementation NFKMistralConversationsBackendTests

- (void)setUp
{
	[super setUp];
	self.backend = [NFKStubConversationsBackend backendWithEndpointURL:[NSURL URLWithString:@"https://api.mistral.ai/v1/conversations"]];
	self.backend.apiKey = @"mistral-key";
	self.backend.modelName = @"mistral-medium-3-5";
}

- (NSString *)textReply:(NSString *)text
{
	return [NSString stringWithFormat:@"{\"object\":\"conversation.response\",\"conversation_id\":\"conv_1\",\"outputs\":["
			"{\"object\":\"entry\",\"type\":\"message.output\",\"id\":\"msg_1\",\"role\":\"assistant\",\"content\":\"%@\"}],"
			"\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":3,\"total_tokens\":15}}", text];
}

#pragma mark The request

- (void)testANewConversationCarriesItsModelInstructionsToolsAndCompletionArgs
{
	[self.backend stage:[self textReply:@"Hi."]];
	NSDictionary *weather = @{ @"name": @"get_weather", @"description": @"Weather", @"parameters": @{ @"type": @"object" } };
	NSArray *messages = @[ @{ @"role": @"system", @"content": @"Be brief." }, @{ @"role": @"user", @"content": @"Weather in Paris?" } ];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputMessages: messages }
															   parameters:@{ NFKParameterTools: @[ weather, @{ @"type": @"web_search" } ],
																			 NFKParameterTemperature: @0.3, NFKParameterMaxTokens: @200,
																			 NFKParameterStopSequences: @[ @"END" ], NFKParameterSeed: @7,
																			 NFKParameterReasoningEffort: NFKReasoningEffortDeep,
																			 NFKParameterJSONSchema: @{ @"type": @"object" },
																			 @"store": @NO }
														   outputModality:NFKModalityText];
	NSError *error = nil;
	XCTAssertNotNil([self.backend runInferenceForRequest:request error:&error], @"%@", error);
	NSURLRequest *sent = self.backend.requests[0];
	XCTAssertEqualObjects(sent.URL.absoluteString, @"https://api.mistral.ai/v1/conversations");
	XCTAssertEqualObjects([sent valueForHTTPHeaderField:@"Authorization"], @"Bearer mistral-key");
	NSDictionary *body = [self.backend bodyAt:0];
	XCTAssertEqualObjects(body[@"model"], @"mistral-medium-3-5");
	XCTAssertEqualObjects(body[@"instructions"], @"Be brief.");
	XCTAssertEqualObjects(body[@"inputs"], (@[ @{ @"type": @"message.input", @"role": @"user", @"content": @"Weather in Paris?" } ]));
	XCTAssertEqualObjects(body[@"tools"][0], (@{ @"type": @"function", @"function": weather }));
	XCTAssertEqualObjects(body[@"tools"][1], (@{ @"type": @"web_search" }));
	NSDictionary *args = body[@"completion_args"];
	XCTAssertEqualObjects(args[@"temperature"], @0.3);
	XCTAssertEqualObjects(args[@"max_tokens"], @200);
	XCTAssertEqualObjects(args[@"stop"], (@[ @"END" ]));
	XCTAssertEqualObjects(args[@"random_seed"], @7);
	XCTAssertEqualObjects(args[@"reasoning_effort"], @"high");
	XCTAssertEqualObjects(args[@"response_format"][@"type"], @"json_schema");
	XCTAssertEqualObjects(body[@"store"], @NO, @"an unnamed parameter folds in by name");
	XCTAssertNil(body[NFKParameterTemperature]);
}

- (void)testALonePromptGoesAsAStringAndAnImageAsAChunk
{
	[self.backend stage:[self textReply:@"One."]];
	[self.backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Hello" }] error:NULL];
	XCTAssertEqualObjects([self.backend bodyAt:0][@"inputs"], @"Hello");

	CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
	CGContextRef context = CGBitmapContextCreate(NULL, 2, 2, 8, 8, colorSpace, kCGBitmapByteOrder32Little | kCGImageAlphaPremultipliedFirst);
	CGColorSpaceRelease(colorSpace);
	CGImageRef image = CGBitmapContextCreateImage(context);
	CGContextRelease(context);
	[self.backend stage:[self textReply:@"A square."]];
	[self.backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"What is this?", NFKInputImage: (__bridge id)image }] error:NULL];
	CGImageRelease(image);
	NSArray *content = [self.backend bodyAt:1][@"inputs"][0][@"content"];
	XCTAssertEqualObjects(content[0], (@{ @"type": @"text", @"text": @"What is this?" }));
	XCTAssertEqualObjects(content[1][@"type"], @"image_url");
	XCTAssertTrue([content[1][@"image_url"] hasPrefix:@"data:image/png;base64,"]);
}

- (void)testAConversationContinuesAndRestartsUnderItsIdentifier
{
	[self.backend stage:[self textReply:@"Again."]];
	NFKInferenceRequest *next = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"And tomorrow?" }
															parameters:@{ NFKParameterPreviousResponseIdentifier: @"conv_1", NFKParameterTools: @[ @{ @"type": @"web_search" } ] }
														outputModality:NFKModalityText];
	XCTAssertNotNil([self.backend runInferenceForRequest:next error:NULL]);
	XCTAssertEqualObjects(self.backend.requests[0].URL.absoluteString, @"https://api.mistral.ai/v1/conversations/conv_1");
	NSDictionary *body = [self.backend bodyAt:0];
	XCTAssertEqualObjects(body[@"inputs"], @"And tomorrow?");
	XCTAssertNil(body[@"model"], @"a continued conversation keeps the model it began with");
	XCTAssertNil(body[@"tools"]);

	[self.backend stage:[self textReply:@"Retried."]];
	NFKInferenceRequest *restart = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Try again." }
															   parameters:@{ NFKParameterPreviousResponseIdentifier: @"conv_1", @"from_entry_id": @"msg_1" }
														   outputModality:NFKModalityText];
	XCTAssertNotNil([self.backend runInferenceForRequest:restart error:NULL]);
	XCTAssertEqualObjects(self.backend.requests[1].URL.absoluteString, @"https://api.mistral.ai/v1/conversations/conv_1/restart");
	XCTAssertEqualObjects([self.backend bodyAt:1][@"from_entry_id"], @"msg_1");
}

#pragma mark The reply

- (void)testEveryEntryKindIsReadBack
{
	[self.backend stage:(@"{\"conversation_id\":\"conv_2\",\"outputs\":["
						 "{\"type\":\"tool.execution\",\"name\":\"web_search\",\"arguments\":\"{\\\"query\\\":\\\"paris weather\\\"}\",\"info\":{\"results\":2}},"
						 "{\"type\":\"agent.handoff\",\"previous_agent_name\":\"router\",\"next_agent_name\":\"weather\"},"
						 "{\"type\":\"message.output\",\"role\":\"assistant\",\"content\":["
						 "{\"type\":\"thinking\",\"thinking\":[{\"type\":\"text\",\"text\":\"Check the forecast.\"}]},"
						 "{\"type\":\"text\",\"text\":\"Sunny, 21°C.\"},"
						 "{\"type\":\"tool_reference\",\"tool\":\"web_search\",\"title\":\"Météo Paris\",\"url\":\"https://meteo.example/paris\"}]}],"
						 "\"usage\":{\"prompt_tokens\":40,\"completion_tokens\":8,\"total_tokens\":48,\"connector_tokens\":200}}")];
	NSError *error = nil;
	NFKInferenceResult *result = [self.backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Weather?" }] error:&error];
	XCTAssertEqualObjects(result.text, @"Sunny, 21°C.", @"%@", error);
	XCTAssertEqualObjects([result outputForKey:NFKOutputReasoning], @"Check the forecast.");
	XCTAssertEqualObjects([result outputForKey:NFKOutputResponseIdentifier], @"conv_2");
	XCTAssertEqualObjects([result outputForKey:NFKOutputFinishReason], NFKFinishReasonStop);
	XCTAssertEqualObjects([result outputForKey:NFKOutputUsage][NFKUsageInputTokens], @40);
	NSDictionary *citation = [[result outputForKey:NFKOutputCitations] firstObject];
	XCTAssertEqualObjects(citation[@"url"], @"https://meteo.example/paris");
	XCTAssertEqualObjects(citation[@"title"], @"Météo Paris");
	NSArray<NSDictionary *> *ran = [result outputForKey:NFKOutputServerToolResults];
	XCTAssertEqualObjects(ran[0][@"name"], @"web_search");
	XCTAssertEqualObjects(ran[0][@"input"], (@{ @"query": @"paris weather" }));
	XCTAssertEqualObjects(ran[0][@"output"], (@{ @"results": @2 }));
	XCTAssertEqualObjects(ran[1][@"name"], @"agent.handoff");
	XCTAssertEqualObjects(ran[1][@"output"], @"weather");
}

// A function call ends the turn; the next request answers it, replaying the turn's own entries.
- (void)testAFunctionCallIsReturnedAndItsAnswerGoesBackAsAFunctionResult
{
	[self.backend stage:@"{\"conversation_id\":\"conv_3\",\"outputs\":[{\"type\":\"function.call\",\"id\":\"fc_1\",\"tool_call_id\":\"call_9\",\"name\":\"get_weather\",\"arguments\":\"{\\\"city\\\":\\\"Paris\\\"}\"}]}"];
	NFKInferenceRequest *ask = [NFKInferenceRequest requestWithInputs:@{ NFKInputMessages: @[ @{ @"role": @"user", @"content": @"Weather?" } ] }];
	NFKInferenceResult *first = [self.backend runInferenceForRequest:ask error:NULL];
	NSDictionary *call = first.toolCalls.firstObject;
	XCTAssertEqualObjects(call[@"id"], @"call_9");
	XCTAssertEqualObjects(call[@"arguments"], (@{ @"city": @"Paris" }));
	XCTAssertEqualObjects([first outputForKey:NFKOutputFinishReason], NFKFinishReasonToolCalls);

	NSDictionary *assistant = [first outputForKey:NFKOutputAssistantMessage];
	NSDictionary *answer = [NFKToolMessages messageForToolCallIdentifier:@"call_9" name:@"get_weather" result:[NFKToolResult resultWithText:@"Sunny"]];
	[self.backend stage:[self textReply:@"It is sunny."]];
	NFKInferenceRequest *reply = [NFKInferenceRequest requestWithInputs:@{ NFKInputMessages: @[ @{ @"role": @"user", @"content": @"Weather?" }, assistant, answer ] }];
	XCTAssertEqualObjects([self.backend runInferenceForRequest:reply error:NULL].text, @"It is sunny.");
	NSArray *inputs = [self.backend bodyAt:1][@"inputs"];
	XCTAssertEqualObjects(inputs[1][@"type"], @"function.call", @"the turn replays its own entry");
	XCTAssertEqualObjects(inputs[1][@"tool_call_id"], @"call_9");
	XCTAssertEqualObjects(inputs[2], (@{ @"type": @"function.result", @"tool_call_id": @"call_9", @"result": @"Sunny" }));
}

- (void)testAnImageTheToolWroteIsDownloadedFromTheFilesAPI
{
	CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
	CGContextRef context = CGBitmapContextCreate(NULL, 2, 2, 8, 8, colorSpace, kCGBitmapByteOrder32Little | kCGImageAlphaPremultipliedFirst);
	CGColorSpaceRelease(colorSpace);
	CGImageRef image = CGBitmapContextCreateImage(context);
	CGContextRelease(context);
	NSData *png = [NFKImageCoding PNGDataForImage:(__bridge id)image];
	CGImageRelease(image);
	[self.backend stage:(@"{\"conversation_id\":\"conv_4\",\"outputs\":[{\"type\":\"tool.execution\",\"name\":\"image_generation\",\"arguments\":\"{}\"},"
						 "{\"type\":\"message.output\",\"content\":[{\"type\":\"text\",\"text\":\"Here it is.\"},"
						 "{\"type\":\"tool_file\",\"tool\":\"image_generation\",\"file_id\":\"file_img\",\"file_name\":\"cat.png\",\"file_type\":\"png\"}]}]}")];
	[self.backend.staged addObject:png];
	NFKInferenceResult *result = [self.backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Draw a cat." }] error:NULL];
	XCTAssertNotNil([result outputForKey:NFKOutputImage]);
	XCTAssertEqualObjects(self.backend.requests[1].URL.absoluteString, @"https://api.mistral.ai/v1/files/file_img/content");
	XCTAssertEqualObjects([[result outputForKey:NFKOutputServerToolResults] lastObject][@"output"], @"file_img");

	self.backend.downloadsImages = NO;
	[self.backend stage:@"{\"outputs\":[{\"type\":\"message.output\",\"content\":[{\"type\":\"tool_file\",\"tool\":\"image_generation\",\"file_id\":\"f\"}]}]}"];
	NSUInteger before = self.backend.requests.count;
	XCTAssertNil([[self.backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Again." }] error:NULL] outputForKey:NFKOutputImage]);
	XCTAssertEqual(self.backend.requests.count, before + 1);
}

#pragma mark The stream

- (void)testTheStreamGathersTextAndAFunctionCallInPieces
{
	self.backend.stagedLines = @[
		@"data: {\"type\":\"conversation.response.started\",\"conversation_id\":\"conv_5\"}",
		@"data: {\"type\":\"message.output.delta\",\"id\":\"m\",\"content\":\"Let me \"}",
		@"data: {\"type\":\"message.output.delta\",\"id\":\"m\",\"content\":\"check.\"}",
		@"data: {\"type\":\"function.call.delta\",\"id\":\"f\",\"tool_call_id\":\"call_1\",\"name\":\"get_weather\",\"arguments\":\"{\\\"city\\\":\"}",
		@"data: {\"type\":\"function.call.delta\",\"id\":\"f\",\"tool_call_id\":\"call_1\",\"name\":\"get_weather\",\"arguments\":\"\\\"Paris\\\"}\"}",
		@"data: {\"type\":\"conversation.response.done\",\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":4}}" ];
	NFKInferenceJob *job = [self.backend submitInferenceJobForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Weather?" }]];
	XCTAssertEqual(job.status, NFKInferenceJobStatusSucceeded, @"%@", job.error);
	XCTAssertEqualObjects(job.result.text, @"Let me check.");
	XCTAssertEqualObjects(job.result.toolCalls.firstObject[@"arguments"], (@{ @"city": @"Paris" }));
	XCTAssertEqualObjects([job.result outputForKey:NFKOutputResponseIdentifier], @"conv_5");
	XCTAssertEqualObjects([self.backend bodyAt:0][@"stream"], @YES);
}

- (void)testAStreamErrorFailsTheJob
{
	self.backend.stagedLines = @[ @"data: {\"type\":\"conversation.response.error\",\"message\":\"model overloaded\",\"code\":3000}" ];
	NFKInferenceJob *job = [self.backend submitInferenceJobForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Hi" }]];
	XCTAssertEqual(job.status, NFKInferenceJobStatusFailed);
	XCTAssertTrue([job.error.localizedDescription containsString:@"model overloaded"]);
}

#pragma mark The conversation, the agent, and the factory

- (void)testHistoryAndDeleteAddressTheConversation
{
	[self.backend stage:@"{\"object\":\"conversation.history\",\"conversation_id\":\"conv_1\",\"entries\":[{\"type\":\"message.input\"},{\"type\":\"message.output\"}]}"];
	XCTAssertEqual([self.backend historyOfConversationWithIdentifier:@"conv_1" error:NULL].count, 2u);
	XCTAssertEqualObjects(self.backend.requests[0].URL.absoluteString, @"https://api.mistral.ai/v1/conversations/conv_1/history");
	XCTAssertTrue([self.backend deleteConversationWithIdentifier:@"conv_1" error:NULL]);
	XCTAssertEqualObjects(self.backend.requests[1].HTTPMethod, @"DELETE");
	XCTAssertEqualObjects(self.backend.requests[1].URL.absoluteString, @"https://api.mistral.ai/v1/conversations/conv_1");
}

- (void)testAnAgentRunsInsteadOfAModelAndTheFactoryServesMistralAlone
{
	self.backend.modelName = nil;
	XCTAssertFalse(self.backend.isReady);
	self.backend.agentIdentifier = @"ag_weather";
	XCTAssertTrue(self.backend.isReady);
	[self.backend stage:[self textReply:@"Hi."]];
	[self.backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Hi" }] error:NULL];
	XCTAssertEqualObjects([self.backend bodyAt:0][@"agent_id"], @"ag_weather");
	XCTAssertNil([self.backend bodyAt:0][@"model"]);

	NFKMistralConversationsBackend *made = [NFKMistralConversationsBackend backendForProvider:NFKRemoteProvider.mistral apiKey:@"k" modelName:@"m"];
	XCTAssertEqualObjects(made.endpointURL.absoluteString, @"https://api.mistral.ai/v1/conversations");
	XCTAssertEqualObjects(made.backendIdentifier, @"mistral-conversations");
	XCTAssertNil([NFKMistralConversationsBackend backendForProvider:NFKRemoteProvider.openAI apiKey:@"k" modelName:@"m"]);
}

#pragma mark Live

// Runs against Mistral's Conversations API when a key is present. POST /v1/conversations answers 401
// without a key and /v1/conversationsx 404, so the path is verified to exist.
- (void)testALiveConversationAnswers
{
	NSString *key = NSProcessInfo.processInfo.environment[@"INFERKIT_MISTRAL_API_KEY"];
	if (key.length == 0) {
		XCTSkip("set INFERKIT_MISTRAL_API_KEY to exercise the live path");
	}
	NFKMistralConversationsBackend *backend = [NFKMistralConversationsBackend backendForProvider:NFKRemoteProvider.mistral apiKey:key modelName:@"mistral-small-latest"];
	NSError *error = nil;
	NFKInferenceResult *result = [backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Reply with the word ready." }
																						 parameters:@{ @"store": @NO } outputModality:NFKModalityText]
														   error:&error];
	XCTAssertGreaterThan(result.text.length, 0u, @"%@", error);
}

@end
