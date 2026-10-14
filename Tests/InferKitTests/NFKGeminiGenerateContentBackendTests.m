//
//  NFKGeminiGenerateContentBackendTests.m
//  InferKitTests
//
//  Gemini's generateContent API through stubbed transports: the contents and parts a request
//  becomes, the generation config, the reply's outputs, a tool conversation's replay, refusals, the
//  stream, and the factory routes. A live test runs against Google when a key is set.
//

#import <XCTest/XCTest.h>
#import <InferKit/NFKGeminiGenerateContentBackend.h>
#import <InferKit/NFKRemoteProvider.h>
#import <InferKit/NFKRemoteBackend.h>
#import <InferKit/NFKImageCoding.h>
#import <InferKit/NFKInferenceRequest.h>
#import <InferKit/NFKInferenceResult.h>
#import <InferKit/NFKInferenceJob.h>
#import <InferKit/NFKInferenceKeys.h>
#import <InferKit/NFKInferenceTool.h>
#import <InferKit/NFKToolMessages.h>
#import <InferKit/NFKErrors.h>

@interface NFKStubGenerateContentBackend : NFKGeminiGenerateContentBackend
@property (nonatomic, strong) NSMutableArray<NSURLRequest *> *requests;
@property (nonatomic, strong) NSMutableArray<NSData *> *stagedReplies;
@property (nonatomic, copy) NSArray<NSString *> *stagedLines;
@property (nonatomic, strong) NSMutableArray<NSArray<NSString *> *> *stagedStreams;
@property (nonatomic, assign) BOOL holdsStreamsOpen;
@property (nonatomic, strong) NSMutableArray *heldCompletions;
@property (nonatomic, copy, nullable) void (^streamStarted)(NSUInteger index);
@property (nonatomic, strong) NSMutableIndexSet *cancelledStreams;
@end

@implementation NFKStubGenerateContentBackend
- (NSData *)sendRequest:(NSURLRequest *)request response:(NSHTTPURLResponse **)outResponse error:(NSError **)outError
{
	if (self.requests == nil) {
		self.requests = [NSMutableArray array];
	}
	[self.requests addObject:request];
	if (outResponse != NULL) {
		*outResponse = [[NSHTTPURLResponse alloc] initWithURL:request.URL statusCode:200 HTTPVersion:@"HTTP/1.1" headerFields:nil];
	}
	NSData *reply = self.stagedReplies.firstObject ?: [@"{}" dataUsingEncoding:NSUTF8StringEncoding];
	if (self.stagedReplies.count > 1) {
		[self.stagedReplies removeObjectAtIndex:0];
	}
	return reply;
}
- (void (^)(void))streamRequest:(NSURLRequest *)request
					lineHandler:(void (^)(NSString *))lineHandler
			  completionHandler:(void (^)(NSHTTPURLResponse *, NSData *, NSError *))completionHandler
{
	if (self.requests == nil) {
		self.requests = [NSMutableArray array];
	}
	[self.requests addObject:request];
	NSUInteger index = self.requests.count - 1;
	if (self.streamStarted != nil) {
		self.streamStarted(index);
	}
	NSArray<NSString *> *lines = self.stagedLines;
	if (self.stagedStreams.count > 0) {
		lines = self.stagedStreams.firstObject;
		[self.stagedStreams removeObjectAtIndex:0];
	}
	for (NSString *line in lines) {
		lineHandler(line);
	}
	NSHTTPURLResponse *response = [[NSHTTPURLResponse alloc] initWithURL:request.URL statusCode:200 HTTPVersion:@"HTTP/1.1" headerFields:nil];
	if (self.holdsStreamsOpen) {
		if (self.heldCompletions == nil) {
			self.heldCompletions = [NSMutableArray array];
		}
		[self.heldCompletions addObject:^{ completionHandler(response, nil, nil); }];
	} else {
		completionHandler(response, nil, nil);
	}
	if (self.cancelledStreams == nil) {
		self.cancelledStreams = [NSMutableIndexSet indexSet];
	}
	NSMutableIndexSet *cancelled = self.cancelledStreams;
	return ^{ [cancelled addIndex:index]; };
}
- (void)stage:(NSString *)reply
{
	if (self.stagedReplies == nil) {
		self.stagedReplies = [NSMutableArray array];
	}
	[self.stagedReplies addObject:[reply dataUsingEncoding:NSUTF8StringEncoding]];
}
- (NSDictionary *)bodyAt:(NSUInteger)index
{
	return [NSJSONSerialization JSONObjectWithData:self.requests[index].HTTPBody options:0 error:NULL];
}
@end

@interface NFKGeminiGenerateContentBackendTests : XCTestCase
@property (nonatomic, strong) NFKStubGenerateContentBackend *backend;
@end

@implementation NFKGeminiGenerateContentBackendTests

- (void)setUp
{
	[super setUp];
	self.backend = [NFKStubGenerateContentBackend backendWithEndpointURL:nil];
	self.backend.apiKey = @"AIza";
	self.backend.modelName = @"gemini-3.8-flash";
}

#pragma mark The request

- (void)testAPromptIsOneUserTurnPostedToTheModelsGenerateContentRoute
{
	[self.backend stage:@"{\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"Hi.\"}]},\"finishReason\":\"STOP\"}]}"];
	NSError *error = nil;
	NFKInferenceResult *result = [self.backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Hello" }] error:&error];
	XCTAssertEqualObjects(result.text, @"Hi.", @"%@", error);
	NSURLRequest *request = self.backend.requests.firstObject;
	XCTAssertEqualObjects(request.URL.absoluteString,
						  @"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent");
	XCTAssertEqualObjects([request valueForHTTPHeaderField:@"x-goog-api-key"], @"AIza");
	XCTAssertNil([request valueForHTTPHeaderField:@"Authorization"]);
	NSArray *expected = @[ @{ @"role": @"user", @"parts": @[ @{ @"text": @"Hello" } ] } ];
	XCTAssertEqualObjects([self.backend bodyAt:0][@"contents"], expected);
}

- (void)testAModelNameWrittenWithItsCollectionIsNotDoubled
{
	self.backend.modelName = @"models/gemini-3.8-flash";
	[self.backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Hello" }] error:NULL];
	XCTAssertEqualObjects(self.backend.requests.firstObject.URL.path, @"/v1beta/models/gemini-3.8-flash:generateContent");
}

- (void)testTheContractParametersReachGenerationConfig
{
	NSDictionary *schema = @{ @"type": @"object", @"properties": @{ @"city": @{ @"type": @"string" } } };
	NSDictionary *parameters = @{ NFKParameterMaxTokens: @256, NFKParameterTemperature: @0.4, NFKParameterTopP: @0.9,
								  NFKParameterTopK: @40, NFKParameterSeed: @7, NFKParameterStopSequences: @[ @"END" ],
								  NFKParameterReasoningEffort: NFKReasoningEffortDeep, NFKParameterJSONSchema: schema,
								  NFKParameterRepetitionPenalty: @1.1, @"candidateCount": @1,
								  @"safetySettings": @[ @{ @"category": @"HARM_CATEGORY_HARASSMENT", @"threshold": @"BLOCK_NONE" } ] };
	[self.backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Where?" } parameters:parameters] error:NULL];
	NSDictionary *body = [self.backend bodyAt:0];
	NSDictionary *generation = body[@"generationConfig"];
	XCTAssertEqualObjects(generation[@"maxOutputTokens"], @256);
	XCTAssertEqualObjects(generation[@"temperature"], @0.4);
	XCTAssertEqualObjects(generation[@"topP"], @0.9);
	XCTAssertEqualObjects(generation[@"topK"], @40);
	XCTAssertEqualObjects(generation[@"seed"], @7);
	XCTAssertEqualObjects(generation[@"stopSequences"], (@[ @"END" ]));
	XCTAssertEqualObjects(generation[@"thinkingConfig"], (@{ @"thinkingLevel": @"HIGH", @"includeThoughts": @YES }));
	XCTAssertEqualObjects(generation[@"responseFormat"], (@{ @"text": @{ @"mimeType": @"APPLICATION_JSON", @"schema": schema } }));
	XCTAssertEqualObjects(generation[@"candidateCount"], @1, @"an unnamed parameter goes into generationConfig");
	XCTAssertNil(generation[NFKParameterRepetitionPenalty], @"a contract key without a field here is dropped");
	XCTAssertNotNil(body[@"safetySettings"], @"safetySettings belongs to the request's top level");
	XCTAssertNil(generation[@"safetySettings"]);
}

- (void)testANumericEffortIsAThinkingBudget
{
	[self.backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Think." }
																	  parameters:@{ NFKParameterReasoningEffort: @"4096" }] error:NULL];
	XCTAssertEqualObjects([self.backend bodyAt:0][@"generationConfig"][@"thinkingConfig"],
						  (@{ @"thinkingBudget": @4096, @"includeThoughts": @YES }));
}

- (void)testToolsBecomeFunctionDeclarationsAndAWireToolPassesThrough
{
	NSDictionary *weather = @{ @"name": @"weather", @"description": @"The weather in a city.",
							   @"parameters": @{ @"type": @"object", @"properties": @{ @"city": @{ @"type": @"string" } } } };
	NSArray *tools = @[ weather, @{ @"googleSearch": @{} } ];
	[self.backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Oslo?" }
																	  parameters:@{ NFKParameterTools: tools }] error:NULL];
	NSArray *wire = [self.backend bodyAt:0][@"tools"];
	XCTAssertEqual(wire.count, 2u);
	NSDictionary *declaration = [wire[0][@"functionDeclarations"] firstObject];
	XCTAssertEqualObjects(declaration[@"name"], @"weather");
	XCTAssertEqualObjects(declaration[@"parametersJsonSchema"], weather[@"parameters"]);
	XCTAssertEqualObjects(wire[1], (@{ @"googleSearch": @{} }));
}

- (void)testASystemTurnIsTheInstructionAndMediaRideOnTheLastUserTurn
{
	CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
	CGContextRef context = CGBitmapContextCreate(NULL, 4, 4, 8, 16, colorSpace, kCGBitmapByteOrder32Little | kCGImageAlphaPremultipliedFirst);
	CGColorSpaceRelease(colorSpace);
	CGImageRef square = CGBitmapContextCreateImage(context);
	CGContextRelease(context);
	NSArray *messages = @[ @{ @"role": @"system", @"content": @"Answer briefly." },
						   @{ @"role": @"user", @"content": @"Hello" },
						   @{ @"role": @"assistant", @"content": @"Hi." },
						   @{ @"role": @"user", @"content": @"What is this?" } ];
	[self.backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputMessages: messages, NFKInputImage: (__bridge id)square }] error:NULL];
	CGImageRelease(square);
	NSDictionary *body = [self.backend bodyAt:0];
	XCTAssertEqualObjects(body[@"systemInstruction"], (@{ @"parts": @[ @{ @"text": @"Answer briefly." } ] }));
	NSArray *contents = body[@"contents"];
	XCTAssertEqual(contents.count, 3u);
	XCTAssertEqualObjects(contents[1], (@{ @"role": @"model", @"parts": @[ @{ @"text": @"Hi." } ] }));
	NSArray *last = contents[2][@"parts"];
	XCTAssertEqualObjects(last[0], (@{ @"text": @"What is this?" }));
	XCTAssertEqualObjects(last[1][@"inlineData"][@"mimeType"], @"image/png");
	XCTAssertGreaterThan([last[1][@"inlineData"][@"data"] length], 0u);
}

- (void)testARequestWithoutAModelOrInputFailsBeforeSending
{
	NSError *error = nil;
	self.backend.modelName = nil;
	XCTAssertFalse(self.backend.isReady);
	XCTAssertNil([self.backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Hello" }] error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceNotReady);
	self.backend.modelName = @"gemini-3.8-flash";
	XCTAssertNil([self.backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{}] error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceMissingInput);
	XCTAssertEqual(self.backend.requests.count, 0u);
}

#pragma mark The reply

- (void)testTheReplyCarriesTextThoughtsCallsStructureAndUsage
{
	[self.backend stage:@"{\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":["
	 "{\"text\":\"Checking the city.\",\"thought\":true},"
	 "{\"text\":\"{\\\"city\\\":\\\"Oslo\\\"}\",\"thoughtSignature\":\"c2ln\"},"
	 "{\"functionCall\":{\"name\":\"weather\",\"args\":{\"city\":\"Oslo\"}}}]},\"finishReason\":\"STOP\"}],"
	 "\"usageMetadata\":{\"promptTokenCount\":10,\"cachedContentTokenCount\":2,\"candidatesTokenCount\":5,\"thoughtsTokenCount\":7}}"];
	NSDictionary *schema = @{ @"type": @"object" };
	NSError *error = nil;
	NFKInferenceResult *result = [self.backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Where?" }
																								 parameters:@{ NFKParameterJSONSchema: schema }] error:&error];
	XCTAssertNotNil(result, @"%@", error);
	XCTAssertEqualObjects(result.text, @"{\"city\":\"Oslo\"}");
	XCTAssertEqualObjects([result outputForKey:NFKOutputStructured], (@{ @"city": @"Oslo" }));
	XCTAssertEqualObjects([result outputForKey:NFKOutputReasoning], @"Checking the city.");
	NSDictionary *call = result.toolCalls.firstObject;
	XCTAssertEqualObjects(call[@"name"], @"weather");
	XCTAssertEqualObjects(call[@"arguments"], (@{ @"city": @"Oslo" }));
	XCTAssertGreaterThan([call[@"id"] length], 0u, @"a call without an id is given one");
	NSDictionary *usage = [result outputForKey:NFKOutputUsage];
	XCTAssertEqualObjects(usage[NFKUsageInputTokens], @10);
	XCTAssertEqualObjects(usage[NFKUsageCachedTokens], @2);
	XCTAssertEqualObjects(usage[NFKUsageOutputTokens], @12, @"the reply's tokens include the thinking ones");
	XCTAssertEqualObjects(usage[NFKUsageReasoningTokens], @7);
	XCTAssertNotNil([result outputForKey:NFKRemoteBackendRawKey]);
}

// The model's parts go back exactly, signatures included; the answers to its calls share one user
// turn, named by the call, and a call the service gave no id is answered without one.
- (void)testAToolConversationReplaysItsOwnPartsAndAnswersInOneUserTurn
{
	[self.backend stage:@"{\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":["
	 "{\"functionCall\":{\"name\":\"weather\",\"args\":{\"city\":\"Oslo\"}},\"thoughtSignature\":\"c2lnMQ==\"},"
	 "{\"functionCall\":{\"id\":\"call-7\",\"name\":\"time\",\"args\":{}}}]},\"finishReason\":\"STOP\"}]}"];
	[self.backend stage:@"{\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"Sunny, noon.\"}]},\"finishReason\":\"STOP\"}]}"];
	NSDictionary *user = @{ @"role": @"user", @"content": @"Weather and time in Oslo?" };
	NFKInferenceResult *first = [self.backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputMessages: @[ user ] }] error:NULL];
	NSArray<NSDictionary *> *calls = first.toolCalls;
	XCTAssertEqual(calls.count, 2u);
	NSDictionary *assistant = [first outputForKey:NFKOutputAssistantMessage];
	NSArray *conversation = @[ user, assistant,
							   [NFKToolMessages messageForToolCallIdentifier:calls[0][@"id"] name:@"weather" result:[NFKToolResult resultWithText:@"Sunny"]],
							   [NFKToolMessages messageForToolCallIdentifier:calls[1][@"id"] name:@"time" result:[NFKToolResult resultWithText:@"12:00"]] ];
	NFKInferenceResult *second = [self.backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputMessages: conversation }] error:NULL];
	XCTAssertEqualObjects(second.text, @"Sunny, noon.");
	NSArray *contents = [self.backend bodyAt:1][@"contents"];
	XCTAssertEqual(contents.count, 3u);
	XCTAssertEqualObjects(contents[1][@"role"], @"model");
	XCTAssertEqualObjects(contents[1][@"parts"][0][@"thoughtSignature"], @"c2lnMQ==");
	NSArray *answers = contents[2][@"parts"];
	XCTAssertEqualObjects(contents[2][@"role"], @"user");
	XCTAssertEqual(answers.count, 2u);
	XCTAssertEqualObjects(answers[0], (@{ @"functionResponse": @{ @"name": @"weather", @"response": @{ @"output": @"Sunny" } } }));
	XCTAssertEqualObjects(answers[1], (@{ @"functionResponse": @{ @"id": @"call-7", @"name": @"time", @"response": @{ @"output": @"12:00" } } }));
}

- (void)testAnotherBackendsAssistantTurnIsRebuiltFromItsTextAndCalls
{
	NSDictionary *assistant = [NFKToolMessages assistantMessageWithText:@"Looking." toolCalls:@[ @{ @"id": @"call_1", @"name": @"weather", @"arguments": @{ @"city": @"Oslo" } } ]
																  record:nil format:nil source:nil];
	NSArray *conversation = @[ @{ @"role": @"user", @"content": @"Oslo?" }, assistant ];
	[self.backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputMessages: conversation }] error:NULL];
	NSArray *expected = @[ @{ @"text": @"Looking." }, @{ @"functionCall": @{ @"id": @"call_1", @"name": @"weather", @"args": @{ @"city": @"Oslo" } } } ];
	XCTAssertEqualObjects([self.backend bodyAt:0][@"contents"][1][@"parts"], expected);
}

- (void)testABlockedPromptAndAWithheldReplyAreRefusals
{
	[self.backend stage:@"{\"promptFeedback\":{\"blockReason\":\"SAFETY\"}}"];
	[self.backend stage:@"{\"candidates\":[{\"finishReason\":\"RECITATION\",\"finishMessage\":\"recited\"}]}"];
	NSError *error = nil;
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Hello" }];
	XCTAssertNil([self.backend runInferenceForRequest:request error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceRefused);
	XCTAssertTrue([error.localizedDescription containsString:@"SAFETY"]);
	XCTAssertNil([self.backend runInferenceForRequest:request error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceRefused);
	XCTAssertTrue([error.localizedDescription containsString:@"recited"]);
}

- (void)testGroundingSupportsAndRecitationSourcesBecomeCitations
{
	[self.backend stage:@"{\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Oslo is in Norway.\"}]},"
	 "\"groundingMetadata\":{\"groundingChunks\":[{\"web\":{\"uri\":\"https://example.test/oslo\",\"title\":\"Oslo\"}}],"
	 "\"groundingSupports\":[{\"segment\":{\"startIndex\":0,\"endIndex\":18,\"text\":\"Oslo is in Norway.\"},\"groundingChunkIndices\":[0]}]},"
	 "\"citationMetadata\":{\"citationSources\":[{\"startIndex\":0,\"endIndex\":4,\"uri\":\"https://example.test/source\"}]}}]}"];
	NFKInferenceResult *result = [self.backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Oslo?" }] error:NULL];
	NSArray<NSDictionary *> *citations = [result outputForKey:NFKOutputCitations];
	XCTAssertEqual(citations.count, 2u);
	XCTAssertEqualObjects(citations[0][@"url"], @"https://example.test/oslo");
	XCTAssertEqualObjects(citations[0][@"title"], @"Oslo");
	XCTAssertEqualObjects(citations[0][@"text"], @"Oslo is in Norway.");
	XCTAssertEqualObjects(citations[1][@"url"], @"https://example.test/source");
	XCTAssertEqualObjects(citations[1][@"text"], @"Oslo", @"a recitation span is measured in bytes of the reply");
}

#pragma mark The stream

- (void)testTheStreamGrowsTheTextAndKeepsTheSignatureOnItsPart
{
	self.backend.stagedLines = @[ @"data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"Thinking.\",\"thought\":true}]}}]}",
								  @"data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"Hel\"}]}}]}",
								  @"data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"lo\"}]}}]}",
								  (@"data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"\",\"thoughtSignature\":\"c2ln\"}]},\"finishReason\":\"STOP\"}],"
								   "\"usageMetadata\":{\"promptTokenCount\":3,\"candidatesTokenCount\":2}}") ];
	NFKInferenceJob *job = [self.backend submitInferenceJobForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"hi" }]];
	XCTAssertEqual(job.status, NFKInferenceJobStatusSucceeded, @"%@", job.error);
	XCTAssertEqualObjects(job.result.text, @"Hello");
	XCTAssertEqualObjects([job.result outputForKey:NFKOutputReasoning], @"Thinking.");
	XCTAssertEqualObjects([job.result outputForKey:NFKOutputUsage][NFKUsageInputTokens], @3);
	XCTAssertEqualObjects(self.backend.requests.firstObject.URL.absoluteString,
						  @"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:streamGenerateContent?alt=sse");
	NSArray *parts = [[NFKToolMessages recordOfMessage:[job.result outputForKey:NFKOutputAssistantMessage]
												format:@"gemini-generatecontent"
												source:self.backend.endpointURL.absoluteString] copy];
	NSArray *expected = @[ @{ @"text": @"Thinking.", @"thought": @YES }, @{ @"text": @"Hello", @"thoughtSignature": @"c2ln" } ];
	XCTAssertEqualObjects(parts, expected);
}

// gemini-4-argon's long decode: a hop that ends in CONTINUATION is resent, unchanged, with its token.
- (void)testAContinuedReplyIsResentWithItsTokenAndTheHopsJoin
{
	self.backend.modelName = @"gemini-4-argon";
	[self.backend stage:@"{\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"Chapter one. \"}]},\"finishReason\":\"CONTINUATION\"}],"
	 "\"continuationToken\":\"tok1\"}"];
	[self.backend stage:@"{\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"Chapter two.\"}]},\"finishReason\":\"STOP\"}],"
	 "\"usageMetadata\":{\"promptTokenCount\":5,\"candidatesTokenCount\":9}}"];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Write it." } parameters:@{ NFKParameterMaxTokens: @100000 }];
	NSError *error = nil;
	NFKInferenceResult *result = [self.backend runInferenceForRequest:request error:&error];
	XCTAssertNotNil(result, @"%@", error);
	XCTAssertEqual(self.backend.requests.count, 2u);
	NSMutableDictionary *second = [[self.backend bodyAt:1] mutableCopy];
	XCTAssertEqualObjects(second[@"continuationToken"], @"tok1");
	[second removeObjectForKey:@"continuationToken"];
	XCTAssertEqualObjects(second, [self.backend bodyAt:0]);
	XCTAssertEqualObjects(result.text, @"Chapter one. Chapter two.");
	XCTAssertEqualObjects([result outputForKey:NFKOutputUsage][NFKUsageInputTokens], @5);
}

- (void)testAReplyCutAtTheCallersLimitIsNotContinued
{
	[self.backend stage:@"{\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Cut\"}]},\"finishReason\":\"MAX_TOKENS\"}],\"continuationToken\":\"tok\"}"];
	NFKInferenceResult *result = [self.backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"hi" }] error:NULL];
	XCTAssertEqual(self.backend.requests.count, 1u);
	XCTAssertEqualObjects(result.text, @"Cut");
}

- (void)testAContinuedStreamOpensTheNextHopAndKeepsGrowingTheText
{
	self.backend.stagedStreams = [NSMutableArray arrayWithArray:@[
		@[ @"data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"Hel\"}]},\"finishReason\":\"CONTINUATION\"}],\"continuationToken\":\"t1\"}" ],
		@[ @"data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"lo\"}]},\"finishReason\":\"STOP\"}]}" ] ]];
	NFKInferenceJob *job = [self.backend submitInferenceJobForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"hi" }]];
	XCTAssertEqual(job.status, NFKInferenceJobStatusSucceeded, @"%@", job.error);
	XCTAssertEqualObjects(job.result.text, @"Hello");
	XCTAssertEqual(self.backend.requests.count, 2u);
	XCTAssertEqualObjects([self.backend bodyAt:1][@"continuationToken"], @"t1");
}

- (void)testAContinuationWithoutATokenFails
{
	[self.backend stage:@"{\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Cut\"}]},\"finishReason\":\"CONTINUATION\"}]}"];
	NSError *error = nil;
	XCTAssertNil([self.backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"hi" }] error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceBackendFailure);
	XCTAssertEqual(self.backend.requests.count, 1u);
}

- (void)testAnErrorObjectInAContinuedHopFailsTheCall
{
	[self.backend stage:@"{\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Chapter one.\"}]},\"finishReason\":\"CONTINUATION\"}],\"continuationToken\":\"t1\"}"];
	[self.backend stage:@"{\"error\":{\"code\":500,\"message\":\"hop lost\",\"status\":\"INTERNAL\"}}"];
	NSError *error = nil;
	XCTAssertNil([self.backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"hi" }] error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceBackendFailure);
	XCTAssertTrue([error.localizedDescription containsString:@"hop lost"], @"%@", error);
}

- (void)testARepeatedContinuationTokenFails
{
	// The stub answers every request after the last staged reply with that reply, the same token each time.
	[self.backend stage:@"{\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Again\"}]},\"finishReason\":\"CONTINUATION\"}],\"continuationToken\":\"same\"}"];
	NSError *error = nil;
	XCTAssertNil([self.backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"hi" }] error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceBackendFailure);
	XCTAssertEqual(self.backend.requests.count, 2u);
}

- (void)testAReplyStillContinuingAfterSixtyFourHopsFails
{
	for (NSUInteger hop = 0; hop <= 64; hop++) {
		[self.backend stage:[NSString stringWithFormat:@"{\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"x\"}]},\"finishReason\":\"CONTINUATION\"}],"
							 "\"continuationToken\":\"t%lu\"}", (unsigned long)hop]];
	}
	NSError *error = nil;
	XCTAssertNil([self.backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"hi" }] error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceBackendFailure);
	XCTAssertEqual(self.backend.requests.count, 65u);
}

- (void)testAStreamThatRepeatsItsContinuationTokenFailsTheJob
{
	NSString *line = @"data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"x\"}]},\"finishReason\":\"CONTINUATION\"}],\"continuationToken\":\"t1\"}";
	self.backend.stagedLines = @[ line ];
	NFKInferenceJob *job = [self.backend submitInferenceJobForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"hi" }]];
	XCTAssertEqual(job.status, NFKInferenceJobStatusFailed);
	XCTAssertEqual(job.error.code, kNFKError_InferenceBackendFailure);
	XCTAssertEqual(self.backend.requests.count, 2u);
}

- (void)testACancelDuringTheHandoverStopsTheNextHop
{
	self.backend.holdsStreamsOpen = YES;
	self.backend.stagedStreams = [NSMutableArray arrayWithArray:@[
		@[ @"data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hel\"}]},\"finishReason\":\"CONTINUATION\"}],\"continuationToken\":\"t1\"}" ],
		@[] ]];
	NFKInferenceJob *job = [self.backend submitInferenceJobForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"hi" }]];
	__weak NFKInferenceJob *weakJob = job;
	self.backend.streamStarted = ^(NSUInteger index) {
		if (index == 1) {
			[weakJob cancel];
		}
	};
	void (^finishFirstHop)(void) = self.backend.heldCompletions.firstObject;
	finishFirstHop();
	XCTAssertEqual(job.status, NFKInferenceJobStatusCancelled);
	XCTAssertTrue([self.backend.cancelledStreams containsIndex:1]);
}

- (void)testAnErrorInTheStreamFailsTheJob
{
	self.backend.stagedLines = @[ @"data: {\"error\":{\"code\":500,\"message\":\"internal\",\"status\":\"INTERNAL\"}}" ];
	NFKInferenceJob *job = [self.backend submitInferenceJobForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"hi" }]];
	XCTAssertEqual(job.status, NFKInferenceJobStatusFailed);
	XCTAssertTrue([job.error.localizedDescription containsString:@"internal"]);
}

#pragma mark The factory

- (void)testTheFactoryReachesGeminiAndTheGatewaysThatRouteToIt
{
	NFKGeminiGenerateContentBackend *google = [NFKGeminiGenerateContentBackend backendForProvider:NFKRemoteProvider.googleGemini apiKey:@"k" modelName:@"gemini-3.8-flash"];
	XCTAssertEqualObjects(google.endpointURL.absoluteString, @"https://generativelanguage.googleapis.com/v1beta/models");
	XCTAssertEqualObjects(google.apiKey, @"k");
	NFKGeminiGenerateContentBackend *zen = [NFKGeminiGenerateContentBackend backendForProvider:NFKRemoteProvider.openCodeZen apiKey:@"k" modelName:@"gemini-3.5-flash"];
	XCTAssertEqualObjects(zen.endpointURL.absoluteString, @"https://opencode.ai/inference/v1/models");
	XCTAssertNil([NFKGeminiGenerateContentBackend backendForProvider:NFKRemoteProvider.openAI apiKey:@"k" modelName:@"m"]);
	XCTAssertNil([NFKGeminiGenerateContentBackend backendForProvider:NFKRemoteProvider.openCodeGo apiKey:@"k" modelName:@"m"],
				 @"Go routes no model to generateContent");
}

#pragma mark Live

// Runs only with INFERKIT_GEMINI_API_KEY set; INFERKIT_GEMINI_MODEL names the model (gemini-3.8-flash
// otherwise).
- (void)testALiveGenerateContentCallAnswers
{
	NSDictionary *environment = NSProcessInfo.processInfo.environment;
	NSString *key = environment[@"INFERKIT_GEMINI_API_KEY"];
	if (key.length == 0) {
		XCTSkip("set INFERKIT_GEMINI_API_KEY to exercise the live path");
	}
	NFKGeminiGenerateContentBackend *backend = [NFKGeminiGenerateContentBackend backendForProvider:NFKRemoteProvider.googleGemini apiKey:key
																						 modelName:environment[@"INFERKIT_GEMINI_MODEL"] ?: @"gemini-3.8-flash"];
	NSError *error = nil;
	NFKInferenceResult *result = [backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Reply with exactly: OK" }
																							  parameters:@{ NFKParameterMaxTokens: @256 }] error:&error];
	XCTAssertNotNil(result, @"%@", error.localizedDescription);
	XCTAssertGreaterThan(result.text.length, 0u);
}

@end
