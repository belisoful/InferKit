//
//  NFKRemoteResponsesBackendTests.m
//  InferKitTests
//
//  The Responses API through stubbed transports: the input items and the contract's parameters on
//  the wire, every output item kind read back, a background job polled to its end, and the stream.
//

#import <XCTest/XCTest.h>
#import <InferKit/NFKRemoteResponsesBackend.h>
#import <InferKit/NFKRemoteProvider.h>
#import <InferKit/NFKImageCoding.h>
#import <InferKit/NFKInferenceRequest.h>
#import <InferKit/NFKInferenceResult.h>
#import <InferKit/NFKInferenceJob.h>
#import <InferKit/NFKInferenceKeys.h>
#import <InferKit/NFKErrors.h>
#import <InferKit/NFKRemoteFileStore.h>
#import <InferKit/NFKModelDescriptor.h>
#import <InferKit/NFKVideoAsset.h>
#import "NFKTestClip.h"

@interface NFKStubResponsesBackend : NFKRemoteResponsesBackend
@property (nonatomic, strong) NSMutableArray<NSURLRequest *> *requests;
@property (nonatomic, strong) NSMutableArray<NSString *> *stagedBodies;
@property (nonatomic, copy) NSArray<NSString *> *stagedLines;
@end

@implementation NFKStubResponsesBackend
- (NSData *)sendRequest:(NSURLRequest *)request response:(NSHTTPURLResponse **)outResponse error:(NSError **)outError
{
	if (self.requests == nil) {
		self.requests = [NSMutableArray array];
	}
	[self.requests addObject:request];
	if (outResponse != NULL) {
		*outResponse = [[NSHTTPURLResponse alloc] initWithURL:request.URL statusCode:200 HTTPVersion:@"HTTP/1.1" headerFields:nil];
	}
	NSString *body = self.stagedBodies.firstObject ?: @"{}";
	if (self.stagedBodies.count > 1) {
		[self.stagedBodies removeObjectAtIndex:0];
	}
	return [body dataUsingEncoding:NSUTF8StringEncoding];
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

@interface NFKRemoteResponsesBackendTests : XCTestCase
@property (nonatomic, strong) NFKStubResponsesBackend *backend;
@end

@implementation NFKRemoteResponsesBackendTests

- (void)setUp
{
	[super setUp];
	self.backend = [NFKStubResponsesBackend backendWithEndpointURL:[NSURL URLWithString:@"https://api.openai.com/v1/responses"]];
	self.backend.modelName = @"gpt-5.6-sol";
	self.backend.pollInterval = 0;
}

- (void)testTheFinishReasonComesFromTheStatus
{
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Hello" }];
	self.backend.stagedBodies = [@[ @"{\"id\":\"r\",\"status\":\"completed\",\"output\":[]}" ] mutableCopy];
	XCTAssertEqualObjects([[self.backend runInferenceForRequest:request error:NULL] outputForKey:NFKOutputFinishReason], NFKFinishReasonStop);
	self.backend.stagedBodies = [@[ @"{\"id\":\"r\",\"status\":\"incomplete\",\"incomplete_details\":{\"reason\":\"max_output_tokens\"},\"output\":[]}" ] mutableCopy];
	XCTAssertEqualObjects([[self.backend runInferenceForRequest:request error:NULL] outputForKey:NFKOutputFinishReason], NFKFinishReasonLength);
	self.backend.stagedBodies = [@[ (@"{\"id\":\"r\",\"status\":\"completed\",\"output\":[{\"type\":\"function_call\",\"call_id\":\"c\","
									 "\"name\":\"get_weather\",\"arguments\":\"{}\"}]}") ] mutableCopy];
	XCTAssertEqualObjects([[self.backend runInferenceForRequest:request error:NULL] outputForKey:NFKOutputFinishReason], NFKFinishReasonToolCalls);
}

- (void)testTheFactoryReachesTheServicesThatServeResponses
{
	XCTAssertEqualObjects([NFKRemoteResponsesBackend backendForProvider:NFKRemoteProvider.xAI apiKey:@"k" modelName:@"m"].endpointURL.absoluteString,
						  @"https://api.x.ai/v1/responses");
	XCTAssertNotNil([NFKRemoteResponsesBackend backendForProvider:NFKRemoteProvider.deepSeek apiKey:@"k" modelName:@"m"]);
	XCTAssertNil([NFKRemoteResponsesBackend backendForProvider:NFKRemoteProvider.anthropic apiKey:@"k" modelName:@"m"]);
	XCTAssertNil([NFKRemoteResponsesBackend backendForProvider:NFKRemoteProvider.mistral apiKey:@"k" modelName:@"m"]);
	XCTAssertEqual([NFKRemoteResponsesBackend backendForProvider:NFKRemoteProvider.openAI apiKey:@"k" modelName:@"m"].dialect,
				   NFKRemoteResponsesDialectStandard);
}

- (void)testTheDescriptorOffersTheInputsAndParametersEachDialectReads
{
	id<NFKInferenceBackend> routed = [NFKRemoteProvider backendForProvider:NFKRemoteProvider.perplexity apiKey:nil modelName:@"sonar-pro"];
	XCTAssertTrue([(NSObject *)routed isKindOfClass:NFKRemoteResponsesBackend.class]);
	NFKModelDescriptor *agent = [NFKModelDescriptor descriptorWithIdentifier:@"remote:perplexity/sonar-pro" derivedFromBackend:routed];
	NSSet *expectedInputs = [NSSet setWithArray:@[ NFKInputPrompt, NFKInputMessages, NFKInputImage, NFKInputImages,
												   NFKInputVideo, NFKInputDocument, NFKInputDocuments ]];
	XCTAssertEqualObjects([NSSet setWithArray:[agent.inputs valueForKey:@"key"]], expectedInputs);
	NSSet *agentOptions = [NSSet setWithArray:[agent.options valueForKey:@"key"]];
	for (NSString *key in @[ NFKParameterTools, NFKParameterJSONSchema, NFKParameterReasoningEffort, NFKParameterMaxTokens,
							 NFKParameterTemperature, NFKParameterTopP, NFKParameterMCPServers, NFKParameterSkills,
							 NFKParameterPreviousResponseIdentifier, NFKParameterVideoFrameCount ]) {
		XCTAssertTrue([agentOptions containsObject:key], @"%@", key);
	}
	NFKModelDescriptor *standard = [NFKModelDescriptor descriptorWithIdentifier:@"m" derivedFromBackend:self.backend];
	XCTAssertEqualObjects([NSSet setWithArray:[standard.inputs valueForKey:@"key"]], expectedInputs);
	NSSet *standardOptions = [NSSet setWithArray:[standard.options valueForKey:@"key"]];
	XCTAssertTrue([standardOptions containsObject:NFKParameterMCPServers]);
	XCTAssertFalse([standardOptions containsObject:NFKParameterSkills]);
}

- (void)testAVideoGoesAsSampledFramesAndItsFrameCountStaysOffTheWire
{
	NSError *error = nil;
	NSURL *clip = [NFKTestClip writeClipWithColors:@[ @[ @1, @0, @0 ], @[ @0, @0, @1 ] ] error:&error];
	XCTAssertNotNil(clip, @"%@", error);
	self.backend.stagedBodies = [@[ @"{\"id\":\"r\",\"status\":\"completed\",\"output\":[]}" ] mutableCopy];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"What changes?",
																			  NFKInputVideo: [NFKVideoAsset videoAssetWithFileURL:clip] }
															   parameters:@{ NFKParameterVideoFrameCount: @2 }];
	XCTAssertNotNil([self.backend runInferenceForRequest:request error:&error], @"%@", error);
	NSDictionary *body = [self.backend bodyAt:0];
	NSArray *parts = body[@"input"][0][@"content"];
	XCTAssertEqualObjects([parts valueForKey:@"type"], (@[ @"input_text", @"input_image", @"input_image" ]));
	XCTAssertNil(body[NFKParameterVideoFrameCount]);
	[NSFileManager.defaultManager removeItemAtURL:clip error:NULL];
}

#pragma mark Perplexity's Agent API

- (NFKStubResponsesBackend *)perplexityAgent
{
	NFKStubResponsesBackend *agent = [NFKStubResponsesBackend backendWithEndpointURL:[NSURL URLWithString:@"https://api.perplexity.ai/v1/agent"]];
	agent.dialect = NFKRemoteResponsesDialectPerplexityAgent;
	agent.modelName = @"openai/gpt-5.5";
	agent.pollInterval = 0;
	return agent;
}

- (void)testThePerplexityPresetReachesTheAgentAPIInItsDialect
{
	NFKRemoteResponsesBackend *agent = [NFKRemoteResponsesBackend backendForProvider:NFKRemoteProvider.perplexity apiKey:@"pplx" modelName:@"openai/gpt-5.5"];
	XCTAssertEqualObjects(agent.endpointURL.absoluteString, @"https://api.perplexity.ai/v1/agent");
	XCTAssertEqual(agent.dialect, NFKRemoteResponsesDialectPerplexityAgent);
	XCTAssertEqualObjects(agent.modelName, @"openai/gpt-5.5");
}

// Perplexity spells the schema as response_format and takes the reasoning effort alone; a preset
// effort without a model folds in under its own name.
- (void)testThePerplexityDialectSpellsTheSchemaAndTheReasoningItsWay
{
	NFKStubResponsesBackend *agent = [self perplexityAgent];
	agent.modelName = nil;
	agent.stagedBodies = [@[ @"{\"id\":\"resp_1\",\"status\":\"completed\",\"output\":[]}" ] mutableCopy];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Who won?" }
															   parameters:@{ NFKParameterJSONSchema: @{ @"type": @"object" },
																			 NFKParameterReasoningEffort: NFKReasoningEffortDeep,
																			 @"effort": @"fast",
																			 NFKParameterTools: @[ @{ @"type": @"web_search", @"max_results": @5 } ] }];
	XCTAssertNotNil([agent runInferenceForRequest:request error:NULL]);
	NSDictionary *body = [agent bodyAt:0];
	XCTAssertEqualObjects(agent.requests[0].URL.absoluteString, @"https://api.perplexity.ai/v1/agent");
	XCTAssertEqualObjects(body[@"response_format"], (@{ @"type": @"json_schema", @"json_schema": @{ @"name": @"response", @"schema": @{ @"type": @"object" } } }));
	XCTAssertNil(body[@"text"]);
	XCTAssertEqualObjects(body[@"reasoning"], (@{ @"effort": @"high" }));
	XCTAssertEqualObjects(body[@"effort"], @"fast");
	XCTAssertNil(body[@"model"]);
	XCTAssertEqualObjects(body[@"tools"], (@[ @{ @"type": @"web_search", @"max_results": @5 } ]));
}

- (void)testPerplexitysToolResultItemsComeBackAsServerToolResults
{
	NFKStubResponsesBackend *agent = [self perplexityAgent];
	agent.stagedBodies = [@[ (@"{\"id\":\"resp_1\",\"status\":\"completed\",\"output\":["
							   "{\"type\":\"search_results\",\"queries\":[\"who won\"],\"results\":[{\"title\":\"T\",\"url\":\"https://a.test\"}]},"
							   "{\"type\":\"sandbox_results\",\"code\":\"print(1)\",\"stdout\":\"1\\n\",\"exit_code\":0},"
							   "{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"A won.\","
							   "\"annotations\":[{\"type\":\"url_citation\",\"url\":\"https://a.test\",\"title\":\"T\",\"start_index\":0,\"end_index\":6}]}]}],"
							   "\"usage\":{\"input_tokens\":30,\"output_tokens\":4}}") ] mutableCopy];
	NFKInferenceResult *result = [agent runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Who won?" }] error:NULL];
	XCTAssertEqualObjects(result.text, @"A won.");
	NSArray<NSDictionary *> *tools = [result outputForKey:NFKOutputServerToolResults];
	XCTAssertEqual(tools.count, 2u);
	XCTAssertEqualObjects(tools[0][@"name"], @"search_results");
	XCTAssertEqualObjects(tools[0][@"input"], (@[ @"who won" ]));
	XCTAssertEqualObjects(tools[0][@"output"][0][@"url"], @"https://a.test");
	XCTAssertEqualObjects(tools[1][@"name"], @"sandbox_results");
	XCTAssertEqualObjects(tools[1][@"input"], @"print(1)");
	XCTAssertEqualObjects(tools[1][@"output"], @"1\n");
	XCTAssertEqualObjects([[result outputForKey:NFKOutputCitations] firstObject][@"url"], @"https://a.test");
}

- (void)testAPerplexityBackgroundJobPollsUnderTheAgentPath
{
	NFKStubResponsesBackend *agent = [self perplexityAgent];
	agent.runsInBackground = YES;
	agent.stagedBodies = [@[ @"{\"id\":\"resp_7\",\"status\":\"queued\"}",
							 @"{\"id\":\"resp_7\",\"status\":\"completed\",\"output\":[]}" ] mutableCopy];
	XCTAssertNotNil([agent runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Research." }] error:NULL]);
	XCTAssertEqualObjects(agent.requests[1].URL.absoluteString, @"https://api.perplexity.ai/v1/agent/resp_7");
	XCTAssertEqualObjects(agent.requests[1].HTTPMethod, @"GET");
}

// Runs against Perplexity's Agent API when a key is present. POST /v1/agent answers 401 without a
// key and /v1/agentx 404, so the path is verified to exist; the reply shape is verified only here.
- (void)testALivePerplexityAgentAnswers
{
	NSString *key = NSProcessInfo.processInfo.environment[@"INFERKIT_PERPLEXITY_API_KEY"];
	if (key.length == 0) {
		XCTSkip("set INFERKIT_PERPLEXITY_API_KEY to exercise the live path");
	}
	NFKRemoteResponsesBackend *agent = [NFKRemoteResponsesBackend backendForProvider:NFKRemoteProvider.perplexity apiKey:key modelName:nil];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Reply with the word ready." }
															   parameters:@{ @"effort": @"fast" }];
	NSError *error = nil;
	NFKInferenceResult *result = [agent runInferenceForRequest:request error:&error];
	XCTAssertGreaterThan(result.text.length, 0u, @"%@", error);
}

- (void)testTheSandboxsFilesAreListedAndDownloaded
{
	NFKStubResponsesBackend *agent = [self perplexityAgent];
	agent.apiKey = @"pplx";
	agent.stagedBodies = [@[ (@"{\"object\":\"list\",\"data\":[{\"id\":\"file_1\",\"object\":\"file\","
							   "\"filename\":\"report.docx\",\"bytes\":2048,\"created_at\":1793000000}]}"), @"PK-bytes" ] mutableCopy];
	NSError *error = nil;
	NSArray<NFKRemoteFile *> *files = [agent outputFilesForResponseIdentifier:@"resp_7" error:&error];
	XCTAssertEqual(files.count, 1u, @"%@", error);
	XCTAssertEqualObjects(files[0].identifier, @"file_1");
	XCTAssertEqualObjects(files[0].filename, @"report.docx");
	XCTAssertEqual(files[0].byteCount, 2048);
	XCTAssertEqualObjects(files[0].createdAt, [NSDate dateWithTimeIntervalSince1970:1793000000]);
	XCTAssertEqualObjects(agent.requests[0].URL.absoluteString, @"https://api.perplexity.ai/v1/agent/resp_7/files");
	XCTAssertEqualObjects([agent.requests[0] valueForHTTPHeaderField:@"Authorization"], @"Bearer pplx");

	NSData *content = [agent contentsOfOutputFileWithIdentifier:@"file_1" responseIdentifier:@"resp_7" error:&error];
	XCTAssertEqualObjects(content, [@"PK-bytes" dataUsingEncoding:NSUTF8StringEncoding], @"%@", error);
	XCTAssertEqualObjects(agent.requests[1].URL.absoluteString, @"https://api.perplexity.ai/v1/agent/resp_7/files/file_1/content");
}

- (void)testTheRequestCarriesInstructionsInputItemsToolsSchemaAndReasoning
{
	CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
	CGContextRef context = CGBitmapContextCreate(NULL, 4, 4, 8, 16, colorSpace, kCGBitmapByteOrder32Little | kCGImageAlphaPremultipliedFirst);
	CGColorSpaceRelease(colorSpace);
	CGImageRef square = CGBitmapContextCreateImage(context);
	CGContextRelease(context);
	self.backend.stagedBodies = [@[ @"{\"id\":\"resp_1\",\"status\":\"completed\",\"output\":[]}" ] mutableCopy];
	NSArray *messages = @[ @{ @"role": @"system", @"content": @"Be exact." }, @{ @"role": @"user", @"content": @"What is this?" } ];
	NSDictionary *weather = @{ @"name": @"get_weather", @"parameters": @{ @"type": @"object" } };
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputMessages: messages, NFKInputImage: (__bridge id)square }
															   parameters:@{ NFKParameterTools: @[ weather, @{ @"type": @"web_search" } ],
																			 NFKParameterJSONSchema: @{ @"type": @"object" },
																			 NFKParameterReasoningEffort: NFKReasoningEffortDeep,
																			 NFKParameterMaxTokens: @2000, NFKParameterTemperature: @0.3,
																			 NFKParameterPreviousResponseIdentifier: @"resp_0" }];
	XCTAssertNotNil([self.backend runInferenceForRequest:request error:NULL]);
	NSDictionary *body = [self.backend bodyAt:0];
	XCTAssertEqualObjects(body[@"instructions"], @"Be exact.");
	NSArray *parts = body[@"input"][0][@"content"];
	XCTAssertEqualObjects(parts[0], (@{ @"type": @"input_text", @"text": @"What is this?" }));
	XCTAssertEqualObjects(parts[1][@"type"], @"input_image");
	XCTAssertEqualObjects(body[@"tools"][0], (@{ @"type": @"function", @"name": @"get_weather", @"parameters": @{ @"type": @"object" } }));
	XCTAssertEqualObjects(body[@"tools"][1], (@{ @"type": @"web_search" }));
	XCTAssertEqualObjects(body[@"text"][@"format"][@"type"], @"json_schema");
	XCTAssertEqualObjects(body[@"reasoning"], (@{ @"effort": @"high", @"summary": @"auto" }));
	XCTAssertEqualObjects(body[@"max_output_tokens"], @2000);
	XCTAssertEqualObjects(body[@"previous_response_id"], @"resp_0");
	XCTAssertNil(body[@"temperature"], @"a GPT-5 family model refuses sampling beside reasoning");
	CGImageRelease(square);
}

// The Responses API reads the conversation key as prompt_cache_key, retention for the models before
// GPT-5.6, and prompt_cache_options, prewarm included, for GPT-5.6 and later.
- (void)testThePromptCacheControlsGoOutInOpenAIsFields
{
	self.backend.stagedBodies = [@[ @"{\"id\":\"resp_1\",\"status\":\"completed\",\"output\":[]}" ] mutableCopy];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Hello" }
															   parameters:@{ NFKParameterConversationKey: @"tenant-4",
																			 NFKParameterPromptCacheRetention: @"in_memory",
																			 NFKParameterPromptCacheTTL: @"12h",
																			 NFKParameterPromptCacheMode: @"implicit",
																			 NFKParameterPromptCachePrewarm: @YES }];
	XCTAssertNotNil([self.backend runInferenceForRequest:request error:NULL]);
	NSDictionary *body = [self.backend bodyAt:0];
	XCTAssertEqualObjects(body[@"prompt_cache_key"], @"tenant-4");
	XCTAssertEqualObjects(body[@"prompt_cache_retention"], @"in_memory");
	XCTAssertEqualObjects(body[@"prompt_cache_options"], (@{ @"mode": @"implicit", @"ttl": @"12h", @"prewarm": @YES }));
	XCTAssertNil(body[NFKParameterConversationKey]);
	XCTAssertNil(body[NFKParameterPromptCachePrewarm]);
}

- (void)testAnUnsetPromptCacheControlSendsNothing
{
	self.backend.stagedBodies = [@[ @"{\"id\":\"resp_1\",\"status\":\"completed\",\"output\":[]}" ] mutableCopy];
	XCTAssertNotNil([self.backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Hello" } parameters:@{}]
												   error:NULL]);
	NSDictionary *body = [self.backend bodyAt:0];
	XCTAssertNil(body[@"prompt_cache_key"]);
	XCTAssertNil(body[@"prompt_cache_retention"]);
	XCTAssertNil(body[@"prompt_cache_options"]);
}

// A breakpoint rides on an input_text part the caller writes, which passes through.
- (void)testACacheBreakpointOnACallersInputTextPartPassesThrough
{
	self.backend.stagedBodies = [@[ @"{\"id\":\"resp_1\",\"status\":\"completed\",\"output\":[]}" ] mutableCopy];
	NSDictionary *part = @{ @"type": @"input_text", @"text": @"A long shared preamble.",
							@"prompt_cache_breakpoint": @{ @"mode": @"explicit", @"ttl": @"12h" } };
	NSArray *messages = @[ @{ @"role": @"developer", @"content": @[ part ] }, @{ @"role": @"user", @"content": @"Go." } ];
	XCTAssertNotNil([self.backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputMessages: messages }
																					 parameters:@{ NFKParameterPromptCacheMode: @"explicit" }]
												   error:NULL]);
	NSDictionary *body = [self.backend bodyAt:0];
	XCTAssertEqualObjects(body[@"input"][0][@"content"], @[ part ]);
	XCTAssertEqualObjects(body[@"prompt_cache_options"], (@{ @"mode": @"explicit" }));
}

- (void)testEveryOutputItemKindIsReadBack
{
	CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
	CGContextRef context = CGBitmapContextCreate(NULL, 4, 4, 8, 16, colorSpace, kCGBitmapByteOrder32Little | kCGImageAlphaPremultipliedFirst);
	CGColorSpaceRelease(colorSpace);
	CGImageRef square = CGBitmapContextCreateImage(context);
	CGContextRelease(context);
	NSString *png = [[NFKImageCoding PNGDataForImage:(__bridge id)square] base64EncodedStringWithOptions:0];
	CGImageRelease(square);
	NSString *reply = [NSString stringWithFormat:@"{\"id\":\"resp_9\",\"status\":\"completed\",\"output\":["
		"{\"type\":\"reasoning\",\"summary\":[{\"type\":\"summary_text\",\"text\":\"Checked the forecast.\"}]},"
		"{\"type\":\"web_search_call\",\"id\":\"ws_1\",\"status\":\"completed\",\"action\":{\"query\":\"weather\"}},"
		"{\"type\":\"function_call\",\"call_id\":\"call_1\",\"name\":\"get_weather\",\"arguments\":\"{\\\"city\\\":\\\"Oslo\\\"}\"},"
		"{\"type\":\"image_generation_call\",\"result\":\"%@\"},"
		"{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"Rain.\","
		"\"annotations\":[{\"type\":\"url_citation\",\"url\":\"https://w.example\",\"title\":\"W\",\"start_index\":0,\"end_index\":5}]}]}],"
		"\"usage\":{\"input_tokens\":10,\"output_tokens\":20,\"output_tokens_details\":{\"reasoning_tokens\":15}}}", png];
	self.backend.stagedBodies = [@[ reply ] mutableCopy];
	NSError *error = nil;
	NFKInferenceResult *result = [self.backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"weather?" }] error:&error];
	XCTAssertNotNil(result, @"%@", error);
	XCTAssertEqualObjects(result.text, @"Rain.");
	XCTAssertEqualObjects([result outputForKey:NFKOutputReasoning], @"Checked the forecast.");
	XCTAssertEqualObjects(result.toolCalls.firstObject[@"arguments"], (@{ @"city": @"Oslo" }));
	XCTAssertEqualObjects([[result outputForKey:NFKOutputCitations] firstObject][@"url"], @"https://w.example");
	NSArray *ran = [result outputForKey:NFKOutputServerToolResults];
	XCTAssertEqualObjects(ran.firstObject[@"name"], @"web_search_call");
	XCTAssertEqualObjects(ran.firstObject[@"input"], (@{ @"query": @"weather" }));
	XCTAssertNotNil([result outputForKey:NFKOutputImage]);
	XCTAssertEqualObjects([result outputForKey:NFKOutputResponseIdentifier], @"resp_9");
	XCTAssertEqualObjects([result outputForKey:NFKOutputUsage][NFKUsageReasoningTokens], @15);
}

- (void)testARefusalAndAFailedStatusAreErrors
{
	self.backend.stagedBodies = [@[ @"{\"status\":\"completed\",\"output\":[{\"type\":\"message\",\"content\":[{\"type\":\"refusal\",\"refusal\":\"no\"}]}]}" ] mutableCopy];
	NSError *error = nil;
	XCTAssertNil([self.backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"x" }] error:&error]);
	XCTAssertEqual(error.code, (NSInteger)kNFKError_InferenceRefused);

	self.backend.stagedBodies = [@[ @"{\"status\":\"failed\",\"error\":{\"message\":\"server fault\"}}" ] mutableCopy];
	XCTAssertNil([self.backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"x" }] error:&error]);
	XCTAssertTrue([error.localizedDescription containsString:@"server fault"]);
}

- (void)testABackgroundJobIsPolledUntilItEnds
{
	self.backend.runsInBackground = YES;
	self.backend.stagedBodies = [@[ @"{\"id\":\"resp_bg\",\"status\":\"queued\"}",
									@"{\"id\":\"resp_bg\",\"status\":\"in_progress\"}",
									@"{\"id\":\"resp_bg\",\"status\":\"completed\",\"output\":[{\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"done\"}]}]}" ] mutableCopy];
	NFKInferenceResult *result = [self.backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"long task" }] error:NULL];
	XCTAssertEqualObjects(result.text, @"done");
	XCTAssertEqualObjects([self.backend bodyAt:0][@"background"], @YES);
	XCTAssertEqual(self.backend.requests.count, 3);
	XCTAssertEqualObjects(self.backend.requests[1].URL.absoluteString, @"https://api.openai.com/v1/responses/resp_bg");
	XCTAssertEqualObjects(self.backend.requests[1].HTTPMethod, @"GET");
}

- (void)testTheStreamGrowsTextAndReasoningAndEndsWithTheCompletedResponse
{
	self.backend.stagedLines = @[ @"data: {\"type\":\"response.reasoning_summary_text.delta\",\"delta\":\"Thinking\"}",
								  @"data: {\"type\":\"response.output_text.delta\",\"delta\":\"Hel\"}",
								  @"data: {\"type\":\"response.output_text.delta\",\"delta\":\"lo\"}",
								  (@"data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_s\",\"status\":\"completed\","
								  "\"output\":[{\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"Hello\"}]}]}}") ];
	NFKInferenceJob *job = [self.backend submitInferenceJobForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"hi" }]];
	XCTAssertEqual(job.status, NFKInferenceJobStatusSucceeded, @"%@", job.error);
	XCTAssertEqualObjects(job.result.text, @"Hello");
	XCTAssertEqualObjects([job.result outputForKey:NFKOutputResponseIdentifier], @"resp_s");
	XCTAssertEqualObjects([self.backend bodyAt:0][@"stream"], @YES);
}

@end
