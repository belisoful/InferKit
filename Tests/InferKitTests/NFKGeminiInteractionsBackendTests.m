//
//  NFKGeminiInteractionsBackendTests.m
//  InferKitTests
//
//  Gemini's Interactions API through stubbed transports: the input steps and content blocks, each
//  output modality's response_format and reply, transcription, background polling, and the stream.
//

#import <XCTest/XCTest.h>
#import <InferKit/NFKGeminiInteractionsBackend.h>
#import <InferKit/NFKImageCoding.h>
#import <InferKit/NFKInferenceRequest.h>
#import <InferKit/NFKInferenceResult.h>
#import <InferKit/NFKInferenceJob.h>
#import <InferKit/NFKInferenceKeys.h>
#import <InferKit/NFKAudioAsset.h>
#import <InferKit/NFKAudioSegment.h>
#import <InferKit/NFKVideoAsset.h>
#import <InferKit/NFKErrors.h>
#import "NFKTestClip.h"

@interface NFKStubInteractionsBackend : NFKGeminiInteractionsBackend
@property (nonatomic, strong) NSMutableArray<NSURLRequest *> *requests;
@property (nonatomic, strong) NSMutableArray<NSData *> *stagedReplies;
@property (nonatomic, copy) NSArray<NSString *> *stagedLines;
@property (nonatomic, strong) NSMutableIndexSet *failingRequests;
@property (nonatomic, strong, nullable) dispatch_semaphore_t firstRequestGate;
@property (nonatomic, copy, nullable) void (^requestSent)(NSURLRequest *request);
@end

@implementation NFKStubInteractionsBackend
- (NSData *)sendRequest:(NSURLRequest *)request response:(NSHTTPURLResponse **)outResponse error:(NSError **)outError
{
	if (self.requests == nil) {
		self.requests = [NSMutableArray array];
	}
	[self.requests addObject:request];
	if (self.requests.count == 1 && self.firstRequestGate != nil) {
		dispatch_semaphore_wait(self.firstRequestGate, DISPATCH_TIME_FOREVER);
	}
	if (self.requestSent != nil) {
		self.requestSent(request);
	}
	if ([self.failingRequests containsIndex:self.requests.count - 1]) {
		if (outError != NULL) {
			*outError = [NSError errorWithDomain:NFKInferenceErrorDomain code:kNFKError_RemoteUnreachable
										userInfo:@{ NSLocalizedDescriptionKey: @"offline" }];
		}
		return nil;
	}
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
	for (NSString *line in self.stagedLines) {
		lineHandler(line);
	}
	completionHandler([[NSHTTPURLResponse alloc] initWithURL:request.URL statusCode:200 HTTPVersion:@"HTTP/1.1" headerFields:nil], nil, nil);
	return ^{};
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

@interface NFKGeminiInteractionsBackendTests : XCTestCase
@property (nonatomic, strong) NFKStubInteractionsBackend *backend;
@property (nonatomic, strong) NSURL *directory;
@property (nonatomic, assign) CGImageRef square;
@property (nonatomic, copy) NSString *squarePNG;
@end

@implementation NFKGeminiInteractionsBackendTests

- (void)setUp
{
	[super setUp];
	self.directory = [[NSURL fileURLWithPath:NSTemporaryDirectory()] URLByAppendingPathComponent:NSUUID.UUID.UUIDString isDirectory:YES];
	self.backend = [NFKStubInteractionsBackend backendWithAPIKey:@"AIza" modelName:@"gemini-3.8-flash"];
	self.backend.pollInterval = 0;
	self.backend.outputDirectoryURL = self.directory;
	CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
	CGContextRef context = CGBitmapContextCreate(NULL, 4, 4, 8, 16, colorSpace, kCGBitmapByteOrder32Little | kCGImageAlphaPremultipliedFirst);
	CGColorSpaceRelease(colorSpace);
	self.square = CGBitmapContextCreateImage(context);
	CGContextRelease(context);
	self.squarePNG = [[NFKImageCoding PNGDataForImage:(__bridge id)self.square] base64EncodedStringWithOptions:0];
}

- (void)tearDown
{
	CGImageRelease(self.square);
	[NSFileManager.defaultManager removeItemAtURL:self.directory error:NULL];
	[super tearDown];
}

- (void)testTextAndAnImageBecomeAUserStepAndTheReplyTextThoughtsAndUsageComeBack
{
	[self.backend stage:@"{\"id\":\"v1_a\",\"status\":\"completed\",\"steps\":["
	 "{\"type\":\"thought\",\"content\":[{\"type\":\"text\",\"text\":\"Looking at the square.\"}]},"
	 "{\"type\":\"model_output\",\"content\":[{\"type\":\"text\",\"text\":\"A blue square.\"}]}],"
	 "\"usage\":{\"total_input_tokens\":12,\"total_output_tokens\":4,\"total_thought_tokens\":9}}"];
	NSArray *messages = @[ @{ @"role": @"system", @"content": @"Answer briefly." }, @{ @"role": @"user", @"content": @"What is this?" } ];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputMessages: messages, NFKInputImage: (__bridge id)self.square }
															   parameters:@{ NFKParameterReasoningEffort: NFKReasoningEffortModerate, @"temperature": @0.2 }];
	NSError *error = nil;
	NFKInferenceResult *result = [self.backend runInferenceForRequest:request error:&error];
	XCTAssertNotNil(result, @"%@", error);
	NSURLRequest *sent = self.backend.requests.firstObject;
	XCTAssertEqualObjects(sent.URL.absoluteString, @"https://generativelanguage.googleapis.com/v1beta/interactions");
	XCTAssertEqualObjects([sent valueForHTTPHeaderField:@"x-goog-api-key"], @"AIza");
	NSDictionary *body = [self.backend bodyAt:0];
	XCTAssertEqualObjects(body[@"system_instruction"], @"Answer briefly.");
	NSDictionary *step = body[@"input"][0];
	XCTAssertEqualObjects(step[@"type"], @"user_input");
	XCTAssertEqualObjects(step[@"content"][0], (@{ @"type": @"text", @"text": @"What is this?" }));
	XCTAssertEqualObjects(step[@"content"][1][@"type"], @"image");
	XCTAssertEqualObjects(body[@"generation_config"][@"thinking_level"], @"medium");
	XCTAssertEqualObjects(body[@"generation_config"][@"temperature"], @0.2);
	XCTAssertEqualObjects(result.text, @"A blue square.");
	XCTAssertEqualObjects([result outputForKey:NFKOutputReasoning], @"Looking at the square.");
	XCTAssertEqualObjects([result outputForKey:NFKOutputUsage][NFKUsageReasoningTokens], @9);
	XCTAssertEqualObjects([result outputForKey:NFKOutputResponseIdentifier], @"v1_a");
}

- (void)testAnImageRequestAsksForAnImageAndDecodesIt
{
	[self.backend stage:[NSString stringWithFormat:@"{\"status\":\"completed\",\"steps\":[{\"type\":\"model_output\",\"content\":["
						 "{\"type\":\"image\",\"mime_type\":\"image/png\",\"data\":\"%@\"}]}]}", self.squarePNG]];
	self.backend.modelName = @"gemini-3.1-flash-image";
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"a square" }
															   parameters:@{ NFKParameterAspectRatio: @"16:9", NFKParameterResolution: @"2K" }
														   outputModality:NFKModalityImage];
	NFKInferenceResult *result = [self.backend runInferenceForRequest:request error:NULL];
	NSDictionary *format = [self.backend bodyAt:0][@"response_format"];
	XCTAssertEqualObjects(format, (@{ @"type": @"image", @"aspect_ratio": @"16:9", @"image_size": @"2K" }));
	XCTAssertEqualObjects([self.backend bodyAt:0][@"input"], @"a square", @"a prompt alone goes as a string");
	XCTAssertNotNil([result outputForKey:NFKOutputImage]);
}

- (void)testTwoSpeakersGoInTheSpeakersObjectAndOverrideTheVoice
{
	[self.backend stage:@"{\"status\":\"completed\",\"steps\":[]}"];
	self.backend.modelName = @"gemini-3.8-flash-tts";
	self.backend.voice = @"Kore";
	NSArray *speakers = @[ @{ @"speaker": @"Joe", @"voice": @"Puck" }, @{ @"speaker": @"Jane", @"voice": @"Kore" } ];
	NFKInferenceRequest *dialogue = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Joe: Hi.\nJane: Hello." }
																parameters:@{ @"speakers": speakers } outputModality:NFKModalityAudio];
	[self.backend runInferenceForRequest:dialogue error:NULL];
	XCTAssertEqualObjects([self.backend bodyAt:0][@"generation_config"][@"speech_config"], (@{ @"speakers": speakers }));

	NSDictionary *wire = @{ @"speakers": speakers, @"mode": @"conversational" };
	NFKInferenceRequest *shaped = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Joe: Hi." }
															  parameters:@{ @"speakers": wire } outputModality:NFKModalityAudio];
	[self.backend runInferenceForRequest:shaped error:NULL];
	XCTAssertEqualObjects([self.backend bodyAt:1][@"generation_config"][@"speech_config"], wire, @"a dictionary passes through");
}

- (void)testSpeechNamesItsVoiceAndThePCMReplyIsWrittenAsWAV
{
	NSData *pcm = [NSData dataWithBytes:"\x01\x00\x02\x00" length:4];
	[self.backend stage:[NSString stringWithFormat:@"{\"status\":\"completed\",\"steps\":[{\"type\":\"model_output\",\"content\":["
						 "{\"type\":\"audio\",\"mime_type\":\"audio/L16;codec=pcm;rate=24000\",\"data\":\"%@\"}]}]}",
						 [pcm base64EncodedStringWithOptions:0]]];
	self.backend.modelName = @"gemini-3.1-flash-tts-preview";
	self.backend.voice = @"Kore";
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Say hi" } parameters:@{} outputModality:NFKModalityAudio];
	NFKInferenceResult *result = [self.backend runInferenceForRequest:request error:NULL];
	NSDictionary *body = [self.backend bodyAt:0];
	XCTAssertEqualObjects(body[@"response_format"], (@{ @"type": @"audio" }));
	XCTAssertEqualObjects(body[@"generation_config"][@"speech_config"], (@[ @{ @"voice": @"Kore" } ]));
	NFKAudioAsset *audio = [result outputForKey:NFKOutputAudio];
	NSData *wav = [NSData dataWithContentsOfURL:audio.fileURL];
	XCTAssertEqualObjects(audio.fileURL.pathExtension, @"wav");
	XCTAssertEqualObjects([wav subdataWithRange:NSMakeRange(0, 4)], [@"RIFF" dataUsingEncoding:NSASCIIStringEncoding]);
	XCTAssertEqualObjects([wav subdataWithRange:NSMakeRange(44, 4)], pcm);
}

- (void)testATranscriptionAsksForVerbatimSpeakersAndReadsTheWords
{
	[self.backend stage:@"{\"status\":\"completed\",\"steps\":[{\"type\":\"model_output\",\"content\":[{\"type\":\"text\",\"text\":\"Hi. Hello.\","
	 "\"annotations\":[{\"type\":\"word_info\",\"text\":\"Hi.\",\"speaker\":\"spk_1\",\"start_offset\":\"0.1s\",\"end_offset\":\"0.4s\"},"
	 "{\"type\":\"word_info\",\"text\":\"Hello.\",\"speaker\":\"spk_2\",\"start_offset\":\"1.0s\",\"end_offset\":\"1.6s\"}]}]}]}"];
	self.backend.modelName = @"gemini-3.5-transcribe";
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputAudio: [@"RIFF" dataUsingEncoding:NSUTF8StringEncoding] }
															   parameters:@{ NFKParameterSpeakerDiarization: @YES, NFKParameterWordTimestamps: @YES,
																			 NFKParameterVocabulary: @[ @"InferKit" ], NFKParameterSourceLanguage: @"en" }];
	NFKInferenceResult *result = [self.backend runInferenceForRequest:request error:NULL];
	NSDictionary *configuration = [self.backend bodyAt:0][@"generation_config"][@"transcription_config"];
	XCTAssertEqualObjects(configuration[@"language_codes"], (@[ @"en" ]));
	XCTAssertEqualObjects(configuration[@"custom_vocabulary"], (@[ @"InferKit" ]));
	XCTAssertEqualObjects(configuration[@"mode"], (@{ @"type": @"verbatim", @"diarization_mode": @"speaker", @"timestamp_granularities": @[ @"word" ] }));
	NSArray<NFKAudioSegment *> *words = [result outputForKey:NFKOutputWords];
	XCTAssertEqual(words.count, 2);
	XCTAssertEqualWithAccuracy(words[1].startSeconds, 1.0, 1e-9);
	NSArray<NFKAudioSegment *> *turns = [result outputForKey:NFKOutputSegments];
	XCTAssertEqual(turns.count, 2);
	XCTAssertEqualObjects(turns[1].speaker, @"spk_2");
}

- (void)testABackgroundVideoIsPolledAndItsURIFetchedWithTheKey
{
	self.backend.runsInBackground = YES;
	self.backend.modelName = @"gemini-omni-1.1-flash";
	[self.backend stage:@"{\"id\":\"v1_vid\",\"status\":\"in_progress\"}"];
	[self.backend stage:@"{\"id\":\"v1_vid\",\"status\":\"completed\",\"steps\":[{\"type\":\"model_output\",\"content\":["
	 "{\"type\":\"video\",\"mime_type\":\"video/mp4\",\"uri\":\"https://generativelanguage.googleapis.com/v1beta/files/abc:download\"}]}]}"];
	[self.backend stage:@"ftypisom"];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"a wave" }
															   parameters:@{ NFKParameterAspectRatio: @"16:9", NFKParameterResolution: @"1080p" }
														   outputModality:NFKModalityVideo];
	NSError *error = nil;
	NFKInferenceResult *result = [self.backend runInferenceForRequest:request error:&error];
	XCTAssertNotNil(result, @"%@", error);
	XCTAssertEqualObjects([self.backend bodyAt:0][@"background"], @YES);
	XCTAssertEqualObjects([self.backend bodyAt:0][@"response_format"], (@{ @"type": @"video", @"aspect_ratio": @"16:9", @"resolution": @"1080p" }));
	XCTAssertEqualObjects(self.backend.requests[1].URL.absoluteString, @"https://generativelanguage.googleapis.com/v1beta/interactions/v1_vid");
	XCTAssertEqualObjects([self.backend.requests[2] valueForHTTPHeaderField:@"x-goog-api-key"], @"AIza");
	NFKVideoAsset *clip = [result outputForKey:NFKOutputVideo];
	XCTAssertEqualObjects([NSData dataWithContentsOfURL:clip.fileURL], [@"ftypisom" dataUsingEncoding:NSUTF8StringEncoding]);
}

- (void)testABackgroundInteractionPastItsTimeoutIsCancelledOnTheService
{
	self.backend.agentName = @"deep-research-preview-04-2026";
	self.backend.backgroundTimeout = 0.01;
	self.backend.pollInterval = 0.002;
	[self.backend stage:@"{\"id\":\"v1_dr\",\"status\":\"in_progress\"}"];
	NSError *error = nil;
	XCTAssertNil([self.backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"research" }] error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceBackendFailure);
	XCTAssertTrue([error.localizedDescription containsString:@"v1_dr"], @"%@", error);
	NSURLRequest *last = self.backend.requests.lastObject;
	XCTAssertEqualObjects(last.HTTPMethod, @"POST");
	XCTAssertEqualObjects(last.URL.absoluteString, @"https://generativelanguage.googleapis.com/v1beta/interactions/v1_dr/cancel");
}

- (void)testACancelDuringTheCreateRequestCancelsTheRunOnTheService
{
	self.backend.agentName = @"deep-research-preview-04-2026";
	self.backend.firstRequestGate = dispatch_semaphore_create(0);
	XCTestExpectation *cancelled = [self expectationWithDescription:@"cancel sent"];
	self.backend.requestSent = ^(NSURLRequest *request) {
		if ([request.URL.absoluteString hasSuffix:@"/v1_dr/cancel"]) {
			[cancelled fulfill];
		}
	};
	[self.backend stage:@"{\"id\":\"v1_dr\",\"status\":\"in_progress\"}"];
	NFKInferenceJob *job = [self.backend submitInferenceJobForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"research" }]];
	[job cancel];
	dispatch_semaphore_signal(self.backend.firstRequestGate);
	[self waitForExpectations:@[ cancelled ] timeout:5.0];
	XCTAssertEqual(job.status, NFKInferenceJobStatusCancelled);
}

- (void)testAFailedPollIsPolledAgain
{
	self.backend.runsInBackground = YES;
	self.backend.failingRequests = [NSMutableIndexSet indexSetWithIndexesInRange:NSMakeRange(1, 2)];
	[self.backend stage:@"{\"id\":\"v1_b\",\"status\":\"in_progress\"}"];
	[self.backend stage:@"{\"id\":\"v1_b\",\"status\":\"completed\",\"steps\":[{\"type\":\"model_output\",\"content\":[{\"type\":\"text\",\"text\":\"Done\"}]}]}"];
	NSError *error = nil;
	NFKInferenceResult *result = [self.backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"hi" }] error:&error];
	XCTAssertEqualObjects(result.text, @"Done", @"%@", error);
	XCTAssertEqual(self.backend.requests.count, 4u);
}

- (void)testThreeFailedPollsInARowEndTheWaitAndLeaveTheRunStored
{
	self.backend.runsInBackground = YES;
	self.backend.failingRequests = [NSMutableIndexSet indexSetWithIndexesInRange:NSMakeRange(1, 3)];
	[self.backend stage:@"{\"id\":\"v1_b\",\"status\":\"in_progress\"}"];
	NSError *error = nil;
	XCTAssertNil([self.backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"hi" }] error:&error]);
	XCTAssertEqual(error.code, kNFKError_RemoteUnreachable);
	XCTAssertTrue([error.localizedDescription containsString:@"v1_b"], @"%@", error);
	XCTAssertEqual(self.backend.requests.count, 4u);
	XCTAssertEqualObjects(self.backend.requests.lastObject.HTTPMethod, @"GET");
}

- (void)testAFunctionCallAndAContinuationAreCarried
{
	[self.backend stage:@"{\"id\":\"v1_f\",\"status\":\"requires_action\",\"steps\":[{\"type\":\"function_call\",\"id\":\"c1\",\"name\":\"get_weather\",\"arguments\":{\"city\":\"Oslo\"}}]}"];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"weather?" }
															   parameters:@{ NFKParameterTools: @[ @{ @"name": @"get_weather", @"parameters": @{ @"type": @"object" } },
																								 @{ @"type": @"google_search" } ],
																			 NFKParameterPreviousResponseIdentifier: @"v1_prev" }];
	NFKInferenceResult *result = [self.backend runInferenceForRequest:request error:NULL];
	NSDictionary *body = [self.backend bodyAt:0];
	XCTAssertEqualObjects(body[@"previous_interaction_id"], @"v1_prev");
	XCTAssertEqualObjects(body[@"tools"][1], (@{ @"type": @"google_search" }));
	XCTAssertEqualObjects(result.toolCalls.firstObject[@"arguments"], (@{ @"city": @"Oslo" }));
}

// An Antigravity agent runs on the backend's model in the background; its own file tool calls come
// back with their results as server tools, its call to the caller's tool as a tool call, and its
// sandbox's id under its own key.
- (void)testAnAgentRunsInTheBackgroundAndSeparatesItsOwnToolCallsFromTheCallers
{
	self.backend.agentName = @"antigravity-preview-09-2026";
	self.backend.modelName = @"gemini-4-argon";
	[self.backend stage:@"{\"id\":\"v1_ag\",\"status\":\"in_progress\"}"];
	[self.backend stage:@"{\"id\":\"v1_ag\",\"status\":\"requires_action\",\"environment_id\":\"140128b2\",\"steps\":["
	 "{\"type\":\"thought\",\"signature\":\"s1\",\"summary\":[{\"type\":\"text\",\"text\":\"Write the mesh script.\"}]},"
	 "{\"type\":\"function_call\",\"id\":\"w1\",\"name\":\"write_to_file\",\"arguments\":{\"TargetFile\":\"stand.py\"}},"
	 "{\"type\":\"function_result\",\"call_id\":\"w1\",\"name\":\"write_to_file\",\"result\":\"ok\"},"
	 "{\"type\":\"function_call\",\"id\":\"c1\",\"name\":\"get_dimensions\",\"arguments\":{\"part\":\"stand\"}}]}"];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Model a laptop stand as an STL." }
															   parameters:@{ NFKParameterTools: @[ @{ @"name": @"get_dimensions", @"parameters": @{ @"type": @"object" } } ],
																			 @"environment": @"remote", @"service_tier": @"flex",
																			 @"agent_config": @{ @"max_total_tokens": @200000 } }];
	NSError *error = nil;
	NFKInferenceResult *result = [self.backend runInferenceForRequest:request error:&error];
	XCTAssertNotNil(result, @"%@", error);
	NSDictionary *body = [self.backend bodyAt:0];
	XCTAssertEqualObjects(body[@"agent"], @"antigravity-preview-09-2026");
	XCTAssertNil(body[@"model"]);
	XCTAssertEqualObjects(body[@"agent_config"], (@{ @"type": @"antigravity", @"model": @"gemini-4-argon", @"max_total_tokens": @200000 }));
	XCTAssertNil(body[@"generation_config"]);
	XCTAssertEqualObjects(body[@"background"], @YES);
	XCTAssertEqualObjects(body[@"store"], @YES);
	XCTAssertEqualObjects(body[@"environment"], @"remote");
	XCTAssertEqualObjects(body[@"service_tier"], @"flex");
	XCTAssertEqual(self.backend.requests.count, 2u);
	XCTAssertEqualObjects([result outputForKey:NFKOutputReasoning], @"Write the mesh script.");
	XCTAssertEqualObjects([result.toolCalls valueForKey:@"id"], @[ @"c1" ]);
	NSArray *serverTools = [result outputForKey:NFKOutputServerToolResults];
	XCTAssertEqualObjects([serverTools valueForKey:@"name"], @[ @"write_to_file" ]);
	XCTAssertEqualObjects(serverTools.firstObject[@"output"], @"ok");
	XCTAssertEqualObjects([result outputForKey:NFKGeminiOutputEnvironmentIdentifier], @"140128b2");
}

// An MCP server becomes an mcp_server tool beside the function tools, its name made one Google accepts.
- (void)testMCPServersBecomeMCPServerTools
{
	self.backend.agentName = @"antigravity-preview-09-2026";
	[self.backend stage:@"{\"id\":\"v1_m\",\"status\":\"completed\",\"steps\":["
	 "{\"type\":\"mcp_server_tool_call\",\"id\":\"m1\",\"name\":\"upscale\",\"server_name\":\"inferkit-mac\",\"arguments\":{}},"
	 "{\"type\":\"mcp_server_tool_result\",\"call_id\":\"m1\",\"name\":\"upscale\",\"server_name\":\"inferkit-mac\",\"result\":\"done\"},"
	 "{\"type\":\"model_output\",\"content\":[{\"type\":\"text\",\"text\":\"Upscaled.\"}]}]}"];
	NSArray *servers = @[ @{ @"name": @"InferKit Mac", @"url": @"https://tunnel.example.com/mcp", @"authorizationToken": @"tok",
							 @"allowedTools": @[ @"upscale" ] } ];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Upscale the render." }
															   parameters:@{ NFKParameterMCPServers: servers,
																			 NFKParameterTools: @[ @{ @"name": @"notify", @"parameters": @{ @"type": @"object" } } ] }];
	NFKInferenceResult *result = [self.backend runInferenceForRequest:request error:NULL];
	NSArray *tools = [self.backend bodyAt:0][@"tools"];
	XCTAssertEqual(tools.count, 2u);
	XCTAssertEqualObjects(tools[0][@"name"], @"notify");
	NSDictionary *expected = @{ @"type": @"mcp_server", @"name": @"inferkit-mac", @"url": @"https://tunnel.example.com/mcp",
								@"headers": @{ @"Authorization": @"Bearer tok" }, @"allowed_tools": @[ @{ @"tools": @[ @"upscale" ] } ] };
	XCTAssertEqualObjects(tools[1], expected);
	XCTAssertEqualObjects([[result outputForKey:NFKOutputServerToolResults] valueForKey:@"name"],
						  (@[ @"mcp_server_tool_call", @"mcp_server_tool_result" ]));
	XCTAssertEqualObjects(result.text, @"Upscaled.");
}

- (void)testDeepResearchTypesItsConfigurationAndAsksForThinkingSummaries
{
	self.backend.agentName = @"deep-research-preview-04-2026";
	[self.backend stage:@"{\"id\":\"v1_dr\",\"status\":\"completed\",\"steps\":[{\"type\":\"model_output\",\"content\":["
	 "{\"type\":\"text\",\"text\":\"Report.\",\"annotations\":[{\"type\":\"url_citation\",\"url\":\"https://example.com\",\"title\":\"Ex\",\"start_index\":0,\"end_index\":6}]}]}]}"];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Research FDM tolerances." }
															   parameters:@{ NFKParameterReasoningEffort: NFKReasoningEffortDeep }];
	NFKInferenceResult *result = [self.backend runInferenceForRequest:request error:NULL];
	NSDictionary *body = [self.backend bodyAt:0];
	XCTAssertEqualObjects(body[@"agent_config"], (@{ @"type": @"deep-research", @"thinking_summaries": @"auto" }));
	XCTAssertNil(body[@"model"]);
	XCTAssertEqualObjects(result.text, @"Report.");
	XCTAssertEqualObjects([result outputForKey:NFKOutputCitations][0][@"url"], @"https://example.com");
}

// gemini-4-argon: more than 65536 output tokens is a background long decode whose hops' text joins.
- (void)testALongDecodeRunsInTheBackgroundAndJoinsItsHops
{
	self.backend.modelName = @"gemini-4-argon";
	[self.backend stage:@"{\"id\":\"v1_ld\",\"status\":\"queued\"}"];
	[self.backend stage:@"{\"id\":\"v1_ld\",\"status\":\"completed\",\"steps\":["
	 "{\"type\":\"model_output\",\"content\":[{\"type\":\"text\",\"text\":\"first hop, \"}]},"
	 "{\"type\":\"model_output\",\"content\":[{\"type\":\"text\",\"text\":\"second hop\"}]}]}"];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Write the book." }
															   parameters:@{ NFKParameterMaxTokens: @200000 }];
	NFKInferenceJob *job = [self.backend submitInferenceJobForRequest:request];
	XCTestExpectation *done = [self expectationWithDescription:@"finished"];
	job.completionHandler = ^(NFKInferenceJob *finished) { [done fulfill]; };
	[self waitForExpectationsWithTimeout:10 handler:nil];
	XCTAssertEqual(job.status, NFKInferenceJobStatusSucceeded, @"%@", job.error);
	NSDictionary *body = [self.backend bodyAt:0];
	XCTAssertEqualObjects(body[@"background"], @YES);
	XCTAssertEqualObjects(body[@"generation_config"][@"max_output_tokens"], @200000);
	XCTAssertFalse([self.backend.requests.firstObject.URL.absoluteString containsString:@"alt=sse"]);
	XCTAssertEqualObjects(job.result.text, @"first hop, second hop");
}

// Agentic video understanding: the processing parameter rides on the video block, and the processing
// steps the model takes come back as server tools.
- (void)testAgenticProcessingRidesOnTheVideoBlock
{
	NSError *error = nil;
	NSURL *clip = [NFKTestClip writeClipWithColors:@[ @[ @1, @0, @0 ] ] error:&error];
	XCTAssertNotNil(clip, @"%@", error);
	[self.backend stage:@"{\"id\":\"v1_v\",\"status\":\"completed\",\"steps\":["
	 "{\"type\":\"processing_call\",\"id\":\"call_01\",\"signature\":\"sig_call_01\"},"
	 "{\"type\":\"processing_result\",\"call_id\":\"call_01\",\"signature\":\"sig_result_01\"},"
	 "{\"type\":\"model_output\",\"content\":[{\"type\":\"text\",\"text\":\"A red frame.\"}]}]}"];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"What happens?",
																			  NFKInputVideo: [NFKVideoAsset videoAssetWithFileURL:clip] }
															   parameters:@{ @"processing": @"agentic" }];
	NFKInferenceResult *result = [self.backend runInferenceForRequest:request error:&error];
	XCTAssertNotNil(result, @"%@", error);
	NSDictionary *body = [self.backend bodyAt:0];
	NSDictionary *video = body[@"input"][0][@"content"][1];
	XCTAssertEqualObjects(video[@"type"], @"video");
	XCTAssertEqualObjects(video[@"processing"], @"agentic");
	XCTAssertNil(body[@"generation_config"][@"processing"]);
	XCTAssertEqualObjects([[result outputForKey:NFKOutputServerToolResults] valueForKey:@"name"], (@[ @"processing_call", @"processing_result" ]));
	XCTAssertEqualObjects(result.text, @"A red frame.");
	[NSFileManager.defaultManager removeItemAtURL:clip error:NULL];
}

- (void)testTheStreamGrowsTheTextAndThenReadsTheFinishedInteraction
{
	self.backend.stagedLines = @[ @"data: {\"event_type\":\"interaction.created\",\"interaction\":{\"id\":\"v1_s\",\"status\":\"in_progress\"}}",
								  @"data: {\"event_type\":\"step.delta\",\"index\":0,\"delta\":{\"type\":\"text\",\"text\":\"Hel\"}}",
								  @"data: {\"event_type\":\"step.delta\",\"index\":0,\"delta\":{\"type\":\"text\",\"text\":\"lo\"}}",
								  @"data: {\"event_type\":\"interaction.completed\",\"interaction\":{\"id\":\"v1_s\",\"status\":\"completed\"}}" ];
	[self.backend stage:@"{\"id\":\"v1_s\",\"status\":\"completed\",\"steps\":[{\"type\":\"model_output\",\"content\":[{\"type\":\"text\",\"text\":\"Hello\"}]}]}"];
	NFKInferenceJob *job = [self.backend submitInferenceJobForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"hi" }]];
	XCTAssertEqual(job.status, NFKInferenceJobStatusSucceeded, @"%@", job.error);
	XCTAssertEqualObjects(job.result.text, @"Hello");
	XCTAssertTrue([self.backend.requests.firstObject.URL.absoluteString hasSuffix:@"interactions?alt=sse"]);
	XCTAssertEqualObjects(self.backend.requests[1].URL.absoluteString, @"https://generativelanguage.googleapis.com/v1beta/interactions/v1_s");
}

@end
