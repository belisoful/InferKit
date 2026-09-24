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

- (void)testTheFactoryReachesTheServicesThatServeResponses
{
	XCTAssertEqualObjects([NFKRemoteResponsesBackend backendForProvider:NFKRemoteProvider.xAI apiKey:@"k" modelName:@"m"].endpointURL.absoluteString,
						  @"https://api.x.ai/v1/responses");
	XCTAssertNotNil([NFKRemoteResponsesBackend backendForProvider:NFKRemoteProvider.deepSeek apiKey:@"k" modelName:@"m"]);
	XCTAssertNil([NFKRemoteResponsesBackend backendForProvider:NFKRemoteProvider.anthropic apiKey:@"k" modelName:@"m"]);
	XCTAssertNil([NFKRemoteResponsesBackend backendForProvider:NFKRemoteProvider.mistral apiKey:@"k" modelName:@"m"]);
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
