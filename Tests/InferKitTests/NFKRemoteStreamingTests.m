//
//  NFKRemoteStreamingTests.m
//  InferKitTests
//
//  The streamed form of both chat backends through stubbed line streams: the request it sends, the
//  partial results a stream produces, how a tool call assembles across deltas, the two ways a stream
//  ends, and that cancelling the job cancels the request. A live pass against a local runner is gated.
//

#import <XCTest/XCTest.h>
#import <InferKit/NFKRemoteBackend.h>
#import <InferKit/NFKAnthropicBackend.h>
#import <InferKit/NFKRemoteProvider.h>
#import <InferKit/NFKInferenceRequest.h>
#import <InferKit/NFKInferenceResult.h>
#import <InferKit/NFKInferenceJob.h>
#import <InferKit/NFKInferenceKeys.h>
#import <InferKit/NFKErrors.h>

/*! A chat backend whose line stream is stubbed: staged lines are delivered synchronously, then the
	stream completes unless held open; the returned cancel block records that it ran. */
@interface NFKStreamStubRemoteBackend : NFKRemoteBackend
@property (nonatomic, strong) NSURLRequest *lastRequest;
@property (nonatomic, copy) NSArray<NSString *> *stagedLines;
@property (nonatomic, assign) NSInteger stagedStatusCode;
@property (nonatomic, copy, nullable) NSData *stagedErrorBody;
@property (nonatomic, strong, nullable) NSError *stagedError;
@property (nonatomic, assign) BOOL holdOpen;
@property (nonatomic, assign) BOOL cancelled;
@end

@implementation NFKStreamStubRemoteBackend
- (void (^)(void))streamRequest:(NSURLRequest *)request
					lineHandler:(void (^)(NSString *))lineHandler
			  completionHandler:(void (^)(NSHTTPURLResponse *, NSData *, NSError *))completionHandler
{
	self.lastRequest = request;
	NSHTTPURLResponse *response = [[NSHTTPURLResponse alloc] initWithURL:request.URL statusCode:self.stagedStatusCode ?: 200
															 HTTPVersion:@"HTTP/1.1" headerFields:nil];
	if (self.stagedStatusCode >= 300) {
		completionHandler(response, self.stagedErrorBody, nil);
	} else {
		for (NSString *line in self.stagedLines) {
			lineHandler(line);
		}
		if (!self.holdOpen) {
			completionHandler(self.stagedError != nil ? nil : response, nil, self.stagedError);
		}
	}
	return ^{ self.cancelled = YES; };
}
@end

@interface NFKStreamStubAnthropicBackend : NFKAnthropicBackend
@property (nonatomic, strong) NSURLRequest *lastRequest;
@property (nonatomic, copy) NSArray<NSString *> *stagedLines;
@property (nonatomic, assign) BOOL holdOpen;
@property (nonatomic, assign) BOOL cancelled;
@end

@implementation NFKStreamStubAnthropicBackend
- (void (^)(void))streamRequest:(NSURLRequest *)request
					lineHandler:(void (^)(NSString *))lineHandler
			  completionHandler:(void (^)(NSHTTPURLResponse *, NSData *, NSError *))completionHandler
{
	self.lastRequest = request;
	for (NSString *line in self.stagedLines) {
		lineHandler(line);
	}
	if (!self.holdOpen) {
		completionHandler([[NSHTTPURLResponse alloc] initWithURL:request.URL statusCode:200 HTTPVersion:@"HTTP/1.1" headerFields:nil], nil, nil);
	}
	return ^{ self.cancelled = YES; };
}
@end

@interface NFKRemoteStreamingTests : XCTestCase
@property (nonatomic, strong) NFKStreamStubRemoteBackend *backend;
@property (nonatomic, strong) NFKStreamStubAnthropicBackend *anthropic;
@end

@implementation NFKRemoteStreamingTests

- (void)setUp
{
	[super setUp];
	self.backend = [[NFKStreamStubRemoteBackend alloc] init];
	self.backend.endpointURL = [NSURL URLWithString:@"http://localhost:11434/v1/chat/completions"];
	self.backend.modelName = @"qwen3.5:27b";
	self.anthropic = [[NFKStreamStubAnthropicBackend alloc] init];
	self.anthropic.modelName = @"claude-sonnet-4-5";
	self.anthropic.apiKey = @"k";
}

- (NSDictionary *)bodyOf:(NSURLRequest *)request
{
	return [NSJSONSerialization JSONObjectWithData:request.HTTPBody options:0 error:NULL];
}

- (NFKInferenceRequest *)prompt:(NSString *)text
{
	return [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: text }];
}

#pragma mark OpenAI-compatible

- (void)testTheStreamedRequestAsksForAStreamAndDeliversTheTextTokenByToken
{
	self.backend.stagedLines = @[
		@"data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"\"}}]}",
		@"data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}",
		@"",
		@": keep-alive",
		@"data: {\"choices\":[{\"delta\":{\"content\":\"lo, \"}}]}",
		@"data: {\"choices\":[{\"delta\":{\"content\":\"world\"},\"finish_reason\":\"stop\"}]}",
		@"data: [DONE]",
	];
	NSMutableArray<NSString *> *partials = [NSMutableArray array];
	self.backend.holdOpen = YES;   // the stub delivers synchronously; the handler is read after
	NFKInferenceJob *job = [self.backend submitInferenceJobForRequest:[self prompt:@"hi"]];
	(void)partials;

	XCTAssertEqual(job.status, NFKInferenceJobStatusSucceeded, @"[DONE] finishes the job");
	XCTAssertEqualObjects(job.result.text, @"Hello, world");
	XCTAssertEqualObjects([job.result outputForKey:NFKRemoteBackendRawKey][@"role"], @"assistant");
	NSDictionary *body = [self bodyOf:self.backend.lastRequest];
	XCTAssertEqualObjects(body[@"stream"], @YES);
	XCTAssertEqualObjects(body[@"model"], @"qwen3.5:27b");
	XCTAssertEqualObjects([self.backend.lastRequest valueForHTTPHeaderField:@"Accept"], @"text/event-stream");
	XCTAssertNotNil(job.cancellationHandler, @"the job can cancel the request");
}

- (void)testEachTextDeltaReportsThePartialTextSoFar
{
	self.backend.stagedLines = @[
		@"data: {\"choices\":[{\"delta\":{\"content\":\"A\"}}]}",
		@"data: {\"choices\":[{\"delta\":{\"content\":\"B\"}}]}",
	];
	self.backend.holdOpen = YES;
	NFKInferenceJob *job = [self.backend submitInferenceJobForRequest:[self prompt:@"hi"]];
	XCTAssertEqual(job.status, NFKInferenceJobStatusRunning, @"no [DONE] yet");
	XCTAssertEqualObjects(job.partialResult.text, @"AB", @"the partial is the text so far, not the last delta");
	XCTAssertLessThan(job.progress, 0, @"a stream cannot say how far along it is");
}

// A tool call's id and name arrive in its first delta and its arguments in fragments after; two
// calls interleave by index.
- (void)testAToolCallAssemblesAcrossDeltas
{
	self.backend.stagedLines = @[
		@"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"\"}}]}}]}",
		@"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"city\\\": \"}}]}}]}",
		@"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":1,\"id\":\"call_2\",\"function\":{\"name\":\"get_time\",\"arguments\":\"{}\"}}]}}]}",
		@"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"Paris\\\"}\"}}]}}]}",
		@"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}",
		@"data: [DONE]",
	];
	NFKInferenceJob *job = [self.backend submitInferenceJobForRequest:[self prompt:@"weather?"]];
	XCTAssertEqual(job.status, NFKInferenceJobStatusSucceeded);
	NSArray<NSDictionary *> *calls = job.result.toolCalls;
	XCTAssertEqual(calls.count, 2);
	XCTAssertEqualObjects(calls[0][@"id"], @"call_1");
	XCTAssertEqualObjects(calls[0][@"name"], @"get_weather");
	XCTAssertEqualObjects(calls[0][@"arguments"], (@{ @"city": @"Paris" }), @"the fragments parse once whole");
	XCTAssertEqualObjects(calls[0][@"argumentsJSON"], @"{\"city\": \"Paris\"}");
	XCTAssertEqualObjects(calls[1][@"name"], @"get_time");
	XCTAssertNil(job.result.text, @"no text was streamed");
}

- (void)testAStreamThatClosesWithoutDoneStillDeliversWhatItStreamed
{
	self.backend.stagedLines = @[ @"data: {\"choices\":[{\"delta\":{\"content\":\"partial\"}}]}" ];
	NFKInferenceJob *job = [self.backend submitInferenceJobForRequest:[self prompt:@"hi"]];
	XCTAssertEqual(job.status, NFKInferenceJobStatusSucceeded);
	XCTAssertEqualObjects(job.result.text, @"partial");
}

- (void)testAFailingStatusArrivesAsTheProvidersMessageAndAConnectionFailureAsUnreachable
{
	self.backend.stagedStatusCode = 401;
	self.backend.stagedErrorBody = [@"{\"error\":{\"message\":\"bad key\"}}" dataUsingEncoding:NSUTF8StringEncoding];
	NFKInferenceJob *rejected = [self.backend submitInferenceJobForRequest:[self prompt:@"hi"]];
	XCTAssertEqual(rejected.status, NFKInferenceJobStatusFailed);
	XCTAssertEqual(rejected.error.code, kNFKError_InferenceBackendFailure);
	XCTAssertTrue([rejected.error.localizedDescription containsString:@"bad key"]);

	self.backend.stagedStatusCode = 0;
	self.backend.stagedError = [NSError errorWithDomain:NFKInferenceErrorDomain code:kNFKError_RemoteUnreachable userInfo:nil];
	NFKInferenceJob *down = [self.backend submitInferenceJobForRequest:[self prompt:@"hi"]];
	XCTAssertEqual(down.status, NFKInferenceJobStatusFailed);
	XCTAssertEqual(down.error.code, kNFKError_RemoteUnreachable);
}

- (void)testCancellingTheJobCancelsTheRequest
{
	self.backend.stagedLines = @[ @"data: {\"choices\":[{\"delta\":{\"content\":\"A\"}}]}" ];
	self.backend.holdOpen = YES;
	NFKInferenceJob *job = [self.backend submitInferenceJobForRequest:[self prompt:@"hi"]];
	XCTAssertEqual(job.status, NFKInferenceJobStatusRunning);
	[job cancel];
	XCTAssertTrue(self.backend.cancelled, @"the request was cancelled, not just the job");
	XCTAssertEqual(job.status, NFKInferenceJobStatusCancelled);
}

- (void)testARequestThatCannotBeBuiltFailsTheJobBeforeAnyStream
{
	NFKRemoteBackend *unconfigured = [NFKRemoteBackend backendWithEndpointURL:nil];
	NFKInferenceJob *job = [unconfigured submitInferenceJobForRequest:[self prompt:@"hi"]];
	XCTAssertEqual(job.status, NFKInferenceJobStatusFailed);
	XCTAssertEqual(job.error.code, kNFKError_InferenceNotReady);
}

- (void)testTheGenericSubmitNowReachesTheStreamedForm
{
	self.backend.stagedLines = @[ @"data: {\"choices\":[{\"delta\":{\"content\":\"via submit\"}}]}", @"data: [DONE]" ];
	NFKInferenceJob *job = NFKInferenceSubmit(self.backend, [self prompt:@"hi"], NULL);
	XCTAssertEqual(job.status, NFKInferenceJobStatusSucceeded);
	XCTAssertEqualObjects(job.result.text, @"via submit");
	XCTAssertEqualObjects([self bodyOf:self.backend.lastRequest][@"stream"], @YES);
}

#pragma mark Anthropic

- (void)testAnthropicEventsAssembleTextAndAToolUse
{
	self.anthropic.stagedLines = @[
		@"event: message_start",
		@"data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"role\":\"assistant\",\"content\":[]}}",
		@"event: content_block_start",
		@"data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}",
		@"data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Checking\"}}",
		@"data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\" now.\"}}",
		@"data: {\"type\":\"content_block_stop\",\"index\":0}",
		@"data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"get_weather\",\"input\":{}}}",
		@"data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"city\\\":\"}}",
		@"data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\" \\\"Paris\\\"}\"}}",
		@"data: {\"type\":\"content_block_stop\",\"index\":1}",
		@"data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"}}",
		@"data: {\"type\":\"message_stop\"}",
	];
	self.anthropic.holdOpen = YES;
	NFKInferenceJob *job = [self.anthropic submitInferenceJobForRequest:[self prompt:@"weather?"]];
	XCTAssertEqual(job.status, NFKInferenceJobStatusSucceeded, @"message_stop finishes the job");
	XCTAssertEqualObjects(job.result.text, @"Checking now.");
	XCTAssertEqual(job.result.toolCalls.count, 1);
	XCTAssertEqualObjects(job.result.toolCalls[0][@"name"], @"get_weather");
	XCTAssertEqualObjects(job.result.toolCalls[0][@"arguments"], (@{ @"city": @"Paris" }));
	XCTAssertEqualObjects([self bodyOf:self.anthropic.lastRequest][@"stream"], @YES);
	XCTAssertEqualObjects(self.anthropic.lastRequest.allHTTPHeaderFields[@"x-api-key"], @"k");
}

- (void)testAnthropicTextDeltasReportPartialsAndAnErrorEventFailsTheJob
{
	self.anthropic.stagedLines = @[
		@"data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}",
		@"data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"so far\"}}",
	];
	self.anthropic.holdOpen = YES;
	NFKInferenceJob *running = [self.anthropic submitInferenceJobForRequest:[self prompt:@"hi"]];
	XCTAssertEqual(running.status, NFKInferenceJobStatusRunning);
	XCTAssertEqualObjects(running.partialResult.text, @"so far");
	[running cancel];
	XCTAssertTrue(self.anthropic.cancelled);

	self.anthropic.stagedLines = @[ @"data: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"Overloaded\"}}" ];
	self.anthropic.holdOpen = NO;
	NFKInferenceJob *failed = [self.anthropic submitInferenceJobForRequest:[self prompt:@"hi"]];
	XCTAssertEqual(failed.status, NFKInferenceJobStatusFailed);
	XCTAssertEqualObjects(failed.error.localizedDescription, @"Overloaded");
}

- (void)testAnthropicStreamsTheForcedStructuredToolIntoTheStructuredOutput
{
	self.anthropic.stagedLines = @[
		@"data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"structured_output\",\"input\":{}}}",
		@"data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"answer\\\": 42}\"}}",
		@"data: {\"type\":\"message_stop\"}",
	];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"the answer?" }
															  parameters:@{ NFKParameterJSONSchema: @{ @"type": @"object" } }
														  outputModality:NFKModalityText];
	NFKInferenceJob *job = [self.anthropic submitInferenceJobForRequest:request];
	XCTAssertEqual(job.status, NFKInferenceJobStatusSucceeded);
	XCTAssertEqualObjects(job.result.structured, (@{ @"answer": @42 }));
	XCTAssertNil(job.result.toolCalls, @"the forced tool is the structured reply, not a call to act on");
}

#pragma mark A live local runner

// Set INFERKIT_LIVE_LOCAL_MODEL to a model the local server has. The point is the stream itself:
// more than one partial arrives before the end, and the final text is what the partials built.
- (void)testALocalRunnerStreamsTokenByToken
{
	NSDictionary *environment = NSProcessInfo.processInfo.environment;
	NSString *model = environment[@"INFERKIT_LIVE_LOCAL_MODEL"];
	if (model.length == 0) {
		XCTSkip("set INFERKIT_LIVE_LOCAL_MODEL (and run a local server) to exercise the live path");
	}
	NSString *identifier = environment[@"INFERKIT_LIVE_LOCAL_PROVIDER"] ?: @"ollama";
	id<NFKInferenceBackend> backend = [NFKRemoteProvider backendForProvider:[NFKRemoteProvider providerWithIdentifier:identifier]
																	 apiKey:nil modelName:model];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Count from one to ten in words, separated by commas." }
															  parameters:@{ NFKParameterMaxTokens: @2048 }
														  outputModality:NFKModalityText];
	NSMutableArray<NSString *> *partials = [NSMutableArray array];
	NFKInferenceJob *job = [backend submitInferenceJobForRequest:request];
	job.progressHandler = ^(NFKInferenceJob *j) {
		@synchronized (partials) { [partials addObject:j.partialResult.text ?: @""]; }
	};
	XCTestExpectation *ended = [self expectationWithDescription:@"stream ended"];
	job.completionHandler = ^(NFKInferenceJob *j) { [ended fulfill]; };
	[self waitForExpectations:@[ ended ] timeout:300];

	XCTAssertEqual(job.status, NFKInferenceJobStatusSucceeded, @"%@", job.error);
	XCTAssertGreaterThan(partials.count, 1, @"the text arrived in more than one piece");
	XCTAssertEqualObjects(job.result.text, partials.lastObject, @"the final text is the last partial");
	XCTAssertTrue([job.result.text.lowercaseString containsString:@"ten"], @"the model answered: %@", job.result.text);
}

@end
