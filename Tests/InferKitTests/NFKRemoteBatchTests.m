//
//  NFKRemoteBatchTests.m
//  InferKitTests
//
//  Every batch style through stub transports: the lines each service reads, the record in the shared
//  vocabulary, the results read back by the submitting backend, pagination, cancel and delete, and the
//  provider factory. A live listing is gated on keys.
//

#import <XCTest/XCTest.h>
#import <InferKit/NFKRemoteBatch.h>
#import <InferKit/NFKRemoteFileStore.h>
#import <InferKit/NFKRemoteProvider.h>
#import <InferKit/NFKRemoteBackend.h>
#import <InferKit/NFKAnthropicBackend.h>
#import <InferKit/NFKRemoteResponsesBackend.h>
#import <InferKit/NFKGeminiGenerateContentBackend.h>
#import <InferKit/NFKRemoteEmbeddingBackend.h>
#import <InferKit/NFKInferenceRequest.h>
#import <InferKit/NFKInferenceResult.h>
#import <InferKit/NFKInferenceKeys.h>
#import <InferKit/NFKErrors.h>
#import <InferKit/NFKRemoteCompletionBackend.h>
#import <InferKit/NFKRemoteOCRBackend.h>
#import <InferKit/NFKRemoteModerationBackend.h>
#import <InferKit/NFKRemoteClassifierBackend.h>
#import <InferKit/NFKRemoteImageBackend.h>
#import <InferKit/NFKRemoteVideoBackend.h>
#import <InferKit/NFKRemoteTranscriptionBackend.h>
#import <InferKit/NFKImageCoding.h>
#import <InferKit/NFKAudioAsset.h>
#import <InferKit/NFKVideoAsset.h>
#import <InferKit/NFKMistralConversationsBackend.h>
#import <CoreGraphics/CoreGraphics.h>

/*! Answers staged bodies in order, each with its status, and records every request. */
@interface NFKStubBatchTransport : NSObject
@property (nonatomic, strong) NSMutableArray<NSURLRequest *> *requests;
@property (nonatomic, strong) NSMutableArray<NSArray *> *staged;
@end

@implementation NFKStubBatchTransport

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_requests = [NSMutableArray array];
		_staged = [NSMutableArray array];
	}
	return self;
}

- (void)stage:(NSString *)body
{
	[self.staged addObject:@[ @200, body ]];
}

- (void)stage:(NSString *)body status:(NSInteger)status
{
	[self.staged addObject:@[ @(status), body ]];
}

- (void)stage:(NSString *)body headers:(NSDictionary<NSString *, NSString *> *)headers
{
	[self.staged addObject:@[ @200, body, headers ]];
}

- (NSData *)answer:(NSURLRequest *)request response:(NSHTTPURLResponse **)outResponse
{
	[self.requests addObject:request];
	NSArray *next = self.staged.firstObject ?: @[ @200, @"{}" ];
	if (self.staged.count > 0) {
		[self.staged removeObjectAtIndex:0];
	}
	if (outResponse != NULL) {
		*outResponse = [[NSHTTPURLResponse alloc] initWithURL:request.URL statusCode:[next[0] integerValue] HTTPVersion:@"HTTP/1.1"
												 headerFields:next.count > 2 ? next[2] : nil];
	}
	return [next[1] dataUsingEncoding:NSUTF8StringEncoding];
}

- (id)bodyAt:(NSUInteger)index
{
	return [NSJSONSerialization JSONObjectWithData:self.requests[index].HTTPBody options:0 error:NULL];
}

@end

@interface NFKStubBatch : NFKRemoteBatch
@property (nonatomic, strong) NFKStubBatchTransport *transport;
@end

@implementation NFKStubBatch
- (NSData *)sendRequest:(NSURLRequest *)request response:(NSHTTPURLResponse **)outResponse error:(NSError **)outError
{
	return [self.transport answer:request response:outResponse];
}
@end

@interface NFKStubBatchFileStore : NFKRemoteFileStore
@property (nonatomic, strong) NFKStubBatchTransport *transport;
@end

@implementation NFKStubBatchFileStore
- (NSData *)sendRequest:(NSURLRequest *)request response:(NSHTTPURLResponse **)outResponse error:(NSError **)outError
{
	return [self.transport answer:request response:outResponse];
}
@end

/*! The inline limit past which Gemini's requests go as a file, lowered here to exercise the file. */
@interface NFKRemoteBatch (NFKTesting)
@property (nonatomic, assign) NSUInteger geminiInlineByteLimit;
@end

@interface NFKBatchStubVideoBackend : NFKRemoteVideoBackend
@property (nonatomic, strong) NSMutableArray<NSURLRequest *> *downloads;
@end

@implementation NFKBatchStubVideoBackend
- (NSData *)sendRequest:(NSURLRequest *)request response:(NSHTTPURLResponse **)outResponse error:(NSError **)outError
{
	if (self.downloads == nil) {
		self.downloads = [NSMutableArray array];
	}
	[self.downloads addObject:request];
	if (outResponse != NULL) {
		*outResponse = [[NSHTTPURLResponse alloc] initWithURL:request.URL statusCode:200 HTTPVersion:@"HTTP/1.1" headerFields:nil];
	}
	const uint8_t header[] = { 0, 0, 0, 0x18, 'f', 't', 'y', 'p', 'm', 'p', '4', '2' };
	return [NSData dataWithBytes:header length:sizeof(header)];
}
@end

@interface NFKRemoteBatchTests : XCTestCase
@property (nonatomic, strong, nullable) NSURL *scratch;
@end

@implementation NFKRemoteBatchTests

- (void)tearDown
{
	if (self.scratch != nil) {
		[NSFileManager.defaultManager removeItemAtURL:self.scratch error:NULL];
	}
	[super tearDown];
}

- (NSString *)onePixelPNGBase64
{
	CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
	CGContextRef context = CGBitmapContextCreate(NULL, 1, 1, 8, 4, colorSpace, kCGBitmapByteOrder32Little | kCGImageAlphaPremultipliedFirst);
	CGColorSpaceRelease(colorSpace);
	CGImageRef image = CGBitmapContextCreateImage(context);
	CGContextRelease(context);
	NSString *encoded = [[NFKImageCoding PNGDataForImage:(__bridge id)image] base64EncodedStringWithOptions:0];
	CGImageRelease(image);
	return encoded;
}

- (NFKStubBatch *)batchFor:(NFKRemoteProvider *)provider
{
	NFKRemoteBatch *made = [NFKRemoteBatch batchForProvider:provider apiKey:@"key"];
	NFKStubBatch *batch = [NFKStubBatch batchWithBaseURL:made.baseURL apiStyle:made.apiStyle];
	batch.apiKey = @"key";
	batch.transport = [[NFKStubBatchTransport alloc] init];
	if (made.fileStore != nil) {
		NFKStubBatchFileStore *store = [[NFKStubBatchFileStore alloc] init];
		store.endpointURL = made.fileStore.endpointURL;
		store.apiStyle = made.fileStore.apiStyle;
		store.defaultPurpose = made.fileStore.defaultPurpose;
		store.uploadPathComponent = made.fileStore.uploadPathComponent;
		store.apiKey = @"key";
		store.transport = batch.transport;
		batch.fileStore = store;
	}
	return batch;
}

- (NSDictionary<NSString *, NFKInferenceRequest *> *)twoRequests
{
	return @{ @"b-2": [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Second?" }],
			  @"a-1": [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"First?" }] };
}

- (NSArray<NSDictionary *> *)linesInUpload:(NSURLRequest *)request
{
	NSString *body = [[NSString alloc] initWithData:request.HTTPBody encoding:NSUTF8StringEncoding];
	NSMutableArray *lines = [NSMutableArray array];
	for (NSString *line in [body componentsSeparatedByString:@"\n"]) {
		id object = [line hasPrefix:@"{"] ? [NSJSONSerialization JSONObjectWithData:[line dataUsingEncoding:NSUTF8StringEncoding] options:0 error:NULL] : nil;
		if (object != nil) {
			[lines addObject:object];
		}
	}
	return lines;
}

#pragma mark OpenAI's file

- (void)testAnOpenAIBatchUploadsItsLinesCreatesTheBatchAndReadsBothFiles
{
	NFKStubBatch *batch = [self batchFor:NFKRemoteProvider.openAI];
	NFKRemoteBackend *backend = [NFKRemoteBackend backendWithEndpointURL:NFKRemoteProvider.openAI.endpointURL];
	backend.modelName = @"gpt-5.6-sol";
	[batch.transport stage:@"{\"id\":\"file-in\",\"object\":\"file\",\"bytes\":10,\"filename\":\"x.jsonl\",\"purpose\":\"batch\"}"];
	[batch.transport stage:@"{\"id\":\"batch_1\",\"object\":\"batch\",\"status\":\"validating\",\"request_counts\":{\"total\":2,\"completed\":0,\"failed\":0},\"created_at\":1792490400}"];
	NSError *error = nil;
	NFKRemoteBatchRecord *record = [batch submitRequests:self.twoRequests backend:backend error:&error];
	XCTAssertEqualObjects(record.identifier, @"batch_1", @"%@", error);
	XCTAssertEqual(record.status, NFKRemoteBatchStatusQueued);
	XCTAssertEqual(record.requestCount, 2);
	XCTAssertFalse(record.isFinished);

	NSURLRequest *upload = batch.transport.requests[0];
	XCTAssertEqualObjects(upload.URL.absoluteString, @"https://api.openai.com/v1/files");
	XCTAssertTrue([[[NSString alloc] initWithData:upload.HTTPBody encoding:NSUTF8StringEncoding] containsString:@"name=\"purpose\"\r\n\r\nbatch\r\n"]);
	NSArray<NSDictionary *> *lines = [self linesInUpload:upload];
	XCTAssertEqualObjects([lines valueForKey:@"custom_id"], (@[ @"a-1", @"b-2" ]), @"the lines go in identifier order");
	XCTAssertEqualObjects(lines[0][@"method"], @"POST");
	XCTAssertEqualObjects(lines[0][@"url"], @"/v1/chat/completions");
	XCTAssertEqualObjects(lines[0][@"body"][@"model"], @"gpt-5.6-sol");
	XCTAssertEqualObjects(lines[0][@"body"][@"messages"][0][@"content"], @"First?");
	NSURLRequest *create = batch.transport.requests[1];
	XCTAssertEqualObjects(create.URL.absoluteString, @"https://api.openai.com/v1/batches");
	XCTAssertEqualObjects([batch.transport bodyAt:1], (@{ @"input_file_id": @"file-in", @"endpoint": @"/v1/chat/completions", @"completion_window": @"24h" }));
	XCTAssertEqualObjects([create valueForHTTPHeaderField:@"Authorization"], @"Bearer key");

	[batch.transport stage:@"{\"id\":\"batch_1\",\"status\":\"completed\",\"output_file_id\":\"file-out\",\"error_file_id\":\"file-err\",\"request_counts\":{\"total\":2,\"completed\":1,\"failed\":1},\"completed_at\":1792494000}"];
	[batch.transport stage:(@"{\"id\":\"batch_req_1\",\"custom_id\":\"a-1\",\"response\":{\"status_code\":200,\"body\":"
							"{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"One.\"},\"finish_reason\":\"stop\"}]}},\"error\":null}\n")];
	[batch.transport stage:@"{\"id\":\"batch_req_2\",\"custom_id\":\"b-2\",\"response\":{\"status_code\":400,\"body\":{\"error\":{\"message\":\"bad\"}}},\"error\":null}\n"];
	NSDictionary<NSString *, NFKRemoteBatchResult *> *results = [batch resultsOfBatchWithIdentifier:@"batch_1" requests:self.twoRequests backend:backend error:&error];
	XCTAssertEqual(results.count, 2u, @"%@", error);
	XCTAssertEqualObjects(results[@"a-1"].result.text, @"One.");
	XCTAssertNil(results[@"a-1"].error);
	XCTAssertNil(results[@"b-2"].result);
	XCTAssertEqual(results[@"b-2"].error.code, kNFKError_InferenceRefused, @"a 400 maps as the transport maps it");
	XCTAssertEqualObjects(batch.transport.requests[3].URL.absoluteString, @"https://api.openai.com/v1/files/file-out/content");
	XCTAssertEqualObjects(batch.transport.requests[4].URL.absoluteString, @"https://api.openai.com/v1/files/file-err/content");
}

// Perplexity batches Agent API requests under /v1/responses, the route its batch reads.
- (void)testAPerplexityBatchSendsAgentRequestsToTheResponsesRoute
{
	NFKStubBatch *batch = [self batchFor:NFKRemoteProvider.perplexity];
	NFKRemoteResponsesBackend *agent = [NFKRemoteResponsesBackend backendForProvider:NFKRemoteProvider.perplexity apiKey:@"k" modelName:@"openai/gpt-5.5"];
	[batch.transport stage:@"{\"id\":\"file_1\",\"object\":\"file\",\"bytes\":10,\"filename\":\"x.jsonl\",\"purpose\":\"batch\"}"];
	[batch.transport stage:@"{\"id\":\"batch_9\",\"object\":\"batch\",\"status\":\"in_progress\",\"request_counts\":{\"total\":2,\"completed\":0,\"failed\":0}}"];
	NSError *error = nil;
	XCTAssertEqual([batch submitRequests:self.twoRequests backend:agent error:&error].status, NFKRemoteBatchStatusInProgress, @"%@", error);
	XCTAssertEqualObjects(batch.transport.requests[0].URL.absoluteString, @"https://api.perplexity.ai/v1/files");
	NSArray<NSDictionary *> *lines = [self linesInUpload:batch.transport.requests[0]];
	XCTAssertEqualObjects(lines[0][@"url"], @"/v1/responses");
	XCTAssertEqualObjects(lines[0][@"body"][@"model"], @"openai/gpt-5.5");
	XCTAssertEqualObjects([batch.transport bodyAt:1][@"endpoint"], @"/v1/responses");
	XCTAssertEqualObjects(batch.transport.requests[1].URL.absoluteString, @"https://api.perplexity.ai/v1/batches");
}

- (void)testTogethersLinesCarryTheBodyAloneAndItsJobRecordReads
{
	NFKStubBatch *batch = [self batchFor:NFKRemoteProvider.together];
	NFKRemoteBackend *backend = [NFKRemoteBackend backendWithEndpointURL:NFKRemoteProvider.together.endpointURL];
	backend.modelName = @"meta-llama/Llama-4";
	[batch.transport stage:@"{\"id\":\"file-t\",\"object\":\"file\",\"bytes\":10,\"filename\":\"x.jsonl\",\"purpose\":\"batch-api\"}"];
	[batch.transport stage:@"{\"job\":{\"id\":\"3fa85f64\",\"status\":\"VALIDATING\",\"created_at\":\"2026-10-20T10:00:00Z\",\"job_deadline\":\"2026-10-21T10:00:00Z\"},\"warning\":null}"];
	NSError *error = nil;
	NFKRemoteBatchRecord *record = [batch submitRequests:self.twoRequests backend:backend error:&error];
	XCTAssertEqualObjects(record.identifier, @"3fa85f64", @"%@", error);
	XCTAssertEqual(record.status, NFKRemoteBatchStatusQueued);
	XCTAssertEqualObjects(record.createdAt, [NSDate dateWithTimeIntervalSince1970:1792490400]);
	XCTAssertNotNil(record.expiresAt);
	XCTAssertEqualObjects(batch.transport.requests[0].URL.absoluteString, @"https://api.together.xyz/v1/files/upload");
	XCTAssertTrue([[[NSString alloc] initWithData:batch.transport.requests[0].HTTPBody encoding:NSUTF8StringEncoding] containsString:@"batch-api"]);
	NSArray<NSDictionary *> *lines = [self linesInUpload:batch.transport.requests[0]];
	XCTAssertEqual(lines[0].allKeys.count, 2u);
	XCTAssertNil(lines[0][@"url"]);
	XCTAssertEqualObjects([batch.transport bodyAt:1], (@{ @"input_file_id": @"file-t", @"endpoint": @"/v1/chat/completions" }));
	NSDictionary *completed = @{ @"id": @"x", @"status": @"COMPLETED" };
	XCTAssertEqual([NFKRemoteBatchRecord recordWithObject:completed apiStyle:NFKRemoteBatchAPIStyleTogether].status, NFKRemoteBatchStatusCompleted);
}

#pragma mark Inline styles

- (void)testAnAnthropicBatchGoesInlineWithItsBetasAndReadsTheResultsStream
{
	NFKStubBatch *batch = [self batchFor:NFKRemoteProvider.anthropic];
	NFKAnthropicBackend *claude = [NFKAnthropicBackend backendWithEndpointURL:NFKRemoteProvider.anthropic.endpointURL];
	claude.modelName = @"claude-opus-5-5";
	claude.maxTokens = 1024;
	NSDictionary *servers = @{ NFKParameterMCPServers: @[ @{ @"url": @"https://tools.example.com/mcp", @"name": @"tools" } ] };
	NSDictionary *requests = @{ @"q1": [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Hi" } parameters:servers outputModality:NFKModalityText] };
	[batch.transport stage:@"{\"id\":\"msgbatch_1\",\"type\":\"message_batch\",\"processing_status\":\"in_progress\",\"request_counts\":{\"processing\":1,\"succeeded\":0,\"errored\":0,\"canceled\":0,\"expired\":0},\"created_at\":\"2026-10-20T10:00:00Z\",\"expires_at\":\"2026-10-21T10:00:00Z\"}"];
	NSError *error = nil;
	NFKRemoteBatchRecord *record = [batch submitRequests:requests backend:claude error:&error];
	XCTAssertEqual(record.status, NFKRemoteBatchStatusInProgress, @"%@", error);
	XCTAssertEqual(record.requestCount, 1);
	NSURLRequest *create = batch.transport.requests[0];
	XCTAssertEqualObjects(create.URL.absoluteString, @"https://api.anthropic.com/v1/messages/batches");
	XCTAssertEqualObjects([create valueForHTTPHeaderField:@"x-api-key"], @"key");
	XCTAssertEqualObjects([create valueForHTTPHeaderField:@"anthropic-beta"], @"mcp-client-2025-11-20");
	NSDictionary *line = [batch.transport bodyAt:0][@"requests"][0];
	XCTAssertEqualObjects(line[@"custom_id"], @"q1");
	XCTAssertEqualObjects(line[@"params"][@"model"], @"claude-opus-5-5");
	XCTAssertEqualObjects(line[@"params"][@"max_tokens"], @1024);
	XCTAssertNil(line[@"params"][@"stream"]);

	[batch.transport stage:@"{\"id\":\"msgbatch_1\",\"processing_status\":\"ended\",\"results_url\":\"https://api.anthropic.com/v1/messages/batches/msgbatch_1/results\",\"request_counts\":{\"processing\":0,\"succeeded\":1,\"errored\":1,\"canceled\":0,\"expired\":1}}"];
	[batch.transport stage:(@"{\"custom_id\":\"q1\",\"result\":{\"type\":\"succeeded\",\"message\":{\"content\":[{\"type\":\"text\",\"text\":\"Hello.\"}],\"stop_reason\":\"end_turn\"}}}\n"
							"{\"custom_id\":\"q2\",\"result\":{\"type\":\"errored\",\"error\":{\"type\":\"error\",\"error\":{\"type\":\"invalid_request_error\",\"message\":\"max_tokens too large\"}}}}\n"
							"{\"custom_id\":\"q3\",\"result\":{\"type\":\"expired\"}}\n")];
	NSDictionary<NSString *, NFKRemoteBatchResult *> *results = [batch resultsOfBatchWithIdentifier:@"msgbatch_1" requests:requests backend:claude error:&error];
	XCTAssertEqualObjects(results[@"q1"].result.text, @"Hello.", @"%@", error);
	XCTAssertTrue([results[@"q2"].error.localizedDescription containsString:@"max_tokens too large"]);
	XCTAssertEqual(results[@"q3"].error.code, kNFKError_InferenceBackendFailure);
	XCTAssertEqualObjects(batch.transport.requests[2].URL.absoluteString, @"https://api.anthropic.com/v1/messages/batches/msgbatch_1/results");
	XCTAssertEqualObjects([batch.transport.requests[2] valueForHTTPHeaderField:@"x-api-key"], @"key");

	NFKRemoteBatchRecord *cancelled = [NFKRemoteBatchRecord recordWithObject:@{ @"id": @"m", @"processing_status": @"ended", @"cancel_initiated_at": @"2026-10-20T10:05:00Z" }
																	apiStyle:NFKRemoteBatchAPIStyleAnthropic];
	XCTAssertEqual(cancelled.status, NFKRemoteBatchStatusCancelled);
}

- (void)testAGeminiBatchKeysItsRequestsAndReadsTheInlinedResponses
{
	NFKStubBatch *batch = [self batchFor:NFKRemoteProvider.googleGemini];
	XCTAssertEqualObjects(batch.baseURL.absoluteString, @"https://generativelanguage.googleapis.com/v1beta");
	NFKGeminiGenerateContentBackend *gemini = [NFKGeminiGenerateContentBackend backendForProvider:NFKRemoteProvider.googleGemini apiKey:@"k" modelName:@"gemini-3.8-flash"];
	[batch.transport stage:@"{\"name\":\"batches/abc\",\"metadata\":{\"@type\":\"type.googleapis.com/google.ai.generativelanguage.v1main.GenerateContentBatch\",\"state\":\"BATCH_STATE_PENDING\",\"batchStats\":{\"requestCount\":\"2\"}}}"];
	NSError *error = nil;
	NFKRemoteBatchRecord *record = [batch submitRequests:self.twoRequests backend:gemini error:&error];
	XCTAssertEqualObjects(record.identifier, @"batches/abc", @"%@", error);
	XCTAssertEqual(record.status, NFKRemoteBatchStatusQueued);
	XCTAssertEqual(record.requestCount, 2);
	NSURLRequest *create = batch.transport.requests[0];
	XCTAssertEqualObjects(create.URL.absoluteString, @"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:batchGenerateContent");
	XCTAssertEqualObjects([create valueForHTTPHeaderField:@"x-goog-api-key"], @"key");
	NSArray *inline_ = [batch.transport bodyAt:0][@"batch"][@"inputConfig"][@"requests"][@"requests"];
	XCTAssertEqualObjects(inline_[0][@"metadata"], @{ @"key": @"a-1" });
	XCTAssertNotNil(inline_[0][@"request"][@"contents"]);

	[batch.transport stage:(@"{\"name\":\"batches/abc\",\"done\":true,\"metadata\":{\"state\":\"BATCH_STATE_SUCCEEDED\"},\"response\":{\"inlinedResponses\":{\"inlinedResponses\":["
							"{\"metadata\":{\"key\":\"a-1\"},\"response\":{\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"One.\"}]},\"finishReason\":\"STOP\"}]}},"
							"{\"error\":{\"code\":400,\"message\":\"blocked\"}}]}}}")];
	NSDictionary<NSString *, NFKRemoteBatchResult *> *results = [batch resultsOfBatchWithIdentifier:@"batches/abc" requests:self.twoRequests backend:gemini error:&error];
	XCTAssertEqualObjects(results[@"a-1"].result.text, @"One.", @"%@", error);
	XCTAssertTrue([results[@"b-2"].error.localizedDescription containsString:@"blocked"], @"an entry without a key is named by its position");
	XCTAssertEqualObjects(batch.transport.requests[1].URL.absoluteString, @"https://generativelanguage.googleapis.com/v1beta/batches/abc");

	XCTAssertTrue([batch cancelBatchWithIdentifier:@"batches/abc" error:NULL]);
	XCTAssertEqualObjects(batch.transport.requests[2].URL.absoluteString, @"https://generativelanguage.googleapis.com/v1beta/batches/abc:cancel");
}

- (void)testAMistralBatchGoesInlineAndReadsItsOutputFile
{
	NFKStubBatch *batch = [self batchFor:NFKRemoteProvider.mistral];
	NFKRemoteBackend *backend = (NFKRemoteBackend *)[NFKRemoteProvider backendForProvider:NFKRemoteProvider.mistral apiKey:@"k" modelName:@"mistral-small-4"];
	[batch.transport stage:@"{\"id\":\"9c2a\",\"object\":\"batch\",\"status\":\"QUEUED\",\"total_requests\":2,\"succeeded_requests\":0,\"failed_requests\":0,\"created_at\":1792490400}"];
	NSError *error = nil;
	XCTAssertEqual([batch submitRequests:self.twoRequests backend:backend error:&error].status, NFKRemoteBatchStatusQueued, @"%@", error);
	XCTAssertEqualObjects(batch.transport.requests[0].URL.absoluteString, @"https://api.mistral.ai/v1/batch/jobs");
	NSDictionary *create = [batch.transport bodyAt:0];
	XCTAssertEqualObjects(create[@"endpoint"], @"/v1/chat/completions");
	XCTAssertEqualObjects(create[@"model"], @"mistral-small-4");
	XCTAssertEqualObjects([create[@"requests"] valueForKey:@"custom_id"], (@[ @"a-1", @"b-2" ]));

	[batch.transport stage:@"{\"id\":\"9c2a\",\"status\":\"SUCCESS\",\"output_file\":\"f-out\",\"error_file\":null,\"total_requests\":2,\"succeeded_requests\":2,\"failed_requests\":0}"];
	[batch.transport stage:@"{\"id\":\"r\",\"custom_id\":\"b-2\",\"response\":{\"status_code\":200,\"body\":{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"Two.\"}}]}},\"error\":null}\n"];
	NSDictionary<NSString *, NFKRemoteBatchResult *> *results = [batch resultsOfBatchWithIdentifier:@"9c2a" requests:nil backend:backend error:&error];
	XCTAssertEqualObjects(results[@"b-2"].result.text, @"Two.", @"%@", error);
	XCTAssertEqualObjects(batch.transport.requests[2].URL.absoluteString, @"https://api.mistral.ai/v1/files/f-out/content");
	NSDictionary *timedOut = @{ @"id": @"x", @"status": @"TIMEOUT_EXCEEDED" };
	XCTAssertEqual([NFKRemoteBatchRecord recordWithObject:timedOut apiStyle:NFKRemoteBatchAPIStyleMistral].status, NFKRemoteBatchStatusExpired);
}

- (void)testAnOpenRouterBatchPutsItsRequestsLastReadsResultsInlineAndCannotCancel
{
	NFKStubBatch *batch = [self batchFor:NFKRemoteProvider.openRouter];
	NFKRemoteBackend *backend = [NFKRemoteBackend backendWithEndpointURL:NFKRemoteProvider.openRouter.endpointURL];
	backend.modelName = @"openai/gpt-5.6-sol";
	[batch.transport stage:@"{\"id\":\"or_1\",\"object\":\"batch\",\"status\":\"validating\",\"request_counts\":{\"total\":2,\"completed\":0,\"failed\":0}}" status:202];
	NSError *error = nil;
	XCTAssertNotNil([batch submitRequests:self.twoRequests backend:backend error:&error], @"%@", error);
	NSString *body = [[NSString alloc] initWithData:batch.transport.requests[0].HTTPBody encoding:NSUTF8StringEncoding];
	XCTAssertLessThan([body rangeOfString:@"\"model\""].location, [body rangeOfString:@"\"requests\""].location,
					  @"OpenRouter reads the fields before the requests");
	XCTAssertEqualObjects([batch.transport bodyAt:0][@"endpoint"], @"/v1/chat/completions");

	[batch.transport stage:[NSString stringWithFormat:@"{\"id\":\"or_1\",\"status\":\"completed\",\"results\":[{\"custom_id\":\"a-1\",\"response\":{\"status_code\":200,\"body\":%@},\"error\":null}]}",
							@"{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"One.\"}}]}"]];
	NSDictionary<NSString *, NFKRemoteBatchResult *> *results = [batch resultsOfBatchWithIdentifier:@"or_1" requests:nil backend:backend error:&error];
	XCTAssertEqualObjects(results[@"a-1"].result.text, @"One.", @"%@", error);
	XCTAssertFalse([batch cancelBatchWithIdentifier:@"or_1" error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceUnsupported);
	XCTAssertTrue([batch deleteBatchWithIdentifier:@"or_1" error:NULL]);
	XCTAssertEqualObjects(batch.transport.requests.lastObject.HTTPMethod, @"DELETE");
}

// xAI's batch is named, its requests added, and its status read from what is pending.
- (void)testAnXAIBatchIsCreatedByNameFilledAndReadFromItsPagedResults
{
	NFKStubBatch *batch = [self batchFor:NFKRemoteProvider.xAI];
	NFKRemoteResponsesBackend *grok = [NFKRemoteResponsesBackend backendForProvider:NFKRemoteProvider.xAI apiKey:@"k" modelName:@"grok-4.7"];
	[batch.transport stage:@"{\"batch_id\":\"xb_1\",\"name\":\"inferkit\",\"state\":{\"num_requests\":0,\"num_pending\":0}}"];
	[batch.transport stage:@"{}"];
	[batch.transport stage:@"{\"batch_id\":\"xb_1\",\"state\":{\"num_requests\":2,\"num_pending\":2,\"num_success\":0,\"num_error\":0,\"num_cancelled\":0}}"];
	NSError *error = nil;
	NFKRemoteBatchRecord *record = [batch submitRequests:self.twoRequests backend:grok error:&error];
	XCTAssertEqual(record.status, NFKRemoteBatchStatusInProgress, @"%@", error);
	XCTAssertNil(record.serviceStatus);
	XCTAssertTrue([[batch.transport bodyAt:0][@"name"] hasPrefix:@"inferkit-"]);
	XCTAssertEqualObjects(batch.transport.requests[1].URL.absoluteString, @"https://api.x.ai/v1/batches/xb_1/requests");
	NSDictionary *added = [batch.transport bodyAt:1][@"batch_requests"][0];
	XCTAssertEqualObjects(added[@"batch_request_id"], @"a-1");
	XCTAssertEqualObjects(added[@"batch_request"][@"responses"][@"model"], @"grok-4.7");

	[batch.transport stage:(@"{\"results\":[{\"batch_request_id\":\"a-1\",\"batch_result\":{\"response\":{\"responses\":{\"id\":\"r\",\"status\":\"completed\","
							"\"output\":[{\"type\":\"message\",\"content\":[{\"type\":\"output_text\",\"text\":\"One.\"}]}]}}}}],\"pagination_token\":\"p2\"}")];
	[batch.transport stage:@"{\"results\":[{\"batch_request_id\":\"b-2\",\"batch_result\":{\"error\":\"rate limited\"}}]}"];
	NSDictionary<NSString *, NFKRemoteBatchResult *> *results = [batch resultsOfBatchWithIdentifier:@"xb_1" requests:nil backend:grok error:&error];
	XCTAssertEqualObjects(results[@"a-1"].result.text, @"One.", @"%@", error);
	XCTAssertTrue([results[@"b-2"].error.localizedDescription containsString:@"rate limited"]);
	XCTAssertEqualObjects(batch.transport.requests[4].URL.absoluteString, @"https://api.x.ai/v1/batches/xb_1/results?limit=1000&pagination_token=p2");

	XCTAssertTrue([batch cancelBatchWithIdentifier:@"xb_1" error:NULL]);
	XCTAssertEqualObjects(batch.transport.requests.lastObject.URL.absoluteString, @"https://api.x.ai/v1/batches/xb_1:cancel");
	NFKRemoteBatchRecord *done = [NFKRemoteBatchRecord recordWithObject:@{ @"batch_id": @"x", @"state": @{ @"num_requests": @2, @"num_pending": @0, @"num_success": @1, @"num_error": @1 } }
															   apiStyle:NFKRemoteBatchAPIStyleXAI];
	XCTAssertEqual(done.status, NFKRemoteBatchStatusCompleted);
	XCTAssertEqual(done.failedCount, 1);
}

#pragma mark Gemini's embeddings and file input

- (void)testAGeminiEmbeddingBatchAsksForAsyncBatchEmbedContentAndReadsTheVectors
{
	NFKStubBatch *batch = [self batchFor:NFKRemoteProvider.googleGemini];
	NFKRemoteEmbeddingBackend *embedder = [NFKRemoteEmbeddingBackend backendForProvider:NFKRemoteProvider.googleGemini apiKey:@"k" modelName:@"gemini-embedding-2"];
	NSDictionary *requests = @{ @"e1": [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"alpha" } parameters:@{ @"dimensions": @3 } outputModality:NFKModalityText] };
	[batch.transport stage:@"{\"name\":\"batches/emb\",\"metadata\":{\"state\":\"BATCH_STATE_PENDING\"}}"];
	NSError *error = nil;
	XCTAssertEqualObjects([batch submitRequests:requests backend:embedder error:&error].identifier, @"batches/emb", @"%@", error);
	XCTAssertEqualObjects(batch.transport.requests[0].URL.absoluteString,
						  @"https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-2:asyncBatchEmbedContent");
	NSDictionary *line = [batch.transport bodyAt:0][@"batch"][@"inputConfig"][@"requests"][@"requests"][0];
	XCTAssertEqualObjects(line[@"request"][@"content"], (@{ @"parts": @[ @{ @"text": @"alpha" } ] }));
	XCTAssertEqualObjects(line[@"request"][@"outputDimensionality"], @3);
	XCTAssertEqualObjects(line[@"metadata"], @{ @"key": @"e1" });

	[batch.transport stage:(@"{\"name\":\"batches/emb\",\"metadata\":{\"state\":\"BATCH_STATE_SUCCEEDED\",\"output\":{\"inlinedResponses\":{\"inlinedResponses\":["
							"{\"metadata\":{\"key\":\"e1\"},\"response\":{\"embedding\":{\"values\":[0.5,-0.25,1]}}}]}}}}")];
	NSDictionary<NSString *, NFKRemoteBatchResult *> *results = [batch resultsOfBatchWithIdentifier:@"batches/emb" requests:requests backend:embedder error:&error];
	XCTAssertEqualObjects([results[@"e1"].result outputForKey:NFKOutputEmbedding], (@[ @0.5, @-0.25, @1 ]), @"%@", error);
}

// Past the inline limit the requests go as a JSONL file through the resumable upload, and the
// responses come back as a file of keyed lines.
- (void)testALargeGeminiBatchGoesAsAFileAndReadsTheResponsesFile
{
	NFKStubBatch *batch = [self batchFor:NFKRemoteProvider.googleGemini];
	batch.geminiInlineByteLimit = 0;
	NFKGeminiGenerateContentBackend *gemini = [NFKGeminiGenerateContentBackend backendForProvider:NFKRemoteProvider.googleGemini apiKey:@"k" modelName:@"gemini-3.8-flash"];
	[batch.transport stage:@"{}" headers:@{ @"X-Goog-Upload-URL": @"https://generativelanguage.googleapis.com/upload/v1beta/files?upload_id=u1" }];
	[batch.transport stage:@"{\"file\":{\"name\":\"files/in1\",\"mimeType\":\"jsonl\",\"state\":\"ACTIVE\"}}"];
	[batch.transport stage:@"{\"name\":\"batches/big\",\"metadata\":{\"state\":\"BATCH_STATE_PENDING\"}}"];
	NSError *error = nil;
	XCTAssertNotNil([batch submitRequests:self.twoRequests backend:gemini error:&error], @"%@", error);
	NSArray<NSDictionary *> *lines = [self linesInUpload:batch.transport.requests[1]];
	XCTAssertEqualObjects([lines valueForKey:@"key"], (@[ @"a-1", @"b-2" ]));
	XCTAssertNotNil(lines[0][@"request"][@"contents"]);
	XCTAssertEqualObjects([batch.transport bodyAt:2][@"batch"][@"inputConfig"], @{ @"fileName": @"files/in1" });

	[batch.transport stage:@"{\"name\":\"batches/big\",\"metadata\":{\"state\":\"BATCH_STATE_SUCCEEDED\",\"output\":{\"responsesFile\":\"files/out1\"}}}"];
	[batch.transport stage:(@"{\"response\":{\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"Two.\"}]},\"finishReason\":\"STOP\"}]},\"key\":\"b-2\"}\n"
							"{\"error\":{\"code\":400,\"message\":\"bad request\"},\"key\":\"a-1\"}\n")];
	NSDictionary<NSString *, NFKRemoteBatchResult *> *results = [batch resultsOfBatchWithIdentifier:@"batches/big" requests:self.twoRequests backend:gemini error:&error];
	XCTAssertEqualObjects(results[@"b-2"].result.text, @"Two.", @"%@", error);
	XCTAssertTrue([results[@"a-1"].error.localizedDescription containsString:@"bad request"]);
	XCTAssertEqualObjects(batch.transport.requests[4].URL.absoluteString,
						  @"https://generativelanguage.googleapis.com/download/v1beta/files/out1:download?alt=media");
}

#pragma mark xAI's media and Groq's audio

- (void)testAnXAIImageBatchNamesItsBodiesByRouteAndReadsTheImages
{
	NFKStubBatch *batch = [self batchFor:NFKRemoteProvider.xAI];
	NFKRemoteImageBackend *images = [NFKRemoteImageBackend backendForProvider:NFKRemoteProvider.xAI apiKey:@"k" modelName:@"grok-imagine-image"];
	NSDictionary *requests = @{ @"i1": [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"A red cube." }] };
	[batch.transport stage:@"{\"batch_id\":\"xb_i\",\"state\":{\"num_requests\":0,\"num_pending\":0}}"];
	[batch.transport stage:@"{}"];
	[batch.transport stage:@"{\"batch_id\":\"xb_i\",\"state\":{\"num_requests\":1,\"num_pending\":1}}"];
	NSError *error = nil;
	XCTAssertNotNil([batch submitRequests:requests backend:images error:&error], @"%@", error);
	NSDictionary *added = [batch.transport bodyAt:1][@"batch_requests"][0];
	XCTAssertEqualObjects(added[@"batch_request"][@"image_generation"][@"prompt"], @"A red cube.");

	[batch.transport stage:[NSString stringWithFormat:@"{\"results\":[{\"batch_request_id\":\"i1\",\"batch_result\":{\"response\":{\"image_generation\":{\"data\":[{\"b64_json\":\"%@\"}]}}}}]}",
							self.onePixelPNGBase64]];
	NSDictionary<NSString *, NFKRemoteBatchResult *> *results = [batch resultsOfBatchWithIdentifier:@"xb_i" requests:requests backend:images error:&error];
	XCTAssertNotNil([results[@"i1"].result outputForKey:NFKOutputImage], @"%@ %@", error, results[@"i1"].error);
}

- (void)testAnXAIVideoBatchAddsTheGenerationAndDownloadsTheFinishedClip
{
	NFKStubBatch *batch = [self batchFor:NFKRemoteProvider.xAI];
	NFKBatchStubVideoBackend *video = [NFKBatchStubVideoBackend backendForProvider:NFKRemoteProvider.xAI apiKey:@"k" modelName:@"grok-imagine-video"];
	self.scratch = [NSFileManager.defaultManager.temporaryDirectory URLByAppendingPathComponent:[@"NFKBatchVideo-" stringByAppendingString:NSUUID.UUID.UUIDString]];
	[NSFileManager.defaultManager createDirectoryAtURL:self.scratch withIntermediateDirectories:YES attributes:nil error:NULL];
	video.outputDirectoryURL = self.scratch;
	NSDictionary *requests = @{ @"v1": [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Waves." } parameters:@{} outputModality:NFKModalityVideo] };
	[batch.transport stage:@"{\"batch_id\":\"xb_v\",\"state\":{}}"];
	[batch.transport stage:@"{}"];
	[batch.transport stage:@"{\"batch_id\":\"xb_v\",\"state\":{\"num_requests\":1,\"num_pending\":1}}"];
	NSError *error = nil;
	XCTAssertNotNil([batch submitRequests:requests backend:video error:&error], @"%@", error);
	XCTAssertEqualObjects([batch.transport bodyAt:1][@"batch_requests"][0][@"batch_request"][@"video_generation"][@"prompt"], @"Waves.");

	[batch.transport stage:@"{\"results\":[{\"batch_request_id\":\"v1\",\"batch_result\":{\"response\":{\"video_generation\":{\"video\":{\"url\":\"https://vidgen.x.ai/clip.mp4\",\"duration\":8},\"model\":\"grok-imagine-video\"}}}}]}"];
	NSDictionary<NSString *, NFKRemoteBatchResult *> *results = [batch resultsOfBatchWithIdentifier:@"xb_v" requests:requests backend:video error:&error];
	XCTAssertTrue([[results[@"v1"].result outputForKey:NFKOutputVideo] isKindOfClass:NFKVideoAsset.class], @"%@ %@", error, results[@"v1"].error);
	XCTAssertEqualObjects(video.downloads.firstObject.URL.absoluteString, @"https://vidgen.x.ai/clip.mp4");
}

// Groq's batch endpoint is chat completions whatever its lines name; an audio line names its route
// and carries the clip's URL.
- (void)testAGroqAudioBatchSendsTheClipsURLAndNamesChatCompletionsAsItsEndpoint
{
	NFKStubBatch *batch = [self batchFor:NFKRemoteProvider.groq];
	NFKRemoteTranscriptionBackend *whisper = [NFKRemoteTranscriptionBackend backendForProvider:NFKRemoteProvider.groq apiKey:@"k" modelName:@"whisper-large-v3"];
	NFKAudioAsset *clip = [NFKAudioAsset audioAssetWithFileURL:[NSURL URLWithString:@"https://example.com/talk.wav"]];
	NSDictionary *requests = @{ @"t1": [NFKInferenceRequest requestWithInputs:@{ NFKInputAudio: clip }
																   parameters:@{ NFKParameterWordTimestamps: @YES, NFKParameterSourceLanguage: @"en" }
															   outputModality:NFKModalityText] };
	[batch.transport stage:@"{\"id\":\"file-g\",\"object\":\"file\",\"bytes\":10,\"filename\":\"x.jsonl\",\"purpose\":\"batch\"}"];
	[batch.transport stage:@"{\"id\":\"batch_g\",\"object\":\"batch\",\"status\":\"validating\"}"];
	NSError *error = nil;
	XCTAssertNotNil([batch submitRequests:requests backend:whisper error:&error], @"%@", error);
	NSDictionary *line = [self linesInUpload:batch.transport.requests[0]].firstObject;
	XCTAssertEqualObjects(line[@"url"], @"/v1/audio/transcriptions");
	XCTAssertEqualObjects(line[@"body"][@"url"], @"https://example.com/talk.wav");
	XCTAssertEqualObjects(line[@"body"][@"model"], @"whisper-large-v3");
	XCTAssertEqualObjects(line[@"body"][@"language"], @"en");
	XCTAssertEqualObjects(line[@"body"][@"timestamp_granularities"], (@[ @"segment", @"word" ]));
	XCTAssertEqualObjects([batch.transport bodyAt:1][@"endpoint"], @"/v1/chat/completions");

	NFKInferenceRequest *local = [NFKInferenceRequest requestWithInputs:@{ NFKInputAudio: [@"RIFF" dataUsingEncoding:NSUTF8StringEncoding] }];
	XCTAssertNil([batch submitRequests:@{ @"t2": local } backend:whisper error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceMissingInput, @"a batched clip goes by its URL");
}

#pragma mark OpenAI's images

// A generation goes as its JSON body; an edit, which a run sends as multipart, goes as the JSON edit
// body with its sources and mask as data URLs.
- (void)testOpenAIImageBatchesSendGenerationsAndJSONEdits
{
	NFKRemoteImageBackend *images = [NFKRemoteImageBackend backendForProvider:NFKRemoteProvider.openAI apiKey:@"k" modelName:@"gpt-image-2.5-flare"];
	NFKStubBatch *batch = [self batchFor:NFKRemoteProvider.openAI];
	[batch.transport stage:@"{\"id\":\"file-i\",\"object\":\"file\",\"bytes\":10,\"filename\":\"x.jsonl\",\"purpose\":\"batch\"}"];
	[batch.transport stage:@"{\"id\":\"batch_i\",\"status\":\"validating\"}"];
	NSError *error = nil;
	NSDictionary *generations = @{ @"g1": [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"A lighthouse." }] };
	XCTAssertNotNil([batch submitRequests:generations backend:images error:&error], @"%@", error);
	NSDictionary *line = [self linesInUpload:batch.transport.requests[0]].firstObject;
	XCTAssertEqualObjects(line[@"url"], @"/v1/images/generations");
	XCTAssertEqualObjects(line[@"body"][@"prompt"], @"A lighthouse.");
	XCTAssertEqualObjects([batch.transport bodyAt:1][@"endpoint"], @"/v1/images/generations");

	CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
	CGContextRef context = CGBitmapContextCreate(NULL, 2, 2, 8, 8, colorSpace, kCGBitmapByteOrder32Little | kCGImageAlphaPremultipliedFirst);
	CGColorSpaceRelease(colorSpace);
	CGImageRef source = CGBitmapContextCreateImage(context);
	CGContextRelease(context);
	NFKStubBatch *edits = [self batchFor:NFKRemoteProvider.openAI];
	[edits.transport stage:@"{\"id\":\"file-e\",\"object\":\"file\",\"bytes\":10,\"filename\":\"x.jsonl\",\"purpose\":\"batch\"}"];
	[edits.transport stage:@"{\"id\":\"batch_e\",\"status\":\"validating\"}"];
	NSDictionary *requests = @{ @"e1": [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Add snow.", NFKInputImage: (__bridge id)source,
																				 NFKInputMask: (__bridge id)source }] };
	CGImageRelease(source);
	XCTAssertNotNil([edits submitRequests:requests backend:images error:&error], @"%@", error);
	NSDictionary *edit = [self linesInUpload:edits.transport.requests[0]].firstObject;
	XCTAssertEqualObjects(edit[@"url"], @"/v1/images/edits");
	XCTAssertTrue([edit[@"body"][@"images"][0][@"image_url"] hasPrefix:@"data:image/png;base64,"]);
	XCTAssertTrue([edit[@"body"][@"mask"][@"image_url"] hasPrefix:@"data:image/png;base64,"]);
	XCTAssertEqualObjects(edit[@"body"][@"model"], @"gpt-image-2.5-flare");
	XCTAssertEqualObjects([edits.transport bodyAt:1][@"endpoint"], @"/v1/images/edits");

	[edits.transport stage:@"{\"id\":\"batch_e\",\"status\":\"completed\",\"output_file_id\":\"file-o\"}"];
	[edits.transport stage:[NSString stringWithFormat:@"{\"custom_id\":\"e1\",\"response\":{\"status_code\":200,\"body\":{\"data\":[{\"b64_json\":\"%@\"}]}}}\n",
							self.onePixelPNGBase64]];
	NFKRemoteBatchResult *result = [edits resultsOfBatchWithIdentifier:@"batch_e" requests:requests backend:images error:&error][@"e1"];
	XCTAssertNotNil([result.result outputForKey:NFKOutputImage], @"%@ %@", error, result.error);
}

#pragma mark Mistral's conversations

- (void)testAMistralConversationsBatchSendsStartBodiesAndReadsTheReplies
{
	NFKMistralConversationsBackend *conversations = [NFKMistralConversationsBackend backendForProvider:NFKRemoteProvider.mistral apiKey:@"k" modelName:@"mistral-medium-3-5"];
	NFKStubBatch *batch = [self batchFor:NFKRemoteProvider.mistral];
	NSDictionary *requests = @{ @"c1": [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Search the news." }
																   parameters:@{ NFKParameterTools: @[ @{ @"type": @"web_search" } ] } outputModality:NFKModalityText] };
	[batch.transport stage:@"{\"id\":\"job_c\",\"status\":\"QUEUED\"}"];
	NSError *error = nil;
	XCTAssertNotNil([batch submitRequests:requests backend:conversations error:&error], @"%@", error);
	NSDictionary *create = [batch.transport bodyAt:0];
	XCTAssertEqualObjects(create[@"endpoint"], @"/v1/conversations");
	NSDictionary *body = create[@"requests"][0][@"body"];
	XCTAssertEqualObjects(body[@"inputs"], @"Search the news.");
	XCTAssertEqualObjects(body[@"tools"], (@[ @{ @"type": @"web_search" } ]));

	[batch.transport stage:@"{\"id\":\"job_c\",\"status\":\"SUCCESS\",\"output_file\":\"f\"}"];
	[batch.transport stage:(@"{\"custom_id\":\"c1\",\"response\":{\"status_code\":200,\"body\":{\"conversation_id\":\"conv_b\",\"outputs\":["
							"{\"type\":\"message.output\",\"content\":\"Headlines.\"}]}},\"error\":null}\n")];
	NFKRemoteBatchResult *result = [batch resultsOfBatchWithIdentifier:@"job_c" requests:requests backend:conversations error:&error][@"c1"];
	XCTAssertEqualObjects(result.result.text, @"Headlines.", @"%@ %@", error, result.error);
	XCTAssertEqualObjects([result.result outputForKey:NFKOutputResponseIdentifier], @"conv_b");

	NFKInferenceRequest *continued = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"More." }
																 parameters:@{ NFKParameterPreviousResponseIdentifier: @"conv_b" } outputModality:NFKModalityText];
	XCTAssertNil([batch submitRequests:@{ @"c2": continued } backend:conversations error:&error], @"a continued conversation is no batch route");
	XCTAssertEqual(error.code, kNFKError_InferenceUnsupported);
}

#pragma mark Mistral's other endpoints

- (void)testMistralBatchesFIMOCRModerationClassificationAndTranscription
{
	NFKAudioAsset *clip = [NFKAudioAsset audioAssetWithFileURL:[NSURL URLWithString:@"https://example.com/talk.mp3"]];
	NSArray<NSArray *> *cases = @[
		@[ [NFKRemoteCompletionBackend backendForProvider:NFKRemoteProvider.mistral apiKey:@"k" modelName:@"codestral-latest"],
		   [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"def add(a, b):", NFKInputSuffix: @"return c" }], @"/v1/fim/completions",
		   @"{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"    c = a + b\"}}]}" ],
		@[ [NFKRemoteOCRBackend backendForProvider:NFKRemoteProvider.mistral apiKey:@"k" modelName:@"mistral-ocr-latest"],
		   [NFKInferenceRequest requestWithInputs:@{ NFKInputDocument: [NSURL URLWithString:@"https://example.com/paper.pdf"] }], @"/v1/ocr",
		   @"{\"pages\":[{\"index\":0,\"markdown\":\"# Title\"}],\"model\":\"mistral-ocr\",\"usage_info\":{\"pages_processed\":1}}" ],
		@[ [NFKRemoteModerationBackend backendForProvider:NFKRemoteProvider.mistral apiKey:@"k" modelName:@"mistral-moderation-2603"],
		   [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"hello" }], @"/v1/moderations",
		   @"{\"id\":\"m\",\"model\":\"mod\",\"results\":[{\"categories\":{\"violence\":false},\"category_scores\":{\"violence\":0.01}}]}" ],
		@[ [NFKRemoteClassifierBackend backendForProvider:NFKRemoteProvider.mistral apiKey:@"k" modelName:@"ft:classifier"],
		   [NFKInferenceRequest requestWithInputs:@{ NFKInputMessages: @[ @{ @"role": @"user", @"content": @"refund?" } ] }], @"/v1/chat/classifications",
		   @"{\"id\":\"c\",\"model\":\"cls\",\"results\":[{\"intent\":{\"scores\":{\"billing\":0.9,\"other\":0.1}}}]}" ],
		@[ [NFKRemoteTranscriptionBackend backendForProvider:NFKRemoteProvider.mistral apiKey:@"k" modelName:@"voxtral-mini-transcribe-3-0"],
		   [NFKInferenceRequest requestWithInputs:@{ NFKInputAudio: clip }], @"/v1/audio/transcriptions",
		   @"{\"model\":\"voxtral\",\"text\":\"Hello there.\",\"language\":\"en\",\"segments\":[]}" ],
	];
	for (NSArray *each in cases) {
		id<NFKInferenceBackend> backend = each[0];
		XCTAssertNotNil(backend, @"%@", each[2]);
		NFKStubBatch *batch = [self batchFor:NFKRemoteProvider.mistral];
		[batch.transport stage:@"{\"id\":\"job\",\"status\":\"QUEUED\"}"];
		NSError *error = nil;
		XCTAssertNotNil([batch submitRequests:@{ @"r": each[1] } backend:backend error:&error], @"%@: %@", each[2], error);
		NSDictionary *create = [batch.transport bodyAt:0];
		XCTAssertEqualObjects(create[@"endpoint"], each[2]);
		XCTAssertNotNil(create[@"requests"][0][@"body"], @"%@", each[2]);
		[batch.transport stage:@"{\"id\":\"job\",\"status\":\"SUCCESS\",\"output_file\":\"f\"}"];
		[batch.transport stage:[NSString stringWithFormat:@"{\"custom_id\":\"r\",\"response\":{\"status_code\":200,\"body\":%@},\"error\":null}\n", each[3]]];
		NFKRemoteBatchResult *result = [batch resultsOfBatchWithIdentifier:@"job" requests:@{ @"r": each[1] } backend:backend error:&error][@"r"];
		XCTAssertNotNil(result.result, @"%@: %@ %@", each[2], error, result.error);
	}
	NFKRemoteTranscriptionBackend *voxtral = [NFKRemoteTranscriptionBackend backendForProvider:NFKRemoteProvider.mistral apiKey:@"k" modelName:@"voxtral-mini-transcribe-3-0"];
	NFKStubBatch *batch = [self batchFor:NFKRemoteProvider.mistral];
	[batch.transport stage:@"{\"id\":\"job\",\"status\":\"QUEUED\"}"];
	[batch submitRequests:@{ @"r": [NFKInferenceRequest requestWithInputs:@{ NFKInputAudio: clip }] } backend:voxtral error:NULL];
	XCTAssertEqualObjects([batch.transport bodyAt:0][@"requests"][0][@"body"][@"file_url"], @"https://example.com/talk.mp3");
}

- (void)testABatchWhoseRequestsNameTwoRoutesIsRefused
{
	NFKStubBatch *batch = [self batchFor:NFKRemoteProvider.mistral];
	NFKRemoteModerationBackend *moderation = [NFKRemoteModerationBackend backendForProvider:NFKRemoteProvider.mistral apiKey:@"k" modelName:@"mistral-moderation-2603"];
	NSDictionary *requests = @{ @"text": [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"hello" }],
								@"chat": [NFKInferenceRequest requestWithInputs:@{ NFKInputMessages: @[ @{ @"role": @"user", @"content": @"hi" } ] }] };
	NSError *error = nil;
	XCTAssertNil([batch submitRequests:requests backend:moderation error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceUnsupported);
	XCTAssertTrue([error.localizedDescription containsString:@"one route"]);
	XCTAssertEqual(batch.transport.requests.count, 0u);
}

#pragma mark Lists, polling, and refusals

- (void)testTheListsFollowEachServicesPages
{
	NFKStubBatch *openAI = [self batchFor:NFKRemoteProvider.openAI];
	[openAI.transport stage:@"{\"data\":[{\"id\":\"b1\",\"status\":\"completed\"}],\"has_more\":true,\"last_id\":\"b1\"}"];
	[openAI.transport stage:@"{\"data\":[{\"id\":\"b2\",\"status\":\"failed\"}],\"has_more\":false}"];
	XCTAssertEqualObjects([[openAI batchesWithError:NULL] valueForKey:@"identifier"], (@[ @"b1", @"b2" ]));
	XCTAssertEqualObjects(openAI.transport.requests[1].URL.absoluteString, @"https://api.openai.com/v1/batches?after=b1&limit=100");

	NFKStubBatch *groq = [self batchFor:NFKRemoteProvider.groq];
	[groq.transport stage:@"{\"data\":[{\"id\":\"g1\",\"status\":\"in_progress\"}],\"paging\":{\"next_cursor\":\"c2\"}}"];
	[groq.transport stage:@"{\"data\":[],\"paging\":{}}"];
	XCTAssertEqual([groq batchesWithError:NULL].count, 1u);
	XCTAssertEqualObjects(groq.transport.requests[1].URL.absoluteString, @"https://api.groq.com/openai/v1/batches?cursor=c2&limit=100");

	NFKStubBatch *together = [self batchFor:NFKRemoteProvider.together];
	[together.transport stage:@"[{\"id\":\"t1\",\"status\":\"IN_PROGRESS\"}]"];
	XCTAssertEqual([together batchesWithError:NULL].firstObject.status, NFKRemoteBatchStatusInProgress);

	NFKStubBatch *gemini = [self batchFor:NFKRemoteProvider.googleGemini];
	[gemini.transport stage:@"{\"operations\":[{\"name\":\"batches/1\",\"metadata\":{\"state\":\"BATCH_STATE_RUNNING\"}}]}"];
	XCTAssertEqualObjects([gemini batchesWithError:NULL].firstObject.identifier, @"batches/1");
}

- (void)testAFinishedBatchEndsThePolling
{
	NFKStubBatch *batch = [self batchFor:NFKRemoteProvider.openAI];
	batch.pollInterval = 0;
	[batch.transport stage:@"{\"id\":\"b\",\"status\":\"in_progress\"}"];
	[batch.transport stage:@"{\"id\":\"b\",\"status\":\"finalizing\"}"];
	[batch.transport stage:@"{\"id\":\"b\",\"status\":\"completed\"}"];
	NFKRemoteBatchRecord *record = [batch recordWhenFinishedOfBatchWithIdentifier:@"b" timeout:60 error:NULL];
	XCTAssertEqual(record.status, NFKRemoteBatchStatusCompleted);
	XCTAssertEqual(batch.transport.requests.count, 3u);
}

- (void)testABackendOrARouteTheServiceDoesNotBatchIsRefusedBeforeAnythingIsSent
{
	NFKStubBatch *anthropic = [self batchFor:NFKRemoteProvider.anthropic];
	NFKRemoteBackend *chat = [NFKRemoteBackend backendWithEndpointURL:NFKRemoteProvider.openAI.endpointURL];
	chat.modelName = @"m";
	NSError *error = nil;
	XCTAssertNil([anthropic submitRequests:self.twoRequests backend:chat error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceUnsupported);
	XCTAssertEqual(anthropic.transport.requests.count, 0u);

	NFKStubBatch *openAI = [self batchFor:NFKRemoteProvider.openAI];
	id<NFKInferenceBackend> notBatchable = [NFKRemoteProvider backendForProvider:NFKRemoteProvider.typeSafe apiKey:@"k" modelName:@"jev-latest"];
	XCTAssertNil([openAI submitRequests:self.twoRequests backend:notBatchable error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceUnsupported);
	XCTAssertNil([openAI submitRequests:@{} backend:chat error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceMissingInput);
	XCTAssertFalse([openAI deleteBatchWithIdentifier:@"b" error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceUnsupported);
	XCTAssertEqual(openAI.transport.requests.count, 0u);
}

- (void)testTheFactoryServesTheBatchingPresetsWithTheirFileStores
{
	NSDictionary<NSString *, NSNumber *> *styles = @{
		@"openai": @(NFKRemoteBatchAPIStyleOpenAI), @"groq": @(NFKRemoteBatchAPIStyleOpenAI), @"modelstudio": @(NFKRemoteBatchAPIStyleOpenAI),
		@"perplexity": @(NFKRemoteBatchAPIStyleOpenAI), @"together": @(NFKRemoteBatchAPIStyleTogether),
		@"anthropic": @(NFKRemoteBatchAPIStyleAnthropic), @"gemini": @(NFKRemoteBatchAPIStyleGemini),
		@"mistral": @(NFKRemoteBatchAPIStyleMistral), @"openrouter": @(NFKRemoteBatchAPIStyleOpenRouter), @"xai": @(NFKRemoteBatchAPIStyleXAI) };
	for (NSString *identifier in styles) {
		NFKRemoteBatch *batch = [NFKRemoteBatch batchForProvider:[NFKRemoteProvider providerWithIdentifier:identifier] apiKey:@"k"];
		XCTAssertEqual(batch.apiStyle, styles[identifier].integerValue, @"%@", identifier);
	}
	NFKRemoteBatch *perplexity = [NFKRemoteBatch batchForProvider:NFKRemoteProvider.perplexity apiKey:@"k"];
	XCTAssertEqualObjects(perplexity.fileStore.endpointURL.absoluteString, @"https://api.perplexity.ai/v1/files");
	XCTAssertEqualObjects(perplexity.fileStore.defaultPurpose, @"batch");
	XCTAssertEqualObjects([NFKRemoteBatch batchForProvider:NFKRemoteProvider.modelStudio apiKey:@"k"].fileStore.endpointURL.absoluteString,
						  @"https://dashscope-intl.aliyuncs.com/compatible-mode/v1/files");
	for (NFKRemoteProvider *provider in @[ NFKRemoteProvider.deepSeek, NFKRemoteProvider.typeSafe, NFKRemoteProvider.ollama ]) {
		XCTAssertNil([NFKRemoteBatch batchForProvider:provider apiKey:@"k"], @"%@ batches nothing", provider.identifier);
	}
}

#pragma mark Live

// Lists the batches when a key is present; listing writes nothing. Each list path answers 401 without
// a key and a path beside it 404, so the paths are verified to exist.
- (void)testALiveListingAnswers
{
	NSDictionary<NSString *, NSString *> *environment = NSProcessInfo.processInfo.environment;
	NSDictionary<NSString *, NFKRemoteProvider *> *keyed = @{ @"INFERKIT_OPENAI_API_KEY": NFKRemoteProvider.openAI,
															  @"INFERKIT_ANTHROPIC_API_KEY": NFKRemoteProvider.anthropic,
															  @"INFERKIT_PERPLEXITY_API_KEY": NFKRemoteProvider.perplexity };
	BOOL ran = NO;
	for (NSString *variable in keyed) {
		NSString *key = environment[variable];
		if (key.length == 0) {
			continue;
		}
		ran = YES;
		NSError *error = nil;
		XCTAssertNotNil([[NFKRemoteBatch batchForProvider:keyed[variable] apiKey:key] batchesWithError:&error], @"%@: %@", variable, error);
	}
	if (!ran) {
		XCTSkip("set INFERKIT_OPENAI_API_KEY, INFERKIT_ANTHROPIC_API_KEY, or INFERKIT_PERPLEXITY_API_KEY to exercise the live path");
	}
}

@end
