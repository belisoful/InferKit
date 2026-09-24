//
//  NFKRemoteCompletionBackendTests.m
//  InferKitTests
//
//  Raw continuation and fill-in-the-middle through stubbed transports: each style's path and field
//  names, the three reply shapes, and the streamed continuation.
//

#import <XCTest/XCTest.h>
#import <InferKit/NFKRemoteCompletionBackend.h>
#import <InferKit/NFKRemoteProvider.h>
#import <InferKit/NFKInferenceRequest.h>
#import <InferKit/NFKInferenceResult.h>
#import <InferKit/NFKInferenceJob.h>
#import <InferKit/NFKInferenceKeys.h>
#import <InferKit/NFKErrors.h>

@interface NFKStubCompletionBackend : NFKRemoteCompletionBackend
@property (nonatomic, strong) NSURLRequest *lastRequest;
@property (nonatomic, copy) NSString *stagedBody;
@property (nonatomic, copy) NSArray<NSString *> *stagedLines;
@end

@implementation NFKStubCompletionBackend

- (NSData *)sendRequest:(NSURLRequest *)request response:(NSHTTPURLResponse **)outResponse error:(NSError **)outError
{
	self.lastRequest = request;
	if (outResponse != NULL) {
		*outResponse = [[NSHTTPURLResponse alloc] initWithURL:request.URL statusCode:200 HTTPVersion:@"HTTP/1.1" headerFields:nil];
	}
	return [self.stagedBody dataUsingEncoding:NSUTF8StringEncoding];
}

- (void (^)(void))streamRequest:(NSURLRequest *)request
					lineHandler:(void (^)(NSString *))lineHandler
			  completionHandler:(void (^)(NSHTTPURLResponse *, NSData *, NSError *))completionHandler
{
	self.lastRequest = request;
	for (NSString *line in self.stagedLines) {
		lineHandler(line);
	}
	completionHandler([[NSHTTPURLResponse alloc] initWithURL:request.URL statusCode:200 HTTPVersion:@"HTTP/1.1" headerFields:nil], nil, nil);
	return ^{};
}

- (NSDictionary *)decodedRequestBody
{
	return [NSJSONSerialization JSONObjectWithData:self.lastRequest.HTTPBody options:0 error:NULL];
}

@end

@interface NFKRemoteCompletionBackendTests : XCTestCase
@end

@implementation NFKRemoteCompletionBackendTests

- (NFKStubCompletionBackend *)stubFor:(NFKRemoteProvider *)provider model:(NSString *)model
{
	NFKRemoteCompletionBackend *made = [NFKRemoteCompletionBackend backendForProvider:provider apiKey:@"k" modelName:model];
	NFKStubCompletionBackend *stub = [NFKStubCompletionBackend backendWithEndpointURL:made.endpointURL];
	stub.apiStyle = made.apiStyle;
	stub.modelName = model;
	stub.apiKey = @"k";
	return stub;
}

- (void)testEachProviderGetsItsOwnPathAndTheChatOnlyOnesNone
{
	XCTAssertEqualObjects([NFKRemoteCompletionBackend backendForProvider:NFKRemoteProvider.deepSeek apiKey:@"k" modelName:@"m"].endpointURL.absoluteString,
						  @"https://api.deepseek.com/beta/completions");
	XCTAssertEqualObjects([NFKRemoteCompletionBackend backendForProvider:NFKRemoteProvider.mistral apiKey:@"k" modelName:@"m"].endpointURL.absoluteString,
						  @"https://api.mistral.ai/v1/fim/completions");
	XCTAssertEqualObjects([NFKRemoteCompletionBackend backendForProvider:NFKRemoteProvider.together apiKey:@"k" modelName:@"m"].endpointURL.absoluteString,
						  @"https://api.together.xyz/v1/completions");
	for (NFKRemoteProvider *provider in @[ NFKRemoteProvider.anthropic, NFKRemoteProvider.googleGemini, NFKRemoteProvider.xAI, NFKRemoteProvider.groq ]) {
		XCTAssertNil([NFKRemoteCompletionBackend backendForProvider:provider apiKey:@"k" modelName:@"m"], @"%@", provider.identifier);
	}
}

- (void)testDeepSeekFillsTheMiddleAndReadsChoicesText
{
	NFKStubCompletionBackend *deepSeek = [self stubFor:NFKRemoteProvider.deepSeek model:@"deepseek-flash"];
	deepSeek.stagedBody = @"{\"object\":\"text_completion\",\"choices\":[{\"text\":\"    return a + b\\n\"}]}";
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"def add(a, b):\n", NFKInputSuffix: @"\nprint(add(1, 2))" }
															   parameters:@{ NFKParameterMaxTokens: @64, NFKParameterStopSequences: @[ @"\n\n" ] }];
	NSError *error = nil;
	NFKInferenceResult *result = [deepSeek runInferenceForRequest:request error:&error];
	XCTAssertNotNil(result, @"%@", error);
	NSDictionary *body = [deepSeek decodedRequestBody];
	XCTAssertEqualObjects(body[@"prompt"], @"def add(a, b):\n");
	XCTAssertEqualObjects(body[@"suffix"], @"\nprint(add(1, 2))");
	XCTAssertEqualObjects(body[@"max_tokens"], @64);
	XCTAssertEqualObjects(body[@"stop"], (@[ @"\n\n" ]));
	XCTAssertEqualObjects(result.text, @"    return a + b\n");
}

- (void)testMistralFIMReadsTheChatShapedReplyAndNamesTheSeedItsWay
{
	NFKStubCompletionBackend *mistral = [self stubFor:NFKRemoteProvider.mistral model:@"codestral-latest"];
	mistral.stagedBody = @"{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"x = 1\"}}]}";
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"# set x\n", NFKInputSuffix: @"\nprint(x)" }
															   parameters:@{ NFKParameterSeed: @7 }];
	NFKInferenceResult *result = [mistral runInferenceForRequest:request error:NULL];
	XCTAssertEqualObjects([mistral decodedRequestBody][@"random_seed"], @7);
	XCTAssertEqualObjects(result.text, @"x = 1");
}

- (void)testLlamaCppInfillsWithPrefixAndSuffixAndContinuesOnCompletion
{
	NFKStubCompletionBackend *llama = [self stubFor:NFKRemoteProvider.llamaCpp model:@"qwen-coder"];
	llama.stagedBody = @"{\"content\":\"middle\",\"stop\":true}";
	NFKInferenceRequest *infill = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"before", NFKInputSuffix: @"after" }
															  parameters:@{ NFKParameterMaxTokens: @16 }];
	NFKInferenceResult *result = [llama runInferenceForRequest:infill error:NULL];
	XCTAssertEqualObjects(llama.lastRequest.URL.absoluteString, @"http://localhost:8080/infill");
	NSDictionary *body = [llama decodedRequestBody];
	XCTAssertEqualObjects(body[@"input_prefix"], @"before");
	XCTAssertEqualObjects(body[@"input_suffix"], @"after");
	XCTAssertEqualObjects(body[@"n_predict"], @16);
	XCTAssertEqualObjects(result.text, @"middle");

	[llama runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Once upon" }] error:NULL];
	XCTAssertEqualObjects(llama.lastRequest.URL.absoluteString, @"http://localhost:8080/completion");
}

- (void)testAStreamedContinuationGrowsThePartialResult
{
	NFKStubCompletionBackend *together = [self stubFor:NFKRemoteProvider.together model:@"m"];
	together.stagedLines = @[ @"data: {\"choices\":[{\"text\":\"Hel\"}]}", @"data: {\"choices\":[{\"text\":\"lo\"}]}", @"data: [DONE]" ];
	NFKInferenceJob *job = [together submitInferenceJobForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Say" }]];
	XCTAssertEqual(job.status, NFKInferenceJobStatusSucceeded);
	XCTAssertEqualObjects(job.result.text, @"Hello");
	XCTAssertEqualObjects([together decodedRequestBody][@"stream"], @YES);
}

- (void)testAMissingPromptOrEndpointIsRefused
{
	NFKStubCompletionBackend *empty = [NFKStubCompletionBackend backendWithEndpointURL:nil];
	NSError *error = nil;
	XCTAssertNil([empty runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"x" }] error:&error]);
	XCTAssertEqual(error.code, (NSInteger)kNFKError_InferenceNotReady);
	NFKStubCompletionBackend *backend = [self stubFor:NFKRemoteProvider.together model:@"m"];
	XCTAssertNil([backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{}] error:&error]);
	XCTAssertEqual(error.code, (NSInteger)kNFKError_InferenceMissingInput);
}

@end
