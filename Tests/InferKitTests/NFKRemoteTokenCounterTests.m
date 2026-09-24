//
//  NFKRemoteTokenCounterTests.m
//  InferKitTests
//
//  Token counting and classification through stubbed transports: each style's path, body, key
//  header, and reply field.
//

#import <XCTest/XCTest.h>
#import <InferKit/NFKRemoteTokenCounter.h>
#import <InferKit/NFKRemoteClassifierBackend.h>
#import <InferKit/NFKRemoteProvider.h>
#import <InferKit/NFKClassification.h>
#import <InferKit/NFKInferenceRequest.h>
#import <InferKit/NFKInferenceResult.h>
#import <InferKit/NFKInferenceKeys.h>
#import <InferKit/NFKErrors.h>

@interface NFKStubTokenCounter : NFKRemoteTokenCounter
@property (nonatomic, strong) NSURLRequest *lastRequest;
@property (nonatomic, copy) NSString *stagedBody;
@end

@implementation NFKStubTokenCounter
- (NSData *)sendRequest:(NSURLRequest *)request response:(NSHTTPURLResponse **)outResponse error:(NSError **)outError
{
	self.lastRequest = request;
	if (outResponse != NULL) {
		*outResponse = [[NSHTTPURLResponse alloc] initWithURL:request.URL statusCode:200 HTTPVersion:@"HTTP/1.1" headerFields:nil];
	}
	return [self.stagedBody dataUsingEncoding:NSUTF8StringEncoding];
}
@end

@interface NFKStubClassifier : NFKRemoteClassifierBackend
@property (nonatomic, strong) NSURLRequest *lastRequest;
@property (nonatomic, copy) NSString *stagedBody;
@end

@implementation NFKStubClassifier
- (NSData *)sendRequest:(NSURLRequest *)request response:(NSHTTPURLResponse **)outResponse error:(NSError **)outError
{
	self.lastRequest = request;
	if (outResponse != NULL) {
		*outResponse = [[NSHTTPURLResponse alloc] initWithURL:request.URL statusCode:200 HTTPVersion:@"HTTP/1.1" headerFields:nil];
	}
	return [self.stagedBody dataUsingEncoding:NSUTF8StringEncoding];
}
@end

@interface NFKRemoteTokenCounterTests : XCTestCase
@end

@implementation NFKRemoteTokenCounterTests

- (NFKStubTokenCounter *)counterFor:(NFKRemoteProvider *)provider model:(NSString *)model
{
	NFKRemoteTokenCounter *made = [NFKRemoteTokenCounter counterForProvider:provider apiKey:@"k" modelName:model];
	NFKStubTokenCounter *stub = [[NFKStubTokenCounter alloc] init];
	stub.endpointURL = made.endpointURL;
	stub.apiStyle = made.apiStyle;
	stub.apiKey = @"k";
	stub.modelName = model;
	return stub;
}

- (NSDictionary *)bodyOf:(NSURLRequest *)request
{
	return [NSJSONSerialization JSONObjectWithData:request.HTTPBody options:0 error:NULL];
}

- (void)testAnthropicCountsTheMessagesWithTheSystemTurnLifted
{
	NFKStubTokenCounter *counter = [self counterFor:NFKRemoteProvider.anthropic model:@"claude-opus-5-5"];
	counter.stagedBody = @"{\"input_tokens\":42}";
	NSArray *messages = @[ @{ @"role": @"system", @"content": @"Be brief." }, @{ @"role": @"user", @"content": @"Hi" } ];
	NSError *error = nil;
	NSNumber *count = [counter tokenCountForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputMessages: messages }] error:&error];
	XCTAssertEqualObjects(count, @42, @"%@", error);
	XCTAssertEqualObjects(counter.lastRequest.URL.absoluteString, @"https://api.anthropic.com/v1/messages/count_tokens");
	XCTAssertEqualObjects([counter.lastRequest valueForHTTPHeaderField:@"x-api-key"], @"k");
	NSDictionary *body = [self bodyOf:counter.lastRequest];
	XCTAssertEqualObjects(body[@"system"], @"Be brief.");
	XCTAssertEqual([body[@"messages"] count], 1);
}

- (void)testGeminiCountsThroughTheModelsPath
{
	NFKStubTokenCounter *counter = [self counterFor:NFKRemoteProvider.googleGemini model:@"gemini-3.8-flash"];
	counter.stagedBody = @"{\"totalTokens\":7}";
	XCTAssertEqualObjects([counter tokenCountForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"hello" }] error:NULL], @7);
	XCTAssertEqualObjects(counter.lastRequest.URL.absoluteString,
						  @"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:countTokens");
	XCTAssertEqualObjects([counter.lastRequest valueForHTTPHeaderField:@"x-goog-api-key"], @"k");
}

- (void)testXAIAndLlamaCppReturnTheIdsAndTheCountIsTheirLength
{
	NFKStubTokenCounter *xai = [self counterFor:NFKRemoteProvider.xAI model:@"grok-4.7"];
	xai.stagedBody = @"{\"token_ids\":[{\"token_id\":1,\"string_token\":\"he\"},{\"token_id\":2,\"string_token\":\"llo\"}]}";
	XCTAssertEqualObjects([xai tokenIdentifiersForText:@"hello" error:NULL], (@[ @1, @2 ]));
	XCTAssertEqualObjects(xai.lastRequest.URL.absoluteString, @"https://api.x.ai/v1/tokenize-text");

	NFKStubTokenCounter *llama = [self counterFor:NFKRemoteProvider.llamaCpp model:@"m"];
	llama.stagedBody = @"{\"tokens\":[5,6,7]}";
	XCTAssertEqualObjects([llama tokenCountForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"abc" }] error:NULL], @3);
	XCTAssertEqualObjects(llama.lastRequest.URL.absoluteString, @"http://localhost:8080/tokenize");

	NSError *error = nil;
	XCTAssertNil([[self counterFor:NFKRemoteProvider.anthropic model:@"m"] tokenIdentifiersForText:@"x" error:&error]);
	XCTAssertEqual(error.code, (NSInteger)kNFKError_InferenceUnsupported);
	XCTAssertNil([NFKRemoteTokenCounter counterForProvider:NFKRemoteProvider.openAI apiKey:@"k" modelName:@"m"]);
}

#pragma mark Classification

- (void)testMistralClassifiesAConversationOnItsChatPathAndNamesSeveralTargets
{
	NFKRemoteClassifierBackend *made = [NFKRemoteClassifierBackend backendForProvider:NFKRemoteProvider.mistral apiKey:@"k" modelName:@"ft:classifier"];
	NFKStubClassifier *classifier = [[NFKStubClassifier alloc] init];
	classifier.endpointURL = made.endpointURL;
	classifier.apiStyle = made.apiStyle;
	classifier.modelName = @"ft:classifier";
	classifier.stagedBody = @"{\"results\":[{\"sentiment\":{\"scores\":{\"positive\":0.9,\"negative\":0.1}},\"topic\":{\"scores\":{\"sports\":0.3}}}]}";
	NSArray *messages = @[ @{ @"role": @"user", @"content": @"Great game!" } ];
	NFKInferenceResult *result = [classifier runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputMessages: messages }] error:NULL];
	XCTAssertEqualObjects(classifier.lastRequest.URL.absoluteString, @"https://api.mistral.ai/v1/chat/classifications");
	XCTAssertEqualObjects([self bodyOf:classifier.lastRequest][@"input"], (@{ @"messages": messages }));
	NSArray<NFKClassification *> *classes = result.classifications;
	XCTAssertEqualObjects(classes.firstObject.label, @"sentiment/positive");
	XCTAssertEqual(classes.count, 3);
}

- (void)testVLLMClassifiesTextAndLabelsTheTopClass
{
	NFKRemoteClassifierBackend *made = [NFKRemoteClassifierBackend backendForProvider:NFKRemoteProvider.vLLM apiKey:nil modelName:@"m"];
	XCTAssertEqualObjects(made.endpointURL.absoluteString, @"http://localhost:8000/classify");
	NFKStubClassifier *classifier = [[NFKStubClassifier alloc] init];
	classifier.endpointURL = made.endpointURL;
	classifier.apiStyle = made.apiStyle;
	classifier.stagedBody = @"{\"data\":[{\"index\":0,\"label\":\"spam\",\"probs\":[0.2,0.8],\"num_classes\":2}]}";
	NFKInferenceResult *result = [classifier runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"WIN NOW" }] error:NULL];
	XCTAssertEqualObjects(result.classifications.firstObject.label, @"spam");
	XCTAssertEqual(result.classifications.firstObject.classIndex, 1);
	XCTAssertNil([NFKRemoteClassifierBackend backendForProvider:NFKRemoteProvider.openAI apiKey:@"k" modelName:@"m"]);
}

@end
