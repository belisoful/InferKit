//
//  NFKRemoteBackendTests.m
//  NFKTests
//
//  Exercises body building and response parsing through a stub transport. A live network
//  call is integration-verified.
//

#import <XCTest/XCTest.h>
#import <InferKit/NFKRemoteBackend.h>
#import <InferKit/NFKInferenceRequest.h>
#import <InferKit/NFKInferenceKeys.h>
#import <InferKit/NFKInferenceResult.h>
#import <InferKit/NFKErrors.h>

/*! A remote backend whose transport is stubbed: it records the request and returns staged data. */
@interface FxRemoteStubBackend : NFKRemoteBackend
@property (nonatomic, strong) NSURLRequest *lastRequest;
@property (nonatomic, strong, nullable) NSData *stagedData;
@property (nonatomic, assign) NSInteger stagedStatusCode;
@end

@implementation FxRemoteStubBackend

- (instancetype)init
{
	self = [super init];
	if (self) {
		_stagedStatusCode = 200;
	}
	return self;
}

- (NSData *)sendRequest:(NSURLRequest *)request response:(NSHTTPURLResponse **)outResponse error:(NSError **)outError
{
	self.lastRequest = request;
	if (outResponse != NULL) {
		*outResponse = [[NSHTTPURLResponse alloc] initWithURL:request.URL
												  statusCode:self.stagedStatusCode
												 HTTPVersion:@"HTTP/1.1"
												headerFields:nil];
	}
	return self.stagedData;
}

- (NSDictionary *)decodedRequestBody
{
	return [NSJSONSerialization JSONObjectWithData:self.lastRequest.HTTPBody options:0 error:NULL];
}

@end

@interface NFKRemoteBackendTests : XCTestCase
@property (nonatomic, strong) FxRemoteStubBackend *backend;
@end

@implementation NFKRemoteBackendTests

- (void)setUp
{
	[super setUp];
	self.backend = FxRemoteStubBackend.new;
	self.backend.endpointURL = [NSURL URLWithString:@"http://localhost:1234/v1/chat/completions"];
	self.backend.modelName = @"test-model";
}

- (void)testTheBackendReportsItsIdentity
{
	XCTAssertEqualObjects(self.backend.backendIdentifier, @"remote");
	XCTAssertTrue(self.backend.isReady);
}

- (void)testAnUnconfiguredBackendIsNotReadyAndFails
{
	NFKRemoteBackend *backend = [NFKRemoteBackend backendWithEndpointURL:nil];
	XCTAssertFalse(backend.isReady);
	NSError *error = nil;
	NFKInferenceResult *result = [backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{}]
															 error:&error];
	XCTAssertNil(result);
	XCTAssertEqual(error.code, (NSInteger)kNFKError_InferenceNotReady);
}

- (void)testAPromptBecomesAUserMessageAndParametersFoldIn
{
	self.backend.stagedData = [@"{\"choices\":[]}" dataUsingEncoding:NSUTF8StringEncoding];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKRemoteBackendPromptKey: @"Hello" }
																	 parameters:@{ @"temperature": @0.5 }];
	[self.backend runInferenceForRequest:request error:NULL];

	NSDictionary *body = [self.backend decodedRequestBody];
	XCTAssertEqualObjects(body[@"model"], @"test-model");
	XCTAssertEqualObjects(body[@"temperature"], @0.5);
	NSArray *messages = body[@"messages"];
	XCTAssertEqual(messages.count, (NSUInteger)1);
	XCTAssertEqualObjects(messages.firstObject[@"role"], @"user");
	XCTAssertEqualObjects(messages.firstObject[@"content"], @"Hello");
}

// The contract's text parameters are camelCase; the endpoint reads underscored names, so a request
// written for any engine reaches this one.
- (void)testTheContractsTextParametersCarryTheEndpointsSpelling
{
	self.backend.stagedData = [@"{\"choices\":[]}" dataUsingEncoding:NSUTF8StringEncoding];
	NSDictionary *parameters = @{ NFKParameterMaxTokens: @64,
								  NFKParameterTopP: @0.9,
								  NFKParameterTopK: @40,
								  NFKParameterStopSequences: @[ @"\n\n" ],
								  NFKParameterTemperature: @0.5,
								  NFKParameterSeed: @7 };
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Hello" }
															   parameters:parameters];
	[self.backend runInferenceForRequest:request error:NULL];

	NSDictionary *body = [self.backend decodedRequestBody];
	XCTAssertEqualObjects(body[@"max_tokens"], @64);
	XCTAssertEqualObjects(body[@"top_p"], @0.9);
	XCTAssertEqualObjects(body[@"top_k"], @40);
	XCTAssertEqualObjects(body[@"stop"], (@[ @"\n\n" ]));
	XCTAssertEqualObjects(body[@"temperature"], @0.5, @"the two spellings already agree");
	XCTAssertEqualObjects(body[@"seed"], @7);
	XCTAssertNil(body[NFKParameterMaxTokens], @"the camelCase key does not also ride along");
	XCTAssertNil(body[NFKParameterTopP]);
	XCTAssertNil(body[NFKParameterStopSequences]);
}

// OpenAI's reasoning models refuse max_tokens on Chat Completions and read the limit under its newer
// name; a router's namespaced name keeps the spelling the router reads.
- (void)testAnOpenAIReasoningModelTakesTheLimitAsMaxCompletionTokens
{
	self.backend.stagedData = [@"{\"choices\":[]}" dataUsingEncoding:NSUTF8StringEncoding];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Hello" }
															   parameters:@{ NFKParameterMaxTokens: @64,
																			 NFKParameterReasoningEffort: @"xhigh" }];
	for (NSString *model in @[ @"gpt-5.6-sol", @"gpt-5.6-luna", @"o3" ]) {
		self.backend.modelName = model;
		[self.backend runInferenceForRequest:request error:NULL];
		NSDictionary *body = [self.backend decodedRequestBody];
		XCTAssertEqualObjects(body[@"max_completion_tokens"], @64, @"%@", model);
		XCTAssertNil(body[@"max_tokens"], @"%@", model);
		XCTAssertEqualObjects(body[@"reasoning_effort"], @"xhigh", @"a level only the provider names passes through");
	}

	self.backend.modelName = @"openai/gpt-5.6-sol";
	[self.backend runInferenceForRequest:request error:NULL];
	XCTAssertEqualObjects([self.backend decodedRequestBody][@"max_tokens"], @64);
}

// OpenAI's reasoning families refuse sampling unless the effort is none, and their default is not.
- (void)testAnOpenAIReasoningModelTakesSamplingOnlyAtNoReasoning
{
	self.backend.stagedData = [@"{\"choices\":[]}" dataUsingEncoding:NSUTF8StringEncoding];
	self.backend.modelName = @"gpt-5.6-luna";
	NSDictionary *sampling = @{ NFKParameterTemperature: @0.2, NFKParameterTopP: @0.9, @"logprobs": @YES };
	[self.backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Hello" } parameters:sampling] error:NULL];
	NSDictionary *body = [self.backend decodedRequestBody];
	XCTAssertNil(body[@"temperature"]);
	XCTAssertNil(body[@"top_p"]);
	XCTAssertNil(body[@"logprobs"]);

	NSMutableDictionary *unreasoned = [sampling mutableCopy];
	unreasoned[NFKParameterReasoningEffort] = @"none";
	[self.backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Hello" } parameters:unreasoned] error:NULL];
	body = [self.backend decodedRequestBody];
	XCTAssertEqualObjects(body[@"temperature"], @0.2);
	XCTAssertEqualObjects(body[@"top_p"], @0.9);
}

// xAI's reasoning models refuse stop sequences, so the backend applies them to the reply.
- (void)testAGrokReasoningModelHasItsStopSequencesAppliedToTheReply
{
	self.backend.stagedData = [@"{\"choices\":[{\"message\":{\"content\":\"one\\nEND\\ntwo\"}}]}" dataUsingEncoding:NSUTF8StringEncoding];
	self.backend.modelName = @"grok-4.7";
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Hello" }
															   parameters:@{ NFKParameterStopSequences: @[ @"END", @"never" ] }];
	NFKInferenceResult *result = [self.backend runInferenceForRequest:request error:NULL];
	XCTAssertNil([self.backend decodedRequestBody][@"stop"]);
	XCTAssertEqualObjects(result.text, @"one\n");

	self.backend.modelName = @"grok-4.20-0309-non-reasoning";
	[self.backend runInferenceForRequest:request error:NULL];
	XCTAssertEqualObjects([self.backend decodedRequestBody][@"stop"], (@[ @"END", @"never" ]));
}

// Mistral answers a reasoning request with typed chunks: the thinking chunk holds text chunks of its own.
- (void)testMistralContentChunksBecomeTheTextAndTheReasoning
{
	NSString *reply = @"{\"choices\":[{\"message\":{\"content\":["
		"{\"type\":\"thinking\",\"thinking\":[{\"type\":\"text\",\"text\":\"two plus two\"}]},"
		"{\"type\":\"text\",\"text\":\"four\"}]}}]}";
	self.backend.stagedData = [reply dataUsingEncoding:NSUTF8StringEncoding];
	NFKInferenceResult *result = [self.backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"2+2?" }]
																error:NULL];
	XCTAssertEqualObjects(result.text, @"four");
	XCTAssertEqualObjects([result outputForKey:NFKOutputReasoning], @"two plus two");
}

// The servers disagree on the name for the same penalty, and none reads both.
- (void)testTheRepetitionPenaltyGoesOutUnderBothSpellings
{
	self.backend.stagedData = [@"{\"choices\":[]}" dataUsingEncoding:NSUTF8StringEncoding];
	NFKInferenceRequest *request =
		[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Hello" }
									parameters:@{ NFKParameterRepetitionPenalty: @1.1 }];
	[self.backend runInferenceForRequest:request error:NULL];

	NSDictionary *body = [self.backend decodedRequestBody];
	XCTAssertEqualObjects(body[@"repetition_penalty"], @1.1);
	XCTAssertEqualObjects(body[@"repeat_penalty"], @1.1);
	XCTAssertNil(body[NFKParameterRepetitionPenalty]);
}

// A caller who writes the endpoint's own name keeps what they wrote: the translation runs first and
// the fold runs after it.
- (void)testAnExplicitWireNameOutranksTheTranslatedCoreKey
{
	self.backend.stagedData = [@"{\"choices\":[]}" dataUsingEncoding:NSUTF8StringEncoding];
	NFKInferenceRequest *request =
		[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Hello" }
									parameters:@{ NFKParameterMaxTokens: @64, @"max_tokens": @128 }];
	[self.backend runInferenceForRequest:request error:NULL];

	XCTAssertEqualObjects([self.backend decodedRequestBody][@"max_tokens"], @128);
}

- (void)testAMessagesArrayIsSentAsIs
{
	self.backend.stagedData = [@"{\"choices\":[]}" dataUsingEncoding:NSUTF8StringEncoding];
	NSArray *messages = @[ @{ @"role": @"system", @"content": @"be terse" }, @{ @"role": @"user", @"content": @"hi" } ];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKRemoteBackendMessagesKey: messages }];
	[self.backend runInferenceForRequest:request error:NULL];

	XCTAssertEqualObjects([self.backend decodedRequestBody][@"messages"], messages);
}

- (void)testTheAssistantContentIsExtracted
{
	self.backend.stagedData = [@"{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"world\"}}]}"
							  dataUsingEncoding:NSUTF8StringEncoding];
	NSError *error = nil;
	NFKInferenceResult *result = [self.backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{}]
																  error:&error];
	XCTAssertNil(error);
	XCTAssertEqualObjects([result outputForKey:NFKRemoteBackendTextKey], @"world");
	XCTAssertNotNil([result outputForKey:NFKRemoteBackendRawKey]);
}

// A reply the provider's filter stopped, or one the model declined under message.refusal, is a
// refusal: the request is the problem and sending it again gives the same answer.
- (void)testAContentFilterStopAndARefusalMessageAreRefusals
{
	self.backend.stagedData = [@"{\"choices\":[{\"finish_reason\":\"content_filter\",\"message\":{\"role\":\"assistant\",\"content\":\"I cannot\"}}]}"
							   dataUsingEncoding:NSUTF8StringEncoding];
	NSError *error = nil;
	XCTAssertNil([self.backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{}] error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceRefused);

	self.backend.stagedData = [@"{\"choices\":[{\"finish_reason\":\"stop\",\"message\":{\"role\":\"assistant\",\"content\":null,\"refusal\":\"I can't help with that.\"}}]}"
							   dataUsingEncoding:NSUTF8StringEncoding];
	XCTAssertNil([self.backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{}] error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceRefused);
	XCTAssertEqualObjects(error.localizedDescription, @"I can't help with that.");

	self.backend.stagedStatusCode = 429;
	self.backend.stagedData = [@"{}" dataUsingEncoding:NSUTF8StringEncoding];
	XCTAssertNil([self.backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{}] error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceRateLimited);
}

- (void)testAnHTTPErrorStatusFailsTheRun
{
	self.backend.stagedData = [@"{}" dataUsingEncoding:NSUTF8StringEncoding];
	self.backend.stagedStatusCode = 500;
	NSError *error = nil;
	NFKInferenceResult *result = [self.backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{}]
																  error:&error];
	XCTAssertNil(result);
	XCTAssertEqual(error.code, (NSInteger)kNFKError_InferenceBackendFailure);
}

#pragma mark Reasoning and usage

- (void)testTheReasoningEffortCarriesTheEndpointsSpelling
{
	self.backend.stagedData = [@"{\"choices\":[]}" dataUsingEncoding:NSUTF8StringEncoding];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"why" }
															  parameters:@{ NFKParameterReasoningEffort: NFKReasoningEffortDeep }];
	[self.backend runInferenceForRequest:request error:NULL];
	XCTAssertEqualObjects([self.backend decodedRequestBody][@"reasoning_effort"], @"high");
	XCTAssertNil([self.backend decodedRequestBody][NFKParameterReasoningEffort], @"the core key is renamed, not folded in beside it");
}

- (void)testALevelTheContractDoesNotNameGoesOutAsWritten
{
	self.backend.stagedData = [@"{\"choices\":[]}" dataUsingEncoding:NSUTF8StringEncoding];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{}
															  parameters:@{ NFKParameterReasoningEffort: @"minimal" }];
	[self.backend runInferenceForRequest:request error:NULL];
	XCTAssertEqualObjects([self.backend decodedRequestBody][@"reasoning_effort"], @"minimal");
}

- (void)testTheReasoningAndTheTokenCountsComeBack
{
	self.backend.stagedData = [@"{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"four\","
								"\"reasoning_content\":\"two plus two\"}}],"
								"\"usage\":{\"prompt_tokens\":11,\"completion_tokens\":7,"
								"\"prompt_tokens_details\":{\"cached_tokens\":8},"
								"\"completion_tokens_details\":{\"reasoning_tokens\":5}}}"
							  dataUsingEncoding:NSUTF8StringEncoding];
	NFKInferenceResult *result = [self.backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{}] error:NULL];
	XCTAssertEqualObjects([result outputForKey:NFKOutputReasoning], @"two plus two");
	NSDictionary *usage = [result outputForKey:NFKOutputUsage];
	XCTAssertEqualObjects(usage[NFKUsageInputTokens], @11);
	XCTAssertEqualObjects(usage[NFKUsageCachedTokens], @8);
	XCTAssertEqualObjects(usage[NFKUsageOutputTokens], @7);
	XCTAssertEqualObjects(usage[NFKUsageReasoningTokens], @5);
}

- (void)testACountTheEndpointLeavesOutIsAbsentRatherThanZero
{
	self.backend.stagedData = [@"{\"choices\":[],\"usage\":{\"prompt_tokens\":3}}" dataUsingEncoding:NSUTF8StringEncoding];
	NFKInferenceResult *result = [self.backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{}] error:NULL];
	NSDictionary *usage = [result outputForKey:NFKOutputUsage];
	XCTAssertEqualObjects(usage[NFKUsageInputTokens], @3);
	XCTAssertNil(usage[NFKUsageOutputTokens]);
}

- (void)testAReplyWithoutReasoningOrCountsCarriesNeitherKey
{
	self.backend.stagedData = [@"{\"choices\":[{\"message\":{\"role\":\"assistant\",\"content\":\"hi\"}}]}"
							  dataUsingEncoding:NSUTF8StringEncoding];
	NFKInferenceResult *result = [self.backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{}] error:NULL];
	XCTAssertNil([result outputForKey:NFKOutputReasoning]);
	XCTAssertNil([result outputForKey:NFKOutputUsage]);
}

- (void)testTheAuthorizationHeaderCarriesTheAPIKey
{
	self.backend.apiKey = @"secret";
	self.backend.stagedData = [@"{\"choices\":[]}" dataUsingEncoding:NSUTF8StringEncoding];
	[self.backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{}] error:NULL];
	XCTAssertEqualObjects([self.backend.lastRequest valueForHTTPHeaderField:@"Authorization"], @"Bearer secret");
}

@end
