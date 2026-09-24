//
//  NFKTypeSafeBackendTests.m
//  InferKitTests
//
//  The System One request and reply through a stub transport: the wire shape, the typed answers,
//  the state fallbacks, the errors, and the provider factory. A live test against the hosted
//  endpoint is gated on a key.
//

#import <XCTest/XCTest.h>
#import <InferKit/NFKTypeSafeBackend.h>
#import <InferKit/NFKRemoteProvider.h>
#import <InferKit/NFKInferenceRequest.h>
#import <InferKit/NFKInferenceResult.h>
#import <InferKit/NFKInferenceKeys.h>
#import <InferKit/NFKErrors.h>

@interface NFKStubTypeSafeBackend : NFKTypeSafeBackend
@property (nonatomic, strong) NSURLRequest *lastRequest;
@property (nonatomic, copy) NSString *stagedBody;
@property (nonatomic, assign) NSInteger stagedStatusCode;
@end

@implementation NFKStubTypeSafeBackend
- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_stagedStatusCode = 200;
	}
	return self;
}

- (NSData *)sendRequest:(NSURLRequest *)request response:(NSHTTPURLResponse **)outResponse error:(NSError **)outError
{
	self.lastRequest = request;
	if (outResponse != NULL) {
		*outResponse = [[NSHTTPURLResponse alloc] initWithURL:request.URL statusCode:self.stagedStatusCode
												  HTTPVersion:@"HTTP/1.1" headerFields:nil];
	}
	return [self.stagedBody dataUsingEncoding:NSUTF8StringEncoding];
}
@end

@interface NFKTypeSafeBackendTests : XCTestCase
@property (nonatomic, strong) NFKStubTypeSafeBackend *backend;
@end

@implementation NFKTypeSafeBackendTests

- (void)setUp
{
	[super setUp];
	self.backend = [[NFKStubTypeSafeBackend alloc] init];
	self.backend.apiKey = @"ts-key";
	self.backend.modelName = @"jev-latest";
	self.backend.stagedBody = @"{\"model\":\"jev-1.13.0\",\"answers\":{"
		"\"department\":{\"type\":\"choice\",\"choice\":\"technical\","
		"\"probabilities\":{\"billing\":0.08,\"technical\":0.85,\"sales\":0.07},\"confidence\":0.82},"
		"\"severity\":{\"type\":\"score\",\"score\":1.6,\"legend\":{\"0\":\"low\",\"1\":\"medium\",\"2\":\"high\"},"
		"\"probabilities\":{\"0\":0.1,\"1\":0.2,\"2\":0.7},\"confidence\":0.61},"
		"\"urgent\":{\"type\":\"noul\",\"noul\":0.91}},"
		"\"usage\":{\"input_tokens\":312,\"output_tokens\":48}}";
}

- (NSDictionary *)decodedRequestBody
{
	return [NSJSONSerialization JSONObjectWithData:self.backend.lastRequest.HTTPBody options:0 error:NULL];
}

- (NSDictionary<NSString *, NFKDecisionQuestion *> *)threeQuestions
{
	return @{
		@"department": [NFKDecisionQuestion choiceQuestionWithInstructions:@"Which team should handle this?"
																   options:@[ @"billing", @"technical", @"sales" ]
															  descriptions:@{ @"billing": @"Payments, invoicing, refunds",
																			  @"technical": @"Bugs, outages, integrations" }],
		@"severity": [NFKDecisionQuestion scoreQuestionWithInstructions:@"How severe is the problem?"
																 levels:@[ @"low", @"medium", @"high" ]],
		@"urgent": [NFKDecisionQuestion noulQuestionWithInstructions:@"The customer needs an answer today."],
	};
}

#pragma mark The wire shape

- (void)testTheRequestCarriesTheModelTheStateAndTheQuestionsWithABearerToken
{
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{
		NFKInputState: @"Help! My payouts have been failing for 3 days.",
		NFKInputQuestions: [self threeQuestions] }];
	NSError *error = nil;
	XCTAssertNotNil([self.backend runInferenceForRequest:request error:&error], @"%@", error);

	XCTAssertEqualObjects(self.backend.lastRequest.URL.absoluteString, @"https://api.typesafe.ai/v1/systemone");
	XCTAssertEqualObjects(self.backend.lastRequest.HTTPMethod, @"POST");
	XCTAssertEqualObjects(self.backend.lastRequest.allHTTPHeaderFields[@"Authorization"], @"Bearer ts-key");
	XCTAssertEqualObjects(self.backend.lastRequest.allHTTPHeaderFields[@"Content-Type"], @"application/json");

	NSDictionary *body = [self decodedRequestBody];
	XCTAssertEqualObjects(body[@"model"], @"jev-latest");
	XCTAssertEqualObjects(body[@"state"], @"Help! My payouts have been failing for 3 days.");
	NSDictionary *questions = body[@"questions"];
	XCTAssertEqual(questions.count, 3);
	XCTAssertEqualObjects(questions[@"department"][@"type"], @"choice");
	XCTAssertEqualObjects(questions[@"department"][@"instructions"], @"Which team should handle this?");
	XCTAssertEqualObjects(questions[@"department"][@"criteria"][@"billing"], @"Payments, invoicing, refunds");
	XCTAssertEqualObjects(questions[@"department"][@"criteria"][@"sales"], NSNull.null, @"an undescribed option is sent as null");
	XCTAssertEqualObjects(questions[@"severity"][@"type"], @"score");
	XCTAssertEqualObjects(questions[@"severity"][@"criteria"], (@[ @"low", @"medium", @"high" ]));
	XCTAssertEqualObjects(questions[@"urgent"][@"type"], @"noul");
	XCTAssertNil(questions[@"urgent"][@"criteria"], @"a noul without meanings sends none");
	XCTAssertEqualObjects(self.backend.backendIdentifier, @"typesafe-systemone");
}

- (void)testAStateThatIsAnObjectOrAListIsSentAsIs
{
	NSDictionary *record = @{ @"subject": @"Refund", @"body": @"Where is my refund?", @"tags": @[ @"billing" ] };
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{
		NFKInputState: record,
		NFKInputQuestions: @{ @"urgent": [NFKDecisionQuestion noulQuestionWithInstructions:@"Urgent?"] } }];
	XCTAssertNotNil([self.backend runInferenceForRequest:request error:NULL]);
	XCTAssertEqualObjects([self decodedRequestBody][@"state"], record);
}

- (void)testThePromptAndThenTheMessagesStandInForAnAbsentState
{
	NSDictionary *questions = @{ @"urgent": [NFKDecisionQuestion noulQuestionWithInstructions:@"Urgent?"] };
	NFKInferenceRequest *prompted = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"the text",
																			   NFKInputQuestions: questions }];
	XCTAssertNotNil([self.backend runInferenceForRequest:prompted error:NULL]);
	XCTAssertEqualObjects([self decodedRequestBody][@"state"], @"the text");

	NSArray *conversation = @[ @{ @"role": @"user", @"content": @"hi" }, @{ @"role": @"assistant", @"content": @"hello" } ];
	NFKInferenceRequest *chatted = [NFKInferenceRequest requestWithInputs:@{ NFKInputMessages: conversation,
																			  NFKInputQuestions: questions }];
	XCTAssertNotNil([self.backend runInferenceForRequest:chatted error:NULL]);
	XCTAssertEqualObjects([self decodedRequestBody][@"state"], conversation);
}

- (void)testAQuestionAlreadyInWireShapePassesThrough
{
	NSDictionary *wire = @{ @"type": @"choice", @"instructions": @"Which?", @"criteria": @{ @"a": NSNull.null, @"b": @"the b" } };
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputState: @"s",
																			  NFKInputQuestions: @{ @"which": wire } }];
	XCTAssertNotNil([self.backend runInferenceForRequest:request error:NULL]);
	XCTAssertEqualObjects([self decodedRequestBody][@"questions"][@"which"], wire);
}

- (void)testAnUnnamedParameterFoldsIntoTheBodyWithoutOverridingTheRequestFields
{
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{
		NFKInputState: @"s", NFKInputQuestions: @{ @"q": [NFKDecisionQuestion noulQuestionWithInstructions:@"?"] } }
		parameters:@{ @"metadata": @{ @"trace": @"t1" }, @"model": @"not-this-one", NFKParameterTemperature: @0.2 }
		outputModality:NFKModalityText];
	XCTAssertNotNil([self.backend runInferenceForRequest:request error:NULL]);
	NSDictionary *body = [self decodedRequestBody];
	XCTAssertEqualObjects(body[@"metadata"][@"trace"], @"t1");
	XCTAssertEqualObjects(body[@"model"], @"jev-latest", @"the backend's model wins over a parameter of that name");
	XCTAssertEqualObjects(body[@"temperature"], @0.2, @"the contract key has no meaning here and is passed as written");
	XCTAssertEqual(self.backend.supportedParameterKeys.count, 0);
	XCTAssertTrue([self.backend.supportedInputKeys containsObject:NFKInputQuestions]);
}

#pragma mark The reply

- (void)testTheReplyBecomesTypedAnswersKeyedAsTheQuestionsWere
{
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputState: @"s",
																			  NFKInputQuestions: [self threeQuestions] }];
	NFKInferenceResult *result = [self.backend runInferenceForRequest:request error:NULL];
	NSDictionary<NSString *, NFKDecisionAnswer *> *answers = result.answers;
	XCTAssertEqual(answers.count, 3);

	NFKDecisionAnswer *department = answers[@"department"];
	XCTAssertEqual(department.type, NFKDecisionTypeChoice);
	XCTAssertEqualObjects(department.choice, @"technical");
	XCTAssertEqualWithAccuracy(department.probabilities[@"technical"].doubleValue, 0.85, 1e-9);
	XCTAssertEqualWithAccuracy(department.confidence, 0.82, 1e-9);
	XCTAssertNil(department.legend);

	NFKDecisionAnswer *severity = answers[@"severity"];
	XCTAssertEqual(severity.type, NFKDecisionTypeScore);
	XCTAssertEqualWithAccuracy(severity.score, 1.6, 1e-9);
	XCTAssertEqualObjects(severity.legend[@"2"], @"high");
	XCTAssertEqualWithAccuracy(severity.probabilities[@"2"].doubleValue, 0.7, 1e-9);
	XCTAssertNil(severity.choice);

	NFKDecisionAnswer *urgent = answers[@"urgent"];
	XCTAssertEqual(urgent.type, NFKDecisionTypeNoul);
	XCTAssertEqualWithAccuracy(urgent.probability, 0.91, 1e-9);
	XCTAssertNil(urgent.probabilities);
	XCTAssertEqualObjects(urgent.raw[@"noul"], @0.91);

	XCTAssertEqualObjects(result.structured[@"model"], @"jev-1.13.0", @"the whole reply rides under structured");
	NSDictionary *usage = [result outputForKey:NFKOutputUsage];
	XCTAssertEqualObjects(usage[NFKUsageInputTokens], @312);
	XCTAssertEqualObjects(usage[NFKUsageOutputTokens], @48);
	XCTAssertNil(usage[NFKUsageReasoningTokens], @"an unreported count is absent");
}

- (void)testTheConvenienceAnswersWithoutARequest
{
	NSError *error = nil;
	NSDictionary<NSString *, NFKDecisionAnswer *> *answers = [self.backend answersForState:@"the state"
																				 questions:[self threeQuestions] error:&error];
	XCTAssertNotNil(answers, @"%@", error);
	XCTAssertEqualObjects(answers[@"department"].choice, @"technical");
	XCTAssertEqualObjects([self decodedRequestBody][@"state"], @"the state");
}

- (void)testAnAnswerOfAnUnknownTypeIsDroppedRatherThanMisread
{
	self.backend.stagedBody = @"{\"answers\":{\"a\":{\"type\":\"noul\",\"noul\":0.5},\"b\":{\"type\":\"ranking\",\"order\":[]}}}";
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputState: @"s",
																			  NFKInputQuestions: [self threeQuestions] }];
	NFKInferenceResult *result = [self.backend runInferenceForRequest:request error:NULL];
	XCTAssertEqual(result.answers.count, 1);
	XCTAssertNotNil(result.answers[@"a"]);
	XCTAssertNotNil(result.structured[@"answers"][@"b"], @"the raw reply keeps what the type does not read");
}

#pragma mark Errors

- (void)testTheBackendIsNotReadyWithoutAModelBecauseTheAPIRequiresOne
{
	self.backend.modelName = nil;
	XCTAssertFalse(self.backend.isReady);
	NSError *error = nil;
	XCTAssertNil([self.backend answersForState:@"s" questions:[self threeQuestions] error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceNotReady);
	XCTAssertNil(self.backend.lastRequest, @"nothing was sent");
}

- (void)testAMissingStateOrMissingQuestionsIsAnErrorBeforeAnythingIsSent
{
	NSError *error = nil;
	NFKInferenceRequest *noState = [NFKInferenceRequest requestWithInputs:@{ NFKInputQuestions: [self threeQuestions] }];
	XCTAssertNil([self.backend runInferenceForRequest:noState error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceMissingInput);

	NFKInferenceRequest *noQuestions = [NFKInferenceRequest requestWithInputs:@{ NFKInputState: @"s" }];
	XCTAssertNil([self.backend runInferenceForRequest:noQuestions error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceMissingInput);

	NFKInferenceRequest *unserializable = [NFKInferenceRequest requestWithInputs:@{ NFKInputState: NSDate.date,
																					 NFKInputQuestions: [self threeQuestions] }];
	XCTAssertNil([self.backend runInferenceForRequest:unserializable error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceMissingInput);
	XCTAssertNil(self.backend.lastRequest);
}

- (void)testAFailingStatusAndAnAnswerlessReplyAreErrors
{
	self.backend.stagedStatusCode = 401;
	self.backend.stagedBody = @"{\"error\":\"invalid api key\"}";
	NSError *error = nil;
	XCTAssertNil([self.backend answersForState:@"s" questions:[self threeQuestions] error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceBackendFailure);
	XCTAssertTrue([error.localizedDescription containsString:@"401"]);
	XCTAssertTrue([error.localizedDescription containsString:@"invalid api key"], @"the body explains the rejection");

	// The service's own codes: 422 is an invalid body, which retrying does not fix; 429 and 529 say
	// back off, and the reset date it names rides with the error.
	self.backend.stagedStatusCode = 422;
	self.backend.stagedBody = @"{\"error\":\"criteria must have between 2 and 10 levels\"}";
	XCTAssertNil([self.backend answersForState:@"s" questions:[self threeQuestions] error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceRefused);
	self.backend.stagedStatusCode = 529;
	XCTAssertNil([self.backend answersForState:@"s" questions:[self threeQuestions] error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceRateLimited);

	self.backend.stagedStatusCode = 200;
	self.backend.stagedBody = @"{\"model\":\"jev-1.13.0\"}";
	XCTAssertNil([self.backend answersForState:@"s" questions:[self threeQuestions] error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceBackendFailure);
}

#pragma mark The provider

- (void)testTheFactoryHandsBackThisBackendPointedAtTheProvidersEndpoint
{
	id<NFKInferenceBackend> backend = [NFKRemoteProvider backendForProvider:NFKRemoteProvider.typeSafe
																	 apiKey:@"k" modelName:@"jev-latest"];
	XCTAssertTrue([backend isKindOfClass:NFKTypeSafeBackend.class]);
	NFKTypeSafeBackend *jev = (NFKTypeSafeBackend *)backend;
	XCTAssertEqualObjects(jev.endpointURL.absoluteString, @"https://api.typesafe.ai/v1/systemone");
	XCTAssertEqualObjects(jev.apiKey, @"k");
	XCTAssertEqualObjects(jev.modelName, @"jev-latest");
	XCTAssertTrue(jev.isReady);

	NFKRemoteProvider *proxied = [NFKRemoteProvider.typeSafe providerWithBaseURL:[NSURL URLWithString:@"http://gateway.test/typesafe/v1"]];
	NFKTypeSafeBackend *behindGateway = (NFKTypeSafeBackend *)[NFKRemoteProvider backendForProvider:proxied apiKey:@"k" modelName:@"m"];
	XCTAssertEqualObjects(behindGateway.endpointURL.absoluteString, @"http://gateway.test/typesafe/v1/systemone");
}

#pragma mark Live

// Runs against the hosted endpoint when a key is present. The endpoint answers 403 to a request
// without a key, so the path is verified to exist; the answer shape is verified only here.
- (void)testALiveEndpointAnswersATypedDecision
{
	NSString *key = NSProcessInfo.processInfo.environment[@"INFERKIT_TYPESAFE_API_KEY"];
	if (key.length == 0) {
		XCTSkip("set INFERKIT_TYPESAFE_API_KEY to exercise the live path");
	}
	NFKTypeSafeBackend *jev = (NFKTypeSafeBackend *)[NFKRemoteProvider backendForProvider:NFKRemoteProvider.typeSafe
																				   apiKey:key modelName:@"jev-latest"];
	NSError *error = nil;
	NSDictionary<NSString *, NFKDecisionAnswer *> *answers = [jev answersForState:@"Help! My payouts have been failing for 3 days."
																		questions:[self threeQuestions] error:&error];
	XCTAssertNotNil(answers, @"%@", error);
	XCTAssertEqualObjects(answers[@"department"].choice, @"technical");
	XCTAssertGreaterThan(answers[@"urgent"].probability, 0.5);
}

@end
