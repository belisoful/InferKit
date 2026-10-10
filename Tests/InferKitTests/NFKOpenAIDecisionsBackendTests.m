//
//  NFKOpenAIDecisionsBackendTests.m
//  InferKitTests
//
//  The Decisions API request and reply through a stub transport: the wire shape, the typed answers,
//  the input forms, refusals, the errors, and the provider factory. A live test against the hosted
//  endpoint is gated on a key.
//

#import <XCTest/XCTest.h>
#import <CoreGraphics/CoreGraphics.h>
#import <InferKit/NFKOpenAIDecisionsBackend.h>
#import <InferKit/NFKRemoteProvider.h>
#import <InferKit/NFKInferenceRequest.h>
#import <InferKit/NFKInferenceResult.h>
#import <InferKit/NFKInferenceKeys.h>
#import <InferKit/NFKErrors.h>

@interface NFKStubOpenAIDecisionsBackend : NFKOpenAIDecisionsBackend
@property (nonatomic, strong) NSURLRequest *lastRequest;
@property (nonatomic, copy) NSString *stagedBody;
@property (nonatomic, assign) NSInteger stagedStatusCode;
@end

@implementation NFKStubOpenAIDecisionsBackend
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

@interface NFKOpenAIDecisionsBackendTests : XCTestCase
@property (nonatomic, strong) NFKStubOpenAIDecisionsBackend *backend;
@end

@implementation NFKOpenAIDecisionsBackendTests

- (void)setUp
{
	[super setUp];
	self.backend = [[NFKStubOpenAIDecisionsBackend alloc] init];
	self.backend.apiKey = @"sk-key";
	self.backend.modelName = @"gpt-6-luna";
	self.backend.stagedBody = @"{\"model\":\"gpt-6-luna\",\"answers\":["
		"{\"type\":\"choice\",\"name\":\"department\",\"choice\":\"technical\",\"confidence\":0.82,"
		"\"probabilities\":[{\"value\":\"billing\",\"probability\":0.08},{\"value\":\"technical\",\"probability\":0.85},"
		"{\"value\":\"sales\",\"probability\":0.07}]},"
		"{\"type\":\"score\",\"name\":\"severity\",\"score\":1.6,\"confidence\":0.61,"
		"\"probabilities\":[{\"value\":0,\"label\":\"low\",\"probability\":0.1},{\"value\":1,\"label\":\"medium\",\"probability\":0.2},"
		"{\"value\":2,\"label\":\"high\",\"probability\":0.7}]},"
		"{\"type\":\"predicate\",\"name\":\"urgent\",\"probability\":0.91}],"
		"\"usage\":{\"input_tokens\":312,\"input_tokens_details\":{\"cached_tokens\":0,\"cache_write_tokens\":0},"
		"\"output_tokens\":0,\"output_tokens_details\":{\"reasoning_tokens\":0},\"total_tokens\":312}}";
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

- (NFKInferenceRequest *)requestWithState:(id)state
{
	return [NFKInferenceRequest requestWithInputs:@{ NFKInputState: state, NFKInputQuestions: [self threeQuestions] }];
}

#pragma mark The wire shape

- (void)testTheRequestCarriesTheModelTheInputAndNamedQuestionsWithABearerToken
{
	NSError *error = nil;
	XCTAssertNotNil([self.backend runInferenceForRequest:[self requestWithState:@"Help! My payouts have been failing for 3 days."]
												   error:&error], @"%@", error);

	XCTAssertEqualObjects(self.backend.lastRequest.URL.absoluteString, @"https://api.openai.com/v1/decisions");
	XCTAssertEqualObjects(self.backend.lastRequest.HTTPMethod, @"POST");
	XCTAssertEqualObjects(self.backend.lastRequest.allHTTPHeaderFields[@"Authorization"], @"Bearer sk-key");
	XCTAssertEqualObjects(self.backend.lastRequest.allHTTPHeaderFields[@"Content-Type"], @"application/json");

	NSDictionary *body = [self decodedRequestBody];
	XCTAssertEqualObjects(body[@"model"], @"gpt-6-luna");
	XCTAssertEqualObjects(body[@"input"], @"Help! My payouts have been failing for 3 days.");
	NSArray *questions = body[@"questions"];
	XCTAssertEqualObjects([questions valueForKey:@"name"], (@[ @"department", @"severity", @"urgent" ]), @"sorted by identifier");
	XCTAssertEqualObjects(questions[0], (@{ @"type": @"choice", @"name": @"department", @"instructions": @"Which team should handle this?",
											@"choices": @[ @{ @"value": @"billing", @"description": @"Payments, invoicing, refunds" },
														   @{ @"value": @"technical", @"description": @"Bugs, outages, integrations" },
														   @{ @"value": @"sales" } ] }), @"an undescribed option carries no description");
	XCTAssertEqualObjects(questions[1][@"type"], @"score");
	XCTAssertEqualObjects(questions[1][@"levels"], (@[ @{ @"label": @"low" }, @{ @"label": @"medium" }, @{ @"label": @"high" } ]));
	XCTAssertEqualObjects(questions[2], (@{ @"type": @"predicate", @"name": @"urgent",
											@"instructions": @"The customer needs an answer today." }));
	XCTAssertEqualObjects(self.backend.backendIdentifier, @"openai-decisions");
}

- (void)testANoulsMeaningsFollowItsInstructions
{
	NFKDecisionQuestion *refund = [NFKDecisionQuestion noulQuestionWithInstructions:@"The customer asks for a refund."
																		trueMeaning:@"money back is requested"
																	   falseMeaning:nil];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputState: @"s", NFKInputQuestions: @{ @"refund": refund } }];
	XCTAssertNotNil([self.backend runInferenceForRequest:request error:NULL]);
	XCTAssertEqualObjects([self decodedRequestBody][@"questions"][0][@"instructions"],
						  @"The customer asks for a refund.\n\nTrue means: money back is requested");
}

- (void)testARecordIsSentAsJSONTextAndAMessageListAsIs
{
	NSDictionary *record = @{ @"subject": @"Refund", @"tags": @[ @"billing" ] };
	XCTAssertNotNil([self.backend runInferenceForRequest:[self requestWithState:record] error:NULL]);
	XCTAssertEqualObjects([self decodedRequestBody][@"input"], @"{\"subject\":\"Refund\",\"tags\":[\"billing\"]}");

	NSArray *messages = @[ @{ @"role": @"user", @"content": @[ @{ @"type": @"input_text", @"text": @"hi" } ] } ];
	XCTAssertNotNil([self.backend runInferenceForRequest:[self requestWithState:messages] error:NULL]);
	XCTAssertEqualObjects([self decodedRequestBody][@"input"], messages);
}

- (void)testThePromptAndThenAConversationTranscriptStandInForAnAbsentState
{
	NSDictionary *questions = [self threeQuestions];
	NFKInferenceRequest *prompted = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"the text", NFKInputQuestions: questions }];
	XCTAssertNotNil([self.backend runInferenceForRequest:prompted error:NULL]);
	XCTAssertEqualObjects([self decodedRequestBody][@"input"], @"the text");

	NSArray *conversation = @[ @{ @"role": @"user", @"content": @"my order is late" },
							   @{ @"role": @"assistant", @"content": @[ @{ @"type": @"text", @"text": @"sorry about that" } ] } ];
	NFKInferenceRequest *chatted = [NFKInferenceRequest requestWithInputs:@{ NFKInputMessages: conversation, NFKInputQuestions: questions }];
	XCTAssertNotNil([self.backend runInferenceForRequest:chatted error:NULL]);
	XCTAssertEqualObjects([self decodedRequestBody][@"input"], @"user: my order is late\nassistant: sorry about that");
}

- (void)testAnImageRidesAsADataURLInAUserMessageBesideTheText
{
	CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
	CGContextRef context = CGBitmapContextCreate(NULL, 2, 2, 8, 8, colorSpace, kCGBitmapByteOrder32Little | kCGImageAlphaPremultipliedFirst);
	CGColorSpaceRelease(colorSpace);
	CGImageRef square = CGBitmapContextCreateImage(context);
	CGContextRelease(context);

	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputState: @"Inspect the product.",
																			 NFKInputImage: (__bridge id)square,
																			 NFKInputQuestions: [self threeQuestions] }];
	XCTAssertNotNil([self.backend runInferenceForRequest:request error:NULL]);
	NSArray *input = [self decodedRequestBody][@"input"];
	XCTAssertEqual(input.count, 1);
	XCTAssertEqualObjects(input[0][@"role"], @"user");
	NSArray *parts = input[0][@"content"];
	XCTAssertEqualObjects(parts[0], (@{ @"type": @"input_text", @"text": @"Inspect the product." }));
	XCTAssertEqualObjects(parts[1][@"type"], @"input_image");
	XCTAssertTrue([parts[1][@"image_url"] hasPrefix:@"data:image/png;base64,"]);
	XCTAssertTrue([self.backend.supportedInputKeys containsObject:NFKInputImages]);
	CGImageRelease(square);
}

- (void)testAQuestionAlreadyInWireShapePassesThroughAndGainsItsName
{
	NSDictionary *wire = @{ @"type": @"predicate", @"instructions": @"Damaged?" };
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputState: @"s", NFKInputQuestions: @{ @"damaged": wire } }];
	XCTAssertNotNil([self.backend runInferenceForRequest:request error:NULL]);
	XCTAssertEqualObjects([self decodedRequestBody][@"questions"][0], (@{ @"type": @"predicate", @"instructions": @"Damaged?", @"name": @"damaged" }));
}

- (void)testAnUnnamedParameterFoldsIntoTheBodyWithoutOverridingTheRequestFields
{
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputState: @"s", NFKInputQuestions: [self threeQuestions] }
															   parameters:@{ @"safety_identifier": @"user-7", @"model": @"not-this-one", @"input": @"nor this" }
														   outputModality:NFKModalityText];
	XCTAssertNotNil([self.backend runInferenceForRequest:request error:NULL]);
	NSDictionary *body = [self decodedRequestBody];
	XCTAssertEqualObjects(body[@"safety_identifier"], @"user-7");
	XCTAssertEqualObjects(body[@"model"], @"gpt-6-luna");
	XCTAssertEqualObjects(body[@"input"], @"s");
	XCTAssertEqual(self.backend.supportedParameterKeys.count, 0);
}

#pragma mark The reply

- (void)testTheReplyBecomesTypedAnswersKeyedAsTheQuestionsWere
{
	NFKInferenceResult *result = [self.backend runInferenceForRequest:[self requestWithState:@"s"] error:NULL];
	NSDictionary<NSString *, NFKDecisionAnswer *> *answers = result.answers;
	XCTAssertEqual(answers.count, 3);

	NFKDecisionAnswer *department = answers[@"department"];
	XCTAssertEqual(department.type, NFKDecisionTypeChoice);
	XCTAssertEqualObjects(department.choice, @"technical");
	XCTAssertEqualWithAccuracy(department.probabilities[@"technical"].doubleValue, 0.85, 1e-9);
	XCTAssertEqualWithAccuracy(department.confidence, 0.82, 1e-9);

	NFKDecisionAnswer *severity = answers[@"severity"];
	XCTAssertEqual(severity.type, NFKDecisionTypeScore);
	XCTAssertEqualWithAccuracy(severity.score, 1.6, 1e-9);
	XCTAssertEqualObjects(severity.legend[@"2"], @"high");
	XCTAssertEqualWithAccuracy(severity.probabilities[@"2"].doubleValue, 0.7, 1e-9);

	NFKDecisionAnswer *urgent = answers[@"urgent"];
	XCTAssertEqual(urgent.type, NFKDecisionTypeNoul);
	XCTAssertEqualWithAccuracy(urgent.probability, 0.91, 1e-9);
	XCTAssertFalse(urgent.isRefused);

	XCTAssertEqualObjects(result.structured[@"model"], @"gpt-6-luna", @"the whole reply rides under structured");
	NSDictionary *usage = [result outputForKey:NFKOutputUsage];
	XCTAssertEqualObjects(usage[NFKUsageInputTokens], @312);
	XCTAssertEqualObjects(usage[NFKUsageCachedTokens], @0);
	XCTAssertEqualObjects(usage[NFKUsageOutputTokens], @0);
}

- (void)testARefusalTakesItsQuestionsTypeAndAnUnnamedAnswerItsPosition
{
	self.backend.stagedBody = @"{\"answers\":[{\"type\":\"refusal\",\"name\":\"department\"},"
		"{\"type\":\"score\",\"score\":2.0,\"confidence\":0.9,\"probabilities\":[]},"
		"{\"type\":\"ranking\",\"name\":\"urgent\"}]}";
	NFKInferenceResult *result = [self.backend runInferenceForRequest:[self requestWithState:@"s"] error:NULL];
	XCTAssertEqual(result.answers.count, 2, @"the answer of an unknown type is dropped");
	XCTAssertTrue(result.answers[@"department"].isRefused);
	XCTAssertEqual(result.answers[@"department"].type, NFKDecisionTypeChoice);
	XCTAssertEqualWithAccuracy(result.answers[@"severity"].score, 2.0, 1e-12);
	XCTAssertNotNil(result.structured[@"answers"][2], @"the raw reply keeps what the type does not read");
}

- (void)testTheConvenienceAnswersWithoutARequest
{
	NSError *error = nil;
	NSDictionary<NSString *, NFKDecisionAnswer *> *answers = [self.backend answersForState:@"the state"
																				 questions:[self threeQuestions] error:&error];
	XCTAssertNotNil(answers, @"%@", error);
	XCTAssertEqualObjects(answers[@"department"].choice, @"technical");
	XCTAssertEqualObjects([self decodedRequestBody][@"input"], @"the state");
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

- (void)testAMissingStateMissingQuestionsOrUnsendableStateIsAnErrorBeforeAnythingIsSent
{
	NSError *error = nil;
	NFKInferenceRequest *noState = [NFKInferenceRequest requestWithInputs:@{ NFKInputQuestions: [self threeQuestions] }];
	XCTAssertNil([self.backend runInferenceForRequest:noState error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceMissingInput);

	NFKInferenceRequest *noQuestions = [NFKInferenceRequest requestWithInputs:@{ NFKInputState: @"s" }];
	XCTAssertNil([self.backend runInferenceForRequest:noQuestions error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceMissingInput);

	XCTAssertNil([self.backend runInferenceForRequest:[self requestWithState:NSDate.date] error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceMissingInput);

	NFKInferenceRequest *withDocument = [NFKInferenceRequest requestWithInputs:@{ NFKInputState: @"s", NFKInputDocument: @"terms",
																				  NFKInputQuestions: [self threeQuestions] }];
	XCTAssertNil([self.backend runInferenceForRequest:withDocument error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceUnsupported);
	XCTAssertNil(self.backend.lastRequest);
}

- (void)testAFailingStatusAndAnAnswerlessReplyAreErrors
{
	self.backend.stagedStatusCode = 403;
	self.backend.stagedBody = @"{\"error\":{\"message\":\"Decision API is not enabled for this user.\",\"type\":\"invalid_request_error\"}}";
	NSError *error = nil;
	XCTAssertNil([self.backend answersForState:@"s" questions:[self threeQuestions] error:&error]);
	XCTAssertTrue([error.localizedDescription containsString:@"403"]);
	XCTAssertTrue([error.localizedDescription containsString:@"not enabled"], @"the body explains the rejection");

	self.backend.stagedStatusCode = 429;
	XCTAssertNil([self.backend answersForState:@"s" questions:[self threeQuestions] error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceRateLimited);

	self.backend.stagedStatusCode = 200;
	self.backend.stagedBody = @"{\"model\":\"gpt-6-luna\"}";
	XCTAssertNil([self.backend answersForState:@"s" questions:[self threeQuestions] error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceBackendFailure);
}

#pragma mark The provider

- (void)testTheFactoryDerivesTheDecisionsURLFromTheOpenAIPresetOnly
{
	NFKOpenAIDecisionsBackend *luna = [NFKOpenAIDecisionsBackend backendForProvider:NFKRemoteProvider.openAI
																			apiKey:@"k" modelName:@"gpt-6-luna"];
	XCTAssertEqualObjects(luna.endpointURL.absoluteString, @"https://api.openai.com/v1/decisions");
	XCTAssertEqualObjects(luna.apiKey, @"k");
	XCTAssertTrue(luna.isReady);

	NFKRemoteProvider *proxied = [NFKRemoteProvider.openAI providerWithBaseURL:[NSURL URLWithString:@"http://gateway.test/openai/v1"]];
	XCTAssertEqualObjects([NFKOpenAIDecisionsBackend backendForProvider:proxied apiKey:@"k" modelName:@"m"].endpointURL.absoluteString,
						  @"http://gateway.test/openai/v1/decisions");
	XCTAssertNil([NFKOpenAIDecisionsBackend backendForProvider:NFKRemoteProvider.typeSafe apiKey:@"k" modelName:@"m"]);
}

#pragma mark Live

- (void)testALiveEndpointAnswersATypedDecision
{
	NSString *key = NSProcessInfo.processInfo.environment[@"INFERKIT_OPENAI_API_KEY"];
	if (key.length == 0) {
		XCTSkip("set INFERKIT_OPENAI_API_KEY to exercise the live path");
	}
	NFKOpenAIDecisionsBackend *luna = [NFKOpenAIDecisionsBackend backendForProvider:NFKRemoteProvider.openAI
																			apiKey:key modelName:@"gpt-6-luna"];
	NSError *error = nil;
	NSDictionary<NSString *, NFKDecisionAnswer *> *answers = [luna answersForState:@"Help! My payouts have been failing for 3 days."
																		 questions:[self threeQuestions] error:&error];
	XCTAssertNotNil(answers, @"%@", error);
	XCTAssertEqual(answers.count, 3);
	XCTAssertNotNil(answers[@"department"].choice);
	XCTAssertGreaterThan(answers[@"urgent"].probability, 0.5);
}

@end
