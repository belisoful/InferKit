//
//  NFKGeminiDecisionsBackendTests.m
//  InferKitTests
//
//  Gemini's Decisions API request and reply through a stub transport: the wire shape, the typed
//  answers, the input forms and content blocks, the errors, and the provider factory. A live test
//  against the hosted endpoint is gated on a key.
//

#import <XCTest/XCTest.h>
#import <CoreGraphics/CoreGraphics.h>
#import <InferKit/NFKGeminiDecisionsBackend.h>
#import <InferKit/NFKRemoteProvider.h>
#import <InferKit/NFKRemoteFileStore.h>
#import <InferKit/NFKInferenceRequest.h>
#import <InferKit/NFKInferenceResult.h>
#import <InferKit/NFKInferenceKeys.h>
#import <InferKit/NFKErrors.h>

@interface NFKStubGeminiDecisionsBackend : NFKGeminiDecisionsBackend
@property (nonatomic, strong) NSURLRequest *lastRequest;
@property (nonatomic, copy) NSString *stagedBody;
@property (nonatomic, assign) NSInteger stagedStatusCode;
@end

@implementation NFKStubGeminiDecisionsBackend
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

@interface NFKGeminiDecisionsBackendTests : XCTestCase
@property (nonatomic, strong) NFKStubGeminiDecisionsBackend *backend;
@end

@implementation NFKGeminiDecisionsBackendTests

- (void)setUp
{
	[super setUp];
	self.backend = [[NFKStubGeminiDecisionsBackend alloc] init];
	self.backend.apiKey = @"g-key";
	self.backend.modelName = @"gemini-decision-1";
	self.backend.stagedBody = @"{\"answers\":{"
		"\"department\":{\"type\":\"choice\",\"choice\":\"technical\",\"confidence\":0.85,"
		"\"probabilities\":{\"billing\":0.08,\"technical\":0.85,\"sales\":0.07}},"
		"\"severity\":{\"type\":\"score\",\"score\":1.6,\"confidence\":0.7,"
		"\"legend\":{\"0\":\"low\",\"1\":\"medium\",\"2\":\"high\"},"
		"\"probabilities\":{\"0\":0.1,\"1\":0.2,\"2\":0.7}},"
		"\"urgent\":{\"type\":\"noul\",\"noul\":0.91,\"confidence\":0.91}},"
		"\"usage\":{\"total_input_tokens\":312,\"input_tokens_by_modality\":[{\"modality\":\"text\",\"tokens\":312}],"
		"\"total_output_tokens\":0,\"total_tokens\":312}}";
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

- (CGImageRef)newSquareImage CF_RETURNS_RETAINED
{
	CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
	CGContextRef context = CGBitmapContextCreate(NULL, 2, 2, 8, 8, colorSpace, kCGBitmapByteOrder32Little | kCGImageAlphaPremultipliedFirst);
	CGColorSpaceRelease(colorSpace);
	CGImageRef square = CGBitmapContextCreateImage(context);
	CGContextRelease(context);
	return square;
}

#pragma mark The wire shape

- (void)testTheRequestCarriesTheModelTheInputAndKeyedQuestionsWithTheGoogleKeyHeader
{
	NSError *error = nil;
	XCTAssertNotNil([self.backend runInferenceForRequest:[self requestWithState:@"Help! My payouts have been failing for 3 days."]
												   error:&error], @"%@", error);

	XCTAssertEqualObjects(self.backend.lastRequest.URL.absoluteString, @"https://generativelanguage.googleapis.com/v1beta/decisions");
	XCTAssertEqualObjects(self.backend.lastRequest.HTTPMethod, @"POST");
	XCTAssertEqualObjects(self.backend.lastRequest.allHTTPHeaderFields[@"x-goog-api-key"], @"g-key");
	XCTAssertNil(self.backend.lastRequest.allHTTPHeaderFields[@"Authorization"]);
	XCTAssertEqualObjects(self.backend.lastRequest.allHTTPHeaderFields[@"Content-Type"], @"application/json");

	NSDictionary *body = [self decodedRequestBody];
	XCTAssertEqualObjects(body[@"model"], @"gemini-decision-1");
	XCTAssertEqualObjects(body[@"input"], @"Help! My payouts have been failing for 3 days.");
	NSDictionary *questions = body[@"questions"];
	NSDictionary *department = @{ @"type": @"choice", @"instructions": @"Which team should handle this?",
								  @"criteria": @{ @"billing": @"Payments, invoicing, refunds",
												  @"technical": @"Bugs, outages, integrations",
												  @"sales": @"" } };
	XCTAssertEqualObjects(questions[@"department"], department, @"an undescribed option carries an empty description");
	NSDictionary *severity = @{ @"type": @"score", @"instructions": @"How severe is the problem?",
								@"criteria": @[ @"low", @"medium", @"high" ] };
	XCTAssertEqualObjects(questions[@"severity"], severity);
	NSDictionary *urgent = @{ @"type": @"noul", @"instructions": @"The customer needs an answer today." };
	XCTAssertEqualObjects(questions[@"urgent"], urgent, @"a noul with no meanings carries no criteria");
	XCTAssertEqualObjects(self.backend.backendIdentifier, @"gemini-decisions");
}

- (void)testANoulsMeaningsRideAsTrueAndFalseDescriptions
{
	NFKDecisionQuestion *refund = [NFKDecisionQuestion noulQuestionWithInstructions:@"The customer asks for a refund."
																		trueMeaning:@"money back is requested"
																	   falseMeaning:nil];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputState: @"s", NFKInputQuestions: @{ @"refund": refund } }];
	XCTAssertNotNil([self.backend runInferenceForRequest:request error:NULL]);
	NSDictionary *criteria = @{ @"true_description": @"money back is requested" };
	XCTAssertEqualObjects([self decodedRequestBody][@"questions"][@"refund"][@"criteria"], criteria);
}

- (void)testARecordIsSentAsJSONTextAndContentBlocksAsIs
{
	NSDictionary *record = @{ @"subject": @"Refund", @"tags": @[ @"billing" ] };
	XCTAssertNotNil([self.backend runInferenceForRequest:[self requestWithState:record] error:NULL]);
	XCTAssertEqualObjects([self decodedRequestBody][@"input"], @"{\"subject\":\"Refund\",\"tags\":[\"billing\"]}");

	NSArray *blocks = @[ @{ @"type": @"text", @"text": @"hi" },
						 @{ @"type": @"image", @"uri": @"https://example.test/parcel.png", @"mime_type": @"image/png" } ];
	XCTAssertNotNil([self.backend runInferenceForRequest:[self requestWithState:blocks] error:NULL]);
	XCTAssertEqualObjects([self decodedRequestBody][@"input"], blocks);
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

- (void)testMediaRidesAsContentBlocksAfterTheTextWithTheRequestsResolution
{
	CGImageRef square = [self newSquareImage];
	NFKRemoteFile *clip = [NFKRemoteFile fileWithGeminiURI:[NSURL URLWithString:@"https://generativelanguage.googleapis.com/v1beta/files/v1"]
												  mimeType:@"video/mp4"];
	NSData *wav = [@"RIFF" dataUsingEncoding:NSASCIIStringEncoding];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputState: @"Inspect the parcel.",
																			 NFKInputImage: (__bridge id)square,
																			 NFKInputAudio: wav,
																			 NFKInputDocuments: (@[ clip, @"Fragile goods ship insured." ]),
																			 NFKInputQuestions: [self threeQuestions] }
															   parameters:@{ @"resolution": @"low" }
														   outputModality:NFKModalityText];
	NSError *error = nil;
	XCTAssertNotNil([self.backend runInferenceForRequest:request error:&error], @"%@", error);
	CGImageRelease(square);

	NSDictionary *body = [self decodedRequestBody];
	XCTAssertNil(body[@"resolution"], @"the resolution belongs to the blocks");
	NSArray<NSDictionary *> *input = body[@"input"];
	XCTAssertEqualObjects([input valueForKey:@"type"], (@[ @"text", @"image", @"audio", @"video", @"text" ]));
	NSDictionary *text = @{ @"type": @"text", @"text": @"Inspect the parcel." };
	XCTAssertEqualObjects(input[0], text);
	XCTAssertEqualObjects(input[1][@"mime_type"], @"image/png");
	XCTAssertEqualObjects(input[1][@"resolution"], @"low");
	XCTAssertGreaterThan([input[1][@"data"] length], 0);
	XCTAssertEqualObjects(input[2][@"mime_type"], @"audio/wav");
	XCTAssertNil(input[2][@"resolution"], @"audio takes no resolution");
	NSDictionary *video = @{ @"type": @"video", @"uri": @"https://generativelanguage.googleapis.com/v1beta/files/v1",
							 @"mime_type": @"video/mp4", @"resolution": @"low" };
	XCTAssertEqualObjects(input[3], video);
	XCTAssertTrue([input[4][@"text"] hasSuffix:@"Fragile goods ship insured."]);
}

- (void)testMediaAloneIsTheInput
{
	CGImageRef square = [self newSquareImage];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputImage: (__bridge id)square,
																			 NFKInputQuestions: [self threeQuestions] }];
	XCTAssertNotNil([self.backend runInferenceForRequest:request error:NULL]);
	CGImageRelease(square);
	NSArray<NSDictionary *> *input = [self decodedRequestBody][@"input"];
	XCTAssertEqualObjects([input valueForKey:@"type"], (@[ @"image" ]));
	XCTAssertNil(input[0][@"resolution"], @"no resolution was asked for");
}

- (void)testAQuestionAlreadyInWireShapePassesThrough
{
	NSDictionary *wire = @{ @"type": @"noul", @"instructions": @"Damaged?", @"criteria": @{ @"true_description": @"a crack or a dent" } };
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputState: @"s", NFKInputQuestions: @{ @"damaged": wire } }];
	XCTAssertNotNil([self.backend runInferenceForRequest:request error:NULL]);
	XCTAssertEqualObjects([self decodedRequestBody][@"questions"][@"damaged"], wire);
}

- (void)testAnUnnamedParameterFoldsIntoTheBodyWithoutOverridingTheRequestFields
{
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputState: @"s", NFKInputQuestions: [self threeQuestions] }
															   parameters:@{ @"labels": @{ @"team": @"support" }, @"model": @"not-this-one", @"input": @"nor this" }
														   outputModality:NFKModalityText];
	XCTAssertNotNil([self.backend runInferenceForRequest:request error:NULL]);
	NSDictionary *body = [self decodedRequestBody];
	XCTAssertEqualObjects(body[@"labels"], (@{ @"team": @"support" }));
	XCTAssertEqualObjects(body[@"model"], @"gemini-decision-1");
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
	XCTAssertEqualWithAccuracy(department.confidence, 0.85, 1e-9);

	NFKDecisionAnswer *severity = answers[@"severity"];
	XCTAssertEqual(severity.type, NFKDecisionTypeScore);
	XCTAssertEqualWithAccuracy(severity.score, 1.6, 1e-9);
	XCTAssertEqualObjects(severity.legend[@"2"], @"high");
	XCTAssertEqualWithAccuracy(severity.probabilities[@"2"].doubleValue, 0.7, 1e-9);

	NFKDecisionAnswer *urgent = answers[@"urgent"];
	XCTAssertEqual(urgent.type, NFKDecisionTypeNoul);
	XCTAssertEqualWithAccuracy(urgent.probability, 0.91, 1e-9);
	XCTAssertEqualWithAccuracy(urgent.confidence, 0.91, 1e-9, @"the service reports a confidence for a noul too");
	XCTAssertFalse(urgent.isRefused);

	XCTAssertNotNil(result.structured[@"answers"], @"the whole reply rides under structured");
	NSDictionary *usage = [result outputForKey:NFKOutputUsage];
	XCTAssertEqualObjects(usage[NFKUsageInputTokens], @312);
	XCTAssertEqualObjects(usage[NFKUsageOutputTokens], @0);
	XCTAssertNil(usage[NFKUsageCachedTokens], @"the service reports no cached count");
}

- (void)testAnAnswerOfAnUnknownTypeIsDroppedAndKeptInTheRawReply
{
	self.backend.stagedBody = @"{\"answers\":{\"department\":{\"type\":\"ranking\",\"order\":[\"sales\"]},"
		"\"urgent\":{\"type\":\"noul\",\"noul\":0.2,\"confidence\":0.8}}}";
	NFKInferenceResult *result = [self.backend runInferenceForRequest:[self requestWithState:@"s"] error:NULL];
	XCTAssertEqual(result.answers.count, 1);
	XCTAssertEqualWithAccuracy(result.answers[@"urgent"].probability, 0.2, 1e-12);
	XCTAssertNotNil(result.structured[@"answers"][@"department"]);
	XCTAssertNil([result outputForKey:NFKOutputUsage], @"a reply without usage reports none");
}

- (void)testABlockedRequestRefusesEveryQuestionWithItsOwnType
{
	self.backend.stagedBody = @"{\"answers\":{},\"prompt_feedback\":{\"block_reason\":\"safety\"}}";
	NFKInferenceResult *result = [self.backend runInferenceForRequest:[self requestWithState:@"s"] error:NULL];
	XCTAssertEqual(result.answers.count, 3);
	XCTAssertTrue(result.answers[@"department"].isRefused);
	XCTAssertEqual(result.answers[@"department"].type, NFKDecisionTypeChoice);
	XCTAssertEqual(result.answers[@"severity"].type, NFKDecisionTypeScore);
	XCTAssertEqual(result.answers[@"urgent"].type, NFKDecisionTypeNoul);
	XCTAssertEqualObjects(result.answers[@"urgent"].raw[@"block_reason"], @"safety");

	self.backend.stagedBody = @"{\"prompt_feedback\":{\"block_reason\":\"blocklist\"}}";
	result = [self.backend runInferenceForRequest:[self requestWithState:@"s"] error:NULL];
	XCTAssertTrue(result.answers[@"severity"].isRefused, @"a blocked reply may carry no answers field at all");

	self.backend.stagedBody = @"{\"answers\":{\"urgent\":{\"type\":\"noul\",\"noul\":0.4,\"confidence\":0.6}},\"prompt_feedback\":{},"
		"\"usage\":{\"total_input_tokens\":90,\"total_cached_tokens\":64,\"total_output_tokens\":0,\"total_thought_tokens\":0}}";
	result = [self.backend runInferenceForRequest:[self requestWithState:@"s"] error:NULL];
	XCTAssertFalse(result.answers[@"urgent"].isRefused, @"feedback without a block reason refuses nothing");
	XCTAssertEqualObjects([result outputForKey:NFKOutputUsage][NFKUsageCachedTokens], @64);
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

- (void)testAMissingInputMissingQuestionsOrUnsendableStateIsAnErrorBeforeAnythingIsSent
{
	NSError *error = nil;
	NFKInferenceRequest *noInput = [NFKInferenceRequest requestWithInputs:@{ NFKInputQuestions: [self threeQuestions] }];
	XCTAssertNil([self.backend runInferenceForRequest:noInput error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceMissingInput);

	NFKInferenceRequest *noQuestions = [NFKInferenceRequest requestWithInputs:@{ NFKInputState: @"s" }];
	XCTAssertNil([self.backend runInferenceForRequest:noQuestions error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceMissingInput);

	XCTAssertNil([self.backend runInferenceForRequest:[self requestWithState:NSDate.date] error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceMissingInput);
	XCTAssertNil(self.backend.lastRequest);
}

- (void)testASafetyBlockAFailingStatusAndAnAnswerlessReplyAreErrors
{
	self.backend.stagedStatusCode = 400;
	self.backend.stagedBody = @"{\"error\":{\"message\":\"The request was blocked by safety filters.\",\"code\":\"safety\"}}";
	NSError *error = nil;
	XCTAssertNil([self.backend answersForState:@"s" questions:[self threeQuestions] error:&error]);
	XCTAssertTrue([error.localizedDescription containsString:@"400"]);
	XCTAssertTrue([error.localizedDescription containsString:@"safety"], @"the body names the block");

	self.backend.stagedStatusCode = 429;
	XCTAssertNil([self.backend answersForState:@"s" questions:[self threeQuestions] error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceRateLimited);

	self.backend.stagedStatusCode = 200;
	self.backend.stagedBody = @"{\"usage\":{\"total_input_tokens\":4}}";
	XCTAssertNil([self.backend answersForState:@"s" questions:[self threeQuestions] error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceBackendFailure);
}

#pragma mark The provider

- (void)testTheFactoryDerivesTheDecisionsURLBesideTheGeminiPresetsOpenAILayerOnly
{
	NFKGeminiDecisionsBackend *decider = [NFKGeminiDecisionsBackend backendForProvider:NFKRemoteProvider.googleGemini
																				apiKey:@"k" modelName:@"gemini-decision-1"];
	XCTAssertEqualObjects(decider.endpointURL.absoluteString, @"https://generativelanguage.googleapis.com/v1beta/decisions");
	XCTAssertEqualObjects(decider.apiKey, @"k");
	XCTAssertTrue(decider.isReady);

	NFKRemoteProvider *proxied = [NFKRemoteProvider.googleGemini providerWithBaseURL:[NSURL URLWithString:@"http://gateway.test/google/v1beta/openai"]];
	XCTAssertEqualObjects([NFKGeminiDecisionsBackend backendForProvider:proxied apiKey:@"k" modelName:@"m"].endpointURL.absoluteString,
						  @"http://gateway.test/google/v1beta/decisions");
	XCTAssertNil([NFKGeminiDecisionsBackend backendForProvider:NFKRemoteProvider.openAI apiKey:@"k" modelName:@"m"]);
}

#pragma mark Live

- (void)testALiveEndpointAnswersATypedDecision
{
	NSString *key = NSProcessInfo.processInfo.environment[@"INFERKIT_GEMINI_API_KEY"];
	if (key.length == 0) {
		XCTSkip("set INFERKIT_GEMINI_API_KEY to exercise the live path");
	}
	NFKGeminiDecisionsBackend *decider = [NFKGeminiDecisionsBackend backendForProvider:NFKRemoteProvider.googleGemini
																				apiKey:key modelName:@"gemini-decision-1"];
	NSError *error = nil;
	NSDictionary<NSString *, NFKDecisionAnswer *> *answers = [decider answersForState:@"Help! My payouts have been failing for 3 days."
																			questions:[self threeQuestions] error:&error];
	XCTAssertNotNil(answers, @"%@", error);
	XCTAssertEqual(answers.count, 3);
	XCTAssertNotNil(answers[@"department"].choice);
	XCTAssertGreaterThan(answers[@"urgent"].probability, 0.5);
}

@end
