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

- (void)testTheAuthorizationHeaderCarriesTheAPIKey
{
	self.backend.apiKey = @"secret";
	self.backend.stagedData = [@"{\"choices\":[]}" dataUsingEncoding:NSUTF8StringEncoding];
	[self.backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{}] error:NULL];
	XCTAssertEqualObjects([self.backend.lastRequest valueForHTTPHeaderField:@"Authorization"], @"Bearer secret");
}

@end
