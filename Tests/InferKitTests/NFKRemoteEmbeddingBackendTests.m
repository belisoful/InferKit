//
//  NFKRemoteEmbeddingBackendTests.m
//  InferKitTests
//
//  The embeddings request and envelope through a stub transport, and the provider factory's URL
//  derivation. A live call is gated on a local runner being present.
//

#import <XCTest/XCTest.h>
#import <InferKit/NFKRemoteEmbeddingBackend.h>
#import <InferKit/NFKRemoteProvider.h>
#import <InferKit/NFKRemoteBackend.h>
#import <InferKit/NFKInferenceRequest.h>
#import <InferKit/NFKInferenceResult.h>
#import <InferKit/NFKInferenceKeys.h>
#import <InferKit/NFKErrors.h>

/*! An embedding backend whose transport is stubbed: it records the request and returns staged data. */
@interface NFKStubEmbeddingBackend : NFKRemoteEmbeddingBackend
@property (nonatomic, strong) NSURLRequest *lastRequest;
@property (nonatomic, strong, nullable) NSData *stagedData;
@property (nonatomic, assign) NSInteger stagedStatusCode;
@end

@implementation NFKStubEmbeddingBackend

- (NSData *)sendRequest:(NSURLRequest *)request response:(NSHTTPURLResponse **)outResponse error:(NSError **)outError
{
	self.lastRequest = request;
	if (outResponse != NULL) {
		*outResponse = [[NSHTTPURLResponse alloc] initWithURL:request.URL
												   statusCode:self.stagedStatusCode ?: 200
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

@interface NFKRemoteEmbeddingBackendTests : XCTestCase
@property (nonatomic, strong) NFKStubEmbeddingBackend *backend;
@end

@implementation NFKRemoteEmbeddingBackendTests

- (void)setUp
{
	[super setUp];
	self.backend = [[NFKStubEmbeddingBackend alloc] init];
	self.backend.endpointURL = [NSURL URLWithString:@"http://localhost:11434/v1/embeddings"];
	self.backend.modelName = @"nomic-embed-text";
	self.backend.stagedData = [@"{\"object\":\"list\",\"data\":[{\"object\":\"embedding\",\"index\":0,"
								"\"embedding\":[0.1,0.2,0.3]}],\"model\":\"nomic-embed-text\"}"
							   dataUsingEncoding:NSUTF8StringEncoding];
}

- (void)testTheBackendReportsItsIdentity
{
	XCTAssertEqualObjects(self.backend.backendIdentifier, @"remote-embedding");
	XCTAssertTrue(self.backend.isReady);
	XCTAssertFalse([NFKRemoteEmbeddingBackend backendWithEndpointURL:nil].isReady);
}

- (void)testAPromptBecomesTheInputAndTheVectorComesBackUnderTheCoreKey
{
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"a red barn" }];
	NSError *error = nil;
	NFKInferenceResult *result = [self.backend runInferenceForRequest:request error:&error];
	XCTAssertNotNil(result, @"%@", error);

	NSDictionary *body = [self.backend decodedRequestBody];
	XCTAssertEqualObjects(body[@"model"], @"nomic-embed-text");
	XCTAssertEqualObjects(body[@"input"], @"a red barn");
	XCTAssertEqualObjects(self.backend.lastRequest.HTTPMethod, @"POST");
	XCTAssertEqualObjects(self.backend.lastRequest.URL.absoluteString, @"http://localhost:11434/v1/embeddings");

	NSArray *vector = result.embedding;
	XCTAssertEqualObjects(vector, (@[ @0.1, @0.2, @0.3 ]));
	XCTAssertNotNil([result outputForKey:NFKRemoteBackendRawKey]);
}

- (void)testMessagesAreJoinedWhenThereIsNoPrompt
{
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputMessages: @[
		@{ @"role": @"user", @"content": @"first" },
		@{ @"role": @"assistant", @"content": @"second" },
	] }];
	[self.backend runInferenceForRequest:request error:NULL];
	XCTAssertEqualObjects([self.backend decodedRequestBody][@"input"], @"first\nsecond");
}

- (void)testParametersFoldIntoTheBodyAndTheKeyIsABearerToken
{
	self.backend.apiKey = @"a-key";
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"x" }
															  parameters:@{ @"dimensions": @256 }
														  outputModality:NFKModalityText];
	[self.backend runInferenceForRequest:request error:NULL];
	XCTAssertEqualObjects([self.backend decodedRequestBody][@"dimensions"], @256);
	XCTAssertEqualObjects(self.backend.lastRequest.allHTTPHeaderFields[@"Authorization"], @"Bearer a-key");
}

- (void)testAnEmptyRequestIsMissingInput
{
	NSError *error = nil;
	XCTAssertNil([self.backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{}] error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceMissingInput);
}

// The provider answers with an index per vector, and a batch is ordered by it rather than by the
// position it arrived in, so the ith vector belongs to the ith text.
- (void)testABatchIsOrderedByIndexNotArrival
{
	self.backend.stagedData = [@"{\"data\":[{\"index\":1,\"embedding\":[2]},{\"index\":0,\"embedding\":[1]}]}"
							   dataUsingEncoding:NSUTF8StringEncoding];
	NSError *error = nil;
	NSArray<NSArray<NSNumber *> *> *vectors = [self.backend embeddingsForTexts:@[ @"one", @"two" ] error:&error];
	XCTAssertNotNil(vectors, @"%@", error);
	XCTAssertEqualObjects(vectors, (@[ @[ @1 ], @[ @2 ] ]));
	XCTAssertEqualObjects([self.backend decodedRequestBody][@"input"], (@[ @"one", @"two" ]));
}

- (void)testABatchWithTheWrongCountBackIsAnError
{
	NSError *error = nil;
	XCTAssertNil(([self.backend embeddingsForTexts:@[ @"one", @"two" ] error:&error]), @"one vector for two texts");
	XCTAssertEqual(error.code, kNFKError_InferenceBackendFailure);

	XCTAssertNil([self.backend embeddingsForTexts:@[] error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceMissingInput);
}

- (void)testAFailingStatusAndAnEmptyEnvelopeAreErrors
{
	self.backend.stagedStatusCode = 404;
	self.backend.stagedData = [@"{\"error\":{\"message\":\"model not found\"}}" dataUsingEncoding:NSUTF8StringEncoding];
	NSError *error = nil;
	XCTAssertNil([self.backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"x" }]
												error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceBackendFailure);
	XCTAssertTrue([error.localizedDescription containsString:@"model not found"]);

	self.backend.stagedStatusCode = 200;
	self.backend.stagedData = [@"{\"data\":[]}" dataUsingEncoding:NSUTF8StringEncoding];
	XCTAssertNil([self.backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"x" }]
												error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceBackendFailure);
}

#pragma mark The provider factory

- (void)testTheFactoryDerivesTheEmbeddingsURLAndDeclinesAnthropic
{
	NFKRemoteEmbeddingBackend *ollama = [NFKRemoteEmbeddingBackend backendForProvider:NFKRemoteProvider.ollama
																				apiKey:nil modelName:@"m"];
	XCTAssertEqualObjects(ollama.endpointURL.absoluteString, @"http://localhost:11434/v1/embeddings");
	XCTAssertEqualObjects(ollama.modelName, @"m");

	NFKRemoteEmbeddingBackend *groq = [NFKRemoteEmbeddingBackend backendForProvider:NFKRemoteProvider.groq
																			  apiKey:@"k" modelName:@"m"];
	XCTAssertEqualObjects(groq.endpointURL.absoluteString, @"https://api.groq.com/openai/v1/embeddings");
	XCTAssertEqualObjects(groq.apiKey, @"k");

	XCTAssertNil([NFKRemoteEmbeddingBackend backendForProvider:NFKRemoteProvider.anthropic apiKey:@"k" modelName:@"m"],
				 @"Anthropic serves no embeddings endpoint");
}

#pragma mark A live local runner

// Set INFERKIT_LIVE_EMBEDDING_MODEL to an embedding model the local server has (nomic-embed-text,
// all-minilm); a chat model also answers on Ollama, at the cost of loading it.
- (void)testALocalRunnerEmbedsWhenOneIsRunning
{
	NSDictionary *environment = NSProcessInfo.processInfo.environment;
	NSString *model = environment[@"INFERKIT_LIVE_EMBEDDING_MODEL"];
	if (model.length == 0) {
		XCTSkip("set INFERKIT_LIVE_EMBEDDING_MODEL (and run a local server) to exercise the live path");
	}
	NSString *identifier = environment[@"INFERKIT_LIVE_LOCAL_PROVIDER"] ?: @"ollama";
	NFKRemoteProvider *provider = [NFKRemoteProvider providerWithIdentifier:identifier];
	NFKRemoteEmbeddingBackend *backend = [NFKRemoteEmbeddingBackend backendForProvider:provider apiKey:nil modelName:model];

	NSError *error = nil;
	NSArray<NSArray<NSNumber *> *> *vectors = [backend embeddingsForTexts:@[ @"a red barn", @"a red barn", @"quarterly revenue" ]
																	error:&error];
	XCTAssertNotNil(vectors, @"%@", error.localizedDescription);
	XCTAssertEqual(vectors.count, 3);
	XCTAssertGreaterThan(vectors[0].count, 0);
	XCTAssertEqualObjects(vectors[0], vectors[1], @"the same text embeds identically");
	XCTAssertNotEqualObjects(vectors[0], vectors[2]);
}

@end
