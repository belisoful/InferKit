//
//  NFKRemoteModelCatalogTests.m
//  InferKitTests
//
//  Model discovery through a stubbed transport: the request each provider style sends, the envelope
//  parse, Anthropic's pagination, and the two failure kinds a caller has to tell apart. A live test
//  against a local runner is gated on one being present.
//

#import <XCTest/XCTest.h>
#import <InferKit/NFKRemoteModelCatalog.h>
#import <InferKit/NFKRemoteModel.h>
#import <InferKit/NFKRemoteProvider.h>
#import <InferKit/NFKRemoteTransport.h>
#import <InferKit/NFKErrors.h>

/*! A catalog whose transport is stubbed: it records every request and returns staged pages in order. */
@interface NFKStubModelCatalog : NFKRemoteModelCatalog
@property (nonatomic, strong) NSMutableArray<NSURLRequest *> *requests;
@property (nonatomic, strong) NSMutableArray<NSString *> *stagedPages;
@property (nonatomic, assign) NSInteger stagedStatusCode;
@property (nonatomic, assign) BOOL unreachable;
@end

@implementation NFKStubModelCatalog

- (NSData *)sendRequest:(NSURLRequest *)request response:(NSHTTPURLResponse **)outResponse error:(NSError **)outError
{
	if (self.requests == nil) {
		self.requests = [NSMutableArray array];
	}
	[self.requests addObject:request];
	if (self.unreachable) {
		if (outError != NULL) {
			NSError *refused = [NSError errorWithDomain:NSURLErrorDomain code:NSURLErrorCannotConnectToHost userInfo:nil];
			*outError = [NSError errorWithDomain:NFKInferenceErrorDomain
											code:kNFKError_RemoteUnreachable
										userInfo:@{ NSUnderlyingErrorKey: refused,
													NSLocalizedDescriptionKey: @"localhost did not answer" }];
		}
		return nil;
	}
	if (outResponse != NULL) {
		*outResponse = [[NSHTTPURLResponse alloc] initWithURL:request.URL
												   statusCode:self.stagedStatusCode ?: 200
												  HTTPVersion:@"HTTP/1.1"
												 headerFields:nil];
	}
	NSString *page = self.stagedPages.firstObject ?: @"{\"data\":[]}";
	if (self.stagedPages.count > 0) {
		[self.stagedPages removeObjectAtIndex:0];
	}
	return [page dataUsingEncoding:NSUTF8StringEncoding];
}

@end

@interface NFKRemoteModelCatalogTests : XCTestCase
@end

@implementation NFKRemoteModelCatalogTests

- (NFKStubModelCatalog *)catalogFor:(NFKRemoteProvider *)provider pages:(NSArray<NSString *> *)pages
{
	NFKStubModelCatalog *catalog = [NFKStubModelCatalog catalogForProvider:provider apiKey:@"a-key"];
	catalog.stagedPages = [pages mutableCopy];
	return catalog;
}

#pragma mark The OpenAI envelope

- (void)testTheOpenAIEnvelopeBecomesModelsInTheProvidersOrder
{
	NFKStubModelCatalog *catalog = [self catalogFor:NFKRemoteProvider.ollama pages:@[
		(@"{\"object\":\"list\",\"data\":["
		 "{\"id\":\"llama3.2:latest\",\"object\":\"model\",\"created\":1700000000,\"owned_by\":\"library\"},"
		 "{\"id\":\"qwen3:8b\",\"object\":\"model\",\"created\":1700000100,\"owned_by\":\"library\"}]}") ]];

	NSError *error = nil;
	NSArray<NFKRemoteModel *> *models = [catalog modelsWithError:&error];
	XCTAssertNotNil(models, @"%@", error);
	XCTAssertEqual(models.count, 2);
	XCTAssertEqualObjects(models[0].identifier, @"llama3.2:latest");
	XCTAssertEqualObjects(models[0].displayName, @"llama3.2:latest", @"no display name published, so the id");
	XCTAssertEqualObjects(models[0].ownedBy, @"library");
	XCTAssertEqualWithAccuracy(models[0].createdAt.timeIntervalSince1970, 1700000000, 0.5);
	XCTAssertNil(models[0].contextLength, @"the OpenAI envelope does not carry one");
	XCTAssertEqualObjects(models[1].identifier, @"qwen3:8b");
	XCTAssertEqualObjects(models[1].raw[@"object"], @"model", @"the entry is kept as received");
}

- (void)testTheRequestIsAGETOfTheModelsURLWithABearerToken
{
	NFKStubModelCatalog *catalog = [self catalogFor:NFKRemoteProvider.openAI pages:@[]];
	[catalog modelsWithError:NULL];

	NSURLRequest *request = catalog.requests.firstObject;
	XCTAssertEqualObjects(request.HTTPMethod, @"GET");
	XCTAssertEqualObjects(request.URL.absoluteString, @"https://api.openai.com/v1/models");
	XCTAssertEqualObjects(request.allHTTPHeaderFields[@"Authorization"], @"Bearer a-key");
	XCTAssertNil(request.allHTTPHeaderFields[@"x-api-key"]);
}

- (void)testALocalRunnerSendsNoCredentialWhenNoneIsGiven
{
	NFKStubModelCatalog *catalog = [NFKStubModelCatalog catalogForProvider:NFKRemoteProvider.lmStudio apiKey:nil];
	[catalog modelsWithError:NULL];
	XCTAssertNil(catalog.requests.firstObject.allHTTPHeaderFields[@"Authorization"]);
}

- (void)testAnEntryWithoutAnIdentifierIsDroppedRatherThanBecomingAnEmptyName
{
	NFKStubModelCatalog *catalog = [self catalogFor:NFKRemoteProvider.ollama pages:@[
		@"{\"data\":[{\"object\":\"model\"},{\"id\":\"\"},{\"id\":\"kept\"},\"not-an-entry\"]}" ]];
	NSArray<NFKRemoteModel *> *models = [catalog modelsWithError:NULL];
	XCTAssertEqual(models.count, 1);
	XCTAssertEqualObjects(models.firstObject.identifier, @"kept");
}

// OpenRouter's entries carry a name and a context length; LM Studio's native list a max_context_length.
- (void)testTheFieldsSomeProvidersPublishAreReadWhenPresent
{
	NFKStubModelCatalog *catalog = [self catalogFor:NFKRemoteProvider.openRouter pages:@[
		(@"{\"data\":[{\"id\":\"meta/llama\",\"name\":\"Llama\",\"context_length\":131072},"
		 "{\"id\":\"local\",\"max_context_length\":4096}]}") ]];
	NSArray<NFKRemoteModel *> *models = [catalog modelsWithError:NULL];
	XCTAssertEqualObjects(models[0].displayName, @"Llama");
	XCTAssertEqualObjects(models[0].contextLength, @131072);
	XCTAssertEqualObjects(models[1].contextLength, @4096);
}

// GET /models/{id} answers the entry itself rather than a list, and a 404 for a name the provider
// does not know, so it is the cheap check before a request carries a name.
- (void)testOneModelByIdentifierReadsTheEntryAndNamesAMissingOne
{
	NFKStubModelCatalog *catalog = [self catalogFor:NFKRemoteProvider.ollama pages:@[
		@"{\"id\":\"llama3.2:latest\",\"object\":\"model\",\"owned_by\":\"library\"}" ]];
	NSError *error = nil;
	NFKRemoteModel *model = [catalog modelWithIdentifier:@"llama3.2:latest" error:&error];
	XCTAssertNotNil(model, @"%@", error);
	XCTAssertEqualObjects(model.identifier, @"llama3.2:latest");
	XCTAssertEqualObjects(catalog.requests.lastObject.URL.absoluteString, @"http://localhost:11434/v1/models/llama3.2:latest");

	catalog.stagedStatusCode = 404;
	catalog.stagedPages = [@[ @"{\"error\":{\"message\":\"model 'nonesuch' not found\"}}" ] mutableCopy];
	XCTAssertNil([catalog modelWithIdentifier:@"nonesuch" error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceBackendFailure);
	XCTAssertTrue([error.localizedDescription containsString:@"nonesuch"]);
}

#pragma mark Anthropic

- (void)testAnthropicSendsItsOwnHeadersAndFollowsThePagesToTheEnd
{
	NFKStubModelCatalog *catalog = [self catalogFor:NFKRemoteProvider.anthropic pages:@[
		(@"{\"data\":[{\"type\":\"model\",\"id\":\"claude-a\",\"display_name\":\"Claude A\","
		 "\"created_at\":\"2024-10-22T00:00:00Z\"}],\"has_more\":true,\"first_id\":\"claude-a\",\"last_id\":\"claude-a\"}"),
		(@"{\"data\":[{\"type\":\"model\",\"id\":\"claude-b\",\"display_name\":\"Claude B\","
		 "\"created_at\":\"2024-06-20T00:00:00Z\"}],\"has_more\":false,\"first_id\":\"claude-b\",\"last_id\":\"claude-b\"}") ]];

	NSError *error = nil;
	NSArray<NFKRemoteModel *> *models = [catalog modelsWithError:&error];
	XCTAssertNotNil(models, @"%@", error);
	XCTAssertEqual(models.count, 2, @"both pages are read");
	XCTAssertEqualObjects(models[0].displayName, @"Claude A");
	XCTAssertEqualObjects(models[1].identifier, @"claude-b");
	XCTAssertNotNil(models[0].createdAt, @"the ISO-8601 stamp parses");

	XCTAssertEqual(catalog.requests.count, 2);
	NSURLRequest *first = catalog.requests[0];
	XCTAssertEqualObjects(first.allHTTPHeaderFields[@"x-api-key"], @"a-key");
	XCTAssertNil(first.allHTTPHeaderFields[@"Authorization"], @"Anthropic does not take a Bearer token");
	XCTAssertEqualObjects(first.allHTTPHeaderFields[@"anthropic-version"], NFKAnthropicAPIVersion);
	XCTAssertTrue([first.URL.query containsString:@"limit=1000"], @"the page size is raised from the default 20");
	XCTAssertFalse([first.URL.query containsString:@"after_id"], @"the first page names no cursor");
	XCTAssertTrue([catalog.requests[1].URL.query containsString:@"after_id=claude-a"],
				  @"the second page continues from last_id");
}

- (void)testAnOpenAICompatibleProviderNeverPaginates
{
	NFKStubModelCatalog *catalog = [self catalogFor:NFKRemoteProvider.openAI pages:@[
		@"{\"data\":[{\"id\":\"m\"}],\"has_more\":true,\"last_id\":\"m\"}" ]];
	XCTAssertEqual([catalog modelsWithError:NULL].count, 1);
	XCTAssertEqual(catalog.requests.count, 1, @"has_more is not part of the OpenAI envelope");
	XCTAssertNil(catalog.requests.firstObject.URL.query);
}

#pragma mark The two failures a caller tells apart

- (void)testARunnerThatIsNotRunningIsUnreachableNotEmpty
{
	NFKStubModelCatalog *catalog = [self catalogFor:NFKRemoteProvider.ollama pages:@[]];
	catalog.unreachable = YES;

	NSError *error = nil;
	XCTAssertNil([catalog modelsWithError:&error]);
	XCTAssertEqualObjects(error.domain, NFKInferenceErrorDomain);
	XCTAssertEqual(error.code, kNFKError_RemoteUnreachable);
	XCTAssertNotNil(error.userInfo[NSUnderlyingErrorKey]);

	XCTAssertFalse([catalog isReachableWithError:&error]);
	XCTAssertEqual(error.code, kNFKError_RemoteUnreachable);
}

- (void)testAServerThatAnswersWithAnErrorIsReachableAndReportsTheStatus
{
	NFKStubModelCatalog *catalog = [self catalogFor:NFKRemoteProvider.openAI
											  pages:@[ @"{\"error\":{\"message\":\"bad key\"}}" ]];
	catalog.stagedStatusCode = 401;

	NSError *error = nil;
	XCTAssertNil([catalog modelsWithError:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceBackendFailure);
	XCTAssertTrue([error.localizedDescription containsString:@"401"]);
	XCTAssertTrue([error.localizedDescription containsString:@"bad key"], @"the body says why");

	catalog.stagedPages = [@[ @"{}" ] mutableCopy];
	XCTAssertTrue([catalog isReachableWithError:NULL], @"a rejected key is still a server that answered");
}

- (void)testABodyWithoutADataArrayIsAnError
{
	NFKStubModelCatalog *catalog = [self catalogFor:NFKRemoteProvider.openAI pages:@[ @"{\"models\":[]}" ]];
	NSError *error = nil;
	XCTAssertNil([catalog modelsWithError:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceBackendFailure);

	catalog.stagedPages = [@[ @"not json" ] mutableCopy];
	XCTAssertNil([catalog modelsWithError:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceBackendFailure);
}

#pragma mark Asynchronous and convenience forms

- (void)testTheCompletionHandlerDeliversTheSameList
{
	NFKStubModelCatalog *catalog = [self catalogFor:NFKRemoteProvider.ollama pages:@[ @"{\"data\":[{\"id\":\"m\"}]}" ]];
	XCTestExpectation *delivered = [self expectationWithDescription:@"models delivered"];
	[catalog modelsWithCompletionHandler:^(NSArray<NFKRemoteModel *> *models, NSError *error) {
		XCTAssertNil(error);
		XCTAssertEqualObjects(models.firstObject.identifier, @"m");
		[delivered fulfill];
	}];
	[self waitForExpectations:@[ delivered ] timeout:5];
}

- (void)testTheTransportReportsARefusedConnectionAsUnreachable
{
	// Port 9 is discard; nothing listens on it here, so the connection is refused rather than timing out.
	NFKRemoteModelCatalog *catalog = [NFKRemoteModelCatalog catalogForProvider:
									  [NFKRemoteProvider.ollama providerWithBaseURL:[NSURL URLWithString:@"http://127.0.0.1:9/v1"]]
																		  apiKey:nil];
	catalog.timeout = 5;
	NSError *error = nil;
	XCTAssertNil([catalog modelsWithError:&error]);
	XCTAssertEqualObjects(error.domain, NFKInferenceErrorDomain);
	XCTAssertEqual(error.code, kNFKError_RemoteUnreachable);
	XCTAssertEqualObjects([error.userInfo[NSUnderlyingErrorKey] domain], NSURLErrorDomain);
}

#pragma mark A live local runner

// Runs only when a local OpenAI-compatible server is listening: the whole path against a real
// implementation. Set INFERKIT_LIVE_LOCAL_MODEL to a model the server has; the list must contain it.
- (void)testALocalRunnerListsTheModelItServes
{
	NSDictionary *environment = NSProcessInfo.processInfo.environment;
	NSString *model = environment[@"INFERKIT_LIVE_LOCAL_MODEL"];
	if (model.length == 0) {
		XCTSkip("set INFERKIT_LIVE_LOCAL_MODEL (and run a local server) to exercise the live path");
	}
	NSString *identifier = environment[@"INFERKIT_LIVE_LOCAL_PROVIDER"] ?: @"ollama";
	NFKRemoteProvider *provider = [NFKRemoteProvider providerWithIdentifier:identifier];
	XCTAssertNotNil(provider, @"unknown provider %@", identifier);

	NSError *error = nil;
	NSArray<NFKRemoteModel *> *models = [provider modelsWithAPIKey:nil error:&error];
	XCTAssertNotNil(models, @"%@", error.localizedDescription);
	NSArray<NSString *> *identifiers = [models valueForKey:@"identifier"];
	XCTAssertTrue([identifiers containsObject:model], @"%@ is not among %@", model, identifiers);
}

@end
