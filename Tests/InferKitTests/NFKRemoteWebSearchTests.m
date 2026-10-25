//
//  NFKRemoteWebSearchTests.m
//  InferKitTests
//
//  Perplexity's Search API through a stub transport: the body, the ranked results, the provider
//  factory, and the errors. A live test against the hosted endpoint is gated on a key.
//

#import <XCTest/XCTest.h>
#import <InferKit/NFKRemoteWebSearch.h>
#import <InferKit/NFKRemoteProvider.h>
#import <InferKit/NFKErrors.h>

@interface NFKStubWebSearch : NFKRemoteWebSearch
@property (nonatomic, strong) NSURLRequest *lastRequest;
@property (nonatomic, copy) NSString *stagedBody;
@property (nonatomic, assign) NSInteger stagedStatusCode;
@end

@implementation NFKStubWebSearch

- (NSData *)sendRequest:(NSURLRequest *)request response:(NSHTTPURLResponse **)outResponse error:(NSError **)outError
{
	self.lastRequest = request;
	if (outResponse != NULL) {
		*outResponse = [[NSHTTPURLResponse alloc] initWithURL:request.URL statusCode:self.stagedStatusCode ?: 200
												  HTTPVersion:@"HTTP/1.1" headerFields:nil];
	}
	return [self.stagedBody dataUsingEncoding:NSUTF8StringEncoding];
}

- (NSDictionary *)decodedRequestBody
{
	return [NSJSONSerialization JSONObjectWithData:self.lastRequest.HTTPBody options:0 error:NULL];
}

@end

@interface NFKRemoteWebSearchTests : XCTestCase
@property (nonatomic, strong) NFKStubWebSearch *search;
@end

@implementation NFKRemoteWebSearchTests

- (void)setUp
{
	[super setUp];
	self.search = [NFKStubWebSearch webSearchWithEndpointURL:[NSURL URLWithString:@"https://api.perplexity.ai/search"]];
	self.search.apiKey = @"pplx-key";
	self.search.stagedBody = @"{\"id\":\"s1\",\"server_time\":null,\"results\":["
		"{\"title\":\"Release notes\",\"url\":\"https://example.test/notes\",\"snippet\":\"Version 2 ships.\",\"date\":\"2026-10-20\",\"last_updated\":\"2026-10-22\"},"
		"{\"title\":\"No snippet\",\"url\":\"https://example.test/empty\"},"
		"{\"title\":\"Blog\",\"url\":\"https://example.test/blog\",\"snippet\":\"Earlier news.\"}]}";
}

- (void)testTheQueryAndTheParametersGoOutWithABearerToken
{
	NSError *error = nil;
	NSDictionary *parameters = @{ @"max_results": @5, @"search_recency_filter": @"week", @"query": @"ignored" };
	XCTAssertNotNil([self.search resultsForQuery:@"release notes" parameters:parameters error:&error], @"%@", error);
	XCTAssertEqualObjects(self.search.lastRequest.HTTPMethod, @"POST");
	XCTAssertEqualObjects(self.search.lastRequest.URL.absoluteString, @"https://api.perplexity.ai/search");
	XCTAssertEqualObjects([self.search.lastRequest valueForHTTPHeaderField:@"Authorization"], @"Bearer pplx-key");
	XCTAssertEqualObjects([self.search decodedRequestBody], (@{ @"query": @"release notes", @"max_results": @5, @"search_recency_filter": @"week" }));
}

- (void)testTheResultsComeBackRankedWithAnIncompleteEntrySkipped
{
	NSError *error = nil;
	NSArray<NFKWebSearchResult *> *results = [self.search resultsForQuery:@"release notes" error:&error];
	XCTAssertEqual(results.count, 2u, @"%@", error);
	XCTAssertEqualObjects(results[0].title, @"Release notes");
	XCTAssertEqualObjects(results[0].URL.absoluteString, @"https://example.test/notes");
	XCTAssertEqualObjects(results[0].snippet, @"Version 2 ships.");
	XCTAssertEqualObjects(results[0].date, @"2026-10-20");
	XCTAssertEqualObjects(results[0].lastUpdated, @"2026-10-22");
	XCTAssertEqualObjects(results[0].raw[@"title"], @"Release notes");
	XCTAssertNil(results[1].date);
	XCTAssertEqualObjects(results[1].title, @"Blog");
}

// Several queries go as one request; a list nested per query reads in order.
- (void)testSeveralQueriesGoAsOneRequestAndANestedListReadsInOrder
{
	self.search.stagedBody = @"{\"id\":\"s2\",\"results\":[[{\"title\":\"A\",\"url\":\"https://a.test\",\"snippet\":\"a\"}],"
		"[{\"title\":\"B\",\"url\":\"https://b.test\",\"snippet\":\"b\"}]]}";
	NSArray<NFKWebSearchResult *> *results = [self.search resultsForQuery:@[ @"first", @"second" ] parameters:nil error:NULL];
	XCTAssertEqualObjects([results valueForKey:@"title"], (@[ @"A", @"B" ]));
	XCTAssertEqualObjects([self.search decodedRequestBody][@"query"], (@[ @"first", @"second" ]));
}

- (void)testAMissingQueryAFailingStatusAndAnEnvelopeWithoutResultsAreErrors
{
	NSError *error = nil;
	XCTAssertNil([self.search resultsForQuery:@"" error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceMissingInput);
	XCTAssertNil(self.search.lastRequest, @"nothing is sent without a query");

	self.search.stagedStatusCode = 422;
	self.search.stagedBody = @"{\"detail\":[{\"loc\":[\"body\",\"max_results\"],\"msg\":\"too large\",\"type\":\"value_error\"}]}";
	XCTAssertNil([self.search resultsForQuery:@"q" error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceRefused);

	self.search.stagedStatusCode = 200;
	self.search.stagedBody = @"{\"id\":\"s3\"}";
	XCTAssertNil([self.search resultsForQuery:@"q" error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceBackendFailure);
}

- (void)testTheFactoryServesPerplexityAtTheRootOfItsHostAndDeclinesTheRest
{
	NFKRemoteWebSearch *search = [NFKRemoteWebSearch webSearchForProvider:NFKRemoteProvider.perplexity apiKey:@"k"];
	XCTAssertEqualObjects(search.endpointURL.absoluteString, @"https://api.perplexity.ai/search");
	XCTAssertEqualObjects(search.apiKey, @"k");
	for (NFKRemoteProvider *provider in @[ NFKRemoteProvider.openAI, NFKRemoteProvider.typeSafe, NFKRemoteProvider.ollama ]) {
		XCTAssertNil([NFKRemoteWebSearch webSearchForProvider:provider apiKey:@"k"], @"%@ serves no search", provider.identifier);
	}
}

#pragma mark Live

// Runs against Perplexity's Search API when a key is present. POST /search answers 401 without a key
// and /searchx 404, so the path is verified to exist; the reply shape is verified only here.
- (void)testALiveSearchAnswersRankedPages
{
	NSString *key = NSProcessInfo.processInfo.environment[@"INFERKIT_PERPLEXITY_API_KEY"];
	if (key.length == 0) {
		XCTSkip("set INFERKIT_PERPLEXITY_API_KEY to exercise the live path");
	}
	NFKRemoteWebSearch *search = [NFKRemoteWebSearch webSearchForProvider:NFKRemoteProvider.perplexity apiKey:key];
	NSError *error = nil;
	NSArray<NFKWebSearchResult *> *results = [search resultsForQuery:@"Objective-C" parameters:@{ @"max_results": @3 } error:&error];
	XCTAssertGreaterThan(results.count, 0u, @"%@", error);
	XCTAssertNotNil(results.firstObject.URL);
}

@end
