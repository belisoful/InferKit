//
//  NFKRemoteRerankerTests.m
//  InferKitTests
//
//  The rerank request and reply through a stub transport: the scores put back in the documents'
//  order, the ranking, and the provider factory.
//

#import <XCTest/XCTest.h>
#import <InferKit/NFKRemoteReranker.h>
#import <InferKit/NFKRemoteProvider.h>
#import <InferKit/NFKErrors.h>

@interface NFKStubReranker : NFKRemoteReranker
@property (nonatomic, strong) NSURLRequest *lastRequest;
@property (nonatomic, copy) NSString *stagedBody;
@property (nonatomic, assign) NSInteger stagedStatusCode;
@end

@implementation NFKStubReranker
- (NSData *)sendRequest:(NSURLRequest *)request response:(NSHTTPURLResponse **)outResponse error:(NSError **)outError
{
	self.lastRequest = request;
	if (outResponse != NULL) {
		*outResponse = [[NSHTTPURLResponse alloc] initWithURL:request.URL statusCode:self.stagedStatusCode ?: 200
												  HTTPVersion:@"HTTP/1.1" headerFields:nil];
	}
	return [self.stagedBody dataUsingEncoding:NSUTF8StringEncoding];
}
@end

@interface NFKRemoteRerankerTests : XCTestCase
@property (nonatomic, strong) NFKStubReranker *reranker;
@property (nonatomic, copy) NSArray<NSString *> *documents;
@end

@implementation NFKRemoteRerankerTests

- (void)setUp
{
	[super setUp];
	self.reranker = [[NFKStubReranker alloc] init];
	self.reranker.endpointURL = [NSURL URLWithString:@"https://api.together.xyz/v1/rerank"];
	self.reranker.modelName = @"Salesforce/Llama-Rank-V1";
	self.reranker.apiKey = @"k";
	self.documents = @[ @"quarterly revenue", @"a red barn", @"barn owls at dusk" ];
	// The service answers in relevance order; the scores must go back to the documents' order.
	self.reranker.stagedBody = @"{\"results\":[{\"index\":1,\"relevance_score\":0.9},{\"index\":2,\"relevance_score\":0.4},{\"index\":0,\"relevance_score\":0.05}]}";
}

- (NSDictionary *)decodedRequestBody
{
	return [NSJSONSerialization JSONObjectWithData:self.reranker.lastRequest.HTTPBody options:0 error:NULL];
}

- (void)testScoresComeBackInTheDocumentsOrderAndTheRequestAsksForEveryOne
{
	NSError *error = nil;
	NSArray<NSNumber *> *scores = [self.reranker scoresForQuery:@"a barn" documents:self.documents error:&error];
	XCTAssertEqualObjects(scores, (@[ @0.05, @0.9, @0.4 ]), @"%@", error);

	NSDictionary *body = [self decodedRequestBody];
	XCTAssertEqualObjects(body[@"query"], @"a barn");
	XCTAssertEqualObjects(body[@"documents"], self.documents);
	XCTAssertEqualObjects(body[@"model"], @"Salesforce/Llama-Rank-V1");
	XCTAssertEqualObjects(body[@"top_n"], @3, @"every document is scored, not a shortlist");
	XCTAssertEqualObjects(body[@"return_documents"], @NO);
	XCTAssertEqualObjects(self.reranker.lastRequest.allHTTPHeaderFields[@"Authorization"], @"Bearer k");
}

- (void)testTheRankingIsByDescendingScore
{
	NSArray<NSNumber *> *ranked = [self.reranker rankedIndicesForQuery:@"a barn" documents:self.documents error:NULL];
	XCTAssertEqualObjects(ranked, (@[ @1, @2, @0 ]));
	XCTAssertEqualObjects([self.reranker scoreForQuery:@"a barn" document:@"a red barn" error:NULL], @0.05,
						  @"a single document takes the single score back, whatever the stub's index");
}

- (void)testAnEmptyListAFailingStatusAndAnEmptyReplyAreErrors
{
	NSError *error = nil;
	XCTAssertNil([self.reranker scoresForQuery:@"q" documents:@[] error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceMissingInput);

	self.reranker.stagedStatusCode = 404;
	self.reranker.stagedBody = @"{\"error\":\"no such model\"}";
	XCTAssertNil([self.reranker scoresForQuery:@"q" documents:self.documents error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceBackendFailure);
	XCTAssertTrue([error.localizedDescription containsString:@"no such model"]);

	self.reranker.stagedStatusCode = 200;
	self.reranker.stagedBody = @"{\"results\":[]}";
	XCTAssertNil([self.reranker scoresForQuery:@"q" documents:self.documents error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceBackendFailure);
}

- (void)testTheFactoryDerivesTheRerankURLAndDeclinesAnthropic
{
	NFKRemoteReranker *together = [NFKRemoteReranker rerankerForProvider:NFKRemoteProvider.together apiKey:@"k" modelName:@"m"];
	XCTAssertEqualObjects(together.endpointURL.absoluteString, @"https://api.together.xyz/v1/rerank");
	NFKRemoteReranker *router = [NFKRemoteReranker rerankerForProvider:NFKRemoteProvider.openRouter apiKey:@"k" modelName:@"m"];
	XCTAssertEqualObjects(router.endpointURL.absoluteString, @"https://openrouter.ai/api/v1/rerank");
	XCTAssertNil([NFKRemoteReranker rerankerForProvider:NFKRemoteProvider.anthropic apiKey:@"k" modelName:@"m"]);
}

@end
