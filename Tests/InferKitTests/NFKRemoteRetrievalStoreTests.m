//
//  NFKRemoteRetrievalStoreTests.m
//  InferKitTests
//
//  Hosted retrieval stores through a scripted transport: each style's store, document, and search
//  calls, xAI's two hosts and keys, and the refusal where a service has no direct search.
//

#import <XCTest/XCTest.h>
#import <InferKit/NFKRemoteRetrievalStore.h>
#import <InferKit/NFKRemoteProvider.h>
#import <InferKit/NFKErrors.h>

@interface NFKScriptedRetrievalStore : NFKRemoteRetrievalStore
@property (nonatomic, strong) NSMutableArray<NSURLRequest *> *requests;
@property (nonatomic, strong) NSMutableArray<NSString *> *replies;
@end

@implementation NFKScriptedRetrievalStore
- (NSData *)sendRequest:(NSURLRequest *)request response:(NSHTTPURLResponse **)outResponse error:(NSError **)outError
{
	if (self.requests == nil) {
		self.requests = [NSMutableArray array];
	}
	[self.requests addObject:request];
	NSString *reply = self.replies.firstObject ?: @"{}";
	if (self.replies.count > 0) {
		[self.replies removeObjectAtIndex:0];
	}
	if (outResponse != NULL) {
		*outResponse = [[NSHTTPURLResponse alloc] initWithURL:request.URL statusCode:200 HTTPVersion:@"HTTP/1.1" headerFields:nil];
	}
	return [reply dataUsingEncoding:NSUTF8StringEncoding];
}
- (NSDictionary *)bodyAt:(NSUInteger)index
{
	return [NSJSONSerialization JSONObjectWithData:self.requests[index].HTTPBody options:0 error:NULL];
}
@end

@interface NFKRemoteRetrievalStoreTests : XCTestCase
@end

@implementation NFKRemoteRetrievalStoreTests

- (NFKScriptedRetrievalStore *)storeFor:(NFKRemoteProvider *)provider
{
	NFKRemoteRetrievalStore *made = [NFKRemoteRetrievalStore retrievalStoreForProvider:provider apiKey:@"k"];
	NFKScriptedRetrievalStore *store = [[NFKScriptedRetrievalStore alloc] init];
	store.endpointURL = made.endpointURL;
	store.managementURL = made.managementURL;
	store.apiStyle = made.apiStyle;
	store.apiKey = @"k";
	store.replies = [NSMutableArray array];
	return store;
}

- (void)testOnlyTheServicesWithStoresGetOne
{
	XCTAssertEqual([NFKRemoteRetrievalStore retrievalStoreForProvider:NFKRemoteProvider.xAI apiKey:@"k"].apiStyle, NFKRemoteRetrievalAPIStyleXAI);
	XCTAssertNil([NFKRemoteRetrievalStore retrievalStoreForProvider:NFKRemoteProvider.anthropic apiKey:@"k"]);
	XCTAssertNil([NFKRemoteRetrievalStore retrievalStoreForProvider:NFKRemoteProvider.together apiKey:@"k"]);
}

- (void)testOpenAICreatesAddsAndSearchesAVectorStore
{
	NFKScriptedRetrievalStore *store = [self storeFor:NFKRemoteProvider.openAI];
	[store.replies addObjectsFromArray:@[
		@"{\"id\":\"vs_1\",\"object\":\"vector_store\",\"name\":\"FAQ\",\"status\":\"completed\",\"file_counts\":{\"total\":0}}",
		@"{\"id\":\"file-1\",\"object\":\"vector_store.file\",\"status\":\"in_progress\"}",
		(@"{\"object\":\"vector_store.search_results.page\",\"data\":[{\"file_id\":\"file-1\",\"filename\":\"faq.pdf\",\"score\":0.9,"
		 "\"attributes\":{\"lang\":\"en\"},\"content\":[{\"type\":\"text\",\"text\":\"Returns within 30 days.\"}]}]}") ]];
	NSError *error = nil;
	NFKRetrievalStoreRecord *created = [store createStoreNamed:@"FAQ" options:@{ @"expires_after": @{ @"anchor": @"last_active_at", @"days": @7 } } error:&error];
	XCTAssertEqualObjects(created.identifier, @"vs_1", @"%@", error);
	XCTAssertEqualObjects(store.requests[0].URL.absoluteString, @"https://api.openai.com/v1/vector_stores");
	XCTAssertEqualObjects([store bodyAt:0][@"name"], @"FAQ");
	XCTAssertEqualObjects([store.requests[0] valueForHTTPHeaderField:@"OpenAI-Beta"], @"assistants=v2");

	NFKRetrievalDocument *added = [store addFileWithIdentifier:@"file-1" toStore:@"vs_1" attributes:@{ @"lang": @"en" } error:&error];
	XCTAssertEqualObjects(added.status, @"in_progress");
	XCTAssertEqualObjects(store.requests[1].URL.absoluteString, @"https://api.openai.com/v1/vector_stores/vs_1/files");
	XCTAssertEqualObjects([store bodyAt:1], (@{ @"file_id": @"file-1", @"attributes": @{ @"lang": @"en" } }));

	NSArray<NFKRetrievalMatch *> *matches = [store searchStores:@[ @"vs_1" ] query:@"return policy" limit:5 filter:nil error:&error];
	XCTAssertEqualObjects(store.requests[2].URL.absoluteString, @"https://api.openai.com/v1/vector_stores/vs_1/search");
	XCTAssertEqualObjects([store bodyAt:2], (@{ @"query": @"return policy", @"max_num_results": @5 }));
	XCTAssertEqualObjects(matches.firstObject.text, @"Returns within 30 days.");
	XCTAssertEqualWithAccuracy(matches.firstObject.score, 0.9, 1e-9);
}

// xAI keeps collections on its management host under the management key, and searches on the API host.
- (void)testXAIManagesCollectionsWithTheManagementKeyAndSearchesWithTheAPIKey
{
	NFKScriptedRetrievalStore *store = [self storeFor:NFKRemoteProvider.xAI];
	store.managementAPIKey = @"mgmt";
	[store.replies addObjectsFromArray:@[
		@"{\"collection_id\":\"collection_9\",\"collection_name\":\"SEC\",\"documents_count\":0}",
		@"{}",
		@"{\"matches\":[{\"file_id\":\"file_1\",\"chunk_id\":\"c1\",\"chunk_content\":\"Revenue rose.\",\"score\":0.8,\"collection_ids\":[\"collection_9\"]}]}" ]];
	[store createStoreNamed:@"SEC" options:nil error:NULL];
	XCTAssertEqualObjects(store.requests[0].URL.absoluteString, @"https://management-api.x.ai/v1/collections");
	XCTAssertEqualObjects([store.requests[0] valueForHTTPHeaderField:@"Authorization"], @"Bearer mgmt");
	XCTAssertEqualObjects([store bodyAt:0][@"collection_name"], @"SEC");

	NFKRetrievalDocument *added = [store addFileWithIdentifier:@"file_1" toStore:@"collection_9" attributes:nil error:NULL];
	XCTAssertEqualObjects(added.identifier, @"file_1");
	XCTAssertEqualObjects(store.requests[1].URL.absoluteString, @"https://management-api.x.ai/v1/collections/collection_9/documents/file_1");

	NSArray<NFKRetrievalMatch *> *matches = [store searchStores:@[ @"collection_9" ] query:@"revenue" limit:3 filter:@"year > 2020" error:NULL];
	XCTAssertEqualObjects(store.requests[2].URL.absoluteString, @"https://api.x.ai/v1/documents/search");
	XCTAssertEqualObjects([store.requests[2] valueForHTTPHeaderField:@"Authorization"], @"Bearer k");
	XCTAssertEqualObjects([store bodyAt:2], (@{ @"query": @"revenue", @"source": @{ @"collection_ids": @[ @"collection_9" ] }, @"limit": @3, @"filter": @"year > 2020" }));
	XCTAssertEqualObjects(matches.firstObject.text, @"Revenue rose.");
}

- (void)testGeminiImportsAFileAndRefusesADirectSearch
{
	NFKScriptedRetrievalStore *store = [self storeFor:NFKRemoteProvider.googleGemini];
	[store.replies addObjectsFromArray:@[ @"{\"name\":\"fileSearchStores/abc\",\"displayName\":\"docs\"}",
										  @"{\"name\":\"fileSearchStores/abc/operations/op1\",\"done\":false}" ]];
	NFKRetrievalStoreRecord *created = [store createStoreNamed:@"docs" options:@{ @"embeddingModel": @"models/gemini-embedding-2" } error:NULL];
	XCTAssertEqualObjects(created.identifier, @"fileSearchStores/abc");
	XCTAssertEqualObjects(created.name, @"docs");
	XCTAssertEqualObjects([store.requests[0] valueForHTTPHeaderField:@"x-goog-api-key"], @"k");
	NFKRetrievalDocument *import = [store addFileWithIdentifier:@"files/f1" toStore:@"fileSearchStores/abc" attributes:nil error:NULL];
	XCTAssertEqualObjects(store.requests[1].URL.absoluteString, @"https://generativelanguage.googleapis.com/v1beta/fileSearchStores/abc:importFile");
	XCTAssertEqualObjects([store bodyAt:1], (@{ @"fileName": @"files/f1" }));
	XCTAssertEqualObjects(import.status, @"pending");
	NSError *error = nil;
	XCTAssertNil([store searchStores:@[ @"fileSearchStores/abc" ] query:@"x" limit:1 filter:nil error:&error]);
	XCTAssertEqual(error.code, (NSInteger)kNFKError_InferenceUnsupported);
}

- (void)testMistralUploadsIntoALibraryAndListsItsDocuments
{
	NFKScriptedRetrievalStore *store = [self storeFor:NFKRemoteProvider.mistral];
	[store.replies addObjectsFromArray:@[ @"{\"id\":\"doc-1\",\"name\":\"manual.pdf\",\"process_status\":\"todo\"}",
										  @"{\"data\":[{\"id\":\"doc-1\",\"name\":\"manual.pdf\",\"process_status\":\"done\"}]}" ]];
	NFKRetrievalDocument *uploaded = [store uploadData:[@"%PDF" dataUsingEncoding:NSUTF8StringEncoding] filename:@"manual.pdf" toStore:@"lib-1" error:NULL];
	XCTAssertEqualObjects(store.requests[0].URL.absoluteString, @"https://api.mistral.ai/v1/libraries/lib-1/documents");
	XCTAssertTrue([[store.requests[0] valueForHTTPHeaderField:@"Content-Type"] hasPrefix:@"multipart/form-data"]);
	XCTAssertEqualObjects(uploaded.filename, @"manual.pdf");
	XCTAssertEqualObjects(uploaded.status, @"todo");
	NSArray *documents = [store documentsInStore:@"lib-1" error:NULL];
	XCTAssertEqual(documents.count, 1);
	NSError *error = nil;
	XCTAssertNil([store addFileWithIdentifier:@"file-x" toStore:@"lib-1" attributes:nil error:&error]);
	XCTAssertEqual(error.code, (NSInteger)kNFKError_InferenceUnsupported);
}

@end
