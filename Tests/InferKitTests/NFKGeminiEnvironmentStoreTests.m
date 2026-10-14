//
//  NFKGeminiEnvironmentStoreTests.m
//  InferKitTests
//
//  The Gemini API's agent environments through a stubbed transport: listing environments and files,
//  reading a file's bytes, writing through the upload prefix, and deleting.
//

#import <XCTest/XCTest.h>
#import <InferKit/NFKGeminiEnvironmentStore.h>
#import <InferKit/NFKRemoteProvider.h>
#import <InferKit/NFKErrors.h>

@interface NFKStubEnvironmentStore : NFKGeminiEnvironmentStore
@property (nonatomic, strong) NSMutableArray<NSURLRequest *> *requests;
@property (nonatomic, strong) NSMutableArray<NSData *> *stagedReplies;
@property (nonatomic, assign) NSInteger status;
@end

@implementation NFKStubEnvironmentStore
- (NSData *)sendRequest:(NSURLRequest *)request response:(NSHTTPURLResponse **)outResponse error:(NSError **)outError
{
	if (self.requests == nil) {
		self.requests = [NSMutableArray array];
	}
	[self.requests addObject:request];
	if (outResponse != NULL) {
		*outResponse = [[NSHTTPURLResponse alloc] initWithURL:request.URL statusCode:self.status ?: 200 HTTPVersion:@"HTTP/1.1" headerFields:nil];
	}
	NSData *reply = self.stagedReplies.firstObject ?: [NSData data];
	if (self.stagedReplies.count > 0) {
		[self.stagedReplies removeObjectAtIndex:0];
	}
	return reply;
}
- (void)stage:(NSString *)reply
{
	if (self.stagedReplies == nil) {
		self.stagedReplies = [NSMutableArray array];
	}
	[self.stagedReplies addObject:[reply dataUsingEncoding:NSUTF8StringEncoding]];
}
@end

@interface NFKGeminiEnvironmentStoreTests : XCTestCase
@property (nonatomic, strong) NFKStubEnvironmentStore *store;
@end

@implementation NFKGeminiEnvironmentStoreTests

- (void)setUp
{
	[super setUp];
	self.store = [[NFKStubEnvironmentStore alloc] init];
	self.store.apiKey = @"AIza";
}

- (void)testTheFactoryReachesOnlyGemini
{
	NFKGeminiEnvironmentStore *store = [NFKGeminiEnvironmentStore environmentStoreForProvider:NFKRemoteProvider.googleGemini apiKey:@"k"];
	XCTAssertEqualObjects(store.endpointURL.absoluteString, @"https://generativelanguage.googleapis.com/v1beta/environments");
	XCTAssertEqualObjects(store.apiKey, @"k");
	XCTAssertNil([NFKGeminiEnvironmentStore environmentStoreForProvider:NFKRemoteProvider.openAI apiKey:@"k"]);
}

- (void)testEnvironmentsAreListedAcrossPages
{
	[self.store stage:@"{\"environments\":[{\"id\":\"e1\",\"status\":\"active\"}],\"next_page_token\":\"p2\"}"];
	[self.store stage:@"{\"environments\":[{\"id\":\"e2\",\"status\":\"idle\"}]}"];
	NSError *error = nil;
	NSArray *environments = [self.store environmentsWithError:&error];
	XCTAssertEqualObjects([environments valueForKey:@"id"], (@[ @"e1", @"e2" ]), @"%@", error);
	XCTAssertEqualObjects(self.store.requests[1].URL.absoluteString, @"https://generativelanguage.googleapis.com/v1beta/environments?page_token=p2");
	XCTAssertEqualObjects([self.store.requests[0] valueForHTTPHeaderField:@"x-goog-api-key"], @"AIza");
}

- (void)testFilesAreListedRecursivelyWithTheirKindsAndSizes
{
	[self.store stage:@"{\"files\":[{\"name\":\"out\",\"path\":\"out\",\"type\":\"DIRECTORY\"},"
	 "{\"name\":\"stand.stl\",\"path\":\"out/stand.stl\",\"type\":\"FILE\",\"size_bytes\":\"684\",\"mime_type\":\"model/stl\"}]}"];
	NSError *error = nil;
	NSArray<NFKGeminiEnvironmentFile *> *files = [self.store filesInEnvironment:@"env1" atPath:nil recursive:YES error:&error];
	XCTAssertEqual(files.count, 2u, @"%@", error);
	XCTAssertEqualObjects(self.store.requests[0].URL.absoluteString,
						  @"https://generativelanguage.googleapis.com/v1beta/environments/env1/files?recursive=true");
	XCTAssertTrue(files[0].isDirectory);
	XCTAssertEqualObjects(files[1].path, @"out/stand.stl");
	XCTAssertEqual(files[1].size, 684ull);
	XCTAssertEqualObjects(files[1].mimeType, @"model/stl");
}

- (void)testAFilesBytesAreReadAsMedia
{
	[self.store stage:@"solid stand\nendsolid stand\n"];
	NSError *error = nil;
	NSData *data = [self.store contentsOfFileAtPath:@"out/stand.stl" inEnvironment:@"env1" recursive:NO error:&error];
	XCTAssertEqualObjects(data, [@"solid stand\nendsolid stand\n" dataUsingEncoding:NSUTF8StringEncoding], @"%@", error);
	XCTAssertEqualObjects(self.store.requests[0].URL.absoluteString,
						  @"https://generativelanguage.googleapis.com/v1beta/environments/env1/files/out/stand.stl?alt=media");
}

- (void)testAWriteGoesThroughTheUploadPrefix
{
	[self.store stage:@"{\"files\":[{\"name\":\"spec.txt\",\"path\":\"workspace/spec.txt\",\"type\":\"FILE\",\"size_bytes\":\"4\"}]}"];
	NSError *error = nil;
	NSArray *written = [self.store writeData:[@"spec" dataUsingEncoding:NSUTF8StringEncoding] toPath:@"workspace/spec.txt"
							   inEnvironment:@"env1" overwrite:YES extract:NO error:&error];
	XCTAssertEqual(written.count, 1u, @"%@", error);
	NSURLRequest *sent = self.store.requests[0];
	XCTAssertEqualObjects(sent.HTTPMethod, @"PUT");
	XCTAssertEqualObjects(sent.URL.absoluteString,
						  @"https://generativelanguage.googleapis.com/upload/v1beta/environments/env1/files/workspace/spec.txt?overwrite=true");
	XCTAssertEqualObjects([sent valueForHTTPHeaderField:@"Content-Type"], @"application/octet-stream");
	XCTAssertEqualObjects(sent.HTTPBody, [@"spec" dataUsingEncoding:NSUTF8StringEncoding]);
}

- (void)testAnArchiveIsExtractedAndADirectoryDownloadsRecursively
{
	[self.store stage:@"{\"files\":[{\"name\":\"a.txt\",\"path\":\"in/a.txt\",\"type\":\"FILE\",\"size_bytes\":\"1\"}]}"];
	[self.store stage:@"tar"];
	NSError *error = nil;
	NSArray *written = [self.store writeData:[@"tar" dataUsingEncoding:NSUTF8StringEncoding] toPath:@"in"
							   inEnvironment:@"env1" overwrite:NO extract:YES error:&error];
	XCTAssertEqual(written.count, 1u, @"%@", error);
	XCTAssertEqualObjects(self.store.requests[0].URL.absoluteString,
						  @"https://generativelanguage.googleapis.com/upload/v1beta/environments/env1/files/in?extract=true");
	XCTAssertEqualObjects([self.store.requests[0] valueForHTTPHeaderField:@"Content-Type"], @"application/x-tar");
	XCTAssertNotNil([self.store contentsOfFileAtPath:@"out" inEnvironment:@"env1" recursive:YES error:&error], @"%@", error);
	XCTAssertEqualObjects(self.store.requests[1].URL.absoluteString,
						  @"https://generativelanguage.googleapis.com/v1beta/environments/env1/files/out?alt=media&recursive=true");
}

- (void)testAWriteToAProxyPutsTheUploadSegmentBeforeTheVersion
{
	self.store.endpointURL = [NSURL URLWithString:@"https://proxy.example/g/v1beta/environments"];
	[self.store stage:@"{\"files\":[]}"];
	NSError *error = nil;
	XCTAssertNotNil([self.store writeData:[NSData data] toPath:@"a.txt" inEnvironment:@"env1" overwrite:NO extract:NO error:&error], @"%@", error);
	XCTAssertEqualObjects(self.store.requests[0].URL.absoluteString, @"https://proxy.example/g/upload/v1beta/environments/env1/files/a.txt");
}

- (void)testAnIdOrPathThatLeavesTheEnvironmentIsRefusedUnsent
{
	NSError *error = nil;
	XCTAssertFalse([self.store deleteEnvironment:@"x/../../files" error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceMissingInput);
	XCTAssertFalse([self.store deleteEnvironment:@".." error:NULL]);
	XCTAssertNil([self.store filesInEnvironment:@"." atPath:nil recursive:NO error:NULL]);
	XCTAssertNil([self.store contentsOfFileAtPath:@"out/../../env2/files/key" inEnvironment:@"env1" recursive:NO error:NULL]);
	XCTAssertNil([self.store writeData:[NSData data] toPath:@"./a" inEnvironment:@"env1" overwrite:YES extract:NO error:NULL]);
	XCTAssertNil([self.store contentsOfFileAtPath:@"out//a.stl" inEnvironment:@"env1" recursive:NO error:NULL]);
	XCTAssertNil([self.store writeData:[NSData data] toPath:@"/" inEnvironment:@"env1" overwrite:YES extract:NO error:NULL]);
	NSString *missing = nil;
	XCTAssertNil([self.store filesInEnvironment:(NSString * _Nonnull)missing atPath:nil recursive:NO error:NULL]);
	XCTAssertEqual(self.store.requests.count, 0u);
}

- (void)testAWriteToAnEndpointWithoutAVersionIsRefusedUnsent
{
	self.store.endpointURL = [NSURL URLWithString:@"https://proxy.example/environments"];
	NSError *error = nil;
	XCTAssertNil([self.store writeData:[NSData data] toPath:@"a.txt" inEnvironment:@"env1" overwrite:NO extract:NO error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceMissingInput);
	XCTAssertEqual(self.store.requests.count, 0u);
	[self.store stage:@"{\"files\":[]}"];
	XCTAssertNotNil([self.store filesInEnvironment:@"env1" atPath:@"" recursive:NO error:&error], @"%@", error);
	XCTAssertEqualObjects(self.store.requests[0].URL.absoluteString, @"https://proxy.example/environments/env1/files");
}

- (void)testAPathComponentIsPercentEncoded
{
	[self.store stage:@"x"];
	NSError *error = nil;
	XCTAssertNotNil([self.store contentsOfFileAtPath:@"out/a b?.stl" inEnvironment:@"env1" recursive:NO error:&error], @"%@", error);
	XCTAssertEqualObjects(self.store.requests[0].URL.absoluteString,
						  @"https://generativelanguage.googleapis.com/v1beta/environments/env1/files/out/a%20b%3F.stl?alt=media");
}

- (void)testAConflictFailsAndADeleteSucceedsOnAnEmptyReply
{
	self.store.status = 409;
	[self.store stage:@"{\"error\":{\"code\":409,\"message\":\"exists\"}}"];
	NSError *error = nil;
	XCTAssertNil([self.store writeData:[NSData data] toPath:@"a.txt" inEnvironment:@"env1" overwrite:NO extract:NO error:&error]);
	XCTAssertNotNil(error);
	self.store.status = 200;
	XCTAssertTrue([self.store deleteEnvironment:@"env1" error:&error], @"%@", error);
	XCTAssertEqualObjects(self.store.requests[1].HTTPMethod, @"DELETE");
	XCTAssertEqualObjects(self.store.requests[1].URL.absoluteString, @"https://generativelanguage.googleapis.com/v1beta/environments/env1");
}

@end
