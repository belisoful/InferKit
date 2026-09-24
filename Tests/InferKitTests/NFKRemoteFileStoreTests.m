//
//  NFKRemoteFileStoreTests.m
//  InferKitTests
//
//  The Files APIs through a scripted transport: each style's upload shape, paging, download, delete,
//  and an uploaded file riding in a request as the service's reference.
//

#import <XCTest/XCTest.h>
#import <InferKit/NFKRemoteFileStore.h>
#import <InferKit/NFKRemoteProvider.h>
#import <InferKit/NFKRemoteBackend.h>
#import <InferKit/NFKAnthropicBackend.h>
#import <InferKit/NFKRemoteResponsesBackend.h>
#import <InferKit/NFKGeminiInteractionsBackend.h>
#import <InferKit/NFKInferenceRequest.h>
#import <InferKit/NFKInferenceResult.h>
#import <InferKit/NFKInferenceKeys.h>
#import <InferKit/NFKErrors.h>

@interface NFKScriptedFileStore : NFKRemoteFileStore
@property (nonatomic, strong) NSMutableArray<NSURLRequest *> *requests;
@property (nonatomic, strong) NSMutableArray *replies;
@end

@implementation NFKScriptedFileStore
- (NSData *)sendRequest:(NSURLRequest *)request response:(NSHTTPURLResponse **)outResponse error:(NSError **)outError
{
	if (self.requests == nil) {
		self.requests = [NSMutableArray array];
	}
	[self.requests addObject:request];
	id reply = self.replies.firstObject ?: @"{}";
	if (self.replies.count > 0) {
		[self.replies removeObjectAtIndex:0];
	}
	NSDictionary *headers = nil;
	if ([reply isKindOfClass:NSArray.class]) {
		headers = reply[1];
		reply = reply[0];
	}
	if (outResponse != NULL) {
		*outResponse = [[NSHTTPURLResponse alloc] initWithURL:request.URL statusCode:200 HTTPVersion:@"HTTP/1.1" headerFields:headers];
	}
	return [reply isKindOfClass:NSData.class] ? reply : [reply dataUsingEncoding:NSUTF8StringEncoding];
}
- (NSString *)formOf:(NSUInteger)index
{
	return [[NSString alloc] initWithData:self.requests[index].HTTPBody encoding:NSISOLatin1StringEncoding];
}
@end

@interface NFKFileStubChat : NFKRemoteBackend
@property (nonatomic, strong) NSURLRequest *lastRequest;
@end
@implementation NFKFileStubChat
- (NSData *)sendRequest:(NSURLRequest *)request response:(NSHTTPURLResponse **)outResponse error:(NSError **)outError
{
	self.lastRequest = request;
	if (outResponse != NULL) {
		*outResponse = [[NSHTTPURLResponse alloc] initWithURL:request.URL statusCode:200 HTTPVersion:@"HTTP/1.1" headerFields:nil];
	}
	if ([request.URL.path hasSuffix:@"/url"]) {
		return [@"{\"url\":\"https://signed.example/doc.pdf\"}" dataUsingEncoding:NSUTF8StringEncoding];
	}
	return [@"{\"choices\":[{\"message\":{\"content\":\"ok\"}}]}" dataUsingEncoding:NSUTF8StringEncoding];
}
@end

@interface NFKFileStubAnthropic : NFKAnthropicBackend
@property (nonatomic, strong) NSURLRequest *lastRequest;
@end
@implementation NFKFileStubAnthropic
- (NSData *)sendRequest:(NSURLRequest *)request response:(NSHTTPURLResponse **)outResponse error:(NSError **)outError
{
	self.lastRequest = request;
	if (outResponse != NULL) {
		*outResponse = [[NSHTTPURLResponse alloc] initWithURL:request.URL statusCode:200 HTTPVersion:@"HTTP/1.1" headerFields:nil];
	}
	return [@"{\"content\":[{\"type\":\"text\",\"text\":\"ok\"}]}" dataUsingEncoding:NSUTF8StringEncoding];
}
@end

@interface NFKRemoteFileStoreTests : XCTestCase
@end

@implementation NFKRemoteFileStoreTests

- (NFKScriptedFileStore *)storeFor:(NFKRemoteProvider *)provider
{
	NFKRemoteFileStore *made = [NFKRemoteFileStore fileStoreForProvider:provider apiKey:@"k"];
	NFKScriptedFileStore *store = [[NFKScriptedFileStore alloc] init];
	store.endpointURL = made.endpointURL;
	store.apiStyle = made.apiStyle;
	store.defaultPurpose = made.defaultPurpose;
	store.uploadPathComponent = made.uploadPathComponent;
	store.apiKey = @"k";
	store.replies = [NSMutableArray array];
	return store;
}

- (NSDictionary *)bodyOf:(NSURLRequest *)request
{
	return [NSJSONSerialization JSONObjectWithData:request.HTTPBody options:0 error:NULL];
}

- (void)testEachProviderGetsItsStoreAndTheFilelessOnesNone
{
	XCTAssertEqualObjects([NFKRemoteFileStore fileStoreForProvider:NFKRemoteProvider.deepSeek apiKey:@"k"].endpointURL.absoluteString,
						  @"https://api.deepseek.com/files");
	XCTAssertEqualObjects([NFKRemoteFileStore fileStoreForProvider:NFKRemoteProvider.mistral apiKey:@"k"].defaultPurpose, @"ocr");
	XCTAssertEqual([NFKRemoteFileStore fileStoreForProvider:NFKRemoteProvider.anthropic apiKey:@"k"].apiStyle, NFKRemoteFileStoreAPIStyleAnthropic);
	XCTAssertNil([NFKRemoteFileStore fileStoreForProvider:NFKRemoteProvider.ollama apiKey:nil]);
}

- (void)testAnOpenAIUploadSendsItsPurposeAndExpiryBeforeTheFileAndReadsTheRecord
{
	NFKScriptedFileStore *store = [self storeFor:NFKRemoteProvider.openAI];
	[store.replies addObject:@"{\"id\":\"file-abc\",\"object\":\"file\",\"bytes\":8,\"created_at\":1700000000,\"filename\":\"brief.pdf\",\"purpose\":\"user_data\",\"expires_at\":1700003600}"];
	NSError *error = nil;
	NFKRemoteFile *file = [store uploadData:[@"%PDF-1.4" dataUsingEncoding:NSUTF8StringEncoding] filename:@"brief.pdf" mimeType:nil
									purpose:nil expiresAfter:3600 error:&error];
	XCTAssertNotNil(file, @"%@", error);
	XCTAssertEqualObjects(store.requests[0].URL.absoluteString, @"https://api.openai.com/v1/files");
	NSString *form = [store formOf:0];
	XCTAssertTrue([form containsString:@"name=\"purpose\"\r\n\r\nuser_data"]);
	XCTAssertTrue([form containsString:@"name=\"expires_after[seconds]\"\r\n\r\n3600"]);
	XCTAssertLessThan([form rangeOfString:@"expires_after"].location, [form rangeOfString:@"name=\"file\""].location);
	XCTAssertTrue([form containsString:@"Content-Type: application/pdf"]);
	XCTAssertEqualObjects(file.identifier, @"file-abc");
	XCTAssertEqual(file.byteCount, 8);
	XCTAssertEqualObjects(file.createdAt, [NSDate dateWithTimeIntervalSince1970:1700000000]);
	XCTAssertTrue(file.isDownloadable);
}

- (void)testAnthropicUploadsWithoutAPurposeAndKeepsItsUploadsFromDownload
{
	NFKScriptedFileStore *store = [self storeFor:NFKRemoteProvider.anthropic];
	[store.replies addObject:@"{\"id\":\"file_011\",\"type\":\"file\",\"filename\":\"a.pdf\",\"mime_type\":\"application/pdf\","
	 "\"size_bytes\":1024,\"created_at\":\"2026-09-22T10:00:00Z\",\"downloadable\":false,\"expires_at\":null}"];
	NFKRemoteFile *file = [store uploadData:[@"%PDF" dataUsingEncoding:NSUTF8StringEncoding] filename:@"a.pdf" mimeType:nil purpose:nil expiresAfter:0 error:NULL];
	XCTAssertFalse([[store formOf:0] containsString:@"purpose"]);
	XCTAssertEqualObjects([store.requests[0] valueForHTTPHeaderField:@"x-api-key"], @"k");
	XCTAssertEqual(file.byteCount, 1024);
	XCTAssertFalse(file.isDownloadable);
	XCTAssertNotNil(file.createdAt);
}

- (void)testGeminiUploadsInTwoStepsAndReportsTheURIAndState
{
	NFKScriptedFileStore *store = [self storeFor:NFKRemoteProvider.googleGemini];
	[store.replies addObject:@[ @"{}", @{ @"X-Goog-Upload-URL": @"https://upload.example/session/1" } ]];
	[store.replies addObject:@"{\"file\":{\"name\":\"files/abc\",\"displayName\":\"clip.mp4\",\"mimeType\":\"video/mp4\",\"sizeBytes\":\"9\","
	 "\"uri\":\"https://generativelanguage.googleapis.com/v1beta/files/abc\",\"state\":\"PROCESSING\"}}"];
	[store.replies addObject:@"{\"name\":\"files/abc\",\"mimeType\":\"video/mp4\",\"state\":\"ACTIVE\",\"uri\":\"https://generativelanguage.googleapis.com/v1beta/files/abc\"}"];
	NSError *error = nil;
	NFKRemoteFile *file = [store uploadData:[@"ftypisom!" dataUsingEncoding:NSUTF8StringEncoding] filename:@"clip.mp4" mimeType:nil purpose:nil expiresAfter:0 error:&error];
	XCTAssertNotNil(file, @"%@", error);
	NSURLRequest *start = store.requests[0];
	XCTAssertEqualObjects(start.URL.absoluteString, @"https://generativelanguage.googleapis.com/upload/v1beta/files");
	XCTAssertEqualObjects([start valueForHTTPHeaderField:@"X-Goog-Upload-Command"], @"start");
	XCTAssertEqualObjects([start valueForHTTPHeaderField:@"X-Goog-Upload-Header-Content-Type"], @"video/mp4");
	XCTAssertEqualObjects([store.requests[1] valueForHTTPHeaderField:@"X-Goog-Upload-Command"], @"upload, finalize");
	XCTAssertEqualObjects(store.requests[1].URL.absoluteString, @"https://upload.example/session/1");
	XCTAssertEqualObjects(file.identifier, @"files/abc");
	XCTAssertEqualObjects(file.state, @"PROCESSING");
	XCTAssertEqual(file.byteCount, 9);

	NFKRemoteFile *ready = [store fileWhenReadyWithIdentifier:file.identifier timeout:5 error:&error];
	XCTAssertEqualObjects(ready.state, @"ACTIVE");
	XCTAssertEqualObjects(store.requests[2].URL.absoluteString, @"https://generativelanguage.googleapis.com/v1beta/files/abc");
	XCTAssertNil([store contentsOfFileWithIdentifier:@"files/abc" error:&error]);
	XCTAssertEqual(error.code, (NSInteger)kNFKError_InferenceUnsupported);
}

- (void)testListingReadsEveryPageAndDownloadAndDeleteHitTheirPaths
{
	NFKScriptedFileStore *store = [self storeFor:NFKRemoteProvider.openAI];
	[store.replies addObject:@"{\"data\":[{\"id\":\"file-1\"},{\"id\":\"file-2\"}],\"has_more\":true,\"last_id\":\"file-2\"}"];
	[store.replies addObject:@"{\"data\":[{\"id\":\"file-3\"}],\"has_more\":false}"];
	[store.replies addObject:[@"bytes" dataUsingEncoding:NSUTF8StringEncoding]];
	[store.replies addObject:@"{\"id\":\"file-1\",\"deleted\":true}"];
	NSArray<NFKRemoteFile *> *files = [store filesWithError:NULL];
	XCTAssertEqual(files.count, 3);
	XCTAssertEqualObjects(store.requests[1].URL.query, @"after=file-2");
	XCTAssertEqualObjects([store contentsOfFileWithIdentifier:@"file-1" error:NULL], [@"bytes" dataUsingEncoding:NSUTF8StringEncoding]);
	XCTAssertEqualObjects(store.requests[2].URL.absoluteString, @"https://api.openai.com/v1/files/file-1/content");
	XCTAssertTrue([store deleteFileWithIdentifier:@"file-1" error:NULL]);
	XCTAssertEqualObjects(store.requests[3].HTTPMethod, @"DELETE");
}

- (void)testTogetherUploadsToItsUploadPathWithAFileName
{
	NFKScriptedFileStore *store = [self storeFor:NFKRemoteProvider.together];
	[store.replies addObject:@"{\"id\":\"file-t\",\"filename\":\"train.jsonl\",\"purpose\":\"fine-tune\"}"];
	[store uploadData:[@"{}" dataUsingEncoding:NSUTF8StringEncoding] filename:@"train.jsonl" mimeType:nil purpose:nil expiresAfter:0 error:NULL];
	XCTAssertEqualObjects(store.requests[0].URL.absoluteString, @"https://api.together.xyz/v1/files/upload");
	XCTAssertTrue([[store formOf:0] containsString:@"name=\"file_name\"\r\n\r\ntrain.jsonl"]);
}

#pragma mark A file in a request

- (void)testAnUploadedFileRidesAsEachServicesReference
{
	NFKRemoteFile *pdf = [NFKRemoteFile fileWithIdentifier:@"file-abc" mimeType:@"application/pdf"];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"summarize", NFKInputDocument: pdf }];

	NFKFileStubChat *openAI = [NFKFileStubChat backendWithEndpointURL:[NSURL URLWithString:@"https://api.openai.com/v1/chat/completions"]];
	[openAI runInferenceForRequest:request error:NULL];
	NSArray *parts = [self bodyOf:openAI.lastRequest][@"messages"][0][@"content"];
	XCTAssertEqualObjects(parts[1], (@{ @"type": @"file", @"file": @{ @"file_id": @"file-abc" } }));

	openAI.chatDialect = NFKRemoteChatDialectDeepSeek;
	[openAI runInferenceForRequest:request error:NULL];
	XCTAssertEqualObjects([self bodyOf:openAI.lastRequest][@"messages"][0][@"content"][1], (@{ @"type": @"file", @"file_id": @"file-abc" }));

	NFKFileStubChat *mistral = [NFKFileStubChat backendWithEndpointURL:[NSURL URLWithString:@"https://api.mistral.ai/v1/chat/completions"]];
	mistral.chatDialect = NFKRemoteChatDialectMistral;
	[mistral runInferenceForRequest:request error:NULL];
	XCTAssertEqualObjects([self bodyOf:mistral.lastRequest][@"messages"][0][@"content"][1],
						  (@{ @"type": @"document_url", @"document_url": @"https://signed.example/doc.pdf" }));

	NFKFileStubAnthropic *claude = [[NFKFileStubAnthropic alloc] init];
	claude.modelName = @"claude-opus-5-5";
	[claude runInferenceForRequest:request error:NULL];
	NSDictionary *block = [self bodyOf:claude.lastRequest][@"messages"][0][@"content"][0];
	XCTAssertEqualObjects(block[@"source"], (@{ @"type": @"file", @"file_id": @"file-abc" }));

	NFKRemoteFile *clip = [NFKRemoteFile fileWithGeminiURI:[NSURL URLWithString:@"https://generativelanguage.googleapis.com/v1beta/files/v1"] mimeType:@"video/mp4"];
	XCTAssertEqualObjects(clip.identifier, @"files/v1");
}

@end
