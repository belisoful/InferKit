//
//  NFKRemoteAttachmentsTests.m
//  InferKitTests
//
//  The media a chat request carries beside its text — audio, documents, a clip's sampled frames —
//  and a spoken reply, through stubbed transports on both chat backends: the wire shape each takes
//  and what it refuses.
//

#import <XCTest/XCTest.h>
#import <InferKit/NFKRemoteBackend.h>
#import <InferKit/NFKAnthropicBackend.h>
#import <InferKit/NFKInferenceRequest.h>
#import <InferKit/NFKInferenceResult.h>
#import <InferKit/NFKInferenceJob.h>
#import <InferKit/NFKInferenceKeys.h>
#import <InferKit/NFKAudioAsset.h>
#import <InferKit/NFKVideoAsset.h>
#import <InferKit/NFKErrors.h>
#import <InferKit/NFKRemoteProvider.h>
#import <InferKit/NFKImageCoding.h>
#import "NFKTestClip.h"

@interface NFKAttachStubRemoteBackend : NFKRemoteBackend
@property (nonatomic, strong) NSURLRequest *lastRequest;
@property (nonatomic, copy) NSString *stagedBody;
@property (nonatomic, copy) NSArray<NSString *> *stagedLines;
@end

@implementation NFKAttachStubRemoteBackend
- (NSData *)sendRequest:(NSURLRequest *)request response:(NSHTTPURLResponse **)outResponse error:(NSError **)outError
{
	self.lastRequest = request;
	if (outResponse != NULL) {
		*outResponse = [[NSHTTPURLResponse alloc] initWithURL:request.URL statusCode:200 HTTPVersion:@"HTTP/1.1" headerFields:nil];
	}
	return [self.stagedBody dataUsingEncoding:NSUTF8StringEncoding];
}
- (void (^)(void))streamRequest:(NSURLRequest *)request lineHandler:(void (^)(NSString *))lineHandler
			  completionHandler:(void (^)(NSHTTPURLResponse *, NSData *, NSError *))completionHandler
{
	self.lastRequest = request;
	for (NSString *line in self.stagedLines) {
		lineHandler(line);
	}
	completionHandler([[NSHTTPURLResponse alloc] initWithURL:request.URL statusCode:200 HTTPVersion:@"HTTP/1.1" headerFields:nil], nil, nil);
	return ^{};
}
@end

@interface NFKAttachStubAnthropicBackend : NFKAnthropicBackend
@property (nonatomic, strong) NSURLRequest *lastRequest;
@property (nonatomic, copy, nullable) NSString *stagedBody;
@end

@implementation NFKAttachStubAnthropicBackend
- (NSData *)sendRequest:(NSURLRequest *)request response:(NSHTTPURLResponse **)outResponse error:(NSError **)outError
{
	self.lastRequest = request;
	if (outResponse != NULL) {
		*outResponse = [[NSHTTPURLResponse alloc] initWithURL:request.URL statusCode:200 HTTPVersion:@"HTTP/1.1" headerFields:nil];
	}
	return [(self.stagedBody ?: @"{\"content\":[{\"type\":\"text\",\"text\":\"ok\"}]}") dataUsingEncoding:NSUTF8StringEncoding];
}
@end

@interface NFKRemoteAttachmentsTests : XCTestCase
@property (nonatomic, strong) NFKAttachStubRemoteBackend *backend;
@property (nonatomic, strong) NFKAttachStubAnthropicBackend *anthropic;
@property (nonatomic, strong) NSURL *wav;
@property (nonatomic, strong) NSURL *pdf;
@property (nonatomic, copy) NSData *wavBytes;
@property (nonatomic, copy) NSData *pdfBytes;
@end

@implementation NFKRemoteAttachmentsTests

- (void)setUp
{
	[super setUp];
	self.backend = [[NFKAttachStubRemoteBackend alloc] init];
	self.backend.endpointURL = [NSURL URLWithString:@"https://api.openai.com/v1/chat/completions"];
	self.backend.modelName = @"gpt-4o-audio-preview";
	self.backend.stagedBody = @"{\"choices\":[{\"message\":{\"content\":\"ok\"}}]}";
	self.anthropic = [[NFKAttachStubAnthropicBackend alloc] init];
	self.anthropic.modelName = @"m";

	NSURL *directory = [NSURL fileURLWithPath:NSTemporaryDirectory()];
	self.wavBytes = [@"RIFF....WAVEfmt " dataUsingEncoding:NSUTF8StringEncoding];
	self.wav = [directory URLByAppendingPathComponent:[NSString stringWithFormat:@"%@.wav", NSUUID.UUID.UUIDString]];
	[self.wavBytes writeToURL:self.wav atomically:YES];
	self.pdfBytes = [@"%PDF-1.4 fake" dataUsingEncoding:NSUTF8StringEncoding];
	self.pdf = [directory URLByAppendingPathComponent:@"brief.pdf"];
	[self.pdfBytes writeToURL:self.pdf atomically:YES];
}

- (void)tearDown
{
	[NSFileManager.defaultManager removeItemAtURL:self.wav error:NULL];
	[NSFileManager.defaultManager removeItemAtURL:self.pdf error:NULL];
	[super tearDown];
}

- (NSDictionary *)bodyOf:(NSURLRequest *)request
{
	return [NSJSONSerialization JSONObjectWithData:request.HTTPBody options:0 error:NULL];
}

- (NSArray *)userPartsOf:(NSURLRequest *)request
{
	return [self bodyOf:request][@"messages"][0][@"content"];
}

#pragma mark Audio in

- (void)testAudioRidesAsAnInputAudioPartWithItsFormat
{
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"What did I say?",
																			  NFKInputAudio: [NFKAudioAsset audioAssetWithFileURL:self.wav] }];
	XCTAssertNotNil([self.backend runInferenceForRequest:request error:NULL]);
	NSArray *parts = [self userPartsOf:self.backend.lastRequest];
	XCTAssertEqual(parts.count, 2);
	XCTAssertEqualObjects(parts[1][@"type"], @"input_audio");
	XCTAssertEqualObjects(parts[1][@"input_audio"][@"format"], @"wav");
	XCTAssertEqualObjects(parts[1][@"input_audio"][@"data"], [self.wavBytes base64EncodedStringWithOptions:0]);
}

- (void)testInMemoryAudioIsSentAsWAVAndAMissingFileIsRefused
{
	[self.backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"x", NFKInputAudio: self.wavBytes }] error:NULL];
	XCTAssertEqualObjects([self userPartsOf:self.backend.lastRequest][1][@"input_audio"][@"format"], @"wav");

	NFKInferenceRequest *missing = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"x",
			NFKInputAudio: [NFKAudioAsset audioAssetWithFileURL:[NSURL fileURLWithPath:@"/nonesuch.wav"]] }];
	NSError *error = nil;
	XCTAssertNil([self.backend runInferenceForRequest:missing error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceMissingInput);
}

- (void)testAnthropicRefusesAudioInEitherDirectionRatherThanDroppingIt
{
	NSError *error = nil;
	NFKInferenceRequest *audioIn = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"x", NFKInputAudio: self.wavBytes }];
	XCTAssertNil([self.anthropic runInferenceForRequest:audioIn error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceUnsupported);
	XCTAssertNil(self.anthropic.lastRequest, @"nothing was sent");

	NFKInferenceRequest *audioOut = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"x" }
															   parameters:@{ NFKParameterAudioOutput: @{ @"voice": @"alloy" } }
														   outputModality:NFKModalityText];
	XCTAssertNil([self.anthropic runInferenceForRequest:audioOut error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceUnsupported);
}

#pragma mark Documents

- (void)testADocumentRidesAsAFilePartAndAsADocumentBlock
{
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Summarize.",
																			  NFKInputDocument: self.pdf,
																			  NFKInputDocuments: @[ self.pdfBytes ] }];
	XCTAssertNotNil([self.backend runInferenceForRequest:request error:NULL]);
	NSArray *parts = [self userPartsOf:self.backend.lastRequest];
	XCTAssertEqual(parts.count, 3, @"the text and two documents");
	XCTAssertEqualObjects(parts[1][@"type"], @"file");
	XCTAssertEqualObjects(parts[1][@"file"][@"filename"], @"brief.pdf");
	XCTAssertEqualObjects(parts[1][@"file"][@"file_data"],
						  [@"data:application/pdf;base64," stringByAppendingString:[self.pdfBytes base64EncodedStringWithOptions:0]]);
	XCTAssertEqualObjects(parts[2][@"file"][@"filename"], @"document.pdf", @"in-memory bytes get a default name");

	XCTAssertNotNil([self.anthropic runInferenceForRequest:request error:NULL]);
	NSArray *blocks = [self userPartsOf:self.anthropic.lastRequest];
	XCTAssertEqual(blocks.count, 3);
	XCTAssertEqualObjects(blocks[0][@"type"], @"document");
	XCTAssertEqualObjects(blocks[0][@"source"][@"media_type"], @"application/pdf");
	XCTAssertEqualObjects(blocks[0][@"title"], @"brief.pdf");
	XCTAssertEqualObjects(blocks[2][@"type"], @"text", @"the text comes after the media");
}

#pragma mark Video

- (void)testAClipIsSampledIntoFramesBesideThePrompt
{
	NSError *error = nil;
	NSURL *clip = [NFKTestClip writeClipWithColors:@[ @[ @1, @0, @0 ], @[ @0, @0, @1 ] ] error:&error];
	XCTAssertNotNil(clip, @"%@", error);
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"What happens?",
																			  NFKInputVideo: [NFKVideoAsset videoAssetWithFileURL:clip] }
															  parameters:@{ NFKParameterVideoFrameCount: @2 }
														  outputModality:NFKModalityText];
	XCTAssertNotNil([self.backend runInferenceForRequest:request error:&error], @"%@", error);
	NSArray *parts = [self userPartsOf:self.backend.lastRequest];
	XCTAssertEqual(parts.count, 3, @"the text and two frames");
	XCTAssertEqualObjects(parts[1][@"type"], @"image_url");
	XCTAssertTrue([parts[2][@"image_url"][@"url"] hasPrefix:@"data:image/png;base64,"]);
	XCTAssertNil([self bodyOf:self.backend.lastRequest][NFKParameterVideoFrameCount], @"the count is consumed, not sent");

	XCTAssertNotNil([self.anthropic runInferenceForRequest:request error:&error], @"%@", error);
	XCTAssertEqualObjects([self userPartsOf:self.anthropic.lastRequest][0][@"type"], @"image");
	[NSFileManager.defaultManager removeItemAtURL:clip error:NULL];
}

#pragma mark Audio out

#pragma mark Dialects

- (void)testMistralReadsAPDFAsADocumentURLAndAudioAsABareString
{
	self.backend.chatDialect = NFKRemoteChatDialectMistral;
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"summarize",
																			  NFKInputDocument: self.pdf,
																			  NFKInputAudio: [NFKAudioAsset audioAssetWithFileURL:self.wav] }];
	XCTAssertNotNil([self.backend runInferenceForRequest:request error:NULL]);
	NSArray *parts = [self userPartsOf:self.backend.lastRequest];
	NSDictionary *audio = parts[1];
	XCTAssertEqualObjects(audio[@"input_audio"], [self.wavBytes base64EncodedStringWithOptions:0]);
	NSDictionary *document = parts[2];
	XCTAssertEqualObjects(document[@"type"], @"document_url");
	XCTAssertTrue([document[@"document_url"] hasPrefix:@"data:application/pdf;base64,"]);
}

- (void)testAPlainTextDocumentRidesAsTextEverywhereAndAsATextSourceOnAnthropic
{
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"what does it say?",
																			  NFKInputDocument: @"The launch is Tuesday." }
															   parameters:@{ NFKParameterCitations: @YES }];
	XCTAssertNotNil([self.backend runInferenceForRequest:request error:NULL]);
	NSDictionary *part = [self userPartsOf:self.backend.lastRequest][1];
	XCTAssertEqualObjects(part[@"type"], @"text");
	XCTAssertTrue([part[@"text"] containsString:@"The launch is Tuesday."]);

	XCTAssertNotNil([self.anthropic runInferenceForRequest:request error:NULL]);
	NSDictionary *block = [self bodyOf:self.anthropic.lastRequest][@"messages"][0][@"content"][0];
	XCTAssertEqualObjects(block[@"source"], (@{ @"type": @"text", @"media_type": @"text/plain", @"data": @"The launch is Tuesday." }));
	XCTAssertEqualObjects(block[@"citations"], (@{ @"enabled": @YES }));
}

// OpenRouter and vLLM read a whole clip as video_url, llama.cpp as input_video; naming a frame
// count asks for sampled frames instead.
- (void)testAWholeClipRidesAsAVideoPartWhereTheDialectReadsOne
{
	NSError *error = nil;
	NSURL *clip = [NFKTestClip writeClipWithColors:@[ @[ @1, @0, @0 ], @[ @0, @0, @1 ] ] error:&error];
	XCTAssertNotNil(clip, @"%@", error);
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"what happens?",
																			  NFKInputVideo: [NFKVideoAsset videoAssetWithFileURL:clip] }];
	self.backend.chatDialect = NFKRemoteChatDialectOpenRouter;
	XCTAssertNotNil([self.backend runInferenceForRequest:request error:NULL]);
	NSArray *parts = [self userPartsOf:self.backend.lastRequest];
	XCTAssertEqual(parts.count, 2);
	XCTAssertEqualObjects(parts[1][@"type"], @"video_url");
	XCTAssertTrue([parts[1][@"video_url"][@"url"] hasPrefix:@"data:video/"]);

	self.backend.chatDialect = NFKRemoteChatDialectLlamaCpp;
	XCTAssertNotNil([self.backend runInferenceForRequest:request error:NULL]);
	XCTAssertEqualObjects([self userPartsOf:self.backend.lastRequest][1][@"type"], @"input_video");

	NFKInferenceRequest *sampled = [NFKInferenceRequest requestWithInputs:request.inputs parameters:@{ NFKParameterVideoFrameCount: @2 }];
	XCTAssertNotNil([self.backend runInferenceForRequest:sampled error:NULL]);
	XCTAssertEqualObjects([self userPartsOf:self.backend.lastRequest][1][@"type"], @"image_url", @"a frame count asks for frames");
	[NSFileManager.defaultManager removeItemAtURL:clip error:NULL];
}

#pragma mark Reply extras

- (void)testAnImageRequestAsksForImageOutputAndDecodesTheReplysImages
{
	CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
	CGContextRef context = CGBitmapContextCreate(NULL, 4, 4, 8, 16, colorSpace, kCGBitmapByteOrder32Little | kCGImageAlphaPremultipliedFirst);
	CGColorSpaceRelease(colorSpace);
	CGImageRef square = CGBitmapContextCreateImage(context);
	CGContextRelease(context);
	NSString *png = [[NFKImageCoding PNGDataForImage:(__bridge id)square] base64EncodedStringWithOptions:0];
	CGImageRelease(square);
	self.backend.stagedBody = [NSString stringWithFormat:@"{\"choices\":[{\"message\":{\"content\":\"here\","
							   "\"images\":[{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,%@\"}}]}}]}", png];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"draw a square" } parameters:@{}
														   outputModality:NFKModalityImage];
	NFKInferenceResult *result = [self.backend runInferenceForRequest:request error:NULL];
	XCTAssertEqualObjects([self bodyOf:self.backend.lastRequest][@"modalities"], (@[ @"image", @"text" ]));
	XCTAssertNotNil([result outputForKey:NFKOutputImage]);
	XCTAssertEqualObjects(result.text, @"here");
}

- (void)testURLCitationsAndExecutedToolsComeBackBesideTheText
{
	self.backend.stagedBody = @"{\"choices\":[{\"message\":{\"content\":\"Rain today.\","
		"\"annotations\":[{\"type\":\"url_citation\",\"url_citation\":{\"url\":\"https://weather.example\",\"title\":\"Forecast\",\"start_index\":0,\"end_index\":11}}],"
		"\"executed_tools\":[{\"index\":0,\"type\":\"browser_search\",\"arguments\":\"{\\\"query\\\":\\\"weather\\\"}\",\"output\":\"rain\"}]}}]}";
	NFKInferenceResult *result = [self.backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"weather?" }] error:NULL];
	NSDictionary *citation = [[result outputForKey:NFKOutputCitations] firstObject];
	XCTAssertEqualObjects(citation[@"url"], @"https://weather.example");
	XCTAssertEqualObjects(citation[@"end"], @11);
	NSDictionary *ran = [[result outputForKey:NFKOutputServerToolResults] firstObject];
	XCTAssertEqualObjects(ran[@"name"], @"browser_search");
	XCTAssertEqualObjects(ran[@"output"], @"rain");
}

- (void)testAnthropicCitationsAndServerToolsComeBackPaired
{
	self.anthropic.stagedBody = @"{\"content\":["
		"{\"type\":\"server_tool_use\",\"id\":\"srv_1\",\"name\":\"web_search\",\"input\":{\"query\":\"launch\"}},"
		"{\"type\":\"web_search_tool_result\",\"tool_use_id\":\"srv_1\",\"content\":[{\"type\":\"web_search_result\",\"url\":\"https://news.example\",\"title\":\"News\"}]},"
		"{\"type\":\"text\",\"text\":\"It launches Tuesday.\",\"citations\":[{\"type\":\"char_location\",\"cited_text\":\"The launch is Tuesday.\",\"document_index\":0,\"start_char_index\":0,\"end_char_index\":22}]}]}";
	NFKInferenceResult *result = [self.anthropic runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"when?" }] error:NULL];
	XCTAssertEqualObjects(result.text, @"It launches Tuesday.");
	NSDictionary *citation = [[result outputForKey:NFKOutputCitations] firstObject];
	XCTAssertEqualObjects(citation[@"text"], @"The launch is Tuesday.");
	XCTAssertEqualObjects(citation[@"end"], @22);
	NSDictionary *ran = [[result outputForKey:NFKOutputServerToolResults] firstObject];
	XCTAssertEqualObjects(ran[@"name"], @"web_search");
	XCTAssertEqualObjects(ran[@"input"], (@{ @"query": @"launch" }));
	XCTAssertEqual([ran[@"output"] count], 1);
}

- (void)testTheProviderFactorySetsEachPresetsDialect
{
	NFKRemoteBackend *mistral = (NFKRemoteBackend *)[NFKRemoteProvider backendForProvider:NFKRemoteProvider.mistral apiKey:@"k" modelName:@"m"];
	XCTAssertEqual(mistral.chatDialect, NFKRemoteChatDialectMistral);
	NFKRemoteBackend *vllm = (NFKRemoteBackend *)[NFKRemoteProvider backendForProvider:NFKRemoteProvider.vLLM apiKey:nil modelName:@"m"];
	XCTAssertEqual(vllm.chatDialect, NFKRemoteChatDialectVLLM);
	NFKRemoteBackend *openAI = (NFKRemoteBackend *)[NFKRemoteProvider backendForProvider:NFKRemoteProvider.openAI apiKey:@"k" modelName:@"m"];
	XCTAssertEqual(openAI.chatDialect, NFKRemoteChatDialectStandard);
}

- (void)testASpokenReplyIsAskedForThroughModalitiesAndComesBackAsAnAudioAsset
{
	NSData *reply = [@"RIFF....WAVEreply" dataUsingEncoding:NSUTF8StringEncoding];
	self.backend.stagedBody = [NSString stringWithFormat:@"{\"choices\":[{\"message\":{\"content\":null,\"audio\":{\"id\":\"a1\","
							   "\"data\":\"%@\",\"transcript\":\"Hello there.\"}}}]}", [reply base64EncodedStringWithOptions:0]];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Say hello." }
															  parameters:@{ NFKParameterAudioOutput: @{ @"voice": @"alloy" } }
														  outputModality:NFKModalityAudio];
	NSError *error = nil;
	NFKInferenceResult *result = [self.backend runInferenceForRequest:request error:&error];
	XCTAssertNotNil(result, @"%@", error);

	NSDictionary *body = [self bodyOf:self.backend.lastRequest];
	XCTAssertEqualObjects(body[@"modalities"], (@[ @"text", @"audio" ]));
	XCTAssertEqualObjects(body[@"audio"], (@{ @"voice": @"alloy", @"format": @"wav" }), @"the format defaults to wav");
	XCTAssertNil(body[NFKParameterAudioOutput]);

	NFKAudioAsset *asset = [result outputForKey:NFKOutputAudio];
	XCTAssertEqualObjects(asset.fileURL.pathExtension, @"wav");
	XCTAssertEqualObjects([NSData dataWithContentsOfURL:asset.fileURL], reply);
	XCTAssertEqualObjects(result.text, @"Hello there.", @"the transcript stands in for the null content");
	[NSFileManager.defaultManager removeItemAtURL:asset.fileURL error:NULL];
}

// The spoken reply streams as base64 chunks that concatenate, beside its transcript's pieces.
- (void)testAStreamedSpokenReplyAssemblesItsChunks
{
	NSData *reply = [@"RIFF....WAVEstreamed" dataUsingEncoding:NSUTF8StringEncoding];
	NSString *base64 = [reply base64EncodedStringWithOptions:0];
	NSString *head = [base64 substringToIndex:8], *tail = [base64 substringFromIndex:8];
	self.backend.stagedLines = @[
		[NSString stringWithFormat:@"data: {\"choices\":[{\"delta\":{\"audio\":{\"id\":\"a1\",\"data\":\"%@\",\"transcript\":\"Hel\"}}}]}", head],
		[NSString stringWithFormat:@"data: {\"choices\":[{\"delta\":{\"audio\":{\"data\":\"%@\",\"transcript\":\"lo.\"}}}]}", tail],
		@"data: [DONE]",
	];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Say hello." }
															  parameters:@{ NFKParameterAudioOutput: @{ @"voice": @"alloy", @"format": @"wav" } }
														  outputModality:NFKModalityAudio];
	NFKInferenceJob *job = [self.backend submitInferenceJobForRequest:request];
	XCTAssertEqual(job.status, NFKInferenceJobStatusSucceeded);
	XCTAssertEqualObjects(job.result.text, @"Hello.");
	NFKAudioAsset *asset = [job.result outputForKey:NFKOutputAudio];
	XCTAssertEqualObjects([NSData dataWithContentsOfURL:asset.fileURL], reply);
	[NSFileManager.defaultManager removeItemAtURL:asset.fileURL error:NULL];
}

@end
