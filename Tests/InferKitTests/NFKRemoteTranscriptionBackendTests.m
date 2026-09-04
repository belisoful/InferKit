//
//  NFKRemoteTranscriptionBackendTests.m
//  InferKitTests
//
//  Exercises audio upload and response parsing through a stub transport. A live call is
//  integration-verified.
//

#import <XCTest/XCTest.h>
#import <InferKit/NFKRemoteTranscriptionBackend.h>
#import <InferKit/NFKInferenceRequest.h>
#import <InferKit/NFKInferenceResult.h>
#import <InferKit/NFKInferenceKeys.h>
#import <InferKit/NFKAudioAsset.h>
#import <InferKit/NFKAudioSegment.h>
#import <InferKit/NFKRemoteProvider.h>
#import <InferKit/NFKErrors.h>

/*! A transcription backend whose transport is stubbed: it records the request and returns staged data. */
@interface StubTranscriptionBackend : NFKRemoteTranscriptionBackend
@property (nonatomic, strong) NSURLRequest *lastRequest;
@property (nonatomic, strong, nullable) NSData *stagedData;
@property (nonatomic, assign) NSInteger stagedStatusCode;
@end

@implementation StubTranscriptionBackend

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

@end

@interface NFKRemoteTranscriptionBackendTests : XCTestCase
@property (nonatomic, strong) StubTranscriptionBackend *backend;
@end

@implementation NFKRemoteTranscriptionBackendTests

- (void)setUp
{
	[super setUp];
	self.backend = [[StubTranscriptionBackend alloc] init];
	self.backend.endpointURL = [NSURL URLWithString:@"https://example.test/v1/audio/transcriptions"];
	self.backend.modelName = @"whisper-1";
}

// The verbose reply's segments become NFKAudioSegments, which is what the on-device Whisper backend
// emits; the decoder's mean log probability becomes the confidence as e to that power.
- (void)testTimestampsAskForTheVerboseReplyAndComeBackAsSegments
{
	self.backend.emitsTimestamps = YES;
	self.backend.stagedData = [@"{\"task\":\"transcribe\",\"text\":\"Thank you. Goodbye.\",\"segments\":["
								"{\"id\":0,\"start\":0.0,\"end\":1.5,\"text\":\" Thank you.\",\"avg_logprob\":-0.1},"
								"{\"id\":1,\"start\":1.5,\"end\":3.0,\"text\":\" Goodbye.\",\"avg_logprob\":-0.7}]}"
							   dataUsingEncoding:NSUTF8StringEncoding];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputAudio: [@"RIFF" dataUsingEncoding:NSUTF8StringEncoding] }];
	NSError *error = nil;
	NFKInferenceResult *result = [self.backend runInferenceForRequest:request error:&error];
	XCTAssertNotNil(result, @"%@", error);

	NSString *body = [[NSString alloc] initWithData:self.backend.lastRequest.HTTPBody encoding:NSISOLatin1StringEncoding];
	XCTAssertTrue([body containsString:@"name=\"response_format\"\r\n\r\nverbose_json"]);

	NSArray<NFKAudioSegment *> *segments = result.segments;
	XCTAssertEqual(segments.count, 2);
	XCTAssertEqualWithAccuracy(segments[0].startSeconds, 0.0, 1e-9);
	XCTAssertEqualWithAccuracy(segments[0].endSeconds, 1.5, 1e-9);
	XCTAssertEqualObjects(segments[0].label, @"Thank you.", @"the leading space the decoder emits is trimmed");
	XCTAssertEqualWithAccuracy(segments[0].confidence, exp(-0.1), 1e-9);
	XCTAssertEqualObjects(segments[1].label, @"Goodbye.");
	XCTAssertEqualObjects(result.text, @"Thank you. Goodbye.");
}

- (void)testACallersOwnResponseFormatWinsOverTheTimestampsDefault
{
	self.backend.emitsTimestamps = YES;
	self.backend.stagedData = [@"1\n00:00:00,000 --> 00:00:01,500\nThank you.\n" dataUsingEncoding:NSUTF8StringEncoding];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputAudio: [@"RIFF" dataUsingEncoding:NSUTF8StringEncoding] }
															  parameters:@{ @"response_format": @"srt" }
														  outputModality:NFKModalityText];
	NFKInferenceResult *result = [self.backend runInferenceForRequest:request error:NULL];
	NSString *body = [[NSString alloc] initWithData:self.backend.lastRequest.HTTPBody encoding:NSISOLatin1StringEncoding];
	XCTAssertFalse([body containsString:@"verbose_json"]);
	XCTAssertTrue([result.text containsString:@"-->"], @"the plain body is the transcript");
	XCTAssertNil(result.segments);
}

// The translations endpoint is the transcriptions one's sibling, reached by switching the path's
// last component.
- (void)testTranslatingSendsTheAudioToTheSiblingEndpoint
{
	self.backend.translates = YES;
	self.backend.stagedData = [@"{\"text\":\"Thank you.\"}" dataUsingEncoding:NSUTF8StringEncoding];
	[self.backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputAudio: [@"RIFF" dataUsingEncoding:NSUTF8StringEncoding] }]
								   error:NULL];
	XCTAssertEqualObjects(self.backend.lastRequest.URL.absoluteString, @"https://example.test/v1/audio/translations");

	NFKRemoteTranscriptionBackend *groq = [NFKRemoteTranscriptionBackend backendForProvider:NFKRemoteProvider.groq apiKey:@"k" modelName:@"whisper-large-v3"];
	XCTAssertEqualObjects(groq.endpointURL.absoluteString, @"https://api.groq.com/openai/v1/audio/transcriptions");
	XCTAssertNil([NFKRemoteTranscriptionBackend backendForProvider:NFKRemoteProvider.anthropic apiKey:@"k" modelName:@"m"]);
}

- (void)testReadinessFollowsTheEndpoint
{
	NFKRemoteTranscriptionBackend *fresh = [NFKRemoteTranscriptionBackend backendWithEndpointURL:nil];
	XCTAssertFalse(fresh.isReady);
	XCTAssertTrue(self.backend.isReady);
	XCTAssertEqualObjects(self.backend.backendIdentifier, @"remote-transcription");
}

- (void)testMissingEndpointFails
{
	self.backend.endpointURL = nil;
	NSError *error = nil;
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputAudio: [NSData data] }];
	XCTAssertNil([self.backend runInferenceForRequest:request error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceNotReady);
}

- (void)testMissingAudioFails
{
	NSError *error = nil;
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"no audio" }];
	XCTAssertNil([self.backend runInferenceForRequest:request error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceMissingInput);
}

- (void)testTranscriptIsReadFromTheResponse
{
	self.backend.stagedData = [@"{\"text\": \"hello world\"}" dataUsingEncoding:NSUTF8StringEncoding];
	NSData *audio = [@"RIFFsome-audio-bytes" dataUsingEncoding:NSUTF8StringEncoding];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputAudio: audio }
															   parameters:@{ @"language": @"en" }
															   outputModality:NFKModalityText];

	NSError *error = nil;
	NFKInferenceResult *result = [self.backend runInferenceForRequest:request error:&error];
	XCTAssertNotNil(result, @"%@", error);
	XCTAssertEqualObjects(result.text, @"hello world");
	XCTAssertEqualObjects([result outputForKey:NFKOutputStructured][@"text"], @"hello world");

	// The multipart body carries the model, the language field, and the audio bytes.
	NSString *body = [[NSString alloc] initWithData:self.backend.lastRequest.HTTPBody encoding:NSUTF8StringEncoding];
	XCTAssertTrue([body containsString:@"name=\"model\""]);
	XCTAssertTrue([body containsString:@"whisper-1"]);
	XCTAssertTrue([body containsString:@"name=\"language\""]);
	XCTAssertTrue([body containsString:@"name=\"file\"; filename="]);
	XCTAssertTrue([body containsString:@"some-audio-bytes"]);
	XCTAssertTrue([self.backend.lastRequest.allHTTPHeaderFields[@"Content-Type"] hasPrefix:@"multipart/form-data; boundary="]);
}

- (void)testAudioAssetFileIsUploaded
{
	NSURL *url = [NSFileManager.defaultManager.temporaryDirectory URLByAppendingPathComponent:@"clip.wav"];
	[[@"RIFFasset-file-bytes" dataUsingEncoding:NSUTF8StringEncoding] writeToURL:url atomically:YES];
	self.backend.stagedData = [@"{\"text\": \"from a file\"}" dataUsingEncoding:NSUTF8StringEncoding];

	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputAudio: [NFKAudioAsset audioAssetWithFileURL:url] }];
	NSError *error = nil;
	NFKInferenceResult *result = [self.backend runInferenceForRequest:request error:&error];
	XCTAssertEqualObjects(result.text, @"from a file", @"%@", error);
	NSString *body = [[NSString alloc] initWithData:self.backend.lastRequest.HTTPBody encoding:NSUTF8StringEncoding];
	XCTAssertTrue([body containsString:@"filename=\"clip.wav\""]);
	XCTAssertTrue([body containsString:@"asset-file-bytes"]);
	[NSFileManager.defaultManager removeItemAtURL:url error:NULL];
}

- (void)testAPlainTextResponseIsTakenAsTheTranscript
{
	// response_format=text returns a bare string, not JSON.
	self.backend.stagedData = [@"just the words" dataUsingEncoding:NSUTF8StringEncoding];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputAudio: [NSData data] }
															   parameters:@{ @"response_format": @"text" }
															   outputModality:NFKModalityText];
	NSError *error = nil;
	NFKInferenceResult *result = [self.backend runInferenceForRequest:request error:&error];
	XCTAssertEqualObjects(result.text, @"just the words", @"%@", error);
	XCTAssertNil([result outputForKey:NFKOutputStructured], @"no structured body for a plain-text response");
}

- (void)testHTTPErrorStatusFails
{
	self.backend.stagedStatusCode = 500;
	self.backend.stagedData = [@"{}" dataUsingEncoding:NSUTF8StringEncoding];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputAudio: [NSData data] }];
	NSError *error = nil;
	XCTAssertNil([self.backend runInferenceForRequest:request error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceBackendFailure);
}

@end
