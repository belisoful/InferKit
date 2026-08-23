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
