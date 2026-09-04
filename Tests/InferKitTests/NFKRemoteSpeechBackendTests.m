//
//  NFKRemoteSpeechBackendTests.m
//  InferKitTests
//
//  The speech request and the file the reply becomes, through a stub transport, and the provider
//  factory's derivation.
//

#import <XCTest/XCTest.h>
#import <InferKit/NFKRemoteSpeechBackend.h>
#import <InferKit/NFKRemoteProvider.h>
#import <InferKit/NFKInferenceRequest.h>
#import <InferKit/NFKInferenceResult.h>
#import <InferKit/NFKInferenceKeys.h>
#import <InferKit/NFKAudioAsset.h>
#import <InferKit/NFKErrors.h>

@interface NFKStubSpeechBackend : NFKRemoteSpeechBackend
@property (nonatomic, strong) NSURLRequest *lastRequest;
@property (nonatomic, strong, nullable) NSData *stagedData;
@property (nonatomic, assign) NSInteger stagedStatusCode;
@end

@implementation NFKStubSpeechBackend

- (NSData *)sendRequest:(NSURLRequest *)request response:(NSHTTPURLResponse **)outResponse error:(NSError **)outError
{
	self.lastRequest = request;
	if (outResponse != NULL) {
		*outResponse = [[NSHTTPURLResponse alloc] initWithURL:request.URL statusCode:self.stagedStatusCode ?: 200
												  HTTPVersion:@"HTTP/1.1" headerFields:nil];
	}
	return self.stagedData;
}

- (NSDictionary *)decodedRequestBody
{
	return [NSJSONSerialization JSONObjectWithData:self.lastRequest.HTTPBody options:0 error:NULL];
}

@end

@interface NFKRemoteSpeechBackendTests : XCTestCase
@property (nonatomic, strong) NFKStubSpeechBackend *backend;
@property (nonatomic, strong) NSURL *directory;
@end

@implementation NFKRemoteSpeechBackendTests

- (void)setUp
{
	[super setUp];
	self.directory = [[NSURL fileURLWithPath:NSTemporaryDirectory()] URLByAppendingPathComponent:NSUUID.UUID.UUIDString isDirectory:YES];
	self.backend = [[NFKStubSpeechBackend alloc] init];
	self.backend.endpointURL = [NSURL URLWithString:@"https://api.openai.com/v1/audio/speech"];
	self.backend.modelName = @"tts-1";
	self.backend.voice = @"alloy";
	self.backend.outputDirectoryURL = self.directory;
	self.backend.stagedData = [@"RIFF....WAVEfmt " dataUsingEncoding:NSUTF8StringEncoding];
}

- (void)tearDown
{
	[NSFileManager.defaultManager removeItemAtURL:self.directory error:NULL];
	[super tearDown];
}

- (void)testTheBackendReportsItsIdentityAndDefaults
{
	XCTAssertEqualObjects(self.backend.backendIdentifier, @"remote-speech");
	XCTAssertTrue(self.backend.isReady);
	XCTAssertEqualObjects(self.backend.responseFormat, @"wav", @"the container the on-device speech backend writes");
	XCTAssertFalse([NFKRemoteSpeechBackend backendWithEndpointURL:nil].isReady);
}

- (void)testThePromptBecomesTheInputAndTheReplyBecomesAnAudioAsset
{
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Hello there." }];
	NSError *error = nil;
	NFKInferenceResult *result = [self.backend runInferenceForRequest:request error:&error];
	XCTAssertNotNil(result, @"%@", error);

	NSDictionary *body = [self.backend decodedRequestBody];
	XCTAssertEqualObjects(body[@"input"], @"Hello there.");
	XCTAssertEqualObjects(body[@"model"], @"tts-1");
	XCTAssertEqualObjects(body[@"voice"], @"alloy");
	XCTAssertEqualObjects(body[@"response_format"], @"wav");
	XCTAssertEqualObjects(self.backend.lastRequest.HTTPMethod, @"POST");

	NFKAudioAsset *asset = [result outputForKey:NFKOutputAudio];
	XCTAssertTrue([asset isKindOfClass:NFKAudioAsset.class]);
	XCTAssertEqualObjects(asset.fileURL.pathExtension, @"wav");
	XCTAssertEqualObjects([NSData dataWithContentsOfURL:asset.fileURL], self.backend.stagedData, @"the bytes land as received");
	XCTAssertTrue([asset.fileURL.path hasPrefix:self.directory.path]);
}

- (void)testParametersFoldInAndAVoiceParameterOverridesTheProperty
{
	self.backend.responseFormat = @"mp3";
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"x" }
															  parameters:@{ @"voice": @"nova", @"speed": @1.2 }
														  outputModality:NFKModalityAudio];
	NFKInferenceResult *result = [self.backend runInferenceForRequest:request error:NULL];
	NSDictionary *body = [self.backend decodedRequestBody];
	XCTAssertEqualObjects(body[@"voice"], @"nova");
	XCTAssertEqualObjects(body[@"speed"], @1.2);
	XCTAssertEqualObjects([[result outputForKey:NFKOutputAudio] fileURL].pathExtension, @"mp3");
}

- (void)testAMissingVoiceOrTextIsRefusedBeforeAnyRequest
{
	self.backend.voice = nil;
	NSError *error = nil;
	XCTAssertNil([self.backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"x" }] error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceNotReady);
	XCTAssertNil(self.backend.lastRequest, @"nothing was sent");

	self.backend.voice = @"alloy";
	XCTAssertNil([self.backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{}] error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceMissingInput);
}

- (void)testAFailingStatusAndAnEmptyReplyAreErrors
{
	self.backend.stagedStatusCode = 400;
	self.backend.stagedData = [@"{\"error\":{\"message\":\"voice not found\"}}" dataUsingEncoding:NSUTF8StringEncoding];
	NSError *error = nil;
	XCTAssertNil([self.backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"x" }] error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceBackendFailure);
	XCTAssertTrue([error.localizedDescription containsString:@"voice not found"]);

	self.backend.stagedStatusCode = 200;
	self.backend.stagedData = [NSData data];
	XCTAssertNil([self.backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"x" }] error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceBackendFailure);
}

- (void)testTheFactoryDerivesTheSpeechURLAndDeclinesAnthropic
{
	NFKRemoteSpeechBackend *groq = [NFKRemoteSpeechBackend backendForProvider:NFKRemoteProvider.groq
																	   apiKey:@"k" modelName:@"playai-tts" voice:@"Fritz-PlayAI"];
	XCTAssertEqualObjects(groq.endpointURL.absoluteString, @"https://api.groq.com/openai/v1/audio/speech");
	XCTAssertEqualObjects(groq.voice, @"Fritz-PlayAI");
	XCTAssertNil([NFKRemoteSpeechBackend backendForProvider:NFKRemoteProvider.anthropic apiKey:@"k" modelName:@"m" voice:@"v"]);
}

@end
