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
#import <InferKit/NFKInferenceJob.h>

@interface NFKStubSpeechBackend : NFKRemoteSpeechBackend
@property (nonatomic, strong) NSURLRequest *lastRequest;
@property (nonatomic, strong, nullable) NSData *stagedData;
@property (nonatomic, assign) NSInteger stagedStatusCode;
@property (nonatomic, strong) NSMutableArray<NSURLRequest *> *requests;
@property (nonatomic, copy, nullable) NSData * _Nullable (^stagedReply)(NSURLRequest *request);
@property (nonatomic, copy, nullable) NSArray<NSString *> *stagedLines;
@end

@implementation NFKStubSpeechBackend

- (NSData *)sendRequest:(NSURLRequest *)request response:(NSHTTPURLResponse **)outResponse error:(NSError **)outError
{
	self.lastRequest = request;
	if (self.requests == nil) {
		self.requests = [NSMutableArray array];
	}
	[self.requests addObject:request];
	if (outResponse != NULL) {
		*outResponse = [[NSHTTPURLResponse alloc] initWithURL:request.URL statusCode:self.stagedStatusCode ?: 200
												  HTTPVersion:@"HTTP/1.1" headerFields:nil];
	}
	return self.stagedReply != nil ? self.stagedReply(request) : self.stagedData;
}

- (void (^)(void))streamRequest:(NSURLRequest *)request
					lineHandler:(void (^)(NSString *))lineHandler
			  completionHandler:(void (^)(NSHTTPURLResponse *, NSData *, NSError *))completionHandler
{
	self.lastRequest = request;
	for (NSString *line in self.stagedLines) {
		lineHandler(line);
	}
	completionHandler([[NSHTTPURLResponse alloc] initWithURL:request.URL statusCode:200 HTTPVersion:@"HTTP/1.1" headerFields:nil], nil, nil);
	return ^{};
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
	XCTAssertEqual(error.code, kNFKError_InferenceRefused, @"a 400 is the request's problem");
	XCTAssertTrue([error.localizedDescription containsString:@"voice not found"]);

	self.backend.stagedStatusCode = 200;
	self.backend.stagedData = [NSData data];
	XCTAssertNil([self.backend runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"x" }] error:&error]);
	XCTAssertEqual(error.code, kNFKError_InferenceBackendFailure);
}

#pragma mark Provider styles

- (NFKStubSpeechBackend *)stubFor:(NFKRemoteProvider *)provider model:(NSString *)model voice:(nullable NSString *)voice
{
	NFKRemoteSpeechBackend *made = [NFKRemoteSpeechBackend backendForProvider:provider apiKey:@"k" modelName:model voice:voice];
	NFKStubSpeechBackend *stub = [[NFKStubSpeechBackend alloc] init];
	stub.endpointURL = made.endpointURL;
	stub.apiStyle = made.apiStyle;
	stub.maximumInputLength = made.maximumInputLength;
	stub.modelName = model;
	stub.voice = voice;
	stub.apiKey = @"k";
	stub.outputDirectoryURL = self.directory;
	return stub;
}

/*! A minimal PCM WAV holding the given sample bytes. */
- (NSData *)wavWithSamples:(NSData *)samples
{
	NSMutableData *wav = [NSMutableData data];
	uint32_t riff = (uint32_t)(36 + samples.length), fmtSize = 16, rate = 24000, byteRate = 48000, dataSize = (uint32_t)samples.length;
	uint16_t pcm = 1, channels = 1, align = 2, bits = 16;
	[wav appendBytes:"RIFF" length:4]; [wav appendBytes:&riff length:4]; [wav appendBytes:"WAVEfmt " length:8];
	[wav appendBytes:&fmtSize length:4]; [wav appendBytes:&pcm length:2]; [wav appendBytes:&channels length:2];
	[wav appendBytes:&rate length:4]; [wav appendBytes:&byteRate length:4]; [wav appendBytes:&align length:2];
	[wav appendBytes:&bits length:2]; [wav appendBytes:"data" length:4]; [wav appendBytes:&dataSize length:4];
	[wav appendData:samples];
	return wav;
}

- (void)testEachProviderGetsItsOwnPathAndTheSilentOnesNone
{
	XCTAssertEqualObjects([NFKRemoteSpeechBackend backendForProvider:NFKRemoteProvider.xAI apiKey:@"k" modelName:nil voice:@"eve"].endpointURL.absoluteString,
						  @"https://api.x.ai/v1/tts");
	XCTAssertEqual([NFKRemoteSpeechBackend backendForProvider:NFKRemoteProvider.groq apiKey:@"k" modelName:@"m" voice:@"troy"].maximumInputLength, 200);
	for (NFKRemoteProvider *provider in @[ NFKRemoteProvider.googleGemini, NFKRemoteProvider.deepSeek, NFKRemoteProvider.ollama ]) {
		XCTAssertNil([NFKRemoteSpeechBackend backendForProvider:provider apiKey:@"k" modelName:@"m" voice:@"v"], @"%@", provider.identifier);
	}
}

- (void)testXAISpeaksThroughTTSWithAnOutputFormatAndAnAutoLanguage
{
	NFKStubSpeechBackend *xai = [self stubFor:NFKRemoteProvider.xAI model:nil voice:nil];
	xai.stagedData = [self wavWithSamples:[NSData dataWithBytes:"\x01\x02" length:2]];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Hello" }
															   parameters:@{ NFKParameterSampleRate: @24000, @"speed": @1.2 }];
	NSError *error = nil;
	XCTAssertNotNil([xai runInferenceForRequest:request error:&error], @"%@", error);
	NSDictionary *body = [xai decodedRequestBody];
	XCTAssertEqualObjects(body[@"text"], @"Hello");
	XCTAssertEqualObjects(body[@"language"], @"auto");
	XCTAssertEqualObjects(body[@"output_format"], (@{ @"codec": @"wav", @"sample_rate": @24000 }));
	XCTAssertEqualObjects(body[@"speed"], @1.2);
	XCTAssertNil(body[@"voice_id"], @"xAI defaults its own voice");
}

- (void)testMistralDecodesTheBase64ReplyAndClonesFromAReferenceClip
{
	NFKStubSpeechBackend *mistral = [self stubFor:NFKRemoteProvider.mistral model:@"voxtral-mini-tts-2603" voice:nil];
	NSData *audio = [@"ID3 fake mp3" dataUsingEncoding:NSUTF8StringEncoding];
	NSString *reply = [NSString stringWithFormat:@"{\"audio_data\":\"%@\"}", [audio base64EncodedStringWithOptions:0]];
	mistral.stagedData = [reply dataUsingEncoding:NSUTF8StringEncoding];
	mistral.responseFormat = @"mp3";
	NSData *clip = [@"RIFFclip" dataUsingEncoding:NSUTF8StringEncoding];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Bonjour", NFKInputVoiceReference: clip }];
	NSError *error = nil;
	NFKInferenceResult *result = [mistral runInferenceForRequest:request error:&error];
	XCTAssertNotNil(result, @"%@", error);
	NSDictionary *body = [mistral decodedRequestBody];
	XCTAssertEqualObjects(body[@"ref_audio"], [clip base64EncodedStringWithOptions:0]);
	XCTAssertEqualObjects(body[@"input"], @"Bonjour");
	NFKAudioAsset *spoken = [result outputForKey:NFKOutputAudio];
	XCTAssertEqualObjects([NSData dataWithContentsOfURL:spoken.fileURL], audio);
}

// Groq's Orpheus voices refuse more than 200 characters: the text is spoken in pieces cut at
// sentence ends and the WAV pieces joined by their samples.
- (void)testLongTextIsSpokenInPiecesAndTheWAVPiecesJoin
{
	NFKStubSpeechBackend *groq = [self stubFor:NFKRemoteProvider.groq model:@"canopylabs/orpheus-v1-english" voice:@"troy"];
	__block uint8_t sample = 0;
	groq.stagedReply = ^NSData *(NSURLRequest *request) {
		sample++;
		return [self wavWithSamples:[NSData dataWithBytes:&sample length:1]];
	};
	NSString *sentence = [@"The" stringByPaddingToLength:120 withString:@" word" startingAtIndex:0];
	NSString *text = [NSString stringWithFormat:@"%@. %@. %@.", sentence, sentence, sentence];
	NSError *error = nil;
	NFKInferenceResult *result = [groq runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: text }] error:&error];
	XCTAssertNotNil(result, @"%@", error);
	XCTAssertEqual(groq.requests.count, 3, @"one request per sentence that fits");
	for (NSURLRequest *request in groq.requests) {
		NSDictionary *body = [NSJSONSerialization JSONObjectWithData:request.HTTPBody options:0 error:NULL];
		XCTAssertLessThanOrEqual([body[@"input"] length], 200);
	}
	NSData *joined = [NSData dataWithContentsOfURL:[[result outputForKey:NFKOutputAudio] fileURL]];
	const uint8_t expected[] = { 1, 2, 3 };
	XCTAssertEqualObjects([joined subdataWithRange:NSMakeRange(joined.length - 3, 3)], [NSData dataWithBytes:expected length:3]);
	uint32_t dataSize = 0;
	[joined getBytes:&dataSize range:NSMakeRange(40, 4)];
	XCTAssertEqual(dataSize, 3);
}

- (void)testAStreamedReplyGrowsThePartialAudioAndFinishesWithTheWhole
{
	NFKStubSpeechBackend *openAI = [self stubFor:NFKRemoteProvider.openAI model:@"gpt-4o-mini-tts" voice:@"marin"];
	openAI.streams = YES;
	openAI.responseFormat = @"pcm";
	NSString *one = [[@"ab" dataUsingEncoding:NSUTF8StringEncoding] base64EncodedStringWithOptions:0];
	NSString *two = [[@"cd" dataUsingEncoding:NSUTF8StringEncoding] base64EncodedStringWithOptions:0];
	openAI.stagedLines = @[ [NSString stringWithFormat:@"data: {\"type\":\"speech.audio.delta\",\"audio\":\"%@\"}", one],
							[NSString stringWithFormat:@"data: {\"type\":\"speech.audio.delta\",\"audio\":\"%@\"}", two],
							@"data: {\"type\":\"speech.audio.done\"}" ];
	NFKInferenceJob *job = [openAI submitInferenceJobForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: @"Hi" }]];
	XCTAssertEqual(job.status, NFKInferenceJobStatusSucceeded, @"%@", job.error);
	XCTAssertEqualObjects([openAI decodedRequestBody][@"stream_format"], @"sse");
	NSData *audio = [NSData dataWithContentsOfURL:[[job.result outputForKey:NFKOutputAudio] fileURL]];
	XCTAssertEqualObjects(audio, [@"abcd" dataUsingEncoding:NSUTF8StringEncoding]);
}

- (void)testVoicesAreListedWhereTheServiceListsThem
{
	NFKStubSpeechBackend *xai = [self stubFor:NFKRemoteProvider.xAI model:nil voice:nil];
	xai.stagedData = [@"{\"voices\":[{\"voice_id\":\"eve\",\"name\":\"Eve\",\"language\":\"en\"}]}" dataUsingEncoding:NSUTF8StringEncoding];
	NSError *error = nil;
	NSArray<NFKRemoteVoice *> *voices = [xai availableVoicesWithError:&error];
	XCTAssertEqualObjects(xai.lastRequest.URL.absoluteString, @"https://api.x.ai/v1/tts/voices");
	XCTAssertEqualObjects(voices.firstObject.identifier, @"eve");
	XCTAssertEqualObjects(voices.firstObject.languages, @[ @"en" ]);

	NFKStubSpeechBackend *together = [self stubFor:NFKRemoteProvider.together model:@"cartesia/sonic" voice:@"v"];
	together.stagedData = [@"{\"data\":[{\"id\":\"v1\"}]}" dataUsingEncoding:NSUTF8StringEncoding];
	XCTAssertEqual([together availableVoicesWithError:NULL].count, 1);
	XCTAssertEqualObjects(together.lastRequest.URL.absoluteString, @"https://api.together.xyz/v1/voices?model=cartesia/sonic");

	NFKStubSpeechBackend *openAI = [self stubFor:NFKRemoteProvider.openAI model:@"tts-1" voice:@"alloy"];
	XCTAssertNil([openAI availableVoicesWithError:&error]);
	XCTAssertEqual(error.code, (NSInteger)kNFKError_InferenceUnsupported);
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
