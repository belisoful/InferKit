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
#import <InferKit/NFKInferenceJob.h>

/*! A transcription backend whose transport is stubbed: it records the request and returns staged data. */
@interface StubTranscriptionBackend : NFKRemoteTranscriptionBackend
@property (nonatomic, strong) NSURLRequest *lastRequest;
@property (nonatomic, strong, nullable) NSData *stagedData;
@property (nonatomic, assign) NSInteger stagedStatusCode;
@property (nonatomic, copy, nullable) NSArray<NSString *> *stagedLines;
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

#pragma mark Provider styles

- (NSString *)formOf:(NSURLRequest *)request
{
	return [[NSString alloc] initWithData:request.HTTPBody encoding:NSISOLatin1StringEncoding];
}

- (StubTranscriptionBackend *)stubFor:(NFKRemoteProvider *)provider model:(NSString *)model
{
	NFKRemoteTranscriptionBackend *made = [NFKRemoteTranscriptionBackend backendForProvider:provider apiKey:@"k" modelName:model];
	StubTranscriptionBackend *stub = [[StubTranscriptionBackend alloc] init];
	stub.endpointURL = made.endpointURL;
	stub.apiStyle = made.apiStyle;
	stub.audioURLFieldName = made.audioURLFieldName;
	stub.modelName = model;
	stub.apiKey = @"k";
	return stub;
}

- (void)testEachProviderGetsItsOwnPathAndTheAudiolessOnesNone
{
	XCTAssertEqualObjects([NFKRemoteTranscriptionBackend backendForProvider:NFKRemoteProvider.xAI apiKey:@"k" modelName:@"m"].endpointURL.absoluteString,
						  @"https://api.x.ai/v1/stt");
	XCTAssertEqual([NFKRemoteTranscriptionBackend backendForProvider:NFKRemoteProvider.mistral apiKey:@"k" modelName:@"m"].apiStyle,
				   NFKRemoteTranscriptionAPIStyleMistral);
	XCTAssertEqualObjects([NFKRemoteTranscriptionBackend backendForProvider:NFKRemoteProvider.groq apiKey:@"k" modelName:@"m"].audioURLFieldName, @"url");
	for (NFKRemoteProvider *provider in @[ NFKRemoteProvider.googleGemini, NFKRemoteProvider.deepSeek, NFKRemoteProvider.anthropic,
											NFKRemoteProvider.ollama, NFKRemoteProvider.llamaCpp ]) {
		XCTAssertNil([NFKRemoteTranscriptionBackend backendForProvider:provider apiKey:@"k" modelName:@"m"], @"%@", provider.identifier);
	}
}

// xAI's /v1/stt times words with speakers; the backend groups them into turns for the segments.
- (void)testXAITranscribesWithDiarizationAndKeyTermsAndTurnsWordsIntoSegments
{
	StubTranscriptionBackend *xai = [self stubFor:NFKRemoteProvider.xAI model:@"grok-voice-transcribe-2.0"];
	xai.stagedData = [(@"{\"text\":\"Hi there. Hello.\",\"language\":\"en\",\"words\":["
						"{\"text\":\"Hi\",\"start\":0.0,\"end\":0.3,\"confidence\":0.9,\"speaker\":0},"
						"{\"text\":\"there.\",\"start\":0.3,\"end\":0.7,\"confidence\":0.8,\"speaker\":0},"
						"{\"text\":\"Hello.\",\"start\":1.0,\"end\":1.5,\"confidence\":0.95,\"speaker\":1}]}")
					  dataUsingEncoding:NSUTF8StringEncoding];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputAudio: [@"RIFF" dataUsingEncoding:NSUTF8StringEncoding] }
															   parameters:@{ NFKParameterSpeakerDiarization: @YES,
																			 NFKParameterVocabulary: @[ @"InferKit" ],
																			 NFKParameterSourceLanguage: @"en" }];
	NSError *error = nil;
	NFKInferenceResult *result = [xai runInferenceForRequest:request error:&error];
	XCTAssertNotNil(result, @"%@", error);
	NSString *form = [self formOf:xai.lastRequest];
	XCTAssertTrue([form containsString:@"name=\"diarize\"\r\n\r\ntrue"]);
	XCTAssertTrue([form containsString:@"name=\"keyterm\"\r\n\r\nInferKit"]);
	XCTAssertTrue([form containsString:@"name=\"language\"\r\n\r\nen"]);
	XCTAssertGreaterThan([form rangeOfString:@"name=\"file\""].location, [form rangeOfString:@"name=\"diarize\""].location,
						 @"xAI reads the fields before the file");
	NSArray<NFKAudioSegment *> *words = [result outputForKey:NFKOutputWords];
	XCTAssertEqual(words.count, 3);
	XCTAssertEqualObjects(words[2].speaker, @"1");
	NSArray<NFKAudioSegment *> *turns = [result outputForKey:NFKOutputSegments];
	XCTAssertEqual(turns.count, 2);
	XCTAssertEqualObjects(turns[0].label, @"Hi there.");
	XCTAssertEqualObjects(turns[0].speaker, @"0");
	XCTAssertEqualWithAccuracy(turns[1].startSeconds, 1.0, 1e-9);
}

// Mistral takes no response_format; it asks for granularities and diarization by field and names
// the speaker on each segment.
- (void)testMistralSendsItsOwnFieldsAndReadsTheSpeakerID
{
	StubTranscriptionBackend *mistral = [self stubFor:NFKRemoteProvider.mistral model:@"voxtral-mini-latest"];
	mistral.emitsTimestamps = YES;
	mistral.stagedData = [(@"{\"model\":\"voxtral\",\"text\":\"Hello\",\"segments\":["
							"{\"type\":\"transcription_segment\",\"text\":\"Hello\",\"start\":0,\"end\":1,\"score\":0.7,\"speaker_id\":\"speaker_1\"}]}")
						  dataUsingEncoding:NSUTF8StringEncoding];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputAudio: [@"RIFF" dataUsingEncoding:NSUTF8StringEncoding] }
															   parameters:@{ NFKParameterSpeakerDiarization: @YES, NFKParameterVocabulary: @[ @"Voxtral" ] }];
	NFKInferenceResult *result = [mistral runInferenceForRequest:request error:NULL];
	NSString *form = [self formOf:mistral.lastRequest];
	XCTAssertFalse([form containsString:@"response_format"]);
	XCTAssertTrue([form containsString:@"name=\"timestamp_granularities\"\r\n\r\nsegment"]);
	XCTAssertTrue([form containsString:@"name=\"context_bias\"\r\n\r\nVoxtral"]);
	NFKAudioSegment *segment = [[result outputForKey:NFKOutputSegments] firstObject];
	XCTAssertEqualObjects(segment.speaker, @"speaker_1");
	XCTAssertEqualWithAccuracy(segment.confidence, 0.7, 1e-9);
}

// OpenAI's diarizing model answers diarized_json; the verbose reply's words land under NFKOutputWords.
- (void)testOpenAIDiarizingModelAsksForDiarizedJSONAndWordsAskForGranularities
{
	StubTranscriptionBackend *openAI = [self stubFor:NFKRemoteProvider.openAI model:@"gpt-4o-transcribe-diarize"];
	openAI.stagedData = [(@"{\"text\":\"A B\",\"segments\":[{\"start\":0,\"end\":1,\"text\":\"A\",\"speaker\":\"A\"}]}")
						 dataUsingEncoding:NSUTF8StringEncoding];
	NSDictionary *diarize = @{ NFKParameterSpeakerDiarization: @YES };
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputAudio: [@"RIFF" dataUsingEncoding:NSUTF8StringEncoding] } parameters:diarize];
	NFKInferenceResult *result = [openAI runInferenceForRequest:request error:NULL];
	NSString *form = [self formOf:openAI.lastRequest];
	XCTAssertTrue([form containsString:@"name=\"response_format\"\r\n\r\ndiarized_json"]);
	XCTAssertTrue([form containsString:@"name=\"chunking_strategy\"\r\n\r\nauto"]);
	NFKAudioSegment *turn = [[result outputForKey:NFKOutputSegments] firstObject];
	XCTAssertEqualObjects(turn.speaker, @"A");

	StubTranscriptionBackend *whisper = [self stubFor:NFKRemoteProvider.groq model:@"whisper-large-v3"];
	whisper.stagedData = [(@"{\"text\":\"hi\",\"words\":[{\"word\":\"hi\",\"start\":0.1,\"end\":0.4}]}") dataUsingEncoding:NSUTF8StringEncoding];
	NFKInferenceRequest *timed = [NFKInferenceRequest requestWithInputs:@{ NFKInputAudio: [@"RIFF" dataUsingEncoding:NSUTF8StringEncoding] }
															 parameters:@{ NFKParameterWordTimestamps: @YES }];
	result = [whisper runInferenceForRequest:timed error:NULL];
	form = [self formOf:whisper.lastRequest];
	XCTAssertTrue([form containsString:@"name=\"response_format\"\r\n\r\nverbose_json"]);
	XCTAssertTrue([form containsString:@"name=\"timestamp_granularities[]\"\r\n\r\nword"]);
	NFKAudioSegment *word = [[result outputForKey:NFKOutputWords] firstObject];
	XCTAssertEqualObjects(word.label, @"hi");
}

// A hosted clip goes by reference where the service takes one, and is fetched and uploaded where not.
- (void)testAHostedClipGoesByReferenceOrIsFetched
{
	StubTranscriptionBackend *groq = [self stubFor:NFKRemoteProvider.groq model:@"whisper-large-v3"];
	groq.stagedData = [@"{\"text\":\"ok\"}" dataUsingEncoding:NSUTF8StringEncoding];
	NFKAudioAsset *hosted = [NFKAudioAsset audioAssetWithFileURL:[NSURL URLWithString:@"https://example.com/talk.mp3"]];
	[groq runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputAudio: hosted }] error:NULL];
	NSString *form = [self formOf:groq.lastRequest];
	XCTAssertTrue([form containsString:@"name=\"url\"\r\n\r\nhttps://example.com/talk.mp3"]);
	XCTAssertFalse([form containsString:@"name=\"file\""]);

	StubTranscriptionBackend *openAI = [self stubFor:NFKRemoteProvider.openAI model:@"whisper-1"];
	openAI.stagedData = [@"{\"text\":\"ok\"}" dataUsingEncoding:NSUTF8StringEncoding];
	XCTAssertNotNil([openAI runInferenceForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputAudio: hosted }] error:NULL]);
	XCTAssertTrue([[self formOf:openAI.lastRequest] containsString:@"name=\"file\"; filename=\"talk.mp3\""]);
}

// A streamed transcription grows the partial result by its deltas and ends with the done event's text.
- (void)testAStreamedTranscriptionReportsDeltasAndFinishesWithTheDoneText
{
	StubTranscriptionBackend *openAI = [self stubFor:NFKRemoteProvider.openAI model:@"gpt-transcribe"];
	openAI.streams = YES;
	openAI.stagedLines = @[ @"data: {\"type\":\"transcript.text.delta\",\"delta\":\"Hel\"}",
							@"data: {\"type\":\"transcript.text.delta\",\"delta\":\"lo\"}",
							@"data: {\"type\":\"transcript.text.done\",\"text\":\"Hello.\"}" ];
	NFKInferenceJob *job = [openAI submitInferenceJobForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputAudio: [@"RIFF" dataUsingEncoding:NSUTF8StringEncoding] }]];
	XCTAssertEqual(job.status, NFKInferenceJobStatusSucceeded);
	XCTAssertEqualObjects([job.result outputForKey:NFKOutputText], @"Hello.");
	XCTAssertTrue([[self formOf:openAI.lastRequest] containsString:@"name=\"stream\"\r\n\r\ntrue"]);
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
