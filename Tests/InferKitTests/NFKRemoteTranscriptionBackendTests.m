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
#import <InferKit/NFKRemoteTransport.h>

/*! A transcription backend whose transport is stubbed: it records the request and returns staged data. */
@interface StubTranscriptionBackend : NFKRemoteTranscriptionBackend
@property (nonatomic, strong) NSURLRequest *lastRequest;
@property (nonatomic, strong, nullable) NSData *stagedData;
@property (nonatomic, assign) NSInteger stagedStatusCode;
@property (nonatomic, copy, nullable) NSArray<NSString *> *stagedLines;
/*! Holds the staged lines until playDeferredStream, so a test sets a progress handler on the job first. */
@property (nonatomic, assign) BOOL defersStream;
@property (nonatomic, copy, nullable) void (^deferredStream)(void);
- (void)playDeferredStream;
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
	NSArray<NSString *> *lines = self.stagedLines;
	void (^stream)(void) = ^{
		for (NSString *line in lines) {
			lineHandler(line);
		}
		completionHandler([[NSHTTPURLResponse alloc] initWithURL:request.URL statusCode:200 HTTPVersion:@"HTTP/1.1" headerFields:nil], nil, nil);
	};
	if (self.defersStream) {
		self.deferredStream = stream;
	} else {
		stream();
	}
	return ^{};
}

- (void)playDeferredStream
{
	void (^stream)(void) = self.deferredStream;
	self.deferredStream = nil;
	if (stream != nil) {
		stream();
	}
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
	NFKRemoteTranscriptionBackend *meta = [NFKRemoteTranscriptionBackend backendForProvider:NFKRemoteProvider.meta apiKey:@"k" modelName:@"m"];
	XCTAssertEqualObjects(meta.endpointURL.absoluteString, @"https://api.meta.ai/v1/asr/transcribe");
	XCTAssertEqual(meta.apiStyle, NFKRemoteTranscriptionAPIStyleMeta);
	for (NFKRemoteProvider *provider in @[ NFKRemoteProvider.googleGemini, NFKRemoteProvider.deepSeek, NFKRemoteProvider.anthropic,
											NFKRemoteProvider.ollama, NFKRemoteProvider.llamaCpp ]) {
		XCTAssertNil([NFKRemoteTranscriptionBackend backendForProvider:provider apiKey:@"k" modelName:@"m"], @"%@", provider.identifier);
	}
}

// Meta's /v1/asr/transcribe takes its settings as one JSON part and answers with turns in milliseconds.
- (void)testMetaSendsAJSONRequestPartAndReadsTheTurns
{
	StubTranscriptionBackend *meta = [self stubFor:NFKRemoteProvider.meta model:@"muse-voice-transcribe-1.0"];
	meta.stagedData = [(@"{\"sessionId\":\"s1\",\"transcript\":\"Hello. Hi there.\",\"audioDurationMs\":2000,\"turns\":["
						 "{\"turnId\":\"t1\",\"startMs\":0,\"endMs\":800,\"transcript\":\"Hello.\",\"speaker\":\"A\"},"
						 "{\"turnId\":\"t2\",\"startMs\":1000,\"endMs\":1900,\"transcript\":\"Hi there.\",\"speaker\":\"B\"}]}")
					   dataUsingEncoding:NSUTF8StringEncoding];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputAudio: [@"RIFF" dataUsingEncoding:NSUTF8StringEncoding],
																			 NFKInputPrompt: @"a prompt the service has no field for" }
															   parameters:@{ NFKParameterSpeakerDiarization: @YES,
																			 NFKParameterVocabulary: @[ @"InferKit" ],
																			 NFKParameterSourceLanguage: @"English",
																			 @"partialMode": @"DELTA" }];
	NSError *error = nil;
	NFKInferenceResult *result = [meta runInferenceForRequest:request error:&error];
	XCTAssertNotNil(result, @"%@", error);
	XCTAssertEqualObjects(meta.lastRequest.allHTTPHeaderFields[@"Authorization"], @"Bearer k");
	NSString *form = [self formOf:meta.lastRequest];
	NSString *settings = @"{\"audioEncoding\":\"WAV\",\"keywords\":[\"InferKit\"],\"languageBias\":[\"English\"],"
		"\"mode\":\"DIARIZATION\",\"model\":\"muse-voice-transcribe-1.0\",\"partialMode\":\"DELTA\"}";
	NSString *requestPart = [@"name=\"request\"\r\nContent-Type: application/json\r\n\r\n" stringByAppendingString:settings];
	XCTAssertTrue([form containsString:requestPart], @"%@", form);
	XCTAssertTrue([form containsString:@"name=\"audio\"; filename=\"audio.wav\"\r\nContent-Type: audio/wav\r\n\r\nRIFF"]);
	XCTAssertFalse([form containsString:@"name=\"file\""]);
	XCTAssertFalse([form containsString:@"a prompt"]);
	XCTAssertGreaterThan([form rangeOfString:@"name=\"audio\""].location, [form rangeOfString:@"name=\"request\""].location);

	XCTAssertEqualObjects(result.text, @"Hello. Hi there.");
	NSArray<NFKAudioSegment *> *turns = [result outputForKey:NFKOutputSegments];
	XCTAssertEqual(turns.count, 2);
	XCTAssertEqualObjects(turns[1].label, @"Hi there.");
	XCTAssertEqualObjects(turns[1].speaker, @"B");
	XCTAssertEqualWithAccuracy(turns[1].startSeconds, 1.0, 1e-9);
	XCTAssertEqualWithAccuracy(turns[1].endSeconds, 1.9, 1e-9);
	XCTAssertEqualObjects(result.structured[@"sessionId"], @"s1");
}

- (void)testMetaAsksForTurnsWhenTimestampsAreWantedAndAStreamedJobAsksForTheEventStream
{
	StubTranscriptionBackend *meta = [self stubFor:NFKRemoteProvider.meta model:@"muse-voice-transcribe-1.0"];
	meta.stagedData = [@"{\"sessionId\":\"s2\",\"transcript\":\"Hello.\",\"turns\":[]}" dataUsingEncoding:NSUTF8StringEncoding];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputAudio: [@"RIFF" dataUsingEncoding:NSUTF8StringEncoding] }];
	XCTAssertNotNil([meta runInferenceForRequest:request error:NULL]);
	XCTAssertFalse([[self formOf:meta.lastRequest] containsString:@"\"mode\""], @"the service's default is push to talk");
	XCTAssertNil(meta.lastRequest.allHTTPHeaderFields[@"Accept"], @"the blocking call takes the buffered reply");

	meta.emitsTimestamps = YES;
	meta.streams = YES;
	meta.stagedLines = [self metaStreamLinesForEvents:@[
		@"{\"type\":\"speechStart\",\"turnId\":1,\"audioProcessedMs\":1200}",
		@"{\"type\":\"transcript\",\"transcript\":\"how is the weather\",\"final\":false,\"audioProcessedMs\":3200}",
		@"{\"type\":\"speechEnd\",\"turnId\":1,\"audioProcessedMs\":3600}",
		@"{\"type\":\"speechComplete\",\"turnId\":1,\"transcript\":\"How is the weather?\",\"audioProcessedMs\":3600}" ]];
	NFKInferenceJob *job = [meta submitInferenceJobForRequest:request];
	XCTAssertEqual(job.status, NFKInferenceJobStatusSucceeded, @"%@", job.error);
	NSString *form = [self formOf:meta.lastRequest];
	XCTAssertTrue([form containsString:@"\"mode\":\"ENDPOINTING\""]);
	XCTAssertFalse([form containsString:@"stream"], @"the Accept header alone selects the stream");
	XCTAssertEqualObjects(meta.lastRequest.allHTTPHeaderFields[@"Accept"], @"text/event-stream");
	XCTAssertEqualObjects(job.result.text, @"How is the weather?");
	NSArray<NFKAudioSegment *> *turns = job.result.segments;
	XCTAssertEqual(turns.count, 1);
	XCTAssertEqualObjects(turns[0].label, @"How is the weather?");
	XCTAssertNil(turns[0].speaker);
	XCTAssertEqualWithAccuracy(turns[0].startSeconds, 1.2, 1e-9);
	XCTAssertEqualWithAccuracy(turns[0].endSeconds, 3.6, 1e-9);
}

#pragma mark Meta's event stream

// Meta's SSE framing: a line naming the event, the data line, and the blank line that ends the event.
- (NSArray<NSString *> *)metaStreamLinesForEvents:(NSArray<NSString *> *)events
{
	NSMutableArray<NSString *> *lines = [NSMutableArray array];
	for (NSString *event in events) {
		NSDictionary *object = [NSJSONSerialization JSONObjectWithData:[event dataUsingEncoding:NSUTF8StringEncoding] options:0 error:NULL];
		[lines addObject:[@"event: " stringByAppendingString:object[@"type"]]];
		[lines addObject:[@"data: " stringByAppendingString:event]];
		[lines addObject:@""];
	}
	return lines;
}

/*! Submits the request on a Meta stub that streams the events, and returns the job finished. texts
	receives the text of each partial result the job reported, in order. */
- (NFKInferenceJob *)metaJobStreamingEvents:(NSArray<NSString *> *)events
								parameters:(NSDictionary<NSString *, id> *)parameters
							  partialTexts:(NSMutableArray<NSString *> *)texts
{
	StubTranscriptionBackend *meta = [self stubFor:NFKRemoteProvider.meta model:@"muse-voice-transcribe-1.0"];
	meta.streams = YES;
	meta.defersStream = YES;
	meta.stagedLines = [self metaStreamLinesForEvents:events];
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputAudio: [@"RIFF" dataUsingEncoding:NSUTF8StringEncoding] }
															   parameters:parameters];
	NFKInferenceJob *job = [meta submitInferenceJobForRequest:request];
	job.progressHandler = ^(NFKInferenceJob *running) {
		if (running.partialResult.text != nil) {
			[texts addObject:running.partialResult.text];
		}
	};
	[meta playDeferredStream];
	self.backend = meta;
	return job;
}

- (void)testMetaStreamsAPushToTalkTranscriptWhoseEachPartialReplacesTheLast
{
	NSMutableArray<NSString *> *partials = [NSMutableArray array];
	NFKInferenceJob *job = [self metaJobStreamingEvents:@[
		@"{\"type\":\"transcript\",\"transcript\":\"i fixed the bag\",\"final\":false,\"audioProcessedMs\":800}",
		@"{\"type\":\"audioProgress\",\"audioProcessedMs\":900}",
		@"{\"type\":\"transcript\",\"transcript\":\"I fixed the bug. By fixed\",\"final\":false,\"audioProcessedMs\":1600}",
		@"{\"type\":\"aLaterEventType\",\"audioProcessedMs\":1700}",
		@"{\"type\":\"transcript\",\"transcript\":\"I fixed the bug. By fixed, I mean it.\",\"final\":true,\"audioProcessedMs\":2400}" ]
											parameters:@{}
										  partialTexts:partials];
	XCTAssertEqual(job.status, NFKInferenceJobStatusSucceeded, @"%@", job.error);
	NSArray<NSString *> *expected = @[ @"i fixed the bag", @"I fixed the bug. By fixed", @"I fixed the bug. By fixed, I mean it." ];
	XCTAssertEqualObjects(partials, expected);
	XCTAssertEqualObjects(job.result.text, @"I fixed the bug. By fixed, I mean it.");
	XCTAssertNil(job.result.segments, @"push to talk reports no turns");
	XCTAssertEqualObjects(job.result.structured[@"turns"], (@[]));
	XCTAssertFalse([[self formOf:self.backend.lastRequest] containsString:@"\"mode\""]);
}

// A later turn opens before the earlier one's speechComplete arrives, so the completion finds its turn
// by id. The speaker event labels the turn the latest speechStart opened.
- (void)testMetaStreamsADiarizedTranscriptAsTurnsWithTheirSpeakers
{
	NSMutableArray<NSString *> *partials = [NSMutableArray array];
	NFKInferenceJob *job = [self metaJobStreamingEvents:@[
		@"{\"type\":\"speechStart\",\"turnId\":1,\"audioProcessedMs\":1200}",
		@"{\"type\":\"transcript\",\"transcript\":\"thanks for calling\",\"final\":false,\"audioProcessedMs\":2400}",
		@"{\"type\":\"speaker\",\"label\":\"A\",\"audioProcessedMs\":2480}",
		@"{\"type\":\"speechEnd\",\"turnId\":1,\"audioProcessedMs\":3600}",
		@"{\"type\":\"speechStart\",\"turnId\":2,\"audioProcessedMs\":4100}",
		@"{\"type\":\"transcript\",\"transcript\":\"i need a new sim\",\"final\":false,\"audioProcessedMs\":5200}",
		@"{\"type\":\"speechComplete\",\"turnId\":1,\"transcript\":\"Thanks for calling.\",\"audioProcessedMs\":5300}",
		@"{\"type\":\"speaker\",\"label\":\"B\",\"audioProcessedMs\":5400}",
		@"{\"type\":\"speechEnd\",\"turnId\":2,\"audioProcessedMs\":6000}",
		@"{\"type\":\"speechComplete\",\"turnId\":2,\"transcript\":\"I need a new SIM.\",\"audioProcessedMs\":6000}" ]
											parameters:@{ NFKParameterSpeakerDiarization: @YES }
										  partialTexts:partials];
	XCTAssertEqual(job.status, NFKInferenceJobStatusSucceeded, @"%@", job.error);
	XCTAssertTrue([[self formOf:self.backend.lastRequest] containsString:@"\"mode\":\"DIARIZATION\""]);
	NSArray<NSString *> *expected = @[ @"thanks for calling", @"thanks for calling i need a new sim",
									   @"Thanks for calling. i need a new sim", @"Thanks for calling. I need a new SIM." ];
	XCTAssertEqualObjects(partials, expected);
	XCTAssertEqualObjects(job.result.text, @"Thanks for calling. I need a new SIM.");
	NSArray<NFKAudioSegment *> *turns = job.result.segments;
	XCTAssertEqual(turns.count, 2);
	XCTAssertEqualObjects(turns[0].label, @"Thanks for calling.");
	XCTAssertEqualObjects(turns[0].speaker, @"A");
	XCTAssertEqualWithAccuracy(turns[0].startSeconds, 1.2, 1e-9);
	XCTAssertEqualWithAccuracy(turns[0].endSeconds, 3.6, 1e-9);
	XCTAssertEqualObjects(turns[1].label, @"I need a new SIM.");
	XCTAssertEqualObjects(turns[1].speaker, @"B");
	XCTAssertEqualWithAccuracy(turns[1].startSeconds, 4.1, 1e-9);
	XCTAssertEqualWithAccuracy(turns[1].endSeconds, 6.0, 1e-9);
	NSDictionary *secondTurn = @{ @"turnId": @2, @"startMs": @4100, @"endMs": @6000, @"transcript": @"I need a new SIM.", @"speaker": @"B" };
	XCTAssertEqualObjects([job.result.structured[@"turns"] lastObject], secondTurn);
}

// Under DELTA a partial carries new text only; the final event carries the complete transcript.
- (void)testMetaAppendsDeltaPartialsAndTakesTheFinalTranscriptWhole
{
	NSMutableArray<NSString *> *partials = [NSMutableArray array];
	NFKInferenceJob *job = [self metaJobStreamingEvents:@[
		@"{\"type\":\"transcript\",\"transcript\":\"I fixed\",\"final\":false,\"audioProcessedMs\":800}",
		@"{\"type\":\"transcript\",\"transcript\":\" the bug\",\"final\":false,\"audioProcessedMs\":1600}",
		@"{\"type\":\"transcript\",\"transcript\":\"I fixed the bug.\",\"final\":true,\"audioProcessedMs\":2000}" ]
											parameters:@{ @"partialMode": @"DELTA" }
										  partialTexts:partials];
	XCTAssertEqual(job.status, NFKInferenceJobStatusSucceeded, @"%@", job.error);
	XCTAssertTrue([[self formOf:self.backend.lastRequest] containsString:@"\"partialMode\":\"DELTA\""]);
	NSArray<NSString *> *expected = @[ @"I fixed", @"I fixed the bug", @"I fixed the bug." ];
	XCTAssertEqualObjects(partials, expected);
	XCTAssertEqualObjects(job.result.text, @"I fixed the bug.");
}

- (void)testMetaReadsTheEventLinesNameForADataObjectWithoutAType
{
	StubTranscriptionBackend *meta = [self stubFor:NFKRemoteProvider.meta model:@"muse-voice-transcribe-1.0"];
	meta.streams = YES;
	meta.stagedLines = @[ @"event: transcript", @"data: {\"transcript\":\"Hello.\",\"final\":true,\"audioProcessedMs\":900}", @"",
						  @"data: {\"transcript\":\"an object no event line names\",\"final\":true}", @"" ];
	NFKInferenceJob *job = [meta submitInferenceJobForRequest:[NFKInferenceRequest requestWithInputs:@{ NFKInputAudio: [@"RIFF" dataUsingEncoding:NSUTF8StringEncoding] }]];
	XCTAssertEqualObjects(job.result.text, @"Hello.", @"%@", job.error);
}

// Once the stream is open the status is 200, so a failure arrives as an event.
- (void)testMetaErrorEventFailsAStreamedJobWithTheServicesMessage
{
	NSMutableArray<NSString *> *partials = [NSMutableArray array];
	NFKInferenceJob *job = [self metaJobStreamingEvents:@[
		@"{\"type\":\"transcript\",\"transcript\":\"how is\",\"final\":false,\"audioProcessedMs\":800}",
		@"{\"type\":\"error\",\"message\":\"Processing exceeded the server budget.\",\"sessionId\":\"s-9\",\"errorType\":\"server_error\"}",
		@"{\"type\":\"transcript\",\"transcript\":\"how is the weather\",\"final\":true,\"audioProcessedMs\":1600}" ]
											parameters:@{}
										  partialTexts:partials];
	XCTAssertEqual(job.status, NFKInferenceJobStatusFailed);
	XCTAssertNil(job.result);
	XCTAssertEqualObjects(job.error.domain, NFKInferenceErrorDomain);
	XCTAssertEqual(job.error.code, (NSInteger)kNFKError_InferenceBackendFailure);
	XCTAssertEqualObjects(job.error.localizedDescription, @"Processing exceeded the server budget.");
	XCTAssertTrue([job.error.userInfo[NFKRemoteErrorBodyKey] containsString:@"\"sessionId\":\"s-9\""]);
	XCTAssertEqualObjects(partials, (@[ @"how is" ]), @"an event after the error is not read");

	job = [self metaJobStreamingEvents:@[ @"{\"type\":\"error\",\"message\":\"Too many sessions.\",\"sessionId\":\"\",\"errorType\":\"rate_limit_error\"}" ]
							parameters:@{}
						  partialTexts:partials];
	XCTAssertEqual(job.error.code, (NSInteger)kNFKError_InferenceRateLimited);
	job = [self metaJobStreamingEvents:@[ @"{\"type\":\"error\",\"message\":\"Unsupported audio.\",\"sessionId\":\"\",\"errorType\":\"invalid_request_error\",\"errorParam\":\"audio\"}" ]
							parameters:@{}
						  partialTexts:partials];
	XCTAssertEqual(job.error.code, (NSInteger)kNFKError_InferenceRefused);
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
