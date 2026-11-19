//
//  NFKRealtimeSessionTests.m
//  InferKitTests
//
//  Realtime sessions against a scripted socket: each style's handshake URL and key, its
//  configuration message, the events the calls send, and what the service's events become on the
//  handlers.
//

#import <XCTest/XCTest.h>
#import <InferKit/NFKRealtimeSession.h>
#import <InferKit/NFKRemoteProvider.h>

/*! A socket that records what is sent and hands the session whatever the test pushes. */
@interface NFKScriptedSocket : NSObject <NFKRealtimeSocket>
@property (nonatomic, strong) NSURLRequest *request;
@property (nonatomic, strong) NSMutableArray<NSDictionary *> *sentEvents;
@property (nonatomic, strong) NSMutableArray<NSData *> *sentData;
/*! Every frame in the order it was sent: an event's dictionary for a text frame, the data for a binary one. */
@property (nonatomic, strong) NSMutableArray *sentFrames;
@property (nonatomic, copy) void (^messageHandler)(NSString *, NSData *);
@property (nonatomic, assign) BOOL closed;
@end

@implementation NFKScriptedSocket
- (void)openWithMessageHandler:(void (^)(NSString *, NSData *))messageHandler closeHandler:(void (^)(NSError *))closeHandler
{
	self.messageHandler = messageHandler;
	self.sentEvents = [NSMutableArray array];
	self.sentData = [NSMutableArray array];
	self.sentFrames = [NSMutableArray array];
}
- (void)sendText:(NSString *)text
{
	NSDictionary *event = [NSJSONSerialization JSONObjectWithData:[text dataUsingEncoding:NSUTF8StringEncoding] options:0 error:NULL];
	[self.sentEvents addObject:event];
	[self.sentFrames addObject:event];
}
- (void)sendData:(NSData *)data
{
	[self.sentData addObject:data];
	[self.sentFrames addObject:data];
}
- (void)close
{
	self.closed = YES;
}
- (void)push:(NSString *)json
{
	self.messageHandler(json, nil);
}
@end

@interface NFKScriptedSession : NFKRealtimeSession
@property (nonatomic, strong) NFKScriptedSocket *scripted;
@end

@implementation NFKScriptedSession
- (id<NFKRealtimeSocket>)socketForRequest:(NSURLRequest *)request
{
	self.scripted = [[NFKScriptedSocket alloc] init];
	self.scripted.request = request;
	return self.scripted;
}
@end

@interface NFKRealtimeWebSocket (NFKRealtimeSessionTests)
- (nullable NSError *)errorForCloseCode:(NSURLSessionWebSocketCloseCode)closeCode reason:(nullable NSData *)reason receiveError:(NSError *)error;
@end

@interface NFKRealtimeSessionTests : XCTestCase
@end

@implementation NFKRealtimeSessionTests

- (NFKScriptedSession *)sessionFor:(NFKRemoteProvider *)provider style:(NFKRealtimeAPIStyle)style model:(NSString *)model
{
	NFKRealtimeSession *made = [NFKRealtimeSession sessionForProvider:provider apiStyle:style apiKey:@"k" modelName:model];
	NFKScriptedSession *session = [[NFKScriptedSession alloc] init];
	session.apiStyle = made.apiStyle;
	session.endpointURL = made.endpointURL;
	session.apiKey = @"k";
	session.modelName = model;
	return session;
}

- (NSString *)base64:(NSString *)text
{
	return [[text dataUsingEncoding:NSUTF8StringEncoding] base64EncodedStringWithOptions:0];
}

- (void)testEachProviderGetsItsSocketAndTheOthersNone
{
	XCTAssertEqualObjects([NFKRealtimeSession sessionForProvider:NFKRemoteProvider.openAI apiStyle:NFKRealtimeAPIStyleOpenAITranslation apiKey:@"k" modelName:@"m"].endpointURL.absoluteString,
						  @"wss://api.openai.com/v1/realtime/translations");
	XCTAssertEqualObjects([NFKRealtimeSession sessionForProvider:NFKRemoteProvider.xAI apiStyle:NFKRealtimeAPIStyleXAITranscription apiKey:@"k" modelName:@"m"].endpointURL.absoluteString,
						  @"wss://api.x.ai/v1/stt");
	XCTAssertEqualObjects([NFKRealtimeSession sessionForProvider:NFKRemoteProvider.vLLM apiStyle:NFKRealtimeAPIStyleVLLMTranscription apiKey:nil modelName:@"m"].endpointURL.absoluteString,
						  @"ws://localhost:8000/v1/realtime");
	XCTAssertTrue([[NFKRealtimeSession sessionForProvider:NFKRemoteProvider.googleGemini apiStyle:NFKRealtimeAPIStyleGeminiLive apiKey:@"k" modelName:@"m"].endpointURL.absoluteString
				   hasSuffix:@"GenerativeService.BidiGenerateContent"]);
	XCTAssertEqualObjects([NFKRealtimeSession sessionForProvider:NFKRemoteProvider.meta apiStyle:NFKRealtimeAPIStyleMetaTranscription apiKey:@"k" modelName:@"m"].endpointURL.absoluteString,
						  @"wss://api.meta.ai/v1/asr/realtime");
	XCTAssertNil([NFKRealtimeSession sessionForProvider:NFKRemoteProvider.openAI apiStyle:NFKRealtimeAPIStyleMetaTranscription apiKey:@"k" modelName:@"m"]);
	XCTAssertNil([NFKRealtimeSession sessionForProvider:NFKRemoteProvider.meta apiStyle:NFKRealtimeAPIStyleOpenAITranscription apiKey:@"k" modelName:@"m"]);
	XCTAssertNil([NFKRealtimeSession sessionForProvider:NFKRemoteProvider.groq apiStyle:NFKRealtimeAPIStyleOpenAIConversation apiKey:@"k" modelName:@"m"]);
	XCTAssertNil([NFKRealtimeSession sessionForProvider:NFKRemoteProvider.openAI apiStyle:NFKRealtimeAPIStyleGeminiLive apiKey:@"k" modelName:@"m"]);
}

// OpenAI Realtime: session.update on connect, audio and text in, audio, transcripts, and a tool call out.
- (void)testAnOpenAIConversationConfiguresItselfAndReportsAudioTranscriptsAndToolCalls
{
	NFKScriptedSession *session = [self sessionFor:NFKRemoteProvider.openAI style:NFKRealtimeAPIStyleOpenAIConversation model:@"gpt-realtime-2.1"];
	session.instructions = @"Be warm.";
	session.voice = @"marin";
	session.tools = @[ @{ @"name": @"get_time", @"parameters": @{ @"type": @"object" } } ];
	NSMutableData *audio = [NSMutableData data];
	NSMutableArray<NSString *> *transcripts = [NSMutableArray array];
	__block NSDictionary *call = nil;
	__block NSUInteger turns = 0;
	session.audioHandler = ^(NSData *pcm) { [audio appendData:pcm]; };
	session.textHandler = ^(NSString *text, NFKRealtimeTextKind kind) {
		if (kind == NFKRealtimeTextKindOutputTranscript) { [transcripts addObject:text]; }
	};
	session.toolCallHandler = ^(NSDictionary *toolCall) { call = toolCall; };
	session.turnHandler = ^{ turns++; };
	[session connect];

	NSURLRequest *handshake = session.scripted.request;
	XCTAssertEqualObjects(handshake.URL.absoluteString, @"wss://api.openai.com/v1/realtime?model=gpt-realtime-2.1");
	XCTAssertEqualObjects([handshake valueForHTTPHeaderField:@"Authorization"], @"Bearer k");
	NSDictionary *update = session.scripted.sentEvents.firstObject;
	XCTAssertEqualObjects(update[@"type"], @"session.update");
	XCTAssertEqualObjects(update[@"session"][@"instructions"], @"Be warm.");
	XCTAssertEqualObjects(update[@"session"][@"audio"][@"output"][@"voice"], @"marin");
	XCTAssertEqualObjects(update[@"session"][@"audio"][@"input"][@"format"], (@{ @"type": @"audio/pcm", @"rate": @24000 }));
	XCTAssertEqualObjects(update[@"session"][@"tools"][0][@"type"], @"function");

	[session appendAudio:[@"pcm" dataUsingEncoding:NSUTF8StringEncoding]];
	[session commitAudio];
	[session requestResponse];
	XCTAssertEqualObjects(session.scripted.sentEvents[1], (@{ @"type": @"input_audio_buffer.append", @"audio": [self base64:@"pcm"] }));
	XCTAssertEqualObjects(session.scripted.sentEvents[2][@"type"], @"input_audio_buffer.commit");
	XCTAssertEqualObjects(session.scripted.sentEvents[3][@"type"], @"response.create");

	[session.scripted push:[NSString stringWithFormat:@"{\"type\":\"response.output_audio.delta\",\"delta\":\"%@\"}", [self base64:@"ab"]]];
	[session.scripted push:@"{\"type\":\"response.output_audio_transcript.delta\",\"delta\":\"Hi\"}"];
	[session.scripted push:@"{\"type\":\"response.function_call_arguments.done\",\"call_id\":\"c1\",\"name\":\"get_time\",\"arguments\":\"{\\\"zone\\\":\\\"UTC\\\"}\"}"];
	[session.scripted push:@"{\"type\":\"response.done\"}"];
	XCTAssertEqualObjects(audio, [@"ab" dataUsingEncoding:NSUTF8StringEncoding]);
	XCTAssertEqualObjects(transcripts, @[ @"Hi" ]);
	XCTAssertEqualObjects(call, (@{ @"id": @"c1", @"name": @"get_time", @"arguments": @{ @"zone": @"UTC" } }));
	XCTAssertEqual(turns, 1);

	[session sendToolResult:@"12:00" forCallIdentifier:@"c1" name:@"get_time"];
	XCTAssertEqualObjects(session.scripted.sentEvents.lastObject[@"item"], (@{ @"type": @"function_call_output", @"call_id": @"c1", @"output": @"12:00" }));
}

// Gemini Live: the key in the query, a setup message, realtimeInput audio, and serverContent back.
- (void)testGeminiLiveSetsUpAndReadsServerContent
{
	NFKScriptedSession *session = [self sessionFor:NFKRemoteProvider.googleGemini style:NFKRealtimeAPIStyleGeminiLive model:@"gemini-3.8-live"];
	session.voice = @"Puck";
	session.instructions = @"Be brief.";
	NSMutableData *audio = [NSMutableData data];
	NSMutableArray *texts = [NSMutableArray array];
	__block BOOL turned = NO;
	session.audioHandler = ^(NSData *pcm) { [audio appendData:pcm]; };
	session.textHandler = ^(NSString *text, NFKRealtimeTextKind kind) { [texts addObject:@[ text, @(kind) ]]; };
	session.turnHandler = ^{ turned = YES; };
	[session connect];
	XCTAssertTrue([session.scripted.request.URL.absoluteString hasSuffix:@"BidiGenerateContent?key=k"]);
	XCTAssertNil([session.scripted.request valueForHTTPHeaderField:@"Authorization"]);
	NSDictionary *setup = session.scripted.sentEvents.firstObject[@"setup"];
	XCTAssertEqualObjects(setup[@"model"], @"models/gemini-3.8-live");
	XCTAssertEqualObjects(setup[@"generationConfig"][@"speechConfig"][@"voiceConfig"][@"prebuiltVoiceConfig"][@"voiceName"], @"Puck");
	XCTAssertEqualObjects(setup[@"systemInstruction"][@"parts"][0][@"text"], @"Be brief.");

	[session appendAudio:[@"pcm" dataUsingEncoding:NSUTF8StringEncoding]];
	XCTAssertEqualObjects(session.scripted.sentEvents[1][@"realtimeInput"][@"audio"][@"mimeType"], @"audio/pcm;rate=16000");

	[session.scripted push:[NSString stringWithFormat:@"{\"serverContent\":{\"modelTurn\":{\"parts\":[{\"inlineData\":{\"mimeType\":\"audio/pcm;rate=24000\",\"data\":\"%@\"}}]},"
							"\"inputTranscription\":{\"text\":\"hello\"},\"outputTranscription\":{\"text\":\"hi there\"},\"turnComplete\":true}}", [self base64:@"xy"]]];
	XCTAssertEqualObjects(audio, [@"xy" dataUsingEncoding:NSUTF8StringEncoding]);
	XCTAssertEqualObjects(texts, (@[ @[ @"hello", @(NFKRealtimeTextKindInputTranscript) ], @[ @"hi there", @(NFKRealtimeTextKindOutputTranscript) ] ]));
	XCTAssertTrue(turned);
}

// gemini-3.8-live-extended-thinking: the spoken filler's turnComplete arrives while the interaction is
// still IN_PROGRESS, and the turn ends when it is IDLE.
- (void)testGeminiLiveWaitsForAThinkingInteractionToGoIdle
{
	NFKScriptedSession *session = [self sessionFor:NFKRemoteProvider.googleGemini style:NFKRealtimeAPIStyleGeminiLive model:@"gemini-3.8-live-extended-thinking"];
	session.extraConfiguration = @{ @"generationConfig": @{ @"thinkingConfig": @{ @"thinkingLevel": @"LOW", @"includeThoughts": @YES } } };
	NSMutableArray *texts = [NSMutableArray array];
	__block NSUInteger turns = 0;
	session.textHandler = ^(NSString *text, NFKRealtimeTextKind kind) { [texts addObject:@[ text, @(kind) ]]; };
	session.turnHandler = ^{ turns += 1; };
	[session connect];
	NSDictionary *generation = session.scripted.sentEvents.firstObject[@"setup"][@"generationConfig"];
	XCTAssertEqualObjects(generation[@"responseModalities"], @[ @"AUDIO" ]);
	XCTAssertEqualObjects(generation[@"thinkingConfig"][@"thinkingLevel"], @"LOW");

	[session.scripted push:@"{\"serverContent\":{\"modelTurn\":{\"parts\":[{\"text\":\"checking flights\",\"thought\":true}]},"
							"\"turnComplete\":true,\"interactionStatus\":\"IN_PROGRESS\"}}"];
	[session.scripted push:@"{\"toolCall\":{\"functionCalls\":[{\"id\":\"call_123\",\"name\":\"searchFlights\",\"args\":{}}]},\"interactionStatus\":\"IN_PROGRESS\"}"];
	XCTAssertEqual(turns, 0u);
	[session.scripted push:@"{\"serverContent\":{\"turnComplete\":true,\"interactionStatus\":\"IDLE\"}}"];
	XCTAssertEqual(turns, 1u);
	XCTAssertEqualObjects(texts, (@[ @[ @"checking flights", @(NFKRealtimeTextKindReasoning) ] ]));
}

- (void)testGeminiLiveReportsTheTurnWhenIdleArrivesAloneOrBesideAToolCall
{
	NFKScriptedSession *session = [self sessionFor:NFKRemoteProvider.googleGemini style:NFKRealtimeAPIStyleGeminiLive model:@"gemini-3.8-live-extended-thinking"];
	__block NSUInteger turns = 0;
	__block NSUInteger calls = 0;
	session.turnHandler = ^{ turns += 1; };
	session.toolCallHandler = ^(NSDictionary *call) { calls += 1; };
	[session connect];
	[session.scripted push:@"{\"serverContent\":{\"turnComplete\":true,\"interactionStatus\":\"IN_PROGRESS\"}}"];
	[session.scripted push:@"{\"interactionStatus\":\"IDLE\"}"];
	XCTAssertEqual(turns, 1u);
	[session.scripted push:@"{\"interactionStatus\":\"IDLE\"}"];
	XCTAssertEqual(turns, 1u);

	[session.scripted push:@"{\"serverContent\":{\"turnComplete\":true,\"interactionStatus\":\"IN_PROGRESS\"}}"];
	[session.scripted push:@"{\"toolCall\":{\"functionCalls\":[{\"id\":\"c1\",\"name\":\"f\",\"args\":{}}]},\"interactionStatus\":\"IDLE\"}"];
	XCTAssertEqual(calls, 1u);
	XCTAssertEqual(turns, 2u);
}

- (void)testAReconnectedGeminiLiveSessionDoesNotFinishTheLastConnectionsTurn
{
	NFKScriptedSession *session = [self sessionFor:NFKRemoteProvider.googleGemini style:NFKRealtimeAPIStyleGeminiLive model:@"gemini-3.8-live-extended-thinking"];
	__block NSUInteger turns = 0;
	session.turnHandler = ^{ turns += 1; };
	[session connect];
	[session.scripted push:@"{\"serverContent\":{\"turnComplete\":true,\"interactionStatus\":\"IN_PROGRESS\"}}"];
	[session close];
	[session connect];
	[session.scripted push:@"{\"interactionStatus\":\"IDLE\"}"];
	XCTAssertEqual(turns, 0u);
}

// gemini-3.5-transcribe-live answers in text: interim pieces, then finished ones.
- (void)testGeminiLiveTranscriptionAsksForTextAndReportsInterimAndFinalPieces
{
	NFKScriptedSession *session = [self sessionFor:NFKRemoteProvider.googleGemini style:NFKRealtimeAPIStyleGeminiLive model:@"gemini-3.5-transcribe-live"];
	session.language = @"en-US";
	NSMutableArray *texts = [NSMutableArray array];
	session.textHandler = ^(NSString *text, NFKRealtimeTextKind kind) { [texts addObject:@[ text, @(kind) ]]; };
	[session connect];
	NSDictionary *setup = session.scripted.sentEvents.firstObject[@"setup"];
	XCTAssertEqualObjects(setup[@"generationConfig"][@"responseModalities"], @[ @"TEXT" ]);
	XCTAssertEqualObjects(setup[@"inputAudioTranscription"][@"languageCodes"], @[ @"en-US" ]);
	XCTAssertNil(setup[@"outputAudioTranscription"]);

	[session.scripted push:@"{\"serverContent\":{\"interimInputTranscription\":{\"text\":\"hel\"}}}"];
	[session.scripted push:@"{\"serverContent\":{\"inputTranscription\":{\"text\":\"hello there\"}}}"];
	XCTAssertEqualObjects(texts, (@[ @[ @"hel", @(NFKRealtimeTextKindInputTranscript) ],
									 @[ @"hello there", @(NFKRealtimeTextKindInputTranscriptFinal) ] ]));
}

// gemini-3.5-live-translate-preview speaks the target language and takes no instructions or tools.
- (void)testGeminiLiveTranslationNamesItsTargetLanguage
{
	NFKScriptedSession *session = [self sessionFor:NFKRemoteProvider.googleGemini style:NFKRealtimeAPIStyleGeminiLive model:@"gemini-3.5-live-translate-preview"];
	session.language = @"pl";
	session.instructions = @"ignored";
	session.tools = @[ @{ @"name": @"lookup", @"parameters": @{ @"type": @"object" } } ];
	[session connect];
	NSDictionary *setup = session.scripted.sentEvents.firstObject[@"setup"];
	XCTAssertEqualObjects(setup[@"generationConfig"][@"translationConfig"], (@{ @"targetLanguageCode": @"pl" }));
	XCTAssertEqualObjects(setup[@"generationConfig"][@"responseModalities"], @[ @"AUDIO" ]);
	XCTAssertNil(setup[@"systemInstruction"]);
	XCTAssertNil(setup[@"tools"]);
}

// An ephemeral token connects to the constrained method as access_token; media other than audio goes to
// mediaHandler.
- (void)testGeminiLiveTakesAnEphemeralTokenAndHandsOnVideoFrames
{
	NFKScriptedSession *session = [self sessionFor:NFKRemoteProvider.googleGemini style:NFKRealtimeAPIStyleGeminiLive model:@"gemini-3.8-live"];
	session.apiKey = @"auth_tokens/abc";
	NSMutableData *audio = [NSMutableData data];
	NSMutableArray *media = [NSMutableArray array];
	session.audioHandler = ^(NSData *pcm) { [audio appendData:pcm]; };
	session.mediaHandler = ^(NSData *data, NSString *mimeType) { [media addObject:@[ data, mimeType ]]; };
	[session connect];
	NSURLComponents *url = [NSURLComponents componentsWithURL:session.scripted.request.URL resolvingAgainstBaseURL:NO];
	XCTAssertTrue([url.path hasSuffix:@"v1beta.GenerativeService.BidiGenerateContentConstrained"]);
	XCTAssertEqualObjects(url.queryItems, @[ [NSURLQueryItem queryItemWithName:@"access_token" value:@"auth_tokens/abc"] ]);

	[session.scripted push:[NSString stringWithFormat:@"{\"serverContent\":{\"modelTurn\":{\"parts\":[{\"inlineData\":{\"mimeType\":\"video/mp4\",\"data\":\"%@\"}},"
							"{\"inlineData\":{\"mimeType\":\"audio/pcm;rate=24000\",\"data\":\"%@\"}}]}}}", [self base64:@"frame"], [self base64:@"pcm"]]];
	XCTAssertEqualObjects(audio, [@"pcm" dataUsingEncoding:NSUTF8StringEncoding]);
	XCTAssertEqualObjects(media, (@[ @[ [@"frame" dataUsingEncoding:NSUTF8StringEncoding], @"video/mp4" ] ]));
}

// xAI's streaming STT takes the audio as binary frames and reports partial and final transcripts.
- (void)testXAIStreamingTranscriptionSendsBinaryAudioAndReportsPartials
{
	NFKScriptedSession *session = [self sessionFor:NFKRemoteProvider.xAI style:NFKRealtimeAPIStyleXAITranscription model:@"grok-voice-transcribe-2.0"];
	session.language = @"en";
	NSMutableArray *texts = [NSMutableArray array];
	session.textHandler = ^(NSString *text, NFKRealtimeTextKind kind) { [texts addObject:@[ text, @(kind) ]]; };
	[session connect];
	NSURLComponents *components = [NSURLComponents componentsWithURL:session.scripted.request.URL resolvingAgainstBaseURL:NO];
	XCTAssertTrue([components.queryItems containsObject:[NSURLQueryItem queryItemWithName:@"sample_rate" value:@"16000"]]);
	XCTAssertTrue([components.queryItems containsObject:[NSURLQueryItem queryItemWithName:@"language" value:@"en"]]);
	XCTAssertEqual(session.scripted.sentEvents.count, 0, @"the handshake query carries the configuration");
	[session appendAudio:[@"pcm" dataUsingEncoding:NSUTF8StringEncoding]];
	XCTAssertEqualObjects(session.scripted.sentData.firstObject, [@"pcm" dataUsingEncoding:NSUTF8StringEncoding]);
	[session finishInput];
	XCTAssertEqualObjects(session.scripted.sentEvents.lastObject, (@{ @"type": @"audio.done" }));
	[session.scripted push:@"{\"type\":\"transcript.partial\",\"text\":\"hel\",\"is_final\":false}"];
	[session.scripted push:@"{\"type\":\"transcript.partial\",\"text\":\"hello\",\"is_final\":true}"];
	XCTAssertEqualObjects(texts, (@[ @[ @"hel", @(NFKRealtimeTextKindInputTranscript) ], @[ @"hello", @(NFKRealtimeTextKindInputTranscriptFinal) ] ]));
}

- (void)testMistralRealtimeTranscriptionSpeaksItsProtocol
{
	NFKScriptedSession *session = [self sessionFor:NFKRemoteProvider.mistral style:NFKRealtimeAPIStyleMistralTranscription model:@"voxtral-mini-transcribe-realtime-2602"];
	NSMutableArray *texts = [NSMutableArray array];
	session.textHandler = ^(NSString *text, NFKRealtimeTextKind kind) { [texts addObject:text]; };
	[session connect];
	XCTAssertEqualObjects(session.scripted.request.URL.absoluteString,
						  @"wss://api.mistral.ai/v1/audio/transcriptions/realtime?model=voxtral-mini-transcribe-realtime-2602");
	XCTAssertEqualObjects(session.scripted.sentEvents.firstObject,
						  (@{ @"type": @"session.update", @"session": @{ @"audio_format": @{ @"encoding": @"pcm_s16le", @"sample_rate": @16000 } } }));
	[session appendAudio:[@"pcm" dataUsingEncoding:NSUTF8StringEncoding]];
	[session commitAudio];
	[session finishInput];
	XCTAssertEqualObjects(session.scripted.sentEvents[1][@"type"], @"input_audio.append");
	XCTAssertEqualObjects(session.scripted.sentEvents[2][@"type"], @"input_audio.flush");
	XCTAssertEqualObjects(session.scripted.sentEvents[3][@"type"], @"input_audio.end");
	[session.scripted push:@"{\"type\":\"transcription.text.delta\",\"text\":\"Bonjour\"}"];
	[session.scripted push:@"{\"type\":\"transcription.done\",\"text\":\"Bonjour.\",\"model\":\"v\"}"];
	XCTAssertEqualObjects(texts, (@[ @"Bonjour", @"Bonjour." ]));
}

// Meta's socket reads the key from the first frame: no header, no query the session adds.
- (void)testMetaTranscriptionSendsItsKeyInTheHandshakeFrameAndNoHeader
{
	NFKScriptedSession *session = [self sessionFor:NFKRemoteProvider.meta style:NFKRealtimeAPIStyleMetaTranscription model:@"muse-voice-transcribe-1.0"];
	[session connect];
	XCTAssertEqualObjects(session.scripted.request.URL.absoluteString, @"wss://api.meta.ai/v1/asr/realtime");
	XCTAssertNil([session.scripted.request valueForHTTPHeaderField:@"Authorization"]);
	XCTAssertEqual(session.inputSampleRate, 24000u);
	NSDictionary *handshake = @{ @"authorization": @{ @"accessToken": @"Bearer k" },
								 @"audioEncoding": @"PCM_24KHZ",
								 @"model": @"muse-voice-transcribe-1.0" };
	XCTAssertEqualObjects(session.scripted.sentFrames, (@[ handshake ]));
}

- (void)testMetaTranscriptionBuildsItsHandshakeFromTheSessionsProperties
{
	NFKScriptedSession *session = [self sessionFor:NFKRemoteProvider.meta style:NFKRealtimeAPIStyleMetaTranscription model:@"muse-voice-transcribe-1.0"];
	session.endpointURL = [NSURL URLWithString:@"wss://api.meta.ai/v1/asr/realtime?sessionId=mic-1"];
	session.inputSampleRate = 16000;
	session.language = @"English";
	session.extraConfiguration = @{ @"mode": @"DIARIZATION", @"partialMode": @"DELTA", @"emitAudioProgress": @NO,
									@"keywords": @[ @"Acme Mobile", @"eSIM" ] };
	[session connect];
	XCTAssertEqualObjects(session.scripted.request.URL.absoluteString, @"wss://api.meta.ai/v1/asr/realtime?sessionId=mic-1");
	XCTAssertNil([session.scripted.request valueForHTTPHeaderField:@"Authorization"]);
	NSDictionary *handshake = @{ @"authorization": @{ @"accessToken": @"Bearer k" },
								 @"audioEncoding": @"PCM_16KHZ",
								 @"model": @"muse-voice-transcribe-1.0",
								 @"languageBias": @[ @"English" ],
								 @"mode": @"DIARIZATION",
								 @"partialMode": @"DELTA",
								 @"emitAudioProgress": @NO,
								 @"keywords": @[ @"Acme Mobile", @"eSIM" ] };
	XCTAssertEqualObjects(session.scripted.sentEvents.firstObject, handshake);

	session.extraConfiguration = @{ @"languageBias": @[ @"English", @"French" ] };
	[session connect];
	NSArray *languages = @[ @"English", @"French" ];
	XCTAssertEqualObjects(session.scripted.sentEvents.firstObject[@"languageBias"], languages);
}

// The service takes audio after its {sessionId}; what is appended earlier goes out, in order, when it arrives.
- (void)testMetaTranscriptionHoldsAudioUntilTheAcknowledgementThenSendsBinaryFrames
{
	NFKScriptedSession *session = [self sessionFor:NFKRemoteProvider.meta style:NFKRealtimeAPIStyleMetaTranscription model:@"muse-voice-transcribe-1.0"];
	NSMutableArray<NSDictionary *> *events = [NSMutableArray array];
	session.eventHandler = ^(NSDictionary *event) { [events addObject:event]; };
	NSData *first = [@"pcm1" dataUsingEncoding:NSUTF8StringEncoding];
	NSData *second = [@"pcm2" dataUsingEncoding:NSUTF8StringEncoding];
	NSData *third = [@"pcm3" dataUsingEncoding:NSUTF8StringEncoding];
	[session connect];
	[session appendAudio:first];
	[session appendAudio:second];
	XCTAssertEqual(session.scripted.sentData.count, 0u);
	XCTAssertEqual(session.scripted.sentEvents.count, 1u);

	[session.scripted push:@"{\"sessionId\":\"550e8400\"}"];
	NSArray *flushed = @[ first, second ];
	XCTAssertEqualObjects(session.scripted.sentData, flushed);
	XCTAssertEqualObjects(events, (@[ @{ @"sessionId": @"550e8400" } ]));

	[session appendAudio:third];
	[session commitAudio];
	[session sendText:@"ignored"];
	[session finishInput];
	NSArray *afterHandshake = @[ first, second, third, @{ @"type": @"endStream" } ];
	XCTAssertEqualObjects([session.scripted.sentFrames subarrayWithRange:NSMakeRange(1, session.scripted.sentFrames.count - 1)], afterHandshake);
}

- (void)testMetaTranscriptionEndsTheStreamAfterTheHeldAudioWhenInputFinishesBeforeTheAcknowledgement
{
	NFKScriptedSession *session = [self sessionFor:NFKRemoteProvider.meta style:NFKRealtimeAPIStyleMetaTranscription model:@"muse-voice-transcribe-1.0"];
	NSData *audio = [@"pcm" dataUsingEncoding:NSUTF8StringEncoding];
	[session connect];
	[session appendAudio:audio];
	[session finishInput];
	XCTAssertEqual(session.scripted.sentFrames.count, 1u);
	[session.scripted push:@"{\"sessionId\":\"s1\"}"];
	[session.scripted push:@"{\"sessionId\":\"s1\"}"];
	NSArray *afterHandshake = @[ audio, @{ @"type": @"endStream" } ];
	XCTAssertEqualObjects([session.scripted.sentFrames subarrayWithRange:NSMakeRange(1, session.scripted.sentFrames.count - 1)], afterHandshake);
}

- (void)testAReconnectedMetaSessionHoldsAudioForItsOwnAcknowledgement
{
	NFKScriptedSession *session = [self sessionFor:NFKRemoteProvider.meta style:NFKRealtimeAPIStyleMetaTranscription model:@"muse-voice-transcribe-1.0"];
	NSData *stale = [@"old" dataUsingEncoding:NSUTF8StringEncoding];
	NSData *fresh = [@"new" dataUsingEncoding:NSUTF8StringEncoding];
	[session connect];
	[session appendAudio:stale];
	[session close];
	[session connect];
	[session.scripted push:@"{\"sessionId\":\"s2\"}"];
	XCTAssertEqual(session.scripted.sentData.count, 0u);

	[session close];
	[session connect];
	[session appendAudio:fresh];
	XCTAssertEqual(session.scripted.sentData.count, 0u);
	[session.scripted push:@"{\"sessionId\":\"s3\"}"];
	XCTAssertEqualObjects(session.scripted.sentData, (@[ fresh ]));
}

// PUSH_TO_TALK, the service's default: partials replace one another and the transcript marked final ends the turn.
- (void)testMetaPushToTalkReportsPartialsAndEndsTheTurnAtTheFinalTranscript
{
	NFKScriptedSession *session = [self sessionFor:NFKRemoteProvider.meta style:NFKRealtimeAPIStyleMetaTranscription model:@"muse-voice-transcribe-1.0"];
	NSMutableArray *texts = [NSMutableArray array];
	NSMutableArray<NSString *> *types = [NSMutableArray array];
	__block NSUInteger turns = 0;
	session.textHandler = ^(NSString *text, NFKRealtimeTextKind kind) { [texts addObject:@[ text, @(kind) ]]; };
	session.eventHandler = ^(NSDictionary *event) { [types addObject:event[@"type"] ?: @"acknowledgement"]; };
	session.turnHandler = ^{ turns += 1; };
	[session connect];
	[session.scripted push:@"{\"sessionId\":\"s1\"}"];
	[session.scripted push:@"{\"type\":\"audioProgress\",\"audioProcessedMs\":800}"];
	[session.scripted push:@"{\"type\":\"transcript\",\"transcript\":\"I fixed the bug\",\"final\":false,\"audioProcessedMs\":1600}"];
	[session.scripted push:@"{\"type\":\"transcript\",\"transcript\":\"I fixed the bug. By fixed\",\"final\":false,\"audioProcessedMs\":2400}"];
	[session.scripted push:@"{\"type\":\"somethingNew\",\"transcript\":\"ignored\"}"];
	XCTAssertEqual(turns, 0u);
	[session.scripted push:@"{\"type\":\"transcript\",\"transcript\":\"I fixed the bug. By fixed, I mean it.\",\"final\":true,\"audioProcessedMs\":4000}"];
	XCTAssertEqualObjects(texts, (@[ @[ @"I fixed the bug", @(NFKRealtimeTextKindInputTranscript) ],
									 @[ @"I fixed the bug. By fixed", @(NFKRealtimeTextKindInputTranscript) ],
									 @[ @"I fixed the bug. By fixed, I mean it.", @(NFKRealtimeTextKindInputTranscriptFinal) ] ]));
	XCTAssertEqual(turns, 1u);
	XCTAssertEqualObjects(types, (@[ @"acknowledgement", @"audioProgress", @"transcript", @"transcript", @"somethingNew", @"transcript" ]));
}

// ENDPOINTING and DIARIZATION: a turn ends at its speechComplete, whose transcript may differ from the last partial.
- (void)testMetaTurnDetectionEndsEachTurnAtItsSpeechComplete
{
	NFKScriptedSession *session = [self sessionFor:NFKRemoteProvider.meta style:NFKRealtimeAPIStyleMetaTranscription model:@"muse-voice-transcribe-1.0"];
	session.extraConfiguration = @{ @"mode": @"DIARIZATION" };
	NSMutableArray *texts = [NSMutableArray array];
	NSMutableArray<NSDictionary *> *events = [NSMutableArray array];
	__block NSUInteger turns = 0;
	session.textHandler = ^(NSString *text, NFKRealtimeTextKind kind) { [texts addObject:@[ text, @(kind) ]]; };
	session.eventHandler = ^(NSDictionary *event) { [events addObject:event]; };
	session.turnHandler = ^{ turns += 1; };
	[session connect];
	[session.scripted push:@"{\"sessionId\":\"s1\"}"];
	[session.scripted push:@"{\"type\":\"speechStart\",\"turnId\":1,\"audioProcessedMs\":1200}"];
	[session.scripted push:@"{\"type\":\"transcript\",\"transcript\":\"thanks for calling\",\"final\":false,\"audioProcessedMs\":2400}"];
	[session.scripted push:@"{\"type\":\"speaker\",\"label\":\"A\",\"audioProcessedMs\":2480}"];
	[session.scripted push:@"{\"type\":\"transcript\",\"transcript\":\"thanks for calling how can i help\",\"final\":true,\"audioProcessedMs\":3400}"];
	[session.scripted push:@"{\"type\":\"speechEnd\",\"turnId\":1,\"audioProcessedMs\":3600}"];
	XCTAssertEqual(turns, 0u);
	[session.scripted push:@"{\"type\":\"speechComplete\",\"turnId\":1,\"transcript\":\"Thanks for calling. How can I help?\",\"audioProcessedMs\":3600}"];
	XCTAssertEqualObjects(texts, (@[ @[ @"thanks for calling", @(NFKRealtimeTextKindInputTranscript) ],
									 @[ @"thanks for calling how can i help", @(NFKRealtimeTextKindInputTranscript) ],
									 @[ @"Thanks for calling. How can I help?", @(NFKRealtimeTextKindInputTranscriptFinal) ] ]));
	XCTAssertEqual(turns, 1u);
	XCTAssertEqualObjects([events valueForKey:@"type"], (@[ NSNull.null, @"speechStart", @"transcript", @"speaker", @"transcript", @"speechEnd", @"speechComplete" ]));
	XCTAssertEqualObjects(events[3][@"label"], @"A");
	XCTAssertEqualObjects(events[1][@"turnId"], @1);
}

- (void)testAMetaErrorFrameReachesTheErrorHandlerWithItsMessage
{
	NFKScriptedSession *session = [self sessionFor:NFKRemoteProvider.meta style:NFKRealtimeAPIStyleMetaTranscription model:@"muse-voice-transcribe-1.0"];
	__block NSError *reported = nil;
	__block NSDictionary *raw = nil;
	session.errorHandler = ^(NSError *error) { reported = error; };
	session.eventHandler = ^(NSDictionary *event) { raw = event; };
	[session connect];
	[session appendAudio:[@"pcm" dataUsingEncoding:NSUTF8StringEncoding]];
	[session.scripted push:@"{\"type\":\"error\",\"message\":\"Unsupported audioEncoding\",\"sessionId\":\"\",\"errorType\":\"invalid_request_error\","
							"\"errorCode\":\"invalid_value\",\"errorParam\":\"audioEncoding\"}"];
	XCTAssertEqualObjects(reported.localizedDescription, @"Unsupported audioEncoding");
	XCTAssertEqualObjects(raw[@"errorParam"], @"audioEncoding");
	XCTAssertEqual(session.scripted.sentData.count, 0u, @"an error frame is not the acknowledgement");
}

// A close frame from the service fails the pending receive whatever its code; only a code other than 1000 is a failure.
- (void)testAServiceCloseWithCode1000IsNotAnErrorAndAnyOtherCodeNamesItself
{
	NSURLRequest *request = [NSURLRequest requestWithURL:[NSURL URLWithString:@"wss://api.meta.ai/v1/asr/realtime"]];
	NFKRealtimeWebSocket *socket = [[NFKRealtimeWebSocket alloc] initWithRequest:request session:nil];
	NSError *notConnected = [NSError errorWithDomain:NSPOSIXErrorDomain code:ENOTCONN userInfo:nil];
	NSData *reason = [@"Max session duration reached" dataUsingEncoding:NSUTF8StringEncoding];
	XCTAssertNil([socket errorForCloseCode:NSURLSessionWebSocketCloseCodeNormalClosure reason:nil receiveError:notConnected]);
	XCTAssertEqualObjects([socket errorForCloseCode:NSURLSessionWebSocketCloseCodeInvalid reason:nil receiveError:notConnected], notConnected);
	XCTAssertEqualObjects([socket errorForCloseCode:NSURLSessionWebSocketCloseCodeInternalServerError reason:reason receiveError:notConnected].localizedDescription,
						  @"the realtime service closed the socket with code 1011: Max session duration reached");
	XCTAssertEqualObjects([socket errorForCloseCode:NSURLSessionWebSocketCloseCodePolicyViolation reason:nil receiveError:notConnected].localizedDescription,
						  @"the realtime service closed the socket with code 1008");
}

- (void)testStreamingSpeechSendsTextAndReportsAudio
{
	NFKScriptedSession *xai = [self sessionFor:NFKRemoteProvider.xAI style:NFKRealtimeAPIStyleXAISpeech model:@"m"];
	xai.voice = @"eve";
	NSMutableData *audio = [NSMutableData data];
	xai.audioHandler = ^(NSData *pcm) { [audio appendData:pcm]; };
	[xai connect];
	XCTAssertTrue([xai.scripted.request.URL.query containsString:@"voice=eve"]);
	[xai sendText:@"Hello"];
	[xai finishInput];
	XCTAssertEqualObjects(xai.scripted.sentEvents[0], (@{ @"type": @"text.delta", @"delta": @"Hello" }));
	XCTAssertEqualObjects(xai.scripted.sentEvents[1], (@{ @"type": @"text.done" }));
	[xai.scripted push:[NSString stringWithFormat:@"{\"type\":\"audio.delta\",\"delta\":\"%@\"}", [self base64:@"s1"]]];
	XCTAssertEqualObjects(audio, [@"s1" dataUsingEncoding:NSUTF8StringEncoding]);

	NFKScriptedSession *together = [self sessionFor:NFKRemoteProvider.together style:NFKRealtimeAPIStyleTogetherSpeech model:@"hexgrad/Kokoro-82M"];
	NSMutableData *spoken = [NSMutableData data];
	together.audioHandler = ^(NSData *pcm) { [spoken appendData:pcm]; };
	[together connect];
	[together sendText:@"Hi"];
	[together finishInput];
	XCTAssertEqualObjects(together.scripted.sentEvents[0], (@{ @"type": @"input_text_buffer.append", @"text": @"Hi" }));
	XCTAssertEqualObjects(together.scripted.sentEvents[1][@"type"], @"input_text_buffer.commit");
	[together.scripted push:[NSString stringWithFormat:@"{\"type\":\"conversation.item.audio_output.delta\",\"delta\":\"%@\"}", [self base64:@"t1"]]];
	XCTAssertEqualObjects(spoken, [@"t1" dataUsingEncoding:NSUTF8StringEncoding]);
}

- (void)testLiveMusicSetsPromptsConfigurationAndPlayback
{
	NFKScriptedSession *music = [self sessionFor:NFKRemoteProvider.googleGemini style:NFKRealtimeAPIStyleGeminiLiveMusic model:@"lyria-realtime-exp"];
	NSMutableData *audio = [NSMutableData data];
	music.audioHandler = ^(NSData *pcm) { [audio appendData:pcm]; };
	[music connect];
	XCTAssertEqual(music.outputSampleRate, 48000);
	XCTAssertTrue([music.scripted.request.URL.absoluteString containsString:@"v1alpha.GenerativeService.BidiGenerateMusic"]);
	[music setMusicPrompts:@[ @{ @"text": @"minimal techno", @"weight": @1.0 } ]];
	[music setMusicConfiguration:@{ @"bpm": @120 }];
	[music playMusic];
	XCTAssertEqualObjects(music.scripted.sentEvents[1][@"clientContent"][@"weightedPrompts"][0][@"text"], @"minimal techno");
	XCTAssertEqualObjects(music.scripted.sentEvents[2], (@{ @"musicGenerationConfig": @{ @"bpm": @120 } }));
	XCTAssertEqualObjects(music.scripted.sentEvents[3], (@{ @"playbackControl": @"PLAY" }));
	[music resetMusicContext];
	XCTAssertEqualObjects(music.scripted.sentEvents[4], (@{ @"playbackControl": @"RESET_CONTEXT" }));
	[music.scripted push:[NSString stringWithFormat:@"{\"serverContent\":{\"audioChunks\":[{\"data\":\"%@\",\"mimeType\":\"audio/l16;rate=48000;channels=2\"}]}}", [self base64:@"mu"]]];
	XCTAssertEqualObjects(audio, [@"mu" dataUsingEncoding:NSUTF8StringEncoding]);
}

- (void)testAnErrorEventReachesTheErrorHandlerAndCloseClosesTheSocket
{
	NFKScriptedSession *session = [self sessionFor:NFKRemoteProvider.together style:NFKRealtimeAPIStyleTogetherTranscription model:@"openai/whisper-large-v3"];
	__block NSError *reported = nil;
	session.errorHandler = ^(NSError *error) { reported = error; };
	[session connect];
	XCTAssertTrue([session.scripted.request.URL.query containsString:@"input_audio_format=pcm_s16le_16000"]);
	[session.scripted push:@"{\"type\":\"error\",\"error\":{\"message\":\"bad audio\"}}"];
	XCTAssertEqualObjects(reported.localizedDescription, @"bad audio");
	[session close];
	XCTAssertTrue(session.scripted.closed);
	XCTAssertFalse(session.isConnected);
}

@end
