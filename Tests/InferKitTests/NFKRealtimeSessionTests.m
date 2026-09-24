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
@property (nonatomic, copy) void (^messageHandler)(NSString *, NSData *);
@property (nonatomic, assign) BOOL closed;
@end

@implementation NFKScriptedSocket
- (void)openWithMessageHandler:(void (^)(NSString *, NSData *))messageHandler closeHandler:(void (^)(NSError *))closeHandler
{
	self.messageHandler = messageHandler;
	self.sentEvents = [NSMutableArray array];
	self.sentData = [NSMutableArray array];
}
- (void)sendText:(NSString *)text
{
	[self.sentEvents addObject:[NSJSONSerialization JSONObjectWithData:[text dataUsingEncoding:NSUTF8StringEncoding] options:0 error:NULL]];
}
- (void)sendData:(NSData *)data
{
	[self.sentData addObject:data];
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
