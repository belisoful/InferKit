//
//  NFKRealtimeSession.m
//  InferKit
//

#import <InferKit/NFKRealtimeSession.h>
#import <InferKit/NFKRemoteProvider.h>
#import <InferKit/NFKRemoteTransport.h>
#import <InferKit/NFKErrors.h>

#pragma mark - WebSocket

@interface NFKRealtimeWebSocket ()
@property (nonatomic, strong) NSURLRequest *request;
@property (nonatomic, strong) NSURLSession *session;
@property (nonatomic, strong, nullable) NSURLSessionWebSocketTask *task;
@property (nonatomic, copy, nullable) void (^messageHandler)(NSString * _Nullable, NSData * _Nullable);
@property (nonatomic, copy, nullable) void (^closeHandler)(NSError * _Nullable);
@end

@implementation NFKRealtimeWebSocket

- (instancetype)initWithRequest:(NSURLRequest *)request session:(nullable NSURLSession *)session
{
	self = [super init];
	if (self != nil) {
		_request = request;
		_session = session ?: NSURLSession.sharedSession;
	}
	return self;
}

- (void)openWithMessageHandler:(void (^)(NSString * _Nullable, NSData * _Nullable))messageHandler
				  closeHandler:(void (^)(NSError * _Nullable))closeHandler
{
	self.messageHandler = messageHandler;
	self.closeHandler = closeHandler;
	self.task = [self.session webSocketTaskWithRequest:self.request];
	[self.task resume];
	[self receive];
}

// One receive per message; the next is asked for once this one is handled.
- (void)receive
{
	__weak NFKRealtimeWebSocket *weakSelf = self;
	[self.task receiveMessageWithCompletionHandler:^(NSURLSessionWebSocketMessage *message, NSError *error) {
		NFKRealtimeWebSocket *socket = weakSelf;
		if (socket == nil) {
			return;
		}
		if (error != nil) {
			[socket finishWithError:error];
			return;
		}
		if (socket.messageHandler != nil) {
			socket.messageHandler(message.string, message.data);
		}
		[socket receive];
	}];
}

- (void)finishWithError:(nullable NSError *)error
{
	void (^handler)(NSError * _Nullable) = self.closeHandler;
	self.closeHandler = nil;
	self.messageHandler = nil;
	if (handler != nil) {
		handler(error);
	}
}

- (void)sendText:(NSString *)text
{
	[self.task sendMessage:[[NSURLSessionWebSocketMessage alloc] initWithString:text] completionHandler:^(NSError *error) {}];
}

- (void)sendData:(NSData *)data
{
	[self.task sendMessage:[[NSURLSessionWebSocketMessage alloc] initWithData:data] completionHandler:^(NSError *error) {}];
}

- (void)close
{
	[self.task cancelWithCloseCode:NSURLSessionWebSocketCloseCodeNormalClosure reason:nil];
	[self finishWithError:nil];
}

@end

#pragma mark - Session

@interface NFKRealtimeSession ()
@property (nonatomic, strong, nullable) id<NFKRealtimeSocket> socket;
@property (nonatomic, readwrite, getter=isConnected) BOOL connected;
@end

@implementation NFKRealtimeSession

+ (nullable instancetype)sessionForProvider:(NFKRemoteProvider *)provider
								   apiStyle:(NFKRealtimeAPIStyle)apiStyle
									 apiKey:(nullable NSString *)apiKey
								  modelName:(nullable NSString *)modelName
{
	NSURL *endpoint = [self endpointForProvider:provider apiStyle:apiStyle];
	if (endpoint == nil) {
		return nil;
	}
	NFKRealtimeSession *session = [[self alloc] init];
	session.apiStyle = apiStyle;
	session.endpointURL = endpoint;
	session.apiKey = apiKey;
	session.modelName = modelName;
	return session;
}

// Each style's socket lives beside its provider's HTTPS base, on the wss (or, locally, ws) scheme.
+ (nullable NSURL *)endpointForProvider:(NFKRemoteProvider *)provider apiStyle:(NFKRealtimeAPIStyle)apiStyle
{
	NSString *identifier = provider.identifier;
	NSDictionary<NSNumber *, NSArray<NSString *> *> *routes = @{
		@(NFKRealtimeAPIStyleOpenAIConversation): @[ @"openai", @"realtime" ],
		@(NFKRealtimeAPIStyleOpenAITranscription): @[ @"openai", @"realtime" ],
		@(NFKRealtimeAPIStyleOpenAITranslation): @[ @"openai", @"realtime/translations" ],
		@(NFKRealtimeAPIStyleXAIConversation): @[ @"xai", @"realtime" ],
		@(NFKRealtimeAPIStyleXAITranscription): @[ @"xai", @"stt" ],
		@(NFKRealtimeAPIStyleXAISpeech): @[ @"xai", @"tts" ],
		@(NFKRealtimeAPIStyleMistralTranscription): @[ @"mistral", @"audio/transcriptions/realtime" ],
		@(NFKRealtimeAPIStyleTogetherTranscription): @[ @"together", @"realtime" ],
		@(NFKRealtimeAPIStyleTogetherSpeech): @[ @"together", @"audio/speech/websocket" ],
		@(NFKRealtimeAPIStyleVLLMTranscription): @[ @"vllm", @"realtime" ],
	};
	if (apiStyle == NFKRealtimeAPIStyleGeminiLive || apiStyle == NFKRealtimeAPIStyleGeminiLiveMusic) {
		if (![identifier isEqualToString:@"gemini"]) {
			return nil;
		}
		NSString *method = apiStyle == NFKRealtimeAPIStyleGeminiLive
			? @"v1beta.GenerativeService.BidiGenerateContent" : @"v1alpha.GenerativeService.BidiGenerateMusic";
		return [NSURL URLWithString:[@"wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage." stringByAppendingString:method]];
	}
	NSArray<NSString *> *route = routes[@(apiStyle)];
	if (route == nil || ![route[0] isEqualToString:identifier]) {
		return nil;
	}
	NSURLComponents *components = [NSURLComponents componentsWithURL:[provider URLForPath:route[1]] resolvingAgainstBaseURL:NO];
	components.scheme = [components.scheme isEqualToString:@"http"] ? @"ws" : @"wss";
	return components.URL;
}

- (NSUInteger)inputSampleRate
{
	if (_inputSampleRate > 0) {
		return _inputSampleRate;
	}
	switch (self.apiStyle) {
		case NFKRealtimeAPIStyleOpenAIConversation:
		case NFKRealtimeAPIStyleOpenAITranscription:
		case NFKRealtimeAPIStyleOpenAITranslation:
		case NFKRealtimeAPIStyleXAIConversation:
			return 24000;
		default:
			return 16000;
	}
}

- (NSUInteger)outputSampleRate
{
	return self.apiStyle == NFKRealtimeAPIStyleGeminiLiveMusic ? 48000 : 24000;
}

#pragma mark Connection

- (void)connect
{
	NSURLRequest *request = [self connectionRequest];
	self.socket = [self socketForRequest:request];
	__weak NFKRealtimeSession *weakSelf = self;
	[self.socket openWithMessageHandler:^(NSString * _Nullable text, NSData * _Nullable data) {
		[weakSelf receiveText:text data:data];
	} closeHandler:^(NSError * _Nullable error) {
		NFKRealtimeSession *session = weakSelf;
		session.connected = NO;
		if (error != nil && session.errorHandler != nil) {
			session.errorHandler(error);
		}
		if (session.closeHandler != nil) {
			session.closeHandler(error);
		}
	}];
	self.connected = YES;
	NSDictionary *configuration = [self configurationMessage];
	if (configuration != nil) {
		[self sendEvent:configuration];
	}
}

- (id<NFKRealtimeSocket>)socketForRequest:(NSURLRequest *)request
{
	return [[NFKRealtimeWebSocket alloc] initWithRequest:request session:nil];
}

// The query each style's handshake reads, and the key: Gemini's in the query, every other's in a
// bearer header.
- (NSURLRequest *)connectionRequest
{
	NSURLComponents *components = [NSURLComponents componentsWithURL:self.endpointURL resolvingAgainstBaseURL:NO];
	NSMutableArray<NSURLQueryItem *> *query = [NSMutableArray arrayWithArray:components.queryItems ?: @[]];
	void (^add)(NSString *, id) = ^(NSString *name, id value) {
		if (value != nil) {
			[query addObject:[NSURLQueryItem queryItemWithName:name value:[value description]]];
		}
	};
	switch (self.apiStyle) {
		case NFKRealtimeAPIStyleOpenAIConversation:
		case NFKRealtimeAPIStyleOpenAITranslation:
		case NFKRealtimeAPIStyleXAIConversation:
		case NFKRealtimeAPIStyleMistralTranscription:
			add(@"model", self.modelName);
			break;
		case NFKRealtimeAPIStyleXAITranscription:
			add(@"sample_rate", @(self.inputSampleRate));
			add(@"encoding", @"pcm");
			add(@"interim_results", @"true");
			add(@"language", self.language);
			break;
		case NFKRealtimeAPIStyleXAISpeech:
			add(@"voice", self.voice);
			add(@"language", self.language ?: @"auto");
			add(@"codec", @"pcm");
			add(@"sample_rate", @(self.outputSampleRate));
			break;
		case NFKRealtimeAPIStyleTogetherTranscription:
			add(@"model", self.modelName);
			add(@"input_audio_format", [NSString stringWithFormat:@"pcm_s16le_%lu", (unsigned long)self.inputSampleRate]);
			break;
		case NFKRealtimeAPIStyleTogetherSpeech:
			add(@"model", self.modelName);
			add(@"voice", self.voice);
			break;
		case NFKRealtimeAPIStyleGeminiLive:
		case NFKRealtimeAPIStyleGeminiLiveMusic:
			add(@"key", self.apiKey);
			break;
		case NFKRealtimeAPIStyleOpenAITranscription:
		case NFKRealtimeAPIStyleVLLMTranscription:
			break;
	}
	components.queryItems = query.count > 0 ? query : nil;
	NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:components.URL ?: self.endpointURL];
	BOOL keyInQuery = self.apiStyle == NFKRealtimeAPIStyleGeminiLive || self.apiStyle == NFKRealtimeAPIStyleGeminiLiveMusic;
	if (!keyInQuery && self.apiKey.length > 0) {
		[request setValue:[@"Bearer " stringByAppendingString:self.apiKey] forHTTPHeaderField:@"Authorization"];
	}
	return request;
}

// The first message a style sends: the session's configuration, or nil for a style whose handshake
// query carries it.
- (nullable NSDictionary *)configurationMessage
{
	NSMutableDictionary *session = [NSMutableDictionary dictionary];
	NSDictionary *pcmIn = @{ @"type": @"audio/pcm", @"rate": @(self.inputSampleRate) };
	NSDictionary *message = nil;
	switch (self.apiStyle) {
		case NFKRealtimeAPIStyleOpenAIConversation:
		case NFKRealtimeAPIStyleXAIConversation: {
			if (self.apiStyle == NFKRealtimeAPIStyleOpenAIConversation) {
				session[@"type"] = @"realtime";
			}
			session[@"instructions"] = self.instructions;
			NSMutableDictionary *output = [NSMutableDictionary dictionaryWithObject:@{ @"type": @"audio/pcm", @"rate": @(self.outputSampleRate) } forKey:@"format"];
			output[@"voice"] = self.voice;
			session[@"audio"] = @{ @"input": @{ @"format": pcmIn }, @"output": output };
			if (self.tools.count > 0) {
				session[@"tools"] = [self functionToolsNamedInline:YES];
			}
			message = @{ @"type": @"session.update", @"session": session };
			break;
		}
		case NFKRealtimeAPIStyleOpenAITranscription: {
			NSMutableDictionary *transcription = [NSMutableDictionary dictionary];
			transcription[@"model"] = self.modelName;
			transcription[@"language"] = self.language;
			session[@"type"] = @"transcription";
			session[@"audio"] = @{ @"input": @{ @"format": pcmIn, @"transcription": transcription } };
			message = @{ @"type": @"session.update", @"session": session };
			break;
		}
		case NFKRealtimeAPIStyleOpenAITranslation:
			session[@"audio"] = @{ @"output": @{ @"language": self.language ?: @"en" } };
			message = @{ @"type": @"session.update", @"session": session };
			break;
		case NFKRealtimeAPIStyleMistralTranscription:
			session[@"audio_format"] = @{ @"encoding": @"pcm_s16le", @"sample_rate": @(self.inputSampleRate) };
			message = @{ @"type": @"session.update", @"session": session };
			break;
		case NFKRealtimeAPIStyleVLLMTranscription:
			message = @{ @"type": @"session.update", @"model": self.modelName ?: @"" };
			break;
		case NFKRealtimeAPIStyleGeminiLive: {
			NSMutableDictionary *setup = [NSMutableDictionary dictionaryWithObject:[@"models/" stringByAppendingString:self.modelName ?: @""] forKey:@"model"];
			NSMutableDictionary *generation = [NSMutableDictionary dictionaryWithObject:@[ @"AUDIO" ] forKey:@"responseModalities"];
			if (self.voice.length > 0) {
				generation[@"speechConfig"] = @{ @"voiceConfig": @{ @"prebuiltVoiceConfig": @{ @"voiceName": self.voice } } };
			}
			setup[@"generationConfig"] = generation;
			if (self.instructions.length > 0) {
				setup[@"systemInstruction"] = @{ @"parts": @[ @{ @"text": self.instructions } ] };
			}
			setup[@"inputAudioTranscription"] = @{};
			setup[@"outputAudioTranscription"] = @{};
			if (self.tools.count > 0) {
				setup[@"tools"] = @[ @{ @"functionDeclarations": [self functionToolsNamedInline:NO] } ];
			}
			[setup addEntriesFromDictionary:self.extraConfiguration ?: @{}];
			return @{ @"setup": setup };
		}
		case NFKRealtimeAPIStyleGeminiLiveMusic: {
			NSMutableDictionary *setup = [NSMutableDictionary dictionaryWithObject:[@"models/" stringByAppendingString:self.modelName ?: @"lyria-realtime-exp"] forKey:@"model"];
			[setup addEntriesFromDictionary:self.extraConfiguration ?: @{}];
			return @{ @"setup": setup };
		}
		case NFKRealtimeAPIStyleXAITranscription:
		case NFKRealtimeAPIStyleXAISpeech:
		case NFKRealtimeAPIStyleTogetherTranscription:
		case NFKRealtimeAPIStyleTogetherSpeech:
			return nil;
	}
	if (self.extraConfiguration.count > 0 && message[@"session"] != nil) {
		NSMutableDictionary *merged = [message[@"session"] mutableCopy];
		[merged addEntriesFromDictionary:self.extraConfiguration];
		NSMutableDictionary *withExtras = [message mutableCopy];
		withExtras[@"session"] = merged;
		return withExtras;
	}
	return message;
}

// OpenAI's realtime tools are {type: function, name, …}; Gemini's are bare declarations.
- (NSArray<NSDictionary *> *)functionToolsNamedInline:(BOOL)inlineType
{
	NSMutableArray<NSDictionary *> *tools = [NSMutableArray array];
	for (NSDictionary *tool in self.tools) {
		NSMutableDictionary *function = [NSMutableDictionary dictionary];
		if (inlineType) {
			function[@"type"] = @"function";
		}
		function[@"name"] = tool[@"name"];
		function[@"description"] = tool[@"description"];
		function[@"parameters"] = tool[@"parameters"] ?: @{ @"type": @"object", @"properties": @{} };
		[tools addObject:function];
	}
	return tools;
}

#pragma mark Sending

- (void)sendEvent:(NSDictionary<NSString *, id> *)event
{
	NSData *json = [NSJSONSerialization dataWithJSONObject:event options:0 error:NULL];
	NSString *text = json != nil ? [[NSString alloc] initWithData:json encoding:NSUTF8StringEncoding] : nil;
	if (text != nil) {
		[self.socket sendText:text];
	}
}

- (void)appendAudio:(NSData *)pcm
{
	NSString *encoded = [pcm base64EncodedStringWithOptions:0];
	switch (self.apiStyle) {
		case NFKRealtimeAPIStyleXAITranscription:
			[self.socket sendData:pcm];
			break;
		case NFKRealtimeAPIStyleGeminiLive: {
			NSString *mimeType = [NSString stringWithFormat:@"audio/pcm;rate=%lu", (unsigned long)self.inputSampleRate];
			[self sendEvent:@{ @"realtimeInput": @{ @"audio": @{ @"data": encoded, @"mimeType": mimeType } } }];
			break;
		}
		case NFKRealtimeAPIStyleOpenAITranslation:
			[self sendEvent:@{ @"type": @"session.input_audio_buffer.append", @"audio": encoded }];
			break;
		case NFKRealtimeAPIStyleMistralTranscription:
			[self sendEvent:@{ @"type": @"input_audio.append", @"audio": encoded }];
			break;
		case NFKRealtimeAPIStyleXAISpeech:
		case NFKRealtimeAPIStyleTogetherSpeech:
		case NFKRealtimeAPIStyleGeminiLiveMusic:
			break;
		default:
			[self sendEvent:@{ @"type": @"input_audio_buffer.append", @"audio": encoded }];
			break;
	}
}

- (void)commitAudio
{
	switch (self.apiStyle) {
		case NFKRealtimeAPIStyleXAITranscription:
			[self sendEvent:@{ @"type": @"finalize" }];
			break;
		case NFKRealtimeAPIStyleMistralTranscription:
			[self sendEvent:@{ @"type": @"input_audio.flush" }];
			break;
		case NFKRealtimeAPIStyleGeminiLive:
			[self sendEvent:@{ @"realtimeInput": @{ @"audioStreamEnd": @YES } }];
			break;
		case NFKRealtimeAPIStyleOpenAITranslation:
		case NFKRealtimeAPIStyleXAISpeech:
		case NFKRealtimeAPIStyleTogetherSpeech:
		case NFKRealtimeAPIStyleGeminiLiveMusic:
			break;
		default:
			[self sendEvent:@{ @"type": @"input_audio_buffer.commit" }];
			break;
	}
}

- (void)finishInput
{
	switch (self.apiStyle) {
		case NFKRealtimeAPIStyleXAITranscription:
			[self sendEvent:@{ @"type": @"audio.done" }];
			break;
		case NFKRealtimeAPIStyleXAISpeech:
			[self sendEvent:@{ @"type": @"text.done" }];
			break;
		case NFKRealtimeAPIStyleTogetherSpeech:
			[self sendEvent:@{ @"type": @"input_text_buffer.commit" }];
			break;
		case NFKRealtimeAPIStyleMistralTranscription:
			[self sendEvent:@{ @"type": @"input_audio.end" }];
			break;
		case NFKRealtimeAPIStyleVLLMTranscription:
			[self sendEvent:@{ @"type": @"input_audio_buffer.commit", @"final": @YES }];
			break;
		case NFKRealtimeAPIStyleOpenAITranslation:
			[self sendEvent:@{ @"type": @"session.close" }];
			break;
		case NFKRealtimeAPIStyleGeminiLive:
			[self sendEvent:@{ @"realtimeInput": @{ @"audioStreamEnd": @YES } }];
			break;
		default:
			[self commitAudio];
			break;
	}
}

- (void)sendText:(NSString *)text
{
	switch (self.apiStyle) {
		case NFKRealtimeAPIStyleXAISpeech:
			[self sendEvent:@{ @"type": @"text.delta", @"delta": text }];
			break;
		case NFKRealtimeAPIStyleTogetherSpeech:
			[self sendEvent:@{ @"type": @"input_text_buffer.append", @"text": text }];
			break;
		case NFKRealtimeAPIStyleGeminiLive:
			[self sendEvent:@{ @"clientContent": @{ @"turns": @[ @{ @"role": @"user", @"parts": @[ @{ @"text": text } ] } ],
													 @"turnComplete": @YES } }];
			break;
		default:
			[self sendEvent:@{ @"type": @"conversation.item.create",
							   @"item": @{ @"type": @"message", @"role": @"user",
										   @"content": @[ @{ @"type": @"input_text", @"text": text } ] } }];
			break;
	}
}

- (void)requestResponse
{
	if (self.apiStyle == NFKRealtimeAPIStyleOpenAIConversation || self.apiStyle == NFKRealtimeAPIStyleXAIConversation) {
		[self sendEvent:@{ @"type": @"response.create" }];
	}
}

- (void)sendToolResult:(NSString *)output forCallIdentifier:(NSString *)callIdentifier name:(nullable NSString *)name
{
	if (self.apiStyle == NFKRealtimeAPIStyleGeminiLive) {
		NSMutableDictionary *response = [NSMutableDictionary dictionaryWithObject:callIdentifier forKey:@"id"];
		response[@"name"] = name;
		response[@"response"] = @{ @"output": output };
		[self sendEvent:@{ @"toolResponse": @{ @"functionResponses": @[ response ] } }];
		return;
	}
	[self sendEvent:@{ @"type": @"conversation.item.create",
					   @"item": @{ @"type": @"function_call_output", @"call_id": callIdentifier, @"output": output } }];
}

- (void)setMusicPrompts:(NSArray<NSDictionary *> *)prompts
{
	[self sendEvent:@{ @"clientContent": @{ @"weightedPrompts": prompts } }];
}

- (void)setMusicConfiguration:(NSDictionary<NSString *, id> *)configuration
{
	[self sendEvent:@{ @"musicGenerationConfig": configuration }];
}

- (void)playMusic
{
	[self sendEvent:@{ @"playbackControl": @"PLAY" }];
}

- (void)pauseMusic
{
	[self sendEvent:@{ @"playbackControl": @"PAUSE" }];
}

- (void)stopMusic
{
	[self sendEvent:@{ @"playbackControl": @"STOP" }];
}

- (void)close
{
	[self.socket close];
	self.connected = NO;
}

#pragma mark Receiving

// Every service sends JSON; Gemini sends it in binary frames too.
- (void)receiveText:(nullable NSString *)text data:(nullable NSData *)data
{
	NSData *payload = data ?: [text dataUsingEncoding:NSUTF8StringEncoding];
	id event = payload != nil ? [NSJSONSerialization JSONObjectWithData:payload options:0 error:NULL] : nil;
	if (![event isKindOfClass:NSDictionary.class]) {
		return;
	}
	if (self.eventHandler != nil) {
		self.eventHandler(event);
	}
	if (self.apiStyle == NFKRealtimeAPIStyleGeminiLive || self.apiStyle == NFKRealtimeAPIStyleGeminiLiveMusic) {
		[self receiveGeminiEvent:event];
		return;
	}
	[self receiveTypedEvent:event];
}

- (void)receiveTypedEvent:(NSDictionary *)event
{
	NSString *type = [event[@"type"] isKindOfClass:NSString.class] ? event[@"type"] : @"";
	NSString *delta = [event[@"delta"] isKindOfClass:NSString.class] ? event[@"delta"] : nil;
	if ([type isEqualToString:@"error"] || [type hasSuffix:@".failed"]) {
		[self reportError:event];
		return;
	}
	// Audio: OpenAI and xAI response.output_audio.delta (response.audio.delta before), OpenAI's
	// translation session.output_audio.delta, xAI's TTS audio.delta, Together's audio_output.delta.
	if ([type hasSuffix:@"audio.delta"] && ![type containsString:@"transcript"]) {
		NSString *encoded = delta ?: ([event[@"audio"] isKindOfClass:NSString.class] ? event[@"audio"] : nil);
		NSData *pcm = encoded != nil ? [[NSData alloc] initWithBase64EncodedString:encoded options:NSDataBase64DecodingIgnoreUnknownCharacters] : nil;
		if (pcm != nil && self.audioHandler != nil) {
			self.audioHandler(pcm);
		}
		return;
	}
	if ([type isEqualToString:@"conversation.item.audio_output.delta"] && [event[@"delta"] isKindOfClass:NSString.class]) {
		NSData *pcm = [[NSData alloc] initWithBase64EncodedString:event[@"delta"] options:NSDataBase64DecodingIgnoreUnknownCharacters];
		if (pcm != nil && self.audioHandler != nil) {
			self.audioHandler(pcm);
		}
		return;
	}
	if ([type isEqualToString:@"response.output_text.delta"] || [type isEqualToString:@"response.text.delta"]) {
		[self reportText:delta kind:NFKRealtimeTextKindResponse];
	} else if ([type hasSuffix:@"output_audio_transcript.delta"] || [type isEqualToString:@"response.audio_transcript.delta"]
			   || [type isEqualToString:@"session.output_transcript.delta"]) {
		[self reportText:delta kind:NFKRealtimeTextKindOutputTranscript];
	} else if ([type isEqualToString:@"conversation.item.input_audio_transcription.delta"] || [type isEqualToString:@"session.input_transcript.delta"]
			   || [type isEqualToString:@"transcription.delta"]) {
		[self reportText:delta kind:NFKRealtimeTextKindInputTranscript];
	} else if ([type isEqualToString:@"transcription.text.delta"]) {
		[self reportText:event[@"text"] kind:NFKRealtimeTextKindInputTranscript];
	} else if ([type isEqualToString:@"transcript.partial"]) {
		NSString *text = event[@"text"] ?: event[@"transcript"];
		[self reportText:text kind:[event[@"is_final"] boolValue] ? NFKRealtimeTextKindInputTranscriptFinal : NFKRealtimeTextKindInputTranscript];
	} else if ([type isEqualToString:@"conversation.item.input_audio_transcription.completed"]) {
		[self reportText:event[@"transcript"] kind:NFKRealtimeTextKindInputTranscriptFinal];
	} else if ([type isEqualToString:@"transcription.done"] || [type isEqualToString:@"transcript.done"]) {
		[self reportText:event[@"text"] ?: event[@"transcript"] kind:NFKRealtimeTextKindInputTranscriptFinal];
		[self reportTurn];
	} else if ([type isEqualToString:@"response.function_call_arguments.done"]) {
		[self reportToolCall:event[@"call_id"] name:event[@"name"] arguments:event[@"arguments"]];
	} else if ([type isEqualToString:@"response.done"] || [type isEqualToString:@"audio.done"]
			   || [type isEqualToString:@"conversation.item.audio_output.done"]) {
		[self reportTurn];
	}
}

// Gemini's server messages are keyed by kind rather than typed: serverContent (model turn parts,
// transcriptions, turnComplete), toolCall, and Lyria's audioChunks.
- (void)receiveGeminiEvent:(NSDictionary *)event
{
	NSDictionary *content = [event[@"serverContent"] isKindOfClass:NSDictionary.class] ? event[@"serverContent"] : nil;
	if (content != nil) {
		NSDictionary *turn = [content[@"modelTurn"] isKindOfClass:NSDictionary.class] ? content[@"modelTurn"] : nil;
		for (NSDictionary *part in [turn[@"parts"] isKindOfClass:NSArray.class] ? turn[@"parts"] : @[]) {
			NSDictionary *inlineData = [part isKindOfClass:NSDictionary.class] && [part[@"inlineData"] isKindOfClass:NSDictionary.class] ? part[@"inlineData"] : nil;
			if ([inlineData[@"data"] isKindOfClass:NSString.class]) {
				[self reportAudio:inlineData[@"data"]];
			} else if ([part isKindOfClass:NSDictionary.class]) {
				[self reportText:part[@"text"] kind:NFKRealtimeTextKindResponse];
			}
		}
		for (NSDictionary *chunk in [content[@"audioChunks"] isKindOfClass:NSArray.class] ? content[@"audioChunks"] : @[]) {
			if ([chunk isKindOfClass:NSDictionary.class]) {
				[self reportAudio:chunk[@"data"]];
			}
		}
		NSDictionary *input = [content[@"inputTranscription"] isKindOfClass:NSDictionary.class] ? content[@"inputTranscription"] : nil;
		[self reportText:input[@"text"] kind:NFKRealtimeTextKindInputTranscript];
		NSDictionary *output = [content[@"outputTranscription"] isKindOfClass:NSDictionary.class] ? content[@"outputTranscription"] : nil;
		[self reportText:output[@"text"] kind:NFKRealtimeTextKindOutputTranscript];
		if ([content[@"turnComplete"] boolValue]) {
			[self reportTurn];
		}
	}
	NSDictionary *toolCall = [event[@"toolCall"] isKindOfClass:NSDictionary.class] ? event[@"toolCall"] : nil;
	for (NSDictionary *call in [toolCall[@"functionCalls"] isKindOfClass:NSArray.class] ? toolCall[@"functionCalls"] : @[]) {
		if ([call isKindOfClass:NSDictionary.class]) {
			[self reportToolCall:call[@"id"] name:call[@"name"] arguments:call[@"args"]];
		}
	}
	if (event[@"error"] != nil) {
		[self reportError:event];
	}
}

- (void)reportAudio:(id)encoded
{
	NSData *pcm = [encoded isKindOfClass:NSString.class]
		? [[NSData alloc] initWithBase64EncodedString:encoded options:NSDataBase64DecodingIgnoreUnknownCharacters] : nil;
	if (pcm != nil && self.audioHandler != nil) {
		self.audioHandler(pcm);
	}
}

- (void)reportText:(id)text kind:(NFKRealtimeTextKind)kind
{
	if ([text isKindOfClass:NSString.class] && [text length] > 0 && self.textHandler != nil) {
		self.textHandler(text, kind);
	}
}

// Arguments arrive as a JSON string (OpenAI, xAI) or an object (Gemini); both are handed on parsed.
- (void)reportToolCall:(id)identifier name:(id)name arguments:(id)arguments
{
	if (self.toolCallHandler == nil) {
		return;
	}
	id parsed = arguments;
	if ([arguments isKindOfClass:NSString.class]) {
		parsed = [NSJSONSerialization JSONObjectWithData:[arguments dataUsingEncoding:NSUTF8StringEncoding] options:0 error:NULL];
	}
	self.toolCallHandler(@{ @"id": [identifier isKindOfClass:NSString.class] ? identifier : @"",
							@"name": [name isKindOfClass:NSString.class] ? name : @"",
							@"arguments": [parsed isKindOfClass:NSDictionary.class] ? parsed : @{} });
}

- (void)reportTurn
{
	if (self.turnHandler != nil) {
		self.turnHandler();
	}
}

- (void)reportError:(NSDictionary *)event
{
	if (self.errorHandler == nil) {
		return;
	}
	id error = event[@"error"];
	NSString *message = [error isKindOfClass:NSDictionary.class] && [error[@"message"] isKindOfClass:NSString.class] ? error[@"message"]
					  : [error isKindOfClass:NSString.class] ? error
					  : [event[@"message"] isKindOfClass:NSString.class] ? event[@"message"] : @"the realtime service reported an error";
	self.errorHandler([NFKRemoteTransport errorWithCode:kNFKError_InferenceBackendFailure reason:message]);
}

@end
