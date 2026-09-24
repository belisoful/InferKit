//
//  NFKRealtimeSession.h
//  InferKit
//

#ifndef NFKRealtimeSession_h
#define NFKRealtimeSession_h

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@class NFKRemoteProvider;

/*!
	@protocol   NFKRealtimeSocket
	@abstract   A bidirectional message channel a realtime session speaks over.
	@discussion NFKRealtimeWebSocket is the shipped one; a test or an alternative transport adopts
				it. The handlers run on the socket's own queue. Introduced in InferKit 0.4.0.
*/
@protocol NFKRealtimeSocket <NSObject>

/*! Opens the channel. messageHandler receives each text frame as text and each binary frame as data;
	closeHandler runs once, with an error when the channel failed. */
- (void)openWithMessageHandler:(void (^)(NSString * _Nullable text, NSData * _Nullable data))messageHandler
				  closeHandler:(void (^)(NSError * _Nullable error))closeHandler;

/*! Sends a text frame. */
- (void)sendText:(NSString *)text;

/*! Sends a binary frame. */
- (void)sendData:(NSData *)data;

/*! Closes the channel. */
- (void)close;

@end

/*!
	@class      NFKRealtimeWebSocket
	@abstract   An NFKRealtimeSocket on NSURLSessionWebSocketTask.
	@discussion Introduced in InferKit 0.4.0.
*/
@interface NFKRealtimeWebSocket : NSObject <NFKRealtimeSocket>

/*! A socket for the request, whose URL is the ws:// or wss:// endpoint and whose headers carry the
	key. */
- (instancetype)initWithRequest:(NSURLRequest *)request session:(nullable NSURLSession *)session NS_DESIGNATED_INITIALIZER;
- (instancetype)init NS_UNAVAILABLE;

@end

/*!
	@enum       NFKRealtimeAPIStyle
	@abstract   The realtime protocol a session speaks.
	@constant   NFKRealtimeAPIStyleOpenAIConversation OpenAI Realtime (wss …/v1/realtime): speech and
				text in, speech and text out, function tools.
	@constant   NFKRealtimeAPIStyleOpenAITranscription OpenAI Realtime in a transcription session.
	@constant   NFKRealtimeAPIStyleOpenAITranslation OpenAI's realtime translation
				(wss …/v1/realtime/translations): speech in one language, speech and text in another.
	@constant   NFKRealtimeAPIStyleXAIConversation xAI's Voice Agent (wss …/v1/realtime), OpenAI's
				event names.
	@constant   NFKRealtimeAPIStyleXAITranscription xAI's streaming speech-to-text (wss …/v1/stt):
				audio as binary frames.
	@constant   NFKRealtimeAPIStyleXAISpeech xAI's streaming text-to-speech (wss …/v1/tts).
	@constant   NFKRealtimeAPIStyleGeminiLive Gemini's Live API (BidiGenerateContent): setup, realtime
				input, server content with audio parts and transcriptions, tool calls.
	@constant   NFKRealtimeAPIStyleGeminiLiveMusic Lyria realtime (BidiGenerateMusic): weighted prompts,
				a generation config, and playback control; 48 kHz stereo audio out.
	@constant   NFKRealtimeAPIStyleMistralTranscription Mistral's realtime transcription
				(wss …/v1/audio/transcriptions/realtime).
	@constant   NFKRealtimeAPIStyleTogetherTranscription Together's realtime transcription
				(wss …/v1/realtime).
	@constant   NFKRealtimeAPIStyleTogetherSpeech Together's streaming text-to-speech
				(wss …/v1/audio/speech/websocket).
	@constant   NFKRealtimeAPIStyleVLLMTranscription vLLM's realtime transcription (ws …/v1/realtime).
	Introduced in InferKit 0.4.0.
*/
typedef NS_ENUM(NSInteger, NFKRealtimeAPIStyle) {
	NFKRealtimeAPIStyleOpenAIConversation = 0,
	NFKRealtimeAPIStyleOpenAITranscription,
	NFKRealtimeAPIStyleOpenAITranslation,
	NFKRealtimeAPIStyleXAIConversation,
	NFKRealtimeAPIStyleXAITranscription,
	NFKRealtimeAPIStyleXAISpeech,
	NFKRealtimeAPIStyleGeminiLive,
	NFKRealtimeAPIStyleGeminiLiveMusic,
	NFKRealtimeAPIStyleMistralTranscription,
	NFKRealtimeAPIStyleTogetherTranscription,
	NFKRealtimeAPIStyleTogetherSpeech,
	NFKRealtimeAPIStyleVLLMTranscription,
};

/*!
	@enum       NFKRealtimeTextKind
	@abstract   What a piece of text a session reports is.
	@constant   NFKRealtimeTextKindResponse The model's text reply.
	@constant   NFKRealtimeTextKindOutputTranscript The transcript of the model's spoken reply.
	@constant   NFKRealtimeTextKindInputTranscript A piece of the transcript of the audio sent in, which
				a later piece may revise.
	@constant   NFKRealtimeTextKindInputTranscriptFinal A finished stretch of the input transcript.
	Introduced in InferKit 0.4.0.
*/
typedef NS_ENUM(NSInteger, NFKRealtimeTextKind) {
	NFKRealtimeTextKindResponse = 0,
	NFKRealtimeTextKindOutputTranscript,
	NFKRealtimeTextKindInputTranscript,
	NFKRealtimeTextKindInputTranscriptFinal,
};

/*!
	@class      NFKRealtimeSession
	@abstract   A live, two-way session with a hosted model over a WebSocket: a spoken conversation,
				streaming transcription, streaming speech, live translation, or live music.
	@discussion The request-and-reply backends answer once; a realtime session stays open and both
				sides talk while it does. One class speaks every provider's protocol, chosen by
				apiStyle, behind the same calls and handlers:

				- appendAudio: sends 16-bit little-endian mono PCM at inputSampleRate
				- commitAudio ends an utterance where the service waits for one; finishInput ends the
				  stream of audio or text
				- sendText: sends a user turn in a conversation, or text to speak in a speech session
				- requestResponse asks a conversation for a reply now
				- sendToolResult:forCallIdentifier: answers a function call
				- setMusicPrompts:, setMusicConfiguration:, and playMusic / pauseMusic / stopMusic
				  steer a live-music session

				What comes back arrives on the handlers: audioHandler with 16-bit PCM at
				outputSampleRate, textHandler with each piece of text and its kind, toolCallHandler
				with {id, name, arguments}, turnHandler when a reply ends, errorHandler, and
				eventHandler with every event as the service sent it. The handlers run on the
				socket's queue; hop to the main queue before touching UI.

				OpenAI's WebRTC-only GPT-Live sessions are not reachable here: WebRTC needs a media
				stack outside the core's dependencies. Introduced in InferKit 0.4.0.
*/
@interface NFKRealtimeSession : NSObject

/*! The protocol the session speaks. */
@property (nonatomic, assign) NFKRealtimeAPIStyle apiStyle;

/*! The service's ws:// or wss:// endpoint, without the query the session adds. */
@property (nonatomic, copy, nullable) NSURL *endpointURL;

/*! The key, sent the way the service reads it. */
@property (nonatomic, copy, nullable) NSString *apiKey;

/*! The model, for example gpt-realtime-2.1, grok-voice-latest, gemini-3.8-live, lyria-realtime-exp. */
@property (nonatomic, copy, nullable) NSString *modelName;

/*! The system instructions of a conversation. */
@property (nonatomic, copy, nullable) NSString *instructions;

/*! The voice a spoken reply uses. */
@property (nonatomic, copy, nullable) NSString *voice;

/*! The language: the spoken language a transcription expects, or the language a translation speaks. */
@property (nonatomic, copy, nullable) NSString *language;

/*! Function tools a conversation may call, in the contract's {name, description, parameters} shape. */
@property (nonatomic, copy, nullable) NSArray<NSDictionary *> *tools;

/*! The sample rate of the audio appendAudio: sends. Each style's default is the rate its service
	expects: 24000 for OpenAI and xAI conversations, 16000 for Gemini Live and the transcription
	services. */
@property (nonatomic, assign) NSUInteger inputSampleRate;

/*! The sample rate of the audio audioHandler receives: 24000, or 48000 (stereo) for live music. */
@property (nonatomic, readonly) NSUInteger outputSampleRate;

/*! Fields merged into the session's configuration message as the service spells them. */
@property (nonatomic, copy, nullable) NSDictionary<NSString *, id> *extraConfiguration;

/*! Whether the socket is open. */
@property (nonatomic, readonly, getter=isConnected) BOOL connected;

@property (nonatomic, copy, nullable) void (^audioHandler)(NSData *pcm);
@property (nonatomic, copy, nullable) void (^textHandler)(NSString *text, NFKRealtimeTextKind kind);
@property (nonatomic, copy, nullable) void (^toolCallHandler)(NSDictionary<NSString *, id> *call);
@property (nonatomic, copy, nullable) void (^turnHandler)(void);
@property (nonatomic, copy, nullable) void (^errorHandler)(NSError *error);
@property (nonatomic, copy, nullable) void (^eventHandler)(NSDictionary<NSString *, id> *event);
@property (nonatomic, copy, nullable) void (^closeHandler)(NSError * _Nullable error);

/*!
	@method     sessionForProvider:apiStyle:apiKey:modelName:
	@abstract   A session pointed at the provider's realtime endpoint for the style, or nil when the
				provider does not serve it.
	@discussion openai serves the three OpenAI styles; xai its conversation, transcription, and
				speech; gemini Live and live music; mistral and vllm transcription; together
				transcription and speech.
*/
+ (nullable instancetype)sessionForProvider:(NFKRemoteProvider *)provider
								   apiStyle:(NFKRealtimeAPIStyle)apiStyle
									 apiKey:(nullable NSString *)apiKey
								  modelName:(nullable NSString *)modelName;

/*! Opens the socket and sends the session's configuration. */
- (void)connect;

/*! Sends a chunk of 16-bit little-endian mono PCM at inputSampleRate. */
- (void)appendAudio:(NSData *)pcm;

/*! Ends an utterance: commits the buffered audio where the service waits for a commit. */
- (void)commitAudio;

/*! Ends the input stream: no more audio (or, for speech, text) follows. */
- (void)finishInput;

/*! Sends a user turn in a conversation, or text to speak in a speech session. */
- (void)sendText:(NSString *)text;

/*! Asks a conversation for a reply now. */
- (void)requestResponse;

/*! Answers the function call the service named by its identifier. */
- (void)sendToolResult:(NSString *)output forCallIdentifier:(NSString *)callIdentifier name:(nullable NSString *)name;

/*! Sets a live-music session's prompts, each {text, weight}. */
- (void)setMusicPrompts:(NSArray<NSDictionary *> *)prompts;

/*! Sets a live-music session's generation config (bpm, density, brightness, scale, guidance, …). */
- (void)setMusicConfiguration:(NSDictionary<NSString *, id> *)configuration;

- (void)playMusic;
- (void)pauseMusic;
- (void)stopMusic;

/*! Sends an event as the service spells it, for a feature the calls above do not name. */
- (void)sendEvent:(NSDictionary<NSString *, id> *)event;

/*! Closes the socket. */
- (void)close;

/*! The socket for the connection request. The default is an NFKRealtimeWebSocket; a test overrides
	it. */
- (id<NFKRealtimeSocket>)socketForRequest:(NSURLRequest *)request;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKRealtimeSession_h */
