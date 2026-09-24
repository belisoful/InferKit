//
//  NFKRemoteSpeechBackend.h
//  InferKit
//

#ifndef NFKRemoteSpeechBackend_h
#define NFKRemoteSpeechBackend_h

#import <Foundation/Foundation.h>
#import "NFKInferenceBackend.h"

NS_ASSUME_NONNULL_BEGIN

@class NFKRemoteProvider;

/*!
	@enum       NFKRemoteSpeechAPIStyle
	@abstract   The wire shape of a hosted text-to-speech service.
	@constant   NFKRemoteSpeechAPIStyleOpenAI POST /audio/speech with model, input, voice, and
				response_format, answered with the audio bytes; the shape OpenAI, Groq, Together, and
				OpenRouter serve. Streamed as SSE audio deltas.
	@constant   NFKRemoteSpeechAPIStyleMistral Mistral's /audio/speech: voice_id or a reference clip
				as ref_audio, answered with JSON {audio_data} in base64; streamed as
				speech.audio.delta events.
	@constant   NFKRemoteSpeechAPIStyleXAI xAI's POST /v1/tts: text, voice_id, language, and an
				output_format object, answered with the audio bytes.
	Introduced in InferKit 0.4.0.
*/
typedef NS_ENUM(NSInteger, NFKRemoteSpeechAPIStyle) {
	NFKRemoteSpeechAPIStyleOpenAI = 0,
	NFKRemoteSpeechAPIStyleMistral,
	NFKRemoteSpeechAPIStyleXAI,
};

/*!
	@class      NFKRemoteVoice
	@abstract   A voice a speech service offers.
	@discussion Introduced in InferKit 0.4.0.
*/
@interface NFKRemoteVoice : NSObject

/*! The identifier a request names the voice by. */
@property (nonatomic, copy, readonly) NSString *identifier;

/*! The display name, or nil when the service gives none. */
@property (nonatomic, copy, readonly, nullable) NSString *name;

/*! The languages the voice speaks, as the service names them. */
@property (nonatomic, copy, readonly) NSArray<NSString *> *languages;

/*! The service's own record of the voice. */
@property (nonatomic, copy, readonly) NSDictionary *raw;

- (instancetype)initWithIdentifier:(NSString *)identifier
							  name:(nullable NSString *)name
						 languages:(NSArray<NSString *> *)languages
							   raw:(NSDictionary *)raw NS_DESIGNATED_INITIALIZER;
- (instancetype)init NS_UNAVAILABLE;

@end

/*!
	@class      NFKRemoteSpeechBackend
	@abstract   An inference backend that speaks text through a hosted text-to-speech endpoint.
	@discussion The request reads NFKInputPrompt (or the joined content of NFKInputMessages) as the
				text. The contract keys map onto every style:

				- voice → the service's voice field (voice, or voice_id)
				- NFKInputVoiceReference → a reference clip the service clones the voice from
				  (Mistral ref_audio, OpenRouter input_references)
				- NFKParameterSourceLanguage → the language of the text, where the service takes one
				- NFKParameterSampleRate → the output sample rate, where the service takes one

				Every other parameter goes out under its own name (speed, instructions). The reply
				is written to a file in outputDirectoryURL and returned as an NFKAudioAsset under
				NFKOutputAudio, which is what the on-device NFKMLXSpeechBackend answers with. The
				format defaults to wav, the container that backend writes. Text longer than
				maximumInputLength is spoken in pieces cut at sentence or word ends and the pieces
				joined. streams asks for the service's event stream; the audio received so far rides
				on the job's partialResult under NFKOutputAudio as NSData. A voice is required by
				every service here except the ones that take a reference clip, and there is no
				default: the names differ per provider and change. runInferenceForRequest: blocks;
				run it off the render thread. Introduced in InferKit 0.3.0.
*/
@interface NFKRemoteSpeechBackend : NSObject <NFKInferenceBackend>

/*! The speech endpoint, for example https://api.openai.com/v1/audio/speech. */
@property (nonatomic, copy, nullable) NSURL *endpointURL;

/*! The wire shape the backend speaks. Defaults to NFKRemoteSpeechAPIStyleOpenAI. Introduced in
	InferKit 0.4.0. */
@property (nonatomic, assign) NFKRemoteSpeechAPIStyle apiStyle;

/*! The bearer token sent as Authorization, when the endpoint needs one. */
@property (nonatomic, copy, nullable) NSString *apiKey;

/*! The model name sent in the request body. */
@property (nonatomic, copy, nullable) NSString *modelName;

/*! The voice sent in the request body. A request parameter named "voice" overrides it. */
@property (nonatomic, copy, nullable) NSString *voice;

/*! The container asked for and the file extension written: wav (default), mp3, opus, aac, flac, or pcm. */
@property (nonatomic, copy) NSString *responseFormat;

/*! The longest text one request carries, or 0 for no limit; longer text is spoken in pieces and
	joined. backendForProvider: sets 200 for Groq, whose Orpheus voices refuse more. Introduced in
	InferKit 0.4.0. */
@property (nonatomic, assign) NSUInteger maximumInputLength;

/*! Asks for the service's event stream when the request is submitted as a job. Off by default.
	Introduced in InferKit 0.4.0. */
@property (nonatomic, assign) BOOL streams;

/*! Where the audio files land. Defaults to an InferKit directory under the temporary directory. */
@property (nonatomic, copy, nullable) NSURL *outputDirectoryURL;

/*! The request timeout in seconds. Defaults to 120. */
@property (nonatomic, assign) NSTimeInterval timeout;

/*! The session used for the call. Defaults to the shared session. */
@property (nonatomic, strong) NSURLSession *session;

+ (instancetype)backendWithEndpointURL:(nullable NSURL *)endpointURL;

/*!
	@method     backendForProvider:apiKey:modelName:voice:
	@abstract   A backend pointed at the provider's speech endpoint in its style, or nil for a
				provider that serves none.
	@discussion openai, groq, together, and openrouter take the OpenAI style; mistral its own; xai its
				/v1/tts. anthropic, gemini (its OpenAI layer has no audio path), deepseek, typesafe,
				and the local runners return nil.
*/
+ (nullable instancetype)backendForProvider:(NFKRemoteProvider *)provider
									 apiKey:(nullable NSString *)apiKey
								  modelName:(nullable NSString *)modelName
									  voice:(nullable NSString *)voice;

/*! Runs the request as a job: streamed when streams is set and the style streams, and otherwise the
	blocking call on a background queue. Introduced in InferKit 0.4.0. */
- (NFKInferenceJob *)submitInferenceJobForRequest:(NFKInferenceRequest *)request;

/*!
	@method     availableVoicesWithError:
	@abstract   The voices the service lists, blocking.
	@discussion xai (GET /v1/tts/voices), mistral (GET /v1/audio/voices), and together
				(GET /v1/voices for modelName) list theirs; a service with a fixed, documented set
				(OpenAI, Groq) has no listing endpoint and answers kNFKError_InferenceUnsupported.
				Introduced in InferKit 0.4.0.
*/
- (nullable NSArray<NFKRemoteVoice *> *)availableVoicesWithError:(NSError * _Nullable *)error;

/*! The transport seam. The default delegates to NFKRemoteTransport; a test overrides it. */
- (nullable NSData *)sendRequest:(NSURLRequest *)request
						response:(NSHTTPURLResponse * _Nullable * _Nullable)outResponse
						   error:(NSError * _Nullable *)outError;

/*! The transport seam for the streamed form; the default delegates to NFKRemoteTransport and
	returns the block that cancels the request. Introduced in InferKit 0.4.0. */
- (void (^)(void))streamRequest:(NSURLRequest *)request
					lineHandler:(void (^)(NSString *line))lineHandler
			  completionHandler:(void (^)(NSHTTPURLResponse * _Nullable response,
										  NSData * _Nullable errorBody,
										  NSError * _Nullable error))completionHandler;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKRemoteSpeechBackend_h */
