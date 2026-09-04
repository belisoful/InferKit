//
//  NFKRemoteSpeechBackend.h
//  InferKit
//

#ifndef NFKRemoteSpeechBackend_h
#define NFKRemoteSpeechBackend_h

#import <Foundation/Foundation.h>
#import <InferKit/NFKInferenceBackend.h>

NS_ASSUME_NONNULL_BEGIN

@class NFKRemoteProvider;

/*!
	@class      NFKRemoteSpeechBackend
	@abstract   An inference backend that calls an OpenAI-compatible text-to-speech endpoint.
	@discussion POST /audio/speech. The request reads NFKInputPrompt (or the joined content of
				NFKInputMessages) as the text, sends the model, voice, and response format, and
				folds the request parameters into the body so a caller sets speed or instructions
				by name. The reply is the audio itself, written to a file in outputDirectoryURL and
				returned as an NFKAudioAsset under NFKOutputAudio, which is what the on-device
				NFKMLXSpeechBackend answers with. The format defaults to wav, the same container
				that backend writes, so the two are interchangeable downstream.

				A voice is required by every service that serves this endpoint and there is no
				default, for the reason a model name has none: the names differ per provider and
				change. runInferenceForRequest: blocks; run it off the render thread. Introduced in
				InferKit 0.3.0.
*/
@interface NFKRemoteSpeechBackend : NSObject <NFKInferenceBackend>

/*! The speech endpoint, for example https://api.openai.com/v1/audio/speech. */
@property (nonatomic, copy, nullable) NSURL *endpointURL;

/*! The bearer token sent as Authorization, when the endpoint needs one. */
@property (nonatomic, copy, nullable) NSString *apiKey;

/*! The model name sent in the request body. */
@property (nonatomic, copy, nullable) NSString *modelName;

/*! The voice name sent in the request body. Required; a request parameter named "voice" overrides it. */
@property (nonatomic, copy, nullable) NSString *voice;

/*! The container asked for and the file extension written: wav (default), mp3, opus, aac, flac, or pcm. */
@property (nonatomic, copy) NSString *responseFormat;

/*! Where the audio files land. Defaults to an InferKit directory under the temporary directory. */
@property (nonatomic, copy, nullable) NSURL *outputDirectoryURL;

/*! The request timeout in seconds. Defaults to 120. */
@property (nonatomic, assign) NSTimeInterval timeout;

/*! The session used for the call. Defaults to the shared session. */
@property (nonatomic, strong) NSURLSession *session;

+ (instancetype)backendWithEndpointURL:(nullable NSURL *)endpointURL;

/*!
	@method     backendForProvider:apiKey:modelName:voice:
	@abstract   A backend pointed at the provider's speech endpoint.
	@discussion Returns nil for Anthropic, which serves no speech endpoint. Of the OpenAI-compatible
				presets, openai, groq, together, xai, mistral, and openrouter were verified to serve
				the path at release; gemini and the local runners do not, and deepseek could not be
				told, so those fail at the first call with the provider's own answer.
*/
+ (nullable instancetype)backendForProvider:(NFKRemoteProvider *)provider
									 apiKey:(nullable NSString *)apiKey
								  modelName:(nullable NSString *)modelName
									  voice:(nullable NSString *)voice;

/*! The transport seam. The default delegates to NFKRemoteTransport; a test overrides it. */
- (nullable NSData *)sendRequest:(NSURLRequest *)request
						response:(NSHTTPURLResponse * _Nullable * _Nullable)outResponse
						   error:(NSError * _Nullable *)outError;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKRemoteSpeechBackend_h */
