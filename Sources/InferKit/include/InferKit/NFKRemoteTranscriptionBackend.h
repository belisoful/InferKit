//
//  NFKRemoteTranscriptionBackend.h
//  InferKit
//

#ifndef NFKRemoteTranscriptionBackend_h
#define NFKRemoteTranscriptionBackend_h

#import <Foundation/Foundation.h>
#import "NFKInferenceBackend.h"

NS_ASSUME_NONNULL_BEGIN

@class NFKRemoteProvider;

/*!
	@enum       NFKRemoteTranscriptionAPIStyle
	@abstract   The wire shape of a hosted speech-to-text service.
	@constant   NFKRemoteTranscriptionAPIStyleOpenAI POST /audio/transcriptions (and /audio/translations)
				as multipart, the shape OpenAI, Groq, Together, OpenRouter, and vLLM serve: the verbose
				reply's segments[] and words[], diarized_json for a diarizing model, and SSE
				transcript.text.delta events when streamed.
	@constant   NFKRemoteTranscriptionAPIStyleMistral Mistral's /audio/transcriptions: no
				response_format field, diarize and context_bias fields, segments[] with speaker_id,
				and transcription.* SSE events when streamed.
	@constant   NFKRemoteTranscriptionAPIStyleXAI xAI's POST /v1/stt: multipart with the file last,
				diarize and keyterm fields, and a reply of words[] with speakers.
	Introduced in InferKit 0.4.0.
*/
typedef NS_ENUM(NSInteger, NFKRemoteTranscriptionAPIStyle) {
	NFKRemoteTranscriptionAPIStyleOpenAI = 0,
	NFKRemoteTranscriptionAPIStyleMistral,
	NFKRemoteTranscriptionAPIStyleXAI,
};

/*!
	@class      NFKRemoteTranscriptionBackend
	@abstract   An inference backend that transcribes audio to text through a hosted or local
				speech-to-text endpoint.
	@discussion Depends only on Foundation, so InferKit ships it. The request supplies audio under
				NFKInputAudio: an NFKAudioAsset (a local file is uploaded; a hosted URL goes by
				reference where the service takes one and is fetched and uploaded where it does
				not) or NSData holding the encoded file. The contract keys map onto every style:

				- NFKInputPrompt → the prompt that primes the decoder, where the service takes one
				- NFKParameterSourceLanguage → the spoken language
				- NFKParameterSpeakerDiarization → the service's diarization; the speaker lands on
				  each NFKAudioSegment's speaker
				- NFKParameterWordTimestamps → word timing; the words come back under NFKOutputWords
				- NFKParameterVocabulary → the service's keyword, key-term, or context-bias list

				Every other parameter goes out as a form field under its own name. The transcript
				comes back under NFKOutputText and the parsed reply under NFKOutputStructured.
				emitsTimestamps adds the segments under NFKOutputSegments as NFKAudioSegments, which
				is what the on-device Whisper backend emits, so the two are interchangeable.
				translates sends the audio to the sibling translations endpoint, which answers in
				English whatever the audio's language. streams asks the service for its event
				stream, and each delta grows the job's partialResult.

				runInferenceForRequest: blocks until the call returns, so a caller runs it off the
				render thread. isReady reports whether an endpoint is set.
*/
@interface NFKRemoteTranscriptionBackend : NSObject <NFKInferenceBackend>

/*! The transcription endpoint, for example a hosted API or a localhost server URL. */
@property (nonatomic, copy, nullable) NSURL *endpointURL;

/*! The wire shape the backend speaks. Defaults to NFKRemoteTranscriptionAPIStyleOpenAI. Introduced
	in InferKit 0.4.0. */
@property (nonatomic, assign) NFKRemoteTranscriptionAPIStyle apiStyle;

/*! The form field that carries a hosted audio URL in place of the file (url on Groq, file on
	Together), or nil for a service that takes an upload only; the backend then fetches the audio and
	uploads it. backendForProvider: sets it. Introduced in InferKit 0.4.0. */
@property (nonatomic, copy, nullable) NSString *audioURLFieldName;

/*! The bearer token sent as Authorization, when the endpoint needs one. */
@property (nonatomic, copy, nullable) NSString *apiKey;

/*! The model name sent as the multipart `model` field, for example a Whisper model name. */
@property (nonatomic, copy, nullable) NSString *modelName;

/*! Asks for per-segment times and adds them under NFKOutputSegments. Off by default. Introduced in InferKit 0.3.0. */
@property (nonatomic, assign) BOOL emitsTimestamps;

/*! Sends the audio to the translations endpoint beside the transcriptions one, for an English
	transcript of any language. The OpenAI style only. Off by default. Introduced in InferKit 0.3.0. */
@property (nonatomic, assign) BOOL translates;

/*! Asks for the service's event stream when the request is submitted as a job, so the transcript
	grows on the job's partialResult as it arrives. The OpenAI and Mistral styles. Off by default,
	because some models refuse it. Introduced in InferKit 0.4.0. */
@property (nonatomic, assign) BOOL streams;

/*! The request timeout in seconds. Defaults to 60. */
@property (nonatomic, assign) NSTimeInterval timeout;

/*! The session used for the call. Defaults to the shared session. */
@property (nonatomic, strong) NSURLSession *session;

+ (instancetype)backendWithEndpointURL:(nullable NSURL *)endpointURL;

/*!
	@method     backendForProvider:apiKey:modelName:
	@abstract   A backend pointed at the provider's speech-to-text endpoint in its style, or nil for a
				provider that serves none.
	@discussion openai, groq, together, openrouter, and vllm take the OpenAI style; mistral its own;
				xai its /v1/stt. anthropic, gemini (its OpenAI layer has no audio path), deepseek,
				typesafe, ollama, lmstudio, and llamacpp return nil. Introduced in InferKit 0.3.0.
*/
+ (nullable instancetype)backendForProvider:(NFKRemoteProvider *)provider
									 apiKey:(nullable NSString *)apiKey
								  modelName:(nullable NSString *)modelName;

/*! Runs the request as a job: streamed when streams is set and the style streams, and otherwise
	the blocking call on a background queue. Introduced in InferKit 0.4.0. */
- (NFKInferenceJob *)submitInferenceJobForRequest:(NFKInferenceRequest *)request;

/*!
	@method     sendRequest:response:error:
	@abstract   Performs the HTTP request synchronously and returns the body data, or nil.
	@discussion The transport seam. The default runs the request on the session and blocks until it
				completes. A test or an alternative transport overrides this.
*/
- (nullable NSData *)sendRequest:(NSURLRequest *)request
					   response:(NSHTTPURLResponse * _Nullable * _Nullable)outResponse
						  error:(NSError * _Nullable *)outError;

/*! The transport seam for the streamed form; the default delegates to NFKRemoteTransport and returns
	the block that cancels the request. Introduced in InferKit 0.4.0. */
- (void (^)(void))streamRequest:(NSURLRequest *)request
					lineHandler:(void (^)(NSString *line))lineHandler
			  completionHandler:(void (^)(NSHTTPURLResponse * _Nullable response,
										  NSData * _Nullable errorBody,
										  NSError * _Nullable error))completionHandler;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKRemoteTranscriptionBackend_h */
