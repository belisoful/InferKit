//
//  NFKGeminiInteractionsBackend.h
//  InferKit
//

#ifndef NFKGeminiInteractionsBackend_h
#define NFKGeminiInteractionsBackend_h

#import <Foundation/Foundation.h>
#import <InferKit/NFKInferenceBackend.h>

NS_ASSUME_NONNULL_BEGIN

/*!
	@class      NFKGeminiInteractionsBackend
	@abstract   An inference backend that calls Gemini's Interactions API (POST /v1beta/interactions),
				Google's native surface for every Gemini mode.
	@discussion One endpoint serves text, image, speech, music, transcription, and video generation;
				the model and the request's outputModality choose what comes back. The request reads
				NFKInputPrompt or NFKInputMessages (a system turn becomes system_instruction, earlier
				turns become steps), with NFKInputImage, NFKInputImages, NFKInputAudio, NFKInputVideo,
				and NFKInputDocument as content blocks on the last user turn.

				- NFKModalityImage → an image reply (Nano Banana models): NFKParameterAspectRatio,
				  NFKParameterResolution (image_size, "1K"…"4K"), NFKParameterOutputFormat
				- NFKModalityAudio → speech (the TTS models, voice) or music (the Lyria models)
				- NFKModalityVideo → a clip (Omni Flash): NFKParameterAspectRatio,
				  NFKParameterResolution
				- otherwise text, with NFKParameterJSONSchema as a JSON response format

				Audio in with NFKParameterSpeakerDiarization, NFKParameterWordTimestamps,
				NFKParameterVocabulary, or NFKParameterSourceLanguage asks for a transcription
				(gemini-3.5-transcribe); the words come back under NFKOutputWords and the speaker
				turns under NFKOutputSegments. NFKParameterReasoningEffort becomes thinking_level with
				thought summaries under NFKOutputReasoning, NFKParameterTools the function tools (a
				wire-shaped entry such as {type: google_search} passes through),
				NFKParameterMaxTokens max_output_tokens, NFKParameterSeed and NFKParameterStopSequences
				theirs, and NFKParameterPreviousResponseIdentifier previous_interaction_id. Every other
				parameter goes into generation_config under its own name.

				The reply's text comes back under NFKOutputText, images under NFKOutputImage and
				NFKOutputImages, speech or music as an NFKAudioAsset under NFKOutputAudio (PCM is
				written as a WAV file), a clip as an NFKVideoAsset under NFKOutputVideo, function
				calls under NFKOutputToolCalls, citations under NFKOutputCitations, the token counts
				under NFKOutputUsage, and the interaction's id under NFKOutputResponseIdentifier.

				runsInBackground creates a background interaction and polls it, which a video or a
				song needs. submitInferenceJobForRequest: streams the text into the job's
				partialResult and then reads the finished interaction. runInferenceForRequest:
				blocks; run it off the render thread. Introduced in InferKit 0.4.0.
*/
@interface NFKGeminiInteractionsBackend : NSObject <NFKInferenceBackend>

/*! The interactions endpoint. Defaults to https://generativelanguage.googleapis.com/v1beta/interactions. */
@property (nonatomic, copy) NSURL *endpointURL;

/*! The Gemini API key, sent as x-goog-api-key. */
@property (nonatomic, copy, nullable) NSString *apiKey;

/*! The model, for example gemini-3.8-flash, gemini-3.1-flash-image, gemini-3.1-flash-tts-preview,
	gemini-3.5-transcribe, lyria-3.5, or gemini-omni-1.1-flash. */
@property (nonatomic, copy, nullable) NSString *modelName;

/*! The voice a speech reply speaks in, for example Kore. A request parameter named "voice"
	overrides it; a "speakers" parameter (NSArray of {speaker, voice}) asks for several. */
@property (nonatomic, copy, nullable) NSString *voice;

/*! Creates the interaction in the background and polls it until it ends. Off by default. */
@property (nonatomic, assign) BOOL runsInBackground;

/*! Seconds between polls of a background interaction. Defaults to 5. */
@property (nonatomic, assign) NSTimeInterval pollInterval;

/*! Where audio and video replies land. Defaults to an InferKit directory under the temporary
	directory. */
@property (nonatomic, copy, nullable) NSURL *outputDirectoryURL;

/*! The request timeout in seconds. Defaults to 600. */
@property (nonatomic, assign) NSTimeInterval timeout;

/*! The session used for the calls. Defaults to the shared session. */
@property (nonatomic, strong) NSURLSession *session;

/*! A backend for the model with the key. */
+ (instancetype)backendWithAPIKey:(nullable NSString *)apiKey modelName:(nullable NSString *)modelName;

/*! Streams the text into the job's partialResult, or, with runsInBackground, polls the background
	interaction; cancelling the job cancels the request or the background interaction. */
- (NFKInferenceJob *)submitInferenceJobForRequest:(NFKInferenceRequest *)request;

/*! The transport seam. The default delegates to NFKRemoteTransport; a test overrides it. */
- (nullable NSData *)sendRequest:(NSURLRequest *)request
						response:(NSHTTPURLResponse * _Nullable * _Nullable)outResponse
						   error:(NSError * _Nullable *)outError;

/*! The transport seam for the streamed form; the default delegates to NFKRemoteTransport and
	returns the block that cancels the request. */
- (void (^)(void))streamRequest:(NSURLRequest *)request
					lineHandler:(void (^)(NSString *line))lineHandler
			  completionHandler:(void (^)(NSHTTPURLResponse * _Nullable response,
										  NSData * _Nullable errorBody,
										  NSError * _Nullable error))completionHandler;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKGeminiInteractionsBackend_h */
