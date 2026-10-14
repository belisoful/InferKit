//
//  NFKGeminiInteractionsBackend.h
//  InferKit
//

#ifndef NFKGeminiInteractionsBackend_h
#define NFKGeminiInteractionsBackend_h

#import <Foundation/Foundation.h>
#import <InferKit/NFKInferenceBackend.h>

NS_ASSUME_NONNULL_BEGIN

/*! The output key under which an agent's sandbox id comes back as an NSString: the environment
	NFKGeminiEnvironmentStore reads the agent's files from, and the "environment" parameter that
	continues in it. Introduced in InferKit 0.4.0. */
extern NSString * const NFKGeminiOutputEnvironmentIdentifier;

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
				theirs, and NFKParameterPreviousResponseIdentifier previous_interaction_id. A
				"processing" parameter (agentic) rides on each video block. service_tier,
				safety_settings, webhook_config, labels, store, environment, and agent_config go at the
				top of the interaction under their own names; every other parameter goes into
				generation_config under its own name.

				agentName runs one of Google's agents in place of a model, always in the background:

				- deep-research-preview-04-2026, deep-research-max-preview-04-2026 → a cited report;
				  NFKParameterReasoningEffort asks for its thinking summaries
				- antigravity-preview-09-2026 → a coding agent in a hosted sandbox on modelName
				  (gemini-3.8-flash unless set, or gemini-4-argon); it writes files there, which
				  NFKGeminiEnvironmentStore lists and downloads by the id under
				  NFKGeminiOutputEnvironmentIdentifier

				NFKParameterMCPServers become mcp_server tools the service calls over streamable HTTP
				(models, Deep Research, and Antigravity), so a hosted agent can call an
				NFKInferenceServer's MCP endpoint behind a public HTTPS tunnel and reach local models;
				the server's name is lowercased to the letters, digits, _ and - Google accepts.

				The agent's own tool calls come back with their results under
				NFKOutputServerToolResults; a call to one of the caller's function tools comes back
				under NFKOutputToolCalls, and its answer continues the interaction through
				NFKParameterPreviousResponseIdentifier and the "environment" parameter.

				NFKParameterMaxTokens above 65536 is a long decode (gemini-4-argon): the interaction
				runs in the background and its text is joined across the model_output steps of every
				hop.

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

/*! The model, for example gemini-4-argon, gemini-3.8-flash, gemini-nano-banana-2.1,
	gemini-3.8-flash-tts, gemini-3.5-transcribe, lyria-3.5, or gemini-omni-1.1-flash. With agentName set
	it is the model an Antigravity agent runs on. */
@property (nonatomic, copy, nullable) NSString *modelName;

/*! An agent to run in place of a model, for example deep-research-preview-04-2026 or
	antigravity-preview-09-2026. Sent as agent, with agent_config in place of generation_config.
	Introduced in InferKit 0.4.0. */
@property (nonatomic, copy, nullable) NSString *agentName;

/*! The voice a speech reply speaks in: a prebuilt name such as Kore, or a custom voice's voice_… id or
	voicekey_… key from NFKGeminiVoiceLibrary. A request parameter named "voice" overrides it; a
	"speakers" parameter (NSArray of {speaker, voice}, prebuilt voices only, at most two) asks for
	several and is sent as speech_config {speakers}. */
@property (nonatomic, copy, nullable) NSString *voice;

/*! Creates the interaction in the background and polls it until it ends. Off by default; an agent and a
	long decode run in the background regardless. */
@property (nonatomic, assign) BOOL runsInBackground;

/*! Seconds between polls of a background interaction. Defaults to 5. */
@property (nonatomic, assign) NSTimeInterval pollInterval;

/*! Seconds a background interaction may run, on the uptime clock. Past it the interaction is cancelled
	on the service and the call fails; the last poll's sleep and request timeout are cut to the time
	left. Defaults to 3600; 0 waits without a limit. Three failed polls in a row also end the
	wait, and that error names the interaction, which the service keeps stored. Introduced in
	InferKit 0.4.0. */
@property (nonatomic, assign) NSTimeInterval backgroundTimeout;

/*! Where audio and video replies land. Defaults to an InferKit directory under the temporary
	directory. */
@property (nonatomic, copy, nullable) NSURL *outputDirectoryURL;

/*! The request timeout in seconds. Defaults to 600. */
@property (nonatomic, assign) NSTimeInterval timeout;

/*! The session used for the calls. Defaults to the shared session. */
@property (nonatomic, strong) NSURLSession *session;

/*! A backend for the model with the key. */
+ (instancetype)backendWithAPIKey:(nullable NSString *)apiKey modelName:(nullable NSString *)modelName;

/*! Streams the text into the job's partialResult, or, for a background interaction, polls it;
	cancelling the job cancels the request or the background interaction. runInferenceForRequest:error:
	blocks for a background interaction's whole run, up to backgroundTimeout, so an agent goes through
	this method when the caller needs to cancel it. */
- (NFKInferenceJob *)submitInferenceJobForRequest:(NFKInferenceRequest *)request;

/*! The transport seam. The default delegates to NFKRemoteTransport; a test overrides it. */
- (nullable NSData *)sendRequest:(NSURLRequest *)request
						response:(NSHTTPURLResponse * _Nullable * _Nullable)outResponse
						   error:(NSError * _Nullable *)outError;

/*! The transport seam for the streamed form; the default delegates to NFKRemoteTransport and
	returns the block that cancels the request. */
- (void (^ NS_SWIFT_SENDABLE)(void))streamRequest:(NSURLRequest *)request
					lineHandler:(void (^ NS_SWIFT_SENDABLE)(NSString *line))lineHandler
			  completionHandler:(void (^ NS_SWIFT_SENDABLE)(NSHTTPURLResponse * _Nullable response,
										  NSData * _Nullable errorBody,
										  NSError * _Nullable error))completionHandler;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKGeminiInteractionsBackend_h */
