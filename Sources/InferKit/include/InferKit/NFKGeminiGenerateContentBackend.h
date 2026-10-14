//
//  NFKGeminiGenerateContentBackend.h
//  InferKit
//

#ifndef NFKGeminiGenerateContentBackend_h
#define NFKGeminiGenerateContentBackend_h

#import <Foundation/Foundation.h>
#import <InferKit/NFKInferenceBackend.h>

NS_ASSUME_NONNULL_BEGIN

@class NFKRemoteProvider;

/*!
	@class      NFKGeminiGenerateContentBackend
	@abstract   An inference backend that calls Gemini's generateContent API
				(POST {endpointURL}/{model}:generateContent).
	@discussion generateContent is the Gemini API's original native chat surface. Google serves it
				beside the Interactions API, and gateways such as OpenCode Zen serve their Gemini models
				on it. The key travels as x-goog-api-key.

				The request reads NFKInputPrompt or NFKInputMessages. A system turn becomes
				systemInstruction, an assistant turn a "model" turn, and a tool answer a
				functionResponse part; consecutive answers share one turn. NFKInputImage,
				NFKInputImages, NFKInputAudio, NFKInputVideo, and NFKInputDocument ride as inlineData
				parts on the last user turn, and an NFKRemoteFile as a fileData part.

				- NFKParameterMaxTokens → maxOutputTokens
				- NFKParameterTemperature, NFKParameterTopP, NFKParameterTopK, NFKParameterSeed →
				  temperature, topP, topK, seed
				- NFKParameterStopSequences → stopSequences
				- NFKParameterReasoningEffort → thinkingConfig: light, moderate, and deep are the LOW,
				  MEDIUM, and HIGH thinking levels, another string is sent as the level, and a number is
				  a thinkingBudget in tokens; thought summaries are asked for
				- NFKParameterJSONSchema → responseFormat.text with the APPLICATION_JSON MIME type
				- NFKParameterTools → functionDeclarations with each tool's JSON Schema as
				  parametersJsonSchema; an entry without a name, such as {googleSearch: {}}, is sent as
				  a tool of its own
				- safetySettings, toolConfig, cachedContent, labels, serviceTier, and store → the
				  request's top level
				- every other parameter → generationConfig under its own name

				The reply's text comes back under NFKOutputText, thought summaries under
				NFKOutputReasoning, function calls under NFKOutputToolCalls, the parsed reply under
				NFKOutputStructured when a schema was asked for, an image part under NFKOutputImage
				(every one under NFKOutputImages), grounding and recitation sources under
				NFKOutputCitations, the token counts under NFKOutputUsage, and the reply body under
				NFKRemoteBackendRawKey. NFKOutputAssistantMessage records the model's parts with their
				thought signatures, which the API requires back on the next turn, so the same backend
				replays them exactly.

				A blocked prompt, and a reply stopped for safety, recitation, or prohibited content
				with no text, fail with kNFKError_InferenceRefused.

				A long decode (gemini-4-argon with maxOutputTokens beyond one hop) stops each hop with
				finishReason CONTINUATION and a continuationToken; the backend resends the request with
				the token until the reply is whole and joins the hops into one result.

				submitInferenceJobForRequest: streams (streamGenerateContent with alt=sse) and reports
				the text as it arrives. runInferenceForRequest: blocks; run it off the render thread.
				Introduced in InferKit 0.4.0.
*/
@interface NFKGeminiGenerateContentBackend : NSObject <NFKInferenceBackend>

/*! The models collection the request names its model under. Defaults to
	https://generativelanguage.googleapis.com/v1beta/models. */
@property (nonatomic, copy) NSURL *endpointURL;

/*! The key, sent as x-goog-api-key. */
@property (nonatomic, copy, nullable) NSString *apiKey;

/*! The model, for example gemini-4-argon or gemini-3.8-flash. Required. */
@property (nonatomic, copy, nullable) NSString *modelName;

/*! The request timeout in seconds. Defaults to 600. */
@property (nonatomic, assign) NSTimeInterval timeout;

/*! The session used for transport. Defaults to the shared session. */
@property (nonatomic, strong) NSURLSession *session;

+ (instancetype)backendWithEndpointURL:(nullable NSURL *)endpointURL;

/*!
	@method     backendForProvider:apiKey:modelName:
	@abstract   A backend on the provider's generateContent route, or nil for a provider without one.
	@discussion gemini serves it at https://generativelanguage.googleapis.com/v1beta/models, beside
				the OpenAI-compatible layer its preset points at. A provider that routes a model to
				NFKRemoteAPIStyleGeminiGenerateContent (OpenCode Zen) serves it under its own base.
				Every other preset returns nil.
*/
+ (nullable instancetype)backendForProvider:(NFKRemoteProvider *)provider
									 apiKey:(nullable NSString *)apiKey
								  modelName:(nullable NSString *)modelName;

/*! Streams the reply into the job's partialResult under NFKOutputText and finishes with the result
	runInferenceForRequest: returns. Cancelling the job cancels the request. */
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

#endif /* NFKGeminiGenerateContentBackend_h */
