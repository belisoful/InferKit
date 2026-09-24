//
//  NFKRemoteResponsesBackend.h
//  InferKit
//

#ifndef NFKRemoteResponsesBackend_h
#define NFKRemoteResponsesBackend_h

#import <Foundation/Foundation.h>
#import <InferKit/NFKInferenceBackend.h>

NS_ASSUME_NONNULL_BEGIN

@class NFKRemoteProvider;

/*!
	@class      NFKRemoteResponsesBackend
	@abstract   An inference backend that calls a Responses API endpoint (POST /responses).
	@discussion The Responses API is the successor to chat completions that OpenAI, xAI, Groq,
				DeepSeek, OpenRouter, and the local runners serve; some models (OpenAI's -pro and
				codex models) are served there only, and the service-run tools live there.

				The request reads NFKInputPrompt or NFKInputMessages (a leading system turn becomes
				instructions), with NFKInputImage, NFKInputImages, and NFKInputDocument as
				input_image and input_file parts on the last user turn. NFKParameterTools become
				function tools; an entry already in the wire shape (it has a type) passes through,
				which is how a caller asks for a built-in tool such as {type: web_search},
				{type: code_interpreter}, or {type: image_generation}. NFKParameterJSONSchema becomes
				text.format, NFKParameterReasoningEffort reasoning.effort with a summary asked for,
				NFKParameterMaxTokens max_output_tokens, and NFKParameterPreviousResponseIdentifier
				previous_response_id, which continues a conversation the service keeps. Every other
				parameter goes out under its own name.

				The reply's text comes back under NFKOutputText, its reasoning summaries under
				NFKOutputReasoning, function calls under NFKOutputToolCalls, a parsed schema reply
				under NFKOutputStructured, url_citation annotations under NFKOutputCitations, images
				from the image_generation tool under NFKOutputImage (and NFKOutputImages), each
				built-in tool's call under NFKOutputServerToolResults, the token counts under
				NFKOutputUsage, and the reply's id under NFKOutputResponseIdentifier. A refusal fails
				with kNFKError_InferenceRefused.

				runsInBackground submits the request as a background job and polls it until it ends,
				which is what a long -pro request needs. submitInferenceJobForRequest: streams the
				text and the reasoning summary into the job's partialResult. runInferenceForRequest:
				blocks; run it off the render thread. Introduced in InferKit 0.4.0.
*/
@interface NFKRemoteResponsesBackend : NSObject <NFKInferenceBackend>

/*! The responses endpoint, for example https://api.openai.com/v1/responses. */
@property (nonatomic, copy, nullable) NSURL *endpointURL;

/*! The bearer token sent as Authorization, when the endpoint needs one. */
@property (nonatomic, copy, nullable) NSString *apiKey;

/*! The model name sent in the request body. */
@property (nonatomic, copy, nullable) NSString *modelName;

/*! Runs the request as a background job the backend polls, rather than holding the connection open.
	Off by default. */
@property (nonatomic, assign) BOOL runsInBackground;

/*! Seconds between polls of a background job. Defaults to 2. */
@property (nonatomic, assign) NSTimeInterval pollInterval;

/*! The request timeout in seconds. Defaults to 600; a reasoning model can think for minutes. */
@property (nonatomic, assign) NSTimeInterval timeout;

/*! The session used for the calls. Defaults to the shared session. */
@property (nonatomic, strong) NSURLSession *session;

+ (instancetype)backendWithEndpointURL:(nullable NSURL *)endpointURL;

/*!
	@method     backendForProvider:apiKey:modelName:
	@abstract   A backend pointed at the provider's responses endpoint, or nil for a provider that
				serves none.
	@discussion openai, xai, groq, deepseek, openrouter, lmstudio, ollama, vllm, and llamacpp serve
				one; anthropic, gemini, mistral, together, and typesafe return nil.
*/
+ (nullable instancetype)backendForProvider:(NFKRemoteProvider *)provider
									 apiKey:(nullable NSString *)apiKey
								  modelName:(nullable NSString *)modelName;

/*! Streams the reply into the job's partialResult, or, with runsInBackground, polls the background
	job; cancelling the job closes the connection or cancels the background job. */
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

#endif /* NFKRemoteResponsesBackend_h */
