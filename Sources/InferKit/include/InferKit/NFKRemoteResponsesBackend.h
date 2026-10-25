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
@class NFKRemoteFile;

/*!
	@enum       NFKRemoteResponsesDialect
	@abstract   How a service spells the parts of the Responses shape it renames.
	@constant   NFKRemoteResponsesDialectStandard OpenAI's spelling, which every other service
				follows.
	@constant   NFKRemoteResponsesDialectPerplexityAgent Perplexity's Agent API (POST /v1/agent): a
				schema goes as response_format {type: json_schema, json_schema: {name, schema}},
				reasoning carries its effort alone, and a background job polls and cancels under
				/v1/agent/{id}. NFKParameterSkills goes out as skills, a custom skill as {type: custom,
				id}. Its search, fetch, and sandbox result items come back under
				NFKOutputServerToolResults.
	@discussion Introduced in InferKit 0.4.0.
*/
typedef NS_ENUM(NSInteger, NFKRemoteResponsesDialect) {
	NFKRemoteResponsesDialectStandard = 0,
	NFKRemoteResponsesDialectPerplexityAgent = 1,
};

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
				previous_response_id, which continues a conversation the service keeps.
				NFKParameterConversationKey becomes prompt_cache_key, NFKParameterPromptCacheRetention
				prompt_cache_retention, and NFKParameterPromptCacheMode, NFKParameterPromptCacheTTL,
				and NFKParameterPromptCachePrewarm prompt_cache_options. A cache breakpoint is a
				prompt_cache_breakpoint on an input_text part the caller writes, which passes
				through; the instructions take none, so a cached preamble goes in a developer turn
				whose content is parts. Every other parameter goes out under its own name.

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

/*! The service's spelling. Defaults to NFKRemoteResponsesDialectStandard;
	backendForProvider:apiKey:modelName: sets it for the perplexity preset. Introduced in InferKit
	0.4.0. */
@property (nonatomic, assign) NFKRemoteResponsesDialect dialect;

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
	@discussion openai, xai, groq, deepseek, openrouter, opencode, opencode-go, lmstudio, ollama,
				vllm, and llamacpp serve one; anthropic, gemini, mistral, together, and typesafe return
				nil. OpenCode answers this route only for the models it routes to Responses (see
				NFKRemoteProvider's modelAPIStyles). perplexity answers with its Agent API at /agent in
				the NFKRemoteResponsesDialectPerplexityAgent dialect, where a request names a model as
				provider/model, or names none and sets Perplexity's effort parameter.
*/
+ (nullable instancetype)backendForProvider:(NFKRemoteProvider *)provider
									 apiKey:(nullable NSString *)apiKey
								  modelName:(nullable NSString *)modelName;

/*!
	@method     outputFilesForResponseIdentifier:error:
	@abstract   The files a finished response's sandbox wrote, or nil with an error.
	@discussion GET {endpoint}/{id}/files, which Perplexity's Agent API serves. Each file's
				identifier names it to contentsOfOutputFileWithIdentifier:responseIdentifier:error:.
				Blocks. Introduced in InferKit 0.4.0.
*/
- (nullable NSArray<NFKRemoteFile *> *)outputFilesForResponseIdentifier:(NSString *)responseIdentifier
																  error:(NSError * _Nullable *)outError;

/*!
	@method     contentsOfOutputFileWithIdentifier:responseIdentifier:error:
	@abstract   The bytes of one file a response's sandbox wrote, or nil with an error.
	@discussion GET {endpoint}/{id}/files/{file_id}/content. Blocks. Introduced in InferKit 0.4.0.
*/
- (nullable NSData *)contentsOfOutputFileWithIdentifier:(NSString *)fileIdentifier
									 responseIdentifier:(NSString *)responseIdentifier
												  error:(NSError * _Nullable *)outError;

/*! Streams the reply into the job's partialResult, or, with runsInBackground, polls the background
	job; cancelling the job closes the connection or cancels the background job. */
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

#endif /* NFKRemoteResponsesBackend_h */
