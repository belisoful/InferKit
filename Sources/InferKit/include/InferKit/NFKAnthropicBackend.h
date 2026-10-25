//
//  NFKAnthropicBackend.h
//  InferKit
//

#import <Foundation/Foundation.h>
#import <InferKit/NFKInferenceBackend.h>

NS_ASSUME_NONNULL_BEGIN

/*!
	@class      NFKAnthropicBackend
	@abstract   An inference backend that calls Anthropic's Messages API.
	@discussion Every other hosted service this toolkit talks to speaks the OpenAI chat-completions
				shape, which NFKRemoteBackend serves. Anthropic differs in four ways that a URL swap
				cannot cover, which is why it is a separate backend:

				- the key travels in an x-api-key header, not as a Bearer token
				- an anthropic-version header is required
				- max_tokens is required rather than optional
				- a system prompt is a top-level field, not a message with role "system"

				The reply carries content as a list of typed blocks; the text blocks are joined.

				Reads NFKInputPrompt or NFKInputMessages and returns NFKOutputText. A message list with
				a leading system role is lifted into the top-level system field, so a caller writes the
				same request here as for an OpenAI-compatible provider. The contract's sampling
				parameters carry this API's spelling: NFKParameterMaxTokens becomes max_tokens,
				NFKParameterTemperature temperature, NFKParameterTopP top_p, NFKParameterTopK top_k,
				and NFKParameterStopSequences stop_sequences. The API has no repetition penalty.
				NFKParameterSkills becomes container.skills, with the code_execution_20260521 tool the
				container runs them in when the request names no code-execution tool; the files a skill
				writes come back as file ids in the code-execution result, which NFKRemoteFileStore
				downloads.

				The request shape follows the model generation named by modelName:

				- before Claude Opus 4.6 (claude-3…, the 4.0, 4.1, and 4.5 models) →
				  NFKParameterReasoningEffort is a thinking budget in tokens (light, moderate, and
				  deep are 2048, 8192, and 16384, or a numeric string), and sampling is sent when
				  the model is not thinking
				- Claude Opus 4.6 and Claude Sonnet 4.6 → adaptive thinking at an output_config
				  effort (low, medium, high, or any level string the API names), and sampling is sent
				  when the model is not thinking
				- every later model, and any name outside those families → the same adaptive effort
				  with a summarized thinking display, and temperature, top_p, and top_k are dropped,
				  because these models refuse them
				- Claude Opus 5.5, Claude Fable 5.1, and Claude Mythos 5.1 refuse a forced tool, so
				  NFKParameterJSONSchema goes out as output_config.format and the reply's JSON text
				  becomes NFKOutputStructured; every other model answers through a forced tool,
				  except beside a thinking budget, which refuses one and takes output_config.format

				A numeric thinking budget sent to an adaptive model is refused with
				kNFKError_InferenceUnsupported. Thinking counts toward max_tokens, so a reasoning
				effort raises it above the matching budget.

				Inference is synchronous and multi-second: run it off the main or render thread, or
				submit a job. isReady reports whether an endpoint and a model are set.
				Introduced in InferKit 0.1.0.
*/
@interface NFKAnthropicBackend : NSObject <NFKInferenceBackend>

/*! The messages endpoint. Defaults to Anthropic's own. */
@property (nonatomic, copy, nullable) NSURL *endpointURL;

/*! The key sent as x-api-key. */
@property (nonatomic, copy, nullable) NSString *apiKey;

/*! The model name sent in the request body. Required by the API. */
@property (nonatomic, copy, nullable) NSString *modelName;

/*! The API version sent as anthropic-version. Defaults to the version this release was written for. */
@property (nonatomic, copy) NSString *apiVersion;

/*!
	@property   maxTokens
	@abstract   The max_tokens a request that sets no NFKParameterMaxTokens sends, which the API requires.
	@discussion 0, the default since InferKit 0.4.0, sends the model's own maximum: the max_tokens the Models
				API states for modelName, asked with apiKey and remembered from a 200 answer. A backend
				without a key, a model the API states none for, or a lookup that fails sends 1024, the
				default before 0.4.0. A model released before Claude 4.5 that receives its maximum is sent
				the beta that ends a reply at the context window in place of refusing the request.
				- submitInferenceJobForRequest: asks the Models API off the caller's thread.
				- runInferenceForRequest:error: asks on its own thread, and reads a reply sent the model's
				  maximum as a stream. That run is not retried on an overload, and its
				  NFKRemoteBackendRawKey holds the assembled assistant turn ({role, content}). A run that
				  names a large limit itself is one response bounded by timeout; a long reply takes
				  submitInferenceJobForRequest:.
*/
@property (nonatomic, assign) NSInteger maxTokens;

/*! Seconds before the request times out. */
@property (nonatomic, assign) NSTimeInterval timeout;

/*! The session used for transport. */
@property (nonatomic, strong) NSURLSession *session;

+ (instancetype)backendWithEndpointURL:(nullable NSURL *)endpointURL;

/*!
	@method     submitInferenceJobForRequest:
	@abstract   Starts a streamed run and returns the job to follow it.
	@discussion The request is sent with stream enabled and the reply read as the Messages API's
				events; each text delta appends to the job's partialResult under
				NFKRemoteBackendTextKey, and a tool use's input assembles across its deltas. The job
				finishes with the same result the blocking form returns. Cancelling the job cancels
				the request. Introduced in InferKit 0.3.0.
*/
- (NFKInferenceJob *)submitInferenceJobForRequest:(NFKInferenceRequest *)request;

/*!
	@method     tokenCountForRequest:error:
	@abstract   The input tokens a run of the request costs, from the Messages API's count_tokens.
	@discussion The request body is the one a run sends (the system turn, the converted tool
				conversation, the attached media, the tools, and the thinking), reduced to the fields
				count_tokens takes, and posted to the endpoint's count_tokens path
				(…/messages/count_tokens). Blocks; run it off the render thread. Introduced in
				InferKit 0.4.0.
*/
- (nullable NSNumber *)tokenCountForRequest:(NFKInferenceRequest *)request error:(NSError * _Nullable *)error;

/*!
	@method     streamRequest:lineHandler:completionHandler:
	@abstract   Performs a request whose reply is read line by line as it arrives.
	@discussion The transport seam for the streamed form; the default delegates to
				NFKRemoteTransport. Returns the block that cancels the request. A test overrides it
				to feed staged lines. Introduced in InferKit 0.3.0.
*/
- (void (^ NS_SWIFT_SENDABLE)(void))streamRequest:(NSURLRequest *)request
					lineHandler:(void (^ NS_SWIFT_SENDABLE)(NSString *line))lineHandler
			  completionHandler:(void (^ NS_SWIFT_SENDABLE)(NSHTTPURLResponse * _Nullable response,
										  NSData * _Nullable errorBody,
										  NSError * _Nullable error))completionHandler;

@end

NS_ASSUME_NONNULL_END
