//
//  NFKAnthropicBackend.h
//  InferKit
//

#import <Foundation/Foundation.h>
#import <InferKit/NFKInferenceBackend.h>

NS_ASSUME_NONNULL_BEGIN

@class NFKRemoteProvider;

/*!
	@const      NFKAnthropicBackendInputTransformationsKey
	@abstract   The output key for what happened to the thinking blocks a request replayed.
	@discussion An NSArray of NSDictionary, absent when nothing happened. Two sources fill it:

				- the backend, after the prefix-check 400 → one {type: thinking_removed, path} per
				  block it removed before sending again (see recoversFromPrefixMismatch)
				- the API, under prefixMismatchBehavior → its input_transformations entries as
				  returned, {type, path, reason}: thinking_dropped for a block it dropped, and
				  thinking_mismatch_allowed for a failing block it let through

				path is messages.{i}.content.{j}, the block's place in the request as first sent. The
				API adds entry types and reasons over time, so a reader skips one it does not know.
				Introduced in InferKit 0.4.0.
*/
extern NSString * const NFKAnthropicBackendInputTransformationsKey;

/*!
	@enum       NFKAnthropicPrefixMismatchBehavior
	@abstract   What a request asks the API to do with a replayed thinking block that fails the prefix
				check.
	@discussion The values are thinking.block_binding.prefix_mismatch_behavior, which the beta header
				thinking-binding-controls-2026-08-01 adds. Introduced in InferKit 0.4.0.
*/
typedef NS_ENUM(NSInteger, NFKAnthropicPrefixMismatchBehavior) {
	/*! Sends neither the setting nor the beta header. The model decides: Claude Fable 5.2 and Claude
		Mythos 5.2 answer 400. */
	NFKAnthropicPrefixMismatchBehaviorUnset = 0,
	/*! "error": a failing block is a 400 on every model that runs the check. */
	NFKAnthropicPrefixMismatchBehaviorError = 1,
	/*! "drop_block": the API drops the failing blocks and answers, and lists them under
		NFKAnthropicBackendInputTransformationsKey. */
	NFKAnthropicPrefixMismatchBehaviorDropBlock = 2,
};

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

				Claude Fable 5.2 and Claude Mythos 5.2 run a prefix check on every request that replays
				thinking blocks, and Claude Fable 5.1, Claude Mythos 5.1, Claude Opus 5.5, Claude
				Sonnet 5.5, and Claude Haiku 5.5 can. A block is valid only while the system prompt,
				the tools, and every earlier message are what they were when the model produced it. A
				conversation that keeps its system prompt and tools and only appends messages passes.
				One that changed any of them answers 400, and the same body fails again.
				recoversFromPrefixMismatch and prefixMismatchBehavior are the two recoveries.

				Meta's Model API serves its Muse models on the same route (POST /v1/messages), which
				backendForProvider:apiKey:modelName: reaches on the meta preset. A model named muse-…
				follows that route's rules:

				- the key travels as a Bearer token (authorizesWithBearerToken)
				- NFKParameterTemperature and NFKParameterTopP are sent beside a reasoning effort,
				  and NFKParameterTopK is dropped, because the route refuses top_k
				- NFKParameterStopSequences stays off the wire, because the route refuses
				  stop_sequences; the text ends at the earliest stop sequence, and
				  NFKOutputFinishReason is then NFKFinishReasonStopSequence
				- NFKParameterJSONSchema goes out as output_config.format, because the route refuses
				  a named tool_choice
				- NFKInputVideo goes whole as a video block, which the route reads with its
				  soundtrack, unless the request names NFKParameterVideoFrameCount, which asks for
				  sampled frames; an NFKRemoteFile whose type is video/… under NFKInputDocument is
				  a video block that names the uploaded file
				- NFKUsageReasoningTokens is the thinking token count the route reports
				- the reasoning is redacted, so NFKOutputReasoning holds the summary when the route
				  returns one, and the encrypted block is replayed with the assistant turn

				Inference is synchronous and multi-second: run it off the main or render thread, or
				submit a job. isReady reports whether an endpoint and a model are set.
				Introduced in InferKit 0.1.0.
*/
@interface NFKAnthropicBackend : NSObject <NFKInferenceBackend>

/*! The messages endpoint. Defaults to Anthropic's own. */
@property (nonatomic, copy, nullable) NSURL *endpointURL;

/*! The key sent as x-api-key, or as a Bearer token when authorizesWithBearerToken is set. */
@property (nonatomic, copy, nullable) NSString *apiKey;

/*!
	@property   authorizesWithBearerToken
	@abstract   Whether apiKey travels as an Authorization Bearer token in place of the x-api-key header.
	@discussion NO by default, which is Anthropic's header. YES is for a service that serves the
				Messages route behind a Bearer key. The factories set it for the meta preset, whose
				documented header is Bearer. The run, count_tokens, and the Models API lookup all
				follow it, and anthropic-version is sent either way. Introduced in InferKit 0.4.0.
*/
@property (nonatomic, assign) BOOL authorizesWithBearerToken;

/*! The model name sent in the request body. Required by the API. */
@property (nonatomic, copy, nullable) NSString *modelName;

/*! The header that carries each request's session, or nil to send none (the default). The session is
	ses_ and a digest of the request's NFKParameterConversationKey, else of its messages through the
	first user message, else an identifier this backend keeps, which a follow-up request (a poll, a
	cancel, a lookup) also carries. The provider factories set it from NFKRemoteProvider's
	sessionHeaderName: x-opencode-session for the OpenCode presets. Introduced in InferKit 0.4.0. */
@property (nonatomic, copy, nullable) NSString *sessionHeaderName;

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

/*!
	@property   recoversFromPrefixMismatch
	@abstract   Whether a request the prefix check refuses is sent once more without the refused thinking.
	@discussion YES by default. The 400 is recognized by its sentence "The block is bound to a different
				conversation". The backend then removes the block the error names and every thinking and
				redacted_thinking block after it, keeps every other block, and sends the request again
				once. Where two neighbouring messages share a role, the named position cannot be read
				against the array, and every thinking block is removed. The result lists the removed
				blocks under NFKAnthropicBackendInputTransformationsKey. A second refusal is returned as
				the error.

				The model answers that turn without the removed reasoning, and the prompt cache misses
				from the first removed block on. The caller's messages still hold the refused blocks, so
				each later run of the conversation is refused and sent again the same way, and loses the
				thinking produced since. Setting prefixMismatchBehavior to
				NFKAnthropicPrefixMismatchBehaviorDropBlock for the rest of that conversation avoids
				both. submitInferenceJobForRequest:, runInferenceForRequest:error:, and
				tokenCountForRequest:error: all recover; a batch item is never refused. NO returns the
				400 as the error. Introduced in InferKit 0.4.0.
*/
@property (nonatomic, assign) BOOL recoversFromPrefixMismatch;

/*!
	@property   prefixMismatchBehavior
	@abstract   The prefix_mismatch_behavior a request carries, with the beta header that adds it.
	@discussion NFKAnthropicPrefixMismatchBehaviorUnset by default, which sends nothing. Any other
				value sends thinking.block_binding.prefix_mismatch_behavior and the anthropic-beta value
				thinking-binding-controls-2026-08-01 on a run, a count, and a batch item, to the models
				that run the prefix check: Claude Fable 5.2 and 5.1, Claude Mythos 5.2 and 5.1, Claude
				Opus 5.5, Claude Sonnet 5.5, and Claude Haiku 5.5. The setting rides on an adaptive
				thinking object, so a request without NFKParameterReasoningEffort gains
				thinking: {type: adaptive}, which is what those models run when a request names none.
				Every other model is sent neither the setting nor the header.

				- NFKAnthropicPrefixMismatchBehaviorDropBlock → the API drops the failing blocks, keeps
				  the valid ones, answers, and names each dropped block under
				  NFKAnthropicBackendInputTransformationsKey. The setting acts on the one request, so
				  it stays set for the rest of a conversation whose history holds the change.
				- NFKAnthropicPrefixMismatchBehaviorError → a failing block is a 400 on every one of
				  those models, which recoversFromPrefixMismatch then recovers from.

				A proxy or platform that drops the header leaves the setting refused with a 400 whose
				text ends "block_binding: Extra inputs are not permitted". Introduced in InferKit 0.4.0.
*/
@property (nonatomic, assign) NFKAnthropicPrefixMismatchBehavior prefixMismatchBehavior;

+ (instancetype)backendWithEndpointURL:(nullable NSURL *)endpointURL;

/*!
	@method     backendForProvider:apiKey:modelName:
	@abstract   A backend pointed at the provider's Messages endpoint, or nil for a provider that
				serves none.
	@discussion anthropic, opencode, opencode-go, meta, and inferkit serve one at the base plus
				/messages; every other preset returns nil. OpenCode answers this route only for the
				models it routes to Messages (see NFKRemoteProvider's modelAPIStyles). The meta preset's
				own style is chat completions, so this factory is how a caller reaches Meta's Messages
				route, with the key as a Bearer token. NFKRemoteProvider's
				backendForProvider:apiKey:modelName: builds the same backend for a model the provider
				routes to NFKRemoteAPIStyleAnthropicMessages. Meta's Models API states no output limit,
				so a run that names no NFKParameterMaxTokens and no maxTokens sends 1024 there.
				Introduced in InferKit 0.4.0.
*/
+ (nullable instancetype)backendForProvider:(NFKRemoteProvider *)provider
									 apiKey:(nullable NSString *)apiKey
								  modelName:(nullable NSString *)modelName;

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
				(…/messages/count_tokens), which Meta's Model API serves for its Muse models too.
				Blocks; run it off the render thread. Introduced in InferKit 0.4.0.
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
