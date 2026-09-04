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
				same request here as for an OpenAI-compatible provider.

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

/*! The max_tokens the API requires. Defaults to 1024; NFKParameterMaxTokens overrides it per request. */
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
	@method     streamRequest:lineHandler:completionHandler:
	@abstract   Performs a request whose reply is read line by line as it arrives.
	@discussion The transport seam for the streamed form; the default delegates to
				NFKRemoteTransport. Returns the block that cancels the request. A test overrides it
				to feed staged lines. Introduced in InferKit 0.3.0.
*/
- (void (^)(void))streamRequest:(NSURLRequest *)request
					lineHandler:(void (^)(NSString *line))lineHandler
			  completionHandler:(void (^)(NSHTTPURLResponse * _Nullable response,
										  NSData * _Nullable errorBody,
										  NSError * _Nullable error))completionHandler;

@end

NS_ASSUME_NONNULL_END
