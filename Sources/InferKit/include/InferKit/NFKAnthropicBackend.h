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

@end

NS_ASSUME_NONNULL_END
