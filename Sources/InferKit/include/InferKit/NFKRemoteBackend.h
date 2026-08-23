//
//  NFKRemoteBackend.h
//  InferKit
//

#ifndef NFKRemoteBackend_h
#define NFKRemoteBackend_h

#import <Foundation/Foundation.h>
#import "NFKInferenceBackend.h"

NS_ASSUME_NONNULL_BEGIN

/*! Request input holding a plain prompt string, wrapped as one user message. */
extern NSString * const NFKRemoteBackendPromptKey;
/*! Request input holding an OpenAI messages array, used as-is when present. */
extern NSString * const NFKRemoteBackendMessagesKey;
/*! Result output holding the assistant's text. */
extern NSString * const NFKRemoteBackendTextKey;
/*! Result output holding the parsed response body. */
extern NSString * const NFKRemoteBackendRawKey;

/*!
	@class      NFKRemoteBackend
	@abstract   An inference backend that calls an OpenAI-compatible chat-completions endpoint.
	@discussion Depends only on Foundation, so InferKit ships it. One
				path serves both a local server (an OpenAI-compatible localhost server such as a
				BaseRT, mlx_lm, or Ollama endpoint) and a true remote API: the difference is only
				the endpoint URL and the key. It speaks the synchronous text protocol; image work
				stays on the InferKit side.

				A request supplies its prompt through NFKRemoteBackendMessagesKey (an OpenAI
				messages array, used as-is) or NFKRemoteBackendPromptKey (a string wrapped as
				one user message). Request parameters fold into the request body, so a caller sets
				temperature, max_tokens, and similar by name. The result exposes the assistant
				text under NFKRemoteBackendTextKey and the parsed body under
				NFKRemoteBackendRawKey.

				runInferenceForRequest: blocks until the call returns, so a caller runs it off the
				render thread. isReady reports whether an endpoint is set.
*/
@interface NFKRemoteBackend : NSObject <NFKInferenceBackend>

/*! The chat-completions endpoint, for example a localhost server or a hosted API URL. */
@property (nonatomic, copy, nullable) NSURL *endpointURL;

/*! The bearer token sent as Authorization, when the endpoint needs one. */
@property (nonatomic, copy, nullable) NSString *apiKey;

/*! The model name sent in the request body. */
@property (nonatomic, copy, nullable) NSString *modelName;

/*! The request timeout in seconds. Defaults to 60. */
@property (nonatomic, assign) NSTimeInterval timeout;

/*! The session used for the call. Defaults to the shared session. */
@property (nonatomic, strong) NSURLSession *session;

+ (instancetype)backendWithEndpointURL:(nullable NSURL *)endpointURL;

/*!
	@method     sendRequest:response:error:
	@abstract   Performs the HTTP request synchronously and returns the body data, or nil.
	@discussion The transport seam. The default runs the request on the session and blocks until
				it completes. A test or an alternative transport overrides this.
*/
- (nullable NSData *)sendRequest:(NSURLRequest *)request
					   response:(NSHTTPURLResponse * _Nullable * _Nullable)outResponse
						  error:(NSError * _Nullable *)outError;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKRemoteBackend_h */
