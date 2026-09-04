//
//  NFKRemoteEmbeddingBackend.h
//  InferKit
//

#ifndef NFKRemoteEmbeddingBackend_h
#define NFKRemoteEmbeddingBackend_h

#import <Foundation/Foundation.h>
#import <InferKit/NFKInferenceBackend.h>

NS_ASSUME_NONNULL_BEGIN

@class NFKRemoteProvider;

/*!
	@class      NFKRemoteEmbeddingBackend
	@abstract   An inference backend that calls an OpenAI-compatible embeddings endpoint.
	@discussion POST /embeddings, which the hosted providers and every local runner serve. The
				request reads NFKInputPrompt, or joins the content of NFKInputMessages, and returns
				the vector under NFKOutputEmbedding (NSArray<NSNumber *>) with the parsed body under
				NFKRemoteBackendRawKey. Request parameters fold into the body, so a caller sets
				dimensions or encoding_format by name. embeddingsForTexts:error: embeds a batch in
				one call, which is how a corpus is indexed.

				This is the remote counterpart of the on-device text embedders in InferKitMLX; both
				answer with the same output key, so a consumer's search or clustering code does not
				change with the engine. runInferenceForRequest: blocks; run it off the render
				thread. Introduced in InferKit 0.3.0.
*/
@interface NFKRemoteEmbeddingBackend : NSObject <NFKInferenceBackend>

/*! The embeddings endpoint, for example http://localhost:11434/v1/embeddings. */
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
	@method     backendForProvider:apiKey:modelName:
	@abstract   A backend pointed at the provider's embeddings endpoint.
	@discussion Returns nil for Anthropic, which serves no embeddings endpoint. Whether another
				provider serves one for the named model is the provider's documentation to say;
				the local runners answer for any embedding model they have.
*/
+ (nullable instancetype)backendForProvider:(NFKRemoteProvider *)provider
									 apiKey:(nullable NSString *)apiKey
								  modelName:(nullable NSString *)modelName;

/*!
	@method     embeddingsForTexts:error:
	@abstract   Embeds several texts in one request, in the order given.
	@discussion The provider answers with an index per vector, and the result is ordered by it,
				so the ith vector belongs to the ith text whatever order the body listed them in.
*/
- (nullable NSArray<NSArray<NSNumber *> *> *)embeddingsForTexts:(NSArray<NSString *> *)texts
														  error:(NSError * _Nullable *)outError;

/*!
	@method     sendRequest:response:error:
	@abstract   Performs the HTTP request synchronously and returns the body data, or nil.
	@discussion The transport seam. The default delegates to NFKRemoteTransport; a test overrides it.
*/
- (nullable NSData *)sendRequest:(NSURLRequest *)request
						response:(NSHTTPURLResponse * _Nullable * _Nullable)outResponse
						   error:(NSError * _Nullable *)outError;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKRemoteEmbeddingBackend_h */
