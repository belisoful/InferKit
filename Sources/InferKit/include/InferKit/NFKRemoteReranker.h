//
//  NFKRemoteReranker.h
//  InferKit
//

#ifndef NFKRemoteReranker_h
#define NFKRemoteReranker_h

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@class NFKRemoteProvider;

/*!
	@class      NFKRemoteReranker
	@abstract   Scores documents against a query through a hosted rerank endpoint.
	@discussion A reranker reads a query and each document together and predicts one relevance
				score per document, which is what reorders an embedder's shortlist. Because it takes
				a query and a list rather than one input, it is a scoring object rather than an
				NFKInferenceBackend, with the same shape as the on-device NFKMLXModernBERTReranker,
				so the two are interchangeable. POST /rerank, verified to be served by together and
				openrouter at release; openai and mistral serve none. Every call blocks; run it off
				the render thread. Introduced in InferKit 0.3.0.
*/
@interface NFKRemoteReranker : NSObject

/*! The rerank endpoint, for example https://api.together.xyz/v1/rerank. */
@property (nonatomic, copy, nullable) NSURL *endpointURL;

/*! The bearer token sent as Authorization. */
@property (nonatomic, copy, nullable) NSString *apiKey;

/*! The model name sent in the request body. Required by every service that serves this endpoint. */
@property (nonatomic, copy, nullable) NSString *modelName;

/*! The request timeout in seconds. Defaults to 60. */
@property (nonatomic, assign) NSTimeInterval timeout;

/*! The session used for the call. Defaults to the shared session. */
@property (nonatomic, strong) NSURLSession *session;

+ (instancetype)rerankerWithEndpointURL:(nullable NSURL *)endpointURL;

/*! A reranker pointed at the provider's rerank endpoint, or nil for Anthropic. */
+ (nullable instancetype)rerankerForProvider:(NFKRemoteProvider *)provider
									  apiKey:(nullable NSString *)apiKey
								   modelName:(nullable NSString *)modelName;

/*! One relevance score per document, in the documents' order, or nil with an error. */
- (nullable NSArray<NSNumber *> *)scoresForQuery:(NSString *)query
									   documents:(NSArray<NSString *> *)documents
										   error:(NSError * _Nullable *)outError;

/*! The documents' indices from most to least relevant, or nil with an error. */
- (nullable NSArray<NSNumber *> *)rankedIndicesForQuery:(NSString *)query
											  documents:(NSArray<NSString *> *)documents
												  error:(NSError * _Nullable *)outError;

/*! One document's relevance to the query, or nil with an error. */
- (nullable NSNumber *)scoreForQuery:(NSString *)query
							document:(NSString *)document
							   error:(NSError * _Nullable *)outError;

/*! The transport seam. The default delegates to NFKRemoteTransport; a test overrides it. */
- (nullable NSData *)sendRequest:(NSURLRequest *)request
						response:(NSHTTPURLResponse * _Nullable * _Nullable)outResponse
						   error:(NSError * _Nullable *)outError;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKRemoteReranker_h */
