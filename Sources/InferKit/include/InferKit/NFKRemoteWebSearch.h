//
//  NFKRemoteWebSearch.h
//  InferKit
//

#ifndef NFKRemoteWebSearch_h
#define NFKRemoteWebSearch_h

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@class NFKRemoteProvider;

/*!
	@class      NFKWebSearchResult
	@abstract   One ranked page a web search returned.
	@discussion The dates are the service's own text (Perplexity writes YYYY-MM-DD), and raw keeps the
				whole entry for a field the class does not read. Introduced in InferKit 0.4.0.
*/
@interface NFKWebSearchResult : NSObject <NSCopying>

/*! The page's title. */
@property (nonatomic, copy, readonly) NSString *title;

/*! The page's address. */
@property (nonatomic, copy, readonly) NSURL *URL;

/*! The text the service extracted from the page, cut to the request's per-page token limit. */
@property (nonatomic, copy, readonly) NSString *snippet;

/*! When the page was published, as the service writes it, or nil where it reports none. */
@property (nonatomic, copy, readonly, nullable) NSString *date;

/*! When the page was last updated, as the service writes it, or nil where it reports none. */
@property (nonatomic, copy, readonly, nullable) NSString *lastUpdated;

/*! The service's entry as it came. */
@property (nonatomic, copy, readonly) NSDictionary<NSString *, id> *raw;

/*! A result from the service's entry, or nil for one without a title, a URL, or a snippet. */
+ (nullable instancetype)resultWithEntry:(NSDictionary<NSString *, id> *)entry;

- (instancetype)init NS_UNAVAILABLE;

@end

/*!
	@class      NFKRemoteWebSearch
	@abstract   Searches the web through a hosted search endpoint and returns ranked pages.
	@discussion A search reads a query and answers a list of pages, each with the text it extracted,
				rather than generating a reply, so it is a query object rather than an
				NFKInferenceBackend, as NFKRemoteReranker is. POST /search with a Bearer token, which
				Perplexity serves at the root of its host (https://api.perplexity.ai/search).

				resultsForQuery:parameters:error: takes one query or several, and the parameters fold
				into the body under their own names: search_type (web, fast, or people), max_results
				(1 to 20 for the web), max_tokens, max_tokens_per_page, country, the language,
				domain, and recency filters, and the published and updated date filters (MM/DD/YYYY).
				Several queries are billed as one request and answered as one ranked list. Every
				call blocks; run it off the render thread. Introduced in InferKit 0.4.0.
*/
@interface NFKRemoteWebSearch : NSObject

/*! The search endpoint, for example https://api.perplexity.ai/search. */
@property (nonatomic, copy, nullable) NSURL *endpointURL;

/*! The bearer token sent as Authorization. */
@property (nonatomic, copy, nullable) NSString *apiKey;

/*! The request timeout in seconds. Defaults to 60. */
@property (nonatomic, assign) NSTimeInterval timeout;

/*! The session used for the call. Defaults to the shared session. */
@property (nonatomic, strong) NSURLSession *session;

+ (instancetype)webSearchWithEndpointURL:(nullable NSURL *)endpointURL;

/*! A search pointed at the provider's search endpoint: perplexity serves one, and every other preset
	returns nil. */
+ (nullable instancetype)webSearchForProvider:(NFKRemoteProvider *)provider apiKey:(nullable NSString *)apiKey;

/*! The pages for one query, most relevant first, or nil with an error. */
- (nullable NSArray<NFKWebSearchResult *> *)resultsForQuery:(NSString *)query error:(NSError * _Nullable *)outError;

/*!
	@method     resultsForQuery:parameters:error:
	@abstract   The pages for a query or a list of queries, most relevant first, or nil with an error.
	@discussion query is an NSString or an NSArray of NSString. parameters fold into the body, and a
				parameter named query does not replace it. An entry without a title, a URL, or a
				snippet is skipped.
*/
- (nullable NSArray<NFKWebSearchResult *> *)resultsForQuery:(id)query
												 parameters:(nullable NSDictionary<NSString *, id> *)parameters
													  error:(NSError * _Nullable *)outError;

/*! The transport seam. The default delegates to NFKRemoteTransport; a test overrides it. */
- (nullable NSData *)sendRequest:(NSURLRequest *)request
						response:(NSHTTPURLResponse * _Nullable * _Nullable)outResponse
						   error:(NSError * _Nullable *)outError;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKRemoteWebSearch_h */
