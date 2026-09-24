//
//  NFKRemoteRetrievalStore.h
//  InferKit
//

#ifndef NFKRemoteRetrievalStore_h
#define NFKRemoteRetrievalStore_h

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@class NFKRemoteProvider;
@class NFKRemoteFileStore;

/*!
	@class      NFKRetrievalStoreRecord
	@abstract   A hosted retrieval store: an OpenAI vector store, an xAI collection, a Gemini file search
				store, or a Mistral library. Introduced in InferKit 0.4.0.
*/
@interface NFKRetrievalStoreRecord : NSObject
/*! The id later calls name the store by. */
@property (nonatomic, copy, readonly) NSString *identifier;
/*! The store's name, or nil. */
@property (nonatomic, copy, readonly, nullable) NSString *name;
/*! The number of documents the service reports, or 0. */
@property (nonatomic, readonly) NSInteger documentCount;
/*! The service's status for the store, or nil. */
@property (nonatomic, copy, readonly, nullable) NSString *status;
/*! The service's own record. */
@property (nonatomic, copy, readonly) NSDictionary<NSString *, id> *raw;
- (instancetype)init NS_UNAVAILABLE;
@end

/*!
	@class      NFKRetrievalDocument
	@abstract   A document inside a retrieval store. Introduced in InferKit 0.4.0.
*/
@interface NFKRetrievalDocument : NSObject
/*! The id the store names the document by: the file id (OpenAI, xAI), the document name (Gemini,
	Mistral), or, for a Gemini import still running, the operation's name. */
@property (nonatomic, copy, readonly) NSString *identifier;
/*! The document's name, or nil. */
@property (nonatomic, copy, readonly, nullable) NSString *filename;
/*! The indexing status (completed, in_progress, DOCUMENT_STATUS_PROCESSED, STATE_ACTIVE, done), or nil. */
@property (nonatomic, copy, readonly, nullable) NSString *status;
/*! The service's own record. */
@property (nonatomic, copy, readonly) NSDictionary<NSString *, id> *raw;
- (instancetype)init NS_UNAVAILABLE;
@end

/*!
	@class      NFKRetrievalMatch
	@abstract   A chunk a store search returned. Introduced in InferKit 0.4.0.
*/
@interface NFKRetrievalMatch : NSObject
/*! The file the chunk came from. */
@property (nonatomic, copy, readonly) NSString *fileIdentifier;
/*! The file's name, or nil. */
@property (nonatomic, copy, readonly, nullable) NSString *filename;
/*! The relevance score; higher is closer under the store's ranking. */
@property (nonatomic, readonly) double score;
/*! The chunk's text. */
@property (nonatomic, copy, readonly) NSString *text;
/*! The document's attributes or fields, or nil. */
@property (nonatomic, copy, readonly, nullable) NSDictionary<NSString *, id> *attributes;
/*! The service's own record. */
@property (nonatomic, copy, readonly) NSDictionary<NSString *, id> *raw;
- (instancetype)init NS_UNAVAILABLE;
@end

/*!
	@enum       NFKRemoteRetrievalAPIStyle
	@abstract   The wire shape of a hosted retrieval store API.
	@constant   NFKRemoteRetrievalAPIStyleOpenAI OpenAI's /vector_stores, /files, and /search.
	@constant   NFKRemoteRetrievalAPIStyleXAI xAI's collections on the management host and
				/documents/search on the API host.
	@constant   NFKRemoteRetrievalAPIStyleGemini Gemini's fileSearchStores, :importFile, and documents;
				no direct search.
	@constant   NFKRemoteRetrievalAPIStyleMistral Mistral's /libraries and their documents; no direct
				search.
	Introduced in InferKit 0.4.0.
*/
typedef NS_ENUM(NSInteger, NFKRemoteRetrievalAPIStyle) {
	NFKRemoteRetrievalAPIStyleOpenAI = 0,
	NFKRemoteRetrievalAPIStyleXAI,
	NFKRemoteRetrievalAPIStyleGemini,
	NFKRemoteRetrievalAPIStyleMistral,
};

/*!
	@class      NFKRemoteRetrievalStore
	@abstract   Creates, fills, searches, and deletes the retrieval stores a hosted service keeps.
	@discussion A retrieval store indexes uploaded files into chunks a query can reach: the hosted half
				of retrieval-augmented generation, beside the on-device embedders and rerankers. A
				store is created, documents are added (by the id of a file uploaded through
				NFKRemoteFileStore, or uploaded here), and it is searched directly where the service
				has a search endpoint (OpenAI, xAI), or named in a model's retrieval tool where it does
				not (Gemini's file_search, Mistral's document_library), whose results come back under
				NFKOutputServerToolResults and NFKOutputCitations. xAI keeps its collections on a
				separate management host with a separate management key. Every call blocks; run it off
				the main thread. Introduced in InferKit 0.4.0.
*/
@interface NFKRemoteRetrievalStore : NSObject

/*! The API root: …/v1 for OpenAI, xAI, and Mistral, …/v1beta for Gemini. */
@property (nonatomic, copy, nullable) NSURL *endpointURL;

/*! xAI's management root (https://management-api.x.ai/v1), where its collections live. */
@property (nonatomic, copy, nullable) NSURL *managementURL;

/*! The wire shape the store speaks. */
@property (nonatomic, assign) NFKRemoteRetrievalAPIStyle apiStyle;

/*! The API key, sent the way the service reads it. */
@property (nonatomic, copy, nullable) NSString *apiKey;

/*! xAI's management key, which its collection calls need; the API key serves search. */
@property (nonatomic, copy, nullable) NSString *managementAPIKey;

/*! The request timeout in seconds. Defaults to 120. */
@property (nonatomic, assign) NSTimeInterval timeout;

/*! The session used for the calls. Defaults to the shared session. */
@property (nonatomic, strong) NSURLSession *session;

/*! A store for the provider's retrieval API: openai, xai, gemini, and mistral serve one; every other
	preset returns nil. */
+ (nullable instancetype)retrievalStoreForProvider:(NFKRemoteProvider *)provider apiKey:(nullable NSString *)apiKey;

/*! Creates a store. options go into the create body under the service's own names (chunking_strategy,
	expires_after, embeddingModel, chunk_size, index_configuration). */
- (nullable NFKRetrievalStoreRecord *)createStoreNamed:(NSString *)name
											   options:(nullable NSDictionary<NSString *, id> *)options
												 error:(NSError * _Nullable *)outError;

/*! Every store the key can see, all pages read. */
- (nullable NSArray<NFKRetrievalStoreRecord *> *)storesWithError:(NSError * _Nullable *)outError;

/*! One store's record. */
- (nullable NFKRetrievalStoreRecord *)storeWithIdentifier:(NSString *)identifier error:(NSError * _Nullable *)outError;

/*! Deletes a store. */
- (BOOL)deleteStoreWithIdentifier:(NSString *)identifier error:(NSError * _Nullable *)outError;

/*! Adds a file the service already keeps to a store. attributes are the document's attributes
	(OpenAI) or fields (xAI); Gemini imports by the file's name and answers with the import's
	operation. Mistral adds only by upload. */
- (nullable NFKRetrievalDocument *)addFileWithIdentifier:(NSString *)fileIdentifier
												 toStore:(NSString *)storeIdentifier
											  attributes:(nullable NSDictionary<NSString *, id> *)attributes
												   error:(NSError * _Nullable *)outError;

/*! Uploads bytes into a store: Mistral's library upload directly, and elsewhere an upload through the
	provider's Files API followed by an add. */
- (nullable NFKRetrievalDocument *)uploadData:(NSData *)data
									 filename:(NSString *)filename
									  toStore:(NSString *)storeIdentifier
										error:(NSError * _Nullable *)outError;

/*! The documents in a store, all pages read. */
- (nullable NSArray<NFKRetrievalDocument *> *)documentsInStore:(NSString *)storeIdentifier error:(NSError * _Nullable *)outError;

/*! Removes a document from a store. */
- (BOOL)removeDocument:(NSString *)documentIdentifier fromStore:(NSString *)storeIdentifier error:(NSError * _Nullable *)outError;

/*! Searches stores for the query, best first. filter is the service's own filter: an OpenAI
	comparison or compound dictionary, or an xAI AIP-160 string. Gemini and Mistral have no direct
	search and answer kNFKError_InferenceUnsupported; name their stores in the model's retrieval tool
	instead. */
- (nullable NSArray<NFKRetrievalMatch *> *)searchStores:(NSArray<NSString *> *)storeIdentifiers
												  query:(NSString *)query
												  limit:(NSInteger)limit
												 filter:(nullable id)filter
												  error:(NSError * _Nullable *)outError;

/*! The Files API store the upload path uses, which shares the store's keys and session. */
- (nullable NFKRemoteFileStore *)fileStore;

/*! The transport seam. The default delegates to NFKRemoteTransport; a test overrides it. */
- (nullable NSData *)sendRequest:(NSURLRequest *)request
						response:(NSHTTPURLResponse * _Nullable * _Nullable)outResponse
						   error:(NSError * _Nullable *)outError;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKRemoteRetrievalStore_h */
