//
//  NFKRemoteFileStore.h
//  InferKit
//

#ifndef NFKRemoteFileStore_h
#define NFKRemoteFileStore_h

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@class NFKRemoteProvider;

/*!
	@class      NFKRemoteFile
	@abstract   A file a hosted service keeps: what it answered when the file was uploaded, listed, or
				fetched.
	@discussion Placed under NFKInputDocument or NFKInputDocuments, a file rides in a request as the
				service's own reference to it (a file id, or Gemini's file URI) rather than as its
				bytes. Introduced in InferKit 0.4.0.
*/
@interface NFKRemoteFile : NSObject

/*! The id a request names the file by: file-…, file_…, or Gemini's files/…. */
@property (nonatomic, copy, readonly) NSString *identifier;

/*! The file's name, or nil when the service keeps none. */
@property (nonatomic, copy, readonly, nullable) NSString *filename;

/*! The size in bytes, or 0 when the service does not say. */
@property (nonatomic, readonly) long long byteCount;

/*! The MIME type, or nil when the service does not say. */
@property (nonatomic, copy, readonly, nullable) NSString *mimeType;

/*! What the file is for (user_data, assistants, ocr, batch, fine-tune), or nil where the service has
	no purposes. */
@property (nonatomic, copy, readonly, nullable) NSString *purpose;

/*! When the service took the file, or nil. */
@property (nonatomic, copy, readonly, nullable) NSDate *createdAt;

/*! When the service deletes the file, or nil when it keeps it. */
@property (nonatomic, copy, readonly, nullable) NSDate *expiresAt;

/*! Gemini's file URI, the form a Gemini request names the file by; nil elsewhere. */
@property (nonatomic, copy, readonly, nullable) NSURL *uri;

/*! The processing state where the service reports one (Gemini: PROCESSING, ACTIVE, FAILED). */
@property (nonatomic, copy, readonly, nullable) NSString *state;

/*! Whether the content can be downloaded. Anthropic and OpenRouter let only the files their tools
	create be downloaded; elsewhere every file can be. */
@property (nonatomic, readonly, getter=isDownloadable) BOOL downloadable;

/*! The service's own record of the file. */
@property (nonatomic, copy, readonly) NSDictionary<NSString *, id> *raw;

/*! A reference to a file the service already keeps, by the id a request names it with. */
+ (instancetype)fileWithIdentifier:(NSString *)identifier mimeType:(nullable NSString *)mimeType;

/*! A reference to a Gemini file by its URI. */
+ (instancetype)fileWithGeminiURI:(NSURL *)uri mimeType:(NSString *)mimeType;

- (instancetype)initWithIdentifier:(NSString *)identifier
						  filename:(nullable NSString *)filename
						 byteCount:(long long)byteCount
						  mimeType:(nullable NSString *)mimeType
						   purpose:(nullable NSString *)purpose
						 createdAt:(nullable NSDate *)createdAt
						 expiresAt:(nullable NSDate *)expiresAt
							   uri:(nullable NSURL *)uri
							 state:(nullable NSString *)state
					  downloadable:(BOOL)downloadable
							   raw:(NSDictionary<NSString *, id> *)raw NS_DESIGNATED_INITIALIZER;
- (instancetype)init NS_UNAVAILABLE;

@end

/*!
	@enum       NFKRemoteFileStoreAPIStyle
	@abstract   The wire shape of a hosted Files API.
	@constant   NFKRemoteFileStoreAPIStyleOpenAI POST /files multipart with a purpose; GET /files,
				/files/{id}, /files/{id}/content; DELETE /files/{id}. OpenAI, Mistral, xAI, DeepSeek,
				Groq, Together (upload at /files/upload), and OpenRouter.
	@constant   NFKRemoteFileStoreAPIStyleAnthropic Anthropic's /files: no purpose, next_page paging,
				and only tool-created files downloadable.
	@constant   NFKRemoteFileStoreAPIStyleGemini Gemini's resumable upload at /upload/v1beta/files,
				files.get / list / delete, no download, and a processing state.
	Introduced in InferKit 0.4.0.
*/
typedef NS_ENUM(NSInteger, NFKRemoteFileStoreAPIStyle) {
	NFKRemoteFileStoreAPIStyleOpenAI = 0,
	NFKRemoteFileStoreAPIStyleAnthropic,
	NFKRemoteFileStoreAPIStyleGemini,
};

/*!
	@class      NFKRemoteFileStore
	@abstract   Uploads, lists, fetches, downloads, and deletes the files a hosted service keeps.
	@discussion A file uploaded once is named by reference in every later request (NFKRemoteFile under
				NFKInputDocument), which is how a large PDF, a long recording, or a video reaches a
				model without riding in each request, and how a tool's output file is fetched.
				What each service keeps a file for differs: OpenAI, Anthropic, Gemini, Mistral (OCR),
				xAI, DeepSeek (images), and OpenRouter take files for prompts; Together and Groq take
				them for fine-tuning and batch jobs. Every call blocks; run it off the main thread.
				Introduced in InferKit 0.4.0.
*/
@interface NFKRemoteFileStore : NSObject

/*! The files endpoint: …/v1/files, or for Gemini the API root (https://generativelanguage.googleapis.com). */
@property (nonatomic, copy, nullable) NSURL *endpointURL;

/*! The wire shape the store speaks. */
@property (nonatomic, assign) NFKRemoteFileStoreAPIStyle apiStyle;

/*! The key, sent the way the service reads it. */
@property (nonatomic, copy, nullable) NSString *apiKey;

/*! The purpose an upload names when the call gives none (user_data on OpenAI and DeepSeek, assistants
	on xAI, ocr on Mistral, batch on Groq, fine-tune on Together). backendForProvider sets it. */
@property (nonatomic, copy, nullable) NSString *defaultPurpose;

/*! The upload path under the endpoint (upload on Together, which posts to /files/upload), or nil. */
@property (nonatomic, copy, nullable) NSString *uploadPathComponent;

/*! The request timeout in seconds. Defaults to 600; an upload is large. */
@property (nonatomic, assign) NSTimeInterval timeout;

/*! The session used for the calls. Defaults to the shared session. */
@property (nonatomic, strong) NSURLSession *session;

/*! A store for the provider's Files API: openai, anthropic, gemini, mistral, xai, deepseek, together,
	openrouter, and groq serve one; every other preset returns nil. */
+ (nullable instancetype)fileStoreForProvider:(NFKRemoteProvider *)provider apiKey:(nullable NSString *)apiKey;

/*! Uploads bytes under a name. purpose nil takes defaultPurpose; expiresAfter 0 keeps the service's
	default lifetime, where the service takes one. */
- (nullable NFKRemoteFile *)uploadData:(NSData *)data
							  filename:(NSString *)filename
							  mimeType:(nullable NSString *)mimeType
							   purpose:(nullable NSString *)purpose
						  expiresAfter:(NSTimeInterval)expiresAfter
								 error:(NSError * _Nullable *)outError;

/*! Uploads a local file, its name and MIME type taken from the URL. */
- (nullable NFKRemoteFile *)uploadFileAtURL:(NSURL *)fileURL
									purpose:(nullable NSString *)purpose
									  error:(NSError * _Nullable *)outError;

/*! Every file the key can see, all pages read. */
- (nullable NSArray<NFKRemoteFile *> *)filesWithError:(NSError * _Nullable *)outError;

/*! One file's record. */
- (nullable NFKRemoteFile *)fileWithIdentifier:(NSString *)identifier error:(NSError * _Nullable *)outError;

/*! Polls a file until its processing ends (Gemini's ACTIVE), or the timeout passes. A service without
	a processing state answers the first fetch. */
- (nullable NFKRemoteFile *)fileWhenReadyWithIdentifier:(NSString *)identifier
												timeout:(NSTimeInterval)timeout
												  error:(NSError * _Nullable *)outError;

/*! A file's content. Refused where the service keeps it from download (an uploaded file on
	Anthropic or OpenRouter; any file on Gemini). */
- (nullable NSData *)contentsOfFileWithIdentifier:(NSString *)identifier error:(NSError * _Nullable *)outError;

/*! Deletes a file. */
- (BOOL)deleteFileWithIdentifier:(NSString *)identifier error:(NSError * _Nullable *)outError;

/*! A time-limited URL to the file's content (Mistral), which a chat request can name as a
	document_url. expiryHours 0 takes the service's default. */
- (nullable NSURL *)signedURLForFileWithIdentifier:(NSString *)identifier
									   expiryHours:(NSInteger)expiryHours
											 error:(NSError * _Nullable *)outError;

/*! The transport seam. The default delegates to NFKRemoteTransport; a test overrides it. */
- (nullable NSData *)sendRequest:(NSURLRequest *)request
						response:(NSHTTPURLResponse * _Nullable * _Nullable)outResponse
						   error:(NSError * _Nullable *)outError;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKRemoteFileStore_h */
