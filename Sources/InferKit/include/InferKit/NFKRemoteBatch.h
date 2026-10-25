//
//  NFKRemoteBatch.h
//  InferKit
//

#ifndef NFKRemoteBatch_h
#define NFKRemoteBatch_h

#import <Foundation/Foundation.h>
#import <InferKit/NFKInferenceBackend.h>

NS_ASSUME_NONNULL_BEGIN

@class NFKRemoteProvider;
@class NFKRemoteFileStore;
@class NFKInferenceRequest;
@class NFKInferenceResult;

/*!
	@enum       NFKRemoteBatchAPIStyle
	@abstract   The wire shape of a hosted batch API.
	@constant   NFKRemoteBatchAPIStyleOpenAI A JSONL file of {custom_id, method, url, body} uploaded with
				purpose batch, POST /batches naming it, and an output and an error file of
				{custom_id, response: {status_code, body}, error}. OpenAI, Groq (chat and audio, its
				batch naming /v1/chat/completions whatever its lines name), Alibaba Model Studio, and
				Perplexity, whose batches take Agent API requests at /v1/responses.
	@constant   NFKRemoteBatchAPIStyleTogether The OpenAI file with lines of {custom_id, body},
				uploaded with purpose batch-api, and Together's own job record.
	@constant   NFKRemoteBatchAPIStyleAnthropic POST /messages/batches with the requests inline as
				{custom_id, params}, and JSONL results at the batch's results_url.
	@constant   NFKRemoteBatchAPIStyleGemini POST /models/{model}:batchGenerateContent, or
				:asyncBatchEmbedContent for embeddings, with the requests inline and keyed through
				their metadata, or past 20 MB as an uploaded JSONL file of {key, request}; the responses
				come inline in input order or as a file of {key, response | error}.
	@constant   NFKRemoteBatchAPIStyleMistral POST /batch/jobs with the endpoint and the model on the
				job and the requests inline as {custom_id, body} (a file past 10,000), and an output
				file: chat, embeddings, FIM, OCR, moderation, classification, transcription, and
				conversations.
	@constant   NFKRemoteBatchAPIStyleOpenRouter POST /batches with the requests inline, and the
				results inline in the finished batch. It serves no cancel; a finished batch deletes.
	@constant   NFKRemoteBatchAPIStyleXAI A named batch the requests are added to, each under its
				route's key (chat_get_completion, responses, image_generation, image_edit,
				video_generation, video_extension), paged results at /batches/{id}/results, and
				POST /batches/{id}:cancel.
	@discussion Introduced in InferKit 0.4.0.
*/
typedef NS_ENUM(NSInteger, NFKRemoteBatchAPIStyle) {
	NFKRemoteBatchAPIStyleOpenAI = 0,
	NFKRemoteBatchAPIStyleTogether,
	NFKRemoteBatchAPIStyleAnthropic,
	NFKRemoteBatchAPIStyleGemini,
	NFKRemoteBatchAPIStyleMistral,
	NFKRemoteBatchAPIStyleOpenRouter,
	NFKRemoteBatchAPIStyleXAI,
};

/*!
	@enum       NFKRemoteBatchStatus
	@abstract   Where a batch is, in one vocabulary over every service's own.
	@constant   NFKRemoteBatchStatusQueued Accepted and not yet running (validating, queued, pending).
	@constant   NFKRemoteBatchStatusInProgress Running.
	@constant   NFKRemoteBatchStatusFinalizing Done running and writing its results.
	@constant   NFKRemoteBatchStatusCompleted Finished; single requests may still have failed.
	@constant   NFKRemoteBatchStatusFailed The batch itself failed.
	@constant   NFKRemoteBatchStatusExpired Its window ended with requests unfinished.
	@constant   NFKRemoteBatchStatusCancelling A cancel is under way.
	@constant   NFKRemoteBatchStatusCancelled Cancelled.
	@constant   NFKRemoteBatchStatusUnknown A status the service names that none of these is.
	@discussion Introduced in InferKit 0.4.0.
*/
typedef NS_ENUM(NSInteger, NFKRemoteBatchStatus) {
	NFKRemoteBatchStatusQueued = 0,
	NFKRemoteBatchStatusInProgress,
	NFKRemoteBatchStatusFinalizing,
	NFKRemoteBatchStatusCompleted,
	NFKRemoteBatchStatusFailed,
	NFKRemoteBatchStatusExpired,
	NFKRemoteBatchStatusCancelling,
	NFKRemoteBatchStatusCancelled,
	NFKRemoteBatchStatusUnknown,
};

/*!
	@class      NFKRemoteBatchRecord
	@abstract   A batch as the service last reported it.
	@discussion The counts are what the service reports, 0 where it reports none (Together reports a
				progress percentage under raw instead). Introduced in InferKit 0.4.0.
*/
@interface NFKRemoteBatchRecord : NSObject <NSCopying>

/*! The identifier the service names the batch by (batches/… on Gemini). */
@property (nonatomic, copy, readonly) NSString *identifier;

@property (nonatomic, readonly) NFKRemoteBatchStatus status;

/*! The status as the service spells it, or nil for xAI, which reports counts alone. */
@property (nonatomic, copy, readonly, nullable) NSString *serviceStatus;

/*! Whether the batch has stopped: completed, failed, expired, or cancelled. */
@property (nonatomic, readonly, getter=isFinished) BOOL finished;

@property (nonatomic, readonly) NSInteger requestCount;
@property (nonatomic, readonly) NSInteger succeededCount;
@property (nonatomic, readonly) NSInteger failedCount;

@property (nonatomic, copy, readonly, nullable) NSDate *createdAt;
@property (nonatomic, copy, readonly, nullable) NSDate *endedAt;
@property (nonatomic, copy, readonly, nullable) NSDate *expiresAt;

/*! The service's record as it came. */
@property (nonatomic, copy, readonly) NSDictionary<NSString *, id> *raw;

/*! A record read from a service's batch object, or nil for one without an identifier. */
+ (nullable instancetype)recordWithObject:(NSDictionary<NSString *, id> *)object apiStyle:(NFKRemoteBatchAPIStyle)apiStyle;

- (instancetype)init NS_UNAVAILABLE;

@end

/*!
	@class      NFKRemoteBatchResult
	@abstract   One batched request's outcome: a result or an error, under the identifier it was sent with.
	@discussion Introduced in InferKit 0.4.0.
*/
@interface NFKRemoteBatchResult : NSObject

/*! The identifier the request was submitted under. */
@property (nonatomic, copy, readonly) NSString *customIdentifier;

/*! The result, read by the backend the batch was submitted with, or nil when the request failed. */
@property (nonatomic, strong, readonly, nullable) NFKInferenceResult *result;

/*! Why the request failed: the service's refusal mapped as NFKRemoteTransport maps a status, or
	kNFKError_InferenceBackendFailure for a request that expired or was cancelled. */
@property (nonatomic, strong, readonly, nullable) NSError *error;

/*! The service's entry for the request as it came. */
@property (nonatomic, copy, readonly) NSDictionary<NSString *, id> *raw;

- (instancetype)init NS_UNAVAILABLE;

@end

/*!
	@class      NFKRemoteBatch
	@abstract   Runs many requests as one hosted batch, which most services price at a discount and finish
				within a day.
	@discussion A batch takes requests keyed by identifiers the caller chooses, and the backend that would
				run one of them:
				- NFKRemoteBackend (chat completions), NFKAnthropicBackend (the Messages API),
				  NFKRemoteResponsesBackend (the Responses API, and Perplexity's Agent API), and
				  NFKGeminiGenerateContentBackend (generateContent)
				- NFKRemoteEmbeddingBackend (text embeddings, Gemini's native ones included)
				- NFKRemoteCompletionBackend (completions and Mistral's FIM), NFKRemoteOCRBackend,
				  NFKRemoteModerationBackend, and NFKRemoteClassifierBackend
				- NFKRemoteTranscriptionBackend (a clip given by its URL, since a batched request
				  carries no file), NFKRemoteImageBackend (an OpenAI edit as the JSON edit body), and
				  NFKRemoteVideoBackend (xAI's)
				- NFKMistralConversationsBackend (a new conversation)
				Each request goes out as the body that backend would send, to the route its URL names,
				and each result comes back read the way that backend reads a reply, so a batched
				request writes and reads like a run. One model and one route serve a whole batch.

				The flow: submitRequests:backend:error: answers the batch's record; recordOfBatch… reads
				it again, and recordWhenFinished… polls until it stops; resultsOfBatch… answers the
				results by identifier. A file-based style (OpenAI, Together, Mistral for its output)
				uploads and downloads through fileStore, which batchForProvider:apiKey: sets.

				Where the services stop:
				- Anthropic and Gemini take the requests inline: 100,000 requests or 256 MB on
				  Anthropic, under 20 MB on Gemini.
				- Mistral takes at most 10,000 inline requests.
				- Perplexity batches only Agent API requests; OpenRouter chat, Responses, Messages,
				  and embeddings; xAI chat, Responses, images, and video.
				- OpenRouter serves no cancel, and Together reports no counts.
				- A Messages request's anthropic-beta headers go on the batch for every request.
				A call the style does not serve fails with kNFKError_InferenceUnsupported. Every call
				blocks; run it off the render thread. Introduced in InferKit 0.4.0.
*/
@interface NFKRemoteBatch : NSObject

/*! The API base the batch routes sit under, for example https://api.openai.com/v1; Gemini's native
	base (…/v1beta) for its style. */
@property (nonatomic, copy, nullable) NSURL *baseURL;

@property (nonatomic, assign) NFKRemoteBatchAPIStyle apiStyle;

/*! The key, sent the way the service reads it. */
@property (nonatomic, copy, nullable) NSString *apiKey;

/*! The store a file-based style uploads its input to and downloads its output from. */
@property (nonatomic, strong, nullable) NFKRemoteFileStore *fileStore;

/*! The request timeout in seconds. Defaults to 600; a batch body is large. */
@property (nonatomic, assign) NSTimeInterval timeout;

/*! Seconds between reads in recordWhenFinishedOfBatchWithIdentifier:timeout:error:. Defaults to 30. */
@property (nonatomic, assign) NSTimeInterval pollInterval;

/*! The session used for the calls. Defaults to the shared session. */
@property (nonatomic, strong) NSURLSession *session;

+ (instancetype)batchWithBaseURL:(nullable NSURL *)baseURL apiStyle:(NFKRemoteBatchAPIStyle)apiStyle;

/*! A batch for the provider: openai, groq, modelstudio, and perplexity (OpenAI style), together,
	anthropic, gemini, mistral, openrouter, and xai serve one; every other preset returns nil. */
+ (nullable instancetype)batchForProvider:(NFKRemoteProvider *)provider apiKey:(nullable NSString *)apiKey;

/*!
	@method     submitRequests:backend:error:
	@abstract   Submits the requests as one batch and answers its record, or nil with an error.
	@discussion The identifiers are the caller's (1 to 64 letters, digits, hyphens, and underscores
				suit every service). backend is one of the five the class names and must name the
				model; its key is not used, the batch's is.
*/
- (nullable NFKRemoteBatchRecord *)submitRequests:(NSDictionary<NSString *, NFKInferenceRequest *> *)requests
										  backend:(id<NFKInferenceBackend>)backend
											error:(NSError * _Nullable *)outError;

/*! The batch as the service reports it now, or nil with an error. */
- (nullable NFKRemoteBatchRecord *)recordOfBatchWithIdentifier:(NSString *)identifier error:(NSError * _Nullable *)outError;

/*! Reads the batch every pollInterval until it stops or the timeout passes, and answers the last
	record; a timeout answers the record unfinished. Blocks for as long as the batch runs. */
- (nullable NFKRemoteBatchRecord *)recordWhenFinishedOfBatchWithIdentifier:(NSString *)identifier
																   timeout:(NSTimeInterval)timeout
																	 error:(NSError * _Nullable *)outError;

/*! The batches the key can see, newest first as the service lists them, every page read. */
- (nullable NSArray<NFKRemoteBatchRecord *> *)batchesWithError:(NSError * _Nullable *)outError;

/*! Asks the service to cancel the batch; the requests already finished keep their results. */
- (BOOL)cancelBatchWithIdentifier:(NSString *)identifier error:(NSError * _Nullable *)outError;

/*! Deletes a finished batch and its results: Anthropic, Gemini, Mistral, and OpenRouter serve it. */
- (BOOL)deleteBatchWithIdentifier:(NSString *)identifier error:(NSError * _Nullable *)outError;

/*!
	@method     resultsOfBatchWithIdentifier:requests:backend:error:
	@abstract   The results of a batch by the identifiers its requests were submitted under.
	@discussion backend reads each reply; requests, the ones submitted, lets it read each against its
				own request (a JSON schema's parsed reply needs it), and nil reads each against an empty
				one. A request that has not finished has no entry. On a finished batch every request
				has one, except where the service reports no outcome for it.
*/
- (nullable NSDictionary<NSString *, NFKRemoteBatchResult *> *)resultsOfBatchWithIdentifier:(NSString *)identifier
																				   requests:(nullable NSDictionary<NSString *, NFKInferenceRequest *> *)requests
																					backend:(id<NFKInferenceBackend>)backend
																					  error:(NSError * _Nullable *)outError;

/*! The transport seam. The default delegates to NFKRemoteTransport; a test overrides it. */
- (nullable NSData *)sendRequest:(NSURLRequest *)request
						response:(NSHTTPURLResponse * _Nullable * _Nullable)outResponse
						   error:(NSError * _Nullable *)outError;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKRemoteBatch_h */
