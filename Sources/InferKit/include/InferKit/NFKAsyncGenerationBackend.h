//
//  NFKAsyncGenerationBackend.h
//  InferKit
//

#ifndef NFKAsyncGenerationBackend_h
#define NFKAsyncGenerationBackend_h

#import <Foundation/Foundation.h>
#import "NFKInferenceBackend.h"

NS_ASSUME_NONNULL_BEGIN

/*!
	@class      NFKAsyncGenerationBackend
	@abstract   A base backend for submit-poll-fetch generation services (image and video).
	@discussion Image and video generation APIs are asynchronous jobs:
				POST a request, receive a job id, poll a status endpoint until it completes, then
				read the output URL. This class owns that lifecycle, driving an NFKInferenceJob with
				progress and cancellation. It is provider-neutral: subclasses override the schema
				hooks to build the submit body and read a provider's status and result shapes, and
				the transport seam to send the requests.

				runInferenceForRequest: satisfies the synchronous protocol contract by running the
				job to completion and blocking, so callers that only have the sync entry still work.
				Prefer submitInferenceJobForRequest: for the progress and cancellation a long
				generation needs.
*/
@interface NFKAsyncGenerationBackend : NSObject <NFKInferenceBackend>

/*! The submit endpoint. isReady is YES when it is set. */
@property (nonatomic, copy, nullable) NSURL *submitURL;

/*! The bearer token sent as Authorization, when the service needs one. */
@property (nonatomic, copy, nullable) NSString *apiKey;

/*! The model name sent in the submit body. */
@property (nonatomic, copy, nullable) NSString *modelName;

/*! Seconds between status polls. Defaults to 2. */
@property (nonatomic, assign) NSTimeInterval pollInterval;

/*! The session used for the calls. Defaults to the shared session. */
@property (nonatomic, strong) NSURLSession *session;

#pragma mark Schema hooks (override per provider)

/*! The JSON submit body. The default sends the model and the request's inputs and parameters. */
- (NSDictionary<NSString *, id> *)submitBodyForRequest:(NFKInferenceRequest *)request;

/*! The provider job id from a submit response. The default reads the "id" key. */
- (nullable NSString *)jobIdentifierFromResponse:(NSDictionary *)response;

/*! The status URL for a job id. The default appends the id to submitURL. */
- (nullable NSURL *)statusURLForJobIdentifier:(NSString *)jobIdentifier;

/*! Progress 0 to 1 from a status response, or negative when unknown. The default reads "progress". */
- (double)progressFromStatusResponse:(NSDictionary *)response;

/*! YES when a status response reports completion. The default matches status "succeeded"/"completed". */
- (BOOL)isSucceededStatusResponse:(NSDictionary *)response;

/*! YES when a status response reports failure. The default matches status "failed"/"error". */
- (BOOL)isFailedStatusResponse:(NSDictionary *)response;

/*! The result from a completed status response. The default reads an output URL and returns an
	NFKVideoAsset under NFKOutputVideo for a video request, or the URL under NFKOutputImage. */
- (nullable NFKInferenceResult *)resultFromStatusResponse:(NSDictionary *)response
											 outputModality:(NFKModality)outputModality;

#pragma mark Transport seam

/*! Sends a request synchronously and returns its JSON object, or nil with an error. Overridden by
	tests and alternative transports. */
- (nullable NSDictionary *)sendJSONRequest:(NSURLRequest *)request error:(NSError * _Nullable *)outError;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKAsyncGenerationBackend_h */
