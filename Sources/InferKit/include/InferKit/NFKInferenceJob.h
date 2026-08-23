//
//  NFKInferenceJob.h
//  InferKit
//

#ifndef NFKInferenceJob_h
#define NFKInferenceJob_h

#import <Foundation/Foundation.h>
#import "NFKInferenceResult.h"

NS_ASSUME_NONNULL_BEGIN

/*!
	@enum       NFKInferenceJobStatus
	@abstract   The lifecycle state of an inference job.
*/
typedef NS_ENUM(NSInteger, NFKInferenceJobStatus) {
	NFKInferenceJobStatusQueued		= 0,
	NFKInferenceJobStatusRunning	= 1,
	NFKInferenceJobStatusSucceeded	= 2,
	NFKInferenceJobStatusFailed		= 3,
	NFKInferenceJobStatusCancelled	= 4,
};

/*!
	@class      NFKInferenceJob
	@abstract   A handle to an asynchronous inference run, for models too slow to run inline.
	@discussion Image and especially video generation take seconds to
				minutes and often run as a submit-poll-fetch job on a remote service, so a caller
				needs progress and cancellation rather than a blocking call. A backend returns a
				job, drives it through the producer methods, and finishes it once; the caller reads
				status and progress, sets completionHandler, and may cancel.

				The job is thread-safe. Terminal states (succeeded, failed, cancelled) are final:
				later producer calls are ignored. Setting completionHandler after the job has
				already finished invokes it immediately, so a caller never misses the result.
*/
@interface NFKInferenceJob : NSObject

#pragma mark Consumer

/*! The current lifecycle state. */
@property (nonatomic, readonly) NFKInferenceJobStatus status;

/*! Progress from 0 to 1, or a negative value when the backend cannot report it. */
@property (nonatomic, readonly) double progress;

/*! The result once the job has succeeded, otherwise nil. */
@property (nonatomic, readonly, nullable) NFKInferenceResult *result;

/*! The latest partial result while the job runs, for a backend that streams (a language model
	emitting text token by token), otherwise nil. Read it in progressHandler. */
@property (nonatomic, readonly, nullable) NFKInferenceResult *partialResult;

/*! The error once the job has failed, otherwise nil. */
@property (nonatomic, readonly, nullable) NSError *error;

/*! Invoked once when the job reaches a terminal state. Set after the job has finished, it fires
	immediately. */
@property (nonatomic, copy, nullable) void (^completionHandler)(NFKInferenceJob *job);

/*! Invoked whenever progress changes, on the thread that reported it. */
@property (nonatomic, copy, nullable) void (^progressHandler)(NFKInferenceJob *job);

/*! Cancels the job. Runs the backend's cancellationHandler and moves to the cancelled state,
	unless the job has already finished. */
- (void)cancel;

#pragma mark Producer (backends)

/*! A block the job runs on cancel so the backend can abort the underlying work. */
@property (nonatomic, copy, nullable) void (^cancellationHandler)(void);

/*! Moves a queued job to running and records progress. Ignored once terminal. */
- (void)reportProgress:(double)progress;

/*! Records progress and the latest partial result (when non-nil), then fires progressHandler.
	Ignored once terminal. A streaming backend reports the text so far after each token. */
- (void)reportProgress:(double)progress partialResult:(nullable NFKInferenceResult *)partialResult;

/*! Finishes the job with a result. Ignored once terminal. */
- (void)finishWithResult:(NFKInferenceResult *)result NS_SWIFT_NAME(finish(with:));

/*! Finishes the job with an error. Ignored once terminal. */
- (void)finishWithError:(NSError *)error NS_SWIFT_NAME(finish(withError:));

@end

NS_ASSUME_NONNULL_END

#endif /* NFKInferenceJob_h */
