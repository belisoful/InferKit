//
//  NFKInferenceBackend.h
//  InferKit
//

#ifndef NFKInferenceBackend_h
#define NFKInferenceBackend_h

#import <Foundation/Foundation.h>
#import "NFKInferenceRequest.h"
#import "NFKInferenceResult.h"
#import "NFKInferenceJob.h"

NS_ASSUME_NONNULL_BEGIN

/*!
	@protocol   NFKInferenceBackend
	@abstract   A swappable inference engine that turns a request into a result.
	@discussion The backend is the seam between an InferKit ML effect
				and whatever runs the model. InferKit ships engines that depend only on Apple
				frameworks (a Core ML runner, a remote OpenAI-compatible client, and a
				passthrough mock); a plugin adopts this protocol to bring a heavier runtime
				(MLX, a C or Rust engine) without InferKit linking it.

				runInferenceForRequest:error: is synchronous by contract. Inference often
				takes seconds, too long for a render call, so a caller runs it off the render
				thread and caches the result by frame. A backend reports readiness through
				isReady; a caller checks it before a run and falls back or defers when a model
				is not yet loaded.
*/
@protocol NFKInferenceBackend <NSObject>

@required

/*! YES when the backend can serve a run now: its model and resources are loaded. */
@property (nonatomic, readonly) BOOL isReady;

/*! A short stable identifier for the engine, for logging and selection (for example
	"passthrough", "coreml", "remote"). */
@property (nonatomic, readonly, copy) NSString *backendIdentifier;

/*!
	@method     runInferenceForRequest:error:
	@abstract   Runs the model synchronously and returns its outputs, or nil with an error.
	@discussion The backend reads the request's inputs and parameters, runs the model, and
				returns a result. It returns nil and sets error when it is not ready, a
				required input is missing, or the run fails. Do not call on the render thread
				for a multi-second model; run it off-thread and cache the result.
*/
- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request
													error:(NSError **)error;

@optional

/*!
	@method     prepareWithError:
	@abstract   Loads the model and resources so a later run is ready; returns YES on success.
	@discussion A caller invokes this once, off the render thread, before the first run. A
				backend with nothing to load (the passthrough mock) does not implement it or
				returns YES immediately.
*/
- (BOOL)prepareWithError:(NSError **)error;

/*!
	@method     submitInferenceJobForRequest:
	@abstract   Starts an asynchronous run and returns a job to track and cancel it.
	@discussion A backend for a slow or remote model — image and especially video generation —
				implements this instead of blocking: it returns a job immediately, reports
				progress, and finishes it once. A backend that only runs fast in-process models
				need not implement it; NFKInferenceSubmit wraps its synchronous method into a job.
*/
- (NFKInferenceJob *)submitInferenceJobForRequest:(NFKInferenceRequest *)request;

@end

/*!
	@function   NFKInferenceSubmit
	@abstract   Submits a request to a backend as a job, uniformly across sync and async backends.
	@discussion When the backend implements submitInferenceJobForRequest:, this returns that job.
				Otherwise it returns a job and runs the backend's synchronous runInferenceForRequest:
				on queue (a default background queue when queue is NULL), finishing the job with the
				result or error. The caller gets a job either way.
*/
NFKInferenceJob *NFKInferenceSubmit(id<NFKInferenceBackend> backend,
									NFKInferenceRequest *request,
									dispatch_queue_t _Nullable queue);

NS_ASSUME_NONNULL_END

#endif /* NFKInferenceBackend_h */
