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

/*!
	@property   supportedParameterKeys
	@abstract   The NFKParameter* keys the backend reads, for a caller that adapts to the engine.
	@discussion A request carries whatever keys a caller sets, and a backend ignores the ones it
				does not implement. A caller that has to know in advance reads this set: a bridge
				that presents the backend to another framework declares the framework's
				capabilities from it (NFKParameterJSONSchema for guided generation,
				NFKParameterTools for tool calling), and a router picks the engine that honors the
				key a request needs.

				A backend declares only the keys it acts on. Absent the property, a caller assumes
				nothing beyond the required protocol. Introduced in InferKit 0.4.0.
*/
@property (nonatomic, readonly, copy) NSSet<NSString *> *supportedParameterKeys;

/*!
	@property   supportedInputKeys
	@abstract   The NFKInput* keys the backend reads, for a caller that adapts to the engine.
	@discussion The counterpart of supportedParameterKeys for inputs: NFKInputImage on a backend
				that takes an image beside the text, NFKInputMessages on one that takes a
				conversation. Introduced in InferKit 0.4.0.
*/
@property (nonatomic, readonly, copy) NSSet<NSString *> *supportedInputKeys;

/*!
	@property   supportedOutputKeys
	@abstract   The NFKOutput* keys the backend writes, for a caller that adapts to the engine.
	@discussion The counterpart of supportedInputKeys for results: NFKOutputText on a backend that
				answers in text, NFKOutputImage and NFKOutputMask on one that returns a matte beside
				the image. A key the backend writes only for some requests is in the set.
				+[NFKModelDescriptor descriptorWithIdentifier:derivedFromBackend:] reads it for a
				descriptor's outputs. Introduced in InferKit 0.4.0.
*/
@property (nonatomic, readonly, copy) NSSet<NSString *> *supportedOutputKeys;

/*!
	@property   modelInfo
	@abstract   What the backend knows about its loaded model, keyed by NFKModelInfo*.
	@discussion The values are strings and numbers: parameter count, weight and storage bytes,
				precision, quantization, context length, key-value bytes per token, compute units,
				and version. A backend fills the keys it knows and leaves the rest out, so a caller
				reads a key that is present rather than trusting a zero. A backend that has not
				loaded its model yet reports what its configuration already fixes, such as its
				compute units. NFKInferenceServer serves it under each model's entry, so a load
				balancer or a client can compare the models different machines host. Introduced in
				InferKit 0.4.0.
*/
@property (nonatomic, readonly, copy) NSDictionary<NSString *, id> *modelInfo;

/*!
	@property   backendStatus
	@abstract   What the backend reports about its state now, such as the caches it holds or the
				servers it spreads requests over.
	@discussion The values are strings, numbers, and booleans, and arrays and dictionaries of them.
				NFKInferenceServer reads it on every status request and serves it under each model's
				"status", leaving out any value of another kind, so a backend answers from state it
				keeps current without waiting on a run in progress. Introduced in InferKit 0.4.0.
*/
@property (nonatomic, readonly, copy) NSDictionary<NSString *, id> *backendStatus;

/*!
	@method     unload
	@abstract   Frees the model's weights and caches and keeps what the backend needs to load them again.
	@discussion The next run, or prepareWithError:, loads the model again from the same source, so an app
				frees a model's memory and keeps the backend with its configuration. The identifier,
				the declared keys, and isReady keep their answers while the model is unloaded.

				It is safe to call from any thread, and it never cuts a run:
				- no run in flight → the backend frees the model before unload returns;
				- runs in flight → unload returns at once, and the backend frees the model when the last
				  of them ends. isLoaded stays YES until then, and a run that starts in the meantime runs
				  on the loaded model.

				Because unload never waits, a run's completion handler may call it. A backend that cannot
				load its model again, such as one built from objects in memory, makes unload a no-op and
				reports isLoaded YES. NFKInferenceUnload calls it uniformly. Introduced in InferKit 0.4.0.
*/
- (void)unload;

/*!
	@property   isLoaded
	@abstract   YES while the backend holds its model in memory.
	@discussion NO from the moment unload frees the model until a run or prepareWithError: loads it again.
				It is safe to read from any thread. NFKInferenceIsLoaded reads it uniformly. Introduced in
				InferKit 0.4.0.
*/
@property (nonatomic, readonly) BOOL isLoaded;

/*!
	@property   residentByteCount
	@abstract   The bytes the loaded model holds in memory, measured when it loaded; 0 while it is not loaded.
	@discussion The backend reads its runtime's allocated memory before the load and after it, with the
				weights read, and reports the difference. The reading covers the whole process, so
				memory that other work allocates or frees during the load falls into the figure, and
				arrays a loader leaves for the first run to read are not in it. A backend that has not
				measured a load reports 0. It is safe to read from any thread. Introduced in InferKit 0.4.0.
*/
@property (nonatomic, readonly) NSInteger residentByteCount;

/*!
	@property   peakByteCount
	@abstract   The most the runtime's allocated memory rose above its level before the load, while the
				model last loaded; 0 before a measured load.
	@discussion It is the memory a load needs, which can exceed what the model then keeps: a loader that
				converts weights holds both forms for a moment. The figure stays after unload, so a caller
				knows what loading again costs. The runtime keeps one peak for the whole process, so a load
				that overlaps other work counts that work's allocations too. It is safe to read from any
				thread. Introduced in InferKit 0.4.0.
*/
@property (nonatomic, readonly) NSInteger peakByteCount;

@end

/*!
	@function   NFKInferencePrepare
	@abstract   Prepares a backend, uniformly across backends that need it and backends that do not.
	@discussion When the backend implements prepareWithError:, this calls it and returns its result.
				Otherwise it returns YES, because a backend with nothing to load is ready. A caller
				that holds a backend through the protocol uses this rather than testing the
				selector itself. Introduced in InferKit 0.4.0.
*/
BOOL NFKInferencePrepare(id<NFKInferenceBackend> backend, NSError * _Nullable * _Nullable error);

/*!
	@function   NFKInferenceUnload
	@abstract   Unloads a backend's model when the backend implements unload, and does nothing otherwise.
	@discussion Introduced in InferKit 0.4.0.
*/
void NFKInferenceUnload(id<NFKInferenceBackend> backend);

/*!
	@function   NFKInferenceIsLoaded
	@abstract   Whether a backend holds its model: its isLoaded, or its isReady when it does not implement
				isLoaded.
	@discussion Introduced in InferKit 0.4.0.
*/
BOOL NFKInferenceIsLoaded(id<NFKInferenceBackend> backend);

/*!
	@function   NFKInferenceResidentByteCount
	@abstract   A backend's residentByteCount, or 0 when it does not implement it.
	@discussion Introduced in InferKit 0.4.0.
*/
NSInteger NFKInferenceResidentByteCount(id<NFKInferenceBackend> backend);

/*!
	@function   NFKInferencePeakByteCount
	@abstract   A backend's peakByteCount, or 0 when it does not implement it.
	@discussion Introduced in InferKit 0.4.0.
*/
NSInteger NFKInferencePeakByteCount(id<NFKInferenceBackend> backend);

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
