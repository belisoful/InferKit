//
//  NFKInferenceBackend.m
//  InferKit
//

#import "NFKInferenceBackend.h"
#import "NFKErrors.h"

BOOL NFKInferencePrepare(id<NFKInferenceBackend> backend, NSError * _Nullable * _Nullable error)
{
	if (![backend respondsToSelector:@selector(prepareWithError:)]) {
		return YES;
	}
	return [backend prepareWithError:error];
}

void NFKInferenceUnload(id<NFKInferenceBackend> backend)
{
	if ([backend respondsToSelector:@selector(unload)]) {
		[backend unload];
	}
}

BOOL NFKInferenceIsLoaded(id<NFKInferenceBackend> backend)
{
	if ([backend respondsToSelector:@selector(isLoaded)]) {
		return backend.isLoaded;
	}
	return backend.isReady;
}

NSInteger NFKInferenceResidentByteCount(id<NFKInferenceBackend> backend)
{
	if (![backend respondsToSelector:@selector(residentByteCount)]) {
		return 0;
	}
	return backend.residentByteCount;
}

NSInteger NFKInferencePeakByteCount(id<NFKInferenceBackend> backend)
{
	if (![backend respondsToSelector:@selector(peakByteCount)]) {
		return 0;
	}
	return backend.peakByteCount;
}

NFKInferenceJob *NFKInferenceSubmit(id<NFKInferenceBackend> backend,
									NFKInferenceRequest *request,
									dispatch_queue_t _Nullable queue)
{
	if ([backend respondsToSelector:@selector(submitInferenceJobForRequest:)]) {
		return [backend submitInferenceJobForRequest:request];
	}

	NFKInferenceJob *job = [[NFKInferenceJob alloc] init];
	dispatch_queue_t runQueue = queue ?: dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);
	dispatch_async(runQueue, ^{
		if (job.status == NFKInferenceJobStatusCancelled) {
			return;
		}
		// A global queue drains no pool per block, so this pool releases what the run autoreleases, such as
		// a wrapper over a C caller's buffer, when the run ends.
		@autoreleasepool {
			[job reportProgress:-1.0];
			NSError *error = nil;
			NFKInferenceResult *result = [backend runInferenceForRequest:request error:&error];
			if (result != nil) {
				[job finishWithResult:result];
			} else {
				[job finishWithError:error ?: [NSError errorWithDomain:NFKInferenceErrorDomain
																   code:kNFKError_InferenceBackendFailure
															   userInfo:@{ NSLocalizedDescriptionKey: @"the inference run failed" }]];
			}
		}
	});
	return job;
}

NSDictionary<NSString *, id> *NFKInferenceLoadOptionValues(id<NFKInferenceBackend> backend)
{
	if (![backend respondsToSelector:@selector(loadOptionValues)]) {
		return @{};
	}
	return backend.loadOptionValues ?: @{};
}
