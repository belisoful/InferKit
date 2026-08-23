//
//  NFKInferenceBackend.m
//  InferKit
//

#import "NFKInferenceBackend.h"
#import "NFKErrors.h"

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
	});
	return job;
}
