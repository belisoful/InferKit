//
//  NFKJobGuard.m
//  InferKit
//

#import "NFKJobGuard.h"
#import <InferKit/NFKErrors.h>

NSError *NFKJobExceptionError(NSException *exception)
{
	NSString *reason = [NSString stringWithFormat:@"the backend raised %@: %@", exception.name, exception.reason ?: @"no reason given"];
	return [NSError errorWithDomain:NFKInferenceErrorDomain code:kNFKError_InferenceBackendFailure
						   userInfo:@{ NSLocalizedDescriptionKey: reason }];
}

void NFKJobRunGuarded(NFKInferenceJob *job, void (NS_NOESCAPE ^work)(void))
{
	@try {
		work();
	} @catch (NSException *exception) {
		[job finishWithError:NFKJobExceptionError(exception)];
	}
}

void NFKJobDispatchGuarded(NFKInferenceJob *job, void (^work)(void))
{
	dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
		NFKJobRunGuarded(job, work);
	});
}
