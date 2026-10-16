//
//  NFKJobGuard.h
//  InferKit
//
//  What the core's job backends use where they run work on a queue of their own: an exception the work
//  raises fails the job with an error, and the process keeps running. Private to the core.
//

#import <Foundation/Foundation.h>
#import <InferKit/NFKInferenceJob.h>

NS_ASSUME_NONNULL_BEGIN

/*! The error a job fails with when its work raised: kNFKError_InferenceBackendFailure in
	NFKInferenceErrorDomain, naming the exception and its reason. */
FOUNDATION_EXPORT NSError *NFKJobExceptionError(NSException *exception);

/*! Runs work, and fails job with NFKJobExceptionError when work raises. A job that already finished
	stays as it finished. */
FOUNDATION_EXPORT void NFKJobRunGuarded(NFKInferenceJob *job, void (NS_NOESCAPE ^work)(void));

/*! Runs work asynchronously on the user-initiated global queue through NFKJobRunGuarded. */
FOUNDATION_EXPORT void NFKJobDispatchGuarded(NFKInferenceJob *job, void (^work)(void));

NS_ASSUME_NONNULL_END
