//
//  NFKTestJobs.h
//  InferKitTests
//
//  A wait for the test backends whose jobs stream partials to a server.
//

#import <Foundation/Foundation.h>
#import <InferKit/NFKInferenceJob.h>

NS_ASSUME_NONNULL_BEGIN

/*! Blocks until a consumer has set the job's completionHandler or the timeout passes, and answers
	whether one was set. A server sets its progressHandler before its completionHandler, so a test job
	that reports after this returns YES reaches the server with each partial as its own report. */
static inline BOOL NFKTestJobAwaitConsumer(NFKInferenceJob *job, NSTimeInterval timeout)
{
	NSDate *deadline = [NSDate dateWithTimeIntervalSinceNow:timeout];
	while (job.completionHandler == nil) {
		if (deadline.timeIntervalSinceNow <= 0) {
			return NO;
		}
		[NSThread sleepForTimeInterval:0.001];
	}
	return YES;
}

NS_ASSUME_NONNULL_END
