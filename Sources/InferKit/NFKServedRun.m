//
//  NFKServedRun.m
//  InferKit
//

#import "NFKServedRun.h"
#import <InferKit/NFKInferenceKeys.h>
#import <InferKit/NFKErrors.h>

NSError *NFKServedCancellation(void)
{
	return [NSError errorWithDomain:NSCocoaErrorDomain code:NSUserCancelledError
						   userInfo:@{ NSLocalizedDescriptionKey: @"the client went away and the run was cancelled" }];
}

BOOL NFKServedIsCancellation(NSError * _Nullable error)
{
	return error.code == NSUserCancelledError && [error.domain isEqualToString:NSCocoaErrorDomain];
}

@implementation NFKServedRun

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_identifier = NSUUID.UUID.UUIDString;
		_route = @"";
		_clientAddress = @"";
	}
	return self;
}

+ (instancetype)tokenCountRunForRequest:(NFKInferenceRequest *)request modelName:(NSString *)modelName
{
	NFKServedRun *run = [[self alloc] init];
	run.request = request;
	run.unmeasured = YES;
	run.work = ^NFKInferenceResult *(id<NFKInferenceBackend> backend, NSError **workError) {
		if (![backend respondsToSelector:@selector(tokenCountForRequest:error:)]) {
			if (workError != NULL) {
				*workError = [NSError errorWithDomain:NFKInferenceErrorDomain code:kNFKError_InferenceUnsupported
											 userInfo:@{ NSLocalizedDescriptionKey: [NSString stringWithFormat:@"the model \"%@\" does not count tokens", modelName] }];
			}
			return nil;
		}
		NSNumber *count = [backend tokenCountForRequest:request error:workError];
		return count != nil ? [NFKInferenceResult resultWithOutputs:@{ NFKOutputUsage: @{ NFKUsageInputTokens: count } }] : nil;
	};
	return run;
}

- (void)cancel
{
	NFKInferenceJob *job = nil;
	@synchronized (self) {
		self.cancelled = YES;
		job = self.job;
	}
	[job cancel];
}

- (void)finishWithResult:(nullable NFKInferenceResult *)result error:(nullable NSError *)error
{
	void (^handler)(NFKInferenceResult * _Nullable, NSError * _Nullable) = nil;
	@synchronized (self) {
		if (self.finished) {
			return;
		}
		self.finished = YES;
		handler = self.completionHandler;
		self.completionHandler = nil;
		self.partialHandler = nil;
	}
	if (result == nil && error == nil) {
		error = self.cancelled ? NFKServedCancellation()
			: [NSError errorWithDomain:NFKInferenceErrorDomain code:kNFKError_InferenceBackendFailure
							  userInfo:@{ NSLocalizedDescriptionKey: @"the backend returned no result and no error" }];
	}
	if (handler != nil) {
		handler(result, error);
	}
	for (NSURL *file in self.temporaryFiles) {
		[NSFileManager.defaultManager removeItemAtURL:file error:NULL];
	}
}

- (void)reportPartial:(nullable NFKInferenceResult *)partial progress:(double)progress
{
	void (^handler)(NFKInferenceResult * _Nullable, double) = nil;
	@synchronized (self) {
		handler = self.finished ? nil : self.partialHandler;
	}
	if (handler != nil) {
		handler(partial, progress);
	}
}

@end
