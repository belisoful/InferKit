//
//  NFKInferenceJob.m
//  InferKit
//

#import "NFKInferenceJob.h"

@implementation NFKInferenceJob
{
	NSLock *_lock;
	NFKInferenceJobStatus _status;
	double _progress;
	NFKInferenceResult *_result;
	NFKInferenceResult *_partialResult;
	NSError *_error;
	void (^_completionHandler)(NFKInferenceJob *);
}

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_lock = [[NSLock alloc] init];
		_status = NFKInferenceJobStatusQueued;
		_progress = -1.0;
	}
	return self;
}

#pragma mark Reads

- (NFKInferenceJobStatus)status
{
	[_lock lock];
	NFKInferenceJobStatus status = _status;
	[_lock unlock];
	return status;
}

- (double)progress
{
	[_lock lock];
	double progress = _progress;
	[_lock unlock];
	return progress;
}

- (NFKInferenceResult *)result
{
	[_lock lock];
	NFKInferenceResult *result = _result;
	[_lock unlock];
	return result;
}

- (NFKInferenceResult *)partialResult
{
	[_lock lock];
	NFKInferenceResult *partialResult = _partialResult;
	[_lock unlock];
	return partialResult;
}

- (NSError *)error
{
	[_lock lock];
	NSError *error = _error;
	[_lock unlock];
	return error;
}

- (BOOL)isTerminalStatus:(NFKInferenceJobStatus)status
{
	return status == NFKInferenceJobStatusSucceeded
		|| status == NFKInferenceJobStatusFailed
		|| status == NFKInferenceJobStatusCancelled;
}

#pragma mark completionHandler (fires immediately if already terminal)

- (void)setCompletionHandler:(void (^)(NFKInferenceJob *))completionHandler
{
	[_lock lock];
	_completionHandler = [completionHandler copy];
	BOOL alreadyFinished = [self isTerminalStatus:_status];
	void (^handler)(NFKInferenceJob *) = _completionHandler;
	[_lock unlock];

	if (alreadyFinished && handler != nil) {
		handler(self);
	}
}

- (void (^)(NFKInferenceJob *))completionHandler
{
	[_lock lock];
	void (^handler)(NFKInferenceJob *) = _completionHandler;
	[_lock unlock];
	return handler;
}

#pragma mark Producer

- (void)reportProgress:(double)progress
{
	[self reportProgress:progress partialResult:nil];
}

- (void)reportProgress:(double)progress partialResult:(nullable NFKInferenceResult *)partialResult
{
	[_lock lock];
	if ([self isTerminalStatus:_status]) {
		[_lock unlock];
		return;
	}
	if (_status == NFKInferenceJobStatusQueued) {
		_status = NFKInferenceJobStatusRunning;
	}
	_progress = progress;
	if (partialResult != nil) {
		_partialResult = partialResult;
	}
	[_lock unlock];

	if (self.progressHandler != nil) {
		self.progressHandler(self);
	}
}

- (void)finishWithResult:(NFKInferenceResult *)result
{
	[self finishWithStatus:NFKInferenceJobStatusSucceeded result:result error:nil];
}

- (void)finishWithError:(NSError *)error
{
	[self finishWithStatus:NFKInferenceJobStatusFailed result:nil error:error];
}

- (void)cancel
{
	[_lock lock];
	if ([self isTerminalStatus:_status]) {
		[_lock unlock];
		return;
	}
	_status = NFKInferenceJobStatusCancelled;
	void (^cancellation)(void) = self.cancellationHandler;
	void (^completion)(NFKInferenceJob *) = _completionHandler;
	[_lock unlock];

	if (cancellation != nil) {
		cancellation();
	}
	if (completion != nil) {
		completion(self);
	}
}

- (void)finishWithStatus:(NFKInferenceJobStatus)status
				  result:(nullable NFKInferenceResult *)result
				   error:(nullable NSError *)error
{
	[_lock lock];
	if ([self isTerminalStatus:_status]) {
		[_lock unlock];
		return;
	}
	_status = status;
	_result = result;
	_error = error;
	if (result != nil) {
		_progress = 1.0;
	}
	void (^completion)(NFKInferenceJob *) = _completionHandler;
	[_lock unlock];

	if (completion != nil) {
		completion(self);
	}
}

@end
