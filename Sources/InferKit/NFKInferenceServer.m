//
//  NFKInferenceServer.m
//  InferKit
//

#import <InferKit/NFKInferenceServer.h>
#import <InferKit/NFKInferenceKeys.h>
#import <InferKit/NFKInferKit.h>
#import <InferKit/NFKErrors.h>
#import <InferKit/NFKRemoteTransport.h>
#import <InferKit/NFKImageCoding.h>
#import "NFKHTTPServer.h"
#import "NFKServedOpenAI.h"
#import "NFKInferenceWireCoding.h"
#import "NFKServedHostStatus.h"

const uint16_t NFKInferenceServerDefaultPort = 11480;
NSString * const NFKInferenceServerServiceType = @"_inferkit._tcp";
NSString * const NFKInferenceServerErrorDomain = @"NFKInferenceServerErrorDomain";

static NSString * const NFKServedBasePath = @"/v1";

static NSError *NFKServedError(NSString *domain, NSInteger code, NSString *reason)
{
	return [NSError errorWithDomain:domain code:code userInfo:@{ NSLocalizedDescriptionKey: reason }];
}

static NSError *NFKServedCancellation(void)
{
	return [NSError errorWithDomain:NSCocoaErrorDomain code:NSUserCancelledError
						   userInfo:@{ NSLocalizedDescriptionKey: @"the client went away and the run was cancelled" }];
}

static BOOL NFKServedIsCancellation(NSError * _Nullable error)
{
	return error.code == NSUserCancelledError && [error.domain isEqualToString:NSCocoaErrorDomain];
}

/*! Monotonic seconds, for durations. */
static NSTimeInterval NFKServedNow(void)
{
	return NSProcessInfo.processInfo.systemUptime;
}

/*! An exponential moving average that weights the newest sample by a fifth; the first sample is the
	average. */
static double NFKServedAveraged(double average, double sample, NSUInteger previousSamples)
{
	return previousSamples == 0 ? sample : average + 0.2 * (sample - average);
}

static NSString *NFKServedJSONString(id object)
{
	NSData *data = [NSJSONSerialization dataWithJSONObject:object options:0 error:NULL];
	return data != nil ? [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding] : @"{}";
}

/*! The suffix of text past prefix, or nil when text does not extend prefix. */
static NSString * _Nullable NFKServedSuffix(NSString * _Nullable text, NSString *prefix)
{
	// hasPrefix: answers NO for an empty prefix, so the first delta is the whole text.
	if (text.length <= prefix.length || (prefix.length > 0 && ![text hasPrefix:prefix])) {
		return nil;
	}
	return [text substringFromIndex:prefix.length];
}

#pragma mark - Runs

/*! One request waiting for, or holding, a slot on its model. Finishes exactly once. */
@interface NFKServedRun : NSObject
@property (nonatomic, strong, nullable) NFKInferenceRequest *request;
@property (nonatomic, copy, nullable) NFKInferenceResult * _Nullable (^work)(id<NFKInferenceBackend> backend, NSError * _Nullable * _Nullable error);
@property (nonatomic, copy, nullable) void (^partialHandler)(NFKInferenceResult * _Nullable partial, double progress);
@property (nonatomic, copy, nullable) void (^completionHandler)(NFKInferenceResult * _Nullable result, NSError * _Nullable error);
@property (nonatomic, strong, nullable) NFKInferenceJob *job;
/*! Files written for the request's inline media, removed once the reply has been built, since a
	result may name one of them. */
@property (nonatomic, copy, nullable) NSArray<NSURL *> *temporaryFiles;
@property (atomic, assign) BOOL cancelled;
@property (atomic, assign) BOOL finished;
@property (nonatomic, assign) NSTimeInterval enqueuedAt;
@property (nonatomic, assign) NSTimeInterval startedAt;
/*! The job's last reported progress, 0 until it reports one. */
@property (atomic, assign) double reportedProgress;
@end

@implementation NFKServedRun

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
			: NFKServedError(NFKInferenceErrorDomain, kNFKError_InferenceBackendFailure, @"the backend returned no result and no error");
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

/*! A hosted backend, the queue of runs waiting for it, and what its runs have measured. Every
	property past backend is read and written on the model's lock. */
@interface NFKServedModel : NSObject
@property (nonatomic, copy) NSString *name;
@property (nonatomic, strong) id<NFKInferenceBackend> backend;
@property (nonatomic, assign) NSUInteger limit;
/*! The most runs that wait for a slot; 0 queues without limit. */
@property (nonatomic, assign) NSUInteger queueLimit;
@property (nonatomic, assign) BOOL removed;
@property (nonatomic, strong) NSMutableArray<NFKServedRun *> *running;
@property (nonatomic, strong) NSMutableArray<NFKServedRun *> *pending;
@property (nonatomic, assign) NSUInteger started;
@property (nonatomic, assign) NSUInteger completed;
@property (nonatomic, assign) NSUInteger failed;
@property (nonatomic, assign) NSUInteger cancelled;
@property (nonatomic, assign) NSUInteger refused;
@property (nonatomic, assign) NSUInteger tokenSamples;
@property (nonatomic, assign) double averageRunSeconds;
@property (nonatomic, assign) double averageWaitSeconds;
@property (nonatomic, assign) double outputTokensPerSecond;
@end

@implementation NFKServedModel

- (void)enqueue:(NFKServedRun *)run
{
	run.enqueuedAt = NFKServedNow();
	BOOL startNow = NO;
	BOOL refused = NO;
	BOOL full = NO;
	@synchronized (self) {
		refused = self.removed;
		if (!refused && self.running.count < self.limit) {
			[self beginRun:run];
			startNow = YES;
		} else if (!refused && self.queueLimit > 0 && [self queuedCount] >= self.queueLimit) {
			self.refused += 1;
			full = YES;
		} else if (!refused) {
			[self.pending addObject:run];
		}
	}
	if (refused) {
		[run finishWithResult:nil error:NFKServedError(NFKInferenceServerErrorDomain, NFKInferenceServerErrorModelNotFound,
													   [NSString stringWithFormat:@"the model \"%@\" is no longer served", self.name])];
	} else if (full) {
		[run finishWithResult:nil error:[self busyError]];
	} else if (startNow) {
		[self start:run];
	}
}

/*! Takes a slot for a run. Runs on the lock. */
- (void)beginRun:(NFKServedRun *)run
{
	run.startedAt = NFKServedNow();
	self.averageWaitSeconds = NFKServedAveraged(self.averageWaitSeconds, run.startedAt - run.enqueuedAt, self.started);
	self.started += 1;
	[self.running addObject:run];
}

- (void)start:(NFKServedRun *)run
{
	if (run.cancelled) {
		[self run:run endedWithResult:nil error:NFKServedCancellation()];
		[run finishWithResult:nil error:NFKServedCancellation()];
		return;
	}
	id<NFKInferenceBackend> backend = self.backend;
	if (run.work == nil && [backend respondsToSelector:@selector(submitInferenceJobForRequest:)]) {
		NFKInferenceJob *job = [backend submitInferenceJobForRequest:run.request];
		BOOL cancelled = NO;
		@synchronized (run) {
			run.job = job;
			cancelled = run.cancelled;
		}
		__weak NFKServedRun *weakRun = run;
		job.progressHandler = ^(NFKInferenceJob *progressed) {
			NFKServedRun *progressing = weakRun;
			double progress = progressed.progress;
			if (isfinite(progress) && progress > 0) {
				progressing.reportedProgress = MIN(progress, 1.0);
			}
			[progressing reportPartial:progressed.partialResult progress:progress];
		};
		job.completionHandler = ^(NFKInferenceJob *finished) {
			NSError *error = finished.status == NFKInferenceJobStatusCancelled ? NFKServedCancellation() : finished.error;
			[self run:run endedWithResult:finished.result error:error];
			[run finishWithResult:finished.result error:error];
		};
		if (cancelled) {
			[job cancel];
		}
		return;
	}
	// A synchronous backend holds its slot until its call returns, cancelled or not: it cannot be
	// interrupted, and the next run must not overlap it.
	dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
		NSError *error = nil;
		NFKInferenceResult *result = nil;
		if (!run.cancelled) {
			result = run.work != nil ? run.work(backend, &error) : [backend runInferenceForRequest:run.request error:&error];
		}
		if (run.cancelled) {
			result = nil;
			error = NFKServedCancellation();
		}
		[self run:run endedWithResult:result error:error];
		[run finishWithResult:result error:error];
	});
}

/*! Releases a run's slot, records how it ended, and starts the next waiting run. */
- (void)run:(NFKServedRun *)run endedWithResult:(nullable NFKInferenceResult *)result error:(nullable NSError *)error
{
	NFKServedRun *next = nil;
	@synchronized (self) {
		[self.running removeObjectIdenticalTo:run];
		[self recordRun:run result:result error:error];
		if (self.pending.count > 0 && self.running.count < self.limit) {
			next = self.pending.firstObject;
			[self.pending removeObjectAtIndex:0];
			[self beginRun:next];
		}
	}
	if (next != nil) {
		[self start:next];
	}
}

/*! Runs on the lock. Only a run that returned a result feeds the averages, so a fast refusal does
	not shorten them. */
- (void)recordRun:(NFKServedRun *)run result:(nullable NFKInferenceResult *)result error:(nullable NSError *)error
{
	if (run.cancelled || NFKServedIsCancellation(error)) {
		self.cancelled += 1;
		return;
	}
	if (result == nil) {
		self.failed += 1;
		return;
	}
	double seconds = NFKServedNow() - run.startedAt;
	self.averageRunSeconds = NFKServedAveraged(self.averageRunSeconds, seconds, self.completed);
	self.completed += 1;
	NSDictionary *usage = [result outputForKey:NFKOutputUsage];
	NSNumber *tokens = [usage isKindOfClass:NSDictionary.class] ? usage[NFKUsageOutputTokens] : nil;
	if ([tokens isKindOfClass:NSNumber.class] && tokens.doubleValue > 0 && seconds > 0) {
		self.outputTokensPerSecond = NFKServedAveraged(self.outputTokensPerSecond, tokens.doubleValue / seconds, self.tokenSamples);
		self.tokenSamples += 1;
	}
}

/*! Waiting runs whose clients are still there. Runs on the lock. */
- (NSUInteger)queuedCount
{
	NSUInteger count = 0;
	for (NFKServedRun *run in self.pending) {
		count += run.cancelled ? 0 : 1;
	}
	return count;
}

/*! Seconds until a running run ends: from its reported progress when it has one, else from the
	average run, or nil before any run has finished. Runs on the lock. */
- (nullable NSNumber *)estimatedRemainingForRun:(NFKServedRun *)run now:(NSTimeInterval)now
{
	double elapsed = now - run.startedAt;
	double progress = run.reportedProgress;
	if (progress > 0 && progress < 1) {
		return @(elapsed * (1 - progress) / progress);
	}
	if (self.completed == 0) {
		return nil;
	}
	return @(MAX(self.averageRunSeconds - elapsed, 0.0));
}

/*! Seconds until a run that arrives now would start, or nil when no estimate exists yet. Each slot
	frees when its run ends, and each waiting run then holds the soonest free slot for an average
	run. Runs on the lock. */
- (nullable NSNumber *)estimatedWaitAt:(NSTimeInterval)now
{
	NSUInteger queued = [self queuedCount];
	if (self.running.count < self.limit && queued == 0) {
		return @0;
	}
	NSMutableArray<NSNumber *> *slots = [NSMutableArray arrayWithCapacity:self.limit];
	for (NFKServedRun *run in self.running) {
		NSNumber *remaining = [self estimatedRemainingForRun:run now:now];
		if (remaining == nil) {
			return nil;
		}
		[slots addObject:remaining];
	}
	while (slots.count < self.limit) {
		[slots addObject:@0];
	}
	if (queued > 0 && self.completed == 0) {
		return nil;
	}
	for (NSUInteger index = 0; index < queued; index++) {
		NSUInteger soonest = [slots indexOfObject:[slots valueForKeyPath:@"@min.self"]];
		slots[soonest] = @(slots[soonest].doubleValue + self.averageRunSeconds);
	}
	return [slots valueForKeyPath:@"@min.self"];
}

- (NSDictionary<NSString *, id> *)loadJSONObject
{
	@synchronized (self) {
		NSMutableDictionary<NSString *, id> *load = [NSMutableDictionary dictionary];
		load[@"limit"] = @(self.limit);
		load[@"running"] = @(self.running.count);
		load[@"queued"] = @([self queuedCount]);
		if (self.queueLimit > 0) {
			load[@"queue_limit"] = @(self.queueLimit);
		}
		load[@"completed"] = @(self.completed);
		load[@"failed"] = @(self.failed);
		load[@"cancelled"] = @(self.cancelled);
		load[@"refused"] = @(self.refused);
		if (self.completed > 0) {
			load[@"average_run_seconds"] = @(self.averageRunSeconds);
		}
		if (self.started > 0) {
			load[@"average_wait_seconds"] = @(self.averageWaitSeconds);
		}
		if (self.tokenSamples > 0) {
			load[@"output_tokens_per_second"] = @(self.outputTokensPerSecond);
		}
		NSTimeInterval now = NFKServedNow();
		load[@"estimated_wait_seconds"] = [self estimatedWaitAt:now];
		NSMutableArray *runs = [NSMutableArray arrayWithCapacity:self.running.count];
		for (NFKServedRun *run in self.running) {
			NSMutableDictionary<NSString *, id> *entry = [NSMutableDictionary dictionary];
			entry[@"elapsed_seconds"] = @(now - run.startedAt);
			if (run.reportedProgress > 0) {
				entry[@"progress"] = @(run.reportedProgress);
			}
			entry[@"estimated_remaining_seconds"] = [self estimatedRemainingForRun:run now:now];
			[runs addObject:entry];
		}
		load[@"runs"] = runs;
		return load;
	}
}

- (NSDictionary<NSString *, NSString *> *)loadHeaders
{
	@synchronized (self) {
		NSMutableDictionary<NSString *, NSString *> *headers = [NSMutableDictionary dictionary];
		headers[@"X-InferKit-Limit"] = [NSString stringWithFormat:@"%lu", (unsigned long)self.limit];
		headers[@"X-InferKit-Running"] = [NSString stringWithFormat:@"%lu", (unsigned long)self.running.count];
		headers[@"X-InferKit-Queued"] = [NSString stringWithFormat:@"%lu", (unsigned long)[self queuedCount]];
		NSNumber *wait = [self estimatedWaitAt:NFKServedNow()];
		if (wait != nil) {
			headers[@"X-InferKit-Estimated-Wait"] = [NSString stringWithFormat:@"%.3f", wait.doubleValue];
		}
		return headers;
	}
}

/*! The refusal for a run that finds the queue full, carrying when a slot is expected to be free. */
- (NSError *)busyError
{
	NSNumber *wait = nil;
	NSUInteger queued = 0;
	@synchronized (self) {
		wait = [self estimatedWaitAt:NFKServedNow()];
		queued = [self queuedCount];
	}
	NSTimeInterval retry = MAX(1.0, ceil(wait.doubleValue));
	NSString *reason = [NSString stringWithFormat:@"the model \"%@\" has %lu runs waiting, the most it queues; retry in about %.0f seconds",
						self.name, (unsigned long)queued, retry];
	return [NSError errorWithDomain:NFKInferenceServerErrorDomain code:NFKInferenceServerErrorBusy
						   userInfo:@{ NSLocalizedDescriptionKey: reason,
									   NFKRemoteErrorRetryAfterKey: [NSDate dateWithTimeIntervalSinceNow:retry] }];
}

- (void)retire
{
	NSArray<NFKServedRun *> *waiting = nil;
	@synchronized (self) {
		self.removed = YES;
		waiting = [self.pending copy];
		[self.pending removeAllObjects];
	}
	NSError *error = NFKServedError(NFKInferenceServerErrorDomain, NFKInferenceServerErrorModelNotFound,
									[NSString stringWithFormat:@"the model \"%@\" is no longer served", self.name]);
	for (NFKServedRun *run in waiting) {
		[run finishWithResult:nil error:error];
	}
}

@end

#pragma mark - Server

@interface NFKInferenceServer ()
@property (nonatomic, strong) NSMutableDictionary<NSString *, NFKServedModel *> *models;
@property (nonatomic, strong, nullable) NFKHTTPListener *listener;
@property (nonatomic, strong, nullable) id identity;
@property (nonatomic, assign, readwrite, getter=isRunning) BOOL running;
@property (nonatomic, assign, readwrite) uint16_t listeningPort;
@property (nonatomic, assign) NSInteger startTime;
@property (nonatomic, strong) NFKServedHostStatus *hostStatus;
@end

@implementation NFKInferenceServer

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_models = [NSMutableDictionary dictionary];
		_port = NFKInferenceServerDefaultPort;
		_requiresAPIKey = YES;
		_advertisesService = YES;
		_maximumRequestBodyBytes = 256 * 1024 * 1024;
		_maximumConcurrentRunsPerModel = 1;
		_reportsHostDetails = YES;
		_hostStatus = [[NFKServedHostStatus alloc] init];
	}
	return self;
}

- (void)dealloc
{
	[_listener stop];
}

- (nullable SecIdentityRef)TLSIdentity
{
	return (__bridge SecIdentityRef)self.identity;
}

- (void)setTLSIdentity:(nullable SecIdentityRef)TLSIdentity
{
	self.identity = (__bridge id)TLSIdentity;
}

- (nullable NSURL *)localBaseURL
{
	if (!self.isRunning) {
		return nil;
	}
	NSString *scheme = self.identity != nil ? @"https" : @"http";
	return [NSURL URLWithString:[NSString stringWithFormat:@"%@://localhost:%u%@", scheme, (unsigned)self.listeningPort, NFKServedBasePath]];
}

#pragma mark Models

- (void)addBackend:(id<NFKInferenceBackend>)backend forModelName:(NSString *)modelName
{
	NFKServedModel *model = [[NFKServedModel alloc] init];
	model.name = modelName;
	model.backend = backend;
	model.limit = MAX(self.maximumConcurrentRunsPerModel, 1);
	model.queueLimit = self.maximumQueuedRunsPerModel;
	model.running = [NSMutableArray array];
	model.pending = [NSMutableArray array];
	NFKServedModel *replaced = nil;
	@synchronized (self.models) {
		replaced = self.models[modelName];
		self.models[modelName] = model;
	}
	[replaced retire];
}

- (void)removeBackendForModelName:(NSString *)modelName
{
	NFKServedModel *removed = nil;
	@synchronized (self.models) {
		removed = self.models[modelName];
		[self.models removeObjectForKey:modelName];
	}
	[removed retire];
}

- (nullable id<NFKInferenceBackend>)backendForModelName:(NSString *)modelName
{
	@synchronized (self.models) {
		return self.models[modelName].backend;
	}
}

- (NSArray<NSString *> *)modelNames
{
	@synchronized (self.models) {
		return [self.models.allKeys sortedArrayUsingSelector:@selector(compare:)];
	}
}

/*! The model a request names, or the only one hosted when it names none. */
- (nullable NFKServedModel *)modelNamed:(nullable id)name error:(NSError * _Nullable *)outError
{
	@synchronized (self.models) {
		if ([name isKindOfClass:NSString.class] && [name length] > 0) {
			NFKServedModel *model = self.models[name];
			if (model == nil && outError != NULL) {
				*outError = NFKServedError(NFKInferenceServerErrorDomain, NFKInferenceServerErrorModelNotFound,
										   [NSString stringWithFormat:@"the model \"%@\" is not served here; GET /v1/models lists what is", name]);
			}
			return model;
		}
		if (self.models.count == 1) {
			return self.models.allValues.firstObject;
		}
	}
	if (outError != NULL) {
		*outError = NFKServedError(NFKInferenceServerErrorDomain, NFKInferenceServerErrorModelNotFound,
								   @"the request names no model and this server hosts more than one; GET /v1/models lists them");
	}
	return nil;
}

#pragma mark Lifecycle

- (BOOL)startWithError:(NSError * _Nullable *)outError
{
	if (self.isRunning) {
		return YES;
	}
	BOOL keyNeeded = (self.requiresAPIKey && !self.loopbackOnly) || self.requiresAPIKeyOnLoopback;
	if (keyNeeded && self.apiKey.length == 0) {
		if (outError != NULL) {
			*outError = NFKServedError(NFKInferenceServerErrorDomain, NFKInferenceServerErrorAPIKeyRequired,
									   @"clients on other machines must present a key and none is set: set apiKey, set loopbackOnly, or set requiresAPIKey to NO to serve without one");
		}
		return NO;
	}
	if (self.maximumConcurrentRunsPerModel == 0 || self.maximumRequestBodyBytes == 0) {
		if (outError != NULL) {
			*outError = NFKServedError(NFKInferenceServerErrorDomain, NFKInferenceServerErrorInvalidConfiguration,
									   @"maximumConcurrentRunsPerModel and maximumRequestBodyBytes must be positive");
		}
		return NO;
	}
	__weak NFKInferenceServer *weakSelf = self;
	NFKHTTPListener *listener = [[NFKHTTPListener alloc] initWithHandler:^(NFKHTTPRequest *request, NFKHTTPResponse *response) {
		NFKInferenceServer *server = weakSelf;
		if (server == nil) {
			[response sendStatus:503 headers:nil body:nil];
			return;
		}
		dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
			[server routeRequest:request response:response];
		});
	}];
	listener.port = self.port;
	listener.loopbackOnly = self.loopbackOnly;
	listener.TLSIdentity = self.identity;
	listener.maximumBodyBytes = self.maximumRequestBodyBytes;
	if (self.advertisesService && !self.loopbackOnly) {
		listener.serviceType = NFKInferenceServerServiceType;
		listener.serviceName = self.serviceName;
		NSMutableDictionary<NSString *, NSString *> *record = [@{ @"path": NFKServedBasePath,
																  @"tls": self.identity != nil ? @"1" : @"0",
																  @"auth": self.requiresAPIKey ? @"1" : @"0",
																  @"version": NFKInferKit.version } mutableCopy];
		if (self.reportsHostDetails) {
			[record addEntriesFromDictionary:[NFKServedHostStatus TXTRecordEntries]];
		}
		listener.TXTRecord = record;
	}
	NSError *listenError = nil;
	if (![listener startWithError:&listenError]) {
		if (outError != NULL) {
			NSMutableDictionary *userInfo = [listenError.userInfo mutableCopy] ?: [NSMutableDictionary dictionary];
			*outError = [NSError errorWithDomain:NFKInferenceServerErrorDomain code:NFKInferenceServerErrorListenFailed userInfo:userInfo];
		}
		return NO;
	}
	self.listener = listener;
	self.listeningPort = listener.boundPort;
	self.startTime = (NSInteger)NSDate.date.timeIntervalSince1970;
	self.running = YES;
	return YES;
}

- (void)stop
{
	[self.listener stop];
	self.listener = nil;
	self.running = NO;
	self.listeningPort = 0;
}

#pragma mark Routing

- (BOOL)authorizes:(NFKHTTPRequest *)request
{
	BOOL keyRequired = request.fromLoopback ? self.requiresAPIKeyOnLoopback : self.requiresAPIKey;
	if (!keyRequired) {
		return YES;
	}
	NSString *expected = self.apiKey;
	NSString *authorization = [request valueForHeader:@"authorization"] ?: @"";
	if (expected.length == 0 || ![authorization.lowercaseString hasPrefix:@"bearer "]) {
		return NO;
	}
	NSData *presented = [[[authorization substringFromIndex:7] stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceCharacterSet]
						 dataUsingEncoding:NSUTF8StringEncoding];
	NSData *secret = [expected dataUsingEncoding:NSUTF8StringEncoding];
	if (presented.length != secret.length) {
		return NO;
	}
	// Compared in constant time, so the reply's timing says nothing about how much of a guess matched.
	const uint8_t *left = presented.bytes;
	const uint8_t *right = secret.bytes;
	uint8_t difference = 0;
	for (NSUInteger index = 0; index < secret.length; index++) {
		difference |= left[index] ^ right[index];
	}
	return difference == 0;
}

- (void)routeRequest:(NFKHTTPRequest *)request response:(NFKHTTPResponse *)response
{
	if (![self authorizes:request]) {
		[self sendError:NFKServedError(NFKInferenceServerErrorDomain, NFKInferenceServerErrorUnauthorized,
									   @"a valid key is required; send it as Authorization: Bearer <key>")
			   response:response];
		return;
	}
	NSString *path = request.path;
	while (path.length > 1 && [path hasSuffix:@"/"]) {
		path = [path substringToIndex:path.length - 1];
	}
	NSString *modelsPrefix = [NFKServedBasePath stringByAppendingString:@"/models/"];
	if ([path hasPrefix:modelsPrefix]) {
		if ([self requireMethod:@"GET" of:request response:response]) {
			[self serveModelNamed:[path substringFromIndex:modelsPrefix.length] response:response];
		}
		return;
	}
	NSDictionary<NSString *, NSString *> *routes = @{ @"/models": @"GET",
													  @"/chat/completions": @"POST",
													  @"/embeddings": @"POST",
													  @"/audio/transcriptions": @"POST",
													  @"/audio/translations": @"POST",
													  @"/audio/speech": @"POST",
													  @"/images/generations": @"POST",
													  @"/images/edits": @"POST",
													  @"/inferkit/run": @"POST",
													  @"/inferkit/status": @"GET" };
	NSString *route = [path hasPrefix:NFKServedBasePath] ? [path substringFromIndex:NFKServedBasePath.length] : nil;
	if (route == nil || routes[route] == nil) {
		[response sendStatus:404 JSONObject:@{ @"error": @{ @"message": [NSString stringWithFormat:@"no route %@ %@", request.method, request.path],
															 @"type": @"invalid_request_error" } } headers:nil];
		return;
	}
	if (![self requireMethod:routes[route] of:request response:response]) {
		return;
	}
	if ([route isEqualToString:@"/models"]) {
		[self serveModelListWithResponse:response];
	} else if ([route isEqualToString:@"/inferkit/status"]) {
		[self serveStatusWithResponse:response];
	} else if ([route isEqualToString:@"/chat/completions"]) {
		[self serveChat:request response:response];
	} else if ([route isEqualToString:@"/embeddings"]) {
		[self serveEmbeddings:request response:response];
	} else if ([route hasPrefix:@"/audio/trans"]) {
		[self serveTranscription:request translating:[route hasSuffix:@"translations"] response:response];
	} else if ([route isEqualToString:@"/audio/speech"]) {
		[self serveSpeech:request response:response];
	} else if ([route hasPrefix:@"/images/"]) {
		[self serveImages:request editing:[route hasSuffix:@"edits"] response:response];
	} else {
		[self serveNativeRun:request response:response];
	}
}

- (BOOL)requireMethod:(NSString *)method of:(NFKHTTPRequest *)request response:(NFKHTTPResponse *)response
{
	if ([request.method isEqualToString:method]) {
		return YES;
	}
	[response sendStatus:405 JSONObject:@{ @"error": @{ @"message": [NSString stringWithFormat:@"%@ takes %@", request.path, method],
														 @"type": @"invalid_request_error" } }
				 headers:@{ @"Allow": method }];
	return NO;
}

#pragma mark Errors

- (NSInteger)statusForError:(NSError *)error
{
	if ([error.domain isEqualToString:NFKInferenceServerErrorDomain]) {
		NSDictionary<NSNumber *, NSNumber *> *statuses = @{ @(NFKInferenceServerErrorModelNotFound): @404,
															@(NFKInferenceServerErrorUnauthorized): @401,
															@(NFKInferenceServerErrorBusy): @503 };
		return statuses[@(error.code)].integerValue ?: 400;
	}
	if (![error.domain isEqualToString:NFKInferenceErrorDomain]) {
		return 500;
	}
	NSDictionary<NSNumber *, NSNumber *> *statuses = @{ @(kNFKError_InferenceNotReady): @503,
														@(kNFKError_InferenceMissingInput): @400,
														@(kNFKError_InferenceUnsupported): @400,
														@(kNFKError_InferenceRefused): @400,
														@(kNFKError_InferenceRateLimited): @429,
														@(kNFKError_RemoteUnreachable): @502 };
	return statuses[@(error.code)].integerValue ?: 500;
}

- (void)sendError:(nullable NSError *)failure response:(NFKHTTPResponse *)response
{
	NSError *error = failure ?: NFKServedError(NFKInferenceErrorDomain, kNFKError_InferenceBackendFailure, @"the request failed");
	NSMutableDictionary<NSString *, NSString *> *headers = [NSMutableDictionary dictionary];
	NSDate *retryAfter = error.userInfo[NFKRemoteErrorRetryAfterKey];
	if ([retryAfter isKindOfClass:NSDate.class]) {
		headers[@"Retry-After"] = [NSString stringWithFormat:@"%ld", (long)MAX(0.0, ceil(retryAfter.timeIntervalSinceNow))];
	}
	if (error.code == NFKInferenceServerErrorUnauthorized && [error.domain isEqualToString:NFKInferenceServerErrorDomain]) {
		headers[@"WWW-Authenticate"] = @"Bearer";
	}
	[response sendStatus:[self statusForError:error] JSONObject:[NFKInferenceWireCoding JSONObjectForError:error] headers:headers];
}

- (nullable NSDictionary *)JSONBodyOf:(NFKHTTPRequest *)request response:(NFKHTTPResponse *)response
{
	id body = request.body.length > 0 ? [NSJSONSerialization JSONObjectWithData:request.body options:0 error:NULL] : nil;
	if (![body isKindOfClass:NSDictionary.class]) {
		[self sendError:NFKServedError(NFKInferenceErrorDomain, kNFKError_InferenceMissingInput, @"the body is not a JSON object") response:response];
		return nil;
	}
	return body;
}

- (void)removeFiles:(NSArray<NSURL *> *)files
{
	for (NSURL *file in files) {
		[NSFileManager.defaultManager removeItemAtURL:file error:NULL];
	}
}

/*! Queues a run for a model, cancelling it when the client goes away. The reply's head carries the
	model's load as it stands when the head is written. */
- (void)enqueue:(NFKServedRun *)run on:(NFKServedModel *)model response:(NFKHTTPResponse *)response
{
	__weak NFKServedRun *weakRun = run;
	response.disconnectHandler = ^{
		[weakRun cancel];
	};
	__weak NFKInferenceServer *weakSelf = self;
	__weak NFKServedModel *weakModel = model;
	response.headerProvider = ^NSDictionary<NSString *, NSString *> *{
		NSMutableDictionary<NSString *, NSString *> *headers = [[weakModel loadHeaders] mutableCopy] ?: [NSMutableDictionary dictionary];
		if (weakSelf.reportsHostDetails) {
			headers[@"X-InferKit-Thermal-State"] = [NFKServedHostStatus thermalStateName];
		}
		return headers;
	};
	[model enqueue:run];
}

#pragma mark Models

- (NSDictionary<NSString *, id> *)entryForModel:(NFKServedModel *)model
{
	id<NFKInferenceBackend> backend = model.backend;
	NSMutableDictionary *description = [NSMutableDictionary dictionary];
	description[@"backend"] = backend.backendIdentifier ?: @"";
	description[@"ready"] = @(backend.isReady);
	if ([backend respondsToSelector:@selector(supportedInputKeys)]) {
		description[@"inputs"] = [backend.supportedInputKeys.allObjects sortedArrayUsingSelector:@selector(compare:)];
	}
	if ([backend respondsToSelector:@selector(supportedParameterKeys)]) {
		description[@"parameters"] = [backend.supportedParameterKeys.allObjects sortedArrayUsingSelector:@selector(compare:)];
	}
	description[@"load"] = [model loadJSONObject];
	return @{ @"id": model.name, @"object": @"model", @"created": @(self.startTime), @"owned_by": @"inferkit",
			  @"inferkit": description };
}

- (void)serveModelListWithResponse:(NFKHTTPResponse *)response
{
	NSMutableArray *entries = [NSMutableArray array];
	for (NSString *name in self.modelNames) {
		NFKServedModel *model = nil;
		@synchronized (self.models) {
			model = self.models[name];
		}
		if (model != nil) {
			[entries addObject:[self entryForModel:model]];
		}
	}
	[response sendStatus:200 JSONObject:@{ @"object": @"list", @"data": entries } headers:nil];
}

- (void)serveModelNamed:(NSString *)name response:(NFKHTTPResponse *)response
{
	NSError *error = nil;
	NFKServedModel *model = [self modelNamed:name error:&error];
	if (model == nil) {
		[self sendError:error response:response];
		return;
	}
	[response sendStatus:200 JSONObject:[self entryForModel:model] headers:nil];
}

- (void)serveStatusWithResponse:(NFKHTTPResponse *)response
{
	NSMutableArray *models = [NSMutableArray array];
	for (NSString *name in self.modelNames) {
		NFKServedModel *model = nil;
		@synchronized (self.models) {
			model = self.models[name];
		}
		if (model != nil) {
			[models addObject:@{ @"id": model.name, @"backend": model.backend.backendIdentifier ?: @"", @"ready": @(model.backend.isReady),
								 @"load": [model loadJSONObject] }];
		}
	}
	NSMutableDictionary<NSString *, id> *status = [NSMutableDictionary dictionary];
	status[@"object"] = @"inferkit.status";
	status[@"server"] = @{ @"version": NFKInferKit.version, @"started": @(self.startTime),
						   @"uptime_seconds": @(MAX(NSDate.date.timeIntervalSince1970 - (NSTimeInterval)self.startTime, 0.0)) };
	status[@"models"] = models;
	if (self.reportsHostDetails) {
		status[@"host"] = [self.hostStatus JSONObjectForStorageURL:self.storageDirectoryURL];
	}
	[response sendStatus:200 JSONObject:status headers:nil];
}

#pragma mark Chat

- (void)serveChat:(NFKHTTPRequest *)request response:(NFKHTTPResponse *)response
{
	NSDictionary *body = [self JSONBodyOf:request response:response];
	if (body == nil) {
		return;
	}
	NSError *error = nil;
	NFKServedModel *model = [self modelNamed:body[@"model"] error:&error];
	NSMutableArray<NSURL *> *files = [NSMutableArray array];
	NFKInferenceRequest *inference = model != nil ? [NFKServedOpenAI chatRequestFromBody:body temporaryFiles:files error:&error] : nil;
	if (inference == nil) {
		[self removeFiles:files];
		[self sendError:error response:response];
		return;
	}
	NSString *identifier = [@"chatcmpl-" stringByAppendingString:NSUUID.UUID.UUIDString];
	NSInteger created = (NSInteger)NSDate.date.timeIntervalSince1970;
	NSString *modelName = model.name;
	NFKServedRun *run = [[NFKServedRun alloc] init];
	run.request = inference;
	run.temporaryFiles = files;

	if (![body[@"stream"] boolValue]) {
		run.completionHandler = ^(NFKInferenceResult *result, NSError *runError) {
			NSString *finishReason = nil;
			NSError *replyError = runError;
			NSDictionary *message = result != nil ? [NFKServedOpenAI chatMessageForResult:result finishReason:&finishReason error:&replyError] : nil;
			if (message == nil) {
				[self sendError:replyError response:response];
				return;
			}
			NSMutableDictionary *reply = [@{ @"id": identifier, @"object": @"chat.completion", @"created": @(created), @"model": modelName,
											 @"choices": @[ @{ @"index": @0, @"message": message, @"finish_reason": finishReason,
															   @"logprobs": NSNull.null } ] } mutableCopy];
			reply[@"usage"] = [NFKServedOpenAI usageForResult:result];
			[response sendStatus:200 JSONObject:reply headers:nil];
		};
		[self enqueue:run on:model response:response];
		return;
	}

	NSMutableString *sentText = [NSMutableString string];
	NSMutableString *sentReasoning = [NSMutableString string];
	void (^emit)(NSDictionary *, id, id) = ^(NSDictionary *delta, id finishReason, id usage) {
		NSMutableDictionary *chunk = [@{ @"id": identifier, @"object": @"chat.completion.chunk", @"created": @(created), @"model": modelName,
										 @"choices": @[ @{ @"index": @0, @"delta": delta, @"finish_reason": finishReason ?: NSNull.null } ] } mutableCopy];
		chunk[@"usage"] = usage;
		[response sendEventData:NFKServedJSONString(chunk)];
	};
	void (^open)(void) = ^{
		if (!response.headersSent) {
			[response beginStreamWithStatus:200 headers:@{ @"Content-Type": @"text/event-stream" }];
			emit(@{ @"role": @"assistant", @"content": @"" }, nil, nil);
		}
	};
	// The deltas past what was sent; a partial that does not extend the text so far is skipped.
	NSDictionary *(^advance)(NFKInferenceResult *) = ^NSDictionary *(NFKInferenceResult *partial) {
		NSMutableDictionary *delta = [NSMutableDictionary dictionary];
		NSString *text = NFKServedSuffix(partial.text, sentText);
		NSString *reasoning = NFKServedSuffix([partial outputForKey:NFKOutputReasoning], sentReasoning);
		if ([reasoning isKindOfClass:NSString.class]) {
			[sentReasoning appendString:reasoning];
			delta[@"reasoning_content"] = reasoning;
		}
		if (text != nil) {
			[sentText appendString:text];
			delta[@"content"] = text;
		}
		return delta;
	};
	NSObject *lock = [[NSObject alloc] init];
	run.partialHandler = ^(NFKInferenceResult *partial, double progress) {
		@synchronized (lock) {
			NSDictionary *delta = partial != nil ? advance(partial) : @{};
			if (delta.count > 0) {
				open();
				emit(delta, nil, nil);
			}
		}
	};
	BOOL wantsUsage = [body[@"stream_options"] isKindOfClass:NSDictionary.class] && [body[@"stream_options"][@"include_usage"] boolValue];
	run.completionHandler = ^(NFKInferenceResult *result, NSError *runError) {
		@synchronized (lock) {
			NSString *finishReason = nil;
			NSError *replyError = runError;
			NSDictionary *message = result != nil ? [NFKServedOpenAI chatMessageForResult:result finishReason:&finishReason error:&replyError] : nil;
			if (message == nil) {
				[self failStream:response error:replyError];
				return;
			}
			open();
			NSMutableDictionary *delta = [advance(result) mutableCopy];
			if (sentText.length == 0 && [message[@"content"] isKindOfClass:NSString.class] && [message[@"content"] length] > 0) {
				delta[@"content"] = message[@"content"];
			}
			NSMutableArray *calls = [NSMutableArray array];
			[message[@"tool_calls"] enumerateObjectsUsingBlock:^(NSDictionary *call, NSUInteger index, BOOL *stop) {
				NSMutableDictionary *indexed = [call mutableCopy];
				indexed[@"index"] = @(index);
				[calls addObject:indexed];
			}];
			if (calls.count > 0) {
				delta[@"tool_calls"] = calls;
			}
			if (message[@"audio"] != nil) {
				delta[@"audio"] = @{ @"data": message[@"audio"][@"data"], @"transcript": message[@"audio"][@"transcript"] };
			}
			if (delta.count > 0) {
				emit(delta, nil, nil);
			}
			NSDictionary *usage = [NFKServedOpenAI usageForResult:result];
			emit(@{}, finishReason, usage);
			if (wantsUsage && usage != nil) {
				NSDictionary *usageChunk = @{ @"id": identifier, @"object": @"chat.completion.chunk", @"created": @(created),
											  @"model": modelName, @"choices": @[], @"usage": usage };
				[response sendEventData:NFKServedJSONString(usageChunk)];
			}
			[response sendEventData:@"[DONE]"];
			[response endStream];
		}
	};
	[self enqueue:run on:model response:response];
}

/*! A failure before the first event is an ordinary error reply; after it, an error event ends the
	stream without its [DONE]. */
- (void)failStream:(NFKHTTPResponse *)response error:(nullable NSError *)failure
{
	if (!response.headersSent) {
		[self sendError:failure response:response];
		return;
	}
	NSError *error = failure ?: NFKServedError(NFKInferenceErrorDomain, kNFKError_InferenceBackendFailure, @"the run failed");
	[response sendEventData:NFKServedJSONString([NFKInferenceWireCoding JSONObjectForError:error])];
	[response endStream];
}

#pragma mark Embeddings

- (void)serveEmbeddings:(NFKHTTPRequest *)request response:(NFKHTTPResponse *)response
{
	NSDictionary *body = [self JSONBodyOf:request response:response];
	if (body == nil) {
		return;
	}
	NSError *error = nil;
	NFKServedModel *model = [self modelNamed:body[@"model"] error:&error];
	NSArray<NSString *> *texts = model != nil ? [NFKServedOpenAI embeddingInputsFromBody:body error:&error] : nil;
	if (texts == nil) {
		[self sendError:error response:response];
		return;
	}
	NSDictionary *parameters = [NFKServedOpenAI embeddingParametersFromBody:body];
	NSMutableArray<NSArray<NSNumber *> *> *vectors = [NSMutableArray arrayWithCapacity:texts.count];
	NFKServedRun *run = [[NFKServedRun alloc] init];
	run.work = ^NFKInferenceResult *(id<NFKInferenceBackend> backend, NSError **workError) {
		for (NSString *text in texts) {
			NFKInferenceRequest *each = [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: text }
																	parameters:parameters outputModality:NFKModalityText];
			NSArray<NSNumber *> *vector = [backend runInferenceForRequest:each error:workError].embedding;
			if (vector == nil) {
				if (workError != NULL && *workError == nil) {
					*workError = NFKServedError(NFKInferenceErrorDomain, kNFKError_InferenceBackendFailure,
												@"the model returned no embedding under NFKOutputEmbedding");
				}
				return nil;
			}
			[vectors addObject:vector];
		}
		return [NFKInferenceResult resultWithOutputs:@{}];
	};
	NSString *modelName = model.name;
	BOOL base64 = [body[@"encoding_format"] isEqual:@"base64"];
	run.completionHandler = ^(NFKInferenceResult *result, NSError *runError) {
		if (result == nil) {
			[self sendError:runError response:response];
			return;
		}
		[response sendStatus:200 JSONObject:[NFKServedOpenAI embeddingReplyForVectors:vectors model:modelName base64:base64] headers:nil];
	};
	[self enqueue:run on:model response:response];
}

#pragma mark Audio

- (void)serveTranscription:(NFKHTTPRequest *)request translating:(BOOL)translating response:(NFKHTTPResponse *)response
{
	NSArray<NFKHTTPFormPart *> *parts = NFKHTTPFormParts(request);
	if (parts == nil) {
		[self sendError:NFKServedError(NFKInferenceErrorDomain, kNFKError_InferenceMissingInput, @"the body must be multipart/form-data")
			   response:response];
		return;
	}
	NSError *error = nil;
	NFKServedModel *model = [self modelNamed:[NFKServedOpenAI partNamed:@"model" in:parts].stringValue error:&error];
	NSMutableArray<NSURL *> *files = [NSMutableArray array];
	NFKInferenceRequest *inference = model != nil
		? [NFKServedOpenAI transcriptionRequestFromParts:parts translating:translating temporaryFiles:files error:&error] : nil;
	if (inference == nil) {
		[self removeFiles:files];
		[self sendError:error response:response];
		return;
	}
	NSString *format = [NFKServedOpenAI partNamed:@"response_format" in:parts].stringValue;
	format = format.length > 0 ? format : @"json";
	BOOL streams = [[NFKServedOpenAI partNamed:@"stream" in:parts].stringValue isEqualToString:@"true"];
	NFKServedRun *run = [[NFKServedRun alloc] init];
	run.request = inference;
	run.temporaryFiles = files;
	NSMutableString *sent = [NSMutableString string];
	NSObject *lock = [[NSObject alloc] init];
	void (^sendDelta)(NSString *) = ^(NSString *text) {
		NSString *delta = NFKServedSuffix(text, sent);
		if (delta == nil) {
			return;
		}
		[sent appendString:delta];
		if (!response.headersSent) {
			[response beginStreamWithStatus:200 headers:@{ @"Content-Type": @"text/event-stream" }];
		}
		[response sendEventData:NFKServedJSONString(@{ @"type": @"transcript.text.delta", @"delta": delta })];
	};
	if (streams) {
		run.partialHandler = ^(NFKInferenceResult *partial, double progress) {
			@synchronized (lock) {
				sendDelta(partial.text);
			}
		};
	}
	run.completionHandler = ^(NFKInferenceResult *result, NSError *runError) {
		if (result == nil) {
			[self failStream:response error:runError];
			return;
		}
		NSString *contentType = nil;
		NSData *reply = [NFKServedOpenAI transcriptionBodyForResult:result format:format translating:translating contentType:&contentType];
		if (!streams) {
			[response sendStatus:200 headers:@{ @"Content-Type": contentType } body:reply];
			return;
		}
		@synchronized (lock) {
			NSString *text = [NFKServedOpenAI textForResult:result] ?: @"";
			sendDelta(text);
			if (!response.headersSent) {
				[response beginStreamWithStatus:200 headers:@{ @"Content-Type": @"text/event-stream" }];
			}
			[response sendEventData:NFKServedJSONString(@{ @"type": @"transcript.text.done", @"text": text })];
			[response endStream];
		}
	};
	[self enqueue:run on:model response:response];
}

- (void)serveSpeech:(NFKHTTPRequest *)request response:(NFKHTTPResponse *)response
{
	NSDictionary *body = [self JSONBodyOf:request response:response];
	if (body == nil) {
		return;
	}
	NSError *error = nil;
	NFKServedModel *model = [self modelNamed:body[@"model"] error:&error];
	NFKInferenceRequest *inference = model != nil ? [NFKServedOpenAI speechRequestFromBody:body error:&error] : nil;
	if (inference == nil) {
		[self sendError:error response:response];
		return;
	}
	NSString *format = [body[@"response_format"] isKindOfClass:NSString.class] ? [body[@"response_format"] lowercaseString] : @"wav";
	BOOL events = [body[@"stream_format"] isEqual:@"sse"];
	NFKServedRun *run = [[NFKServedRun alloc] init];
	run.request = inference;
	run.completionHandler = ^(NFKInferenceResult *result, NSError *runError) {
		NSString *contentType = nil;
		NSError *replyError = runError;
		NSData *audio = result != nil ? [NFKServedOpenAI audioDataForResult:result format:format contentType:&contentType error:&replyError] : nil;
		if (audio == nil) {
			[self sendError:replyError response:response];
			return;
		}
		if (!events) {
			[response sendStatus:200 headers:@{ @"Content-Type": contentType } body:audio];
			return;
		}
		[response beginStreamWithStatus:200 headers:@{ @"Content-Type": @"text/event-stream" }];
		NSUInteger piece = 48 * 1024;
		for (NSUInteger offset = 0; offset < audio.length; offset += piece) {
			NSData *slice = [audio subdataWithRange:NSMakeRange(offset, MIN(piece, audio.length - offset))];
			[response sendEventData:NFKServedJSONString(@{ @"type": @"speech.audio.delta", @"audio": [slice base64EncodedStringWithOptions:0] })];
		}
		[response sendEventData:NFKServedJSONString(@{ @"type": @"speech.audio.done" })];
		[response endStream];
	};
	[self enqueue:run on:model response:response];
}

#pragma mark Images

- (void)serveImages:(NFKHTTPRequest *)request editing:(BOOL)editing response:(NFKHTTPResponse *)response
{
	NSArray<NFKHTTPFormPart *> *parts = editing ? NFKHTTPFormParts(request) : nil;
	NSDictionary *body = nil;
	if (parts == nil) {
		body = [self JSONBodyOf:request response:response];
		if (body == nil) {
			return;
		}
	}
	id modelName = parts != nil ? [NFKServedOpenAI partNamed:@"model" in:parts].stringValue : body[@"model"];
	NSError *error = nil;
	NFKServedModel *model = [self modelNamed:modelName error:&error];
	NFKInferenceRequest *inference = nil;
	if (model != nil && parts != nil) {
		inference = [NFKServedOpenAI imageEditRequestFromParts:parts error:&error];
	} else if (model != nil && editing) {
		[self sendError:NFKServedError(NFKInferenceErrorDomain, kNFKError_InferenceUnsupported,
									   @"an edit takes its images as multipart/form-data; this server fetches no image URLs") response:response];
		return;
	} else if (model != nil) {
		inference = [NFKServedOpenAI imageGenerationRequestFromBody:body error:&error];
	}
	if (inference == nil) {
		[self sendError:error response:response];
		return;
	}
	NSString *(^field)(NSString *) = ^NSString *(NSString *name) {
		id value = parts != nil ? [NFKServedOpenAI partNamed:name in:parts].stringValue : body[name];
		return [value isKindOfClass:NSString.class] ? value : nil;
	};
	NSString *responseFormat = field(@"response_format");
	NSString *outputFormat = field(@"output_format");
	BOOL streams = parts != nil ? [field(@"stream") isEqualToString:@"true"] : [body[@"stream"] boolValue];
	NFKServedRun *run = [[NFKServedRun alloc] init];
	run.request = inference;
	if (streams) {
		run.partialHandler = ^(NFKInferenceResult *partial, double progress) {
			NSData *png = [partial outputForKey:NFKOutputImage] != nil ? [NFKImageCoding PNGDataForImage:[partial outputForKey:NFKOutputImage]] : nil;
			if (png == nil) {
				return;
			}
			if (!response.headersSent) {
				[response beginStreamWithStatus:200 headers:@{ @"Content-Type": @"text/event-stream" }];
			}
			[response sendEventData:NFKServedJSONString(@{ @"type": editing ? @"image_edit.partial_image" : @"image_generation.partial_image",
														   @"b64_json": [png base64EncodedStringWithOptions:0] })];
		};
	}
	run.completionHandler = ^(NFKInferenceResult *result, NSError *runError) {
		NSError *replyError = runError;
		NSDictionary *reply = result != nil
			? [NFKServedOpenAI imageReplyForResult:result responseFormat:streams ? nil : responseFormat outputFormat:outputFormat error:&replyError] : nil;
		if (reply == nil) {
			[self failStream:response error:replyError];
			return;
		}
		if (!streams) {
			[response sendStatus:200 JSONObject:reply headers:nil];
			return;
		}
		if (!response.headersSent) {
			[response beginStreamWithStatus:200 headers:@{ @"Content-Type": @"text/event-stream" }];
		}
		for (NSDictionary *entry in reply[@"data"]) {
			[response sendEventData:NFKServedJSONString(@{ @"type": editing ? @"image_edit.completed" : @"image_generation.completed",
														   @"b64_json": entry[@"b64_json"] ?: @"" })];
		}
		[response endStream];
	};
	[self enqueue:run on:model response:response];
}

#pragma mark Native

- (void)serveNativeRun:(NFKHTTPRequest *)request response:(NFKHTTPResponse *)response
{
	NSDictionary *body = [self JSONBodyOf:request response:response];
	if (body == nil) {
		return;
	}
	NSError *error = nil;
	NFKServedModel *model = [self modelNamed:body[@"model"] error:&error];
	NFKInferenceWireCoding *decoder = [[NFKInferenceWireCoding alloc] initWithTemporaryDirectory:nil];
	NFKInferenceRequest *inference = nil;
	if (model != nil && body[@"request"] == nil) {
		error = NFKServedError(NFKInferenceErrorDomain, kNFKError_InferenceMissingInput, @"the body carries no request");
	} else if (model != nil) {
		inference = [decoder requestFromJSONObject:body[@"request"] error:&error];
	}
	if (inference == nil) {
		[self removeFiles:decoder.temporaryFileURLs];
		[self sendError:error response:response];
		return;
	}
	BOOL streams = [body[@"stream"] boolValue];
	NFKServedRun *run = [[NFKServedRun alloc] init];
	run.request = inference;
	run.temporaryFiles = decoder.temporaryFileURLs;
	NFKInferenceWireCoding *encoder = [[NFKInferenceWireCoding alloc] initWithTemporaryDirectory:nil];
	NSObject *lock = [[NSObject alloc] init];
	__block NFKInferenceResult *lastPartial = nil;
	if (streams) {
		run.partialHandler = ^(NFKInferenceResult *partial, double progress) {
			@synchronized (lock) {
				NSMutableDictionary *event = [@{ @"type": @"progress", @"progress": @(isfinite(progress) ? progress : -1.0) } mutableCopy];
				if (partial != nil && partial != lastPartial) {
					lastPartial = partial;
					event[@"partial"] = [encoder JSONObjectForResult:partial error:NULL];
				}
				if (!response.headersSent) {
					[response beginStreamWithStatus:200 headers:@{ @"Content-Type": @"text/event-stream" }];
				}
				[response sendEventData:NFKServedJSONString(event)];
			}
		};
	}
	run.completionHandler = ^(NFKInferenceResult *result, NSError *runError) {
		NSError *replyError = runError;
		id encoded = result != nil ? [encoder JSONObjectForResult:result error:&replyError] : nil;
		if (!streams) {
			if (encoded == nil) {
				[self sendError:replyError response:response];
			} else {
				[response sendStatus:200 JSONObject:@{ @"result": encoded } headers:nil];
			}
			return;
		}
		@synchronized (lock) {
			if (!response.headersSent) {
				[response beginStreamWithStatus:200 headers:@{ @"Content-Type": @"text/event-stream" }];
			}
			NSDictionary *event = encoded != nil
				? @{ @"type": @"result", @"result": encoded }
				: @{ @"type": @"error", @"error": [NFKInferenceWireCoding JSONObjectForError:replyError][@"error"] };
			[response sendEventData:NFKServedJSONString(event)];
			[response sendEventData:@"[DONE]"];
			[response endStream];
		}
	};
	[self enqueue:run on:model response:response];
}

@end
