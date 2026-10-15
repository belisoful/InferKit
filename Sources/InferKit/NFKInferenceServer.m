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
#import <InferKit/NFKServerIdentity.h>
#import "NFKHTTPServer.h"
#import "NFKServedOpenAI.h"
#import "NFKServedMCP.h"
#import "NFKInferenceWireCoding.h"
#import "NFKServedHostStatus.h"
#import "NFKServerRoutePrivate.h"
#import "NFKServerObservationPrivate.h"
#import <CommonCrypto/CommonDigest.h>
#import <unistd.h>
#import <arpa/inet.h>

const uint16_t NFKInferenceServerDefaultPort = 11480;
NSString * const NFKInferenceServerServiceType = @"_inferkit._tcp";
NSString * const NFKInferenceServerErrorDomain = @"NFKInferenceServerErrorDomain";
NSString * const NFKInferenceServerDefaultAPIKeyName = @"default";

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

/*! value reduced to what a client can read as JSON: strings, finite numbers, and booleans, and arrays
	and string-keyed dictionaries of them, nested at most depth levels. Nil when nothing is left. */
static id _Nullable NFKServedJSONObject(id _Nullable value, NSUInteger depth)
{
	if ([value isKindOfClass:NSString.class]) {
		return value;
	}
	if ([value isKindOfClass:NSNumber.class]) {
		return isfinite([value doubleValue]) ? value : nil;
	}
	if (depth == 0) {
		return nil;
	}
	if ([value isKindOfClass:NSArray.class]) {
		NSMutableArray *elements = [NSMutableArray array];
		for (id element in (NSArray *)value) {
			id kept = NFKServedJSONObject(element, depth - 1);
			if (kept != nil) {
				[elements addObject:kept];
			}
		}
		return elements;
	}
	if ([value isKindOfClass:NSDictionary.class]) {
		NSMutableDictionary *entries = [NSMutableDictionary dictionary];
		[(NSDictionary *)value enumerateKeysAndObjectsUsingBlock:^(id key, id entry, BOOL *stop) {
			id kept = [key isKindOfClass:NSString.class] ? NFKServedJSONObject(entry, depth - 1) : nil;
			if (kept != nil) {
				entries[key] = kept;
			}
		}];
		return entries;
	}
	return nil;
}

/*! A token count a result's usage reports under key, or nil when it reports none. */
static NSNumber * _Nullable NFKServedTokenCount(id usage, NSString *key)
{
	id count = [usage isKindOfClass:NSDictionary.class] ? usage[key] : nil;
	return [count isKindOfClass:NSNumber.class] && [count longLongValue] >= 0 ? count : nil;
}

/*! The entries of a dictionary a client can read as JSON: string keys with string or finite number
	values. */
static NSDictionary<NSString *, id> *NFKServedJSONValues(id _Nullable dictionary)
{
	NSMutableDictionary<NSString *, id> *values = [NSMutableDictionary dictionary];
	if (![dictionary isKindOfClass:NSDictionary.class]) {
		return values;
	}
	[(NSDictionary *)dictionary enumerateKeysAndObjectsUsingBlock:^(id key, id value, BOOL *stop) {
		BOOL finiteNumber = [value isKindOfClass:NSNumber.class] && isfinite([value doubleValue]);
		if ([key isKindOfClass:NSString.class] && ([value isKindOfClass:NSString.class] || finiteNumber)) {
			values[key] = value;
		}
	}];
	return values;
}

static NSMutableArray<NSString *> *NFKServedRuntimeStatusClassNames(void)
{
	static NSMutableArray<NSString *> *names = nil;
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		names = [NSMutableArray arrayWithObject:@"NFKMLXGPU"];
	});
	return names;
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

/*! The SHA-256 digest of text's UTF-8 bytes. */
static NSData *NFKServedKeyDigest(NSString *text)
{
	NSData *bytes = [text dataUsingEncoding:NSUTF8StringEncoding] ?: [NSData data];
	unsigned char digest[CC_SHA256_DIGEST_LENGTH];
	CC_SHA256(bytes.bytes, (CC_LONG)bytes.length, digest);
	return [NSData dataWithBytes:digest length:sizeof(digest)];
}

/*! Whether two digests of equal length are equal, compared in constant time so the reply's timing says
	nothing about how much of a guess matched. */
static BOOL NFKServedDigestsEqual(NSData *left, NSData *right)
{
	if (left.length != right.length) {
		return NO;
	}
	const uint8_t *leftBytes = left.bytes;
	const uint8_t *rightBytes = right.bytes;
	uint8_t difference = 0;
	for (NSUInteger index = 0; index < left.length; index++) {
		difference |= leftBytes[index] ^ rightBytes[index];
	}
	return difference == 0;
}

#pragma mark - Keys

/*! One named key: its scopes and the digest a presented key is compared with. */
@interface NFKServedAPIKey : NSObject
@property (nonatomic, copy) NSString *name;
/*! Kept for apiKey, which reads the default key back. */
@property (nonatomic, copy) NSString *secret;
@property (nonatomic, copy) NSData *digest;
@property (nonatomic, assign) NFKServerKeyScope scopes;
@end

@implementation NFKServedAPIKey
@end

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
/*! What the run's events report: an identifier shared by its start and end, and the request's route,
	client, and key. */
@property (nonatomic, copy) NSString *identifier;
@property (nonatomic, copy) NSString *route;
@property (nonatomic, copy) NSString *clientAddress;
@property (nonatomic, copy, nullable) NSString *keyName;
/*! Whether the run's work began, which its start event reported. */
@property (atomic, assign) BOOL began;
@end

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

/*! What a request does on arrival. */
typedef NS_ENUM(NSInteger, NFKServedAdmission) {
	NFKServedAdmissionStart,
	NFKServedAdmissionQueue,
	NFKServedAdmissionQueueAndBuild,
	NFKServedAdmissionFull,
	NFKServedAdmissionRemoved,
};

static NSString *NFKServedResidencyName(NFKServerModelResidency residency)
{
	switch (residency) {
		case NFKServerModelResidencyLoading:
			return @"loading";
		case NFKServerModelResidencyResident:
			return @"resident";
		default:
			return @"not_loaded";
	}
}

/*! A hosted model: its backend or the provider that builds it, the queue of runs waiting for it, and
	what its runs have measured. Every property past uptime is read and written on the model's lock. */
@interface NFKServedModel : NSObject
@property (nonatomic, copy) NSString *name;
/*! The server's monotonic clock. */
@property (nonatomic, copy) NSTimeInterval (^uptime)(void);
/*! Called after the backend is released by an unload, off the model's lock. */
@property (nonatomic, copy, nullable) void (^unloadHandler)(void);
/*! Reports a run's start or end, off the model's lock. Set before the model is hosted. */
@property (nonatomic, copy, nullable) void (^runObserver)(NFKServedRun *run, NFKServerRunState state, NSError * _Nullable error);
/*! Reports a load, a failed load, or an unload, off the model's lock. Set before the model is hosted. */
@property (nonatomic, copy, nullable) void (^modelObserver)(NFKServerModelEventKind kind, NFKServerModelEventReason reason,
															 NSTimeInterval loadSeconds, NSError * _Nullable error);
/*! Why the build in progress started. */
@property (nonatomic, assign) NFKServerModelEventReason buildReason;
@property (nonatomic, strong, nullable) id<NFKInferenceBackend> backend;
/*! Builds the backend; nil for a model hosted with a ready backend, which is never unloaded. */
@property (nonatomic, copy, nullable) NFKServedBackendProvider provider;
@property (nonatomic, assign) NFKServerModelResidency residency;
/*! An unload waiting for the model's runs and build to end. */
@property (nonatomic, assign) BOOL unloadRequested;
@property (nonatomic, strong) NSMutableArray<void (^)(NSError * _Nullable)> *loadHandlers;
/*! The backend, inputs, parameters, outputs, and model entries of the backend the model last held. */
@property (nonatomic, copy) NSDictionary<NSString *, id> *lastDescription;
@property (nonatomic, assign) NSTimeInterval loadStartedAt;
@property (nonatomic, assign) NSTimeInterval loadedAt;
@property (nonatomic, assign) NSTimeInterval lastUsedAt;
@property (nonatomic, assign) NSTimeInterval lastLoadSeconds;
@property (nonatomic, copy, nullable) NSString *loadError;
@property (nonatomic, assign) NSUInteger loads;
@property (nonatomic, assign) NSUInteger unloads;
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
@property (nonatomic, assign) unsigned long long inputTokens;
@property (nonatomic, assign) unsigned long long cachedInputTokens;
/*! The input tokens of the runs that reported a cached count, which the cached share divides by. */
@property (nonatomic, assign) unsigned long long cachedRunInputTokens;
@end

@implementation NFKServedModel

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_running = [NSMutableArray array];
		_pending = [NSMutableArray array];
		_loadHandlers = [NSMutableArray array];
		_lastDescription = @{ @"backend": @"", @"model": @{} };
		_uptime = ^NSTimeInterval {
			return NFKServedNow();
		};
	}
	return self;
}

- (NSTimeInterval)now
{
	return self.uptime();
}

/*! What a backend says about itself in a model entry: its identifier, inputs, parameters, outputs, and
	model. */
+ (NSDictionary<NSString *, id> *)descriptionOfBackend:(id<NFKInferenceBackend>)backend
{
	NSMutableDictionary<NSString *, id> *description = [NSMutableDictionary dictionary];
	description[@"backend"] = backend.backendIdentifier ?: @"";
	if ([backend respondsToSelector:@selector(supportedInputKeys)]) {
		description[@"inputs"] = [backend.supportedInputKeys.allObjects sortedArrayUsingSelector:@selector(compare:)];
	}
	if ([backend respondsToSelector:@selector(supportedParameterKeys)]) {
		description[@"parameters"] = [backend.supportedParameterKeys.allObjects sortedArrayUsingSelector:@selector(compare:)];
	}
	if ([backend respondsToSelector:@selector(supportedOutputKeys)]) {
		description[@"outputs"] = [backend.supportedOutputKeys.allObjects sortedArrayUsingSelector:@selector(compare:)];
	}
	description[@"model"] = [backend respondsToSelector:@selector(modelInfo)] ? NFKServedJSONValues(backend.modelInfo) : @{};
	return description;
}

- (void)reportRun:(NFKServedRun *)run state:(NFKServerRunState)state error:(nullable NSError *)error
{
	if (self.runObserver != nil) {
		self.runObserver(run, state, error);
	}
}

- (void)reportModelEvent:(NFKServerModelEventKind)kind reason:(NFKServerModelEventReason)reason
			 loadSeconds:(NSTimeInterval)loadSeconds error:(nullable NSError *)error
{
	if (self.modelObserver != nil) {
		self.modelObserver(kind, reason, loadSeconds, error);
	}
}

- (NSError *)notServedError
{
	return NFKServedError(NFKInferenceServerErrorDomain, NFKInferenceServerErrorModelNotFound,
						  [NSString stringWithFormat:@"the model \"%@\" is no longer served", self.name]);
}

#pragma mark Admission

- (void)enqueue:(NFKServedRun *)run
{
	run.enqueuedAt = [self now];
	NFKServedAdmission admission = NFKServedAdmissionRemoved;
	@synchronized (self) {
		admission = [self admit:run];
	}
	switch (admission) {
		case NFKServedAdmissionRemoved: {
			NSError *error = [self notServedError];
			[self reportRun:run state:NFKServerRunStateFailed error:error];
			[run finishWithResult:nil error:error];
			break;
		}
		case NFKServedAdmissionFull: {
			NSError *error = [self busyError];
			[self reportRun:run state:NFKServerRunStateRefused error:error];
			[run finishWithResult:nil error:error];
			break;
		}
		case NFKServedAdmissionStart:
			[self start:run];
			break;
		case NFKServedAdmissionQueueAndBuild:
			[self build];
			break;
		case NFKServedAdmissionQueue:
			break;
	}
}

/*! Starts, queues, or refuses an arriving run. Runs on the lock. */
- (NFKServedAdmission)admit:(NFKServedRun *)run
{
	if (self.removed) {
		return NFKServedAdmissionRemoved;
	}
	BOOL resident = self.residency == NFKServerModelResidencyResident;
	if (resident && self.running.count < self.limit) {
		[self beginRun:run];
		return NFKServedAdmissionStart;
	}
	// Runs that take a slot as soon as the build ends do not count against the queue limit.
	NSUInteger room = self.queueLimit + (resident ? 0 : self.limit);
	if (self.queueLimit > 0 && [self queuedCount] >= room) {
		self.refused += 1;
		return NFKServedAdmissionFull;
	}
	[self.pending addObject:run];
	return [self beginBuildIfNeededFor:NFKServerModelEventReasonRequest] ? NFKServedAdmissionQueueAndBuild : NFKServedAdmissionQueue;
}

/*! Takes a slot for a run. Runs on the lock. */
- (void)beginRun:(NFKServedRun *)run
{
	run.startedAt = [self now];
	self.lastUsedAt = run.startedAt;
	self.averageWaitSeconds = NFKServedAveraged(self.averageWaitSeconds, run.startedAt - run.enqueuedAt, self.started);
	self.started += 1;
	[self.running addObject:run];
}

/*! Gives each free slot to the next waiting run while the backend is resident, and returns the runs to
	start. Runs on the lock. */
- (NSArray<NFKServedRun *> *)takeStartableRuns
{
	NSMutableArray<NFKServedRun *> *starting = [NSMutableArray array];
	while (self.residency == NFKServerModelResidencyResident && self.pending.count > 0 && self.running.count < self.limit) {
		NFKServedRun *next = self.pending.firstObject;
		[self.pending removeObjectAtIndex:0];
		[self beginRun:next];
		[starting addObject:next];
	}
	return starting;
}

/*! Changes the model's concurrency and queue limits. A raised concurrency limit starts waiting runs at
	once; a lowered one takes effect as running runs end, and a lowered queue limit refuses only later
	arrivals. */
- (void)applyLimit:(NSUInteger)limit queueLimit:(NSUInteger)queueLimit
{
	NSArray<NFKServedRun *> *starting = nil;
	@synchronized (self) {
		self.limit = MAX(limit, 1);
		self.queueLimit = queueLimit;
		starting = [self takeStartableRuns];
	}
	for (NFKServedRun *run in starting) {
		[self start:run];
	}
}

#pragma mark Building

/*! Marks a build as started for reason when the model is not loaded and can be built. Runs on the lock. */
- (BOOL)beginBuildIfNeededFor:(NFKServerModelEventReason)reason
{
	if (self.residency != NFKServerModelResidencyNotLoaded || self.provider == nil) {
		return NO;
	}
	self.residency = NFKServerModelResidencyLoading;
	self.loadStartedAt = [self now];
	self.buildReason = reason;
	return YES;
}

- (void)build
{
	NFKServedBackendProvider provider = nil;
	@synchronized (self) {
		provider = self.provider;
	}
	dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
		NSError *error = nil;
		id<NFKInferenceBackend> backend = provider(&error);
		[self buildEndedWithBackend:backend error:error];
	});
}

- (void)buildEndedWithBackend:(nullable id<NFKInferenceBackend>)backend error:(nullable NSError *)error
{
	NSDictionary<NSString *, id> *description = backend != nil ? [NFKServedModel descriptionOfBackend:backend] : nil;
	NSError *failure = nil;
	if (backend == nil) {
		failure = error ?: NFKServedError(NFKInferenceErrorDomain, kNFKError_InferenceBackendFailure,
										  [NSString stringWithFormat:@"the provider for the model \"%@\" returned no backend and no error", self.name]);
	}
	NSArray<NFKServedRun *> *refused = @[];
	NSMutableArray<NFKServedRun *> *starting = [NSMutableArray array];
	NSArray<void (^)(NSError * _Nullable)> *handlers = nil;
	BOOL reportsBuild = NO;
	NFKServerModelEventReason reason = NFKServerModelEventReasonRequest;
	NSTimeInterval loadSeconds = 0;
	@synchronized (self) {
		handlers = [self.loadHandlers copy];
		[self.loadHandlers removeAllObjects];
		reportsBuild = !self.removed;
		reason = self.buildReason;
		loadSeconds = [self now] - self.loadStartedAt;
		if (self.removed) {
			failure = failure ?: [self notServedError];
		} else if (failure != nil) {
			self.residency = NFKServerModelResidencyNotLoaded;
			self.loadError = failure.localizedDescription;
			self.unloadRequested = NO;
			refused = [self.pending copy];
			[self.pending removeAllObjects];
			self.failed += refused.count;
		} else {
			[self becomeResidentWithBackend:backend description:description];
			[starting addObjectsFromArray:[self takeStartableRuns]];
		}
		if (reportsBuild) {
			[self reportModelEvent:failure != nil ? NFKServerModelEventKindLoadFailed : NFKServerModelEventKindLoaded
							reason:reason loadSeconds:loadSeconds error:failure];
		}
	}
	for (NFKServedRun *run in refused) {
		[self reportRun:run state:NFKServerRunStateFailed error:failure];
		[run finishWithResult:nil error:failure];
	}
	for (NFKServedRun *run in starting) {
		[self start:run];
	}
	for (void (^handler)(NSError * _Nullable) in handlers) {
		handler(failure);
	}
	[self unloadIfRequested];
}

/*! Runs on the lock. */
- (void)becomeResidentWithBackend:(id<NFKInferenceBackend>)backend description:(NSDictionary<NSString *, id> *)description
{
	NSTimeInterval now = [self now];
	self.backend = backend;
	self.lastDescription = description;
	self.residency = NFKServerModelResidencyResident;
	self.loadedAt = now;
	self.lastUsedAt = now;
	self.lastLoadSeconds = now - self.loadStartedAt;
	self.loads += 1;
	self.loadError = nil;
}

- (void)loadWithCompletionHandler:(nullable void (^)(NSError * _Nullable error))completionHandler
{
	BOOL settled = NO;
	BOOL build = NO;
	NSError *error = nil;
	@synchronized (self) {
		self.unloadRequested = NO;
		if (self.removed) {
			error = [self notServedError];
			settled = YES;
		} else if (self.residency == NFKServerModelResidencyResident) {
			settled = YES;
		} else {
			if (completionHandler != nil) {
				[self.loadHandlers addObject:completionHandler];
			}
			build = [self beginBuildIfNeededFor:NFKServerModelEventReasonExplicit];
		}
	}
	if (settled && completionHandler != nil) {
		dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
			completionHandler(error);
		});
	}
	if (build) {
		[self build];
	}
}

#pragma mark Unloading

/*! Whether the model could release its backend now. Runs on the lock. */
- (BOOL)isUnloadableNow
{
	return self.provider != nil && !self.removed && self.residency == NFKServerModelResidencyResident
		&& self.running.count == 0 && self.pending.count == 0;
}

/*! Whether a run holds a slot or a backend is being built, which a runtime's cache release must not
	overlap. */
- (BOOL)isBusy
{
	@synchronized (self) {
		return self.running.count > 0 || self.residency == NFKServerModelResidencyLoading;
	}
}

- (NSTimeInterval)lastUse
{
	@synchronized (self) {
		return self.lastUsedAt;
	}
}

/*! Takes the backend out of the model when the model could release it now and test, run on the lock,
	agrees, reports the unload with reason, and returns the backend; nil when it did not. The event is
	reported on the lock, as a finished build's is, so the delegate receives a model's events in the
	order its residency changed. The caller releases the backend, then calls finishUnload. */
- (nullable id<NFKInferenceBackend>)takeBackendPassingTest:(BOOL (^)(NFKServedModel *model))test
											  reason:(NFKServerModelEventReason)reason
{
	@synchronized (self) {
		if (![self isUnloadableNow] || !test(self)) {
			return nil;
		}
		id<NFKInferenceBackend> taken = self.backend;
		self.backend = nil;
		self.residency = NFKServerModelResidencyNotLoaded;
		self.unloads += 1;
		self.unloadRequested = NO;
		if (taken != nil) {
			[self reportModelEvent:NFKServerModelEventKindUnloaded reason:reason loadSeconds:0 error:nil];
		}
		return taken;
	}
}

/*! Runs the unload handler; called once the taken backend is released. */
- (void)finishUnload
{
	if (self.unloadHandler != nil) {
		self.unloadHandler();
	}
}

/*! Releases the backend when takeBackendPassingTest:reason: takes it; YES when it did. The backend is
	released off the lock. */
- (BOOL)unloadPassingTest:(BOOL (^)(NFKServedModel *model))test reason:(NFKServerModelEventReason)reason
{
	BOOL unloaded = NO;
	@autoreleasepool {
		unloaded = [self takeBackendPassingTest:test reason:reason] != nil;
	}
	if (unloaded) {
		[self finishUnload];
	}
	return unloaded;
}

- (void)unloadIfRequested
{
	[self unloadPassingTest:^BOOL(NFKServedModel *model) {
		return model.unloadRequested;
	} reason:NFKServerModelEventReasonExplicit];
}

/*! Unloads now, or once the model's runs and build end. NO when the model cannot be built again. */
- (BOOL)requestUnload
{
	@synchronized (self) {
		if (self.provider == nil || self.removed) {
			return NO;
		}
		self.unloadRequested = self.residency != NFKServerModelResidencyNotLoaded;
	}
	[self unloadIfRequested];
	return YES;
}

/*! Takes the backend when the model has gone unused for at least idle seconds and was built at least
	grace seconds ago, as takeBackendPassingTest:reason: does. */
- (nullable id<NFKInferenceBackend>)takeBackendIfUnusedFor:(NSTimeInterval)idle grace:(NSTimeInterval)grace at:(NSTimeInterval)now
													reason:(NFKServerModelEventReason)reason
{
	return [self takeBackendPassingTest:^BOOL(NFKServedModel *model) {
		return now - model.loadedAt >= grace && now - model.lastUsedAt >= idle;
	} reason:reason];
}

/*! Whether takeBackendIfUnusedFor:grace:at:reason: would take the backend. */
- (BOOL)isUnusedFor:(NSTimeInterval)idle grace:(NSTimeInterval)grace at:(NSTimeInterval)now
{
	@synchronized (self) {
		return [self isUnloadableNow] && now - self.loadedAt >= grace && now - self.lastUsedAt >= idle;
	}
}

- (BOOL)isUnloadable
{
	@synchronized (self) {
		return self.provider != nil;
	}
}

- (NFKServerModelResidency)currentResidency
{
	@synchronized (self) {
		return self.residency;
	}
}

- (nullable id<NFKInferenceBackend>)residentBackend
{
	@synchronized (self) {
		return self.backend;
	}
}

- (NSDictionary<NSString *, id> *)lastBackendDescription
{
	@synchronized (self) {
		return self.lastDescription;
	}
}

#pragma mark Runs

- (void)start:(NFKServedRun *)run
{
	id<NFKInferenceBackend> backend = [self residentBackend];
	if (run.cancelled || backend == nil) {
		NSError *error = run.cancelled ? NFKServedCancellation()
			: NFKServedError(NFKInferenceErrorDomain, kNFKError_InferenceNotReady,
							 [NSString stringWithFormat:@"the model \"%@\" is not loaded", self.name]);
		[self run:run endedWithResult:nil error:error];
		[run finishWithResult:nil error:error];
		return;
	}
	run.began = YES;
	[self reportRun:run state:NFKServerRunStateStarted error:nil];
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

/*! Releases a run's slot, records how it ended, and starts the next waiting run, or performs a
	requested unload when none waits. */
- (void)run:(NFKServedRun *)run endedWithResult:(nullable NFKInferenceResult *)result error:(nullable NSError *)error
{
	NSArray<NFKServedRun *> *starting = nil;
	@synchronized (self) {
		[self.running removeObjectIdenticalTo:run];
		[self recordRun:run result:result error:error];
		self.lastUsedAt = [self now];
		starting = [self takeStartableRuns];
	}
	NFKServerRunState state = NFKServerRunStateCompleted;
	if (run.cancelled || NFKServedIsCancellation(error)) {
		state = NFKServerRunStateCancelled;
	} else if (result == nil) {
		state = NFKServerRunStateFailed;
	}
	[self reportRun:run state:state error:state == NFKServerRunStateCompleted ? nil : error];
	for (NFKServedRun *next in starting) {
		[self start:next];
	}
	if (starting.count == 0) {
		[self unloadIfRequested];
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
	double seconds = [self now] - run.startedAt;
	self.averageRunSeconds = NFKServedAveraged(self.averageRunSeconds, seconds, self.completed);
	self.completed += 1;
	NSDictionary *usage = [result outputForKey:NFKOutputUsage];
	NSNumber *tokens = [usage isKindOfClass:NSDictionary.class] ? usage[NFKUsageOutputTokens] : nil;
	if ([tokens isKindOfClass:NSNumber.class] && tokens.doubleValue > 0 && seconds > 0) {
		self.outputTokensPerSecond = NFKServedAveraged(self.outputTokensPerSecond, tokens.doubleValue / seconds, self.tokenSamples);
		self.tokenSamples += 1;
	}
	NSNumber *input = NFKServedTokenCount(usage, NFKUsageInputTokens);
	if (input == nil) {
		return;
	}
	self.inputTokens += input.unsignedLongLongValue;
	NSNumber *cached = NFKServedTokenCount(usage, NFKUsageCachedTokens);
	if (cached != nil) {
		// A cached count above the input count is clamped, so the share stays within 0...1.
		self.cachedInputTokens += MIN(cached.unsignedLongLongValue, input.unsignedLongLongValue);
		self.cachedRunInputTokens += input.unsignedLongLongValue;
	}
}

#pragma mark Load

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

/*! Seconds until the backend is resident: 0 when it is, else the last build's duration less the time
	this build has taken, or nil before any build has finished. Runs on the lock. */
- (nullable NSNumber *)estimatedBuildRemainingAt:(NSTimeInterval)now
{
	if (self.residency == NFKServerModelResidencyResident) {
		return @0;
	}
	if (self.loads == 0) {
		return nil;
	}
	NSTimeInterval elapsed = self.residency == NFKServerModelResidencyLoading ? now - self.loadStartedAt : 0;
	return @(MAX(self.lastLoadSeconds - elapsed, 0.0));
}

/*! Seconds until a run that arrives now would start, or nil when no estimate exists yet. The backend is
	resident first; then each slot frees when its run ends, and each waiting run holds the soonest
	free slot for an average run. Runs on the lock. */
- (nullable NSNumber *)estimatedWaitAt:(NSTimeInterval)now
{
	NSNumber *building = [self estimatedBuildRemainingAt:now];
	if (building == nil) {
		return nil;
	}
	NSUInteger queued = [self queuedCount];
	if (self.running.count < self.limit && queued == 0) {
		return building;
	}
	NSMutableArray<NSNumber *> *slots = [NSMutableArray arrayWithCapacity:self.running.count];
	for (NFKServedRun *run in self.running) {
		NSNumber *remaining = [self estimatedRemainingForRun:run now:now];
		if (remaining == nil) {
			return nil;
		}
		[slots addObject:remaining];
	}
	// After a lowered limit more runs hold slots than the limit allows: the first slot opens only when
	// the surplus runs and one more have ended, so the soonest-ending surplus is dropped.
	if (slots.count > self.limit) {
		[slots sortUsingSelector:@selector(compare:)];
		[slots removeObjectsInRange:NSMakeRange(0, slots.count - self.limit)];
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
	return @([[slots valueForKeyPath:@"@min.self"] doubleValue] + building.doubleValue);
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
		if (self.inputTokens > 0) {
			load[@"input_tokens"] = @(self.inputTokens);
		}
		if (self.cachedRunInputTokens > 0) {
			load[@"cached_input_tokens"] = @(self.cachedInputTokens);
			load[@"cached_input_share"] = @((double)self.cachedInputTokens / (double)self.cachedRunInputTokens);
		}
		NSTimeInterval now = [self now];
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

/*! The model's residency for a status reading. timeout is its idle timeout now, INFINITY for none. */
- (NSDictionary<NSString *, id> *)residencyJSONObjectWithTimeout:(NSTimeInterval)timeout pinned:(BOOL)pinned
{
	@synchronized (self) {
		NSMutableDictionary<NSString *, id> *residency = [NSMutableDictionary dictionary];
		residency[@"state"] = NFKServedResidencyName(self.residency);
		residency[@"unloadable"] = @(self.provider != nil);
		residency[@"pinned"] = @(pinned);
		if (self.loads > 0 || self.provider == nil) {
			NSTimeInterval idle = MAX([self now] - self.lastUsedAt, 0.0);
			residency[@"last_used"] = @(NSDate.date.timeIntervalSince1970 - idle);
		}
		if (self.provider != nil && isfinite(timeout)) {
			residency[@"idle_timeout_seconds"] = @(timeout);
		}
		if (self.loads > 0) {
			residency[@"load_seconds"] = @(self.lastLoadSeconds);
		}
		residency[@"load_error"] = self.loadError;
		residency[@"loads"] = @(self.loads);
		residency[@"unloads"] = @(self.unloads);
		return residency;
	}
}

- (NSDictionary<NSString *, NSString *> *)loadHeaders
{
	@synchronized (self) {
		NSMutableDictionary<NSString *, NSString *> *headers = [NSMutableDictionary dictionary];
		headers[@"X-InferKit-Limit"] = [NSString stringWithFormat:@"%lu", (unsigned long)self.limit];
		headers[@"X-InferKit-Running"] = [NSString stringWithFormat:@"%lu", (unsigned long)self.running.count];
		headers[@"X-InferKit-Queued"] = [NSString stringWithFormat:@"%lu", (unsigned long)[self queuedCount]];
		NSNumber *wait = [self estimatedWaitAt:[self now]];
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
		wait = [self estimatedWaitAt:[self now]];
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
	NSArray<void (^)(NSError * _Nullable)> *handlers = nil;
	BOOL wasResident = NO;
	@synchronized (self) {
		self.removed = YES;
		wasResident = self.residency == NFKServerModelResidencyResident;
		waiting = [self.pending copy];
		[self.pending removeAllObjects];
		handlers = [self.loadHandlers copy];
		[self.loadHandlers removeAllObjects];
		if (wasResident) {
			[self reportModelEvent:NFKServerModelEventKindUnloaded reason:NFKServerModelEventReasonRemoved loadSeconds:0 error:nil];
		}
	}
	NSError *error = [self notServedError];
	for (NFKServedRun *run in waiting) {
		[self reportRun:run state:NFKServerRunStateFailed error:error];
		[run finishWithResult:nil error:error];
	}
	for (void (^handler)(NSError * _Nullable) in handlers) {
		handler(error);
	}
}

@end

#pragma mark - Server

@interface NFKInferenceServer ()
@property (nonatomic, strong) NSMutableDictionary<NSString *, NFKServedModel *> *models;
@property (nonatomic, strong) NSMutableArray<NFKInferenceTool *> *servedTools;
/*! The keys by name; guarded by itself. */
@property (nonatomic, strong) NSMutableDictionary<NSString *, NFKServedAPIKey *> *keys;
/*! The app routes by normalized prefix; guarded by itself. */
@property (nonatomic, strong) NSMutableDictionary<NSString *, NFKServerRouteHandler> *appRoutes;
@property (nonatomic, strong, nullable) NFKHTTPListener *listener;
/*! The plain-HTTP loopback listener, while servesPlainHTTPOnLoopback has it running. */
@property (nonatomic, strong, nullable) NFKHTTPListener *plainListener;
@property (nonatomic, assign, readwrite) uint16_t plainLoopbackListeningPort;
@property (nonatomic, strong, nullable) id identity;
@property (nonatomic, assign, readwrite, getter=isRunning) BOOL running;
@property (nonatomic, assign, readwrite) uint16_t listeningPort;
@property (nonatomic, assign) NSInteger startTime;
@property (nonatomic, strong) NFKServedHostStatus *hostStatus;
/*! The monotonic clock every timing reads. */
@property (nonatomic, copy) NSTimeInterval (^uptimeProvider)(void);
/*! Guards the unloading state below and the per-model idle intervals. */
@property (nonatomic, strong) NSObject *residencyLock;
@property (nonatomic, strong) NSMutableDictionary<NSString *, NSNumber *> *idleUnloadIntervals;
@property (nonatomic, strong) NSMutableSet<NSString *> *pinnedNames;
@property (nonatomic, assign) double heldPressure;
@property (nonatomic, assign) NSTimeInterval heldPressureAt;
@property (nonatomic, assign) BOOL hasHeldPressure;
@property (nonatomic, assign) NSTimeInterval lastPressureUnloadAt;
@property (nonatomic, assign) BOOL hasPressureUnloaded;
@property (nonatomic, strong) dispatch_queue_t residencyQueue;
@property (nonatomic, strong, nullable) dispatch_source_t residencyTimer;
@property (nonatomic, copy, nullable) double (^customMemoryPressureProvider)(void);
@property (nonatomic, strong) NFKServedMCPSessionStore *mcpSessions;
/*! The delegate's queue unless the host sets its own. */
@property (nonatomic, strong) dispatch_queue_t eventQueue;
@property (atomic, strong, nullable) dispatch_queue_t customDelegateQueue;
@end

@implementation NFKInferenceServer

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_models = [NSMutableDictionary dictionary];
		_servedTools = [NSMutableArray array];
		_appRoutes = [NSMutableDictionary dictionary];
		_keys = [NSMutableDictionary dictionary];
		_loopbackExemptScopes = NFKServerKeyScopeAll;
		_port = NFKInferenceServerDefaultPort;
		_requiresAPIKey = YES;
		_advertisesService = YES;
		_maximumRequestBodyBytes = 256 * 1024 * 1024;
		_maximumConcurrentRunsPerModel = 1;
		_reportsHostDetails = YES;
		_hostStatus = [[NFKServedHostStatus alloc] init];
		_uptimeProvider = ^NSTimeInterval {
			return NFKServedNow();
		};
		_residencyLock = [[NSObject alloc] init];
		_idleUnloadIntervals = [NSMutableDictionary dictionary];
		_pinnedNames = [NSMutableSet set];
		_memoryPressureOnset = 0.6;
		_shortestIdleUnloadInterval = 10;
		_pressureUnloadInterval = 600;
		_unloadGracePeriod = 30;
		_pressureUnloadHysteresis = 10;
		_residencyCheckInterval = 1;
		_residencyQueue = dispatch_queue_create("NFKInferenceServer.residency",
												dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_UTILITY, 0));
		_eventQueue = dispatch_queue_create("NFKInferenceServer.events",
											dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_UTILITY, 0));
		_mcpSessions = [[NFKServedMCPSessionStore alloc] init];
		__weak NFKInferenceServer *weakSelf = self;
		_mcpSessions.eventHandler = ^(NFKServerMCPSession *session, BOOL ended) {
			[weakSelf reportMCPSession:session ended:ended];
		};
		[self scheduleResidencyChecks];
	}
	return self;
}

- (void)dealloc
{
	[_listener stop];
	[_plainListener stop];
	if (_residencyTimer != nil) {
		dispatch_source_cancel(_residencyTimer);
	}
}

- (nullable NSString *)TLSCertificateSHA256Fingerprint
{
	if (self.identity == nil) {
		return nil;
	}
	SecCertificateRef certificate = NULL;
	if (SecIdentityCopyCertificate((__bridge SecIdentityRef)self.identity, &certificate) != errSecSuccess) {
		return nil;
	}
	NSData *der = CFBridgingRelease(SecCertificateCopyData(certificate));
	CFRelease(certificate);
	return [NFKServerIdentity SHA256FingerprintOfCertificateData:der];
}

- (nullable SecIdentityRef)TLSIdentity
{
	return (__bridge SecIdentityRef)self.identity;
}

- (void)setTLSIdentity:(nullable SecIdentityRef)TLSIdentity
{
	self.identity = (__bridge id)TLSIdentity;
}

/*! scheme://host:port followed by path. */
+ (nullable NSURL *)originWithScheme:(NSString *)scheme host:(NSString *)host port:(uint16_t)port path:(NSString *)path
{
	NSURLComponents *components = [[NSURLComponents alloc] init];
	components.scheme = scheme;
	components.host = host;
	components.port = @(port);
	components.path = path;
	return components.URL;
}

/*! The URL at path on the listener a client on this machine uses, or nil while stopped. */
- (nullable NSURL *)localURLWithPath:(NSString *)path
{
	if (!self.isRunning) {
		return nil;
	}
	if (self.plainLoopbackListeningPort != 0) {
		return [NFKInferenceServer originWithScheme:@"http" host:@"localhost" port:self.plainLoopbackListeningPort path:path];
	}
	return [NFKInferenceServer originWithScheme:self.identity != nil ? @"https" : @"http" host:@"localhost"
										   port:self.listeningPort path:path];
}

/*! The URL at path on the main listener as another machine names it, or nil while stopped or
	loopbackOnly. */
- (nullable NSURL *)networkURLWithPath:(NSString *)path
{
	if (!self.isRunning || self.loopbackOnly) {
		return nil;
	}
	return [NFKInferenceServer originWithScheme:self.identity != nil ? @"https" : @"http" host:[NFKInferenceServer networkHostName]
										   port:self.listeningPort path:path];
}

/*! This machine's host name, with the mDNS domain added to a bare name so another machine resolves it. */
+ (NSString *)networkHostName
{
	char name[256] = { 0 };
	if (gethostname(name, sizeof(name) - 1) != 0 || name[0] == '\0') {
		return @"localhost";
	}
	NSString *host = [NSString stringWithUTF8String:name] ?: @"localhost";
	return [host containsString:@"."] ? host : [host stringByAppendingString:@".local"];
}

- (nullable NSURL *)localBaseURL
{
	return [self localURLWithPath:NFKServedBasePath];
}

- (nullable NSURL *)localMCPURL
{
	return [self localURLWithPath:@"/mcp"];
}

- (nullable NSURL *)networkBaseURL
{
	return [self networkURLWithPath:NFKServedBasePath];
}

- (nullable NSURL *)networkMCPURL
{
	return [self networkURLWithPath:@"/mcp"];
}

#pragma mark Tools

- (void)addTool:(NFKInferenceTool *)tool
{
	@synchronized (self.servedTools) {
		NSUInteger existing = [self.servedTools indexOfObjectPassingTest:^BOOL(NFKInferenceTool *served, NSUInteger index, BOOL *stop) {
			return [served.name isEqualToString:tool.name];
		}];
		if (existing != NSNotFound) {
			self.servedTools[existing] = tool;
		} else {
			[self.servedTools addObject:tool];
		}
	}
}

- (void)removeToolNamed:(NSString *)name
{
	@synchronized (self.servedTools) {
		[self.servedTools filterUsingPredicate:[NSPredicate predicateWithBlock:^BOOL(NFKInferenceTool *served, NSDictionary *bindings) {
			return ![served.name isEqualToString:name];
		}]];
	}
}

- (NSArray<NFKInferenceTool *> *)tools
{
	@synchronized (self.servedTools) {
		return [self.servedTools copy];
	}
}

#pragma mark Observation

- (dispatch_queue_t)delegateQueue
{
	return self.customDelegateQueue ?: self.eventQueue;
}

- (void)setDelegateQueue:(nullable dispatch_queue_t)delegateQueue
{
	self.customDelegateQueue = delegateQueue;
}

/*! Runs call with the delegate on the delegate's queue when the delegate implements selector. */
- (void)notifyDelegateImplementing:(SEL)selector call:(void (^)(id<NFKInferenceServerDelegate> delegate))call
{
	id<NFKInferenceServerDelegate> delegate = self.delegate;
	if (![delegate respondsToSelector:selector]) {
		return;
	}
	dispatch_async(self.delegateQueue, ^{
		call(delegate);
	});
}

- (BOOL)delegateImplements:(SEL)selector
{
	return [self.delegate respondsToSelector:selector];
}

- (void)reportRun:(NFKServedRun *)run ofModelNamed:(NSString *)modelName state:(NFKServerRunState)state error:(nullable NSError *)error
{
	SEL selector = state == NFKServerRunStateStarted ? @selector(inferenceServer:didStartRun:) : @selector(inferenceServer:didFinishRun:);
	if (![self delegateImplements:selector]) {
		return;
	}
	NSTimeInterval now = [self now];
	NFKServerRunEvent *event = [[NFKServerRunEvent alloc] initPrivately];
	event.identifier = run.identifier;
	event.modelName = modelName;
	event.route = run.route;
	event.clientAddress = run.clientAddress;
	event.keyName = run.keyName;
	event.state = state;
	event.waitSeconds = MAX((run.began ? run.startedAt : now) - run.enqueuedAt, 0.0);
	event.durationSeconds = state != NFKServerRunStateStarted && run.began ? MAX(now - run.startedAt, 0.0) : 0;
	event.error = error;
	event.date = [NSDate date];
	__weak NFKInferenceServer *weakSelf = self;
	[self notifyDelegateImplementing:selector call:^(id<NFKInferenceServerDelegate> delegate) {
		NFKInferenceServer *server = weakSelf;
		if (server == nil) {
			return;
		}
		if (state == NFKServerRunStateStarted) {
			[delegate inferenceServer:server didStartRun:event];
		} else {
			[delegate inferenceServer:server didFinishRun:event];
		}
	}];
}

- (void)reportModelNamed:(NSString *)modelName kind:(NFKServerModelEventKind)kind reason:(NFKServerModelEventReason)reason
			 loadSeconds:(NSTimeInterval)loadSeconds error:(nullable NSError *)error
{
	if (![self delegateImplements:@selector(inferenceServer:didChangeModel:)]) {
		return;
	}
	NFKServerModelEvent *event = [[NFKServerModelEvent alloc] initPrivately];
	event.modelName = modelName;
	event.kind = kind;
	event.reason = reason;
	event.loadSeconds = loadSeconds;
	event.error = error;
	event.date = [NSDate date];
	__weak NFKInferenceServer *weakSelf = self;
	[self notifyDelegateImplementing:@selector(inferenceServer:didChangeModel:) call:^(id<NFKInferenceServerDelegate> delegate) {
		NFKInferenceServer *server = weakSelf;
		if (server != nil) {
			[delegate inferenceServer:server didChangeModel:event];
		}
	}];
}

- (void)reportClientOf:(NFKHTTPRequest *)request scope:(NFKServerKeyScope)scope
{
	if (![self delegateImplements:@selector(inferenceServer:didConnectClient:)]) {
		return;
	}
	NFKServerClientEvent *event = [[NFKServerClientEvent alloc] initPrivately];
	event.clientAddress = request.clientAddress ?: @"";
	event.fromLoopback = request.fromLoopback;
	event.keyName = request.keyName;
	event.scope = scope;
	event.path = request.path;
	event.date = [NSDate date];
	__weak NFKInferenceServer *weakSelf = self;
	[self notifyDelegateImplementing:@selector(inferenceServer:didConnectClient:) call:^(id<NFKInferenceServerDelegate> delegate) {
		NFKInferenceServer *server = weakSelf;
		if (server != nil) {
			[delegate inferenceServer:server didConnectClient:event];
		}
	}];
}

- (void)reportMCPSession:(NFKServerMCPSession *)session ended:(BOOL)ended
{
	SEL selector = ended ? @selector(inferenceServer:didEndMCPSession:) : @selector(inferenceServer:didStartMCPSession:);
	__weak NFKInferenceServer *weakSelf = self;
	[self notifyDelegateImplementing:selector call:^(id<NFKInferenceServerDelegate> delegate) {
		NFKInferenceServer *server = weakSelf;
		if (server == nil) {
			return;
		}
		if (ended) {
			[delegate inferenceServer:server didEndMCPSession:session];
		} else {
			[delegate inferenceServer:server didStartMCPSession:session];
		}
	}];
}

- (NFKServerStatus *)currentStatus
{
	return [NFKServerStatus statusWithJSONObject:[self statusJSONObject]];
}

#pragma mark MCP sessions

- (NSTimeInterval)MCPSessionIdleTimeout
{
	return self.mcpSessions.idleTimeout;
}

- (void)setMCPSessionIdleTimeout:(NSTimeInterval)MCPSessionIdleTimeout
{
	self.mcpSessions.idleTimeout = MAX(MCPSessionIdleTimeout, 0.0);
	[self.mcpSessions settingsChanged];
}

- (NSTimeInterval)MCPPingInterval
{
	return self.mcpSessions.pingInterval;
}

- (void)setMCPPingInterval:(NSTimeInterval)MCPPingInterval
{
	self.mcpSessions.pingInterval = MAX(MCPPingInterval, 0.0);
	[self.mcpSessions settingsChanged];
}

- (NSUInteger)MCPMissedPingLimit
{
	return self.mcpSessions.missedPingLimit;
}

- (void)setMCPMissedPingLimit:(NSUInteger)MCPMissedPingLimit
{
	self.mcpSessions.missedPingLimit = MAX(MCPMissedPingLimit, (NSUInteger)1);
}

- (NSArray<NFKServerMCPSession *> *)MCPSessions
{
	return [self.mcpSessions sessions];
}

- (BOOL)endMCPSessionWithIdentifier:(NSString *)identifier
{
	return [self.mcpSessions endSessionWithIdentifier:identifier reason:NFKServerMCPSessionEndReasonEndedByHost];
}

#pragma mark Keys

- (void)addAPIKey:(NSString *)key named:(NSString *)name scopes:(NFKServerKeyScope)scopes
{
	NFKServedAPIKey *entry = [[NFKServedAPIKey alloc] init];
	entry.name = name;
	entry.secret = key;
	entry.digest = NFKServedKeyDigest(key);
	entry.scopes = scopes;
	@synchronized (self.keys) {
		self.keys[name] = entry;
	}
}

- (void)removeAPIKeyNamed:(NSString *)name
{
	@synchronized (self.keys) {
		[self.keys removeObjectForKey:name];
	}
	[self.mcpSessions endSessionsWithKeyName:name reason:NFKServerMCPSessionEndReasonRevoked];
}

- (NSArray<NSString *> *)apiKeyNames
{
	@synchronized (self.keys) {
		return [self.keys.allKeys sortedArrayUsingSelector:@selector(compare:)];
	}
}

- (NFKServerKeyScope)scopesForAPIKeyNamed:(NSString *)name
{
	@synchronized (self.keys) {
		return self.keys[name].scopes;
	}
}

- (nullable NSString *)apiKey
{
	@synchronized (self.keys) {
		return self.keys[NFKInferenceServerDefaultAPIKeyName].secret;
	}
}

- (void)setApiKey:(nullable NSString *)apiKey
{
	if (apiKey.length > 0) {
		[self addAPIKey:apiKey named:NFKInferenceServerDefaultAPIKeyName scopes:NFKServerKeyScopeAll];
	} else {
		[self removeAPIKeyNamed:NFKInferenceServerDefaultAPIKeyName];
	}
}

- (BOOL)requiresAPIKeyOnLoopback
{
	return (self.loopbackExemptScopes & NFKServerKeyScopeAll) != NFKServerKeyScopeAll;
}

- (void)setRequiresAPIKeyOnLoopback:(BOOL)requiresAPIKeyOnLoopback
{
	self.loopbackExemptScopes = requiresAPIKeyOnLoopback ? 0 : NFKServerKeyScopeAll;
}

- (BOOL)hasAPIKeys
{
	@synchronized (self.keys) {
		return self.keys.count > 0;
	}
}

/*! The key a request presents as a Bearer token, or nil when it presents none the server holds. Every
	key is compared, so the time taken does not depend on which one matched. */
- (nullable NFKServedAPIKey *)keyPresentedBy:(NFKHTTPRequest *)request
{
	NSString *authorization = [request valueForHeader:@"authorization"] ?: @"";
	if (![authorization.lowercaseString hasPrefix:@"bearer "]) {
		return nil;
	}
	NSString *token = [[authorization substringFromIndex:7] stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceCharacterSet];
	if (token.length == 0) {
		return nil;
	}
	NSData *presented = NFKServedKeyDigest(token);
	NSArray<NFKServedAPIKey *> *keys = nil;
	@synchronized (self.keys) {
		keys = [self.keys.allValues sortedArrayUsingComparator:^NSComparisonResult(NFKServedAPIKey *left, NFKServedAPIKey *right) {
			return [left.name compare:right.name];
		}];
	}
	NFKServedAPIKey *matched = nil;
	for (NFKServedAPIKey *key in keys) {
		if (NFKServedDigestsEqual(presented, key.digest) && matched == nil) {
			matched = key;
		}
	}
	return matched;
}

#pragma mark App routes

- (BOOL)addRouteForPathPrefix:(NSString *)pathPrefix handler:(NFKServerRouteHandler)handler error:(NSError * _Nullable *)error
{
	NSString *prefix = NFKServerRouteNormalizedPrefix(pathPrefix, error);
	if (prefix == nil) {
		return NO;
	}
	@synchronized (self.appRoutes) {
		self.appRoutes[prefix] = [handler copy];
	}
	return YES;
}

- (void)removeRouteForPathPrefix:(NSString *)pathPrefix
{
	NSString *prefix = NFKServerRouteNormalizedPrefix(pathPrefix, NULL);
	if (prefix == nil) {
		return;
	}
	@synchronized (self.appRoutes) {
		[self.appRoutes removeObjectForKey:prefix];
	}
}

- (NSArray<NSString *> *)routePathPrefixes
{
	@synchronized (self.appRoutes) {
		return [self.appRoutes.allKeys sortedArrayUsingSelector:@selector(compare:)];
	}
}

/*! The longest app-route prefix path lies under, or nil. */
- (nullable NSString *)appRoutePrefixForPath:(NSString *)path
{
	NSString *matched = nil;
	@synchronized (self.appRoutes) {
		for (NSString *prefix in self.appRoutes) {
			if (NFKServerRoutePathMatchesPrefix(path, prefix) && prefix.length > matched.length) {
				matched = prefix;
			}
		}
	}
	return matched;
}

- (void)serveAppRoute:(NSString *)prefix request:(NFKHTTPRequest *)request response:(NFKHTTPResponse *)response
{
	NFKServerRouteHandler handler = nil;
	@synchronized (self.appRoutes) {
		handler = self.appRoutes[prefix];
	}
	if (handler == nil) {
		[self sendNoRouteFor:request response:response];
		return;
	}
	handler([NFKServerRouteRequest requestWithHTTPRequest:request pathPrefix:prefix],
			[NFKServerRouteResponse responseWithHTTPResponse:response]);
}

- (void)sendNoRouteFor:(NFKHTTPRequest *)request response:(NFKHTTPResponse *)response
{
	[response sendStatus:404 JSONObject:@{ @"error": @{ @"message": [NSString stringWithFormat:@"no route %@ %@", request.method, request.path],
														 @"type": @"invalid_request_error" } } headers:nil];
}

#pragma mark Models

- (NSTimeInterval)now
{
	return self.uptimeProvider();
}

/*! A model under a name with the server's limits and clock, not yet hosted. */
- (NFKServedModel *)servedModelNamed:(NSString *)modelName
{
	NFKServedModel *model = [[NFKServedModel alloc] init];
	model.name = modelName;
	model.limit = MAX(self.maximumConcurrentRunsPerModel, 1);
	model.queueLimit = self.maximumQueuedRunsPerModel;
	__weak NFKInferenceServer *weakSelf = self;
	model.uptime = ^NSTimeInterval {
		NFKInferenceServer *server = weakSelf;
		return server != nil ? [server now] : NFKServedNow();
	};
	model.unloadHandler = ^{
		[weakSelf releaseRuntimeCachesWhenIdle];
	};
	model.runObserver = ^(NFKServedRun *run, NFKServerRunState state, NSError *error) {
		[weakSelf reportRun:run ofModelNamed:modelName state:state error:error];
	};
	model.modelObserver = ^(NFKServerModelEventKind kind, NFKServerModelEventReason reason, NSTimeInterval loadSeconds, NSError *error) {
		[weakSelf reportModelNamed:modelName kind:kind reason:reason loadSeconds:loadSeconds error:error];
	};
	return model;
}

- (void)host:(NFKServedModel *)model
{
	NFKServedModel *replaced = nil;
	@synchronized (self.models) {
		replaced = self.models[model.name];
		self.models[model.name] = model;
	}
	// A limit set while the model was being made reaches it here.
	[model applyLimit:self.maximumConcurrentRunsPerModel queueLimit:self.maximumQueuedRunsPerModel];
	[replaced retire];
}

- (void)addBackend:(id<NFKInferenceBackend>)backend forModelName:(NSString *)modelName
{
	NFKServedModel *model = [self servedModelNamed:modelName];
	NSTimeInterval now = [self now];
	model.backend = backend;
	model.lastDescription = [NFKServedModel descriptionOfBackend:backend];
	model.residency = NFKServerModelResidencyResident;
	model.loadedAt = now;
	model.lastUsedAt = now;
	[self host:model];
}

- (void)addBackendProvider:(NFKServedBackendProvider)provider forModelName:(NSString *)modelName
{
	NFKServedModel *model = [self servedModelNamed:modelName];
	model.provider = provider;
	model.residency = NFKServerModelResidencyNotLoaded;
	[self host:model];
}

- (void)addBackendBuilder:(id<NFKServedBackendBuilder>)builder forModelName:(NSString *)modelName
{
	NSString *name = [modelName copy];
	[self addBackendProvider:^id<NFKInferenceBackend>(NSError **error) {
		return [builder backendForModelName:name error:error];
	} forModelName:name];
}

- (nullable NFKServedModel *)hostedModelNamed:(NSString *)modelName
{
	@synchronized (self.models) {
		return self.models[modelName];
	}
}

- (NSArray<NFKServedModel *> *)hostedModels
{
	@synchronized (self.models) {
		return [self.models.allValues copy];
	}
}

- (void)loadModelNamed:(NSString *)modelName completionHandler:(nullable void (^)(NSError * _Nullable error))completionHandler
{
	NFKServedModel *model = [self hostedModelNamed:modelName];
	if (model != nil) {
		[model loadWithCompletionHandler:completionHandler];
		return;
	}
	if (completionHandler != nil) {
		NSError *error = NFKServedError(NFKInferenceServerErrorDomain, NFKInferenceServerErrorModelNotFound,
										[NSString stringWithFormat:@"the model \"%@\" is not served here", modelName]);
		dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
			completionHandler(error);
		});
	}
}

- (BOOL)unloadModelNamed:(NSString *)modelName
{
	return [[self hostedModelNamed:modelName] requestUnload];
}

- (NFKServerModelResidency)residencyForModelName:(NSString *)modelName
{
	NFKServedModel *model = [self hostedModelNamed:modelName];
	return model != nil ? [model currentResidency] : NFKServerModelResidencyUnknown;
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

#pragma mark Unloading

- (void)setResidencyCheckInterval:(NSTimeInterval)residencyCheckInterval
{
	_residencyCheckInterval = residencyCheckInterval;
	[self scheduleResidencyChecks];
}

- (void)scheduleResidencyChecks
{
	dispatch_source_t previous = self.residencyTimer;
	if (previous != nil) {
		dispatch_source_cancel(previous);
	}
	self.residencyTimer = nil;
	NSTimeInterval interval = self.residencyCheckInterval;
	if (!(interval > 0)) {
		return;
	}
	dispatch_source_t timer = dispatch_source_create(DISPATCH_SOURCE_TYPE_TIMER, 0, 0, self.residencyQueue);
	uint64_t nanoseconds = (uint64_t)(interval * NSEC_PER_SEC);
	dispatch_source_set_timer(timer, dispatch_time(DISPATCH_TIME_NOW, (int64_t)nanoseconds), nanoseconds, nanoseconds / 10);
	__weak NFKInferenceServer *weakSelf = self;
	dispatch_source_set_event_handler(timer, ^{
		[weakSelf unloadIdleModels];
	});
	dispatch_resume(timer);
	self.residencyTimer = timer;
}

- (double (^)(void))memoryPressureProvider
{
	double (^provider)(void) = self.customMemoryPressureProvider;
	if (provider != nil) {
		return provider;
	}
	NFKServedHostStatus *host = self.hostStatus;
	return ^double {
		return [host memoryPressureFraction];
	};
}

- (void)setMemoryPressureProvider:(nullable double (^)(void))memoryPressureProvider
{
	self.customMemoryPressureProvider = memoryPressureProvider;
}

- (double)memoryPressure
{
	if (!self.unloadsUnderMemoryPressure) {
		return 0;
	}
	double reading = self.memoryPressureProvider();
	reading = isfinite(reading) ? MIN(MAX(reading, 0.0), 1.0) : 0.0;
	NSTimeInterval now = [self now];
	@synchronized (self.residencyLock) {
		BOOL holding = self.hasHeldPressure && reading < self.heldPressure
			&& now - self.heldPressureAt < self.pressureUnloadHysteresis;
		if (!holding) {
			self.heldPressure = reading;
			self.heldPressureAt = now;
			self.hasHeldPressure = YES;
		}
		return self.heldPressure;
	}
}

- (void)setMaximumConcurrentRunsPerModel:(NSUInteger)maximumConcurrentRunsPerModel
{
	_maximumConcurrentRunsPerModel = maximumConcurrentRunsPerModel;
	[self applyLimitsToHostedModels];
}

- (void)setMaximumQueuedRunsPerModel:(NSUInteger)maximumQueuedRunsPerModel
{
	_maximumQueuedRunsPerModel = maximumQueuedRunsPerModel;
	[self applyLimitsToHostedModels];
}

- (void)applyLimitsToHostedModels
{
	NSUInteger limit = self.maximumConcurrentRunsPerModel;
	NSUInteger queueLimit = self.maximumQueuedRunsPerModel;
	for (NFKServedModel *model in [self hostedModels]) {
		[model applyLimit:limit queueLimit:queueLimit];
	}
}

- (void)setPinned:(BOOL)pinned forModelName:(NSString *)modelName
{
	@synchronized (self.residencyLock) {
		if (pinned) {
			[self.pinnedNames addObject:modelName];
		} else {
			[self.pinnedNames removeObject:modelName];
		}
	}
}

- (NSArray<NSString *> *)pinnedModelNames
{
	@synchronized (self.residencyLock) {
		return [self.pinnedNames.allObjects sortedArrayUsingSelector:@selector(compare:)];
	}
}

- (BOOL)isPinnedModelNamed:(NSString *)modelName
{
	@synchronized (self.residencyLock) {
		return [self.pinnedNames containsObject:modelName];
	}
}

- (void)setIdleUnloadInterval:(nullable NSNumber *)seconds forModelName:(NSString *)modelName
{
	@synchronized (self.residencyLock) {
		self.idleUnloadIntervals[modelName] = seconds;
	}
}

/*! The model's idle interval, its own or the server's, or INFINITY when idle unloading is off for it. */
- (NSTimeInterval)idleIntervalForModelNamed:(NSString *)modelName
{
	NSNumber *own = nil;
	@synchronized (self.residencyLock) {
		own = self.idleUnloadIntervals[modelName];
	}
	NSTimeInterval interval = own != nil ? own.doubleValue : self.idleUnloadInterval;
	return interval > 0 ? interval : INFINITY;
}

/*! The idle timeout for an idle interval at a pressure: the interval, shortened once pressure reaches
	the onset, linearly to the shortest interval at full pressure. */
- (NSTimeInterval)timeoutForIdleInterval:(NSTimeInterval)idle pressure:(double)pressure
{
	double onset = self.memoryPressureOnset;
	if (!self.unloadsUnderMemoryPressure || pressure < onset) {
		return idle;
	}
	NSTimeInterval base = isfinite(idle) ? idle : MAX(self.pressureUnloadInterval, 0.0);
	NSTimeInterval shortest = MIN(MAX(self.shortestIdleUnloadInterval, 0.0), base);
	double fraction = onset < 1 ? MIN(MAX((pressure - onset) / (1 - onset), 0.0), 1.0) : 1.0;
	return MIN(idle, base - (base - shortest) * fraction);
}

- (NSTimeInterval)idleTimeoutForModel:(NFKServedModel *)model
{
	if (![model isUnloadable] || [self isPinnedModelNamed:model.name]) {
		return INFINITY;
	}
	return [self timeoutForIdleInterval:[self idleIntervalForModelNamed:model.name] pressure:self.memoryPressure];
}

- (NSTimeInterval)idleTimeoutForModelName:(NSString *)modelName
{
	NFKServedModel *model = [self hostedModelNamed:modelName];
	return model != nil ? [self idleTimeoutForModel:model] : INFINITY;
}

- (NSArray<NSString *> *)unloadIdleModels
{
	NSMutableArray<NSString *> *unloaded = [NSMutableArray array];
	NSMutableArray<id<NFKInferenceBackend>> *released = [NSMutableArray array];
	NSMutableArray<NFKServedModel *> *unloadedModels = [NSMutableArray array];
	double pressure = self.memoryPressure;
	NSTimeInterval grace = MAX(self.unloadGracePeriod, 0.0);
	@synchronized (self.residencyLock) {
		NSTimeInterval now = [self now];
		NSMutableArray<NFKServedModel *> *pressured = [NSMutableArray array];
		NSMutableDictionary<NSString *, NSNumber *> *timeouts = [NSMutableDictionary dictionary];
		for (NFKServedModel *model in [self hostedModels]) {
			if (![model isUnloadable] || [self isPinnedModelNamed:model.name]) {
				continue;
			}
			NSTimeInterval idle = [self idleIntervalForModelNamed:model.name];
			NSTimeInterval timeout = [self timeoutForIdleInterval:idle pressure:pressure];
			id<NFKInferenceBackend> taken = [model takeBackendIfUnusedFor:idle grace:grace at:now reason:NFKServerModelEventReasonIdle];
			if (taken != nil) {
				[released addObject:taken];
				[unloadedModels addObject:model];
				[unloaded addObject:model.name];
			} else if (timeout < idle && [model isUnusedFor:timeout grace:grace at:now]) {
				[pressured addObject:model];
				timeouts[model.name] = @(timeout);
			}
		}
		BOOL spaced = !self.hasPressureUnloaded || now - self.lastPressureUnloadAt >= self.pressureUnloadHysteresis;
		NSArray<NFKServedModel *> *leastRecentFirst = [pressured sortedArrayUsingComparator:^NSComparisonResult(NFKServedModel *left, NFKServedModel *right) {
			return [@([left lastUse]) compare:@([right lastUse])];
		}];
		for (NFKServedModel *model in spaced ? leastRecentFirst : @[]) {
			id<NFKInferenceBackend> taken = [model takeBackendIfUnusedFor:timeouts[model.name].doubleValue grace:grace at:now
																   reason:NFKServerModelEventReasonPressure];
			if (taken != nil) {
				[released addObject:taken];
				[unloadedModels addObject:model];
				[unloaded addObject:model.name];
				self.lastPressureUnloadAt = now;
				self.hasPressureUnloaded = YES;
				break;
			}
		}
	}
	// A backend's teardown can take seconds (an MLX model frees its weights), so it runs off the lock.
	@autoreleasepool {
		[released removeAllObjects];
	}
	for (NFKServedModel *model in unloadedModels) {
		[model finishUnload];
	}
	return unloaded;
}

/*! Asks each registered runtime to return its cache once no hosted model has a run in flight or a
	backend being built, since a runtime may free buffers in-flight work still reads. */
- (void)releaseRuntimeCachesWhenIdle
{
	__weak NFKInferenceServer *weakSelf = self;
	dispatch_async(self.residencyQueue, ^{
		NFKInferenceServer *server = weakSelf;
		if (server == nil) {
			return;
		}
		for (NFKServedModel *model in [server hostedModels]) {
			if ([model isBusy]) {
				return;
			}
		}
		for (Class<NFKServingRuntimeStatus> runtime in [NFKInferenceServer runtimeStatusClasses]) {
			if ([(Class)runtime respondsToSelector:@selector(releaseCachedMemory)]) {
				[runtime releaseCachedMemory];
			}
		}
	});
}

+ (void)registerRuntimeStatusClassName:(NSString *)className
{
	NSMutableArray<NSString *> *names = NFKServedRuntimeStatusClassNames();
	@synchronized (names) {
		if (![names containsObject:className]) {
			[names addObject:className];
		}
	}
}

/*! The registered runtime classes that are linked and adopt NFKServingRuntimeStatus. */
+ (NSArray<Class<NFKServingRuntimeStatus>> *)runtimeStatusClasses
{
	NSArray<NSString *> *names = nil;
	NSMutableArray<NSString *> *registered = NFKServedRuntimeStatusClassNames();
	@synchronized (registered) {
		names = [registered copy];
	}
	NSMutableArray<Class<NFKServingRuntimeStatus>> *classes = [NSMutableArray array];
	for (NSString *name in names) {
		Class runtime = NSClassFromString(name);
		if (runtime != Nil && [runtime conformsToProtocol:@protocol(NFKServingRuntimeStatus)]) {
			[classes addObject:runtime];
		}
	}
	return classes;
}

/*! Each linked runtime's report, keyed by its name. */
+ (NSDictionary<NSString *, id> *)runtimeStatuses
{
	NSMutableDictionary<NSString *, id> *statuses = [NSMutableDictionary dictionary];
	for (Class<NFKServingRuntimeStatus> reporter in [self runtimeStatusClasses]) {
		statuses[[reporter runtimeStatusName]] = NFKServedJSONValues([reporter runtimeStatus]);
	}
	return statuses;
}

/*! What a backend reports about its state now, reduced to JSON, or nil when it reports nothing. */
+ (nullable NSDictionary<NSString *, id> *)statusOfBackend:(id<NFKInferenceBackend>)backend
{
	if (![backend respondsToSelector:@selector(backendStatus)]) {
		return nil;
	}
	NSDictionary *status = NFKServedJSONObject(backend.backendStatus, 8);
	return status.count > 0 ? status : nil;
}

- (nullable id<NFKInferenceBackend>)backendForModelName:(NSString *)modelName
{
	return [[self hostedModelNamed:modelName] residentBackend];
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
	if (keyNeeded && ![self hasAPIKeys]) {
		if (outError != NULL) {
			*outError = NFKServedError(NFKInferenceServerErrorDomain, NFKInferenceServerErrorAPIKeyRequired,
									   @"some clients must present a key and none is set: add a key (apiKey or addAPIKey:named:scopes:), set loopbackOnly, or set requiresAPIKey to NO and exempt every scope on loopback to serve without one");
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
	NFKHTTPListener *listener = [self makeListener];
	listener.port = self.port;
	listener.loopbackOnly = self.loopbackOnly;
	listener.TLSIdentity = self.identity;
	if (self.advertisesService && !self.loopbackOnly) {
		listener.serviceType = NFKInferenceServerServiceType;
		listener.serviceName = self.serviceName;
		NSMutableDictionary<NSString *, NSString *> *record = [@{ @"path": NFKServedBasePath,
																  @"tls": self.identity != nil ? @"1" : @"0",
																  @"auth": self.requiresAPIKey ? @"1" : @"0",
																  @"mcp": @"/mcp",
																  @"version": NFKInferKit.version } mutableCopy];
		NSString *fingerprint = [self TLSCertificateSHA256Fingerprint];
		if (fingerprint != nil) {
			record[@"cert_sha256"] = fingerprint;
		}
		if (self.reportsHostDetails) {
			[record addEntriesFromDictionary:[NFKServedHostStatus TXTRecordEntries]];
		}
		listener.TXTRecord = record;
	}
	NFKHTTPListener *plainListener = nil;
	if (self.servesPlainHTTPOnLoopback) {
		plainListener = [self makeListener];
		plainListener.port = self.plainLoopbackPort;
		plainListener.loopbackOnly = YES;
	}
	NSError *listenError = nil;
	if (![listener startWithError:&listenError]) {
		return [NFKInferenceServer listenFailure:listenError error:outError];
	}
	if (plainListener != nil && ![plainListener startWithError:&listenError]) {
		[listener stop];
		return [NFKInferenceServer listenFailure:listenError error:outError];
	}
	self.listener = listener;
	self.listeningPort = listener.boundPort;
	self.plainListener = plainListener;
	self.plainLoopbackListeningPort = plainListener.boundPort;
	self.startTime = (NSInteger)NSDate.date.timeIntervalSince1970;
	self.running = YES;
	return YES;
}

/*! A listener that routes every request to this server and takes its body limit. */
- (NFKHTTPListener *)makeListener
{
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
	listener.maximumBodyBytes = self.maximumRequestBodyBytes;
	return listener;
}

/*! Reports a listener's failure to start as NFKInferenceServerErrorListenFailed; always NO. */
+ (BOOL)listenFailure:(nullable NSError *)listenError error:(NSError * _Nullable *)outError
{
	if (outError != NULL) {
		NSMutableDictionary *userInfo = [listenError.userInfo mutableCopy] ?: [NSMutableDictionary dictionary];
		*outError = [NSError errorWithDomain:NFKInferenceServerErrorDomain code:NFKInferenceServerErrorListenFailed userInfo:userInfo];
	}
	return NO;
}

- (void)stop
{
	[self.mcpSessions endAllSessionsWithReason:NFKServerMCPSessionEndReasonServerStopped];
	[self.listener stop];
	[self.plainListener stop];
	self.listener = nil;
	self.plainListener = nil;
	self.running = NO;
	self.listeningPort = 0;
	self.plainLoopbackListeningPort = 0;
}

#pragma mark Routing

/*! Whether a Host header names this machine's loopback: localhost, an address in 127.0.0.0/8, or ::1,
	with or without a port. A request without the header names no other host either. */
static BOOL NFKHostNamesLoopback(NSString * _Nullable host)
{
	if (host.length == 0) {
		return YES;
	}
	NSString *name = host.lowercaseString;
	if ([name hasPrefix:@"["]) {
		NSRange close = [name rangeOfString:@"]"];
		name = close.location != NSNotFound ? [name substringWithRange:NSMakeRange(1, close.location - 1)] : name;
	} else if ([name componentsSeparatedByString:@":"].count == 2) {
		name = [name componentsSeparatedByString:@":"].firstObject;
	}
	if ([name isEqualToString:@"localhost"] || [name isEqualToString:@"localhost."]) {
		return YES;
	}
	struct in_addr v4;
	if (inet_pton(AF_INET, name.UTF8String, &v4) == 1) {
		return (ntohl(v4.s_addr) >> 24) == 127;
	}
	struct in6_addr v6;
	return inet_pton(AF_INET6, name.UTF8String, &v6) == 1 && IN6_IS_ADDR_LOOPBACK(&v6);
}

/*! The refusal of a request for a route of scope, or nil when it may reach the route, recording the name
	of the key it presented. A client exempt from the check passes whatever it presents. A loopback
	request whose Host names another host, which is how a browser page reaches this machine by DNS
	rebinding, is served only with a key that opens the route. */
- (nullable NSError *)refusalOf:(NFKHTTPRequest *)request scope:(NFKServerKeyScope)scope
{
	NFKServedAPIKey *key = [self keyPresentedBy:request];
	request.keyName = key.name;
	if (request.fromLoopback && !NFKHostNamesLoopback([request valueForHeader:@"host"])) {
		if ((key.scopes & scope) != 0) {
			return nil;
		}
		return NFKServedError(NFKInferenceServerErrorDomain, NFKInferenceServerErrorMisdirected,
							  @"a request from this machine names another host; connect to localhost or present a key");
	}
	BOOL keyRequired = request.fromLoopback ? (self.loopbackExemptScopes & scope) == 0 : self.requiresAPIKey;
	if (!keyRequired || (key.scopes & scope) != 0) {
		return nil;
	}
	if (key == nil) {
		return NFKServedError(NFKInferenceServerErrorDomain, NFKInferenceServerErrorUnauthorized,
							  @"a valid key for this route is required; send it as Authorization: Bearer <key>");
	}
	return NFKServedError(NFKInferenceServerErrorDomain, NFKInferenceServerErrorInsufficientScope,
						  @"the key does not open this route");
}

/*! The scope a path's route belongs to, and the app route's prefix when an app route claims it. */
- (NFKServerKeyScope)scopeOfPath:(NSString *)path appRoutePrefix:(NSString * _Nullable * _Nonnull)outPrefix
{
	*outPrefix = nil;
	if ([path isEqualToString:@"/mcp"]) {
		return NFKServerKeyScopeMCP;
	}
	if ([path isEqualToString:NFKServedBasePath] || [path hasPrefix:[NFKServedBasePath stringByAppendingString:@"/"]]) {
		return [path isEqualToString:[NFKServedBasePath stringByAppendingString:@"/inferkit/status"]] ? NFKServerKeyScopeStatus : NFKServerKeyScopeInference;
	}
	*outPrefix = [self appRoutePrefixForPath:path];
	return NFKServerKeyScopeAppRoutes;
}

- (void)routeRequest:(NFKHTTPRequest *)request response:(NFKHTTPResponse *)response
{
	NSString *path = request.path;
	while (path.length > 1 && [path hasSuffix:@"/"]) {
		path = [path substringToIndex:path.length - 1];
	}
	NSString *appRoutePrefix = nil;
	NFKServerKeyScope scope = [self scopeOfPath:path appRoutePrefix:&appRoutePrefix];
	NSError *refusal = [self refusalOf:request scope:scope];
	if (refusal != nil) {
		[self sendError:refusal response:response];
		return;
	}
	if (request.firstOnConnection) {
		[self reportClientOf:request scope:scope];
	}
	if (appRoutePrefix != nil) {
		[self serveAppRoute:appRoutePrefix request:request response:response];
		return;
	}
	if ([path isEqualToString:@"/mcp"]) {
		[NFKServedMCP serveRequest:request response:response tools:self.tools instructions:self.MCPInstructions sessions:self.mcpSessions];
		return;
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
	BOOL underBase = [path isEqualToString:NFKServedBasePath] || [path hasPrefix:[NFKServedBasePath stringByAppendingString:@"/"]];
	NSString *route = underBase ? [path substringFromIndex:NFKServedBasePath.length] : nil;
	if (route == nil || routes[route] == nil) {
		[self sendNoRouteFor:request response:response];
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
															@(NFKInferenceServerErrorInsufficientScope): @403,
															@(NFKInferenceServerErrorMisdirected): @421,
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
	if ([error.domain isEqualToString:NFKInferenceServerErrorDomain]) {
		NSDictionary<NSNumber *, NSString *> *challenges = @{ @(NFKInferenceServerErrorUnauthorized): @"Bearer",
															  @(NFKInferenceServerErrorInsufficientScope): @"Bearer error=\"insufficient_scope\"" };
		headers[@"WWW-Authenticate"] = challenges[@(error.code)];
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
- (void)enqueue:(NFKServedRun *)run on:(NFKServedModel *)model request:(NFKHTTPRequest *)request response:(NFKHTTPResponse *)response
{
	run.route = request.path;
	run.clientAddress = request.clientAddress ?: @"";
	run.keyName = request.keyName;
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

/*! A model's description: its backend's identifier, readiness, inputs, parameters, outputs, model, and
	status, with its load and residency. A model that is not resident describes the backend it last held
	and reports ready, since a request builds it. */
- (NSMutableDictionary<NSString *, id> *)descriptionOfModel:(NFKServedModel *)model
{
	NSTimeInterval timeout = [self idleTimeoutForModel:model];
	id<NFKInferenceBackend> backend = [model residentBackend];
	NSMutableDictionary<NSString *, id> *description = nil;
	if (backend != nil) {
		description = [[NFKServedModel descriptionOfBackend:backend] mutableCopy];
		description[@"ready"] = @(backend.isReady);
		description[@"status"] = [NFKInferenceServer statusOfBackend:backend];
	} else {
		description = [[model lastBackendDescription] mutableCopy];
		description[@"ready"] = @YES;
	}
	description[@"load"] = [model loadJSONObject];
	description[@"residency"] = [model residencyJSONObjectWithTimeout:timeout pinned:[self isPinnedModelNamed:model.name]];
	return description;
}

- (NSDictionary<NSString *, id> *)entryForModel:(NFKServedModel *)model
{
	return @{ @"id": model.name, @"object": @"model", @"created": @(self.startTime), @"owned_by": @"inferkit",
			  @"inferkit": [self descriptionOfModel:model] };
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
	[response sendStatus:200 JSONObject:[self statusJSONObject] headers:nil];
}

/*! The status route's reply. */
- (NSDictionary<NSString *, id> *)statusJSONObject
{
	NSMutableArray *models = [NSMutableArray array];
	for (NSString *name in self.modelNames) {
		NFKServedModel *model = nil;
		@synchronized (self.models) {
			model = self.models[name];
		}
		if (model != nil) {
			NSMutableDictionary<NSString *, id> *entry = [self descriptionOfModel:model];
			[entry removeObjectsForKeys:@[ @"inputs", @"parameters", @"outputs" ]];
			entry[@"id"] = model.name;
			[models addObject:entry];
		}
	}
	NSMutableDictionary<NSString *, id> *status = [NSMutableDictionary dictionary];
	status[@"object"] = @"inferkit.status";
	NSMutableDictionary<NSString *, id> *server = [NSMutableDictionary dictionaryWithObject:NFKInferKit.version forKey:@"version"];
	if (self.startTime > 0) {
		server[@"started"] = @(self.startTime);
	}
	server[@"uptime_seconds"] = @(self.running ? MAX(NSDate.date.timeIntervalSince1970 - (NSTimeInterval)self.startTime, 0.0) : 0.0);
	status[@"server"] = server;
	status[@"models"] = models;
	if (self.reportsHostDetails) {
		NSMutableDictionary<NSString *, id> *host = [[self.hostStatus JSONObjectForStorageURL:self.storageDirectoryURL] mutableCopy];
		host[@"runtimes"] = [NFKInferenceServer runtimeStatuses];
		status[@"host"] = host;
	}
	return status;
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
		[self enqueue:run on:model request:request response:response];
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
	[self enqueue:run on:model request:request response:response];
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
	[self enqueue:run on:model request:request response:response];
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
	[self enqueue:run on:model request:request response:response];
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
	[self enqueue:run on:model request:request response:response];
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
	[self enqueue:run on:model request:request response:response];
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
	[self enqueue:run on:model request:request response:response];
}

@end
