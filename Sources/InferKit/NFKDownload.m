//
//  NFKDownload.m
//  InferKit
//

#import "NFKDownload.h"
#import "NFKDownloadInternal.h"

NSErrorDomain const NFKDownloadErrorDomain = @"NFKDownloadErrorDomain";
NSString * const NFKDownloadHTTPStatusCodeKey = @"NFKDownloadHTTPStatusCode";
const long long NFKDownloadUnknownByteCount = -1;

static NSString * const NFKDownloadCurrentStackKey = @"NFKDownloadCurrentStack";
// Marks a download's report queue, so a wait called from a progress handler does not wait on itself.
static char NFKDownloadReportQueueKey;
static const NSTimeInterval NFKDownloadReportInterval = 0.1;
static const NSTimeInterval NFKDownloadRateWindow = 3.0;

#pragma mark - File progress

@interface NFKDownloadFileProgress ()
@property (nonatomic, copy) NSString *path;
@property (nonatomic, copy) NSURL *localURL;
@property (nonatomic, assign) long long receivedByteCount;
@property (nonatomic, assign) long long expectedByteCount;
@property (nonatomic, assign) BOOL completed;
@property (nonatomic, assign) BOOL cached;
@end

@implementation NFKDownloadFileProgress

- (instancetype)initWithPath:(NSString *)path localURL:(NSURL *)localURL expectedByteCount:(long long)expected
{
	self = [super init];
	if (self != nil) {
		_path = [path copy];
		_localURL = [localURL copy];
		_expectedByteCount = expected;
	}
	return self;
}

- (instancetype)snapshotCopy
{
	NFKDownloadFileProgress *copy = [[NFKDownloadFileProgress alloc] initWithPath:self.path
																		 localURL:self.localURL
																expectedByteCount:self.expectedByteCount];
	copy.receivedByteCount = self.receivedByteCount;
	copy.completed = self.completed;
	copy.cached = self.cached;
	return copy;
}

- (NSString *)description
{
	return [NSString stringWithFormat:@"<%@ %@ %lld/%lld%@>", self.class, self.path, self.receivedByteCount,
			self.expectedByteCount, self.completed ? @" completed" : @""];
}

@end

#pragma mark - Progress snapshot

@interface NFKDownloadProgress ()
@property (nonatomic, assign) NFKDownloadState state;
@property (nonatomic, assign) long long receivedByteCount;
@property (nonatomic, assign) long long expectedByteCount;
@property (nonatomic, assign) NSUInteger fileCount;
@property (nonatomic, assign) NSUInteger completedFileCount;
@property (nonatomic, strong, nullable) NFKDownloadFileProgress *currentFile;
@property (nonatomic, copy) NSArray<NFKDownloadFileProgress *> *files;
@property (nonatomic, assign) double bytesPerSecond;
@property (nonatomic, copy, nullable) NSNumber *estimatedSecondsRemaining;
@end

@implementation NFKDownloadProgress

- (instancetype)initEmpty
{
	self = [super init];
	if (self != nil) {
		_files = @[];
	}
	return self;
}

- (double)fractionCompleted
{
	if (self.state == NFKDownloadStateCompleted) {
		return 1.0;
	}
	if (self.expectedByteCount <= 0) {
		return 0.0;
	}
	return MIN(1.0, (double)self.receivedByteCount / (double)self.expectedByteCount);
}

- (NSString *)description
{
	return [NSString stringWithFormat:@"<%@ state %ld, %lld/%lld bytes, %lu/%lu files>", self.class, (long)self.state,
			self.receivedByteCount, self.expectedByteCount, (unsigned long)self.completedFileCount,
			(unsigned long)self.fileCount];
}

@end

#pragma mark - Download

/*! Snapshot folders that unfinished downloads write into, counted per download. */
static NSCountedSet<NSString *> *NFKDownloadActiveSnapshots(void)
{
	static NSCountedSet<NSString *> *snapshots = nil;
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		snapshots = [NSCountedSet set];
	});
	return snapshots;
}

static NSRecursiveLock *NFKDownloadRegistryLock(void)
{
	static NSRecursiveLock *lock = nil;
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		lock = [[NSRecursiveLock alloc] init];
	});
	return lock;
}

/*! The snapshot's path with the links of its deepest existing folder resolved, so the key is the same
	before the folder exists and after. */
static NSString *NFKDownloadSnapshotKey(NSURL *snapshotURL)
{
	NSURL *existing = snapshotURL.URLByStandardizingPath;
	NSMutableArray<NSString *> *missing = [NSMutableArray array];
	while (existing.path.length > 1 && ![NSFileManager.defaultManager fileExistsAtPath:existing.path]) {
		[missing insertObject:existing.lastPathComponent atIndex:0];
		existing = existing.URLByDeletingLastPathComponent;
	}
	NSString *path = existing.URLByResolvingSymlinksInPath.path;
	for (NSString *component in missing) {
		path = [path stringByAppendingPathComponent:component];
	}
	return path;
}

@interface NFKDownload ()
@property (nonatomic, copy, nullable) NFKDownloadProgressHandler progressHandler;
@property (nonatomic, assign) NFKDownloadState state;
@property (nonatomic, assign, getter=isCancelled) BOOL cancelled;
@property (nonatomic, copy, nullable) NSError *error;
@property (nonatomic, copy, nullable) NFKDownloadPlan *plan;
@property (nonatomic, copy, nullable) NSURL *localURL;
@property (nonatomic, strong) NSProgress *progress;
@property (nonatomic, strong) NSObject *lock;
@property (nonatomic, strong) dispatch_queue_t reportQueue;
@property (nonatomic, strong) NSCondition *finishedCondition;
/*! YES once the final report and the completion handler have run. Read and written under
	finishedCondition. */
@property (nonatomic, assign) BOOL finalReportDelivered;
@property (nonatomic, strong) NSMutableArray<NFKDownloadFileProgress *> *fileRecords;
@property (nonatomic, strong) NSMutableDictionary<NSString *, NFKDownloadFileProgress *> *fileRecordsByLocalPath;
@property (nonatomic, strong) NSMutableSet<NSString *> *declaredLocalPaths;
@property (nonatomic, copy, nullable) NSString *currentLocalPath;
@property (nonatomic, strong) NSMutableDictionary<NSUUID *, dispatch_block_t> *cancellationHandlers;
@property (nonatomic, strong) NSMutableSet<NSString *> *snapshotKeys;
@property (nonatomic, strong) NSMutableArray<NSArray<NSNumber *> *> *rateSamples;
@property (nonatomic, assign) NSInteger activityCount;
@property (nonatomic, copy, nullable) NSError *firstActivityError;
@property (nonatomic, copy, nullable) NSURL *resultURL;
@property (nonatomic, assign) CFAbsoluteTime lastReportTime;
@end

@implementation NFKDownload

@synthesize completionHandler = _completionHandler;

+ (instancetype)downloadWithProgressHandler:(nullable NFKDownloadProgressHandler)progressHandler
{
	return [[self alloc] initWithProgressHandler:progressHandler];
}

- (instancetype)init
{
	return [self initWithProgressHandler:nil];
}

- (instancetype)initWithProgressHandler:(nullable NFKDownloadProgressHandler)progressHandler
{
	self = [super init];
	if (self != nil) {
		_progressHandler = [progressHandler copy];
		_state = NFKDownloadStatePending;
		_lock = [[NSObject alloc] init];
		_reportQueue = dispatch_queue_create("com.inferkit.download.progress",
											 dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INITIATED, 0));
		dispatch_queue_set_specific(_reportQueue, &NFKDownloadReportQueueKey, &NFKDownloadReportQueueKey, NULL);
		_finishedCondition = [[NSCondition alloc] init];
		_fileRecords = [NSMutableArray array];
		_fileRecordsByLocalPath = [NSMutableDictionary dictionary];
		_declaredLocalPaths = [NSMutableSet set];
		_cancellationHandlers = [NSMutableDictionary dictionary];
		_snapshotKeys = [NSMutableSet set];
		_rateSamples = [NSMutableArray array];
		_progress = [NSProgress discreteProgressWithTotalUnitCount:0];
		_progress.kind = NSProgressKindFile;
		[_progress setUserInfoObject:NSProgressFileOperationKindDownloading forKey:NSProgressFileOperationKindKey];
		_progress.cancellable = YES;
		__weak NFKDownload *weakSelf = self;
		_progress.cancellationHandler = ^{
			[weakSelf cancel];
		};
	}
	return self;
}

- (BOOL)isFinished
{
	@synchronized (self.lock) {
		return [self stateIsFinal:self.state];
	}
}

- (BOOL)stateIsFinal:(NFKDownloadState)state
{
	return state == NFKDownloadStateCompleted || state == NFKDownloadStateFailed || state == NFKDownloadStateCancelled;
}

- (NFKDownloadState)state
{
	@synchronized (self.lock) {
		return _state;
	}
}

- (BOOL)isCancelled
{
	@synchronized (self.lock) {
		return _cancelled;
	}
}

- (nullable NSError *)error
{
	@synchronized (self.lock) {
		return _error;
	}
}

- (nullable NFKDownloadPlan *)plan
{
	@synchronized (self.lock) {
		return _plan;
	}
}

- (nullable NSURL *)localURL
{
	@synchronized (self.lock) {
		return _localURL;
	}
}

- (NFKDownloadProgress *)currentProgress
{
	@synchronized (self.lock) {
		return [self snapshotLocked];
	}
}

#pragma mark Snapshots and reports

- (NFKDownloadProgress *)snapshotLocked
{
	NFKDownloadProgress *snapshot = [[NFKDownloadProgress alloc] initEmpty];
	snapshot.state = self.state;
	long long received = 0;
	long long expected = 0;
	NSUInteger completed = 0;
	NSMutableArray<NFKDownloadFileProgress *> *files = [NSMutableArray arrayWithCapacity:self.fileRecords.count];
	for (NFKDownloadFileProgress *record in self.fileRecords) {
		[files addObject:[record snapshotCopy]];
		if (record.cached) {
			completed++;
			continue;
		}
		received += record.receivedByteCount;
		expected += MAX(record.expectedByteCount, record.receivedByteCount);
		if (record.completed) {
			completed++;
		}
	}
	snapshot.files = files;
	snapshot.fileCount = self.fileRecords.count;
	snapshot.completedFileCount = completed;
	snapshot.receivedByteCount = received;
	snapshot.expectedByteCount = MAX(expected, received);
	NFKDownloadFileProgress *current = self.currentLocalPath != nil ? self.fileRecordsByLocalPath[self.currentLocalPath] : nil;
	snapshot.currentFile = [current snapshotCopy];
	snapshot.bytesPerSecond = [self bytesPerSecondLocked];
	if (snapshot.bytesPerSecond > 0 && expected > 0 && ![self stateIsFinal:self.state]) {
		snapshot.estimatedSecondsRemaining = @((double)(snapshot.expectedByteCount - received) / snapshot.bytesPerSecond);
	}
	return snapshot;
}

- (double)bytesPerSecondLocked
{
	if (self.rateSamples.count < 2) {
		return 0;
	}
	NSArray<NSNumber *> *first = self.rateSamples.firstObject;
	NSArray<NSNumber *> *last = self.rateSamples.lastObject;
	double elapsed = last[0].doubleValue - first[0].doubleValue;
	if (elapsed <= 0) {
		return 0;
	}
	return (last[1].doubleValue - first[1].doubleValue) / elapsed;
}

- (void)sampleRateLocked
{
	long long received = 0;
	for (NFKDownloadFileProgress *record in self.fileRecords) {
		if (!record.cached) {
			received += record.receivedByteCount;
		}
	}
	CFAbsoluteTime now = CFAbsoluteTimeGetCurrent();
	[self.rateSamples addObject:@[ @(now), @(received) ]];
	while (self.rateSamples.count > 2 && now - self.rateSamples[1][0].doubleValue > NFKDownloadRateWindow) {
		[self.rateSamples removeObjectAtIndex:0];
	}
}

/*! Delivers a snapshot to the handler and the NSProgress. Called with the lock held. */
- (void)reportLockedForced:(BOOL)forced
{
	CFAbsoluteTime now = CFAbsoluteTimeGetCurrent();
	if (!forced && now - self.lastReportTime < NFKDownloadReportInterval) {
		return;
	}
	self.lastReportTime = now;
	NFKDownloadProgress *snapshot = [self snapshotLocked];
	NFKDownloadProgressHandler handler = self.progressHandler;
	NSProgress *progress = self.progress;
	dispatch_async(self.reportQueue, ^{
		progress.totalUnitCount = MAX(snapshot.expectedByteCount, 1);
		progress.completedUnitCount = snapshot.state == NFKDownloadStateCompleted ? progress.totalUnitCount
																				  : snapshot.receivedByteCount;
		progress.fileTotalCount = @(snapshot.fileCount);
		progress.fileCompletedCount = @(snapshot.completedFileCount);
		progress.throughput = @((NSUInteger)snapshot.bytesPerSecond);
		progress.estimatedTimeRemaining = snapshot.estimatedSecondsRemaining;
		if (handler != nil) {
			handler(snapshot);
		}
	});
}

#pragma mark Hub-driven state

- (void)markPlanning
{
	@synchronized (self.lock) {
		if (self.state != NFKDownloadStatePending) {
			return;
		}
		self.state = NFKDownloadStatePlanning;
		[self reportLockedForced:YES];
	}
}

- (void)adoptPlan:(NFKDownloadPlan *)plan
{
	@synchronized (self.lock) {
		self.plan = plan;
		if (self.state == NFKDownloadStatePending || self.state == NFKDownloadStatePlanning) {
			self.state = NFKDownloadStateDownloading;
		}
	}
	[self addPlan:plan];
}

- (void)setResultURL:(nullable NSURL *)URL
{
	@synchronized (self.lock) {
		_resultURL = [URL copy];
	}
}

#pragma mark Collecting work

+ (nullable NFKDownload *)currentDownload
{
	NSMutableArray<NFKDownload *> *stack = NSThread.currentThread.threadDictionary[NFKDownloadCurrentStackKey];
	return stack.lastObject;
}

- (void)becomeCurrent
{
	NSMutableDictionary *threadDictionary = NSThread.currentThread.threadDictionary;
	NSMutableArray<NFKDownload *> *stack = threadDictionary[NFKDownloadCurrentStackKey];
	if (stack == nil) {
		stack = [NSMutableArray array];
		threadDictionary[NFKDownloadCurrentStackKey] = stack;
	}
	[stack addObject:self];
}

- (void)resignCurrent
{
	NSMutableArray<NFKDownload *> *stack = NSThread.currentThread.threadDictionary[NFKDownloadCurrentStackKey];
	NSUInteger index = [stack indexOfObjectIdenticalTo:self];
	if (index == NSNotFound) {
		return;
	}
	[stack removeObjectAtIndex:index];
}

- (void)performAsCurrent:(void (NS_NOESCAPE ^)(void))block
{
	[self becomeCurrent];
	@try {
		block();
	} @finally {
		[self resignCurrent];
	}
}

- (void)addPlan:(NFKDownloadPlan *)plan
{
	@synchronized (self.lock) {
		if ([self stateIsFinal:self.state]) {
			return;
		}
		for (NFKDownloadPlanFile *file in plan.files) {
			NSString *key = file.localURL.path;
			[self.declaredLocalPaths addObject:key];
			NFKDownloadFileProgress *record = self.fileRecordsByLocalPath[key];
			if (record != nil) {
				if (record.expectedByteCount < 0 && file.byteCount >= 0) {
					record.expectedByteCount = file.byteCount;
				}
				continue;
			}
			record = [[NFKDownloadFileProgress alloc] initWithPath:file.path localURL:file.localURL expectedByteCount:file.byteCount];
			record.cached = file.cached;
			record.completed = file.cached;
			[self.fileRecords addObject:record];
			self.fileRecordsByLocalPath[key] = record;
		}
		[self reportLockedForced:YES];
	}
}

- (void)beginActivity
{
	@synchronized (self.lock) {
		if ([self stateIsFinal:self.state]) {
			return;
		}
		self.activityCount += 1;
		if (self.state == NFKDownloadStatePending) {
			self.state = NFKDownloadStateDownloading;
			[self reportLockedForced:YES];
		}
	}
}

- (void)endActivityWithError:(nullable NSError *)error
{
	@synchronized (self.lock) {
		if ([self stateIsFinal:self.state] || self.activityCount == 0) {
			return;
		}
		if (error != nil && self.firstActivityError == nil) {
			self.firstActivityError = error;
		}
		self.activityCount -= 1;
		if (self.activityCount > 0) {
			return;
		}
	}
	[self finish];
}

/*! Moves to the final state the activities and cancellation call for, once. */
- (void)finish
{
	NFKDownloadCompletionHandler completion = nil;
	NSURL *resultURL = nil;
	NSError *finalError = nil;
	NSArray<NSString *> *snapshotKeys = nil;
	@synchronized (self.lock) {
		if ([self stateIsFinal:self.state]) {
			return;
		}
		if (self.cancelled) {
			self.state = NFKDownloadStateCancelled;
			self.error = [NFKDownload cancellationError];
		} else if (self.firstActivityError != nil) {
			BOOL cancelledError = [self.firstActivityError.domain isEqualToString:NFKDownloadErrorDomain]
				&& self.firstActivityError.code == NFKDownloadErrorCancelled;
			self.state = cancelledError ? NFKDownloadStateCancelled : NFKDownloadStateFailed;
			self.error = self.firstActivityError;
		} else {
			self.state = NFKDownloadStateCompleted;
			self.localURL = self.resultURL;
		}
		self.currentLocalPath = nil;
		finalError = self.error;
		resultURL = self.localURL;
		completion = self.completionHandler;
		self.completionHandler = nil;
		[self.cancellationHandlers removeAllObjects];
		snapshotKeys = self.snapshotKeys.allObjects;
		[self.snapshotKeys removeAllObjects];
		[self reportLockedForced:YES];
	}
	[NFKDownloadRegistryLock() lock];
	for (NSString *key in snapshotKeys) {
		[NFKDownloadActiveSnapshots() removeObject:key];
	}
	[NFKDownloadRegistryLock() unlock];

	// Queued behind the final report, so a finished wait has seen every report and the completion.
	NSCondition *finishedCondition = self.finishedCondition;
	dispatch_async(self.reportQueue, ^{
		if (completion != nil) {
			completion(resultURL, finalError);
		}
		[finishedCondition lock];
		self.finalReportDelivered = YES;
		[finishedCondition broadcast];
		[finishedCondition unlock];
	});
}

- (void)cancel
{
	NSArray<dispatch_block_t> *handlers = nil;
	BOOL finishNow = NO;
	@synchronized (self.lock) {
		if (self.cancelled || [self stateIsFinal:self.state]) {
			return;
		}
		self.cancelled = YES;
		handlers = self.cancellationHandlers.allValues;
		finishNow = self.activityCount == 0;
	}
	for (dispatch_block_t handler in handlers) {
		handler();
	}
	if (!self.progress.isCancelled) {
		[self.progress cancel];
	}
	if (finishNow) {
		[self finish];
	}
}

- (BOOL)waitUntilFinishedWithTimeout:(NSTimeInterval)timeout
{
	if (dispatch_get_specific(&NFKDownloadReportQueueKey) != NULL) {
		// On a report queue the final report cannot be delivered while this waits.
		return self.isFinished;
	}
	NSDate *deadline = timeout < 0 ? NSDate.distantFuture : [NSDate dateWithTimeIntervalSinceNow:timeout];
	[self.finishedCondition lock];
	BOOL delivered = self.finalReportDelivered;
	while (!delivered) {
		if (![self.finishedCondition waitUntilDate:deadline]) {
			delivered = self.finalReportDelivered;
			break;
		}
		delivered = self.finalReportDelivered;
	}
	[self.finishedCondition unlock];
	return delivered;
}

#pragma mark File reports

- (NFKDownloadFileProgress *)recordLockedForLocalURL:(NSURL *)localURL path:(NSString *)path expected:(long long)expected
{
	NSString *key = localURL.path;
	NFKDownloadFileProgress *record = self.fileRecordsByLocalPath[key];
	if (record == nil) {
		record = [[NFKDownloadFileProgress alloc] initWithPath:path localURL:localURL expectedByteCount:expected];
		[self.fileRecords addObject:record];
		self.fileRecordsByLocalPath[key] = record;
	}
	return record;
}

- (void)fileWillStartAtLocalURL:(NSURL *)localURL path:(NSString *)path expectedByteCount:(long long)expected
{
	@synchronized (self.lock) {
		if ([self stateIsFinal:self.state]) {
			return;
		}
		if (self.state == NFKDownloadStatePending || self.state == NFKDownloadStatePlanning) {
			self.state = NFKDownloadStateDownloading;
		}
		NFKDownloadFileProgress *record = [self recordLockedForLocalURL:localURL path:path expected:expected];
		if (record.expectedByteCount < 0 && expected >= 0) {
			record.expectedByteCount = expected;
		}
		record.cached = NO;
		record.completed = NO;
		self.currentLocalPath = localURL.path;
		[self sampleRateLocked];
		[self reportLockedForced:YES];
	}
}

- (void)fileAtLocalURL:(NSURL *)localURL didWriteByteCount:(long long)byteCount expectedByteCount:(long long)expected
{
	@synchronized (self.lock) {
		NFKDownloadFileProgress *record = self.fileRecordsByLocalPath[localURL.path];
		if (record == nil || [self stateIsFinal:self.state]) {
			return;
		}
		if (expected >= 0) {
			record.expectedByteCount = expected;
		}
		// A restarted transfer counts again only past the bytes it already reported, so the total
		// never decreases.
		record.receivedByteCount = MAX(record.receivedByteCount, byteCount);
		[self sampleRateLocked];
		[self reportLockedForced:NO];
	}
}

- (void)fileDidFinishAtLocalURL:(NSURL *)localURL path:(NSString *)path byteCount:(long long)byteCount cached:(BOOL)cached
{
	@synchronized (self.lock) {
		if ([self stateIsFinal:self.state]) {
			return;
		}
		NFKDownloadFileProgress *record = [self recordLockedForLocalURL:localURL path:path expected:NFKDownloadUnknownByteCount];
		if (record.completed) {
			return;
		}
		record.completed = YES;
		if (cached && record.receivedByteCount == 0) {
			record.cached = YES;
		} else {
			// A transport that reports no bytes (an override of the fetch seam) still lands the file.
			record.receivedByteCount = MAX(record.receivedByteCount, byteCount);
			record.expectedByteCount = record.receivedByteCount;
		}
		if ([self.currentLocalPath isEqualToString:localURL.path]) {
			self.currentLocalPath = nil;
		}
		[self sampleRateLocked];
		[self reportLockedForced:YES];
	}
}

- (void)fileDidFailAtLocalURL:(NSURL *)localURL
{
	@synchronized (self.lock) {
		NSString *key = localURL.path;
		if ([self.currentLocalPath isEqualToString:key]) {
			self.currentLocalPath = nil;
		}
		// A fetch no plan declared that fails, such as an optional file the release does not serve,
		// was never part of the download.
		NFKDownloadFileProgress *record = self.fileRecordsByLocalPath[key];
		if (record != nil && ![self.declaredLocalPaths containsObject:key] && record.receivedByteCount == 0) {
			[self.fileRecords removeObjectIdenticalTo:record];
			[self.fileRecordsByLocalPath removeObjectForKey:key];
		}
		[self reportLockedForced:YES];
	}
}

#pragma mark Cancellation and snapshots

- (id)addCancellationHandler:(dispatch_block_t)handler
{
	NSUUID *token = [NSUUID UUID];
	BOOL runNow = NO;
	@synchronized (self.lock) {
		if (self.cancelled) {
			runNow = YES;
		} else {
			self.cancellationHandlers[token] = [handler copy];
		}
	}
	if (runNow) {
		handler();
	}
	return token;
}

- (void)removeCancellationHandler:(id)token
{
	@synchronized (self.lock) {
		[self.cancellationHandlers removeObjectForKey:token];
	}
}

- (void)registerSnapshotURL:(NSURL *)snapshotURL
{
	NSString *key = NFKDownloadSnapshotKey(snapshotURL);
	[NFKDownloadRegistryLock() lock];
	@synchronized (self.lock) {
		if (![self stateIsFinal:self.state] && ![self.snapshotKeys containsObject:key]) {
			[self.snapshotKeys addObject:key];
			[NFKDownloadActiveSnapshots() addObject:key];
		}
	}
	[NFKDownloadRegistryLock() unlock];
}

+ (BOOL)isSnapshotURLInUse:(NSURL *)snapshotURL
{
	NSString *key = NFKDownloadSnapshotKey(snapshotURL);
	[NFKDownloadRegistryLock() lock];
	BOOL inUse = [NFKDownloadActiveSnapshots() countForObject:key] > 0;
	[NFKDownloadRegistryLock() unlock];
	return inUse;
}

+ (NSString *)holdSnapshotURL:(NSURL *)snapshotURL
{
	NSString *key = NFKDownloadSnapshotKey(snapshotURL);
	[NFKDownloadRegistryLock() lock];
	[NFKDownloadActiveSnapshots() addObject:key];
	[NFKDownloadRegistryLock() unlock];
	return key;
}

+ (void)endHoldOfSnapshotKey:(NSString *)key
{
	[NFKDownloadRegistryLock() lock];
	[NFKDownloadActiveSnapshots() removeObject:key];
	[NFKDownloadRegistryLock() unlock];
}

+ (void)performWithSnapshotRegistryLocked:(void (NS_NOESCAPE ^)(void))block
{
	[NFKDownloadRegistryLock() lock];
	@try {
		block();
	} @finally {
		[NFKDownloadRegistryLock() unlock];
	}
}

+ (NSError *)cancellationError
{
	return [self errorWithCode:NFKDownloadErrorCancelled reason:@"the download was cancelled"];
}

+ (NSError *)errorWithCode:(NFKDownloadError)code reason:(NSString *)reason
{
	return [NSError errorWithDomain:NFKDownloadErrorDomain code:code userInfo:@{ NSLocalizedDescriptionKey: reason }];
}

@end
