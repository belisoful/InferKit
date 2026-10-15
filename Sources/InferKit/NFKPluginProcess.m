//
//  NFKPluginProcess.m
//  InferKit
//

#import "NFKPluginProcess.h"
#import "NFKPluginBackend.h"
#import "NFKPluginWire.h"

#if TARGET_OS_OSX

#import "NFKPluginChannel.h"
#import "NFKPluginProcessPrivate.h"
#import "NFKPluginToolProxy.h"
#import "NFKServedHostStatus.h"
#import "NFKSubprocessPrivate.h"
#import <InferKit/NFKErrors.h>
#import <InferKit/NFKInferKit.h>
#import <signal.h>
#import <string.h>
#import <unistd.h>

#if __has_include(<InferKitC/NFKPluginABI.h>)
#import <InferKitC/NFKPluginABI.h>
#elif __has_include(<InferKit/NFKPluginABI.h>)
#import <InferKit/NFKPluginABI.h>
#else
#import "../InferKitC/include/InferKitC/NFKPluginABI.h"
#endif

/*! How long a shutdown waits for the plugin's reply before the channel closes. */
static const NSTimeInterval NFKPluginShutdownReplyTimeout = 2;

/*! How long cancelled runs may take to answer after a restart's drain timeout. */
static const NSTimeInterval NFKPluginCancelGrace = 5;

/*! The longest wait for a launcher's stop: NFKSubprocess's escalation takes at most about 10.5 s. */
static const NSTimeInterval NFKPluginLauncherStopTimeout = 15;

/*! How much of the process's standard error an exit error carries, in characters. */
static const NSUInteger NFKPluginStandardErrorTail = 2048;

/*! The idle exit interval memory pressure starts from when the configured one is infinite, as the
	server's pressureUnloadInterval. */
static const NSTimeInterval NFKPluginPressureBaseInterval = 600;

NSError *NFKPluginCancellationError(void)
{
	return [NSError errorWithDomain:NSCocoaErrorDomain code:NSUserCancelledError
						   userInfo:@{ NSLocalizedDescriptionKey: @"the plugin cancelled the run" }];
}

NSError *NFKPluginErrorFromReply(NSDictionary<NSString *, id> *reply)
{
	id cancelled = reply[@"cancelled"];
	if ([cancelled isKindOfClass:NSNumber.class] && [cancelled boolValue]) {
		return NFKPluginCancellationError();
	}
	id body = reply[@"error"];
	if (body == nil || body == NSNull.null) {
		return nil;
	}
	NSError *decoded = [NFKInferenceWireCoding errorFromJSONObject:@{ @"error": body }];
	if (decoded != nil) {
		return decoded;
	}
	NSDictionary *fields = [body isKindOfClass:NSDictionary.class] ? body : @{};
	NSString *message = [fields[@"message"] isKindOfClass:NSString.class] ? fields[@"message"] : @"the plugin reported an error";
	NSString *domain = [fields[@"domain"] isKindOfClass:NSString.class] ? fields[@"domain"] : nil;
	NSNumber *code = [fields[@"code"] isKindOfClass:NSNumber.class] ? fields[@"code"] : nil;
	if (domain.length > 0 && code != nil) {
		return [NSError errorWithDomain:domain code:code.integerValue userInfo:@{ NSLocalizedDescriptionKey: message }];
	}
	return [NSError errorWithDomain:NFKInferenceErrorDomain code:kNFKError_InferenceBackendFailure
						   userInfo:@{ NSLocalizedDescriptionKey: message }];
}

/*! An NFKPluginErrorDomain error with an underlying error. */
static NSError *NFKPluginWrappedError(NSInteger code, NSString *reason, NSError * _Nullable underlying)
{
	NSMutableDictionary *userInfo = [NSMutableDictionary dictionary];
	userInfo[NSLocalizedDescriptionKey] = underlying.localizedDescription.length > 0
		? [NSString stringWithFormat:@"%@: %@", reason, underlying.localizedDescription] : reason;
	userInfo[NSUnderlyingErrorKey] = underlying;
	return [NSError errorWithDomain:NFKPluginErrorDomain code:code userInfo:userInfo];
}

static NSError *NFKPluginMakeError(NSInteger code, NSString *reason)
{
	return [NSError errorWithDomain:NFKPluginErrorDomain code:code userInfo:@{ NSLocalizedDescriptionKey: reason }];
}

static NSString *NFKPluginHandleKey(id handle)
{
	return [NSString stringWithFormat:@"%@", handle];
}

static NSTimeInterval NFKPluginNow(void)
{
	return NSProcessInfo.processInfo.systemUptime;
}

/*! A dispatch deadline timeout seconds from now, or forever when timeout is not positive. */
static dispatch_time_t NFKPluginDeadline(NSTimeInterval timeout)
{
	return timeout > 0 ? dispatch_time(DISPATCH_TIME_NOW, (int64_t)(timeout * NSEC_PER_SEC)) : DISPATCH_TIME_FOREVER;
}

/*! The keys of a descriptor dictionary's ports or options. */
static NSSet<NSString *> *NFKPluginDescribedKeys(id entries)
{
	NSMutableSet<NSString *> *keys = [NSMutableSet set];
	if ([entries isKindOfClass:NSArray.class]) {
		for (id entry in entries) {
			if ([entry isKindOfClass:NSDictionary.class] && [entry[@"key"] isKindOfClass:NSString.class]) {
				[keys addObject:entry[@"key"]];
			}
		}
	}
	return keys;
}

#pragma mark - Pending request

@interface NFKPluginPendingRequest : NSObject
@property (nonatomic, assign) NSInteger identifier;
@property (nonatomic, copy) NSString *type;
@property (nonatomic, copy, nullable) NFKPluginProgressHandler progressHandler;
@property (nonatomic, copy, nullable) NFKPluginReplyHandler replyHandler;
@end

@implementation NFKPluginPendingRequest
@end

#pragma mark - Admission

/*! A request's place behind a restart. */
@interface NFKPluginAdmission : NSObject
@property (nonatomic, assign) BOOL admitted;
@property (nonatomic, assign) BOOL finished;
@property (nonatomic, strong, nullable) NSError *error;
@end

@implementation NFKPluginAdmission
@end

#pragma mark - Subprocess launcher

/*! Runs the plugin host, or an executable plugin's own program, through NFKSubprocess. */
@interface NFKPluginSubprocessLauncher : NSObject <NFKPluginLauncher>
@property (atomic, readwrite, strong, nullable) NFKSubprocess *subprocess;
@end

@implementation NFKPluginSubprocessLauncher

- (BOOL)launchPluginProcess:(NFKPluginProcess *)process channelDescriptor:(int)descriptor error:(NSError **)error
{
	NSArray<NSString *> *arguments = nil;
	NSString *command = [self commandForProcess:process arguments:&arguments error:error];
	if (command == nil) {
		return NO;
	}
	NFKSubprocess *subprocess = [[NFKSubprocess alloc] initWithCommand:command arguments:arguments];
	subprocess.environment = @{ NFKPluginChannelEnvironmentVariable: [NSString stringWithFormat:@"%d", NFKPluginChannelDescriptorNumber] };
	subprocess.ownerInheritedDescriptors = @{ @(NFKPluginChannelDescriptorNumber): @(descriptor) };
	subprocess.ownerQualityOfService = QOS_CLASS_USER_INITIATED;
	subprocess.restartPolicy = NFKSubprocessRestartPolicyNever;
	self.subprocess = subprocess;
	return [subprocess startWithError:error];
}

/*! The program and its arguments: the plugin host with the bundle for "objc" and "c-abi", or the
	bundle's own executable for "executable". */
- (nullable NSString *)commandForProcess:(NFKPluginProcess *)process arguments:(NSArray<NSString *> **)outArguments error:(NSError **)error
{
	NFKPluginManifest *manifest = process.manifest;
	NSURL *bundle = manifest.bundleURL;
	if (bundle == nil) {
		NFKPluginSetError(error, NFKPluginErrorLaunchFailed, [NSString stringWithFormat:@"the manifest of %@ names no bundle", manifest.identifier]);
		return nil;
	}
	if (![manifest.entryKind isEqualToString:NFKPluginEntryKindExecutable]) {
		if (process.hostExecutableURL == nil) {
			NFKPluginSetError(error, NFKPluginErrorLaunchFailed, @"no plugin host executable is set");
			return nil;
		}
		NSMutableArray<NSString *> *arguments = [NSMutableArray arrayWithObjects:@"--bundle", bundle.path, nil];
		NFKPluginSignaturePolicy *policy = process.signaturePolicy;
		if (policy.teamIdentifier.length > 0) {
			[arguments addObjectsFromArray:@[ @"--team-id", policy.teamIdentifier ]];
		}
		if (policy.requirementString.length > 0) {
			[arguments addObjectsFromArray:@[ @"--requirement", policy.requirementString ]];
		}
		if (policy.allowsAdHocSignature) {
			[arguments addObject:@"--allow-ad-hoc"];
		}
		if (!policy.requiresValidSignature) {
			// The host honors it only beside --allow-ad-hoc; alone it refuses to start, with the reason
			// on standard error.
			[arguments addObject:@"--allow-unsigned"];
		}
		*outArguments = arguments;
		return process.hostExecutableURL.path;
	}
	NSURL *executable = nil;
	if (manifest.executablePath.length > 0) {
		NSURL *directory = [bundle URLByAppendingPathComponent:@"Contents/MacOS" isDirectory:YES].URLByStandardizingPath;
		executable = [directory URLByAppendingPathComponent:manifest.executablePath].URLByStandardizingPath;
		NSString *prefix = [directory.path stringByAppendingString:@"/"];
		if (![executable.path hasPrefix:prefix]) {
			NFKPluginSetError(error, NFKPluginErrorLaunchFailed, @"the manifest's executable lies outside the bundle");
			return nil;
		}
	} else {
		executable = [NSBundle bundleWithURL:bundle].executableURL;
	}
	if (executable == nil) {
		NFKPluginSetError(error, NFKPluginErrorLaunchFailed, [NSString stringWithFormat:@"%@ names no executable", bundle.lastPathComponent]);
		return nil;
	}
	*outArguments = manifest.arguments;
	return executable.path;
}

- (void)stopWithCompletionHandler:(void (^)(void))completionHandler
{
	NFKSubprocess *subprocess = self.subprocess;
	if (subprocess == nil) {
		completionHandler();
		return;
	}
	[subprocess stopWithCompletionHandler:completionHandler];
}

- (pid_t)processIdentifier
{
	return self.subprocess.processIdentifier;
}

- (NFKSubprocessTerminationReason)terminationReason
{
	return self.subprocess.terminationReason;
}

- (int)terminationStatus
{
	return self.subprocess.terminationStatus;
}

- (NSString *)recentStandardError
{
	return self.subprocess.recentStandardError ?: @"";
}

@end

#pragma mark - Process interface

@interface NFKPluginProcess ()
@property (atomic, readwrite, copy) NFKPluginManifest *manifest;
@property (atomic, readwrite) NFKPluginProcessState state;
@property (atomic, readwrite, copy, nullable) NSError *failureError;
@property (atomic, readwrite) NFKSubprocessTerminationReason terminationReason;
@property (atomic, readwrite) int terminationStatus;
@property (atomic, readwrite) pid_t processIdentifier;
@property (atomic, readwrite, strong, nullable) NFKSubprocess *subprocess;
@property (atomic, readwrite) NSUInteger generation;
@property (atomic, strong, nullable) NFKPluginConnection *connection;
@property (nonatomic, strong) NSObject *stateLock;
@property (nonatomic, strong) NSLock *startLock;
@property (nonatomic, strong) NSCondition *admissionCondition;
@property (nonatomic, strong) NSMutableArray<NFKPluginAdmission *> *heldAdmissions;
@property (nonatomic, assign) BOOL holding;
@property (nonatomic, assign) NSUInteger pendingRestarts;
@property (nonatomic, strong) NSMutableArray<NSNumber *> *crashTimes;
@property (nonatomic, strong) dispatch_queue_t eventQueue;
@property (nonatomic, strong) dispatch_queue_t controlQueue;
@property (nonatomic, strong) dispatch_queue_t idleQueue;
@property (nonatomic, strong, nullable) dispatch_source_t idleTimer;
@property (nonatomic, strong, nullable) NFKServedHostStatus *hostStatus;
- (void)connectionDidEnd:(NFKPluginConnection *)connection error:(nullable NSError *)error;
@end

#pragma mark - Connection

@interface NFKPluginConnection ()
@property (nonatomic, readwrite) NSUInteger generation;
@property (nonatomic, readwrite) NSUInteger inlineByteLimit;
@property (atomic, readwrite, copy) NSDictionary<NSString *, id> *helloReply;
@property (atomic, readwrite, getter=isClosed) BOOL closed;
/*! Runs when the plugin host reports that it has verified the bundle. */
@property (atomic, copy, nullable) dispatch_block_t verificationHandler;
- (void)abandonRequest:(NSInteger)identifier;
@property (nonatomic, strong) NFKPluginChannel *channel;
@property (nonatomic, strong) id<NFKPluginLauncher> launcher;
@property (nonatomic, weak) NFKPluginProcess *process;
@property (nonatomic, strong) NSObject *lock;
@property (nonatomic, strong) NSMutableDictionary<NSNumber *, NFKPluginPendingRequest *> *pending;
@property (nonatomic, assign) NSInteger nextIdentifier;
@property (nonatomic, strong) NSMapTable<NSString *, NFKPluginBackend *> *backends;
@property (nonatomic, assign) NSInteger liveBackendCount;
@property (nonatomic, assign) NSInteger activeUses;
@property (nonatomic, assign) NSTimeInterval lastActivity;
@property (nonatomic, assign) BOOL retiring;
@property (nonatomic, assign) BOOL intentional;
@property (nonatomic, strong, nullable) NSError *closeError;
@end

@implementation NFKPluginConnection

- (instancetype)initWithChannel:(NFKPluginChannel *)channel
					 generation:(NSUInteger)generation
					   launcher:(id<NFKPluginLauncher>)launcher
						process:(NFKPluginProcess *)process
{
	self = [super init];
	if (self != nil) {
		_channel = channel;
		_generation = generation;
		_launcher = launcher;
		_process = process;
		_inlineByteLimit = NFKPluginDefaultInlineByteLimit;
		_helloReply = @{};
		_lock = [[NSObject alloc] init];
		_pending = [NSMutableDictionary dictionary];
		_backends = [NSMapTable strongToWeakObjectsMapTable];
		_lastActivity = NFKPluginNow();
		// The handlers hold the connection until the channel closes; channelDidCloseWithError: lets go.
		channel.messageHandler = ^(NSDictionary<NSString *, id> *message, NSArray<NFKPluginDescriptor *> *descriptors) {
			[self handleMessage:message descriptors:descriptors];
		};
		channel.closeHandler = ^(NSError * _Nullable error) {
			[self channelDidCloseWithError:error];
		};
	}
	return self;
}

#pragma mark Requests

- (NSInteger)sendRequest:(NSDictionary<NSString *, id> *)message
			 descriptors:(nullable NSArray<NSNumber *> *)descriptors
		 progressHandler:(nullable NFKPluginProgressHandler)progressHandler
			replyHandler:(NFKPluginReplyHandler)replyHandler
				   error:(NSError **)error
{
	NFKPluginPendingRequest *pending = [[NFKPluginPendingRequest alloc] init];
	pending.type = [message[@"type"] isKindOfClass:NSString.class] ? message[@"type"] : @"";
	pending.progressHandler = progressHandler;
	pending.replyHandler = replyHandler;
	NSInteger identifier = 0;
	@synchronized (self.lock) {
		if (self.closed) {
			NFKPluginSetError(error, NFKPluginErrorStopped, @"the plugin's process is not running");
			return 0;
		}
		self.nextIdentifier += 1;
		identifier = self.nextIdentifier;
		pending.identifier = identifier;
		self.pending[@(identifier)] = pending;
		self.lastActivity = NFKPluginNow();
	}
	NSMutableDictionary *framed = [message mutableCopy];
	framed[@"id"] = @(identifier);
	NSError *sendError = nil;
	if ([self.channel sendMessage:framed descriptors:descriptors error:&sendError]) {
		return identifier;
	}
	// A close that raced the send already answered the request; only a request still pending is the
	// caller's to fail.
	BOOL removed = NO;
	@synchronized (self.lock) {
		removed = self.pending[@(identifier)] != nil;
		[self.pending removeObjectForKey:@(identifier)];
	}
	if (!removed) {
		return identifier;
	}
	if (error != NULL) {
		*error = sendError;
	}
	return 0;
}

- (nullable id)sendRequestAndWait:(NSDictionary<NSString *, id> *)message timeout:(NSTimeInterval)timeout error:(NSError **)error
{
	__block id result = nil;
	__block NSError *failure = nil;
	dispatch_semaphore_t done = dispatch_semaphore_create(0);
	NSInteger identifier = [self sendRequest:message descriptors:nil progressHandler:nil
								replyHandler:^(NSDictionary<NSString *, id> * _Nullable reply, NSArray<NFKPluginDescriptor *> *descriptors, NSError * _Nullable replyError) {
		if (reply == nil) {
			failure = replyError;
		} else {
			failure = NFKPluginErrorFromReply(reply);
			result = failure == nil ? (reply[@"result"] ?: NSNull.null) : nil;
		}
		dispatch_semaphore_signal(done);
	} error:error];
	if (identifier == 0) {
		return nil;
	}
	dispatch_time_t deadline = timeout > 0 ? dispatch_time(DISPATCH_TIME_NOW, (int64_t)(timeout * NSEC_PER_SEC)) : DISPATCH_TIME_FOREVER;
	if (dispatch_semaphore_wait(done, deadline) != 0) {
		@synchronized (self.lock) {
			[self.pending removeObjectForKey:@(identifier)];
		}
		NFKPluginSetError(error, NFKPluginErrorProtocolViolation,
						  [NSString stringWithFormat:@"the plugin did not answer %@ within %g seconds", message[@"type"], timeout]);
		return nil;
	}
	if (result == nil && error != NULL) {
		*error = failure ?: NFKPluginMakeError(NFKPluginErrorStopped, @"the plugin's process is not running");
	}
	return result;
}

- (BOOL)sendMessage:(NSDictionary<NSString *, id> *)message error:(NSError **)error
{
	return [self.channel sendMessage:message descriptors:nil error:error];
}

/*! Forgets a request whose reply the caller no longer waits for. */
- (void)abandonRequest:(NSInteger)identifier
{
	@synchronized (self.lock) {
		[self.pending removeObjectForKey:@(identifier)];
	}
}

- (NSUInteger)pendingCount
{
	@synchronized (self.lock) {
		return self.pending.count;
	}
}

/*! Asks the plugin to cancel every run in flight. */
- (void)cancelRuns
{
	NSArray<NFKPluginPendingRequest *> *runs = nil;
	@synchronized (self.lock) {
		runs = self.pending.allValues;
	}
	for (NFKPluginPendingRequest *request in runs) {
		if ([request.type isEqualToString:NFKPluginMessageRun]) {
			[self sendMessage:@{ @"type": NFKPluginMessageCancel, @"id": @(request.identifier) } error:NULL];
		}
	}
}

/*! Waits until no request is in flight, up to timeout. Returns whether none is. */
- (BOOL)waitForRequestsToDrainWithin:(NSTimeInterval)timeout
{
	NSTimeInterval deadline = NFKPluginNow() + MAX(timeout, 0.0);
	while (self.pendingCount > 0) {
		if (NFKPluginNow() >= deadline) {
			return NO;
		}
		[NSThread sleepForTimeInterval:0.01];
	}
	return YES;
}

- (void)failAllPendingWithError:(NSError *)error
{
	NSArray<NFKPluginPendingRequest *> *requests = nil;
	@synchronized (self.lock) {
		requests = [self.pending.allValues sortedArrayUsingComparator:^NSComparisonResult(NFKPluginPendingRequest *left, NFKPluginPendingRequest *right) {
			return left.identifier < right.identifier ? NSOrderedAscending : NSOrderedDescending;
		}];
		[self.pending removeAllObjects];
	}
	for (NFKPluginPendingRequest *request in requests) {
		request.replyHandler(nil, @[], error);
	}
}

#pragma mark Use

/*! Counts a caller that is about to send. NO once the connection retires. */
- (BOOL)beginUse
{
	@synchronized (self.lock) {
		if (self.closed || self.retiring) {
			return NO;
		}
		self.activeUses += 1;
		return YES;
	}
}

- (void)endUse
{
	@synchronized (self.lock) {
		self.activeUses = MAX(self.activeUses - 1, 0);
		self.lastActivity = NFKPluginNow();
	}
}

/*! Retires the connection when nothing has used it for interval: no caller, no request, no live
	backend. */
- (BOOL)retireIfIdleFor:(NSTimeInterval)interval
{
	@synchronized (self.lock) {
		BOOL busy = self.closed || self.retiring || self.activeUses > 0 || self.pending.count > 0 || self.liveBackendCount > 0;
		if (busy || NFKPluginNow() - self.lastActivity < interval) {
			return NO;
		}
		self.retiring = YES;
		self.intentional = YES;
		return YES;
	}
}

/*! Takes no new callers. The requests in flight continue. */
- (void)retire
{
	@synchronized (self.lock) {
		self.retiring = YES;
		self.intentional = YES;
	}
}

/*! Retires the connection and fails its requests in flight with error. */
- (void)retireWithError:(NSError *)error
{
	@synchronized (self.lock) {
		self.retiring = YES;
		self.intentional = YES;
		self.closeError = error;
	}
	[self failAllPendingWithError:error];
}

- (void)closeChannel
{
	@synchronized (self.lock) {
		self.intentional = YES;
	}
	[self.channel close];
}

#pragma mark Backends

- (void)registerBackend:(NFKPluginBackend *)backend handle:(id)handle
{
	@synchronized (self.lock) {
		[self.backends setObject:backend forKey:NFKPluginHandleKey(handle)];
		self.liveBackendCount += 1;
	}
}

- (void)releaseBackendHandle:(id)handle
{
	BOOL sends = NO;
	@synchronized (self.lock) {
		[self.backends removeObjectForKey:NFKPluginHandleKey(handle)];
		self.liveBackendCount = MAX(self.liveBackendCount - 1, 0);
		self.lastActivity = NFKPluginNow();
		sends = !self.closed && !self.retiring;
	}
	if (sends) {
		[self sendRequest:@{ @"type": NFKPluginMessageRelease, @"backend": handle } descriptors:nil progressHandler:nil
			 replyHandler:^(NSDictionary<NSString *, id> * _Nullable reply, NSArray<NFKPluginDescriptor *> *descriptors, NSError * _Nullable error) {
		} error:NULL];
	}
}

#pragma mark Messages

// Runs on the channel's handler queue.
- (void)handleMessage:(NSDictionary<NSString *, id> *)message descriptors:(NSArray<NFKPluginDescriptor *> *)descriptors
{
	NSString *type = message[@"type"];
	id identifier = message[@"id"];
	if ([type isEqualToString:NFKPluginMessageReply]) {
		[self completeRequest:identifier reply:message descriptors:descriptors];
	} else if ([type isEqualToString:NFKPluginMessageProgress]) {
		NFKPluginPendingRequest *request = nil;
		@synchronized (self.lock) {
			request = [identifier isKindOfClass:NSNumber.class] ? self.pending[identifier] : nil;
			self.lastActivity = NFKPluginNow();
		}
		if (request.progressHandler != nil) {
			request.progressHandler(message, descriptors);
		}
	} else if ([type isEqualToString:NFKPluginMessageEvent]) {
		[self routeEvent:message];
	} else if ([type isEqualToString:NFKPluginMessageLog]) {
		NSInteger level = [message[@"level"] isKindOfClass:NSNumber.class] ? [message[@"level"] integerValue] : NFKPluginLogLevelInfo;
		NSString *text = [message[@"message"] isKindOfClass:NSString.class] ? message[@"message"] : @"";
		[self.process logMessage:text level:(NFKPluginLogLevel)MIN(MAX(level, NFKPluginLogLevelDebug), NFKPluginLogLevelError)];
	} else if ([identifier isKindOfClass:NSNumber.class]) {
		NSError *unsupported = NFKPluginMakeError(NFKPluginErrorProtocolUnsupported,
											  [NSString stringWithFormat:@"the host does not support %@", type]);
		NSDictionary *body = [NFKInferenceWireCoding JSONObjectForError:unsupported][@"error"];
		[self sendMessage:@{ @"type": NFKPluginMessageReply, @"id": identifier, @"error": body } error:NULL];
	}
}

- (void)completeRequest:(id)identifier reply:(NSDictionary<NSString *, id> *)reply descriptors:(NSArray<NFKPluginDescriptor *> *)descriptors
{
	NFKPluginPendingRequest *request = nil;
	@synchronized (self.lock) {
		if ([identifier isKindOfClass:NSNumber.class]) {
			request = self.pending[identifier];
			[self.pending removeObjectForKey:identifier];
		}
		self.lastActivity = NFKPluginNow();
	}
	if (request != nil) {
		request.replyHandler(reply, descriptors, nil);
	}
}

- (void)routeEvent:(NSDictionary<NSString *, id> *)event
{
	if ([event[@"event"] isEqual:NFKPluginEventVerified]) {
		dispatch_block_t handler = self.verificationHandler;
		if (handler != nil) {
			handler();
		}
		return;
	}
	id handle = event[@"backend"];
	if (handle == nil || handle == NSNull.null) {
		return;
	}
	NFKPluginBackend *backend = nil;
	@synchronized (self.lock) {
		backend = [self.backends objectForKey:NFKPluginHandleKey(handle)];
	}
	[backend handleEvent:event];
}

// Runs on the channel's handler queue, once.
- (void)channelDidCloseWithError:(nullable NSError *)error
{
	self.channel.messageHandler = nil;
	self.channel.closeHandler = nil;
	BOOL intentional = NO;
	NSError *closeError = nil;
	@synchronized (self.lock) {
		self.closed = YES;
		intentional = self.intentional;
		closeError = self.closeError;
	}
	if (intentional) {
		[self failAllPendingWithError:closeError ?: NFKPluginMakeError(NFKPluginErrorStopped, @"the plugin was stopped")];
		return;
	}
	NFKPluginProcess *process = self.process;
	if (process != nil) {
		[process connectionDidEnd:self error:error];
	} else {
		[self failAllPendingWithError:NFKPluginMakeError(NFKPluginErrorProcessExited, @"the plugin's process exited")];
	}
}

@end

#pragma mark - Process

@implementation NFKPluginProcess

- (instancetype)initWithManifest:(NFKPluginManifest *)manifest hostExecutableURL:(nullable NSURL *)hostExecutableURL
{
	self = [super init];
	if (self != nil) {
		_manifest = [manifest copy];
		_hostExecutableURL = [hostExecutableURL copy];
		_signaturePolicy = [NFKPluginSignaturePolicy defaultPolicy];
		_handshakeTimeout = 10;
		_verificationTimeout = 120;
		_idleExitInterval = manifest.idleExitInterval;
		_restartDrainTimeout = 120;
		_crashLimit = 3;
		_crashLimitInterval = 60;
		_memoryPressureOnset = 0.6;
		_shortestIdleExitInterval = 10;
		_pluginOptions = @{};
		_stateLock = [[NSObject alloc] init];
		_startLock = [[NSLock alloc] init];
		_admissionCondition = [[NSCondition alloc] init];
		_heldAdmissions = [NSMutableArray array];
		_crashTimes = [NSMutableArray array];
		dispatch_queue_attr_t attributes = dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INITIATED, 0);
		_eventQueue = dispatch_queue_create("com.inferkit.plugin-process.events", attributes);
		_controlQueue = dispatch_queue_create("com.inferkit.plugin-process.control", attributes);
		_idleQueue = dispatch_queue_create("com.inferkit.plugin-process.idle",
										   dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_UTILITY, 0));
	}
	return self;
}

- (void)dealloc
{
	if (_idleTimer != nil) {
		dispatch_source_cancel(_idleTimer);
	}
	NFKPluginConnection *connection = _connection;
	if (connection != nil) {
		[connection closeChannel];
		[connection.launcher stopWithCompletionHandler:^{
		}];
	}
}

#pragma mark State

- (void)enterState:(NFKPluginProcessState)state
{
	@synchronized (self.stateLock) {
		if (self.state == state) {
			return;
		}
		self.state = state;
		NFKPluginProcessStateHandler handler = self.stateHandler;
		if (handler != nil) {
			dispatch_async(self.eventQueue, ^{
				handler(self, state);
			});
		}
	}
}

- (void)logMessage:(NSString *)message level:(NFKPluginLogLevel)level
{
	NFKPluginProcessLogHandler handler = self.logHandler;
	if (handler != nil) {
		dispatch_async(self.eventQueue, ^{
			handler(self, level, message);
		});
	}
}

#pragma mark Starting

- (BOOL)startWithError:(NSError **)error
{
	return [self performWithReservation:nil clearingFailure:YES block:^BOOL(NFKPluginConnection *connection, NSError **blockError) {
		return YES;
	} error:error];
}

- (nullable id)reserveAdmission
{
	[self.admissionCondition lock];
	NFKPluginAdmission *admission = nil;
	if (self.holding) {
		admission = [[NFKPluginAdmission alloc] init];
		[self.heldAdmissions addObject:admission];
	}
	[self.admissionCondition unlock];
	return admission;
}

- (void)abandonReservation:(nullable id)reservation
{
	NFKPluginAdmission *admission = reservation;
	if (admission == nil) {
		return;
	}
	[self.admissionCondition lock];
	admission.finished = YES;
	[self.admissionCondition broadcast];
	[self.admissionCondition unlock];
}

- (BOOL)performWithReservation:(nullable id)reservation
						 block:(BOOL (^)(NFKPluginConnection *connection, NSError **error))block
						 error:(NSError **)error
{
	return [self performWithReservation:reservation clearingFailure:NO block:block error:error];
}

- (BOOL)performWithReservation:(nullable id)reservation
			   clearingFailure:(BOOL)clearingFailure
						 block:(BOOL (^)(NFKPluginConnection *connection, NSError **error))block
						 error:(NSError **)error
{
	NFKPluginAdmission *admission = reservation ?: [self reserveAdmission];
	NSError *failure = nil;
	if (admission != nil) {
		[self.admissionCondition lock];
		while (!admission.admitted) {
			[self.admissionCondition wait];
		}
		failure = admission.error;
		[self.admissionCondition unlock];
	}
	BOOL performed = NO;
	if (failure == nil) {
		performed = [self performWithConnectionClearingFailure:clearingFailure block:block error:&failure];
	}
	if (admission != nil) {
		[self.admissionCondition lock];
		admission.finished = YES;
		[self.admissionCondition broadcast];
		[self.admissionCondition unlock];
	}
	if (!performed && error != NULL) {
		*error = failure;
	}
	return performed;
}

/*! Runs block with a live connection. A connection that an idle exit retires between the start and the
	use is replaced once. */
- (BOOL)performWithConnectionClearingFailure:(BOOL)clearingFailure
									   block:(BOOL (^)(NFKPluginConnection *connection, NSError **error))block
									   error:(NSError **)error
{
	for (NSUInteger attempt = 0; attempt < 2; attempt++) {
		NFKPluginConnection *connection = [self connectionStartingIfNeededClearingFailure:clearingFailure error:error];
		if (connection == nil) {
			return NO;
		}
		if (![connection beginUse]) {
			continue;
		}
		BOOL performed = block(connection, error);
		[connection endUse];
		return performed;
	}
	return NFKPluginSetError(error, NFKPluginErrorStopped, @"the plugin's process exited before the request was sent");
}

- (nullable NFKPluginConnection *)connectionStartingIfNeededClearingFailure:(BOOL)clearingFailure error:(NSError **)error
{
	[self.startLock lock];
	NFKPluginConnection *connection = self.connection;
	if (connection != nil && !connection.closed && !connection.retiring) {
		[self.startLock unlock];
		return connection;
	}
	if (self.state == NFKPluginProcessStateFailed && !clearingFailure) {
		if (error != NULL) {
			*error = self.failureError ?: NFKPluginMakeError(NFKPluginErrorLaunchFailed, @"the plugin failed");
		}
		[self.startLock unlock];
		return nil;
	}
	if (clearingFailure) {
		[self.crashTimes removeAllObjects];
	}
	connection = [self launchRestarting:NO error:error];
	[self.startLock unlock];
	return connection;
}

- (nullable NFKPluginConnection *)readyConnection
{
	NFKPluginConnection *connection = self.connection;
	if (connection == nil || connection.closed || connection.retiring || self.state != NFKPluginProcessStateReady) {
		return nil;
	}
	[self.admissionCondition lock];
	BOOL holding = self.holding;
	[self.admissionCondition unlock];
	return holding ? nil : connection;
}

// Runs under the start lock.
- (nullable NFKPluginConnection *)launchRestarting:(BOOL)restarting error:(NSError **)error
{
	if (!restarting) {
		[self enterState:NFKPluginProcessStateStarting];
	}
	NSError *failure = nil;
	NFKPluginConnection *connection = [self verifyCodeWithError:&failure] ? [self connectWithError:&failure] : nil;
	if (connection == nil) {
		self.failureError = failure;
		[self enterState:NFKPluginProcessStateFailed];
		if (error != NULL) {
			*error = failure;
		}
		return nil;
	}
	self.failureError = nil;
	self.connection = connection;
	[self enterState:NFKPluginProcessStateReady];
	[self scheduleIdleTimer];
	return connection;
}

- (BOOL)isExecutablePlugin
{
	return [self.manifest.entryKind isEqualToString:NFKPluginEntryKindExecutable];
}

/*! The policy for the plugin host: the plugin's policy without its team or requirement, which name the
	plugin's signer and not the app's. */
- (NFKPluginSignaturePolicy *)hostSignaturePolicy
{
	NFKPluginSignaturePolicy *policy = [self.signaturePolicy copy];
	policy.teamIdentifier = nil;
	policy.requirementString = nil;
	return policy;
}

/*! Checks the code the launch runs: the plugin's bundle under the policy, and the plugin host under the
	host policy. */
- (BOOL)verifyCodeWithError:(NSError **)error
{
	NSURL *bundle = self.manifest.bundleURL;
	if (bundle != nil && ![self.signaturePolicy validateCodeAtURL:bundle error:error]) {
		return NO;
	}
	NSURL *host = self.hostExecutableURL;
	if (!self.isExecutablePlugin && host != nil && ![self.hostSignaturePolicy validateCodeAtURL:host error:error]) {
		return NO;
	}
	return YES;
}

/*! Launches a process, takes its channel, checks its peer, and greets it. */
- (nullable NFKPluginConnection *)connectWithError:(NSError **)error
{
	int hostEnd = -1;
	int childEnd = -1;
	if (![NFKPluginChannel createSocketPair:&hostEnd second:&childEnd error:error]) {
		return nil;
	}
	NFKPluginLauncherFactory factory = self.launcherFactory;
	id<NFKPluginLauncher> launcher = factory != nil ? factory(self) : [[NFKPluginSubprocessLauncher alloc] init];
	NSError *launchError = nil;
	BOOL launched = [launcher launchPluginProcess:self channelDescriptor:childEnd error:&launchError];
	close(childEnd);
	if ([launcher respondsToSelector:@selector(subprocess)]) {
		self.subprocess = launcher.subprocess;
	}
	if (!launched) {
		close(hostEnd);
		[self recordTerminationOfLauncher:launcher];
		if (error != NULL) {
			*error = [launchError.domain isEqualToString:NFKPluginErrorDomain] ? launchError
				: NFKPluginWrappedError(NFKPluginErrorLaunchFailed, @"the plugin's process could not be launched", launchError);
		}
		return nil;
	}
	self.processIdentifier = launcher.processIdentifier;
	NFKPluginChannel *channel = [NFKPluginChannel channelByAcceptingHandOffOnDescriptor:hostEnd timeout:self.handshakeTimeout error:error];
	if (channel == nil || ![self validatePeerOfChannel:channel launcher:launcher error:error]) {
		[channel close];
		[self stopLauncher:launcher];
		return nil;
	}
	NFKPluginConnection *connection = [[NFKPluginConnection alloc] initWithChannel:channel generation:self.generation + 1
																		  launcher:launcher process:self];
	[channel resume];
	if (![self greetConnection:connection error:error]) {
		[connection retire];
		[connection closeChannel];
		[self stopLauncher:launcher];
		return nil;
	}
	self.generation = connection.generation;
	return connection;
}

- (BOOL)validatePeerOfChannel:(NFKPluginChannel *)channel launcher:(id<NFKPluginLauncher>)launcher error:(NSError **)error
{
	audit_token_t token;
	if (![channel getPeerAuditToken:&token error:error]) {
		return NO;
	}
	if ([launcher respondsToSelector:@selector(validatePeerWithAuditToken:process:error:)]) {
		return [launcher validatePeerWithAuditToken:token process:self error:error];
	}
	pid_t expected = launcher.processIdentifier;
	pid_t peer = channel.peerProcessIdentifier;
	if (expected != 0 && peer != expected) {
		return NFKPluginSetError(error, NFKPluginErrorHandshakeFailed,
								 [NSString stringWithFormat:@"the channel's peer is process %d, not the launched process %d", peer, expected]);
	}
	NFKPluginSignaturePolicy *policy = self.isExecutablePlugin ? self.signaturePolicy : self.hostSignaturePolicy;
	return [policy validateProcessWithAuditToken:token error:error];
}

/*! The hello exchange and the describe check. */
- (BOOL)greetConnection:(NFKPluginConnection *)connection error:(NSError **)error
{
	NFKPluginManifest *manifest = self.manifest;
	NSDictionary *hello = @{
		@"type": NFKPluginMessageHello,
		@"protocol": @{ @"major": @(NFKPluginProtocolMajorVersion), @"minor": @(NFKPluginProtocolMinorVersion) },
		@"inferkitAPI": NFKInferKit.APIVersion,
		@"pluginABI": [NSString stringWithFormat:@"%d.%d", NFK_PLUGIN_ABI_VERSION_MAJOR, NFK_PLUGIN_ABI_VERSION_MINOR],
		@"capabilities": @[],
		@"options": self.pluginOptions ?: @{},
		@"limits": @{ @"inlineBytes": @(NFKPluginDefaultInlineByteLimit) },
	};
	NSError *replyError = nil;
	id reply = [self sendHello:hello onConnection:connection error:&replyError];
	if (reply == nil) {
		if (error != NULL) {
			*error = NFKPluginWrappedError(NFKPluginErrorHandshakeFailed, @"the plugin did not complete the hello exchange", replyError);
		}
		return NO;
	}
	NSDictionary *result = [reply isKindOfClass:NSDictionary.class] ? reply : @{};
	NSDictionary *protocol = [result[@"protocol"] isKindOfClass:NSDictionary.class] ? result[@"protocol"] : @{};
	NSNumber *major = [protocol[@"major"] isKindOfClass:NSNumber.class] ? protocol[@"major"] : nil;
	if (major.integerValue != NFKPluginProtocolMajorVersion) {
		return NFKPluginSetError(error, NFKPluginErrorProtocolUnsupported,
								 [NSString stringWithFormat:@"the plugin speaks protocol %@.%@ and this host speaks %ld.%ld",
								  major ?: @"?", protocol[@"minor"] ?: @"?", (long)NFKPluginProtocolMajorVersion, (long)NFKPluginProtocolMinorVersion]);
	}
	NSDictionary *plugin = [result[@"plugin"] isKindOfClass:NSDictionary.class] ? result[@"plugin"] : @{};
	if (![plugin[@"id"] isEqual:manifest.identifier]) {
		return NFKPluginSetError(error, NFKPluginErrorHandshakeFailed,
								 [NSString stringWithFormat:@"the process is plugin %@, not %@", plugin[@"id"] ?: @"(unnamed)", manifest.identifier]);
	}
	if (![plugin[@"version"] isEqual:manifest.version]) {
		return NFKPluginSetError(error, NFKPluginErrorHandshakeFailed,
								 [NSString stringWithFormat:@"the process runs version %@ of %@ and the manifest names %@",
								  plugin[@"version"] ?: @"(none)", manifest.identifier, manifest.version]);
	}
	NSDictionary *limits = [result[@"limits"] isKindOfClass:NSDictionary.class] ? result[@"limits"] : @{};
	NSInteger inlineBytes = [limits[@"inlineBytes"] isKindOfClass:NSNumber.class] ? [limits[@"inlineBytes"] integerValue] : 0;
	if (inlineBytes > 0 && (NSUInteger)inlineBytes < connection.inlineByteLimit) {
		connection.inlineByteLimit = (NSUInteger)inlineBytes;
	}
	connection.helloReply = result;
	return [self checkDescriptionOfConnection:connection error:error];
}

/*! Sends hello and waits for its reply. The plugin host verifies the bundle before it reads hello, so for
	an "objc" or "c-abi" plugin the wait is verificationTimeout until the host reports the verification,
	then handshakeTimeout. An executable plugin verifies nothing and has handshakeTimeout alone. */
- (nullable id)sendHello:(NSDictionary<NSString *, id> *)hello onConnection:(NFKPluginConnection *)connection error:(NSError **)error
{
	dispatch_semaphore_t verified = dispatch_semaphore_create(0);
	dispatch_semaphore_t answered = dispatch_semaphore_create(0);
	__block id result = nil;
	__block NSError *failure = nil;
	connection.verificationHandler = ^{
		dispatch_semaphore_signal(verified);
	};
	NSInteger identifier = [connection sendRequest:hello descriptors:nil progressHandler:nil
									  replyHandler:^(NSDictionary<NSString *, id> * _Nullable reply, NSArray<NFKPluginDescriptor *> *descriptors, NSError * _Nullable replyError) {
		if (reply == nil) {
			failure = replyError;
		} else {
			failure = NFKPluginErrorFromReply(reply);
			result = failure == nil ? (reply[@"result"] ?: NSNull.null) : nil;
		}
		dispatch_semaphore_signal(answered);
		dispatch_semaphore_signal(verified);
	} error:error];
	if (identifier == 0) {
		connection.verificationHandler = nil;
		return nil;
	}
	BOOL timedOut = NO;
	NSString *stage = @"hello";
	NSTimeInterval limit = self.handshakeTimeout;
	if (!self.isExecutablePlugin && dispatch_semaphore_wait(verified, NFKPluginDeadline(self.verificationTimeout)) != 0) {
		timedOut = YES;
		stage = @"the verification of its bundle";
		limit = self.verificationTimeout;
	}
	if (!timedOut && dispatch_semaphore_wait(answered, NFKPluginDeadline(self.handshakeTimeout)) != 0) {
		timedOut = YES;
	}
	connection.verificationHandler = nil;
	if (timedOut) {
		[connection abandonRequest:identifier];
		NFKPluginSetError(error, NFKPluginErrorProtocolViolation,
						  [NSString stringWithFormat:@"the plugin did not finish %@ within %g seconds", stage, limit]);
		return nil;
	}
	if (result == nil && error != NULL) {
		*error = failure ?: NFKPluginMakeError(NFKPluginErrorStopped, @"the plugin's process is not running");
	}
	return result;
}

/*! Requires every model the manifest lists in the plugin's description, and logs the ports and options
	that differ. */
- (BOOL)checkDescriptionOfConnection:(NFKPluginConnection *)connection error:(NSError **)error
{
	NSError *replyError = nil;
	id reply = [connection sendRequestAndWait:@{ @"type": NFKPluginMessageDescribe } timeout:self.handshakeTimeout error:&replyError];
	if (reply == nil) {
		if (error != NULL) {
			*error = NFKPluginWrappedError(NFKPluginErrorHandshakeFailed, @"the plugin did not describe its models", replyError);
		}
		return NO;
	}
	NSDictionary *description = [reply isKindOfClass:NSDictionary.class] ? reply : @{};
	NSMutableDictionary<NSString *, NSDictionary *> *described = [NSMutableDictionary dictionary];
	for (id entry in [description[@"models"] isKindOfClass:NSArray.class] ? description[@"models"] : @[]) {
		if ([entry isKindOfClass:NSDictionary.class] && [entry[@"id"] isKindOfClass:NSString.class]) {
			described[entry[@"id"]] = entry;
		} else if ([entry isKindOfClass:NSString.class]) {
			described[entry] = @{};
		}
	}
	for (NFKModelDescriptor *model in self.manifest.models) {
		NSDictionary *entry = described[model.identifier];
		if (entry == nil) {
			return NFKPluginSetError(error, NFKPluginErrorUnknownModel,
									 [NSString stringWithFormat:@"plugin %@ does not describe model %@", self.manifest.identifier, model.identifier]);
		}
		[self logDifferencesOfModel:model fromDescription:entry];
	}
	return YES;
}

- (void)logDifferencesOfModel:(NFKModelDescriptor *)model fromDescription:(NSDictionary *)entry
{
	if (entry.count == 0) {
		return;
	}
	NSDictionary<NSString *, NSArray *> *declared = @{
		@"inputs": [model.inputs valueForKey:@"key"],
		@"outputs": [model.outputs valueForKey:@"key"],
		@"parameters": [model.parameters valueForKey:@"key"],
	};
	for (NSString *field in @[ @"inputs", @"outputs", @"parameters" ]) {
		if (entry[field] == nil) {
			continue;
		}
		NSSet<NSString *> *manifestKeys = [NSSet setWithArray:declared[field]];
		NSSet<NSString *> *describedKeys = NFKPluginDescribedKeys(entry[field]);
		if (![manifestKeys isEqualToSet:describedKeys]) {
			NSString *message = [NSString stringWithFormat:@"model %@ describes %@ %@ and its manifest lists %@", model.identifier, field,
								 [describedKeys.allObjects sortedArrayUsingSelector:@selector(compare:)],
								 [manifestKeys.allObjects sortedArrayUsingSelector:@selector(compare:)]];
			[self logMessage:message level:NFKPluginLogLevelWarning];
		}
	}
}

#pragma mark Ending

/*! Stops a launcher and waits for its exit, up to NFKPluginLauncherStopTimeout, then records how the
	process ended. */
- (void)stopLauncher:(id<NFKPluginLauncher>)launcher
{
	dispatch_semaphore_t stopped = dispatch_semaphore_create(0);
	[launcher stopWithCompletionHandler:^{
		dispatch_semaphore_signal(stopped);
	}];
	dispatch_semaphore_wait(stopped, dispatch_time(DISPATCH_TIME_NOW, (int64_t)(NFKPluginLauncherStopTimeout * NSEC_PER_SEC)));
	[self recordTerminationOfLauncher:launcher];
}

- (void)recordTerminationOfLauncher:(id<NFKPluginLauncher>)launcher
{
	self.terminationReason = launcher.terminationReason;
	self.terminationStatus = launcher.terminationStatus;
	self.processIdentifier = 0;
}

/*! NFKPluginErrorProcessExited with the exit status or signal and the end of standard error. */
- (NSError *)exitErrorOfLauncher:(id<NFKPluginLauncher>)launcher
{
	NSString *reason = nil;
	switch (launcher.terminationReason) {
		case NFKSubprocessTerminationReasonExit:
			reason = [NSString stringWithFormat:@"the plugin's process exited with status %d", launcher.terminationStatus];
			break;
		case NFKSubprocessTerminationReasonSignal: {
			const char *name = launcher.terminationStatus > 0 && launcher.terminationStatus < NSIG ? sys_signame[launcher.terminationStatus] : "unknown";
			reason = [NSString stringWithFormat:@"the plugin's process was ended by signal %d (SIG%s)", launcher.terminationStatus,
					  [[NSString stringWithUTF8String:name] uppercaseString].UTF8String];
			break;
		}
		default:
			reason = @"the plugin's process closed its channel";
			break;
	}
	NSMutableDictionary *userInfo = [NSMutableDictionary dictionaryWithObject:reason forKey:NSLocalizedDescriptionKey];
	NSString *standardError = [launcher.recentStandardError stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceAndNewlineCharacterSet];
	if (standardError.length > 0) {
		NSUInteger start = standardError.length > NFKPluginStandardErrorTail ? standardError.length - NFKPluginStandardErrorTail : 0;
		userInfo[NSLocalizedFailureReasonErrorKey] = [standardError substringFromIndex:start];
	}
	return [NSError errorWithDomain:NFKPluginErrorDomain code:NFKPluginErrorProcessExited userInfo:userInfo];
}

// Runs on the ended channel's handler queue, for a channel the supervisor did not close.
- (void)connectionDidEnd:(NFKPluginConnection *)connection error:(nullable NSError *)error
{
	[self.startLock lock];
	BOOL current = self.connection == connection;
	if (current) {
		self.connection = nil;
		[self cancelIdleTimer];
	}
	[self stopLauncher:connection.launcher];
	NSError *exited = [self exitErrorOfLauncher:connection.launcher];
	if (current && self.state != NFKPluginProcessStateRestarting) {
		[self recordCrashWithError:exited];
	}
	[self.startLock unlock];
	[connection failAllPendingWithError:exited];
}

// Runs under the start lock.
- (void)recordCrashWithError:(NSError *)exited
{
	NSTimeInterval now = NFKPluginNow();
	[self.crashTimes addObject:@(now)];
	NSTimeInterval window = self.crashLimitInterval;
	NSIndexSet *old = [self.crashTimes indexesOfObjectsPassingTest:^BOOL(NSNumber *time, NSUInteger index, BOOL *stop) {
		return now - time.doubleValue > window;
	}];
	[self.crashTimes removeObjectsAtIndexes:old];
	NSUInteger limit = self.crashLimit;
	if (limit > 0 && self.crashTimes.count >= limit) {
		NSString *reason = [NSString stringWithFormat:@"the plugin's process ended %lu times within %g seconds",
							(unsigned long)self.crashTimes.count, window];
		self.failureError = NFKPluginWrappedError(NFKPluginErrorProcessExited, reason, exited);
		[self enterState:NFKPluginProcessStateFailed];
		return;
	}
	[self enterState:NFKPluginProcessStateNotRunning];
}

/*! Retires a connection, fails its requests with error, asks the plugin to shut down, closes the
	channel, and stops the process. */
- (void)shutDownConnection:(NFKPluginConnection *)connection error:(NSError *)error
{
	[connection retireWithError:error];
	if (!connection.closed) {
		[connection sendRequestAndWait:@{ @"type": NFKPluginMessageShutdown } timeout:NFKPluginShutdownReplyTimeout error:NULL];
	}
	[connection closeChannel];
	[self stopLauncher:connection.launcher];
}

- (void)stopWithCompletionHandler:(nullable void (^)(void))completionHandler
{
	dispatch_async(self.controlQueue, ^{
		[self.startLock lock];
		NFKPluginConnection *connection = self.connection;
		self.connection = nil;
		[self cancelIdleTimer];
		if (connection != nil) {
			[self shutDownConnection:connection error:NFKPluginMakeError(NFKPluginErrorStopped, @"the plugin was stopped")];
		}
		if (self.state != NFKPluginProcessStateFailed) {
			[self enterState:NFKPluginProcessStateNotRunning];
		}
		[self.startLock unlock];
		if (completionHandler != nil) {
			dispatch_async(self.eventQueue, completionHandler);
		}
	});
}

#pragma mark Restarting

- (void)restartWithManifest:(NFKPluginManifest *)manifest completionHandler:(nullable void (^)(NSError * _Nullable))completionHandler
{
	// Holding starts before this returns, so a request made after the call waits for the new process.
	[self.admissionCondition lock];
	self.holding = YES;
	self.pendingRestarts += 1;
	[self.admissionCondition unlock];
	NFKPluginManifest *next = [manifest copy];
	dispatch_async(self.controlQueue, ^{
		NSError *failure = [self replaceProcessWithManifest:next];
		[self releaseHeldAdmissionsWithError:failure];
		if (completionHandler != nil) {
			dispatch_async(self.eventQueue, ^{
				completionHandler(failure);
			});
		}
	});
}

/*! Drains the runs in flight, stops the old process, and starts the new version. Returns the error that
	failed the start, or nil. */
- (nullable NSError *)replaceProcessWithManifest:(NFKPluginManifest *)manifest
{
	[self enterState:NFKPluginProcessStateRestarting];
	NFKPluginConnection *old = self.connection;
	if (old != nil) {
		[old retire];
		if (![old waitForRequestsToDrainWithin:self.restartDrainTimeout]) {
			[old cancelRuns];
			[old waitForRequestsToDrainWithin:NFKPluginCancelGrace];
		}
	}
	[self.startLock lock];
	if (old != nil && self.connection == old) {
		self.connection = nil;
		[self cancelIdleTimer];
	}
	if (old != nil) {
		[self shutDownConnection:old error:NFKPluginMakeError(NFKPluginErrorStopped, @"the plugin restarted before the run finished")];
	}
	if (self.idleExitInterval == self.manifest.idleExitInterval) {
		self.idleExitInterval = manifest.idleExitInterval;
	}
	self.manifest = manifest;
	[self.crashTimes removeAllObjects];
	NSError *failure = nil;
	NFKPluginConnection *fresh = [self launchRestarting:YES error:&failure];
	[self.startLock unlock];
	return fresh != nil ? nil : failure;
}

/*! Admits the held requests one at a time, in order, each after the one before it has sent. Admits only
	when no other restart is waiting. */
- (void)releaseHeldAdmissionsWithError:(nullable NSError *)failure
{
	[self.admissionCondition lock];
	self.pendingRestarts -= 1;
	if (self.pendingRestarts == 0) {
		while (self.heldAdmissions.count > 0) {
			NFKPluginAdmission *admission = self.heldAdmissions.firstObject;
			[self.heldAdmissions removeObjectAtIndex:0];
			admission.error = failure;
			admission.admitted = YES;
			[self.admissionCondition broadcast];
			NSDate *deadline = [NSDate dateWithTimeIntervalSinceNow:MAX(self.restartDrainTimeout, self.handshakeTimeout)];
			while (!admission.finished && [self.admissionCondition waitUntilDate:deadline]) {
			}
		}
		self.holding = NO;
	}
	[self.admissionCondition unlock];
}

#pragma mark Idle exit

- (NSTimeInterval)effectiveIdleExitInterval
{
	NSTimeInterval idle = self.idleExitInterval;
	if (!(idle > 0)) {
		return INFINITY;
	}
	double pressure = [self memoryPressure];
	double onset = self.memoryPressureOnset;
	if (pressure < onset) {
		return idle;
	}
	NSTimeInterval base = isfinite(idle) ? idle : NFKPluginPressureBaseInterval;
	NSTimeInterval shortest = MIN(MAX(self.shortestIdleExitInterval, 0.0), base);
	double fraction = onset < 1 ? MIN(MAX((pressure - onset) / (1 - onset), 0.0), 1.0) : 1.0;
	return MIN(idle, base - (base - shortest) * fraction);
}

- (double)memoryPressure
{
	double (^provider)(void) = self.memoryPressureProvider;
	double reading = 0;
	if (provider != nil) {
		reading = provider();
	} else {
		NFKServedHostStatus *host = nil;
		@synchronized (self.stateLock) {
			if (self.hostStatus == nil) {
				self.hostStatus = [[NFKServedHostStatus alloc] init];
			}
			host = self.hostStatus;
		}
		reading = [host memoryPressureFraction];
	}
	return isfinite(reading) ? MIN(MAX(reading, 0.0), 1.0) : 0.0;
}

// Runs under the start lock.
- (void)scheduleIdleTimer
{
	[self cancelIdleTimer];
	NSTimeInterval idle = self.idleExitInterval;
	if (!(idle > 0)) {
		return;
	}
	NSTimeInterval shortest = MIN(isfinite(idle) ? idle : NFKPluginPressureBaseInterval, MAX(self.shortestIdleExitInterval, 0.0));
	NSTimeInterval period = MIN(MAX(shortest / 5.0, 0.05), 1.0);
	dispatch_source_t timer = dispatch_source_create(DISPATCH_SOURCE_TYPE_TIMER, 0, 0, self.idleQueue);
	dispatch_source_set_timer(timer, dispatch_time(DISPATCH_TIME_NOW, (int64_t)(period * NSEC_PER_SEC)),
							  (uint64_t)(period * NSEC_PER_SEC), (uint64_t)(period * 0.1 * NSEC_PER_SEC));
	__weak NFKPluginProcess *weakSelf = self;
	dispatch_source_set_event_handler(timer, ^{
		[weakSelf exitIfIdle];
	});
	self.idleTimer = timer;
	dispatch_resume(timer);
}

- (void)cancelIdleTimer
{
	if (self.idleTimer != nil) {
		dispatch_source_cancel(self.idleTimer);
		self.idleTimer = nil;
	}
}

// Runs on the idle queue.
- (void)exitIfIdle
{
	NFKPluginConnection *connection = self.connection;
	if (connection == nil || self.state != NFKPluginProcessStateReady) {
		return;
	}
	NSTimeInterval interval = [self effectiveIdleExitInterval];
	if (!isfinite(interval) || ![connection retireIfIdleFor:interval]) {
		return;
	}
	[self.startLock lock];
	// A request that found the connection retiring may have launched its successor in the meantime.
	BOOL current = self.connection == connection;
	if (current) {
		self.connection = nil;
		[self cancelIdleTimer];
	}
	[self shutDownConnection:connection error:NFKPluginMakeError(NFKPluginErrorStopped, @"the plugin's process exited while idle")];
	if (current && self.state == NFKPluginProcessStateReady) {
		[self enterState:NFKPluginProcessStateNotRunning];
	}
	[self.startLock unlock];
}

#pragma mark Backends and tools

- (nullable NFKPluginBackend *)backendForModelIdentifier:(NSString *)modelIdentifier
												 options:(NSDictionary<NSString *, id> *)options
												   error:(NSError **)error
{
	NFKPluginManifest *manifest = self.manifest;
	for (NFKModelDescriptor *descriptor in manifest.models) {
		if ([descriptor.identifier isEqualToString:modelIdentifier]) {
			return [[NFKPluginBackend alloc] initWithProcess:self descriptor:descriptor options:options ?: @{}];
		}
	}
	NFKPluginSetError(error, NFKPluginErrorUnknownModel,
					  [NSString stringWithFormat:@"plugin %@ has no model %@", manifest.identifier, modelIdentifier]);
	return nil;
}

- (NSArray<NFKInferenceTool *> *)tools
{
	NSMutableArray<NFKInferenceTool *> *tools = [NSMutableArray array];
	for (NSDictionary<NSString *, id> *declaration in self.manifest.tools) {
		NFKPluginToolProxy *tool = [[NFKPluginToolProxy alloc] initWithProcess:self declaration:declaration];
		if (tool != nil) {
			[tools addObject:tool];
		}
	}
	return tools;
}

@end

#endif
