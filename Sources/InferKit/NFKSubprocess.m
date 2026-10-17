//
//  NFKSubprocess.m
//  InferKit
//

#import "NFKSubprocess.h"
#import "NFKSubprocessPrivate.h"
#import "NFKAgentCLIPrivate.h"

NSString * const NFKSubprocessErrorDomain = @"NFKSubprocessErrorDomain";

#if TARGET_OS_OSX

#import <spawn.h>
#import <pthread/spawn.h>
#import <signal.h>
#import <sys/wait.h>
#import <fcntl.h>
#import <poll.h>
#import <unistd.h>
#import <errno.h>
#import <string.h>

/*! How long output may trail the process's exit before the run ends without it. A grandchild that
	inherited the pipes can hold them open after the child exits. */
static const NSTimeInterval NFKSubprocessDrainGrace = 0.5;

/*! The first and the longest wait between reap attempts after the exit event finds no exited child. */
static const NSTimeInterval NFKSubprocessReapRetryFirst = 0.001;
static const NSTimeInterval NFKSubprocessReapRetryLongest = 0.1;

/*! How often a blocked write wakes to see whether standard input is closing, in milliseconds. */
static const int NFKSubprocessWritePollMilliseconds = 100;

static const NSTimeInterval NFKSubprocessDefaultWriteTimeout = 30;

static void *NFKSubprocessQueueKey = &NFKSubprocessQueueKey;

/*! One launch: the process, its pipes, and the output not yet framed into a line. Touched only on
	the event queue. */
@interface NFKSubprocessRun : NSObject
@property (nonatomic, assign) pid_t processIdentifier;
@property (nonatomic, assign) int outputDescriptor;
@property (nonatomic, assign) int errorDescriptor;
@property (nonatomic, strong, nullable) dispatch_source_t outputSource;
@property (nonatomic, strong, nullable) dispatch_source_t errorSource;
@property (nonatomic, strong, nullable) dispatch_source_t exitSource;
@property (nonatomic, strong) NSMutableData *pendingOutput;
@property (nonatomic, strong) NSMutableData *pendingError;
@property (nonatomic, assign) BOOL discardingOutput;
@property (nonatomic, assign) BOOL discardingError;
@property (nonatomic, assign) BOOL exited;
@property (nonatomic, assign) BOOL statusKnown;
@property (nonatomic, assign) int waitStatus;
@property (nonatomic, assign) BOOL stopping;
@property (nonatomic, assign) BOOL finished;
@property (nonatomic, strong) NSDate *launchDate;
@end

@implementation NFKSubprocessRun

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_outputDescriptor = -1;
		_errorDescriptor = -1;
		_pendingOutput = [NSMutableData data];
		_pendingError = [NSMutableData data];
		_launchDate = [NSDate date];
	}
	return self;
}

@end

@interface NFKSubprocess ()
@property (atomic, readwrite) NFKSubprocessState state;
@property (atomic, readwrite) pid_t processIdentifier;
@property (atomic, readwrite) NFKSubprocessTerminationReason terminationReason;
@property (atomic, readwrite) int terminationStatus;
@property (atomic, readwrite, copy, nullable) NSError *launchError;
@property (atomic, readwrite) NSUInteger restartCount;
@property (atomic, readwrite, getter=isRestartPending) BOOL restartPending;
@property (nonatomic, strong) dispatch_queue_t eventQueue;
@property (nonatomic, strong, nullable) NFKSubprocessRun *run;
@property (nonatomic, assign) NSUInteger consecutiveCrashes;
@property (nonatomic, assign) NSUInteger restartGeneration;
@property (nonatomic, strong) NSMutableArray<void (^)(void)> *stopCompletionHandlers;
@property (nonatomic, strong) NSMutableData *standardErrorBuffer;
@property (nonatomic, strong) NSLock *inputLock;
@property (atomic, assign) int inputDescriptor;
@property (atomic, assign) BOOL inputClosing;
/*! Child number → this subprocess's own duplicate, or the negated errno of a duplication that failed. */
@property (nonatomic, strong) NSDictionary<NSNumber *, NSNumber *> *ownedInheritedDescriptors;
@end

@implementation NFKSubprocess

- (instancetype)initWithCommand:(NSString *)command arguments:(NSArray<NSString *> *)arguments
{
	self = [super init];
	if (self != nil) {
		_command = [command copy];
		_arguments = [arguments copy];
		_environment = @{};
		_inheritsEnvironment = YES;
		_standardErrorCapacity = 65536;
		_maximumLineLength = 256u * 1024u * 1024u;
		_writeTimeout = NFKSubprocessDefaultWriteTimeout;
		_stopInputTimeout = 2;
		_stopTerminateTimeout = 3;
		_restartPolicy = NFKSubprocessRestartPolicyNever;
		_maximumRestarts = 5;
		_restartDelay = 0.5;
		_maximumRestartDelay = 30;
		_restartResetInterval = 60;
		_inputDescriptor = -1;
		_stopCompletionHandlers = [NSMutableArray array];
		_standardErrorBuffer = [NSMutableData data];
		_inputLock = [[NSLock alloc] init];
		dispatch_queue_attr_t attributes = dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INITIATED, 0);
		_eventQueue = dispatch_queue_create("com.inferkit.subprocess", attributes);
		dispatch_queue_set_specific(_eventQueue, NFKSubprocessQueueKey, (__bridge void *)self, NULL);
	}
	return self;
}

- (NSString *)description
{
	return [NSString stringWithFormat:@"<%@ %@ pid %d state %ld>", self.class, self.command, self.processIdentifier, (long)self.state];
}

- (void)dealloc
{
	[self closeOwnedInheritedDescriptors];
}

#pragma mark Inherited descriptors

- (nullable NSDictionary<NSNumber *, NSNumber *> *)ownerInheritedDescriptors
{
	@synchronized (self) {
		return self.ownedInheritedDescriptors;
	}
}

- (void)setOwnerInheritedDescriptors:(nullable NSDictionary<NSNumber *, NSNumber *> *)descriptors
{
	NSMutableDictionary<NSNumber *, NSNumber *> *owned = descriptors != nil ? [NSMutableDictionary dictionaryWithCapacity:descriptors.count] : nil;
	[descriptors enumerateKeysAndObjectsUsingBlock:^(NSNumber *target, NSNumber *source, BOOL *stop) {
		int duplicate = fcntl(source.intValue, F_DUPFD_CLOEXEC, 0);
		owned[target] = duplicate >= 0 ? @(duplicate) : @(-errno);
	}];
	@synchronized (self) {
		[self closeOwnedInheritedDescriptors];
		self.ownedInheritedDescriptors = owned;
	}
}

// Runs under the lock on self, or in dealloc.
- (void)closeOwnedInheritedDescriptors
{
	for (NSNumber *duplicate in _ownedInheritedDescriptors.allValues) {
		if (duplicate.intValue >= 0) {
			close(duplicate.intValue);
		}
	}
	_ownedInheritedDescriptors = nil;
}

#pragma mark Starting

- (BOOL)startWithError:(NSError **)error
{
	__block BOOL launched = NO;
	__block NSError *failure = nil;
	[self performOnEventQueue:^{
		NFKSubprocessState state = self.state;
		if (state == NFKSubprocessStateStarting || state == NFKSubprocessStateRunning) {
			launched = YES;
			return;
		}
		[self cancelScheduledRestart];
		self.consecutiveCrashes = 0;
		NSError *launchFailure = nil;
		launched = [self launchWithError:&launchFailure];
		failure = launchFailure;
	}];
	if (!launched && error != NULL) {
		*error = failure;
	}
	return launched;
}

- (BOOL)isOnEventQueue
{
	return dispatch_get_specific(NFKSubprocessQueueKey) == (__bridge void *)self;
}

- (void)performOnEventQueue:(dispatch_block_t)block
{
	if (self.isOnEventQueue) {
		block();
	} else {
		dispatch_sync(self.eventQueue, block);
	}
}

// Runs on the event queue.
- (BOOL)launchWithError:(NSError **)error
{
	[self enterState:NFKSubprocessStateStarting];
	NSMutableDictionary<NSString *, NSString *> *environment = self.childEnvironment;
	NSArray<NSString *> *searched = @[];
	NSString *path = [self executablePathInEnvironment:environment searched:&searched];
	if (path == nil) {
		NSString *reason = searched.count > 0
			? [NSString stringWithFormat:@"%@ is not found in %@", self.command, [searched componentsJoinedByString:@", "]]
			: [NSString stringWithFormat:@"%@ is not found", self.command];
		return [self failLaunchWithCode:ENOENT reason:reason error:error];
	}
	int input[2] = { -1, -1 };
	int output[2] = { -1, -1 };
	int errorOutput[2] = { -1, -1 };
	if (pipe(input) != 0 || pipe(output) != 0 || pipe(errorOutput) != 0) {
		int code = errno;
		int descriptors[6] = { input[0], input[1], output[0], output[1], errorOutput[0], errorOutput[1] };
		for (int index = 0; index < 6; index++) {
			[self closeDescriptor:descriptors[index]];
		}
		return [self failLaunchWithCode:code reason:@"the pipes could not be created" error:error];
	}
	// macOS has no pipe2, so close-on-exec follows each pipe at once, before the spawn, to keep a process
	// another thread spawns from inheriting an end. The child's copies are dup2 targets, which do not keep
	// the flag; an end that already has its target's number keeps none, since dup2 onto itself changes no flag.
	int ends[6] = { input[0], input[1], output[0], output[1], errorOutput[0], errorOutput[1] };
	int streams[6] = { STDIN_FILENO, -1, -1, STDOUT_FILENO, -1, STDERR_FILENO };
	for (int index = 0; index < 6; index++) {
		if (ends[index] != streams[index]) {
			fcntl(ends[index], F_SETFD, FD_CLOEXEC);
		}
	}
	pid_t processIdentifier = 0;
	NSArray<NSNumber *> *inherited = nil;
	NSArray<NSNumber *> *targets = nil;
	int status = [self prepareInheritedDescriptors:&inherited targets:&targets];
	if (status == 0) {
		status = [self spawnPath:path environment:environment input:input[0] output:output[1] errorOutput:errorOutput[1]
					   inherited:inherited targets:targets processIdentifier:&processIdentifier];
	}
	for (NSNumber *descriptor in inherited) {
		close(descriptor.intValue);
	}
	close(input[0]);
	close(output[1]);
	close(errorOutput[1]);
	if (status != 0) {
		close(input[1]);
		close(output[0]);
		close(errorOutput[0]);
		return [self failLaunchWithCode:status reason:[NSString stringWithFormat:@"%@ could not be launched", path] error:error];
	}
	[self configureParentDescriptor:input[1] writing:YES];
	[self configureParentDescriptor:output[0] writing:NO];
	[self configureParentDescriptor:errorOutput[0] writing:NO];

	NFKSubprocessRun *run = [[NFKSubprocessRun alloc] init];
	run.processIdentifier = processIdentifier;
	run.outputDescriptor = output[0];
	run.errorDescriptor = errorOutput[0];
	self.run = run;
	[self.inputLock lock];
	self.inputDescriptor = input[1];
	[self.inputLock unlock];
	self.processIdentifier = processIdentifier;
	self.terminationReason = NFKSubprocessTerminationReasonNone;
	self.terminationStatus = 0;
	self.launchError = nil;
	run.outputSource = [self readSourceForDescriptor:output[0] run:run output:YES];
	run.errorSource = [self readSourceForDescriptor:errorOutput[0] run:run output:NO];
	[self observeExitOfRun:run];
	[self enterState:NFKSubprocessStateRunning];
	return YES;
}

- (NSMutableDictionary<NSString *, NSString *> *)childEnvironment
{
	NSMutableDictionary<NSString *, NSString *> *environment = [NSMutableDictionary dictionary];
	if (self.inheritsEnvironment) {
		NSDictionary<NSString *, NSString *> *inherited = NSProcessInfo.processInfo.environment;
		NSArray<NSString *> *names = self.inheritedEnvironmentVariables ?: inherited.allKeys;
		for (NSString *name in names) {
			if (inherited[name] != nil) {
				environment[name] = inherited[name];
			}
		}
	}
	[environment addEntriesFromDictionary:self.environment];
	return environment;
}

/*! The executable to launch, or nil with searched set to every directory looked in. A command found
	only in a tool directory adds that directory to environment's PATH. */
- (nullable NSString *)executablePathInEnvironment:(NSMutableDictionary<NSString *, NSString *> *)environment
										  searched:(NSArray<NSString *> * _Nonnull * _Nonnull)searched
{
	NSString *command = self.command;
	if ([command containsString:@"/"]) {
		if (command.isAbsolutePath) {
			return command;
		}
		NSString *base = self.workingDirectoryURL.path ?: NSFileManager.defaultManager.currentDirectoryPath;
		return [base stringByAppendingPathComponent:command];
	}
	NSString *searchPath = environment[@"PATH"] ?: @"/usr/bin:/bin:/usr/sbin:/sbin";
	NSMutableOrderedSet<NSString *> *directories = [NSMutableOrderedSet orderedSet];
	for (NSString *directory in [searchPath componentsSeparatedByString:@":"]) {
		if (directory.length == 0 || [directories containsObject:directory]) {
			continue;
		}
		[directories addObject:directory];
		NSString *candidate = [self executableNamed:command inDirectory:directory];
		if (candidate != nil) {
			return candidate;
		}
	}
	for (NSString *directory in NFKAgentCLIToolDirectories()) {
		if ([directories containsObject:directory]) {
			continue;
		}
		[directories addObject:directory];
		NSString *candidate = [self executableNamed:command inDirectory:directory];
		if (candidate != nil) {
			environment[@"PATH"] = [NSString stringWithFormat:@"%@:%@", searchPath, directory];
			return candidate;
		}
	}
	*searched = directories.array;
	return nil;
}

- (nullable NSString *)executableNamed:(NSString *)name inDirectory:(NSString *)directory
{
	NSString *candidate = [directory stringByAppendingPathComponent:name];
	BOOL isDirectory = NO;
	BOOL found = [NSFileManager.defaultManager fileExistsAtPath:candidate isDirectory:&isDirectory] && !isDirectory
		&& [NSFileManager.defaultManager isExecutableFileAtPath:candidate];
	return found ? candidate : nil;
}

/*! Duplicates each owned inherited descriptor above every target number, so no duplication in the
	child overwrites a source another one still reads. Sets inherited to the duplicates and targets to
	their child numbers, both in sorted target order from one reading of the property, and returns 0 or
	the POSIX error. The caller closes the duplicates, once, after the spawn. */
- (int)prepareInheritedDescriptors:(NSArray<NSNumber *> * _Nullable * _Nonnull)inherited
						   targets:(NSArray<NSNumber *> * _Nullable * _Nonnull)outTargets
{
	*inherited = @[];
	*outTargets = @[];
	@synchronized (self) {
		return [self lockedPrepareInheritedDescriptors:inherited targets:outTargets];
	}
}

- (int)lockedPrepareInheritedDescriptors:(NSArray<NSNumber *> * _Nonnull * _Nonnull)inherited
								 targets:(NSArray<NSNumber *> * _Nonnull * _Nonnull)outTargets
{
	NSDictionary<NSNumber *, NSNumber *> *requested = self.ownedInheritedDescriptors;
	NSArray<NSNumber *> *targets = [requested.allKeys sortedArrayUsingSelector:@selector(compare:)];
	if (targets.count == 0) {
		return 0;
	}
	if (targets.firstObject.intValue <= STDERR_FILENO) {
		return EINVAL;
	}
	for (NSNumber *target in targets) {
		if (requested[target].intValue < 0) {
			return -requested[target].intValue;
		}
	}
	int floor = targets.lastObject.intValue + 1;
	NSMutableArray<NSNumber *> *duplicates = [NSMutableArray arrayWithCapacity:targets.count];
	for (NSNumber *target in targets) {
		int duplicate = fcntl(requested[target].intValue, F_DUPFD_CLOEXEC, floor);
		if (duplicate < 0) {
			int code = errno;
			for (NSNumber *made in duplicates) {
				close(made.intValue);
			}
			return code;
		}
		[duplicates addObject:@(duplicate)];
	}
	*inherited = duplicates;
	*outTargets = targets;
	return 0;
}

- (int)spawnPath:(NSString *)path environment:(NSDictionary<NSString *, NSString *> *)environment
		   input:(int)input output:(int)output errorOutput:(int)errorOutput inherited:(NSArray<NSNumber *> *)inherited
		 targets:(NSArray<NSNumber *> *)targets processIdentifier:(pid_t *)processIdentifier
{
	posix_spawn_file_actions_t actions;
	posix_spawn_file_actions_init(&actions);
	posix_spawn_file_actions_adddup2(&actions, input, STDIN_FILENO);
	posix_spawn_file_actions_adddup2(&actions, output, STDOUT_FILENO);
	posix_spawn_file_actions_adddup2(&actions, errorOutput, STDERR_FILENO);
	// The standard streams go first: a pipe can hold a number an inherited descriptor targets. Every
	// source here lies above every target, so the order among them does not matter.
	for (NSUInteger index = 0; index < inherited.count && index < targets.count; index++) {
		posix_spawn_file_actions_adddup2(&actions, inherited[index].intValue, targets[index].intValue);
	}
	if (self.workingDirectoryURL != nil) {
		// The macOS 26 SDK deprecates the _np form for posix_spawn_file_actions_addchdir, which
		// earlier systems lack. The _np form runs on every supported system.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
		posix_spawn_file_actions_addchdir_np(&actions, self.workingDirectoryURL.fileSystemRepresentation);
#pragma clang diagnostic pop
	}
	posix_spawnattr_t attributes;
	posix_spawnattr_init(&attributes);
	// Only the dup2 targets stay open in the child, and its process group is its own, so a
	// stop signals the server's own children too.
	posix_spawnattr_setflags(&attributes, POSIX_SPAWN_CLOEXEC_DEFAULT | POSIX_SPAWN_SETPGROUP | POSIX_SPAWN_SETSIGDEF | POSIX_SPAWN_SETSIGMASK);
	posix_spawnattr_setpgroup(&attributes, 0);
	sigset_t noSignals;
	sigemptyset(&noSignals);
	posix_spawnattr_setsigmask(&attributes, &noSignals);
	sigset_t allSignals;
	sigfillset(&allSignals);
	posix_spawnattr_setsigdefault(&attributes, &allSignals);
	qos_class_t qualityOfService = self.ownerQualityOfService;
	if (qualityOfService != QOS_CLASS_UNSPECIFIED) {
		posix_spawnattr_set_qos_class_np(&attributes, qualityOfService);
	}

	NSMutableArray<NSString *> *argumentStrings = [NSMutableArray arrayWithObject:self.command];
	[argumentStrings addObjectsFromArray:self.arguments];
	NSMutableArray<NSString *> *environmentStrings = [NSMutableArray array];
	[environment enumerateKeysAndObjectsUsingBlock:^(NSString *name, NSString *value, BOOL *stop) {
		[environmentStrings addObject:[NSString stringWithFormat:@"%@=%@", name, value]];
	}];
	char **argv = [self copyCStrings:argumentStrings];
	char **envp = [self copyCStrings:environmentStrings];
	int status = posix_spawn(processIdentifier, path.fileSystemRepresentation, &actions, &attributes, argv, envp);
	[self freeCStrings:argv];
	[self freeCStrings:envp];
	posix_spawn_file_actions_destroy(&actions);
	posix_spawnattr_destroy(&attributes);
	return status;
}

- (char **)copyCStrings:(NSArray<NSString *> *)strings
{
	char **copies = calloc(strings.count + 1, sizeof(char *));
	for (NSUInteger index = 0; index < strings.count; index++) {
		copies[index] = strdup(strings[index].UTF8String ?: "");
	}
	return copies;
}

- (void)freeCStrings:(char **)strings
{
	for (char **cursor = strings; *cursor != NULL; cursor++) {
		free(*cursor);
	}
	free(strings);
}

- (void)configureParentDescriptor:(int)descriptor writing:(BOOL)writing
{
	fcntl(descriptor, F_SETFD, FD_CLOEXEC);
	fcntl(descriptor, F_SETFL, fcntl(descriptor, F_GETFL) | O_NONBLOCK);
	if (writing) {
		fcntl(descriptor, F_SETNOSIGPIPE, 1);
	}
}

- (void)closeDescriptor:(int)descriptor
{
	if (descriptor >= 0) {
		close(descriptor);
	}
}

- (BOOL)failLaunchWithCode:(int)code reason:(NSString *)reason error:(NSError **)error
{
	NSError *underlying = [NSError errorWithDomain:NSPOSIXErrorDomain code:code userInfo:nil];
	NSString *description = [NSString stringWithFormat:@"%@: %s", reason, strerror(code)];
	NSError *failure = [NSError errorWithDomain:NFKSubprocessErrorDomain code:NFKSubprocessErrorLaunchFailed
									   userInfo:@{ NSLocalizedDescriptionKey: description, NSUnderlyingErrorKey: underlying }];
	self.launchError = failure;
	self.processIdentifier = 0;
	[self enterState:NFKSubprocessStateFailedToLaunch];
	if (error != NULL) {
		*error = failure;
	}
	return NO;
}

- (void)enterState:(NFKSubprocessState)state
{
	self.state = state;
	NFKSubprocessStateHandler owner = self.ownerStateHandler;
	if (owner != nil) {
		owner(self, state);
	}
	NFKSubprocessStateHandler handler = self.stateHandler;
	if (handler != nil) {
		handler(self, state);
	}
}

#pragma mark Output

- (dispatch_source_t)readSourceForDescriptor:(int)descriptor run:(NFKSubprocessRun *)run output:(BOOL)isOutput
{
	dispatch_source_t source = dispatch_source_create(DISPATCH_SOURCE_TYPE_READ, (uintptr_t)descriptor, 0, self.eventQueue);
	dispatch_source_set_event_handler(source, ^{
		[self readDescriptor:descriptor run:run output:isOutput];
	});
	dispatch_source_set_cancel_handler(source, ^{
		close(descriptor);
	});
	dispatch_resume(source);
	return source;
}

- (void)readDescriptor:(int)descriptor run:(NFKSubprocessRun *)run output:(BOOL)isOutput
{
	if ((isOutput ? run.outputSource : run.errorSource) == nil) {
		return;
	}
	uint8_t buffer[65536];
	for (;;) {
		ssize_t count = read(descriptor, buffer, sizeof buffer);
		if (count > 0) {
			[self receiveBytes:buffer length:(size_t)count run:run output:isOutput];
			continue;
		}
		if (count < 0 && errno == EINTR) {
			continue;
		}
		if (count < 0 && errno == EAGAIN) {
			return;
		}
		[self closeStreamOfRun:run output:isOutput];
		return;
	}
}

- (void)closeStreamOfRun:(NFKSubprocessRun *)run output:(BOOL)isOutput
{
	dispatch_source_t source = isOutput ? run.outputSource : run.errorSource;
	if (source == nil) {
		return;
	}
	dispatch_source_cancel(source);
	if (isOutput) {
		run.outputSource = nil;
	} else {
		run.errorSource = nil;
	}
	[self finishRunIfDrained:run];
}

- (void)receiveBytes:(const uint8_t *)bytes length:(size_t)length run:(NFKSubprocessRun *)run output:(BOOL)isOutput
{
	if (isOutput) {
		BOOL discarding = run.discardingOutput;
		NSArray<NSData *> *lines = [self linesByAppendingBytes:bytes length:length pending:run.pendingOutput discarding:&discarding];
		run.discardingOutput = discarding;
		for (NSData *line in lines) {
			[self deliverOutputLine:line];
		}
		return;
	}
	[self appendStandardError:bytes length:length];
	BOOL discarding = run.discardingError;
	NSArray<NSData *> *lines = [self linesByAppendingBytes:bytes length:length pending:run.pendingError discarding:&discarding];
	run.discardingError = discarding;
	for (NSData *line in lines) {
		[self deliverErrorLine:line];
	}
}

/*! Appends bytes to the unframed output and returns each line they complete. pending never holds a
	newline, so only the new bytes are scanned. */
- (NSArray<NSData *> *)linesByAppendingBytes:(const uint8_t *)bytes length:(size_t)length
									 pending:(NSMutableData *)pending discarding:(BOOL *)discarding
{
	NSMutableArray<NSData *> *lines = [NSMutableArray array];
	size_t start = 0;
	for (;;) {
		const uint8_t *newline = memchr(bytes + start, '\n', length - start);
		size_t end = newline != NULL ? (size_t)(newline - bytes) : length;
		if (!*discarding) {
			[pending appendBytes:bytes + start length:end - start];
			if (pending.length > self.maximumLineLength) {
				*discarding = YES;
				pending.length = 0;
			}
		}
		if (newline == NULL) {
			break;
		}
		if (!*discarding) {
			[lines addObject:[self lineFromPending:pending]];
		}
		*discarding = NO;
		pending.length = 0;
		start = end + 1;
	}
	return lines;
}

- (NSData *)lineFromPending:(NSData *)pending
{
	NSUInteger length = pending.length;
	if (length > 0 && ((const uint8_t *)pending.bytes)[length - 1] == '\r') {
		length -= 1;
	}
	return [NSData dataWithBytes:pending.bytes length:length];
}

- (void)deliverOutputLine:(NSData *)line
{
	NFKSubprocessLineHandler owner = self.ownerLineHandler;
	if (owner != nil) {
		owner(line);
	}
	NFKSubprocessLineHandler handler = self.lineHandler;
	if (handler != nil) {
		handler(line);
	}
}

- (void)deliverErrorLine:(NSData *)line
{
	NFKSubprocessTextLineHandler handler = self.standardErrorLineHandler;
	if (handler != nil) {
		handler([self stringFromData:line]);
	}
}

- (void)appendStandardError:(const uint8_t *)bytes length:(size_t)length
{
	NSMutableData *buffer = self.standardErrorBuffer;
	NSUInteger capacity = self.standardErrorCapacity;
	@synchronized (buffer) {
		[buffer appendBytes:bytes length:length];
		if (buffer.length > capacity) {
			[buffer replaceBytesInRange:NSMakeRange(0, buffer.length - capacity) withBytes:NULL length:0];
		}
	}
}

- (NSString *)recentStandardError
{
	NSMutableData *buffer = self.standardErrorBuffer;
	NSData *copy = nil;
	@synchronized (buffer) {
		copy = [buffer copy];
	}
	// The ring can begin inside a multi-byte character; its continuation bytes are skipped.
	const uint8_t *bytes = copy.bytes;
	NSUInteger start = 0;
	while (start < copy.length && start < 3 && (bytes[start] & 0xC0) == 0x80) {
		start += 1;
	}
	return [self stringFromData:[copy subdataWithRange:NSMakeRange(start, copy.length - start)]];
}

- (NSString *)stringFromData:(NSData *)data
{
	return [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding]
		?: [[NSString alloc] initWithData:data encoding:NSISOLatin1StringEncoding] ?: @"";
}

#pragma mark Exit

- (void)observeExitOfRun:(NFKSubprocessRun *)run
{
	dispatch_source_t source = dispatch_source_create(DISPATCH_SOURCE_TYPE_PROC, (uintptr_t)run.processIdentifier, DISPATCH_PROC_EXIT, self.eventQueue);
	dispatch_source_set_event_handler(source, ^{
		[self reapRun:run retryingAfter:NFKSubprocessReapRetryFirst];
	});
	run.exitSource = source;
	dispatch_resume(source);
	// A process that exited before the source was registered is reaped here.
	dispatch_async(self.eventQueue, ^{
		[self reapRun:run];
	});
}

/*! Reaps after the exit event, which the kernel can deliver before waitpid reports the child: measured on
	macOS 26 with codesign children started on six threads at once, waitid found no exited child 6 to 14 ms
	after the event in 8 of 16,000 runs, no second event followed, and each child stayed a zombie with its
	subprocess Running for good. Until the child is reaped, this tries again after delay, doubling up to
	NFKSubprocessReapRetryLongest; with it, 12,000 runs left none, 8 of them reaped on a retry. */
- (void)reapRun:(NFKSubprocessRun *)run retryingAfter:(NSTimeInterval)delay
{
	[self reapRun:run];
	if (run.exited) {
		return;
	}
	dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(delay * NSEC_PER_SEC)), self.eventQueue, ^{
		[self reapRun:run retryingAfter:MIN(delay * 2, NFKSubprocessReapRetryLongest)];
	});
}

- (void)reapRun:(NFKSubprocessRun *)run
{
	if (run.exited) {
		return;
	}
	int status = 0;
	pid_t reaped = 0;
	do {
		reaped = waitpid(run.processIdentifier, &status, WNOHANG);
	} while (reaped < 0 && errno == EINTR);
	if (reaped == 0) {
		return;
	}
	run.exited = YES;
	run.statusKnown = reaped > 0;
	run.waitStatus = status;
	if (run.exitSource != nil) {
		dispatch_source_cancel(run.exitSource);
		run.exitSource = nil;
	}
	[self finishRunIfDrained:run];
	if (!run.finished) {
		dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(NFKSubprocessDrainGrace * NSEC_PER_SEC)), self.eventQueue, ^{
			[self drainAndFinishRun:run];
		});
	}
}

- (void)drainAndFinishRun:(NFKSubprocessRun *)run
{
	if (run.finished) {
		return;
	}
	[self readDescriptor:run.outputDescriptor run:run output:YES];
	[self readDescriptor:run.errorDescriptor run:run output:NO];
	[self closeStreamOfRun:run output:YES];
	[self closeStreamOfRun:run output:NO];
	[self finishRunIfDrained:run];
}

- (void)finishRunIfDrained:(NFKSubprocessRun *)run
{
	if (run.exited && run.outputSource == nil && run.errorSource == nil) {
		[self finishRun:run];
	}
}

- (void)finishRun:(NFKSubprocessRun *)run
{
	if (run.finished) {
		return;
	}
	run.finished = YES;
	if (run.pendingOutput.length > 0 && !run.discardingOutput) {
		[self deliverOutputLine:[self lineFromPending:run.pendingOutput]];
	}
	if (run.pendingError.length > 0 && !run.discardingError) {
		[self deliverErrorLine:[self lineFromPending:run.pendingError]];
	}
	[self closeStandardInput];
	int status = run.waitStatus;
	if (run.statusKnown && WIFEXITED(status)) {
		self.terminationReason = NFKSubprocessTerminationReasonExit;
		self.terminationStatus = WEXITSTATUS(status);
	} else if (run.statusKnown && WIFSIGNALED(status)) {
		self.terminationReason = NFKSubprocessTerminationReasonSignal;
		self.terminationStatus = WTERMSIG(status);
	} else {
		self.terminationReason = NFKSubprocessTerminationReasonNone;
		self.terminationStatus = 0;
	}
	self.processIdentifier = 0;
	if (self.run == run) {
		self.run = nil;
	}
	[self scheduleRestartAfterRun:run];
	[self enterState:NFKSubprocessStateExited];
	NSArray<void (^)(void)> *completions = [self.stopCompletionHandlers copy];
	[self.stopCompletionHandlers removeAllObjects];
	for (void (^completion)(void) in completions) {
		completion();
	}
}

#pragma mark Restarting

- (void)scheduleRestartAfterRun:(NFKSubprocessRun *)run
{
	BOOL cleanExit = self.terminationReason == NFKSubprocessTerminationReasonExit && self.terminationStatus == 0;
	if (run.stopping || cleanExit || self.restartPolicy != NFKSubprocessRestartPolicyOnCrash) {
		return;
	}
	if (-run.launchDate.timeIntervalSinceNow >= self.restartResetInterval) {
		self.consecutiveCrashes = 0;
	}
	if (self.consecutiveCrashes >= self.maximumRestarts) {
		return;
	}
	NSTimeInterval delay = MIN(self.restartDelay * pow(2.0, (double)self.consecutiveCrashes), self.maximumRestartDelay);
	self.consecutiveCrashes += 1;
	self.restartGeneration += 1;
	self.restartPending = YES;
	NSUInteger generation = self.restartGeneration;
	dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(MAX(delay, 0.0) * NSEC_PER_SEC)), self.eventQueue, ^{
		if (!self.isRestartPending || self.restartGeneration != generation) {
			return;
		}
		self.restartPending = NO;
		self.restartCount += 1;
		[self launchWithError:NULL];
	});
}

- (void)cancelScheduledRestart
{
	self.restartPending = NO;
	self.restartGeneration += 1;
}

#pragma mark Input

- (BOOL)writeLine:(NSData *)line error:(NSError **)error
{
	NSMutableData *data = [NSMutableData dataWithCapacity:line.length + 1];
	[data appendData:line];
	[data appendBytes:"\n" length:1];
	return [self writeData:data error:error];
}

- (BOOL)writeData:(NSData *)data error:(NSError **)error
{
	[self.inputLock lock];
	BOOL written = [self writeLockedData:data error:error];
	[self.inputLock unlock];
	return written;
}

- (BOOL)writeLockedData:(NSData *)data error:(NSError **)error
{
	int descriptor = self.inputDescriptor;
	if (descriptor < 0) {
		return [self fail:error code:NFKSubprocessErrorNotRunning posixCode:0 reason:@"the process is not running or its standard input is closed"];
	}
	const uint8_t *bytes = data.bytes;
	NSUInteger offset = 0;
	NSTimeInterval timeout = self.writeTimeout > 0 ? self.writeTimeout : NFKSubprocessDefaultWriteTimeout;
	NSDate *deadline = [NSDate dateWithTimeIntervalSinceNow:timeout];
	while (offset < data.length) {
		if (self.inputClosing) {
			return [self fail:error code:NFKSubprocessErrorNotRunning posixCode:0 reason:@"the process's standard input closed during the write"];
		}
		ssize_t count = write(descriptor, bytes + offset, data.length - offset);
		if (count > 0) {
			offset += (NSUInteger)count;
			continue;
		}
		if (count < 0 && errno == EINTR) {
			continue;
		}
		if (count < 0 && errno == EAGAIN) {
			if (deadline.timeIntervalSinceNow <= 0) {
				return [self fail:error code:NFKSubprocessErrorWriteTimedOut posixCode:0
						   reason:[NSString stringWithFormat:@"the process read nothing for %g seconds", timeout]];
			}
			struct pollfd writable = { .fd = descriptor, .events = POLLOUT, .revents = 0 };
			poll(&writable, 1, NFKSubprocessWritePollMilliseconds);
			continue;
		}
		int code = count < 0 ? errno : EIO;
		NSString *reason = code == EPIPE ? @"the process closed its standard input" : @"the write to the process failed";
		return [self fail:error code:NFKSubprocessErrorWriteFailed posixCode:code reason:reason];
	}
	return YES;
}

- (void)closeStandardInput
{
	self.inputClosing = YES;
	[self.inputLock lock];
	[self closeDescriptor:self.inputDescriptor];
	self.inputDescriptor = -1;
	self.inputClosing = NO;
	[self.inputLock unlock];
}

- (BOOL)fail:(NSError **)error code:(NFKSubprocessError)code posixCode:(int)posixCode reason:(NSString *)reason
{
	if (error != NULL) {
		NSMutableDictionary *userInfo = [NSMutableDictionary dictionaryWithObject:reason forKey:NSLocalizedDescriptionKey];
		if (posixCode != 0) {
			userInfo[NSUnderlyingErrorKey] = [NSError errorWithDomain:NSPOSIXErrorDomain code:posixCode userInfo:nil];
		}
		*error = [NSError errorWithDomain:NFKSubprocessErrorDomain code:code userInfo:userInfo];
	}
	return NO;
}

#pragma mark Stopping

- (void)stopWithCompletionHandler:(nullable void (^)(void))completionHandler
{
	dispatch_async(self.eventQueue, ^{
		[self beginStopWithCompletionHandler:completionHandler];
	});
}

- (BOOL)stop
{
	if (self.isOnEventQueue) {
		[self beginStopWithCompletionHandler:nil];
		return NO;
	}
	dispatch_semaphore_t stopped = dispatch_semaphore_create(0);
	[self stopWithCompletionHandler:^{
		dispatch_semaphore_signal(stopped);
	}];
	// SIGKILL cannot be ignored, so the wait ends; the bound covers a process the kernel holds.
	NSTimeInterval bound = MAX(self.stopInputTimeout, 0.0) + MAX(self.stopTerminateTimeout, 0.0) + NFKSubprocessDrainGrace + 5;
	return dispatch_semaphore_wait(stopped, dispatch_time(DISPATCH_TIME_NOW, (int64_t)(bound * NSEC_PER_SEC))) == 0;
}

- (void)beginStopWithCompletionHandler:(nullable void (^)(void))completionHandler
{
	[self cancelScheduledRestart];
	NFKSubprocessRun *run = self.run;
	if (run == nil || run.finished) {
		if (completionHandler != nil) {
			completionHandler();
		}
		return;
	}
	if (completionHandler != nil) {
		[self.stopCompletionHandlers addObject:[completionHandler copy]];
	}
	if (run.stopping) {
		return;
	}
	run.stopping = YES;
	[self closeStandardInput];
	NSTimeInterval terminateTimeout = self.stopTerminateTimeout;
	[self after:self.stopInputTimeout perform:^{
		if (run.exited) {
			return;
		}
		[self sendSignal:SIGTERM toRun:run];
		[self after:terminateTimeout perform:^{
			if (!run.exited) {
				[self sendSignal:SIGKILL toRun:run];
			}
		}];
	}];
}

- (void)after:(NSTimeInterval)delay perform:(dispatch_block_t)block
{
	dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(MAX(delay, 0.0) * NSEC_PER_SEC)), self.eventQueue, block);
}

// Runs on the event queue before the run is reaped, so the identifier still names this process.
- (void)sendSignal:(int)signalNumber toRun:(NFKSubprocessRun *)run
{
	kill(-run.processIdentifier, signalNumber);
	kill(run.processIdentifier, signalNumber);
}

@end

#endif
