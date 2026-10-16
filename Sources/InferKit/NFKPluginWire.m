//
//  NFKPluginWire.m
//  InferKit
//

#import "NFKPluginWire.h"
#import "NFKPluginChannel.h"
#import "NFKPluginSharedMemory.h"

#if TARGET_OS_OSX
#import <execinfo.h>
#import <pthread.h>
#import <os/lock.h>
#import <copyfile.h>
#import <fcntl.h>
#import <limits.h>
#import <stdlib.h>
#import <unistd.h>
#endif

#if TARGET_OS_OSX

NSString * const NFKPluginChannelEnvironmentVariable = @"INFERKIT_PLUGIN_CHANNEL_FD";

NSString * const NFKPluginMessageChannel = @"channel";
NSString * const NFKPluginMessageHello = @"hello";
NSString * const NFKPluginMessageDescribe = @"describe";
NSString * const NFKPluginMessageBuild = @"build";
NSString * const NFKPluginMessagePrepare = @"prepare";
NSString * const NFKPluginMessageRun = @"run";
NSString * const NFKPluginMessageProgress = @"progress";
NSString * const NFKPluginMessageCancel = @"cancel";
NSString * const NFKPluginMessageUnload = @"unload";
NSString * const NFKPluginMessageRelease = @"release";
NSString * const NFKPluginMessageStatus = @"status";
NSString * const NFKPluginMessageCallTool = @"callTool";
NSString * const NFKPluginMessagePing = @"ping";
NSString * const NFKPluginMessageShutdown = @"shutdown";
NSString * const NFKPluginMessageEvent = @"event";
NSString * const NFKPluginMessageLog = @"log";
NSString * const NFKPluginMessageReply = @"reply";
NSString * const NFKPluginEventVerified = @"verified";
NSString * const NFKPluginCapabilityFileDescriptors = @"file-descriptors";

#pragma mark - Descriptor ownership

#if DEBUG

static os_unfair_lock NFKPluginDescriptorRegistryLock = OS_UNFAIR_LOCK_INIT;

static NSMutableDictionary<NSNumber *, NSString *> *NFKPluginDescriptorRegistry(void)
{
	static NSMutableDictionary<NSNumber *, NSString *> *registry = nil;
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		registry = [NSMutableDictionary dictionary];
	});
	return registry;
}

/*! The last events each descriptor number went through, with their callers, so a reused number can be
	traced. Guarded by the registry lock. */
static NSMutableDictionary<NSNumber *, NSMutableArray<NSString *> *> *NFKPluginDescriptorHistories(void)
{
	static NSMutableDictionary<NSNumber *, NSMutableArray<NSString *> *> *histories = nil;
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		histories = [NSMutableDictionary dictionary];
	});
	return histories;
}

static const NSUInteger NFKPluginDescriptorHistoryDepth = 8;

NSString *NFKPluginDebugBacktrace(NSUInteger skip, NSUInteger count)
{
	void *frames[32];
	int captured = backtrace(frames, 32);
	char **symbols = backtrace_symbols(frames, captured);
	NSMutableArray<NSString *> *names = [NSMutableArray array];
	for (int index = (int)skip + 1; index < captured && names.count < count; index++) {
		NSString *line = symbols != NULL ? @(symbols[index]) : @"?";
		// "N  Image  0xADDRESS  symbol + offset": keep the symbol.
		NSRange address = [line rangeOfString:@" 0x"];
		NSRange symbol = address.location != NSNotFound ? [line rangeOfString:@" " options:0 range:NSMakeRange(address.location + 1, line.length - address.location - 1)] : address;
		NSString *name = symbol.location != NSNotFound ? [line substringFromIndex:symbol.location + 1] : line;
		NSRange offset = [name rangeOfString:@" + " options:NSBackwardsSearch];
		[names addObject:offset.location != NSNotFound ? [name substringToIndex:offset.location] : name];
	}
	free(symbols);
	return [names componentsJoinedByString:@" < "];
}

BOOL NFKPluginDebugTracing(void)
{
	static BOOL tracing = NO;
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		const char *setting = getenv("INFERKIT_PLUGIN_TRACE");
		tracing = setting != NULL && strcmp(setting, "1") == 0;
	});
	return tracing;
}

void NFKPluginDebugTrace(NSString *format, ...)
{
	if (!NFKPluginDebugTracing()) {
		return;
	}
	va_list arguments;
	va_start(arguments, format);
	NSString *message = [[NSString alloc] initWithFormat:format arguments:arguments];
	va_end(arguments);
	uint64_t thread = 0;
	pthread_threadid_np(NULL, &thread);
	fprintf(stderr, "[inferkit-plugin-trace pid %d thread %llu %.6f] %s\n", getpid(), thread,
			NSDate.date.timeIntervalSince1970, message.UTF8String);
}

// Runs under the registry lock.
static void NFKPluginRecordDescriptorEvent(int descriptor, NSString *event)
{
	NSMutableArray<NSString *> *history = NFKPluginDescriptorHistories()[@(descriptor)];
	if (history == nil) {
		history = [NSMutableArray array];
		NFKPluginDescriptorHistories()[@(descriptor)] = history;
	}
	// Symbolicating every event slows descriptor-heavy work, so callers are recorded only while tracing.
	[history addObject:NFKPluginDebugTracing() ? [NSString stringWithFormat:@"%@ from %@", event, NFKPluginDebugBacktrace(2, 4)] : event];
	if (history.count > NFKPluginDescriptorHistoryDepth) {
		[history removeObjectAtIndex:0];
	}
}

NSString *NFKPluginDebugDescriptorHistory(int descriptor)
{
	os_unfair_lock_lock(&NFKPluginDescriptorRegistryLock);
	NSString *owner = NFKPluginDescriptorRegistry()[@(descriptor)] ?: @"no owner";
	NSArray<NSString *> *history = [NFKPluginDescriptorHistories()[@(descriptor)] copy] ?: @[];
	os_unfair_lock_unlock(&NFKPluginDescriptorRegistryLock);
	return [NSString stringWithFormat:@"descriptor %d (now %@): %@", descriptor, owner, [history componentsJoinedByString:@" | "]];
}

static void NFKPluginDescriptorViolation(NSString *message, int descriptor)
{
	fprintf(stderr, "InferKit descriptor ownership: %s\n  %s\n", message.UTF8String, NFKPluginDebugDescriptorHistory(descriptor).UTF8String);
	abort();
}

void NFKPluginAdoptDescriptor(int descriptor, const char *owner)
{
	if (descriptor < 0) {
		return;
	}
	if (fcntl(descriptor, F_GETFD) == -1) {
		NFKPluginDescriptorViolation([NSString stringWithFormat:@"%s adopts descriptor %d, which is not open", owner, descriptor], descriptor);
	}
	os_unfair_lock_lock(&NFKPluginDescriptorRegistryLock);
	NSString *holder = NFKPluginDescriptorRegistry()[@(descriptor)];
	NFKPluginDescriptorRegistry()[@(descriptor)] = @(owner);
	NFKPluginRecordDescriptorEvent(descriptor, [NSString stringWithFormat:@"adopt by %s", owner]);
	os_unfair_lock_unlock(&NFKPluginDescriptorRegistryLock);
	if (holder != nil) {
		NFKPluginDescriptorViolation([NSString stringWithFormat:@"%s adopts descriptor %d while %@ still owns it: code that did not own %@'s descriptor closed it",
									  owner, descriptor, holder, holder], descriptor);
	}
}

void NFKPluginDisownDescriptor(int descriptor)
{
	if (descriptor < 0) {
		return;
	}
	os_unfair_lock_lock(&NFKPluginDescriptorRegistryLock);
	[NFKPluginDescriptorRegistry() removeObjectForKey:@(descriptor)];
	NFKPluginRecordDescriptorEvent(descriptor, @"disown");
	os_unfair_lock_unlock(&NFKPluginDescriptorRegistryLock);
}

void NFKPluginCloseDescriptor(int *descriptor)
{
	int number = *descriptor;
	if (number < 0) {
		return;
	}
	*descriptor = -1;
	os_unfair_lock_lock(&NFKPluginDescriptorRegistryLock);
	NSString *holder = NFKPluginDescriptorRegistry()[@(number)];
	[NFKPluginDescriptorRegistry() removeObjectForKey:@(number)];
	NFKPluginRecordDescriptorEvent(number, [NSString stringWithFormat:@"close by %@", holder ?: @"no owner"]);
	os_unfair_lock_unlock(&NFKPluginDescriptorRegistryLock);
	if (holder == nil) {
		NFKPluginDescriptorViolation([NSString stringWithFormat:@"descriptor %d is closed twice, or by an owner that never adopted it", number], number);
	}
	if (fcntl(number, F_GETFD) == -1) {
		NFKPluginDescriptorViolation([NSString stringWithFormat:@"%@ closes descriptor %d, which code that did not own it already closed",
									  holder, number], number);
	}
	close(number);
}

#else

void NFKPluginAdoptDescriptor(int descriptor, const char *owner)
{
}

void NFKPluginDisownDescriptor(int descriptor)
{
}

void NFKPluginCloseDescriptor(int *descriptor)
{
	if (*descriptor >= 0) {
		close(*descriptor);
		*descriptor = -1;
	}
}

#endif

#endif

BOOL NFKPluginSetError(NSError **error, NSInteger code, NSString *reason)
{
	if (error != NULL) {
		*error = [NSError errorWithDomain:NFKPluginErrorDomain code:code userInfo:@{ NSLocalizedDescriptionKey: reason }];
	}
	return NO;
}

BOOL NFKPluginSetNotImplemented(NSError **error, SEL selector)
{
	return NFKPluginSetError(error, NFKPluginErrorNotImplemented,
							 [NSString stringWithFormat:@"%@ is not implemented yet", NSStringFromSelector(selector)]);
}

#if TARGET_OS_OSX

#pragma mark - Run slots

@interface NFKPluginRunSlots ()
@property (nonatomic, assign) NSInteger limit;
@property (atomic, readwrite) NSInteger activeCount;
@property (nonatomic, strong) NSMutableArray<dispatch_block_t> *waiting;
@end

@implementation NFKPluginRunSlots

- (instancetype)initWithLimit:(NSInteger)limit
{
	self = [super init];
	if (self != nil) {
		_limit = MAX(limit, 1);
		_waiting = [NSMutableArray array];
	}
	return self;
}

- (void)admit:(dispatch_block_t)start
{
	BOOL startsNow = NO;
	@synchronized (self) {
		if (self.activeCount < self.limit) {
			self.activeCount += 1;
			startsNow = YES;
		} else {
			[self.waiting addObject:[start copy]];
		}
	}
	if (startsNow) {
		start();
	}
}

- (void)finish
{
	dispatch_block_t next = nil;
	@synchronized (self) {
		if (self.waiting.count > 0) {
			next = self.waiting.firstObject;
			[self.waiting removeObjectAtIndex:0];
		} else if (self.activeCount > 0) {
			self.activeCount -= 1;
		}
	}
	if (next != nil) {
		next();
	}
}

- (NSInteger)waitingCount
{
	@synchronized (self) {
		return (NSInteger)self.waiting.count;
	}
}

@end

#pragma mark - Attachment writer

@interface NFKPluginAttachmentWriter ()
@property (nonatomic, strong) NSMutableArray *attachments;
@end

@implementation NFKPluginAttachmentWriter

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_attachments = [NSMutableArray array];
	}
	return self;
}

- (NSArray<NSNumber *> *)descriptors
{
	NSMutableArray<NSNumber *> *descriptors = [NSMutableArray arrayWithCapacity:self.attachments.count];
	for (id attachment in self.attachments) {
		int descriptor = [attachment isKindOfClass:NFKPluginSharedMemory.class] ? [(NFKPluginSharedMemory *)attachment descriptor]
																				: [(NFKPluginDescriptor *)attachment descriptor];
		[descriptors addObject:@(descriptor)];
	}
	return descriptors;
}

- (nullable void *)attachmentRegionOfLength:(NSUInteger)length index:(NSInteger *)outIndex error:(NSError **)outError
{
	if (![self hasRoomWithError:outError]) {
		return NULL;
	}
	NFKPluginSharedMemory *region = [NFKPluginSharedMemory regionWithLength:length namePrefix:self.namePrefix error:outError];
	if (region == nil) {
		return NULL;
	}
	*outIndex = (NSInteger)self.attachments.count;
	[self.attachments addObject:region];
	return region.mutableBytes;
}

- (BOOL)attachFileAtURL:(NSURL *)url index:(NSInteger *)outIndex error:(NSError **)outError
{
	*outIndex = -1;
	if (!self.passesFilesAsDescriptors) {
		return YES;
	}
	if (![self hasRoomWithError:outError]) {
		return NO;
	}
	int descriptor = open(url.fileSystemRepresentation, O_RDONLY | O_CLOEXEC);
	if (descriptor < 0) {
		int code = errno;
		if (outError != NULL) {
			*outError = [NSError errorWithDomain:NSPOSIXErrorDomain code:code
										userInfo:@{ NSLocalizedDescriptionKey: [NSString stringWithFormat:@"%@ could not be opened: %s",
																				url.lastPathComponent, strerror(code)] }];
		}
		return NO;
	}
	*outIndex = (NSInteger)self.attachments.count;
	[self.attachments addObject:[[NFKPluginDescriptor alloc] initWithDescriptor:descriptor]];
	return YES;
}

- (BOOL)hasRoomWithError:(NSError **)outError
{
	if (self.attachments.count < NFKPluginFrameMaximumDescriptors) {
		return YES;
	}
	return NFKPluginSetError(outError, NFKPluginErrorProtocolViolation,
							 [NSString stringWithFormat:@"a message carries at most %u attachments", NFKPluginFrameMaximumDescriptors]);
}

@end

#pragma mark - Attachment reader

@interface NFKPluginAttachmentReader ()
@property (nonatomic, copy) NSArray<NFKPluginDescriptor *> *descriptors;
@property (nonatomic, strong) NSMutableDictionary<NSNumber *, NFKPluginSharedMemory *> *regions;
@end

@implementation NFKPluginAttachmentReader

- (instancetype)initWithDescriptors:(NSArray<NFKPluginDescriptor *> *)descriptors
{
	self = [super init];
	if (self != nil) {
		_descriptors = [descriptors copy];
		_regions = [NSMutableDictionary dictionary];
		_copiesFileRegions = YES;
	}
	return self;
}

- (nullable NSData *)attachmentDataAtIndex:(NSInteger)index error:(NSError **)outError
{
	@synchronized (self) {
		NFKPluginSharedMemory *region = self.regions[@(index)];
		if (region == nil) {
			int descriptor = [self takeDescriptorAtIndex:index error:outError];
			region = descriptor >= 0 ? [NFKPluginSharedMemory regionWithDescriptor:descriptor copiesFiles:self.copiesFileRegions error:outError] : nil;
			if (region == nil) {
				return nil;
			}
			self.regions[@(index)] = region;
		}
		return region.data;
	}
}

- (nullable NSURL *)fileURLForAttachmentAtIndex:(NSInteger)index name:(NSString *)name error:(NSError **)outError
{
	int source = -1;
	@synchronized (self) {
		source = [self takeDescriptorAtIndex:index error:outError];
	}
	if (source < 0) {
		return nil;
	}
	NFKPluginAdoptDescriptor(source, "NFKPluginAttachmentReader file copy");
	NSURL *directory = self.temporaryDirectoryURL
		?: [[NSURL fileURLWithPath:NSTemporaryDirectory()] URLByAppendingPathComponent:@"InferKit" isDirectory:YES];
	NSError *directoryError = nil;
	if (![NSFileManager.defaultManager createDirectoryAtURL:directory withIntermediateDirectories:YES attributes:nil error:&directoryError]) {
		NFKPluginCloseDescriptor(&source);
		if (outError != NULL) {
			*outError = directoryError;
		}
		return nil;
	}
	NSString *fileName = [NSString stringWithFormat:@"plugin-%@.%@", NSUUID.UUID.UUIDString, NFKServedFileExtension(name.pathExtension)];
	NSURL *target = [directory URLByAppendingPathComponent:fileName];
	int destination = open(target.fileSystemRepresentation, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, S_IRUSR | S_IWUSR);
	NFKPluginAdoptDescriptor(destination, "NFKPluginAttachmentReader file copy");
	BOOL copied = destination >= 0 && fcopyfile(source, destination, NULL, COPYFILE_DATA) == 0;
	int code = errno;
	NFKPluginCloseDescriptor(&source);
	NFKPluginCloseDescriptor(&destination);
	if (!copied) {
		[NSFileManager.defaultManager removeItemAtURL:target error:NULL];
		if (outError != NULL) {
			*outError = [NSError errorWithDomain:NSPOSIXErrorDomain code:code
										userInfo:@{ NSLocalizedDescriptionKey: [NSString stringWithFormat:@"a file attachment could not be copied: %s", strerror(code)] }];
		}
		return nil;
	}
	return target;
}

// The check compares resolved paths, and the URL it returns names the resolved file under the allowed
// directory's own spelling, so a symbolic link inside the directory cannot name a file outside it.
- (nullable NSURL *)fileURLForPath:(NSString *)path error:(NSError **)outError
{
	NSURL *url = [NSURL fileURLWithPath:path].URLByStandardizingPath;
	NSURL *allowed = self.allowedFileDirectoryURL;
	if (allowed == nil) {
		return url;
	}
	char resolvedFile[PATH_MAX];
	char resolvedDirectory[PATH_MAX];
	if (realpath(allowed.fileSystemRepresentation, resolvedDirectory) == NULL) {
		NFKPluginSetError(outError, NFKPluginErrorProtocolViolation,
						  [NSString stringWithFormat:@"a file reference lies outside %@, which does not exist", allowed.lastPathComponent]);
		return nil;
	}
	if (!path.isAbsolutePath || realpath(url.fileSystemRepresentation, resolvedFile) == NULL) {
		NFKPluginSetError(outError, NFKPluginErrorProtocolViolation,
						  [NSString stringWithFormat:@"a file reference names no file inside %@", allowed.lastPathComponent]);
		return nil;
	}
	NSString *file = [NSString stringWithUTF8String:resolvedFile];
	NSString *directory = [NSString stringWithUTF8String:resolvedDirectory];
	NSString *prefix = [directory hasSuffix:@"/"] ? directory : [directory stringByAppendingString:@"/"];
	if (![file hasPrefix:prefix]) {
		NFKPluginSetError(outError, NFKPluginErrorProtocolViolation,
						  [NSString stringWithFormat:@"a file reference lies outside %@", allowed.lastPathComponent]);
		return nil;
	}
	return [allowed URLByAppendingPathComponent:[file substringFromIndex:prefix.length] isDirectory:NO];
}

// Runs under the reader's lock.
- (int)takeDescriptorAtIndex:(NSInteger)index error:(NSError **)outError
{
	int descriptor = index >= 0 && (NSUInteger)index < self.descriptors.count ? [self.descriptors[(NSUInteger)index] takeDescriptor] : -1;
	if (descriptor < 0) {
		NFKPluginSetError(outError, NFKPluginErrorProtocolViolation,
						  [NSString stringWithFormat:@"attachment %ld is missing or was already used", (long)index]);
	}
	return descriptor;
}

@end

#endif
