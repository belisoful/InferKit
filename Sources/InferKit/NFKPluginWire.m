//
//  NFKPluginWire.m
//  InferKit
//

#import "NFKPluginWire.h"
#import "NFKPluginChannel.h"
#import "NFKPluginSharedMemory.h"

#if TARGET_OS_OSX
#import <copyfile.h>
#import <fcntl.h>
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
	NSURL *directory = self.temporaryDirectoryURL
		?: [[NSURL fileURLWithPath:NSTemporaryDirectory()] URLByAppendingPathComponent:@"InferKit" isDirectory:YES];
	NSError *directoryError = nil;
	if (![NSFileManager.defaultManager createDirectoryAtURL:directory withIntermediateDirectories:YES attributes:nil error:&directoryError]) {
		close(source);
		if (outError != NULL) {
			*outError = directoryError;
		}
		return nil;
	}
	NSString *fileName = [NSString stringWithFormat:@"plugin-%@.%@", NSUUID.UUID.UUIDString, NFKServedFileExtension(name.pathExtension)];
	NSURL *target = [directory URLByAppendingPathComponent:fileName];
	int destination = open(target.fileSystemRepresentation, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, S_IRUSR | S_IWUSR);
	BOOL copied = destination >= 0 && fcopyfile(source, destination, NULL, COPYFILE_DATA) == 0;
	int code = errno;
	close(source);
	if (destination >= 0) {
		close(destination);
	}
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

- (nullable NSURL *)fileURLForPath:(NSString *)path error:(NSError **)outError
{
	NSURL *url = [NSURL fileURLWithPath:path].URLByStandardizingPath;
	NSURL *allowed = self.allowedFileDirectoryURL.URLByStandardizingPath;
	if (allowed == nil) {
		return url;
	}
	NSString *prefix = [allowed.path hasSuffix:@"/"] ? allowed.path : [allowed.path stringByAppendingString:@"/"];
	if (![url.path hasPrefix:prefix]) {
		NFKPluginSetError(outError, NFKPluginErrorProtocolViolation,
						  [NSString stringWithFormat:@"a file reference lies outside %@", allowed.lastPathComponent]);
		return nil;
	}
	return url;
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
