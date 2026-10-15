//
//  NFKPluginSharedMemory.m
//  InferKit
//

#import "NFKPluginSharedMemory.h"
#import "NFKPluginWire.h"

#if TARGET_OS_OSX

#import <sys/mman.h>
#import <sys/stat.h>
#import <fcntl.h>
#import <unistd.h>
#import <errno.h>

/*! The longest shared-memory object name the system accepts (PSHMNAMLEN). */
static const NSUInteger NFKSharedMemoryMaximumNameLength = 31;

/*! The fewest random characters a name carries, so names do not collide. */
static const NSUInteger NFKSharedMemoryMinimumRandomLength = 10;

@interface NFKPluginSharedMemory ()
@property (nonatomic, readwrite) int descriptor;
@property (nonatomic, readwrite, nullable) void *mutableBytes;
@property (nonatomic, readwrite) NSUInteger length;
@property (nonatomic, readwrite, getter=isSharedMemoryObject) BOOL sharedMemoryObject;
@property (nonatomic, readwrite, getter=isCopied) BOOL copied;
@property (nonatomic, assign) BOOL mapped;
@end

@implementation NFKPluginSharedMemory

+ (nullable instancetype)regionWithLength:(NSUInteger)length namePrefix:(nullable NSString *)prefix error:(NSError **)outError
{
	BOOL sharedMemoryObject = YES;
	int descriptor = [self openSharedMemoryObjectWithPrefix:prefix];
	if (descriptor < 0) {
		sharedMemoryObject = NO;
		descriptor = [self openUnlinkedTemporaryFile];
	}
	if (descriptor < 0) {
		[self setPOSIXError:outError code:errno reason:@"no shared-memory object or temporary file could be created"];
		return nil;
	}
	if (length > 0 && ftruncate(descriptor, (off_t)length) != 0) {
		int code = errno;
		close(descriptor);
		[self setPOSIXError:outError code:code reason:[NSString stringWithFormat:@"a region of %lu bytes could not be sized", (unsigned long)length]];
		return nil;
	}
	NFKPluginSharedMemory *region = [[self alloc] initWithDescriptor:descriptor];
	region.sharedMemoryObject = sharedMemoryObject;
	return [region mapLength:length error:outError] ? region : nil;
}

+ (nullable instancetype)regionWithDescriptor:(int)descriptor copiesFiles:(BOOL)copiesFiles error:(NSError **)outError
{
	struct stat status;
	if (fstat(descriptor, &status) != 0) {
		int code = errno;
		close(descriptor);
		[self setPOSIXError:outError code:code reason:@"an attachment's descriptor could not be read"];
		return nil;
	}
	// A POSIX shared-memory object reports no file type; a regular file is the temporary-file fallback.
	// A pipe, socket, device, or directory never names a region.
	BOOL sharedMemoryObject = (status.st_mode & S_IFMT) == 0;
	BOOL regularFile = S_ISREG(status.st_mode);
	if ((!sharedMemoryObject && !regularFile) || status.st_size < 0) {
		close(descriptor);
		NFKPluginSetError(outError, NFKPluginErrorProtocolViolation, @"an attachment's descriptor is neither shared memory nor a file");
		return nil;
	}
	NSUInteger length = (NSUInteger)status.st_size;
	NFKPluginSharedMemory *region = [[self alloc] initWithDescriptor:descriptor];
	region.sharedMemoryObject = sharedMemoryObject;
	if (regularFile && copiesFiles) {
		return [region copyFileOfLength:length error:outError] ? region : nil;
	}
	return [region mapLength:length error:outError] ? region : nil;
}

- (instancetype)initWithDescriptor:(int)descriptor
{
	self = [super init];
	if (self != nil) {
		_descriptor = descriptor;
	}
	return self;
}

- (void)dealloc
{
	if (self.mapped && self.mutableBytes != NULL) {
		munmap(self.mutableBytes, self.length);
	} else if (self.copied) {
		free(self.mutableBytes);
	}
	if (self.descriptor >= 0) {
		close(self.descriptor);
	}
}

- (NSData *)data
{
	if (self.length == 0) {
		return [NSData data];
	}
	NFKPluginSharedMemory *region = self;
	return [[NSData alloc] initWithBytesNoCopy:self.mutableBytes length:self.length deallocator:^(void *bytes, NSUInteger length) {
		(void)region;
	}];
}

#pragma mark Creating

+ (int)openSharedMemoryObjectWithPrefix:(nullable NSString *)prefix
{
	NSString *head = prefix ?: @"nfk.";
	if (head.length + NFKSharedMemoryMinimumRandomLength > NFKSharedMemoryMaximumNameLength) {
		errno = ENAMETOOLONG;
		return -1;
	}
	NSUInteger randomLength = NFKSharedMemoryMaximumNameLength - head.length;
	for (int attempt = 0; attempt < 8; attempt++) {
		NSString *name = [head stringByAppendingString:[self randomNameOfLength:randomLength]];
		int descriptor = shm_open(name.UTF8String, O_RDWR | O_CREAT | O_EXCL, S_IRUSR | S_IWUSR);
		if (descriptor >= 0) {
			shm_unlink(name.UTF8String);
			fcntl(descriptor, F_SETFD, FD_CLOEXEC);
			return descriptor;
		}
		if (errno != EEXIST) {
			return -1;
		}
	}
	return -1;
}

+ (NSString *)randomNameOfLength:(NSUInteger)length
{
	static const char alphabet[] = "abcdefghijklmnopqrstuvwxyz0123456789";
	NSMutableString *name = [NSMutableString stringWithCapacity:length];
	for (NSUInteger index = 0; index < length; index++) {
		[name appendFormat:@"%c", alphabet[arc4random_uniform((uint32_t)(sizeof alphabet - 1))]];
	}
	return name;
}

+ (int)openUnlinkedTemporaryFile
{
	NSString *pattern = [NSTemporaryDirectory() stringByAppendingPathComponent:@"inferkit-region-XXXXXXXX"];
	char *path = strdup(pattern.fileSystemRepresentation);
	int descriptor = mkostemp(path, O_CLOEXEC);
	if (descriptor >= 0) {
		unlink(path);
	}
	free(path);
	return descriptor;
}

#pragma mark Mapping

- (BOOL)mapLength:(NSUInteger)length error:(NSError **)outError
{
	self.length = length;
	if (length == 0) {
		return YES;
	}
	void *bytes = mmap(NULL, length, PROT_READ | PROT_WRITE, MAP_SHARED, self.descriptor, 0);
	if (bytes != MAP_FAILED) {
		self.mutableBytes = bytes;
		self.mapped = YES;
		return YES;
	}
	// A descriptor opened read-only maps only for reading; its bytes are copied so the region stays
	// writable like every other.
	bytes = mmap(NULL, length, PROT_READ, MAP_SHARED, self.descriptor, 0);
	void *copy = bytes != MAP_FAILED ? malloc(length) : NULL;
	if (copy == NULL) {
		int code = bytes == MAP_FAILED ? errno : ENOMEM;
		if (bytes != MAP_FAILED) {
			munmap(bytes, length);
		}
		return [self.class setPOSIXError:outError code:code reason:[NSString stringWithFormat:@"a region of %lu bytes could not be mapped", (unsigned long)length]];
	}
	memcpy(copy, bytes, length);
	munmap(bytes, length);
	self.mutableBytes = copy;
	self.copied = YES;
	return YES;
}

- (BOOL)copyFileOfLength:(NSUInteger)length error:(NSError **)outError
{
	void *bytes = length > 0 ? malloc(length) : NULL;
	if (length > 0 && bytes == NULL) {
		close(self.descriptor);
		self.descriptor = -1;
		return [self.class setPOSIXError:outError code:ENOMEM reason:@"no memory for a copied region"];
	}
	NSUInteger offset = 0;
	while (offset < length) {
		ssize_t count = pread(self.descriptor, (char *)bytes + offset, length - offset, (off_t)offset);
		if (count < 0 && errno == EINTR) {
			continue;
		}
		if (count <= 0) {
			int code = count < 0 ? errno : EIO;
			free(bytes);
			close(self.descriptor);
			self.descriptor = -1;
			return [self.class setPOSIXError:outError code:code reason:@"a file region ended before its length"];
		}
		offset += (NSUInteger)count;
	}
	close(self.descriptor);
	self.descriptor = -1;
	self.mutableBytes = bytes;
	self.length = length;
	self.copied = YES;
	return YES;
}

+ (BOOL)setPOSIXError:(NSError **)outError code:(int)code reason:(NSString *)reason
{
	if (outError != NULL) {
		*outError = [NSError errorWithDomain:NSPOSIXErrorDomain code:code
									userInfo:@{ NSLocalizedDescriptionKey: [NSString stringWithFormat:@"%@: %s", reason, strerror(code)] }];
	}
	return NO;
}

@end

#endif
