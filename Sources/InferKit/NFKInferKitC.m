//
//  NFKInferKitC.m
//  InferKit
//
//  The C interface of NFKInferKitC.h over the catalog, the backends, and the wire coding's attachment
//  mode. Every entry point catches Objective-C exceptions, so none reaches a C caller.
//

#if __has_include(<InferKitC/NFKInferKitC.h>)
#import <InferKitC/NFKInferKitC.h>
#elif __has_include(<InferKit/NFKInferKitC.h>)
#import <InferKit/NFKInferKitC.h>
#else
#import "../InferKitC/include/InferKitC/NFKInferKitC.h"
#endif

#import "NFKInferKit.h"
#import "NFKInferKitCDelivery.h"
#import "NFKErrors.h"
#import "NFKInferenceBackend.h"
#import "NFKInferenceJob.h"
#import "NFKInferenceRequest.h"
#import "NFKInferenceResult.h"
#import "NFKInferenceWireCoding.h"
#import "NFKModelBuilding.h"
#import "NFKModelCatalog.h"
#import "NFKModelOptions.h"
#if TARGET_OS_OSX
#import "NFKPluginRegistry.h"
#endif

#import <IOSurface/IOSurfaceRef.h>
#import <errno.h>
#import <stdlib.h>
#import <string.h>
#import <sys/stat.h>
#import <unistd.h>

NS_ASSUME_NONNULL_BEGIN

#pragma mark - Strings and handles

/*! A UTF-8 string with its terminating NUL, for a pointer a C caller reads. */
static NSData *NFKCString(NSString *string)
{
	NSMutableData *data = [[string dataUsingEncoding:NSUTF8StringEncoding] mutableCopy] ?: [NSMutableData data];
	[data appendBytes:"" length:1];
	return data;
}

/*! A copy of data with a terminating NUL, in memory nfk_string_free releases, or NULL. */
static char * _Nullable NFKCCopyCString(NSData *data)
{
	char *string = malloc(data.length + 1);
	if (string == NULL) {
		return NULL;
	}
	memcpy(string, data.bytes, data.length);
	string[data.length] = '\0';
	return string;
}

/*! A handle that owns one reference to object; its *_free function releases it. */
static void *NFKCRetainHandle(id object)
{
	return (__bridge_retained void *)object;
}

/*! The object behind a handle when it is of class cls, else nil, so a handle of another type reads as
	NULL. */
static id _Nullable NFKCObjectOfClass(const void * _Nullable handle, Class cls)
{
	if (handle == NULL) {
		return nil;
	}
	id object = (__bridge id)handle;
	return [object isKindOfClass:cls] ? object : nil;
}

static void NFKCReleaseHandle(void * _Nullable handle, Class cls)
{
	if (NFKCObjectOfClass(handle, cls) != nil) {
		CFBridgingRelease(handle);
	}
}

/*! The part of value JSON carries: strings, finite numbers, null, and arrays and string-keyed
	dictionaries of them. */
static id _Nullable NFKCJSONSafe(id _Nullable value)
{
	if ([value isKindOfClass:NSString.class] || value == NSNull.null) {
		return value;
	}
	if ([value isKindOfClass:NSNumber.class]) {
		return isfinite([value doubleValue]) ? value : nil;
	}
	if ([value isKindOfClass:NSArray.class]) {
		NSMutableArray *safe = [NSMutableArray arrayWithCapacity:[value count]];
		for (id element in value) {
			id kept = NFKCJSONSafe(element);
			if (kept != nil) {
				[safe addObject:kept];
			}
		}
		return safe;
	}
	if ([value isKindOfClass:NSDictionary.class]) {
		NSMutableDictionary *safe = [NSMutableDictionary dictionaryWithCapacity:[value count]];
		for (id key in value) {
			id kept = [key isKindOfClass:NSString.class] ? NFKCJSONSafe(value[key]) : nil;
			if (kept != nil) {
				safe[key] = kept;
			}
		}
		return safe;
	}
	return nil;
}

#pragma mark - Errors

/*! An error as the C interface hands it out. */
@interface NFKCError : NSObject
@property (nonatomic, readonly) NSError *error;
@property (nonatomic, readonly) NSData *domain;
@property (nonatomic, readonly) NSData *message;
@property (nonatomic, readonly) int64_t code;
- (instancetype)initWithError:(NSError *)error;
@end

@implementation NFKCError

- (instancetype)initWithError:(NSError *)error
{
	self = [super init];
	if (self != nil) {
		_error = error;
		_domain = NFKCString(error.domain ?: @"");
		_message = NFKCString(error.localizedDescription ?: @"");
		_code = (int64_t)error.code;
	}
	return self;
}

@end

/*! The part of a user-info value JSON carries: an error as {domain, code, message, details}, its details
	without the description the message already holds, an option
	problem as {key, code, message, value}, an object with a dictionary representation as that
	representation, and JSON values as themselves. Anything else is left out. */
static id _Nullable NFKCDetailValue(id _Nullable value)
{
	if ([value isKindOfClass:NSError.class]) {
		NSError *error = value;
		NSMutableDictionary<NSString *, id> *details = [NSMutableDictionary dictionary];
		[error.userInfo enumerateKeysAndObjectsUsingBlock:^(NSString *key, id entry, BOOL *stop) {
			BOOL repeatsMessage = [key isEqual:NSLocalizedDescriptionKey];
			id kept = [key isKindOfClass:NSString.class] && !repeatsMessage ? NFKCDetailValue(entry) : nil;
			if (kept != nil) {
				details[key] = kept;
			}
		}];
		return @{ @"domain": error.domain ?: @"", @"code": @(error.code), @"message": error.localizedDescription ?: @"",
				  @"details": details };
	}
	if ([value isKindOfClass:NFKModelOptionProblem.class]) {
		NFKModelOptionProblem *problem = value;
		NSMutableDictionary<NSString *, id> *entry = [@{ @"key": problem.key, @"code": problem.code, @"message": problem.message } mutableCopy];
		entry[@"value"] = NFKCJSONSafe(problem.value);
		return entry;
	}
	if ([value isKindOfClass:NSArray.class]) {
		NSMutableArray *kept = [NSMutableArray arrayWithCapacity:[value count]];
		for (id element in value) {
			id converted = NFKCDetailValue(element);
			if (converted != nil) {
				[kept addObject:converted];
			}
		}
		return kept;
	}
	if ([value isKindOfClass:NSDictionary.class]) {
		NSMutableDictionary *kept = [NSMutableDictionary dictionaryWithCapacity:[value count]];
		for (id key in value) {
			id converted = [key isKindOfClass:NSString.class] ? NFKCDetailValue(value[key]) : nil;
			if (converted != nil) {
				kept[key] = converted;
			}
		}
		return kept;
	}
	if ([value respondsToSelector:@selector(dictionaryRepresentation)]) {
		id representation = [value performSelector:@selector(dictionaryRepresentation)];
		return [representation isKindOfClass:NSDictionary.class] ? NFKCJSONSafe(representation) : nil;
	}
	return NFKCJSONSafe(value);
}

static NSError *NFKCMakeError(NSString *domain, NSInteger code, NSString *message)
{
	return [NSError errorWithDomain:domain code:code userInfo:@{ NSLocalizedDescriptionKey: message }];
}

static NSError *NFKCExceptionError(NSException *exception)
{
	NSString *message = [NSString stringWithFormat:@"InferKit raised %@: %@", exception.name, exception.reason ?: @"no reason"];
	return NFKCMakeError(NFKInferenceErrorDomain, kNFKError_InferenceBackendFailure, message);
}

static NSError *NFKCCancellationError(void)
{
	return NFKCMakeError(NSCocoaErrorDomain, NSUserCancelledError, @"The run was cancelled.");
}

static nfk_error *NFKCErrorHandle(NSError *error)
{
	return (nfk_error *)NFKCRetainHandle([[NFKCError alloc] initWithError:error]);
}

static void NFKCSetError(nfk_error * _Nullable * _Nullable outError, NSError *error)
{
	if (outError != NULL) {
		*outError = NFKCErrorHandle(error);
	}
}

static void NFKCClearError(nfk_error * _Nullable * _Nullable outError)
{
	if (outError != NULL) {
		*outError = NULL;
	}
}

#pragma mark - Run files

/*! The directory a run's decoded files land in, removed when the last owner releases it: the request
	decoding, the job, and the result that may name the files. */
@interface NFKCRunFiles : NSObject
@property (nonatomic, readonly, copy) NSURL *directoryURL;
@end

@implementation NFKCRunFiles

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		NSURL *root = [[NSURL fileURLWithPath:NSTemporaryDirectory() isDirectory:YES] URLByAppendingPathComponent:@"InferKit/c-api"
																									   isDirectory:YES];
		_directoryURL = [root URLByAppendingPathComponent:NSUUID.UUID.UUIDString isDirectory:YES];
	}
	return self;
}

- (void)dealloc
{
	[NSFileManager.defaultManager removeItemAtURL:_directoryURL error:NULL];
}

@end

#pragma mark - Request

/*! The buffer at index of a payload, striding by the first buffer's struct_size, or NULL. */
static const nfk_buffer * _Nullable NFKCBufferAtIndex(const nfk_payload *payload, NSInteger index)
{
	if (index < 0 || (uint32_t)index >= payload->buffer_count || payload->buffers == NULL) {
		return NULL;
	}
	uint32_t stride = payload->buffers[0].struct_size;
	if (stride < sizeof(nfk_buffer) || stride % _Alignof(nfk_buffer) != 0) {
		return NULL;
	}
	const nfk_buffer *buffer = (const nfk_buffer *)((const uint8_t *)payload->buffers + (size_t)stride * (size_t)index);
	return buffer->struct_size >= sizeof(nfk_buffer) ? buffer : NULL;
}

/*! The attachment source over a caller's payload. Bytes and surfaces are wrapped without a copy; each
	wrapper enters group while it lives, so the group empties once nothing reads the caller's memory. */
@interface NFKCPayloadSource : NSObject <NFKWireAttachmentSource>
@property (nonatomic, readonly) const nfk_payload *payload;
@property (nonatomic, readonly, nullable) dispatch_group_t group;
@property (nonatomic, readonly) NFKCRunFiles *files;
- (instancetype)initWithPayload:(const nfk_payload *)payload group:(nullable dispatch_group_t)group files:(NFKCRunFiles *)files;
@end

@implementation NFKCPayloadSource

- (instancetype)initWithPayload:(const nfk_payload *)payload group:(nullable dispatch_group_t)group files:(NFKCRunFiles *)files
{
	self = [super init];
	if (self != nil) {
		_payload = payload;
		_group = group;
		_files = files;
	}
	return self;
}

- (BOOL)fail:(NSError **)outError reason:(NSString *)reason
{
	if (outError != NULL) {
		*outError = NFKCMakeError(NFKInferenceErrorDomain, kNFKError_InferenceMissingInput, reason);
	}
	return NO;
}

- (nullable NSData *)attachmentDataAtIndex:(NSInteger)index error:(NSError **)outError
{
	const nfk_buffer *buffer = NFKCBufferAtIndex(self.payload, index);
	if (buffer == NULL) {
		[self fail:outError reason:[NSString stringWithFormat:@"the request has no buffer %ld", (long)index]];
		return nil;
	}
	switch (buffer->kind) {
		case NFK_BUFFER_BYTES:
			return [self dataOfBytesBuffer:buffer error:outError];
		case NFK_BUFFER_SURFACE:
			return [self dataOfSurfaceBuffer:buffer error:outError];
		case NFK_BUFFER_FILE:
			return [self dataOfFileBuffer:buffer error:outError];
		default:
			[self fail:outError reason:[NSString stringWithFormat:@"buffer %ld has the unknown kind %d", (long)index, buffer->kind]];
			return nil;
	}
}

/*! The caller's memory as data that holds group entered until it is released, then runs release. */
- (NSData *)borrowedBytes:(const void *)bytes length:(NSUInteger)length release:(nullable void (^)(void))release
{
	dispatch_group_t group = self.group;
	if (group != nil) {
		dispatch_group_enter(group);
	}
	return [[NSData alloc] initWithBytesNoCopy:(void *)bytes length:length deallocator:^(void *unused, NSUInteger unusedLength) {
		if (release != nil) {
			release();
		}
		if (group != nil) {
			dispatch_group_leave(group);
		}
	}];
}

- (nullable NSData *)dataOfBytesBuffer:(const nfk_buffer *)buffer error:(NSError **)outError
{
	if (buffer->length == 0) {
		return [NSData data];
	}
	if (buffer->bytes == NULL) {
		[self fail:outError reason:@"a bytes buffer has a length and no bytes"];
		return nil;
	}
	return [self borrowedBytes:buffer->bytes length:(NSUInteger)buffer->length release:nil];
}

- (nullable NSData *)dataOfSurfaceBuffer:(const nfk_buffer *)buffer error:(NSError **)outError
{
	IOSurfaceRef surface = (IOSurfaceRef)buffer->surface;
	if (surface == NULL || CFGetTypeID(surface) != IOSurfaceGetTypeID()) {
		[self fail:outError reason:@"a surface buffer carries no IOSurface"];
		return nil;
	}
	if (IOSurfaceLock(surface, kIOSurfaceLockReadOnly, NULL) != KERN_SUCCESS) {
		[self fail:outError reason:@"a surface buffer could not be locked"];
		return nil;
	}
	size_t size = IOSurfaceGetAllocSize(surface);
	NSUInteger length = buffer->length > 0 && buffer->length <= size ? (NSUInteger)buffer->length : size;
	CFRetain(surface);
	return [self borrowedBytes:IOSurfaceGetBaseAddress(surface) length:length release:^{
		IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
		CFRelease(surface);
	}];
}

- (nullable NSData *)dataOfFileBuffer:(const nfk_buffer *)buffer error:(NSError **)outError
{
	if (buffer->path != NULL) {
		NSString *path = [NSString stringWithUTF8String:buffer->path];
		if (path == nil) {
			[self fail:outError reason:@"a file buffer's path is not UTF-8"];
			return nil;
		}
		return [NSData dataWithContentsOfFile:path options:NSDataReadingMappedIfSafe error:outError];
	}
	if (buffer->descriptor < 0) {
		[self fail:outError reason:@"a file buffer names no path and no descriptor"];
		return nil;
	}
	struct stat status;
	if (fstat(buffer->descriptor, &status) != 0 || !S_ISREG(status.st_mode)) {
		[self fail:outError reason:@"a file buffer's descriptor is not a regular file"];
		return nil;
	}
	NSMutableData *data = [NSMutableData dataWithLength:(NSUInteger)status.st_size];
	size_t done = 0;
	while (done < data.length) {
		ssize_t count = pread(buffer->descriptor, (uint8_t *)data.mutableBytes + done, data.length - done, (off_t)done);
		if (count < 0 && errno == EINTR) {
			continue;
		}
		if (count <= 0) {
			[self fail:outError reason:@"a file buffer's descriptor could not be read"];
			return nil;
		}
		done += (size_t)count;
	}
	return data;
}

- (nullable NSURL *)fileURLForAttachmentAtIndex:(NSInteger)index name:(NSString *)name error:(NSError **)outError
{
	const nfk_buffer *buffer = NFKCBufferAtIndex(self.payload, index);
	if (buffer != NULL && buffer->kind == NFK_BUFFER_FILE && buffer->path != NULL) {
		NSString *path = [NSString stringWithUTF8String:buffer->path];
		if (!path.isAbsolutePath) {
			[self fail:outError reason:@"a file buffer's path is not absolute"];
			return nil;
		}
		return [NSURL fileURLWithPath:path];
	}
	NSData *data = [self attachmentDataAtIndex:index error:outError];
	if (data == nil) {
		return nil;
	}
	NSURL *directory = self.files.directoryURL;
	if (![NSFileManager.defaultManager createDirectoryAtURL:directory withIntermediateDirectories:YES attributes:nil error:outError]) {
		return nil;
	}
	NSString *fileName = [NSString stringWithFormat:@"input-%@.%@", NSUUID.UUID.UUIDString, NFKServedFileExtension(name.pathExtension)];
	NSURL *target = [directory URLByAppendingPathComponent:fileName];
	return [data writeToURL:target options:NSDataWritingAtomic error:outError] ? target : nil;
}

// The caller names files in its own process, so a reference by path is taken as it is.
- (nullable NSURL *)fileURLForPath:(NSString *)path error:(NSError **)outError
{
	return [NSURL fileURLWithPath:path];
}

@end

/*! The request a payload describes, or nil with an error. group, when given, holds an entry for each
	wrapper over the caller's memory. */
static NFKInferenceRequest * _Nullable NFKCRequestFromPayload(const nfk_payload * _Nullable payload,
															   dispatch_group_t _Nullable group,
															   NFKCRunFiles *files,
															   NSError **error)
{
	if (payload == NULL || payload->json == NULL) {
		*error = NFKCMakeError(NFKInferenceErrorDomain, kNFKError_InferenceMissingInput, @"The request carries no JSON.");
		return nil;
	}
	if (payload->struct_size < sizeof(nfk_payload)) {
		NSString *reason = [NSString stringWithFormat:@"The request's struct_size %u is smaller than nfk_payload's %zu bytes.",
							payload->struct_size, sizeof(nfk_payload)];
		*error = NFKCMakeError(NFKInferenceErrorDomain, kNFKError_InferenceMissingInput, reason);
		return nil;
	}
	if (payload->buffer_count > 0 && payload->buffers == NULL) {
		*error = NFKCMakeError(NFKInferenceErrorDomain, kNFKError_InferenceMissingInput,
							   @"The request counts buffers and passes no array of them.");
		return nil;
	}
	NSData *JSON = [NSData dataWithBytesNoCopy:(void *)payload->json length:strlen(payload->json) freeWhenDone:NO];
	NSError *parseError = nil;
	id object = [NSJSONSerialization JSONObjectWithData:JSON options:0 error:&parseError];
	if (![object isKindOfClass:NSDictionary.class]) {
		NSString *reason = parseError != nil
			? [NSString stringWithFormat:@"The request is not JSON: %@", parseError.localizedDescription]
			: @"The request's JSON is not an object.";
		*error = NFKCMakeError(NFKInferenceErrorDomain, kNFKError_InferenceMissingInput, reason);
		return nil;
	}
	NFKInferenceWireCoding *coder = [[NFKInferenceWireCoding alloc] initWithTemporaryDirectory:files.directoryURL];
	coder.attachmentSource = [[NFKCPayloadSource alloc] initWithPayload:payload group:group files:files];
	return [coder requestFromJSONObject:object error:error];
}

#pragma mark - Result

/*! A result in the C form: its tagged JSON and one bytes buffer per attachment, all owned until
	nfk_result_free. Every value with bytes is attached, so a C reader decodes no base64 for them. */
@interface NFKCResult : NSObject <NFKWireAttachmentSink>
@property (nonatomic, readonly) NSMutableArray<NSMutableData *> *regions;
@property (nonatomic, strong, nullable) NSData *JSON;
@property (nonatomic, strong, nullable) NSMutableData *table;
@property (nonatomic, strong, nullable) NFKInferenceResult *result;
@property (nonatomic, strong, nullable) NFKCRunFiles *files;
@property (nonatomic, readonly, nullable) const char *JSONString;
@property (nonatomic, readonly) uint32_t bufferCount;
+ (nullable instancetype)resultEncoding:(NFKInferenceResult *)result error:(NSError **)error;
- (nullable const nfk_buffer *)bufferAtIndex:(uint32_t)index;
@end

@implementation NFKCResult

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_regions = [NSMutableArray array];
	}
	return self;
}

+ (nullable instancetype)resultEncoding:(NFKInferenceResult *)result error:(NSError **)error
{
	NFKCResult *encoded = [[self alloc] init];
	return [encoded encodeResult:result error:error] ? encoded : nil;
}

- (nullable void *)attachmentRegionOfLength:(NSUInteger)length index:(NSInteger *)outIndex error:(NSError **)outError
{
	NSMutableData *region = [NSMutableData dataWithLength:length];
	if (region == nil) {
		if (outError != NULL) {
			*outError = NFKCMakeError(NFKInferenceErrorDomain, kNFKError_InferenceBackendFailure,
									  [NSString stringWithFormat:@"%lu bytes could not be allocated", (unsigned long)length]);
		}
		return NULL;
	}
	*outIndex = (NSInteger)self.regions.count;
	[self.regions addObject:region];
	return region.mutableBytes;
}

- (BOOL)encodeResult:(NFKInferenceResult *)result error:(NSError **)error
{
	NFKInferenceWireCoding *coder = [[NFKInferenceWireCoding alloc] initWithTemporaryDirectory:nil];
	coder.attachmentSink = self;
	coder.inlineByteLimit = 0;
	coder.encodesValueObjectsAsJSON = YES;
	id object = [coder JSONObjectForResult:result error:error];
	NSData *JSON = object != nil
		? [NSJSONSerialization dataWithJSONObject:object options:NSJSONWritingSortedKeys | NSJSONWritingWithoutEscapingSlashes error:error]
		: nil;
	if (JSON == nil) {
		return NO;
	}
	NSMutableData *terminated = [JSON mutableCopy];
	[terminated appendBytes:"" length:1];
	self.JSON = terminated;
	self.table = [NSMutableData dataWithLength:sizeof(nfk_buffer) * MAX(self.regions.count, (NSUInteger)1)];
	nfk_buffer *buffers = self.table.mutableBytes;
	[self.regions enumerateObjectsUsingBlock:^(NSMutableData *region, NSUInteger index, BOOL *stop) {
		buffers[index] = (nfk_buffer){ .struct_size = sizeof(nfk_buffer), .kind = NFK_BUFFER_BYTES, .bytes = region.bytes,
									   .length = region.length, .surface = NULL, .path = NULL, .descriptor = -1, .reserved = 0 };
	}];
	self.result = result;
	return YES;
}

- (nullable const char *)JSONString
{
	return self.JSON.bytes;
}

- (uint32_t)bufferCount
{
	return (uint32_t)self.regions.count;
}

- (nullable const nfk_buffer *)bufferAtIndex:(uint32_t)index
{
	if (index >= self.regions.count) {
		return NULL;
	}
	return (const nfk_buffer *)self.table.bytes + index;
}

@end

/*! A result in the C form holding the run's files, or nil with an error. */
static NFKCResult * _Nullable NFKCEncodedResult(NFKInferenceResult *result, NFKCRunFiles *files, NSError **outError)
{
	NSError *error = nil;
	NFKCResult *encoded = [NFKCResult resultEncoding:result error:&error];
	if (encoded == nil) {
		*outError = error ?: NFKCMakeError(NFKInferenceErrorDomain, kNFKError_InferenceBackendFailure, @"The result could not be encoded.");
		return nil;
	}
	encoded.files = files;
	return encoded;
}

/*! An encoded result as a handle the caller owns, or NULL with an error handle. */
static nfk_result * _Nullable NFKCResultHandle(NFKInferenceResult *result, NFKCRunFiles *files, nfk_error * _Nullable * _Nullable outError)
{
	NSError *error = nil;
	NFKCResult *encoded = NFKCEncodedResult(result, files, &error);
	if (encoded == nil) {
		NFKCSetError(outError, error);
		return NULL;
	}
	return (nfk_result *)NFKCRetainHandle(encoded);
}

#pragma mark - Backend

/*! A backend as the C interface hands it out. */
@interface NFKCBackend : NSObject
@property (nonatomic, readonly) id<NFKInferenceBackend> backend;
- (instancetype)initWithBackend:(id<NFKInferenceBackend>)backend;
@end

@implementation NFKCBackend

- (instancetype)initWithBackend:(id<NFKInferenceBackend>)backend
{
	self = [super init];
	if (self != nil) {
		_backend = backend;
	}
	return self;
}

- (NSDictionary<NSString *, id> *)describe
{
	id<NFKInferenceBackend> backend = self.backend;
	NSMutableDictionary<NSString *, id> *description = [NSMutableDictionary dictionary];
	description[@"identifier"] = backend.backendIdentifier ?: @"";
	description[@"ready"] = @(backend.isReady);
	description[@"loaded"] = @(NFKInferenceIsLoaded(backend));
	description[@"residentBytes"] = @(NFKInferenceResidentByteCount(backend));
	description[@"peakBytes"] = @(NFKInferencePeakByteCount(backend));
	if ([backend respondsToSelector:@selector(supportedInputKeys)]) {
		description[@"inputs"] = [backend.supportedInputKeys.allObjects sortedArrayUsingSelector:@selector(compare:)];
	}
	if ([backend respondsToSelector:@selector(supportedParameterKeys)]) {
		description[@"parameters"] = [backend.supportedParameterKeys.allObjects sortedArrayUsingSelector:@selector(compare:)];
	}
	if ([backend respondsToSelector:@selector(supportedOutputKeys)]) {
		description[@"outputs"] = [backend.supportedOutputKeys.allObjects sortedArrayUsingSelector:@selector(compare:)];
	}
	description[@"modelInfo"] = [backend respondsToSelector:@selector(modelInfo)] ? (NFKCJSONSafe(backend.modelInfo) ?: @{}) : @{};
	return description;
}

@end

#pragma mark - Job

static const void * const NFKCJobQueueKey = &NFKCJobQueueKey;

/*! A submitted run as the C interface hands it out. Its callbacks run one at a time on its serial queue.
	freed, delivering, and completed are read and written under condition. */
@interface NFKCJob : NSObject
@property (nonatomic, readonly) dispatch_queue_t queue;
@property (nonatomic, readonly) NSCondition *condition;
@property (nonatomic, readonly, nullable) nfk_progress_callback progress;
@property (nonatomic, readonly) nfk_completion_callback completion;
@property (nonatomic, readonly, nullable) void *context;
@property (atomic, strong, nullable) NFKInferenceJob *job;
@property (nonatomic) BOOL freed;
@property (nonatomic) BOOL delivering;
@property (nonatomic) BOOL completed;
@property (nonatomic, strong, nullable) NFKInferenceResult *lastPartial;
@property (nonatomic, strong, nullable) NFKCResult *lastEncodedPartial;
@end

@implementation NFKCJob

- (instancetype)initWithProgress:(nullable nfk_progress_callback)progress
					  completion:(nfk_completion_callback)completion
						 context:(nullable void *)context
{
	self = [super init];
	if (self != nil) {
		dispatch_queue_attr_t attributes = dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INITIATED, 0);
		_queue = dispatch_queue_create("com.inferkit.c-api.job", attributes);
		dispatch_queue_set_specific(_queue, NFKCJobQueueKey, (__bridge void *)self, NULL);
		_condition = [[NSCondition alloc] init];
		_progress = progress;
		_completion = completion;
		_context = context;
	}
	return self;
}

/*! Marks a callback as running and answers YES, or answers NO once the job is freed or completed. */
- (BOOL)beginCallbackCompleting:(BOOL)completing
{
	[self.condition lock];
	BOOL allowed = !self.freed && !self.completed;
	if (allowed) {
		self.delivering = YES;
		self.completed = completing;
	}
	[self.condition unlock];
	return allowed;
}

- (void)endCallback
{
	[self.condition lock];
	self.delivering = NO;
	[self.condition broadcast];
	[self.condition unlock];
}

/*! Marks the job freed, waits for a callback running on another thread, and answers whether the run had
	not completed. */
- (BOOL)markFreed
{
	BOOL onQueue = dispatch_get_specific(NFKCJobQueueKey) == (__bridge void *)self;
	[self.condition lock];
	self.freed = YES;
	while (self.delivering && !onQueue) {
		[self.condition wait];
	}
	BOOL completed = self.completed;
	[self.condition unlock];
	return !completed;
}

// Runs on the queue.
- (void)deliverProgress:(double)fraction partial:(nullable NFKInferenceResult *)partial
{
	nfk_progress_callback progress = self.progress;
	if (progress == NULL || ![self beginCallbackCompleting:NO]) {
		return;
	}
	if (partial != nil && partial != self.lastPartial) {
		self.lastPartial = partial;
		@try {
			self.lastEncodedPartial = [NFKCResult resultEncoding:partial error:NULL];
		} @catch (NSException *exception) {
			self.lastEncodedPartial = nil;
		}
	}
	NFKCResult *encoded = partial != nil ? self.lastEncodedPartial : nil;
	progress(self.context, fraction, (__bridge const nfk_result *)encoded);
	[self endCallback];
}

// Runs on the queue.
- (void)finishWithResult:(nullable NFKInferenceResult *)result error:(nullable NSError *)error files:(NFKCRunFiles *)files
{
	if (![self beginCallbackCompleting:YES]) {
		return;
	}
	NFKCResult *encoded = nil;
	NSError *failure = nil;
	@try {
		encoded = result != nil ? NFKCEncodedResult(result, files, &failure) : nil;
		if (result == nil) {
			failure = error ?: NFKCMakeError(NFKInferenceErrorDomain, kNFKError_InferenceBackendFailure, @"The run failed and reported no error.");
		}
	} @catch (NSException *exception) {
		encoded = nil;
		failure = NFKCExceptionError(exception);
	}
	NFKCError *failureObject = failure != nil ? [[NFKCError alloc] initWithError:failure] : nil;
	NFKCDeliverCompletion(self.completion, self.context, encoded != nil ? CFBridgingRetain(encoded) : NULL,
						  failureObject != nil ? CFBridgingRetain(failureObject) : NULL);
	self.job = nil;
	self.lastPartial = nil;
	self.lastEncodedPartial = nil;
	[self endCallback];
}

/*! Relays job's progress and completion to the queue. A run that ends without a result completes once
	group is empty, which is when nothing reads the caller's buffers any more. */
- (void)observeJob:(NFKInferenceJob *)job group:(dispatch_group_t)group files:(NFKCRunFiles *)files
{
	self.job = job;
	__weak NFKCJob *weakSelf = self;
	if (self.progress != NULL) {
		job.progressHandler = ^(NFKInferenceJob *observed) {
			NFKCJob *strongSelf = weakSelf;
			double fraction = observed.progress;
			NFKInferenceResult *partial = observed.partialResult;
			if (strongSelf != nil) {
				dispatch_async(strongSelf.queue, ^{
					[strongSelf deliverProgress:fraction partial:partial];
				});
			}
		};
	}
	job.completionHandler = ^(NFKInferenceJob *observed) {
		NFKCJob *strongSelf = weakSelf;
		if (strongSelf == nil) {
			return;
		}
		NFKInferenceResult *result = observed.result;
		NSError *error = observed.status == NFKInferenceJobStatusCancelled ? NFKCCancellationError() : observed.error;
		void (^finish)(void) = ^{
			[strongSelf finishWithResult:result error:error files:files];
		};
		if (result != nil) {
			dispatch_async(strongSelf.queue, finish);
		} else {
			dispatch_group_notify(group, strongSelf.queue, finish);
		}
	};
}

@end

NS_ASSUME_NONNULL_END

#pragma mark - Library

uint32_t nfk_api_version(void)
{
	@try {
		return (uint32_t)(((NSUInteger)NFKInferKit.APIMajorVersion << 16) | (NSUInteger)NFKInferKit.APIMinorVersion);
	} @catch (NSException *exception) {
		return NFK_API_VERSION;
	}
}

const char *nfk_release_version(void)
{
	static char *version;
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		@autoreleasepool {
			@try {
				version = strdup(NFKInferKit.releaseVersion.UTF8String ?: "");
			} @catch (NSException *exception) {
				version = NULL;
			}
		}
	});
	return version != NULL ? version : "";
}

char *nfk_catalog_copy_json(nfk_error **out_error)
{
	NFKCClearError(out_error);
	@autoreleasepool {
		@try {
			NSError *error = nil;
			NSData *manifest = [NFKModelCatalog.sharedCatalog manifestDataWithError:&error];
			if (manifest == nil) {
				NFKCSetError(out_error, error ?: NFKCMakeError(NFKModelCatalogErrorDomain, NFKModelCatalogErrorMalformedManifest,
															   @"The catalog could not be written."));
				return NULL;
			}
			return NFKCCopyCString(manifest);
		} @catch (NSException *exception) {
			NFKCSetError(out_error, NFKCExceptionError(exception));
			return NULL;
		}
	}
}

int32_t nfk_plugins_add_directory(const char *path, nfk_error **out_error)
{
	NFKCClearError(out_error);
	@autoreleasepool {
		@try {
#if TARGET_OS_OSX
			NSString *directory = path != NULL ? [NSString stringWithUTF8String:path] : nil;
			if (directory.length == 0) {
				NFKCSetError(out_error, NFKCMakeError(NFKInferenceErrorDomain, kNFKError_InferenceMissingInput,
													  @"No plugin directory was named."));
				return -1;
			}
			NSArray<NSError *> *errors = nil;
			NSArray<NFKPluginManifest *> *added = [NFKPluginRegistry.sharedRegistry addPluginsInDirectoryURL:[NSURL fileURLWithPath:directory isDirectory:YES]
																									 errors:&errors];
			if (errors.count > 0) {
				NFKCSetError(out_error, errors.firstObject);
			}
			return (int32_t)MIN(added.count, (NSUInteger)INT32_MAX);
#else
			NFKCSetError(out_error, NFKCMakeError(NFKInferenceErrorDomain, kNFKError_InferenceUnsupported,
												  @"Plugins load on macOS only."));
			return -1;
#endif
		} @catch (NSException *exception) {
			NFKCSetError(out_error, NFKCExceptionError(exception));
			return -1;
		}
	}
}

#pragma mark - Backends

nfk_backend *nfk_backend_create(const char *catalog_identifier, const char *options_json, nfk_error **out_error)
{
	NFKCClearError(out_error);
	@autoreleasepool {
		@try {
			NSString *identifier = catalog_identifier != NULL ? [NSString stringWithUTF8String:catalog_identifier] : nil;
			if (identifier.length == 0) {
				NFKCSetError(out_error, NFKCMakeError(NFKModelCatalogErrorDomain, NFKModelCatalogErrorUnknownModel,
													  @"No catalog identifier was named."));
				return NULL;
			}
			NSDictionary<NSString *, id> *options = nil;
			if (options_json != NULL && options_json[0] != '\0') {
				NSData *JSON = [NSData dataWithBytesNoCopy:(void *)options_json length:strlen(options_json) freeWhenDone:NO];
				NSError *parseError = nil;
				id object = [NSJSONSerialization JSONObjectWithData:JSON options:0 error:&parseError];
				if (![object isKindOfClass:NSDictionary.class]) {
					NSString *reason = parseError != nil
						? [NSString stringWithFormat:@"The options are not JSON: %@", parseError.localizedDescription]
						: @"The options are not a JSON object.";
					NFKCSetError(out_error, NFKCMakeError(NFKModelCatalogErrorDomain, NFKModelCatalogErrorInvalidOptions, reason));
					return NULL;
				}
				options = object;
			}
			NSError *error = nil;
			id<NFKInferenceBackend> backend = [NFKModelCatalog.sharedCatalog backendForModelIdentifier:identifier options:options error:&error];
			if (backend == nil) {
				NFKCSetError(out_error, error ?: NFKCMakeError(NFKModelCatalogErrorDomain, NFKModelCatalogErrorNoBuilder,
															   @"The catalog built no backend."));
				return NULL;
			}
			return (nfk_backend *)NFKCRetainHandle([[NFKCBackend alloc] initWithBackend:backend]);
		} @catch (NSException *exception) {
			NFKCSetError(out_error, NFKCExceptionError(exception));
			return NULL;
		}
	}
}

char *nfk_backend_copy_description(nfk_backend *backend)
{
	@autoreleasepool {
		@try {
			NFKCBackend *object = NFKCObjectOfClass(backend, NFKCBackend.class);
			if (object == nil) {
				return NULL;
			}
			NSData *JSON = [NSJSONSerialization dataWithJSONObject:[object describe]
														   options:NSJSONWritingSortedKeys | NSJSONWritingWithoutEscapingSlashes error:NULL];
			return JSON != nil ? NFKCCopyCString(JSON) : NULL;
		} @catch (NSException *exception) {
			return NULL;
		}
	}
}

int32_t nfk_backend_prepare(nfk_backend *backend, nfk_error **out_error)
{
	NFKCClearError(out_error);
	@autoreleasepool {
		@try {
			NFKCBackend *object = NFKCObjectOfClass(backend, NFKCBackend.class);
			if (object == nil) {
				NFKCSetError(out_error, NFKCMakeError(NFKInferenceErrorDomain, kNFKError_InferenceNotReady, @"No backend was passed."));
				return 0;
			}
			NSError *error = nil;
			if (!NFKInferencePrepare(object.backend, &error)) {
				NFKCSetError(out_error, error ?: NFKCMakeError(NFKInferenceErrorDomain, kNFKError_InferenceNotReady,
															   @"The backend could not load its model and reported no error."));
				return 0;
			}
			return 1;
		} @catch (NSException *exception) {
			NFKCSetError(out_error, NFKCExceptionError(exception));
			return 0;
		}
	}
}

void nfk_backend_unload(nfk_backend *backend)
{
	@autoreleasepool {
		@try {
			NFKCBackend *object = NFKCObjectOfClass(backend, NFKCBackend.class);
			if (object != nil) {
				NFKInferenceUnload(object.backend);
			}
		} @catch (NSException *exception) {
		}
	}
}

void nfk_backend_free(nfk_backend *backend)
{
	@autoreleasepool {
		@try {
			NFKCReleaseHandle(backend, NFKCBackend.class);
		} @catch (NSException *exception) {
		}
	}
}

nfk_result *nfk_backend_run(nfk_backend *backend, const nfk_payload *request, nfk_error **out_error)
{
	NFKCClearError(out_error);
	@autoreleasepool {
		@try {
			NFKCBackend *object = NFKCObjectOfClass(backend, NFKCBackend.class);
			if (object == nil) {
				NFKCSetError(out_error, NFKCMakeError(NFKInferenceErrorDomain, kNFKError_InferenceNotReady, @"No backend was passed."));
				return NULL;
			}
			NFKCRunFiles *files = [[NFKCRunFiles alloc] init];
			NSError *error = nil;
			NFKInferenceRequest *decoded = NFKCRequestFromPayload(request, nil, files, &error);
			NFKInferenceResult *result = decoded != nil ? [object.backend runInferenceForRequest:decoded error:&error] : nil;
			if (result == nil) {
				NFKCSetError(out_error, error ?: NFKCMakeError(NFKInferenceErrorDomain, kNFKError_InferenceBackendFailure,
															   @"The run failed and reported no error."));
				return NULL;
			}
			return NFKCResultHandle(result, files, out_error);
		} @catch (NSException *exception) {
			NFKCSetError(out_error, NFKCExceptionError(exception));
			return NULL;
		}
	}
}

nfk_job *nfk_backend_submit(nfk_backend *backend, const nfk_payload *request, nfk_progress_callback progress,
							nfk_completion_callback completion, void *context)
{
	@autoreleasepool {
		@try {
			NFKCBackend *object = NFKCObjectOfClass(backend, NFKCBackend.class);
			if (object == nil || request == NULL || completion == NULL) {
				return NULL;
			}
			NFKCJob *job = [[NFKCJob alloc] initWithProgress:progress completion:completion context:context];
			dispatch_group_t group = dispatch_group_create();
			NFKCRunFiles *files = [[NFKCRunFiles alloc] init];
			NSError *error = nil;
			NFKInferenceRequest *decoded = NFKCRequestFromPayload(request, group, files, &error);
			if (decoded == nil) {
				dispatch_group_notify(group, job.queue, ^{
					[job finishWithResult:nil error:error files:files];
				});
				return (nfk_job *)NFKCRetainHandle(job);
			}
			NFKInferenceJob *inference = NFKInferenceSubmit(object.backend, decoded, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0));
			[job observeJob:inference group:group files:files];
			return (nfk_job *)NFKCRetainHandle(job);
		} @catch (NSException *exception) {
			return NULL;
		}
	}
}

void nfk_job_cancel(nfk_job *job)
{
	@autoreleasepool {
		@try {
			NFKCJob *object = NFKCObjectOfClass(job, NFKCJob.class);
			[object.job cancel];
		} @catch (NSException *exception) {
		}
	}
}

void nfk_job_free(nfk_job *job)
{
	@autoreleasepool {
		@try {
			NFKCJob *object = NFKCObjectOfClass(job, NFKCJob.class);
			if (object == nil) {
				return;
			}
			if ([object markFreed]) {
				[object.job cancel];
			}
			NFKCReleaseHandle(job, NFKCJob.class);
		} @catch (NSException *exception) {
		}
	}
}

#pragma mark - Results, errors, strings

const char *nfk_result_json(const nfk_result *result)
{
	@try {
		NFKCResult *object = NFKCObjectOfClass(result, NFKCResult.class);
		return object.JSONString;
	} @catch (NSException *exception) {
		return NULL;
	}
}

uint32_t nfk_result_buffer_count(const nfk_result *result)
{
	@try {
		NFKCResult *object = NFKCObjectOfClass(result, NFKCResult.class);
		return object.bufferCount;
	} @catch (NSException *exception) {
		return 0;
	}
}

const nfk_buffer *nfk_result_buffer(const nfk_result *result, uint32_t index)
{
	@try {
		NFKCResult *object = NFKCObjectOfClass(result, NFKCResult.class);
		return [object bufferAtIndex:index];
	} @catch (NSException *exception) {
		return NULL;
	}
}

void nfk_result_free(nfk_result *result)
{
	@autoreleasepool {
		@try {
			NFKCReleaseHandle(result, NFKCResult.class);
		} @catch (NSException *exception) {
		}
	}
}

const char *nfk_error_domain(const nfk_error *error)
{
	@try {
		NFKCError *object = NFKCObjectOfClass(error, NFKCError.class);
		return object.domain.bytes;
	} @catch (NSException *exception) {
		return NULL;
	}
}

int64_t nfk_error_code(const nfk_error *error)
{
	@try {
		NFKCError *object = NFKCObjectOfClass(error, NFKCError.class);
		return object.code;
	} @catch (NSException *exception) {
		return 0;
	}
}

const char *nfk_error_message(const nfk_error *error)
{
	@try {
		NFKCError *object = NFKCObjectOfClass(error, NFKCError.class);
		return object.message.bytes;
	} @catch (NSException *exception) {
		return NULL;
	}
}

char *nfk_error_copy_details_json(const nfk_error *error)
{
	@autoreleasepool {
		@try {
			NFKCError *object = NFKCObjectOfClass(error, NFKCError.class);
			if (object == nil) {
				return NULL;
			}
			NSDictionary *details = [NFKCDetailValue(object.error) objectForKey:@"details"] ?: @{};
			NSData *JSON = [NSJSONSerialization dataWithJSONObject:details
														   options:NSJSONWritingSortedKeys | NSJSONWritingWithoutEscapingSlashes error:NULL];
			return JSON != nil ? NFKCCopyCString(JSON) : NULL;
		} @catch (NSException *exception) {
			return NULL;
		}
	}
}

void nfk_error_free(nfk_error *error)
{
	@autoreleasepool {
		@try {
			NFKCReleaseHandle(error, NFKCError.class);
		} @catch (NSException *exception) {
		}
	}
}

void nfk_string_free(char *string)
{
	free(string);
}
