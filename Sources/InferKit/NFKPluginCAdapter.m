//
//  NFKPluginCAdapter.m
//  InferKit
//

#import "NFKPluginCAdapter.h"
#import "NFKPluginLoaderPrivate.h"
#import "NFKPluginManifestPrivate.h"
#import "NFKPluginWire.h"
#import "NFKInferenceWireCoding.h"
#import "NFKErrors.h"
#import "NFKInferenceTool.h"
#import "NFKModelVocabulary.h"

#if __has_include(<InferKitC/NFKAPIVersion.h>)
#import <InferKitC/NFKAPIVersion.h>
#elif __has_include(<InferKit/NFKAPIVersion.h>)
#import <InferKit/NFKAPIVersion.h>
#else
#import "../InferKitC/include/InferKitC/NFKAPIVersion.h"
#endif

#if TARGET_OS_OSX

#import <IOSurface/IOSurfaceRef.h>
#import <copyfile.h>
#import <fcntl.h>
#import <sys/stat.h>
#import <unistd.h>

@class NFKPluginCRun;

static NSError *NFKPluginCError(NSString *domain, NSInteger code, NSString *reason)
{
	return [NSError errorWithDomain:domain code:code userInfo:@{ NSLocalizedDescriptionKey: reason }];
}

/*! A UTF-8 string with its terminating NUL, for a pointer the plugin reads. */
static NSData *NFKPluginCString(NSString *string)
{
	NSMutableData *data = [[string dataUsingEncoding:NSUTF8StringEncoding] mutableCopy] ?: [NSMutableData data];
	[data appendBytes:"" length:1];
	return data;
}

/*! The JSON a plugin wrote, or nil. */
static id _Nullable NFKPluginCParseJSON(NSString * _Nullable string)
{
	NSData *data = [string dataUsingEncoding:NSUTF8StringEncoding];
	return data != nil ? [NSJSONSerialization JSONObjectWithData:data options:0 error:NULL] : nil;
}

/*! The buffer at index of a payload, striding by the first buffer's struct_size so a plugin built
	against a later minor version, whose buffers are longer, reads correctly. NULL when out of range. */
static const nfk_plugin_buffer *NFKPluginCBufferAtIndex(const nfk_plugin_payload *payload, NSInteger index)
{
	if (index < 0 || (uint32_t)index >= payload->buffer_count || payload->buffers == NULL) {
		return NULL;
	}
	uint32_t stride = payload->buffers[0].struct_size;
	if (stride < sizeof(nfk_plugin_buffer)) {
		return NULL;
	}
	const nfk_plugin_buffer *buffer = (const nfk_plugin_buffer *)((const uint8_t *)payload->buffers + (size_t)stride * (size_t)index);
	return buffer->struct_size >= sizeof(nfk_plugin_buffer) ? buffer : NULL;
}

#pragma mark - Adapter interface

@interface NFKPluginCAdapter ()
{
	// The plugin keeps this table's address for its life, so it is an instance variable and never moves.
	nfk_plugin_host _hostTable;
}
@property (nonatomic, readwrite, copy) NFKPluginManifest *manifest;
@property (nonatomic, readwrite, strong) NFKPluginHostContext *context;
@property (nonatomic, assign) const nfk_plugin *table;
@property (nonatomic, strong) NSData *optionsJSON;
@property (nonatomic, strong) NSData *temporaryDirectoryPath;
@property (atomic, readwrite, getter=isShutDown) BOOL shutDown;
@property (nonatomic, copy) NSArray<NFKModelDescriptor *> *describedModels;
@property (nonatomic, copy) NSArray<NFKInferenceTool *> *describedTools;
@property (nonatomic, copy) NSArray<NFKModelOption *> *describedOptions;

- (nullable NSString *)takeString:(nullable char *)string;
- (NSError *)errorForStatus:(nfk_plugin_status)status errorString:(nullable char *)errorString operation:(NSString *)operation;
- (nullable NFKInferenceResult *)resultFromPayload:(const nfk_plugin_payload *)payload
										  copying:(BOOL)copying
											owner:(nullable id)owner
											error:(NSError **)error;
- (nullable NFKInferenceResult *)resultForStatus:(nfk_plugin_status)status
										 payload:(nullable nfk_plugin_payload *)payload
									 errorString:(nullable char *)errorString
										   error:(NSError **)error;
@end

#pragma mark - Run

/*! One run in progress: what the host table answers for it. The plugin sees it as an opaque
	nfk_plugin_run pointer, valid until its run returns. */
@interface NFKPluginCRun : NSObject
@property (atomic, assign) BOOL cancelled;
@property (atomic, assign) BOOL inFlight;
@property (nonatomic, strong) NSLock *lock;
@property (nonatomic, weak) NFKPluginCAdapter *adapter;
@property (nonatomic, copy, nullable) NSURL *outputDirectoryURL;
@property (nonatomic, strong, nullable) NSData *outputDirectoryPath;
@property (nonatomic, copy, nullable) void (^progressHandler)(double fraction, NFKInferenceResult * _Nullable partial);
@end

@implementation NFKPluginCRun

- (instancetype)initWithAdapter:(NFKPluginCAdapter *)adapter outputDirectoryURL:(nullable NSURL *)outputDirectoryURL
{
	self = [super init];
	if (self != nil) {
		_lock = [[NSLock alloc] init];
		_adapter = adapter;
		_outputDirectoryURL = [outputDirectoryURL copy];
	}
	return self;
}

/*! The run's output directory as a C string, made on first use: the caller's, or a new directory under
	the plugin's temporary directory. Valid while the run lives. */
- (const char *)outputDirectoryCString
{
	@synchronized (self) {
		if (self.outputDirectoryPath == nil) {
			NSURL *directory = self.outputDirectoryURL;
			if (directory == nil) {
				NSURL *runs = [self.adapter.context.temporaryDirectoryURL URLByAppendingPathComponent:@"runs" isDirectory:YES];
				directory = [runs URLByAppendingPathComponent:NSUUID.UUID.UUIDString isDirectory:YES];
				self.outputDirectoryURL = directory;
			}
			[NSFileManager.defaultManager createDirectoryAtURL:directory withIntermediateDirectories:YES attributes:nil error:NULL];
			self.outputDirectoryPath = NFKPluginCString(directory.path);
		}
		return self.outputDirectoryPath.bytes;
	}
}

- (void)reportProgress:(double)fraction partial:(nullable const nfk_plugin_payload *)partial
{
	void (^handler)(double, NFKInferenceResult *) = self.progressHandler;
	NFKPluginCAdapter *adapter = self.adapter;
	if (handler == nil || adapter == nil || self.cancelled) {
		return;
	}
	NFKInferenceResult *partialResult = partial != NULL ? [adapter resultFromPayload:partial copying:YES owner:nil error:NULL] : nil;
	handler(isfinite(fraction) ? fraction : -1, partialResult);
}

@end

#pragma mark - Host table

static void NFKPluginCHostLog(void *context, nfk_plugin_log_level level, const char *message)
{
	if (context == NULL || message == NULL) {
		return;
	}
	@autoreleasepool {
		NFKPluginCAdapter *adapter = (__bridge NFKPluginCAdapter *)context;
		if (adapter.isShutDown) {
			return;
		}
		NFKPluginLogLevel mapped = (NFKPluginLogLevel)MIN(MAX(level, NFK_PLUGIN_LOG_DEBUG), NFK_PLUGIN_LOG_ERROR);
		[adapter.context logMessage:[NSString stringWithUTF8String:message] ?: @"(a message that is not UTF-8)" level:mapped];
	}
}

static void NFKPluginCHostReportProgress(void *context, nfk_plugin_run *run, double fraction, const nfk_plugin_payload *partial)
{
	if (context == NULL || run == NULL) {
		return;
	}
	@autoreleasepool {
		NFKPluginCRun *current = (__bridge NFKPluginCRun *)(void *)run;
		[current reportProgress:fraction partial:partial];
	}
}

static int32_t NFKPluginCHostIsCancelled(void *context, nfk_plugin_run *run)
{
	if (run == NULL) {
		return 0;
	}
	return ((__bridge NFKPluginCRun *)(void *)run).cancelled ? 1 : 0;
}

static const char *NFKPluginCHostOutputDirectory(void *context, nfk_plugin_run *run)
{
	if (run == NULL) {
		return NULL;
	}
	@autoreleasepool {
		return [(__bridge NFKPluginCRun *)(void *)run outputDirectoryCString];
	}
}

#pragma mark - Request payload

/*! A request in the plugin's form: the JSON and one bytes buffer per attachment, owned until the run
	that reads them returns. */
@interface NFKPluginCRequestPayload : NSObject <NFKWireAttachmentSink>
{
	// The plugin reads this structure through a pointer during its run.
	nfk_plugin_payload _payload;
}
@property (nonatomic, strong) NSMutableArray<NSMutableData *> *regions;
@property (nonatomic, strong, nullable) NSData *JSON;
@property (nonatomic, strong, nullable) NSMutableData *buffers;
@end

@implementation NFKPluginCRequestPayload

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_regions = [NSMutableArray array];
	}
	return self;
}

- (nullable void *)attachmentRegionOfLength:(NSUInteger)length index:(NSInteger *)outIndex error:(NSError **)outError
{
	NSMutableData *region = [NSMutableData dataWithLength:length];
	if (region == nil) {
		NFKPluginSetError(outError, NFKPluginErrorLoadFailed, [NSString stringWithFormat:@"%lu bytes could not be allocated", (unsigned long)length]);
		return NULL;
	}
	*outIndex = (NSInteger)self.regions.count;
	[self.regions addObject:region];
	return region.mutableBytes;
}

- (BOOL)attachFileAtURL:(NSURL *)url index:(NSInteger *)outIndex error:(NSError **)outError
{
	*outIndex = -1;
	return YES;
}

- (BOOL)finishWithJSONObject:(id)object error:(NSError **)error
{
	NSData *JSON = [NSJSONSerialization dataWithJSONObject:object options:0 error:error];
	if (JSON == nil) {
		return NO;
	}
	NSMutableData *terminated = [JSON mutableCopy];
	[terminated appendBytes:"" length:1];
	self.JSON = terminated;
	self.buffers = [NSMutableData dataWithLength:sizeof(nfk_plugin_buffer) * MAX(self.regions.count, 1u)];
	nfk_plugin_buffer *buffers = self.buffers.mutableBytes;
	[self.regions enumerateObjectsUsingBlock:^(NSMutableData *region, NSUInteger index, BOOL *stop) {
		buffers[index] = (nfk_plugin_buffer){ .struct_size = sizeof(nfk_plugin_buffer), .kind = NFK_PLUGIN_BUFFER_BYTES,
											  .bytes = region.bytes, .length = region.length, .surface = NULL, .path = NULL,
											  .descriptor = -1, .reserved = 0 };
	}];
	_payload = (nfk_plugin_payload){ .struct_size = sizeof(nfk_plugin_payload), .buffer_count = (uint32_t)self.regions.count,
									 .json = self.JSON.bytes, .buffers = self.regions.count > 0 ? buffers : NULL };
	return YES;
}

- (const nfk_plugin_payload *)payloadPointer
{
	return &_payload;
}

@end

#pragma mark - Result payload

/*! Owns a payload the plugin returned and frees it through free_payload when released. Every value
	wrapped over the payload's memory retains it. */
@interface NFKPluginCPayloadOwner : NSObject
@property (nonatomic, strong) NFKPluginCAdapter *adapter;
@property (nonatomic, assign) nfk_plugin_payload *payload;
@end

@implementation NFKPluginCPayloadOwner

- (void)dealloc
{
	NFKPluginCAdapter *adapter = _adapter;
	if (_payload != NULL && !adapter.isShutDown) {
		adapter.table->free_payload(adapter.table->context, _payload);
	}
}

@end

/*! The attachment source over a plugin's payload. copying makes every value its own copy, for a payload
	valid only during a call. */
@interface NFKPluginCPayloadSource : NSObject <NFKWireAttachmentSource>
@property (nonatomic, assign) const nfk_plugin_payload *payload;
@property (nonatomic, assign) BOOL copying;
@property (nonatomic, strong, nullable) id owner;
@property (nonatomic, copy) NSURL *temporaryDirectoryURL;
@end

@implementation NFKPluginCPayloadSource

- (nullable NSData *)attachmentDataAtIndex:(NSInteger)index error:(NSError **)outError
{
	const nfk_plugin_buffer *buffer = NFKPluginCBufferAtIndex(self.payload, index);
	if (buffer == NULL) {
		NFKPluginSetError(outError, NFKPluginErrorProtocolViolation, [NSString stringWithFormat:@"the payload has no buffer %ld", (long)index]);
		return nil;
	}
	switch (buffer->kind) {
		case NFK_PLUGIN_BUFFER_BYTES:
			return [self dataOfBytesBuffer:buffer error:outError];
		case NFK_PLUGIN_BUFFER_SURFACE:
			return [self dataOfSurfaceBuffer:buffer error:outError];
		case NFK_PLUGIN_BUFFER_FILE:
			return [self dataOfFileBuffer:buffer error:outError];
		default:
			NFKPluginSetError(outError, NFKPluginErrorProtocolViolation, [NSString stringWithFormat:@"buffer %ld has the unknown kind %d", (long)index, buffer->kind]);
			return nil;
	}
}

- (nullable NSData *)dataOfBytesBuffer:(const nfk_plugin_buffer *)buffer error:(NSError **)outError
{
	if (buffer->length == 0) {
		return [NSData data];
	}
	if (buffer->bytes == NULL) {
		NFKPluginSetError(outError, NFKPluginErrorProtocolViolation, @"a bytes buffer has a length and no bytes");
		return nil;
	}
	if (self.copying) {
		return [NSData dataWithBytes:buffer->bytes length:(NSUInteger)buffer->length];
	}
	id owner = self.owner;
	return [[NSData alloc] initWithBytesNoCopy:(void *)buffer->bytes length:(NSUInteger)buffer->length deallocator:^(void *bytes, NSUInteger length) {
		(void)owner;
	}];
}

- (nullable NSData *)dataOfSurfaceBuffer:(const nfk_plugin_buffer *)buffer error:(NSError **)outError
{
	IOSurfaceRef surface = (IOSurfaceRef)buffer->surface;
	if (surface == NULL || CFGetTypeID(surface) != IOSurfaceGetTypeID()) {
		NFKPluginSetError(outError, NFKPluginErrorProtocolViolation, @"a surface buffer carries no IOSurface");
		return nil;
	}
	if (IOSurfaceLock(surface, kIOSurfaceLockReadOnly, NULL) != kIOReturnSuccess) {
		NFKPluginSetError(outError, NFKPluginErrorProtocolViolation, @"a surface buffer could not be locked");
		return nil;
	}
	size_t size = IOSurfaceGetAllocSize(surface);
	NSUInteger length = buffer->length > 0 && buffer->length <= size ? (NSUInteger)buffer->length : size;
	void *bytes = IOSurfaceGetBaseAddress(surface);
	if (self.copying) {
		NSData *copy = [NSData dataWithBytes:bytes length:length];
		IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
		return copy;
	}
	CFRetain(surface);
	id owner = self.owner;
	return [[NSData alloc] initWithBytesNoCopy:bytes length:length deallocator:^(void *unused, NSUInteger unusedLength) {
		(void)owner;
		IOSurfaceUnlock(surface, kIOSurfaceLockReadOnly, NULL);
		CFRelease(surface);
	}];
}

- (nullable NSData *)dataOfFileBuffer:(const nfk_plugin_buffer *)buffer error:(NSError **)outError
{
	if (buffer->path != NULL) {
		NSString *path = [NSString stringWithUTF8String:buffer->path];
		NSDataReadingOptions options = self.copying ? 0 : NSDataReadingMappedIfSafe;
		return path != nil ? [NSData dataWithContentsOfFile:path options:options error:outError] : nil;
	}
	if (buffer->descriptor < 0) {
		NFKPluginSetError(outError, NFKPluginErrorProtocolViolation, @"a file buffer names no path and no descriptor");
		return nil;
	}
	struct stat status;
	if (fstat(buffer->descriptor, &status) != 0 || !S_ISREG(status.st_mode)) {
		NFKPluginSetError(outError, NFKPluginErrorProtocolViolation, @"a file buffer's descriptor is not a regular file");
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
			NFKPluginSetError(outError, NFKPluginErrorProtocolViolation, @"a file buffer's descriptor could not be read");
			return nil;
		}
		done += (size_t)count;
	}
	return data;
}

- (nullable NSURL *)fileURLForAttachmentAtIndex:(NSInteger)index name:(NSString *)name error:(NSError **)outError
{
	const nfk_plugin_buffer *buffer = NFKPluginCBufferAtIndex(self.payload, index);
	if (buffer != NULL && buffer->kind == NFK_PLUGIN_BUFFER_FILE && buffer->path != NULL) {
		NSString *path = [NSString stringWithUTF8String:buffer->path];
		return path.isAbsolutePath ? [NSURL fileURLWithPath:path] : nil;
	}
	NSData *data = [self attachmentDataAtIndex:index error:outError];
	if (data == nil) {
		return nil;
	}
	NSString *fileName = [NSString stringWithFormat:@"plugin-%@.%@", NSUUID.UUID.UUIDString, NFKServedFileExtension(name.pathExtension)];
	NSURL *target = [self.temporaryDirectoryURL URLByAppendingPathComponent:fileName];
	[NSFileManager.defaultManager createDirectoryAtURL:self.temporaryDirectoryURL withIntermediateDirectories:YES attributes:nil error:NULL];
	return [data writeToURL:target options:NSDataWritingAtomic error:outError] ? target : nil;
}

- (nullable NSURL *)fileURLForPath:(NSString *)path error:(NSError **)outError
{
	return [NSURL fileURLWithPath:path];
}

@end

#pragma mark - Backend

@interface NFKPluginCBackend ()
@property (nonatomic, strong) NFKPluginCAdapter *adapter;
@property (nonatomic, assign) nfk_plugin_backend *handle;
@property (nonatomic, readwrite, copy) NSString *modelIdentifier;
@property (nonatomic, copy) NSString *identifier;
@property (nonatomic, strong) NFKPluginRunSlots *slots;
@property (nonatomic, copy) NSSet<NSString *> *inputKeys;
@property (nonatomic, copy) NSSet<NSString *> *parameterKeys;
@property (nonatomic, copy) NSSet<NSString *> *outputKeys;
@end

@implementation NFKPluginCBackend

- (instancetype)initWithAdapter:(NFKPluginCAdapter *)adapter handle:(nfk_plugin_backend *)handle modelIdentifier:(NSString *)modelIdentifier
{
	self = [super init];
	if (self != nil) {
		_adapter = adapter;
		_handle = handle;
		_modelIdentifier = [modelIdentifier copy];
		_identifier = [NSString stringWithFormat:@"plugin:%@/%@", adapter.manifest.identifier, modelIdentifier];
		_slots = [[NFKPluginRunSlots alloc] initWithLimit:adapter.manifest.maximumConcurrentRuns];
		[self readKeys];
	}
	return self;
}

- (void)dealloc
{
	NFKPluginCAdapter *adapter = _adapter;
	if (_handle != NULL && !adapter.isShutDown) {
		adapter.table->destroy_backend(adapter.table->context, _handle);
	}
}

/*! The key sets: the plugin's description of the backend, else the manifest's ports. */
- (void)readKeys
{
	NSDictionary *description = [self backendDescription];
	NFKModelDescriptor *model = [self.adapter.manifest modelWithIdentifier:self.modelIdentifier];
	NSMutableSet<NSString *> *inputs = [NSMutableSet set];
	NSMutableSet<NSString *> *parameters = [NSMutableSet set];
	NSMutableSet<NSString *> *outputs = [NSMutableSet set];
	for (NFKModelPort *port in model.inputs) {
		[inputs addObject:port.key];
	}
	for (NFKModelOption *option in model.options) {
		[parameters addObject:option.key];
	}
	for (NFKModelPort *port in model.outputs) {
		[outputs addObject:port.key];
	}
	self.inputKeys = [self keysIn:description[@"inputs"]] ?: inputs;
	self.parameterKeys = [self keysIn:description[@"parameters"]] ?: parameters;
	self.outputKeys = [self keysIn:description[@"outputs"]] ?: outputs;
}

- (nullable NSSet<NSString *> *)keysIn:(nullable id)value
{
	if (![value isKindOfClass:NSArray.class]) {
		return nil;
	}
	NSMutableSet<NSString *> *keys = [NSMutableSet set];
	for (id key in value) {
		if ([key isKindOfClass:NSString.class]) {
			[keys addObject:key];
		}
	}
	return keys;
}

/*! The plugin's describe_backend object, or an empty dictionary when it has none. */
- (NSDictionary<NSString *, id> *)backendDescription
{
	NFKPluginCAdapter *adapter = self.adapter;
	const nfk_plugin *table = adapter.table;
	if (table->describe_backend == NULL || adapter.isShutDown) {
		return @{};
	}
	char *JSON = NULL;
	nfk_plugin_status status = table->describe_backend(table->context, self.handle, &JSON);
	NSString *string = [adapter takeString:JSON];
	id object = status == NFK_PLUGIN_STATUS_OK ? NFKPluginCParseJSON(string) : nil;
	return [object isKindOfClass:NSDictionary.class] ? object : @{};
}

#pragma mark NFKInferenceBackend

- (BOOL)isReady
{
	return YES;
}

- (NSString *)backendIdentifier
{
	return self.identifier;
}

- (NSSet<NSString *> *)supportedInputKeys
{
	return self.inputKeys;
}

- (NSSet<NSString *> *)supportedParameterKeys
{
	return self.parameterKeys;
}

- (NSSet<NSString *> *)supportedOutputKeys
{
	return self.outputKeys;
}

- (NSDictionary<NSString *, id> *)modelInfo
{
	id info = [self backendDescription][@"modelInfo"];
	return [info isKindOfClass:NSDictionary.class] ? info : @{};
}

- (BOOL)isLoaded
{
	id loaded = [self backendDescription][@"loaded"];
	return [loaded isKindOfClass:NSNumber.class] ? [loaded boolValue] : YES;
}

- (NSInteger)residentByteCount
{
	id bytes = [self backendDescription][@"residentBytes"];
	return [bytes isKindOfClass:NSNumber.class] ? [bytes integerValue] : 0;
}

- (NSInteger)peakByteCount
{
	id bytes = [self backendDescription][@"peakBytes"];
	return [bytes isKindOfClass:NSNumber.class] ? [bytes integerValue] : 0;
}

- (void)unload
{
	NFKPluginCAdapter *adapter = self.adapter;
	if (adapter.table->unload != NULL && !adapter.isShutDown) {
		adapter.table->unload(adapter.table->context, self.handle);
	}
}

- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request error:(NSError **)error
{
	NFKPluginCRun *run = [[NFKPluginCRun alloc] initWithAdapter:self.adapter outputDirectoryURL:nil];
	// A synchronous run blocks its caller by contract, and only its caller, until a slot frees.
	dispatch_semaphore_t admitted = dispatch_semaphore_create(0);
	[self.slots admit:^{
		dispatch_semaphore_signal(admitted);
	}];
	dispatch_semaphore_wait(admitted, DISPATCH_TIME_FOREVER);
	NFKInferenceResult *result = [self runRequest:request run:run error:error];
	[self.slots finish];
	return result;
}

- (NFKInferenceJob *)submitInferenceJobForRequest:(NFKInferenceRequest *)request
{
	return [self submitRequest:request outputDirectoryURL:nil];
}

- (NFKInferenceJob *)submitInferenceJobForRequest:(NFKInferenceRequest *)request outputDirectoryURL:(NSURL *)outputDirectoryURL
{
	return [self submitRequest:request outputDirectoryURL:outputDirectoryURL];
}

#pragma mark Runs

- (NFKInferenceJob *)submitRequest:(NFKInferenceRequest *)request outputDirectoryURL:(nullable NSURL *)outputDirectoryURL
{
	NFKInferenceJob *job = [[NFKInferenceJob alloc] init];
	NFKPluginCRun *run = [[NFKPluginCRun alloc] initWithAdapter:self.adapter outputDirectoryURL:outputDirectoryURL];
	__weak NFKInferenceJob *weakJob = job;
	run.progressHandler = ^(double fraction, NFKInferenceResult *partial) {
		[weakJob reportProgress:fraction partialResult:partial];
	};
	__weak NFKPluginCBackend *weakSelf = self;
	job.cancellationHandler = ^{
		[weakSelf cancelRun:run];
	};
	NFKPluginRunSlots *slots = self.slots;
	[slots admit:^{
		dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
			NSError *error = nil;
			NFKInferenceResult *result = [self runRequest:request run:run error:&error];
			[slots finish];
			if (result != nil) {
				[job finishWithResult:result];
			} else {
				[job finishWithError:error ?: NFKPluginCError(NFKInferenceErrorDomain, kNFKError_InferenceBackendFailure, @"the plugin's run failed")];
			}
		});
	}];
	return job;
}

- (void)cancelRun:(NFKPluginCRun *)run
{
	run.cancelled = YES;
	NFKPluginCAdapter *adapter = self.adapter;
	// Under the run's lock, so the plugin's cancel never sees a run that has already returned.
	[run.lock lock];
	if (run.inFlight && adapter.table->cancel != NULL && !adapter.isShutDown) {
		adapter.table->cancel(adapter.table->context, self.handle, (nfk_plugin_run *)(__bridge void *)run);
	}
	[run.lock unlock];
}

// Runs while the caller holds one of the backend's run slots.
- (nullable NFKInferenceResult *)runRequest:(NFKInferenceRequest *)request run:(NFKPluginCRun *)run error:(NSError **)error
{
	NFKPluginCAdapter *adapter = self.adapter;
	NFKPluginCRequestPayload *payload NS_VALID_UNTIL_END_OF_SCOPE = [[NFKPluginCRequestPayload alloc] init];
	NFKInferenceWireCoding *coder = [[NFKInferenceWireCoding alloc] initWithTemporaryDirectory:adapter.context.temporaryDirectoryURL];
	coder.attachmentSink = payload;
	coder.encodesValueObjectsAsJSON = YES;
	id JSON = [coder JSONObjectForRequest:request error:error];
	if (JSON == nil || ![payload finishWithJSONObject:JSON error:error]) {
		return nil;
	}
	nfk_plugin_payload *result = NULL;
	char *errorString = NULL;
	nfk_plugin_status status = NFK_PLUGIN_STATUS_CANCELLED;
	[run.lock lock];
	BOOL starts = !run.cancelled && !adapter.isShutDown;
	run.inFlight = starts;
	[run.lock unlock];
	if (starts) {
		const nfk_plugin *table = adapter.table;
		status = table->run(table->context, self.handle, (nfk_plugin_run *)(__bridge void *)run, payload.payloadPointer, &result, &errorString);
	}
	[run.lock lock];
	run.inFlight = NO;
	[run.lock unlock];
	if (!starts && adapter.isShutDown) {
		NFKPluginSetError(error, NFKPluginErrorStopped, @"the plugin has shut down");
		return nil;
	}
	return [adapter resultForStatus:status payload:result errorString:errorString error:error];
}

@end

#pragma mark - Adapter

@implementation NFKPluginCAdapter

- (instancetype)initWithContext:(id<NFKPluginContext>)context
{
	// A C plugin is created by its entry function, which initWithEntry:manifest:context:error: calls.
	return nil;
}

- (nullable instancetype)initWithEntry:(nfk_plugin_entry_function)entry
							  manifest:(NFKPluginManifest *)manifest
							   context:(NFKPluginHostContext *)context
								 error:(NSError **)error
{
	self = [super init];
	if (self == nil) {
		return nil;
	}
	_manifest = [manifest copy];
	_context = context;
	NSData *options = [NSJSONSerialization isValidJSONObject:context.options]
		? [NSJSONSerialization dataWithJSONObject:context.options options:0 error:NULL] : nil;
	_optionsJSON = NFKPluginCString(options != nil ? [[NSString alloc] initWithData:options encoding:NSUTF8StringEncoding] : @"{}");
	_temporaryDirectoryPath = NFKPluginCString(context.temporaryDirectoryURL.path);
	_hostTable = (nfk_plugin_host){ .struct_size = sizeof(nfk_plugin_host), .abi_version = NFK_PLUGIN_ABI_VERSION,
									.context = (__bridge void *)self, .inferkit_api_version = NFK_API_VERSION, .reserved = 0,
									.options_json = _optionsJSON.bytes, .temporary_directory = _temporaryDirectoryPath.bytes,
									.log = NFKPluginCHostLog, .report_progress = NFKPluginCHostReportProgress,
									.is_cancelled = NFKPluginCHostIsCancelled, .output_directory = NFKPluginCHostOutputDirectory };
	const nfk_plugin *table = NULL;
	nfk_plugin_status status = entry(NFK_PLUGIN_ABI_VERSION, &_hostTable, &table);
	if (![self acceptTable:table status:status error:error]) {
		return nil;
	}
	if (![self readDescriptionWithError:error]) {
		[self shutdown];
		return nil;
	}
	return self;
}

- (BOOL)acceptTable:(nullable const nfk_plugin *)table status:(nfk_plugin_status)status error:(NSError **)error
{
	NSString *name = self.manifest.identifier;
	if (status == NFK_PLUGIN_STATUS_INCOMPATIBLE) {
		return NFKPluginSetError(error, NFKPluginErrorIncompatible,
								 [NSString stringWithFormat:@"the plugin %@ refused plugin ABI %d.%d", name, NFK_PLUGIN_ABI_VERSION_MAJOR, NFK_PLUGIN_ABI_VERSION_MINOR]);
	}
	if (status != NFK_PLUGIN_STATUS_OK || table == NULL) {
		return NFKPluginSetError(error, NFKPluginErrorLoadFailed,
								 [NSString stringWithFormat:@"the plugin %@'s entry returned status %d and no function table", name, status]);
	}
	// The version is read before the size, since a table of another major version has another layout.
	if (table->struct_size < offsetof(nfk_plugin, context) || (table->abi_version >> 16) != NFK_PLUGIN_ABI_VERSION_MAJOR) {
		return NFKPluginSetError(error, NFKPluginErrorIncompatible,
								 [NSString stringWithFormat:@"the plugin %@ implements plugin ABI %u.%u, and this library implements %d.%d", name,
								  table->abi_version >> 16, table->abi_version & 0xFFFF, NFK_PLUGIN_ABI_VERSION_MAJOR, NFK_PLUGIN_ABI_VERSION_MINOR]);
	}
	if (table->struct_size < sizeof(nfk_plugin)) {
		return NFKPluginSetError(error, NFKPluginErrorIncompatible,
								 [NSString stringWithFormat:@"the plugin %@'s function table is %u bytes, shorter than plugin ABI %d.0's %zu",
								  name, table->struct_size, NFK_PLUGIN_ABI_VERSION_MAJOR, sizeof(nfk_plugin)]);
	}
	BOOL complete = table->describe != NULL && table->create_backend != NULL && table->run != NULL && table->destroy_backend != NULL
		&& table->free_payload != NULL && table->free_string != NULL && table->shutdown != NULL;
	if (!complete) {
		if (table->shutdown != NULL) {
			table->shutdown(table->context);
		}
		return NFKPluginSetError(error, NFKPluginErrorLoadFailed,
								 [NSString stringWithFormat:@"the plugin %@'s function table lacks a required function", name]);
	}
	self.table = table;
	return YES;
}

- (BOOL)readDescriptionWithError:(NSError **)error
{
	char *JSON = NULL;
	nfk_plugin_status status = self.table->describe(self.table->context, &JSON);
	NSString *string = [self takeString:JSON];
	if (status != NFK_PLUGIN_STATUS_OK) {
		return NFKPluginSetError(error, NFKPluginErrorLoadFailed,
								 [NSString stringWithFormat:@"the plugin %@'s describe returned status %d", self.manifest.identifier, status]);
	}
	id description = NFKPluginCParseJSON(string);
	if (![description isKindOfClass:NSDictionary.class]) {
		return NFKPluginSetError(error, NFKPluginErrorLoadFailed,
								 [NSString stringWithFormat:@"the plugin %@'s describe wrote no JSON object", self.manifest.identifier]);
	}
	NSArray<NFKModelDescriptor *> *models = [self modelsIn:description[@"models"] error:error];
	NSArray<NFKModelOption *> *options = models != nil ? [self optionsIn:description[@"options"] error:error] : nil;
	if (options == nil) {
		return NO;
	}
	self.describedModels = models;
	self.describedOptions = options;
	self.describedTools = [self toolsIn:description[@"tools"]];
	return YES;
}

- (nullable NSArray<NFKModelDescriptor *> *)modelsIn:(nullable id)value error:(NSError **)error
{
	NSMutableArray<NFKModelDescriptor *> *models = [NSMutableArray array];
	for (id dictionary in [value isKindOfClass:NSArray.class] ? value : @[]) {
		NSMutableDictionary *withEngine = [dictionary isKindOfClass:NSDictionary.class] ? [dictionary mutableCopy] : nil;
		if (withEngine[@"engine"] == nil) {
			withEngine[@"engine"] = NFKModelEnginePlugin;
		}
		NSError *modelError = nil;
		NFKModelDescriptor *model = withEngine != nil ? [NFKModelDescriptor descriptorWithDictionary:withEngine error:&modelError] : nil;
		if (model == nil) {
			NFKPluginSetError(error, NFKPluginErrorLoadFailed,
							  [NSString stringWithFormat:@"the plugin %@ describes a malformed model: %@", self.manifest.identifier,
							   modelError.localizedDescription ?: @"not an object"]);
			return nil;
		}
		[models addObject:model];
	}
	return models;
}

- (nullable NSArray<NFKModelOption *> *)optionsIn:(nullable id)value error:(NSError **)error
{
	NSMutableArray<NFKModelOption *> *options = [NSMutableArray array];
	for (id dictionary in [value isKindOfClass:NSArray.class] ? value : @[]) {
		NSError *optionError = nil;
		NFKModelOption *option = [dictionary isKindOfClass:NSDictionary.class] ? [NFKModelOption optionWithDictionary:dictionary error:&optionError] : nil;
		if (option == nil) {
			NFKPluginSetError(error, NFKPluginErrorLoadFailed,
							  [NSString stringWithFormat:@"the plugin %@ describes a malformed option: %@", self.manifest.identifier,
							   optionError.localizedDescription ?: @"not an object"]);
			return nil;
		}
		[options addObject:option];
	}
	return options;
}

- (NSArray<NFKInferenceTool *> *)toolsIn:(nullable id)value
{
	if (self.table->call_tool == NULL) {
		return @[];
	}
	NSMutableArray<NFKInferenceTool *> *tools = [NSMutableArray array];
	__weak NFKPluginCAdapter *weakSelf = self;
	for (id declaration in [value isKindOfClass:NSArray.class] ? value : @[]) {
		NSString *name = [declaration isKindOfClass:NSDictionary.class] && [declaration[@"name"] isKindOfClass:NSString.class] ? declaration[@"name"] : nil;
		if (name == nil) {
			continue;
		}
		NSString *summary = [declaration[@"description"] isKindOfClass:NSString.class] ? declaration[@"description"] : @"";
		NSDictionary *parameters = [declaration[@"parameters"] isKindOfClass:NSDictionary.class] ? declaration[@"parameters"]
			: @{ @"type": @"object", @"properties": @{} };
		[tools addObject:[NFKInferenceTool toolWithName:name description:summary parameters:parameters
												handler:^NFKToolResult *(NSDictionary<NSString *, id> *arguments, NSError **error) {
			NFKPluginCAdapter *adapter = weakSelf;
			if (adapter == nil || adapter.isShutDown) {
				NFKPluginSetError(error, NFKPluginErrorStopped, @"the plugin has shut down");
				return nil;
			}
			return [adapter callToolNamed:name arguments:arguments error:error];
		}]];
	}
	return tools;
}

- (nullable NFKToolResult *)callToolNamed:(NSString *)name arguments:(NSDictionary<NSString *, id> *)arguments error:(NSError **)error
{
	if (![NSJSONSerialization isValidJSONObject:arguments]) {
		if (error != NULL) {
			*error = NFKPluginCError(NFKInferenceErrorDomain, kNFKError_InferenceUnsupported, @"the tool's arguments are not JSON");
		}
		return nil;
	}
	NSData *argumentsJSON = NFKPluginCString([[NSString alloc] initWithData:[NSJSONSerialization dataWithJSONObject:arguments options:0 error:NULL]
																   encoding:NSUTF8StringEncoding]);
	char *resultString = NULL;
	char *errorString = NULL;
	nfk_plugin_status status = self.table->call_tool(self.table->context, name.UTF8String, argumentsJSON.bytes, &resultString, &errorString);
	NSString *result = [self takeString:resultString];
	if (status != NFK_PLUGIN_STATUS_OK) {
		NSError *failure = [self errorForStatus:status errorString:errorString operation:[NSString stringWithFormat:@"tool %@", name]];
		if (error != NULL) {
			*error = failure;
		}
		return nil;
	}
	[self takeString:errorString];
	return [self toolResultFromString:result error:error];
}

/*! A tool result from the plugin's JSON: an object {"text", "isError", "outputs", "images"}, a JSON
	string, or any other text as the result's text. */
- (nullable NFKToolResult *)toolResultFromString:(nullable NSString *)string error:(NSError **)error
{
	id object = NFKPluginCParseJSON(string);
	if ([object isKindOfClass:NSString.class]) {
		return [NFKToolResult resultWithText:object];
	}
	if (![object isKindOfClass:NSDictionary.class]) {
		return [NFKToolResult resultWithText:string ?: @""];
	}
	NSMutableDictionary *entry = [object mutableCopy];
	entry[@"$nfk"] = @"toolResult";
	if (![entry[@"text"] isKindOfClass:NSString.class]) {
		entry[@"text"] = @"";
	}
	NFKInferenceWireCoding *coder = [[NFKInferenceWireCoding alloc] initWithTemporaryDirectory:self.context.temporaryDirectoryURL];
	id result = [coder valueFromJSONValue:entry key:@"toolResult" error:error];
	return [result isKindOfClass:NFKToolResult.class] ? result : nil;
}

#pragma mark Strings, errors, and results

- (const nfk_plugin_host *)hostTable
{
	return &_hostTable;
}

- (nullable NSString *)takeString:(nullable char *)string
{
	if (string == NULL) {
		return nil;
	}
	NSString *copy = [NSString stringWithUTF8String:string];
	if (!self.isShutDown) {
		self.table->free_string(self.table->context, string);
	}
	return copy;
}

- (NSError *)errorForStatus:(nfk_plugin_status)status errorString:(nullable char *)errorString operation:(NSString *)operation
{
	id body = NFKPluginCParseJSON([self takeString:errorString]);
	NSDictionary *described = [body isKindOfClass:NSDictionary.class] ? body : @{};
	NSString *message = [described[@"message"] isKindOfClass:NSString.class] ? described[@"message"]
		: [NSString stringWithFormat:@"the plugin %@'s %@ failed with status %d", self.manifest.identifier, operation, status];
	if (status == NFK_PLUGIN_STATUS_CANCELLED) {
		return NFKPluginCError(NSCocoaErrorDomain, NSUserCancelledError, message);
	}
	if ([described[@"domain"] isKindOfClass:NSString.class] && [described[@"code"] isKindOfClass:NSNumber.class]) {
		return NFKPluginCError(described[@"domain"], [described[@"code"] integerValue], message);
	}
	switch (status) {
		case NFK_PLUGIN_STATUS_UNKNOWN_MODEL:
			return NFKPluginCError(NFKPluginErrorDomain, NFKPluginErrorUnknownModel, message);
		case NFK_PLUGIN_STATUS_UNKNOWN_TOOL:
			return NFKPluginCError(NFKPluginErrorDomain, NFKPluginErrorUnknownTool, message);
		case NFK_PLUGIN_STATUS_INCOMPATIBLE:
			return NFKPluginCError(NFKPluginErrorDomain, NFKPluginErrorIncompatible, message);
		case NFK_PLUGIN_STATUS_MISSING_INPUT:
			return NFKPluginCError(NFKInferenceErrorDomain, kNFKError_InferenceMissingInput, message);
		case NFK_PLUGIN_STATUS_UNSUPPORTED:
		case NFK_PLUGIN_STATUS_INVALID_ARGUMENT:
			return NFKPluginCError(NFKInferenceErrorDomain, kNFKError_InferenceUnsupported, message);
		case NFK_PLUGIN_STATUS_NOT_READY:
			return NFKPluginCError(NFKInferenceErrorDomain, kNFKError_InferenceNotReady, message);
		default:
			return NFKPluginCError(NFKInferenceErrorDomain, kNFKError_InferenceBackendFailure, message);
	}
}

- (nullable NFKInferenceResult *)resultForStatus:(nfk_plugin_status)status
										 payload:(nullable nfk_plugin_payload *)payload
									 errorString:(nullable char *)errorString
										   error:(NSError **)error
{
	NFKPluginCPayloadOwner *owner = nil;
	if (payload != NULL) {
		owner = [[NFKPluginCPayloadOwner alloc] init];
		owner.adapter = self;
		owner.payload = payload;
	}
	if (status != NFK_PLUGIN_STATUS_OK) {
		NSError *failure = [self errorForStatus:status errorString:errorString operation:@"run"];
		if (error != NULL) {
			*error = failure;
		}
		return nil;
	}
	[self takeString:errorString];
	if (payload == NULL) {
		NFKPluginSetError(error, NFKPluginErrorProtocolViolation, [NSString stringWithFormat:@"the plugin %@'s run returned no result", self.manifest.identifier]);
		return nil;
	}
	return [self resultFromPayload:payload copying:NO owner:owner error:error];
}

- (nullable NFKInferenceResult *)resultFromPayload:(const nfk_plugin_payload *)payload
										  copying:(BOOL)copying
											owner:(nullable id)owner
											error:(NSError **)error
{
	if (payload->struct_size < sizeof(nfk_plugin_payload) || payload->json == NULL) {
		NFKPluginSetError(error, NFKPluginErrorProtocolViolation, @"a payload is shorter than plugin ABI 1.0's or carries no JSON");
		return nil;
	}
	id object = NFKPluginCParseJSON([NSString stringWithUTF8String:payload->json]);
	if (![object isKindOfClass:NSDictionary.class]) {
		NFKPluginSetError(error, NFKPluginErrorProtocolViolation, @"a payload's JSON is not an object");
		return nil;
	}
	NFKPluginCPayloadSource *source = [[NFKPluginCPayloadSource alloc] init];
	source.payload = payload;
	source.copying = copying;
	source.owner = owner;
	source.temporaryDirectoryURL = self.context.temporaryDirectoryURL;
	NFKInferenceWireCoding *coder = [[NFKInferenceWireCoding alloc] initWithTemporaryDirectory:self.context.temporaryDirectoryURL];
	coder.attachmentSource = source;
	return [coder resultFromJSONObject:object error:error];
}

#pragma mark NFKPlugin

- (NSArray<NFKModelDescriptor *> *)modelDescriptors
{
	return self.describedModels;
}

- (NSArray<NFKInferenceTool *> *)tools
{
	return self.describedTools;
}

- (NSArray<NFKModelOption *> *)pluginOptions
{
	return self.describedOptions;
}

- (nullable id<NFKInferenceBackend>)backendForModelIdentifier:(NSString *)modelIdentifier
													  options:(NSDictionary<NSString *, id> *)options
														error:(NSError **)error
{
	if (self.isShutDown) {
		NFKPluginSetError(error, NFKPluginErrorStopped, @"the plugin has shut down");
		return nil;
	}
	if (![NSJSONSerialization isValidJSONObject:options]) {
		if (error != NULL) {
			*error = NFKPluginCError(NFKInferenceErrorDomain, kNFKError_InferenceUnsupported, @"the backend's options are not JSON");
		}
		return nil;
	}
	NSData *optionsJSON = NFKPluginCString([[NSString alloc] initWithData:[NSJSONSerialization dataWithJSONObject:options options:0 error:NULL]
																 encoding:NSUTF8StringEncoding]);
	nfk_plugin_backend *handle = NULL;
	char *errorString = NULL;
	nfk_plugin_status status = self.table->create_backend(self.table->context, modelIdentifier.UTF8String, optionsJSON.bytes, &handle, &errorString);
	if (status != NFK_PLUGIN_STATUS_OK || handle == NULL) {
		NSError *failure = status == NFK_PLUGIN_STATUS_OK
			? NFKPluginCError(NFKPluginErrorDomain, NFKPluginErrorLoadFailed, [NSString stringWithFormat:@"the plugin %@ made no backend for %@", self.manifest.identifier, modelIdentifier])
			: [self errorForStatus:status errorString:errorString operation:[NSString stringWithFormat:@"backend for %@", modelIdentifier]];
		if (status == NFK_PLUGIN_STATUS_OK) {
			[self takeString:errorString];
		}
		if (error != NULL) {
			*error = failure;
		}
		return nil;
	}
	[self takeString:errorString];
	return [[NFKPluginCBackend alloc] initWithAdapter:self handle:handle modelIdentifier:modelIdentifier];
}

- (void)willUnload
{
	[self shutdown];
}

- (void)shutdown
{
	@synchronized (self) {
		if (self.isShutDown || self.table == NULL) {
			return;
		}
		self.table->shutdown(self.table->context);
		self.shutDown = YES;
	}
}

@end

#endif
