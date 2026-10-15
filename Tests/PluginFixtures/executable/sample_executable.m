//
//  sample_executable.m
//  InferKit plugin fixtures
//
//  An executable plugin that uses Foundation and no InferKit. It speaks the plugin protocol by hand,
//  which shows a plugin in any language can implement it from Docs/agent-reference/plugins.md:
//  - It hands the host a socket of its own as its first frame, {"type": "channel"}, on the descriptor
//    INFERKIT_PLUGIN_CHANNEL_FD names, and speaks on the other end of that socket.
//  - It frames each message as a 12-byte little-endian header (magic "NFKP", JSON length, descriptor
//    count, flags) and the JSON, with the descriptors in the same sendmsg as SCM_RIGHTS.
//  - It reads attachments from the shared memory the host passes, and returns large results in
//    shared memory of its own.
//
//  The models and the tool behave as the other fixtures' do:
//  - "invert" inverts a 32BGRA pixel buffer under "image" (each color byte becomes 255 minus itself,
//    alpha kept) and a float32 or float64 array under "values" (each element becomes 1 minus itself).
//  - "slow" counts "steps" steps of "stepMilliseconds" each, reports progress with a partial result
//    {"step": n} after each, and stops when the run is cancelled. "exitAtStep" ends the process there.
//  - "echo" returns the "text" argument after the plugin's "echoPrefix" option.
//  Every result carries "pluginVersion".
//

#import <Foundation/Foundation.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#ifndef SAMPLE_PLUGIN_VERSION
#define SAMPLE_PLUGIN_VERSION "1.0.0"
#endif

static NSString * const SampleIdentifier = @"com.example.inferkit.sample-executable";
static NSString * const SampleVersion = @SAMPLE_PLUGIN_VERSION;
static NSString * const SamplePluginErrorDomain = @"NFKPluginErrorDomain";
static NSString * const SampleInferenceErrorDomain = @"NFKInferenceErrorDomain";

static const uint32_t SampleFrameMagic = 0x504B464E;
static const uint32_t SampleMaximumJSONLength = 16u * 1024u * 1024u;
static const uint16_t SampleMaximumDescriptors = 64;
static const double SampleFormat32BGRA = 1111970369.0;
static const NSInteger SampleWeightBytes = 4096;

enum {
	SampleFrameHeaderLength = 12,
	SamplePluginErrorProtocolUnsupported = 12,
	SamplePluginErrorUnknownModel = 14,
	SamplePluginErrorUnknownTool = 15,
	SampleInferenceErrorMissingInput = 2,
	SampleInferenceErrorUnsupported = 4,
};

/*! The inner error object of an InferKit error envelope, which keeps the domain and code intact. */
static NSDictionary *SampleError(NSString *domain, NSInteger code, NSString *message)
{
	return @{ @"message": message, @"type": @"server_error", @"inferkit_domain": domain, @"inferkit_code": @(code),
			  @"inferkit_user_info": @{} };
}

static void SampleConfigureSocket(int descriptor)
{
	int on = 1;
	int size = 1 << 20;
	setsockopt(descriptor, SOL_SOCKET, SO_NOSIGPIPE, &on, sizeof on);
	setsockopt(descriptor, SOL_SOCKET, SO_SNDBUF, &size, sizeof size);
	setsockopt(descriptor, SOL_SOCKET, SO_RCVBUF, &size, sizeof size);
	fcntl(descriptor, F_SETFD, FD_CLOEXEC);
}

/*! Sends one frame: the header and the JSON in one sendmsg whose control data carries descriptors. */
static BOOL SampleSendFrame(int socket, NSDictionary *message, NSArray<NSNumber *> *descriptors)
{
	NSData *JSON = [NSJSONSerialization dataWithJSONObject:message options:0 error:NULL];
	if (JSON == nil || JSON.length > SampleMaximumJSONLength || descriptors.count > SampleMaximumDescriptors) {
		return NO;
	}
	uint32_t length = (uint32_t)JSON.length;
	uint16_t count = (uint16_t)descriptors.count;
	uint8_t header[SampleFrameHeaderLength] = {
		(uint8_t)SampleFrameMagic, (uint8_t)(SampleFrameMagic >> 8), (uint8_t)(SampleFrameMagic >> 16), (uint8_t)(SampleFrameMagic >> 24),
		(uint8_t)length, (uint8_t)(length >> 8), (uint8_t)(length >> 16), (uint8_t)(length >> 24),
		(uint8_t)count, (uint8_t)(count >> 8), 0, 0,
	};
	NSMutableData *frame = [NSMutableData dataWithBytes:header length:sizeof header];
	[frame appendData:JSON];
	size_t controlLength = count > 0 ? CMSG_SPACE(count * sizeof(int)) : 0;
	NSMutableData *control = [NSMutableData dataWithLength:controlLength];
	if (count > 0) {
		struct cmsghdr *part = (struct cmsghdr *)control.mutableBytes;
		part->cmsg_level = SOL_SOCKET;
		part->cmsg_type = SCM_RIGHTS;
		part->cmsg_len = CMSG_LEN(count * sizeof(int));
		int *slots = (int *)CMSG_DATA(part);
		for (NSUInteger index = 0; index < count; index++) {
			slots[index] = descriptors[index].intValue;
		}
	}
	size_t sent = 0;
	while (sent < frame.length) {
		struct iovec vector = { (uint8_t *)frame.mutableBytes + sent, frame.length - sent };
		struct msghdr outgoing = { 0 };
		outgoing.msg_iov = &vector;
		outgoing.msg_iovlen = 1;
		// The descriptors ride the frame's first byte, so only the first sendmsg carries them.
		if (sent == 0 && controlLength > 0) {
			outgoing.msg_control = control.mutableBytes;
			outgoing.msg_controllen = (socklen_t)controlLength;
		}
		ssize_t written = sendmsg(socket, &outgoing, 0);
		if (written < 0 && errno == EINTR) {
			continue;
		}
		if (written <= 0) {
			return NO;
		}
		sent += (size_t)written;
	}
	return YES;
}

/*! Maps a region the host passed, and closes the descriptor; the mapping outlives it. */
static NSData *SampleMapRegion(int descriptor)
{
	struct stat information;
	if (fstat(descriptor, &information) != 0 || information.st_size <= 0) {
		close(descriptor);
		return nil;
	}
	size_t length = (size_t)information.st_size;
	void *bytes = mmap(NULL, length, PROT_READ, MAP_SHARED, descriptor, 0);
	close(descriptor);
	if (bytes == MAP_FAILED) {
		return nil;
	}
	return [[NSData alloc] initWithBytesNoCopy:bytes length:length deallocator:^(void *mapped, NSUInteger mappedLength) {
		munmap(mapped, mappedLength);
	}];
}

/*! A shared-memory region holding bytes, unlinked at once so only the descriptor reaches it, or -1. */
static int SampleMakeRegion(NSData *bytes)
{
	for (int attempt = 0; attempt < 8; attempt++) {
		char name[32];
		snprintf(name, sizeof name, "/nfks-%08x%08x", arc4random(), arc4random());
		int descriptor = shm_open(name, O_RDWR | O_CREAT | O_EXCL, 0600);
		if (descriptor < 0 && errno == EEXIST) {
			continue;
		}
		if (descriptor < 0) {
			return -1;
		}
		shm_unlink(name);
		fcntl(descriptor, F_SETFD, FD_CLOEXEC);
		void *mapped = ftruncate(descriptor, (off_t)bytes.length) == 0
			? mmap(NULL, bytes.length, PROT_READ | PROT_WRITE, MAP_SHARED, descriptor, 0) : MAP_FAILED;
		if (mapped == MAP_FAILED) {
			close(descriptor);
			return -1;
		}
		memcpy(mapped, bytes.bytes, bytes.length);
		munmap(mapped, bytes.length);
		return descriptor;
	}
	return -1;
}

#pragma mark - Backends

@interface SampleBackend : NSObject
@property (nonatomic, copy) NSString *model;
@property (atomic, assign) BOOL loaded;
@end

@implementation SampleBackend
@end

/*! What one run returns: its result object and the regions its attachments are in. */
@interface SampleRunOutput : NSObject
@property (nonatomic, strong) NSMutableDictionary *outputs;
@property (nonatomic, strong) NSMutableArray<NSNumber *> *regions;
@property (nonatomic, strong, nullable) NSDictionary *error;
@property (nonatomic, assign) BOOL cancelled;
@end

@implementation SampleRunOutput

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_outputs = [NSMutableDictionary dictionaryWithObject:SampleVersion forKey:@"pluginVersion"];
		_regions = [NSMutableArray array];
	}
	return self;
}

/*! Adds bytes as the next attachment and returns its index, or -1. */
- (NSInteger)attach:(NSData *)bytes
{
	int region = SampleMakeRegion(bytes);
	if (region < 0) {
		return -1;
	}
	[self.regions addObject:@(region)];
	return (NSInteger)self.regions.count - 1;
}

- (void)failWithDomain:(NSString *)domain code:(NSInteger)code message:(NSString *)message
{
	self.error = SampleError(domain, code, message);
}

@end

#pragma mark - Session

@interface SampleSession : NSObject
@property (nonatomic, assign) int socket;
@property (nonatomic, strong) NSObject *sendLock;
@property (nonatomic, strong) NSMutableDictionary<NSNumber *, SampleBackend *> *backends;
@property (nonatomic, strong) NSMutableSet<NSNumber *> *cancelledRuns;
@property (nonatomic, assign) NSInteger nextHandle;
@property (nonatomic, copy) NSString *echoPrefix;
@property (nonatomic, strong) dispatch_queue_t runQueue;
@end

@implementation SampleSession

- (instancetype)initWithSocket:(int)socket
{
	self = [super init];
	if (self != nil) {
		_socket = socket;
		_sendLock = [[NSObject alloc] init];
		_backends = [NSMutableDictionary dictionary];
		_cancelledRuns = [NSMutableSet set];
		_nextHandle = 1;
		_echoPrefix = @"";
		dispatch_queue_attr_t attributes = dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_CONCURRENT, QOS_CLASS_USER_INITIATED, 0);
		_runQueue = dispatch_queue_create("com.example.inferkit.sample-executable.runs", attributes);
	}
	return self;
}

- (void)send:(NSDictionary *)message descriptors:(NSArray<NSNumber *> *)descriptors
{
	@synchronized (self.sendLock) {
		if (!SampleSendFrame(self.socket, message, descriptors)) {
			_exit(3);
		}
	}
}

- (void)reply:(id)identifier result:(NSDictionary *)result
{
	[self send:@{ @"type": @"reply", @"id": identifier, @"result": result } descriptors:@[]];
}

- (void)reply:(id)identifier error:(NSDictionary *)error
{
	[self send:@{ @"type": @"reply", @"id": identifier, @"error": error } descriptors:@[]];
}

#pragma mark Reading

/*! Reads frames until the host closes the socket, and exits. */
- (void)run
{
	NSMutableData *pending = [NSMutableData data];
	NSMutableArray<NSNumber *> *queuedDescriptors = [NSMutableArray array];
	NSMutableData *buffer = [NSMutableData dataWithLength:65536];
	NSMutableData *control = [NSMutableData dataWithLength:CMSG_SPACE(SampleMaximumDescriptors * sizeof(int))];
	for (;;) {
		@autoreleasepool {
			struct iovec vector = { buffer.mutableBytes, buffer.length };
			struct msghdr header = { 0 };
			header.msg_iov = &vector;
			header.msg_iovlen = 1;
			header.msg_control = control.mutableBytes;
			header.msg_controllen = (socklen_t)control.length;
			ssize_t received = recvmsg(self.socket, &header, 0);
			if (received < 0 && errno == EINTR) {
				continue;
			}
			if (received <= 0) {
				exit(0);
			}
			if ((header.msg_flags & MSG_CTRUNC) != 0) {
				_exit(2);
			}
			for (struct cmsghdr *part = CMSG_FIRSTHDR(&header); part != NULL; part = CMSG_NXTHDR(&header, part)) {
				if (part->cmsg_level != SOL_SOCKET || part->cmsg_type != SCM_RIGHTS) {
					continue;
				}
				size_t count = (part->cmsg_len - CMSG_LEN(0)) / sizeof(int);
				const int *descriptors = (const int *)CMSG_DATA(part);
				for (size_t index = 0; index < count; index++) {
					fcntl(descriptors[index], F_SETFD, FD_CLOEXEC);
					[queuedDescriptors addObject:@(descriptors[index])];
				}
			}
			[pending appendBytes:buffer.bytes length:(NSUInteger)received];
			[self deliverFramesIn:pending descriptors:queuedDescriptors];
		}
	}
}

- (void)deliverFramesIn:(NSMutableData *)pending descriptors:(NSMutableArray<NSNumber *> *)queuedDescriptors
{
	while (pending.length >= SampleFrameHeaderLength) {
		const uint8_t *bytes = pending.bytes;
		uint32_t magic = (uint32_t)bytes[0] | (uint32_t)bytes[1] << 8 | (uint32_t)bytes[2] << 16 | (uint32_t)bytes[3] << 24;
		uint32_t length = (uint32_t)bytes[4] | (uint32_t)bytes[5] << 8 | (uint32_t)bytes[6] << 16 | (uint32_t)bytes[7] << 24;
		uint16_t count = (uint16_t)(bytes[8] | bytes[9] << 8);
		if (magic != SampleFrameMagic || length > SampleMaximumJSONLength || count > SampleMaximumDescriptors) {
			_exit(2);
		}
		if (pending.length < SampleFrameHeaderLength + length) {
			return;
		}
		if (queuedDescriptors.count < count) {
			_exit(2);
		}
		NSArray<NSNumber *> *descriptors = [queuedDescriptors subarrayWithRange:NSMakeRange(0, count)];
		[queuedDescriptors removeObjectsInRange:NSMakeRange(0, count)];
		NSData *JSON = [pending subdataWithRange:NSMakeRange(SampleFrameHeaderLength, length)];
		[pending replaceBytesInRange:NSMakeRange(0, SampleFrameHeaderLength + length) withBytes:NULL length:0];
		NSDictionary *message = [NSJSONSerialization JSONObjectWithData:JSON options:0 error:NULL];
		if (![message isKindOfClass:NSDictionary.class] || ![message[@"type"] isKindOfClass:NSString.class]) {
			_exit(2);
		}
		[self handleMessage:message descriptors:descriptors];
	}
}

#pragma mark Messages

- (void)handleMessage:(NSDictionary *)message descriptors:(NSArray<NSNumber *> *)descriptors
{
	NSString *type = message[@"type"];
	id identifier = [message[@"id"] isKindOfClass:NSNumber.class] ? message[@"id"] : nil;
	if ([type isEqualToString:@"run"] && identifier != nil) {
		[self startRun:message identifier:identifier descriptors:descriptors];
		return;
	}
	for (NSNumber *descriptor in descriptors) {
		close(descriptor.intValue);
	}
	if ([type isEqualToString:@"cancel"]) {
		[self cancelRun:identifier];
		return;
	}
	if (identifier == nil) {
		return;
	}
	if ([type isEqualToString:@"hello"]) {
		[self hello:message identifier:identifier];
	} else if ([type isEqualToString:@"describe"]) {
		[self describeWithIdentifier:identifier];
	} else if ([type isEqualToString:@"build"]) {
		[self build:message identifier:identifier];
	} else if ([type isEqualToString:@"prepare"]) {
		[self prepare:message identifier:identifier];
	} else if ([type isEqualToString:@"unload"]) {
		[self unload:message identifier:identifier];
	} else if ([type isEqualToString:@"release"]) {
		[self releaseBackend:message identifier:identifier];
	} else if ([type isEqualToString:@"status"]) {
		[self status:message identifier:identifier];
	} else if ([type isEqualToString:@"callTool"]) {
		[self callTool:message identifier:identifier];
	} else if ([type isEqualToString:@"ping"]) {
		[self reply:identifier result:@{}];
	} else if ([type isEqualToString:@"shutdown"]) {
		[self reply:identifier result:@{}];
		exit(0);
	} else {
		[self reply:identifier error:SampleError(SamplePluginErrorDomain, SamplePluginErrorProtocolUnsupported,
												 [NSString stringWithFormat:@"the plugin does not support \"%@\"", type])];
	}
}

- (void)hello:(NSDictionary *)message identifier:(id)identifier
{
	NSDictionary *protocol = [message[@"protocol"] isKindOfClass:NSDictionary.class] ? message[@"protocol"] : @{};
	if ([protocol[@"major"] integerValue] != 1) {
		[self reply:identifier error:SampleError(SamplePluginErrorDomain, SamplePluginErrorProtocolUnsupported, @"the plugin speaks protocol 1")];
		return;
	}
	NSDictionary *options = [message[@"options"] isKindOfClass:NSDictionary.class] ? message[@"options"] : @{};
	self.echoPrefix = [options[@"echoPrefix"] isKindOfClass:NSString.class] ? options[@"echoPrefix"] : @"";
	[self reply:identifier result:@{
		@"protocol": @{ @"major": @1, @"minor": @0 },
		@"plugin": @{ @"id": SampleIdentifier, @"version": SampleVersion },
		@"inferkitAPI": @"1.0",
		@"capabilities": @[ @"attachments" ],
		@"pid": @(getpid()),
	}];
	NSString *ready = [NSString stringWithFormat:@"sample executable %@ ready", SampleVersion];
	[self send:@{ @"type": @"log", @"level": @1, @"message": ready } descriptors:@[]];
}

- (void)describeWithIdentifier:(id)identifier
{
	NSDictionary *pluginVersion = @{ @"key": @"pluginVersion", @"valueType": @"text", @"required": @YES };
	NSDictionary *invert = @{
		@"id": @"invert", @"displayName": @"Invert", @"family": @"sample-invert", @"engine": @"plugin",
		@"summary": @"Inverts a 32BGRA image or a float array.", @"tasks": @[ @"image-to-image" ],
		@"inputs": @[ @{ @"key": @"image", @"valueType": @"image" }, @{ @"key": @"values", @"valueType": @"embedding" } ],
		@"outputs": @[ @{ @"key": @"image", @"valueType": @"image" }, @{ @"key": @"values", @"valueType": @"embedding" }, pluginVersion ],
	};
	NSDictionary *slow = @{
		@"id": @"slow", @"displayName": @"Slow", @"family": @"sample-slow", @"engine": @"plugin",
		@"summary": @"Counts steps, reporting progress after each, until it finishes or is cancelled.",
		@"outputs": @[ @{ @"key": @"steps", @"valueType": @"integer", @"required": @YES }, pluginVersion ],
		@"parameters": @[
			@{ @"key": @"steps", @"kind": @"integer", @"default": @10, @"minimum": @1 },
			@{ @"key": @"stepMilliseconds", @"kind": @"integer", @"default": @20, @"minimum": @0 },
			@{ @"key": @"exitAtStep", @"kind": @"integer", @"minimum": @1,
			   @"help": @"Ends the plugin's process at this step. Only for a plugin in a process of its own." },
		],
	};
	NSDictionary *echo = @{
		@"name": @"echo", @"description": @"Returns the text it is given, after the echoPrefix option.",
		@"parameters": @{ @"type": @"object", @"properties": @{ @"text": @{ @"type": @"string" } }, @"required": @[ @"text" ] },
	};
	NSDictionary *prefix = @{ @"key": @"echoPrefix", @"kind": @"text", @"default": @"", @"title": @"Echo prefix" };
	[self reply:identifier result:@{ @"models": @[ invert, slow ], @"tools": @[ echo ], @"options": @[ prefix ] }];
}

- (nullable SampleBackend *)backendFor:(NSDictionary *)message
{
	@synchronized (self) {
		return [message[@"backend"] isKindOfClass:NSNumber.class] ? self.backends[message[@"backend"]] : nil;
	}
}

- (NSDictionary *)stateOfBackend:(SampleBackend *)backend
{
	return @{ @"loaded": @(backend.loaded), @"residentBytes": @(backend.loaded ? SampleWeightBytes : 0), @"peakBytes": @(SampleWeightBytes) };
}

- (void)build:(NSDictionary *)message identifier:(id)identifier
{
	NSString *model = [message[@"model"] isKindOfClass:NSString.class] ? message[@"model"] : @"";
	BOOL invert = [model isEqualToString:@"invert"];
	if (!invert && ![model isEqualToString:@"slow"]) {
		[self reply:identifier error:SampleError(SamplePluginErrorDomain, SamplePluginErrorUnknownModel,
												 [NSString stringWithFormat:@"the plugin has no model \"%@\"", model])];
		return;
	}
	SampleBackend *backend = [[SampleBackend alloc] init];
	backend.model = model;
	NSNumber *handle;
	@synchronized (self) {
		handle = @(self.nextHandle);
		self.nextHandle += 1;
		self.backends[handle] = backend;
	}
	NSMutableDictionary *result = [[self stateOfBackend:backend] mutableCopy];
	result[@"backend"] = handle;
	result[@"ready"] = @YES;
	result[@"inputs"] = invert ? @[ @"image", @"values" ] : @[];
	result[@"parameters"] = invert ? @[] : @[ @"steps", @"stepMilliseconds", @"exitAtStep" ];
	result[@"outputs"] = invert ? @[ @"image", @"values", @"pluginVersion" ] : @[ @"steps", @"pluginVersion" ];
	result[@"modelInfo"] = @{ @"architecture": invert ? @"sample-invert" : @"sample-slow" };
	[self reply:identifier result:result];
}

- (BOOL)replyUnknownBackend:(nullable SampleBackend *)backend identifier:(id)identifier
{
	if (backend != nil) {
		return NO;
	}
	[self reply:identifier error:SampleError(SamplePluginErrorDomain, SamplePluginErrorUnknownModel, @"the plugin has no such backend")];
	return YES;
}

- (void)loadBackend:(SampleBackend *)backend handle:(id)handle
{
	@synchronized (backend) {
		if (backend.loaded) {
			return;
		}
		backend.loaded = YES;
	}
	[self send:@{ @"type": @"event", @"event": @"loaded", @"backend": handle, @"residentBytes": @(SampleWeightBytes) } descriptors:@[]];
}

- (void)prepare:(NSDictionary *)message identifier:(id)identifier
{
	SampleBackend *backend = [self backendFor:message];
	if ([self replyUnknownBackend:backend identifier:identifier]) {
		return;
	}
	[self loadBackend:backend handle:message[@"backend"]];
	[self reply:identifier result:[self stateOfBackend:backend]];
}

- (void)unload:(NSDictionary *)message identifier:(id)identifier
{
	SampleBackend *backend = [self backendFor:message];
	if ([self replyUnknownBackend:backend identifier:identifier]) {
		return;
	}
	BOOL wasLoaded;
	@synchronized (backend) {
		wasLoaded = backend.loaded;
		backend.loaded = NO;
	}
	[self reply:identifier result:@{}];
	if (wasLoaded) {
		[self send:@{ @"type": @"event", @"event": @"unloaded", @"backend": message[@"backend"], @"residentBytes": @0 } descriptors:@[]];
	}
}

- (void)releaseBackend:(NSDictionary *)message identifier:(id)identifier
{
	@synchronized (self) {
		if ([message[@"backend"] isKindOfClass:NSNumber.class]) {
			[self.backends removeObjectForKey:message[@"backend"]];
		}
	}
	[self reply:identifier result:@{}];
}

- (void)status:(NSDictionary *)message identifier:(id)identifier
{
	SampleBackend *backend = [self backendFor:message];
	if ([self replyUnknownBackend:backend identifier:identifier]) {
		return;
	}
	NSMutableDictionary *result = [[self stateOfBackend:backend] mutableCopy];
	result[@"backendStatus"] = @{ @"model": backend.model };
	[self reply:identifier result:result];
}

- (void)callTool:(NSDictionary *)message identifier:(id)identifier
{
	if (![message[@"tool"] isEqual:@"echo"]) {
		[self reply:identifier error:SampleError(SamplePluginErrorDomain, SamplePluginErrorUnknownTool, @"the plugin has no such tool")];
		return;
	}
	NSDictionary *arguments = [message[@"arguments"] isKindOfClass:NSDictionary.class] ? message[@"arguments"] : @{};
	if (![arguments[@"text"] isKindOfClass:NSString.class]) {
		[self reply:identifier error:SampleError(SampleInferenceErrorDomain, SampleInferenceErrorMissingInput, @"echo needs a string \"text\"")];
		return;
	}
	NSString *text = [self.echoPrefix stringByAppendingString:arguments[@"text"]];
	NSDictionary *toolResult = @{ @"$nfk": @"toolResult", @"text": text, @"isError": @NO, @"images": @[], @"outputs": @{} };
	[self reply:identifier result:@{ @"toolResult": toolResult }];
}

- (void)cancelRun:(nullable id)identifier
{
	if (identifier == nil) {
		return;
	}
	@synchronized (self) {
		[self.cancelledRuns addObject:identifier];
	}
}

- (BOOL)isRunCancelled:(id)identifier
{
	@synchronized (self) {
		return [self.cancelledRuns containsObject:identifier];
	}
}

#pragma mark Runs

- (void)startRun:(NSDictionary *)message identifier:(id)identifier descriptors:(NSArray<NSNumber *> *)descriptors
{
	NSMutableArray<NSData *> *attachments = [NSMutableArray arrayWithCapacity:descriptors.count];
	for (NSNumber *descriptor in descriptors) {
		[attachments addObject:SampleMapRegion(descriptor.intValue) ?: [NSData data]];
	}
	SampleBackend *backend = [self backendFor:message];
	if ([self replyUnknownBackend:backend identifier:identifier]) {
		return;
	}
	dispatch_async(self.runQueue, ^{
		[self loadBackend:backend handle:message[@"backend"]];
		NSDictionary *request = [message[@"request"] isKindOfClass:NSDictionary.class] ? message[@"request"] : @{};
		SampleRunOutput *output = [[SampleRunOutput alloc] init];
		if ([backend.model isEqualToString:@"invert"]) {
			[self invert:request attachments:attachments output:output];
		} else {
			[self countSteps:request identifier:identifier output:output];
		}
		[self finishRun:identifier output:output];
		@synchronized (self) {
			[self.cancelledRuns removeObject:identifier];
		}
	});
}

- (void)finishRun:(id)identifier output:(SampleRunOutput *)output
{
	if (output.cancelled) {
		[self send:@{ @"type": @"reply", @"id": identifier, @"cancelled": @YES } descriptors:@[]];
	} else if (output.error != nil) {
		[self reply:identifier error:output.error];
	} else {
		[self send:@{ @"type": @"reply", @"id": identifier, @"result": @{ @"outputs": output.outputs } } descriptors:output.regions];
	}
	for (NSNumber *region in output.regions) {
		close(region.intValue);
	}
}

/*! A tagged value's bytes: base64 inline, or the attachment its index names. */
- (nullable NSData *)bytesOf:(NSDictionary *)entry base64:(nullable id)base64 attachments:(NSArray<NSData *> *)attachments
{
	if ([entry[@"$nfk"] isEqual:@"buffer"]) {
		NSInteger index = [entry[@"index"] isKindOfClass:NSNumber.class] ? [entry[@"index"] integerValue] : -1;
		NSUInteger length = [entry[@"length"] unsignedIntegerValue];
		if (index < 0 || index >= (NSInteger)attachments.count || length > attachments[index].length) {
			return nil;
		}
		return [attachments[index] subdataWithRange:NSMakeRange(0, length)];
	}
	return [base64 isKindOfClass:NSString.class] ? [[NSData alloc] initWithBase64EncodedString:base64 options:0] : nil;
}

- (void)invert:(NSDictionary *)request attachments:(NSArray<NSData *> *)attachments output:(SampleRunOutput *)output
{
	NSDictionary *inputs = [request[@"inputs"] isKindOfClass:NSDictionary.class] ? request[@"inputs"] : @{};
	NSDictionary *image = [inputs[@"image"] isKindOfClass:NSDictionary.class] ? inputs[@"image"] : nil;
	NSDictionary *values = [inputs[@"values"] isKindOfClass:NSDictionary.class] ? inputs[@"values"] : nil;
	if (image == nil && values == nil) {
		[output failWithDomain:SampleInferenceErrorDomain code:SampleInferenceErrorMissingInput message:@"invert needs \"image\" or \"values\""];
		return;
	}
	if (image != nil) {
		[self invertImage:image attachments:attachments output:output];
	}
	if (values != nil && output.error == nil) {
		[self invertValues:values attachments:attachments output:output];
	}
}

- (void)invertImage:(NSDictionary *)entry attachments:(NSArray<NSData *> *)attachments output:(SampleRunOutput *)output
{
	BOOL attached = [entry[@"$nfk"] isEqual:@"buffer"] && [entry[@"as"] isEqual:@"pixelBuffer"];
	BOOL inlined = [entry[@"$nfk"] isEqual:@"pixelBuffer"];
	NSDictionary *plane = [entry[@"planes"] isKindOfClass:NSArray.class] ? [entry[@"planes"] firstObject] : nil;
	NSUInteger width = [entry[@"width"] unsignedIntegerValue];
	NSUInteger height = [entry[@"height"] unsignedIntegerValue];
	NSUInteger bytesPerRow = [plane[@"bytesPerRow"] unsignedIntegerValue];
	NSUInteger rows = [plane[@"rows"] unsignedIntegerValue];
	NSUInteger offset = [plane[@"offset"] unsignedIntegerValue];
	NSData *source = (attached || inlined) ? [self bytesOf:entry base64:plane[@"base64"] attachments:attachments] : nil;
	NSUInteger length = bytesPerRow * rows;
	if (source == nil || [entry[@"format"] doubleValue] != SampleFormat32BGRA || width == 0 || bytesPerRow < width * 4
		|| rows < height || offset + length > source.length) {
		[output failWithDomain:SampleInferenceErrorDomain code:SampleInferenceErrorUnsupported message:@"invert takes a 32BGRA pixel buffer under \"image\""];
		return;
	}
	NSMutableData *inverted = [NSMutableData dataWithLength:length];
	const uint8_t *from = (const uint8_t *)source.bytes + offset;
	uint8_t *to = inverted.mutableBytes;
	for (NSUInteger at = 0; at < length; at++) {
		NSUInteger column = at % bytesPerRow;
		BOOL color = column < width * 4 && column % 4 != 3;
		to[at] = color ? (uint8_t)(255 - from[at]) : from[at];
	}
	NSMutableDictionary *result = [@{ @"format": entry[@"format"], @"width": @(width), @"height": @(height) } mutableCopy];
	if (attached) {
		NSInteger index = [output attach:inverted];
		result[@"$nfk"] = @"buffer";
		result[@"as"] = @"pixelBuffer";
		result[@"index"] = @(index);
		result[@"length"] = @(length);
		result[@"planes"] = @[ @{ @"offset": @0, @"bytesPerRow": @(bytesPerRow), @"rows": @(rows) } ];
	} else {
		result[@"$nfk"] = @"pixelBuffer";
		result[@"planes"] = @[ @{ @"bytesPerRow": @(bytesPerRow), @"rows": @(rows), @"base64": [inverted base64EncodedStringWithOptions:0] } ];
	}
	output.outputs[@"image"] = result;
}

- (void)invertValues:(NSDictionary *)entry attachments:(NSArray<NSData *> *)attachments output:(SampleRunOutput *)output
{
	BOOL attached = [entry[@"$nfk"] isEqual:@"buffer"];
	NSString *kind = attached ? entry[@"as"] : entry[@"$nfk"];
	BOOL singles = [kind isEqual:@"float32Array"];
	BOOL doubles = [kind isEqual:@"float64Array"];
	NSData *source = singles || doubles ? [self bytesOf:entry base64:entry[@"base64"] attachments:attachments] : nil;
	NSUInteger elementSize = singles ? sizeof(float) : sizeof(double);
	if (source == nil || source.length % elementSize != 0) {
		[output failWithDomain:SampleInferenceErrorDomain code:SampleInferenceErrorUnsupported message:@"invert takes a float32 or float64 array under \"values\""];
		return;
	}
	NSUInteger count = source.length / elementSize;
	NSMutableData *inverted = [NSMutableData dataWithLength:source.length];
	for (NSUInteger index = 0; index < count; index++) {
		if (singles) {
			float value;
			memcpy(&value, (const uint8_t *)source.bytes + index * elementSize, sizeof value);
			value = 1.0f - value;
			memcpy((uint8_t *)inverted.mutableBytes + index * elementSize, &value, sizeof value);
		} else {
			double value;
			memcpy(&value, (const uint8_t *)source.bytes + index * elementSize, sizeof value);
			value = 1.0 - value;
			memcpy((uint8_t *)inverted.mutableBytes + index * elementSize, &value, sizeof value);
		}
	}
	if (attached) {
		NSInteger index = [output attach:inverted];
		output.outputs[@"values"] = @{ @"$nfk": @"buffer", @"as": kind, @"index": @(index), @"length": @(inverted.length), @"count": @(count) };
	} else {
		output.outputs[@"values"] = @{ @"$nfk": kind, @"base64": [inverted base64EncodedStringWithOptions:0] };
	}
}

- (void)countSteps:(NSDictionary *)request identifier:(id)identifier output:(SampleRunOutput *)output
{
	NSDictionary *parameters = [request[@"parameters"] isKindOfClass:NSDictionary.class] ? request[@"parameters"] : @{};
	NSInteger steps = [parameters[@"steps"] isKindOfClass:NSNumber.class] ? [parameters[@"steps"] integerValue] : 10;
	NSInteger stepMilliseconds = [parameters[@"stepMilliseconds"] isKindOfClass:NSNumber.class] ? [parameters[@"stepMilliseconds"] integerValue] : 20;
	NSInteger exitAtStep = [parameters[@"exitAtStep"] isKindOfClass:NSNumber.class] ? [parameters[@"exitAtStep"] integerValue] : 0;
	if (steps < 1 || stepMilliseconds < 0) {
		[output failWithDomain:SampleInferenceErrorDomain code:SampleInferenceErrorUnsupported message:@"steps is at least 1 and stepMilliseconds at least 0"];
		return;
	}
	for (NSInteger step = 1; step <= steps; step++) {
		[NSThread sleepForTimeInterval:stepMilliseconds / 1000.0];
		if ([self isRunCancelled:identifier]) {
			output.cancelled = YES;
			return;
		}
		if (step == exitAtStep) {
			_exit(70);
		}
		[self send:@{ @"type": @"progress", @"id": identifier, @"progress": @((double)step / (double)steps),
					  @"partial": @{ @"outputs": @{ @"step": @(step) } } } descriptors:@[]];
	}
	output.outputs[@"steps"] = @(steps);
}

@end

#pragma mark - Entry

int main(int argc, const char *argv[])
{
	@autoreleasepool {
		signal(SIGPIPE, SIG_IGN);
		const char *named = getenv("INFERKIT_PLUGIN_CHANNEL_FD");
		int inherited = named != NULL ? atoi(named) : 3;
		int pair[2];
		if (socketpair(AF_UNIX, SOCK_STREAM, 0, pair) != 0) {
			return 1;
		}
		SampleConfigureSocket(pair[0]);
		SampleConfigureSocket(pair[1]);
		BOOL handedOff = SampleSendFrame(inherited, @{ @"type": @"channel" }, @[ @(pair[1]) ]);
		close(inherited);
		close(pair[1]);
		if (!handedOff) {
			return 1;
		}
		[[[SampleSession alloc] initWithSocket:pair[0]] run];
	}
	return 0;
}
