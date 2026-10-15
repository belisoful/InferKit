//
//  NFKPluginChannel.m
//  InferKit
//

#import "NFKPluginChannel.h"
#import "NFKPluginWire.h"

#if TARGET_OS_OSX

#import <sys/socket.h>
#import <sys/un.h>
#import <sys/stat.h>
#import <fcntl.h>
#import <poll.h>
#import <unistd.h>
#import <errno.h>

/*! The bytes one recvmsg reads at most. */
static const size_t NFKPluginChannelReadChunk = 256u * 1024u;

/*! The socket buffer size each end asks for, so a large frame moves in few system calls. */
static const int NFKPluginChannelSocketBuffer = 1024 * 1024;

static const NSTimeInterval NFKPluginChannelDefaultSendTimeout = 30;

/*! How often a blocked send wakes to see whether the channel closed, in milliseconds. */
static const int NFKPluginChannelPollMilliseconds = 100;

static BOOL NFKPluginChannelPOSIXError(NSError **outError, NSInteger code, int posixCode, NSString *reason)
{
	if (outError != NULL) {
		NSString *description = posixCode != 0 ? [NSString stringWithFormat:@"%@: %s", reason, strerror(posixCode)] : reason;
		NSMutableDictionary *userInfo = [NSMutableDictionary dictionaryWithObject:description forKey:NSLocalizedDescriptionKey];
		if (posixCode != 0) {
			userInfo[NSUnderlyingErrorKey] = [NSError errorWithDomain:NSPOSIXErrorDomain code:posixCode userInfo:nil];
		}
		*outError = [NSError errorWithDomain:NFKPluginErrorDomain code:code userInfo:userInfo];
	}
	return NO;
}

/*! Sets the options every channel socket carries: close-on-exec, SIGPIPE suppressed, and larger buffers.
	nonblocking also sets O_NONBLOCK, which a channel needs and an inheriting child's end does not. */
static void NFKPluginChannelConfigureSocket(int descriptor, BOOL nonblocking)
{
	fcntl(descriptor, F_SETFD, FD_CLOEXEC);
	int on = 1;
	setsockopt(descriptor, SOL_SOCKET, SO_NOSIGPIPE, &on, sizeof on);
	int size = NFKPluginChannelSocketBuffer;
	setsockopt(descriptor, SOL_SOCKET, SO_SNDBUF, &size, sizeof size);
	setsockopt(descriptor, SOL_SOCKET, SO_RCVBUF, &size, sizeof size);
	if (nonblocking) {
		fcntl(descriptor, F_SETFL, fcntl(descriptor, F_GETFL) | O_NONBLOCK);
	}
}

/*! Whether descriptor is a Unix stream socket, the only thing a hand-off may carry. */
static BOOL NFKPluginChannelIsStreamSocket(int descriptor)
{
	struct stat status;
	int type = 0;
	socklen_t typeLength = sizeof type;
	struct sockaddr_storage address;
	socklen_t addressLength = sizeof address;
	return fstat(descriptor, &status) == 0 && S_ISSOCK(status.st_mode)
		&& getsockopt(descriptor, SOL_SOCKET, SO_TYPE, &type, &typeLength) == 0 && type == SOCK_STREAM
		&& getsockname(descriptor, (struct sockaddr *)&address, &addressLength) == 0 && address.ss_family == AF_UNIX;
}

static void NFKPluginChannelWriteLittleEndian32(uint8_t *bytes, uint32_t value)
{
	bytes[0] = (uint8_t)value;
	bytes[1] = (uint8_t)(value >> 8);
	bytes[2] = (uint8_t)(value >> 16);
	bytes[3] = (uint8_t)(value >> 24);
}

static uint32_t NFKPluginChannelReadLittleEndian32(const uint8_t *bytes)
{
	return (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8) | ((uint32_t)bytes[2] << 16) | ((uint32_t)bytes[3] << 24);
}

static uint16_t NFKPluginChannelReadLittleEndian16(const uint8_t *bytes)
{
	return (uint16_t)(bytes[0] | (bytes[1] << 8));
}

/*! Writes one frame to a socket: the header and JSON in one sendmsg carrying descriptors, then any
	remainder. Waits for the socket to drain up to timeout, or until shouldStop answers YES. */
static BOOL NFKPluginChannelWriteFrame(int socket, NSData *JSON, NSArray<NSNumber *> *descriptors, NSTimeInterval timeout,
									   BOOL (^ _Nullable shouldStop)(void), NSError **outError)
{
	uint8_t header[NFKPluginFrameHeaderLength] = { 0 };
	NFKPluginChannelWriteLittleEndian32(header, NFKPluginFrameMagic);
	NFKPluginChannelWriteLittleEndian32(header + 4, (uint32_t)JSON.length);
	header[8] = (uint8_t)descriptors.count;
	header[9] = (uint8_t)(descriptors.count >> 8);

	size_t controlLength = descriptors.count > 0 ? CMSG_SPACE(sizeof(int) * descriptors.count) : 0;
	NSMutableData *control = [NSMutableData dataWithLength:controlLength];
	if (descriptors.count > 0) {
		struct cmsghdr *message = (struct cmsghdr *)control.mutableBytes;
		message->cmsg_level = SOL_SOCKET;
		message->cmsg_type = SCM_RIGHTS;
		message->cmsg_len = (socklen_t)CMSG_LEN(sizeof(int) * descriptors.count);
		int *slots = (int *)CMSG_DATA(message);
		for (NSUInteger index = 0; index < descriptors.count; index++) {
			slots[index] = descriptors[index].intValue;
		}
	}

	size_t total = NFKPluginFrameHeaderLength + JSON.length;
	size_t sent = 0;
	NSDate *deadline = [NSDate dateWithTimeIntervalSinceNow:timeout > 0 ? timeout : NFKPluginChannelDefaultSendTimeout];
	while (sent < total) {
		if (shouldStop != nil && shouldStop()) {
			return NFKPluginChannelPOSIXError(outError, NFKPluginErrorStopped, 0, @"the channel closed during a send");
		}
		struct iovec pieces[2];
		int pieceCount = 0;
		if (sent < NFKPluginFrameHeaderLength) {
			pieces[pieceCount++] = (struct iovec){ .iov_base = header + sent, .iov_len = NFKPluginFrameHeaderLength - sent };
			pieces[pieceCount++] = (struct iovec){ .iov_base = (void *)JSON.bytes, .iov_len = JSON.length };
		} else {
			size_t offset = sent - NFKPluginFrameHeaderLength;
			pieces[pieceCount++] = (struct iovec){ .iov_base = (char *)JSON.bytes + offset, .iov_len = JSON.length - offset };
		}
		struct msghdr message = { 0 };
		message.msg_iov = pieces;
		message.msg_iovlen = pieceCount;
		// The descriptors ride the frame's first byte, so they go only with the sendmsg that sends it.
		if (sent == 0 && controlLength > 0) {
			message.msg_control = control.mutableBytes;
			message.msg_controllen = (socklen_t)controlLength;
		}
		ssize_t count = sendmsg(socket, &message, 0);
		if (count > 0) {
			sent += (size_t)count;
			continue;
		}
		if (count < 0 && errno == EINTR) {
			continue;
		}
		if (count < 0 && (errno == EAGAIN || errno == ENOBUFS)) {
			if (deadline.timeIntervalSinceNow <= 0) {
				return NFKPluginChannelPOSIXError(outError, NFKPluginErrorProtocolViolation, 0,
												  [NSString stringWithFormat:@"the peer read nothing for %g seconds", timeout]);
			}
			struct pollfd writable = { .fd = socket, .events = POLLOUT, .revents = 0 };
			poll(&writable, 1, NFKPluginChannelPollMilliseconds);
			continue;
		}
		int code = count < 0 ? errno : EIO;
		NSInteger errorCode = (code == EPIPE || code == ECONNRESET || code == ENOTCONN) ? NFKPluginErrorStopped : NFKPluginErrorProtocolViolation;
		return NFKPluginChannelPOSIXError(outError, errorCode, code, @"the frame could not be sent");
	}
	return YES;
}

#pragma mark - Descriptor

@interface NFKPluginDescriptor ()
@property (atomic, readwrite) int descriptor;
@end

@implementation NFKPluginDescriptor

- (instancetype)initWithDescriptor:(int)descriptor
{
	self = [super init];
	if (self != nil) {
		_descriptor = descriptor;
	}
	return self;
}

- (int)takeDescriptor
{
	@synchronized (self) {
		int descriptor = self.descriptor;
		self.descriptor = -1;
		return descriptor;
	}
}

- (void)dealloc
{
	if (_descriptor >= 0) {
		close(_descriptor);
	}
}

- (NSString *)description
{
	return [NSString stringWithFormat:@"<%@ %d>", self.class, self.descriptor];
}

@end

#pragma mark - Frame reader

/*! Accumulates a socket's bytes and descriptors and cuts them into frames. Not thread-safe; a channel
	touches it on its read queue only. */
@interface NFKPluginFrameReader : NSObject
@property (nonatomic, strong) NSMutableData *pending;
@property (nonatomic, strong) NSMutableArray<NFKPluginDescriptor *> *descriptors;
@property (nonatomic, assign) uint8_t *chunk;
@property (nonatomic, assign) void *control;
@end

@implementation NFKPluginFrameReader

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_pending = [NSMutableData data];
		_descriptors = [NSMutableArray array];
		_chunk = malloc(NFKPluginChannelReadChunk);
		_control = calloc(1, CMSG_SPACE(sizeof(int) * NFKPluginFrameMaximumDescriptors));
	}
	return self;
}

- (void)dealloc
{
	free(_chunk);
	free(_control);
}

/*! Reads once from socket. Returns the bytes read, 0 at end of file, or -1: with outError nil when the
	socket has nothing to read now, or with an error. */
- (ssize_t)receiveFromSocket:(int)socket error:(NSError **)outError
{
	size_t controlLength = CMSG_SPACE(sizeof(int) * NFKPluginFrameMaximumDescriptors);
	void *control = self.control;
	for (;;) {
		struct iovec piece = { .iov_base = self.chunk, .iov_len = NFKPluginChannelReadChunk };
		struct msghdr message = { 0 };
		message.msg_iov = &piece;
		message.msg_iovlen = 1;
		message.msg_control = control;
		message.msg_controllen = (socklen_t)controlLength;
		ssize_t count = recvmsg(socket, &message, 0);
		if (count < 0 && errno == EINTR) {
			continue;
		}
		if (count < 0) {
			if (outError != NULL) {
				*outError = nil;
			}
			if (errno == EAGAIN) {
				return -1;
			}
			NFKPluginChannelPOSIXError(outError, NFKPluginErrorProtocolViolation, errno, @"the channel could not be read");
			return -1;
		}
		[self collectDescriptorsOfMessage:&message];
		if ((message.msg_flags & MSG_CTRUNC) != 0) {
			NFKPluginChannelPOSIXError(outError, NFKPluginErrorProtocolViolation, 0,
									   [NSString stringWithFormat:@"a frame carried more than %u descriptors", NFKPluginFrameMaximumDescriptors]);
			return -1;
		}
		[self.pending appendBytes:self.chunk length:(NSUInteger)count];
		return count;
	}
}

- (void)collectDescriptorsOfMessage:(struct msghdr *)message
{
	for (struct cmsghdr *header = CMSG_FIRSTHDR(message); header != NULL; header = CMSG_NXTHDR(message, header)) {
		if (header->cmsg_level != SOL_SOCKET || header->cmsg_type != SCM_RIGHTS) {
			continue;
		}
		size_t count = (header->cmsg_len - CMSG_LEN(0)) / sizeof(int);
		const int *slots = (const int *)CMSG_DATA(header);
		for (size_t index = 0; index < count; index++) {
			fcntl(slots[index], F_SETFD, FD_CLOEXEC);
			[self.descriptors addObject:[[NFKPluginDescriptor alloc] initWithDescriptor:slots[index]]];
		}
	}
}

/*! The next complete frame's message with its descriptors, or nil: with outError nil while the frame is
	incomplete, or with the violation the bytes show. */
- (nullable NSDictionary<NSString *, id> *)nextMessageWithDescriptors:(NSArray<NFKPluginDescriptor *> * _Nullable * _Nonnull)outDescriptors
																 error:(NSError **)outError
{
	*outDescriptors = nil;
	if (outError != NULL) {
		*outError = nil;
	}
	NSUInteger available = self.pending.length;
	if (available < NFKPluginFrameHeaderLength) {
		return nil;
	}
	const uint8_t *bytes = self.pending.bytes;
	uint32_t magic = NFKPluginChannelReadLittleEndian32(bytes);
	uint32_t length = NFKPluginChannelReadLittleEndian32(bytes + 4);
	uint16_t count = NFKPluginChannelReadLittleEndian16(bytes + 8);
	if (magic != NFKPluginFrameMagic) {
		NFKPluginChannelPOSIXError(outError, NFKPluginErrorProtocolViolation, 0,
								   [NSString stringWithFormat:@"a frame began with 0x%08X, not the magic number", magic]);
		return nil;
	}
	if (length > NFKPluginFrameMaximumJSONLength || count > NFKPluginFrameMaximumDescriptors) {
		NFKPluginChannelPOSIXError(outError, NFKPluginErrorProtocolViolation, 0,
								   [NSString stringWithFormat:@"a frame declared %u bytes and %u descriptors", length, count]);
		return nil;
	}
	if (available < NFKPluginFrameHeaderLength + length) {
		return nil;
	}
	// The kernel delivers a frame's descriptors with its first byte, which has been read, so a frame
	// whose descriptors are not queued now never receives them.
	if (self.descriptors.count < count) {
		NFKPluginChannelPOSIXError(outError, NFKPluginErrorProtocolViolation, 0,
								   [NSString stringWithFormat:@"a frame declared %u descriptors and %lu arrived", count, (unsigned long)self.descriptors.count]);
		return nil;
	}
	NSData *JSON = [self.pending subdataWithRange:NSMakeRange(NFKPluginFrameHeaderLength, length)];
	[self.pending replaceBytesInRange:NSMakeRange(0, NFKPluginFrameHeaderLength + length) withBytes:NULL length:0];
	NSArray<NFKPluginDescriptor *> *descriptors = [self.descriptors subarrayWithRange:NSMakeRange(0, count)];
	[self.descriptors removeObjectsInRange:NSMakeRange(0, count)];
	id message = [NSJSONSerialization JSONObjectWithData:JSON options:0 error:NULL];
	if (![message isKindOfClass:NSDictionary.class] || ![message[@"type"] isKindOfClass:NSString.class]) {
		NFKPluginChannelPOSIXError(outError, NFKPluginErrorProtocolViolation, 0, @"a frame's JSON is not an object with a string type");
		return nil;
	}
	*outDescriptors = descriptors;
	return message;
}

@end

#pragma mark - Channel

@interface NFKPluginChannel ()
@property (atomic, readwrite, getter=isClosed) BOOL closed;
@property (atomic, assign) int socket;
@property (nonatomic, strong) dispatch_queue_t readQueue;
@property (nonatomic, readwrite, strong) dispatch_queue_t handlerQueue;
@property (nonatomic, strong, nullable) dispatch_source_t readSource;
@property (nonatomic, strong) NFKPluginFrameReader *reader;
@property (nonatomic, strong) NSLock *sendLock;
@property (nonatomic, assign) BOOL ended;
@end

@implementation NFKPluginChannel

+ (BOOL)createSocketPair:(int *)firstDescriptor second:(int *)secondDescriptor error:(NSError **)outError
{
	int pair[2] = { -1, -1 };
	if (socketpair(AF_UNIX, SOCK_STREAM, 0, pair) != 0) {
		return NFKPluginChannelPOSIXError(outError, NFKPluginErrorLaunchFailed, errno, @"the channel's socket pair could not be created");
	}
	NFKPluginChannelConfigureSocket(pair[0], NO);
	NFKPluginChannelConfigureSocket(pair[1], NO);
	*firstDescriptor = pair[0];
	*secondDescriptor = pair[1];
	return YES;
}

+ (nullable instancetype)channelByHandingOffDescriptor:(int)inheritedDescriptor error:(NSError **)outError
{
	int kept = -1;
	int handed = -1;
	if (![self createSocketPair:&kept second:&handed error:outError]) {
		close(inheritedDescriptor);
		return nil;
	}
	NSData *JSON = [NSJSONSerialization dataWithJSONObject:@{ @"type": NFKPluginMessageChannel } options:0 error:NULL];
	BOOL sent = NFKPluginChannelWriteFrame(inheritedDescriptor, JSON, @[ @(handed) ], NFKPluginChannelDefaultSendTimeout, nil, outError);
	close(handed);
	close(inheritedDescriptor);
	if (!sent) {
		close(kept);
		return nil;
	}
	return [[self alloc] initWithDescriptor:kept];
}

+ (nullable instancetype)channelByAcceptingHandOffOnDescriptor:(int)descriptor timeout:(NSTimeInterval)timeout error:(NSError **)outError
{
	NFKPluginChannelConfigureSocket(descriptor, YES);
	NFKPluginFrameReader *reader = [[NFKPluginFrameReader alloc] init];
	NSDate *deadline = [NSDate dateWithTimeIntervalSinceNow:timeout];
	NSDictionary *message = nil;
	NSArray<NFKPluginDescriptor *> *descriptors = nil;
	NSError *failure = nil;
	while (message == nil && failure == nil) {
		message = [reader nextMessageWithDescriptors:&descriptors error:&failure];
		if (message != nil || failure != nil) {
			break;
		}
		NSTimeInterval remaining = deadline.timeIntervalSinceNow;
		if (remaining <= 0) {
			NFKPluginChannelPOSIXError(&failure, NFKPluginErrorHandshakeFailed, 0,
									   [NSString stringWithFormat:@"the plugin handed over no channel within %g seconds", timeout]);
			break;
		}
		struct pollfd readable = { .fd = descriptor, .events = POLLIN, .revents = 0 };
		poll(&readable, 1, (int)MIN(remaining * 1000.0 + 1.0, (double)INT_MAX));
		NSError *readError = nil;
		ssize_t count = [reader receiveFromSocket:descriptor error:&readError];
		if (count == 0) {
			NFKPluginChannelPOSIXError(&failure, NFKPluginErrorHandshakeFailed, 0, @"the plugin closed its channel before handing one over");
		} else if (count < 0 && readError != nil) {
			failure = readError;
		}
	}
	close(descriptor);
	BOOL handedOver = message != nil && [message[@"type"] isEqual:NFKPluginMessageChannel] && descriptors.count == 1
		&& NFKPluginChannelIsStreamSocket(descriptors.firstObject.descriptor);
	if (!handedOver) {
		if (failure == nil) {
			NFKPluginChannelPOSIXError(&failure, NFKPluginErrorHandshakeFailed, 0, @"the plugin's first frame is not a channel hand-off carrying one Unix stream socket");
		}
		if (outError != NULL) {
			*outError = failure;
		}
		return nil;
	}
	return [[self alloc] initWithDescriptor:[descriptors.firstObject takeDescriptor]];
}

- (instancetype)initWithDescriptor:(int)descriptor
{
	self = [super init];
	if (self != nil) {
		_socket = descriptor;
		_sendTimeout = NFKPluginChannelDefaultSendTimeout;
		_reader = [[NFKPluginFrameReader alloc] init];
		_sendLock = [[NSLock alloc] init];
		dispatch_queue_attr_t attributes = dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INITIATED, 0);
		_readQueue = dispatch_queue_create("com.inferkit.plugin-channel.read", attributes);
		_handlerQueue = dispatch_queue_create("com.inferkit.plugin-channel.handler", attributes);
		NFKPluginChannelConfigureSocket(descriptor, YES);
	}
	return self;
}

- (void)dealloc
{
	if (_readSource == nil && _socket >= 0) {
		close(_socket);
	}
}

#pragma mark Reading

- (void)resume
{
	dispatch_async(self.readQueue, ^{
		if (self.readSource != nil || self.ended) {
			return;
		}
		int socket = self.socket;
		if (socket < 0 || self.closed) {
			[self endWithError:nil];
			return;
		}
		dispatch_source_t source = dispatch_source_create(DISPATCH_SOURCE_TYPE_READ, (uintptr_t)socket, 0, self.readQueue);
		dispatch_source_set_event_handler(source, ^{
			[self readAvailableFrames];
		});
		dispatch_source_set_cancel_handler(source, ^{
			[self closeSocket];
		});
		self.readSource = source;
		dispatch_resume(source);
	});
}

// Runs on the read queue.
- (void)readAvailableFrames
{
	if (self.ended) {
		return;
	}
	for (;;) {
		NSError *readError = nil;
		ssize_t count = [self.reader receiveFromSocket:self.socket error:&readError];
		if (count == 0) {
			[self deliverFrames];
			[self endWithError:nil];
			return;
		}
		if (count < 0) {
			if (readError != nil) {
				[self endWithError:readError];
			} else {
				[self deliverFrames];
			}
			return;
		}
		if (![self deliverFrames]) {
			return;
		}
	}
}

/*! Delivers every complete frame. Returns NO when a frame violated the protocol and ended the channel. */
- (BOOL)deliverFrames
{
	for (;;) {
		NSArray<NFKPluginDescriptor *> *descriptors = nil;
		NSError *violation = nil;
		NSDictionary *message = [self.reader nextMessageWithDescriptors:&descriptors error:&violation];
		if (violation != nil) {
			[self endWithError:violation];
			return NO;
		}
		if (message == nil) {
			return YES;
		}
		NFKPluginChannelMessageHandler handler = self.messageHandler;
		dispatch_async(self.handlerQueue, ^{
			if (handler != nil) {
				handler(message, descriptors);
			}
		});
	}
}

// Runs on the read queue. The close handler runs once, after the messages already delivered.
- (void)endWithError:(nullable NSError *)error
{
	if (self.ended) {
		return;
	}
	self.ended = YES;
	self.closed = YES;
	[self.reader.descriptors removeAllObjects];
	if (self.readSource != nil) {
		dispatch_source_cancel(self.readSource);
		self.readSource = nil;
	} else {
		[self closeSocket];
	}
	NFKPluginChannelCloseHandler handler = self.closeHandler;
	dispatch_async(self.handlerQueue, ^{
		if (handler != nil) {
			handler(error);
		}
	});
}

- (void)closeSocket
{
	[self.sendLock lock];
	if (self.socket >= 0) {
		close(self.socket);
		self.socket = -1;
	}
	[self.sendLock unlock];
}

- (void)close
{
	if (self.closed) {
		return;
	}
	self.closed = YES;
	// shutdown wakes the read source and tells the peer at once. Under the send lock, so the descriptor
	// cannot be closed and its number reused in between; a blocked send sees closed within one poll.
	[self.sendLock lock];
	if (self.socket >= 0) {
		shutdown(self.socket, SHUT_RDWR);
	}
	[self.sendLock unlock];
	dispatch_async(self.readQueue, ^{
		[self endWithError:nil];
	});
}

#pragma mark Sending

- (BOOL)sendMessage:(NSDictionary<NSString *, id> *)message error:(NSError **)outError
{
	return [self sendMessage:message descriptors:nil error:outError];
}

- (BOOL)sendMessage:(NSDictionary<NSString *, id> *)message descriptors:(nullable NSArray<NSNumber *> *)descriptors error:(NSError **)outError
{
	NSArray<NSNumber *> *carried = descriptors ?: @[];
	if (carried.count > NFKPluginFrameMaximumDescriptors) {
		return NFKPluginChannelPOSIXError(outError, NFKPluginErrorProtocolViolation, 0,
										  [NSString stringWithFormat:@"a frame carries at most %u descriptors", NFKPluginFrameMaximumDescriptors]);
	}
	if (![message[@"type"] isKindOfClass:NSString.class] || ![NSJSONSerialization isValidJSONObject:message]) {
		return NFKPluginChannelPOSIXError(outError, NFKPluginErrorProtocolViolation, 0, @"a message is a JSON object with a string type");
	}
	NSData *JSON = [NSJSONSerialization dataWithJSONObject:message options:0 error:outError];
	if (JSON == nil) {
		return NO;
	}
	if (JSON.length > NFKPluginFrameMaximumJSONLength) {
		return NFKPluginChannelPOSIXError(outError, NFKPluginErrorProtocolViolation, 0,
										  [NSString stringWithFormat:@"a message of %lu bytes exceeds the frame limit", (unsigned long)JSON.length]);
	}
	[self.sendLock lock];
	BOOL sent = NO;
	if (self.closed || self.socket < 0) {
		NFKPluginChannelPOSIXError(outError, NFKPluginErrorStopped, 0, @"the channel is closed");
	} else {
		sent = NFKPluginChannelWriteFrame(self.socket, JSON, carried, self.sendTimeout, ^BOOL {
			return self.closed;
		}, outError);
	}
	[self.sendLock unlock];
	return sent;
}

#pragma mark Peer

- (BOOL)getPeerAuditToken:(audit_token_t *)token error:(NSError **)outError
{
	[self.sendLock lock];
	socklen_t length = sizeof *token;
	int status = self.socket >= 0 ? getsockopt(self.socket, SOL_LOCAL, LOCAL_PEERTOKEN, token, &length) : -1;
	int code = status != 0 ? (self.socket >= 0 ? errno : EBADF) : 0;
	[self.sendLock unlock];
	if (status != 0) {
		return NFKPluginChannelPOSIXError(outError, NFKPluginErrorHandshakeFailed, code, @"the peer's audit token could not be read");
	}
	return YES;
}

- (pid_t)peerProcessIdentifier
{
	pid_t identifier = 0;
	socklen_t length = sizeof identifier;
	[self.sendLock lock];
	int status = self.socket >= 0 ? getsockopt(self.socket, SOL_LOCAL, LOCAL_PEERPID, &identifier, &length) : -1;
	[self.sendLock unlock];
	return status == 0 ? identifier : 0;
}

@end

#endif
