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

/*! How long the plugin side holds a handed socket for a host that never reads the hand-off. */
static const NSTimeInterval NFKPluginHandOffHoldLimit = 300;

/*!
	Keeps the plugin side's copy of a handed socket, and the transport it crossed, open until the host has
	received it, then closes both. The host closes the transport once it has read the hand-off, so end of
	file on the transport marks receipt.
	A socket in flight whose sender has closed every descriptor of it can be collected by the kernel's
	Unix-domain garbage collector before the receiver installs it, and arrives already hung up: measured
	on macOS 26 at 18 of 300 hand-offs with no delay before the receive and 300 of 300 with 2 ms.
	Holding the sender's copy until receipt, measured the same way, hung up none.
	Takes ownership of transport and handed, and returns at once.
*/
static void NFKPluginChannelHoldUntilReceived(int transport, int handed)
{
	NFKPluginChannelConfigureSocket(transport, YES);
	dispatch_queue_attr_t attributes = dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INITIATED, 0);
	dispatch_queue_t queue = dispatch_queue_create("com.inferkit.plugin-channel.hand-off", attributes);
	dispatch_source_t source = dispatch_source_create(DISPATCH_SOURCE_TYPE_READ, (uintptr_t)transport, 0, queue);
	__block int heldTransport = transport;
	__block int heldHanded = handed;
	dispatch_source_set_event_handler(source, ^{
		char drained[256];
		for (;;) {
			ssize_t count = read(heldTransport, drained, sizeof drained);
			if (count > 0 || (count < 0 && errno == EINTR)) {
				continue;
			}
			if (count < 0 && errno == EAGAIN) {
				return;
			}
			dispatch_source_cancel(source);
			return;
		}
	});
	dispatch_source_set_cancel_handler(source, ^{
		NFKPluginCloseDescriptor(&heldHanded);
		NFKPluginCloseDescriptor(&heldTransport);
	});
	dispatch_after(dispatch_time(DISPATCH_TIME_NOW, (int64_t)(NFKPluginHandOffHoldLimit * NSEC_PER_SEC)), queue, ^{
		dispatch_source_cancel(source);
	});
	dispatch_resume(source);
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
		NFKPluginAdoptDescriptor(descriptor, "NFKPluginDescriptor");
	}
	return self;
}

- (int)takeDescriptor
{
	@synchronized (self) {
		int descriptor = self.descriptor;
		self.descriptor = -1;
		NFKPluginDisownDescriptor(descriptor);
		return descriptor;
	}
}

- (void)dealloc
{
	NFKPluginCloseDescriptor(&_descriptor);
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
/*! Why the channel ended, for the DEBUG trace: end of file, a local close, or the error. */
@property (atomic, copy, nullable) NSString *endReason;
/*! The socket object the descriptor named when the channel adopted it, for the DEBUG trace. */
@property (nonatomic, assign) dev_t socketDevice;
@property (nonatomic, assign) ino_t socketInode;
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
	int inherited = inheritedDescriptor;
	NFKPluginAdoptDescriptor(inherited, "a channel hand-off");
	int kept = -1;
	int handed = -1;
	if (![self createSocketPair:&kept second:&handed error:outError]) {
		NFKPluginCloseDescriptor(&inherited);
		return nil;
	}
	NFKPluginAdoptDescriptor(kept, "a channel hand-off");
	NFKPluginAdoptDescriptor(handed, "a channel hand-off");
	NSData *JSON = [NSJSONSerialization dataWithJSONObject:@{ @"type": NFKPluginMessageChannel } options:0 error:NULL];
	BOOL sent = NFKPluginChannelWriteFrame(inherited, JSON, @[ @(handed) ], NFKPluginChannelDefaultSendTimeout, nil, outError);
#if DEBUG
	NFKPluginDebugTrace(@"plugin side hands socket %d over descriptor %d and keeps %d: %@", handed, inherited, kept,
						sent ? @"sent" : (outError != NULL ? (*outError).localizedDescription : @"not sent"));
#endif
	if (!sent) {
		NFKPluginCloseDescriptor(&handed);
		NFKPluginCloseDescriptor(&inherited);
		NFKPluginCloseDescriptor(&kept);
		return nil;
	}
	NFKPluginChannelHoldUntilReceived(inherited, handed);
	NFKPluginDisownDescriptor(kept);
	return [[self alloc] initWithDescriptor:kept];
}

+ (nullable instancetype)channelByAcceptingHandOffOnDescriptor:(int)inheritedDescriptor timeout:(NSTimeInterval)timeout error:(NSError **)outError
{
	int descriptor = inheritedDescriptor;
	NFKPluginAdoptDescriptor(descriptor, "a channel hand-off");
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
			NFKPluginChannelPOSIXError(&failure, NFKPluginErrorTimedOut, 0,
									   [NSString stringWithFormat:@"the channel hand-off did not arrive within its %g-second limit", timeout]);
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
#if DEBUG
	NFKPluginDebugTrace(@"host side reads the hand-off on descriptor %d: %@ with %lu descriptors, %@", descriptor,
						message[@"type"] ?: @"no frame", (unsigned long)descriptors.count, failure.localizedDescription ?: @"no error");
#endif
	NFKPluginCloseDescriptor(&descriptor);
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
		// A pool per block: a weak table read autoreleases the object it returns, such as the backend an
		// event names, and a queue's thread drains no pool of its own until it exits, so a released backend
		// would stay alive, and keep its plugin's process from exiting, for as long as the thread lives.
		dispatch_queue_attr_t attributes = dispatch_queue_attr_make_with_autorelease_frequency(
			dispatch_queue_attr_make_with_qos_class(DISPATCH_QUEUE_SERIAL, QOS_CLASS_USER_INITIATED, 0),
			DISPATCH_AUTORELEASE_FREQUENCY_WORK_ITEM);
		_readQueue = dispatch_queue_create("com.inferkit.plugin-channel.read", attributes);
		_handlerQueue = dispatch_queue_create("com.inferkit.plugin-channel.handler", attributes);
		NFKPluginAdoptDescriptor(descriptor, "NFKPluginChannel");
		NFKPluginChannelConfigureSocket(descriptor, YES);
#if DEBUG
		struct stat status;
		if (fstat(descriptor, &status) == 0) {
			_socketDevice = status.st_dev;
			_socketInode = status.st_ino;
		}
		if (NFKPluginDebugTracing()) {
			NFKPluginDebugTrace(@"channel %p adopts socket %d (inode %llu, peer pid %d) from %@", self, descriptor,
								(unsigned long long)_socketInode, self.peerProcessIdentifier, NFKPluginDebugBacktrace(1, 6));
		}
#endif
	}
	return self;
}

#if DEBUG
/*! What the channel's descriptor names now: the socket it adopted, another object, or nothing. */
- (NSString *)debugDescriptorState:(int)descriptor
{
	struct stat status;
	if (descriptor < 0) {
		return @"closed by the channel";
	}
	if (fstat(descriptor, &status) != 0) {
		return [NSString stringWithFormat:@"NOT OPEN (%s): closed by code that did not own it", strerror(errno)];
	}
	if (status.st_dev != self.socketDevice || status.st_ino != self.socketInode) {
		return [NSString stringWithFormat:@"ANOTHER OBJECT (inode %llu): closed and reused by code that did not own it",
				(unsigned long long)status.st_ino];
	}
	return @"the adopted socket";
}
#endif

// A resumed channel's read source holds the channel until its cancel handler has closed the socket,
// so only a channel that never read can reach here owning it.
- (void)dealloc
{
	NFKPluginCloseDescriptor(&_socket);
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
			self.endReason = self.endReason ?: @"end of file: the peer's socket closed";
			[self endWithError:nil];
			return;
		}
		if (count < 0) {
			if (readError != nil) {
				self.endReason = readError.localizedDescription;
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
#if DEBUG
	if (NFKPluginDebugTracing()) {
		NFKPluginDebugTrace(@"channel %p socket %d ends: %@; error %@; the descriptor names %@; %@", self, self.socket,
							self.endReason ?: (error != nil ? error.localizedDescription : @"unknown"), error,
							[self debugDescriptorState:self.socket], NFKPluginDebugDescriptorHistory(self.socket));
	}
#endif
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
	int socket = self.socket;
	NFKPluginCloseDescriptor(&socket);
	self.socket = socket;
	[self.sendLock unlock];
}

- (void)close
{
	if (self.closed) {
		return;
	}
	self.closed = YES;
	self.endReason = @"close() called locally";
#if DEBUG
	if (NFKPluginDebugTracing()) {
		NFKPluginDebugTrace(@"channel %p socket %d close() from %@", self, self.socket, NFKPluginDebugBacktrace(1, 8));
	}
#endif
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
