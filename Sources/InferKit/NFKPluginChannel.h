//
//  NFKPluginChannel.h
//  InferKit
//
//  The framed socket a host and a plugin's process talk over. Docs/agent-reference/plugins.md
//  specifies the frame, the hand-off, and the messages.
//

#import <Foundation/Foundation.h>
#include <mach/message.h>

NS_ASSUME_NONNULL_BEGIN

#if TARGET_OS_OSX

/*!
	@class      NFKPluginDescriptor
	@abstract   A file descriptor a frame carried, closed when the object is released unless it was taken.
*/
@interface NFKPluginDescriptor : NSObject

/*! Wraps descriptor and takes ownership of it. */
- (instancetype)initWithDescriptor:(int)descriptor NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;

/*! The descriptor, or -1 once taken. */
@property (atomic, readonly) int descriptor;

/*! Returns the descriptor and gives up ownership of it, or -1 when it was already taken. */
- (int)takeDescriptor;

@end

/*! Receives each message with the descriptors its frame carried, on the channel's handler queue. */
typedef void (^NFKPluginChannelMessageHandler)(NSDictionary<NSString *, id> *message, NSArray<NFKPluginDescriptor *> *descriptors);

/*! Receives the channel's end once: nil when the peer closed its end or close was called, or the error
	that closed it. Runs on the handler queue after every message received before the end. */
typedef void (^NFKPluginChannelCloseHandler)(NSError * _Nullable error);

/*!
	@class      NFKPluginChannel
	@abstract   Frames JSON messages and file descriptors over a Unix stream socket.
	@discussion A frame is a 12-byte header (uint32 magic 0x504B464E, uint32 JSON length at most 16 MiB,
				uint16 descriptor count at most 64, uint16 flags written 0 and ignored, all little-endian)
				and the UTF-8 JSON of one object with a string "type". The header and the JSON go out in
				one sendmsg whose SCM_RIGHTS carries the descriptors; the kernel attaches them to the
				frame's first byte. The receiver queues every descriptor it receives and hands a frame
				as many as its header counts.

				A wrong magic number, an oversized frame, a frame whose descriptors did not arrive, a
				truncated descriptor list, or JSON that is not an object with a type closes the channel
				with NFKPluginErrorProtocolViolation.

				The channel reads on a queue of its own and delivers on a serial handler queue, so a
				handler that blocks in a send does not stop the reading that would unblock it. A send
				blocks while the socket is full, up to sendTimeout, and sends are serialized. A resumed
				channel keeps itself alive until it closes.

				A socketpair's end reports, as its peer, the process that created the pair, so an
				inherited socket names the host on both sides. The plugin side therefore hands the host a
				socket it creates itself (channelByHandingOffDescriptor:error:), and the host reads the
				plugin's audit token from that socket.
*/
@interface NFKPluginChannel : NSObject

/*! Creates a connected pair of Unix stream sockets, both close-on-exec with SIGPIPE suppressed. The
	caller owns both descriptors. */
+ (BOOL)createSocketPair:(int *)firstDescriptor second:(int *)secondDescriptor error:(NSError * _Nullable *)outError;

/*! The plugin side of the hand-off: creates a socket pair, sends one end to the host as the first frame
	on inheritedDescriptor, closes inheritedDescriptor, and returns a channel on the other end. */
+ (nullable instancetype)channelByHandingOffDescriptor:(int)inheritedDescriptor error:(NSError * _Nullable *)outError;

/*! The host side of the hand-off: waits up to timeout for the first frame on descriptor, which must be
	{"type": "channel"} carrying one Unix stream socket, closes descriptor, and returns a channel on the
	received socket. Fails with NFKPluginErrorHandshakeFailed otherwise. */
+ (nullable instancetype)channelByAcceptingHandOffOnDescriptor:(int)descriptor
													   timeout:(NSTimeInterval)timeout
														 error:(NSError * _Nullable *)outError;

/*! A channel on a connected Unix stream socket, which it takes ownership of. */
- (instancetype)initWithDescriptor:(int)descriptor NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;

/*! Receives each message. Set before resume. */
@property (atomic, copy, nullable) NFKPluginChannelMessageHandler messageHandler;

/*! Receives the channel's end. Set before resume. */
@property (atomic, copy, nullable) NFKPluginChannelCloseHandler closeHandler;

/*! The serial queue the handlers run on. */
@property (nonatomic, readonly, strong) dispatch_queue_t handlerQueue;

/*! How long a send waits for the peer to read, in seconds. Defaults to 30. */
@property (atomic, assign) NSTimeInterval sendTimeout;

/*! Whether the channel has closed. */
@property (atomic, readonly, getter=isClosed) BOOL closed;

/*! Starts reading. */
- (void)resume;

/*! Sends message, which must serialize as a JSON object. */
- (BOOL)sendMessage:(NSDictionary<NSString *, id> *)message error:(NSError * _Nullable *)outError;

/*! Sends message with descriptors in its frame. The caller keeps its own descriptors open; the peer
	receives duplicates. */
- (BOOL)sendMessage:(NSDictionary<NSString *, id> *)message
		descriptors:(nullable NSArray<NSNumber *> *)descriptors
			  error:(NSError * _Nullable *)outError;

/*! Closes the channel. A send in progress fails, and the close handler runs with nil. */
- (void)close;

/*! The audit token of the process that created the peer's socket. */
- (BOOL)getPeerAuditToken:(audit_token_t *)token error:(NSError * _Nullable *)outError;

/*! The identifier of the process that created the peer's socket, or 0. */
@property (nonatomic, readonly) pid_t peerProcessIdentifier;

@end

#endif

NS_ASSUME_NONNULL_END
