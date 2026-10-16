//
//  NFKPluginWire.h
//  InferKit
//
//  The plugin protocol's constants, its error helpers, and the attachment sink and source that carry a
//  message's large values in shared memory. Docs/agent-reference/plugins.md specifies the protocol.
//

#import <Foundation/Foundation.h>
#import <InferKit/NFKPlugin.h>
#import "NFKInferenceWireCoding.h"

NS_ASSUME_NONNULL_BEGIN

/*! Sets error to an NFKPluginErrorDomain error with code and reason, and returns NO. */
BOOL NFKPluginSetError(NSError * _Nullable * _Nullable error, NSInteger code, NSString *reason);

#if TARGET_OS_OSX

@class NFKPluginDescriptor;

/*! The descriptor number the plugin's process inherits its channel on. */
static const int NFKPluginChannelDescriptorNumber = 3;

/*! The environment variable that names NFKPluginChannelDescriptorNumber in the plugin's process. */
extern NSString * const NFKPluginChannelEnvironmentVariable;

/*! The protocol version this library speaks. */
static const NSInteger NFKPluginProtocolMajorVersion = 1;
static const NSInteger NFKPluginProtocolMinorVersion = 0;

/*! The frame header's first word, "NFKP" read as little-endian ASCII. */
static const uint32_t NFKPluginFrameMagic = 0x504B464E;

/*! The frame header's length: magic, JSON length, descriptor count, and flags. An enumeration
	constant, so it sizes an array. */
enum { NFKPluginFrameHeaderLength = 12 };

/*! The longest JSON a frame carries, in bytes. */
static const uint32_t NFKPluginFrameMaximumJSONLength = 16u * 1024u * 1024u;

/*! The most descriptors a frame carries. */
static const uint16_t NFKPluginFrameMaximumDescriptors = 64;

/*! The largest value a message carries inline, in bytes, unless hello sets limits.inlineBytes. */
static const NSUInteger NFKPluginDefaultInlineByteLimit = 64u * 1024u;

/*!
	@functiongroup Descriptor ownership
	@discussion Every descriptor a plugin object holds has one owner. The owner adopts it when ownership
				arrives, closes it through NFKPluginCloseDescriptor exactly once, or disowns it when
				ownership leaves for another owner. In a DEBUG build a registry checks every step and stops
				the process at the first violation, naming the owner: adopting a number another owner still
				holds means that owner's descriptor was closed by code that did not own it, and closing a
				descriptor that is no longer open means it was closed twice.
*/

/*! Records that owner holds descriptor. */
void NFKPluginAdoptDescriptor(int descriptor, const char *owner);

/*! Records that the descriptor's ownership left for code that closes it. */
void NFKPluginDisownDescriptor(int descriptor);

/*! Closes *descriptor once and sets it to -1. A negative descriptor is left alone. */
void NFKPluginCloseDescriptor(int *descriptor);

#if DEBUG
/*! Whether INFERKIT_PLUGIN_TRACE=1 asks for the channel and descriptor trace. DEBUG builds only. */
BOOL NFKPluginDebugTracing(void);

/*! Writes one diagnostic line to standard error, with the process, the thread, and the time, while
	NFKPluginDebugTracing() answers YES. DEBUG builds only. */
void NFKPluginDebugTrace(NSString *format, ...) NS_FORMAT_FUNCTION(1, 2);

/*! The caller's frames as symbol names, innermost first, after skipping skip frames. */
NSString *NFKPluginDebugBacktrace(NSUInteger skip, NSUInteger count);

/*! The descriptor number's current owner and its recent adopt, disown, and close events with their
	callers. */
NSString *NFKPluginDebugDescriptorHistory(int descriptor);
#endif

/*! Message types, the "type" field of every message. */
extern NSString * const NFKPluginMessageChannel;
extern NSString * const NFKPluginMessageHello;
extern NSString * const NFKPluginMessageDescribe;
extern NSString * const NFKPluginMessageBuild;
extern NSString * const NFKPluginMessagePrepare;
extern NSString * const NFKPluginMessageRun;
extern NSString * const NFKPluginMessageProgress;
extern NSString * const NFKPluginMessageCancel;
extern NSString * const NFKPluginMessageUnload;
extern NSString * const NFKPluginMessageRelease;
extern NSString * const NFKPluginMessageStatus;
extern NSString * const NFKPluginMessageCallTool;
extern NSString * const NFKPluginMessagePing;
extern NSString * const NFKPluginMessageShutdown;
extern NSString * const NFKPluginMessageEvent;
extern NSString * const NFKPluginMessageLog;
extern NSString * const NFKPluginMessageReply;

/*! The event the plugin host sends once it has verified the bundle, before it reads hello. */
extern NSString * const NFKPluginEventVerified;

/*! The capability a host lists in hello to exchange files as open descriptors beside their paths. A
	sandboxed host offers it with its app group's sharedMemoryPrefix. */
extern NSString * const NFKPluginCapabilityFileDescriptors;

/*!
	@class      NFKPluginRunSlots
	@abstract   A backend's run slots: at most limit runs at once, and the rest wait in order.
	@discussion A waiting run is a block in a FIFO, so it holds no thread. admit: calls start at once on the
				calling thread when a slot is free, and otherwise when an earlier run's finish frees one, on
				the thread that calls finish. Every admitted start is matched by one finish.
*/
@interface NFKPluginRunSlots : NSObject

- (instancetype)initWithLimit:(NSInteger)limit NS_DESIGNATED_INITIALIZER;
- (instancetype)init NS_UNAVAILABLE;

/*! Calls start once a slot is free. */
- (void)admit:(dispatch_block_t)start;

/*! Frees the caller's slot, handing it to the oldest waiting start. */
- (void)finish;

/*! The runs that hold a slot now. */
@property (atomic, readonly) NSInteger activeCount;

/*! The starts waiting for a slot now. */
@property (atomic, readonly) NSInteger waitingCount;

@end

/*!
	@class      NFKPluginAttachmentWriter
	@abstract   The attachment sink for one outgoing message: each attachment is a shared-memory region,
				and descriptors lists them in index order for the frame.
	@discussion A region is created with shm_open and a random name, and unlinked at once. A sandboxed
				process passes its app group as namePrefix. When no shared-memory object can be made the
				region is an unlinked temporary file. The writer keeps its regions mapped until it is
				released, so it lives until the frame is sent.
*/
@interface NFKPluginAttachmentWriter : NSObject <NFKWireAttachmentSink>

/*! The app group prefix of a sandboxed process, such as "TEAMID.group.example/", or nil. */
@property (nonatomic, copy, nullable) NSString *namePrefix;

/*! Whether a file crosses as an open descriptor besides its path, for a sandboxed peer that cannot open
	the path. Defaults to NO. */
@property (nonatomic, assign) BOOL passesFilesAsDescriptors;

/*! The descriptors to send with the frame, in attachment index order. The writer still owns them. */
@property (nonatomic, readonly, copy) NSArray<NSNumber *> *descriptors;

@end

/*!
	@class      NFKPluginAttachmentReader
	@abstract   The attachment source for one incoming message, over the descriptors its frame carried.
	@discussion Each region maps on first use and its bytes are wrapped without a copy. A region that is
				a regular file, the temporary-file fallback, is copied when copiesFileRegions is YES: a
				peer that truncates the file would otherwise fault the mapping.
*/
@interface NFKPluginAttachmentReader : NSObject <NFKWireAttachmentSource>

/*! A reader over descriptors, which it takes ownership of. */
- (instancetype)initWithDescriptors:(NSArray<NFKPluginDescriptor *> *)descriptors NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;

/*! Whether a region backed by a regular file is copied before use. The host sets it for what a plugin
	sends. Defaults to YES. */
@property (nonatomic, assign) BOOL copiesFileRegions;

/*! The directory a file passed as a descriptor is copied into, or nil for the InferKit temporary
	directory. */
@property (nonatomic, copy, nullable) NSURL *temporaryDirectoryURL;

/*! The directory every file reference by path must lie inside, or nil to accept any absolute path. */
@property (nonatomic, copy, nullable) NSURL *allowedFileDirectoryURL;

@end

/*! The object that owns the files a plugin's result names, or nil. The files last as long as it does, so a
	holder of the result's file paths keeps it, and need not keep the result's values. */
id _Nullable NFKPluginResultFileOwner(NFKInferenceResult *result);

#endif

NS_ASSUME_NONNULL_END
