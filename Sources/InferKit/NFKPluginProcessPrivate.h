//
//  NFKPluginProcessPrivate.h
//  InferKit
//
//  The host side's internals: the launcher seam a test replaces to attach an in-test peer, the live
//  connection to a plugin's process, and the initializers a supervisor and a registry make backends
//  with. Docs/agent-reference/plugins.md describes the protocol they speak.
//

#import <Foundation/Foundation.h>
#import <InferKit/NFKPluginProcess.h>
#import <InferKit/NFKPluginBackend.h>
#include <mach/message.h>

NS_ASSUME_NONNULL_BEGIN

#if TARGET_OS_OSX

@class NFKPluginDescriptor;
@class NFKPluginConnection;

/*!
	@protocol   NFKPluginLauncher
	@abstract   Starts and stops one process of a plugin.
	@discussion The supervisor asks its launcherFactory for a launcher per launch. The default launcher
				runs the plugin host or the plugin's own executable through NFKSubprocess. A test supplies
				one that runs an in-test peer on the descriptor instead, which performs the same channel
				hand-off a plugin's process performs.
*/
@protocol NFKPluginLauncher <NSObject>

/*! Launches the process so that it inherits descriptor as its channel, the descriptor number
	NFKPluginChannelDescriptorNumber in a spawned child. Returns once the process runs. The launcher
	neither takes nor closes descriptor; the supervisor closes it when this returns. */
- (BOOL)launchPluginProcess:(NFKPluginProcess *)process channelDescriptor:(int)descriptor error:(NSError **)error;

/*! Stops the process, with escalation, and calls completionHandler once it has exited. Calls it at once
	when the process has already exited. */
- (void)stopWithCompletionHandler:(void (^)(void))completionHandler;

/*! The process's identifier, or 0 when the launcher runs no separate process. */
@property (atomic, readonly) pid_t processIdentifier;

/*! How the process ended. */
@property (atomic, readonly) NFKSubprocessTerminationReason terminationReason;

/*! The exit status or the signal, per terminationReason. */
@property (atomic, readonly) int terminationStatus;

/*! The end of the process's standard error. */
@property (atomic, readonly, copy) NSString *recentStandardError;

@optional

/*! The subprocess the launcher runs, or nil. */
@property (atomic, readonly, strong, nullable) NFKSubprocess *subprocess;

/*! Checks the process on the other end of the channel in place of the signature policy. */
- (BOOL)validatePeerWithAuditToken:(audit_token_t)auditToken process:(NFKPluginProcess *)process error:(NSError **)error;

@end

/*! Makes the launcher for one launch. */
typedef id<NFKPluginLauncher> _Nonnull (^NFKPluginLauncherFactory)(NFKPluginProcess *process);

/*! Receives a progress message of a request, with the descriptors its frame carried. */
typedef void (^NFKPluginProgressHandler)(NSDictionary<NSString *, id> *message, NSArray<NFKPluginDescriptor *> *descriptors);

/*! Receives a request's reply once: the reply message with its descriptors, or the error that ended the
	request first. */
typedef void (^NFKPluginReplyHandler)(NSDictionary<NSString *, id> * _Nullable reply,
									  NSArray<NFKPluginDescriptor *> *descriptors,
									  NSError * _Nullable error);

/*!
	@class      NFKPluginConnection
	@abstract   One launched process of a plugin: its channel, its requests in flight, and its backends.
	@discussion A connection belongs to one generation. Every launch makes a new one with the next
				generation, so a backend handle from an earlier process is recognized as stale. A
				request's reply handler runs on the channel's handler queue.
*/
@interface NFKPluginConnection : NSObject

/*! The generation, counted from 1 per supervisor. */
@property (nonatomic, readonly) NSUInteger generation;

/*! The largest value a message inlines, in bytes. */
@property (nonatomic, readonly) NSUInteger inlineByteLimit;

/*! The plugin's hello reply. */
@property (atomic, readonly, copy) NSDictionary<NSString *, id> *helloReply;

/*! Whether the connection has closed or is closing. */
@property (atomic, readonly, getter=isClosed) BOOL closed;

/*! Sends a request and returns its identifier, or 0 with an error. message carries no "id"; the
	connection adds one. progressHandler receives the request's progress messages. */
- (NSInteger)sendRequest:(NSDictionary<NSString *, id> *)message
			 descriptors:(nullable NSArray<NSNumber *> *)descriptors
		 progressHandler:(nullable NFKPluginProgressHandler)progressHandler
			replyHandler:(NFKPluginReplyHandler)replyHandler
				   error:(NSError **)error;

/*! Sends a request and waits for its reply up to timeout (no limit when not positive). Returns the
	reply's "result", NSNull when it has none, or nil with the reply's error. */
- (nullable id)sendRequestAndWait:(NSDictionary<NSString *, id> *)message
						  timeout:(NSTimeInterval)timeout
							error:(NSError **)error;

/*! Stops counting a request as in flight, so it holds neither an idle exit nor a restart's drain, and
	still delivers its reply, when one comes, to its reply handler. */
- (void)detachRequest:(NSInteger)identifier;

/*! Sends a message that expects no reply. */
- (BOOL)sendMessage:(NSDictionary<NSString *, id> *)message error:(NSError **)error;

/*! Routes the plugin's events for handle to backend, and counts it as a live backend. */
- (void)registerBackend:(NFKPluginBackend *)backend handle:(id)handle;

/*! Releases the plugin's backend for handle and stops counting it. Sends nothing once closed. */
- (void)releaseBackendHandle:(id)handle;

@end

@interface NFKPluginProcess ()

/*! Makes the launcher for each launch. nil uses the NFKSubprocess launcher. */
@property (atomic, copy, nullable) NFKPluginLauncherFactory launcherFactory;

/*! The memory pressure reading in 0...1. nil reads NFKServedHostStatus. */
@property (atomic, copy, nullable) double (^memoryPressureProvider)(void);

/*! The pressure at which the idle exit interval starts to shorten. Defaults to 0.6, as the server's. */
@property (atomic, assign) double memoryPressureOnset;

/*! The idle exit interval at full pressure, in seconds. Defaults to 10, as the server's. */
@property (atomic, assign) NSTimeInterval shortestIdleExitInterval;

/*! The values of the plugin's own options, sent in hello. */
@property (atomic, copy) NSDictionary<NSString *, id> *pluginOptions;

/*! The generation of the running process, or of the last one. 0 before the first launch. */
- (NSUInteger)generation;

/*! The idle exit interval memory pressure leaves now. */
- (NSTimeInterval)effectiveIdleExitInterval;

/*! Runs block with the live connection on the calling thread, starting the process when none runs.
	During a restart it waits its turn first. block sends what it needs and returns; the next held
	request waits for it. */
- (BOOL)performWithReservation:(nullable id)reservation
						 block:(BOOL (^)(NFKPluginConnection *connection, NSError **error))block
						 error:(NSError **)error;

/*! Runs block as performWithReservation:block:error: does, without blocking the caller, and calls
	completion once with whether it ran and the error that stopped it. During a restart the request takes
	its place in order at the call, and the restart runs block in its turn on its own queue, so a held
	request needs no thread while it waits. Otherwise block runs on the process's submission queue. */
- (void)performAsynchronouslyWithBlock:(BOOL (^)(NFKPluginConnection *connection, NSError **error))block
							completion:(void (^)(BOOL performed, NSError * _Nullable error))completion;

/*! The supervisor's state, the restart's phase, the held requests, and the live connection's requests
	in flight, for a test's failure message. */
- (NSString *)diagnosticDescription;

/*! The live connection when the process is Ready and no restart holds requests, else nil. Starts
	nothing. */
- (nullable NFKPluginConnection *)readyConnection;

/*! Reports a log message through logHandler. */
- (void)logMessage:(NSString *)message level:(NFKPluginLogLevel)level;

@end

@interface NFKPluginBackend ()

/*! A backend for one of the plugin's models, run in process's process. */
- (instancetype)initWithProcess:(NFKPluginProcess *)process
					 descriptor:(NFKModelDescriptor *)descriptor
						options:(NSDictionary<NSString *, id> *)options;

/*! A backend that forwards every call to a plugin's own backend in this process. */
- (instancetype)initWithPluginIdentifier:(NSString *)pluginIdentifier
							  descriptor:(NFKModelDescriptor *)descriptor
								 options:(NSDictionary<NSString *, id> *)options
						   pluginBackend:(id<NFKInferenceBackend>)pluginBackend;

/*! The generation the plugin's backend was built in, or 0 before a build. */
- (NSUInteger)builtGeneration;

/*! Applies an event the plugin sent for this backend. Runs on the channel's handler queue. */
- (void)handleEvent:(NSDictionary<NSString *, id> *)event;

@end

/*! The error a run the plugin cancelled fails with: NSCocoaErrorDomain, NSUserCancelledError. */
NSError *NFKPluginCancellationError(void);

/*! The error a reply carries: its decoded "error", the cancellation error for "cancelled", or nil. */
NSError * _Nullable NFKPluginErrorFromReply(NSDictionary<NSString *, id> *reply);

#endif

NS_ASSUME_NONNULL_END
