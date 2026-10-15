//
//  NFKPluginTestPeer.h
//  InferKitTests
//
//  An in-test plugin process: a launcher that, in place of spawning a process, runs a peer in the
//  test process that performs the channel hand-off and answers the protocol. NFKPluginProcessTests
//  and NFKPluginBackendTests drive the supervisor and the backend through it.
//

#import <Foundation/Foundation.h>
#import <InferKit/InferKit.h>

#if TARGET_OS_OSX

#import "../../Sources/InferKit/NFKPluginChannel.h"
#import "../../Sources/InferKit/NFKPluginProcessPrivate.h"
#import "../../Sources/InferKit/NFKPluginWire.h"

NS_ASSUME_NONNULL_BEGIN

@class NFKPluginTestPeer;

/*! Answers one request in place of the default answer. Runs on a background queue, so it may block. */
typedef void (^NFKPluginTestPeerHandler)(NFKPluginTestPeer *peer, NSDictionary<NSString *, id> *message, NSArray<NFKPluginDescriptor *> *descriptors);

/*!
	@class      NFKPluginTestPeer
	@abstract   One launch of an in-test plugin.
	@discussion By default it answers hello with the manifest's identity, describe with the manifest's
				models, build with a new handle, prepare and status with a loaded model, run with the
				request's prompt prefixed by its launch number ("1:hello"), unload with an "unloaded"
				event, callTool with "<tool> called", and shutdown by closing its channel. Handlers set
				per message type replace the default answer.
*/
@interface NFKPluginTestPeer : NSObject <NFKPluginLauncher>

/*! The launch this peer serves, counted from 1. */
@property (nonatomic, assign) NSUInteger launchNumber;

/*! The plugin identifier and version the hello reply names. Default: the manifest's at launch. */
@property (atomic, copy, nullable) NSString *pluginIdentifier;
@property (atomic, copy, nullable) NSString *pluginVersion;

/*! The protocol the hello reply names. Default: {1, 0}. */
@property (atomic, copy) NSDictionary<NSString *, id> *protocol;

/*! The model identifiers describe lists. Default: the manifest's. */
@property (atomic, copy, nullable) NSArray<NSString *> *describedModelIdentifiers;

/*! Handlers by message type that replace the default answer. */
@property (atomic, copy) NSDictionary<NSString *, NFKPluginTestPeerHandler> *handlers;

/*! Whether a launch fails. */
@property (atomic, assign) BOOL failsLaunch;

/*! What the peer reports after it ends. */
@property (atomic, readwrite) NFKSubprocessTerminationReason terminationReason;
@property (atomic, readwrite) int terminationStatus;
@property (atomic, readwrite, copy) NSString *recentStandardError;

/*! Whether the supervisor stopped the peer. */
@property (atomic, readonly) BOOL stopped;

/*! The process the audit token the supervisor read names. */
@property (atomic, readonly) pid_t auditedProcessIdentifier;

/*! Every message received, in order. */
- (NSArray<NSDictionary<NSString *, id> *> *)receivedMessages;

/*! The type of every message received, in order. */
- (NSArray<NSString *> *)receivedTypes;

/*! The received messages of one type. */
- (NSArray<NSDictionary<NSString *, id> *> *)receivedMessagesOfType:(NSString *)type;

/*! Waits until a message of type has arrived, up to timeout. */
- (nullable NSDictionary<NSString *, id> *)waitForMessageOfType:(NSString *)type timeout:(NSTimeInterval)timeout;

/*! Replies to a request. */
- (void)replyTo:(NSDictionary<NSString *, id> *)request result:(id)result;
- (void)replyTo:(NSDictionary<NSString *, id> *)request error:(NSError *)error;
- (void)replyCancelledTo:(NSDictionary<NSString *, id> *)request;

/*! Replies to a run with result, its large values as attachments. */
- (void)replyTo:(NSDictionary<NSString *, id> *)request inferenceResult:(NFKInferenceResult *)result;

/*! Sends a run's progress, with a partial result when one is given. */
- (void)sendProgress:(double)progress partial:(nullable NFKInferenceResult *)partial forRequest:(NSDictionary<NSString *, id> *)request;

/*! Sends a message that expects no reply. */
- (void)sendMessage:(NSDictionary<NSString *, id> *)message;

/*! The inference request a run message carries, with its attachments. */
- (nullable NFKInferenceRequest *)requestOfRun:(NSDictionary<NSString *, id> *)run descriptors:(NSArray<NFKPluginDescriptor *> *)descriptors;

/*! Ends the peer as a process a signal killed: it closes its channel without a shutdown. */
- (void)crashWithSignal:(int)signal standardError:(NSString *)standardError;

@end

/*!
	@class      NFKPluginTestHarness
	@abstract   The launcher factory of a test: one new peer per launch, configured before it starts.
*/
@interface NFKPluginTestHarness : NSObject

/*! Configures each peer before it launches. */
@property (atomic, copy, nullable) void (^configure)(NFKPluginTestPeer *peer);

/*! Every peer launched, in order. */
@property (atomic, readonly, copy) NSArray<NFKPluginTestPeer *> *peers;

/*! The peer of the launch at index, counted from 0, once that launch has happened, up to timeout. A
	submitted job launches on a background queue, so its peer appears after the call returns. */
- (nullable NFKPluginTestPeer *)peerAtIndex:(NSUInteger)index timeout:(NSTimeInterval)timeout;

/*! The factory to set as the supervisor's launcherFactory. */
- (NFKPluginLauncherFactory)factory;

@end

/*! A manifest for plugin com.example.echo at version, with model "echo" (prompt in, text out, a request
	parameter "temperature", and a load option "size") and tool "shout". */
NFKPluginManifest *NFKPluginTestManifest(NSString *version, NSTimeInterval idleExitSeconds);

/*! Polls condition until it holds or timeout passes. */
BOOL NFKPluginTestWaitUntil(NSTimeInterval timeout, BOOL (^condition)(void));

NS_ASSUME_NONNULL_END

#endif
