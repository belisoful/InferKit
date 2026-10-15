//
//  NFKPluginProcess.h
//  InferKit
//

#ifndef NFKPluginProcess_h
#define NFKPluginProcess_h

#import <Foundation/Foundation.h>
#import <InferKit/NFKInferenceTool.h>
#import <InferKit/NFKPlugin.h>
#import <InferKit/NFKPluginManifest.h>
#import <InferKit/NFKPluginSignaturePolicy.h>
#import <InferKit/NFKSubprocess.h>

NS_ASSUME_NONNULL_BEGIN

@class NFKPluginBackend;
@class NFKPluginProcess;

/*!
	@enum       NFKPluginProcessState
	@abstract   Where a plugin's process is in its life.
	@constant   NFKPluginProcessStateNotRunning No process runs. The next request starts one.
	@constant   NFKPluginProcessStateStarting The process is verified, launched, and greeted.
	@constant   NFKPluginProcessStateReady The process answered the hello exchange and serves requests.
	@constant   NFKPluginProcessStateRestarting The process is replaced by a new version. New requests
				wait, in order, for the new process.
	@constant   NFKPluginProcessStateFailed The process cannot run. failureError says why. A start
				clears it.
	Introduced in InferKit 0.4.0.
*/
typedef NS_ENUM(NSInteger, NFKPluginProcessState) {
	NFKPluginProcessStateNotRunning = 0,
	NFKPluginProcessStateStarting = 1,
	NFKPluginProcessStateReady = 2,
	NFKPluginProcessStateRestarting = 3,
	NFKPluginProcessStateFailed = 4,
};

/*! Receives each state a plugin process enters, on the process's event queue. Introduced in InferKit
	0.4.0. */
typedef void (^ NS_SWIFT_SENDABLE NFKPluginProcessStateHandler)(NFKPluginProcess *process, NFKPluginProcessState state) API_UNAVAILABLE(ios, tvos, watchos);

/*! Receives the messages a plugin's process logs and the supervisor's own notices, on the process's
	event queue. Introduced in InferKit 0.4.0. */
typedef void (^ NS_SWIFT_SENDABLE NFKPluginProcessLogHandler)(NFKPluginProcess *process, NFKPluginLogLevel level, NSString *message) API_UNAVAILABLE(ios, tvos, watchos);

/*!
	@class      NFKPluginProcess
	@abstract   Supervises the process one plugin runs in.
	@discussion The process is the plugin host serving an "objc" or "c-abi" bundle, or an "executable"
				plugin's own program. A start verifies the code signature, launches the process with a
				socket on descriptor 3, completes the hello exchange, and checks that the process on the
				other end runs the verified code. A process that exits is not relaunched at once: the
				next request starts it again. crashLimit crashes within crashLimitInterval leave it
				Failed.

				A restart replaces the process with a new version while the app runs. New requests wait
				in order, runs in flight finish (up to restartDrainTimeout, then they are cancelled),
				the old process stops, and the new one starts. A launch failure fails the waiting
				requests with its reason. Backends made before the restart rebuild themselves before
				their next run.

				An idle process with no live backends exits after idleExitInterval. The class is macOS
				only. Introduced in InferKit 0.4.0.
*/
API_UNAVAILABLE(ios, tvos, watchos)
@interface NFKPluginProcess : NSObject

/*! A supervisor for the plugin manifest describes. hostExecutableURL is the plugin host that serves an
	"objc" or "c-abi" plugin; an "executable" plugin runs its own program and ignores it. */
- (instancetype)initWithManifest:(NFKPluginManifest *)manifest
			   hostExecutableURL:(nullable NSURL *)hostExecutableURL NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

/*! The manifest of the version that runs, or that the next start runs. */
@property (atomic, readonly, copy) NFKPluginManifest *manifest;

/*! The plugin host executable, or nil. */
@property (nonatomic, readonly, copy, nullable) NSURL *hostExecutableURL;

/*! The signatures the process's code must carry. Defaults to +[NFKPluginSignaturePolicy defaultPolicy]. */
@property (atomic, copy) NFKPluginSignaturePolicy *signaturePolicy;

/*! How long the hello exchange may take, in seconds. Defaults to 10. For an "objc" or "c-abi" plugin
	the clock starts once the plugin host has verified the bundle. */
@property (atomic, assign) NSTimeInterval handshakeTimeout;

/*! How long the plugin host may take to verify the bundle after it hands over its channel, in seconds.
	Defaults to 120. Verification hashes every file the bundle's seal covers, so a plugin keeps large
	weights outside its bundle. */
@property (atomic, assign) NSTimeInterval verificationTimeout;

/*! How long an idle process with no live backends runs before it exits, in seconds. Defaults to the
	manifest's idleExitInterval. Memory pressure shortens it as it shortens NFKInferenceServer's idle
	unload. A value that is not positive keeps an idle process running. */
@property (atomic, assign) NSTimeInterval idleExitInterval;

/*! How long a restart waits for runs in flight before it cancels them, in seconds. Defaults to 120. */
@property (atomic, assign) NSTimeInterval restartDrainTimeout;

/*! How many crashes within crashLimitInterval leave the process Failed. Defaults to 3. */
@property (atomic, assign) NSUInteger crashLimit;

/*! The window crashLimit counts crashes in, in seconds. Defaults to 60. */
@property (atomic, assign) NSTimeInterval crashLimitInterval;

/*! The current state. */
@property (atomic, readonly) NFKPluginProcessState state;

/*! Why the process is Failed, or why the last start failed, or nil. */
@property (atomic, readonly, copy, nullable) NSError *failureError;

/*! How the last process ended. */
@property (atomic, readonly) NFKSubprocessTerminationReason terminationReason;

/*! The last process's exit status, or the signal that ended it, per terminationReason. */
@property (atomic, readonly) int terminationStatus;

/*! The running process's identifier, or 0. */
@property (atomic, readonly) pid_t processIdentifier;

/*! The subprocess that runs the plugin, or nil before the first start. */
@property (atomic, readonly, strong, nullable) NFKSubprocess *subprocess;

/*! Receives every state change. */
@property (atomic, copy, nullable) NFKPluginProcessStateHandler stateHandler;

/*! Receives the plugin's log messages, and the differences the supervisor finds between the manifest
	and what the running plugin describes. */
@property (atomic, copy, nullable) NFKPluginProcessLogHandler logHandler;

/*! Starts the process and returns once it is Ready, or returns NO with an error. A Ready process is left
	alone. Blocks for up to verificationTimeout and handshakeTimeout after the launch. */
- (BOOL)startWithError:(NSError **)error;

/*! Asks the process to shut down, then stops it with NFKSubprocess's escalation, and calls
	completionHandler once it has exited. Runs in flight fail with NFKPluginErrorStopped. */
- (void)stopWithCompletionHandler:(nullable void (^ NS_SWIFT_SENDABLE)(void))completionHandler;

/*! Replaces the process with the version manifest describes. completionHandler receives nil once the new
	process is Ready, or the error that failed it. */
- (void)restartWithManifest:(NFKPluginManifest *)manifest
		  completionHandler:(nullable void (^ NS_SWIFT_SENDABLE)(NSError * _Nullable error))completionHandler;

/*! A backend for one of the plugin's models, configured with options. Starts nothing: the backend
	starts the process when it prepares or runs. Returns nil with NFKPluginErrorUnknownModel when the
	manifest lists no such model. */
- (nullable NFKPluginBackend *)backendForModelIdentifier:(NSString *)modelIdentifier
												 options:(NSDictionary<NSString *, id> *)options
												   error:(NSError **)error;

/*! The plugin's tools, from its manifest. Calling one starts the process. */
@property (nonatomic, readonly, copy) NSArray<NFKInferenceTool *> *tools;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKPluginProcess_h */
