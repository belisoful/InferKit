//
//  NFKSubprocess.h
//  InferKit
//

#ifndef NFKSubprocess_h
#define NFKSubprocess_h

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

/*! The error domain for launching, writing to, and stopping a subprocess. Introduced in InferKit 0.4.0. */
extern NSString * const NFKSubprocessErrorDomain;

/*!
	@enum       NFKSubprocessError
	@abstract   Error codes in NFKSubprocessErrorDomain.
	@constant   NFKSubprocessErrorLaunchFailed The executable is not found or the system refused to
				start it. The POSIX error is under NSUnderlyingErrorKey.
	@constant   NFKSubprocessErrorNotRunning The process is not running, or its standard input is
				closed.
	@constant   NFKSubprocessErrorWriteFailed The process closed its standard input, or the write
				failed. The POSIX error is under NSUnderlyingErrorKey.
	@constant   NFKSubprocessErrorWriteTimedOut The process read none of the data within writeTimeout.
	Introduced in InferKit 0.4.0.
*/
typedef NS_ERROR_ENUM(NFKSubprocessErrorDomain, NFKSubprocessError) {
	NFKSubprocessErrorLaunchFailed = 1,
	NFKSubprocessErrorNotRunning = 2,
	NFKSubprocessErrorWriteFailed = 3,
	NFKSubprocessErrorWriteTimedOut = 4,
};

/*!
	@enum       NFKSubprocessState
	@abstract   Where a subprocess is in its life.
	@constant   NFKSubprocessStateIdle Not started yet.
	@constant   NFKSubprocessStateStarting Launching.
	@constant   NFKSubprocessStateRunning Launched and not yet exited.
	@constant   NFKSubprocessStateExited Exited. terminationReason and terminationStatus say how, and
				restartPending says whether a relaunch is scheduled.
	@constant   NFKSubprocessStateFailedToLaunch The launch failed. launchError says why.
	Introduced in InferKit 0.4.0.
*/
typedef NS_ENUM(NSInteger, NFKSubprocessState) {
	NFKSubprocessStateIdle = 0,
	NFKSubprocessStateStarting = 1,
	NFKSubprocessStateRunning = 2,
	NFKSubprocessStateExited = 3,
	NFKSubprocessStateFailedToLaunch = 4,
};

/*!
	@enum       NFKSubprocessTerminationReason
	@abstract   How a subprocess ended.
	@constant   NFKSubprocessTerminationReasonNone It has not ended, or its status could not be read.
	@constant   NFKSubprocessTerminationReasonExit It exited. terminationStatus is its exit status.
	@constant   NFKSubprocessTerminationReasonSignal A signal ended it. terminationStatus is the signal
				number.
	Introduced in InferKit 0.4.0.
*/
typedef NS_ENUM(NSInteger, NFKSubprocessTerminationReason) {
	NFKSubprocessTerminationReasonNone = 0,
	NFKSubprocessTerminationReasonExit = 1,
	NFKSubprocessTerminationReasonSignal = 2,
};

/*!
	@enum       NFKSubprocessRestartPolicy
	@abstract   Whether a subprocess launches again after it ends on its own.
	@constant   NFKSubprocessRestartPolicyNever It stays exited.
	@constant   NFKSubprocessRestartPolicyOnCrash A nonzero exit status or a signal launches it again
				after a backoff, up to maximumRestarts consecutive times. An exit with status 0 leaves
				it exited, and so does an exit that stop caused.
	Introduced in InferKit 0.4.0.
*/
typedef NS_ENUM(NSInteger, NFKSubprocessRestartPolicy) {
	NFKSubprocessRestartPolicyNever = 0,
	NFKSubprocessRestartPolicyOnCrash = 1,
};

@class NFKSubprocess;

/*! Receives each line a subprocess writes to standard output, without its line ending. Called on the
	subprocess's serial event queue. Introduced in InferKit 0.4.0. */
typedef void (^ NS_SWIFT_SENDABLE NFKSubprocessLineHandler)(NSData *line);

/*! Receives each line a subprocess writes to standard error, decoded as UTF-8. Called on the
	subprocess's serial event queue. Introduced in InferKit 0.4.0. */
typedef void (^ NS_SWIFT_SENDABLE NFKSubprocessTextLineHandler)(NSString *line);

/*! Receives each state a subprocess enters. Called on the subprocess's serial event queue, after
	every standard output line the run wrote. Introduced in InferKit 0.4.0. */
typedef void (^ NS_SWIFT_SENDABLE NFKSubprocessStateHandler)(NFKSubprocess *subprocess, NFKSubprocessState state) API_UNAVAILABLE(ios, tvos, watchos);

/*!
	@class      NFKSubprocess
	@abstract   Launches a program, frames its output into lines, and supervises it until it stops.
	@discussion A subprocess runs one executable with an argument array. No shell parses the command.
				A command without a slash is found on the PATH of the environment the child receives.
				A GUI application's PATH is /usr/bin:/bin:/usr/sbin:/sbin, so a command PATH does not
				hold is then searched in the directories NFKAgentCLI searches: ~/.local/bin, Homebrew's
				prefixes, and the npm-global, Bun, Volta, Cargo, and nvm bins. A command found there
				has its directory appended to the child's PATH, so a script it runs through
				/usr/bin/env finds its interpreter beside it. An absolute path is used as given. A
				command found nowhere fails the launch with an error naming every directory searched.

				The child starts in its own process group, with default signal dispositions and only
				its three standard streams open. Standard output arrives line by line through
				lineHandler. Standard error goes into a buffer holding the last
				standardErrorCapacity bytes, read through recentStandardError, and to
				standardErrorLineHandler when one is set. writeData:error: blocks while the child's
				pipe is full, which is the backpressure a reader that falls behind applies.

				stop closes standard input, waits stopInputTimeout for the child to exit, sends
				SIGTERM to its process group, waits stopTerminateTimeout, and sends SIGKILL.

				A running subprocess keeps itself alive until its process exits. The command,
				environment, and working directory apply at each launch, the timeouts at each stop,
				the restart settings at each exit, and the handlers at once. The class is macOS only.
				Introduced in InferKit 0.4.0.
*/
API_UNAVAILABLE(ios, tvos, watchos)
@interface NFKSubprocess : NSObject

/*! The executable: an absolute path, a path relative to workingDirectoryURL, or a name searched on
	the child's PATH. */
@property (nonatomic, readonly, copy) NSString *command;

/*! The arguments after the executable. */
@property (nonatomic, readonly, copy) NSArray<NSString *> *arguments;

/*! Variables set in the child's environment. With inheritsEnvironment, they override this process's
	variables of the same name. */
@property (nonatomic, copy) NSDictionary<NSString *, NSString *> *environment;

/*! Whether the child starts from this process's environment. NO starts it from environment alone.
	Defaults to YES. */
@property (nonatomic, assign) BOOL inheritsEnvironment;

/*! The names of this process's variables the child inherits, or nil for every one. A name this
	process does not set is skipped. Applies when inheritsEnvironment is YES. Defaults to nil. */
@property (nonatomic, copy, nullable) NSArray<NSString *> *inheritedEnvironmentVariables;

/*! The child's working directory, or nil for this process's. */
@property (nonatomic, copy, nullable) NSURL *workingDirectoryURL;

/*! Receives standard output line by line. A line ends at a newline, and a carriage return before it
	is dropped. Output left without a newline at exit arrives as a final line. */
@property (atomic, copy, nullable) NFKSubprocessLineHandler lineHandler;

/*! Receives standard error line by line, in addition to the recentStandardError buffer. */
@property (atomic, copy, nullable) NFKSubprocessTextLineHandler standardErrorLineHandler;

/*! Receives every state change. */
@property (atomic, copy, nullable) NFKSubprocessStateHandler stateHandler;

/*! How many bytes of standard error recentStandardError keeps. Defaults to 65536. */
@property (nonatomic, assign) NSUInteger standardErrorCapacity;

/*! The longest line delivered, in bytes. A longer line is dropped through its newline. Defaults to
	268435456 (256 MiB). */
@property (nonatomic, assign) NSUInteger maximumLineLength;

/*! How long writeData:error: waits for the child to read, in seconds. 0 waits until the child reads
	or exits. Defaults to 30. */
@property (nonatomic, assign) NSTimeInterval writeTimeout;

/*! How long stop waits after closing standard input before it sends SIGTERM, in seconds. Defaults to 2. */
@property (nonatomic, assign) NSTimeInterval stopInputTimeout;

/*! How long stop waits after SIGTERM before it sends SIGKILL, in seconds. Defaults to 3. */
@property (nonatomic, assign) NSTimeInterval stopTerminateTimeout;

/*! Whether the process launches again after it crashes. Defaults to NFKSubprocessRestartPolicyNever. */
@property (nonatomic, assign) NFKSubprocessRestartPolicy restartPolicy;

/*! How many consecutive crashes relaunch the process before it stays exited. Defaults to 5. */
@property (nonatomic, assign) NSUInteger maximumRestarts;

/*! The wait before the first relaunch, in seconds. Each consecutive crash doubles it, up to
	maximumRestartDelay. Defaults to 0.5. */
@property (nonatomic, assign) NSTimeInterval restartDelay;

/*! The longest wait before a relaunch, in seconds. Defaults to 30. */
@property (nonatomic, assign) NSTimeInterval maximumRestartDelay;

/*! A run that lasts this long, in seconds, resets the consecutive crash count. Defaults to 60. */
@property (nonatomic, assign) NSTimeInterval restartResetInterval;

/*! The current state. */
@property (atomic, readonly) NFKSubprocessState state;

/*! The running process's identifier, which is also its process group, or 0 when none runs. */
@property (atomic, readonly) pid_t processIdentifier;

/*! How the last run ended. */
@property (atomic, readonly) NFKSubprocessTerminationReason terminationReason;

/*! The last run's exit status, or the signal that ended it, per terminationReason. */
@property (atomic, readonly) int terminationStatus;

/*! Why the last launch failed, or nil. */
@property (atomic, readonly, copy, nullable) NSError *launchError;

/*! How many times the restart policy has launched the process again. */
@property (atomic, readonly) NSUInteger restartCount;

/*! Whether a relaunch is scheduled after the last crash. */
@property (atomic, readonly, getter=isRestartPending) BOOL restartPending;

/*! The last standardErrorCapacity bytes of standard error, across runs, decoded as UTF-8. */
@property (atomic, readonly, copy) NSString *recentStandardError;

- (instancetype)initWithCommand:(NSString *)command arguments:(NSArray<NSString *> *)arguments NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;

/*! Launches the process and returns once it runs, or returns NO when the launch fails. A running
	process is left alone, and a scheduled relaunch happens now. */
- (BOOL)startWithError:(NSError * _Nullable *)error;

/*! Writes data to standard input. Blocks while the pipe is full, up to writeTimeout. Calls are
	serialized, so concurrent writes do not interleave. */
- (BOOL)writeData:(NSData *)data error:(NSError * _Nullable *)error;

/*! Writes line and a newline as one write. */
- (BOOL)writeLine:(NSData *)line error:(NSError * _Nullable *)error;

/*! Closes standard input, which tells a child that reads to its end that no more input comes. */
- (void)closeStandardInput;

/*!
	@abstract   Stops the process and calls completionHandler once it has exited.
	@discussion Closes standard input, then sends SIGTERM after stopInputTimeout and SIGKILL after
				stopTerminateTimeout, each to the process group and only while the process runs. A
				scheduled relaunch is canceled, and the restart policy does not relaunch a stopped
				process. completionHandler runs on the subprocess's event queue, at once when no
				process runs.
*/
- (void)stopWithCompletionHandler:(nullable void (^ NS_SWIFT_SENDABLE)(void))completionHandler;

/*! Stops the process and waits until it has exited. Called from a state or line handler, it starts
	the stop and returns without waiting. */
- (void)stop;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKSubprocess_h */
