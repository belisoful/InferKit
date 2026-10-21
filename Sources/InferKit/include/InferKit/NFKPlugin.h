//
//  NFKPlugin.h
//  InferKit
//

#ifndef NFKPlugin_h
#define NFKPlugin_h

#import <Foundation/Foundation.h>
#import <InferKit/NFKInferenceBackend.h>
#import <InferKit/NFKInferenceTool.h>
#import <InferKit/NFKModelDescriptor.h>

NS_ASSUME_NONNULL_BEGIN

/*! The error domain for reading, verifying, loading, launching, and talking to a plugin. Introduced in
	InferKit 0.4.0. */
extern NSString * const NFKPluginErrorDomain;

/*!
	@enum       NFKPluginError
	@abstract   Error codes in NFKPluginErrorDomain.
	@constant   NFKPluginErrorMalformedManifest The manifest is not valid JSON, or a field is missing or
				has the wrong type.
	@constant   NFKPluginErrorUnsupportedManifestVersion The manifest's schemaVersion is newer than this
				library reads.
	@constant   NFKPluginErrorManifestMismatch The manifest disagrees with its bundle: the bundle
				identifier, the short version, or the executable differs.
	@constant   NFKPluginErrorIncompatible The plugin needs an API version, plugin ABI version, protocol
				version, system version, or architecture this host does not provide.
	@constant   NFKPluginErrorSignatureInvalid The plugin's code signature is missing, broken, or ad hoc
				under a policy that refuses ad hoc signatures.
	@constant   NFKPluginErrorTeamMismatch The plugin is signed by a team the policy does not allow.
	@constant   NFKPluginErrorLoadFailed The plugin's code could not be loaded, or its entry point is
				missing.
	@constant   NFKPluginErrorDuplicateRuntime The plugin carries its own copy of InferKit. One copy of
				InferKit runs in a process.
	@constant   NFKPluginErrorLaunchFailed The plugin's process could not be launched.
	@constant   NFKPluginErrorHandshakeFailed The plugin's process did not complete the hello exchange
				within the handshake timeout, or its answer was refused.
	@constant   NFKPluginErrorProtocolViolation The other side sent a frame or a message the protocol does
				not allow.
	@constant   NFKPluginErrorProtocolUnsupported The other side does not support the message, or speaks
				another major protocol version.
	@constant   NFKPluginErrorProcessExited The plugin's process exited during the operation. The error
				carries the exit status or signal and the end of its standard error.
	@constant   NFKPluginErrorUnknownModel The plugin has no model with the identifier.
	@constant   NFKPluginErrorUnknownTool The plugin has no tool with the name.
	@constant   NFKPluginErrorNotTrusted The operation needs a plugin the app trusts to run in its own
				process, or a plugin kind that can run there.
	@constant   NFKPluginErrorStopped The plugin was stopped before the operation finished.
	@constant   NFKPluginErrorTimedOut The plugin did not answer within a deadline: the hello exchange
				(handshakeTimeout), the host's verification of the bundle (verificationTimeout), a build or a
				preparation (buildTimeout), or a run that a restart cancelled and that did not end. The
				description names the step and the limit. Introduced in InferKit 0.4.0.
	@constant   NFKPluginErrorIdentifierInUse The catalog already lists a model under one of the plugin's
				catalog identifiers, the registry lists a plugin with the same identifier from another
				bundle, or another listed plugin declares a tool of the same name. Introduced in InferKit
				0.4.0.
	@constant   NFKPluginErrorUnknownPlugin The registry lists no plugin with the identifier. Introduced
				in InferKit 0.4.0.
	@constant   NFKPluginErrorInProcessUnloadable The plugin runs in the app's process, and a loaded bundle
				stays loaded until the app quits, so it cannot be stopped. Introduced in InferKit 0.4.0.
	Introduced in InferKit 0.4.0.
*/
typedef NS_ERROR_ENUM(NFKPluginErrorDomain, NFKPluginError) {
	NFKPluginErrorMalformedManifest = 1,
	NFKPluginErrorUnsupportedManifestVersion = 2,
	NFKPluginErrorManifestMismatch = 3,
	NFKPluginErrorIncompatible = 4,
	NFKPluginErrorSignatureInvalid = 5,
	NFKPluginErrorTeamMismatch = 6,
	NFKPluginErrorLoadFailed = 7,
	NFKPluginErrorDuplicateRuntime = 8,
	NFKPluginErrorLaunchFailed = 9,
	NFKPluginErrorHandshakeFailed = 10,
	NFKPluginErrorProtocolViolation = 11,
	NFKPluginErrorProtocolUnsupported = 12,
	NFKPluginErrorProcessExited = 13,
	NFKPluginErrorUnknownModel = 14,
	NFKPluginErrorUnknownTool = 15,
	NFKPluginErrorNotTrusted = 16,
	NFKPluginErrorStopped = 17,
	NFKPluginErrorTimedOut = 18,
	NFKPluginErrorIdentifierInUse = 19,
	NFKPluginErrorUnknownPlugin = 20,
	NFKPluginErrorInProcessUnloadable = 21,
};

/*!
	@enum       NFKPluginLogLevel
	@abstract   The severity of a message a plugin logs. The values match the C interface's
				NFK_PLUGIN_LOG_* constants.
	Introduced in InferKit 0.4.0.
*/
typedef NS_ENUM(NSInteger, NFKPluginLogLevel) {
	NFKPluginLogLevelDebug = 0,
	NFKPluginLogLevelInfo = 1,
	NFKPluginLogLevelWarning = 2,
	NFKPluginLogLevelError = 3,
};

/*!
	@protocol   NFKPluginContext
	@abstract   What the host gives a plugin when it creates it.
	@discussion The same context reaches a plugin in its own process and in the app's process. Introduced
				in InferKit 0.4.0.
*/
API_UNAVAILABLE(ios, tvos, watchos)
@protocol NFKPluginContext <NSObject>

/*! Writes a message to the host's log. Safe to call from any thread. */
- (void)logMessage:(NSString *)message level:(NFKPluginLogLevel)level;

/*! A directory the plugin may write. The host removes it when the plugin unloads. */
@property (nonatomic, readonly, copy) NSURL *temporaryDirectoryURL;

/*! The API version of the InferKit library the host runs, as "major.minor". */
@property (nonatomic, readonly, copy) NSString *hostAPIVersion;

/*! The values the app chose for the options the plugin declares, keyed by option key. */
@property (nonatomic, readonly, copy) NSDictionary<NSString *, id> *options;

/*! YES when the plugin runs in the app's own process, NO when it runs in a plugin host process. */
@property (nonatomic, readonly) BOOL runsInHostProcess;

@end

/*!
	@protocol   NFKPlugin
	@abstract   The principal class of an Objective-C or Swift plugin bundle.
	@discussion The host creates one instance per process with initWithContext:, reads its models and
				tools, and asks it for a backend per model. The plugin links InferKit's interface and
				does not embed its own copy: it is built with -undefined dynamic_lookup, or against the
				InferKit framework the app ships. The loader refuses a bundle that defines any NFK
				class. Introduced in InferKit 0.4.0.
*/
API_UNAVAILABLE(ios, tvos, watchos)
@protocol NFKPlugin <NSObject>

@required

/*! Creates the plugin. It keeps context for its life. */
- (instancetype)initWithContext:(id<NFKPluginContext>)context;

/*! The models the plugin runs. Each identifier matches a model the manifest lists. */
- (NSArray<NFKModelDescriptor *> *)modelDescriptors;

/*! A backend for one of the plugin's models, configured with the option values in options, or nil with
	an error. */
- (nullable id<NFKInferenceBackend>)backendForModelIdentifier:(NSString *)modelIdentifier
													  options:(NSDictionary<NSString *, id> *)options
														error:(NSError **)error;

@optional

/*! The tools the plugin offers. Each name matches a tool the manifest lists. */
- (NSArray<NFKInferenceTool *> *)tools;

/*! The options the plugin reads from its context, beyond each model's own parameters. */
- (NSArray<NFKModelOption *> *)pluginOptions;

/*! Called before the host releases the plugin, so it can free what it holds. */
- (void)willUnload;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKPlugin_h */
