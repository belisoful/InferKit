//
//  NFKPluginRegistry.h
//  InferKit
//

#ifndef NFKPluginRegistry_h
#define NFKPluginRegistry_h

#import <Foundation/Foundation.h>
#import <InferKit/NFKInferenceTool.h>
#import <InferKit/NFKModelCatalog.h>
#import <InferKit/NFKPluginBackend.h>
#import <InferKit/NFKPluginManifest.h>
#import <InferKit/NFKPluginProcess.h>
#import <InferKit/NFKPluginSignaturePolicy.h>

NS_ASSUME_NONNULL_BEGIN

/*!
	@class      NFKPluginStatus
	@abstract   Where a listed plugin runs and whether it is running, at one moment.
	@discussion A plugin in its own process reports its process's state. A plugin the app trusts reports
				NFKPluginProcessStateReady once its code is loaded into the app's process, which it stays
				in until the app quits, NFKPluginProcessStateFailed when its load failed, and
				NFKPluginProcessStateNotRunning before. The class is macOS only. Introduced in InferKit
				0.4.0.
*/
API_UNAVAILABLE(ios, tvos, watchos)
@interface NFKPluginStatus : NSObject

/*! The plugin's identifier. */
@property (nonatomic, readonly, copy) NSString *pluginIdentifier;
/*! The version the plugin runs, or the version listed when it is not running. */
@property (nonatomic, readonly, copy) NSString *version;
/*! Where the plugin runs: in its own process, or in the app's process for a plugin the app trusts. */
@property (nonatomic, readonly) NFKPluginIsolation isolation;
/*! Whether the plugin is running. */
@property (nonatomic, readonly) NFKPluginProcessState processState;
/*! Why the last start or load failed, such as a refused signature; nil unless processState is failed. */
@property (nonatomic, readonly, copy, nullable) NSError *failureError;
/*! The plugin process's identifier, or 0 when none runs or the plugin runs in the app's process. */
@property (nonatomic, readonly) pid_t processIdentifier;
/*! When the running process started, or nil when none runs. */
@property (nonatomic, readonly, copy, nullable) NSDate *launchDate;

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

@end

/*! Posted by a registry, the object, when a plugin's status changes. userInfo holds the new status under
	NFKPluginRegistryStatusKey. Delivered on a background queue. Introduced in InferKit 0.4.0. */
extern NSNotificationName const NFKPluginRegistryStatusDidChangeNotification API_UNAVAILABLE(ios, tvos, watchos);

/*! The NFKPluginStatus in an NFKPluginRegistryStatusDidChangeNotification's userInfo. Introduced in
	InferKit 0.4.0. */
extern NSString * const NFKPluginRegistryStatusKey API_UNAVAILABLE(ios, tvos, watchos);

/*! A block a registry calls with a plugin's new status. Introduced in InferKit 0.4.0. */
typedef void (^NFKPluginStatusHandler)(NFKPluginStatus *status) API_UNAVAILABLE(ios, tvos, watchos);

/*! A warning an app shows before it trusts a plugin to run in its own process: the plugin stays loaded
	until the app quits. Introduced in InferKit 0.4.0. */
extern NSString * const NFKPluginInProcessWarningNoUnload API_UNAVAILABLE(ios, tvos, watchos);

/*! A warning an app shows before it trusts a plugin to run in its own process: a crash in the plugin
	ends the app. Introduced in InferKit 0.4.0. */
extern NSString * const NFKPluginInProcessWarningCrashTakesDownApp API_UNAVAILABLE(ios, tvos, watchos);

/*! A warning an app shows before it trusts a plugin to run in its own process: the plugin has every
	access the app has. Introduced in InferKit 0.4.0. */
extern NSString * const NFKPluginInProcessWarningFullAccess API_UNAVAILABLE(ios, tvos, watchos);

/*!
	@class      NFKPluginRegistry
	@abstract   The plugins an app has installed, listed in the model catalog from their manifests.
	@discussion Adding a plugin reads its manifest and loads no code. Each model joins the catalog as
				"<plugin identifier>/<model identifier>" with engine "plugin", registryName the model
				identifier, and extensions.plugin {id, version, kind}. A plugin cannot replace a model
				already in the catalog under that identifier.

				The registry registers the builder of engine "plugin" with
				+[NFKModelCatalog registerBackendBuilder:forEngine:], so
				-[NFKModelCatalog backendForModelIdentifier:options:error:] builds a plugin's model after
				it checks the options, the machine, and the memory estimate, as it does for any model.
				The load options reach the plugin's build; the request options become request defaults.

				A plugin runs in its own process by default. The process starts on first use, and exits
				after its idle interval once no backend holds it. A plugin the app trusts runs in the
				app's process from its next load; it then stays loaded until the app quits. The app
				shows the NFKPluginInProcessWarning* texts before trusting one, and persists the
				choice itself.

				A server hosts plugin models lazily with
				-[NFKInferenceServer addBackendProvider:forModelName:], or with
				-[NFKInferenceServer addModelWithIdentifier:options:catalog:forModelName:error:]; its idle
				unload releases the backend, and the plugin's process then exits on its own. Every method
				is safe to call from any thread. The class is macOS only. Introduced in InferKit 0.4.0.
*/
API_UNAVAILABLE(ios, tvos, watchos)
@interface NFKPluginRegistry : NSObject

/*! The registry that lists its plugins in +[NFKModelCatalog sharedCatalog]. */
@property (class, nonatomic, readonly) NFKPluginRegistry *sharedRegistry NS_SWIFT_NAME(shared);

/*! A registry that lists its plugins in catalog. */
- (instancetype)initWithCatalog:(NFKModelCatalog *)catalog NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

/*! The catalog the registry lists plugins in. */
@property (nonatomic, readonly, strong) NFKModelCatalog *catalog;

/*! The signatures a plugin must carry to load or launch. Defaults to
	+[NFKPluginSignaturePolicy defaultPolicy]. Setting it applies to every plugin's next launch or load,
	except a plugin given a policy of its own. */
@property (atomic, copy) NFKPluginSignaturePolicy *signaturePolicy;

/*! Gives one plugin a signature policy of its own, from its next launch or load, or nil to return it to
	signaturePolicy. The registry keeps the choice in memory only, through a reload; the app persists it.
	Removing the plugin drops the choice, so a bundle listed later under the identifier starts under
	signaturePolicy. Introduced in InferKit 0.4.0. */
- (void)setSignaturePolicy:(nullable NFKPluginSignaturePolicy *)signaturePolicy
	   forPluginIdentifier:(NSString *)identifier;

/*! The policy a plugin launches or loads under: its own, else signaturePolicy. Introduced in InferKit
	0.4.0. */
- (NFKPluginSignaturePolicy *)signaturePolicyForPluginIdentifier:(NSString *)identifier;

/*! The plugin host executable, inferkit-plugin-host, that serves an "objc" or "c-abi" plugin in its own
	process. Defaults to the first executable found: Contents/Helpers in the main bundle, the main
	bundle's auxiliary executable, one beside the main executable, then one beside the image that
	contains InferKit. nil when none is found. Setting nil restores the search. A process takes the value
	when the registry first makes it. */
@property (atomic, copy, nullable) NSURL *hostExecutableURL;

/*! The handshakeTimeout of each plugin process the registry makes, from the process's next start.
	Defaults to 30 seconds. A value that is not positive sets the default. Introduced in InferKit 0.4.0. */
@property (atomic, assign) NSTimeInterval handshakeTimeout;

/*! The verificationTimeout of each plugin process the registry makes, from the process's next start.
	Defaults to 120 seconds. A value that is not positive sets the default. Introduced in InferKit 0.4.0. */
@property (atomic, assign) NSTimeInterval verificationTimeout;

/*! The buildTimeout of each plugin process the registry makes, from the next build or preparation.
	Defaults to 300 seconds. A value that is not positive waits without a limit, as the process's does.
	Introduced in InferKit 0.4.0. */
@property (atomic, assign) NSTimeInterval buildTimeout;

/*! The app group prefix of a sandboxed app, such as "TEAMID.group.example/", or nil for an unsandboxed
	app, the default. Every plugin process takes it as its sharedMemoryPrefix, from its next start. */
@property (atomic, copy, nullable) NSString *sharedMemoryPrefix;

/*!
	@method     addPluginAtURL:error:
	@abstract   Reads the manifest of the plugin bundle at url and lists its models in the catalog.
	@discussion Loads no code and checks no signature; the first launch or load checks it. Returns the
				manifest, or nil with an error:
				- the manifest's own errors (NFKPluginErrorMalformedManifest, …ManifestMismatch, …);
				- NFKPluginErrorIncompatible when this process cannot run the plugin;
				- NFKPluginErrorLoadFailed when another account could change the bundle: the folder that
				  holds it, the bundle, and everything inside it must belong to this user or root and be
				  writable by neither group nor others;
				- NFKPluginErrorIdentifierInUse when the catalog lists a model under one of its catalog
				  identifiers, another bundle is listed under its plugin identifier, or another listed
				  plugin declares a tool of the same name. The description names the identifier or the
				  tool and the plugin that holds it.
				Adding a bundle that is already listed reloads it.
*/
- (nullable NFKPluginManifest *)addPluginAtURL:(NSURL *)url error:(NSError **)error NS_SWIFT_NAME(addPlugin(at:));

/*! Adds every .inferkitplugin bundle directly inside directoryURL, in name order. Returns the manifests
	added, and sets errors to one error per bundle that failed, or to the error that the folder could not
	be read. */
- (NSArray<NFKPluginManifest *> *)addPluginsInDirectoryURL:(NSURL *)directoryURL
													errors:(NSArray<NSError *> * _Nullable * _Nullable)errors
	NS_SWIFT_NAME(addPlugins(inDirectory:errors:));

/*! Adds every plugin in directoryURL as addPluginsInDirectoryURL:errors: does, and gives each one it adds
	signaturePolicy as its own, as setSignaturePolicy:forPluginIdentifier: does: a folder of local builds
	takes +[NFKPluginSignaturePolicy developmentPolicy] while the installed plugins keep the registry's.
	Introduced in InferKit 0.4.0. */
- (NSArray<NFKPluginManifest *> *)addPluginsInDirectoryURL:(NSURL *)directoryURL
										   signaturePolicy:(NFKPluginSignaturePolicy *)signaturePolicy
													errors:(NSArray<NSError *> * _Nullable * _Nullable)errors
	NS_SWIFT_NAME(addPlugins(inDirectory:signaturePolicy:errors:));

/*! Removes a plugin and its models from the catalog. Its process stops once no backend or tool holds it.
	Code loaded into the app's process stays until the app quits. */
- (void)removePluginWithIdentifier:(NSString *)identifier;

/*!
	@method     reloadPluginWithIdentifier:error:
	@abstract   Reads the plugin's manifest from its bundle again.
	@discussion The catalog's descriptors follow the new manifest. A running process restarts with it:
				new requests wait in order, runs in flight finish, and the new version serves the waiting
				requests. A process that is not running starts the new version on its next request.
				Returns NO with NFKPluginErrorUnknownPlugin for a plugin the registry does not list, with
				any error addPluginAtURL:error: reports, and with NFKPluginErrorLoadFailed when the
				plugin's code runs in the app's process at another version, which stays until the app
				quits. A failed reload changes nothing. The catalog is updated when this returns, and the
				restart is still under way; reloadPluginWithIdentifier:completionHandler: reports its end.
*/
- (BOOL)reloadPluginWithIdentifier:(NSString *)identifier error:(NSError **)error;

/*! Reloads the plugin as reloadPluginWithIdentifier:error: does, and calls completionHandler on a
	background queue with nil once the restarted process is ready, or at once when no process runs, or
	with the error that failed the reload or the restart. Introduced in InferKit 0.4.0. */
- (void)reloadPluginWithIdentifier:(NSString *)identifier
				 completionHandler:(void (^ NS_SWIFT_SENDABLE)(NSError * _Nullable error))completionHandler;

/*! The manifests of every plugin added, sorted by plugin identifier. */
@property (nonatomic, readonly, copy) NSArray<NFKPluginManifest *> *manifests;

/*! The manifest of the plugin that serves a catalog identifier, or nil. */
- (nullable NFKPluginManifest *)manifestForModelIdentifier:(NSString *)catalogIdentifier;

/*! Trusts a plugin to run in the app's process, or withdraws that trust, from the plugin's next load.
	Backends made before keep running where they run. The registry keeps the choice in memory only. */
- (void)setTrusted:(BOOL)trusted forPluginIdentifier:(NSString *)identifier;

/*! Whether the app trusts a plugin to run in its process. */
- (BOOL)isTrustedPluginIdentifier:(NSString *)identifier;

/*!
	@method     setOptionValues:forPluginIdentifier:error:
	@abstract   Sets the values of a plugin's own options, the manifest's top-level options.
	@discussion The values resolve against the manifest's options as model options do: a string converts
				to an option's kind, and a choice's text to its value. A value that does not resolve fails
				with NFKModelCatalogErrorInvalidOptions in NFKModelCatalogErrorDomain, its problems under
				NFKModelOptionProblemsKey, and changes nothing; a plugin the registry does not list fails
				with NFKPluginErrorUnknownPlugin. nil clears the values. The plugin reads them from
				-[NFKPluginContext options]: its process receives them at its next start, and a running
				process restarts to receive them. A plugin already loaded into the app's process keeps the
				values it was loaded with until the app quits. The registry keeps the values in memory
				only, through a reload or a removal; the app persists them. When the plugin is added again
				or reloaded, each value is checked against the new manifest: a key it no longer declares,
				and a value that no longer resolves, are dropped. Introduced in InferKit 0.4.0.
*/
- (BOOL)setOptionValues:(nullable NSDictionary<NSString *, id> *)values
	forPluginIdentifier:(NSString *)identifier
				  error:(NSError **)error;

/*! The values setOptionValues:forPluginIdentifier:error: set, converted to each option's kind, or an
	empty dictionary. Introduced in InferKit 0.4.0. */
- (NSDictionary<NSString *, id> *)optionValuesForPluginIdentifier:(NSString *)identifier;

/*!
	@method     backendForModelIdentifier:options:error:
	@abstract   A backend for a catalog identifier "<plugin identifier>/<model identifier>".
	@discussion The options reach the plugin's build as given; the catalog's build entry checks them
				first. An untrusted plugin's backend runs in the plugin's process and starts nothing
				until it prepares or runs. A trusted plugin's code loads into the app's process now,
				after its signature passes signaturePolicy. Returns nil with an error:
				- NFKPluginErrorUnknownModel for an identifier no listed plugin serves;
				- NFKPluginErrorNotTrusted for a trusted plugin whose manifest does not allow it to run in
				  the app's process, which includes every "executable" plugin;
				- the loader's error for a trusted plugin that does not load.
*/
- (nullable NFKPluginBackend *)backendForModelIdentifier:(NSString *)catalogIdentifier
												 options:(nullable NSDictionary<NSString *, id> *)options
												   error:(NSError **)error
	NS_SWIFT_NAME(backend(forModelIdentifier:options:));

#pragma mark Running

/*! The status of a listed plugin, or nil for one the registry does not list. Introduced in InferKit
	0.4.0. */
- (nullable NFKPluginStatus *)statusForPluginIdentifier:(NSString *)identifier;

/*! Called on a background queue with a plugin's new status whenever it changes, whatever started or
	stopped it: a backend's run, a tool call, startPlugin…, stopPlugin…, an idle exit, a crash, a
	reload, or a refused signature at a start. The registry also posts
	NFKPluginRegistryStatusDidChangeNotification. A plugin process's own stateHandler stays the app's.
	Introduced in InferKit 0.4.0. */
@property (atomic, copy, nullable) NFKPluginStatusHandler statusHandler;

/*!
	@method     startPluginWithIdentifier:error:
	@abstract   Starts a plugin ahead of its first request.
	@discussion Blocks until the plugin's process is ready and runs its signature checks then, so a
				refused signature shows at the start. A plugin the app trusts loads into the app's process. Returns NO with NFKPluginErrorUnknownPlugin for a
				plugin the registry does not list, or with the error that failed the start, which the
				status then reports. A running plugin returns YES at once. The process exits after its
				idle interval as one a request started does. Introduced in InferKit 0.4.0.
*/
- (BOOL)startPluginWithIdentifier:(NSString *)identifier error:(NSError **)error;

/*! Starts a plugin as startPluginWithIdentifier:error: does without blocking, and calls completionHandler
	on a background queue with nil once it is ready, or with the error that failed it. Introduced in
	InferKit 0.4.0. */
- (void)startPluginWithIdentifier:(NSString *)identifier
				completionHandler:(void (^ NS_SWIFT_SENDABLE)(NSError * _Nullable error))completionHandler;

/*!
	@method     stopPluginWithIdentifier:completionHandler:
	@abstract   Stops a plugin's process now.
	@discussion The stop retires the process as a reload does: requests made from the call on wait, the
				runs in flight get the process's restartDrainTimeout to finish and are then cancelled,
				and the process ends. The waiting requests then start the plugin again, and each backend
				rebuilds its plugin backend on its next run. completionHandler runs on a background queue
				once the process has ended, at once when none runs, with NFKPluginErrorUnknownPlugin for a
				plugin the registry does not list, or with NFKPluginErrorInProcessUnloadable for a
				plugin loaded into the app's process, which stays until the app quits. Introduced in
				InferKit 0.4.0.
*/
- (void)stopPluginWithIdentifier:(NSString *)identifier
			   completionHandler:(nullable void (^ NS_SWIFT_SENDABLE)(NSError * _Nullable error))completionHandler;

/*! Every plugin's tools, read from the manifests, sorted by plugin identifier. Listing them starts no
	process and loads no code: calling a tool starts the plugin's process, or loads a trusted plugin. A
	trusted plugin that cannot run in the app's process fails each call with NFKPluginErrorNotTrusted, as
	its backends do. */
@property (nonatomic, readonly, copy) NSArray<NFKInferenceTool *> *tools;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKPluginRegistry_h */
