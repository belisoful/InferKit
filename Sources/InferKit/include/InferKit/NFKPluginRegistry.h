//
//  NFKPluginRegistry.h
//  InferKit
//

#ifndef NFKPluginRegistry_h
#define NFKPluginRegistry_h

#import <Foundation/Foundation.h>
#import <InferKit/NFKInferenceTool.h>
#import <InferKit/NFKModelCatalog.h>
#import <InferKit/NFKPluginManifest.h>
#import <InferKit/NFKPluginSignaturePolicy.h>

NS_ASSUME_NONNULL_BEGIN

@class NFKPluginBackend;

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

				A plugin runs in its own process by default. The process starts on first use, and exits
				after its idle interval once no backend holds it. A plugin the app trusts runs in the
				app's process from its next load; it then stays loaded until the app quits. The app
				shows the NFKPluginInProcessWarning* texts before trusting one, and persists the
				choice itself.

				A server hosts plugin models lazily with
				-[NFKInferenceServer addBackendProvider:forModelName:]; its idle unload releases the
				backend, and the plugin's process then exits on its own. The class is macOS only.
				Introduced in InferKit 0.4.0.
*/
API_UNAVAILABLE(ios, tvos, watchos)
@interface NFKPluginRegistry : NSObject

/*! The registry that lists its plugins in +[NFKModelCatalog sharedCatalog]. */
@property (class, nonatomic, readonly) NFKPluginRegistry *sharedRegistry;

/*! A registry that lists its plugins in catalog. */
- (instancetype)initWithCatalog:(NFKModelCatalog *)catalog NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

/*! The catalog the registry lists plugins in. */
@property (nonatomic, readonly, strong) NFKModelCatalog *catalog;

/*! The signatures a plugin must carry to load or launch. Defaults to
	+[NFKPluginSignaturePolicy defaultPolicy]. */
@property (atomic, copy) NFKPluginSignaturePolicy *signaturePolicy;

/*! The plugin host executable. Defaults to the main bundle's auxiliary executable
	inferkit-plugin-host, then one beside the main executable, then one beside the image that contains
	InferKit. nil when none is found. */
@property (atomic, copy, nullable) NSURL *hostExecutableURL;

/*! Reads the manifest of the plugin bundle at url and lists its models. Returns the manifest, or nil
	with an error. */
- (nullable NFKPluginManifest *)addPluginAtURL:(NSURL *)url error:(NSError **)error;

/*! Adds every .inferkitplugin bundle directly inside directoryURL. Returns the manifests added, and
	sets errors to one error per bundle that failed. */
- (NSArray<NFKPluginManifest *> *)addPluginsInDirectoryURL:(NSURL *)directoryURL
													errors:(NSArray<NSError *> * _Nullable * _Nullable)errors;

/*! Removes a plugin and its models from the catalog, and stops its process once no backend holds it. */
- (void)removePluginWithIdentifier:(NSString *)identifier;

/*! Reads the plugin's manifest from disk again. A new version updates the catalog's descriptors and
	restarts a running process with it. */
- (BOOL)reloadPluginWithIdentifier:(NSString *)identifier error:(NSError **)error;

/*! The manifests of every plugin added. */
@property (nonatomic, readonly, copy) NSArray<NFKPluginManifest *> *manifests;

/*! The manifest of the plugin that serves a catalog identifier, or nil. */
- (nullable NFKPluginManifest *)manifestForModelIdentifier:(NSString *)catalogIdentifier;

/*! Trusts a plugin to run in the app's process, or withdraws that trust, from the plugin's next load. */
- (void)setTrusted:(BOOL)trusted forPluginIdentifier:(NSString *)identifier;

/*! Whether the app trusts a plugin to run in its process. */
- (BOOL)isTrustedPluginIdentifier:(NSString *)identifier;

/*! A backend for a catalog identifier "<plugin identifier>/<model identifier>", in the plugin's process
	or, for a trusted plugin, in the app's. Returns nil with an error for an unknown identifier. */
- (nullable NFKPluginBackend *)backendForModelIdentifier:(NSString *)catalogIdentifier
												 options:(NSDictionary<NSString *, id> *)options
												   error:(NSError **)error;

/*! Every plugin's tools, from their manifests. */
@property (nonatomic, readonly, copy) NSArray<NFKInferenceTool *> *tools;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKPluginRegistry_h */
