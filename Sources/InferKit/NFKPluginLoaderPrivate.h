//
//  NFKPluginLoaderPrivate.h
//  InferKit
//
//  The loader's internals: the context a loaded plugin receives, the plugin object behind a loaded
//  plugin, and the load entry the plugin host calls with the options a hello carries.
//

#import <Foundation/Foundation.h>
#import <InferKit/NFKPlugin.h>
#import <InferKit/NFKPluginLoader.h>

NS_ASSUME_NONNULL_BEGIN

#if TARGET_OS_OSX

/*! Refuses a bundle another account could change between its verification and its load: the directory
	that holds the bundle, the bundle, and everything inside it must each be owned by this user or root
	and writable by neither group nor others. Returns NO with NFKPluginErrorLoadFailed naming the path and
	the reason. */
BOOL NFKPluginCheckBundleOwnership(NSURL *bundleURL, NSError * _Nullable * _Nullable error);

/*! Receives a message a plugin logs. Called on the plugin's thread. */
typedef void (^NFKPluginLogHandler)(NFKPluginLogLevel level, NSString *message);

/*!
	@class      NFKPluginHostContext
	@abstract   The NFKPluginContext a loader gives the plugin it loads.
	@discussion The temporary directory is made when the context is, under the InferKit temporary
				directory. A message goes to logHandler, or to the unified log when there is none.
*/
@interface NFKPluginHostContext : NSObject <NFKPluginContext>

- (instancetype)initWithPluginIdentifier:(NSString *)pluginIdentifier
								 options:(NSDictionary<NSString *, id> *)options
					   runsInHostProcess:(BOOL)runsInHostProcess
							  logHandler:(nullable NFKPluginLogHandler)logHandler NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;

/*! The plugin the context belongs to. */
@property (nonatomic, readonly, copy) NSString *pluginIdentifier;

/*! Removes the temporary directory. The host calls it after the plugin unloads. */
- (void)removeTemporaryDirectory;

@end

@interface NFKLoadedPlugin ()

/*! The plugin object: the principal class's instance, or the C adapter. */
@property (nonatomic, readonly, strong) id<NFKPlugin> plugin;

/*! The context the plugin received. */
@property (nonatomic, readonly, strong) NFKPluginHostContext *context;

@end

@interface NFKPluginLoader ()

/*! Loads the bundle at url as loadPluginAtURL:signaturePolicy:error: does, giving the plugin options and
	logHandler through its context. runsInHostProcess is YES for an app that loads a plugin it trusts and
	NO for the plugin host. A plugin the process already loaded keeps the context it was loaded with. */
+ (nullable NFKLoadedPlugin *)loadPluginAtURL:(NSURL *)url
							  signaturePolicy:(NFKPluginSignaturePolicy *)policy
									  options:(NSDictionary<NSString *, id> *)options
							runsInHostProcess:(BOOL)runsInHostProcess
								   logHandler:(nullable NFKPluginLogHandler)logHandler
										error:(NSError **)error;

@end

#endif

NS_ASSUME_NONNULL_END
