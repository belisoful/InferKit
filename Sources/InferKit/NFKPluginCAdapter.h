//
//  NFKPluginCAdapter.h
//  InferKit
//
//  A C plugin's function table as an NFKPlugin, and its backends as NFKInferenceBackends. The adapter
//  implements the host function table the plugin calls back through. Docs/agent-reference/plugins.md
//  describes the ownership rules it keeps.
//

#import <Foundation/Foundation.h>
#import <InferKit/NFKPlugin.h>
#import <InferKit/NFKPluginManifest.h>

#if __has_include(<InferKitC/NFKPluginABI.h>)
#import <InferKitC/NFKPluginABI.h>
#elif __has_include(<InferKit/NFKPluginABI.h>)
#import <InferKit/NFKPluginABI.h>
#else
#import "../InferKitC/include/InferKitC/NFKPluginABI.h"
#endif

NS_ASSUME_NONNULL_BEGIN

#if TARGET_OS_OSX

@class NFKPluginHostContext;

/*!
	@protocol   NFKPluginOutputDirectoryBackend
	@abstract   A backend that writes a run's result files into a directory its caller names.
	@discussion The plugin service passes the host's per-run output directory through it. A backend
				without it writes its files where it chooses, and the service copies them.
*/
@protocol NFKPluginOutputDirectoryBackend <NSObject>

/*! Starts a run whose result files go into outputDirectoryURL. */
- (NFKInferenceJob *)submitInferenceJobForRequest:(NFKInferenceRequest *)request
							   outputDirectoryURL:(NSURL *)outputDirectoryURL;

@end

/*!
	@class      NFKPluginCAdapter
	@abstract   A C plugin's function table, as the NFKPlugin a service or a loaded plugin serves.
	@discussion The adapter calls the entry with its host table, checks the plugin's table, and reads
				describe once. It owns the host table, which stays at one address for the adapter's life,
				and it never closes the library. A backend it makes is an NFKPluginCBackend. Once shutdown
				returns, nothing calls the plugin again.
*/
@interface NFKPluginCAdapter : NSObject <NFKPlugin>

/*! Calls entry and returns the adapter over the table it gives, or nil with NFKPluginErrorIncompatible
	or NFKPluginErrorLoadFailed. */
- (nullable instancetype)initWithEntry:(nfk_plugin_entry_function)entry
							  manifest:(NFKPluginManifest *)manifest
							   context:(NFKPluginHostContext *)context
								 error:(NSError **)error;

- (instancetype)init NS_UNAVAILABLE;

/*! The manifest of the plugin. */
@property (nonatomic, readonly, copy) NFKPluginManifest *manifest;

/*! The context the host table reports from. */
@property (nonatomic, readonly, strong) NFKPluginHostContext *context;

/*! Whether shutdown has run. */
@property (atomic, readonly, getter=isShutDown) BOOL shutDown;

/*! Calls the plugin's shutdown once. Backends still alive stop calling the plugin. */
- (void)shutdown;

@end

/*!
	@class      NFKPluginCBackend
	@abstract   One backend a C plugin made.
	@discussion Runs are limited to the manifest's maximumConcurrentRuns. A request crosses in the
				wire coding's attachment form: each large value is a bytes buffer the adapter owns until
				run returns, and each file passes by its path. A result's buffers are wrapped without a
				copy, and the plugin's payload is freed when the last value over it is released. A partial
				result is copied during report_progress. Releasing the backend calls destroy_backend.
*/
@interface NFKPluginCBackend : NSObject <NFKInferenceBackend, NFKPluginOutputDirectoryBackend>

- (instancetype)init NS_UNAVAILABLE;

/*! The model the backend runs. */
@property (nonatomic, readonly, copy) NSString *modelIdentifier;

@end

#endif

NS_ASSUME_NONNULL_END
