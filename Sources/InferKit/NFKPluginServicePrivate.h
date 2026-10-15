//
//  NFKPluginServicePrivate.h
//  InferKit
//
//  The service mode the plugin host runs: the plugin loads when hello arrives, with the options hello
//  carries, and a load failure answers hello with its error.
//

#import <InferKit/NFKPluginService.h>
#import "NFKPluginLoaderPrivate.h"

NS_ASSUME_NONNULL_BEGIN

#if TARGET_OS_OSX

/*! Loads the plugin a service serves, with the options a hello carries and a handler that sends the
	plugin's log messages over the channel. Returns nil with an error to refuse the hello. */
typedef id<NFKPlugin> _Nullable (^NFKPluginServiceLoader)(NSDictionary<NSString *, id> *options,
														  NFKPluginLogHandler logHandler,
														  NSError * _Nullable * _Nullable error);

@interface NFKPluginService ()

/*! A service whose plugin loader makes when hello arrives. plugin is nil until then. */
- (instancetype)initWithManifest:(NFKPluginManifest *)manifest loader:(NFKPluginServiceLoader)loader NS_DESIGNATED_INITIALIZER;

/*! Runs after the channel frame is sent and before any message is read: the plugin host verifies the
	bundle here, so the host's hello timeout does not cover the verification. The service then sends
	{"type": "event", "event": "verified"}. An error it returns answers hello. */
@property (nonatomic, copy, nullable) NSError * _Nullable (^verifier)(void);

/*! How long serving waits for runs a shutdown cancelled before it returns, in seconds. Defaults to 5. */
@property (atomic, assign) NSTimeInterval drainTimeout;

@end

#endif

NS_ASSUME_NONNULL_END
