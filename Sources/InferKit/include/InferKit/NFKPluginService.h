//
//  NFKPluginService.h
//  InferKit
//

#ifndef NFKPluginService_h
#define NFKPluginService_h

#import <Foundation/Foundation.h>
#import <InferKit/NFKPlugin.h>
#import <InferKit/NFKPluginManifest.h>

NS_ASSUME_NONNULL_BEGIN

/*!
	@class      NFKPluginService
	@abstract   Serves one plugin over the plugin protocol, on the plugin's side of the channel.
	@discussion The plugin host runs one for the bundle it loads. A program that hosts a plugin by
				other means runs one too. The service answers hello, describe, build, prepare, run,
				cancel, unload, release, status, callTool, ping, and shutdown, and reports progress,
				partial results, and load events. The class is macOS only. Introduced in InferKit 0.4.0.
*/
API_UNAVAILABLE(ios, tvos, watchos)
@interface NFKPluginService : NSObject

/*! A service for plugin, which manifest describes. */
- (instancetype)initWithPlugin:(id<NFKPlugin>)plugin manifest:(NFKPluginManifest *)manifest NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

/*! The plugin served. */
@property (nonatomic, readonly, strong) id<NFKPlugin> plugin;

/*! The plugin's manifest. */
@property (nonatomic, readonly, copy) NFKPluginManifest *manifest;

/*! Serves the plugin on descriptor, the socket the host passed, until the host sends shutdown or closes
	its end. The first frame hands the host a socket the service creates, and the service speaks on that
	socket from then on. Takes ownership of descriptor. Returns YES on an orderly end, or NO with an
	error. */
- (BOOL)runWithChannelDescriptor:(int)descriptor error:(NSError **)error;

@end

/*!
	@function   NFKPluginHostMain
	@abstract   The whole of the inferkit-plugin-host program.
	@discussion Arguments: --bundle <path> [--team-id <team>] [--requirement <requirement>]
				[--allow-ad-hoc [--allow-unsigned]]. The flags name the app's signature policy;
				--allow-unsigned is for local development and needs --allow-ad-hoc, the development
				policy. The program hands the host its channel on descriptor 3
				(INFERKIT_PLUGIN_CHANNEL_FD), then checks the bundle's ownership and verifies its
				signature in this process, then loads the plugin with NFKPluginLoader when hello arrives
				and serves it with NFKPluginService. Returns the process's exit status. Introduced in
				InferKit 0.4.0.
*/
FOUNDATION_EXPORT int NFKPluginHostMain(int argc, char * _Nullable * _Nonnull argv) API_UNAVAILABLE(ios, tvos, watchos);

NS_ASSUME_NONNULL_END

#endif /* NFKPluginService_h */
