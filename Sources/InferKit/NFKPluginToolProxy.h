//
//  NFKPluginToolProxy.h
//  InferKit
//
//  A tool a plugin declares in its manifest, run in the plugin's process by a callTool request.
//

#import <Foundation/Foundation.h>
#import <InferKit/NFKInferenceTool.h>

NS_ASSUME_NONNULL_BEGIN

#if TARGET_OS_OSX

@class NFKPluginProcess;

/*!
	@class      NFKPluginToolProxy
	@abstract   One of a plugin's tools, with its name, description, and parameters from the manifest.
	@discussion A call sends callTool with the arguments and returns the plugin's toolResult. The call
				starts the plugin's process when none runs, and waits behind a restart in order. An
				error the plugin reports keeps its domain and code.
*/
@interface NFKPluginToolProxy : NFKInferenceTool

/*! A proxy for the manifest's tool declaration {name, description, parameters}, or nil when the
	declaration names no tool. */
- (nullable instancetype)initWithProcess:(NFKPluginProcess *)process declaration:(NSDictionary<NSString *, id> *)declaration;

- (instancetype)initWithName:(NSString *)name
				 description:(NSString *)toolDescription
				  parameters:(NSDictionary<NSString *, id> *)parameters NS_UNAVAILABLE;

/*! The process the tool runs in. */
@property (nonatomic, readonly, strong) NFKPluginProcess *process;

@end

#endif

NS_ASSUME_NONNULL_END
