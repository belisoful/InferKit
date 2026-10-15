//
//  NFKPluginBackend.h
//  InferKit
//

#ifndef NFKPluginBackend_h
#define NFKPluginBackend_h

#import <Foundation/Foundation.h>
#import <InferKit/NFKInferenceBackend.h>
#import <InferKit/NFKModelDescriptor.h>

NS_ASSUME_NONNULL_BEGIN

@class NFKPluginProcess;

/*!
	@enum       NFKPluginIsolation
	@abstract   Where a plugin's model runs.
	@constant   NFKPluginIsolationProcess In the plugin's own process.
	@constant   NFKPluginIsolationInProcess In the app's process, for a plugin the app trusts.
	Introduced in InferKit 0.4.0.
*/
typedef NS_ENUM(NSInteger, NFKPluginIsolation) {
	NFKPluginIsolationProcess = 0,
	NFKPluginIsolationInProcess = 1,
};

/*!
	@class      NFKPluginBackend
	@abstract   One of a plugin's models as an inference backend.
	@discussion Its identifier is "plugin:<plugin identifier>/<model identifier>". Its keys and model
				information come from the manifest's descriptor, so a caller reads them before the
				plugin runs.

				In a plugin process, prepareWithError: starts the process and builds the model there.
				A run relays the request, and a submitted job relays progress and partial results;
				cancelling the job cancels the run in the plugin. unload, isLoaded, and
				residentByteCount follow the plugin's messages and events. Releasing the backend
				releases the plugin's backend. After the process restarts, the backend builds its
				model again before its next run.

				In the app's process, every call goes to the plugin's own backend. The class is macOS
				only. Introduced in InferKit 0.4.0.
*/
API_UNAVAILABLE(ios, tvos, watchos)
@interface NFKPluginBackend : NSObject <NFKInferenceBackend>

/*! The plugin's identifier. */
@property (nonatomic, readonly, copy) NSString *pluginIdentifier;

/*! The model's identifier within the plugin. */
@property (nonatomic, readonly, copy) NSString *modelIdentifier;

/*! The option values the backend was built with. */
@property (nonatomic, readonly, copy) NSDictionary<NSString *, id> *options;

/*! The model's descriptor from the plugin's manifest. */
@property (nonatomic, readonly, copy) NFKModelDescriptor *descriptor;

/*! Where the model runs. */
@property (nonatomic, readonly) NFKPluginIsolation isolation;

/*! The process the model runs in, or nil in the app's process. */
@property (nonatomic, readonly, strong, nullable) NFKPluginProcess *process;

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKPluginBackend_h */
