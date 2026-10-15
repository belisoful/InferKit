//
//  NFKPluginLoader.h
//  InferKit
//

#ifndef NFKPluginLoader_h
#define NFKPluginLoader_h

#import <Foundation/Foundation.h>
#import <InferKit/NFKInferenceBackend.h>
#import <InferKit/NFKInferenceTool.h>
#import <InferKit/NFKModelDescriptor.h>
#import <InferKit/NFKPluginManifest.h>
#import <InferKit/NFKPluginSignaturePolicy.h>

NS_ASSUME_NONNULL_BEGIN

/*!
	@class      NFKLoadedPlugin
	@abstract   A plugin whose code runs in this process.
	@discussion NFKPluginLoader makes it. It stays loaded for the life of the process, since unloading
				Objective-C or C code a process has called is not safe. Introduced in InferKit 0.4.0.
*/
API_UNAVAILABLE(ios, tvos, watchos)
@interface NFKLoadedPlugin : NSObject

/*! The manifest the plugin was loaded from. */
@property (nonatomic, readonly, copy) NFKPluginManifest *manifest;

/*! The models the plugin's code describes. */
@property (nonatomic, readonly, copy) NSArray<NFKModelDescriptor *> *describedModels;

/*! The tools the plugin's code offers. */
@property (nonatomic, readonly, copy) NSArray<NFKInferenceTool *> *tools;

/*! A backend for one of the plugin's models, or nil with an error. */
- (nullable id<NFKInferenceBackend>)backendForModelIdentifier:(NSString *)modelIdentifier
													  options:(NSDictionary<NSString *, id> *)options
														error:(NSError **)error;

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

@end

/*!
	@class      NFKPluginLoader
	@abstract   Loads a plugin's code into this process.
	@discussion Any process calls it: the plugin host loads the plugin it serves, and an app loads a
				plugin it trusts to run in its own process.
				- "objc" → the bundle loads through NSBundle and its principal class, which adopts
				  NFKPlugin, is created with a context. A bundle that defines any class whose name
				  begins with NFK carries its own copy of InferKit and is refused with
				  NFKPluginErrorDuplicateRuntime.
				- "c-abi" → the library opens with dlopen(RTLD_LOCAL), and nfk_plugin_entry is called
				  with the host's function table.
				- "executable" → refused with NFKPluginErrorNotTrusted, since such a plugin runs only as
				  its own process.
				The signature is checked before any code loads. Loading the same bundle again returns
				the same object, and loading a second version of a loaded plugin identifier fails. The
				class is macOS only. Introduced in InferKit 0.4.0.
*/
API_UNAVAILABLE(ios, tvos, watchos)
@interface NFKPluginLoader : NSObject

/*! Verifies the bundle at url under policy, loads its code, and returns the loaded plugin, or nil with
	an error. */
+ (nullable NFKLoadedPlugin *)loadPluginAtURL:(NSURL *)url
							  signaturePolicy:(NFKPluginSignaturePolicy *)policy
										error:(NSError **)error;

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKPluginLoader_h */
