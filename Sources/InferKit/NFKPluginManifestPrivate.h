//
//  NFKPluginManifestPrivate.h
//  InferKit
//
//  What the loaders and the plugin host read from a manifest beyond its public properties.
//

#import <InferKit/NFKPluginManifest.h>

NS_ASSUME_NONNULL_BEGIN

/*! The manifest's file inside a plugin bundle, relative to its Contents directory. */
extern NSString * const NFKPluginManifestRelativePath;

@interface NFKPluginManifest ()

/*! The file the plugin's code is entered through: Contents/MacOS/ joined with executablePath, or with the
	bundle's CFBundleExecutable when executablePath is nil. nil for a manifest read from JSON alone. */
@property (nonatomic, readonly, copy, nullable) NSURL *executableURL;

/*! The bundle's CFBundleExecutable, or nil. */
@property (nonatomic, readonly, copy, nullable) NSString *bundleExecutableName;

/*! The model descriptor with the identifier, or nil. */
- (nullable NFKModelDescriptor *)modelWithIdentifier:(NSString *)identifier;

#if TARGET_OS_OSX
/*! Whether this process runs the plugin: the InferKit API version, the plugin protocol's major version,
	the plugin ABI's major version for a "c-abi" plugin, the macOS version, and the architecture. Returns
	NO with NFKPluginErrorIncompatible otherwise. */
- (BOOL)checkHostCompatibilityWithError:(NSError **)error;
#endif

@end

NS_ASSUME_NONNULL_END
