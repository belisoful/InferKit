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
	bundle's CFBundleExecutable when executablePath is nil, with every symbolic link resolved. nil for a
	manifest read from JSON alone, and for a file that is missing or resolves outside Contents/MacOS. */
@property (nonatomic, readonly, copy, nullable) NSURL *executableURL;

/*! executableURL checked against the bundle on disk now, or nil with NFKPluginErrorManifestMismatch: a
	name with a "." or ".." component or a leading "/", a missing file, or a file whose resolved path lies
	outside the resolved bundle's Contents/MacOS. A loader calls it immediately before it loads or
	launches the file, and uses the URL it returns. */
- (nullable NSURL *)containedExecutableURLWithError:(NSError **)error;

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
