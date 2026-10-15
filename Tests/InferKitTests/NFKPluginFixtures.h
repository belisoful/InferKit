//
//  NFKPluginFixtures.h
//  InferKitTests
//
//  The sample plugins in Tests/PluginFixtures, built at test time with xcrun clang, laid out as
//  .inferkitplugin bundles, and signed ad hoc. Docs/agent-reference/plugins.md ("Building plugins")
//  gives the recipe.
//

#import <Foundation/Foundation.h>
#import <InferKit/InferKit.h>
#import <CoreVideo/CoreVideo.h>

NS_ASSUME_NONNULL_BEGIN

#if TARGET_OS_OSX

/*! The error domain of a fixture build or a tool run that failed. The error's description carries the
	tool's output. */
extern NSString * const NFKPluginFixturesErrorDomain;

/*! The model that inverts a 32BGRA image under "image" and a float array under "values", and mirrors a
	detection under "region" into "detections": its box mirrored left to right (x becomes
	1 - x - width) and its confidence replaced by 1 minus itself. */
extern NSString * const NFKPluginFixtureInvertModel;

/*! The model that counts steps, reports progress after each, and honors cancellation. */
extern NSString * const NFKPluginFixtureSlowModel;

/*! The tool that returns its "text" argument after the echo prefix option. */
extern NSString * const NFKPluginFixtureEchoTool;

/*! The plugin option the echo tool reads. */
extern NSString * const NFKPluginFixtureEchoPrefixOption;

/*! The invert model's float array input and output key. */
extern NSString * const NFKPluginFixtureValuesKey;

/*! The invert model's detection input key. */
extern NSString * const NFKPluginFixtureRegionKey;

/*! The output key every fixture result carries, holding the plugin's version. */
extern NSString * const NFKPluginFixtureVersionKey;

/*! The slow model's step count parameter and output key. */
extern NSString * const NFKPluginFixtureStepsKey;

/*! The slow model's step duration parameter, in milliseconds. */
extern NSString * const NFKPluginFixtureStepMillisecondsKey;

/*! The slow model's parameter that ends the plugin's process at that step with status 70. */
extern NSString * const NFKPluginFixtureExitAtStepKey;

/*! The function the C fixture exports for its count of live allocations: int64_t (*)(void). */
extern NSString * const NFKPluginFixtureAllocationCounterSymbol;

@interface NFKPluginFixtures : NSObject

/*! The repository root, found from this file's path. */
@property (class, nonatomic, readonly) NSURL *sourceRootURL;

/*! The plugin identifier of the fixture of kind: NFKPluginEntryKindObjC, NFKPluginEntryKindCABI, or
	NFKPluginEntryKindExecutable. */
+ (NSString *)pluginIdentifierForKind:(NFKPluginEntryKind)kind;

/*! The fixture of kind at version (such as "1.0.0" or "2.0.0"), built on first use and cached in the
	temporary directory under a hash of its sources, flags, and version, or nil with an error. */
+ (nullable NSURL *)bundleURLForKind:(NFKPluginEntryKind)kind version:(NSString *)version error:(NSError **)error;

/*! An Objective-C fixture that defines its own NFKInferenceRequest, as a bundle embedding InferKit would. */
+ (nullable NSURL *)duplicateRuntimeBundleURLWithError:(NSError **)error;

/*! The inferkit-plugin-host executable: NFK_PLUGIN_HOST_PATH when set, else the one beside the test
	bundle. nil when neither is an executable file. */
+ (nullable NSURL *)hostExecutableURL;

/*! A copy of a signed bundle in a new temporary directory, its manifest's summary edited after signing,
	so its signature no longer verifies. */
+ (nullable NSURL *)tamperedCopyOfBundleAtURL:(NSURL *)bundleURL error:(NSError **)error;

/*! A new folder in the temporary directory, mode 0755, holding a copy of each bundle under its own name,
	or nil with an error. The caller removes the folder. */
+ (nullable NSURL *)folderWithCopiesOfBundles:(NSArray<NSURL *> *)bundles error:(NSError **)error;

/*! Rewrites the manifest of the bundle at url through edit. The bundle's signature then no longer
	verifies, so it serves listing, or is signed again. */
+ (BOOL)editManifestOfBundleAtURL:(NSURL *)url edit:(void (^)(NSMutableDictionary *manifest))edit error:(NSError **)error;

/*! A side × side 32BGRA pixel buffer whose pixel at (column, row) is blue column, green 20, red row, and
	alpha 255, each byte the low 8 bits. */
+ (CVPixelBufferRef)newPatternImageOfSide:(size_t)side CF_RETURNS_RETAINED;

/*! A 32BGRA pixel buffer's pixels, row padding left out, or nil for a value that is not a pixel buffer. */
+ (nullable NSData *)pixelsOfImage:(nullable id)image;

/*! Contents/MacOS/<CFBundleExecutable> of a bundle, or nil when its Info.plist names none. */
+ (nullable NSURL *)executableURLOfBundleAtURL:(NSURL *)bundleURL;

/*! Runs a program to its end and returns its standard output and standard error, or nil with an error
	carrying them when it fails to launch, exits with a nonzero status, or runs past five minutes. */
+ (nullable NSString *)runProgram:(NSString *)path arguments:(NSArray<NSString *> *)arguments error:(NSError **)error;

@end

#endif

NS_ASSUME_NONNULL_END
