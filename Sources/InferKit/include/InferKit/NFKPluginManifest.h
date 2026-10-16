//
//  NFKPluginManifest.h
//  InferKit
//

#ifndef NFKPluginManifest_h
#define NFKPluginManifest_h

#import <Foundation/Foundation.h>
#import <InferKit/NFKModelDescriptor.h>

NS_ASSUME_NONNULL_BEGIN

/*! How a plugin's code is entered. Introduced in InferKit 0.4.0. */
typedef NSString * NFKPluginEntryKind NS_TYPED_ENUM;

/*! An Objective-C or Swift bundle whose principal class adopts NFKPlugin: "objc". Introduced in
	InferKit 0.4.0. */
extern NFKPluginEntryKind const NFKPluginEntryKindObjC;

/*! A library that exports nfk_plugin_entry from NFKPluginABI.h: "c-abi". Introduced in InferKit 0.4.0. */
extern NFKPluginEntryKind const NFKPluginEntryKindCABI;

/*! A program that speaks the plugin protocol on the descriptor it inherits: "executable". Introduced in
	InferKit 0.4.0. */
extern NFKPluginEntryKind const NFKPluginEntryKindExecutable;

/*!
	@class      NFKPluginManifest
	@abstract   What a plugin is and offers, read without loading its code.
	@discussion A plugin is a .inferkitplugin bundle whose Contents/Resources/inferkit-plugin.json holds
				the manifest. The bundle's signature seals the file. The manifest names the plugin, its
				version and compatibility, how its code is entered, the models it runs in the model
				catalog's descriptor schema, its tools, and its options. A host lists a plugin from its
				manifest alone and starts the plugin on first use.

				The JSON form:

				{"schemaVersion": 1, "kind": "inferkit.plugin", "id", "version", "displayName",
				"summary", "compatibility": {"inferkitAPI", "pluginABI", "protocol", "minimumOS",
				"architectures"}, "entry": {"kind", "executable", "symbol", "principalClass",
				"arguments"}, "process": {"idleExitSeconds", "allowsInProcess",
				"maximumConcurrentRuns"}, "options": […], "models": […], "tools": [{"name",
				"description", "parameters"}], "extensions": {…}}

				Unknown fields are ignored. The class is available on every platform, so an app lists
				plugins wherever it runs; loading and launching them is macOS only. Introduced in
				InferKit 0.4.0.
*/
@interface NFKPluginManifest : NSObject <NSCopying>

/*! The manifest schema version this library reads and writes. */
@property (class, nonatomic, readonly) NSInteger schemaVersion;

/*! Reads Contents/Resources/inferkit-plugin.json from a bundle and checks it against the bundle's
	Info.plist: the identifier and the short version must match, and the executable, entry.executable or
	else CFBundleExecutable, must be a file whose path with symbolic links resolved lies in the bundle's
	Contents/MacOS. A mismatch is NFKPluginErrorManifestMismatch. Loads no code. */
+ (nullable instancetype)manifestWithBundleURL:(NSURL *)bundleURL error:(NSError **)error;

/*! Reads a manifest from its JSON form. bundleURL is nil. */
+ (nullable instancetype)manifestWithJSONData:(NSData *)data error:(NSError **)error;

/*! The manifest in its JSON form, keys sorted. */
- (nullable NSData *)JSONDataWithError:(NSError **)error;

/*! The plugin's identifier, a reverse-DNS name such as "com.example.denoise". */
@property (nonatomic, readonly, copy) NSString *identifier;

/*! The plugin's version, a semantic-version string. */
@property (nonatomic, readonly, copy) NSString *version;

/*! The name an app shows. */
@property (nonatomic, readonly, copy) NSString *displayName;

/*! One sentence on what the plugin does, or nil. */
@property (nonatomic, readonly, copy, nullable) NSString *summary;

/*! How the plugin's code is entered. */
@property (nonatomic, readonly, copy) NFKPluginEntryKind entryKind;

/*! The executable or library, relative to the bundle's Contents/MacOS, or nil for the bundle's own
	executable. */
@property (nonatomic, readonly, copy, nullable) NSString *executablePath;

/*! The principal class of an "objc" plugin, or nil for the bundle's NSPrincipalClass. */
@property (nonatomic, readonly, copy, nullable) NSString *principalClassName;

/*! The entry symbol of a "c-abi" plugin, or nil for nfk_plugin_entry. */
@property (nonatomic, readonly, copy, nullable) NSString *entrySymbol;

/*! The arguments an "executable" plugin is launched with. */
@property (nonatomic, readonly, copy) NSArray<NSString *> *arguments;

/*! The lowest InferKit API version the plugin runs on, as "major.minor". */
@property (nonatomic, readonly, copy) NSString *minimumAPIVersion;

/*! The plugin ABI version a "c-abi" plugin was built against, as "major.minor", or nil. */
@property (nonatomic, readonly, copy, nullable) NSString *pluginABIVersion;

/*! The plugin protocol version the plugin speaks, as "major.minor". */
@property (nonatomic, readonly, copy) NSString *protocolVersion;

/*! The lowest system version per platform, keyed by "macOS". */
@property (nonatomic, readonly, copy) NSDictionary<NSString *, NSString *> *minimumOSVersions;

/*! The architectures the plugin's code carries, such as "arm64" and "x86_64". */
@property (nonatomic, readonly, copy) NSArray<NSString *> *architectures;

/*! The models the plugin runs, in the model catalog's schema, with engine "plugin". */
@property (nonatomic, readonly, copy) NSArray<NFKModelDescriptor *> *models;

/*! The tools the plugin offers, each a declaration {name, description, parameters}. */
@property (nonatomic, readonly, copy) NSArray<NSDictionary<NSString *, id> *> *tools;

/*! The options the plugin reads, beyond each model's parameters. */
@property (nonatomic, readonly, copy) NSArray<NFKModelOption *> *options;

/*! How long an idle plugin process with no live backends runs before it exits, in seconds. 300 when
	the manifest names none. */
@property (nonatomic, readonly) NSTimeInterval idleExitInterval;

/*! Whether the plugin supports running in the app's own process once the app trusts it. */
@property (nonatomic, readonly) BOOL allowsInProcess;

/*! How many runs of one backend may be in flight at once. 1 serializes them. */
@property (nonatomic, readonly) NSInteger maximumConcurrentRuns;

/*! Fields outside the schema, kept for a reader that knows them. */
@property (nonatomic, readonly, copy) NSDictionary<NSString *, id> *extensions;

/*! The bundle the manifest was read from, or nil. */
@property (nonatomic, readonly, copy, nullable) NSURL *bundleURL;

/*! Whether a library of APIVersion ("major.minor") runs the plugin: the same major version and a minor
	version at least the manifest's minimumAPIVersion. Returns NO with NFKPluginErrorIncompatible
	otherwise. */
- (BOOL)checkCompatibilityWithAPIVersion:(NSString *)APIVersion error:(NSError **)error;

- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END

#endif /* NFKPluginManifest_h */
