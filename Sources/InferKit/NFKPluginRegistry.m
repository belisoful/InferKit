//
//  NFKPluginRegistry.m
//  InferKit
//

#import "NFKPluginRegistry.h"
#import "NFKPluginBackend.h"
#import "NFKPluginWire.h"

#if TARGET_OS_OSX

NSString * const NFKPluginInProcessWarningNoUnload =
	@"A plugin that runs in the app's process stays loaded until the app quits.";
NSString * const NFKPluginInProcessWarningCrashTakesDownApp =
	@"A crash in a plugin that runs in the app's process ends the app.";
NSString * const NFKPluginInProcessWarningFullAccess =
	@"A plugin that runs in the app's process has every access the app has.";

@implementation NFKPluginRegistry

+ (NFKPluginRegistry *)sharedRegistry
{
	static NFKPluginRegistry *shared = nil;
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		shared = [[NFKPluginRegistry alloc] initWithCatalog:NFKModelCatalog.sharedCatalog];
	});
	return shared;
}

- (instancetype)initWithCatalog:(NFKModelCatalog *)catalog
{
	self = [super init];
	if (self != nil) {
		_catalog = catalog;
		_signaturePolicy = [NFKPluginSignaturePolicy defaultPolicy];
	}
	return self;
}

- (nullable NFKPluginManifest *)addPluginAtURL:(NSURL *)url error:(NSError **)error
{
	NFKPluginSetNotImplemented(error, _cmd);
	return nil;
}

- (NSArray<NFKPluginManifest *> *)addPluginsInDirectoryURL:(NSURL *)directoryURL errors:(NSArray<NSError *> **)errors
{
	if (errors != NULL) {
		NSError *error = nil;
		NFKPluginSetNotImplemented(&error, _cmd);
		*errors = @[ error ];
	}
	return @[];
}

- (void)removePluginWithIdentifier:(NSString *)identifier
{
}

- (BOOL)reloadPluginWithIdentifier:(NSString *)identifier error:(NSError **)error
{
	return NFKPluginSetNotImplemented(error, _cmd);
}

- (NSArray<NFKPluginManifest *> *)manifests
{
	return @[];
}

- (nullable NFKPluginManifest *)manifestForModelIdentifier:(NSString *)catalogIdentifier
{
	return nil;
}

- (void)setTrusted:(BOOL)trusted forPluginIdentifier:(NSString *)identifier
{
}

- (BOOL)isTrustedPluginIdentifier:(NSString *)identifier
{
	return NO;
}

- (nullable NFKPluginBackend *)backendForModelIdentifier:(NSString *)catalogIdentifier
												 options:(NSDictionary<NSString *, id> *)options
												   error:(NSError **)error
{
	NFKPluginSetNotImplemented(error, _cmd);
	return nil;
}

- (NSArray<NFKInferenceTool *> *)tools
{
	return @[];
}

@end

#endif
