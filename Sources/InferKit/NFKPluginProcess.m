//
//  NFKPluginProcess.m
//  InferKit
//

#import "NFKPluginProcess.h"
#import "NFKPluginBackend.h"
#import "NFKPluginWire.h"

#if TARGET_OS_OSX

@implementation NFKPluginProcess

- (instancetype)initWithManifest:(NFKPluginManifest *)manifest hostExecutableURL:(nullable NSURL *)hostExecutableURL
{
	self = [super init];
	if (self != nil) {
		_manifest = [manifest copy];
		_hostExecutableURL = [hostExecutableURL copy];
		_signaturePolicy = [NFKPluginSignaturePolicy defaultPolicy];
		_handshakeTimeout = 10;
		_idleExitInterval = manifest.idleExitInterval;
		_restartDrainTimeout = 120;
		_crashLimit = 3;
		_crashLimitInterval = 60;
	}
	return self;
}

- (BOOL)startWithError:(NSError **)error
{
	return NFKPluginSetNotImplemented(error, _cmd);
}

- (void)stopWithCompletionHandler:(nullable void (^)(void))completionHandler
{
	if (completionHandler != nil) {
		completionHandler();
	}
}

- (void)restartWithManifest:(NFKPluginManifest *)manifest completionHandler:(nullable void (^)(NSError * _Nullable))completionHandler
{
	NSError *error = nil;
	NFKPluginSetNotImplemented(&error, _cmd);
	if (completionHandler != nil) {
		completionHandler(error);
	}
}

- (nullable NFKPluginBackend *)backendForModelIdentifier:(NSString *)modelIdentifier
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
