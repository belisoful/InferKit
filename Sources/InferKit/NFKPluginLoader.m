//
//  NFKPluginLoader.m
//  InferKit
//

#import "NFKPluginLoader.h"
#import "NFKPluginWire.h"

#if TARGET_OS_OSX

@implementation NFKLoadedPlugin

- (nullable id<NFKInferenceBackend>)backendForModelIdentifier:(NSString *)modelIdentifier
													  options:(NSDictionary<NSString *, id> *)options
														error:(NSError **)error
{
	NFKPluginSetNotImplemented(error, _cmd);
	return nil;
}

@end

@implementation NFKPluginLoader

+ (nullable NFKLoadedPlugin *)loadPluginAtURL:(NSURL *)url
							  signaturePolicy:(NFKPluginSignaturePolicy *)policy
										error:(NSError **)error
{
	NFKPluginSetNotImplemented(error, _cmd);
	return nil;
}

@end

#endif
