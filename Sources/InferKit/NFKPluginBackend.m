//
//  NFKPluginBackend.m
//  InferKit
//

#import "NFKPluginBackend.h"
#import "NFKPluginProcess.h"
#import "NFKPluginWire.h"

#if TARGET_OS_OSX

@implementation NFKPluginBackend

- (BOOL)isReady
{
	return YES;
}

- (NSString *)backendIdentifier
{
	return [NSString stringWithFormat:@"plugin:%@/%@", self.pluginIdentifier, self.modelIdentifier];
}

- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request error:(NSError **)error
{
	NFKPluginSetNotImplemented(error, _cmd);
	return nil;
}

@end

#endif
