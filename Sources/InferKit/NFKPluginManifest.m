//
//  NFKPluginManifest.m
//  InferKit
//

#import "NFKPluginManifest.h"
#import "NFKPluginWire.h"

NFKPluginEntryKind const NFKPluginEntryKindObjC = @"objc";
NFKPluginEntryKind const NFKPluginEntryKindCABI = @"c-abi";
NFKPluginEntryKind const NFKPluginEntryKindExecutable = @"executable";

@implementation NFKPluginManifest

+ (NSInteger)schemaVersion
{
	return 1;
}

+ (nullable instancetype)manifestWithBundleURL:(NSURL *)bundleURL error:(NSError **)error
{
	NFKPluginSetNotImplemented(error, _cmd);
	return nil;
}

+ (nullable instancetype)manifestWithJSONData:(NSData *)data error:(NSError **)error
{
	NFKPluginSetNotImplemented(error, _cmd);
	return nil;
}

- (nullable NSData *)JSONDataWithError:(NSError **)error
{
	NFKPluginSetNotImplemented(error, _cmd);
	return nil;
}

- (BOOL)checkCompatibilityWithAPIVersion:(NSString *)APIVersion error:(NSError **)error
{
	return NFKPluginSetNotImplemented(error, _cmd);
}

- (id)copyWithZone:(nullable NSZone *)zone
{
	return self;
}

@end
