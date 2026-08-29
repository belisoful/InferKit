//
//  NFKInferKit.m
//  InferKit
//

#import "NFKInferKit.h"

@implementation NFKInferKit

// The single in-code source of truth for the core version. On release, bump this together with
// `s.version` in InferKit.podspec and the `vX.Y.Z` git tag.
+ (NSString *)version
{
	return @"0.2.0";
}

@end
