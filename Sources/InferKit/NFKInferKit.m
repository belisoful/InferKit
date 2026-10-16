//
//  NFKInferKit.m
//  InferKit
//

#import "NFKInferKit.h"

@implementation NFKInferKit

// The single in-code source of truth for the release version. On release, bump this together with
// `s.version` in InferKit.podspec and the `vX.Y.Z` git tag. Tools/xcframework/inferkit-version.sh
// reads this line, and a release build refuses a version that disagrees with the podspec or the tags.
+ (NSString *)releaseVersion
{
	return @"0.3.1";
}

+ (NSString *)APIVersion
{
	return [NSString stringWithFormat:@"%ld.%ld", (long)self.APIMajorVersion, (long)self.APIMinorVersion];
}

+ (NSInteger)APIMajorVersion
{
	return NFK_API_VERSION_MAJOR;
}

+ (NSInteger)APIMinorVersion
{
	return NFK_API_VERSION_MINOR;
}

@end
