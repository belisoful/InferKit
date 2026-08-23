//
//  NFKVideoAsset.m
//  InferKit
//

#import "NFKVideoAsset.h"

@implementation NFKVideoAsset

+ (instancetype)videoAssetWithFileURL:(NSURL *)fileURL
{
	return [[self alloc] initWithFileURL:fileURL durationSeconds:0 framesPerSecond:0 dimensions:CGSizeZero];
}

+ (instancetype)videoAssetWithFileURL:(nullable NSURL *)fileURL
					  durationSeconds:(double)durationSeconds
					   framesPerSecond:(double)framesPerSecond
							dimensions:(CGSize)dimensions
{
	return [[self alloc] initWithFileURL:fileURL
						 durationSeconds:durationSeconds
						  framesPerSecond:framesPerSecond
							   dimensions:dimensions];
}

- (instancetype)initWithFileURL:(nullable NSURL *)fileURL
				durationSeconds:(double)durationSeconds
				 framesPerSecond:(double)framesPerSecond
					  dimensions:(CGSize)dimensions
{
	self = [super init];
	if (self != nil) {
		_fileURL = [fileURL copy];
		_durationSeconds = durationSeconds;
		_framesPerSecond = framesPerSecond;
		_dimensions = dimensions;
	}
	return self;
}

- (id)copyWithZone:(NSZone *)zone
{
	// Immutable.
	return self;
}

- (BOOL)isEqual:(id)object
{
	if (self == object) {
		return YES;
	}
	if (![object isKindOfClass:NFKVideoAsset.class]) {
		return NO;
	}
	NFKVideoAsset *other = object;
	return (self.fileURL == other.fileURL || [self.fileURL isEqual:other.fileURL])
		&& self.durationSeconds == other.durationSeconds
		&& self.framesPerSecond == other.framesPerSecond
		&& CGSizeEqualToSize(self.dimensions, other.dimensions);
}

- (NSUInteger)hash
{
	return self.fileURL.hash ^ (NSUInteger)self.durationSeconds;
}

@end
