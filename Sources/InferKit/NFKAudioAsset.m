//
//  NFKAudioAsset.m
//  InferKit
//

#import "NFKAudioAsset.h"

@implementation NFKAudioAsset

+ (instancetype)audioAssetWithFileURL:(NSURL *)fileURL
{
	return [[self alloc] initWithFileURL:fileURL durationSeconds:0 sampleRate:0 channelCount:0];
}

+ (instancetype)audioAssetWithFileURL:(nullable NSURL *)fileURL
					  durationSeconds:(double)durationSeconds
						   sampleRate:(double)sampleRate
						  channelCount:(NSInteger)channelCount
{
	return [[self alloc] initWithFileURL:fileURL
						 durationSeconds:durationSeconds
							  sampleRate:sampleRate
							 channelCount:channelCount];
}

- (instancetype)initWithFileURL:(nullable NSURL *)fileURL
				durationSeconds:(double)durationSeconds
					 sampleRate:(double)sampleRate
					channelCount:(NSInteger)channelCount
{
	self = [super init];
	if (self != nil) {
		_fileURL = [fileURL copy];
		_durationSeconds = durationSeconds;
		_sampleRate = sampleRate;
		_channelCount = channelCount;
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
	if (![object isKindOfClass:NFKAudioAsset.class]) {
		return NO;
	}
	NFKAudioAsset *other = object;
	return (self.fileURL == other.fileURL || [self.fileURL isEqual:other.fileURL])
		&& self.durationSeconds == other.durationSeconds
		&& self.sampleRate == other.sampleRate
		&& self.channelCount == other.channelCount;
}

- (NSUInteger)hash
{
	return self.fileURL.hash ^ (NSUInteger)self.durationSeconds ^ (NSUInteger)self.channelCount;
}

@end
