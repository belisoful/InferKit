//
//  NFKAudioSegment.m
//  InferKit
//

#import "NFKAudioSegment.h"

@implementation NFKAudioSegment

+ (instancetype)segmentWithStartSeconds:(double)startSeconds
							endSeconds:(double)endSeconds
								 label:(nullable NSString *)label
							confidence:(double)confidence
{
	return [[self alloc] initWithStartSeconds:startSeconds endSeconds:endSeconds label:label confidence:confidence];
}

- (instancetype)initWithStartSeconds:(double)startSeconds
						 endSeconds:(double)endSeconds
							  label:(nullable NSString *)label
						 confidence:(double)confidence
{
	self = [super init];
	if (self != nil) {
		_startSeconds = startSeconds;
		_endSeconds = endSeconds;
		_label = [label copy];
		_confidence = confidence;
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
	if (![object isKindOfClass:NFKAudioSegment.class]) {
		return NO;
	}
	NFKAudioSegment *other = object;
	return self.startSeconds == other.startSeconds
		&& self.endSeconds == other.endSeconds
		&& (self.label == other.label || [self.label isEqual:other.label])
		&& self.confidence == other.confidence;
}

- (NSUInteger)hash
{
	return (NSUInteger)(self.startSeconds * 1000) ^ (NSUInteger)(self.endSeconds * 1000) ^ self.label.hash;
}

- (NSString *)description
{
	return [NSString stringWithFormat:@"<%@ %.3f–%.3fs %@ %.2f>",
			NSStringFromClass(self.class), self.startSeconds, self.endSeconds, self.label ?: @"speech", self.confidence];
}

@end
