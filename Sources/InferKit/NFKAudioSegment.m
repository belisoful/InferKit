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
	return [[self alloc] initWithStartSeconds:startSeconds endSeconds:endSeconds label:label confidence:confidence speaker:nil];
}

+ (instancetype)segmentWithStartSeconds:(double)startSeconds
							endSeconds:(double)endSeconds
								 label:(nullable NSString *)label
							confidence:(double)confidence
							   speaker:(nullable NSString *)speaker
{
	return [[self alloc] initWithStartSeconds:startSeconds endSeconds:endSeconds label:label confidence:confidence speaker:speaker];
}

- (instancetype)initWithStartSeconds:(double)startSeconds
						 endSeconds:(double)endSeconds
							  label:(nullable NSString *)label
						 confidence:(double)confidence
{
	return [self initWithStartSeconds:startSeconds endSeconds:endSeconds label:label confidence:confidence speaker:nil];
}

- (instancetype)initWithStartSeconds:(double)startSeconds
						 endSeconds:(double)endSeconds
							  label:(nullable NSString *)label
						 confidence:(double)confidence
							speaker:(nullable NSString *)speaker
{
	self = [super init];
	if (self != nil) {
		_startSeconds = startSeconds;
		_endSeconds = endSeconds;
		_label = [label copy];
		_confidence = confidence;
		_speaker = [speaker copy];
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
		&& self.confidence == other.confidence
		&& (self.speaker == other.speaker || [self.speaker isEqual:other.speaker]);
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

#pragma mark NSSecureCoding

+ (BOOL)supportsSecureCoding
{
	return YES;
}

- (void)encodeWithCoder:(NSCoder *)coder
{
	[coder encodeDouble:self.startSeconds forKey:@"startSeconds"];
	[coder encodeDouble:self.endSeconds forKey:@"endSeconds"];
	[coder encodeObject:self.label forKey:@"label"];
	[coder encodeDouble:self.confidence forKey:@"confidence"];
	[coder encodeObject:self.speaker forKey:@"speaker"];
}

- (nullable instancetype)initWithCoder:(NSCoder *)coder
{
	return [self initWithStartSeconds:[coder decodeDoubleForKey:@"startSeconds"]
						  endSeconds:[coder decodeDoubleForKey:@"endSeconds"]
							   label:[coder decodeObjectOfClass:NSString.class forKey:@"label"]
						  confidence:[coder decodeDoubleForKey:@"confidence"]
							 speaker:[coder decodeObjectOfClass:NSString.class forKey:@"speaker"]];
}

@end
