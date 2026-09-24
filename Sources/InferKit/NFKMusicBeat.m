//
//  NFKMusicBeat.m
//  InferKit
//

#import "NFKMusicBeat.h"

@implementation NFKMusicBeat

+ (instancetype)beatWithTimeSeconds:(double)timeSeconds positionInBar:(NSInteger)positionInBar
{
	return [[self alloc] initWithTimeSeconds:timeSeconds positionInBar:positionInBar];
}

- (instancetype)initWithTimeSeconds:(double)timeSeconds positionInBar:(NSInteger)positionInBar
{
	self = [super init];
	if (self != nil) {
		_timeSeconds = timeSeconds;
		_positionInBar = positionInBar;
	}
	return self;
}

- (BOOL)isDownbeat
{
	return self.positionInBar == 1;
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
	if (![object isKindOfClass:NFKMusicBeat.class]) {
		return NO;
	}
	NFKMusicBeat *other = object;
	return self.timeSeconds == other.timeSeconds && self.positionInBar == other.positionInBar;
}

- (NSUInteger)hash
{
	return (NSUInteger)(self.timeSeconds * 1000) ^ (NSUInteger)self.positionInBar;
}

- (NSString *)description
{
	return [NSString stringWithFormat:@"<%@ %.3fs %ld>", NSStringFromClass(self.class), self.timeSeconds, (long)self.positionInBar];
}

#pragma mark NSSecureCoding

+ (BOOL)supportsSecureCoding
{
	return YES;
}

- (void)encodeWithCoder:(NSCoder *)coder
{
	[coder encodeDouble:self.timeSeconds forKey:@"timeSeconds"];
	[coder encodeInteger:self.positionInBar forKey:@"positionInBar"];
}

- (nullable instancetype)initWithCoder:(NSCoder *)coder
{
	// isDownbeat reads the position, so it needs nothing of its own in the archive.
	return [self initWithTimeSeconds:[coder decodeDoubleForKey:@"timeSeconds"]
					   positionInBar:[coder decodeIntegerForKey:@"positionInBar"]];
}

@end
