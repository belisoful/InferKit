//
//  NFKMIDINote.m
//  InferKit
//

#import "NFKMIDINote.h"

@implementation NFKMIDINote

+ (instancetype)noteWithPitch:(NSInteger)pitch
				 startSeconds:(double)startSeconds
				   endSeconds:(double)endSeconds
					 velocity:(NSInteger)velocity
{
	return [[self alloc] initWithPitch:pitch
						  startSeconds:startSeconds
							endSeconds:endSeconds
							  velocity:velocity
							   program:0
							percussion:NO
							 pitchBend:nil];
}

- (instancetype)initWithPitch:(NSInteger)pitch
				 startSeconds:(double)startSeconds
				   endSeconds:(double)endSeconds
					 velocity:(NSInteger)velocity
					  program:(NSInteger)program
				   percussion:(BOOL)percussion
					pitchBend:(nullable NSArray<NSNumber *> *)pitchBend
{
	self = [super init];
	if (self != nil) {
		_pitch = pitch;
		_startSeconds = startSeconds;
		_endSeconds = endSeconds;
		_velocity = velocity;
		_program = program;
		_percussion = percussion;
		_pitchBend = [pitchBend copy];
	}
	return self;
}

- (double)durationSeconds
{
	return self.endSeconds - self.startSeconds;
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
	if (![object isKindOfClass:NFKMIDINote.class]) {
		return NO;
	}
	NFKMIDINote *other = object;
	return self.pitch == other.pitch
		&& self.startSeconds == other.startSeconds
		&& self.endSeconds == other.endSeconds
		&& self.velocity == other.velocity
		&& self.program == other.program
		&& self.isPercussion == other.isPercussion
		&& (self.pitchBend == other.pitchBend || [self.pitchBend isEqual:other.pitchBend]);
}

- (NSUInteger)hash
{
	return (NSUInteger)self.pitch ^ (NSUInteger)(self.startSeconds * 1000) ^ (NSUInteger)(self.endSeconds * 1000);
}

- (NSString *)description
{
	return [NSString stringWithFormat:@"<%@ pitch %ld %.3f–%.3fs velocity %ld%@>",
			NSStringFromClass(self.class), (long)self.pitch, self.startSeconds, self.endSeconds,
			(long)self.velocity, self.pitchBend != nil ? @" bent" : @""];
}

#pragma mark NSSecureCoding

+ (BOOL)supportsSecureCoding
{
	return YES;
}

- (void)encodeWithCoder:(NSCoder *)coder
{
	[coder encodeInteger:self.pitch forKey:@"pitch"];
	[coder encodeDouble:self.startSeconds forKey:@"startSeconds"];
	[coder encodeDouble:self.endSeconds forKey:@"endSeconds"];
	[coder encodeInteger:self.velocity forKey:@"velocity"];
	[coder encodeInteger:self.program forKey:@"program"];
	[coder encodeBool:self.isPercussion forKey:@"percussion"];
	[coder encodeObject:self.pitchBend forKey:@"pitchBend"];
}

- (nullable instancetype)initWithCoder:(NSCoder *)coder
{
	NSSet *bendClasses = [NSSet setWithObjects:NSArray.class, NSNumber.class, nil];
	return [self initWithPitch:[coder decodeIntegerForKey:@"pitch"]
				  startSeconds:[coder decodeDoubleForKey:@"startSeconds"]
					endSeconds:[coder decodeDoubleForKey:@"endSeconds"]
					  velocity:[coder decodeIntegerForKey:@"velocity"]
					   program:[coder decodeIntegerForKey:@"program"]
					percussion:[coder decodeBoolForKey:@"percussion"]
					 pitchBend:[coder decodeObjectOfClasses:bendClasses forKey:@"pitchBend"]];
}

@end
