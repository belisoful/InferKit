//
//  NFKKeypoint.m
//  InferKit
//

#import "NFKKeypoint.h"

@implementation NFKKeypoint

+ (instancetype)keypointWithName:(nullable NSString *)name
						   index:(NSInteger)index
						position:(CGPoint)position
					  confidence:(double)confidence
{
	return [[self alloc] initWithName:name index:index position:position confidence:confidence];
}

- (instancetype)initWithName:(nullable NSString *)name
					  index:(NSInteger)index
				   position:(CGPoint)position
				 confidence:(double)confidence
{
	self = [super init];
	if (self != nil) {
		_name = [name copy];
		_index = index;
		_position = position;
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
	if (![object isKindOfClass:NFKKeypoint.class]) {
		return NO;
	}
	NFKKeypoint *other = object;
	return (self.name == other.name || [self.name isEqual:other.name])
		&& self.index == other.index
		&& CGPointEqualToPoint(self.position, other.position)
		&& self.confidence == other.confidence;
}

- (NSUInteger)hash
{
	return self.name.hash ^ (NSUInteger)self.index ^ (NSUInteger)(self.confidence * 1000);
}

- (NSString *)description
{
	return [NSString stringWithFormat:@"<%@ %@[%ld] (%.3f, %.3f) %.2f>",
			NSStringFromClass(self.class), self.name ?: @"?", (long)self.index,
			self.position.x, self.position.y, self.confidence];
}

@end
