//
//  NFKQuadrilateral.m
//  InferKit
//

#import "NFKQuadrilateral.h"

@implementation NFKQuadrilateral

+ (instancetype)quadrilateralWithTopLeft:(CGPoint)topLeft
								topRight:(CGPoint)topRight
							  bottomLeft:(CGPoint)bottomLeft
							 bottomRight:(CGPoint)bottomRight
{
	return [[self alloc] initWithTopLeft:topLeft topRight:topRight bottomLeft:bottomLeft bottomRight:bottomRight];
}

- (instancetype)initWithTopLeft:(CGPoint)topLeft
					   topRight:(CGPoint)topRight
					 bottomLeft:(CGPoint)bottomLeft
					bottomRight:(CGPoint)bottomRight
{
	self = [super init];
	if (self != nil) {
		_topLeft = topLeft;
		_topRight = topRight;
		_bottomLeft = bottomLeft;
		_bottomRight = bottomRight;
	}
	return self;
}

- (CGRect)boundingBox
{
	CGFloat minX = MIN(MIN(self.topLeft.x, self.topRight.x), MIN(self.bottomLeft.x, self.bottomRight.x));
	CGFloat maxX = MAX(MAX(self.topLeft.x, self.topRight.x), MAX(self.bottomLeft.x, self.bottomRight.x));
	CGFloat minY = MIN(MIN(self.topLeft.y, self.topRight.y), MIN(self.bottomLeft.y, self.bottomRight.y));
	CGFloat maxY = MAX(MAX(self.topLeft.y, self.topRight.y), MAX(self.bottomLeft.y, self.bottomRight.y));
	return CGRectMake(minX, minY, maxX - minX, maxY - minY);
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
	if (![object isKindOfClass:NFKQuadrilateral.class]) {
		return NO;
	}
	NFKQuadrilateral *other = object;
	return CGPointEqualToPoint(self.topLeft, other.topLeft)
		&& CGPointEqualToPoint(self.topRight, other.topRight)
		&& CGPointEqualToPoint(self.bottomLeft, other.bottomLeft)
		&& CGPointEqualToPoint(self.bottomRight, other.bottomRight);
}

- (NSUInteger)hash
{
	return (NSUInteger)(self.topLeft.x * 1000) ^ (NSUInteger)(self.topLeft.y * 1000)
		 ^ (NSUInteger)(self.bottomRight.x * 1000) ^ (NSUInteger)(self.bottomRight.y * 1000);
}

- (NSString *)description
{
	return [NSString stringWithFormat:@"<%@ (%.3f, %.3f) (%.3f, %.3f) (%.3f, %.3f) (%.3f, %.3f)>",
			NSStringFromClass(self.class),
			self.topLeft.x, self.topLeft.y, self.topRight.x, self.topRight.y,
			self.bottomLeft.x, self.bottomLeft.y, self.bottomRight.x, self.bottomRight.y];
}

#pragma mark NSSecureCoding

+ (BOOL)supportsSecureCoding
{
	return YES;
}

- (void)encodeWithCoder:(NSCoder *)coder
{
	// Eight numbers, which every platform decodes the same way.
	[coder encodeDouble:self.topLeft.x forKey:@"topLeftX"];
	[coder encodeDouble:self.topLeft.y forKey:@"topLeftY"];
	[coder encodeDouble:self.topRight.x forKey:@"topRightX"];
	[coder encodeDouble:self.topRight.y forKey:@"topRightY"];
	[coder encodeDouble:self.bottomLeft.x forKey:@"bottomLeftX"];
	[coder encodeDouble:self.bottomLeft.y forKey:@"bottomLeftY"];
	[coder encodeDouble:self.bottomRight.x forKey:@"bottomRightX"];
	[coder encodeDouble:self.bottomRight.y forKey:@"bottomRightY"];
}

- (nullable instancetype)initWithCoder:(NSCoder *)coder
{
	return [self initWithTopLeft:CGPointMake([coder decodeDoubleForKey:@"topLeftX"],
											 [coder decodeDoubleForKey:@"topLeftY"])
						topRight:CGPointMake([coder decodeDoubleForKey:@"topRightX"],
											 [coder decodeDoubleForKey:@"topRightY"])
					  bottomLeft:CGPointMake([coder decodeDoubleForKey:@"bottomLeftX"],
											 [coder decodeDoubleForKey:@"bottomLeftY"])
					 bottomRight:CGPointMake([coder decodeDoubleForKey:@"bottomRightX"],
											 [coder decodeDoubleForKey:@"bottomRightY"])];
}

@end
