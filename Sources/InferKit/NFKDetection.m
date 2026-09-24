//
//  NFKDetection.m
//  InferKit
//

#import "NFKDetection.h"

@implementation NFKDetection

+ (instancetype)detectionWithLabel:(nullable NSString *)label
						classIndex:(NSInteger)classIndex
						confidence:(double)confidence
					   boundingBox:(CGRect)boundingBox
{
	return [[self alloc] initWithLabel:label classIndex:classIndex confidence:confidence boundingBox:boundingBox];
}

- (instancetype)initWithLabel:(nullable NSString *)label
				   classIndex:(NSInteger)classIndex
				   confidence:(double)confidence
				  boundingBox:(CGRect)boundingBox
{
	return [self initWithLabel:label
					classIndex:classIndex
					confidence:confidence
				   boundingBox:boundingBox
				 quadrilateral:nil];
}

+ (instancetype)detectionWithLabel:(nullable NSString *)label
						classIndex:(NSInteger)classIndex
						confidence:(double)confidence
					 quadrilateral:(NFKQuadrilateral *)quadrilateral
{
	return [[self alloc] initWithLabel:label
							classIndex:classIndex
							confidence:confidence
						   boundingBox:quadrilateral.boundingBox
						 quadrilateral:quadrilateral];
}

- (instancetype)initWithLabel:(nullable NSString *)label
				   classIndex:(NSInteger)classIndex
				   confidence:(double)confidence
				  boundingBox:(CGRect)boundingBox
				quadrilateral:(nullable NFKQuadrilateral *)quadrilateral
{
	self = [super init];
	if (self != nil) {
		_label = [label copy];
		_classIndex = classIndex;
		_confidence = confidence;
		_boundingBox = boundingBox;
		_quadrilateral = quadrilateral;
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
	if (![object isKindOfClass:NFKDetection.class]) {
		return NO;
	}
	NFKDetection *other = object;
	return (self.label == other.label || [self.label isEqual:other.label])
		&& self.classIndex == other.classIndex
		&& self.confidence == other.confidence
		&& CGRectEqualToRect(self.boundingBox, other.boundingBox)
		&& (self.quadrilateral == other.quadrilateral || [self.quadrilateral isEqual:other.quadrilateral]);
}

- (NSUInteger)hash
{
	return self.label.hash ^ (NSUInteger)self.classIndex ^ (NSUInteger)(self.confidence * 1000);
}

- (NSString *)description
{
	return [NSString stringWithFormat:@"<%@ %@[%ld] %.2f (%.3f, %.3f, %.3f, %.3f)>",
			NSStringFromClass(self.class), self.label ?: @"?", (long)self.classIndex, self.confidence,
			self.boundingBox.origin.x, self.boundingBox.origin.y,
			self.boundingBox.size.width, self.boundingBox.size.height];
}

#pragma mark NSSecureCoding

+ (BOOL)supportsSecureCoding
{
	return YES;
}

- (void)encodeWithCoder:(NSCoder *)coder
{
	[coder encodeObject:self.label forKey:@"label"];
	[coder encodeInteger:self.classIndex forKey:@"classIndex"];
	[coder encodeDouble:self.confidence forKey:@"confidence"];
	// The rect goes out as four numbers, which every platform decodes the same way.
	[coder encodeDouble:self.boundingBox.origin.x forKey:@"x"];
	[coder encodeDouble:self.boundingBox.origin.y forKey:@"y"];
	[coder encodeDouble:self.boundingBox.size.width forKey:@"width"];
	[coder encodeDouble:self.boundingBox.size.height forKey:@"height"];
	[coder encodeObject:self.quadrilateral forKey:@"quadrilateral"];
}

- (nullable instancetype)initWithCoder:(NSCoder *)coder
{
	CGRect boundingBox = CGRectMake([coder decodeDoubleForKey:@"x"],
									[coder decodeDoubleForKey:@"y"],
									[coder decodeDoubleForKey:@"width"],
									[coder decodeDoubleForKey:@"height"]);
	return [self initWithLabel:[coder decodeObjectOfClass:NSString.class forKey:@"label"]
					classIndex:[coder decodeIntegerForKey:@"classIndex"]
					confidence:[coder decodeDoubleForKey:@"confidence"]
				   boundingBox:boundingBox
				 quadrilateral:[coder decodeObjectOfClass:NFKQuadrilateral.class forKey:@"quadrilateral"]];
}

@end
