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
	self = [super init];
	if (self != nil) {
		_label = [label copy];
		_classIndex = classIndex;
		_confidence = confidence;
		_boundingBox = boundingBox;
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
		&& CGRectEqualToRect(self.boundingBox, other.boundingBox);
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

@end
