//
//  NFKClassification.m
//  InferKit
//

#import "NFKClassification.h"

@implementation NFKClassification

+ (instancetype)classificationWithLabel:(nullable NSString *)label
							 classIndex:(NSInteger)classIndex
							 confidence:(double)confidence
{
	return [[self alloc] initWithLabel:label classIndex:classIndex confidence:confidence];
}

- (instancetype)initWithLabel:(nullable NSString *)label
				   classIndex:(NSInteger)classIndex
				   confidence:(double)confidence
{
	self = [super init];
	if (self != nil) {
		_label = [label copy];
		_classIndex = classIndex;
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
	if (![object isKindOfClass:NFKClassification.class]) {
		return NO;
	}
	NFKClassification *other = object;
	return (self.label == other.label || [self.label isEqual:other.label])
		&& self.classIndex == other.classIndex
		&& self.confidence == other.confidence;
}

- (NSUInteger)hash
{
	return self.label.hash ^ (NSUInteger)self.classIndex ^ (NSUInteger)(self.confidence * 1000);
}

- (NSString *)description
{
	return [NSString stringWithFormat:@"<%@ %@[%ld] %.3f>",
			NSStringFromClass(self.class), self.label ?: @"?", (long)self.classIndex, self.confidence];
}

@end
