//
//  NFKDecisionAnswer.m
//  InferKit
//

#import "NFKDecisionAnswer.h"

@implementation NFKDecisionAnswer

+ (nullable instancetype)answerWithDictionary:(NSDictionary<NSString *, id> *)dictionary
{
	if (![dictionary isKindOfClass:NSDictionary.class]) {
		return nil;
	}
	NSString *name = [dictionary[@"type"] isKindOfClass:NSString.class] ? dictionary[@"type"] : nil;
	NFKDecisionType type;
	if ([name isEqualToString:@"choice"]) {
		type = NFKDecisionTypeChoice;
	} else if ([name isEqualToString:@"score"]) {
		type = NFKDecisionTypeScore;
	} else if ([name isEqualToString:@"noul"]) {
		type = NFKDecisionTypeNoul;
	} else {
		return nil;
	}
	NSString *choice = [dictionary[@"choice"] isKindOfClass:NSString.class] ? dictionary[@"choice"] : nil;
	NSDictionary *probabilities = [dictionary[@"probabilities"] isKindOfClass:NSDictionary.class] ? dictionary[@"probabilities"] : nil;
	NSDictionary *legend = [dictionary[@"legend"] isKindOfClass:NSDictionary.class] ? dictionary[@"legend"] : nil;
	return [[self alloc] initWithType:type
							   choice:type == NFKDecisionTypeChoice ? choice : nil
								score:type == NFKDecisionTypeScore ? [self numberIn:dictionary forKey:@"score"] : 0
						  probability:type == NFKDecisionTypeNoul ? [self numberIn:dictionary forKey:@"noul"] : 0
						   confidence:[self numberIn:dictionary forKey:@"confidence"]
						probabilities:type == NFKDecisionTypeNoul ? nil : probabilities
							   legend:type == NFKDecisionTypeScore ? legend : nil
								  raw:dictionary];
}

+ (double)numberIn:(NSDictionary *)dictionary forKey:(NSString *)key
{
	id value = dictionary[key];
	return [value isKindOfClass:NSNumber.class] ? [value doubleValue] : 0;
}

- (instancetype)initWithType:(NFKDecisionType)type
					  choice:(nullable NSString *)choice
					   score:(double)score
				 probability:(double)probability
				  confidence:(double)confidence
			   probabilities:(nullable NSDictionary<NSString *, NSNumber *> *)probabilities
					  legend:(nullable NSDictionary<NSString *, NSString *> *)legend
						 raw:(nullable NSDictionary<NSString *, id> *)raw
{
	self = [super init];
	if (self != nil) {
		_type = type;
		_choice = [choice copy];
		_score = score;
		_probability = probability;
		_confidence = confidence;
		_probabilities = [probabilities copy];
		_legend = [legend copy];
		_raw = [raw copy] ?: @{};
	}
	return self;
}

- (id)copyWithZone:(nullable NSZone *)zone
{
	// Immutable.
	return self;
}

- (BOOL)isEqual:(id)object
{
	if (self == object) {
		return YES;
	}
	if (![object isKindOfClass:NFKDecisionAnswer.class]) {
		return NO;
	}
	NFKDecisionAnswer *other = object;
	return self.type == other.type
		&& (self.choice == other.choice || [self.choice isEqualToString:other.choice])
		&& self.score == other.score
		&& self.probability == other.probability
		&& self.confidence == other.confidence
		&& (self.probabilities == other.probabilities || [self.probabilities isEqualToDictionary:other.probabilities])
		&& (self.legend == other.legend || [self.legend isEqualToDictionary:other.legend]);
}

- (NSUInteger)hash
{
	return (NSUInteger)self.type ^ self.choice.hash ^ (NSUInteger)(self.score * 1000)
		^ (NSUInteger)(self.probability * 1000) ^ (NSUInteger)(self.confidence * 1000);
}

- (NSString *)description
{
	switch (self.type) {
		case NFKDecisionTypeChoice:
			return [NSString stringWithFormat:@"<%@ choice %@ %.3f>", NSStringFromClass(self.class), self.choice ?: @"?", self.confidence];
		case NFKDecisionTypeScore:
			return [NSString stringWithFormat:@"<%@ score %.3f %.3f>", NSStringFromClass(self.class), self.score, self.confidence];
		case NFKDecisionTypeNoul:
			return [NSString stringWithFormat:@"<%@ noul %.3f>", NSStringFromClass(self.class), self.probability];
	}
	return [super description];
}

#pragma mark NSSecureCoding

+ (BOOL)supportsSecureCoding
{
	return YES;
}

- (void)encodeWithCoder:(NSCoder *)coder
{
	[coder encodeInteger:self.type forKey:@"type"];
	[coder encodeObject:self.choice forKey:@"choice"];
	[coder encodeDouble:self.score forKey:@"score"];
	[coder encodeDouble:self.probability forKey:@"probability"];
	[coder encodeDouble:self.confidence forKey:@"confidence"];
	[coder encodeObject:self.probabilities forKey:@"probabilities"];
	[coder encodeObject:self.legend forKey:@"legend"];
	[coder encodeObject:self.raw forKey:@"raw"];
}

- (nullable instancetype)initWithCoder:(NSCoder *)coder
{
	NSSet *plist = [NSSet setWithArray:@[ NSDictionary.class, NSArray.class, NSString.class, NSNumber.class, NSNull.class ]];
	NSSet *stringMap = [NSSet setWithArray:@[ NSDictionary.class, NSString.class ]];
	NSSet *numberMap = [NSSet setWithArray:@[ NSDictionary.class, NSString.class, NSNumber.class ]];
	return [self initWithType:[coder decodeIntegerForKey:@"type"]
					   choice:[coder decodeObjectOfClass:NSString.class forKey:@"choice"]
						score:[coder decodeDoubleForKey:@"score"]
				  probability:[coder decodeDoubleForKey:@"probability"]
				   confidence:[coder decodeDoubleForKey:@"confidence"]
				probabilities:[coder decodeObjectOfClasses:numberMap forKey:@"probabilities"]
					   legend:[coder decodeObjectOfClasses:stringMap forKey:@"legend"]
						  raw:[coder decodeObjectOfClasses:plist forKey:@"raw"]];
}

@end
