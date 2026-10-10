//
//  NFKDecisionAnswer.m
//  InferKit
//

#import "NFKDecisionAnswer.h"

@interface NFKDecisionAnswer ()
@property (nonatomic, readwrite, getter=isRefused) BOOL refused;
@end

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
	} else if ([name isEqualToString:@"noul"] || [name isEqualToString:@"predicate"]) {
		type = NFKDecisionTypeNoul;
	} else {
		return nil;
	}
	id choice = dictionary[@"choice"];
	NSDictionary *probabilities = [self probabilitiesIn:dictionary[@"probabilities"]];
	NSDictionary *legend = [dictionary[@"legend"] isKindOfClass:NSDictionary.class] ? dictionary[@"legend"] : [self legendIn:dictionary[@"probabilities"]];
	double probability = dictionary[@"noul"] != nil ? [self numberIn:dictionary forKey:@"noul"] : [self numberIn:dictionary forKey:@"probability"];
	return [[self alloc] initWithType:type
							   choice:type == NFKDecisionTypeChoice && choice != nil ? [self keyForValue:choice] : nil
								score:type == NFKDecisionTypeScore ? [self numberIn:dictionary forKey:@"score"] : 0
						  probability:type == NFKDecisionTypeNoul ? probability : 0
						   confidence:[self numberIn:dictionary forKey:@"confidence"]
						probabilities:type == NFKDecisionTypeNoul ? nil : probabilities
							   legend:type == NFKDecisionTypeScore ? legend : nil
								  raw:dictionary];
}

+ (instancetype)refusalForType:(NFKDecisionType)type raw:(nullable NSDictionary<NSString *, id> *)raw
{
	NFKDecisionAnswer *answer = [[self alloc] initWithType:type choice:nil score:0 probability:0 confidence:0
											 probabilities:nil legend:nil raw:raw];
	answer.refused = YES;
	return answer;
}

/*! A map passes through; an array of {value, probability} entries is keyed by its values. */
+ (nullable NSDictionary<NSString *, NSNumber *> *)probabilitiesIn:(id)wire
{
	if ([wire isKindOfClass:NSDictionary.class]) {
		return wire;
	}
	if (![wire isKindOfClass:NSArray.class]) {
		return nil;
	}
	NSMutableDictionary<NSString *, NSNumber *> *probabilities = [NSMutableDictionary dictionary];
	for (NSDictionary *entry in wire) {
		if ([entry isKindOfClass:NSDictionary.class] && entry[@"value"] != nil && [entry[@"probability"] isKindOfClass:NSNumber.class]) {
			probabilities[[self keyForValue:entry[@"value"]]] = entry[@"probability"];
		}
	}
	return probabilities;
}

/*! The labels of an array of {value, label, probability} score entries, keyed by level index. */
+ (nullable NSDictionary<NSString *, NSString *> *)legendIn:(id)wire
{
	if (![wire isKindOfClass:NSArray.class]) {
		return nil;
	}
	NSMutableDictionary<NSString *, NSString *> *legend = [NSMutableDictionary dictionary];
	for (NSDictionary *entry in wire) {
		if ([entry isKindOfClass:NSDictionary.class] && entry[@"value"] != nil && [entry[@"label"] isKindOfClass:NSString.class]) {
			legend[[self keyForValue:entry[@"value"]]] = entry[@"label"];
		}
	}
	return legend.count > 0 ? legend : nil;
}

/*! An option or level value as a key: a string as is, a JSON boolean as "true" or "false", and a
	number as its decimal text. */
+ (NSString *)keyForValue:(id)value
{
	if ([value isKindOfClass:NSString.class]) {
		return value;
	}
	if ([value isKindOfClass:NSNumber.class] && CFGetTypeID((__bridge CFTypeRef)value) == CFBooleanGetTypeID()) {
		return [value boolValue] ? @"true" : @"false";
	}
	return [value description];
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
		&& self.refused == other.refused
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
	if (self.refused) {
		return [NSString stringWithFormat:@"<%@ %@ refused>", NSStringFromClass(self.class), [NFKDecisionQuestion nameForType:self.type]];
	}
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
	[coder encodeBool:self.refused forKey:@"refused"];
}

- (nullable instancetype)initWithCoder:(NSCoder *)coder
{
	NSSet *plist = [NSSet setWithArray:@[ NSDictionary.class, NSArray.class, NSString.class, NSNumber.class, NSNull.class ]];
	NSSet *stringMap = [NSSet setWithArray:@[ NSDictionary.class, NSString.class ]];
	NSSet *numberMap = [NSSet setWithArray:@[ NSDictionary.class, NSString.class, NSNumber.class ]];
	self = [self initWithType:[coder decodeIntegerForKey:@"type"]
					   choice:[coder decodeObjectOfClass:NSString.class forKey:@"choice"]
						score:[coder decodeDoubleForKey:@"score"]
				  probability:[coder decodeDoubleForKey:@"probability"]
				   confidence:[coder decodeDoubleForKey:@"confidence"]
				probabilities:[coder decodeObjectOfClasses:numberMap forKey:@"probabilities"]
					   legend:[coder decodeObjectOfClasses:stringMap forKey:@"legend"]
						  raw:[coder decodeObjectOfClasses:plist forKey:@"raw"]];
	if (self != nil) {
		self.refused = [coder decodeBoolForKey:@"refused"];
	}
	return self;
}

@end
