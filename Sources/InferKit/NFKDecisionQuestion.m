//
//  NFKDecisionQuestion.m
//  InferKit
//

#import "NFKDecisionQuestion.h"

@implementation NFKDecisionQuestion

+ (instancetype)choiceQuestionWithInstructions:(NSString *)instructions
									   options:(NSArray<NSString *> *)options
{
	return [self choiceQuestionWithInstructions:instructions options:options descriptions:nil];
}

+ (instancetype)choiceQuestionWithInstructions:(NSString *)instructions
									   options:(NSArray<NSString *> *)options
								  descriptions:(nullable NSDictionary<NSString *, NSString *> *)descriptions
{
	return [[self alloc] initWithType:NFKDecisionTypeChoice instructions:instructions
							  options:options descriptions:descriptions];
}

+ (instancetype)scoreQuestionWithInstructions:(NSString *)instructions
									   levels:(NSArray<NSString *> *)levels
{
	return [[self alloc] initWithType:NFKDecisionTypeScore instructions:instructions
							  options:levels descriptions:nil];
}

+ (instancetype)noulQuestionWithInstructions:(NSString *)instructions
{
	return [self noulQuestionWithInstructions:instructions trueMeaning:nil falseMeaning:nil];
}

+ (instancetype)noulQuestionWithInstructions:(NSString *)instructions
								trueMeaning:(nullable NSString *)trueMeaning
							   falseMeaning:(nullable NSString *)falseMeaning
{
	NSMutableDictionary<NSString *, NSString *> *meanings = [NSMutableDictionary dictionary];
	if (trueMeaning != nil) {
		meanings[@"true"] = trueMeaning;
	}
	if (falseMeaning != nil) {
		meanings[@"false"] = falseMeaning;
	}
	return [[self alloc] initWithType:NFKDecisionTypeNoul instructions:instructions
							  options:nil descriptions:meanings];
}

- (instancetype)initWithType:(NFKDecisionType)type
				instructions:(NSString *)instructions
					 options:(nullable NSArray<NSString *> *)options
				descriptions:(nullable NSDictionary<NSString *, NSString *> *)descriptions
{
	self = [super init];
	if (self != nil) {
		_type = type;
		_instructions = [instructions copy] ?: @"";
		_options = [options copy] ?: @[];
		_descriptions = [descriptions copy] ?: @{};
	}
	return self;
}

+ (nullable instancetype)questionWithDictionary:(NSDictionary<NSString *, id> *)dictionary
{
	if (![dictionary isKindOfClass:NSDictionary.class]) {
		return nil;
	}
	NSString *name = [dictionary[@"type"] isKindOfClass:NSString.class] ? dictionary[@"type"] : nil;
	NSString *instructions = [dictionary[@"instructions"] isKindOfClass:NSString.class] ? dictionary[@"instructions"] : nil;
	if (instructions == nil) {
		return nil;
	}
	id criteria = dictionary[@"criteria"];
	if ([name isEqualToString:@"choice"]) {
		NSDictionary *map = [criteria isKindOfClass:NSDictionary.class] ? criteria : @{};
		NSArray<NSString *> *options = [map.allKeys sortedArrayUsingSelector:@selector(compare:)];
		NSMutableDictionary<NSString *, NSString *> *descriptions = [NSMutableDictionary dictionary];
		for (NSString *option in options) {
			if ([map[option] isKindOfClass:NSString.class]) {
				descriptions[option] = map[option];
			}
		}
		return [self choiceQuestionWithInstructions:instructions options:options descriptions:descriptions];
	}
	if ([name isEqualToString:@"score"]) {
		NSMutableArray<NSString *> *levels = [NSMutableArray array];
		for (id level in [criteria isKindOfClass:NSArray.class] ? criteria : @[]) {
			[levels addObject:[level isKindOfClass:NSString.class] ? level : [level description]];
		}
		return [self scoreQuestionWithInstructions:instructions levels:levels];
	}
	if ([name isEqualToString:@"noul"]) {
		NSDictionary *meanings = [criteria isKindOfClass:NSDictionary.class] ? criteria : @{};
		return [self noulQuestionWithInstructions:instructions
									  trueMeaning:[meanings[@"true"] isKindOfClass:NSString.class] ? meanings[@"true"] : nil
									 falseMeaning:[meanings[@"false"] isKindOfClass:NSString.class] ? meanings[@"false"] : nil];
	}
	return nil;
}

+ (NSString *)nameForType:(NFKDecisionType)type
{
	switch (type) {
		case NFKDecisionTypeChoice: return @"choice";
		case NFKDecisionTypeScore: return @"score";
		case NFKDecisionTypeNoul: return @"noul";
	}
	return @"choice";
}

- (NSDictionary<NSString *, id> *)dictionaryRepresentation
{
	NSMutableDictionary<NSString *, id> *wire = [NSMutableDictionary dictionary];
	wire[@"type"] = [self.class nameForType:self.type];
	wire[@"instructions"] = self.instructions;
	id criteria = [self criteria];
	if (criteria != nil) {
		wire[@"criteria"] = criteria;
	}
	return wire;
}

- (nullable id)criteria
{
	switch (self.type) {
		case NFKDecisionTypeChoice: {
			NSMutableDictionary<NSString *, id> *criteria = [NSMutableDictionary dictionary];
			for (NSString *option in self.options) {
				criteria[option] = self.descriptions[option] ?: NSNull.null;
			}
			return criteria;
		}
		case NFKDecisionTypeScore:
			return self.options;
		case NFKDecisionTypeNoul:
			return self.descriptions.count > 0 ? self.descriptions : nil;
	}
	return nil;
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
	if (![object isKindOfClass:NFKDecisionQuestion.class]) {
		return NO;
	}
	NFKDecisionQuestion *other = object;
	return self.type == other.type
		&& [self.instructions isEqualToString:other.instructions]
		&& [self.options isEqualToArray:other.options]
		&& [self.descriptions isEqualToDictionary:other.descriptions];
}

- (NSUInteger)hash
{
	return (NSUInteger)self.type ^ self.instructions.hash ^ self.options.hash;
}

- (NSString *)description
{
	return [NSString stringWithFormat:@"<%@ %@ \"%@\" %@>", NSStringFromClass(self.class),
			[self.class nameForType:self.type], self.instructions, self.options];
}

@end
