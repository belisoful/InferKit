//
//  NFKModelOptions.m
//  InferKit
//

#import "NFKModelOptions.h"
#import "NFKModelDescriptorPrivate.h"
#import "NFKModelCatalog.h"
#import "NFKHardwareProfile.h"

NSString * const NFKModelOptionProblemUnknownKey		= @"unknown-key";
NSString * const NFKModelOptionProblemWrongType			= @"wrong-type";
NSString * const NFKModelOptionProblemBelowMinimum		= @"below-minimum";
NSString * const NFKModelOptionProblemAboveMaximum		= @"above-maximum";
NSString * const NFKModelOptionProblemOffStep			= @"off-step";
NSString * const NFKModelOptionProblemNotAChoice		= @"not-a-choice";
NSString * const NFKModelOptionProblemUnavailable		= @"unavailable";
NSString * const NFKModelOptionProblemWrongStage		= @"wrong-stage";
NSString * const NFKModelOptionProblemUnknownModel		= @"unknown-model";
NSString * const NFKModelOptionProblemMissingFile		= @"missing-file";

NSString * const NFKModelOptionProblemsKey = @"NFKModelOptionProblems";

NSString *NFKModelOptionValueText(id _Nullable value)
{
	if (value == nil || value == NSNull.null) {
		return @"nothing";
	}
	if ([value isKindOfClass:NSString.class]) {
		return [NSString stringWithFormat:@"\"%@\"", value];
	}
	if (NFKModelIsBoolean(value)) {
		return [value boolValue] ? @"true" : @"false";
	}
	if ([value isKindOfClass:NSNumber.class]) {
		return [value stringValue];
	}
	if ([value isKindOfClass:NSURL.class]) {
		return [NSString stringWithFormat:@"\"%@\"", [value path] ?: [value absoluteString]];
	}
	if ([NSJSONSerialization isValidJSONObject:value]) {
		NSData *data = [NSJSONSerialization dataWithJSONObject:value options:NSJSONWritingSortedKeys error:NULL];
		NSString *text = data != nil ? [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding] : nil;
		if (text != nil) {
			return text;
		}
	}
	return [value description];
}

#pragma mark - NFKModelOptionProblem

@implementation NFKModelOptionProblem

- (instancetype)initWithKey:(NSString *)key code:(NSString *)code message:(NSString *)message value:(nullable id)value
{
	self = [super init];
	if (self != nil) {
		_key = [key copy];
		_code = [code copy];
		_message = [message copy];
		_value = value;
	}
	return self;
}

- (NSDictionary<NSString *, id> *)dictionaryRepresentation
{
	NSMutableDictionary<NSString *, id> *dictionary = [NSMutableDictionary dictionary];
	dictionary[@"key"] = self.key;
	dictionary[@"code"] = self.code;
	dictionary[@"message"] = self.message;
	id value = self.value;
	BOOL scalar = [value isKindOfClass:NSString.class] || [value isKindOfClass:NSNumber.class];
	if (scalar || (value != nil && [NSJSONSerialization isValidJSONObject:value])) {
		dictionary[@"value"] = value;
	}
	return [dictionary copy];
}

- (id)copyWithZone:(nullable NSZone *)zone
{
	return self;
}

- (BOOL)isEqual:(id)object
{
	if (![object isKindOfClass:NFKModelOptionProblem.class]) {
		return NO;
	}
	NFKModelOptionProblem *other = object;
	return [other.key isEqualToString:self.key] && [other.code isEqualToString:self.code]
		&& [other.message isEqualToString:self.message];
}

- (NSUInteger)hash
{
	return self.key.hash ^ self.code.hash;
}

- (NSString *)description
{
	return [NSString stringWithFormat:@"<%@ %@ %@: %@>", NSStringFromClass(self.class), self.key, self.code, self.message];
}

@end

#pragma mark - Coercion

// The noun a wrong-type message names for a kind.
static NSString *NFKModelKindNoun(NSString *kind)
{
	NSDictionary<NSString *, NSString *> *nouns = @{
		NFKModelOptionKindInteger: @"an integer", NFKModelOptionKindNumber: @"a number",
		NFKModelOptionKindBoolean: @"a boolean", NFKModelOptionKindText: @"a string",
		NFKModelOptionKindTextList: @"a list of strings", NFKModelOptionKindFileOrFolder: @"a path",
		NFKModelOptionKindModelReference: @"a model identifier", NFKModelOptionKindChoice: @"one of its choices",
	};
	return nouns[kind] ?: @"a JSON value";
}

static NSNumber * _Nullable NFKModelNumberFromString(NSString *string, BOOL integer)
{
	NSString *trimmed = [string stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceCharacterSet];
	NSScanner *scanner = [NSScanner scannerWithString:trimmed];
	scanner.locale = nil;
	if (integer) {
		long long value = 0;
		if ([scanner scanLongLong:&value] && scanner.isAtEnd) {
			return @(value);
		}
		return nil;
	}
	double value = 0;
	if ([scanner scanDouble:&value] && scanner.isAtEnd) {
		return @(value);
	}
	return nil;
}

static NSNumber * _Nullable NFKModelBooleanFrom(id value)
{
	if ([value isKindOfClass:NSNumber.class]) {
		return @([value boolValue]);
	}
	if (![value isKindOfClass:NSString.class]) {
		return nil;
	}
	NSString *lowered = [value lowercaseString];
	if ([@[ @"true", @"yes", @"1" ] containsObject:lowered]) {
		return @YES;
	}
	if ([@[ @"false", @"no", @"0" ] containsObject:lowered]) {
		return @NO;
	}
	return nil;
}

// The value converted to a scalar kind, or nil when it does not convert.
static id _Nullable NFKModelCoerceValue(id value, NSString *kind)
{
	if ([kind isEqualToString:NFKModelOptionKindInteger]) {
		if ([value isKindOfClass:NSNumber.class] && !NFKModelIsBoolean(value)) {
			double number = [value doubleValue];
			return number == floor(number) ? @((long long)number) : nil;
		}
		return [value isKindOfClass:NSString.class] ? NFKModelNumberFromString(value, YES) : nil;
	}
	if ([kind isEqualToString:NFKModelOptionKindNumber]) {
		if ([value isKindOfClass:NSNumber.class] && !NFKModelIsBoolean(value)) {
			return value;
		}
		return [value isKindOfClass:NSString.class] ? NFKModelNumberFromString(value, NO) : nil;
	}
	if ([kind isEqualToString:NFKModelOptionKindBoolean]) {
		return NFKModelBooleanFrom(value);
	}
	if ([kind isEqualToString:NFKModelOptionKindText] || [kind isEqualToString:NFKModelOptionKindModelReference]) {
		return [value isKindOfClass:NSString.class] ? value : nil;
	}
	if ([kind isEqualToString:NFKModelOptionKindTextList]) {
		if ([value isKindOfClass:NSString.class]) {
			return @[ value ];
		}
		if (![value isKindOfClass:NSArray.class]) {
			return nil;
		}
		for (id element in value) {
			if (![element isKindOfClass:NSString.class]) {
				return nil;
			}
		}
		return value;
	}
	if ([kind isEqualToString:NFKModelOptionKindFileOrFolder]) {
		if ([value isKindOfClass:NSURL.class] && [value isFileURL]) {
			return [value path];
		}
		return [value isKindOfClass:NSString.class] ? [value stringByExpandingTildeInPath] : nil;
	}
	BOOL scalar = [value isKindOfClass:NSString.class] || [value isKindOfClass:NSNumber.class];
	return (scalar || [NSJSONSerialization isValidJSONObject:value]) ? value : nil;
}

#pragma mark - Validation

// The outcome of resolving one value: the value to use, or the problem.
@interface NFKModelOptionResolution : NSObject
@property (nonatomic, nullable) id value;
@property (nonatomic, nullable) NFKModelOptionProblem *problem;
@end

@implementation NFKModelOptionResolution
@end

@implementation NFKModelDescriptor (NFKModelCatalogInternal)

- (NSSet<NSString *> *)requestOverridableOptionKeys
{
	NFKModelMemoryModel *model = self.memoryModel;
	NSMutableSet<NSString *> *keys = [NSMutableSet set];
	for (NSString *key in @[ model.contextLengthOptionKey ?: @"", model.cacheBitsOptionKey ?: @"",
							 model.cacheGroupSizeOptionKey ?: @"" ]) {
		if (key.length > 0) {
			[keys addObject:key];
		}
	}
	return keys;
}

- (NFKModelOptionResolution *)problem:(NSString *)code key:(NSString *)key value:(nullable id)value
							  message:(NSString *)message
{
	NFKModelOptionResolution *resolution = [[NFKModelOptionResolution alloc] init];
	resolution.problem = [[NFKModelOptionProblem alloc] initWithKey:key code:code message:message value:value];
	return resolution;
}

- (NFKModelOptionResolution *)resolveValue:(id)value
									forKey:(NSString *)key
									 stage:(nullable NSString *)stage
								   options:(NSDictionary<NSString *, id> *)options
								   catalog:(NFKModelCatalog *)catalog
								   profile:(nullable NFKHardwareProfile *)profile
{
	if ([key isEqualToString:NFKModelLoadOptionIgnoreMemoryBudget]) {
		return [self resolveIgnoreBudget:value stage:stage];
	}
	NFKModelOption *option = [self optionWithKey:key];
	if (option == nil) {
		return [self problem:NFKModelOptionProblemUnknownKey key:key value:value
					 message:[NSString stringWithFormat:@"%@ has no option \"%@\".", self.identifier, key]];
	}
	NFKModelOptionResolution *wrongStage = [self wrongStageOf:option value:value stage:stage];
	if (wrongStage != nil) {
		return wrongStage;
	}
	NSString *optionProblem = profile != nil ? [option.requirements problemForHardwareProfile:profile] : nil;
	if (optionProblem != nil) {
		return [self problem:NFKModelOptionProblemUnavailable key:key value:value
					 message:[NSString stringWithFormat:@"\"%@\" is unavailable on this machine. %@", key, optionProblem]];
	}
	if ([option.kind isEqualToString:NFKModelOptionKindChoice]) {
		return [self resolveChoice:value option:option profile:profile];
	}
	if ([option.kind isEqualToString:NFKModelOptionKindModelReference]) {
		return [self resolveModelReference:value option:option options:options catalog:catalog];
	}
	id coerced = NFKModelCoerceValue(value, option.kind);
	if (coerced == nil) {
		return [self problem:NFKModelOptionProblemWrongType key:key value:value
					 message:[NSString stringWithFormat:@"\"%@\" takes %@, and %@ is not one.", key,
							  NFKModelKindNoun(option.kind), NFKModelOptionValueText(value)]];
	}
	if ([option.kind isEqualToString:NFKModelOptionKindInteger] || [option.kind isEqualToString:NFKModelOptionKindNumber]) {
		return [self resolveNumber:coerced option:option];
	}
	if ([option.kind isEqualToString:NFKModelOptionKindFileOrFolder]
		&& ![NSFileManager.defaultManager fileExistsAtPath:coerced]) {
		return [self problem:NFKModelOptionProblemMissingFile key:key value:value
					 message:[NSString stringWithFormat:@"\"%@\" names %@, which does not exist.", key,
							  NFKModelOptionValueText(coerced)]];
	}
	NFKModelOptionResolution *resolution = [[NFKModelOptionResolution alloc] init];
	resolution.value = coerced;
	return resolution;
}

- (NFKModelOptionResolution *)resolveIgnoreBudget:(id)value stage:(nullable NSString *)stage
{
	NSString *key = NFKModelLoadOptionIgnoreMemoryBudget;
	if ([stage isEqualToString:NFKModelOptionStageRequest]) {
		return [self problem:NFKModelOptionProblemWrongStage key:key value:value
					 message:[NSString stringWithFormat:@"\"%@\" is set when the model loads.", key]];
	}
	NSNumber *flag = NFKModelBooleanFrom(value);
	if (flag == nil) {
		return [self problem:NFKModelOptionProblemWrongType key:key value:value
					 message:[NSString stringWithFormat:@"\"%@\" takes a boolean, and %@ is not one.", key,
							  NFKModelOptionValueText(value)]];
	}
	NFKModelOptionResolution *resolution = [[NFKModelOptionResolution alloc] init];
	resolution.value = flag;
	return resolution;
}

- (nullable NFKModelOptionResolution *)wrongStageOf:(NFKModelOption *)option value:(id)value stage:(nullable NSString *)stage
{
	if (stage == nil || [option.stage isEqualToString:stage]) {
		return nil;
	}
	BOOL overridable = [stage isEqualToString:NFKModelOptionStageRequest]
		&& [self.requestOverridableOptionKeys containsObject:option.key];
	if (overridable) {
		return nil;
	}
	NSString *message = [option.stage isEqualToString:NFKModelOptionStageLoad]
		? [NSString stringWithFormat:@"\"%@\" is set when the model loads.", option.key]
		: [NSString stringWithFormat:@"\"%@\" is set on each request.", option.key];
	return [self problem:NFKModelOptionProblemWrongStage key:option.key value:value message:message];
}

- (NFKModelOptionResolution *)resolveChoice:(id)value option:(NFKModelOption *)option profile:(nullable NFKHardwareProfile *)profile
{
	NFKModelOptionChoice *choice = [option choiceWithValue:value];
	if (choice == nil) {
		NSMutableArray<NSString *> *listed = [NSMutableArray array];
		for (NFKModelOptionChoice *each in option.choices) {
			[listed addObject:NFKModelOptionValueText(each.value)];
		}
		return [self problem:NFKModelOptionProblemNotAChoice key:option.key value:value
					 message:[NSString stringWithFormat:@"\"%@\" takes one of %@, and %@ is none of them.", option.key,
							  [listed componentsJoinedByString:@", "], NFKModelOptionValueText(value)]];
	}
	NSString *problem = profile != nil ? [choice.requirements problemForHardwareProfile:profile] : nil;
	if (problem != nil) {
		return [self problem:NFKModelOptionProblemUnavailable key:option.key value:value
					 message:[NSString stringWithFormat:@"\"%@\" cannot be %@ on this machine. %@", option.key,
							  NFKModelOptionValueText(choice.value), problem]];
	}
	NFKModelOptionResolution *resolution = [[NFKModelOptionResolution alloc] init];
	resolution.value = choice.value;
	return resolution;
}

- (NFKModelOptionResolution *)resolveNumber:(NSNumber *)number option:(NFKModelOption *)option
{
	NSString *key = option.key;
	if (option.minimum != nil && [number compare:option.minimum] == NSOrderedAscending) {
		return [self problem:NFKModelOptionProblemBelowMinimum key:key value:number
					 message:[NSString stringWithFormat:@"\"%@\" takes %@ or more, and %@ is less.", key,
							  NFKModelOptionValueText(option.minimum), NFKModelOptionValueText(number)]];
	}
	if (option.maximum != nil && [number compare:option.maximum] == NSOrderedDescending) {
		return [self problem:NFKModelOptionProblemAboveMaximum key:key value:number
					 message:[NSString stringWithFormat:@"\"%@\" takes %@ or less, and %@ is more.", key,
							  NFKModelOptionValueText(option.maximum), NFKModelOptionValueText(number)]];
	}
	double step = option.step.doubleValue;
	if (step > 0) {
		double base = option.minimum.doubleValue;
		double steps = (number.doubleValue - base) / step;
		if (fabs(steps - round(steps)) > 1e-9) {
			return [self problem:NFKModelOptionProblemOffStep key:key value:number
						 message:[NSString stringWithFormat:@"\"%@\" takes steps of %@ from %@, and %@ is between them.", key,
								  NFKModelOptionValueText(option.step), NFKModelOptionValueText(option.minimum ?: @0),
								  NFKModelOptionValueText(number)]];
		}
	}
	NFKModelOptionResolution *resolution = [[NFKModelOptionResolution alloc] init];
	resolution.value = number;
	return resolution;
}

- (NFKModelOptionResolution *)resolveModelReference:(id)value
											 option:(NFKModelOption *)option
											options:(NSDictionary<NSString *, id> *)options
											catalog:(NFKModelCatalog *)catalog
{
	NSString *key = option.key;
	if (![value isKindOfClass:NSString.class]) {
		return [self problem:NFKModelOptionProblemWrongType key:key value:value
					 message:[NSString stringWithFormat:@"\"%@\" takes a model identifier, and %@ is not one.", key,
							  NFKModelOptionValueText(value)]];
	}
	BOOL isDraft = [key isEqualToString:self.memoryModel.draftOptionKey ?: @""];
	NFKModelOptionResolution *resolution = [[NFKModelOptionResolution alloc] init];
	if (isDraft && [value length] == 0) {
		return resolution;
	}
	if ([catalog descriptorWithIdentifier:value] == nil) {
		return [self problem:NFKModelOptionProblemUnknownModel key:key value:value
					 message:[NSString stringWithFormat:@"\"%@\" names %@, which is not in the catalog.", key,
							  NFKModelOptionValueText(value)]];
	}
	if (option.choices.count > 0) {
		NFKModelOptionResolution *choice = [self resolveChoice:value option:option profile:nil];
		return choice;
	}
	if (isDraft) {
		NFKModelVariant *variant = [self variantWithIdentifier:[options[NFKModelLoadOptionVariant] description] ?: @""];
		NSArray<NFKModelDescriptor *> *drafts = [self draftDescriptorsForVariant:variant inCatalog:catalog];
		if (![[drafts valueForKey:@"identifier"] containsObject:value]) {
			return [self problem:NFKModelOptionProblemNotAChoice key:key value:value
						 message:[NSString stringWithFormat:@"\"%@\" names %@, which cannot draft for %@.", key,
								  NFKModelOptionValueText(value), self.identifier]];
		}
	}
	resolution.value = value;
	return resolution;
}

- (NSArray<NFKModelOptionProblem *> *)problemsWithOptions:(nullable NSDictionary<NSString *, id> *)options
													stage:(nullable NSString *)stage
												  catalog:(NFKModelCatalog *)catalog
												  profile:(nullable NFKHardwareProfile *)profile
{
	NSMutableArray<NFKModelOptionProblem *> *problems = [NSMutableArray array];
	for (NSString *key in [options.allKeys sortedArrayUsingSelector:@selector(compare:)]) {
		NFKModelOptionResolution *resolution = [self resolveValue:options[key] forKey:key stage:stage options:options
														  catalog:catalog profile:profile];
		if (resolution.problem != nil) {
			[problems addObject:resolution.problem];
		}
	}
	return problems;
}

- (nullable NSDictionary<NSString *, id> *)resolvedOptions:(nullable NSDictionary<NSString *, id> *)options
													 stage:(nullable NSString *)stage
												   catalog:(NFKModelCatalog *)catalog
												   profile:(nullable NFKHardwareProfile *)profile
													 error:(NSError **)error
{
	NSMutableDictionary<NSString *, id> *resolved = [NSMutableDictionary dictionary];
	NSMutableArray<NFKModelOptionProblem *> *problems = [NSMutableArray array];
	for (NSString *key in [options.allKeys sortedArrayUsingSelector:@selector(compare:)]) {
		NFKModelOptionResolution *resolution = [self resolveValue:options[key] forKey:key stage:stage options:options
														  catalog:catalog profile:profile];
		if (resolution.problem != nil) {
			[problems addObject:resolution.problem];
		} else if (resolution.value != nil && ![key isEqualToString:NFKModelLoadOptionIgnoreMemoryBudget]) {
			resolved[key] = resolution.value;
		}
	}
	if (problems.count > 0) {
		if (error != NULL) {
			*error = [NSError errorWithDomain:NFKModelCatalogErrorDomain code:NFKModelCatalogErrorInvalidOptions
									 userInfo:@{ NSLocalizedDescriptionKey: problems.firstObject.message,
												 NFKModelOptionProblemsKey: [problems copy] }];
		}
		return nil;
	}
	for (NFKModelOption *option in [self optionsOfStage:stage]) {
		if (resolved[option.key] != nil || options[option.key] != nil || option.defaultValue == nil) {
			continue;
		}
		resolved[option.key] = NFKModelCoerceValue(option.defaultValue, option.kind) ?: option.defaultValue;
	}
	return [resolved copy];
}

- (NSArray<NFKModelOption *> *)optionsOfStage:(nullable NSString *)stage
{
	if ([stage isEqualToString:NFKModelOptionStageLoad]) {
		return self.loadOptions;
	}
	if ([stage isEqualToString:NFKModelOptionStageRequest]) {
		return self.requestOptions;
	}
	return [self.loadOptions arrayByAddingObjectsFromArray:self.requestOptions];
}

@end

@implementation NFKModelDescriptor (NFKModelOptionValidation)

- (NSArray<NFKModelOptionProblem *> *)problemsWithOptions:(nullable NSDictionary<NSString *, id> *)options
													stage:(nullable NSString *)stage
{
	return [self problemsWithOptions:options stage:stage catalog:NFKModelCatalog.sharedCatalog
							 profile:NFKHardwareProfile.currentProfile];
}

- (nullable NSDictionary<NSString *, id> *)resolvedOptions:(nullable NSDictionary<NSString *, id> *)options
													 stage:(nullable NSString *)stage
													 error:(NSError **)error
{
	return [self resolvedOptions:options stage:stage catalog:NFKModelCatalog.sharedCatalog
						 profile:NFKHardwareProfile.currentProfile error:error];
}

- (nullable NSString *)canonicalLoadOptionsKey:(nullable NSDictionary<NSString *, id> *)loadOptions
{
	NSDictionary<NSString *, id> *resolved = [self resolvedOptions:loadOptions stage:NFKModelOptionStageLoad
														   catalog:NFKModelCatalog.sharedCatalog profile:nil error:NULL];
	if (resolved == nil || ![NSJSONSerialization isValidJSONObject:resolved]) {
		return nil;
	}
	NSData *data = [NSJSONSerialization dataWithJSONObject:resolved
												   options:NSJSONWritingSortedKeys | NSJSONWritingWithoutEscapingSlashes
													 error:NULL];
	return data != nil ? [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding] : nil;
}

- (NSArray<NFKModelDescriptor *> *)draftDescriptorsForVariant:(nullable NFKModelVariant *)variant
													inCatalog:(NFKModelCatalog *)catalog
{
	NFKModelVariant *drafted = variant ?: self.defaultVariant;
	NSNumber *vocabulary = drafted.memoryFacts.vocabularySize;
	NSNumber *parameters = drafted.parameterCount;
	if (vocabulary == nil || parameters == nil) {
		return @[];
	}
	NSMutableArray<NFKModelDescriptor *> *drafts = [NSMutableArray array];
	for (NFKModelDescriptor *candidate in catalog.descriptors) {
		NFKModelVariant *candidateVariant = candidate.defaultVariant;
		BOOL sameFamily = ![candidate.identifier isEqualToString:self.identifier]
			&& [candidate.family isEqualToString:self.family];
		BOOL sameVocabulary = [candidateVariant.memoryFacts.vocabularySize isEqualToNumber:vocabulary];
		BOOL smaller = candidateVariant.parameterCount != nil
			&& [candidateVariant.parameterCount compare:parameters] == NSOrderedAscending;
		if (sameFamily && sameVocabulary && smaller) {
			[drafts addObject:candidate];
		}
	}
	return drafts;
}

@end
