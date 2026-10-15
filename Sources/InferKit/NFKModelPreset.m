//
//  NFKModelPreset.m
//  InferKit
//

#import "NFKModelPreset.h"
#import "NFKModelDescriptorPrivate.h"

NSString * const NFKModelPresetSeparator = @"@";
NSString * const NFKModelToolNameSeparator = @"__";

static const NSUInteger NFKModelToolNameLength = 64;

NSString *NFKModelPresetModelName(NSString *modelIdentifier, NSString *presetIdentifier)
{
	return [NSString stringWithFormat:@"%@%@%@", modelIdentifier, NFKModelPresetSeparator, presetIdentifier];
}

BOOL NFKModelNameSplitAtPreset(NSString *name,
							   BOOL (^ _Nullable isPreset)(NSString *model, NSString *preset),
							   NSString * _Nullable * _Nullable model,
							   NSString * _Nullable * _Nullable preset)
{
	if (model != NULL) {
		*model = name;
	}
	if (preset != NULL) {
		*preset = nil;
	}
	NSRange separator = [name rangeOfString:NFKModelPresetSeparator options:NSBackwardsSearch];
	if (separator.location == NSNotFound) {
		return NO;
	}
	NSString *modelPart = [name substringToIndex:separator.location];
	NSString *presetPart = [name substringFromIndex:NSMaxRange(separator)];
	if (modelPart.length == 0 || presetPart.length == 0) {
		return NO;
	}
	if (isPreset != nil && !isPreset(modelPart, presetPart)) {
		return NO;
	}
	if (model != NULL) {
		*model = modelPart;
	}
	if (preset != NULL) {
		*preset = presetPart;
	}
	return YES;
}

NSString *NFKModelToolName(NSString *modelName)
{
	NSString *separated = [modelName stringByReplacingOccurrencesOfString:NFKModelPresetSeparator
															   withString:NFKModelToolNameSeparator];
	NSMutableString *name = [NSMutableString stringWithCapacity:separated.length];
	for (NSUInteger index = 0; index < separated.length; index += 1) {
		unichar character = [separated characterAtIndex:index];
		BOOL allowed = (character >= 'a' && character <= 'z') || (character >= 'A' && character <= 'Z')
			|| (character >= '0' && character <= '9') || character == '_' || character == '-';
		[name appendString:allowed ? [NSString stringWithCharacters:&character length:1] : @"_"];
	}
	if (name.length > NFKModelToolNameLength) {
		return [name substringToIndex:NFKModelToolNameLength];
	}
	return [name copy];
}

static NSArray<NSString *> *NFKModelPresetFields(void)
{
	return @[ @"id", @"title", @"help", @"base", @"loadOptions", @"requestDefaults", @"suggested" ];
}

@implementation NFKModelPreset

- (instancetype)initWithIdentifier:(NSString *)identifier
							 title:(nullable NSString *)title
							  help:(nullable NSString *)help
			   baseModelIdentifier:(nullable NSString *)baseModelIdentifier
					   loadOptions:(nullable NSDictionary<NSString *, id> *)loadOptions
				   requestDefaults:(nullable NSDictionary<NSString *, id> *)requestDefaults
						 suggested:(BOOL)suggested
						extensions:(nullable NSDictionary<NSString *, id> *)extensions
{
	self = [super init];
	if (self != nil) {
		_identifier = [identifier copy];
		_title = [title copy] ?: _identifier;
		_help = [help copy];
		_baseModelIdentifier = [baseModelIdentifier copy];
		_loadOptions = [loadOptions copy] ?: @{};
		_requestDefaults = [requestDefaults copy] ?: @{};
		_suggested = suggested;
		_extensions = NFKModelExtensionsWithout(extensions, NFKModelPresetFields());
	}
	return self;
}

- (BOOL)changesLoadOptions
{
	return self.loadOptions.count > 0;
}

- (NSString *)modelNameForBaseModelIdentifier:(nullable NSString *)baseModelIdentifier
{
	return NFKModelPresetModelName(self.baseModelIdentifier ?: baseModelIdentifier ?: @"", self.identifier);
}

+ (nullable instancetype)presetWithDictionary:(NSDictionary<NSString *, id> *)dictionary error:(NSError **)error
{
	return [self presetFromObject:dictionary path:@"preset" error:error];
}

+ (nullable instancetype)presetFromObject:(id)object path:(NSString *)path error:(NSError **)error
{
	NFKModelFieldReader *reader = [NFKModelFieldReader readerWithObject:object path:path error:error];
	NSString *identifier = nil, *title = nil, *help = nil, *base = nil;
	NSDictionary<NSString *, id> *loadOptions = nil, *requestDefaults = nil;
	BOOL suggested = NO;
	if (reader == nil
		|| ![reader readString:@"id" required:YES into:&identifier error:error]
		|| ![reader readString:@"title" required:NO into:&title error:error]
		|| ![reader readString:@"help" required:NO into:&help error:error]
		|| ![reader readString:@"base" required:NO into:&base error:error]
		|| ![reader readJSONDictionary:@"loadOptions" into:&loadOptions error:error]
		|| ![reader readJSONDictionary:@"requestDefaults" into:&requestDefaults error:error]
		|| ![reader readBoolean:@"suggested" into:&suggested error:error]) {
		return nil;
	}
	return [[self alloc] initWithIdentifier:identifier title:title help:help baseModelIdentifier:base
								loadOptions:loadOptions requestDefaults:requestDefaults suggested:suggested
								 extensions:[reader extensionsWithoutFields:NFKModelPresetFields()]];
}

- (NSDictionary<NSString *, id> *)dictionaryRepresentation
{
	NSMutableDictionary<NSString *, id> *dictionary = [self.extensions mutableCopy];
	dictionary[@"id"] = self.identifier;
	dictionary[@"title"] = self.title;
	dictionary[@"help"] = self.help;
	dictionary[@"base"] = self.baseModelIdentifier;
	if (self.loadOptions.count > 0) {
		dictionary[@"loadOptions"] = self.loadOptions;
	}
	if (self.requestDefaults.count > 0) {
		dictionary[@"requestDefaults"] = self.requestDefaults;
	}
	dictionary[@"suggested"] = @(self.isSuggested);
	return [dictionary copy];
}

- (id)copyWithZone:(nullable NSZone *)zone
{
	return self;
}

- (BOOL)isEqual:(id)object
{
	return NFKModelIsEqualDescription(self, object, NFKModelPreset.class);
}

- (NSUInteger)hash
{
	return self.identifier.hash;
}

@end
