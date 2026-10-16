//
//  NFKModelDescriptor.m
//  InferKit
//

#import "NFKModelDescriptor.h"
#import "NFKModelDescriptorPrivate.h"
#import "NFKModelCatalog.h"
#import "NFKModelMemory.h"
#import "NFKModelPreset.h"
#import "NFKHardwareProfile.h"
#import "NFKInferenceKeys.h"

BOOL NFKModelCatalogFail(NFKModelCatalogError code, NSString *reason, NSError **error)
{
	if (error != NULL) {
		*error = [NSError errorWithDomain:NFKModelCatalogErrorDomain code:code
								 userInfo:@{ NSLocalizedDescriptionKey: reason }];
	}
	return NO;
}

BOOL NFKModelIsBoolean(id _Nullable value)
{
	return [value isKindOfClass:NSNumber.class] && CFGetTypeID((__bridge CFTypeRef)value) == CFBooleanGetTypeID();
}

NSDictionary<NSString *, id> *NFKModelExtensionsWithout(NSDictionary<NSString *, id> * _Nullable extensions,
														NSArray<NSString *> *knownFields)
{
	if (extensions.count == 0) {
		return @{};
	}
	NSMutableDictionary<NSString *, id> *kept = [extensions mutableCopy];
	[kept removeObjectsForKeys:knownFields];
	return [kept copy];
}

// The option kind that edits a value of the type. The five scalar kinds share their value type's string.
static NSString *NFKModelOptionKindForValueType(NSString * _Nullable valueType)
{
	NSArray<NSString *> *shared = @[ NFKModelValueTypeBoolean, NFKModelValueTypeInteger, NFKModelValueTypeNumber,
									 NFKModelValueTypeText, NFKModelValueTypeTextList ];
	if (valueType != nil && [shared containsObject:valueType]) {
		return valueType;
	}
	return NFKModelOptionKindJSON;
}

static NSArray<NSString *> *NFKModelValueTypesOfPorts(NSArray<NFKModelPort *> *ports)
{
	NSMutableOrderedSet<NSString *> *types = [NSMutableOrderedSet orderedSet];
	for (NFKModelPort *port in ports) {
		if (port.valueType != nil) {
			[types addObject:port.valueType];
		}
	}
	return types.array;
}

static NSArray<NSString *> *NFKModelSortedKeys(NSSet<NSString *> * _Nullable keys)
{
	return [keys.allObjects sortedArrayUsingSelector:@selector(compare:)] ?: @[];
}

NSArray *NFKModelDictionariesOf(NSArray *values)
{
	NSMutableArray *dictionaries = [NSMutableArray arrayWithCapacity:values.count];
	for (id value in values) {
		[dictionaries addObject:[value dictionaryRepresentation]];
	}
	return dictionaries;
}

BOOL NFKModelIsEqualDescription(id object, id other, Class expectedClass)
{
	if (object == other) {
		return YES;
	}
	if (![other isKindOfClass:expectedClass]) {
		return NO;
	}
	return [[object dictionaryRepresentation] isEqualToDictionary:[other dictionaryRepresentation]];
}

#pragma mark - Field reader

@implementation NFKModelFieldReader

+ (nullable instancetype)readerWithObject:(id)object path:(NSString *)path error:(NSError **)error
{
	if (![object isKindOfClass:NSDictionary.class]) {
		NFKModelCatalogFail(NFKModelCatalogErrorMalformedDescriptor,
							[NSString stringWithFormat:@"%@ is not a JSON object", path], error);
		return nil;
	}
	return [[self alloc] initWithDictionary:object path:path];
}

- (instancetype)initWithDictionary:(NSDictionary<NSString *, id> *)dictionary path:(NSString *)path
{
	self = [super init];
	if (self != nil) {
		_dictionary = [dictionary copy];
		_path = [path copy];
	}
	return self;
}

- (NSString *)pathForField:(NSString *)field
{
	return [NSString stringWithFormat:@"%@.%@", self.path, field];
}

- (BOOL)failField:(NSString *)field reason:(NSString *)reason error:(NSError **)error
{
	return NFKModelCatalogFail(NFKModelCatalogErrorMalformedDescriptor,
							   [NSString stringWithFormat:@"%@ %@", [self pathForField:field], reason], error);
}

- (NSDictionary<NSString *, id> *)extensionsWithoutFields:(NSArray<NSString *> *)knownFields
{
	return NFKModelExtensionsWithout(self.dictionary, knownFields);
}

- (BOOL)readString:(NSString *)field required:(BOOL)required into:(NSString * _Nullable * _Nonnull)value error:(NSError **)error
{
	id raw = self.dictionary[field];
	*value = nil;
	if (raw == nil) {
		return required ? [self failField:field reason:@"is missing" error:error] : YES;
	}
	if (![raw isKindOfClass:NSString.class]) {
		return [self failField:field reason:@"is not a string" error:error];
	}
	*value = raw;
	return YES;
}

- (BOOL)readNumber:(NSString *)field into:(NSNumber * _Nullable * _Nonnull)value error:(NSError **)error
{
	id raw = self.dictionary[field];
	*value = nil;
	if (raw == nil) {
		return YES;
	}
	if (![raw isKindOfClass:NSNumber.class] || NFKModelIsBoolean(raw)) {
		return [self failField:field reason:@"is not a number" error:error];
	}
	*value = raw;
	return YES;
}

- (BOOL)readBoolean:(NSString *)field into:(BOOL *)value error:(NSError **)error
{
	id raw = self.dictionary[field];
	*value = NO;
	if (raw == nil) {
		return YES;
	}
	if (![raw isKindOfClass:NSNumber.class]) {
		return [self failField:field reason:@"is not a boolean" error:error];
	}
	*value = [raw boolValue];
	return YES;
}

- (BOOL)readURL:(NSString *)field into:(NSURL * _Nullable * _Nonnull)value error:(NSError **)error
{
	NSString *string = nil;
	*value = nil;
	if (![self readString:field required:NO into:&string error:error]) {
		return NO;
	}
	if (string == nil) {
		return YES;
	}
	NSURL *URL = [NSURL URLWithString:string];
	if (URL == nil) {
		return [self failField:field reason:@"is not a URL" error:error];
	}
	*value = URL;
	return YES;
}

- (BOOL)readStringArray:(NSString *)field into:(NSArray<NSString *> * _Nullable * _Nonnull)value error:(NSError **)error
{
	id raw = self.dictionary[field];
	*value = nil;
	if (raw == nil) {
		return YES;
	}
	if (![raw isKindOfClass:NSArray.class]) {
		return [self failField:field reason:@"is not an array of strings" error:error];
	}
	for (id element in raw) {
		if (![element isKindOfClass:NSString.class]) {
			return [self failField:field reason:@"is not an array of strings" error:error];
		}
	}
	*value = raw;
	return YES;
}

- (BOOL)readStringDictionary:(NSString *)field
						into:(NSDictionary<NSString *, NSString *> * _Nullable * _Nonnull)value
					   error:(NSError **)error
{
	id raw = self.dictionary[field];
	*value = nil;
	if (raw == nil) {
		return YES;
	}
	if (![raw isKindOfClass:NSDictionary.class]) {
		return [self failField:field reason:@"is not an object of strings" error:error];
	}
	for (id element in [raw allValues]) {
		if (![element isKindOfClass:NSString.class]) {
			return [self failField:field reason:@"is not an object of strings" error:error];
		}
	}
	*value = raw;
	return YES;
}

- (BOOL)readNumberDictionary:(NSString *)field
						into:(NSDictionary<NSString *, NSNumber *> * _Nullable * _Nonnull)value
					   error:(NSError **)error
{
	id raw = self.dictionary[field];
	*value = nil;
	if (raw == nil) {
		return YES;
	}
	if (![raw isKindOfClass:NSDictionary.class]) {
		return [self failField:field reason:@"is not an object of numbers" error:error];
	}
	for (id element in [raw allValues]) {
		if (![element isKindOfClass:NSNumber.class] || NFKModelIsBoolean(element)) {
			return [self failField:field reason:@"is not an object of numbers" error:error];
		}
	}
	*value = raw;
	return YES;
}

- (BOOL)readJSONDictionary:(NSString *)field
					  into:(NSDictionary<NSString *, id> * _Nullable * _Nonnull)value
					 error:(NSError **)error
{
	id raw = self.dictionary[field];
	*value = nil;
	if (raw == nil) {
		return YES;
	}
	if (![raw isKindOfClass:NSDictionary.class] || ![NSJSONSerialization isValidJSONObject:raw]) {
		return [self failField:field reason:@"is not a JSON object" error:error];
	}
	*value = raw;
	return YES;
}

- (BOOL)readObjectDictionary:(NSString *)field
				  withReader:(NFKModelElementReader)reader
						into:(NSDictionary<NSString *, id> * _Nullable * _Nonnull)value
					   error:(NSError **)error
{
	id raw = self.dictionary[field];
	*value = nil;
	if (raw == nil) {
		return YES;
	}
	if (![raw isKindOfClass:NSDictionary.class]) {
		return [self failField:field reason:@"is not an object" error:error];
	}
	NSMutableDictionary<NSString *, id> *read = [NSMutableDictionary dictionaryWithCapacity:[raw count]];
	for (NSString *key in [[raw allKeys] sortedArrayUsingSelector:@selector(compare:)]) {
		NSString *elementPath = [NSString stringWithFormat:@"%@.%@", [self pathForField:field], key];
		id object = reader(raw[key], elementPath, error);
		if (object == nil) {
			return NO;
		}
		read[key] = object;
	}
	*value = read;
	return YES;
}

- (BOOL)requireUniqueElements:(nullable NSArray *)elements
					  ofField:(NSString *)field
					  keyPath:(NSString *)keyPath
						 name:(NSString *)name
						error:(NSError **)error
{
	NSMutableDictionary<id, NSNumber *> *indexByValue = [NSMutableDictionary dictionaryWithCapacity:elements.count];
	NSString *fieldPath = [self pathForField:field];
	NSUInteger index = 0;
	for (id element in elements) {
		id value = [element valueForKey:keyPath];
		NSNumber *earlier = value != nil ? indexByValue[value] : nil;
		if (earlier != nil) {
			NSString *reason = [NSString stringWithFormat:@"%@[%lu].%@ \"%@\" repeats %@[%lu].%@", fieldPath,
								(unsigned long)index, name, value, fieldPath, (unsigned long)earlier.unsignedIntegerValue, name];
			return NFKModelCatalogFail(NFKModelCatalogErrorMalformedDescriptor, reason, error);
		}
		if (value != nil) {
			indexByValue[value] = @(index);
		}
		index += 1;
	}
	return YES;
}

- (BOOL)readObject:(NSString *)field
		 withReader:(NFKModelElementReader)reader
			   into:(id _Nullable * _Nonnull)value
			  error:(NSError **)error
{
	id raw = self.dictionary[field];
	*value = nil;
	if (raw == nil) {
		return YES;
	}
	id read = reader(raw, [self pathForField:field], error);
	if (read == nil) {
		return NO;
	}
	*value = read;
	return YES;
}

- (BOOL)readObjects:(NSString *)field
		 withReader:(NFKModelElementReader)reader
			   into:(NSArray * _Nullable * _Nonnull)value
			  error:(NSError **)error
{
	id raw = self.dictionary[field];
	*value = nil;
	if (raw == nil) {
		return YES;
	}
	if (![raw isKindOfClass:NSArray.class]) {
		return [self failField:field reason:@"is not an array" error:error];
	}
	NSMutableArray *read = [NSMutableArray arrayWithCapacity:[raw count]];
	NSUInteger index = 0;
	for (id element in raw) {
		NSString *elementPath = [NSString stringWithFormat:@"%@[%lu]", [self pathForField:field], (unsigned long)index];
		id object = reader(element, elementPath, error);
		if (object == nil) {
			return NO;
		}
		[read addObject:object];
		index += 1;
	}
	*value = read;
	return YES;
}

@end

#pragma mark - NFKModelPort

static NSArray<NSString *> *NFKModelPortFields(void)
{
	return @[ @"key", @"valueType", @"required", @"title", @"help" ];
}

@implementation NFKModelPort

- (instancetype)initWithKey:(NSString *)key
				  valueType:(nullable NSString *)valueType
				   required:(BOOL)required
					  title:(nullable NSString *)title
					   help:(nullable NSString *)help
				 extensions:(nullable NSDictionary<NSString *, id> *)extensions
{
	self = [super init];
	if (self != nil) {
		_key = [key copy];
		_valueType = [valueType copy] ?: NFKModelValueTypeForKey(key);
		_required = required;
		_title = [title copy];
		_help = [help copy];
		_extensions = NFKModelExtensionsWithout(extensions, NFKModelPortFields());
	}
	return self;
}

+ (instancetype)portWithKey:(NSString *)key required:(BOOL)required
{
	return [[self alloc] initWithKey:key valueType:nil required:required title:nil help:nil extensions:nil];
}

+ (instancetype)outputPortWithKey:(NSString *)key required:(BOOL)required
{
	return [[self alloc] initWithKey:key valueType:NFKModelValueTypeForOutputKey(key) required:required
							   title:nil help:nil extensions:nil];
}

+ (nullable instancetype)portWithDictionary:(NSDictionary<NSString *, id> *)dictionary error:(NSError **)error
{
	return [self portFromObject:dictionary path:@"port" output:NO error:error];
}

+ (nullable instancetype)portFromObject:(id)object path:(NSString *)path output:(BOOL)output error:(NSError **)error
{
	NFKModelFieldReader *reader = [NFKModelFieldReader readerWithObject:object path:path error:error];
	NSString *key = nil, *valueType = nil, *title = nil, *help = nil;
	BOOL required = NO;
	if (reader == nil
		|| ![reader readString:@"key" required:YES into:&key error:error]
		|| ![reader readString:@"valueType" required:NO into:&valueType error:error]
		|| ![reader readBoolean:@"required" into:&required error:error]
		|| ![reader readString:@"title" required:NO into:&title error:error]
		|| ![reader readString:@"help" required:NO into:&help error:error]) {
		return nil;
	}
	if (valueType == nil && output) {
		valueType = NFKModelValueTypeForOutputKey(key);
	}
	return [[self alloc] initWithKey:key valueType:valueType required:required title:title help:help
						  extensions:[reader extensionsWithoutFields:NFKModelPortFields()]];
}

- (NSDictionary<NSString *, id> *)dictionaryRepresentation
{
	NSMutableDictionary<NSString *, id> *dictionary = [self.extensions mutableCopy];
	dictionary[@"key"] = self.key;
	dictionary[@"valueType"] = self.valueType;
	dictionary[@"required"] = @(self.isRequired);
	dictionary[@"title"] = self.title;
	dictionary[@"help"] = self.help;
	return [dictionary copy];
}

- (id)copyWithZone:(nullable NSZone *)zone
{
	return self;
}

- (BOOL)isEqual:(id)object
{
	return NFKModelIsEqualDescription(self, object, NFKModelPort.class);
}

- (NSUInteger)hash
{
	return self.key.hash;
}

@end

#pragma mark - NFKModelOptionChoice

static NSArray<NSString *> *NFKModelOptionChoiceFields(void)
{
	return @[ @"value", @"title", @"requires", @"memory", @"memoryByVariant" ];
}

static id _Nullable NFKModelReadRequirements(id element, NSString *elementPath, NSError **elementError);

@implementation NFKModelOptionChoice

- (instancetype)initWithValue:(id)value
						title:(nullable NSString *)title
				 requirements:(nullable NFKModelRequirements *)requirements
					   memory:(nullable NFKModelMemoryFigure *)memory
			  memoryByVariant:(nullable NSDictionary<NSString *, NFKModelMemoryFigure *> *)memoryByVariant
				   extensions:(nullable NSDictionary<NSString *, id> *)extensions
{
	self = [super init];
	if (self != nil) {
		_value = [value copy];
		_title = [title copy] ?: [value description];
		_requirements = [requirements copy];
		_memory = [memory copy];
		_memoryByVariant = [memoryByVariant copy] ?: @{};
		_extensions = NFKModelExtensionsWithout(extensions, NFKModelOptionChoiceFields());
	}
	return self;
}

- (instancetype)initWithValue:(id)value
						title:(nullable NSString *)title
				   extensions:(nullable NSDictionary<NSString *, id> *)extensions
{
	return [self initWithValue:value title:title requirements:nil memory:nil memoryByVariant:nil extensions:extensions];
}

- (nullable NFKModelMemoryFigure *)memoryForVariant:(nullable NSString *)variantIdentifier
{
	NFKModelMemoryFigure *figure = variantIdentifier != nil ? self.memoryByVariant[variantIdentifier] : nil;
	return figure ?: self.memory;
}

+ (instancetype)choiceWithValue:(id)value title:(nullable NSString *)title
{
	return [[self alloc] initWithValue:value title:title extensions:nil];
}

+ (nullable instancetype)choiceWithDictionary:(NSDictionary<NSString *, id> *)dictionary error:(NSError **)error
{
	return [self choiceFromObject:dictionary path:@"choice" error:error];
}

+ (nullable instancetype)choiceFromObject:(id)object path:(NSString *)path error:(NSError **)error
{
	NFKModelFieldReader *reader = [NFKModelFieldReader readerWithObject:object path:path error:error];
	if (reader == nil) {
		return nil;
	}
	id value = reader.dictionary[@"value"];
	if (value == nil) {
		[reader failField:@"value" reason:@"is missing" error:error];
		return nil;
	}
	if (![value isKindOfClass:NSString.class] && ![value isKindOfClass:NSNumber.class]) {
		[reader failField:@"value" reason:@"is not a string or a number" error:error];
		return nil;
	}
	NSString *title = nil;
	id requirements = nil, memory = nil;
	NSDictionary *memoryByVariant = nil;
	NFKModelElementReader figureReader = ^id (id element, NSString *elementPath, NSError **elementError) {
		return [NFKModelMemoryFigure figureFromObject:element path:elementPath error:elementError];
	};
	NFKModelElementReader requirementsReader = ^id (id element, NSString *elementPath, NSError **elementError) {
		return NFKModelReadRequirements(element, elementPath, elementError);
	};
	if (![reader readString:@"title" required:NO into:&title error:error]
		|| ![reader readObject:@"requires" withReader:requirementsReader into:&requirements error:error]
		|| ![reader readObject:@"memory" withReader:figureReader into:&memory error:error]
		|| ![reader readObjectDictionary:@"memoryByVariant" withReader:figureReader into:&memoryByVariant error:error]) {
		return nil;
	}
	return [[self alloc] initWithValue:value title:title requirements:requirements memory:memory
					   memoryByVariant:memoryByVariant
							extensions:[reader extensionsWithoutFields:NFKModelOptionChoiceFields()]];
}

- (NSDictionary<NSString *, id> *)dictionaryRepresentation
{
	NSMutableDictionary<NSString *, id> *dictionary = [self.extensions mutableCopy];
	dictionary[@"value"] = self.value;
	dictionary[@"title"] = self.title;
	dictionary[@"requires"] = self.requirements.dictionaryRepresentation;
	dictionary[@"memory"] = self.memory.dictionaryRepresentation;
	if (self.memoryByVariant.count > 0) {
		NSMutableDictionary<NSString *, id> *byVariant = [NSMutableDictionary dictionary];
		[self.memoryByVariant enumerateKeysAndObjectsUsingBlock:^(NSString *key, NFKModelMemoryFigure *figure, BOOL *stop) {
			byVariant[key] = figure.dictionaryRepresentation;
		}];
		dictionary[@"memoryByVariant"] = byVariant;
	}
	return [dictionary copy];
}

- (id)copyWithZone:(nullable NSZone *)zone
{
	return self;
}

- (BOOL)isEqual:(id)object
{
	return NFKModelIsEqualDescription(self, object, NFKModelOptionChoice.class);
}

- (NSUInteger)hash
{
	return [self.value hash];
}

@end

#pragma mark - NFKModelOption

static NSArray<NSString *> *NFKModelOptionFields(void)
{
	return @[ @"key", @"kind", @"title", @"help", @"default", @"minimum", @"maximum", @"step", @"choices",
			  @"stage", @"group", @"requires", @"property", @"memoryBytesPerUnit" ];
}

@implementation NFKModelOption

- (instancetype)initWithKey:(NSString *)key
					   kind:(NSString *)kind
					  title:(nullable NSString *)title
					   help:(nullable NSString *)help
			   defaultValue:(nullable id)defaultValue
					minimum:(nullable NSNumber *)minimum
					maximum:(nullable NSNumber *)maximum
					   step:(nullable NSNumber *)step
					choices:(nullable NSArray<NFKModelOptionChoice *> *)choices
					  stage:(nullable NSString *)stage
					  group:(nullable NSString *)group
			   requirements:(nullable NFKModelRequirements *)requirements
			propertyKeyPath:(nullable NSString *)propertyKeyPath
		 memoryBytesPerUnit:(nullable NSNumber *)memoryBytesPerUnit
				 extensions:(nullable NSDictionary<NSString *, id> *)extensions
{
	self = [super init];
	if (self != nil) {
		_key = [key copy];
		_kind = [kind copy];
		_title = [title copy];
		_help = [help copy];
		_defaultValue = [defaultValue copy];
		_minimum = [minimum copy];
		_maximum = [maximum copy];
		_step = [step copy];
		_choices = [choices copy] ?: @[];
		_stage = [stage copy] ?: NFKModelOptionStageRequest;
		_group = [group copy];
		_requirements = [requirements copy];
		_propertyKeyPath = [propertyKeyPath copy];
		_memoryBytesPerUnit = [memoryBytesPerUnit copy];
		_extensions = NFKModelExtensionsWithout(extensions, NFKModelOptionFields());
	}
	return self;
}

- (instancetype)initWithKey:(NSString *)key
					   kind:(NSString *)kind
					  title:(nullable NSString *)title
					   help:(nullable NSString *)help
			   defaultValue:(nullable id)defaultValue
					minimum:(nullable NSNumber *)minimum
					maximum:(nullable NSNumber *)maximum
					   step:(nullable NSNumber *)step
					choices:(nullable NSArray<NFKModelOptionChoice *> *)choices
					  stage:(nullable NSString *)stage
					  group:(nullable NSString *)group
				 extensions:(nullable NSDictionary<NSString *, id> *)extensions
{
	return [self initWithKey:key kind:kind title:title help:help defaultValue:defaultValue minimum:minimum
					 maximum:maximum step:step choices:choices stage:stage group:group requirements:nil
			 propertyKeyPath:nil memoryBytesPerUnit:nil extensions:extensions];
}

- (nullable NFKModelOptionChoice *)choiceWithValue:(id)value
{
	for (NFKModelOptionChoice *choice in self.choices) {
		if ([choice.value isEqual:value]) {
			return choice;
		}
	}
	NSString *text = [value description];
	for (NFKModelOptionChoice *choice in self.choices) {
		if ([[choice.value description] isEqualToString:text]) {
			return choice;
		}
	}
	return nil;
}

+ (instancetype)optionWithKey:(NSString *)key kind:(NSString *)kind
{
	return [[self alloc] initWithKey:key kind:kind title:nil help:nil defaultValue:nil minimum:nil maximum:nil
								step:nil choices:nil stage:nil group:nil extensions:nil];
}

+ (instancetype)optionWithKey:(NSString *)key
{
	return [self optionWithKey:key kind:NFKModelOptionKindForValueType(NFKModelValueTypeForKey(key))];
}

+ (nullable instancetype)optionWithDictionary:(NSDictionary<NSString *, id> *)dictionary error:(NSError **)error
{
	return [self optionFromObject:dictionary path:@"option" error:error];
}

+ (nullable instancetype)optionFromObject:(id)object path:(NSString *)path error:(NSError **)error
{
	NFKModelFieldReader *reader = [NFKModelFieldReader readerWithObject:object path:path error:error];
	NSString *key = nil, *kind = nil, *title = nil, *help = nil, *stage = nil, *group = nil, *property = nil;
	NSNumber *minimum = nil, *maximum = nil, *step = nil, *memoryBytesPerUnit = nil;
	NSArray *choices = nil;
	id requirements = nil;
	NFKModelElementReader requirementsReader = ^id (id element, NSString *elementPath, NSError **elementError) {
		return NFKModelReadRequirements(element, elementPath, elementError);
	};
	NFKModelElementReader choiceReader = ^id (id element, NSString *elementPath, NSError **elementError) {
		return [NFKModelOptionChoice choiceFromObject:element path:elementPath error:elementError];
	};
	if (reader == nil
		|| ![reader readString:@"key" required:YES into:&key error:error]
		|| ![reader readString:@"kind" required:NO into:&kind error:error]
		|| ![reader readString:@"title" required:NO into:&title error:error]
		|| ![reader readString:@"help" required:NO into:&help error:error]
		|| ![reader readNumber:@"minimum" into:&minimum error:error]
		|| ![reader readNumber:@"maximum" into:&maximum error:error]
		|| ![reader readNumber:@"step" into:&step error:error]
		|| ![reader readObjects:@"choices" withReader:choiceReader into:&choices error:error]
		|| ![reader readString:@"stage" required:NO into:&stage error:error]
		|| ![reader readString:@"group" required:NO into:&group error:error]
		|| ![reader readObject:@"requires" withReader:requirementsReader into:&requirements error:error]
		|| ![reader readString:@"property" required:NO into:&property error:error]
		|| ![reader readNumber:@"memoryBytesPerUnit" into:&memoryBytesPerUnit error:error]) {
		return nil;
	}
	id defaultValue = reader.dictionary[@"default"];
	if (defaultValue != nil && ![self isDefaultValue:defaultValue]) {
		[reader failField:@"default" reason:@"is not a string, number, boolean, array, or object" error:error];
		return nil;
	}
	return [[self alloc] initWithKey:key
								kind:kind ?: NFKModelOptionKindForValueType(NFKModelValueTypeForKey(key))
							   title:title
								help:help
						defaultValue:defaultValue
							 minimum:minimum
							 maximum:maximum
								step:step
							 choices:choices								   stage:stage
								   group:group
							requirements:requirements
						 propertyKeyPath:property
					  memoryBytesPerUnit:memoryBytesPerUnit
							  extensions:[reader extensionsWithoutFields:NFKModelOptionFields()]];
}

+ (BOOL)isDefaultValue:(id)value
{
	if ([value isKindOfClass:NSString.class] || [value isKindOfClass:NSNumber.class]) {
		return YES;
	}
	return ([value isKindOfClass:NSArray.class] || [value isKindOfClass:NSDictionary.class])
		&& [NSJSONSerialization isValidJSONObject:value];
}

- (NSDictionary<NSString *, id> *)dictionaryRepresentation
{
	NSMutableDictionary<NSString *, id> *dictionary = [self.extensions mutableCopy];
	dictionary[@"key"] = self.key;
	dictionary[@"kind"] = self.kind;
	dictionary[@"title"] = self.title;
	dictionary[@"help"] = self.help;
	dictionary[@"default"] = self.defaultValue;
	dictionary[@"minimum"] = self.minimum;
	dictionary[@"maximum"] = self.maximum;
	dictionary[@"step"] = self.step;
	if (self.choices.count > 0) {
		dictionary[@"choices"] = NFKModelDictionariesOf(self.choices);
	}
	dictionary[@"stage"] = self.stage;
	dictionary[@"group"] = self.group;
	dictionary[@"requires"] = self.requirements.dictionaryRepresentation;
	dictionary[@"property"] = self.propertyKeyPath;
	dictionary[@"memoryBytesPerUnit"] = self.memoryBytesPerUnit;
	return [dictionary copy];
}

- (id)copyWithZone:(nullable NSZone *)zone
{
	return self;
}

- (BOOL)isEqual:(id)object
{
	return NFKModelIsEqualDescription(self, object, NFKModelOption.class);
}

- (NSUInteger)hash
{
	return self.key.hash;
}

@end

#pragma mark - NFKModelLicense

static NSArray<NSString *> *NFKModelLicenseFields(void)
{
	return @[ @"id", @"name", @"url", @"gated", @"restrictsCommercialUse", @"note" ];
}

@implementation NFKModelLicense

- (instancetype)initWithIdentifier:(nullable NSString *)identifier
							  name:(NSString *)name
							   URL:(nullable NSURL *)URL
							 gated:(BOOL)gated
			restrictsCommercialUse:(BOOL)restrictsCommercialUse
							  note:(nullable NSString *)note
						extensions:(nullable NSDictionary<NSString *, id> *)extensions
{
	self = [super init];
	if (self != nil) {
		_identifier = [identifier copy];
		_name = [name copy];
		_URL = [URL copy];
		_gated = gated;
		_restrictsCommercialUse = restrictsCommercialUse;
		_note = [note copy];
		_extensions = NFKModelExtensionsWithout(extensions, NFKModelLicenseFields());
	}
	return self;
}

+ (instancetype)licenseWithIdentifier:(nullable NSString *)identifier name:(NSString *)name URL:(nullable NSURL *)URL
{
	return [[self alloc] initWithIdentifier:identifier name:name URL:URL gated:NO restrictsCommercialUse:NO
									   note:nil extensions:nil];
}

+ (nullable instancetype)licenseWithDictionary:(NSDictionary<NSString *, id> *)dictionary error:(NSError **)error
{
	return [self licenseFromObject:dictionary path:@"license" error:error];
}

+ (nullable instancetype)licenseFromObject:(id)object path:(NSString *)path error:(NSError **)error
{
	NFKModelFieldReader *reader = [NFKModelFieldReader readerWithObject:object path:path error:error];
	NSString *identifier = nil, *name = nil, *note = nil;
	NSURL *URL = nil;
	BOOL gated = NO, restrictsCommercialUse = NO;
	if (reader == nil
		|| ![reader readString:@"id" required:NO into:&identifier error:error]
		|| ![reader readString:@"name" required:NO into:&name error:error]
		|| ![reader readURL:@"url" into:&URL error:error]
		|| ![reader readBoolean:@"gated" into:&gated error:error]
		|| ![reader readBoolean:@"restrictsCommercialUse" into:&restrictsCommercialUse error:error]
		|| ![reader readString:@"note" required:NO into:&note error:error]) {
		return nil;
	}
	NSString *resolvedName = name ?: identifier;
	if (resolvedName == nil) {
		[reader failField:@"name" reason:@"is missing, and so is id" error:error];
		return nil;
	}
	return [[self alloc] initWithIdentifier:identifier name:resolvedName URL:URL gated:gated
					 restrictsCommercialUse:restrictsCommercialUse note:note
								 extensions:[reader extensionsWithoutFields:NFKModelLicenseFields()]];
}

- (NSDictionary<NSString *, id> *)dictionaryRepresentation
{
	NSMutableDictionary<NSString *, id> *dictionary = [self.extensions mutableCopy];
	dictionary[@"id"] = self.identifier;
	dictionary[@"name"] = self.name;
	dictionary[@"url"] = self.URL.absoluteString;
	dictionary[@"gated"] = @(self.isGated);
	dictionary[@"restrictsCommercialUse"] = @(self.restrictsCommercialUse);
	dictionary[@"note"] = self.note;
	return [dictionary copy];
}

- (id)copyWithZone:(nullable NSZone *)zone
{
	return self;
}

- (BOOL)isEqual:(id)object
{
	return NFKModelIsEqualDescription(self, object, NFKModelLicense.class);
}

- (NSUInteger)hash
{
	return self.name.hash;
}

@end

#pragma mark - NFKModelFile

static NSArray<NSString *> *NFKModelFileFields(void)
{
	return @[ @"path", @"bytes", @"sha256", @"role", @"optional", @"repository", @"revision", @"url" ];
}

@implementation NFKModelFile

- (instancetype)initWithPath:(NSString *)path
				   byteCount:(nullable NSNumber *)byteCount
					  sha256:(nullable NSString *)sha256
						role:(nullable NSString *)role
					optional:(BOOL)optional
				  repository:(nullable NSString *)repository
					revision:(nullable NSString *)revision
						 URL:(nullable NSURL *)URL
				  extensions:(nullable NSDictionary<NSString *, id> *)extensions
{
	self = [super init];
	if (self != nil) {
		_path = [path copy];
		_byteCount = [byteCount copy];
		_sha256 = [sha256 copy];
		_role = [role copy] ?: NFKModelFileRoleOther;
		_optional = optional;
		_repository = [repository copy];
		_revision = [revision copy];
		_URL = [URL copy];
		_extensions = NFKModelExtensionsWithout(extensions, NFKModelFileFields());
	}
	return self;
}

- (instancetype)initWithPath:(NSString *)path
				   byteCount:(nullable NSNumber *)byteCount
					  sha256:(nullable NSString *)sha256
						role:(nullable NSString *)role
					optional:(BOOL)optional
				  extensions:(nullable NSDictionary<NSString *, id> *)extensions
{
	return [self initWithPath:path byteCount:byteCount sha256:sha256 role:role optional:optional repository:nil
					 revision:nil URL:nil extensions:extensions];
}

+ (instancetype)fileWithPath:(NSString *)path role:(NSString *)role
{
	return [[self alloc] initWithPath:path byteCount:nil sha256:nil role:role optional:NO extensions:nil];
}

- (BOOL)overridesSource
{
	return self.repository != nil || self.revision != nil || self.URL != nil;
}

+ (nullable instancetype)fileWithDictionary:(NSDictionary<NSString *, id> *)dictionary error:(NSError **)error
{
	return [self fileFromObject:dictionary path:@"file" error:error];
}

+ (nullable instancetype)fileFromObject:(id)object path:(NSString *)path error:(NSError **)error
{
	NFKModelFieldReader *reader = [NFKModelFieldReader readerWithObject:object path:path error:error];
	NSString *filePath = nil, *sha256 = nil, *role = nil, *repository = nil, *revision = nil;
	NSNumber *byteCount = nil;
	NSURL *URL = nil;
	BOOL optional = NO;
	if (reader == nil
		|| ![reader readString:@"path" required:YES into:&filePath error:error]
		|| ![reader readNumber:@"bytes" into:&byteCount error:error]
		|| ![reader readString:@"sha256" required:NO into:&sha256 error:error]
		|| ![reader readString:@"role" required:NO into:&role error:error]
		|| ![reader readBoolean:@"optional" into:&optional error:error]
		|| ![reader readString:@"repository" required:NO into:&repository error:error]
		|| ![reader readString:@"revision" required:NO into:&revision error:error]
		|| ![reader readURL:@"url" into:&URL error:error]) {
		return nil;
	}
	return [[self alloc] initWithPath:filePath byteCount:byteCount sha256:sha256 role:role optional:optional
						   repository:repository revision:revision URL:URL
						   extensions:[reader extensionsWithoutFields:NFKModelFileFields()]];
}

- (NSDictionary<NSString *, id> *)dictionaryRepresentation
{
	NSMutableDictionary<NSString *, id> *dictionary = [self.extensions mutableCopy];
	dictionary[@"path"] = self.path;
	dictionary[@"bytes"] = self.byteCount;
	dictionary[@"sha256"] = self.sha256;
	dictionary[@"role"] = self.role;
	dictionary[@"optional"] = @(self.isOptional);
	dictionary[@"repository"] = self.repository;
	dictionary[@"revision"] = self.revision;
	dictionary[@"url"] = self.URL.absoluteString;
	return [dictionary copy];
}

- (id)copyWithZone:(nullable NSZone *)zone
{
	return self;
}

- (BOOL)isEqual:(id)object
{
	return NFKModelIsEqualDescription(self, object, NFKModelFile.class);
}

- (NSUInteger)hash
{
	return self.path.hash;
}

@end

#pragma mark - NFKModelSource

static NSArray<NSString *> *NFKModelSourceFields(void)
{
	return @[ @"kind", @"repository", @"revision", @"url", @"layout", @"files", @"requiresConversion" ];
}

@implementation NFKModelSource

- (instancetype)initWithKind:(NSString *)kind
				  repository:(nullable NSString *)repository
					revision:(nullable NSString *)revision
						 URL:(nullable NSURL *)URL
					  layout:(nullable NSString *)layout
					   files:(nullable NSArray<NFKModelFile *> *)files
		  requiresConversion:(BOOL)requiresConversion
				  extensions:(nullable NSDictionary<NSString *, id> *)extensions
{
	self = [super init];
	if (self != nil) {
		_kind = [kind copy];
		_repository = [repository copy];
		_revision = [revision copy];
		_URL = [URL copy];
		_layout = [layout copy] ?: NFKModelSourceLayoutNone;
		_files = [files copy] ?: @[];
		_requiresConversion = requiresConversion;
		_extensions = NFKModelExtensionsWithout(extensions, NFKModelSourceFields());
	}
	return self;
}

+ (instancetype)huggingFaceSourceWithRepository:(NSString *)repository
									   revision:(nullable NSString *)revision
										 layout:(NSString *)layout
{
	return [[self alloc] initWithKind:NFKModelSourceKindHuggingFace repository:repository revision:revision URL:nil
							   layout:layout files:nil requiresConversion:NO extensions:nil];
}

+ (nullable instancetype)sourceWithDictionary:(NSDictionary<NSString *, id> *)dictionary error:(NSError **)error
{
	return [self sourceFromObject:dictionary path:@"source" error:error];
}

+ (nullable instancetype)sourceFromObject:(id)object path:(NSString *)path error:(NSError **)error
{
	NFKModelFieldReader *reader = [NFKModelFieldReader readerWithObject:object path:path error:error];
	NSString *kind = nil, *repository = nil, *revision = nil, *layout = nil;
	NSURL *URL = nil;
	NSArray *files = nil;
	BOOL requiresConversion = NO;
	NFKModelElementReader fileReader = ^id (id element, NSString *elementPath, NSError **elementError) {
		return [NFKModelFile fileFromObject:element path:elementPath error:elementError];
	};
	if (reader == nil
		|| ![reader readString:@"kind" required:YES into:&kind error:error]
		|| ![reader readString:@"repository" required:NO into:&repository error:error]
		|| ![reader readString:@"revision" required:NO into:&revision error:error]
		|| ![reader readURL:@"url" into:&URL error:error]
		|| ![reader readString:@"layout" required:NO into:&layout error:error]
		|| ![reader readObjects:@"files" withReader:fileReader into:&files error:error]
		|| ![reader readBoolean:@"requiresConversion" into:&requiresConversion error:error]) {
		return nil;
	}
	return [[self alloc] initWithKind:kind repository:repository revision:revision URL:URL layout:layout files:files
				   requiresConversion:requiresConversion
						   extensions:[reader extensionsWithoutFields:NFKModelSourceFields()]];
}

- (NSDictionary<NSString *, id> *)dictionaryRepresentation
{
	NSMutableDictionary<NSString *, id> *dictionary = [self.extensions mutableCopy];
	dictionary[@"kind"] = self.kind;
	dictionary[@"repository"] = self.repository;
	dictionary[@"revision"] = self.revision;
	dictionary[@"url"] = self.URL.absoluteString;
	dictionary[@"layout"] = self.layout;
	dictionary[@"files"] = NFKModelDictionariesOf(self.files);
	dictionary[@"requiresConversion"] = @(self.requiresConversion);
	return [dictionary copy];
}

- (id)copyWithZone:(nullable NSZone *)zone
{
	return self;
}

- (BOOL)isEqual:(id)object
{
	return NFKModelIsEqualDescription(self, object, NFKModelSource.class);
}

- (NSUInteger)hash
{
	return self.kind.hash ^ self.repository.hash;
}

@end

#pragma mark - NFKModelVariant

static NSArray<NSString *> *NFKModelVariantFields(void)
{
	return @[ @"id", @"displayName", @"isDefault", @"registryName", @"source", @"downloadBytes", @"residentBytes",
			  @"peakBytes", @"parameterCount", @"precision", @"minimumMemoryBytes", @"license", @"memoryFacts" ];
}

@implementation NFKModelVariant

- (instancetype)initWithIdentifier:(NSString *)identifier
					   displayName:(nullable NSString *)displayName
						 isDefault:(BOOL)isDefault
					  registryName:(nullable NSString *)registryName
							source:(nullable NFKModelSource *)source
				 downloadByteCount:(nullable NSNumber *)downloadByteCount
				 residentByteCount:(nullable NSNumber *)residentByteCount
					 peakByteCount:(nullable NSNumber *)peakByteCount
					parameterCount:(nullable NSNumber *)parameterCount
						 precision:(nullable NSString *)precision
			minimumMemoryByteCount:(nullable NSNumber *)minimumMemoryByteCount
						   license:(nullable NFKModelLicense *)license
					   memoryFacts:(nullable NFKModelMemoryFacts *)memoryFacts
						extensions:(nullable NSDictionary<NSString *, id> *)extensions
{
	self = [super init];
	if (self != nil) {
		_identifier = [identifier copy];
		_displayName = [displayName copy] ?: _identifier;
		_isDefault = isDefault;
		_registryName = [registryName copy];
		_source = [source copy];
		_downloadByteCount = [downloadByteCount copy];
		_residentByteCount = [residentByteCount copy];
		_peakByteCount = [peakByteCount copy];
		_parameterCount = [parameterCount copy];
		_precision = [precision copy];
		_minimumMemoryByteCount = [minimumMemoryByteCount copy];
		_license = [license copy];
		_memoryFacts = [memoryFacts copy];
		_extensions = NFKModelExtensionsWithout(extensions, NFKModelVariantFields());
	}
	return self;
}

- (instancetype)initWithIdentifier:(NSString *)identifier
					   displayName:(nullable NSString *)displayName
						 isDefault:(BOOL)isDefault
					  registryName:(nullable NSString *)registryName
							source:(nullable NFKModelSource *)source
				 downloadByteCount:(nullable NSNumber *)downloadByteCount
				 residentByteCount:(nullable NSNumber *)residentByteCount
					 peakByteCount:(nullable NSNumber *)peakByteCount
					parameterCount:(nullable NSNumber *)parameterCount
						 precision:(nullable NSString *)precision
			minimumMemoryByteCount:(nullable NSNumber *)minimumMemoryByteCount
						   license:(nullable NFKModelLicense *)license
						extensions:(nullable NSDictionary<NSString *, id> *)extensions
{
	return [self initWithIdentifier:identifier displayName:displayName isDefault:isDefault registryName:registryName
							 source:source downloadByteCount:downloadByteCount residentByteCount:residentByteCount
					  peakByteCount:peakByteCount parameterCount:parameterCount precision:precision
			 minimumMemoryByteCount:minimumMemoryByteCount license:license memoryFacts:nil extensions:extensions];
}

- (instancetype)initWithIdentifier:(NSString *)identifier
					   displayName:(nullable NSString *)displayName
						 isDefault:(BOOL)isDefault
					  registryName:(nullable NSString *)registryName
							source:(nullable NFKModelSource *)source
				 downloadByteCount:(nullable NSNumber *)downloadByteCount
				 residentByteCount:(nullable NSNumber *)residentByteCount
					 peakByteCount:(nullable NSNumber *)peakByteCount
					parameterCount:(nullable NSNumber *)parameterCount
						 precision:(nullable NSString *)precision
			minimumMemoryByteCount:(nullable NSNumber *)minimumMemoryByteCount
						extensions:(nullable NSDictionary<NSString *, id> *)extensions
{
	return [self initWithIdentifier:identifier displayName:displayName isDefault:isDefault registryName:registryName
							 source:source downloadByteCount:downloadByteCount residentByteCount:residentByteCount
					  peakByteCount:peakByteCount parameterCount:parameterCount precision:precision
			 minimumMemoryByteCount:minimumMemoryByteCount license:nil extensions:extensions];
}

+ (instancetype)variantWithIdentifier:(NSString *)identifier displayName:(nullable NSString *)displayName
{
	return [[self alloc] initWithIdentifier:identifier displayName:displayName isDefault:NO registryName:nil
									 source:nil downloadByteCount:nil residentByteCount:nil peakByteCount:nil
							 parameterCount:nil precision:nil minimumMemoryByteCount:nil extensions:nil];
}

+ (nullable instancetype)variantWithDictionary:(NSDictionary<NSString *, id> *)dictionary error:(NSError **)error
{
	return [self variantFromObject:dictionary path:@"variant" error:error];
}

+ (nullable instancetype)variantFromObject:(id)object path:(NSString *)path error:(NSError **)error
{
	NFKModelFieldReader *reader = [NFKModelFieldReader readerWithObject:object path:path error:error];
	NSString *identifier = nil, *displayName = nil, *registryName = nil, *precision = nil;
	NSNumber *downloadByteCount = nil, *residentByteCount = nil, *peakByteCount = nil, *parameterCount = nil;
	NSNumber *minimumMemoryByteCount = nil;
	id source = nil, license = nil, memoryFacts = nil;
	BOOL isDefault = NO;
	NFKModelElementReader sourceReader = ^id (id element, NSString *elementPath, NSError **elementError) {
		return [NFKModelSource sourceFromObject:element path:elementPath error:elementError];
	};
	NFKModelElementReader licenseReader = ^id (id element, NSString *elementPath, NSError **elementError) {
		return [NFKModelLicense licenseFromObject:element path:elementPath error:elementError];
	};
	NFKModelElementReader factsReader = ^id (id element, NSString *elementPath, NSError **elementError) {
		return [NFKModelMemoryFacts factsFromObject:element path:elementPath error:elementError];
	};
	if (reader == nil
		|| ![reader readString:@"id" required:YES into:&identifier error:error]
		|| ![reader readString:@"displayName" required:NO into:&displayName error:error]
		|| ![reader readBoolean:@"isDefault" into:&isDefault error:error]
		|| ![reader readString:@"registryName" required:NO into:&registryName error:error]
		|| ![reader readObject:@"source" withReader:sourceReader into:&source error:error]
		|| ![reader readNumber:@"downloadBytes" into:&downloadByteCount error:error]
		|| ![reader readNumber:@"residentBytes" into:&residentByteCount error:error]
		|| ![reader readNumber:@"peakBytes" into:&peakByteCount error:error]
		|| ![reader readNumber:@"parameterCount" into:&parameterCount error:error]
		|| ![reader readString:@"precision" required:NO into:&precision error:error]
		|| ![reader readNumber:@"minimumMemoryBytes" into:&minimumMemoryByteCount error:error]
		|| ![reader readObject:@"license" withReader:licenseReader into:&license error:error]
		|| ![reader readObject:@"memoryFacts" withReader:factsReader into:&memoryFacts error:error]) {
		return nil;
	}
	return [[self alloc] initWithIdentifier:identifier displayName:displayName isDefault:isDefault
							   registryName:registryName source:source downloadByteCount:downloadByteCount
						  residentByteCount:residentByteCount peakByteCount:peakByteCount
							 parameterCount:parameterCount precision:precision
					 minimumMemoryByteCount:minimumMemoryByteCount license:license memoryFacts:memoryFacts
								 extensions:[reader extensionsWithoutFields:NFKModelVariantFields()]];
}

- (NSDictionary<NSString *, id> *)dictionaryRepresentation
{
	NSMutableDictionary<NSString *, id> *dictionary = [self.extensions mutableCopy];
	dictionary[@"id"] = self.identifier;
	dictionary[@"displayName"] = self.displayName;
	dictionary[@"isDefault"] = @(self.isDefault);
	dictionary[@"registryName"] = self.registryName;
	dictionary[@"source"] = self.source.dictionaryRepresentation;
	dictionary[@"downloadBytes"] = self.downloadByteCount;
	dictionary[@"residentBytes"] = self.residentByteCount;
	dictionary[@"peakBytes"] = self.peakByteCount;
	dictionary[@"parameterCount"] = self.parameterCount;
	dictionary[@"precision"] = self.precision;
	dictionary[@"minimumMemoryBytes"] = self.minimumMemoryByteCount;
	dictionary[@"license"] = self.license.dictionaryRepresentation;
	dictionary[@"memoryFacts"] = self.memoryFacts.dictionaryRepresentation;
	return [dictionary copy];
}

- (id)copyWithZone:(nullable NSZone *)zone
{
	return self;
}

- (BOOL)isEqual:(id)object
{
	return NFKModelIsEqualDescription(self, object, NFKModelVariant.class);
}

- (NSUInteger)hash
{
	return self.identifier.hash;
}

@end

#pragma mark - NFKModelRequirements

static NSArray<NSString *> *NFKModelRequirementsFields(void)
{
	return @[ @"minimumOS", @"requiresAppleSilicon", @"minimumMemoryBytes", @"requiredFeatures", @"unmetReason" ];
}

static id _Nullable NFKModelReadRequirements(id element, NSString *elementPath, NSError **elementError)
{
	return [NFKModelRequirements requirementsFromObject:element path:elementPath error:elementError];
}

// Compares dotted version strings component by component, a missing component reading as 0.
static NSComparisonResult NFKModelCompareVersions(NSString *version, NSString *other)
{
	NSArray<NSString *> *first = [version componentsSeparatedByString:@"."];
	NSArray<NSString *> *second = [other componentsSeparatedByString:@"."];
	NSUInteger count = MAX(first.count, second.count);
	for (NSUInteger index = 0; index < count; index += 1) {
		NSInteger a = index < first.count ? first[index].integerValue : 0;
		NSInteger b = index < second.count ? second[index].integerValue : 0;
		if (a != b) {
			return a < b ? NSOrderedAscending : NSOrderedDescending;
		}
	}
	return NSOrderedSame;
}

@implementation NFKModelRequirements

- (instancetype)init
{
	return [self initWithMinimumOSVersions:nil requiresAppleSilicon:NO minimumMemoryByteCount:nil requiredFeatures:nil
							   unmetReason:nil extensions:nil];
}

- (instancetype)initWithMinimumOSVersions:(nullable NSDictionary<NSString *, NSString *> *)minimumOSVersions
					 requiresAppleSilicon:(BOOL)requiresAppleSilicon
				   minimumMemoryByteCount:(nullable NSNumber *)minimumMemoryByteCount
						 requiredFeatures:(nullable NSArray<NSString *> *)requiredFeatures
							  unmetReason:(nullable NSString *)unmetReason
							   extensions:(nullable NSDictionary<NSString *, id> *)extensions
{
	self = [super init];
	if (self != nil) {
		_minimumOSVersions = [minimumOSVersions copy] ?: @{};
		_requiresAppleSilicon = requiresAppleSilicon;
		_minimumMemoryByteCount = [minimumMemoryByteCount copy];
		_requiredFeatures = [requiredFeatures copy] ?: @[];
		_unmetReason = [unmetReason copy];
		_extensions = NFKModelExtensionsWithout(extensions, NFKModelRequirementsFields());
	}
	return self;
}

- (instancetype)initWithMinimumOSVersions:(nullable NSDictionary<NSString *, NSString *> *)minimumOSVersions
					 requiresAppleSilicon:(BOOL)requiresAppleSilicon
				   minimumMemoryByteCount:(nullable NSNumber *)minimumMemoryByteCount
							   extensions:(nullable NSDictionary<NSString *, id> *)extensions
{
	return [self initWithMinimumOSVersions:minimumOSVersions requiresAppleSilicon:requiresAppleSilicon
					minimumMemoryByteCount:minimumMemoryByteCount requiredFeatures:nil unmetReason:nil
								extensions:extensions];
}

- (nullable NSString *)problemForHardwareProfile:(NFKHardwareProfile *)profile
{
	NSString *problem = [self composedProblemForHardwareProfile:profile];
	if (problem == nil) {
		return nil;
	}
	return self.unmetReason ?: problem;
}

- (nullable NSString *)composedProblemForHardwareProfile:(NFKHardwareProfile *)profile
{
	NSString *platform = profile.operatingSystemName;
	if (self.minimumOSVersions.count > 0 && platform.length > 0) {
		NSString *minimum = self.minimumOSVersions[platform];
		if (minimum == nil) {
			return [NSString stringWithFormat:@"This does not run on %@.", platform];
		}
		if (NFKModelCompareVersions(profile.operatingSystemVersion, minimum) == NSOrderedAscending) {
			return [NSString stringWithFormat:@"This needs %@ %@ or later.", platform, minimum];
		}
	}
	BOOL needsAppleSilicon = self.requiresAppleSilicon || [self.requiredFeatures containsObject:NFKModelFeatureAppleSilicon];
	if (needsAppleSilicon && !profile.isAppleSilicon) {
		return @"This needs Apple silicon.";
	}
	for (NSString *feature in self.requiredFeatures) {
		NSString *problem = [self problemForFeature:feature profile:profile];
		if (problem != nil) {
			return problem;
		}
	}
	return nil;
}

- (nullable NSString *)problemForFeature:(NSString *)feature profile:(NFKHardwareProfile *)profile
{
	if ([feature isEqualToString:NFKModelFeatureAppleSilicon]) {
		return nil;
	}
	if ([feature isEqualToString:NFKModelFeatureNeuralAccelerators]) {
		return profile.hasNeuralAccelerators ? nil : @"This needs a GPU with neural accelerators.";
	}
	return [NSString stringWithFormat:@"This needs the %@ feature.", feature];
}

+ (nullable instancetype)requirementsWithDictionary:(NSDictionary<NSString *, id> *)dictionary error:(NSError **)error
{
	return [self requirementsFromObject:dictionary path:@"requirements" error:error];
}

+ (nullable instancetype)requirementsFromObject:(id)object path:(NSString *)path error:(NSError **)error
{
	NFKModelFieldReader *reader = [NFKModelFieldReader readerWithObject:object path:path error:error];
	NSDictionary<NSString *, NSString *> *minimumOSVersions = nil;
	NSNumber *minimumMemoryByteCount = nil;
	NSArray<NSString *> *requiredFeatures = nil;
	NSString *unmetReason = nil;
	BOOL requiresAppleSilicon = NO;
	if (reader == nil
		|| ![reader readStringDictionary:@"minimumOS" into:&minimumOSVersions error:error]
		|| ![reader readBoolean:@"requiresAppleSilicon" into:&requiresAppleSilicon error:error]
		|| ![reader readNumber:@"minimumMemoryBytes" into:&minimumMemoryByteCount error:error]
		|| ![reader readStringArray:@"requiredFeatures" into:&requiredFeatures error:error]
		|| ![reader readString:@"unmetReason" required:NO into:&unmetReason error:error]) {
		return nil;
	}
	return [[self alloc] initWithMinimumOSVersions:minimumOSVersions requiresAppleSilicon:requiresAppleSilicon
							minimumMemoryByteCount:minimumMemoryByteCount requiredFeatures:requiredFeatures
									   unmetReason:unmetReason
										extensions:[reader extensionsWithoutFields:NFKModelRequirementsFields()]];
}

- (NSDictionary<NSString *, id> *)dictionaryRepresentation
{
	NSMutableDictionary<NSString *, id> *dictionary = [self.extensions mutableCopy];
	dictionary[@"minimumOS"] = self.minimumOSVersions;
	dictionary[@"requiresAppleSilicon"] = @(self.requiresAppleSilicon);
	dictionary[@"minimumMemoryBytes"] = self.minimumMemoryByteCount;
	if (self.requiredFeatures.count > 0) {
		dictionary[@"requiredFeatures"] = self.requiredFeatures;
	}
	dictionary[@"unmetReason"] = self.unmetReason;
	return [dictionary copy];
}

- (id)copyWithZone:(nullable NSZone *)zone
{
	return self;
}

- (BOOL)isEqual:(id)object
{
	return NFKModelIsEqualDescription(self, object, NFKModelRequirements.class);
}

- (NSUInteger)hash
{
	return self.minimumOSVersions.hash ^ (NSUInteger)self.requiresAppleSilicon;
}

@end

#pragma mark - NFKModelDescriptor

static NSArray<NSString *> *NFKModelDescriptorFields(void)
{
	return @[ @"id", @"displayName", @"family", @"summary", @"engine", @"registryName", @"providerClass", @"tasks",
			  @"inputModalities", @"outputModalities", @"inputs", @"outputs", @"options", @"variants",
			  @"license", @"requirements", @"residencies", @"memoryModel", @"presets", @"standIn",
			  @"introducedVersion" ];
}

@interface NFKModelDescriptor ()
@property (nonatomic, readonly, nullable) NFKModelOption *derivedVariantOption;
@end

@implementation NFKModelDescriptor

- (instancetype)initWithIdentifier:(NSString *)identifier
					   displayName:(nullable NSString *)displayName
							family:(NSString *)family
						   summary:(nullable NSString *)summary
							engine:(NSString *)engine
					  registryName:(nullable NSString *)registryName
				 providerClassName:(nullable NSString *)providerClassName
							 tasks:(nullable NSArray<NSString *> *)tasks
				   inputModalities:(nullable NSArray<NSString *> *)inputModalities
				  outputModalities:(nullable NSArray<NSString *> *)outputModalities
							inputs:(nullable NSArray<NFKModelPort *> *)inputs
						   outputs:(nullable NSArray<NFKModelPort *> *)outputs
						   options:(nullable NSArray<NFKModelOption *> *)options
						  variants:(nullable NSArray<NFKModelVariant *> *)variants
						   license:(nullable NFKModelLicense *)license
					  requirements:(nullable NFKModelRequirements *)requirements
					   residencies:(nullable NSArray<NSString *> *)residencies
					   memoryModel:(nullable NFKModelMemoryModel *)memoryModel
						   presets:(nullable NSArray<NFKModelPreset *> *)presets
						   standIn:(BOOL)standIn
				 introducedVersion:(nullable NSString *)introducedVersion
						extensions:(nullable NSDictionary<NSString *, id> *)extensions
{
	self = [super init];
	if (self != nil) {
		_identifier = [identifier copy];
		_displayName = [displayName copy] ?: _identifier;
		_family = [family copy];
		_summary = [summary copy];
		_engine = [engine copy];
		_registryName = [registryName copy];
		_providerClassName = [providerClassName copy];
		_tasks = [tasks copy] ?: @[];
		_inputs = [inputs copy] ?: @[];
		_outputs = [outputs copy] ?: @[];
		_inputModalities = [inputModalities copy] ?: NFKModelValueTypesOfPorts(_inputs);
		_outputModalities = [outputModalities copy] ?: NFKModelValueTypesOfPorts(_outputs);
		_options = [options copy] ?: @[];
		_variants = [variants copy] ?: @[];
		_license = [license copy];
		_requirements = [requirements copy] ?: [[NFKModelRequirements alloc] init];
		_residencies = [residencies copy] ?: @[];
		_memoryModel = [memoryModel copy];
		_presets = [presets copy] ?: @[];
		_standIn = standIn;
		_introducedVersion = [introducedVersion copy];
		_extensions = NFKModelExtensionsWithout(extensions, NFKModelDescriptorFields());
		_derivedVariantOption = [self makeDerivedVariantOption];
	}
	return self;
}

- (instancetype)initWithIdentifier:(NSString *)identifier
					   displayName:(nullable NSString *)displayName
							family:(NSString *)family
						   summary:(nullable NSString *)summary
							engine:(NSString *)engine
					  registryName:(nullable NSString *)registryName
				 providerClassName:(nullable NSString *)providerClassName
							 tasks:(nullable NSArray<NSString *> *)tasks
				   inputModalities:(nullable NSArray<NSString *> *)inputModalities
				  outputModalities:(nullable NSArray<NSString *> *)outputModalities
							inputs:(nullable NSArray<NFKModelPort *> *)inputs
						   outputs:(nullable NSArray<NFKModelPort *> *)outputs
						   options:(nullable NSArray<NFKModelOption *> *)options
						  variants:(nullable NSArray<NFKModelVariant *> *)variants
						   license:(nullable NFKModelLicense *)license
					  requirements:(nullable NFKModelRequirements *)requirements
					   residencies:(nullable NSArray<NSString *> *)residencies
						   standIn:(BOOL)standIn
				 introducedVersion:(nullable NSString *)introducedVersion
						extensions:(nullable NSDictionary<NSString *, id> *)extensions
{
	return [self initWithIdentifier:identifier displayName:displayName family:family summary:summary engine:engine
					   registryName:registryName providerClassName:providerClassName tasks:tasks
					inputModalities:inputModalities outputModalities:outputModalities inputs:inputs outputs:outputs
							options:options variants:variants license:license requirements:requirements
						residencies:residencies memoryModel:nil presets:nil standIn:standIn
				  introducedVersion:introducedVersion extensions:extensions];
}

// The variant choice a descriptor with several variants offers when it declares none of its own.
- (nullable NFKModelOption *)makeDerivedVariantOption
{
	if (self.variants.count < 2) {
		return nil;
	}
	for (NFKModelOption *option in self.options) {
		if ([option.key isEqualToString:NFKModelLoadOptionVariant]) {
			return nil;
		}
	}
	NSMutableArray<NFKModelOptionChoice *> *choices = [NSMutableArray arrayWithCapacity:self.variants.count];
	for (NFKModelVariant *variant in self.variants) {
		NFKModelMemoryFigure *figure = nil;
		if (variant.residentByteCount != nil) {
			figure = [[NFKModelMemoryFigure alloc] initWithResidentByteCount:variant.residentByteCount.integerValue
															   peakByteCount:(variant.peakByteCount ?: variant.residentByteCount).integerValue];
		}
		[choices addObject:[[NFKModelOptionChoice alloc] initWithValue:variant.identifier title:variant.displayName
														  requirements:nil memory:figure memoryByVariant:nil
															extensions:nil]];
	}
	return [[NFKModelOption alloc] initWithKey:NFKModelLoadOptionVariant kind:NFKModelOptionKindChoice title:@"Variant"
										  help:@"The release of the model to load." defaultValue:self.defaultVariant.identifier
									   minimum:nil maximum:nil step:nil choices:choices stage:NFKModelOptionStageLoad
										 group:nil requirements:nil propertyKeyPath:nil memoryBytesPerUnit:nil
									extensions:nil];
}

- (NSArray<NFKModelOption *> *)loadOptions
{
	NSMutableArray<NFKModelOption *> *options = [NSMutableArray array];
	if (self.derivedVariantOption != nil) {
		[options addObject:self.derivedVariantOption];
	}
	for (NFKModelOption *option in self.options) {
		if ([option.stage isEqualToString:NFKModelOptionStageLoad]) {
			[options addObject:option];
		}
	}
	return [options copy];
}

- (NSArray<NFKModelOption *> *)requestOptions
{
	NSMutableArray<NFKModelOption *> *options = [NSMutableArray array];
	for (NFKModelOption *option in self.options) {
		if ([option.stage isEqualToString:NFKModelOptionStageRequest]) {
			[options addObject:option];
		}
	}
	return [options copy];
}

- (nullable NFKModelOption *)optionWithKey:(NSString *)key
{
	for (NFKModelOption *option in self.options) {
		if ([option.key isEqualToString:key]) {
			return option;
		}
	}
	if ([key isEqualToString:NFKModelLoadOptionVariant]) {
		return self.derivedVariantOption;
	}
	return nil;
}

- (nullable NFKModelPreset *)presetWithIdentifier:(NSString *)identifier
{
	for (NFKModelPreset *preset in self.presets) {
		if ([preset.identifier isEqualToString:identifier]) {
			return preset;
		}
	}
	return nil;
}

+ (instancetype)descriptorWithIdentifier:(NSString *)identifier derivedFromBackend:(id<NFKInferenceBackend>)backend
{
	NSMutableArray<NFKModelPort *> *inputs = [NSMutableArray array];
	if ([backend respondsToSelector:@selector(supportedInputKeys)]) {
		for (NSString *key in NFKModelSortedKeys(backend.supportedInputKeys)) {
			[inputs addObject:[NFKModelPort portWithKey:key required:NO]];
		}
	}
	NSMutableArray<NFKModelPort *> *outputs = [NSMutableArray array];
	if ([backend respondsToSelector:@selector(supportedOutputKeys)]) {
		for (NSString *key in NFKModelSortedKeys(backend.supportedOutputKeys)) {
			[outputs addObject:[NFKModelPort outputPortWithKey:key required:NO]];
		}
	}
	NSMutableArray<NFKModelOption *> *options = [NSMutableArray array];
	if ([backend respondsToSelector:@selector(supportedParameterKeys)]) {
		for (NSString *key in NFKModelSortedKeys(backend.supportedParameterKeys)) {
			[options addObject:[NFKModelOption optionWithKey:key]];
		}
	}
	return [[self alloc] initWithIdentifier:identifier displayName:nil family:[self familyOfBackend:backend]
									summary:nil engine:NFKModelEnginePlugin registryName:nil providerClassName:nil
									  tasks:nil inputModalities:nil outputModalities:nil inputs:inputs
									outputs:outputs options:options variants:nil license:nil
							   requirements:nil residencies:nil standIn:NO introducedVersion:nil extensions:nil];
}

+ (NSString *)familyOfBackend:(id<NFKInferenceBackend>)backend
{
	if ([backend respondsToSelector:@selector(modelInfo)]) {
		id architecture = backend.modelInfo[NFKModelInfoArchitecture];
		if ([architecture isKindOfClass:NSString.class] && [architecture length] > 0) {
			return architecture;
		}
	}
	return backend.backendIdentifier;
}

+ (nullable instancetype)descriptorWithDictionary:(NSDictionary<NSString *, id> *)dictionary error:(NSError **)error
{
	return [self descriptorFromObject:dictionary path:@"model" error:error];
}

+ (nullable instancetype)descriptorFromObject:(id)object path:(NSString *)path error:(NSError **)error
{
	NFKModelFieldReader *reader = [NFKModelFieldReader readerWithObject:object path:path error:error];
	NSString *identifier = nil, *displayName = nil, *family = nil, *summary = nil, *engine = nil;
	NSString *registryName = nil, *providerClassName = nil, *introducedVersion = nil;
	NSArray<NSString *> *tasks = nil, *inputModalities = nil, *outputModalities = nil, *residencies = nil;
	NSArray *inputs = nil, *outputs = nil, *options = nil, *variants = nil, *presets = nil;
	id license = nil, requirements = nil, memoryModel = nil;
	BOOL standIn = NO;
	NFKModelElementReader inputReader = ^id (id element, NSString *elementPath, NSError **elementError) {
		return [NFKModelPort portFromObject:element path:elementPath output:NO error:elementError];
	};
	NFKModelElementReader outputReader = ^id (id element, NSString *elementPath, NSError **elementError) {
		return [NFKModelPort portFromObject:element path:elementPath output:YES error:elementError];
	};
	NFKModelElementReader optionReader = ^id (id element, NSString *elementPath, NSError **elementError) {
		return [NFKModelOption optionFromObject:element path:elementPath error:elementError];
	};
	NFKModelElementReader variantReader = ^id (id element, NSString *elementPath, NSError **elementError) {
		return [NFKModelVariant variantFromObject:element path:elementPath error:elementError];
	};
	NFKModelElementReader licenseReader = ^id (id element, NSString *elementPath, NSError **elementError) {
		return [NFKModelLicense licenseFromObject:element path:elementPath error:elementError];
	};
	NFKModelElementReader requirementsReader = ^id (id element, NSString *elementPath, NSError **elementError) {
		return [NFKModelRequirements requirementsFromObject:element path:elementPath error:elementError];
	};
	NFKModelElementReader memoryModelReader = ^id (id element, NSString *elementPath, NSError **elementError) {
		return [NFKModelMemoryModel memoryModelFromObject:element path:elementPath error:elementError];
	};
	NFKModelElementReader presetReader = ^id (id element, NSString *elementPath, NSError **elementError) {
		return [NFKModelPreset presetFromObject:element path:elementPath error:elementError];
	};
	if (reader == nil
		|| ![reader readString:@"id" required:YES into:&identifier error:error]
		|| ![reader readString:@"displayName" required:NO into:&displayName error:error]
		|| ![reader readString:@"family" required:YES into:&family error:error]
		|| ![reader readString:@"summary" required:NO into:&summary error:error]
		|| ![reader readString:@"engine" required:YES into:&engine error:error]
		|| ![reader readString:@"registryName" required:NO into:&registryName error:error]
		|| ![reader readString:@"providerClass" required:NO into:&providerClassName error:error]
		|| ![reader readStringArray:@"tasks" into:&tasks error:error]
		|| ![reader readStringArray:@"inputModalities" into:&inputModalities error:error]
		|| ![reader readStringArray:@"outputModalities" into:&outputModalities error:error]
		|| ![reader readObjects:@"inputs" withReader:inputReader into:&inputs error:error]
		|| ![reader readObjects:@"outputs" withReader:outputReader into:&outputs error:error]
		|| ![reader readObjects:@"options" withReader:optionReader into:&options error:error]
		|| ![reader readObjects:@"variants" withReader:variantReader into:&variants error:error]
		|| ![reader readObject:@"license" withReader:licenseReader into:&license error:error]
		|| ![reader readObject:@"requirements" withReader:requirementsReader into:&requirements error:error]
		|| ![reader readStringArray:@"residencies" into:&residencies error:error]
		|| ![reader readObject:@"memoryModel" withReader:memoryModelReader into:&memoryModel error:error]
		|| ![reader readObjects:@"presets" withReader:presetReader into:&presets error:error]
		|| ![reader readBoolean:@"standIn" into:&standIn error:error]
		|| ![reader readString:@"introducedVersion" required:NO into:&introducedVersion error:error]
		|| ![reader requireUniqueElements:inputs ofField:@"inputs" keyPath:@"key" name:@"key" error:error]
		|| ![reader requireUniqueElements:outputs ofField:@"outputs" keyPath:@"key" name:@"key" error:error]
		|| ![reader requireUniqueElements:options ofField:@"options" keyPath:@"key" name:@"key" error:error]
		|| ![reader requireUniqueElements:variants ofField:@"variants" keyPath:@"identifier" name:@"id" error:error]
		|| ![reader requireUniqueElements:presets ofField:@"presets" keyPath:@"identifier" name:@"id" error:error]) {
		return nil;
	}
	return [[self alloc] initWithIdentifier:identifier displayName:displayName family:family summary:summary
									 engine:engine registryName:registryName providerClassName:providerClassName
									  tasks:tasks inputModalities:inputModalities outputModalities:outputModalities
									 inputs:inputs outputs:outputs options:options variants:variants
									license:license requirements:requirements residencies:residencies
								memoryModel:memoryModel presets:presets standIn:standIn
						  introducedVersion:introducedVersion
								 extensions:[reader extensionsWithoutFields:NFKModelDescriptorFields()]];
}

- (nullable NFKModelVariant *)defaultVariant
{
	for (NFKModelVariant *variant in self.variants) {
		if (variant.isDefault) {
			return variant;
		}
	}
	return self.variants.firstObject;
}

- (nullable NFKModelVariant *)variantWithIdentifier:(NSString *)identifier
{
	for (NFKModelVariant *variant in self.variants) {
		if ([variant.identifier isEqualToString:identifier]) {
			return variant;
		}
	}
	return nil;
}

- (nullable NFKModelLicense *)licenseForVariant:(NFKModelVariant *)variant
{
	return variant.license ?: self.license;
}

- (NSDictionary<NSString *, id> *)dictionaryRepresentation
{
	NSMutableDictionary<NSString *, id> *dictionary = [self.extensions mutableCopy];
	dictionary[@"id"] = self.identifier;
	dictionary[@"displayName"] = self.displayName;
	dictionary[@"family"] = self.family;
	dictionary[@"summary"] = self.summary;
	dictionary[@"engine"] = self.engine;
	dictionary[@"registryName"] = self.registryName;
	dictionary[@"providerClass"] = self.providerClassName;
	dictionary[@"tasks"] = self.tasks;
	dictionary[@"inputModalities"] = self.inputModalities;
	dictionary[@"outputModalities"] = self.outputModalities;
	dictionary[@"inputs"] = NFKModelDictionariesOf(self.inputs);
	dictionary[@"outputs"] = NFKModelDictionariesOf(self.outputs);
	dictionary[@"options"] = NFKModelDictionariesOf(self.options);
	dictionary[@"variants"] = NFKModelDictionariesOf(self.variants);
	dictionary[@"license"] = self.license.dictionaryRepresentation;
	dictionary[@"requirements"] = self.requirements.dictionaryRepresentation;
	dictionary[@"residencies"] = self.residencies;
	dictionary[@"memoryModel"] = self.memoryModel.dictionaryRepresentation;
	if (self.presets.count > 0) {
		dictionary[@"presets"] = NFKModelDictionariesOf(self.presets);
	}
	dictionary[@"standIn"] = @(self.isStandIn);
	dictionary[@"introducedVersion"] = self.introducedVersion;
	return [dictionary copy];
}

- (id)copyWithZone:(nullable NSZone *)zone
{
	return self;
}

- (BOOL)isEqual:(id)object
{
	return NFKModelIsEqualDescription(self, object, NFKModelDescriptor.class);
}

- (NSUInteger)hash
{
	return self.identifier.hash;
}

- (NSString *)description
{
	return [NSString stringWithFormat:@"<%@ %@ %@ %@>", NSStringFromClass(self.class), self.identifier, self.engine,
			self.tasks];
}

@end
