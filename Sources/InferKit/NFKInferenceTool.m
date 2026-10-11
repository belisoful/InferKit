//
//  NFKInferenceTool.m
//  InferKit
//

#import "NFKInferenceTool.h"

@implementation NFKToolResult

+ (instancetype)resultWithText:(NSString *)text
{
	return [[self alloc] initWithText:text outputs:nil imagePNGs:nil isError:NO];
}

+ (instancetype)resultWithText:(NSString *)text
					   outputs:(nullable NSDictionary<NSString *, id> *)outputs
					 imagePNGs:(nullable NSArray<NSData *> *)imagePNGs
{
	return [[self alloc] initWithText:text outputs:outputs imagePNGs:imagePNGs isError:NO];
}

+ (instancetype)errorResultWithText:(NSString *)text
{
	return [[self alloc] initWithText:text outputs:nil imagePNGs:nil isError:YES];
}

- (instancetype)initWithText:(NSString *)text
					 outputs:(nullable NSDictionary<NSString *, id> *)outputs
				   imagePNGs:(nullable NSArray<NSData *> *)imagePNGs
					 isError:(BOOL)isError
{
	self = [super init];
	if (self != nil) {
		_text = [text copy];
		_outputs = [outputs copy] ?: @{};
		_imagePNGs = [imagePNGs copy] ?: @[];
		_isError = isError;
	}
	return self;
}

- (NSString *)description
{
	return [NSString stringWithFormat:@"<%@ %@%@>", NSStringFromClass(self.class), self.isError ? @"error: " : @"", self.text];
}

@end

@interface NFKInferenceTool ()
@property (nonatomic, copy, nullable) NFKToolHandler handler;
@end

@implementation NFKInferenceTool

+ (instancetype)toolWithName:(NSString *)name
				 description:(NSString *)toolDescription
				  parameters:(NSDictionary<NSString *, id> *)parameters
					 handler:(NFKToolHandler)handler
{
	NFKInferenceTool *tool = [[self alloc] initWithName:name description:toolDescription parameters:parameters];
	tool.handler = handler;
	return tool;
}

- (instancetype)initWithName:(NSString *)name
				 description:(NSString *)toolDescription
				  parameters:(NSDictionary<NSString *, id> *)parameters
{
	self = [super init];
	if (self != nil) {
		_name = [name copy];
		_toolDescription = [toolDescription copy];
		_parameters = [parameters copy];
	}
	return self;
}

- (NSDictionary<NSString *, id> *)declaration
{
	return @{ @"name": self.name, @"description": self.toolDescription, @"parameters": self.parameters };
}

- (nullable NFKToolResult *)resultForArguments:(NSDictionary<NSString *, id> *)arguments error:(NSError **)error
{
	NFKToolHandler handler = self.handler;
	if (handler == nil) {
		return [NFKToolResult errorResultWithText:[NSString stringWithFormat:@"the tool %@ has nothing to run", self.name]];
	}
	return handler(arguments, error);
}

- (NFKToolResult *)invokeWithArguments:(NSDictionary<NSString *, id> *)arguments
{
	NSArray<NSString *> *problems = [self problemsWithArguments:arguments];
	if (problems.count > 0) {
		return [NFKToolResult errorResultWithText:[NSString stringWithFormat:@"The call to %@ was not run: %@.",
												   self.name, [problems componentsJoinedByString:@"; "]]];
	}
	NSError *error = nil;
	NFKToolResult *result = [self resultForArguments:arguments error:&error];
	if (result != nil) {
		return result;
	}
	NSString *reason = error.localizedDescription ?: @"the tool reported no reason";
	return [NFKToolResult errorResultWithText:[NSString stringWithFormat:@"%@ failed: %@", self.name, reason]];
}

- (NSArray<NSString *> *)problemsWithArguments:(NSDictionary<NSString *, id> *)arguments
{
	NSMutableArray<NSString *> *problems = [NSMutableArray array];
	if (![arguments isKindOfClass:NSDictionary.class]) {
		[problems addObject:@"the arguments are not a JSON object"];
		return problems;
	}
	NSArray *required = [self.parameters[@"required"] isKindOfClass:NSArray.class] ? self.parameters[@"required"] : @[];
	for (NSString *name in required) {
		if ([name isKindOfClass:NSString.class] && (arguments[name] == nil || arguments[name] == NSNull.null)) {
			[problems addObject:[NSString stringWithFormat:@"the required argument \"%@\" is missing", name]];
		}
	}
	NSDictionary *properties = [self.parameters[@"properties"] isKindOfClass:NSDictionary.class] ? self.parameters[@"properties"] : @{};
	for (NSString *name in arguments) {
		NSDictionary *property = [properties[name] isKindOfClass:NSDictionary.class] ? properties[name] : nil;
		NSString *type = [property[@"type"] isKindOfClass:NSString.class] ? property[@"type"] : nil;
		if (type == nil || arguments[name] == NSNull.null) {
			continue;
		}
		if (![self value:arguments[name] matchesJSONType:type]) {
			[problems addObject:[NSString stringWithFormat:@"the argument \"%@\" must be a JSON %@", name, type]];
		}
	}
	return problems;
}

// A JSON boolean decodes as an NSNumber too, so the boolean and number checks tell them apart by the
// number's own type.
- (BOOL)value:(id)value matchesJSONType:(NSString *)type
{
	BOOL isBoolean = [value isKindOfClass:NSNumber.class] && CFGetTypeID((__bridge CFTypeRef)value) == CFBooleanGetTypeID();
	if ([type isEqualToString:@"string"]) {
		return [value isKindOfClass:NSString.class];
	}
	if ([type isEqualToString:@"boolean"]) {
		return isBoolean;
	}
	if ([type isEqualToString:@"number"]) {
		return [value isKindOfClass:NSNumber.class] && !isBoolean;
	}
	if ([type isEqualToString:@"integer"]) {
		return [value isKindOfClass:NSNumber.class] && !isBoolean && [value doubleValue] == floor([value doubleValue]);
	}
	if ([type isEqualToString:@"object"]) {
		return [value isKindOfClass:NSDictionary.class];
	}
	if ([type isEqualToString:@"array"]) {
		return [value isKindOfClass:NSArray.class];
	}
	return YES;
}

- (NSString *)description
{
	return [NSString stringWithFormat:@"<%@ %@>", NSStringFromClass(self.class), self.name];
}

@end
