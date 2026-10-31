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

@implementation NFKToolCaller

- (instancetype)initWithSessionIdentifier:(nullable NSString *)sessionIdentifier
							   clientName:(nullable NSString *)clientName
							clientVersion:(nullable NSString *)clientVersion
								  keyName:(nullable NSString *)keyName
							clientAddress:(NSString *)clientAddress
								  headers:(nullable NSDictionary<NSString *, NSString *> *)headers
{
	self = [super init];
	if (self != nil) {
		_sessionIdentifier = [sessionIdentifier copy];
		_clientName = [clientName copy];
		_clientVersion = [clientVersion copy];
		_keyName = [keyName copy];
		_clientAddress = [clientAddress copy];
		NSMutableDictionary<NSString *, NSString *> *lowercased = [NSMutableDictionary dictionaryWithCapacity:headers.count];
		[headers enumerateKeysAndObjectsUsingBlock:^(NSString *name, NSString *value, BOOL *stop) {
			lowercased[name.lowercaseString] = value;
		}];
		_headers = [lowercased copy];
	}
	return self;
}

- (nullable NSString *)valueForHeader:(NSString *)name
{
	return self.headers[name.lowercaseString];
}

- (NSString *)description
{
	return [NSString stringWithFormat:@"<%@ %@ %@ session %@>", NSStringFromClass(self.class), self.clientName ?: @"(unnamed)",
			self.clientAddress, self.sessionIdentifier ?: @"(none)"];
}

@end

@interface NFKInferenceTool ()
@property (nonatomic, copy, nullable) NFKToolHandler handler;
@property (nonatomic, copy, nullable) NFKToolCallerHandler callerHandler;
@property (nonatomic, copy, nullable) NFKToolStartHandler startHandler;
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

+ (instancetype)toolWithName:(NSString *)name
				 description:(NSString *)toolDescription
				  parameters:(NSDictionary<NSString *, id> *)parameters
			   callerHandler:(NFKToolCallerHandler)handler
{
	NFKInferenceTool *tool = [[self alloc] initWithName:name description:toolDescription parameters:parameters];
	tool.callerHandler = handler;
	return tool;
}

+ (instancetype)toolWithName:(NSString *)name
				 description:(NSString *)toolDescription
				  parameters:(NSDictionary<NSString *, id> *)parameters
				startHandler:(NFKToolStartHandler)handler
{
	NFKInferenceTool *tool = [[self alloc] initWithName:name description:toolDescription parameters:parameters];
	tool.startHandler = handler;
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
	if (self.callerHandler != nil || self.startHandler != nil) {
		return [self resultForArguments:arguments caller:nil error:error];
	}
	NFKToolHandler handler = self.handler;
	if (handler == nil) {
		return [NFKToolResult errorResultWithText:[NSString stringWithFormat:@"the tool %@ has nothing to run", self.name]];
	}
	return handler(arguments, error);
}

- (nullable NFKToolResult *)resultForArguments:(NSDictionary<NSString *, id> *)arguments
										caller:(nullable NFKToolCaller *)caller
										 error:(NSError **)error
{
	NFKToolCallerHandler callerHandler = self.callerHandler;
	if (callerHandler != nil) {
		return callerHandler(arguments, caller, error);
	}
	NFKToolStartHandler startHandler = self.startHandler;
	if (startHandler == nil) {
		return [self resultForArguments:arguments error:error];
	}
	dispatch_semaphore_t finished = dispatch_semaphore_create(0);
	__block NFKToolResult *result = nil;
	__block BOOL completed = NO;
	startHandler(arguments, caller, ^(NFKToolResult *finalResult) {
		@synchronized (finished) {
			if (completed) {
				return;
			}
			completed = YES;
			result = finalResult;
		}
		dispatch_semaphore_signal(finished);
	});
	dispatch_semaphore_wait(finished, DISPATCH_TIME_FOREVER);
	return result;
}

- (NFKToolResult *)invokeWithArguments:(NSDictionary<NSString *, id> *)arguments
{
	return [self invokeWithArguments:arguments caller:nil];
}

- (NFKToolResult *)invokeWithArguments:(NSDictionary<NSString *, id> *)arguments caller:(nullable NFKToolCaller *)caller
{
	NFKToolResult *refusal = [self refusalOfArguments:arguments];
	if (refusal != nil) {
		return refusal;
	}
	NSError *error = nil;
	NFKToolResult *result = [self resultForArguments:arguments caller:caller error:&error];
	if (result != nil) {
		return result;
	}
	NSString *reason = error.localizedDescription ?: @"the tool reported no reason";
	return [NFKToolResult errorResultWithText:[NSString stringWithFormat:@"%@ failed: %@", self.name, reason]];
}

- (nullable dispatch_block_t)startWithArguments:(NSDictionary<NSString *, id> *)arguments
										 caller:(nullable NFKToolCaller *)caller
									 completion:(void (^)(NFKToolResult *result))completion
{
	NFKToolStartHandler startHandler = self.startHandler;
	if (startHandler == nil) {
		completion([self invokeWithArguments:arguments caller:caller]);
		return nil;
	}
	NFKToolResult *refusal = [self refusalOfArguments:arguments];
	if (refusal != nil) {
		completion(refusal);
		return nil;
	}
	return startHandler(arguments, caller, completion);
}

/*! The error result of arguments that fail the schema check, or nil. */
- (nullable NFKToolResult *)refusalOfArguments:(NSDictionary<NSString *, id> *)arguments
{
	NSArray<NSString *> *problems = [self problemsWithArguments:arguments];
	if (problems.count == 0) {
		return nil;
	}
	return [NFKToolResult errorResultWithText:[NSString stringWithFormat:@"The call to %@ was not run: %@.",
											   self.name, [problems componentsJoinedByString:@"; "]]];
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
