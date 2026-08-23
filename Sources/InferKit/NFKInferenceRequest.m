//
//  NFKInferenceRequest.m
//  InferKit
//

#import "NFKInferenceRequest.h"
#import "NFKInferenceKeys.h"
#import "NFK_ARC.h"

@implementation NFKInferenceRequest

+ (instancetype)requestWithInputs:(NSDictionary<NSString *, id> *)inputs
					   parameters:(nullable NSDictionary<NSString *, id> *)parameters
				   outputModality:(NFKModality)outputModality
{
	return NARC_AUTORELEASE([[self alloc] initWithInputs:inputs parameters:parameters outputModality:outputModality]);
}

+ (instancetype)requestWithInputs:(NSDictionary<NSString *, id> *)inputs
					   parameters:(nullable NSDictionary<NSString *, id> *)parameters
{
	return [self requestWithInputs:inputs parameters:parameters outputModality:NFKModalityImage];
}

+ (instancetype)requestWithInputs:(NSDictionary<NSString *, id> *)inputs
{
	return [self requestWithInputs:inputs parameters:nil outputModality:NFKModalityImage];
}

- (instancetype)initWithInputs:(NSDictionary<NSString *, id> *)inputs
					parameters:(nullable NSDictionary<NSString *, id> *)parameters
{
	return [self initWithInputs:inputs parameters:parameters outputModality:NFKModalityImage];
}

- (instancetype)initWithInputs:(NSDictionary<NSString *, id> *)inputs
					parameters:(nullable NSDictionary<NSString *, id> *)parameters
				outputModality:(NFKModality)outputModality
{
	self = [super init];
	if (self != nil) {
		_inputs = [(inputs ?: @{}) copy];
		_parameters = [(parameters ?: @{}) copy];
		_outputModality = outputModality;
	}
	return self;
}

- (void)dealloc
{
	NARC_RELEASE(_inputs);
	NARC_RELEASE(_parameters);
	SUPER_DEALLOC();
}

- (nullable id)inputForKey:(NSString *)key
{
	return key != nil ? _inputs[key] : nil;
}

- (nullable id)parameterForKey:(NSString *)key
{
	return key != nil ? _parameters[key] : nil;
}

- (nullable NSString *)prompt
{
	id value = _inputs[NFKInputPrompt];
	return [value isKindOfClass:NSString.class] ? value : nil;
}

- (nullable NSString *)negativePrompt
{
	id value = _inputs[NFKInputNegativePrompt];
	return [value isKindOfClass:NSString.class] ? value : nil;
}

- (nullable NSArray<NSDictionary<NSString *, id> *> *)messages
{
	id value = _inputs[NFKInputMessages];
	return [value isKindOfClass:NSArray.class] ? value : nil;
}

- (id)copyWithZone:(NSZone *)zone
{
	// Immutable.
	return NARC_RETAIN(self);
}

- (BOOL)isEqual:(id)object
{
	if (self == object) {
		return YES;
	}
	if (![object isKindOfClass:NFKInferenceRequest.class]) {
		return NO;
	}
	NFKInferenceRequest *other = object;
	return other->_outputModality == _outputModality
		&& [other->_inputs isEqualToDictionary:_inputs]
		&& [other->_parameters isEqualToDictionary:_parameters];
}

- (NSUInteger)hash
{
	return _inputs.hash ^ _parameters.hash ^ (NSUInteger)_outputModality;
}

@end
