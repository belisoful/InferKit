//
//  NFKServedRequestChecks.m
//  InferKit
//

#import "NFKServedRequestChecks.h"
#import <InferKit/NFKInferenceServer.h>
#import <InferKit/NFKModelVocabulary.h>
#import <InferKit/NFKErrors.h>

NSError *NFKServedInvalidRequest(NSString *reason)
{
	return [NSError errorWithDomain:NFKInferenceServerErrorDomain code:NFKInferenceServerErrorInvalidRequest
						   userInfo:@{ NSLocalizedDescriptionKey: reason }];
}

BOOL NFKServedIsBoolean(id _Nullable value)
{
	return value != nil && CFGetTypeID((__bridge CFTypeRef)value) == CFBooleanGetTypeID();
}

static BOOL NFKServedIsNumber(id _Nullable value)
{
	return [value isKindOfClass:NSNumber.class] && !NFKServedIsBoolean(value) && isfinite([value doubleValue]);
}

static BOOL NFKServedIsInteger(id _Nullable value)
{
	if (!NFKServedIsNumber(value)) {
		return NO;
	}
	double number = [value doubleValue];
	return number == trunc(number) && fabs(number) <= 9223372036854775807.0;
}

static BOOL NFKServedValueIsOfOneType(id _Nullable value, NSString *type)
{
	if ([type isEqualToString:@"boolean"]) {
		return NFKServedIsBoolean(value);
	}
	if ([type isEqualToString:@"string"]) {
		return [value isKindOfClass:NSString.class];
	}
	if ([type isEqualToString:@"number"]) {
		return NFKServedIsNumber(value);
	}
	if ([type isEqualToString:@"integer"]) {
		return NFKServedIsInteger(value);
	}
	if ([type isEqualToString:@"object"]) {
		return [value isKindOfClass:NSDictionary.class];
	}
	if ([type isEqualToString:@"array"]) {
		return [value isKindOfClass:NSArray.class];
	}
	return [type isEqualToString:@"null"] && value == NSNull.null;
}

BOOL NFKServedValueIsOfType(id _Nullable value, NSString *type)
{
	for (NSString *each in [type componentsSeparatedByString:@"|"]) {
		if (NFKServedValueIsOfOneType(value, each)) {
			return YES;
		}
	}
	return NO;
}

/*! A type as the reason names it: "a boolean", "a string or an array". */
static NSString *NFKServedTypeDescription(NSString *type)
{
	NSDictionary<NSString *, NSString *> *names = @{ @"boolean": @"a boolean", @"string": @"a string", @"number": @"a number",
													  @"integer": @"an integer", @"object": @"an object", @"array": @"an array",
													  @"null": @"null" };
	NSMutableArray<NSString *> *described = [NSMutableArray array];
	for (NSString *each in [type componentsSeparatedByString:@"|"]) {
		[described addObject:names[each] ?: each];
	}
	return [described componentsJoinedByString:@" or "];
}

NSError * _Nullable NFKServedValueTypeError(id _Nullable value, NSString *type, NSString *path)
{
	if (value == nil || value == NSNull.null || NFKServedValueIsOfType(value, type)) {
		return nil;
	}
	return NFKServedInvalidRequest([NSString stringWithFormat:@"%@ must be %@", path, NFKServedTypeDescription(type)]);
}

NSError * _Nullable NFKServedFieldTypeError(id _Nullable object, NSDictionary<NSString *, NSString *> *types, NSString * _Nullable path)
{
	if (object == nil || object == NSNull.null) {
		return nil;
	}
	if (![object isKindOfClass:NSDictionary.class]) {
		return NFKServedInvalidRequest([NSString stringWithFormat:@"%@ must be an object", path ?: @"the body"]);
	}
	for (NSString *name in [types.allKeys sortedArrayUsingSelector:@selector(compare:)]) {
		NSString *named = path.length > 0 ? [NSString stringWithFormat:@"%@.%@", path, name] : name;
		NSError *error = NFKServedValueTypeError(((NSDictionary *)object)[name], types[name], named);
		if (error != nil) {
			return error;
		}
	}
	return nil;
}

/*! The field type a contract value type is checked as, or nil for one a route decodes itself. */
static NSString * _Nullable NFKServedFieldTypeForValueType(NSString * _Nullable valueType)
{
	NSDictionary<NSString *, NSString *> *types = @{ NFKModelValueTypeText: @"string", NFKModelValueTypeNumber: @"number",
													  NFKModelValueTypeInteger: @"integer", NFKModelValueTypeBoolean: @"boolean",
													  NFKModelValueTypeTextList: @"array", NFKModelValueTypeMessages: @"array" };
	return valueType != nil ? types[valueType] : nil;
}

/*! The refusal of one contract value named role "key", or nil. */
static NSError * _Nullable NFKServedContractValueError(NSString *role, NSString *key, id value)
{
	NSString *valueType = NFKModelValueTypeForKey(key);
	NSString *fieldType = NFKServedFieldTypeForValueType(valueType);
	if (fieldType == nil) {
		return nil;
	}
	NSString *path = [NSString stringWithFormat:@"the %@ \"%@\"", role, key];
	NSError *error = NFKServedValueTypeError(value, fieldType, path);
	if (error != nil || ![value isKindOfClass:NSArray.class]) {
		return error;
	}
	BOOL messages = [valueType isEqualToString:NFKModelValueTypeMessages];
	for (id element in (NSArray *)value) {
		BOOL fits = messages ? [element isKindOfClass:NSDictionary.class] : [element isKindOfClass:NSString.class];
		if (!fits) {
			return NFKServedInvalidRequest([NSString stringWithFormat:@"%@ must be %@", path,
											messages ? @"an array of message objects" : @"an array of strings"]);
		}
	}
	return nil;
}

NSDictionary *NFKServedWithoutNulls(NSDictionary *object)
{
	NSSet *nulls = [object keysOfEntriesPassingTest:^BOOL(id key, id value, BOOL *stop) {
		return value == NSNull.null;
	}];
	if (nulls.count == 0) {
		return object;
	}
	NSMutableDictionary *kept = [object mutableCopy];
	[kept removeObjectsForKeys:nulls.allObjects];
	return kept;
}

/*! entries without the null values under contract keys. */
static NSDictionary<NSString *, id> *NFKServedWithoutNullContractValues(NSDictionary<NSString *, id> *entries)
{
	NSSet *nulls = [entries keysOfEntriesPassingTest:^BOOL(NSString *key, id value, BOOL *stop) {
		return value == NSNull.null && NFKModelValueTypeForKey(key) != nil;
	}];
	if (nulls.count == 0) {
		return entries;
	}
	NSMutableDictionary *kept = [entries mutableCopy];
	[kept removeObjectsForKeys:nulls.allObjects];
	return kept;
}

NFKInferenceRequest *NFKServedRequestWithoutNullContractValues(NFKInferenceRequest *request)
{
	NSDictionary *inputs = NFKServedWithoutNullContractValues(request.inputs);
	NSDictionary *parameters = NFKServedWithoutNullContractValues(request.parameters);
	if (inputs == request.inputs && parameters == request.parameters) {
		return request;
	}
	return [[NFKInferenceRequest alloc] initWithInputs:inputs parameters:parameters outputModality:request.outputModality];
}

NSError * _Nullable NFKServedContractTypeError(NSDictionary<NSString *, id> *inputs, NSDictionary<NSString *, id> *parameters)
{
	for (NSString *key in [inputs.allKeys sortedArrayUsingSelector:@selector(compare:)]) {
		NSError *error = NFKServedContractValueError(@"input", key, inputs[key]);
		if (error != nil) {
			return error;
		}
	}
	for (NSString *key in [parameters.allKeys sortedArrayUsingSelector:@selector(compare:)]) {
		NSError *error = NFKServedContractValueError(@"parameter", key, parameters[key]);
		if (error != nil) {
			return error;
		}
	}
	return nil;
}

id _Nullable NFKServedFormValue(NSString *name, NSString *text, NSError * _Nullable * _Nullable error)
{
	NSString *valueType = NFKModelValueTypeForKey(name);
	NSScanner *scanner = [NSScanner scannerWithString:[text stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceCharacterSet]];
	scanner.charactersToBeSkipped = nil;
	id value = text;
	if ([valueType isEqualToString:NFKModelValueTypeInteger]) {
		long long integer = 0;
		value = [scanner scanLongLong:&integer] && scanner.atEnd ? @(integer) : nil;
	} else if ([valueType isEqualToString:NFKModelValueTypeNumber]) {
		double number = 0;
		value = [scanner scanDouble:&number] && scanner.atEnd && isfinite(number) ? @(number) : nil;
	} else if ([valueType isEqualToString:NFKModelValueTypeBoolean]) {
		NSDictionary<NSString *, NSNumber *> *booleans = @{ @"true": @YES, @"false": @NO };
		value = booleans[text.lowercaseString];
	} else if ([valueType isEqualToString:NFKModelValueTypeTextList]) {
		value = @[ text ];
	}
	if (value == nil && error != NULL) {
		*error = NFKServedInvalidRequest([NSString stringWithFormat:@"the form field %@ must be %@", name,
										  NFKServedTypeDescription(NFKServedFieldTypeForValueType(valueType) ?: @"string")]);
	}
	return value;
}

BOOL NFKServedContentTypeAcceptsJSON(NSString * _Nullable contentType)
{
	if ([contentType stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceCharacterSet].length == 0) {
		return YES;
	}
	NSString *mediaType = [[contentType componentsSeparatedByString:@";"].firstObject
						   stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceCharacterSet].lowercaseString;
	if ([mediaType isEqualToString:@"application/json"]) {
		return YES;
	}
	NSArray<NSString *> *sides = [mediaType componentsSeparatedByString:@"/"];
	return sides.count == 2 && sides[0].length > 0 && [sides[1] hasSuffix:@"+json"];
}

/*! The host a Host header names, lowercased and without its port or an IPv6 address's brackets. */
static NSString *NFKServedHostOfHeader(NSString * _Nullable header)
{
	NSString *host = header.lowercaseString ?: @"";
	if ([host hasPrefix:@"["]) {
		NSRange close = [host rangeOfString:@"]"];
		return close.location != NSNotFound ? [host substringWithRange:NSMakeRange(1, close.location - 1)] : host;
	}
	NSArray<NSString *> *sides = [host componentsSeparatedByString:@":"];
	return sides.count == 2 ? sides[0] : host;
}

BOOL NFKServedOriginIsAllowed(NFKHTTPRequest *request)
{
	NSString *origin = [request valueForHeader:@"origin"];
	if (origin.length == 0) {
		return YES;
	}
	NSString *host = [NSURL URLWithString:origin].host.lowercaseString;
	if ([host hasPrefix:@"["] && [host hasSuffix:@"]"]) {
		host = [host substringWithRange:NSMakeRange(1, host.length - 2)];
	}
	if (host.length == 0) {
		return NO;
	}
	if ([@[ @"localhost", @"127.0.0.1", @"::1" ] containsObject:host]) {
		return YES;
	}
	return [host isEqualToString:NFKServedHostOfHeader([request valueForHeader:@"host"])];
}

NSError *NFKServedExceptionError(NSException *exception)
{
	NSString *reason = [NSString stringWithFormat:@"the server raised %@ while serving the request: %@",
						exception.name, exception.reason ?: @"no reason given"];
	return [NSError errorWithDomain:NFKInferenceErrorDomain code:kNFKError_InferenceBackendFailure
						   userInfo:@{ NSLocalizedDescriptionKey: reason }];
}
