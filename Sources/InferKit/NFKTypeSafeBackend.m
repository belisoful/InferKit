//
//  NFKTypeSafeBackend.m
//  InferKit
//

#import <InferKit/NFKTypeSafeBackend.h>
#import <InferKit/NFKRemoteProvider.h>
#import <InferKit/NFKRemoteTransport.h>
#import <InferKit/NFKInferenceRequest.h>
#import <InferKit/NFKInferenceResult.h>
#import <InferKit/NFKInferenceKeys.h>
#import <InferKit/NFKErrors.h>

/*! The body fields this backend writes itself; a request parameter of the same name does not
	overwrite them. */
static NSSet<NSString *> *NFKTypeSafeReservedFields(void)
{
	static NSSet<NSString *> *fields;
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		fields = [NSSet setWithArray:@[ @"model", @"state", @"questions" ]];
	});
	return fields;
}

@implementation NFKTypeSafeBackend

@synthesize session = _session;

+ (instancetype)backendWithEndpointURL:(nullable NSURL *)endpointURL
{
	NFKTypeSafeBackend *backend = [[self alloc] init];
	if (endpointURL != nil) {
		backend.endpointURL = endpointURL;
	}
	return backend;
}

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_endpointURL = NFKRemoteProvider.typeSafe.endpointURL;
		_timeout = 30.0;
	}
	return self;
}

- (NSURLSession *)session
{
	if (_session == nil) {
		_session = [NSURLSession sharedSession];
	}
	return _session;
}

#pragma mark NFKInferenceBackend

- (BOOL)isReady
{
	return self.endpointURL != nil && self.modelName.length > 0;
}

- (NSString *)backendIdentifier
{
	return @"typesafe-systemone";
}

- (NSSet<NSString *> *)supportedParameterKeys
{
	// A decision is not sampled, so no contract parameter has a meaning here.
	return [NSSet set];
}

- (NSSet<NSString *> *)supportedInputKeys
{
	return [NSSet setWithArray:@[ NFKInputState, NFKInputQuestions, NFKInputPrompt, NFKInputMessages ]];
}

- (nullable NSDictionary<NSString *, NFKDecisionAnswer *> *)answersForState:(id)state
																   questions:(NSDictionary<NSString *, NFKDecisionQuestion *> *)questions
																	   error:(NSError * _Nullable *)outError
{
	NFKInferenceRequest *request = [NFKInferenceRequest requestWithInputs:@{ NFKInputState: state,
																			 NFKInputQuestions: questions }
															   parameters:@{}
														   outputModality:NFKModalityText];
	return [self runInferenceForRequest:request error:outError].answers;
}

- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request
												  error:(NSError * _Nullable *)outError
{
	if (!self.isReady) {
		return [self failWithCode:kNFKError_InferenceNotReady
						   reason:self.endpointURL == nil ? @"no endpoint URL is set" : @"no model name is set; the API requires one"
							error:outError];
	}
	id state = [self stateForRequest:request];
	if (state == nil) {
		return [self failWithCode:kNFKError_InferenceMissingInput
						   reason:@"the request carries no state under NFKInputState, NFKInputPrompt, or NFKInputMessages" error:outError];
	}
	NSDictionary *questions = [self wireQuestionsForRequest:request];
	if (questions.count == 0) {
		return [self failWithCode:kNFKError_InferenceMissingInput
						   reason:@"the request carries no questions under NFKInputQuestions" error:outError];
	}

	NSMutableDictionary<NSString *, id> *body = [NSMutableDictionary dictionary];
	for (NSString *key in request.parameters) {
		if (![NFKTypeSafeReservedFields() containsObject:key]) {
			body[key] = request.parameters[key];
		}
	}
	body[@"model"] = self.modelName;
	body[@"state"] = state;
	body[@"questions"] = questions;

	NSError *encodeError = nil;
	if (![NSJSONSerialization isValidJSONObject:body]) {
		return [self failWithCode:kNFKError_InferenceMissingInput
						   reason:@"the state is not a string or a JSON-serializable dictionary or array" error:outError];
	}
	NSData *payload = [NSJSONSerialization dataWithJSONObject:body options:0 error:&encodeError];
	if (payload == nil) {
		if (outError != NULL) { *outError = encodeError; }
		return nil;
	}
	NSMutableURLRequest *urlRequest = [NSMutableURLRequest requestWithURL:self.endpointURL];
	urlRequest.HTTPMethod = @"POST";
	urlRequest.timeoutInterval = self.timeout;
	urlRequest.HTTPBody = payload;
	[urlRequest setValue:@"application/json" forHTTPHeaderField:@"Content-Type"];
	[NFKRemoteTransport authorizeRequest:urlRequest apiKey:self.apiKey style:NFKRemoteAPIStyleSystemOne];

	NSHTTPURLResponse *response = nil;
	NSError *sendError = nil;
	NSData *data = [self sendRequest:urlRequest response:&response error:&sendError];
	if (data == nil) {
		if (outError != NULL) { *outError = sendError; }
		return nil;
	}
	NSError *statusError = [NFKRemoteTransport errorForResponse:response data:data];
	if (statusError != nil) {
		if (outError != NULL) { *outError = statusError; }
		return nil;
	}
	id reply = [NSJSONSerialization JSONObjectWithData:data options:0 error:NULL];
	NSDictionary *wireAnswers = [reply isKindOfClass:NSDictionary.class] ? reply[@"answers"] : nil;
	if (![wireAnswers isKindOfClass:NSDictionary.class]) {
		return [self failWithCode:kNFKError_InferenceBackendFailure reason:@"the response carries no answers" error:outError];
	}
	NSMutableDictionary<NSString *, NFKDecisionAnswer *> *answers = [NSMutableDictionary dictionary];
	for (NSString *identifier in wireAnswers) {
		NFKDecisionAnswer *answer = [NFKDecisionAnswer answerWithDictionary:wireAnswers[identifier]];
		if (answer != nil) {
			answers[identifier] = answer;
		}
	}
	NSMutableDictionary<NSString *, id> *outputs = [NSMutableDictionary dictionary];
	outputs[NFKOutputAnswers] = answers;
	outputs[NFKOutputStructured] = reply;
	NSDictionary *usage = [self usageInReply:reply];
	if (usage != nil) {
		outputs[NFKOutputUsage] = usage;
	}
	return [NFKInferenceResult resultWithOutputs:outputs];
}

#pragma mark The request

- (nullable id)stateForRequest:(NFKInferenceRequest *)request
{
	id state = [request inputForKey:NFKInputState];
	if (state != nil) {
		return state;
	}
	if (request.prompt.length > 0) {
		return request.prompt;
	}
	return request.messages;
}

/*! The questions in wire shape: an NFKDecisionQuestion is converted, and a dictionary is taken as
	already in that shape. Anything else is skipped. */
- (NSDictionary<NSString *, id> *)wireQuestionsForRequest:(NFKInferenceRequest *)request
{
	id questions = [request inputForKey:NFKInputQuestions];
	if (![questions isKindOfClass:NSDictionary.class]) {
		return @{};
	}
	NSMutableDictionary<NSString *, id> *wire = [NSMutableDictionary dictionary];
	for (NSString *identifier in questions) {
		id question = questions[identifier];
		if ([question isKindOfClass:NFKDecisionQuestion.class]) {
			wire[identifier] = [question dictionaryRepresentation];
		} else if ([question isKindOfClass:NSDictionary.class]) {
			wire[identifier] = question;
		}
	}
	return wire;
}

#pragma mark The reply

- (nullable NSDictionary<NSString *, NSNumber *> *)usageInReply:(NSDictionary *)reply
{
	NSDictionary *usage = [reply[@"usage"] isKindOfClass:NSDictionary.class] ? reply[@"usage"] : nil;
	if (usage == nil) {
		return nil;
	}
	NSMutableDictionary<NSString *, NSNumber *> *counts = [NSMutableDictionary dictionary];
	if ([usage[@"input_tokens"] isKindOfClass:NSNumber.class]) {
		counts[NFKUsageInputTokens] = usage[@"input_tokens"];
	}
	if ([usage[@"output_tokens"] isKindOfClass:NSNumber.class]) {
		counts[NFKUsageOutputTokens] = usage[@"output_tokens"];
	}
	return counts.count > 0 ? counts : nil;
}

#pragma mark Transport

- (nullable NSData *)sendRequest:(NSURLRequest *)request
						response:(NSHTTPURLResponse * _Nullable * _Nullable)outResponse
						   error:(NSError * _Nullable *)outError
{
	return [NFKRemoteTransport sendRequest:request session:self.session response:outResponse error:outError];
}

#pragma mark Errors

- (nullable NFKInferenceResult *)failWithCode:(NFKInferenceError)code reason:(NSString *)reason error:(NSError * _Nullable *)outError
{
	if (outError != NULL) {
		*outError = [NFKRemoteTransport errorWithCode:code reason:reason];
	}
	return nil;
}

@end
