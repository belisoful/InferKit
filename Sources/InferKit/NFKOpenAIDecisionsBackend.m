//
//  NFKOpenAIDecisionsBackend.m
//  InferKit
//

#import <InferKit/NFKOpenAIDecisionsBackend.h>
#import <InferKit/NFKRemoteProvider.h>
#import <InferKit/NFKRemoteTransport.h>
#import <InferKit/NFKInferenceRequest.h>
#import <InferKit/NFKInferenceResult.h>
#import <InferKit/NFKInferenceKeys.h>
#import <InferKit/NFKErrors.h>
#import "NFKRemoteMediaSupport.h"

/*! The body fields this backend writes itself; a request parameter of the same name does not
	overwrite them. */
static NSSet<NSString *> *NFKOpenAIDecisionsReservedFields(void)
{
	static NSSet<NSString *> *fields;
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		fields = [NSSet setWithArray:@[ @"model", @"input", @"questions" ]];
	});
	return fields;
}

@implementation NFKOpenAIDecisionsBackend

@synthesize session = _session;

+ (instancetype)backendWithEndpointURL:(nullable NSURL *)endpointURL
{
	NFKOpenAIDecisionsBackend *backend = [[self alloc] init];
	if (endpointURL != nil) {
		backend.endpointURL = endpointURL;
	}
	return backend;
}

+ (nullable instancetype)backendForProvider:(NFKRemoteProvider *)provider
									 apiKey:(nullable NSString *)apiKey
								  modelName:(nullable NSString *)modelName
{
	if (![provider.identifier isEqualToString:@"openai"]) {
		return nil;
	}
	NFKOpenAIDecisionsBackend *backend = [self backendWithEndpointURL:[provider URLForPath:@"decisions"]];
	backend.apiKey = apiKey;
	backend.modelName = modelName;
	return backend;
}

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_endpointURL = [NFKRemoteProvider.openAI URLForPath:@"decisions"];
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
	return @"openai-decisions";
}

- (NSSet<NSString *> *)supportedParameterKeys
{
	// A decision is not sampled, so no contract parameter has a meaning here.
	return [NSSet set];
}

- (NSSet<NSString *> *)supportedInputKeys
{
	return [NSSet setWithArray:@[ NFKInputState, NFKInputQuestions, NFKInputPrompt, NFKInputMessages,
								  NFKInputImage, NFKInputImages ]];
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
	NSDictionary *questionsByIdentifier = [self questionsForRequest:request];
	if (questionsByIdentifier.count == 0) {
		return [self failWithCode:kNFKError_InferenceMissingInput
						   reason:@"the request carries no questions under NFKInputQuestions" error:outError];
	}
	id input = [self inputForRequest:request error:outError];
	if (input == nil) {
		return nil;
	}
	NSArray<NSString *> *identifiers = [questionsByIdentifier.allKeys sortedArrayUsingSelector:@selector(compare:)];

	NSMutableDictionary<NSString *, id> *body = [NSMutableDictionary dictionary];
	for (NSString *key in request.parameters) {
		if (![NFKOpenAIDecisionsReservedFields() containsObject:key]) {
			body[key] = request.parameters[key];
		}
	}
	body[@"model"] = self.modelName;
	body[@"input"] = input;
	body[@"questions"] = [self wireQuestions:questionsByIdentifier identifiers:identifiers];

	if (![NSJSONSerialization isValidJSONObject:body]) {
		return [self failWithCode:kNFKError_InferenceMissingInput
						   reason:@"the input or a parameter is not JSON-serializable" error:outError];
	}
	NSError *encodeError = nil;
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
	[NFKRemoteTransport authorizeRequest:urlRequest apiKey:self.apiKey style:NFKRemoteAPIStyleOpenAIChat];

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
	NSArray *wireAnswers = [reply isKindOfClass:NSDictionary.class] ? reply[@"answers"] : nil;
	if (![wireAnswers isKindOfClass:NSArray.class]) {
		return [self failWithCode:kNFKError_InferenceBackendFailure reason:@"the response carries no answers" error:outError];
	}
	NSMutableDictionary<NSString *, id> *outputs = [NSMutableDictionary dictionary];
	outputs[NFKOutputAnswers] = [self answersIn:wireAnswers questions:questionsByIdentifier identifiers:identifiers];
	outputs[NFKOutputStructured] = reply;
	NSDictionary *usage = [self usageInReply:reply];
	if (usage != nil) {
		outputs[NFKOutputUsage] = usage;
	}
	return [NFKInferenceResult resultWithOutputs:outputs];
}

#pragma mark The input

- (nullable id)inputForRequest:(NFKInferenceRequest *)request error:(NSError * _Nullable *)outError
{
	id input = [self evidenceForRequest:request];
	if (input == nil) {
		[self failWithCode:kNFKError_InferenceMissingInput
					reason:@"the request carries no state under NFKInputState, NFKInputPrompt, or NFKInputMessages" error:outError];
		return nil;
	}
	if ([input isKindOfClass:NSDictionary.class]) {
		input = [self JSONTextForRecord:input];
		if (input == nil) {
			[self failWithCode:kNFKError_InferenceMissingInput reason:@"the state is not a JSON-serializable record" error:outError];
			return nil;
		}
	}
	if (![input isKindOfClass:NSString.class] && ![input isKindOfClass:NSArray.class]) {
		[self failWithCode:kNFKError_InferenceMissingInput
					reason:@"the state is not a string, a JSON-serializable record, or an array of user messages" error:outError];
		return nil;
	}
	NFKRemoteAttachments *attachments = [NFKRemoteAttachments attachmentsForRequest:request error:outError];
	if (attachments == nil) {
		return nil;
	}
	if (attachments.audioData != nil || attachments.documents.count > 0) {
		[self failWithCode:kNFKError_InferenceUnsupported
					reason:@"the Decisions API reads text and images only; audio and documents are not accepted" error:outError];
		return nil;
	}
	if (attachments.imagePNGs.count == 0) {
		return input;
	}
	NSMutableArray *parts = [NSMutableArray array];
	if ([input isKindOfClass:NSString.class]) {
		[parts addObject:@{ @"type": @"input_text", @"text": input }];
	}
	for (NSData *png in attachments.imagePNGs) {
		[parts addObject:@{ @"type": @"input_image",
							@"image_url": [@"data:image/png;base64," stringByAppendingString:[png base64EncodedStringWithOptions:0]] }];
	}
	NSDictionary *message = @{ @"role": @"user", @"content": parts };
	return [input isKindOfClass:NSArray.class] ? [input arrayByAddingObject:message] : @[ message ];
}

- (nullable id)evidenceForRequest:(NFKInferenceRequest *)request
{
	id state = [request inputForKey:NFKInputState];
	if (state != nil) {
		return state;
	}
	if (request.prompt.length > 0) {
		return request.prompt;
	}
	return request.messages.count > 0 ? [self transcriptOfMessages:request.messages] : nil;
}

/*! A conversation as one text, a "role: content" line per turn; array content contributes its text parts. */
- (NSString *)transcriptOfMessages:(NSArray<NSDictionary<NSString *, id> *> *)messages
{
	NSMutableArray<NSString *> *lines = [NSMutableArray array];
	for (NSDictionary *message in messages) {
		NSString *role = [message[@"role"] isKindOfClass:NSString.class] ? message[@"role"] : @"user";
		id content = message[@"content"];
		NSString *text = [content isKindOfClass:NSString.class] ? content : nil;
		if ([content isKindOfClass:NSArray.class]) {
			NSMutableArray<NSString *> *texts = [NSMutableArray array];
			for (NSDictionary *part in content) {
				if ([part isKindOfClass:NSDictionary.class] && [part[@"text"] isKindOfClass:NSString.class]) {
					[texts addObject:part[@"text"]];
				}
			}
			text = [texts componentsJoinedByString:@"\n"];
		}
		if (text.length > 0) {
			[lines addObject:[NSString stringWithFormat:@"%@: %@", role, text]];
		}
	}
	return [lines componentsJoinedByString:@"\n"];
}

- (nullable NSString *)JSONTextForRecord:(NSDictionary *)record
{
	if (![NSJSONSerialization isValidJSONObject:record]) {
		return nil;
	}
	NSData *data = [NSJSONSerialization dataWithJSONObject:record options:NSJSONWritingSortedKeys error:NULL];
	return data == nil ? nil : [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding];
}

#pragma mark The questions

/*! The request's NFKDecisionQuestions and wire-shape dictionaries; anything else is skipped. */
- (NSDictionary<NSString *, id> *)questionsForRequest:(NFKInferenceRequest *)request
{
	id questions = [request inputForKey:NFKInputQuestions];
	if (![questions isKindOfClass:NSDictionary.class]) {
		return @{};
	}
	NSMutableDictionary<NSString *, id> *kept = [NSMutableDictionary dictionary];
	for (NSString *identifier in questions) {
		id question = questions[identifier];
		if ([question isKindOfClass:NFKDecisionQuestion.class] || [question isKindOfClass:NSDictionary.class]) {
			kept[identifier] = question;
		}
	}
	return kept;
}

- (NSArray<NSDictionary *> *)wireQuestions:(NSDictionary<NSString *, id> *)questions identifiers:(NSArray<NSString *> *)identifiers
{
	NSMutableArray<NSDictionary *> *wire = [NSMutableArray array];
	for (NSString *identifier in identifiers) {
		id question = questions[identifier];
		if ([question isKindOfClass:NFKDecisionQuestion.class]) {
			[wire addObject:[self wireQuestion:question name:identifier]];
		} else if (question[@"name"] != nil) {
			[wire addObject:question];
		} else {
			NSMutableDictionary *named = [question mutableCopy];
			named[@"name"] = identifier;
			[wire addObject:named];
		}
	}
	return wire;
}

- (NSDictionary *)wireQuestion:(NFKDecisionQuestion *)question name:(NSString *)name
{
	switch (question.type) {
		case NFKDecisionTypeChoice: {
			NSMutableArray *choices = [NSMutableArray array];
			for (NSString *option in question.options) {
				NSMutableDictionary *choice = [NSMutableDictionary dictionaryWithObject:option forKey:@"value"];
				if (question.descriptions[option] != nil) {
					choice[@"description"] = question.descriptions[option];
				}
				[choices addObject:choice];
			}
			return @{ @"type": @"choice", @"name": name, @"instructions": question.instructions, @"choices": choices };
		}
		case NFKDecisionTypeScore: {
			NSMutableArray *levels = [NSMutableArray array];
			for (NSString *level in question.options) {
				[levels addObject:@{ @"label": level }];
			}
			return @{ @"type": @"score", @"name": name, @"instructions": question.instructions, @"levels": levels };
		}
		case NFKDecisionTypeNoul:
			return @{ @"type": @"predicate", @"name": name, @"instructions": [self predicateInstructionsFor:question] };
	}
	return @{};
}

/*! The service has no field for what "true" and "false" mean, so given meanings follow the
	instructions as two lines. */
- (NSString *)predicateInstructionsFor:(NFKDecisionQuestion *)question
{
	NSMutableString *instructions = [question.instructions mutableCopy];
	NSString *trueMeaning = question.descriptions[@"true"];
	NSString *falseMeaning = question.descriptions[@"false"];
	if (trueMeaning != nil || falseMeaning != nil) {
		[instructions appendString:@"\n"];
	}
	if (trueMeaning != nil) {
		[instructions appendFormat:@"\nTrue means: %@", trueMeaning];
	}
	if (falseMeaning != nil) {
		[instructions appendFormat:@"\nFalse means: %@", falseMeaning];
	}
	return instructions;
}

- (NFKDecisionType)typeOfQuestion:(id)question
{
	if ([question isKindOfClass:NFKDecisionQuestion.class]) {
		return [(NFKDecisionQuestion *)question type];
	}
	NSString *name = question[@"type"];
	if ([name isEqual:@"score"]) {
		return NFKDecisionTypeScore;
	}
	if ([name isEqual:@"predicate"] || [name isEqual:@"noul"]) {
		return NFKDecisionTypeNoul;
	}
	return NFKDecisionTypeChoice;
}

#pragma mark The reply

/*! Each answer is matched to its question by name; an unnamed answer by its position, because the
	service answers in the order it was asked. An answer of an unknown type is dropped. */
- (NSDictionary<NSString *, NFKDecisionAnswer *> *)answersIn:(NSArray *)wireAnswers
												   questions:(NSDictionary<NSString *, id> *)questions
												 identifiers:(NSArray<NSString *> *)identifiers
{
	NSMutableDictionary<NSString *, NFKDecisionAnswer *> *answers = [NSMutableDictionary dictionary];
	[wireAnswers enumerateObjectsUsingBlock:^(NSDictionary *wire, NSUInteger index, BOOL *stop) {
		if (![wire isKindOfClass:NSDictionary.class]) {
			return;
		}
		NSString *identifier = [wire[@"name"] isKindOfClass:NSString.class] ? wire[@"name"]
			: (index < identifiers.count ? identifiers[index] : nil);
		if (identifier == nil) {
			return;
		}
		NFKDecisionAnswer *answer = [wire[@"type"] isEqual:@"refusal"]
			? [NFKDecisionAnswer refusalForType:[self typeOfQuestion:questions[identifier]] raw:wire]
			: [NFKDecisionAnswer answerWithDictionary:wire];
		if (answer != nil) {
			answers[identifier] = answer;
		}
	}];
	return answers;
}

- (nullable NSDictionary<NSString *, NSNumber *> *)usageInReply:(NSDictionary *)reply
{
	NSDictionary *usage = [reply[@"usage"] isKindOfClass:NSDictionary.class] ? reply[@"usage"] : nil;
	if (usage == nil) {
		return nil;
	}
	NSDictionary *input = [usage[@"input_tokens_details"] isKindOfClass:NSDictionary.class] ? usage[@"input_tokens_details"] : nil;
	NSDictionary *output = [usage[@"output_tokens_details"] isKindOfClass:NSDictionary.class] ? usage[@"output_tokens_details"] : nil;
	return NFKRemoteUsage(usage[@"input_tokens"], input[@"cached_tokens"], usage[@"output_tokens"], output[@"reasoning_tokens"]);
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
