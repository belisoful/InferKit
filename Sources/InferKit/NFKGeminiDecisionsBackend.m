//
//  NFKGeminiDecisionsBackend.m
//  InferKit
//

#import <InferKit/NFKGeminiDecisionsBackend.h>
#import <InferKit/NFKRemoteProvider.h>
#import <InferKit/NFKRemoteTransport.h>
#import <InferKit/NFKInferenceRequest.h>
#import <InferKit/NFKInferenceResult.h>
#import <InferKit/NFKInferenceKeys.h>
#import <InferKit/NFKErrors.h>
#import "NFKRemoteMediaSupport.h"
#import "NFKRemoteTransportPrivate.h"

static NSString * const NFKGeminiDecisionsResolutionField = @"resolution";

/*! The fields this backend writes itself or places on a block; a request parameter of the same name
	does not reach the body's top level. */
static NSSet<NSString *> *NFKGeminiDecisionsReservedFields(void)
{
	static NSSet<NSString *> *fields;
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		fields = [NSSet setWithArray:@[ @"model", @"input", @"questions", NFKGeminiDecisionsResolutionField ]];
	});
	return fields;
}

@implementation NFKGeminiDecisionsBackend

@synthesize session = _session;

+ (instancetype)backendWithEndpointURL:(nullable NSURL *)endpointURL
{
	NFKGeminiDecisionsBackend *backend = [[self alloc] init];
	if (endpointURL != nil) {
		backend.endpointURL = endpointURL;
	}
	return backend;
}

+ (nullable instancetype)backendForProvider:(NFKRemoteProvider *)provider
									 apiKey:(nullable NSString *)apiKey
								  modelName:(nullable NSString *)modelName
{
	if (![provider.identifier isEqualToString:@"gemini"]) {
		return nil;
	}
	// The preset's base is the OpenAI layer under v1beta; the decisions route is its sibling.
	NSURL *endpoint = [provider.baseURL.URLByDeletingLastPathComponent URLByAppendingPathComponent:@"decisions"];
	NFKGeminiDecisionsBackend *backend = [self backendWithEndpointURL:endpoint];
	NSURLSession *pinned = NFKRemotePinnedSessionForProvider(provider);
	if (pinned != nil) {
		backend.session = pinned;
	}
	backend.apiKey = apiKey;
	backend.modelName = modelName;
	return backend;
}

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_endpointURL = [NSURL URLWithString:@"https://generativelanguage.googleapis.com/v1beta/decisions"];
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
	return @"gemini-decisions";
}

- (NSSet<NSString *> *)supportedParameterKeys
{
	// A decision is not sampled, so no contract parameter has a meaning here.
	return [NSSet set];
}

- (NSSet<NSString *> *)supportedInputKeys
{
	return [NSSet setWithArray:@[ NFKInputState, NFKInputQuestions, NFKInputPrompt, NFKInputMessages,
								  NFKInputImage, NFKInputImages, NFKInputAudio, NFKInputVideo,
								  NFKInputDocument, NFKInputDocuments ]];
}

- (NSSet<NSString *> *)supportedOutputKeys
{
	return [NSSet setWithArray:@[ NFKOutputAnswers, NFKOutputStructured, NFKOutputUsage ]];
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
	NSDictionary<NSString *, NSDictionary *> *questions = [self wireQuestionsForRequest:request];
	if (questions.count == 0) {
		return [self failWithCode:kNFKError_InferenceMissingInput
						   reason:@"the request carries no questions under NFKInputQuestions" error:outError];
	}
	id input = [self inputForRequest:request error:outError];
	if (input == nil) {
		return nil;
	}

	NSMutableDictionary<NSString *, id> *body = [NSMutableDictionary dictionary];
	for (NSString *key in request.parameters) {
		if (![NFKGeminiDecisionsReservedFields() containsObject:key]) {
			body[key] = request.parameters[key];
		}
	}
	body[@"model"] = self.modelName;
	body[@"input"] = input;
	body[@"questions"] = questions;

	if (![NSJSONSerialization isValidJSONObject:body]) {
		return [self failWithCode:kNFKError_InferenceMissingInput
						   reason:@"the input, a question, or a parameter is not JSON-serializable" error:outError];
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
	if (self.apiKey.length > 0) {
		[urlRequest setValue:self.apiKey forHTTPHeaderField:@"x-goog-api-key"];
	}

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
	NSDictionary *feedback = [reply isKindOfClass:NSDictionary.class] ? [self blockFeedbackIn:reply] : nil;
	NSDictionary *wireAnswers = [reply isKindOfClass:NSDictionary.class] ? reply[@"answers"] : nil;
	if (feedback == nil && ![wireAnswers isKindOfClass:NSDictionary.class]) {
		return [self failWithCode:kNFKError_InferenceBackendFailure reason:@"the response carries no answers" error:outError];
	}
	NSMutableDictionary<NSString *, id> *outputs = [NSMutableDictionary dictionary];
	outputs[NFKOutputAnswers] = feedback != nil ? [self refusalsForQuestions:questions feedback:feedback]
												: [self answersIn:wireAnswers];
	outputs[NFKOutputStructured] = reply;
	NSDictionary *usage = [self usageInReply:reply];
	if (usage != nil) {
		outputs[NFKOutputUsage] = usage;
	}
	return [NFKInferenceResult resultWithOutputs:outputs];
}

#pragma mark The input

/*! The evidence as the service reads it: a string when the request carries text alone, the caller's
	own content blocks, or the text block followed by a block per attachment. */
- (nullable id)inputForRequest:(NFKInferenceRequest *)request error:(NSError * _Nullable *)outError
{
	id input = [self evidenceForRequest:request];
	if ([input isKindOfClass:NSDictionary.class]) {
		input = [self JSONTextForRecord:input];
		if (input == nil) {
			[self failWithCode:kNFKError_InferenceMissingInput reason:@"the state is not a JSON-serializable record" error:outError];
			return nil;
		}
	}
	if (input != nil && ![input isKindOfClass:NSString.class] && ![input isKindOfClass:NSArray.class]) {
		[self failWithCode:kNFKError_InferenceMissingInput
					reason:@"the state is not a string, a JSON-serializable record, or an array of content blocks" error:outError];
		return nil;
	}
	NFKRemoteAttachments *attachments = [NFKRemoteAttachments attachmentsForRequest:request keepsVideo:YES error:outError];
	if (attachments == nil) {
		return nil;
	}
	NSArray<NSDictionary *> *media = [self mediaBlocksFor:attachments resolution:request.parameters[NFKGeminiDecisionsResolutionField]];
	if (input == nil && media.count == 0) {
		[self failWithCode:kNFKError_InferenceMissingInput
					reason:@"the request carries no state under NFKInputState, NFKInputPrompt, or NFKInputMessages, and no media" error:outError];
		return nil;
	}
	if (media.count == 0) {
		return input;
	}
	NSMutableArray *blocks = [NSMutableArray array];
	if ([input isKindOfClass:NSArray.class]) {
		[blocks addObjectsFromArray:input];
	} else if ([input length] > 0) {
		[blocks addObject:@{ @"type": @"text", @"text": input }];
	}
	[blocks addObjectsFromArray:media];
	return blocks;
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

// The service sets detail per image and video block, so the request's one resolution rides on each.
- (NSArray<NSDictionary *> *)mediaBlocksFor:(NFKRemoteAttachments *)attachments resolution:(nullable id)resolution
{
	NSArray<NSMutableDictionary<NSString *, id> *> *blocks = NFKGeminiContentBlocks(attachments);
	if ([resolution isKindOfClass:NSString.class]) {
		for (NSMutableDictionary<NSString *, id> *block in blocks) {
			if ([block[@"type"] isEqual:@"image"] || [block[@"type"] isEqual:@"video"]) {
				block[NFKGeminiDecisionsResolutionField] = resolution;
			}
		}
	}
	return blocks;
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

/*! The questions in wire shape: an NFKDecisionQuestion is converted, and a dictionary is taken as
	already in that shape. Anything else is skipped. */
- (NSDictionary<NSString *, NSDictionary *> *)wireQuestionsForRequest:(NFKInferenceRequest *)request
{
	id questions = [request inputForKey:NFKInputQuestions];
	if (![questions isKindOfClass:NSDictionary.class]) {
		return @{};
	}
	NSMutableDictionary<NSString *, NSDictionary *> *wire = [NSMutableDictionary dictionary];
	for (NSString *identifier in questions) {
		id question = questions[identifier];
		if ([question isKindOfClass:NFKDecisionQuestion.class]) {
			wire[identifier] = [self wireQuestion:question];
		} else if ([question isKindOfClass:NSDictionary.class]) {
			wire[identifier] = question;
		}
	}
	return wire;
}

- (NSDictionary *)wireQuestion:(NFKDecisionQuestion *)question
{
	NSMutableDictionary<NSString *, id> *wire = [NSMutableDictionary dictionary];
	wire[@"type"] = [NFKDecisionQuestion nameForType:question.type];
	wire[@"instructions"] = question.instructions;
	id criteria = [self criteriaFor:question];
	if (criteria != nil) {
		wire[@"criteria"] = criteria;
	}
	return wire;
}

- (nullable id)criteriaFor:(NFKDecisionQuestion *)question
{
	switch (question.type) {
		case NFKDecisionTypeChoice: {
			// The service takes a description string per option; an empty one stands for none.
			NSMutableDictionary<NSString *, NSString *> *criteria = [NSMutableDictionary dictionary];
			for (NSString *option in question.options) {
				criteria[option] = question.descriptions[option] ?: @"";
			}
			return criteria;
		}
		case NFKDecisionTypeScore:
			return question.options;
		case NFKDecisionTypeNoul: {
			NSMutableDictionary<NSString *, NSString *> *criteria = [NSMutableDictionary dictionary];
			criteria[@"true_description"] = question.descriptions[@"true"];
			criteria[@"false_description"] = question.descriptions[@"false"];
			return criteria.count > 0 ? criteria : nil;
		}
	}
	return nil;
}

- (NFKDecisionType)typeOfWireQuestion:(NSDictionary *)question
{
	NSString *name = [question[@"type"] isKindOfClass:NSString.class] ? question[@"type"] : nil;
	if ([name isEqualToString:@"score"]) {
		return NFKDecisionTypeScore;
	}
	return [name isEqualToString:@"noul"] ? NFKDecisionTypeNoul : NFKDecisionTypeChoice;
}

#pragma mark The reply

- (NSDictionary<NSString *, NFKDecisionAnswer *> *)answersIn:(NSDictionary *)wireAnswers
{
	NSMutableDictionary<NSString *, NFKDecisionAnswer *> *answers = [NSMutableDictionary dictionary];
	for (NSString *identifier in wireAnswers) {
		NFKDecisionAnswer *answer = [NFKDecisionAnswer answerWithDictionary:wireAnswers[identifier]];
		if (answer != nil) {
			answers[identifier] = answer;
		}
	}
	return answers;
}

/*! The reply's prompt_feedback when it names a block reason: the service evaluated no question. */
- (nullable NSDictionary *)blockFeedbackIn:(NSDictionary *)reply
{
	NSDictionary *feedback = [reply[@"prompt_feedback"] isKindOfClass:NSDictionary.class] ? reply[@"prompt_feedback"] : nil;
	return [feedback[@"block_reason"] isKindOfClass:NSString.class] ? feedback : nil;
}

/*! A refused answer per question asked, each of its question's type and carrying the feedback. */
- (NSDictionary<NSString *, NFKDecisionAnswer *> *)refusalsForQuestions:(NSDictionary<NSString *, NSDictionary *> *)questions
															   feedback:(NSDictionary *)feedback
{
	NSMutableDictionary<NSString *, NFKDecisionAnswer *> *refusals = [NSMutableDictionary dictionary];
	for (NSString *identifier in questions) {
		refusals[identifier] = [NFKDecisionAnswer refusalForType:[self typeOfWireQuestion:questions[identifier]] raw:feedback];
	}
	return refusals;
}

- (nullable NSDictionary<NSString *, NSNumber *> *)usageInReply:(NSDictionary *)reply
{
	NSDictionary *usage = [reply[@"usage"] isKindOfClass:NSDictionary.class] ? reply[@"usage"] : nil;
	return usage == nil ? nil : NFKRemoteUsage(usage[@"total_input_tokens"], usage[@"total_cached_tokens"],
											   usage[@"total_output_tokens"], usage[@"total_thought_tokens"]);
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
