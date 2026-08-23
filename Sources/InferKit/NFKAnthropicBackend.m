//
//  NFKAnthropicBackend.m
//  InferKit
//

#import <InferKit/NFKAnthropicBackend.h>
#import <InferKit/NFKInferenceRequest.h>
#import <InferKit/NFKInferenceResult.h>
#import <InferKit/NFKInferenceKeys.h>
#import <InferKit/NFKErrors.h>
#import <InferKit/NFKRemoteBackend.h>
#import "NFK_ARC.h"

/*! The version this release was written against. A newer one is set through apiVersion. */
static NSString * const NFKAnthropicDefaultVersion = @"2023-06-01";

@implementation NFKAnthropicBackend

+ (instancetype)backendWithEndpointURL:(nullable NSURL *)endpointURL
{
	NFKAnthropicBackend *backend = [[self alloc] init];
	if (endpointURL != nil) {
		backend.endpointURL = endpointURL;
	}
	return backend;
}

- (instancetype)init
{
	self = [super init];
	if (self) {
		_endpointURL = [NSURL URLWithString:@"https://api.anthropic.com/v1/messages"];
		_apiVersion = NFKAnthropicDefaultVersion;
		_maxTokens = 1024;
		_timeout = 120;
		_session = NSURLSession.sharedSession;
	}
	return self;
}

- (NSString *)backendIdentifier
{
	return @"anthropic-messages";
}

- (BOOL)isReady
{
	return self.endpointURL != nil && self.modelName.length > 0;
}

#pragma mark Inference

- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request
												  error:(NSError * _Nullable *)outError
{
	if (!self.isReady) {
		return [self failWithCode:kNFKError_InferenceNotReady
					  description:@"the Anthropic backend needs an endpoint and a model name"
							error:outError];
	}

	NSMutableDictionary *body = [self bodyForRequest:request];
	if (body == nil) {
		return [self failWithCode:kNFKError_InferenceMissingInput
					  description:@"the request carries neither a prompt nor messages"
							error:outError];
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
	// Anthropic authenticates with its own header rather than a Bearer token, and requires a version.
	if (self.apiKey.length > 0) {
		[urlRequest setValue:self.apiKey forHTTPHeaderField:@"x-api-key"];
	}
	[urlRequest setValue:self.apiVersion forHTTPHeaderField:@"anthropic-version"];

	NSHTTPURLResponse *response = nil;
	NSError *sendError = nil;
	NSData *responseData = [self sendRequest:urlRequest response:&response error:&sendError];
	if (responseData == nil) {
		if (outError != NULL) { *outError = sendError; }
		return nil;
	}
	if (response != nil && (response.statusCode < 200 || response.statusCode > 299)) {
		NSString *detail = [[NSString alloc] initWithData:responseData encoding:NSUTF8StringEncoding];
		return [self failWithCode:kNFKError_InferenceBackendFailure
					  description:[NSString stringWithFormat:@"the endpoint returned %ld: %@",
								   (long)response.statusCode, detail ?: @"no body"]
							error:outError];
	}

	NSDictionary *responseBody = [NSJSONSerialization JSONObjectWithData:responseData options:0 error:NULL];
	if (![responseBody isKindOfClass:NSDictionary.class]) {
		return [self failWithCode:kNFKError_InferenceBackendFailure
					  description:@"the endpoint returned a body that is not a JSON object"
							error:outError];
	}
	return [self resultForResponseBody:responseBody];
}

/*! Builds the Messages request body, lifting a system message into the top-level field. */
- (nullable NSMutableDictionary *)bodyForRequest:(NFKInferenceRequest *)request
{
	NSMutableDictionary *body = [NSMutableDictionary dictionary];
	body[@"model"] = self.modelName;

	NSNumber *maxTokens = [request parameterForKey:NFKParameterMaxTokens];
	body[@"max_tokens"] = [maxTokens isKindOfClass:NSNumber.class] ? maxTokens : @(self.maxTokens);

	NSNumber *temperature = [request parameterForKey:NFKParameterTemperature];
	if ([temperature isKindOfClass:NSNumber.class]) {
		body[@"temperature"] = temperature;
	}

	NSArray *messages = request.messages;
	if (messages != nil) {
		NSMutableArray *conversation = [NSMutableArray array];
		for (NSDictionary *message in messages) {
			// A system turn is a top-level field here, not a message with a role.
			if ([message[@"role"] isEqualToString:@"system"]) {
				body[@"system"] = message[@"content"];
				continue;
			}
			[conversation addObject:message];
		}
		if (conversation.count == 0) {
			return nil;
		}
		body[@"messages"] = conversation;
		return body;
	}

	NSString *prompt = request.prompt;
	if (prompt.length == 0) {
		return nil;
	}
	body[@"messages"] = @[ @{ @"role": @"user", @"content": prompt } ];
	return body;
}

/*! Joins the text blocks of the reply. The API returns a list of typed blocks, not one string. */
- (NFKInferenceResult *)resultForResponseBody:(NSDictionary *)responseBody
{
	NSMutableDictionary *outputs = [NSMutableDictionary dictionary];
	NSArray *content = responseBody[@"content"];
	if ([content isKindOfClass:NSArray.class]) {
		NSMutableArray<NSString *> *pieces = [NSMutableArray array];
		for (NSDictionary *block in content) {
			if ([block isKindOfClass:NSDictionary.class] &&
				[block[@"type"] isEqualToString:@"text"] &&
				[block[@"text"] isKindOfClass:NSString.class]) {
				[pieces addObject:block[@"text"]];
			}
		}
		if (pieces.count > 0) {
			outputs[NFKRemoteBackendTextKey] = [pieces componentsJoinedByString:@""];
		}
	}
	outputs[NFKRemoteBackendRawKey] = responseBody;
	return [NFKInferenceResult resultWithOutputs:outputs];
}

#pragma mark Transport

/*! Overridden in tests to stage a response without a network. */
- (nullable NSData *)sendRequest:(NSURLRequest *)request
						response:(NSHTTPURLResponse * _Nullable * _Nullable)outResponse
						   error:(NSError * _Nullable *)outError
{
	__block NSData *resultData = nil;
	__block NSURLResponse *resultResponse = nil;
	__block NSError *resultError = nil;
	dispatch_semaphore_t semaphore = dispatch_semaphore_create(0);

	NSURLSessionDataTask *task = [self.session dataTaskWithRequest:request
												completionHandler:^(NSData *data, NSURLResponse *response, NSError *error) {
		resultData = NARC_RETAIN(data);
		resultResponse = NARC_RETAIN(response);
		resultError = NARC_RETAIN(error);
		dispatch_semaphore_signal(semaphore);
	}];
	[task resume];
	dispatch_semaphore_wait(semaphore, DISPATCH_TIME_FOREVER);

	if (outResponse != NULL && [resultResponse isKindOfClass:NSHTTPURLResponse.class]) {
		*outResponse = (NSHTTPURLResponse *)resultResponse;
	}
	if (resultData == nil && outError != NULL) {
		*outError = resultError != nil ? resultError
									   : [NSError errorWithDomain:NFKInferenceErrorDomain
															 code:kNFKError_InferenceBackendFailure
														 userInfo:@{ NSLocalizedDescriptionKey: @"the request returned no data" }];
	}
	return resultData;
}

#pragma mark Errors

- (nullable NFKInferenceResult *)failWithCode:(NSInteger)code
								  description:(NSString *)description
										error:(NSError * _Nullable *)outError
{
	if (outError != NULL) {
		*outError = [NSError errorWithDomain:NFKInferenceErrorDomain
										code:code
									userInfo:@{ NSLocalizedDescriptionKey: description }];
	}
	return nil;
}

@end
