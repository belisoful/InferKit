//
//  NFKRemoteTokenCounter.m
//  InferKit
//

#import <InferKit/NFKRemoteTokenCounter.h>
#import <InferKit/NFKRemoteProvider.h>
#import <InferKit/NFKRemoteTransport.h>
#import <InferKit/NFKInferenceRequest.h>
#import <InferKit/NFKErrors.h>

@implementation NFKRemoteTokenCounter

@synthesize session = _session;

+ (nullable instancetype)counterForProvider:(NFKRemoteProvider *)provider
									 apiKey:(nullable NSString *)apiKey
								  modelName:(nullable NSString *)modelName
{
	NSString *identifier = provider.identifier;
	NSURL *versionRoot = provider.baseURL.URLByDeletingLastPathComponent;
	NFKRemoteTokenCounter *counter = [[self alloc] init];
	if ([identifier isEqualToString:@"anthropic"]) {
		counter.endpointURL = [provider URLForPath:@"messages/count_tokens"];
		counter.apiStyle = NFKRemoteTokenCounterAPIStyleAnthropic;
	} else if ([identifier isEqualToString:@"gemini"]) {
		counter.endpointURL = [versionRoot URLByAppendingPathComponent:@"models"];
		counter.apiStyle = NFKRemoteTokenCounterAPIStyleGemini;
	} else if ([identifier isEqualToString:@"xai"]) {
		counter.endpointURL = [provider URLForPath:@"tokenize-text"];
		counter.apiStyle = NFKRemoteTokenCounterAPIStyleXAI;
	} else if ([identifier isEqualToString:@"llamacpp"]) {
		counter.endpointURL = [versionRoot URLByAppendingPathComponent:@"tokenize"];
		counter.apiStyle = NFKRemoteTokenCounterAPIStyleLlamaCpp;
	} else {
		return nil;
	}
	counter.apiKey = apiKey;
	counter.modelName = modelName;
	return counter;
}

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
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

#pragma mark Counting

- (nullable NSNumber *)tokenCountForRequest:(NFKInferenceRequest *)request error:(NSError * _Nullable *)outError
{
	NSDictionary *reply = nil;
	if (self.apiStyle == NFKRemoteTokenCounterAPIStyleAnthropic) {
		NSDictionary *body = [self anthropicBodyForRequest:request];
		if (body == nil) {
			[self fail:outError code:kNFKError_InferenceMissingInput reason:@"the request carries neither a prompt nor messages"];
			return nil;
		}
		reply = [self replyForBody:body URL:self.endpointURL error:outError];
		return [self numberIn:reply key:@"input_tokens" error:outError];
	}
	NSString *text = [self textForRequest:request];
	if (text == nil) {
		[self fail:outError code:kNFKError_InferenceMissingInput reason:@"the request carries neither a prompt nor messages"];
		return nil;
	}
	if (self.apiStyle == NFKRemoteTokenCounterAPIStyleGemini) {
		NSURL *url = [self.endpointURL URLByAppendingPathComponent:[(self.modelName ?: @"") stringByAppendingString:@":countTokens"]];
		reply = [self replyForBody:@{ @"contents": @[ @{ @"parts": @[ @{ @"text": text } ] } ] } URL:url error:outError];
		return [self numberIn:reply key:@"totalTokens" error:outError];
	}
	NSArray<NSNumber *> *identifiers = [self tokenIdentifiersForText:text error:outError];
	return identifiers != nil ? @(identifiers.count) : nil;
}

- (nullable NSArray<NSNumber *> *)tokenIdentifiersForText:(NSString *)text error:(NSError * _Nullable *)outError
{
	if (self.apiStyle == NFKRemoteTokenCounterAPIStyleXAI) {
		NSMutableDictionary *body = [NSMutableDictionary dictionaryWithObject:text forKey:@"text"];
		body[@"model"] = self.modelName;
		NSDictionary *reply = [self replyForBody:body URL:self.endpointURL error:outError];
		NSArray *tokens = [reply[@"token_ids"] isKindOfClass:NSArray.class] ? reply[@"token_ids"] : nil;
		if (tokens == nil) {
			return reply == nil ? nil : [self fail:outError code:kNFKError_InferenceBackendFailure reason:@"the reply carries no token_ids"];
		}
		NSMutableArray<NSNumber *> *identifiers = [NSMutableArray array];
		for (id token in tokens) {
			id identifier = [token isKindOfClass:NSDictionary.class] ? token[@"token_id"] : token;
			if ([identifier isKindOfClass:NSNumber.class]) {
				[identifiers addObject:identifier];
			}
		}
		return identifiers;
	}
	if (self.apiStyle == NFKRemoteTokenCounterAPIStyleLlamaCpp) {
		NSDictionary *reply = [self replyForBody:@{ @"content": text } URL:self.endpointURL error:outError];
		NSArray *tokens = [reply[@"tokens"] isKindOfClass:NSArray.class] ? reply[@"tokens"] : nil;
		if (tokens == nil) {
			return reply == nil ? nil : [self fail:outError code:kNFKError_InferenceBackendFailure reason:@"the reply carries no tokens"];
		}
		NSMutableArray<NSNumber *> *identifiers = [NSMutableArray array];
		for (id token in tokens) {
			id identifier = [token isKindOfClass:NSDictionary.class] ? token[@"id"] : token;
			if ([identifier isKindOfClass:NSNumber.class]) {
				[identifiers addObject:identifier];
			}
		}
		return identifiers;
	}
	return [self fail:outError code:kNFKError_InferenceUnsupported reason:@"this service counts tokens but does not return their ids"];
}

#pragma mark Request

// The Messages body as the count endpoint takes it: a leading system turn is the top-level field.
- (nullable NSDictionary *)anthropicBodyForRequest:(NFKInferenceRequest *)request
{
	NSMutableDictionary *body = [NSMutableDictionary dictionary];
	body[@"model"] = self.modelName;
	NSMutableArray *messages = [NSMutableArray array];
	for (NSDictionary *message in request.messages) {
		if (![message isKindOfClass:NSDictionary.class]) {
			continue;
		}
		if ([message[@"role"] isEqual:@"system"]) {
			body[@"system"] = message[@"content"];
			continue;
		}
		[messages addObject:message];
	}
	if (messages.count == 0 && request.prompt.length > 0) {
		[messages addObject:@{ @"role": @"user", @"content": request.prompt }];
	}
	if (messages.count == 0) {
		return nil;
	}
	body[@"messages"] = messages;
	return body;
}

- (nullable NSString *)textForRequest:(NFKInferenceRequest *)request
{
	if (request.prompt.length > 0) {
		return request.prompt;
	}
	NSMutableArray<NSString *> *pieces = [NSMutableArray array];
	for (NSDictionary *message in request.messages) {
		if ([message isKindOfClass:NSDictionary.class] && [message[@"content"] isKindOfClass:NSString.class]) {
			[pieces addObject:message[@"content"]];
		}
	}
	return pieces.count > 0 ? [pieces componentsJoinedByString:@"\n"] : nil;
}

- (nullable NSDictionary *)replyForBody:(NSDictionary *)body URL:(nullable NSURL *)url error:(NSError * _Nullable *)outError
{
	if (url == nil) {
		return [self fail:outError code:kNFKError_InferenceNotReady reason:@"no endpoint URL is set"];
	}
	NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:url];
	request.HTTPMethod = @"POST";
	request.timeoutInterval = self.timeout;
	request.HTTPBody = [NSJSONSerialization dataWithJSONObject:body options:0 error:NULL];
	[request setValue:@"application/json" forHTTPHeaderField:@"Content-Type"];
	if (self.apiStyle == NFKRemoteTokenCounterAPIStyleAnthropic) {
		[NFKRemoteTransport authorizeRequest:request apiKey:self.apiKey style:NFKRemoteAPIStyleAnthropicMessages];
	} else if (self.apiStyle == NFKRemoteTokenCounterAPIStyleGemini) {
		if (self.apiKey.length > 0) {
			[request setValue:self.apiKey forHTTPHeaderField:@"x-goog-api-key"];
		}
	} else {
		[NFKRemoteTransport authorizeRequest:request apiKey:self.apiKey style:NFKRemoteAPIStyleOpenAIChat];
	}
	NSHTTPURLResponse *response = nil;
	NSError *sendError = nil;
	NSData *data = [self sendRequest:request response:&response error:&sendError];
	NSError *failure = data == nil ? sendError : [NFKRemoteTransport errorForResponse:response data:data];
	if (failure != nil || data == nil) {
		if (outError != NULL) {
			*outError = failure ?: [NFKRemoteTransport errorWithCode:kNFKError_InferenceBackendFailure reason:@"the count call failed"];
		}
		return nil;
	}
	id reply = [NSJSONSerialization JSONObjectWithData:data options:0 error:NULL];
	return [reply isKindOfClass:NSDictionary.class] ? reply
		: [self fail:outError code:kNFKError_InferenceBackendFailure reason:@"the response is not a JSON object"];
}

- (nullable NSNumber *)numberIn:(nullable NSDictionary *)reply key:(NSString *)key error:(NSError * _Nullable *)outError
{
	if (reply == nil) {
		return nil;
	}
	return [reply[key] isKindOfClass:NSNumber.class] ? reply[key]
		: [self fail:outError code:kNFKError_InferenceBackendFailure reason:[NSString stringWithFormat:@"the reply carries no %@", key]];
}

#pragma mark Transport

- (nullable NSData *)sendRequest:(NSURLRequest *)request
						response:(NSHTTPURLResponse * _Nullable * _Nullable)outResponse
						   error:(NSError * _Nullable *)outError
{
	return [NFKRemoteTransport sendRequest:request session:self.session response:outResponse error:outError];
}

#pragma mark Errors

- (nullable id)fail:(NSError * _Nullable *)outError code:(NFKInferenceError)code reason:(NSString *)reason
{
	if (outError != NULL) {
		*outError = [NFKRemoteTransport errorWithCode:code reason:reason];
	}
	return nil;
}

@end
