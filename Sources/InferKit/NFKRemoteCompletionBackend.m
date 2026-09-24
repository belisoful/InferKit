//
//  NFKRemoteCompletionBackend.m
//  InferKit
//

#import <InferKit/NFKRemoteCompletionBackend.h>
#import <InferKit/NFKRemoteProvider.h>
#import <InferKit/NFKRemoteTransport.h>
#import <InferKit/NFKRemoteBackend.h>
#import <InferKit/NFKInferenceRequest.h>
#import <InferKit/NFKInferenceResult.h>
#import <InferKit/NFKInferenceJob.h>
#import <InferKit/NFKInferenceKeys.h>
#import <InferKit/NFKErrors.h>

@implementation NFKRemoteCompletionBackend

@synthesize session = _session;

+ (instancetype)backendWithEndpointURL:(nullable NSURL *)endpointURL
{
	NFKRemoteCompletionBackend *backend = [[self alloc] init];
	backend.endpointURL = endpointURL;
	return backend;
}

+ (nullable instancetype)backendForProvider:(NFKRemoteProvider *)provider
									 apiKey:(nullable NSString *)apiKey
								  modelName:(nullable NSString *)modelName
{
	NSString *identifier = provider.identifier;
	NSURL *versionRoot = provider.baseURL.URLByDeletingLastPathComponent;
	NFKRemoteCompletionBackend *backend = nil;
	if ([@[ @"openai", @"together", @"vllm", @"ollama", @"lmstudio" ] containsObject:identifier]) {
		backend = [self backendWithEndpointURL:[provider URLForPath:@"completions"]];
	} else if ([identifier isEqualToString:@"deepseek"]) {
		backend = [self backendWithEndpointURL:[versionRoot URLByAppendingPathComponent:@"beta/completions"]];
	} else if ([identifier isEqualToString:@"mistral"]) {
		backend = [self backendWithEndpointURL:[provider URLForPath:@"fim/completions"]];
		backend.apiStyle = NFKRemoteCompletionAPIStyleMistral;
	} else if ([identifier isEqualToString:@"llamacpp"]) {
		backend = [self backendWithEndpointURL:versionRoot];
		backend.apiStyle = NFKRemoteCompletionAPIStyleLlamaCpp;
	}
	backend.apiKey = apiKey;
	backend.modelName = modelName;
	return backend;
}

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_timeout = 120.0;
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
	return self.endpointURL != nil;
}

- (NSString *)backendIdentifier
{
	return @"remote-completion";
}

- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request
												  error:(NSError * _Nullable *)outError
{
	NSMutableURLRequest *urlRequest = [self urlRequestForRequest:request streaming:NO error:outError];
	if (urlRequest == nil) {
		return nil;
	}
	NSHTTPURLResponse *response = nil;
	NSError *sendError = nil;
	NSData *data = [self sendRequest:urlRequest response:&response error:&sendError];
	NSError *failure = data == nil ? sendError : [NFKRemoteTransport errorForResponse:response data:data];
	if (failure != nil || data == nil) {
		if (outError != NULL) {
			*outError = failure ?: [NFKRemoteTransport errorWithCode:kNFKError_InferenceBackendFailure reason:@"the completion call failed"];
		}
		return nil;
	}
	id body = [NSJSONSerialization JSONObjectWithData:data options:0 error:NULL];
	if (![body isKindOfClass:NSDictionary.class]) {
		return [self failWithCode:kNFKError_InferenceBackendFailure reason:@"the response is not a JSON object" error:outError];
	}
	NSString *text = [self textInBody:body];
	NSMutableDictionary<NSString *, id> *outputs = [NSMutableDictionary dictionaryWithObject:body forKey:NFKRemoteBackendRawKey];
	outputs[NFKOutputText] = text;
	return [NFKInferenceResult resultWithOutputs:outputs];
}

- (NFKInferenceJob *)submitInferenceJobForRequest:(NFKInferenceRequest *)request
{
	NFKInferenceJob *job = [[NFKInferenceJob alloc] init];
	NSError *error = nil;
	NSMutableURLRequest *urlRequest = [self urlRequestForRequest:request streaming:YES error:&error];
	if (urlRequest == nil) {
		[job finishWithError:error ?: [NFKRemoteTransport errorWithCode:kNFKError_InferenceBackendFailure reason:@"the completion call failed"]];
		return job;
	}
	[job reportProgress:-1.0];
	NSMutableString *text = [NSMutableString string];
	__block BOOL finished = NO;
	void (^cancel)(void) = [self streamRequest:urlRequest lineHandler:^(NSString *line) {
		NSString *payload = [NFKRemoteTransport SSEDataForLine:line];
		if (finished || payload == nil || [payload isEqualToString:@"[DONE]"]) {
			return;
		}
		id chunk = [NSJSONSerialization JSONObjectWithData:[payload dataUsingEncoding:NSUTF8StringEncoding] options:0 error:NULL];
		NSString *piece = [chunk isKindOfClass:NSDictionary.class] ? [self textInBody:chunk] : nil;
		if (piece.length > 0) {
			[text appendString:piece];
			[job reportProgress:-1.0 partialResult:[NFKInferenceResult resultWithOutputs:@{ NFKOutputText: [text copy] }]];
		}
	} completionHandler:^(NSHTTPURLResponse * _Nullable response, NSData * _Nullable errorBody, NSError * _Nullable streamError) {
		if (finished || job.status == NFKInferenceJobStatusCancelled) {
			return;
		}
		finished = YES;
		NSError *failure = streamError ?: [NFKRemoteTransport errorForResponse:response data:errorBody];
		if (failure != nil) {
			[job finishWithError:failure];
			return;
		}
		[job finishWithResult:[NFKInferenceResult resultWithOutputs:@{ NFKOutputText: [text copy] }]];
	}];
	job.cancellationHandler = cancel;
	return job;
}

#pragma mark Request

- (nullable NSMutableURLRequest *)urlRequestForRequest:(NFKInferenceRequest *)request
											 streaming:(BOOL)streaming
												 error:(NSError * _Nullable *)outError
{
	if (self.endpointURL == nil) {
		[self failWithCode:kNFKError_InferenceNotReady reason:@"no endpoint URL is set" error:outError];
		return nil;
	}
	NSString *prompt = request.prompt;
	if (prompt == nil) {
		[self failWithCode:kNFKError_InferenceMissingInput reason:@"the request carries no prompt" error:outError];
		return nil;
	}
	id suffix = [request inputForKey:NFKInputSuffix];
	NSString *suffixText = [suffix isKindOfClass:NSString.class] ? suffix : nil;
	NSMutableDictionary<NSString *, id> *body = [self bodyForPrompt:prompt suffix:suffixText request:request];
	if (streaming) {
		body[@"stream"] = @YES;
	}
	NSURL *url = self.endpointURL;
	if (self.apiStyle == NFKRemoteCompletionAPIStyleLlamaCpp) {
		url = [self.endpointURL URLByAppendingPathComponent:suffixText != nil ? @"infill" : @"completion"];
	}
	NSError *encodeError = nil;
	NSData *payload = [NSJSONSerialization dataWithJSONObject:body options:0 error:&encodeError];
	if (payload == nil) {
		if (outError != NULL) { *outError = encodeError; }
		return nil;
	}
	NSMutableURLRequest *urlRequest = [NSMutableURLRequest requestWithURL:url];
	urlRequest.HTTPMethod = @"POST";
	urlRequest.timeoutInterval = self.timeout;
	urlRequest.HTTPBody = payload;
	[urlRequest setValue:@"application/json" forHTTPHeaderField:@"Content-Type"];
	if (streaming) {
		[urlRequest setValue:@"text/event-stream" forHTTPHeaderField:@"Accept"];
	}
	[NFKRemoteTransport authorizeRequest:urlRequest apiKey:self.apiKey style:NFKRemoteAPIStyleOpenAIChat];
	return urlRequest;
}

// The contract's sampling keys in each style's spelling; llama.cpp names the length n_predict and
// the fill-in-the-middle halves input_prefix and input_suffix, Mistral the seed random_seed.
- (NSMutableDictionary<NSString *, id> *)bodyForPrompt:(NSString *)prompt suffix:(nullable NSString *)suffix request:(NFKInferenceRequest *)request
{
	NSDictionary *parameters = request.parameters;
	NSMutableDictionary<NSString *, id> *body = [NSMutableDictionary dictionary];
	BOOL llama = self.apiStyle == NFKRemoteCompletionAPIStyleLlamaCpp;
	if (self.modelName.length > 0) {
		body[@"model"] = self.modelName;
	}
	if (llama && suffix != nil) {
		body[@"input_prefix"] = prompt;
		body[@"input_suffix"] = suffix;
	} else {
		body[@"prompt"] = prompt;
		body[@"suffix"] = suffix;
	}
	body[llama ? @"n_predict" : @"max_tokens"] = parameters[NFKParameterMaxTokens];
	body[@"temperature"] = parameters[NFKParameterTemperature];
	body[@"top_p"] = parameters[NFKParameterTopP];
	body[@"top_k"] = parameters[NFKParameterTopK];
	body[@"stop"] = parameters[NFKParameterStopSequences];
	body[self.apiStyle == NFKRemoteCompletionAPIStyleMistral ? @"random_seed" : @"seed"] = parameters[NFKParameterSeed];
	NSSet<NSString *> *mapped = [NSSet setWithArray:@[ NFKParameterMaxTokens, NFKParameterTemperature, NFKParameterTopP,
													   NFKParameterTopK, NFKParameterStopSequences, NFKParameterSeed ]];
	for (NSString *key in parameters) {
		if (![mapped containsObject:key]) {
			body[key] = parameters[key];
		}
	}
	return body;
}

#pragma mark Response

// choices[].text (OpenAI style), choices[].message.content or a streamed choices[].delta.content
// (Mistral), or content (llama.cpp).
- (nullable NSString *)textInBody:(NSDictionary *)body
{
	if ([body[@"content"] isKindOfClass:NSString.class]) {
		return body[@"content"];
	}
	NSArray *choices = [body[@"choices"] isKindOfClass:NSArray.class] ? body[@"choices"] : nil;
	NSDictionary *choice = [choices.firstObject isKindOfClass:NSDictionary.class] ? choices.firstObject : nil;
	if ([choice[@"text"] isKindOfClass:NSString.class]) {
		return choice[@"text"];
	}
	for (NSString *key in @[ @"message", @"delta" ]) {
		NSDictionary *message = [choice[key] isKindOfClass:NSDictionary.class] ? choice[key] : nil;
		if ([message[@"content"] isKindOfClass:NSString.class]) {
			return message[@"content"];
		}
	}
	return nil;
}

#pragma mark Transport

- (nullable NSData *)sendRequest:(NSURLRequest *)request
						response:(NSHTTPURLResponse * _Nullable * _Nullable)outResponse
						   error:(NSError * _Nullable *)outError
{
	return [NFKRemoteTransport sendRequest:request session:self.session response:outResponse error:outError];
}

- (void (^)(void))streamRequest:(NSURLRequest *)request
					lineHandler:(void (^)(NSString *))lineHandler
			  completionHandler:(void (^)(NSHTTPURLResponse * _Nullable, NSData * _Nullable, NSError * _Nullable))completionHandler
{
	return [NFKRemoteTransport streamRequest:request session:self.session lineHandler:lineHandler completionHandler:completionHandler];
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
