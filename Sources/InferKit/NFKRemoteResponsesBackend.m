//
//  NFKRemoteResponsesBackend.m
//  InferKit
//

#import <InferKit/NFKRemoteResponsesBackend.h>
#import <InferKit/NFKRemoteProvider.h>
#import <InferKit/NFKRemoteTransport.h>
#import <InferKit/NFKRemoteBackend.h>
#import <InferKit/NFKImageCoding.h>
#import <InferKit/NFKInferenceRequest.h>
#import <InferKit/NFKInferenceResult.h>
#import <InferKit/NFKInferenceJob.h>
#import <InferKit/NFKInferenceKeys.h>
#import <InferKit/NFKErrors.h>
#import "NFKRemoteMediaSupport.h"
#import <InferKit/NFKRemoteFileStore.h>

/*! The contract keys this backend translates; every other parameter goes out under its own name. */
static NSSet<NSString *> *NFKResponsesMappedParameters(void)
{
	static NSSet<NSString *> *mapped;
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		mapped = [NSSet setWithArray:@[ NFKParameterTools, NFKParameterJSONSchema, NFKParameterReasoningEffort,
										NFKParameterMaxTokens, NFKParameterTemperature, NFKParameterTopP,
										NFKParameterPreviousResponseIdentifier ]];
	});
	return mapped;
}

/*! The status a background job ends in; any other status is still running. */
static BOOL NFKResponsesIsTerminalStatus(id status)
{
	return [@[ @"completed", @"failed", @"cancelled", @"incomplete" ] containsObject:status ?: @""];
}

@implementation NFKRemoteResponsesBackend

@synthesize session = _session;

+ (instancetype)backendWithEndpointURL:(nullable NSURL *)endpointURL
{
	NFKRemoteResponsesBackend *backend = [[self alloc] init];
	backend.endpointURL = endpointURL;
	return backend;
}

+ (nullable instancetype)backendForProvider:(NFKRemoteProvider *)provider
									 apiKey:(nullable NSString *)apiKey
								  modelName:(nullable NSString *)modelName
{
	NSArray<NSString *> *serving = @[ @"openai", @"xai", @"groq", @"deepseek", @"openrouter",
									  @"lmstudio", @"ollama", @"vllm", @"llamacpp" ];
	if (![serving containsObject:provider.identifier]) {
		return nil;
	}
	NFKRemoteResponsesBackend *backend = [self backendWithEndpointURL:[provider URLForPath:@"responses"]];
	backend.apiKey = apiKey;
	backend.modelName = modelName;
	return backend;
}

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_timeout = 600.0;
		_pollInterval = 2.0;
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
	return @"remote-responses";
}

- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request
												  error:(NSError * _Nullable *)outError
{
	NSMutableURLRequest *urlRequest = [self urlRequestForRequest:request streaming:NO error:outError];
	if (urlRequest == nil) {
		return nil;
	}
	NSDictionary *reply = [self JSONForRequest:urlRequest error:outError];
	while (reply != nil && self.runsInBackground && !NFKResponsesIsTerminalStatus(reply[@"status"])) {
		[NSThread sleepForTimeInterval:self.pollInterval];
		reply = [self JSONForRequest:[self pollRequestForReply:reply] error:outError];
	}
	return reply != nil ? [self resultFromReply:reply expectsStructured:[self expectsStructured:request] error:outError] : nil;
}

- (NFKInferenceJob *)submitInferenceJobForRequest:(NFKInferenceRequest *)request
{
	NFKInferenceJob *job = [[NFKInferenceJob alloc] init];
	if (self.runsInBackground) {
		dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
			[self runBackgroundJob:job forRequest:request];
		});
		return job;
	}
	NSError *error = nil;
	NSMutableURLRequest *urlRequest = [self urlRequestForRequest:request streaming:YES error:&error];
	if (urlRequest == nil) {
		[job finishWithError:error ?: [NFKRemoteTransport errorWithCode:kNFKError_InferenceBackendFailure reason:@"the responses call failed"]];
		return job;
	}
	[job reportProgress:-1.0];
	BOOL expectsStructured = [self expectsStructured:request];
	NSMutableString *text = [NSMutableString string];
	NSMutableString *reasoning = [NSMutableString string];
	__block NSDictionary *completed = nil;
	__block NSString *streamFailure = nil;
	__block BOOL finished = NO;
	void (^cancel)(void) = [self streamRequest:urlRequest lineHandler:^(NSString *line) {
		NSString *payload = [NFKRemoteTransport SSEDataForLine:line];
		id event = payload != nil ? [NSJSONSerialization JSONObjectWithData:[payload dataUsingEncoding:NSUTF8StringEncoding] options:0 error:NULL] : nil;
		if (finished || ![event isKindOfClass:NSDictionary.class]) {
			return;
		}
		NSString *type = [event[@"type"] isKindOfClass:NSString.class] ? event[@"type"] : @"";
		NSString *delta = [event[@"delta"] isKindOfClass:NSString.class] ? event[@"delta"] : nil;
		if ([type isEqualToString:@"response.output_text.delta"] && delta != nil) {
			[text appendString:delta];
		} else if ([type hasPrefix:@"response.reasoning"] && [type hasSuffix:@".delta"] && delta != nil) {
			[reasoning appendString:delta];
		} else if ([type isEqualToString:@"response.completed"] || [type isEqualToString:@"response.incomplete"]) {
			completed = [event[@"response"] isKindOfClass:NSDictionary.class] ? event[@"response"] : nil;
			return;
		} else if ([type isEqualToString:@"response.failed"] || [type isEqualToString:@"error"]) {
			NSDictionary *response = [event[@"response"] isKindOfClass:NSDictionary.class] ? event[@"response"] : event;
			streamFailure = [self failureReasonInReply:response] ?: @"the stream reported an error";
			return;
		} else {
			return;
		}
		NSMutableDictionary<NSString *, id> *partial = [NSMutableDictionary dictionaryWithObject:[text copy] forKey:NFKOutputText];
		if (reasoning.length > 0) {
			partial[NFKOutputReasoning] = [reasoning copy];
		}
		[job reportProgress:-1.0 partialResult:[NFKInferenceResult resultWithOutputs:partial]];
	} completionHandler:^(NSHTTPURLResponse * _Nullable response, NSData * _Nullable errorBody, NSError * _Nullable streamError) {
		if (finished || job.status == NFKInferenceJobStatusCancelled) {
			return;
		}
		finished = YES;
		NSError *failure = streamError ?: [NFKRemoteTransport errorForResponse:response data:errorBody];
		if (failure == nil && streamFailure != nil) {
			failure = [NFKRemoteTransport errorWithCode:kNFKError_InferenceBackendFailure reason:streamFailure];
		}
		NFKInferenceResult *result = nil;
		if (failure == nil) {
			result = completed != nil
				? [self resultFromReply:completed expectsStructured:expectsStructured error:&failure]
				: [NFKInferenceResult resultWithOutputs:@{ NFKOutputText: [text copy] }];
		}
		if (result != nil) {
			[job finishWithResult:result];
		} else {
			[job finishWithError:failure ?: [NFKRemoteTransport errorWithCode:kNFKError_InferenceBackendFailure reason:@"the responses call failed"]];
		}
	}];
	job.cancellationHandler = cancel;
	return job;
}

// A background job is created, then polled until it ends; cancelling the job cancels it on the
// service too.
- (void)runBackgroundJob:(NFKInferenceJob *)job forRequest:(NFKInferenceRequest *)request
{
	NSError *error = nil;
	NSMutableURLRequest *urlRequest = [self urlRequestForRequest:request streaming:NO error:&error];
	NSDictionary *reply = urlRequest != nil ? [self JSONForRequest:urlRequest error:&error] : nil;
	NSString *identifier = [reply[@"id"] isKindOfClass:NSString.class] ? reply[@"id"] : nil;
	if (identifier != nil) {
		job.cancellationHandler = ^{
			NSMutableURLRequest *cancel = [NSMutableURLRequest requestWithURL:[[self.endpointURL URLByAppendingPathComponent:identifier]
																				URLByAppendingPathComponent:@"cancel"]];
			cancel.HTTPMethod = @"POST";
			[NFKRemoteTransport authorizeRequest:cancel apiKey:self.apiKey style:NFKRemoteAPIStyleOpenAIChat];
			[self sendRequest:cancel response:NULL error:NULL];
		};
	}
	while (reply != nil && !NFKResponsesIsTerminalStatus(reply[@"status"])) {
		if (job.status == NFKInferenceJobStatusCancelled) {
			return;
		}
		[job reportProgress:-1.0];
		[NSThread sleepForTimeInterval:self.pollInterval];
		reply = [self JSONForRequest:[self pollRequestForReply:reply] error:&error];
	}
	NFKInferenceResult *result = reply != nil ? [self resultFromReply:reply expectsStructured:[self expectsStructured:request] error:&error] : nil;
	if (result != nil) {
		[job finishWithResult:result];
	} else {
		[job finishWithError:error ?: [NFKRemoteTransport errorWithCode:kNFKError_InferenceBackendFailure reason:@"the background job failed"]];
	}
}

#pragma mark Request

- (nullable NSMutableURLRequest *)urlRequestForRequest:(NFKInferenceRequest *)request
											 streaming:(BOOL)streaming
												 error:(NSError * _Nullable *)outError
{
	if (self.endpointURL == nil) {
		[self fail:outError code:kNFKError_InferenceNotReady reason:@"no endpoint URL is set"];
		return nil;
	}
	NFKRemoteAttachments *attachments = [NFKRemoteAttachments attachmentsForRequest:request error:outError];
	if (attachments == nil) {
		return nil;
	}
	if (attachments.audioData != nil) {
		[self fail:outError code:kNFKError_InferenceUnsupported reason:@"the Responses API takes no audio; use a chat backend with an audio model"];
		return nil;
	}
	NSMutableDictionary<NSString *, id> *body = [NSMutableDictionary dictionary];
	body[@"model"] = self.modelName;
	if (![self addInputForRequest:request attachments:attachments to:body]) {
		[self fail:outError code:kNFKError_InferenceMissingInput reason:@"the request carries neither a prompt nor messages"];
		return nil;
	}
	[self addParametersOfRequest:request to:body];
	if (streaming) {
		body[@"stream"] = @YES;
	}
	if (self.runsInBackground) {
		body[@"background"] = @YES;
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
	if (streaming) {
		[urlRequest setValue:@"text/event-stream" forHTTPHeaderField:@"Accept"];
	}
	[NFKRemoteTransport authorizeRequest:urlRequest apiKey:self.apiKey style:NFKRemoteAPIStyleOpenAIChat];
	return urlRequest;
}

// A system turn becomes instructions; each other turn an input message whose text is input_text
// (output_text for the assistant's); the media ride on the last user turn.
- (BOOL)addInputForRequest:(NFKInferenceRequest *)request attachments:(NFKRemoteAttachments *)attachments to:(NSMutableDictionary *)body
{
	NSArray *messages = request.messages;
	if (messages.count == 0 && request.prompt.length > 0) {
		messages = @[ @{ @"role": @"user", @"content": request.prompt } ];
	}
	NSMutableArray *items = [NSMutableArray array];
	for (NSDictionary *message in messages) {
		if (![message isKindOfClass:NSDictionary.class]) {
			continue;
		}
		if ([message[@"role"] isEqual:@"system"] && [message[@"content"] isKindOfClass:NSString.class]) {
			body[@"instructions"] = message[@"content"];
			continue;
		}
		NSString *partType = [message[@"role"] isEqual:@"assistant"] ? @"output_text" : @"input_text";
		id content = message[@"content"];
		NSArray *parts = [content isKindOfClass:NSString.class] ? @[ @{ @"type": partType, @"text": content } ]
					   : [content isKindOfClass:NSArray.class] ? content : @[];
		[items addObject:@{ @"role": message[@"role"] ?: @"user", @"content": parts }];
	}
	if (items.count == 0) {
		return NO;
	}
	if (!attachments.isEmpty) {
		NSInteger last = -1;
		for (NSInteger index = (NSInteger)items.count - 1; index >= 0; index--) {
			if ([items[index][@"role"] isEqual:@"user"]) {
				last = index;
				break;
			}
		}
		if (last >= 0) {
			NSMutableArray *parts = [items[last][@"content"] mutableCopy];
			for (NSData *png in attachments.imagePNGs) {
				[parts addObject:@{ @"type": @"input_image",
									@"image_url": [@"data:image/png;base64," stringByAppendingString:[png base64EncodedStringWithOptions:0]] }];
			}
			for (NSDictionary *document in attachments.documents) {
				[parts addObject:[self filePartForDocument:document]];
			}
			NSMutableDictionary *message = [items[last] mutableCopy];
			message[@"content"] = parts;
			items[last] = message;
		}
	}
	body[@"input"] = items;
	return YES;
}

- (NSDictionary *)filePartForDocument:(NSDictionary *)document
{
	NFKRemoteFile *file = document[@"fileReference"];
	if (file != nil) {
		return [file.mimeType hasPrefix:@"image/"] ? @{ @"type": @"input_image", @"file_id": file.identifier }
												   : @{ @"type": @"input_file", @"file_id": file.identifier };
	}
	NSData *data = document[@"data"];
	if ([document[@"mediaType"] isEqual:@"text/plain"]) {
		NSString *text = [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding] ?: @"";
		return @{ @"type": @"input_text", @"text": [NSString stringWithFormat:@"%@:\n%@", document[@"filename"], text] };
	}
	return @{ @"type": @"input_file", @"filename": document[@"filename"],
			  @"file_data": [@"data:application/pdf;base64," stringByAppendingString:[data base64EncodedStringWithOptions:0]] };
}

- (void)addParametersOfRequest:(NFKInferenceRequest *)request to:(NSMutableDictionary *)body
{
	NSDictionary *parameters = request.parameters;
	NSArray *tools = parameters[NFKParameterTools];
	if ([tools isKindOfClass:NSArray.class]) {
		NSMutableArray *wire = [NSMutableArray array];
		for (NSDictionary *tool in tools) {
			if (![tool isKindOfClass:NSDictionary.class]) {
				continue;
			}
			if (tool[@"type"] != nil) {
				[wire addObject:tool];
				continue;
			}
			NSMutableDictionary *function = [NSMutableDictionary dictionaryWithObject:@"function" forKey:@"type"];
			function[@"name"] = tool[@"name"];
			function[@"description"] = tool[@"description"];
			function[@"parameters"] = tool[@"parameters"] ?: @{ @"type": @"object", @"properties": @{} };
			[wire addObject:function];
		}
		body[@"tools"] = wire;
	}
	NSDictionary *schema = parameters[NFKParameterJSONSchema];
	if ([schema isKindOfClass:NSDictionary.class]) {
		body[@"text"] = @{ @"format": @{ @"type": @"json_schema", @"name": @"response", @"schema": schema } };
	}
	NSString *effort = [parameters[NFKParameterReasoningEffort] isKindOfClass:NSString.class] ? parameters[NFKParameterReasoningEffort] : nil;
	if (effort != nil) {
		NSDictionary<NSString *, NSString *> *levels = @{ NFKReasoningEffortLight: @"low", NFKReasoningEffortModerate: @"medium",
														  NFKReasoningEffortDeep: @"high" };
		body[@"reasoning"] = @{ @"effort": levels[effort] ?: effort, @"summary": @"auto" };
	}
	body[@"max_output_tokens"] = parameters[NFKParameterMaxTokens];
	body[@"previous_response_id"] = parameters[NFKParameterPreviousResponseIdentifier];
	// OpenAI's reasoning families refuse sampling unless the effort is none, and their default is not.
	BOOL reasoningFamily = NO;
	for (NSString *prefix in @[ @"gpt-5", @"gpt-6", @"o1", @"o3", @"o4" ]) {
		reasoningFamily = reasoningFamily || [self.modelName hasPrefix:prefix];
	}
	if (!reasoningFamily || [effort isEqualToString:@"none"]) {
		body[@"temperature"] = parameters[NFKParameterTemperature];
		body[@"top_p"] = parameters[NFKParameterTopP];
	}
	NSSet<NSString *> *mapped = NFKResponsesMappedParameters();
	for (NSString *key in parameters) {
		if (![mapped containsObject:key]) {
			body[key] = parameters[key];
		}
	}
}

- (BOOL)expectsStructured:(NFKInferenceRequest *)request
{
	return [request.parameters[NFKParameterJSONSchema] isKindOfClass:NSDictionary.class];
}

- (NSURLRequest *)pollRequestForReply:(NSDictionary *)reply
{
	NSString *identifier = [reply[@"id"] isKindOfClass:NSString.class] ? reply[@"id"] : @"";
	NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:[self.endpointURL URLByAppendingPathComponent:identifier]];
	request.HTTPMethod = @"GET";
	request.timeoutInterval = self.timeout;
	[NFKRemoteTransport authorizeRequest:request apiKey:self.apiKey style:NFKRemoteAPIStyleOpenAIChat];
	return request;
}

- (nullable NSDictionary *)JSONForRequest:(NSURLRequest *)request error:(NSError * _Nullable *)outError
{
	NSHTTPURLResponse *response = nil;
	NSError *sendError = nil;
	NSData *data = [self sendRequest:request response:&response error:&sendError];
	NSError *failure = data == nil ? sendError : [NFKRemoteTransport errorForResponse:response data:data];
	if (failure != nil || data == nil) {
		if (outError != NULL) {
			*outError = failure ?: [NFKRemoteTransport errorWithCode:kNFKError_InferenceBackendFailure reason:@"the responses call failed"];
		}
		return nil;
	}
	id reply = [NSJSONSerialization JSONObjectWithData:data options:0 error:NULL];
	return [reply isKindOfClass:NSDictionary.class] ? reply
		: [self fail:outError code:kNFKError_InferenceBackendFailure reason:@"the response is not a JSON object"];
}

#pragma mark Response

// The reply is a list of output items: messages, reasoning, function calls, and each built-in tool's
// call with its result.
- (nullable NFKInferenceResult *)resultFromReply:(NSDictionary *)reply expectsStructured:(BOOL)expectsStructured error:(NSError * _Nullable *)outError
{
	if ([reply[@"status"] isEqual:@"failed"]) {
		return [self fail:outError code:kNFKError_InferenceBackendFailure reason:[self failureReasonInReply:reply] ?: @"the response failed"];
	}
	NSMutableArray<NSString *> *texts = [NSMutableArray array];
	NSMutableArray<NSString *> *summaries = [NSMutableArray array];
	NSMutableArray<NSDictionary *> *toolCalls = [NSMutableArray array];
	NSMutableArray<NSDictionary *> *citations = [NSMutableArray array];
	NSMutableArray<NSDictionary *> *serverTools = [NSMutableArray array];
	NSMutableArray *images = [NSMutableArray array];
	for (NSDictionary *item in [reply[@"output"] isKindOfClass:NSArray.class] ? reply[@"output"] : @[]) {
		if (![item isKindOfClass:NSDictionary.class]) {
			continue;
		}
		NSString *type = [item[@"type"] isKindOfClass:NSString.class] ? item[@"type"] : @"";
		if ([type isEqualToString:@"message"]) {
			for (NSDictionary *part in [item[@"content"] isKindOfClass:NSArray.class] ? item[@"content"] : @[]) {
				if ([part[@"type"] isEqual:@"refusal"]) {
					NSString *refusal = [part[@"refusal"] isKindOfClass:NSString.class] ? part[@"refusal"] : @"the model declined to answer";
					return [self fail:outError code:kNFKError_InferenceRefused reason:refusal];
				}
				if ([part[@"text"] isKindOfClass:NSString.class]) {
					[texts addObject:part[@"text"]];
				}
				[self addCitationsOfPart:part to:citations];
			}
		} else if ([type isEqualToString:@"reasoning"]) {
			for (NSDictionary *summary in [item[@"summary"] isKindOfClass:NSArray.class] ? item[@"summary"] : @[]) {
				if ([summary isKindOfClass:NSDictionary.class] && [summary[@"text"] isKindOfClass:NSString.class]) {
					[summaries addObject:summary[@"text"]];
				}
			}
		} else if ([type isEqualToString:@"function_call"]) {
			NSString *arguments = [item[@"arguments"] isKindOfClass:NSString.class] ? item[@"arguments"] : @"{}";
			id parsed = [NSJSONSerialization JSONObjectWithData:[arguments dataUsingEncoding:NSUTF8StringEncoding] options:0 error:NULL];
			[toolCalls addObject:@{ @"id": item[@"call_id"] ?: item[@"id"] ?: @"", @"name": item[@"name"] ?: @"",
									@"arguments": [parsed isKindOfClass:NSDictionary.class] ? parsed : @{}, @"argumentsJSON": arguments }];
		} else if ([type isEqualToString:@"image_generation_call"]) {
			NSData *bytes = [item[@"result"] isKindOfClass:NSString.class]
				? [[NSData alloc] initWithBase64EncodedString:item[@"result"] options:NSDataBase64DecodingIgnoreUnknownCharacters] : nil;
			CVPixelBufferRef pixelBuffer = bytes != nil ? [NFKImageCoding pixelBufferWithImageData:bytes] : NULL;
			if (pixelBuffer != NULL) {
				[images addObject:(__bridge id)pixelBuffer];
				CVPixelBufferRelease(pixelBuffer);
			}
			[serverTools addObject:@{ @"name": type, @"input": item[@"revised_prompt"] ?: @"", @"raw": item }];
		} else if ([type hasSuffix:@"_call"]) {
			NSMutableDictionary *entry = [NSMutableDictionary dictionaryWithObject:item forKey:@"raw"];
			entry[@"name"] = type;
			entry[@"input"] = item[@"action"] ?: item[@"code"] ?: item[@"queries"] ?: item[@"arguments"];
			entry[@"output"] = item[@"results"] ?: item[@"outputs"] ?: item[@"output"];
			[serverTools addObject:entry];
		}
	}
	NSMutableDictionary<NSString *, id> *outputs = [NSMutableDictionary dictionaryWithObject:reply forKey:NFKRemoteBackendRawKey];
	NSString *text = [texts componentsJoinedByString:@""];
	if (text.length > 0) {
		outputs[NFKOutputText] = text;
	}
	if (expectsStructured && text.length > 0) {
		id structured = [NSJSONSerialization JSONObjectWithData:[text dataUsingEncoding:NSUTF8StringEncoding] options:0 error:NULL];
		if ([structured isKindOfClass:NSDictionary.class]) {
			outputs[NFKOutputStructured] = structured;
		}
	}
	if (summaries.count > 0) {
		outputs[NFKOutputReasoning] = [summaries componentsJoinedByString:@"\n"];
	}
	if (toolCalls.count > 0) {
		outputs[NFKOutputToolCalls] = toolCalls;
	}
	if (citations.count > 0) {
		outputs[NFKOutputCitations] = citations;
	}
	if (serverTools.count > 0) {
		outputs[NFKOutputServerToolResults] = serverTools;
	}
	if (images.count > 0) {
		outputs[NFKOutputImage] = images.firstObject;
	}
	if (images.count > 1) {
		outputs[NFKOutputImages] = images;
	}
	if ([reply[@"id"] isKindOfClass:NSString.class]) {
		outputs[NFKOutputResponseIdentifier] = reply[@"id"];
	}
	NSDictionary *usage = [reply[@"usage"] isKindOfClass:NSDictionary.class] ? reply[@"usage"] : nil;
	if (usage != nil) {
		NSDictionary *input = [usage[@"input_tokens_details"] isKindOfClass:NSDictionary.class] ? usage[@"input_tokens_details"] : nil;
		NSDictionary *output = [usage[@"output_tokens_details"] isKindOfClass:NSDictionary.class] ? usage[@"output_tokens_details"] : nil;
		NSDictionary *counts = NFKRemoteUsage(usage[@"input_tokens"], input[@"cached_tokens"], usage[@"output_tokens"], output[@"reasoning_tokens"]);
		if (counts != nil) {
			outputs[NFKOutputUsage] = counts;
		}
	}
	return [NFKInferenceResult resultWithOutputs:outputs];
}

- (void)addCitationsOfPart:(NSDictionary *)part to:(NSMutableArray<NSDictionary *> *)citations
{
	for (NSDictionary *annotation in [part[@"annotations"] isKindOfClass:NSArray.class] ? part[@"annotations"] : @[]) {
		if (![annotation isKindOfClass:NSDictionary.class] || ![annotation[@"type"] isEqual:@"url_citation"]) {
			continue;
		}
		NSMutableDictionary *citation = [NSMutableDictionary dictionaryWithObject:annotation forKey:@"raw"];
		citation[@"url"] = annotation[@"url"];
		citation[@"title"] = annotation[@"title"];
		citation[@"start"] = annotation[@"start_index"];
		citation[@"end"] = annotation[@"end_index"];
		[citations addObject:citation];
	}
}

- (nullable NSString *)failureReasonInReply:(NSDictionary *)reply
{
	id error = reply[@"error"];
	if ([error isKindOfClass:NSDictionary.class] && [error[@"message"] isKindOfClass:NSString.class]) {
		return error[@"message"];
	}
	return [error isKindOfClass:NSString.class] ? error : nil;
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

- (nullable id)fail:(NSError * _Nullable *)outError code:(NFKInferenceError)code reason:(NSString *)reason
{
	if (outError != NULL) {
		*outError = [NFKRemoteTransport errorWithCode:code reason:reason];
	}
	return nil;
}

@end
