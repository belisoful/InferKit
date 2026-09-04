//
//  NFKRemoteBackend.m
//  InferKit
//

#import "NFKRemoteBackend.h"
#import "NFKRemoteTransport.h"
#import "NFKRemoteMediaSupport.h"
#import "NFKInferenceRequest.h"
#import "NFKInferenceResult.h"
#import "NFKInferenceJob.h"
#import "NFKInferenceKeys.h"
#import "NFKAudioAsset.h"
#import "NFKErrors.h"

NSString * const NFKRemoteBackendPromptKey	= @"prompt";
NSString * const NFKRemoteBackendMessagesKey	= @"messages";
NSString * const NFKRemoteBackendTextKey		= @"text";
NSString * const NFKRemoteBackendRawKey		= @"raw";

/*! What a streamed reply assembles into: the text, the tool calls keyed by their index, and the
	spoken reply's base64 chunks and transcript. */
@interface NFKRemoteStreamState : NSObject
@property (nonatomic, strong) NSMutableString *text;
@property (nonatomic, strong) NSMutableDictionary<NSNumber *, NSMutableDictionary *> *toolCallsByIndex;
@property (nonatomic, strong) NSMutableString *audioBase64;
@property (nonatomic, strong) NSMutableString *transcript;
@property (nonatomic, assign) BOOL finished;
@end

@implementation NFKRemoteStreamState
- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_text = [NSMutableString string];
		_toolCallsByIndex = [NSMutableDictionary dictionary];
		_audioBase64 = [NSMutableString string];
		_transcript = [NSMutableString string];
	}
	return self;
}
@end

@implementation NFKRemoteBackend

@synthesize endpointURL = _endpointURL;
@synthesize apiKey = _apiKey;
@synthesize modelName = _modelName;
@synthesize session = _session;

+ (instancetype)backendWithEndpointURL:(nullable NSURL *)endpointURL
{
	NFKRemoteBackend *backend = [[self alloc] init];
	backend.endpointURL = endpointURL;
	return backend;
}

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_timeout = 60.0;
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
	return @"remote";
}

- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request
													error:(NSError * _Nullable *)outError
{
	NSError *error = nil;
	NSMutableURLRequest *urlRequest = [self urlRequestForRequest:request streaming:NO error:&error];
	if (urlRequest == nil) {
		[self propagateError:error to:outError];
		return nil;
	}

	NSHTTPURLResponse *response = nil;
	NSData *responseData = [self sendRequest:urlRequest response:&response error:&error];
	if (responseData == nil) {
		[self propagateError:error to:outError];
		return nil;
	}
	NSError *statusError = [NFKRemoteTransport errorForResponse:response data:responseData];
	if (statusError != nil) {
		[self propagateError:statusError to:outError];
		return nil;
	}
	return [self resultFromResponseData:responseData request:request error:outError];
}

- (NFKInferenceJob *)submitInferenceJobForRequest:(NFKInferenceRequest *)request
{
	NFKInferenceJob *job = [[NFKInferenceJob alloc] init];
	NSError *error = nil;
	NSMutableURLRequest *urlRequest = [self urlRequestForRequest:request streaming:YES error:&error];
	if (urlRequest == nil) {
		[job finishWithError:error];
		return job;
	}
	[job reportProgress:-1.0];

	NFKRemoteStreamState *state = [[NFKRemoteStreamState alloc] init];
	BOOL expectsStructured = [self requestExpectsStructuredReply:request];
	NSString *audioFormat = [self audioOutputFormatForRequest:request];
	void (^cancel)(void) = [self streamRequest:urlRequest lineHandler:^(NSString *line) {
		if (state.finished) {
			return;
		}
		NSString *payload = [NFKRemoteTransport SSEDataForLine:line];
		if (payload == nil) {
			return;
		}
		if ([payload isEqualToString:@"[DONE]"]) {
			state.finished = YES;
			[self finishJob:job fromStreamState:state expectsStructured:expectsStructured audioFormat:audioFormat];
			return;
		}
		id chunk = [NSJSONSerialization JSONObjectWithData:[payload dataUsingEncoding:NSUTF8StringEncoding] options:0 error:NULL];
		if ([chunk isKindOfClass:NSDictionary.class] && [self applyStreamChunk:chunk toState:state]) {
			NSString *text = state.text.length > 0 ? [state.text copy] : [state.transcript copy];
			[job reportProgress:-1.0 partialResult:[NFKInferenceResult resultWithOutputs:@{ NFKRemoteBackendTextKey: text }]];
		}
	} completionHandler:^(NSHTTPURLResponse * _Nullable response, NSData * _Nullable errorBody, NSError * _Nullable streamError) {
		if (state.finished || job.status == NFKInferenceJobStatusCancelled) {
			return;
		}
		state.finished = YES;
		NSError *failure = streamError ?: [NFKRemoteTransport errorForResponse:response data:errorBody];
		if (failure != nil) {
			[job finishWithError:failure];
			return;
		}
		// A stream that closes without [DONE] still delivered what it delivered.
		[self finishJob:job fromStreamState:state expectsStructured:expectsStructured audioFormat:audioFormat];
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
		[self setError:outError code:kNFKError_InferenceNotReady reason:@"no endpoint URL is set"];
		return nil;
	}
	NSDictionary<NSString *, id> *body = [self requestBodyForRequest:request streaming:streaming error:outError];
	if (body == nil) {
		return nil;
	}
	NSError *encodeError = nil;
	NSData *bodyData = [NSJSONSerialization dataWithJSONObject:body options:0 error:&encodeError];
	if (bodyData == nil) {
		[self propagateError:encodeError to:outError];
		return nil;
	}
	NSMutableURLRequest *urlRequest = [NSMutableURLRequest requestWithURL:self.endpointURL];
	urlRequest.HTTPMethod = @"POST";
	urlRequest.timeoutInterval = self.timeout;
	urlRequest.HTTPBody = bodyData;
	[urlRequest setValue:@"application/json" forHTTPHeaderField:@"Content-Type"];
	if (streaming) {
		[urlRequest setValue:@"text/event-stream" forHTTPHeaderField:@"Accept"];
	}
	[NFKRemoteTransport authorizeRequest:urlRequest apiKey:self.apiKey style:NFKRemoteAPIStyleOpenAIChat];
	return urlRequest;
}

- (nullable NSDictionary<NSString *, id> *)requestBodyForRequest:(NFKInferenceRequest *)request
													   streaming:(BOOL)streaming
														   error:(NSError * _Nullable *)outError
{
	NSMutableDictionary<NSString *, id> *body = [NSMutableDictionary dictionary];
	if (self.modelName.length > 0) {
		body[@"model"] = self.modelName;
	}

	NSArray *messages = [request inputForKey:NFKRemoteBackendMessagesKey];
	if (![messages isKindOfClass:NSArray.class]) {
		id prompt = [request inputForKey:NFKRemoteBackendPromptKey];
		NSString *text = [prompt isKindOfClass:NSString.class] ? prompt : @"";
		messages = @[ @{ @"role": @"user", @"content": text } ];
	}
	NFKRemoteAttachments *attachments = [NFKRemoteAttachments attachmentsForRequest:request error:outError];
	if (attachments == nil) {
		return nil;
	}
	if (!attachments.isEmpty) {
		messages = [self messages:messages attaching:attachments error:outError];
		if (messages == nil) {
			return nil;
		}
	}
	body[@"messages"] = messages;

	// The contract's tools and schema become the endpoint's shapes; everything else folds in by name.
	NSArray *tools = request.parameters[NFKParameterTools];
	if ([tools isKindOfClass:NSArray.class]) {
		// The contract's {name, description, parameters} is wrapped; an entry already in the wire
		// shape ({type: function, function: …}) passes through, so a caller who wrote it is not
		// wrapped twice.
		NSMutableArray *functions = [NSMutableArray array];
		for (NSDictionary *tool in tools) {
			if (![tool isKindOfClass:NSDictionary.class]) {
				continue;
			}
			[functions addObject:tool[@"type"] != nil ? tool : @{ @"type": @"function", @"function": tool }];
		}
		body[@"tools"] = functions;
	}
	NSDictionary *schema = request.parameters[NFKParameterJSONSchema];
	if ([schema isKindOfClass:NSDictionary.class]) {
		body[@"response_format"] = @{ @"type": @"json_schema",
									  @"json_schema": @{ @"name": @"response", @"schema": schema } };
	}
	// A spoken reply is asked for through the modalities list and the voice; the format defaults
	// to the container the speech backend writes.
	NSDictionary *audioOutput = request.parameters[NFKParameterAudioOutput];
	if ([audioOutput isKindOfClass:NSDictionary.class]) {
		NSMutableDictionary *audio = [audioOutput mutableCopy];
		if (audio[@"format"] == nil) {
			audio[@"format"] = @"wav";
		}
		body[@"audio"] = audio;
		body[@"modalities"] = @[ @"text", @"audio" ];
	}
	NSSet<NSString *> *translated = [NSSet setWithArray:@[ NFKParameterTools, NFKParameterJSONSchema,
														   NFKParameterAudioOutput, NFKParameterVideoFrameCount ]];
	for (NSString *key in request.parameters) {
		if (![translated containsObject:key]) {
			body[key] = request.parameters[key];
		}
	}
	if (streaming) {
		body[@"stream"] = @YES;
	}
	return body;
}

// Media rides on the last user turn as content parts beside the text: images inline as data
// URLs, audio as input_audio, documents as file parts — the shapes the endpoint reads.
- (nullable NSArray *)messages:(NSArray *)messages attaching:(NFKRemoteAttachments *)attachments error:(NSError * _Nullable *)outError
{
	NSInteger index = [self indexOfLastUserMessageIn:messages];
	if (index < 0) {
		[self setError:outError code:kNFKError_InferenceMissingInput reason:@"no user message to attach the media to"];
		return nil;
	}
	NSDictionary *message = messages[index];
	NSMutableArray *parts = [NSMutableArray array];
	if ([message[@"content"] isKindOfClass:NSString.class]) {
		[parts addObject:@{ @"type": @"text", @"text": message[@"content"] }];
	} else if ([message[@"content"] isKindOfClass:NSArray.class]) {
		[parts addObjectsFromArray:message[@"content"]];
	}
	for (NSData *png in attachments.imagePNGs) {
		NSString *dataURL = [@"data:image/png;base64," stringByAppendingString:[png base64EncodedStringWithOptions:0]];
		[parts addObject:@{ @"type": @"image_url", @"image_url": @{ @"url": dataURL } }];
	}
	if (attachments.audioData != nil) {
		[parts addObject:@{ @"type": @"input_audio",
							@"input_audio": @{ @"data": [attachments.audioData base64EncodedStringWithOptions:0],
											   @"format": attachments.audioFormat ?: @"wav" } }];
	}
	for (NSDictionary *document in attachments.documents) {
		NSString *dataURL = [@"data:application/pdf;base64," stringByAppendingString:[document[@"data"] base64EncodedStringWithOptions:0]];
		[parts addObject:@{ @"type": @"file", @"file": @{ @"filename": document[@"filename"], @"file_data": dataURL } }];
	}
	NSMutableDictionary *attached = [message mutableCopy];
	attached[@"content"] = parts;
	NSMutableArray *result = [messages mutableCopy];
	result[index] = attached;
	return result;
}

- (NSInteger)indexOfLastUserMessageIn:(NSArray *)messages
{
	for (NSInteger index = (NSInteger)messages.count - 1; index >= 0; index--) {
		NSDictionary *message = messages[index];
		if ([message isKindOfClass:NSDictionary.class] && [message[@"role"] isEqual:@"user"]) {
			return index;
		}
	}
	return -1;
}

// A schema, or a response_format the caller folded in by name, is a promise the reply is JSON.
- (BOOL)requestExpectsStructuredReply:(NFKInferenceRequest *)request
{
	if ([request.parameters[NFKParameterJSONSchema] isKindOfClass:NSDictionary.class]) {
		return YES;
	}
	NSDictionary *format = request.parameters[@"response_format"];
	NSString *type = [format isKindOfClass:NSDictionary.class] ? format[@"type"] : nil;
	return [type isEqual:@"json_object"] || [type isEqual:@"json_schema"];
}

- (nullable NSString *)audioOutputFormatForRequest:(NFKInferenceRequest *)request
{
	NSDictionary *audioOutput = request.parameters[NFKParameterAudioOutput];
	if (![audioOutput isKindOfClass:NSDictionary.class]) {
		return nil;
	}
	return [audioOutput[@"format"] isKindOfClass:NSString.class] ? audioOutput[@"format"] : @"wav";
}

#pragma mark Response

- (nullable NFKInferenceResult *)resultFromResponseData:(NSData *)data
												request:(NFKInferenceRequest *)request
												  error:(NSError * _Nullable *)outError
{
	id object = [NSJSONSerialization JSONObjectWithData:data options:0 error:NULL];
	if (![object isKindOfClass:NSDictionary.class]) {
		[self setError:outError code:kNFKError_InferenceBackendFailure reason:@"the response is not a JSON object"];
		return nil;
	}
	NSDictionary *responseBody = object;
	NSDictionary *message = nil;
	NSString *content = nil;
	NSArray *choices = responseBody[@"choices"];
	if ([choices isKindOfClass:NSArray.class] && choices.count > 0) {
		NSDictionary *choice = choices.firstObject;
		if ([choice isKindOfClass:NSDictionary.class]) {
			message = [choice[@"message"] isKindOfClass:NSDictionary.class] ? choice[@"message"] : nil;
			if ([message[@"content"] isKindOfClass:NSString.class]) {
				content = message[@"content"];
			} else if ([choice[@"text"] isKindOfClass:NSString.class]) {
				content = choice[@"text"];
			}
		}
	}

	// A spoken reply carries its bytes and transcript under message.audio; the transcript stands
	// in for the text, which the endpoint leaves null then.
	NSDictionary *audio = [message[@"audio"] isKindOfClass:NSDictionary.class] ? message[@"audio"] : nil;
	NSString *audioBase64 = [audio[@"data"] isKindOfClass:NSString.class] ? audio[@"data"] : nil;
	if (content == nil && [audio[@"transcript"] isKindOfClass:NSString.class]) {
		content = audio[@"transcript"];
	}
	return [self resultWithText:content
					  wireCalls:message[@"tool_calls"]
					audioBase64:audioBase64
					audioFormat:[self audioOutputFormatForRequest:request]
			  expectsStructured:[self requestExpectsStructuredReply:request]
							raw:responseBody
						  error:outError];
}

- (nullable NFKInferenceResult *)resultWithText:(nullable NSString *)text
									  wireCalls:(nullable NSArray *)wireCalls
									audioBase64:(nullable NSString *)audioBase64
									audioFormat:(nullable NSString *)audioFormat
							  expectsStructured:(BOOL)expectsStructured
											raw:(id)raw
										  error:(NSError * _Nullable *)outError
{
	NSMutableDictionary<NSString *, id> *outputs = [NSMutableDictionary dictionary];
	if (text.length > 0) {
		outputs[NFKRemoteBackendTextKey] = text;
	}
	NSArray *toolCalls = [self toolCallsFromWireCalls:wireCalls];
	if (toolCalls.count > 0) {
		outputs[NFKOutputToolCalls] = toolCalls;
	}
	if (expectsStructured) {
		NSDictionary *structured = [self structuredObjectInText:text];
		if (structured != nil) {
			outputs[NFKOutputStructured] = structured;
		}
	}
	if (audioBase64.length > 0) {
		NSData *bytes = [[NSData alloc] initWithBase64EncodedString:audioBase64 options:NSDataBase64DecodingIgnoreUnknownCharacters];
		NSURL *fileURL = bytes != nil ? NFKRemoteWriteMediaFile(bytes, @"reply", audioFormat ?: @"wav", nil, outError) : nil;
		if (fileURL == nil) {
			if (bytes != nil) {
				return nil;		// the write failed and reported why
			}
			[self setError:outError code:kNFKError_InferenceBackendFailure reason:@"the spoken reply could not be decoded"];
			return nil;
		}
		outputs[NFKOutputAudio] = [NFKAudioAsset audioAssetWithFileURL:fileURL];
	}
	outputs[NFKRemoteBackendRawKey] = raw;
	return [NFKInferenceResult resultWithOutputs:outputs];
}

// The wire form is {id, type, function: {name, arguments}} with arguments a JSON string.
- (NSArray<NSDictionary *> *)toolCallsFromWireCalls:(nullable NSArray *)wireCalls
{
	NSMutableArray<NSDictionary *> *calls = [NSMutableArray array];
	if (![wireCalls isKindOfClass:NSArray.class]) {
		return calls;
	}
	for (NSDictionary *call in wireCalls) {
		if (![call isKindOfClass:NSDictionary.class]) {
			continue;
		}
		NSDictionary *function = [call[@"function"] isKindOfClass:NSDictionary.class] ? call[@"function"] : @{};
		NSString *name = [function[@"name"] isKindOfClass:NSString.class] ? function[@"name"] : nil;
		if (name == nil) {
			continue;
		}
		NSString *argumentsJSON = [function[@"arguments"] isKindOfClass:NSString.class] ? function[@"arguments"] : @"{}";
		NSMutableDictionary *entry = [NSMutableDictionary dictionary];
		entry[@"id"] = [call[@"id"] isKindOfClass:NSString.class] ? call[@"id"] : @"";
		entry[@"name"] = name;
		entry[@"arguments"] = [self structuredObjectInText:argumentsJSON] ?: @{};
		entry[@"argumentsJSON"] = argumentsJSON;
		[calls addObject:entry];
	}
	return calls;
}

- (nullable NSDictionary *)structuredObjectInText:(nullable NSString *)text
{
	if (text.length == 0) {
		return nil;
	}
	id object = [NSJSONSerialization JSONObjectWithData:[text dataUsingEncoding:NSUTF8StringEncoding] options:0 error:NULL];
	return [object isKindOfClass:NSDictionary.class] ? object : nil;
}

#pragma mark Streaming

// Returns whether the chunk carried text or transcript, which is what a partial result is worth
// reporting for.
- (BOOL)applyStreamChunk:(NSDictionary *)chunk toState:(NFKRemoteStreamState *)state
{
	NSArray *choices = chunk[@"choices"];
	NSDictionary *choice = [choices isKindOfClass:NSArray.class] && choices.count > 0 ? choices.firstObject : nil;
	NSDictionary *delta = [choice isKindOfClass:NSDictionary.class] && [choice[@"delta"] isKindOfClass:NSDictionary.class]
		? choice[@"delta"] : nil;
	if (delta == nil) {
		return NO;
	}
	BOOL carriedText = NO;
	if ([delta[@"content"] isKindOfClass:NSString.class] && [delta[@"content"] length] > 0) {
		[state.text appendString:delta[@"content"]];
		carriedText = YES;
	}
	// A spoken reply streams as base64 chunks that concatenate, beside its transcript's pieces.
	NSDictionary *audio = [delta[@"audio"] isKindOfClass:NSDictionary.class] ? delta[@"audio"] : nil;
	if ([audio[@"data"] isKindOfClass:NSString.class]) {
		[state.audioBase64 appendString:audio[@"data"]];
	}
	if ([audio[@"transcript"] isKindOfClass:NSString.class] && [audio[@"transcript"] length] > 0) {
		[state.transcript appendString:audio[@"transcript"]];
		carriedText = YES;
	}
	// A tool call's id and name arrive in its first delta and its arguments in fragments after.
	NSArray *calls = delta[@"tool_calls"];
	if ([calls isKindOfClass:NSArray.class]) {
		for (NSDictionary *call in calls) {
			if (![call isKindOfClass:NSDictionary.class]) {
				continue;
			}
			NSNumber *index = [call[@"index"] isKindOfClass:NSNumber.class] ? call[@"index"] : @(state.toolCallsByIndex.count);
			NSMutableDictionary *assembled = state.toolCallsByIndex[index];
			if (assembled == nil) {
				assembled = [@{ @"id": @"", @"function": [@{ @"name": @"", @"arguments": @"" } mutableCopy] } mutableCopy];
				state.toolCallsByIndex[index] = assembled;
			}
			if ([call[@"id"] isKindOfClass:NSString.class]) {
				assembled[@"id"] = call[@"id"];
			}
			NSDictionary *function = [call[@"function"] isKindOfClass:NSDictionary.class] ? call[@"function"] : nil;
			NSMutableDictionary *assembledFunction = assembled[@"function"];
			if ([function[@"name"] isKindOfClass:NSString.class]) {
				assembledFunction[@"name"] = function[@"name"];
			}
			if ([function[@"arguments"] isKindOfClass:NSString.class]) {
				assembledFunction[@"arguments"] = [assembledFunction[@"arguments"] stringByAppendingString:function[@"arguments"]];
			}
		}
	}
	return carriedText;
}

- (void)finishJob:(NFKInferenceJob *)job
   fromStreamState:(NFKRemoteStreamState *)state
 expectsStructured:(BOOL)expectsStructured
	   audioFormat:(nullable NSString *)audioFormat
{
	NSArray *orderedIndexes = [state.toolCallsByIndex.allKeys sortedArrayUsingSelector:@selector(compare:)];
	NSMutableArray *wireCalls = [NSMutableArray array];
	for (NSNumber *index in orderedIndexes) {
		[wireCalls addObject:state.toolCallsByIndex[index]];
	}
	NSString *text = state.text.length > 0 ? [state.text copy] : [state.transcript copy];
	NSMutableDictionary *message = [NSMutableDictionary dictionaryWithObject:@"assistant" forKey:@"role"];
	message[@"content"] = [state.text copy];
	if (wireCalls.count > 0) {
		message[@"tool_calls"] = wireCalls;
	}
	NSError *error = nil;
	NFKInferenceResult *result = [self resultWithText:text
											wireCalls:wireCalls
										  audioBase64:state.audioBase64.length > 0 ? [state.audioBase64 copy] : nil
										  audioFormat:audioFormat
									expectsStructured:expectsStructured
												  raw:message
												error:&error];
	if (result != nil) {
		[job finishWithResult:result];
	} else {
		[job finishWithError:error ?: [NFKRemoteTransport errorWithCode:kNFKError_InferenceBackendFailure
																 reason:@"the streamed reply could not be assembled"]];
	}
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

- (BOOL)setError:(NSError * _Nullable *)outError code:(NSInteger)code reason:(NSString *)reason
{
	if (outError != NULL) {
		*outError = [NSError errorWithDomain:NFKInferenceErrorDomain
										code:code
									userInfo:@{ NSLocalizedDescriptionKey: reason }];
	}
	return NO;
}

- (BOOL)propagateError:(nullable NSError *)error to:(NSError * _Nullable *)outError
{
	if (outError == NULL) {
		return NO;
	}
	*outError = error != nil ? error
							: [NSError errorWithDomain:NFKInferenceErrorDomain
												  code:kNFKError_InferenceBackendFailure
											  userInfo:@{ NSLocalizedDescriptionKey: @"the remote call failed" }];
	return NO;
}

@end
