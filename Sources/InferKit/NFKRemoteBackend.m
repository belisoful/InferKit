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
#import "NFKImageCoding.h"
#import "NFKRemoteFileStore.h"
#import "NFKErrors.h"

/*! The endpoint's spelling for each core text parameter the contract names. The core keys are
	camelCase and an OpenAI-compatible service reads underscored names, so the key is renamed and the
	value passes through unchanged. The repetition penalty goes out under both spellings the servers
	use: they mean the same multiplicative penalty, and no server reads both. */
static NSDictionary<NSString *, NSArray<NSString *> *> *NFKRemoteWireNames(void)
{
	static NSDictionary<NSString *, NSArray<NSString *> *> *names;
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		names = @{ NFKParameterMaxTokens: @[ @"max_tokens" ],
				   NFKParameterTopP: @[ @"top_p" ],
				   NFKParameterTopK: @[ @"top_k" ],
				   NFKParameterStopSequences: @[ @"stop" ],
				   NFKParameterRepetitionPenalty: @[ @"repetition_penalty", @"repeat_penalty" ] };
	});
	return names;
}

/*! The endpoint's spelling for each reasoning effort the contract names. An OpenAI-compatible
	service reads low / medium / high under reasoning_effort, so the three core levels are renamed and
	any other string goes out as written, which reaches a service that names its own levels. */
static NSDictionary<NSString *, NSString *> *NFKRemoteReasoningEfforts(void)
{
	static NSDictionary<NSString *, NSString *> *efforts;
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		efforts = @{ NFKReasoningEffortLight: @"low",
					 NFKReasoningEffortModerate: @"medium",
					 NFKReasoningEffortDeep: @"high" };
	});
	return efforts;
}

/*! The model-name prefixes of OpenAI's reasoning families, which refuse max_tokens on Chat
	Completions and read the limit as max_completion_tokens. A prefix rather than a substring, so a
	router's namespaced name (openai/gpt-5.6-sol) keeps the spelling the router reads. */
static NSArray<NSString *> *NFKRemoteCompletionTokenModelPrefixes(void)
{
	return @[ @"gpt-5", @"gpt-6", @"o1", @"o3", @"o4" ];
}

/*! The sampling fields OpenAI's reasoning families refuse unless the reasoning effort is none. */
static NSArray<NSString *> *NFKRemoteReasoningRefusedSamplingFields(void)
{
	return @[ @"temperature", @"top_p", @"logprobs", @"top_logprobs" ];
}

/*! Whether xAI serves the model as a reasoning model, which refuses stop sequences: Grok 4 onward
	and Grok 3 mini, except the variants named non-reasoning. */
static BOOL NFKRemoteModelRefusesStopSequences(NSString * _Nullable model)
{
	if ([model containsString:@"non-reasoning"]) {
		return NO;
	}
	return [model hasPrefix:@"grok-4"] || [model hasPrefix:@"grok-3-mini"];
}

/*! The text up to the earliest of the stop sequences, which is what the endpoint returns when it
	applies them itself. */
static NSString *NFKRemoteTextBeforeStops(NSString *text, NSArray<NSString *> *stops)
{
	NSUInteger end = text.length;
	for (NSString *stop in stops) {
		if (![stop isKindOfClass:NSString.class] || stop.length == 0) {
			continue;
		}
		NSRange found = [text rangeOfString:stop];
		if (found.location != NSNotFound && found.location < end) {
			end = found.location;
		}
	}
	return [text substringToIndex:end];
}

NSString * const NFKRemoteBackendPromptKey	= @"prompt";
NSString * const NFKRemoteBackendMessagesKey	= @"messages";
NSString * const NFKRemoteBackendTextKey		= @"text";
NSString * const NFKRemoteBackendRawKey		= @"raw";

/*! What a streamed reply assembles into: the text, the reasoning shown before it, the tool calls
	keyed by their index, the spoken reply's base64 chunks and transcript, and the token counts the
	last chunk reports. */
@interface NFKRemoteStreamState : NSObject
@property (nonatomic, strong) NSMutableString *text;
@property (nonatomic, strong) NSMutableString *reasoning;
@property (nonatomic, copy, nullable) NSDictionary<NSString *, NSNumber *> *usage;
@property (nonatomic, strong) NSMutableDictionary<NSNumber *, NSMutableDictionary *> *toolCallsByIndex;
@property (nonatomic, strong) NSMutableString *audioBase64;
@property (nonatomic, strong) NSMutableString *transcript;
@property (nonatomic, copy, nullable) NSString *finishReason;
@property (nonatomic, copy, nullable) NSString *refusal;
@property (nonatomic, copy, nullable) NSArray<NSString *> *clientStops;
@property (nonatomic, assign) BOOL finished;
@end

@implementation NFKRemoteStreamState
- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_text = [NSMutableString string];
		_reasoning = [NSMutableString string];
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

- (NSSet<NSString *> *)supportedParameterKeys
{
	// Tools, the schema, the audio reply, and the frame count become the endpoint's own shapes; the
	// text parameters are renamed to the endpoint's spelling; temperature and seed already carry it.
	NSMutableSet<NSString *> *keys = [NSMutableSet setWithArray:@[ NFKParameterTools, NFKParameterJSONSchema,
																   NFKParameterAudioOutput, NFKParameterVideoFrameCount,
																   NFKParameterTemperature, NFKParameterSeed,
																   NFKParameterReasoningEffort ]];
	[keys addObjectsFromArray:NFKRemoteWireNames().allKeys];
	return keys;
}

- (NSSet<NSString *> *)supportedInputKeys
{
	return [NSSet setWithArray:@[ NFKInputPrompt, NFKInputMessages, NFKInputImage, NFKInputImages,
								  NFKInputVideo, NFKInputAudio, NFKInputDocument, NFKInputDocuments ]];
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
	state.clientStops = [self clientStopsForRequest:request];
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
			NSMutableDictionary<NSString *, id> *partial = [NSMutableDictionary dictionaryWithObject:text forKey:NFKRemoteBackendTextKey];
			if (state.reasoning.length > 0) {
				partial[NFKOutputReasoning] = [state.reasoning copy];
			}
			[job reportProgress:-1.0 partialResult:[NFKInferenceResult resultWithOutputs:partial]];
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

- (NSArray<NSString *> *)wireNamesForKey:(NSString *)key among:(NSDictionary<NSString *, NSArray<NSString *> *> *)wireNames
{
	if ([key isEqualToString:NFKParameterMaxTokens] && self.isOpenAIReasoningModel) {
		return @[ @"max_completion_tokens" ];
	}
	return wireNames[key];
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
	BOOL readsWholeVideo = self.chatDialect == NFKRemoteChatDialectOpenRouter || self.chatDialect == NFKRemoteChatDialectVLLM
		|| self.chatDialect == NFKRemoteChatDialectLlamaCpp;
	NFKRemoteAttachments *attachments = [NFKRemoteAttachments attachmentsForRequest:request keepsVideo:readsWholeVideo error:outError];
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
	if (request.outputModality == NFKModalityImage && request.parameters[@"modalities"] == nil) {
		body[@"modalities"] = @[ @"image", @"text" ];
	}

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
	// The contract's text parameters carry the endpoint's spelling. These are written before the fold
	// below, so a caller who sets the endpoint's own name keeps the value they wrote.
	NSDictionary<NSString *, NSArray<NSString *> *> *wireNames = NFKRemoteWireNames();
	for (NSString *key in wireNames) {
		id value = request.parameters[key];
		if (value == nil) {
			continue;
		}
		for (NSString *wireName in [self wireNamesForKey:key among:wireNames]) {
			body[wireName] = value;
		}
	}

	// The reasoning effort is renamed and its value translated, so the contract's levels reach a
	// service that names its own.
	NSString *effort = request.parameters[NFKParameterReasoningEffort];
	if ([effort isKindOfClass:NSString.class]) {
		body[@"reasoning_effort"] = NFKRemoteReasoningEfforts()[effort] ?: effort;
	}

	NSMutableSet<NSString *> *translated = [NSMutableSet setWithArray:@[ NFKParameterTools, NFKParameterJSONSchema,
																		 NFKParameterAudioOutput, NFKParameterVideoFrameCount,
																		   NFKParameterReasoningEffort ]];
	[translated addObjectsFromArray:wireNames.allKeys];
	for (NSString *key in request.parameters) {
		if (![translated containsObject:key]) {
			body[key] = request.parameters[key];
		}
	}
	[self removeFieldsTheModelRefusesFromBody:body];
	if (streaming) {
		body[@"stream"] = @YES;
	}
	return body;
}

/*! Removes what the named model answers with a 400. OpenAI's reasoning families take no sampling
	unless the effort is none, and their default effort is not none. xAI's reasoning models take no
	stop sequences, which the backend applies to the reply itself instead. */
- (void)removeFieldsTheModelRefusesFromBody:(NSMutableDictionary<NSString *, id> *)body
{
	if (self.isOpenAIReasoningModel && ![body[@"reasoning_effort"] isEqual:@"none"]) {
		[body removeObjectsForKeys:NFKRemoteReasoningRefusedSamplingFields()];
	}
	if (NFKRemoteModelRefusesStopSequences(self.modelName)) {
		[body removeObjectForKey:@"stop"];
	}
}

- (BOOL)isOpenAIReasoningModel
{
	for (NSString *prefix in NFKRemoteCompletionTokenModelPrefixes()) {
		if ([self.modelName hasPrefix:prefix]) {
			return YES;
		}
	}
	return NO;
}

/*! The stop sequences the backend applies to the reply, for a model that refuses them on the wire. */
- (nullable NSArray<NSString *> *)clientStopsForRequest:(NFKInferenceRequest *)request
{
	if (!NFKRemoteModelRefusesStopSequences(self.modelName)) {
		return nil;
	}
	id stops = request.parameters[NFKParameterStopSequences] ?: request.parameters[@"stop"];
	if ([stops isKindOfClass:NSString.class]) {
		return @[ stops ];
	}
	return [stops isKindOfClass:NSArray.class] ? stops : nil;
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
	if (attachments.videoData != nil) {
		[parts addObject:[self videoPartForData:attachments.videoData format:attachments.videoFormat ?: @"mp4"]];
	}
	if (attachments.audioData != nil) {
		NSString *audio = [attachments.audioData base64EncodedStringWithOptions:0];
		[parts addObject:self.chatDialect == NFKRemoteChatDialectMistral
			? @{ @"type": @"input_audio", @"input_audio": audio }
			: @{ @"type": @"input_audio", @"input_audio": @{ @"data": audio, @"format": attachments.audioFormat ?: @"wav" } }];
	}
	for (NSDictionary *document in attachments.documents) {
		NSDictionary *part = [self partForDocument:document error:outError];
		if (part == nil) {
			return nil;
		}
		[parts addObject:part];
	}
	NSMutableDictionary *attached = [message mutableCopy];
	attached[@"content"] = parts;
	NSMutableArray *result = [messages mutableCopy];
	result[index] = attached;
	return result;
}

// OpenRouter and vLLM read a whole clip as video_url; llama.cpp reads input_video beside input_audio.
- (NSDictionary *)videoPartForData:(NSData *)video format:(NSString *)format
{
	NSString *encoded = [video base64EncodedStringWithOptions:0];
	if (self.chatDialect == NFKRemoteChatDialectLlamaCpp) {
		return @{ @"type": @"input_video", @"input_video": @{ @"data": encoded, @"format": format } };
	}
	NSString *dataURL = [NSString stringWithFormat:@"data:video/%@;base64,%@", format, encoded];
	return @{ @"type": @"video_url", @"video_url": @{ @"url": dataURL } };
}

// A PDF is a file part (a document_url on Mistral); a plain-text document is a text part, since no
// chat-completions service reads text files as parts.
- (nullable NSDictionary *)partForDocument:(NSDictionary *)document error:(NSError * _Nullable *)outError
{
	NFKRemoteFile *file = document[@"fileReference"];
	if (file != nil) {
		return [self partForFile:file error:outError];
	}
	NSData *data = document[@"data"];
	if ([document[@"mediaType"] isEqual:@"text/plain"]) {
		NSString *text = [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding] ?: @"";
		return @{ @"type": @"text", @"text": [NSString stringWithFormat:@"%@:\n%@", document[@"filename"], text] };
	}
	NSString *dataURL = [@"data:application/pdf;base64," stringByAppendingString:[data base64EncodedStringWithOptions:0]];
	if (self.chatDialect == NFKRemoteChatDialectMistral) {
		return @{ @"type": @"document_url", @"document_url": dataURL, @"document_name": document[@"filename"] };
	}
	return @{ @"type": @"file", @"file": @{ @"filename": document[@"filename"], @"file_data": dataURL } };
}

// A file the service keeps rides by its id; Mistral's chat reads documents by URL only, so its file is
// named by a signed URL fetched from the files endpoint beside the chat one.
- (nullable NSDictionary *)partForFile:(NFKRemoteFile *)file error:(NSError * _Nullable *)outError
{
	switch (self.chatDialect) {
		case NFKRemoteChatDialectDeepSeek:
			return @{ @"type": @"file", @"file_id": file.identifier };
		case NFKRemoteChatDialectMistral: {
			NSURL *chatRoot = self.endpointURL.URLByDeletingLastPathComponent.URLByDeletingLastPathComponent;
			NSURL *signing = [[[chatRoot URLByAppendingPathComponent:@"files"] URLByAppendingPathComponent:file.identifier] URLByAppendingPathComponent:@"url"];
			NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:signing];
			request.timeoutInterval = self.timeout;
			[NFKRemoteTransport authorizeRequest:request apiKey:self.apiKey style:NFKRemoteAPIStyleOpenAIChat];
			NSHTTPURLResponse *response = nil;
			NSError *sendError = nil;
			NSData *data = [self sendRequest:request response:&response error:&sendError];
			NSError *failure = data == nil ? sendError : [NFKRemoteTransport errorForResponse:response data:data];
			id reply = failure == nil ? [NSJSONSerialization JSONObjectWithData:data options:0 error:NULL] : nil;
			NSString *signedURL = [reply isKindOfClass:NSDictionary.class] && [reply[@"url"] isKindOfClass:NSString.class] ? reply[@"url"] : nil;
			if (signedURL == nil) {
				[self setError:outError code:kNFKError_InferenceBackendFailure
						reason:failure.localizedDescription ?: @"Mistral signed no URL for the file"];
				return nil;
			}
			return @{ @"type": @"document_url", @"document_url": signedURL };
		}
		default:
			return @{ @"type": @"file", @"file": @{ @"file_id": file.identifier } };
	}
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
	NSMutableString *chunkReasoning = [NSMutableString string];
	NSArray *choices = responseBody[@"choices"];
	if ([choices isKindOfClass:NSArray.class] && choices.count > 0) {
		NSDictionary *choice = choices.firstObject;
		if ([choice isKindOfClass:NSDictionary.class]) {
			message = [choice[@"message"] isKindOfClass:NSDictionary.class] ? choice[@"message"] : nil;
			NSString *refusal = [self refusalInChoice:choice message:message];
			if (refusal != nil) {
				[self setError:outError code:kNFKError_InferenceRefused reason:refusal];
				return nil;
			}
			if ([message[@"content"] isKindOfClass:NSString.class]) {
				content = message[@"content"];
			} else if ([message[@"content"] isKindOfClass:NSArray.class]) {
				NSMutableString *text = [NSMutableString string];
				[self appendContentChunks:message[@"content"] toText:text reasoning:chunkReasoning];
				content = text;
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
	NSArray<NSString *> *clientStops = [self clientStopsForRequest:request];
	if (content != nil && clientStops != nil) {
		content = NFKRemoteTextBeforeStops(content, clientStops);
	}
	NFKInferenceResult *result = [self resultWithText:content
											reasoning:[self reasoningInMessage:message] ?: (chunkReasoning.length > 0 ? chunkReasoning : nil)
												usage:[self usageInResponseBody:responseBody]
											wireCalls:message[@"tool_calls"]
										  audioBase64:audioBase64
										  audioFormat:[self audioOutputFormatForRequest:request]
									expectsStructured:[self requestExpectsStructuredReply:request]
												  raw:responseBody
												error:outError];
	NSDictionary<NSString *, id> *extras = [self extraOutputsInMessage:message];
	if (result == nil || extras.count == 0) {
		return result;
	}
	NSMutableDictionary<NSString *, id> *outputs = [result.outputs mutableCopy];
	[outputs addEntriesFromDictionary:extras];
	return [NFKInferenceResult resultWithOutputs:outputs];
}

// What a reply carries beside its text: generated images (OpenRouter's message.images), the sources
// its url_citation annotations cite, and the tools the service ran itself (Groq's executed_tools).
- (NSDictionary<NSString *, id> *)extraOutputsInMessage:(nullable NSDictionary *)message
{
	NSMutableDictionary<NSString *, id> *extras = [NSMutableDictionary dictionary];
	NSMutableArray *images = [NSMutableArray array];
	for (NSDictionary *entry in [message[@"images"] isKindOfClass:NSArray.class] ? message[@"images"] : @[]) {
		NSDictionary *imageURL = [entry isKindOfClass:NSDictionary.class] && [entry[@"image_url"] isKindOfClass:NSDictionary.class] ? entry[@"image_url"] : nil;
		NSString *url = [imageURL[@"url"] isKindOfClass:NSString.class] ? imageURL[@"url"] : nil;
		NSRange comma = [url rangeOfString:@","];
		if (![url hasPrefix:@"data:"] || comma.location == NSNotFound) {
			continue;
		}
		NSData *bytes = [[NSData alloc] initWithBase64EncodedString:[url substringFromIndex:comma.location + 1]
															options:NSDataBase64DecodingIgnoreUnknownCharacters];
		CVPixelBufferRef pixelBuffer = bytes != nil ? [NFKImageCoding pixelBufferWithImageData:bytes] : NULL;
		if (pixelBuffer != NULL) {
			[images addObject:(__bridge id)pixelBuffer];
			CVPixelBufferRelease(pixelBuffer);
		}
	}
	if (images.count > 0) {
		extras[NFKOutputImage] = images.firstObject;
	}
	if (images.count > 1) {
		extras[NFKOutputImages] = images;
	}
	NSMutableArray<NSDictionary *> *citations = [NSMutableArray array];
	for (NSDictionary *annotation in [message[@"annotations"] isKindOfClass:NSArray.class] ? message[@"annotations"] : @[]) {
		NSDictionary *cited = [annotation isKindOfClass:NSDictionary.class] && [annotation[@"url_citation"] isKindOfClass:NSDictionary.class]
			? annotation[@"url_citation"] : nil;
		if (cited == nil) {
			continue;
		}
		NSMutableDictionary *citation = [NSMutableDictionary dictionaryWithObject:annotation forKey:@"raw"];
		citation[@"url"] = cited[@"url"];
		citation[@"title"] = cited[@"title"];
		citation[@"text"] = cited[@"content"];
		citation[@"start"] = cited[@"start_index"];
		citation[@"end"] = cited[@"end_index"];
		[citations addObject:citation];
	}
	if (citations.count > 0) {
		extras[NFKOutputCitations] = citations;
	}
	NSMutableArray<NSDictionary *> *executed = [NSMutableArray array];
	for (NSDictionary *tool in [message[@"executed_tools"] isKindOfClass:NSArray.class] ? message[@"executed_tools"] : @[]) {
		if (![tool isKindOfClass:NSDictionary.class]) {
			continue;
		}
		NSMutableDictionary *entry = [NSMutableDictionary dictionaryWithObject:tool forKey:@"raw"];
		entry[@"name"] = tool[@"name"] ?: tool[@"type"];
		entry[@"input"] = tool[@"arguments"];
		entry[@"output"] = tool[@"output"] ?: tool[@"search_results"] ?: tool[@"code_results"];
		[executed addObject:entry];
	}
	if (executed.count > 0) {
		extras[NFKOutputServerToolResults] = executed;
	}
	return extras;
}

/*! Why a reply was refused, or nil for one that was not: the provider's content filter ended the
	reply (finish_reason content_filter), or the model declined and said so under message.refusal,
	which is where a structured-output refusal arrives instead of content. */
- (nullable NSString *)refusalInChoice:(nullable NSDictionary *)choice message:(nullable NSDictionary *)message
{
	NSString *refusal = [message[@"refusal"] isKindOfClass:NSString.class] ? message[@"refusal"] : nil;
	if (refusal.length > 0) {
		return refusal;
	}
	if ([choice[@"finish_reason"] isEqual:@"content_filter"]) {
		return @"the provider's content filter stopped the reply";
	}
	return nil;
}

// Mistral returns a reasoning reply as typed chunks: text chunks carry the answer, and a thinking
// chunk carries the chain as a list of text chunks of its own.
- (void)appendContentChunks:(NSArray *)chunks toText:(NSMutableString *)text reasoning:(NSMutableString *)reasoning
{
	for (NSDictionary *chunk in chunks) {
		if (![chunk isKindOfClass:NSDictionary.class]) {
			continue;
		}
		if ([chunk[@"type"] isEqual:@"text"] && [chunk[@"text"] isKindOfClass:NSString.class]) {
			[text appendString:chunk[@"text"]];
			continue;
		}
		if (![chunk[@"type"] isEqual:@"thinking"]) {
			continue;
		}
		id thinking = chunk[@"thinking"];
		if ([thinking isKindOfClass:NSString.class]) {
			[reasoning appendString:thinking];
		} else if ([thinking isKindOfClass:NSArray.class]) {
			NSMutableString *ignored = [NSMutableString string];
			[self appendContentChunks:thinking toText:reasoning reasoning:ignored];
		}
	}
}

// A reasoning model returns its chain beside the answer. The field has two spellings across the
// OpenAI-compatible servers and they mean the same thing, so both are read.
- (nullable NSString *)reasoningInMessage:(nullable NSDictionary *)message
{
	for (NSString *key in @[ @"reasoning_content", @"reasoning" ]) {
		id reasoning = message[key];
		if ([reasoning isKindOfClass:NSString.class] && [reasoning length] > 0) {
			return reasoning;
		}
	}
	return nil;
}

// The counts ride under usage, with the cached and reasoning breakdowns one level further down.
- (nullable NSDictionary<NSString *, NSNumber *> *)usageInResponseBody:(nullable NSDictionary *)body
{
	NSDictionary *usage = [body[@"usage"] isKindOfClass:NSDictionary.class] ? body[@"usage"] : nil;
	if (usage == nil) {
		return nil;
	}
	NSDictionary *inputDetails = [usage[@"prompt_tokens_details"] isKindOfClass:NSDictionary.class] ? usage[@"prompt_tokens_details"] : nil;
	NSDictionary *outputDetails = [usage[@"completion_tokens_details"] isKindOfClass:NSDictionary.class] ? usage[@"completion_tokens_details"] : nil;
	return NFKRemoteUsage(usage[@"prompt_tokens"], inputDetails[@"cached_tokens"],
						  usage[@"completion_tokens"], outputDetails[@"reasoning_tokens"]);
}

- (nullable NFKInferenceResult *)resultWithText:(nullable NSString *)text
									  reasoning:(nullable NSString *)reasoning
										  usage:(nullable NSDictionary<NSString *, NSNumber *> *)usage
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
	if (reasoning.length > 0) {
		outputs[NFKOutputReasoning] = reasoning;
	}
	if (usage != nil) {
		outputs[NFKOutputUsage] = usage;
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
	// A server that reports the token counts puts them on a chunk of their own at the end. OpenAI
	// sends them only when the request asked, which a caller does by setting stream_options in the
	// request parameters; a server that sends them unasked is read either way.
	NSDictionary<NSString *, NSNumber *> *usage = [self usageInResponseBody:chunk];
	if (usage != nil) {
		state.usage = usage;
	}

	NSArray *choices = chunk[@"choices"];
	NSDictionary *choice = [choices isKindOfClass:NSArray.class] && choices.count > 0 ? choices.firstObject : nil;
	if ([choice isKindOfClass:NSDictionary.class] && [choice[@"finish_reason"] isKindOfClass:NSString.class]) {
		state.finishReason = choice[@"finish_reason"];
	}
	NSDictionary *delta = [choice isKindOfClass:NSDictionary.class] && [choice[@"delta"] isKindOfClass:NSDictionary.class]
		? choice[@"delta"] : nil;
	if (delta == nil) {
		return NO;
	}
	if ([delta[@"refusal"] isKindOfClass:NSString.class] && [delta[@"refusal"] length] > 0) {
		state.refusal = [(state.refusal ?: @"") stringByAppendingString:delta[@"refusal"]];
	}
	BOOL carriedText = NO;
	if ([delta[@"content"] isKindOfClass:NSString.class] && [delta[@"content"] length] > 0) {
		[state.text appendString:delta[@"content"]];
		carriedText = YES;
	} else if ([delta[@"content"] isKindOfClass:NSArray.class]) {
		NSUInteger before = state.text.length + state.reasoning.length;
		[self appendContentChunks:delta[@"content"] toText:state.text reasoning:state.reasoning];
		carriedText = state.text.length + state.reasoning.length > before;
	}
	NSString *reasoning = [self reasoningInMessage:delta];
	if (reasoning != nil) {
		[state.reasoning appendString:reasoning];
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
	NSString *refusal = [self refusalInChoice:@{ @"finish_reason": state.finishReason ?: @"" }
									  message:@{ @"refusal": state.refusal ?: @"" }];
	if (refusal != nil) {
		[job finishWithError:[NFKRemoteTransport errorWithCode:kNFKError_InferenceRefused reason:refusal]];
		return;
	}
	NSArray *orderedIndexes = [state.toolCallsByIndex.allKeys sortedArrayUsingSelector:@selector(compare:)];
	NSMutableArray *wireCalls = [NSMutableArray array];
	for (NSNumber *index in orderedIndexes) {
		[wireCalls addObject:state.toolCallsByIndex[index]];
	}
	NSString *text = state.text.length > 0 ? [state.text copy] : [state.transcript copy];
	if (state.clientStops != nil) {
		text = NFKRemoteTextBeforeStops(text, state.clientStops);
	}
	NSMutableDictionary *message = [NSMutableDictionary dictionaryWithObject:@"assistant" forKey:@"role"];
	message[@"content"] = [state.text copy];
	if (wireCalls.count > 0) {
		message[@"tool_calls"] = wireCalls;
	}
	NSError *error = nil;
	NFKInferenceResult *result = [self resultWithText:text
											reasoning:state.reasoning.length > 0 ? [state.reasoning copy] : nil
												usage:state.usage
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
