//
//  NFKGeminiInteractionsBackend.m
//  InferKit
//

#import <InferKit/NFKGeminiInteractionsBackend.h>
#import <InferKit/NFKRemoteTransport.h>
#import <InferKit/NFKRemoteBackend.h>
#import <InferKit/NFKImageCoding.h>
#import <InferKit/NFKInferenceRequest.h>
#import <InferKit/NFKInferenceResult.h>
#import <InferKit/NFKInferenceJob.h>
#import <InferKit/NFKInferenceKeys.h>
#import <InferKit/NFKAudioAsset.h>
#import <InferKit/NFKAudioSegment.h>
#import <InferKit/NFKVideoAsset.h>
#import <InferKit/NFKErrors.h>
#import "NFKRemoteMediaSupport.h"
#import <InferKit/NFKRemoteFileStore.h>

/*! The contract keys this backend translates; every other parameter goes into generation_config. */
static NSSet<NSString *> *NFKInteractionsMappedParameters(void)
{
	static NSSet<NSString *> *mapped;
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		mapped = [NSSet setWithArray:@[ NFKParameterTools, NFKParameterJSONSchema, NFKParameterReasoningEffort,
										NFKParameterMaxTokens, NFKParameterSeed, NFKParameterStopSequences,
										NFKParameterPreviousResponseIdentifier, NFKParameterAspectRatio,
										NFKParameterResolution, NFKParameterOutputFormat, NFKParameterSpeakerDiarization,
										NFKParameterWordTimestamps, NFKParameterVocabulary, NFKParameterSourceLanguage,
										NFKParameterVideoFrameCount, @"voice", @"speakers" ]];
	});
	return mapped;
}

static BOOL NFKInteractionsIsTerminalStatus(id status)
{
	return [@[ @"completed", @"failed", @"cancelled", @"incomplete", @"requires_action" ] containsObject:status ?: @""];
}

/*! A 16-bit PCM stream wrapped in a WAV header. */
static NSData *NFKInteractionsWAVFromPCM(NSData *pcm, uint32_t sampleRate, uint16_t channels)
{
	NSMutableData *wav = [NSMutableData data];
	uint32_t dataSize = (uint32_t)pcm.length, riffSize = 36 + dataSize, formatSize = 16;
	uint32_t byteRate = sampleRate * channels * 2;
	uint16_t format = 1, blockAlign = (uint16_t)(channels * 2), bits = 16;
	[wav appendBytes:"RIFF" length:4];
	[wav appendBytes:&riffSize length:4];
	[wav appendBytes:"WAVEfmt " length:8];
	[wav appendBytes:&formatSize length:4];
	[wav appendBytes:&format length:2];
	[wav appendBytes:&channels length:2];
	[wav appendBytes:&sampleRate length:4];
	[wav appendBytes:&byteRate length:4];
	[wav appendBytes:&blockAlign length:2];
	[wav appendBytes:&bits length:2];
	[wav appendBytes:"data" length:4];
	[wav appendBytes:&dataSize length:4];
	[wav appendData:pcm];
	return wav;
}

/*! Seconds from an offset written as a number or as a duration string ("1.25s"). */
static double NFKInteractionsSeconds(id offset)
{
	if ([offset isKindOfClass:NSNumber.class]) {
		return [offset doubleValue];
	}
	if ([offset isKindOfClass:NSString.class]) {
		return [[offset stringByTrimmingCharactersInSet:NSCharacterSet.letterCharacterSet] doubleValue];
	}
	return 0;
}

@implementation NFKGeminiInteractionsBackend

@synthesize session = _session;

+ (instancetype)backendWithAPIKey:(nullable NSString *)apiKey modelName:(nullable NSString *)modelName
{
	NFKGeminiInteractionsBackend *backend = [[self alloc] init];
	backend.apiKey = apiKey;
	backend.modelName = modelName;
	return backend;
}

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_endpointURL = [NSURL URLWithString:@"https://generativelanguage.googleapis.com/v1beta/interactions"];
		_timeout = 600.0;
		_pollInterval = 5.0;
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
	return @"gemini-interactions";
}

- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request
												  error:(NSError * _Nullable *)outError
{
	NSMutableURLRequest *urlRequest = [self urlRequestForRequest:request streaming:NO error:outError];
	if (urlRequest == nil) {
		return nil;
	}
	NSDictionary *interaction = [self JSONForRequest:urlRequest error:outError];
	while (interaction != nil && !NFKInteractionsIsTerminalStatus(interaction[@"status"])) {
		[NSThread sleepForTimeInterval:self.pollInterval];
		interaction = [self JSONForRequest:[self requestForInteraction:interaction[@"id"] suffix:nil method:@"GET"] error:outError];
	}
	return interaction != nil ? [self resultFromInteraction:interaction request:request error:outError] : nil;
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
		[job finishWithError:error ?: [NFKRemoteTransport errorWithCode:kNFKError_InferenceBackendFailure reason:@"the interaction failed"]];
		return job;
	}
	[job reportProgress:-1.0];
	NSMutableString *text = [NSMutableString string];
	__block NSString *identifier = nil;
	__block NSDictionary *finalInteraction = nil;
	__block NSString *streamFailure = nil;
	__block BOOL finished = NO;
	void (^cancel)(void) = [self streamRequest:urlRequest lineHandler:^(NSString *line) {
		NSString *payload = [NFKRemoteTransport SSEDataForLine:line];
		id event = payload != nil ? [NSJSONSerialization JSONObjectWithData:[payload dataUsingEncoding:NSUTF8StringEncoding] options:0 error:NULL] : nil;
		if (finished || ![event isKindOfClass:NSDictionary.class]) {
			return;
		}
		NSString *type = [event[@"event_type"] isKindOfClass:NSString.class] ? event[@"event_type"] : @"";
		NSDictionary *interaction = [event[@"interaction"] isKindOfClass:NSDictionary.class] ? event[@"interaction"] : nil;
		if ([interaction[@"id"] isKindOfClass:NSString.class]) {
			identifier = interaction[@"id"];
		}
		if ([type isEqualToString:@"interaction.completed"] && interaction[@"steps"] != nil) {
			finalInteraction = interaction;
		} else if ([type isEqualToString:@"error"]) {
			NSDictionary *detail = [event[@"error"] isKindOfClass:NSDictionary.class] ? event[@"error"] : @{};
			streamFailure = [detail[@"message"] isKindOfClass:NSString.class] ? detail[@"message"] : @"the stream reported an error";
		} else if ([type isEqualToString:@"step.delta"]) {
			NSDictionary *delta = [event[@"delta"] isKindOfClass:NSDictionary.class] ? event[@"delta"] : nil;
			if ([delta[@"type"] isEqual:@"text"] && [delta[@"text"] isKindOfClass:NSString.class]) {
				[text appendString:delta[@"text"]];
				[job reportProgress:-1.0 partialResult:[NFKInferenceResult resultWithOutputs:@{ NFKOutputText: [text copy] }]];
			}
		}
	} completionHandler:^(NSHTTPURLResponse * _Nullable response, NSData * _Nullable errorBody, NSError * _Nullable streamError) {
		if (finished || job.status == NFKInferenceJobStatusCancelled) {
			return;
		}
		finished = YES;
		NSError *failure = streamError ?: [NFKRemoteTransport errorForResponse:response data:errorBody];
		if (failure == nil && streamFailure != nil) {
			failure = [NFKRemoteTransport errorWithCode:kNFKError_InferenceBackendFailure reason:streamFailure];
		}
		// The stream carries the text; the finished interaction, read back by id, carries the rest.
		NSDictionary *interaction = finalInteraction;
		if (failure == nil && interaction == nil && identifier != nil) {
			interaction = [self JSONForRequest:[self requestForInteraction:identifier suffix:nil method:@"GET"] error:&failure];
		}
		NFKInferenceResult *result = nil;
		if (failure == nil) {
			result = interaction != nil ? [self resultFromInteraction:interaction request:request error:&failure]
										: [NFKInferenceResult resultWithOutputs:@{ NFKOutputText: [text copy] }];
		}
		if (result != nil) {
			[job finishWithResult:result];
		} else {
			[job finishWithError:failure ?: [NFKRemoteTransport errorWithCode:kNFKError_InferenceBackendFailure reason:@"the interaction failed"]];
		}
	}];
	job.cancellationHandler = cancel;
	return job;
}

- (void)runBackgroundJob:(NFKInferenceJob *)job forRequest:(NFKInferenceRequest *)request
{
	NSError *error = nil;
	NSMutableURLRequest *urlRequest = [self urlRequestForRequest:request streaming:NO error:&error];
	NSDictionary *interaction = urlRequest != nil ? [self JSONForRequest:urlRequest error:&error] : nil;
	NSString *identifier = [interaction[@"id"] isKindOfClass:NSString.class] ? interaction[@"id"] : nil;
	if (identifier != nil) {
		job.cancellationHandler = ^{
			[self sendRequest:[self requestForInteraction:identifier suffix:@"cancel" method:@"POST"] response:NULL error:NULL];
		};
	}
	while (interaction != nil && !NFKInteractionsIsTerminalStatus(interaction[@"status"])) {
		if (job.status == NFKInferenceJobStatusCancelled) {
			return;
		}
		[job reportProgress:-1.0];
		[NSThread sleepForTimeInterval:self.pollInterval];
		interaction = [self JSONForRequest:[self requestForInteraction:identifier suffix:nil method:@"GET"] error:&error];
	}
	NFKInferenceResult *result = interaction != nil ? [self resultFromInteraction:interaction request:request error:&error] : nil;
	if (result != nil) {
		[job finishWithResult:result];
	} else {
		[job finishWithError:error ?: [NFKRemoteTransport errorWithCode:kNFKError_InferenceBackendFailure reason:@"the background interaction failed"]];
	}
}

#pragma mark Request

- (nullable NSMutableURLRequest *)urlRequestForRequest:(NFKInferenceRequest *)request
											 streaming:(BOOL)streaming
												 error:(NSError * _Nullable *)outError
{
	if (self.modelName.length == 0) {
		[self fail:outError code:kNFKError_InferenceNotReady reason:@"the Interactions API needs a model name"];
		return nil;
	}
	NFKRemoteAttachments *attachments = [NFKRemoteAttachments attachmentsForRequest:request keepsVideo:YES error:outError];
	if (attachments == nil) {
		return nil;
	}
	NSMutableDictionary<NSString *, id> *body = [NSMutableDictionary dictionaryWithObject:self.modelName forKey:@"model"];
	if (![self addInputForRequest:request attachments:attachments to:body]) {
		[self fail:outError code:kNFKError_InferenceMissingInput reason:@"the request carries neither a prompt, messages, nor media"];
		return nil;
	}
	[self addConfigurationForRequest:request attachments:attachments to:body];
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
	NSURL *url = self.endpointURL;
	if (streaming) {
		NSURLComponents *components = [NSURLComponents componentsWithURL:url resolvingAgainstBaseURL:NO];
		components.queryItems = @[ [NSURLQueryItem queryItemWithName:@"alt" value:@"sse"] ];
		url = components.URL ?: url;
	}
	NSMutableURLRequest *urlRequest = [NSMutableURLRequest requestWithURL:url];
	urlRequest.HTTPMethod = @"POST";
	urlRequest.timeoutInterval = self.timeout;
	urlRequest.HTTPBody = payload;
	[urlRequest setValue:@"application/json" forHTTPHeaderField:@"Content-Type"];
	[self authorize:urlRequest];
	return urlRequest;
}

// A prompt alone goes as a string. Otherwise each turn is a step (user_input or model_output) of
// content blocks, the system turn is system_instruction, and the media ride on the last user turn.
- (BOOL)addInputForRequest:(NFKInferenceRequest *)request attachments:(NFKRemoteAttachments *)attachments to:(NSMutableDictionary *)body
{
	NSArray *messages = request.messages;
	if (messages.count == 0 && attachments.isEmpty) {
		if (request.prompt.length == 0) {
			return NO;
		}
		body[@"input"] = request.prompt;
		return YES;
	}
	if (messages.count == 0) {
		messages = @[ @{ @"role": @"user", @"content": request.prompt ?: @"" } ];
	}
	NSMutableArray *steps = [NSMutableArray array];
	for (NSDictionary *message in messages) {
		if (![message isKindOfClass:NSDictionary.class]) {
			continue;
		}
		NSString *content = [message[@"content"] isKindOfClass:NSString.class] ? message[@"content"] : nil;
		if ([message[@"role"] isEqual:@"system"]) {
			body[@"system_instruction"] = content;
			continue;
		}
		NSMutableArray *blocks = [NSMutableArray array];
		if (content.length > 0) {
			[blocks addObject:@{ @"type": @"text", @"text": content }];
		}
		[steps addObject:@{ @"type": [message[@"role"] isEqual:@"assistant"] ? @"model_output" : @"user_input", @"content": blocks }];
	}
	NSInteger last = -1;
	for (NSInteger index = (NSInteger)steps.count - 1; index >= 0; index--) {
		if ([steps[index][@"type"] isEqual:@"user_input"]) {
			last = index;
			break;
		}
	}
	if (last < 0) {
		[steps addObject:@{ @"type": @"user_input", @"content": @[] }];
		last = (NSInteger)steps.count - 1;
	}
	NSMutableArray *blocks = [steps[last][@"content"] mutableCopy];
	[blocks addObjectsFromArray:[self mediaBlocksFor:attachments]];
	steps[last] = @{ @"type": @"user_input", @"content": blocks };
	body[@"input"] = steps;
	return YES;
}

- (NSArray<NSDictionary *> *)mediaBlocksFor:(NFKRemoteAttachments *)attachments
{
	NSMutableArray<NSDictionary *> *blocks = [NSMutableArray array];
	for (NSData *png in attachments.imagePNGs) {
		[blocks addObject:@{ @"type": @"image", @"mime_type": @"image/png", @"data": [png base64EncodedStringWithOptions:0] }];
	}
	if (attachments.audioData != nil) {
		NSString *format = [attachments.audioFormat isEqualToString:@"mp3"] ? @"mpeg" : attachments.audioFormat ?: @"wav";
		[blocks addObject:@{ @"type": @"audio", @"mime_type": [@"audio/" stringByAppendingString:format],
							 @"data": [attachments.audioData base64EncodedStringWithOptions:0] }];
	}
	if (attachments.videoData != nil) {
		NSString *format = [attachments.videoFormat isEqualToString:@"mov"] ? @"quicktime" : attachments.videoFormat ?: @"mp4";
		[blocks addObject:@{ @"type": @"video", @"mime_type": [@"video/" stringByAppendingString:format],
							 @"data": [attachments.videoData base64EncodedStringWithOptions:0] }];
	}
	for (NSDictionary *document in attachments.documents) {
		NFKRemoteFile *file = document[@"fileReference"];
		if (file != nil) {
			NSString *mimeType = file.mimeType ?: @"application/pdf";
			NSString *type = [mimeType hasPrefix:@"image/"] ? @"image" : [mimeType hasPrefix:@"audio/"] ? @"audio"
						   : [mimeType hasPrefix:@"video/"] ? @"video" : @"document";
			[blocks addObject:@{ @"type": type, @"uri": file.uri.absoluteString ?: file.identifier, @"mime_type": mimeType }];
			continue;
		}
		if ([document[@"mediaType"] isEqual:@"text/plain"]) {
			NSString *text = [[NSString alloc] initWithData:document[@"data"] encoding:NSUTF8StringEncoding] ?: @"";
			[blocks addObject:@{ @"type": @"text", @"text": [NSString stringWithFormat:@"%@:\n%@", document[@"filename"], text] }];
			continue;
		}
		[blocks addObject:@{ @"type": @"document", @"mime_type": @"application/pdf",
							 @"data": [document[@"data"] base64EncodedStringWithOptions:0] }];
	}
	return blocks;
}

- (void)addConfigurationForRequest:(NFKInferenceRequest *)request attachments:(NFKRemoteAttachments *)attachments to:(NSMutableDictionary *)body
{
	NSDictionary *parameters = request.parameters;
	NSMutableDictionary *generation = [NSMutableDictionary dictionary];
	NSMutableDictionary *format = [NSMutableDictionary dictionary];
	switch (request.outputModality) {
		case NFKModalityImage:
			format[@"type"] = @"image";
			format[@"aspect_ratio"] = parameters[NFKParameterAspectRatio];
			format[@"image_size"] = parameters[NFKParameterResolution];
			if ([parameters[NFKParameterOutputFormat] isKindOfClass:NSString.class]) {
				format[@"mime_type"] = [@"image/" stringByAppendingString:parameters[NFKParameterOutputFormat]];
			}
			break;
		case NFKModalityAudio: {
			format[@"type"] = @"audio";
			NSArray *speakers = [parameters[@"speakers"] isKindOfClass:NSArray.class] ? parameters[@"speakers"] : nil;
			NSString *voice = [parameters[@"voice"] isKindOfClass:NSString.class] ? parameters[@"voice"] : self.voice;
			if (speakers != nil) {
				generation[@"speech_config"] = speakers;
			} else if (voice.length > 0) {
				generation[@"speech_config"] = @[ @{ @"voice": voice } ];
			}
			if ([parameters[NFKParameterOutputFormat] isKindOfClass:NSString.class]) {
				format[@"mime_type"] = [@"audio/" stringByAppendingString:parameters[NFKParameterOutputFormat]];
			}
			break;
		}
		case NFKModalityVideo:
			format[@"type"] = @"video";
			format[@"aspect_ratio"] = parameters[NFKParameterAspectRatio];
			format[@"resolution"] = parameters[NFKParameterResolution];
			break;
		default: {
			NSDictionary *schema = [parameters[NFKParameterJSONSchema] isKindOfClass:NSDictionary.class] ? parameters[NFKParameterJSONSchema] : nil;
			if (schema != nil) {
				format[@"type"] = @"text";
				format[@"mime_type"] = @"application/json";
				format[@"schema"] = schema;
			}
			break;
		}
	}
	if (format.count > 0) {
		body[@"response_format"] = format;
	}
	NSDictionary *transcription = [self transcriptionConfigurationFor:parameters attachments:attachments];
	if (transcription != nil) {
		generation[@"transcription_config"] = transcription;
	}
	NSString *effort = [parameters[NFKParameterReasoningEffort] isKindOfClass:NSString.class] ? parameters[NFKParameterReasoningEffort] : nil;
	if (effort != nil) {
		NSDictionary<NSString *, NSString *> *levels = @{ NFKReasoningEffortLight: @"low", NFKReasoningEffortModerate: @"medium",
														  NFKReasoningEffortDeep: @"high" };
		generation[@"thinking_level"] = levels[effort] ?: effort;
		generation[@"thinking_summaries"] = @"auto";
	}
	generation[@"max_output_tokens"] = parameters[NFKParameterMaxTokens];
	generation[@"seed"] = parameters[NFKParameterSeed];
	generation[@"stop_sequences"] = parameters[NFKParameterStopSequences];
	NSSet<NSString *> *mapped = NFKInteractionsMappedParameters();
	for (NSString *key in parameters) {
		if (![mapped containsObject:key]) {
			generation[key] = parameters[key];
		}
	}
	if (generation.count > 0) {
		body[@"generation_config"] = generation;
	}
	body[@"previous_interaction_id"] = parameters[NFKParameterPreviousResponseIdentifier];
	NSArray *tools = [parameters[NFKParameterTools] isKindOfClass:NSArray.class] ? parameters[NFKParameterTools] : nil;
	if (tools != nil) {
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
}

// Audio in with any of the transcription keys asks for a transcription: verbatim with speakers or
// word times, smart otherwise.
- (nullable NSDictionary *)transcriptionConfigurationFor:(NSDictionary *)parameters attachments:(NFKRemoteAttachments *)attachments
{
	BOOL diarizes = [parameters[NFKParameterSpeakerDiarization] boolValue];
	BOOL words = [parameters[NFKParameterWordTimestamps] boolValue];
	NSArray *vocabulary = [parameters[NFKParameterVocabulary] isKindOfClass:NSArray.class] ? parameters[NFKParameterVocabulary] : nil;
	NSString *language = [parameters[NFKParameterSourceLanguage] isKindOfClass:NSString.class] ? parameters[NFKParameterSourceLanguage] : nil;
	if (attachments.audioData == nil || (!diarizes && !words && vocabulary == nil && language == nil)) {
		return nil;
	}
	NSMutableDictionary *configuration = [NSMutableDictionary dictionary];
	if (language != nil) {
		configuration[@"language_codes"] = @[ language ];
	}
	configuration[@"custom_vocabulary"] = vocabulary;
	if (diarizes || words) {
		NSMutableDictionary *mode = [NSMutableDictionary dictionaryWithObject:@"verbatim" forKey:@"type"];
		if (diarizes) {
			mode[@"diarization_mode"] = @"speaker";
		}
		if (words) {
			mode[@"timestamp_granularities"] = @[ @"word" ];
		}
		configuration[@"mode"] = mode;
	}
	return configuration;
}

- (NSURLRequest *)requestForInteraction:(nullable NSString *)identifier suffix:(nullable NSString *)suffix method:(NSString *)method
{
	NSURL *url = [self.endpointURL URLByAppendingPathComponent:identifier ?: @""];
	if (suffix != nil) {
		url = [url URLByAppendingPathComponent:suffix];
	}
	NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:url];
	request.HTTPMethod = method;
	request.timeoutInterval = self.timeout;
	[self authorize:request];
	return request;
}

- (void)authorize:(NSMutableURLRequest *)request
{
	if (self.apiKey.length > 0) {
		[request setValue:self.apiKey forHTTPHeaderField:@"x-goog-api-key"];
	}
}

- (nullable NSDictionary *)JSONForRequest:(NSURLRequest *)request error:(NSError * _Nullable *)outError
{
	NSHTTPURLResponse *response = nil;
	NSError *sendError = nil;
	NSData *data = [self sendRequest:request response:&response error:&sendError];
	NSError *failure = data == nil ? sendError : [NFKRemoteTransport errorForResponse:response data:data];
	if (failure != nil || data == nil) {
		if (outError != NULL) {
			*outError = failure ?: [NFKRemoteTransport errorWithCode:kNFKError_InferenceBackendFailure reason:@"the interaction failed"];
		}
		return nil;
	}
	id reply = [NSJSONSerialization JSONObjectWithData:data options:0 error:NULL];
	return [reply isKindOfClass:NSDictionary.class] ? reply
		: [self fail:outError code:kNFKError_InferenceBackendFailure reason:@"the response is not a JSON object"];
}

#pragma mark Response

- (nullable NFKInferenceResult *)resultFromInteraction:(NSDictionary *)interaction request:(NFKInferenceRequest *)request error:(NSError * _Nullable *)outError
{
	if ([interaction[@"status"] isEqual:@"failed"] || [interaction[@"status"] isEqual:@"cancelled"]) {
		id error = interaction[@"error"];
		NSString *reason = [error isKindOfClass:NSDictionary.class] && [error[@"message"] isKindOfClass:NSString.class] ? error[@"message"]
			: [NSString stringWithFormat:@"the interaction ended %@", interaction[@"status"]];
		return [self fail:outError code:kNFKError_InferenceBackendFailure reason:reason];
	}
	NSMutableArray<NSString *> *texts = [NSMutableArray array];
	NSMutableArray<NSString *> *thoughts = [NSMutableArray array];
	NSMutableArray *images = [NSMutableArray array];
	NSMutableArray<NFKAudioSegment *> *words = [NSMutableArray array];
	NSMutableArray<NSDictionary *> *citations = [NSMutableArray array];
	NSMutableArray<NSDictionary *> *toolCalls = [NSMutableArray array];
	NSMutableArray<NSDictionary *> *serverTools = [NSMutableArray array];
	NSMutableDictionary<NSString *, id> *outputs = [NSMutableDictionary dictionaryWithObject:interaction forKey:NFKRemoteBackendRawKey];
	for (NSDictionary *step in [interaction[@"steps"] isKindOfClass:NSArray.class] ? interaction[@"steps"] : @[]) {
		if (![step isKindOfClass:NSDictionary.class]) {
			continue;
		}
		NSString *type = [step[@"type"] isKindOfClass:NSString.class] ? step[@"type"] : @"";
		NSArray *content = [step[@"content"] isKindOfClass:NSArray.class] ? step[@"content"] : @[];
		if ([type isEqualToString:@"thought"]) {
			for (NSDictionary *block in content) {
				if ([block isKindOfClass:NSDictionary.class] && [block[@"text"] isKindOfClass:NSString.class]) {
					[thoughts addObject:block[@"text"]];
				}
			}
		} else if ([type isEqualToString:@"function_call"]) {
			NSDictionary *arguments = [step[@"arguments"] isKindOfClass:NSDictionary.class] ? step[@"arguments"] : @{};
			NSData *json = [NSJSONSerialization dataWithJSONObject:arguments options:0 error:NULL];
			[toolCalls addObject:@{ @"id": step[@"id"] ?: @"", @"name": step[@"name"] ?: @"", @"arguments": arguments,
									@"argumentsJSON": json != nil ? [[NSString alloc] initWithData:json encoding:NSUTF8StringEncoding] : @"{}" }];
		} else if ([type isEqualToString:@"model_output"]) {
			for (NSDictionary *block in content) {
				if (![block isKindOfClass:NSDictionary.class]) {
					continue;
				}
				if (![self readBlock:block into:outputs texts:texts images:images words:words citations:citations error:outError]) {
					return nil;
				}
			}
		} else if (![type isEqualToString:@"user_input"]) {
			NSMutableDictionary *entry = [NSMutableDictionary dictionaryWithObject:step forKey:@"raw"];
			entry[@"name"] = type;
			entry[@"input"] = step[@"arguments"];
			entry[@"output"] = step[@"result"] ?: content;
			[serverTools addObject:entry];
		}
	}
	NSString *text = [texts componentsJoinedByString:@""];
	if (text.length > 0) {
		outputs[NFKOutputText] = text;
		if ([request.parameters[NFKParameterJSONSchema] isKindOfClass:NSDictionary.class]) {
			id structured = [NSJSONSerialization JSONObjectWithData:[text dataUsingEncoding:NSUTF8StringEncoding] options:0 error:NULL];
			if ([structured isKindOfClass:NSDictionary.class]) {
				outputs[NFKOutputStructured] = structured;
			}
		}
	}
	if (thoughts.count > 0) {
		outputs[NFKOutputReasoning] = [thoughts componentsJoinedByString:@"\n"];
	}
	if (images.count > 0) {
		outputs[NFKOutputImage] = images.firstObject;
	}
	if (images.count > 1) {
		outputs[NFKOutputImages] = images;
	}
	if (words.count > 0) {
		outputs[NFKOutputWords] = words;
		outputs[NFKOutputSegments] = [self turnsFromWords:words];
	}
	if (citations.count > 0) {
		outputs[NFKOutputCitations] = citations;
	}
	if (toolCalls.count > 0) {
		outputs[NFKOutputToolCalls] = toolCalls;
	}
	if (serverTools.count > 0) {
		outputs[NFKOutputServerToolResults] = serverTools;
	}
	if ([interaction[@"id"] isKindOfClass:NSString.class]) {
		outputs[NFKOutputResponseIdentifier] = interaction[@"id"];
	}
	NSDictionary *usage = [interaction[@"usage"] isKindOfClass:NSDictionary.class] ? interaction[@"usage"] : nil;
	NSDictionary *counts = NFKRemoteUsage(usage[@"total_input_tokens"], usage[@"total_cached_tokens"],
										  usage[@"total_output_tokens"], usage[@"total_thought_tokens"]);
	if (counts != nil) {
		outputs[NFKOutputUsage] = counts;
	}
	return [NFKInferenceResult resultWithOutputs:outputs];
}

// One output block: text (with its annotations), an image, an audio clip, or a video.
- (BOOL)readBlock:(NSDictionary *)block
			 into:(NSMutableDictionary<NSString *, id> *)outputs
			texts:(NSMutableArray<NSString *> *)texts
		   images:(NSMutableArray *)images
			words:(NSMutableArray<NFKAudioSegment *> *)words
		citations:(NSMutableArray<NSDictionary *> *)citations
			error:(NSError * _Nullable *)outError
{
	NSString *type = [block[@"type"] isKindOfClass:NSString.class] ? block[@"type"] : @"";
	NSString *mimeType = [block[@"mime_type"] isKindOfClass:NSString.class] ? block[@"mime_type"] : @"";
	if ([type isEqualToString:@"text"]) {
		if ([block[@"text"] isKindOfClass:NSString.class]) {
			[texts addObject:block[@"text"]];
		}
		[self readAnnotations:block[@"annotations"] words:words citations:citations];
		return YES;
	}
	NSData *bytes = [self bytesOfBlock:block error:outError];
	if (bytes == nil) {
		// Only a URI that could not be fetched fails the result; a block without media is skipped.
		return ![block[@"uri"] isKindOfClass:NSString.class] || [block[@"data"] isKindOfClass:NSString.class];
	}
	if ([type isEqualToString:@"image"]) {
		CVPixelBufferRef pixelBuffer = [NFKImageCoding pixelBufferWithImageData:bytes];
		if (pixelBuffer != NULL) {
			[images addObject:(__bridge id)pixelBuffer];
			CVPixelBufferRelease(pixelBuffer);
		}
		return YES;
	}
	if ([type isEqualToString:@"audio"]) {
		BOOL encoded = [mimeType containsString:@"mp3"] || [mimeType containsString:@"mpeg"] || [mimeType containsString:@"wav"];
		NSString *extension = [mimeType containsString:@"mp3"] || [mimeType containsString:@"mpeg"] ? @"mp3" : @"wav";
		uint32_t rate = [block[@"sample_rate"] isKindOfClass:NSNumber.class] ? [block[@"sample_rate"] unsignedIntValue] : 24000;
		uint16_t channels = [block[@"channels"] isKindOfClass:NSNumber.class] ? [block[@"channels"] unsignedShortValue] : 1;
		NSData *file = encoded ? bytes : NFKInteractionsWAVFromPCM(bytes, rate, channels);
		NSURL *url = NFKRemoteWriteMediaFile(file, @"interaction", extension, self.outputDirectoryURL, outError);
		if (url == nil) {
			return NO;
		}
		outputs[NFKOutputAudio] = [NFKAudioAsset audioAssetWithFileURL:url];
		return YES;
	}
	if ([type isEqualToString:@"video"]) {
		NSString *extension = [mimeType containsString:@"webm"] ? @"webm" : @"mp4";
		NSURL *url = NFKRemoteWriteMediaFile(bytes, @"interaction", extension, self.outputDirectoryURL, outError);
		if (url == nil) {
			return NO;
		}
		outputs[NFKOutputVideo] = [NFKVideoAsset videoAssetWithFileURL:url];
	}
	return YES;
}

// Inline data decodes; a URI is fetched, with the key when it is on Google's own host.
- (nullable NSData *)bytesOfBlock:(NSDictionary *)block error:(NSError * _Nullable *)outError
{
	if ([block[@"data"] isKindOfClass:NSString.class]) {
		return [[NSData alloc] initWithBase64EncodedString:block[@"data"] options:NSDataBase64DecodingIgnoreUnknownCharacters];
	}
	NSURL *url = [block[@"uri"] isKindOfClass:NSString.class] ? [NSURL URLWithString:block[@"uri"]] : nil;
	if (url == nil) {
		return nil;
	}
	NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:url];
	request.timeoutInterval = self.timeout;
	if ([url.host isEqualToString:self.endpointURL.host]) {
		[self authorize:request];
	}
	NSHTTPURLResponse *response = nil;
	NSError *sendError = nil;
	NSData *data = [self sendRequest:request response:&response error:&sendError];
	NSError *failure = data == nil ? sendError : [NFKRemoteTransport errorForResponse:response data:data];
	if (failure != nil) {
		if (outError != NULL) { *outError = failure; }
		return nil;
	}
	return data;
}

// word_info annotations time each word with its speaker; the rest cite sources.
- (void)readAnnotations:(id)annotations words:(NSMutableArray<NFKAudioSegment *> *)words citations:(NSMutableArray<NSDictionary *> *)citations
{
	for (NSDictionary *annotation in [annotations isKindOfClass:NSArray.class] ? annotations : @[]) {
		if (![annotation isKindOfClass:NSDictionary.class]) {
			continue;
		}
		if ([annotation[@"type"] isEqual:@"word_info"]) {
			NSString *speaker = [annotation[@"speaker"] isKindOfClass:NSString.class] ? annotation[@"speaker"] : nil;
			[words addObject:[NFKAudioSegment segmentWithStartSeconds:NFKInteractionsSeconds(annotation[@"start_offset"])
														   endSeconds:NFKInteractionsSeconds(annotation[@"end_offset"])
																label:annotation[@"text"]
														   confidence:1.0
															  speaker:speaker]];
			continue;
		}
		NSMutableDictionary *citation = [NSMutableDictionary dictionaryWithObject:annotation forKey:@"raw"];
		citation[@"url"] = annotation[@"url"] ?: annotation[@"uri"];
		citation[@"title"] = annotation[@"title"];
		citation[@"start"] = annotation[@"start_index"];
		citation[@"end"] = annotation[@"end_index"];
		[citations addObject:citation];
	}
}

// Words grouped into turns at a speaker change.
- (NSArray<NFKAudioSegment *> *)turnsFromWords:(NSArray<NFKAudioSegment *> *)words
{
	NSMutableArray<NFKAudioSegment *> *turns = [NSMutableArray array];
	NSMutableArray<NSString *> *pieces = [NSMutableArray array];
	NFKAudioSegment *first = nil;
	for (NSUInteger index = 0; index < words.count; index++) {
		NFKAudioSegment *word = words[index];
		first = first ?: word;
		[pieces addObject:word.label ?: @""];
		NFKAudioSegment *next = index + 1 < words.count ? words[index + 1] : nil;
		if (next == nil || !(next.speaker == word.speaker || [next.speaker isEqual:word.speaker])) {
			[turns addObject:[NFKAudioSegment segmentWithStartSeconds:first.startSeconds endSeconds:word.endSeconds
																label:[pieces componentsJoinedByString:@" "] confidence:1.0 speaker:first.speaker]];
			[pieces removeAllObjects];
			first = nil;
		}
	}
	return turns;
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
