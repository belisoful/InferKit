//
//  NFKRemoteSpeechBackend.m
//  InferKit
//

#import <InferKit/NFKRemoteSpeechBackend.h>
#import <InferKit/NFKRemoteProvider.h>
#import <InferKit/NFKRemoteTransport.h>
#import <InferKit/NFKInferenceRequest.h>
#import <InferKit/NFKInferenceResult.h>
#import <InferKit/NFKInferenceJob.h>
#import <InferKit/NFKInferenceKeys.h>
#import <InferKit/NFKAudioAsset.h>
#import <InferKit/NFKErrors.h>

/*! The contract keys this backend translates; every other parameter goes out under its own name. */
static NSSet<NSString *> *NFKSpeechMappedParameters(void)
{
	static NSSet<NSString *> *mapped;
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		mapped = [NSSet setWithArray:@[ NFKParameterSourceLanguage, NFKParameterSampleRate ]];
	});
	return mapped;
}

/*! The chunk with the given four-character id in a RIFF file, as its range, or NSNotFound. A data
	chunk whose declared size runs past the file (a streamed WAV) ends at the file's end. */
static NSRange NFKSpeechWAVChunk(NSData *wav, const char *identifier)
{
	const uint8_t *bytes = wav.bytes;
	NSUInteger offset = 12;
	while (wav.length >= 12 && offset + 8 <= wav.length) {
		uint32_t size = (uint32_t)bytes[offset + 4] | ((uint32_t)bytes[offset + 5] << 8)
			| ((uint32_t)bytes[offset + 6] << 16) | ((uint32_t)bytes[offset + 7] << 24);
		NSUInteger start = offset + 8;
		NSUInteger length = MIN((NSUInteger)size, wav.length - start);
		if (memcmp(bytes + offset, identifier, 4) == 0) {
			return NSMakeRange(start, length);
		}
		offset = start + length + (length % 2);
	}
	return NSMakeRange(NSNotFound, 0);
}

static void NFKSpeechAppendUInt32(NSMutableData *data, uint32_t value)
{
	uint8_t bytes[4] = { (uint8_t)value, (uint8_t)(value >> 8), (uint8_t)(value >> 16), (uint8_t)(value >> 24) };
	[data appendBytes:bytes length:4];
}

@implementation NFKRemoteVoice

- (instancetype)initWithIdentifier:(NSString *)identifier
							  name:(nullable NSString *)name
						 languages:(NSArray<NSString *> *)languages
							   raw:(NSDictionary *)raw
{
	self = [super init];
	if (self != nil) {
		_identifier = [identifier copy];
		_name = [name copy];
		_languages = [languages copy];
		_raw = [raw copy];
	}
	return self;
}

@end

/*! What a streamed reply assembles into: the decoded audio so far. */
@interface NFKSpeechStreamState : NSObject
@property (nonatomic, strong) NSMutableData *audio;
@property (nonatomic, assign) BOOL finished;
@end

@implementation NFKSpeechStreamState
@end

@implementation NFKRemoteSpeechBackend

@synthesize session = _session;

+ (instancetype)backendWithEndpointURL:(nullable NSURL *)endpointURL
{
	NFKRemoteSpeechBackend *backend = [[self alloc] init];
	backend.endpointURL = endpointURL;
	return backend;
}

+ (nullable instancetype)backendForProvider:(NFKRemoteProvider *)provider
									 apiKey:(nullable NSString *)apiKey
								  modelName:(nullable NSString *)modelName
									  voice:(nullable NSString *)voice
{
	NSString *identifier = provider.identifier;
	NFKRemoteSpeechBackend *backend = nil;
	if ([@[ @"openai", @"groq", @"together", @"openrouter" ] containsObject:identifier]) {
		backend = [self backendWithEndpointURL:[provider URLForPath:@"audio/speech"]];
		backend.maximumInputLength = [identifier isEqualToString:@"groq"] ? 200 : 0;
	} else if ([identifier isEqualToString:@"mistral"]) {
		backend = [self backendWithEndpointURL:[provider URLForPath:@"audio/speech"]];
		backend.apiStyle = NFKRemoteSpeechAPIStyleMistral;
	} else if ([identifier isEqualToString:@"xai"]) {
		backend = [self backendWithEndpointURL:[provider URLForPath:@"tts"]];
		backend.apiStyle = NFKRemoteSpeechAPIStyleXAI;
	}
	backend.apiKey = apiKey;
	backend.modelName = modelName;
	backend.voice = voice;
	return backend;
}

- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_responseFormat = @"wav";
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
	return @"remote-speech";
}

- (nullable NFKInferenceResult *)runInferenceForRequest:(NFKInferenceRequest *)request
												  error:(NSError * _Nullable *)outError
{
	NSString *text = [self textForRequest:request];
	if (![self validateRequest:request text:text error:outError]) {
		return nil;
	}
	NSMutableData *joined = [NSMutableData data];
	NSMutableArray<NSData *> *pieces = [NSMutableArray array];
	for (NSString *piece in [self piecesOfText:text]) {
		NSData *audio = [self audioForText:piece request:request error:outError];
		if (audio == nil) {
			return nil;
		}
		[pieces addObject:audio];
	}
	NSString *format = [self formatForRequest:request];
	[joined appendData:[self joinedAudio:pieces format:format]];
	return [self resultForAudio:joined format:format error:outError];
}

- (NFKInferenceJob *)submitInferenceJobForRequest:(NFKInferenceRequest *)request
{
	NFKInferenceJob *job = [[NFKInferenceJob alloc] init];
	NSString *text = [self textForRequest:request];
	BOOL streamable = self.streams && self.apiStyle != NFKRemoteSpeechAPIStyleXAI
		&& (self.maximumInputLength == 0 || text.length <= self.maximumInputLength);
	if (!streamable) {
		dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
			if (job.status == NFKInferenceJobStatusCancelled) {
				return;
			}
			[job reportProgress:-1.0];
			NSError *error = nil;
			NFKInferenceResult *result = [self runInferenceForRequest:request error:&error];
			if (result != nil) {
				[job finishWithResult:result];
			} else {
				[job finishWithError:error ?: [NFKRemoteTransport errorWithCode:kNFKError_InferenceBackendFailure reason:@"the speech call failed"]];
			}
		});
		return job;
	}
	NSError *error = nil;
	if (![self validateRequest:request text:text error:&error]) {
		[job finishWithError:error ?: [NFKRemoteTransport errorWithCode:kNFKError_InferenceBackendFailure reason:@"the speech call failed"]];
		return job;
	}
	NSMutableURLRequest *urlRequest = [self urlRequestForText:text request:request streaming:YES error:&error];
	if (urlRequest == nil) {
		[job finishWithError:error ?: [NFKRemoteTransport errorWithCode:kNFKError_InferenceBackendFailure reason:@"the speech call failed"]];
		return job;
	}
	[job reportProgress:-1.0];
	NFKSpeechStreamState *state = [[NFKSpeechStreamState alloc] init];
	state.audio = [NSMutableData data];
	NSString *format = [self formatForRequest:request];
	void (^cancel)(void) = [self streamRequest:urlRequest lineHandler:^(NSString *line) {
		if (state.finished) {
			return;
		}
		NSString *payload = [NFKRemoteTransport SSEDataForLine:line];
		id event = payload != nil ? [NSJSONSerialization JSONObjectWithData:[payload dataUsingEncoding:NSUTF8StringEncoding] options:0 error:NULL] : nil;
		NSData *chunk = [event isKindOfClass:NSDictionary.class] ? [self audioInStreamEvent:event] : nil;
		if (chunk.length > 0) {
			[state.audio appendData:chunk];
			[job reportProgress:-1.0 partialResult:[NFKInferenceResult resultWithOutputs:@{ NFKOutputAudio: [state.audio copy] }]];
		}
	} completionHandler:^(NSHTTPURLResponse * _Nullable response, NSData * _Nullable errorBody, NSError * _Nullable streamError) {
		if (state.finished || job.status == NFKInferenceJobStatusCancelled) {
			return;
		}
		state.finished = YES;
		NSError *failure = streamError ?: [NFKRemoteTransport errorForResponse:response data:errorBody];
		NFKInferenceResult *result = failure == nil ? [self resultForAudio:state.audio format:format error:&failure] : nil;
		if (result != nil) {
			[job finishWithResult:result];
		} else {
			[job finishWithError:failure ?: [NFKRemoteTransport errorWithCode:kNFKError_InferenceBackendFailure reason:@"the speech call failed"]];
		}
	}];
	job.cancellationHandler = cancel;
	return job;
}

#pragma mark Request

- (BOOL)validateRequest:(NFKInferenceRequest *)request text:(nullable NSString *)text error:(NSError * _Nullable *)outError
{
	if (self.endpointURL == nil) {
		return [self failWithCode:kNFKError_InferenceNotReady reason:@"no endpoint URL is set" error:outError] != nil;
	}
	if (text == nil) {
		return [self failWithCode:kNFKError_InferenceMissingInput
						   reason:@"the request carries neither a prompt nor messages" error:outError] != nil;
	}
	BOOL hasVoice = [self voiceForRequest:request].length > 0;
	BOOL hasReference = [self referenceClipForRequest:request] != nil;
	if (!hasVoice && !hasReference && self.apiStyle != NFKRemoteSpeechAPIStyleXAI) {
		return [self failWithCode:kNFKError_InferenceNotReady
						   reason:@"no voice is set; every speech service here requires one or a reference clip" error:outError] != nil;
	}
	return YES;
}

- (nullable NSString *)voiceForRequest:(NFKInferenceRequest *)request
{
	id voice = request.parameters[@"voice"] ?: request.parameters[@"voice_id"];
	if ([voice isKindOfClass:NSString.class] && [voice length] > 0) {
		return voice;
	}
	return self.voice;
}

- (NSString *)formatForRequest:(NFKInferenceRequest *)request
{
	id format = request.parameters[@"response_format"];
	return [format isKindOfClass:NSString.class] && [format length] > 0 ? format : self.responseFormat;
}

- (nullable NSData *)referenceClipForRequest:(NFKInferenceRequest *)request
{
	id clip = [request inputForKey:NFKInputVoiceReference];
	if ([clip isKindOfClass:NSData.class]) {
		return clip;
	}
	NSURL *url = [clip isKindOfClass:NFKAudioAsset.class] ? [(NFKAudioAsset *)clip fileURL] : nil;
	return url.isFileURL ? [NSData dataWithContentsOfURL:url] : nil;
}

- (NSDictionary<NSString *, id> *)bodyForText:(NSString *)text request:(NFKInferenceRequest *)request streaming:(BOOL)streaming
{
	NSMutableDictionary<NSString *, id> *body = [NSMutableDictionary dictionary];
	NSDictionary *parameters = request.parameters;
	NSString *voice = [self voiceForRequest:request];
	NSString *format = [self formatForRequest:request];
	NSData *reference = [self referenceClipForRequest:request];
	switch (self.apiStyle) {
		case NFKRemoteSpeechAPIStyleOpenAI:
			body[@"input"] = text;
			body[@"voice"] = voice;
			body[@"response_format"] = format;
			body[@"language"] = parameters[NFKParameterSourceLanguage];
			body[@"sample_rate"] = parameters[NFKParameterSampleRate];
			if (reference != nil) {
				body[@"input_references"] = @[ @{ @"type": @"input_audio",
												  @"input_audio": @{ @"data": [reference base64EncodedStringWithOptions:0] } } ];
			}
			if (streaming) {
				body[@"stream_format"] = @"sse";
				body[@"stream"] = @YES;
			}
			break;
		case NFKRemoteSpeechAPIStyleMistral:
			body[@"input"] = text;
			body[@"voice_id"] = voice;
			body[@"response_format"] = format;
			if (reference != nil) {
				body[@"ref_audio"] = [reference base64EncodedStringWithOptions:0];
			}
			if (streaming) {
				body[@"stream"] = @YES;
			}
			break;
		case NFKRemoteSpeechAPIStyleXAI: {
			body[@"text"] = text;
			body[@"voice_id"] = voice;
			body[@"language"] = parameters[NFKParameterSourceLanguage] ?: @"auto";
			NSMutableDictionary *output = [NSMutableDictionary dictionaryWithObject:format forKey:@"codec"];
			output[@"sample_rate"] = parameters[NFKParameterSampleRate];
			body[@"output_format"] = output;
			break;
		}
	}
	if (self.modelName.length > 0) {
		body[@"model"] = self.modelName;
	}
	NSSet<NSString *> *mapped = NFKSpeechMappedParameters();
	for (NSString *key in parameters) {
		if (![mapped containsObject:key] && ![key isEqualToString:@"voice"] && ![key isEqualToString:@"response_format"]) {
			body[key] = parameters[key];
		}
	}
	return body;
}

- (nullable NSMutableURLRequest *)urlRequestForText:(NSString *)text
											request:(NFKInferenceRequest *)request
										  streaming:(BOOL)streaming
											  error:(NSError * _Nullable *)outError
{
	NSError *encodeError = nil;
	NSData *payload = [NSJSONSerialization dataWithJSONObject:[self bodyForText:text request:request streaming:streaming]
													  options:0 error:&encodeError];
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

// Mistral answers with the audio in base64 inside JSON; the other styles answer with the bytes.
- (nullable NSData *)audioForText:(NSString *)text request:(NFKInferenceRequest *)request error:(NSError * _Nullable *)outError
{
	NSMutableURLRequest *urlRequest = [self urlRequestForText:text request:request streaming:NO error:outError];
	if (urlRequest == nil) {
		return nil;
	}
	NSHTTPURLResponse *response = nil;
	NSError *sendError = nil;
	NSData *reply = [self sendRequest:urlRequest response:&response error:&sendError];
	if (reply == nil) {
		if (outError != NULL) { *outError = sendError; }
		return nil;
	}
	NSError *statusError = [NFKRemoteTransport errorForResponse:response data:reply];
	if (statusError != nil) {
		if (outError != NULL) { *outError = statusError; }
		return nil;
	}
	if (self.apiStyle == NFKRemoteSpeechAPIStyleMistral) {
		id body = [NSJSONSerialization JSONObjectWithData:reply options:0 error:NULL];
		NSString *encoded = [body isKindOfClass:NSDictionary.class] ? body[@"audio_data"] : nil;
		reply = [encoded isKindOfClass:NSString.class]
			? [[NSData alloc] initWithBase64EncodedString:encoded options:NSDataBase64DecodingIgnoreUnknownCharacters] : nil;
	}
	if (reply.length == 0) {
		[self failWithCode:kNFKError_InferenceBackendFailure reason:@"the endpoint returned no audio" error:outError];
		return nil;
	}
	return reply;
}

// OpenAI's speech.audio.delta carries audio, Together's audio.tts.chunk b64, Mistral's
// speech.audio.delta audio_data.
- (nullable NSData *)audioInStreamEvent:(NSDictionary *)event
{
	for (NSString *key in @[ @"audio", @"b64", @"audio_data" ]) {
		if ([event[key] isKindOfClass:NSString.class]) {
			return [[NSData alloc] initWithBase64EncodedString:event[key] options:NSDataBase64DecodingIgnoreUnknownCharacters];
		}
	}
	return nil;
}

#pragma mark Text and audio

- (nullable NSString *)textForRequest:(NFKInferenceRequest *)request
{
	NSString *prompt = request.prompt;
	if (prompt.length > 0) {
		return prompt;
	}
	NSMutableArray<NSString *> *pieces = [NSMutableArray array];
	for (NSDictionary *message in request.messages) {
		if ([message isKindOfClass:NSDictionary.class] && [message[@"content"] isKindOfClass:NSString.class]) {
			[pieces addObject:message[@"content"]];
		}
	}
	return pieces.count > 0 ? [pieces componentsJoinedByString:@"\n"] : nil;
}

// Pieces no longer than the limit, cut after a sentence where one fits, after a word where a
// sentence does not, and mid-word only for a word longer than the limit.
- (NSArray<NSString *> *)piecesOfText:(NSString *)text
{
	NSUInteger limit = self.maximumInputLength;
	if (limit == 0 || text.length <= limit) {
		return @[ text ];
	}
	NSMutableArray<NSString *> *units = [NSMutableArray array];
	[text enumerateSubstringsInRange:NSMakeRange(0, text.length) options:NSStringEnumerationBySentences
						  usingBlock:^(NSString *sentence, NSRange range, NSRange enclosing, BOOL *stop) {
		NSString *trimmed = [[text substringWithRange:enclosing] stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceAndNewlineCharacterSet];
		if (trimmed.length <= limit) {
			[units addObject:trimmed];
			return;
		}
		for (NSString *word in [trimmed componentsSeparatedByCharactersInSet:NSCharacterSet.whitespaceAndNewlineCharacterSet]) {
			for (NSUInteger start = 0; start < word.length; start += limit) {
				[units addObject:[word substringWithRange:NSMakeRange(start, MIN(limit, word.length - start))]];
			}
		}
	}];
	NSMutableArray<NSString *> *pieces = [NSMutableArray array];
	NSMutableString *current = [NSMutableString string];
	for (NSString *unit in units) {
		if (unit.length == 0) {
			continue;
		}
		if (current.length > 0 && current.length + 1 + unit.length > limit) {
			[pieces addObject:[current copy]];
			[current setString:@""];
		}
		if (current.length > 0) {
			[current appendString:@" "];
		}
		[current appendString:unit];
	}
	if (current.length > 0) {
		[pieces addObject:[current copy]];
	}
	return pieces;
}

// WAV pieces are joined by their data chunks under the first piece's format; the other containers
// concatenate as they stand (MP3 frames and raw PCM both do).
- (NSData *)joinedAudio:(NSArray<NSData *> *)pieces format:(NSString *)format
{
	if (pieces.count == 1 || ![format.lowercaseString isEqualToString:@"wav"]) {
		NSMutableData *joined = [NSMutableData data];
		for (NSData *piece in pieces) {
			[joined appendData:piece];
		}
		return joined;
	}
	NSRange formatChunk = NFKSpeechWAVChunk(pieces.firstObject, "fmt ");
	if (formatChunk.location == NSNotFound) {
		return pieces.firstObject;
	}
	NSMutableData *samples = [NSMutableData data];
	for (NSData *piece in pieces) {
		NSRange data = NFKSpeechWAVChunk(piece, "data");
		if (data.location != NSNotFound) {
			[samples appendData:[piece subdataWithRange:data]];
		}
	}
	NSData *formatBytes = [pieces.firstObject subdataWithRange:formatChunk];
	NSMutableData *wav = [NSMutableData data];
	[wav appendBytes:"RIFF" length:4];
	NFKSpeechAppendUInt32(wav, (uint32_t)(4 + 8 + formatBytes.length + 8 + samples.length));
	[wav appendBytes:"WAVEfmt " length:8];
	NFKSpeechAppendUInt32(wav, (uint32_t)formatBytes.length);
	[wav appendData:formatBytes];
	[wav appendBytes:"data" length:4];
	NFKSpeechAppendUInt32(wav, (uint32_t)samples.length);
	[wav appendData:samples];
	return wav;
}

- (nullable NFKInferenceResult *)resultForAudio:(NSData *)audio format:(NSString *)format error:(NSError * _Nullable *)outError
{
	if (audio.length == 0) {
		return [self failWithCode:kNFKError_InferenceBackendFailure reason:@"the endpoint returned no audio" error:outError];
	}
	NSURL *fileURL = [self writeAudio:audio format:format error:outError];
	if (fileURL == nil) {
		return nil;
	}
	return [NFKInferenceResult resultWithOutputs:@{ NFKOutputAudio: [NFKAudioAsset audioAssetWithFileURL:fileURL] }];
}

- (nullable NSURL *)writeAudio:(NSData *)audio format:(NSString *)format error:(NSError * _Nullable *)outError
{
	NSURL *directory = self.outputDirectoryURL
		?: [[NSURL fileURLWithPath:NSTemporaryDirectory()] URLByAppendingPathComponent:@"InferKit" isDirectory:YES];
	NSError *error = nil;
	if (![NSFileManager.defaultManager createDirectoryAtURL:directory withIntermediateDirectories:YES attributes:nil error:&error]) {
		if (outError != NULL) { *outError = error; }
		return nil;
	}
	NSString *extension = format.length > 0 ? format : @"wav";
	NSURL *fileURL = [directory URLByAppendingPathComponent:[NSString stringWithFormat:@"speech-%@.%@", NSUUID.UUID.UUIDString, extension]];
	if (![audio writeToURL:fileURL options:NSDataWritingAtomic error:&error]) {
		if (outError != NULL) { *outError = error; }
		return nil;
	}
	return fileURL;
}

#pragma mark Voices

- (nullable NSArray<NFKRemoteVoice *> *)availableVoicesWithError:(NSError * _Nullable *)outError
{
	NSURL *url = [self voicesURL];
	if (url == nil) {
		[self failWithCode:kNFKError_InferenceUnsupported reason:@"this service lists no voices; its set is fixed and documented" error:outError];
		return nil;
	}
	NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:url];
	request.timeoutInterval = self.timeout;
	[NFKRemoteTransport authorizeRequest:request apiKey:self.apiKey style:NFKRemoteAPIStyleOpenAIChat];
	NSHTTPURLResponse *response = nil;
	NSError *sendError = nil;
	NSData *data = [self sendRequest:request response:&response error:&sendError];
	NSError *failure = data == nil ? sendError : [NFKRemoteTransport errorForResponse:response data:data];
	if (failure != nil) {
		if (outError != NULL) { *outError = failure; }
		return nil;
	}
	id body = [NSJSONSerialization JSONObjectWithData:data options:0 error:NULL];
	NSArray *entries = [body isKindOfClass:NSArray.class] ? body : nil;
	for (NSString *key in @[ @"voices", @"data", @"items" ]) {
		if (entries == nil && [body isKindOfClass:NSDictionary.class] && [body[key] isKindOfClass:NSArray.class]) {
			entries = body[key];
		}
	}
	NSMutableArray<NFKRemoteVoice *> *voices = [NSMutableArray array];
	for (NSDictionary *entry in entries) {
		NFKRemoteVoice *voice = [entry isKindOfClass:NSDictionary.class] ? [self voiceFromEntry:entry] : nil;
		if (voice != nil) {
			[voices addObject:voice];
		}
	}
	return voices;
}

// xAI lists at /tts/voices, Mistral at /audio/voices, Together at /voices?model=.
- (nullable NSURL *)voicesURL
{
	switch (self.apiStyle) {
		case NFKRemoteSpeechAPIStyleXAI:
			return [self.endpointURL URLByAppendingPathComponent:@"voices"];
		case NFKRemoteSpeechAPIStyleMistral:
			return [self.endpointURL.URLByDeletingLastPathComponent URLByAppendingPathComponent:@"voices"];
		case NFKRemoteSpeechAPIStyleOpenAI: {
			if (![self.endpointURL.host containsString:@"together"]) {
				return nil;
			}
			NSURLComponents *components = [NSURLComponents componentsWithURL:[self.endpointURL.URLByDeletingLastPathComponent.URLByDeletingLastPathComponent
																			  URLByAppendingPathComponent:@"voices"]
												   resolvingAgainstBaseURL:NO];
			if (self.modelName.length > 0) {
				components.queryItems = @[ [NSURLQueryItem queryItemWithName:@"model" value:self.modelName] ];
			}
			return components.URL;
		}
	}
	return nil;
}

- (nullable NFKRemoteVoice *)voiceFromEntry:(NSDictionary *)entry
{
	id identifier = entry[@"voice_id"] ?: entry[@"id"] ?: entry[@"name"];
	if (![identifier isKindOfClass:NSString.class]) {
		return nil;
	}
	NSMutableArray<NSString *> *languages = [NSMutableArray array];
	id language = entry[@"languages"] ?: entry[@"language"];
	if ([language isKindOfClass:NSString.class]) {
		[languages addObject:language];
	} else if ([language isKindOfClass:NSArray.class]) {
		for (id each in language) {
			if ([each isKindOfClass:NSString.class]) {
				[languages addObject:each];
			}
		}
	}
	NSString *name = [entry[@"name"] isKindOfClass:NSString.class] ? entry[@"name"] : nil;
	return [[NFKRemoteVoice alloc] initWithIdentifier:identifier name:name languages:languages raw:entry];
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
