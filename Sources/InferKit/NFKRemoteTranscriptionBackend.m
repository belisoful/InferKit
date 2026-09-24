//
//  NFKRemoteTranscriptionBackend.m
//  InferKit
//

#import "NFKRemoteTranscriptionBackend.h"
#import "NFKRemoteTransport.h"
#import "NFKRemoteProvider.h"
#import "NFKInferenceRequest.h"
#import "NFKInferenceResult.h"
#import "NFKInferenceJob.h"
#import "NFKInferenceKeys.h"
#import "NFKAudioAsset.h"
#import "NFKAudioSegment.h"
#import "NFKErrors.h"

/*! The contract keys this backend translates; every other parameter goes out as a form field. */
static NSSet<NSString *> *NFKTranscriptionMappedParameters(void)
{
	static NSSet<NSString *> *mapped;
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		mapped = [NSSet setWithArray:@[ NFKParameterSourceLanguage, NFKParameterSpeakerDiarization,
										NFKParameterWordTimestamps, NFKParameterVocabulary ]];
	});
	return mapped;
}

/*! The audio a request carries: its bytes, or the hosted URL they live at. */
@interface NFKTranscriptionAudio : NSObject
@property (nonatomic, copy, nullable) NSData *data;
@property (nonatomic, copy, nullable) NSURL *remoteURL;
@property (nonatomic, copy) NSString *filename;
@end

@implementation NFKTranscriptionAudio
@end

/*! What a streamed transcription assembles into. */
@interface NFKTranscriptionStreamState : NSObject
@property (nonatomic, strong) NSMutableString *text;
@property (nonatomic, strong) NSMutableArray<NSDictionary *> *segments;
@property (nonatomic, copy, nullable) NSDictionary *finalBody;
@property (nonatomic, assign) BOOL finished;
@end

@implementation NFKTranscriptionStreamState
- (instancetype)init
{
	self = [super init];
	if (self != nil) {
		_text = [NSMutableString string];
		_segments = [NSMutableArray array];
	}
	return self;
}
@end

@implementation NFKRemoteTranscriptionBackend

+ (instancetype)backendWithEndpointURL:(nullable NSURL *)endpointURL
{
	NFKRemoteTranscriptionBackend *backend = [[self alloc] init];
	backend.endpointURL = endpointURL;
	return backend;
}

+ (nullable instancetype)backendForProvider:(NFKRemoteProvider *)provider
									 apiKey:(nullable NSString *)apiKey
								  modelName:(nullable NSString *)modelName
{
	NSString *identifier = provider.identifier;
	NFKRemoteTranscriptionBackend *backend = nil;
	if ([@[ @"openai", @"groq", @"together", @"openrouter", @"vllm" ] containsObject:identifier]) {
		backend = [self backendWithEndpointURL:[provider URLForPath:@"audio/transcriptions"]];
		NSDictionary<NSString *, NSString *> *urlFields = @{ @"groq": @"url", @"together": @"file" };
		backend.audioURLFieldName = urlFields[identifier];
	} else if ([identifier isEqualToString:@"mistral"]) {
		backend = [self backendWithEndpointURL:[provider URLForPath:@"audio/transcriptions"]];
		backend.apiStyle = NFKRemoteTranscriptionAPIStyleMistral;
		backend.audioURLFieldName = @"file_url";
	} else if ([identifier isEqualToString:@"xai"]) {
		backend = [self backendWithEndpointURL:[provider URLForPath:@"stt"]];
		backend.apiStyle = NFKRemoteTranscriptionAPIStyleXAI;
		backend.audioURLFieldName = @"url";
	}
	backend.apiKey = apiKey;
	backend.modelName = modelName;
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
	return @"remote-transcription";
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
	NSData *responseData = [self sendRequest:urlRequest response:&response error:&sendError];
	if (responseData == nil) {
		[self propagateError:sendError to:outError];
		return nil;
	}
	NSError *statusError = [NFKRemoteTransport errorForResponse:response data:responseData];
	if (statusError != nil) {
		[self propagateError:statusError to:outError];
		return nil;
	}
	return [self resultFromResponseData:responseData error:outError];
}

- (NFKInferenceJob *)submitInferenceJobForRequest:(NFKInferenceRequest *)request
{
	NFKInferenceJob *job = [[NFKInferenceJob alloc] init];
	if (!self.streams || self.apiStyle == NFKRemoteTranscriptionAPIStyleXAI) {
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
				[job finishWithError:error ?: [NFKRemoteTransport errorWithCode:kNFKError_InferenceBackendFailure reason:@"the transcription call failed"]];
			}
		});
		return job;
	}
	NSError *error = nil;
	NSMutableURLRequest *urlRequest = [self urlRequestForRequest:request streaming:YES error:&error];
	if (urlRequest == nil) {
		[job finishWithError:error ?: [NFKRemoteTransport errorWithCode:kNFKError_InferenceBackendFailure reason:@"the transcription call failed"]];
		return job;
	}
	[job reportProgress:-1.0];
	NFKTranscriptionStreamState *state = [[NFKTranscriptionStreamState alloc] init];
	void (^cancel)(void) = [self streamRequest:urlRequest lineHandler:^(NSString *line) {
		if (state.finished) {
			return;
		}
		NSString *payload = [NFKRemoteTransport SSEDataForLine:line];
		id event = payload != nil ? [NSJSONSerialization JSONObjectWithData:[payload dataUsingEncoding:NSUTF8StringEncoding] options:0 error:NULL] : nil;
		if (![event isKindOfClass:NSDictionary.class]) {
			return;
		}
		if ([self applyStreamEvent:event toState:state]) {
			[job reportProgress:-1.0 partialResult:[NFKInferenceResult resultWithOutputs:@{ NFKOutputText: [state.text copy] }]];
		}
	} completionHandler:^(NSHTTPURLResponse * _Nullable response, NSData * _Nullable errorBody, NSError * _Nullable streamError) {
		if (state.finished || job.status == NFKInferenceJobStatusCancelled) {
			return;
		}
		state.finished = YES;
		NSError *failure = streamError ?: [NFKRemoteTransport errorForResponse:response data:errorBody];
		if (failure != nil) {
			[job finishWithError:failure ?: [NFKRemoteTransport errorWithCode:kNFKError_InferenceBackendFailure reason:@"the transcription call failed"]];
			return;
		}
		[job finishWithResult:[self resultFromStreamState:state]];
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
	if (self.translates && self.apiStyle != NFKRemoteTranscriptionAPIStyleOpenAI) {
		[self setError:outError code:kNFKError_InferenceUnsupported reason:@"only the OpenAI-style services translate"];
		return nil;
	}
	NFKTranscriptionAudio *audio = [self audioForRequest:request];
	if (audio == nil) {
		[self setError:outError code:kNFKError_InferenceMissingInput reason:@"no audio is set under NFKInputAudio"];
		return nil;
	}
	if (audio.data == nil && self.audioURLFieldName == nil) {
		audio.data = [self fetchAudioAtURL:audio.remoteURL];
		if (audio.data == nil) {
			[self setError:outError code:kNFKError_InferenceMissingInput reason:@"the hosted audio could not be fetched"];
			return nil;
		}
	}

	NSString *boundary = [@"InferKitBoundary-" stringByAppendingString:NSUUID.UUID.UUIDString];
	NSMutableURLRequest *urlRequest = [NSMutableURLRequest requestWithURL:[self requestURL]];
	urlRequest.HTTPMethod = @"POST";
	urlRequest.timeoutInterval = self.timeout;
	urlRequest.HTTPBody = [self multipartBodyForAudio:audio fields:[self fieldsForRequest:request streaming:streaming] boundary:boundary];
	[urlRequest setValue:[NSString stringWithFormat:@"multipart/form-data; boundary=%@", boundary] forHTTPHeaderField:@"Content-Type"];
	if (streaming) {
		[urlRequest setValue:@"text/event-stream" forHTTPHeaderField:@"Accept"];
	}
	[NFKRemoteTransport authorizeRequest:urlRequest apiKey:self.apiKey style:NFKRemoteAPIStyleOpenAIChat];
	return urlRequest;
}

// The translations endpoint is the transcriptions one's sibling, so the switch replaces the path's
// last component rather than asking for a second URL.
- (NSURL *)requestURL
{
	if (!self.translates) {
		return self.endpointURL;
	}
	NSURL *base = [self.endpointURL URLByDeletingLastPathComponent];
	return [base URLByAppendingPathComponent:@"translations"];
}

// Ordered name–value pairs; a list repeats its name, with [] where the style reads that spelling.
- (NSArray<NSArray<NSString *> *> *)fieldsForRequest:(NFKInferenceRequest *)request streaming:(BOOL)streaming
{
	NSMutableArray<NSArray<NSString *> *> *fields = [NSMutableArray array];
	void (^add)(NSString *, id) = ^(NSString *name, id value) {
		if (value != nil) {
			[fields addObject:@[ name, [value description] ]];
		}
	};
	NSDictionary *parameters = request.parameters;
	BOOL diarizes = [parameters[NFKParameterSpeakerDiarization] boolValue];
	BOOL words = [parameters[NFKParameterWordTimestamps] boolValue];
	NSArray *vocabulary = [parameters[NFKParameterVocabulary] isKindOfClass:NSArray.class] ? parameters[NFKParameterVocabulary] : nil;
	NSString *prompt = [request.prompt isKindOfClass:NSString.class] && request.prompt.length > 0 ? request.prompt : nil;

	if (self.modelName.length > 0) {
		add(@"model", self.modelName);
	}
	add(@"language", parameters[NFKParameterSourceLanguage]);
	switch (self.apiStyle) {
		case NFKRemoteTranscriptionAPIStyleOpenAI: {
			add(@"prompt", prompt);
			BOOL diarizingModel = [self.modelName containsString:@"diarize"];
			if (parameters[@"response_format"] == nil) {
				if (diarizes && diarizingModel) {
					add(@"response_format", @"diarized_json");
					add(@"chunking_strategy", @"auto");
				} else if (self.emitsTimestamps || words || diarizes) {
					add(@"response_format", @"verbose_json");
				}
			}
			if (self.emitsTimestamps || words) {
				add(@"timestamp_granularities[]", @"segment");
			}
			if (words) {
				add(@"timestamp_granularities[]", @"word");
			}
			if (diarizes && !diarizingModel) {
				add(@"diarize", @"true");
			}
			for (NSString *term in vocabulary) {
				add(@"keywords[]", term);
			}
			break;
		}
		case NFKRemoteTranscriptionAPIStyleMistral:
			if (diarizes) {
				add(@"diarize", @"true");
			}
			if (self.emitsTimestamps || diarizes) {
				add(@"timestamp_granularities", @"segment");
			}
			if (words) {
				add(@"timestamp_granularities", @"word");
			}
			for (NSString *term in vocabulary) {
				add(@"context_bias", term);
			}
			break;
		case NFKRemoteTranscriptionAPIStyleXAI:
			if (diarizes) {
				add(@"diarize", @"true");
			}
			for (NSString *term in vocabulary) {
				add(@"keyterm", term);
			}
			break;
	}
	if (streaming) {
		add(@"stream", @"true");
	}
	NSSet<NSString *> *mapped = NFKTranscriptionMappedParameters();
	for (NSString *key in parameters) {
		if (![mapped containsObject:key]) {
			add(key, parameters[key]);
		}
	}
	return fields;
}

#pragma mark Audio and body

- (nullable NFKTranscriptionAudio *)audioForRequest:(NFKInferenceRequest *)request
{
	id input = [request inputForKey:NFKInputAudio];
	NFKTranscriptionAudio *audio = [[NFKTranscriptionAudio alloc] init];
	audio.filename = @"audio.wav";
	if ([input isKindOfClass:NSData.class]) {
		audio.data = input;
		return audio;
	}
	NSURL *url = [input isKindOfClass:NFKAudioAsset.class] ? [(NFKAudioAsset *)input fileURL] : nil;
	if (url == nil) {
		return nil;
	}
	if (url.lastPathComponent.length > 0) {
		audio.filename = url.lastPathComponent;
	}
	if (url.isFileURL) {
		audio.data = [NSData dataWithContentsOfURL:url];
		return audio.data != nil ? audio : nil;
	}
	audio.remoteURL = url;
	return audio;
}

- (nullable NSData *)fetchAudioAtURL:(NSURL *)url
{
	NSMutableURLRequest *request = [NSMutableURLRequest requestWithURL:url];
	request.timeoutInterval = self.timeout;
	NSHTTPURLResponse *response = nil;
	NSData *data = [self sendRequest:request response:&response error:NULL];
	return data != nil && [NFKRemoteTransport errorForResponse:response data:data] == nil ? data : nil;
}

// The file goes last: xAI reads the fields before it and refuses them after.
- (NSData *)multipartBodyForAudio:(NFKTranscriptionAudio *)audio
						   fields:(NSArray<NSArray<NSString *> *> *)fields
						 boundary:(NSString *)boundary
{
	NSMutableData *body = [NSMutableData data];
	NSData *dashBoundary = [[NSString stringWithFormat:@"--%@\r\n", boundary] dataUsingEncoding:NSUTF8StringEncoding];
	void (^appendField)(NSString *, NSString *) = ^(NSString *name, NSString *value) {
		[body appendData:dashBoundary];
		NSString *disposition = [NSString stringWithFormat:@"Content-Disposition: form-data; name=\"%@\"\r\n\r\n%@\r\n", name, value];
		[body appendData:[disposition dataUsingEncoding:NSUTF8StringEncoding]];
	};
	for (NSArray<NSString *> *field in fields) {
		appendField(field[0], field[1]);
	}
	if (audio.data == nil) {
		appendField(self.audioURLFieldName, audio.remoteURL.absoluteString);
	} else {
		[body appendData:dashBoundary];
		NSString *fileHeader = [NSString stringWithFormat:@"Content-Disposition: form-data; name=\"file\"; filename=\"%@\"\r\nContent-Type: application/octet-stream\r\n\r\n", audio.filename];
		[body appendData:[fileHeader dataUsingEncoding:NSUTF8StringEncoding]];
		[body appendData:audio.data];
		[body appendData:[@"\r\n" dataUsingEncoding:NSUTF8StringEncoding]];
	}
	[body appendData:[[NSString stringWithFormat:@"--%@--\r\n", boundary] dataUsingEncoding:NSUTF8StringEncoding]];
	return body;
}

#pragma mark Response

- (nullable NFKInferenceResult *)resultFromResponseData:(NSData *)data error:(NSError * _Nullable *)outError
{
	id object = [NSJSONSerialization JSONObjectWithData:data options:0 error:NULL];
	if ([object isKindOfClass:NSDictionary.class]) {
		return [self resultFromBody:object];
	}
	// response_format=text/srt/vtt returns a plain body, not JSON; take it as the transcript.
	NSString *text = [[NSString alloc] initWithData:data encoding:NSUTF8StringEncoding];
	if (text == nil) {
		[self setError:outError code:kNFKError_InferenceBackendFailure reason:@"the response is neither JSON nor text"];
		return nil;
	}
	return [NFKInferenceResult resultWithOutputs:@{ NFKOutputText: text }];
}

- (NFKInferenceResult *)resultFromBody:(NSDictionary *)body
{
	NSMutableDictionary<NSString *, id> *outputs = [NSMutableDictionary dictionary];
	if ([body[@"text"] isKindOfClass:NSString.class]) {
		outputs[NFKOutputText] = body[@"text"];
	}
	outputs[NFKOutputStructured] = body;
	NSArray<NFKAudioSegment *> *words = [self spansIn:body[@"words"] textKey:@"word" fallbackTextKey:@"text"];
	if (words.count > 0) {
		outputs[NFKOutputWords] = words;
	}
	NSArray<NFKAudioSegment *> *segments = [self spansIn:body[@"segments"] textKey:@"text" fallbackTextKey:nil];
	if (segments == nil && words.count > 0) {
		segments = [self turnsFromWords:words];
	}
	if (segments != nil) {
		outputs[NFKOutputSegments] = segments;
	}
	return [NFKInferenceResult resultWithOutputs:outputs];
}

// A span's start, end, text, and speaker, under whichever of the services' spellings it carries. The
// decoder's mean log probability becomes the confidence as e to that power; a score or confidence
// is taken as it stands.
- (nullable NSArray<NFKAudioSegment *> *)spansIn:(id)entries textKey:(NSString *)textKey fallbackTextKey:(nullable NSString *)fallbackTextKey
{
	if (![entries isKindOfClass:NSArray.class]) {
		return nil;
	}
	NSMutableArray<NFKAudioSegment *> *spans = [NSMutableArray array];
	for (NSDictionary *entry in entries) {
		if (![entry isKindOfClass:NSDictionary.class] ||
			![entry[@"start"] isKindOfClass:NSNumber.class] || ![entry[@"end"] isKindOfClass:NSNumber.class]) {
			continue;
		}
		id rawText = entry[textKey] ?: (fallbackTextKey != nil ? entry[fallbackTextKey] : nil);
		NSString *text = [rawText isKindOfClass:NSString.class]
			? [rawText stringByTrimmingCharactersInSet:NSCharacterSet.whitespaceCharacterSet] : nil;
		double confidence = 1.0;
		if ([entry[@"avg_logprob"] isKindOfClass:NSNumber.class]) {
			confidence = MIN(1.0, exp([entry[@"avg_logprob"] doubleValue]));
		} else if ([entry[@"confidence"] isKindOfClass:NSNumber.class]) {
			confidence = [entry[@"confidence"] doubleValue];
		} else if ([entry[@"score"] isKindOfClass:NSNumber.class]) {
			confidence = [entry[@"score"] doubleValue];
		}
		id speaker = entry[@"speaker"] ?: entry[@"speaker_id"];
		NSString *speakerName = [speaker isKindOfClass:NSString.class] ? speaker
							  : [speaker isKindOfClass:NSNumber.class] ? [speaker stringValue] : nil;
		[spans addObject:[NFKAudioSegment segmentWithStartSeconds:[entry[@"start"] doubleValue]
													   endSeconds:[entry[@"end"] doubleValue]
															label:text
													   confidence:confidence
														  speaker:speakerName]];
	}
	return spans;
}

// A reply that times words but not segments (xAI) is grouped into turns: a turn ends where the
// speaker changes or a word closes a sentence.
- (NSArray<NFKAudioSegment *> *)turnsFromWords:(NSArray<NFKAudioSegment *> *)words
{
	NSMutableArray<NFKAudioSegment *> *turns = [NSMutableArray array];
	NSMutableArray<NSString *> *pieces = [NSMutableArray array];
	NFKAudioSegment *first = nil;
	double confidenceSum = 0;
	NSCharacterSet *sentenceEnds = [NSCharacterSet characterSetWithCharactersInString:@".?!"];
	for (NSUInteger index = 0; index < words.count; index++) {
		NFKAudioSegment *word = words[index];
		first = first ?: word;
		[pieces addObject:word.label ?: @""];
		confidenceSum += word.confidence;
		NFKAudioSegment *next = index + 1 < words.count ? words[index + 1] : nil;
		BOOL speakerChanges = next != nil && !(next.speaker == word.speaker || [next.speaker isEqual:word.speaker]);
		BOOL closesSentence = word.label.length > 0
			&& [sentenceEnds characterIsMember:[word.label characterAtIndex:word.label.length - 1]];
		if (next == nil || speakerChanges || closesSentence) {
			[turns addObject:[NFKAudioSegment segmentWithStartSeconds:first.startSeconds
														   endSeconds:word.endSeconds
																label:[pieces componentsJoinedByString:@" "]
														   confidence:confidenceSum / pieces.count
															  speaker:first.speaker]];
			[pieces removeAllObjects];
			first = nil;
			confidenceSum = 0;
		}
	}
	return turns;
}

#pragma mark Streaming

// OpenAI streams transcript.text.delta and ends with transcript.text.done; Mistral streams
// transcription.text.delta and transcription.segment and ends with transcription.done. Returns
// whether the event grew the text.
- (BOOL)applyStreamEvent:(NSDictionary *)event toState:(NFKTranscriptionStreamState *)state
{
	NSString *type = [event[@"type"] isKindOfClass:NSString.class] ? event[@"type"] : @"";
	if ([type hasSuffix:@".text.delta"] && [event[@"delta"] isKindOfClass:NSString.class]) {
		[state.text appendString:event[@"delta"]];
		return YES;
	}
	if ([type isEqualToString:@"transcription.segment"]) {
		[state.segments addObject:event];
		return NO;
	}
	if ([type isEqualToString:@"transcript.text.done"] || [type isEqualToString:@"transcription.done"]) {
		state.finalBody = event;
	}
	return NO;
}

- (NFKInferenceResult *)resultFromStreamState:(NFKTranscriptionStreamState *)state
{
	NSMutableDictionary *body = [state.finalBody mutableCopy] ?: [NSMutableDictionary dictionary];
	if (![body[@"text"] isKindOfClass:NSString.class]) {
		body[@"text"] = [state.text copy];
	}
	if (body[@"segments"] == nil && state.segments.count > 0) {
		body[@"segments"] = [state.segments copy];
	}
	return [self resultFromBody:body];
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
											  userInfo:@{ NSLocalizedDescriptionKey: @"the transcription call failed" }];
	return NO;
}

@end
