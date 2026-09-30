//
//  NFKServedOpenAI.m
//  InferKit
//

#import "NFKServedOpenAI.h"
#import "NFKRemoteMediaSupport.h"
#import "NFKInferenceWireCoding.h"
#import <InferKit/NFKInferenceKeys.h>
#import <InferKit/NFKImageCoding.h>
#import <InferKit/NFKAudioAsset.h>
#import <InferKit/NFKVideoAsset.h>
#import <InferKit/NFKAudioSegment.h>
#import <InferKit/NFKErrors.h>
#import <AVFoundation/AVFoundation.h>
#import <ImageIO/ImageIO.h>

/*! The contract key for each chat field the client renames, the inverse of the client's wire names. */
static NSDictionary<NSString *, NSString *> *NFKServedChatParameterNames(void)
{
	static NSDictionary<NSString *, NSString *> *names;
	static dispatch_once_t once;
	dispatch_once(&once, ^{
		names = @{ @"max_tokens": NFKParameterMaxTokens,
				   @"max_completion_tokens": NFKParameterMaxTokens,
				   @"top_p": NFKParameterTopP,
				   @"top_k": NFKParameterTopK,
				   @"repetition_penalty": NFKParameterRepetitionPenalty,
				   @"repeat_penalty": NFKParameterRepetitionPenalty };
	});
	return names;
}

/*! The contract key for each image field the image client renames. */
static NSDictionary<NSString *, NSString *> *NFKServedImageParameterNames(void)
{
	return @{ @"n": NFKParameterSampleCount, @"seed": NFKParameterSeed, @"steps": NFKParameterSteps,
			  @"guidance_scale": NFKParameterGuidanceScale, @"aspect_ratio": NFKParameterAspectRatio,
			  @"resolution": NFKParameterResolution, @"output_format": NFKParameterOutputFormat };
}

/*! The bytes and media type of a data URL, or nil for anything else. */
static NSData * _Nullable NFKServedDataURLBytes(id url, NSString * _Nullable * _Nullable outMediaType)
{
	if (![url isKindOfClass:NSString.class] || ![url hasPrefix:@"data:"]) {
		return nil;
	}
	NSRange comma = [url rangeOfString:@","];
	if (comma.location == NSNotFound) {
		return nil;
	}
	NSString *header = [url substringWithRange:NSMakeRange(5, comma.location - 5)];
	if (outMediaType != NULL) {
		*outMediaType = [header componentsSeparatedByString:@";"].firstObject;
	}
	NSString *payload = [url substringFromIndex:comma.location + 1];
	if ([header hasSuffix:@";base64"]) {
		return [[NSData alloc] initWithBase64EncodedString:payload options:NSDataBase64DecodingIgnoreUnknownCharacters];
	}
	return [[payload stringByRemovingPercentEncoding] dataUsingEncoding:NSUTF8StringEncoding];
}

/*! The file extension for a media type such as audio/wav or video/mp4. */
static NSString *NFKServedExtensionForMediaType(NSString * _Nullable mediaType, NSString *fallback)
{
	NSString *subtype = [mediaType componentsSeparatedByString:@"/"].lastObject.lowercaseString;
	NSDictionary<NSString *, NSString *> *aliases = @{ @"mpeg": @"mp3", @"x-wav": @"wav", @"wave": @"wav",
													   @"quicktime": @"mov", @"x-m4a": @"m4a", @"mp4": @"mp4" };
	if (subtype.length == 0) {
		return fallback;
	}
	return aliases[subtype] ?: subtype;
}

@implementation NFKServedOpenAI

#pragma mark Chat

+ (nullable NFKInferenceRequest *)chatRequestFromBody:(NSDictionary *)body
									   temporaryFiles:(NSMutableArray<NSURL *> *)temporaryFiles
												error:(NSError * _Nullable *)outError
{
	NSArray *wireMessages = [body[@"messages"] isKindOfClass:NSArray.class] ? body[@"messages"] : nil;
	if (wireMessages.count == 0) {
		[self setError:outError code:kNFKError_InferenceMissingInput reason:@"the request carries no messages"];
		return nil;
	}
	NSMutableArray *messages = [NSMutableArray arrayWithCapacity:wireMessages.count];
	NSMutableArray *images = [NSMutableArray array];
	NSMutableArray *audios = [NSMutableArray array];
	NSMutableArray *videos = [NSMutableArray array];
	NSMutableArray *documents = [NSMutableArray array];
	for (NSDictionary *message in wireMessages) {
		if (![message isKindOfClass:NSDictionary.class]) {
			[self setError:outError code:kNFKError_InferenceMissingInput reason:@"a message is not an object"];
			return nil;
		}
		if (![message[@"content"] isKindOfClass:NSArray.class]) {
			[messages addObject:message];
			continue;
		}
		NSMutableArray<NSString *> *texts = [NSMutableArray array];
		for (NSDictionary *part in message[@"content"]) {
			if (![self readContentPart:part texts:texts images:images audios:audios videos:videos documents:documents
						temporaryFiles:temporaryFiles error:outError]) {
				return nil;
			}
		}
		NSMutableDictionary *flattened = [message mutableCopy];
		flattened[@"content"] = [texts componentsJoinedByString:@"\n"];
		[messages addObject:flattened];
	}

	NSMutableDictionary<NSString *, id> *inputs = [NSMutableDictionary dictionaryWithObject:messages forKey:NFKInputMessages];
	NSDictionary *only = messages.count == 1 ? messages.firstObject : nil;
	if ([only[@"role"] isEqual:@"user"] && [only[@"content"] isKindOfClass:NSString.class]) {
		inputs[NFKInputPrompt] = only[@"content"];
	}
	[self addMedia:images firstKey:NFKInputImage restKey:NFKInputImages to:inputs];
	[self addMedia:documents firstKey:NFKInputDocument restKey:NFKInputDocuments to:inputs];
	inputs[NFKInputAudio] = audios.firstObject;
	inputs[NFKInputVideo] = videos.firstObject;

	NSArray *modalities = [body[@"modalities"] isKindOfClass:NSArray.class] ? body[@"modalities"] : @[];
	NFKModality modality = [modalities containsObject:@"image"] ? NFKModalityImage
		: ([modalities containsObject:@"audio"] ? NFKModalityAudio : NFKModalityText);
	return [NFKInferenceRequest requestWithInputs:inputs parameters:[self chatParametersFromBody:body] outputModality:modality];
}

+ (void)addMedia:(NSArray *)media firstKey:(NSString *)firstKey restKey:(NSString *)restKey to:(NSMutableDictionary *)inputs
{
	if (media.count > 0) {
		inputs[firstKey] = media.firstObject;
	}
	if (media.count > 1) {
		inputs[restKey] = [media subarrayWithRange:NSMakeRange(1, media.count - 1)];
	}
}

// The parts the chat client writes: text, image_url, input_audio (an object, or Mistral's bare
// string), video_url and input_video, and a PDF as file.file_data or document_url.
+ (BOOL)readContentPart:(NSDictionary *)part
				  texts:(NSMutableArray<NSString *> *)texts
				 images:(NSMutableArray *)images
				 audios:(NSMutableArray *)audios
				 videos:(NSMutableArray *)videos
			  documents:(NSMutableArray *)documents
		 temporaryFiles:(NSMutableArray<NSURL *> *)temporaryFiles
				  error:(NSError * _Nullable *)outError
{
	NSString *type = [part isKindOfClass:NSDictionary.class] && [part[@"type"] isKindOfClass:NSString.class] ? part[@"type"] : nil;
	if ([type isEqualToString:@"text"]) {
		[texts addObject:[part[@"text"] isKindOfClass:NSString.class] ? part[@"text"] : @""];
		return YES;
	}
	if ([type isEqualToString:@"image_url"]) {
		id reference = part[@"image_url"];
		id url = [reference isKindOfClass:NSDictionary.class] ? reference[@"url"] : reference;
		NSData *bytes = NFKServedDataURLBytes(url, NULL);
		CVPixelBufferRef buffer = bytes != nil ? [NFKImageCoding pixelBufferWithImageData:bytes] : NULL;
		if (buffer == NULL) {
			return [self setError:outError code:kNFKError_InferenceUnsupported
						   reason:@"an image part must be an inline data URL of an image this server can decode"];
		}
		[images addObject:(__bridge_transfer id)buffer];
		return YES;
	}
	if ([type isEqualToString:@"input_audio"]) {
		id audio = part[@"input_audio"];
		NSString *encoded = [audio isKindOfClass:NSDictionary.class] ? audio[@"data"] : audio;
		NSString *format = [audio isKindOfClass:NSDictionary.class] && [audio[@"format"] isKindOfClass:NSString.class] ? audio[@"format"] : @"wav";
		NSData *bytes = [encoded isKindOfClass:NSString.class] ? [[NSData alloc] initWithBase64EncodedString:encoded options:NSDataBase64DecodingIgnoreUnknownCharacters] : nil;
		NSURL *file = bytes != nil ? [self writeTemporary:bytes extension:format files:temporaryFiles error:outError] : nil;
		if (file == nil) {
			return bytes == nil ? [self setError:outError code:kNFKError_InferenceMissingInput reason:@"an audio part carries no base64 data"] : NO;
		}
		[audios addObject:[NFKAudioAsset audioAssetWithFileURL:file]];
		return YES;
	}
	if ([type isEqualToString:@"video_url"] || [type isEqualToString:@"input_video"]) {
		NSData *bytes = nil;
		NSString *extension = @"mp4";
		if ([type isEqualToString:@"video_url"]) {
			id reference = part[@"video_url"];
			NSString *mediaType = nil;
			bytes = NFKServedDataURLBytes([reference isKindOfClass:NSDictionary.class] ? reference[@"url"] : reference, &mediaType);
			extension = NFKServedExtensionForMediaType(mediaType, @"mp4");
		} else if ([part[@"input_video"] isKindOfClass:NSDictionary.class] && [part[@"input_video"][@"data"] isKindOfClass:NSString.class]) {
			bytes = [[NSData alloc] initWithBase64EncodedString:part[@"input_video"][@"data"] options:NSDataBase64DecodingIgnoreUnknownCharacters];
			extension = [part[@"input_video"][@"format"] isKindOfClass:NSString.class] ? part[@"input_video"][@"format"] : @"mp4";
		}
		NSURL *file = bytes != nil ? [self writeTemporary:bytes extension:extension files:temporaryFiles error:outError] : nil;
		if (file == nil) {
			return bytes == nil ? [self setError:outError code:kNFKError_InferenceUnsupported reason:@"a video part must be inline data"] : NO;
		}
		[videos addObject:[NFKVideoAsset videoAssetWithFileURL:file]];
		return YES;
	}
	if ([type isEqualToString:@"file"] || [type isEqualToString:@"document_url"]) {
		id source = [type isEqualToString:@"file"]
			? ([part[@"file"] isKindOfClass:NSDictionary.class] ? part[@"file"][@"file_data"] : nil)
			: part[@"document_url"];
		NSData *bytes = NFKServedDataURLBytes(source, NULL);
		if (bytes == nil) {
			return [self setError:outError code:kNFKError_InferenceUnsupported
						   reason:@"a document part must be inline data; this server keeps no files"];
		}
		[documents addObject:bytes];
		return YES;
	}
	return [self setError:outError code:kNFKError_InferenceUnsupported
				   reason:[NSString stringWithFormat:@"the content part type \"%@\" is not served", type ?: @"(none)"]];
}

+ (NSDictionary<NSString *, id> *)chatParametersFromBody:(NSDictionary *)body
{
	NSMutableDictionary<NSString *, id> *parameters = [NSMutableDictionary dictionary];
	NSSet<NSString *> *consumed = [NSSet setWithArray:@[ @"model", @"messages", @"stream", @"stream_options", @"n",
														 @"modalities", @"audio", @"tools", @"response_format",
														 @"stop", @"reasoning_effort" ]];
	NSDictionary<NSString *, NSString *> *renamed = NFKServedChatParameterNames();
	for (NSString *key in body) {
		if (renamed[key] != nil) {
			parameters[renamed[key]] = body[key];
		} else if (![consumed containsObject:key]) {
			parameters[key] = body[key];
		}
	}
	id stop = body[@"stop"];
	if ([stop isKindOfClass:NSString.class]) {
		parameters[NFKParameterStopSequences] = @[ stop ];
	} else if ([stop isKindOfClass:NSArray.class]) {
		parameters[NFKParameterStopSequences] = stop;
	}
	NSString *effort = body[@"reasoning_effort"];
	if ([effort isKindOfClass:NSString.class]) {
		NSDictionary<NSString *, NSString *> *levels = @{ @"low": NFKReasoningEffortLight,
														  @"medium": NFKReasoningEffortModerate,
														  @"high": NFKReasoningEffortDeep };
		parameters[NFKParameterReasoningEffort] = levels[effort] ?: effort;
	}
	if ([body[@"tools"] isKindOfClass:NSArray.class]) {
		NSMutableArray *tools = [NSMutableArray array];
		for (NSDictionary *tool in body[@"tools"]) {
			BOOL function = [tool isKindOfClass:NSDictionary.class] && [tool[@"type"] isEqual:@"function"]
				&& [tool[@"function"] isKindOfClass:NSDictionary.class];
			[tools addObject:function ? tool[@"function"] : tool];
		}
		parameters[NFKParameterTools] = tools;
	}
	NSDictionary *format = [body[@"response_format"] isKindOfClass:NSDictionary.class] ? body[@"response_format"] : nil;
	NSDictionary *schema = [format[@"json_schema"] isKindOfClass:NSDictionary.class] ? format[@"json_schema"][@"schema"] : nil;
	if ([format[@"type"] isEqual:@"json_schema"] && [schema isKindOfClass:NSDictionary.class]) {
		parameters[NFKParameterJSONSchema] = schema;
	} else if (format != nil) {
		parameters[@"response_format"] = format;
	}
	if ([body[@"audio"] isKindOfClass:NSDictionary.class]) {
		parameters[NFKParameterAudioOutput] = body[@"audio"];
	}
	return parameters;
}

+ (nullable NSString *)textForResult:(nullable NFKInferenceResult *)result
{
	if (result.text != nil) {
		return result.text;
	}
	NSDictionary *structured = result.structured;
	if (structured == nil || ![NSJSONSerialization isValidJSONObject:structured]) {
		return nil;
	}
	NSData *JSON = [NSJSONSerialization dataWithJSONObject:structured options:NSJSONWritingSortedKeys error:NULL];
	return JSON != nil ? [[NSString alloc] initWithData:JSON encoding:NSUTF8StringEncoding] : nil;
}

+ (nullable NSDictionary<NSString *, id> *)chatMessageForResult:(NFKInferenceResult *)result
												   finishReason:(NSString * _Nullable * _Nullable)outFinishReason
														  error:(NSError * _Nullable *)outError
{
	NSMutableDictionary<NSString *, id> *message = [NSMutableDictionary dictionaryWithObject:@"assistant" forKey:@"role"];
	NSString *text = [self textForResult:result];
	NSArray *toolCalls = [self wireToolCallsForResult:result];
	message[@"content"] = text ?: (toolCalls.count > 0 ? (id)NSNull.null : @"");
	NSString *reasoning = [result outputForKey:NFKOutputReasoning];
	if ([reasoning isKindOfClass:NSString.class] && reasoning.length > 0) {
		message[@"reasoning_content"] = reasoning;
	}
	if (toolCalls.count > 0) {
		message[@"tool_calls"] = toolCalls;
	}
	NSMutableArray *images = [NSMutableArray array];
	for (id image in [self imagesForResult:result]) {
		NSString *dataURL = [NFKImageCoding dataURLForImage:image];
		if (dataURL == nil) {
			[self setError:outError code:kNFKError_InferenceUnsupported reason:@"a generated image's pixel format cannot be encoded as PNG"];
			return nil;
		}
		[images addObject:@{ @"type": @"image_url", @"image_url": @{ @"url": dataURL } }];
	}
	if (images.count > 0) {
		message[@"images"] = images;
	}
	id audio = [result outputForKey:NFKOutputAudio];
	if (audio != nil) {
		NSString *format = nil;
		NSData *bytes = [self bytesForAudio:audio format:&format];
		if (bytes == nil) {
			[self setError:outError code:kNFKError_InferenceUnsupported reason:@"the spoken reply cannot be read"];
			return nil;
		}
		message[@"audio"] = @{ @"id": [@"audio_" stringByAppendingString:NSUUID.UUID.UUIDString],
							   @"data": [bytes base64EncodedStringWithOptions:0],
							   @"transcript": text ?: @"",
							   @"format": format ?: @"wav" };
	}
	NSArray *annotations = [self annotationsForResult:result];
	if (annotations.count > 0) {
		message[@"annotations"] = annotations;
	}
	if (outFinishReason != NULL) {
		*outFinishReason = toolCalls.count > 0 ? @"tool_calls" : @"stop";
	}
	return message;
}

+ (NSArray *)imagesForResult:(NFKInferenceResult *)result
{
	NSArray *images = [result outputForKey:NFKOutputImages];
	if ([images isKindOfClass:NSArray.class] && images.count > 0) {
		return images;
	}
	id image = [result outputForKey:NFKOutputImage];
	return image != nil ? @[ image ] : @[];
}

+ (NSArray<NSDictionary *> *)wireToolCallsForResult:(NFKInferenceResult *)result
{
	NSMutableArray<NSDictionary *> *calls = [NSMutableArray array];
	for (NSDictionary *call in result.toolCalls ?: @[]) {
		if (![call isKindOfClass:NSDictionary.class] || ![call[@"name"] isKindOfClass:NSString.class]) {
			continue;
		}
		NSString *arguments = [call[@"argumentsJSON"] isKindOfClass:NSString.class] ? call[@"argumentsJSON"] : nil;
		if (arguments == nil && [NSJSONSerialization isValidJSONObject:call[@"arguments"] ?: @{}]) {
			NSData *JSON = [NSJSONSerialization dataWithJSONObject:call[@"arguments"] ?: @{} options:0 error:NULL];
			arguments = [[NSString alloc] initWithData:JSON encoding:NSUTF8StringEncoding];
		}
		NSString *identifier = [call[@"id"] isKindOfClass:NSString.class] && [call[@"id"] length] > 0
			? call[@"id"] : [@"call_" stringByAppendingString:NSUUID.UUID.UUIDString];
		[calls addObject:@{ @"id": identifier, @"type": @"function",
							@"function": @{ @"name": call[@"name"], @"arguments": arguments ?: @"{}" } }];
	}
	return calls;
}

+ (NSArray<NSDictionary *> *)annotationsForResult:(NFKInferenceResult *)result
{
	NSArray *citations = [result outputForKey:NFKOutputCitations];
	NSMutableArray<NSDictionary *> *annotations = [NSMutableArray array];
	for (NSDictionary *citation in [citations isKindOfClass:NSArray.class] ? citations : @[]) {
		if (![citation isKindOfClass:NSDictionary.class] || ![citation[@"url"] isKindOfClass:NSString.class]) {
			continue;
		}
		NSMutableDictionary *cited = [NSMutableDictionary dictionaryWithObject:citation[@"url"] forKey:@"url"];
		cited[@"title"] = citation[@"title"];
		cited[@"content"] = citation[@"text"];
		cited[@"start_index"] = citation[@"start"];
		cited[@"end_index"] = citation[@"end"];
		[annotations addObject:@{ @"type": @"url_citation", @"url_citation": cited }];
	}
	return annotations;
}

+ (nullable NSDictionary<NSString *, id> *)usageForResult:(NFKInferenceResult *)result
{
	NSDictionary *usage = [result outputForKey:NFKOutputUsage];
	if (![usage isKindOfClass:NSDictionary.class] || usage.count == 0) {
		return nil;
	}
	NSInteger input = [usage[NFKUsageInputTokens] integerValue];
	NSInteger output = [usage[NFKUsageOutputTokens] integerValue];
	NSMutableDictionary *wire = [@{ @"prompt_tokens": @(input), @"completion_tokens": @(output),
									@"total_tokens": @(input + output) } mutableCopy];
	if (usage[NFKUsageCachedTokens] != nil) {
		wire[@"prompt_tokens_details"] = @{ @"cached_tokens": usage[NFKUsageCachedTokens] };
	}
	if (usage[NFKUsageReasoningTokens] != nil) {
		wire[@"completion_tokens_details"] = @{ @"reasoning_tokens": usage[NFKUsageReasoningTokens] };
	}
	return wire;
}

#pragma mark Embeddings

+ (nullable NSArray<NSString *> *)embeddingInputsFromBody:(NSDictionary *)body error:(NSError * _Nullable *)outError
{
	id input = body[@"input"];
	NSArray *inputs = [input isKindOfClass:NSString.class] ? @[ input ] : ([input isKindOfClass:NSArray.class] ? input : nil);
	if (inputs.count == 0) {
		[self setError:outError code:kNFKError_InferenceMissingInput reason:@"the request carries no input"];
		return nil;
	}
	for (id entry in inputs) {
		if (![entry isKindOfClass:NSString.class]) {
			[self setError:outError code:kNFKError_InferenceUnsupported
					reason:@"only text inputs are served; token arrays and media parts are not"];
			return nil;
		}
	}
	return inputs;
}

+ (NSDictionary<NSString *, id> *)embeddingParametersFromBody:(NSDictionary *)body
{
	NSMutableDictionary<NSString *, id> *parameters = [body mutableCopy];
	[parameters removeObjectsForKeys:@[ @"model", @"input", @"encoding_format", @"user" ]];
	return parameters;
}

+ (NSDictionary<NSString *, id> *)embeddingReplyForVectors:(NSArray<NSArray<NSNumber *> *> *)vectors
													 model:(NSString *)model
													base64:(BOOL)base64
{
	NSMutableArray *data = [NSMutableArray arrayWithCapacity:vectors.count];
	[vectors enumerateObjectsUsingBlock:^(NSArray<NSNumber *> *vector, NSUInteger index, BOOL *stop) {
		id embedding = vector;
		if (base64) {
			NSMutableData *floats = [NSMutableData dataWithLength:vector.count * sizeof(float)];
			float *values = floats.mutableBytes;
			for (NSUInteger component = 0; component < vector.count; component++) {
				values[component] = vector[component].floatValue;
			}
			embedding = [floats base64EncodedStringWithOptions:0];
		}
		[data addObject:@{ @"object": @"embedding", @"index": @(index), @"embedding": embedding }];
	}];
	return @{ @"object": @"list", @"data": data, @"model": model,
			  @"usage": @{ @"prompt_tokens": @0, @"total_tokens": @0 } };
}

#pragma mark Transcription

+ (nullable NFKHTTPFormPart *)partNamed:(NSString *)name in:(NSArray<NFKHTTPFormPart *> *)parts
{
	NSString *bracketed = [name stringByAppendingString:@"[]"];
	for (NFKHTTPFormPart *part in parts) {
		if ([part.name isEqualToString:name] || [part.name isEqualToString:bracketed]) {
			return part;
		}
	}
	return nil;
}

+ (NSArray<NSString *> *)valuesNamed:(NSString *)name in:(NSArray<NFKHTTPFormPart *> *)parts
{
	NSString *bracketed = [name stringByAppendingString:@"[]"];
	NSMutableArray<NSString *> *values = [NSMutableArray array];
	for (NFKHTTPFormPart *part in parts) {
		if ([part.name isEqualToString:name] || [part.name isEqualToString:bracketed]) {
			[values addObject:part.stringValue];
		}
	}
	return values;
}

+ (nullable NFKInferenceRequest *)transcriptionRequestFromParts:(NSArray<NFKHTTPFormPart *> *)parts
													translating:(BOOL)translating
												 temporaryFiles:(NSMutableArray<NSURL *> *)temporaryFiles
														  error:(NSError * _Nullable *)outError
{
	NFKHTTPFormPart *file = [self partNamed:@"file" in:parts];
	if (file == nil || file.data.length == 0) {
		[self setError:outError code:kNFKError_InferenceMissingInput reason:@"the request carries no audio file"];
		return nil;
	}
	NSString *extension = file.filename.pathExtension.length > 0 ? file.filename.pathExtension
		: NFKServedExtensionForMediaType(file.contentType, @"wav");
	NSURL *fileURL = [self writeTemporary:file.data extension:extension files:temporaryFiles error:outError];
	if (fileURL == nil) {
		return nil;
	}
	NSMutableDictionary<NSString *, id> *inputs = [NSMutableDictionary dictionaryWithObject:[NFKAudioAsset audioAssetWithFileURL:fileURL]
																					 forKey:NFKInputAudio];
	NSString *prompt = [self partNamed:@"prompt" in:parts].stringValue;
	if (prompt.length > 0) {
		inputs[NFKInputPrompt] = prompt;
	}

	NSMutableDictionary<NSString *, id> *parameters = [NSMutableDictionary dictionary];
	NSSet<NSString *> *consumed = [NSSet setWithArray:@[ @"file", @"model", @"prompt", @"language", @"response_format",
														 @"timestamp_granularities", @"timestamp_granularities[]",
														 @"stream", @"diarize", @"keywords", @"keywords[]", @"temperature" ]];
	for (NFKHTTPFormPart *part in parts) {
		if (![consumed containsObject:part.name] && part.filename == nil) {
			parameters[part.name] = part.stringValue;
		}
	}
	NSString *language = [self partNamed:@"language" in:parts].stringValue;
	if (language.length > 0) {
		parameters[NFKParameterSourceLanguage] = language;
	}
	if (translating) {
		parameters[NFKParameterTargetLanguage] = @"en";
	}
	NSString *temperature = [self partNamed:@"temperature" in:parts].stringValue;
	if (temperature.length > 0) {
		parameters[NFKParameterTemperature] = @(temperature.doubleValue);
	}
	if ([[self valuesNamed:@"timestamp_granularities" in:parts] containsObject:@"word"]) {
		parameters[NFKParameterWordTimestamps] = @YES;
	}
	NSString *format = [self partNamed:@"response_format" in:parts].stringValue;
	if ([[self partNamed:@"diarize" in:parts].stringValue isEqualToString:@"true"] || [format isEqualToString:@"diarized_json"]) {
		parameters[NFKParameterSpeakerDiarization] = @YES;
	}
	NSArray<NSString *> *keywords = [self valuesNamed:@"keywords" in:parts];
	if (keywords.count > 0) {
		parameters[NFKParameterVocabulary] = keywords;
	}
	return [NFKInferenceRequest requestWithInputs:inputs parameters:parameters outputModality:NFKModalityText];
}

+ (NSData *)transcriptionBodyForResult:(NFKInferenceResult *)result
								format:(NSString *)format
						   translating:(BOOL)translating
						   contentType:(NSString * _Nonnull * _Nonnull)outContentType
{
	NSArray<NFKAudioSegment *> *segments = [result.segments isKindOfClass:NSArray.class] ? result.segments : @[];
	NSArray<NFKAudioSegment *> *words = [[result outputForKey:NFKOutputWords] isKindOfClass:NSArray.class] ? [result outputForKey:NFKOutputWords] : @[];
	NSString *text = result.text;
	if (text == nil) {
		text = [[segments valueForKey:@"label"] componentsJoinedByString:@" "];
	}
	if ([format isEqualToString:@"text"]) {
		*outContentType = @"text/plain; charset=utf-8";
		return [text dataUsingEncoding:NSUTF8StringEncoding];
	}
	if ([format isEqualToString:@"srt"] || [format isEqualToString:@"vtt"]) {
		*outContentType = @"text/plain; charset=utf-8";
		return [[self subtitlesForSegments:segments vtt:[format isEqualToString:@"vtt"]] dataUsingEncoding:NSUTF8StringEncoding];
	}
	NSMutableDictionary<NSString *, id> *body = [NSMutableDictionary dictionaryWithObject:text forKey:@"text"];
	if ([format isEqualToString:@"verbose_json"] || [format isEqualToString:@"diarized_json"]) {
		BOOL diarized = [format isEqualToString:@"diarized_json"];
		NSMutableArray *wireSegments = [NSMutableArray array];
		[segments enumerateObjectsUsingBlock:^(NFKAudioSegment *segment, NSUInteger index, BOOL *stop) {
			NSMutableDictionary *entry = [@{ @"id": diarized ? (id)[NSString stringWithFormat:@"seg_%lu", (unsigned long)index] : @(index),
											 @"start": @(segment.startSeconds), @"end": @(segment.endSeconds),
											 @"text": segment.label ?: @"" } mutableCopy];
			entry[@"speaker"] = segment.speaker;
			if (diarized) {
				entry[@"type"] = @"transcript.text.segment";
			}
			[wireSegments addObject:entry];
		}];
		body[@"segments"] = wireSegments;
		if (!diarized) {
			NSMutableArray *wireWords = [NSMutableArray array];
			for (NFKAudioSegment *word in words) {
				[wireWords addObject:@{ @"word": word.label ?: @"", @"start": @(word.startSeconds), @"end": @(word.endSeconds) }];
			}
			body[@"words"] = wireWords;
			body[@"task"] = translating ? @"translate" : @"transcribe";
			body[@"duration"] = @([segments.lastObject endSeconds]);
		}
	}
	*outContentType = @"application/json";
	return [NSJSONSerialization dataWithJSONObject:body options:0 error:NULL] ?: [NSData data];
}

+ (NSString *)subtitlesForSegments:(NSArray<NFKAudioSegment *> *)segments vtt:(BOOL)vtt
{
	NSString *(^stamp)(double) = ^NSString *(double seconds) {
		long milliseconds = lround(seconds * 1000.0);
		return [NSString stringWithFormat:vtt ? @"%02ld:%02ld:%02ld.%03ld" : @"%02ld:%02ld:%02ld,%03ld",
				milliseconds / 3600000, (milliseconds / 60000) % 60, (milliseconds / 1000) % 60, milliseconds % 1000];
	};
	NSMutableString *subtitles = [NSMutableString stringWithString:vtt ? @"WEBVTT\n\n" : @""];
	[segments enumerateObjectsUsingBlock:^(NFKAudioSegment *segment, NSUInteger index, BOOL *stop) {
		if (!vtt) {
			[subtitles appendFormat:@"%lu\n", (unsigned long)index + 1];
		}
		[subtitles appendFormat:@"%@ --> %@\n%@\n\n", stamp(segment.startSeconds), stamp(segment.endSeconds), segment.label ?: @""];
	}];
	return subtitles;
}

#pragma mark Speech

+ (nullable NFKInferenceRequest *)speechRequestFromBody:(NSDictionary *)body error:(NSError * _Nullable *)outError
{
	NSString *input = [body[@"input"] isKindOfClass:NSString.class] ? body[@"input"] : nil;
	if (input.length == 0) {
		[self setError:outError code:kNFKError_InferenceMissingInput reason:@"the request carries no input text"];
		return nil;
	}
	NSMutableDictionary<NSString *, id> *parameters = [body mutableCopy];
	[parameters removeObjectsForKeys:@[ @"model", @"input", @"response_format", @"stream_format", @"language", @"sample_rate" ]];
	parameters[NFKParameterSourceLanguage] = body[@"language"];
	parameters[NFKParameterSampleRate] = body[@"sample_rate"];
	return [NFKInferenceRequest requestWithInputs:@{ NFKInputPrompt: input } parameters:parameters outputModality:NFKModalityAudio];
}

+ (nullable NSData *)bytesForAudio:(id)audio format:(NSString * _Nullable * _Nullable)outFormat
{
	if ([audio isKindOfClass:NSData.class]) {
		return audio;
	}
	NSURL *file = [audio isKindOfClass:NFKAudioAsset.class] ? [(NFKAudioAsset *)audio fileURL] : ([audio isKindOfClass:NSURL.class] ? audio : nil);
	if (outFormat != NULL) {
		*outFormat = file.pathExtension.lowercaseString;
	}
	return file.isFileURL ? [NSData dataWithContentsOfURL:file] : nil;
}

+ (NSString *)contentTypeForAudioFormat:(NSString *)format
{
	NSDictionary<NSString *, NSString *> *types = @{ @"wav": @"audio/wav", @"mp3": @"audio/mpeg", @"flac": @"audio/flac",
													 @"aac": @"audio/aac", @"m4a": @"audio/mp4", @"opus": @"audio/opus",
													 @"pcm": @"audio/pcm", @"caf": @"audio/x-caf" };
	return types[format] ?: @"application/octet-stream";
}

+ (nullable NSData *)audioDataForResult:(NFKInferenceResult *)result
								 format:(NSString *)format
							contentType:(NSString * _Nullable * _Nullable)outContentType
								  error:(NSError * _Nullable *)outError
{
	id audio = [result outputForKey:NFKOutputAudio];
	if (outContentType != NULL) {
		*outContentType = [self contentTypeForAudioFormat:format];
	}
	if ([audio isKindOfClass:NSData.class]) {
		return audio;
	}
	AVAudioPCMBuffer *samples = [audio isKindOfClass:AVAudioPCMBuffer.class] ? audio : nil;
	NSURL *source = [audio isKindOfClass:NFKAudioAsset.class] ? [(NFKAudioAsset *)audio fileURL] : ([audio isKindOfClass:NSURL.class] ? audio : nil);
	if (samples == nil && !source.isFileURL) {
		[self setError:outError code:kNFKError_InferenceBackendFailure reason:@"the model returned no audio under NFKOutputAudio"];
		return nil;
	}
	if (samples == nil && [source.pathExtension.lowercaseString isEqualToString:format]) {
		return [NSData dataWithContentsOfURL:source options:0 error:outError];
	}
	if (samples == nil) {
		AVAudioFile *file = [[AVAudioFile alloc] initForReading:source error:outError];
		samples = file != nil ? [[AVAudioPCMBuffer alloc] initWithPCMFormat:file.processingFormat frameCapacity:(AVAudioFrameCount)MAX(file.length, 1)] : nil;
		if (samples == nil || ![file readIntoBuffer:samples error:outError]) {
			return nil;
		}
	}
	return [self encodeSamples:samples format:format error:outError];
}

// Apple's encoders write WAV, FLAC, AAC, and CAF; MP3 and Ogg Opus have none, so those formats are
// served only when the model itself produced them.
+ (nullable NSData *)encodeSamples:(AVAudioPCMBuffer *)samples format:(NSString *)format error:(NSError * _Nullable *)outError
{
	if ([format isEqualToString:@"pcm"]) {
		return [self int16SamplesFrom:samples];
	}
	NSDictionary<NSString *, NSNumber *> *formatIDs = @{ @"wav": @(kAudioFormatLinearPCM), @"flac": @(kAudioFormatFLAC),
														 @"aac": @(kAudioFormatMPEG4AAC), @"m4a": @(kAudioFormatMPEG4AAC),
														 @"caf": @(kAudioFormatLinearPCM) };
	NSNumber *formatID = formatIDs[format];
	if (formatID == nil) {
		[self setError:outError code:kNFKError_InferenceUnsupported
				reason:[NSString stringWithFormat:@"this server encodes wav, pcm, flac, aac, m4a, and caf; the model's audio is not %@", format]];
		return nil;
	}
	NSMutableDictionary *settings = [@{ AVFormatIDKey: formatID,
										AVSampleRateKey: @(samples.format.sampleRate),
										AVNumberOfChannelsKey: @(samples.format.channelCount) } mutableCopy];
	if (formatID.unsignedIntValue == kAudioFormatLinearPCM) {
		[settings addEntriesFromDictionary:@{ AVLinearPCMBitDepthKey: @16, AVLinearPCMIsFloatKey: @NO,
											  AVLinearPCMIsBigEndianKey: @NO, AVLinearPCMIsNonInterleaved: @NO }];
	}
	NSURL *destination = [[NSURL fileURLWithPath:NSTemporaryDirectory() isDirectory:YES]
						  URLByAppendingPathComponent:[NSString stringWithFormat:@"inferkit-served-%@.%@", NSUUID.UUID.UUIDString,
													   [format isEqualToString:@"aac"] ? @"m4a" : format]];
	NSData *encoded = nil;
	NSError *failure = nil;
	// The file is closed, and its last packets written, when the pool releases it.
	@autoreleasepool {
		NSError *writeError = nil;
		AVAudioFile *file = [[AVAudioFile alloc] initForWriting:destination settings:settings
												   commonFormat:samples.format.commonFormat
													interleaved:samples.format.isInterleaved error:&writeError];
		if (file == nil || ![file writeFromBuffer:samples error:&writeError]) {
			failure = writeError;
		}
	}
	if (failure == nil) {
		encoded = [NSData dataWithContentsOfURL:destination options:0 error:&failure];
	}
	[NSFileManager.defaultManager removeItemAtURL:destination error:NULL];
	if (encoded == nil && outError != NULL) {
		*outError = failure;
	}
	return encoded;
}

/*! Raw 16-bit little-endian interleaved samples, OpenAI's pcm format at the model's own rate. */
+ (NSData *)int16SamplesFrom:(AVAudioPCMBuffer *)samples
{
	AVAudioFormat *target = [[AVAudioFormat alloc] initWithCommonFormat:AVAudioPCMFormatInt16 sampleRate:samples.format.sampleRate
															   channels:samples.format.channelCount interleaved:YES];
	AVAudioConverter *converter = [[AVAudioConverter alloc] initFromFormat:samples.format toFormat:target];
	AVAudioPCMBuffer *converted = [[AVAudioPCMBuffer alloc] initWithPCMFormat:target frameCapacity:MAX(samples.frameLength, 1)];
	if (converter == nil || converted == nil || ![converter convertToBuffer:converted fromBuffer:samples error:NULL]) {
		return [NSData data];
	}
	return [NSData dataWithBytes:converted.int16ChannelData[0] length:converted.frameLength * target.streamDescription->mBytesPerFrame];
}

#pragma mark Images

+ (nullable NFKInferenceRequest *)imageGenerationRequestFromBody:(NSDictionary *)body error:(NSError * _Nullable *)outError
{
	NSString *prompt = [body[@"prompt"] isKindOfClass:NSString.class] ? body[@"prompt"] : nil;
	if (prompt.length == 0) {
		[self setError:outError code:kNFKError_InferenceMissingInput reason:@"the request carries no prompt"];
		return nil;
	}
	NSMutableDictionary<NSString *, id> *inputs = [NSMutableDictionary dictionaryWithObject:prompt forKey:NFKInputPrompt];
	if ([body[@"negative_prompt"] isKindOfClass:NSString.class]) {
		inputs[NFKInputNegativePrompt] = body[@"negative_prompt"];
	}
	NSMutableDictionary<NSString *, id> *fields = [body mutableCopy];
	[fields removeObjectsForKeys:@[ @"prompt", @"negative_prompt" ]];
	return [NFKInferenceRequest requestWithInputs:inputs parameters:[self imageParametersFromFields:fields] outputModality:NFKModalityImage];
}

+ (NSDictionary<NSString *, id> *)imageParametersFromFields:(NSDictionary<NSString *, id> *)fields
{
	NSMutableDictionary<NSString *, id> *parameters = [NSMutableDictionary dictionary];
	NSDictionary<NSString *, NSString *> *renamed = NFKServedImageParameterNames();
	NSSet<NSString *> *consumed = [NSSet setWithArray:@[ @"model", @"size", @"response_format", @"stream" ]];
	for (NSString *key in fields) {
		if (renamed[key] != nil) {
			parameters[renamed[key]] = fields[key];
		} else if (![consumed containsObject:key]) {
			parameters[key] = fields[key];
		}
	}
	NSArray<NSString *> *size = [fields[@"size"] isKindOfClass:NSString.class] ? [fields[@"size"] componentsSeparatedByString:@"x"] : nil;
	if (size.count == 2 && size[0].integerValue > 0 && size[1].integerValue > 0) {
		parameters[NFKParameterWidth] = @(size[0].integerValue);
		parameters[NFKParameterHeight] = @(size[1].integerValue);
	}
	return parameters;
}

+ (nullable NFKInferenceRequest *)imageEditRequestFromParts:(NSArray<NFKHTTPFormPart *> *)parts error:(NSError * _Nullable *)outError
{
	NSMutableArray *images = [NSMutableArray array];
	id mask = nil;
	NSMutableDictionary<NSString *, id> *fields = [NSMutableDictionary dictionary];
	NSSet<NSString *> *numeric = [NSSet setWithArray:@[ @"n", @"seed", @"steps", @"guidance_scale" ]];
	for (NFKHTTPFormPart *part in parts) {
		BOOL image = [part.name isEqualToString:@"image"] || [part.name isEqualToString:@"image[]"];
		if (image || [part.name isEqualToString:@"mask"]) {
			CVPixelBufferRef buffer = [NFKImageCoding pixelBufferWithImageData:part.data];
			if (buffer == NULL) {
				[self setError:outError code:kNFKError_InferenceUnsupported
						reason:[NSString stringWithFormat:@"the %@ part is not an image this server can decode", part.name]];
				return nil;
			}
			id decoded = (__bridge_transfer id)buffer;
			if (image) {
				[images addObject:decoded];
			} else {
				mask = decoded;
			}
			continue;
		}
		fields[part.name] = [numeric containsObject:part.name] ? @(part.stringValue.doubleValue) : part.stringValue;
	}
	if (images.count == 0) {
		[self setError:outError code:kNFKError_InferenceMissingInput reason:@"the request carries no image to edit"];
		return nil;
	}
	NSMutableDictionary<NSString *, id> *inputs = [NSMutableDictionary dictionary];
	[self addMedia:images firstKey:NFKInputImage restKey:NFKInputImages to:inputs];
	inputs[NFKInputMask] = mask;
	inputs[NFKInputPrompt] = fields[@"prompt"];
	inputs[NFKInputNegativePrompt] = fields[@"negative_prompt"];
	[fields removeObjectsForKeys:@[ @"prompt", @"negative_prompt" ]];
	return [NFKInferenceRequest requestWithInputs:inputs parameters:[self imageParametersFromFields:fields] outputModality:NFKModalityImage];
}

+ (nullable NSDictionary<NSString *, id> *)imageReplyForResult:(NFKInferenceResult *)result
											   responseFormat:(nullable NSString *)responseFormat
												 outputFormat:(nullable NSString *)outputFormat
														error:(NSError * _Nullable *)outError
{
	NSArray *images = [self imagesForResult:result];
	if (images.count == 0) {
		[self setError:outError code:kNFKError_InferenceBackendFailure reason:@"the model returned no image"];
		return nil;
	}
	BOOL jpeg = [outputFormat isEqualToString:@"jpeg"] || [outputFormat isEqualToString:@"jpg"];
	NSMutableArray *data = [NSMutableArray arrayWithCapacity:images.count];
	for (id image in images) {
		NSData *bytes = jpeg ? [self JPEGDataForImage:image] : [NFKImageCoding PNGDataForImage:image];
		if (bytes == nil) {
			[self setError:outError code:kNFKError_InferenceUnsupported reason:@"a generated image's pixel format cannot be encoded"];
			return nil;
		}
		NSString *encoded = [bytes base64EncodedStringWithOptions:0];
		// No file is hosted, so a url reply is the image as a data URL.
		[data addObject:[responseFormat isEqualToString:@"url"]
			? @{ @"url": [NSString stringWithFormat:@"data:image/%@;base64,%@", jpeg ? @"jpeg" : @"png", encoded] }
			: @{ @"b64_json": encoded }];
	}
	return @{ @"created": @((NSInteger)NSDate.date.timeIntervalSince1970), @"data": data, @"output_format": jpeg ? @"jpeg" : @"png" };
}

+ (nullable NSData *)JPEGDataForImage:(id)image
{
	CGImageRef cgImage = [NFKImageCoding CGImageForImage:image];
	if (cgImage == NULL) {
		return nil;
	}
	NSMutableData *data = [NSMutableData data];
	CGImageDestinationRef destination = CGImageDestinationCreateWithData((__bridge CFMutableDataRef)data, CFSTR("public.jpeg"), 1, NULL);
	BOOL written = NO;
	if (destination != NULL) {
		CGImageDestinationAddImage(destination, cgImage, (__bridge CFDictionaryRef)@{ (id)kCGImageDestinationLossyCompressionQuality: @0.9 });
		written = CGImageDestinationFinalize(destination);
		CFRelease(destination);
	}
	CGImageRelease(cgImage);
	return written ? data : nil;
}

#pragma mark Helpers

+ (nullable NSURL *)writeTemporary:(NSData *)data extension:(NSString *)extension files:(NSMutableArray<NSURL *> *)files error:(NSError * _Nullable *)outError
{
	NSURL *url = NFKRemoteWriteMediaFile(data, @"served", NFKServedFileExtension(extension), nil, outError);
	if (url != nil) {
		[files addObject:url];
	}
	return url;
}

+ (BOOL)setError:(NSError * _Nullable *)outError code:(NSInteger)code reason:(NSString *)reason
{
	if (outError != NULL) {
		*outError = [NSError errorWithDomain:NFKInferenceErrorDomain code:code userInfo:@{ NSLocalizedDescriptionKey: reason }];
	}
	return NO;
}

@end
